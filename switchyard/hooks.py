"""The LiteLLM proxy plugin. Register it in litellm-config.yaml:

    litellm_settings:
      callbacks: ["switchyard.hooks.switchyard_handler"]

Deliberately built on LiteLLM's *stable* extension points (CustomLogger hooks)
rather than the beta custom-routing-strategy API, so a LiteLLM upgrade cannot
silently change how capacity is allocated. The pre-call hook rewrites a lane
name into a concrete deployment; from LiteLLM's point of view every request
names exactly one provider and it never has to make a routing decision.

A failed request returns to the caller fast (num_retries=0 in the generated
config). The caller's 429 retry re-enters the proxy through this pre-call
hook, and the picker places against the fresh cooldowns the failure just set
- which is strictly better placement than any in-router retry could give.
The pinned litellm's `async_pre_routing_hook` is the internal auto-router
hook and never iterates registered CustomLogger callbacks, so a Switchyard
re-pick there was never going to fire even with num_retries=1.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import threading
import time
from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from redis.asyncio import Redis

from . import models
from .classify import Outcome, classify, escalated_cooldown, inspect_success_payload
from .models import Group
from .picker import LaneSaturated, Picker
from .policy import CapacityPolicy
from .reasoning import ReasoningSplitter
from .session import derive as derive_session
from .slots import SlotTable
from .usage import Ledger

log = logging.getLogger("switchyard")


def _configure_logging() -> None:
    """Attach our own handler at our own level.

    LiteLLM owns the root logger configuration, and under it our INFO records
    were dropped — so the startup banner and every `lane=... -> plan` line went
    missing while routing worked fine. A silently invisible router is worse than
    a noisy one: you cannot tell it apart from one that never loaded.
    """
    level = os.environ.get("SWITCHYARD_LOG", "INFO").upper()
    log.setLevel(getattr(logging, level, logging.INFO))
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("switchyard: %(message)s"))
        log.addHandler(handler)
    # Don't also hand records to the root logger, or every line appears twice.
    log.propagate = False


_configure_logging()

META_KEY = "switchyard"

# Call types that LiteLLM does NOT log via async_log_success_event /
# async_log_failure_event (LiteLLM 1.101.0 verified mapping). For these,
# slot release + usage accounting runs from async_post_call_success_hook
# (buffered) or async_post_call_streaming_iterator_hook (streamed). Buffered
# and streamed variants both belong on the post-call side; the success path's
# exactly-once marker covers the rare case where both happen to fire.
#
# Everything else stays owned by normal success logging -- the four classic
# chat/text call types (completion, acompletion, text_completion,
# atext_completion) keep firing async_log_success_event on success and
# async_log_failure_event on failure, and we MUST NOT do the work twice.
UNLOGGED_CALL_TYPES = frozenset({"anthropic_messages", "aanthropic_messages"})

# Per-call ctx marker. The first path to finish sets the marker; whichever
# finishes second is a no-op. The value is "success" or "failure" so a future
# log can show which side ran (mostly for debugging a double-fire surprise).
_TERMINATED = "_terminated"


class SwitchyardHandler(CustomLogger):
    def __init__(self) -> None:
        self.registry = models.load()
        self._redis: Redis | None = None
        self._slots: SlotTable | None = None
        self._picker: Picker | None = None
        self._ledger: Ledger | None = None
        self._policy: CapacityPolicy | None = None
        self._plans_mtime: float | None = self._stat_plans()
        self._watcher: threading.Thread | None = None
        self._sync_redis_client: Any = None
        # request_id -> heartbeat task, so a live request keeps its slot and a
        # dead one loses it within inflight_max_age_seconds.
        self._beats: dict[str, asyncio.Task] = {}
        log.info(
            "%d plans, lanes=%s, tool-capable=%d",
            len(self.registry.plans), ",".join(self.registry.lanes),
            sum(1 for p in self.registry.plans.values() if p.can_use_tools),
        )
        self._start_watcher()

    # -- wiring ------------------------------------------------------------
    @property
    def redis(self) -> Redis:
        if self._redis is None:
            self._redis = Redis.from_url(
                os.environ.get("SWITCHYARD_REDIS_URL", "redis://redis:6379/1")
            )
        return self._redis

    @property
    def slots(self) -> SlotTable:
        if self._slots is None:
            self._slots = SlotTable(
                self.redis, self.registry.settings.inflight_max_age_seconds
            )
        return self._slots

    @property
    def policy(self) -> CapacityPolicy:
        if self._policy is None:
            self._policy = CapacityPolicy(self.redis, self.registry.settings, self.ledger)
        return self._policy

    @property
    def picker(self) -> Picker:
        if self._picker is None:
            self._picker = Picker(self.registry, self.slots, self.policy)
        return self._picker

    @property
    def ledger(self) -> Ledger:
        if self._ledger is None:
            self._ledger = Ledger(self.redis)
        return self._ledger

    # -- live config reload ------------------------------------------------
    # All state that matters lives in Redis — learned caps, pacing, cooldowns,
    # session leases, usage — so rebuilding the registry costs nothing but a
    # yaml parse. The LiteLLM router is the exception: it is built at startup
    # from the generated config, which bakes in model strings, api_base,
    # credentials and context windows. router_signature() draws exactly that
    # line: a policy-only edit swaps in place within HOT_RELOAD_SECONDS, while
    # an edit the router cannot follow keeps the old registry (so the running
    # router and the routing policy never disagree) and says so loudly.
    #
    # A plain daemon thread, not an asyncio task: the handler is constructed at
    # config-parse time, possibly before any event loop exists, and a watcher
    # spawned from the pre-call hook would not start at all on an idle gateway
    # — which is exactly when you edit the config. Blocking IO in a thread we
    # own is fine; the swap is GIL-atomic attribute assignment, and every
    # worker of a multi-worker gateway runs its own watcher against the same
    # mounted file and converges on the same registry.
    HOT_RELOAD_SECONDS = 5.0
    ROUTER_SIG_KEY = "switchyard:router_sig"

    def _start_watcher(self) -> None:
        if self._watcher is not None:
            return
        self._watcher = threading.Thread(
            target=self._watch_loop, name="switchyard-config-watcher", daemon=True)
        self._watcher.start()

    def _sync_redis(self):
        if self._sync_redis_client is None:
            import redis as sync_redis
            self._sync_redis_client = sync_redis.Redis.from_url(
                os.environ.get("SWITCHYARD_REDIS_URL", "redis://redis:6379/1"))
        return self._sync_redis_client

    def _publish_sig(self) -> None:
        try:
            self._sync_redis().set(self.ROUTER_SIG_KEY,
                                    models.router_signature(self.registry))
        except Exception as exc:
            log.warning("could not publish router signature to redis: %r", exc)

    def _stat_plans(self) -> float | None:
        try:
            return os.stat(models.CONFIG_PATH).st_mtime
        except OSError:
            return None

    def _watch_loop(self) -> None:
        # Publish first, every iteration, not just at startup: redis flush,
        # gateway restart, or a refused router-shaped edit all leave the key
        # holding what is actually routing, and reload.sh compares against it.
        while True:
            try:
                self._maybe_reload()
            except Exception as exc:
                log.warning("config watcher iteration failed: %r", exc)
            self._publish_sig()
            time.sleep(self.HOT_RELOAD_SECONDS)

    def _maybe_reload(self) -> None:
        mtime = self._stat_plans()
        if mtime is None or mtime == self._plans_mtime:
            return
        try:
            fresh = models.load()
        except Exception as exc:
            # Advance the stamp so a broken file is reported once, not every
            # cycle; the next edit gets a new stamp and is tried again.
            self._plans_mtime = mtime
            log.warning("plans.yaml reload skipped — %s: %s", type(exc).__name__, exc)
            return
        self._plans_mtime = mtime
        if models.router_signature(fresh) != models.router_signature(self.registry):
            log.warning(
                "plans.yaml changed model strings / credentials / lanes — "
                "the LiteLLM router is built at startup, KEEPING the current "
                "routing; run scripts/reload.sh (or docker compose restart "
                "gateway) to apply it.")
            return
        self._swap_registry(fresh)
        log.info(
            "plans.yaml reloaded in place: %d plans, lanes=%s, tool-capable=%d",
            len(fresh.plans), ",".join(fresh.lanes),
            sum(1 for p in fresh.plans.values() if p.can_use_tools))

    def _swap_registry(self, fresh: models.Registry) -> None:
        # Build the replacements before promoting any of them, so the swap is
        # a sequence of assignments rather than a half-built state. The slot
        # table and ledger carry over: they are plan-keyed Redis state, not
        # registry state, and a request in flight is using them right now.
        if self._slots is not None:
            fresh_ttl = fresh.settings.inflight_max_age_seconds
            if self._slots.inflight_max_age != fresh_ttl:
                self._slots = SlotTable(self.redis, fresh_ttl)
        fresh_slots = self._slots
        fresh_ledger = self._ledger or Ledger(self.redis)
        policy = CapacityPolicy(self.redis, fresh.settings, fresh_ledger)
        picker = Picker(fresh, fresh_slots, policy)
        self.registry = fresh
        self._policy = policy
        self._picker = picker
        self._ledger = fresh_ledger

    # -- inbound: choose a provider ---------------------------------------
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        lane = data.get("model")
        direct = None
        if lane not in self.registry.lanes:
            # A caller naming one deployment still gets slot accounting, quota
            # gating and usage recording -- everything except lane spill.
            direct = self.registry.model_for_deployment(str(lane))
            if direct is None:
                return data      # not ours at all: a raw LiteLLM model name

        key_hash = None
        token = getattr(user_api_key_dict, "api_key", None)
        if token:
            key_hash = hashlib.sha256(str(token).encode()).hexdigest()[:8]
        session = derive_session(data, key_hash)

        # A request carrying tool definitions cannot go to a plan marked
        # supports_tools: false; see Plan.can_use_tools.
        needs_tools = bool(data.get("tools"))

        # A request carrying tool *results* is mid-loop. While its session
        # lease is alive it returns to the plan that minted its tool_call_ids
        # -- the prompt cache and the loop's quota are both there. Past the
        # lease TTL the pin is gone and it places fresh, which is safe because
        # any bridge rebuilds a lost session from the request itself. See
        # Picker.pick.
        pinned = _carries_tool_results(data.get("messages"))

        try:
            pick = (await self.picker.pick_direct(direct, session) if direct
                    else await self.picker.pick(lane, session, needs_tools, pinned))
        except LaneSaturated as exc:
            # Surfacing this as a 429 is what lets clients back off instead of
            # hammering a lane whose paid capacity is genuinely gone.
            from fastapi import HTTPException
            raise HTTPException(
                status_code=429,
                detail={"error": str(exc), "lane": lane,
                        "switchyard": "lane_saturated"},
                headers={"Retry-After": "20"},
            ) from exc

        self._start_heartbeat(pick.plan.key, pick.model.ref, pick.request_id)
        data["model"] = pick.model.deployment
        meta = data.setdefault("metadata", {})
        meta[META_KEY] = {
            "lane": lane,
            "plan": pick.plan.key,
            "model": pick.model.ref,
            "request_id": pick.request_id,
            "session": session,
            "sticky": pick.sticky,
            "claimed_at": time.time(),
            "cap": pick.cap,
            # direct / needs_tools / pinned are stamped so any downstream
            # failure path can tell why the picker picked what it picked and
            # so the post-call hooks can decide which plan to book the call
            # against. The caller's retry re-enters through this hook, not
            # through a router-level retry, so the picker gets a fresh
            # pick against whatever cooldowns the previous attempt set.
            "direct": bool(direct),
            "needs_tools": needs_tools,
            "pinned": pinned,
            # Group that produced the pick, when the lane walked one. Affinity
            # pins (picked_group is None) leave the field blank so a pinned
            # follow-up does not look like a group selection.
            "picked_group_gid": (pick.picked_group.gid
                                 if pick.picked_group is not None else ""),
            "picked_group_strategy": (pick.picked_group.strategy
                                      if pick.picked_group is not None else ""),
            # LiteLLM's call_type ("acompletion", "anthropic_messages", ...).
            # Stamped here so every downstream hook sees the same value and can
            # decide which side of LiteLLM's split logging path owns the slot
            # release. Buffered /v1/messages (anthropic_messages) reaches
            # async_post_call_success_hook only, streamed /v1/messages reaches
            # async_post_call_streaming_iterator_hook only -- the normal
            # async_log_success_event does NOT fire for either (LiteLLM 1.101.0
            # verified mapping).
            "call_type": call_type,
        }
        log.info(
            "lane=%s -> %s [%s]%s%s%s",
            lane, pick.model.ref, _reason_with_group(pick),
            " tools" if needs_tools else "",
            " (sticky)" if pick.sticky else "",
            f" skipped={','.join(pick.considered)}" if pick.considered else "",
        )
        return data

    # -- keeping a live claim alive ----------------------------------------
    def _start_heartbeat(self, plan_key: str, model_ref: str, request_id: str) -> None:
        interval = max(5, self.registry.settings.heartbeat_seconds)

        async def beat() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    alive = await self.slots.touch(plan_key, request_id, model_ref)
                    if not alive:
                        # The sweep already removed it; nothing to keep alive.
                        return
            except asyncio.CancelledError:
                raise
            except Exception:
                log.debug("heartbeat for %s stopped", request_id, exc_info=True)

        task = asyncio.create_task(beat())
        self._beats[request_id] = task

    def _stop_heartbeat(self, request_id: str) -> None:
        task = self._beats.pop(request_id, None)
        if task:
            task.cancel()

    # -- outbound: release the slot, record usage --------------------------
    def _ctx(self, kwargs: dict) -> dict | None:
        meta = (kwargs.get("litellm_params", {}) or {}).get("metadata") or kwargs.get("metadata") or {}
        ctx = meta.get(META_KEY)
        return ctx if isinstance(ctx, dict) else None

    def _check_served_deployment(self, ctx: dict, kwargs: dict):
        """Which plan actually served this, when it is not the one we picked.

        LiteLLM's router-level retries and context-window fallbacks run after the
        pre-call hook, so they can move a request to a deployment the picker
        never chose. Booking its tokens against the pick would spend one
        subscription's quota and debit another's, so the served plan is
        identified and used instead — and the move is logged, because it means
        something bypassed the routing rules (tool capability, the session
        lease, the mid-loop pin) and is worth seeing rather than absorbing.

        Returns (plan, served_ref): the plan to attribute usage to (None when
        it cannot be resolved) and the ref of the deployment that answered, so
        model-scoped economics land on the model that actually spent them.
        """
        picked = self.registry.plans.get(ctx["plan"])
        served_name = kwargs.get("model")
        if not served_name:
            return picked, ctx.get("model")
        served = self.registry.model_for_deployment(str(served_name))
        if served is None or served.ref == ctx.get("model"):
            return picked, ctx.get("model")
        actual = self.registry.plan_of(served)
        log.warning(
            "request was picked for %s but served by %s — LiteLLM moved it after "
            "the pre-call hook, so it bypassed the routing rules. Booking usage "
            "to %s, the plan whose quota it actually spent.",
            ctx.get("model"), served.ref, actual.key,
        )
        return actual, served.ref

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Buffered /v1/chat/completions, /v1/completions and friends finish here.

        LiteLLM 1.101.0 keeps the normal success logging on
        ``completion / acompletion / text_completion / atext_completion`` -- this
        is the only hook that fires for them. Buffered ``/v1/messages`` does
        NOT reach here (verified), and is owned by async_post_call_success_hook
        instead, gated on call_type. Either way the exactly-once marker inside
        ``_finish_success`` keeps both paths from running the work twice on the
        rare LiteLLM build where both happen to fire.
        """
        ctx = self._ctx(kwargs)
        if not ctx:
            return
        # Defensive gate. The unlogged call types do not normally reach this
        # hook, but if a future LiteLLM version starts firing it for them,
        # the post-call side still owns the slot -- we must not release twice.
        if ctx.get("call_type") in UNLOGGED_CALL_TYPES:
            return
        await self._finish_success(
            ctx, response_obj=response_obj,
            start_time=start_time, end_time=end_time,
            kwargs=kwargs,
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Same ownership story as ``async_log_success_event``, for the failure side.

        Failure accounting for ``/v1/messages`` is owned by
        ``async_post_call_failure_hook``; this hook stays out of the way for
        those call types and runs ``_finish_failure`` for everything else.
        """
        ctx = self._ctx(kwargs)
        if not ctx:
            return
        if ctx.get("call_type") in UNLOGGED_CALL_TYPES:
            return
        await self._finish_failure(
            ctx, kwargs.get("exception") or kwargs.get("original_exception"),
        )

    async def _finish_success(
        self, ctx: dict, *,
        response_obj: Any = None,
        start_time: Any = None,
        end_time: Any = None,
        kwargs: dict | None = None,
        usage: dict | None = None,
    ) -> bool:
        """Common success accounting. Idempotent via the ctx marker.

        Exactly-once guarantee: the first call sets ``ctx[_TERMINATED]``; any
        later call (success or failure path, post-call or normal) is a no-op.
        Without this guard, LiteLLM builds that fire both
        ``async_post_call_success_hook`` and ``async_log_success_event`` for
        buffered OpenAI routes would book the same slot release twice and the
        ledger would double-count.

        Returns True when this call did the work, False when it short-circuited.

        ``usage`` is an explicit override for streamed responses where the
        caller has already extracted usage out of the SSE chunks; absent, the
        method pulls it off ``response_obj.usage`` in either OpenAI shape
        (``prompt_tokens`` / ``completion_tokens``) or Anthropic shape
        (``input_tokens`` / ``output_tokens``).
        """
        if ctx.get(_TERMINATED):
            return False
        ctx[_TERMINATED] = "success"
        kwargs = kwargs or {}
        self._stop_heartbeat(ctx["request_id"])
        # Always release what we claimed, whoever ended up serving it.
        await self.picker.release(ctx["plan"], ctx["request_id"], ctx["model"])
        plan, served_ref = self._check_served_deployment(ctx, kwargs)
        if not plan:
            return True

        # A 200 is not proof of success. MiniMax can return HTTP 200 with the
        # real failure in base_resp.status_code; recording that as a success
        # would leave a quota-dead plan looking healthy and collecting every
        # request the lane can give it.
        verdict = inspect_success_payload(plan.provider_family, _payload_of(response_obj))
        if verdict is not None:
            log.warning(
                "plan=%s returned HTTP 200 carrying a failure: %s",
                plan.key, verdict.detail,
            )
            await self.ledger.record(plan, failed=True, model=served_ref)
            await self._apply_verdict(plan, verdict, ctx)
            return True

        prompt_tokens, completion_tokens = _prompt_completion_tokens(response_obj, usage)
        # Only metered providers have a per-request cost. On a subscription the
        # fee is fixed and the marginal cost of a request is zero; recording
        # LiteLLM's notional price would inflate month_cost and corrupt the
        # effective $/Mtok figure, which is the number used to judge whether the
        # subscription is worth renewing.
        cost = float(kwargs.get("response_cost") or 0.0) if plan.metered else 0.0
        await self.ledger.record(
            plan, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, cost=cost,
            model=served_ref,
        )

        # A genuine 200 against this plan resets its transient-failure
        # streak: the plan is talking to us again, so the next 5xx starts at
        # base (60s) rather than at the top of the ladder. Reset on the
        # SERVING plan (the one whose quota we just spent), not the picked
        # one — the served plan is the one whose streak actually moved.
        await self.slots.reset_transient_failures(plan.key)

        # Per-slot throughput drives the pacer: one busy slot delivered this
        # many units in this many seconds.
        try:
            seconds = (end_time - start_time).total_seconds()
        except (TypeError, AttributeError):
            seconds = max(0.0, time.time() - float(ctx.get("claimed_at") or time.time()))
        units = cost if plan.quota.kind == "dollars" else prompt_tokens + completion_tokens
        await self.policy.pacer.note_throughput(plan, units, seconds)

        await self._absorb_limit_headers(plan.key, kwargs)
        return True

    async def _finish_failure(self, ctx: dict, exception: Exception | None) -> bool:
        """Common failure accounting. Idempotent via the ctx marker.

        Same marker as ``_finish_success``: whichever path runs first wins, so
        a slot cannot be released twice when both ``async_log_failure_event``
        and ``async_post_call_failure_hook`` fire for the same request (which
        LiteLLM does for the buffered OpenAI routes on a router-level retry).
        """
        if ctx.get(_TERMINATED):
            return False
        ctx[_TERMINATED] = "failure"
        self._stop_heartbeat(ctx["request_id"])
        await self.picker.release(ctx["plan"], ctx["request_id"], ctx["model"])
        plan = self.registry.plans.get(ctx["plan"])
        # No failed_ref marker: the routed-to deployment IS the served one,
        # there is no router-level retry to feed it into (num_retries=0), and
        # the caller's retry re-enters through async_pre_call_hook which gets
        # its own fresh pick against the cooldowns this failure just set.
        if plan:
            await self.ledger.record(plan, failed=True, model=ctx["model"])
        await self._handle_failure(ctx, exception)
        return True

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response, **_: Any
    ):
        """Tell the caller which plan served it AND finish ``/v1/messages`` accounting.

        The body's ``model`` echoes what was asked for -- usually a lane name
        like ``judge`` -- so a caller doing its own token or cost accounting
        cannot see which subscription the tokens came out of. LiteLLM puts the
        answer in response *headers* (x-litellm-model-group and friends), which
        is easy to miss and lost by any client that only keeps the JSON. So it
        goes in the body too, under one namespaced key that a strict client
        will ignore.

        For call types LiteLLM does NOT route through ``async_log_success_event``
        (``anthropic_messages`` / ``aanthropic_messages``, per the 1.101.0
        mapping) this hook also runs ``_finish_success`` -- the slot release,
        ledger booking, transient-failure reset, throughput note and limit
        headers. For OpenAI-shaped buffered routes that call BOTH hooks the
        exactly-once marker keeps the work from running twice.
        """
        ctx = (((data or {}).get("metadata") or {}).get(META_KEY))
        if not isinstance(ctx, dict):
            return response
        stamp = {"lane": ctx.get("lane"), "plan": ctx.get("plan"),
                 "model": ctx.get("model"), "sticky": bool(ctx.get("sticky"))}
        try:
            if isinstance(response, dict):
                response["switchyard"] = stamp
            else:
                setattr(response, "switchyard", stamp)
        except Exception:                 # never fail a served request over a label
            log.debug("could not stamp response with switchyard routing info")

        if ctx.get("call_type") in UNLOGGED_CALL_TYPES:
            kwargs = (data or {}).get("litellm_params") or {}
            await self._finish_success(ctx, response_obj=response, kwargs=kwargs)
        return response

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: UserAPIKeyAuth, response: Any, request_data: dict
    ):
        """Lift inline ``<think>`` reasoning and finish streamed ``/v1/messages`` accounting.

        Two responsibilities, kept in one hook because LiteLLM only gives us
        one place in the streamed path:

        1. **Reasoning split**: a streamed chunk that arrives with inline
           ``<think>...</think>`` tags is rewritten so the reasoning moves
           into ``reasoning_content`` (matching the buffered path). Reasoning
           is moved, never dropped.
        2. **Finish streamed ``/v1/messages``**: LiteLLM does NOT fire
           ``async_log_success_event`` for streamed ``/v1/messages``, so the
           slot release + ledger booking + transient-failure reset + throughput
           note + limit headers have to happen here. ``_stream_chunks`` wraps
           the upstream iterator with usage parsing and a detached finalize
           task, so client cancellation cannot leave the slot claimed. OpenAI
           streamed routes are still owned by ``async_log_success_event``; the
           call_type gate below keeps us out of their way.

        Cancellation safety: the finalize is a detached asyncio task scheduled
        from the iterator's `try/except/else` arms, so the iterator itself
        never `await`s after the last `yield` -- the FastAPI/Starlette
        streaming response does not stall waiting for Redis writes, and a
        client disconnect mid-stream still produces a slot release.
        """
        if not self.registry.settings.split_reasoning_tags:
            async for chunk in self._stream_chunks(response, request_data):
                yield chunk
            return

        splitter = ReasoningSplitter()
        async for chunk in self._stream_chunks(response, request_data):
            try:
                choices = getattr(chunk, "choices", None) or []
                delta = getattr(choices[0], "delta", None) if choices else None
                text = getattr(delta, "content", None) if delta is not None else None
                if text:
                    content, reasoning = splitter.feed(text)
                    delta.content = content
                    if reasoning:
                        # The field may not exist on the model; set it either way
                        # so the caller sees the same shape as a buffered reply.
                        prior = getattr(delta, "reasoning_content", None) or ""
                        delta.reasoning_content = prior + reasoning
            except Exception:                 # never break a stream over this
                log.debug("reasoning split skipped for one chunk", exc_info=True)
            yield chunk

        trailing_content, trailing_reasoning = splitter.flush()
        if trailing_content or trailing_reasoning:
            log.debug("stream ended mid-tag; flushed %d content, %d reasoning chars",
                      len(trailing_content), len(trailing_reasoning))

    async def _stream_chunks(self, response: Any, request_data: dict):
        """Yield every byte/chunk from `response` and finish streamed accounting.

        The hook is a `yield`-per-chunk iterator, so the FastAPI/Starlette
        streaming response stays unblocked. We:

        * Parse every chunk for Anthropic SSE usage (``message_start`` carries
          the input-token total; successive ``message_delta`` frames carry
          the running output/cache totals). Tolerate chunks shaped as bytes,
          str, dict or pydantic objects; a malformed chunk is logged and
          skipped -- never propagated to the client.
        * Schedule a detached task from the success, exception and cancel
          arms so the slot is released whether the stream completes cleanly,
          errors out, or has its consumer disconnect mid-iteration.
        """
        ctx = ((request_data or {}).get("metadata") or {}).get(META_KEY)
        if not isinstance(ctx, dict):
            # Not one of ours -- /v1/chat/completions streamed, etc. Pass the
            # iterator through unchanged; ``async_log_success_event`` owns the
            # finish. The reasoning split above still runs (a no-op for chunks
            # that lack a ``.choices`` attribute).
            async for chunk in response:
                yield chunk
            return

        # Only this call type needs streamed finish. OpenAI-shaped routes
        # still own their success logging through async_log_success_event, so
        # we yield through and let the hook above do the reasoning split.
        if ctx.get("call_type") not in UNLOGGED_CALL_TYPES:
            async for chunk in response:
                yield chunk
            return

        collected: dict[str, int] = {}
        try:
            async for chunk in response:
                try:
                    self._absorb_chunk_usage(chunk, collected)
                except Exception:
                    log.debug("stream usage parse skipped for one chunk",
                              exc_info=True)
                yield chunk
        except BaseException:
            self._schedule_stream_finalize(ctx, collected, request_data)
            raise
        else:
            self._schedule_stream_finalize(ctx, collected, request_data)

    def _schedule_stream_finalize(
        self, ctx: dict, collected: dict, request_data: dict,
    ) -> None:
        """Detached finish task so iterator cleanup is never blocked on Redis.

        Awaiting the finish inside the streaming hook would delay the final
        yield to the client (FastAPI's StreamingResponse holds the generator
        open until it returns) and risk GeneratorExit swallowing the release.
        A detached task runs even if the hook's generator is GC'd -- the slot
        is released whether the stream completed, errored, or had the client
        hang up.

        The task wrapper swallows every exception: a finalize-time error
        (``Redis is down``, the registry has been hot-swapped, the model
        request id no longer exists) is logged at exception level rather than
        left as an unobserved task exception that would surface as
        ``Task exception was never retrieved``.
        """
        async def _run() -> None:
            try:
                kwargs = ((request_data or {}).get("litellm_params") or {})
                await self._finish_success(
                    ctx, kwargs=kwargs,
                    usage=collected or None,
                )
            except Exception:
                log.exception(
                    "stream finalize failed for request_id=%s",
                    ctx.get("request_id"),
                )
        try:
            asyncio.create_task(_run())
        except RuntimeError:
            # No running loop (synchronous caller, e.g. a test). Run inline.
            try:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(_run())
                finally:
                    loop.close()
            except Exception:
                log.exception(
                    "stream inline finalize failed for request_id=%s",
                    ctx.get("request_id"),
                )

    @staticmethod
    def _absorb_chunk_usage(chunk: Any, collected: dict) -> None:
        """Extract Anthropic usage from one streamed chunk. Never raises.

        Tolerated shapes:

        * ``bytes`` / ``bytearray``: raw SSE frame(s), as the provider
          passthrough emits for ``/v1/messages``. Frames are ``\n\n``
          terminated; each ``data:`` line carries a JSON event.
        * ``str``: same content, decoded.
        * ``dict``: a pre-parsed event (``message_start`` etc.). ``response.usage``
          is checked on the dict directly so a synthetic-stream or
          agentic-loop chunk is also covered.
        * Pydantic model with ``model_dump``: same as dict.

        Other shapes are skipped. Anything that throws while parsing is
        swallowed at the caller (`_stream_chunks`) so a malformed chunk
        cannot break the client stream.
        """
        if chunk is None:
            return
        if isinstance(chunk, dict):
            _collect_anthropic_event_usage(chunk, collected)
            return
        if isinstance(chunk, (bytes, bytearray)):
            try:
                text = bytes(chunk).decode("utf-8", errors="ignore")
            except Exception:
                return
        elif isinstance(chunk, str):
            text = chunk
        else:
            fn = getattr(chunk, "model_dump", None)
            if callable(fn):
                try:
                    dumped = fn()
                    if isinstance(dumped, dict):
                        _collect_anthropic_event_usage(dumped, collected)
                except Exception:
                    return
            return
        if not text:
            return
        # A single chunk can carry one or more `\n\n`-terminated frames. Walk
        # each, find the `data:` line, parse the JSON event.
        for frame in text.split("\n\n"):
            if not frame.strip():
                continue
            payload: str | None = None
            for line in frame.splitlines():
                stripped = line.strip()
                if stripped.startswith("data:"):
                    candidate = stripped[len("data:"):].strip()
                    if candidate and candidate != "[DONE]":
                        payload = candidate
                    break
            if payload is None:
                continue
            try:
                event = json.loads(payload)
            except (ValueError, TypeError):
                continue
            if isinstance(event, dict):
                _collect_anthropic_event_usage(event, collected)

    async def async_post_call_failure_hook(
        self, request_data: dict, original_exception: Exception, user_api_key_dict: UserAPIKeyAuth, **_: Any
    ) -> None:
        """Failure-side counterpart of ``async_post_call_success_hook``.

        ``/v1/messages`` failures reach here (LiteLLM does NOT fire
        ``async_log_failure_event`` for them, verified 1.101.0); OpenAI-shaped
        routes can call BOTH hooks, in which case ``_finish_failure`` short-
        circuits on the second arrival.
        """
        meta = (request_data or {}).get("metadata") or {}
        ctx = meta.get(META_KEY)
        if isinstance(ctx, dict):
            await self._finish_failure(ctx, original_exception)

    async def _handle_failure(self, ctx: dict, exc: Exception | None) -> None:
        plan = self.registry.plans.get(ctx.get("plan", ""))
        if plan is None or exc is None:
            return
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        try:
            status = int(status) if status is not None else None
        except (TypeError, ValueError):
            status = None
        retry_after = _retry_after(exc)
        verdict = classify(
            status, str(getattr(exc, "message", None) or exc),
            family=plan.provider_family,
            body=_error_body(exc),
            retry_after=retry_after,
            default_cooldown=self.registry.settings.default_cooldown_seconds,
        )
        await self._apply_verdict(plan, verdict, ctx)

    async def _apply_verdict(self, plan, verdict, ctx: dict) -> None:
        """Act on a classified failure: cool the plan, learn, release the lease."""
        if verdict.is_our_fault:
            return  # a bad prompt is not the provider's problem

        # CRITICAL double-count guard. async_log_failure_event fires PER
        # ATTEMPT and async_post_call_failure_hook fires once more after the
        # router's retries exhaust — both call _apply_verdict with the same
        # ctx object, because the router reuses the kwargs dict across
        # attempts. Without a per-attempt marker, a single 5xx would look
        # like two, doubling the streak and the cooldown ladder. The new
        # request_id minted for the re-pick is the natural key.
        marker = ctx.get("_verdict_applied")
        rid = ctx.get("request_id")
        if marker and marker == rid:
            return  # already applied for this attempt
        if rid:
            ctx["_verdict_applied"] = rid

        if verdict.outcome is Outcome.QUOTA_EXHAUSTED:
            # Which window did we hit? The reset time tells us, and attributing
            # it correctly keeps a 5-hour wall out of the weekly figures.
            window = await self.ledger.note_exhaustion(
                plan, time.time() + verdict.cooldown_seconds)
            log.warning(
                "plan=%s exhausted its %s window (%s) — dropping its %d slots for %ds",
                plan.key, window.label, verdict.detail, plan.max_parallel,
                verdict.cooldown_seconds,
            )
        elif verdict.outcome is Outcome.PLAN_DEAD:
            log.error(
                "plan=%s is over (%s) — out of every lane for %dh. "
                "Remove it from config/plans.yaml.",
                plan.key, verdict.detail, verdict.cooldown_seconds // 3600,
            )
        elif verdict.outcome is Outcome.CONCURRENCY:
            # The provider says we opened too many connections. Teach the
            # learner where the ceiling actually is, for this hour of the day.
            at_cap = int(ctx.get("cap") or plan.max_parallel)
            await self.ledger.note_concurrency_rejection(plan)
            learned = await self.policy.learner.note_rejection(plan, at_cap)
            log.warning(
                "plan=%s refused on connection limit at %d — learned cap now %d",
                plan.key, at_cap, learned,
            )

        # Apply the cooldown. TRANSIENT uses the escalated ladder so a broken
        # seat stops getting re-fed every 60s; every other outcome keeps the
        # verdict's own cooldown (a quota wall already knows how long it
        # should last).
        cooldown = verdict.cooldown_seconds
        did_cool_atomic = False
        if verdict.outcome is Outcome.TRANSIENT:
            breaker = self.registry.settings.transient_breaker
            if breaker.enabled:
                # Atomic: INCR + EXPIRE streak, compute cooldown from new
                # streak, SET cooldown key. The Lua uses the same formula as
                # `escalated_cooldown`, so the cooldown stored in Redis and
                # the one Python computes below for the log line always
                # agree. Returns the post-INCR streak. Concurrent TRANSIENT
                # failures on the same plan can no longer leave the picker
                # observing a cooldown that undershoots the final streak.
                streak = await self.slots.bump_and_cool(
                    plan.key,
                    base=cooldown,
                    cap=breaker.max_seconds,
                    reason=verdict.outcome.value,
                )
                cooldown = escalated_cooldown(
                    cooldown, streak, cap=breaker.max_seconds)
                log.warning(
                    "plan=%s transient failure streak=%d -> cooldown %ds",
                    plan.key, streak, cooldown,
                )
                did_cool_atomic = True
        if verdict.should_cool and not did_cool_atomic:
            await self.slots.cool_down(plan.key, cooldown, verdict.outcome.value)
        if verdict.outcome in (Outcome.QUOTA_EXHAUSTED, Outcome.PLAN_DEAD, Outcome.AUTH) and ctx.get("session"):
            # Do not strand the session on dead capacity; let it re-lease.
            await self.slots.drop_lease(ctx["session"])

    async def _absorb_limit_headers(self, plan_key: str, kwargs: dict) -> None:
        """Prefer a provider's own remaining-quota headers over our estimate."""
        plan = self.registry.plans.get(plan_key)
        if not plan or not any(q.headers for q in plan.quotas):
            return
        headers = {}
        for src in ("response_headers", "_response_headers"):
            h = kwargs.get(src) or (kwargs.get("additional_args") or {}).get(src)
            if isinstance(h, dict):
                headers.update({k.lower(): v for k, v in h.items()})
        if not headers:
            return
        # Each window may name its own headers, so a provider that reports both
        # a 5-hour and a weekly remaining count populates both.
        for q in plan.quotas:
            if not q.headers:
                continue
            remaining = _as_float(headers.get((q.headers.get("remaining") or "").lower()))
            reset = _as_float(headers.get((q.headers.get("reset") or "").lower()))
            limit = _as_float(headers.get((q.headers.get("limit") or "").lower()))
            if remaining is not None or reset is not None or limit is not None:
                await self.ledger.note_reported(plan.key, remaining, reset,
                                                window=q.label, limit=limit)


def _reason_with_group(pick) -> str:
    """The bracketed reason in the gateway log line.

    The base string is the picker's own cap_reason ("configured",
    "quota spent", "paced 2 of 4", ...). When the lane walked a group to
    reach this pick, the group's gid (and strategy, when distinct from the
    plain per-lane shape) is appended INSIDE the brackets, so a script that
    grep'd the previous `\[(\w+)\]` regex still extracts the FIRST token —
    the original cap_reason — and now also gets the routing context.

    A flat lane (no group walk) produces the same bracketed reason it always
    did, so the smoke.py grep pattern `lane=<lane> ->` and the existing
    per-row test parsing keep working bit-for-bit.
    """
    reason = pick.cap_reason or "configured"
    group = pick.picked_group
    if group is None:
        return reason
    if isinstance(group, Group) and group.strategy:
        return f"{reason} group={group.gid},strategy={group.strategy}"
    return f"{reason} group={group.gid}"


def _carries_tool_results(messages: Any) -> bool:
    """Is this a mid-tool-loop follow-up, in EITHER wire format?

    The two protocols disagree about where a tool result lives, and checking
    only one of them is a silent failure rather than a loud one: an
    Anthropic-shaped follow-up looked like a fresh request, so it was free to
    spill to another plan, whose bridge had never minted those ids and answered
    400 "tool results must all belong to exactly one live mcp_bridge session".

      OpenAI:     {"role": "tool", "tool_call_id": "..."}
      Anthropic:  {"role": "user", "content": [{"type": "tool_result",
                                                "tool_use_id": "..."}]}
    """
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" and message.get("tool_call_id"):
            return True
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
    return False


def _payload_of(response_obj: Any) -> Any:
    """Best-effort dict view of a response, whatever shape LiteLLM handed us."""
    if isinstance(response_obj, dict):
        return response_obj
    for attr in ("model_dump", "dict", "json"):
        fn = getattr(response_obj, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, (dict, str)):
                    return out
            except Exception:
                continue
    hidden = getattr(response_obj, "_hidden_params", None)
    return hidden if isinstance(hidden, dict) else None


def _error_body(exc: Exception) -> Any:
    """The raw error body, where the vendor business code actually lives."""
    for attr in ("body", "response_body", "error"):
        val = getattr(exc, attr, None)
        if isinstance(val, (dict, str)):
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("json", "text"):
            fn = getattr(resp, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    continue
            elif isinstance(fn, str):
                return fn
    # Last resort: the message itself often carries "(1008)".
    return str(getattr(exc, "message", None) or exc)


def _retry_after(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or getattr(exc, "headers", None) or {}
    try:
        headers = {k.lower(): v for k, v in dict(headers).items()}
    except Exception:
        return None
    return _as_float(headers.get("retry-after"))


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _prompt_completion_tokens(
    response_obj: Any, usage_override: dict | None,
) -> tuple[int, int]:
    """Pull prompt/completion token counts out of whatever shape arrived.

    Anthropic uses ``input_tokens`` / ``output_tokens`` on the ``usage`` block
    of a buffered /v1/messages response (and the same keys appear inside
    message_start/message_delta SSE frames). OpenAI uses
    ``prompt_tokens`` / ``completion_tokens``. LiteLLM's chat-completion path
    exposes both OpenAI keys. The buffered Anthropic response does not.

    The streamed path passes its collected usage dict explicitly, so it does
    not have to look up attributes on a finished object. The buffered path
    reads ``response_obj.usage`` -- which LiteLLM serves as a dict or a pydantic
    Usage object depending on the provider -- and tries both key names per
    field. Cache tokens (``cache_creation_input_tokens``,
    ``cache_read_input_tokens``) are folded into ``prompt_tokens`` here so the
    ledger, which only knows the OpenAI shape, sees the full input cost; the
    gateway CLI bridge and bridge wrappers already do the same conversion
    when folding to OpenAI shape, and double-counting a cache read as a fresh
    input is the original Anthropic-flavoured bug this preserves.
    """
    src: Any = usage_override
    if src is None and response_obj is not None:
        if isinstance(response_obj, dict):
            src = response_obj.get("usage") or {}
        else:
            src = getattr(response_obj, "usage", None) or {}
    if isinstance(src, dict):
        prompt = int(src.get("prompt_tokens") or src.get("input_tokens") or 0)
        completion = int(src.get("completion_tokens") or src.get("output_tokens") or 0)
        cache_read = int(src.get("cache_read_input_tokens") or 0)
        cache_creation = int(src.get("cache_creation_input_tokens") or 0)
    else:
        prompt = int(
            getattr(src, "prompt_tokens", None)
            or getattr(src, "input_tokens", None)
            or 0
        )
        completion = int(
            getattr(src, "completion_tokens", None)
            or getattr(src, "output_tokens", None)
            or 0
        )
        cache_read = int(getattr(src, "cache_read_input_tokens", None) or 0)
        cache_creation = int(getattr(src, "cache_creation_input_tokens", None) or 0)
    return prompt + cache_read + cache_creation, completion


def _collect_anthropic_event_usage(event: dict, collected: dict) -> None:
    """Take the totals from a single Anthropic SSE event.

    ``message_start`` carries the initial input-token count on ``message.usage``;
    a subsequent ``message_delta`` carries the running output and cache totals
    directly on the top-level ``usage``. Both are *totals*, not deltas -- a
    later ``message_delta`` is the latest authoritative reading, not an
    addition. We overwrite the running figure so a fragmented stream (or a
    provider that emits a ``message_delta`` before any tokens are counted)
    cannot inflate the total.

    The function is intentionally narrow: ``message_start`` only writes the
    input/cache fields it actually carries, ``message_delta`` only the output/
    cache fields it carries, and any other event type is ignored. Other call
    shapes (text completions, embeddings, etc.) reach a different hook path.
    """
    typ = event.get("type")
    if typ == "message_start":
        msg = event.get("message") or {}
        usage = msg.get("usage") if isinstance(msg, dict) else None
        if isinstance(usage, dict):
            for key in ("input_tokens", "cache_creation_input_tokens",
                        "cache_read_input_tokens"):
                value = usage.get(key)
                if value is not None:
                    try:
                        collected[key] = int(value)
                    except (TypeError, ValueError):
                        pass
            # Some providers emit output_tokens in message_start (often the
            # ``1`` placeholder Anthropic writes for "at least one"). Preserve
            # it; later message_delta frames overwrite with the real count.
            value = usage.get("output_tokens")
            if value is not None:
                try:
                    collected["output_tokens"] = int(value)
                except (TypeError, ValueError):
                    pass
        return
    if typ == "message_delta":
        usage = event.get("usage")
        if isinstance(usage, dict):
            for key in ("output_tokens", "cache_creation_input_tokens",
                        "cache_read_input_tokens"):
                value = usage.get(key)
                if value is not None:
                    try:
                        collected[key] = int(value)
                    except (TypeError, ValueError):
                        pass


switchyard_handler = SwitchyardHandler()
