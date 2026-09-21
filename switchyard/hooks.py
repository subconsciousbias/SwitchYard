"""The LiteLLM proxy plugin. Register it in litellm-config.yaml:

    litellm_settings:
      callbacks: ["switchyard.hooks.switchyard_handler"]

Deliberately built on LiteLLM's *stable* extension points (CustomLogger hooks)
rather than the beta custom-routing-strategy API, so a LiteLLM upgrade cannot
silently change how capacity is allocated. The pre-call hook rewrites a lane
name into a concrete deployment; from LiteLLM's point of view every request
names exactly one provider and it never has to make a routing decision.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import time
from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from redis.asyncio import Redis

from . import models
from .classify import Outcome, classify, inspect_success_payload
from .picker import LaneSaturated, Picker
from .policy import CapacityPolicy
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


class SwitchyardHandler(CustomLogger):
    def __init__(self) -> None:
        self.registry = models.load()
        self._redis: Redis | None = None
        self._slots: SlotTable | None = None
        self._picker: Picker | None = None
        self._ledger: Ledger | None = None
        self._policy: CapacityPolicy | None = None
        # request_id -> heartbeat task, so a live request keeps its slot and a
        # dead one loses it within inflight_max_age_seconds.
        self._beats: dict[str, asyncio.Task] = {}
        log.info(
            "%d plans, lanes=%s, tool-capable=%d",
            len(self.registry.plans), ",".join(self.registry.lanes),
            sum(1 for p in self.registry.plans.values() if p.can_use_tools),
        )

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

    def reload(self) -> None:
        """Pick up edits to plans.yaml without restarting the proxy.

        Learned concurrency and pacing state live in Redis, so toggling pacing
        or changing an allowance takes effect on the next request without
        losing what the learner already knows.
        """
        self.registry = models.load()
        self._policy = CapacityPolicy(self.redis, self.registry.settings, self.ledger)
        self._picker = Picker(self.registry, self.slots, self._policy)

    # -- inbound: choose a provider ---------------------------------------
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        lane = data.get("model")
        if lane not in self.registry.lanes:
            return data  # a caller naming a deployment directly bypasses us

        key_hash = None
        token = getattr(user_api_key_dict, "api_key", None)
        if token:
            key_hash = hashlib.sha256(str(token).encode()).hexdigest()[:8]
        session = derive_session(data, key_hash)

        # A request carrying tool definitions cannot go to a plan marked
        # supports_tools: false; see Plan.can_use_tools.
        needs_tools = bool(data.get("tools"))

        # A request carrying tool *results* is mid-loop: its tool_call_ids were
        # minted by one plan's bridge, so it must go back to that same plan
        # rather than spill to a peer that cannot read them. See Picker.pick.
        pinned = any(m.get("role") == "tool" and m.get("tool_call_id")
                     for m in (data.get("messages") or []) if isinstance(m, dict))

        try:
            pick = await self.picker.pick(lane, session, needs_tools, pinned)
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
        }
        log.info(
            "lane=%s -> %s [%s]%s%s%s",
            lane, pick.model.ref, pick.cap_reason,
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

        Returns the plan to attribute usage to, or None if it cannot be resolved.
        """
        picked = self.registry.plans.get(ctx["plan"])
        served_name = kwargs.get("model")
        if not served_name:
            return picked
        served = self.registry.model_for_deployment(str(served_name))
        if served is None or served.ref == ctx.get("model"):
            return picked
        actual = self.registry.plan_of(served)
        log.warning(
            "request was picked for %s but served by %s — LiteLLM moved it after "
            "the pre-call hook, so it bypassed the routing rules. Booking usage "
            "to %s, the plan whose quota it actually spent.",
            ctx.get("model"), served.ref, actual.key,
        )
        return actual

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        ctx = self._ctx(kwargs)
        if not ctx:
            return
        self._stop_heartbeat(ctx["request_id"])
        # Always release what we claimed, whoever ended up serving it.
        await self.picker.release(ctx["plan"], ctx["request_id"], ctx["model"])
        plan = self._check_served_deployment(ctx, kwargs)
        if not plan:
            return

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
            await self.ledger.record(plan, failed=True)
            await self._apply_verdict(plan, verdict, ctx)
            return

        usage = getattr(response_obj, "usage", None) or {}
        get = (lambda k: usage.get(k, 0)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, 0) or 0)
        prompt_tokens = int(get("prompt_tokens") or 0)
        completion_tokens = int(get("completion_tokens") or 0)
        # Only metered providers have a per-request cost. On a subscription the
        # fee is fixed and the marginal cost of a request is zero; recording
        # LiteLLM's notional price would inflate month_cost and corrupt the
        # effective $/Mtok figure, which is the number used to judge whether the
        # subscription is worth renewing.
        cost = float(kwargs.get("response_cost") or 0.0) if plan.metered else 0.0
        await self.ledger.record(
            plan, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, cost=cost,
        )

        # Per-slot throughput drives the pacer: one busy slot delivered this
        # many units in this many seconds.
        try:
            seconds = (end_time - start_time).total_seconds()
        except (TypeError, AttributeError):
            seconds = max(0.0, time.time() - float(ctx.get("claimed_at") or time.time()))
        units = cost if plan.quota.kind == "dollars" else prompt_tokens + completion_tokens
        await self.policy.pacer.note_throughput(plan, units, seconds)

        await self._absorb_limit_headers(plan.key, kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        ctx = self._ctx(kwargs)
        if not ctx:
            return
        self._stop_heartbeat(ctx["request_id"])
        await self.picker.release(ctx["plan"], ctx["request_id"], ctx["model"])
        plan = self.registry.plans.get(ctx["plan"])
        if plan:
            await self.ledger.record(plan, failed=True)
        await self._handle_failure(ctx, kwargs.get("exception") or kwargs.get("original_exception"))

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response, **_: Any
    ):
        """Tell the caller which plan actually served it.

        The body's `model` echoes what was asked for — usually a lane name like
        `judge` — so a caller doing its own token or cost accounting cannot see
        which subscription the tokens came out of. LiteLLM puts the answer in
        response *headers* (x-litellm-model-group and friends), which is easy to
        miss and lost by any client that only keeps the JSON. So it goes in the
        body too, under one namespaced key that a strict client will ignore.
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
        return response

    async def async_post_call_failure_hook(
        self, request_data: dict, original_exception: Exception, user_api_key_dict: UserAPIKeyAuth, **_: Any
    ) -> None:
        meta = (request_data or {}).get("metadata") or {}
        ctx = meta.get(META_KEY)
        if isinstance(ctx, dict):
            self._stop_heartbeat(ctx["request_id"])
            await self.picker.release(ctx["plan"], ctx["request_id"], ctx["model"])
            await self._handle_failure(ctx, original_exception)

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

        if verdict.should_cool:
            await self.slots.cool_down(plan.key, verdict.cooldown_seconds, verdict.outcome.value)
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


switchyard_handler = SwitchyardHandler()
