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
import copy
import hashlib
import json
import logging
import os
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from redis.asyncio import Redis

from . import caller_env, models
from .classify import Outcome, Verdict, classify, escalated_cooldown, extract_no_text_tokens, inspect_success_payload
from .models import Group
from .picker import LaneSaturated, Picker
from .policy import CapacityPolicy
from .reasoning import ReasoningSplitter
from .session import derive as derive_session, identify as identify_cli
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

# CLIs whose first request gets a system message injected, telling the model
# that its native tools are unavailable in this environment. The set is
# permissive on purpose — adding a new CLI to the gateway without listing it
# here would mean resumed sessions keep reasoning in their local tools and
# fail when those calls never reach the user's workspace. `codex` is included
# even though its native tools do not touch the filesystem today, so a future
# CLI tool there (e.g. a `web_search` bridge) inherits the same gate.
INJECT_CLIS = frozenset({"opencode", "claude-code", "codex", "gemini"})

# Short system note appended to the first turn of a recognised CLI's session.
# Deliberately a system message (not a user one) so it sits in front of every
# subsequent turn and survives the bridge's leading-system-message preservation
# in cli_bridge.fold_system / mcp_bridge.flatten_with_tool_history. ~60 tokens.
INJECTION_NOTE = (
    "You are running through SwitchYard, an LLM proxy. Your native tools "
    "(file edits, shell, etc.) are stripped and unavailable in this "
    "environment — they fail silently when called. Use the MCP tools exposed "
    "by this session to interact with the user's workspace; native CLI "
    "commands will not reach the user's machine."
)

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

# Claude-backed plans are named with a `claude-` prefix by configuration
# convention. The prefix is what tells us a plan routes through an Anthropic
# sidecar (and therefore accepts the Claude-side request params -- `thinking`,
# `output_config`, etc.) rather than through an OpenAI / Codex sidecar (which
# expects the OpenAI/Responses shape: `reasoning`, `tool_choice`, etc.). The
# spillover path uses this distinction to drop `thinking` from a request that
# is being re-routed onto an OpenAI pick; without the strip the OpenAI
# sidecar's strict request validation rejects `thinking` as an unknown body
# field and 404s the request.
_CLAUDE_PLAN_PREFIX = "claude-"


def is_claude_plan(plan) -> bool:
    """A Claude-backed plan by the configured plan-key convention.

    Plans whose key starts with `claude-` are reached through the Claude CLI
    sidecar (oauth- or CLI-backed Anthropic). They accept Claude-shape
    request params (`thinking`, `output_config`, ...); an OpenAI-shaped
    spillover that retains those params 404s at the sidecar. The hook uses
    this predicate to decide whether to strip Claude-only fields after the
    pick lands on a non-Claude deployment.
    """
    key = getattr(plan, "key", "") or ""
    return key.startswith(_CLAUDE_PLAN_PREFIX)


# -------------------------------------------- what a CLI sidecar must receive ---
# Two things a CLI-backed plan needs never arrived through LiteLLM (issue #264,
# measured against the pinned LiteLLM 1.101.0 with a loopback capture):
#
#   * `metadata` is not forwarded to the provider at all, so the caller_env
#     stamp above never reached the sidecar;
#   * a reasoning effort is forwarded to claude/opencode sidecars and then
#     ignored, and for gpt-5.4+ model names (the codex plans) function tools
#     plus an effort make LiteLLM switch to its Responses-API bridge --
#     POST {api_base}/responses, which no sidecar serves: every OpenCode tool
#     turn (it always sends reasoning_effort) and every Claude Code tool turn
#     with output_config.effort 404'd on codex.
#
# `extra_body` IS forwarded verbatim. So for a CLI-backed plan the effort is
# taken out of the parameters LiteLLM interprets and carried, with the
# caller_env stamp, in `extra_body.switchyard`; the bridges map the effort to
# the CLI's own switch (claude --effort, codex model_reasoning_effort,
# opencode --variant). API plans are untouched.
#
# Issue #292: the explicit reasoning-request / display policy travels here
# too. A Claude Code caller says `thinking: {type: enabled, display: ...}`
# (or `thinking: {type: adaptive}` on adaptive models), and the bridges map
# that to the CLI's own thinking switch -- claude's `--include-partial-messages`
# + the agent's reasoning event channel, codex's `--reasoning-effort` override,
# opencode's `--variant` family. The display policy (`thinking.display`)
# names whether the assistant text is rendered alongside the reasoning or
# only the reasoning is shown (`omitted`); the bridges honour that by
# suppressing the visible content when "omitted" is requested, while
# preserving the reasoning itself on the payload's `reasoning_content` so
# the gateway/LiteLLM can adapt it for chat / Messages / Responses callers.
def carry_to_cli_sidecar(data: dict, caller_env_stamp: dict | None) -> None:
    effort = data.pop("reasoning_effort", None)
    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        effort = effort or reasoning.get("effort")
        data.pop("reasoning", None)
    output_config = data.get("output_config")
    if isinstance(output_config, dict) and "effort" in output_config:
        effort = effort or output_config.get("effort")
        rest = {k: v for k, v in output_config.items() if k != "effort"}
        if rest:
            data["output_config"] = rest
        else:
            data.pop("output_config", None)
    # Issue #292: lift `thinking` (Anthropic shape) onto the carrier.
    # `thinking.type` is the enable signal (enabled/adaptive/disabled);
    # `thinking.display` is the visibility policy (summarized/full/omitted).
    # Neither is interpreted by LiteLLM on a CLI plan, but both go to the
    # sidecar in `extra_body.switchyard.thinking` so each CLI's argv builder
    # can switch the model's reasoning on AND pick the right rendering.
    # An explicit `thinking.type == "disabled"` is the user's "off, do not
    # lift" choice; we drop the entire policy in that case so the sidecar's
    # `thinking` key never carries a disabled marker it would have to
    # second-guess.
    #
    # The whitelist also carries `budget_tokens` (Anthropic's thinking-budget
    # spelling for adaptive models) and `enabled` (a future Anthropic field).
    # Today the sidecar reads only `type` and `display` -- the other two ride
    # the carrier anyway so a future reader does not need a gateway-side
    # change to act on them.
    thinking: dict | None = None
    raw_thinking = data.pop("thinking", None) if isinstance(data.get("thinking"), dict) else None
    if isinstance(raw_thinking, dict) and raw_thinking.get("type") != "disabled":
        thinking = {k: v for k, v in raw_thinking.items()
                    if k in ("type", "display", "budget_tokens", "enabled")}
    if thinking is None:
        # OpenAI shape spelled `reasoning` -- we already popped one above if
        # it carried `effort`. A second one with `display` only is still
        # possible; lift those keys too so the sidecar has the full picture.
        prior = data.get("reasoning")
        if isinstance(prior, dict) and "display" in prior:
            thinking = {"display": prior["display"]}
    extra = dict(data.get("extra_body") or {})
    carried = dict(extra.get("switchyard") or {})
    if effort:
        carried["reasoning_effort"] = effort
    if thinking:
        # Normalise: a caller that only sent `display` keeps just that key.
        carried["thinking"] = thinking
    if caller_env_stamp:
        carried["caller_env"] = caller_env_stamp
    if carried:
        extra["switchyard"] = carried
        data["extra_body"] = extra


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
        # A reload can land before the first request has touched the lazy
        # `slots` property (the watcher starts in __init__). Building the
        # table here keeps the new picker from being handed slots=None — once
        # `_picker` is set the lazy property never runs again, and every
        # request 500'd on `get_lease` (issue #196).
        fresh_ttl = fresh.settings.inflight_max_age_seconds
        if self._slots is None or self._slots.inflight_max_age != fresh_ttl:
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

        # A request whose `x-switchyard-cli` header names a known harness has
        # its native file/shell tools filtered out before the picker runs.
        # Those tools reach the sidecar's container rather than the caller's
        # workspace, so a successful "edit" call there silently does the wrong
        # thing — worse than failing loudly. The list comes from
        # Settings.cli_tool_block (see switchyard/models.py), keyed by the
        # lowercased header value. The filtered list is only written back when
        # the request actually had a `tools` key AND at least one entry was
        # dropped, so a tool-less request is not promoted into `[]` (which
        # would still register as needing tools and the picker would 429 a
        # lane whose every member is `supports_tools: false`).
        cli = identify_cli(data)
        if cli:
            tools = data.get("tools")
            if isinstance(tools, list) and tools:
                blocked = {n.lower() for n in
                           self.registry.settings.cli_tool_block.get(cli, ())}
                if blocked:
                    kept, dropped = [], []
                    for tool in tools:
                        name = _tool_name(tool)
                        if name and name.lower() in blocked:
                            dropped.append(name)
                        else:
                            kept.append(tool)
                    if dropped:
                        log.info(
                            "cli=%s tool_block: dropped %d/%d (%s)",
                            cli, len(dropped), len(tools),
                            ",".join(sorted(set(dropped))),
                        )
                        data["tools"] = kept

        # A request carrying tool definitions cannot go to a plan marked
        # supports_tools: false; see Plan.can_use_tools. Derived AFTER the
        # blocklist filter above so an emptied request routes like a plain
        # request — the picker skips supports_tools filtering on a request
        # whose `tools` was emptied here.
        needs_tools = bool(data.get("tools"))

        # An image-bearing request cannot land on a model that cannot serve
        # images, under a lane whose image_routing is `always`. The check
        # has to inspect the message list directly because the presence of
        # an image is data-dependent — providers differ on what wire shape
        # they accept, and no top-level request key flags it. See
        # Picker._members / Picker._affinity for the matching gates.
        needs_images = _has_image_block(data.get("messages"))

        # A request carrying tool *results* is mid-loop. While its session
        # lease is alive it returns to the plan that minted its tool_call_ids
        # -- the prompt cache and the loop's quota are both there. Past the
        # lease TTL the pin is gone and it places fresh, which is safe because
        # any bridge rebuilds a lost session from the request itself. See
        # Picker.pick.
        pinned = _carries_tool_results(data.get("messages"))

        try:
            # Thread the request body into the picker so the
            # `enforce_context_window` gate (issue #104) can consult each
            # peer's `context_window` against the inbound `messages`. Without
            # `data=data` here, `_estimate_input_tokens(None)` short-circuits
            # to 0 and the gate never fires from `/v1/messages`; the
            # picker-side gate is opt-in, so non-opted operators are
            # unaffected.
            pick = (await self.picker.pick_direct(direct, session, data=data) if direct
                    else await self.picker.pick(
                        lane, session, needs_tools, pinned, None, needs_images,
                        data=data))
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
        # Claude-side request params must not survive a spillover onto a
        # non-Claude deployment. A Claude Code (Anthropic-shape) request
        # spilling onto the OpenAI seat carries `thinking`: the OpenAI
        # sidecar's request validation rejects it as an unknown body field
        # and 404s the call. Strip the field here, on the way out, only for
        # non-Claude picks; a Claude pick keeps it intact because that is
        # the sidecar's own contract. The strip is keyed off the configured
        # `claude-` plan-key convention so a future Claude-backed plan
        # inherits the predicate without any code change here.
        if not is_claude_plan(pick.plan):
            data.pop("thinking", None)
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
            "needs_images": needs_images,
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
        # The caller-environment stamp is best-effort and never fails the
        # request. Belt-and-braces: hooks.py proves metadata lives on the
        # request body, but nothing in-tree proves LiteLLM forwards it to
        # the sidecar (the gateway->sidecar transport is not yet verified
        # in production). The mcp_bridge / cli_bridge re-resolve the env
        # from the request itself on their side -- the stamp here is just
        # the easy path. Wrapped in try/except so a parse hiccup cannot
        # poison every request that hits this hook.
        try:
            ce = caller_env.resolve(data, self.registry.settings.caller_environment)
            meta[META_KEY]["caller_env"] = {
                "cwd": ce.cwd,
                "platform": ce.platform,
                "shell": ce.shell,
                "source": ce.source,
                "git": ce.git,
            }
        except Exception:                       # never fail the request over a label
            log.debug("caller_env resolution skipped for this request", exc_info=True)
        if pick.plan.is_cli_backed:
            carry_to_cli_sidecar(data, meta[META_KEY].get("caller_env"))
        log.info(
            "lane=%s -> %s [%s]%s%s%s%s%s",
            lane, pick.model.ref, _reason_with_group(pick),
            " tools" if needs_tools else "",
            " images" if needs_images else "",
            " (sticky)" if pick.sticky else "",
            f" drain=({pick.drain_reason})" if pick.drain_reason else "",
            f" skipped={','.join(pick.considered)}" if pick.considered else "",
        )

        # First-turn system injection for known CLI harnesses. The note tells
        # the model that its native tools are stripped in this environment so
        # the resumed session — whose earlier turns reasoned against the
        # harness's local tool surface — stops calling tools that would never
        # reach the user's workspace. Inserted AFTER the leading run of system
        # messages (the while-loop below) so a caller-supplied system prompt
        # still wins precedence: the note sits between the caller's system
        # block and the rest of the conversation.
        #
        # NOT gated on `pick.sticky`. In the production incident the entire
        # stack restarted, Redis died with it, the resumed pick is non-sticky,
        # and a sticky gate would never fire — which is precisely the bug.
        if cli and session and cli in INJECT_CLIS \
                and not await self._already_injected(session):
            messages = data.get("messages")
            if isinstance(messages, list):
                insert_at = 0
                while insert_at < len(messages):
                    msg = messages[insert_at]
                    if isinstance(msg, dict) and msg.get("role") == "system":
                        insert_at += 1
                        continue
                    break
                messages.insert(insert_at, {"role": "system",
                                            "content": INJECTION_NOTE})
                log.info(
                    "cli=%s session=%s injection: appended system note at index %d",
                    cli, session, insert_at,
                )
                await self._mark_injected(session)
        return data

    # -- keeping a live claim alive ----------------------------------------
    async def _already_injected(self, session: str) -> bool:
        """Has this session already had the first-turn system note appended?

        Tracked in the slot table so it survives a stack restart along with
        the session lease: a resumed CLI session whose lease is still alive
        must NOT be re-injected (the note is already in its message log), and
        one whose lease expired must be (its lease renewal is what re-runs
        the hook, and `mark_injected` writes a fresh TTL).
        """
        return await self.slots.injected(session)

    async def _mark_injected(self, session: str) -> None:
        """Stamp the slot table so the next turn skips re-injection.

        TTL matches the lease so the two expire together; drop_lease clears
        both keys at once, which is how a hard plan rejection re-enables
        injection on the next pick.
        """
        await self.slots.mark_injected(
            session, self.registry.settings.lease_ttl_seconds)

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
        lease, the mid-tool-loop pin) and is worth seeing rather than absorbing.

        Returns (plan, served_ref): the plan to attribute usage to (None when
        it cannot be resolved) and the ref of the deployment that answered, so
        model-scoped economics land on the model that actually spent them.

        ``kwargs["model"]`` can arrive in two shapes:

          * the deployment string ``sy.{plan}.{model}`` -- pre-call
            ``data["model"]`` for a direct-pick caller (hooks.py:398) and
            what ``model_for_deployment`` resolves against;
          * the router id ``_hidden_params["model_id"]`` -- the value the
            proxy stamps on every response and that ``_call_facts`` writes
            through (issue #184). SwitchYard's ``gen_litellm.py`` now
            stamps ``model_info["id"]`` with a deterministic
            ``Model.router_id`` (``sy.{plan}.{model}.id``) so this lookup
            resolves back to a Model; a stale hash from before the stamp
            still falls through to the picked model.
        """
        picked = self.registry.plans.get(ctx["plan"])
        served_name = kwargs.get("model")
        if not served_name:
            return picked, ctx.get("model")
        served_str = str(served_name)
        served = (self.registry.model_for_deployment(served_str)
                  or self.registry.model_for_router_id(served_str))
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

        ``kwargs`` is threaded into ``_finish_failure`` so the failure side
        can see the per-hop state ``run_async_fallback`` stamps on it
        (``kwargs["litellm_params"]["metadata"]["attempted_fallbacks"]`` for
        the per-hop dedupe key, ``kwargs["model"]`` for resolving which
        deployment actually produced the error on a multi-hop walk).
        """
        ctx = self._ctx(kwargs)
        if not ctx:
            return
        if ctx.get("call_type") in UNLOGGED_CALL_TYPES:
            return
        await self._finish_failure(
            ctx,
            kwargs.get("exception") or kwargs.get("original_exception"),
            kwargs=kwargs,
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
            model=served_ref, session=ctx.get("session"),
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

    async def _finish_failure(
        self, ctx: dict, exception: Exception | None,
        kwargs: dict | None = None,
    ) -> bool:
        """Common failure accounting. Idempotent via the per-hop dedupe set.

        Per-hop dedupe keys (``hop``) come from the router's
        ``attempted_fallbacks`` stamp on ``kwargs["litellm_params"]
        ["metadata"]``. The router writes this stamp in two places:

          * in-place on the metadata at walk entry, ``0`` (the entry
            stamp) -- ``router.py::async_function_with_fallbacks:7582``;
          * per-hop on a rebinding copy of the metadata,
            ``fallback_depth`` (1 / 2 / …) -- ``router_utils/
            fallback_event_handlers.run_async_fallback``.

        The rebinding copy is what ``async_log_failure_event`` sees
        on each hop; the proxy-side ``request_data`` for
        ``async_post_call_failure_hook`` keeps the in-place entry
        stamp (``0``) because the per-hop rebinds are intentional
        detached copies. ``hop = 0`` and ``hop >= 1`` therefore
        correspond to real production shapes. ``hop = -1`` is the
        no-router-stamp sentinel for callers that pass ``kwargs``
        without ``attempted_fallbacks`` (older-litellm shapes,
        internal callers with ``kwargs=None``, the proxy-side
        ``async_post_call_failure_hook`` in builds without the
        in-place entry stamp).

        Dedupe has two layers working together:

          * ``_TERMINATED`` -- the cross-hook idempotency marker.
            Non-CONTEXT primary-like arms (``hop <= 0``: the router's
            entry stamp on hop 0, or the no-router-metadata sentinel
            ``-1``) set it to ``"failure"`` to suppress a same-attempt
            late success; CONTEXT verdicts and non-CONTEXT mid-walk
            arms (``hop >= 1``) leave it unset so a sibling success
            on a later hop can still book. The success side sets it
            to ``"success"``.

          * ``_ctx_fail_booked_hops`` (set of int) -- per-hop dedupe.
            ``run_async_fallback`` catches any ``Exception`` per hop,
            so a context-window-fallbacks walk that exhausts every
            sibling and bubbles the last error here fires
            ``async_log_failure_event`` once per attempt. Each attempt
            carries the router's stamp so every hop gets its own
            slot (``{0, 1, 2, …}``). The OpenAI-shape double-fire
            (``async_log`` + ``async_post_call`` for one attempt)
            dedupes via the same entry stamp (``0``): the proxy-side
            ``request_data`` carries the in-place ``0``, the
            ``async_log`` hop also stamps ``0``, the set collapses
            them. ``/v1/messages`` failures (only
            ``async_post_call`` fires) take whatever slot the proxy
            stamped.

        The per-hop de-dupe has no special branches: every shape --
        int stamp, sentinel stamp, proxy in-place stamp -- slots into
        the same set.

        CONTEXT-window failures are the documented exception to the
        claim-marker rule for the mid-walk extension: a non-CONTEXT
        failure on hop 1 is no evidence the walk is done -- the
        router can still try hop 2 on another sibling, succeed, and
        expect ``_finish_success`` to book the served plan. Claiming
        ``_TERMINATED`` here would re-open the original #186 ledger
        hole: hop 2 succeeds → ``_finish_success`` first-line checks
        the marker → returns False → served plan's tokens never
        booked. Non-CONTEXT mid-walk therefore mirrors CONTEXT's
        no-claim behavior on the marker write too (passes
        ``skip_marker=True`` to ``_apply_verdict``); only the
        primary-like arms (hop <= 0) write the marker. A late
        verdict from a later hop is then free to apply its own
        cooldown / ladder bump rather than getting silently
        suppressed by the earlier hop's marker.

        For a streamed ``/v1/messages`` that errored mid-stream, the
        iterator stashes any partial usage it managed to parse on ctx under
        ``_stream_collected_usage``. We book those tokens as part of the
        failure record so a partial stream is not silently dropped on the
        ledger -- the bounded loss the reviewer's design accepts -- and we
        do NOT reset the transient-failure streak (this is the failure
        path, not success). The verdict and cooldown ladder still run as
        usual.
        """
        if ctx.get(_TERMINATED):
            return False

        # Detect the per-hop key from kwargs. ``hop`` is the router's
        # ``attempted_fallbacks`` stamp when present, or ``-1`` when
        # the caller passed kwargs without that stamp (no-router-
        # metadata path). Every shape -- real production int stamps
        # (``0`` for entry, ``>= 1`` for siblings) and the sentinel --
        # slots into one ``_ctx_fail_booked_hops`` set below.
        hop_meta = (
            (kwargs.get("litellm_params") or {}).get("metadata")
            if kwargs is not None else None
        )
        if not isinstance(hop_meta, dict):
            hop_meta = (
                (kwargs.get("metadata") if kwargs is not None else None)
                or {}
            )
        if not isinstance(hop_meta, dict):
            hop_meta = {}
        attempted = hop_meta.get("attempted_fallbacks")
        hop: int = attempted if isinstance(attempted, int) else -1
        booked_hops: set[int] = ctx.setdefault("_ctx_fail_booked_hops", set())

        if hop == -1:
            # No-router-metadata sentinel. kwargs arrived without a
            # router ``attempted_fallbacks`` stamp -- internal callers
            # with ``kwargs=None``, legacy fixtures, the proxy side of
            # ``async_post_call_failure_hook`` in builds without the
            # in-place entry stamp. The router always stamps ``0`` at
            # walk entry in both litellm 1.101.0 and 1.102.1, so
            # production router traffic never reaches this branch.
            # Book into the dedupe set; the set dedupes both shapes
            # uniformly, so a same-shape second call short-circuits
            # via ``-1 in booked_hops`` rather than a per-shape flag.
            if -1 in booked_hops:
                return False
            booked_hops.add(-1)
        else:
            # Per-hop dedupe for multi-hop walks. Each
            # ``async_log_failure_event`` carries a distinct
            # ``attempted_fallbacks`` stamp the router wrote per hop,
            # so a multi-hop context-window-fallbacks walk books every
            # hop exactly once.
            if hop in booked_hops:
                return False
            booked_hops.add(hop)

        # INTERNAL short-circuit. A bare exception -- Redis
        # ConnectionError, asyncio CancelledError, our own bookkeeping
        # bug -- has no provider-shaped signal (no SDK APIError
        # subclass, no int-coercible ``status_code``). Sending those
        # through ``classify`` would land them in the string-and-HTTP
        # classifier's TRANSIENT branch and cool the plan on a
        # problem we caused. ``_provider_error`` gates that: a
        # non-provider exception is logged once, the dedupe set has
        # already claimed this hop so the second arm of an OpenAI
        # double-fire short-circuits above, and the path below
        # applies ``Verdict(Outcome.INTERNAL, 0, ...)``.
        if exception is not None and not _provider_error(exception):
            plan_key = ctx.get("plan", "?")
            rid = ctx.get("request_id", "?")
            log.exception(
                "switchyard internal fault on plan=%s request_id=%s: %r",
                plan_key, rid, exception,
            )
            # Claim the terminal marker here for non-provider
            # exceptions too. The INTERNAL verdict's
            # ``is_our_fault=True`` makes ``_apply_verdict`` short-
            # circuit at the top guard (no cooldown, no streak bump);
            # the marker still has to be set so a same-attempt late
            # success can't double-book and so the OpenAI double-fire
            # tail (``async_post_call_failure_hook`` after
            # ``async_log_failure_event``) takes the second-line check
            # and short-circuits via ``_TERMINATED`` instead of
            # double-firing the slot release + log. The
            # ``hop <= 0`` rule from cycle 3 covers this: primary-like
            # arms claim, mid-walk arms (``hop >= 1``) do not.
            if hop <= 0:
                ctx[_TERMINATED] = "failure"
            served_plan, served_ref = (
                self._resolve_served_failure_plan(ctx, kwargs)
            )
            target = served_plan or self.registry.plans.get(plan_key)
            # Symmetric with the classify-driven path below:
            # ``_stop_heartbeat`` cancels the per-request beat task
            # (idempotent -- ``self._beats.pop`` returns silently if no
            # task was registered) so the InnoDB-style touch loop
            # stops touching Redis on the next tick. Skipping the
            # call here would leak at most one interval of useless
            # touches (the beat self-stops when ``slots.touch``
            # observes the claim gone), but every other terminal arm
            # in this function stops the beat for symmetry, so this
            # branch does too.
            self._stop_heartbeat(ctx["request_id"])
            await self.picker.release(
                ctx["plan"], ctx["request_id"], ctx["model"],
            )
            if target is not None:
                await self._record_failure(
                    target, exception, ctx, served_ref=served_ref,
                )
                await self._apply_verdict(
                    target,
                    Verdict(Outcome.INTERNAL, 0, "switchyard internal fault"),
                    ctx,
                )
            return True

        # Resolve the serving deployment for THIS hop's failure, the way
        # ``_finish_success`` does for the success side. A hop's
        # ``kwargs["model"]`` is the deployment string the router picked for
        # that hop (``run_async_fallback`` rewrites ``kwargs["model"]``
        # per iteration, ``router.py:_update_kwargs_before_fallbacks``
        # stamps it on the hop's metadata), so:
        #   * hop 0 (no fallback): ``kwargs["model"]`` is the picked
        #     deployment; the ctx-equality branch returns the picked
        #     plan, and ``target`` collapses to ``ctx["plan"]``.
        #   * hop 1+ (sibling): ``kwargs["model"]`` is a sibling
        #     deployment; ``target`` is the sibling's plan -- the plan
        #     whose actual 400 / 5xx produced the row. A failure row on
        #     the wrong plan would corrupt that plan's quota / breaker
        #     streak (an AUTH hop on a sibling cools the sibling for
        #     1800s instead of the picked plan; a TRANSIENT hop bumps
        #     the sibling's ladder instead of the picked plan's).
        served_plan, served_ref = self._resolve_served_failure_plan(ctx, kwargs)
        target = served_plan or self.registry.plans.get(ctx.get("plan", ""))

        # Classify against ``target``'s provider family (the plan that
        # actually produced the error), not the picked plan's family.
        # The vendor-code table is family-keyed: a MiniMax 1008
        # insufficiency arriving as HTTP 500 on a sibling should
        # classify as QUOTA_EXHAUSTED, not TRANSIENT, and that needs
        # the sibling plan's family in the classify call. Without this
        # re-class the verdict lands on the right plan but with the
        # wrong semantics (e.g. ``note_exhaustion`` against a plan that
        # is not actually exhausted). ``family=None`` when the target
        # plan is gone (registry hot-swap); the prose-only CONTEXT
        # regex still classifies correctly without a vendor-code table.
        classify_family = target.provider_family if target else None
        verdict = (
            self._classify_failure(exception, classify_family)
            if exception is not None else None
        )
        is_context = (
            verdict is not None and verdict.outcome is Outcome.CONTEXT
        )

        # Marker claim. CONTEXT never claims (a sibling success may
        # still arrive). The mid-walk call (hop >= 1) likewise never
        # claims: ``run_async_fallback`` walks up to ``max_fallbacks``
        # siblings and catches any exception per hop, so a 5xx on hop
        # 1 is no evidence the walk is exhausted, and claiming
        # ``_TERMINATED`` would suppress hop 2's ``_finish_success``
        # and re-open the original #186 ledger hole. The primary-like
        # arms (hop <= 0 -- the router's entry stamp on hop 0, or the
        # no-router-metadata sentinel ``-1``) are safe to claim: no
        # walked fallback can still fire after them.
        if not is_context and hop <= 0:
            ctx[_TERMINATED] = "failure"
        self._stop_heartbeat(ctx["request_id"])
        # Slot release always goes against ctx["plan"]: the slot was
        # claimed on the picked plan only; sibling slots were never
        # claimed in placement, so a release against ctx["plan"] is
        # the only one with anything to release. picker.release is
        # idempotent (zrem of a missing key), so a no-op release on
        # the picked plan after a multi-hop walk that already
        # heart-beat-stopped is safe.
        await self.picker.release(
            ctx["plan"], ctx["request_id"], ctx["model"],
        )
        if target is not None:
            await self._record_failure(
                target, exception, ctx, served_ref=served_ref,
            )
            if verdict is not None:
                # ``skip_marker`` is True for CONTEXT verdicts (whose
                # sibling success may still be in flight) AND for any
                # mid-walk hop (hop >= 1, where ctx["request_id"] is
                # shared with subsequent hops and a marker would suppress
                # a legitimate later verdict). Only primary-like arms
                # (hop <= 0) write the marker.
                await self._apply_verdict(
                    target, verdict, ctx,
                    skip_marker=is_context or hop > 0,
                )
        return True

    def _resolve_served_failure_plan(
        self, ctx: dict, kwargs: dict | None,
    ):
        """Which plan produced THIS hop's failure, when the router rewrote
        the request across a context-window-fallbacks walk.

        Mirrors ``_check_served_deployment``'s logic but for the failure
        side: ``kwargs["model"]`` on a failed attempt is the deployment
        the router was trying on this hop, which may be a sibling
        deployment on a different plan when the walk moved on.

        Returns ``(plan, ref)``:

          * ``plan`` is the serving plan -- ``None`` when no plan owns the
            hop's ``kwargs["model"]`` (registry hot-swap raced, unrecognised
            deployment string). The caller falls back to ``ctx["plan"]``.
          * ``ref`` is the deployment that actually returned the error,
            so the ledger row keys the failure to the right model on the
            right plan.

        ``ctx["plan"]`` / ``ctx["model"]`` are the picked plan and model;
        on hop 0 (no fallback), ``kwargs["model"]`` equals ``ctx["model"]``
        and the ctx-equality branch returns the picked pair unchanged.
        """
        picked = self.registry.plans.get(ctx.get("plan", ""))
        if kwargs is None:
            return picked, ctx.get("model")
        served_name = kwargs.get("model")
        if not served_name:
            return picked, ctx.get("model")
        served_str = str(served_name)
        served = (self.registry.model_for_deployment(served_str)
                  or self.registry.model_for_router_id(served_str))
        if served is None or served.ref == ctx.get("model"):
            return picked, ctx.get("model")
        actual = self.registry.plan_of(served)
        log.warning(
            "failure was picked for %s but produced by %s -- the "
            "context-window-fallbacks walk moved the request across "
            "plans mid-failure. Booking the row + verdict to %s.",
            ctx.get("model"), served.ref, actual.key,
        )
        return actual, served.ref

    async def _record_failure(
        self, plan, exception: Exception | None, ctx: dict,
        served_ref: str | None = None,
    ) -> None:
        """Book the failure row onto the ledger. Pulled out of
        ``_finish_failure`` so the CONTEXT and non-CONTEXT paths share the
        same TEXT_LOST / partial-usage / bare-record logic.

        ``served_ref`` is the deployment that actually produced the error
        on this hop (the plan was resolved by ``_resolve_served_failure_plan``
        from ``kwargs["model"]``). Defaults to ``ctx["model"]`` when the
        caller did not resolve a served ref -- the picked-model fallback
        when ``kwargs`` carries no per-hop info, which keeps
        ``/v1/messages`` failures (only the post-call hook fires there)
        attributed the same way as before.
        """
        model = served_ref or ctx["model"]
        msg = (
            str(getattr(exception, "message", None) or exception)
            if exception is not None else ""
        )
        tokens = extract_no_text_tokens(msg) if msg else None
        if tokens is not None:
            prompt_tokens, completion_tokens = tokens
            await self.ledger.record(
                plan, failed=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model,
            )
            return
        partial = ctx.get("_stream_collected_usage")
        if isinstance(partial, dict) and partial:
            prompt_tokens, completion_tokens = _prompt_completion_tokens(
                None, partial,
            )
            await self.ledger.record(
                plan,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=0.0,
                failed=True,
                model=model,
            )
            return
        await self.ledger.record(plan, failed=True, model=model)

    def _classify_failure(
        self, exception: Exception, family: str | None,
    ) -> Any:
        """Classify a failure exception. Pulled out so the CONTEXT branch in
        ``_finish_failure`` can classify without going through
        ``_handle_failure`` (which re-runs the classify anyway). Returns the
        same ``Verdict`` the existing flow would produce.
        """
        status = (
            getattr(exception, "status_code", None)
            or getattr(exception, "code", None)
        )
        try:
            status = int(status) if status is not None else None
        except (TypeError, ValueError):
            status = None
        retry_after = _retry_after(exception)
        return classify(
            status,
            str(getattr(exception, "message", None) or exception),
            family=family,
            body=_error_body(exception),
            retry_after=retry_after,
            default_cooldown=self.registry.settings.default_cooldown_seconds,
        )

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
                response.switchyard = stamp
        except Exception:                 # never fail a served request over a label
            log.debug("could not stamp response with switchyard routing info",
                      exc_info=True)

        if ctx.get("call_type") in UNLOGGED_CALL_TYPES:
            # LiteLLM 1.101.0 does not populate ``data["litellm_params"]`` on
            # the /v1/messages path, so reading ``response_cost`` /
            # ``model`` / ``response_headers`` from there would always land
            # $0 cost, skip the served-deployment re-attribution, and miss
            # the provider's quota headers. Build the kwargs from
            # ``response._hidden_params`` instead -- the field provenance
            # comment on ``_call_facts`` enumerates the exact 1.101.0
            # locations.
            kwargs = _call_facts(response, data)
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
           streamed routes ALSO get iterator-side disconnect cleanup -- the
           bare pass-through below is wrapped in ``try/except BaseException``
           that schedules the marker-free detached slot release -- but their
           success accounting is still owned by ``async_log_success_event``;
           the call_type gate keeps the streamed-success finalize out of
           their way.

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
        last_chunk: Any = None
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
            last_chunk = chunk
            yield chunk

        trailing_content, trailing_reasoning = splitter.flush()
        if trailing_content or trailing_reasoning:
            log.debug("stream ended mid-tag; flushed %d content, %d reasoning chars",
                      len(trailing_content), len(trailing_reasoning))
            # The flush() tail got logged but never yielded before this
            # change -- every chunk we already yielded above has been
            # written to the response, so dropping this loses the last
            # character (or two, or seven) of the answer silently. Stamp
            # the trailing content/reasoning onto one synthetic chunk
            # cloned from the last frame so the real terminal chunk
            # keeps its `finish_reason` and the client sees the full
            # model output. The synthesis mirrors the chunk loop's own
            # "never break a stream over this" contract (just above): a
            # raise in ``_tail_chunk`` -- a frozen Pydantic chunk with
            # non-copyable internals, or any chunk shape this hook does
            # not tolerate -- falls back to the SimpleNamespace skeleton
            # path so the streaming response still completes with the
            # tail carried on a recognizable frame.
            try:
                tail = self._tail_chunk(last_chunk, trailing_content, trailing_reasoning)
            except Exception:                 # never break a stream over this
                log.debug("tail chunk clone failed; falling back to skeleton",
                          exc_info=True)
                tail = self._tail_chunk(None, trailing_content, trailing_reasoning)
            yield tail

    def _tail_chunk(self, last_chunk: Any, content: str, reasoning: str) -> Any:
        """Build a synthetic OpenAI-shaped chunk carrying the splitter's flush tail.

        Mirrors the shape of the last yielded chunk so the client receives the
        trailing content / reasoning on a same-shape frame (the real terminal
        chunk keeps its ``finish_reason`` -- the synthetic one carries
        ``finish_reason=None`` so the producer's stop marker still terminates
        the stream). Falls back to a minimal OpenAI skeleton when the stream
        emitted no frames of its own (the zero-chunk / all-headers case) or
        when the deep-copy path failed and the caller wants the trailing text
        still delivered on a recognizable frame.

        For ``n > 1`` streams (the proxy serves multi-choice via
        ``litellm.completion(n=...)`` -- see
        ``tests/test_request_params.py::test_streamed_n_emits_every_choice``),
        the deep-copied frame keeps every other choice from ``last_chunk``
        untouched; stamping only ``choices[0]`` would leave choices[1..]'s
        final ``delta.content`` re-delivered on this synthetic frame (a
        duplicated tail) and their real ``finish_reason`` (e.g. ``"stop"``)
        on a non-terminal chunk (a duplicated terminal signal). The splitter
        itself only ever reads ``choices[0]`` (just above), so the splitter
        tail for choices[1..] is silently unattributable in any case; strip
        the secondary choices' delta content and finish_reason so this
        synthetic frame never re-delivers a secondary choice's tail.
        """
        if last_chunk is None:
            delta = SimpleNamespace(
                content=content,
                reasoning_content=(reasoning if reasoning else None),
                role=None,
            )
            choice = SimpleNamespace(
                index=0, delta=delta, finish_reason=None,
            )
            return SimpleNamespace(choices=[choice])
        cloned = copy.deepcopy(last_chunk)
        choices = getattr(cloned, "choices", None) or []
        if not choices:
            choices.append(SimpleNamespace(
                index=0, delta=SimpleNamespace(), finish_reason=None,
            ))
            cloned.choices = choices
        frame = choices[0]
        delta = getattr(frame, "delta", None)
        if delta is None:
            delta = SimpleNamespace()
            frame.delta = delta
        delta.content = content
        if reasoning:
            delta.reasoning_content = reasoning
        frame.finish_reason = None
        # Blank every secondary choice so the synthetic frame does not
        # re-deliver their previous tail / finish_reason on a frame that
        # the splitter never wrote content for. ``choices[0]`` keeps the
        # flushed content + ``finish_reason=None``; ``choices[1..]`` keep
        # their index/delta object but lose their content and finish.
        for secondary in choices[1:]:
            secondary_delta = getattr(secondary, "delta", None)
            if secondary_delta is not None:
                # ``setattr`` (rather than rebind) keeps pydantic field
                # validators happy where the field type expects ``str``
                # but previously carried ``None`` is acceptable on the
                # wire -- the API sends ``"content": null`` for empty
                # deltas already.
                try:
                    secondary_delta.content = None
                except Exception:
                    pass
            secondary.finish_reason = None
        return cloned

    async def _stream_chunks(self, response: Any, request_data: dict):
        """Yield every byte/chunk from `response` and finish streamed accounting.

        The hook is a `yield`-per-chunk iterator, so the FastAPI/Starlette
        streaming response stays unblocked. We:

        * Parse every chunk for Anthropic SSE usage (``message_start`` carries
          the input-token total; successive ``message_delta`` frames carry
          the running output/cache totals). Tolerate chunks shaped as bytes,
          str, dict or pydantic objects; a malformed chunk is logged and
          skipped -- never propagated to the client.
        * Schedule cleanup tasks on a deterministic split:
          - **Clean completion (else arm):** the existing
            ``_schedule_stream_finalize`` task runs ``_finish_success``, which
            claims the ``_TERMINATED`` marker before its first await and does
            the slot release / ledger book / transient-failure reset /
            throughput note / limit-header capture.
          - **Mid-stream interruption (except BaseException arm):** a
            slot-RELEASE-ONLY detached task frees the slot; the partial
            usage parsed so far is stashed on ctx under
            ``_stream_collected_usage`` so the post-call failure hook can
            book it as part of the failure record. The iterator does NOT
            claim ``_TERMINATED`` here, so the failure path is the sole
            owner of failure semantics -- LiteLLM's
            ``post_call_failure_hook`` runs ``_finish_failure`` and the
            transient-failure streak is not reset.

          The split closes the reviewer's race: with Slack alerting
          configured, ``proxy_logging_obj.post_call_failure_hook``
          (utils.py:2525) awaits ``update_request_status`` before iterating
          callbacks. That intermediate await would have let the pre-fix
          detached ``_finish_success`` claim ``_TERMINATED`` first, which
          suppressed the failure record. The post-fix design makes the
          failure path deterministic regardless of which event-loop turn
          runs first.

          For pure client cancellation (``asyncio.CancelledError`` /
          ``GeneratorExit``), LiteLLM's ``async_streaming_data_generator``
          does NOT fire ``post_call_failure_hook``
          (common_request_processing.py:3670-3686 just re-raises); the
          slot-release-only task is the sole cleanup, which is what we want
          for a non-provider exit (no usage, no failure ledger entry, no
          verdict / cooldown).
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

        # OpenAI-shaped routes (and any other non-``anthropic_messages``
        # call type) still own their success logging through
        # ``async_log_success_event`` -- but the streaming iterator is the
        # only place that observes a mid-stream client disconnect, so wrap
        # the bare pass-through in ``try/except BaseException`` that
        # schedules the marker-free detached slot release. The helper
        # intentionally does NOT claim ``ctx[_TERMINATED]`` (one documented
        # deviation from the issue's sketch, mirroring the sibling
        # ``/v1/messages`` design at hooks.py:925-934 / hooks.py:989-1051) so
        # ``async_log_success_event`` can still book partial usage when
        # chunks were non-empty; on the zero-chunk CLI-sidecar leak (the
        # reported case) this arm is the sole cleanup. ``picker.release``
        # and ``_stop_heartbeat`` are idempotent, so a double-release is a
        # no-op if ``async_log_success_event`` later wins the race.
        if ctx.get("call_type") not in UNLOGGED_CALL_TYPES:
            try:
                async for chunk in response:
                    yield chunk
            except BaseException:
                self._schedule_stream_slot_release(ctx, request_data)
                raise
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
            # Stash the partial usage for the post-call failure hook to
            # consult. ``dict(collected)`` snapshots so a later mutation of
            # the local cannot surprise the failure record.
            ctx["_stream_collected_usage"] = dict(collected)
            # Slot-release-only cleanup. Idempotent with picker.release and
            # _stop_heartbeat so it is race-safe with whichever side of
            # LiteLLM's proxy eventually drives the bookkeeping.
            self._schedule_stream_slot_release(ctx, request_data)
            raise
        else:
            # Clean completion. The success finalize claims the marker in
            # its first synchronous step before any await, so a stray
            # post-call failure hook (if any) sees ``_TERMINATED == "success"``
            # and short-circuits -- preserving exactly-once across the rare
            # double-fire case the design calls out.
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
                # The response wrapper is gone by the time finalize runs --
                # the streaming generator has been fully consumed. Read
                # cost + model off ``logging_obj.model_call_details``
                # (which the streaming handler populates:
                # ``litellm_core_utils/streaming_handler.py:2386-2388``)
                # and accept headers from the same place if they ever
                # land there. ``_call_facts``'s None-response_obj branch
                # is the documented shape for this caller.
                kwargs = _call_facts(None, request_data)
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

    def _schedule_stream_slot_release(
        self, ctx: dict, request_data: dict,
    ) -> None:
        """Detached slot-release cleanup so the iterator never strands the seat.

        Scheduled from ``_stream_chunks``'s ``except BaseException`` arm,
        which means any of: client ``CancelledError`` / ``GeneratorExit``
        mid-iteration, an upstream ``Exception`` raised by the provider
        iterator, or a non-Exception ``BaseException`` (rare; SystemExit).
        The task does ONLY the slot release + heartbeat stop -- no ledger
        record, no verdict, no usage booking, no ``_TERMINATED`` claim -- so:

        * LiteLLM's ``post_call_failure_hook`` (which fires for ``Exception``
          subclasses in ``async_streaming_data_generator``'s
          ``except Exception`` arm) still gets to run ``_finish_failure``
          unblocked, claim the marker, and write the failure ledger entry
          + verdict. Without this split, the pre-fix race let the detached
          ``_finish_success`` claim ``_TERMINATED`` during the intermediate
          ``update_request_status`` await inside
          ``proxy_logging.post_call_failure_hook``, suppressing the failure
          record.
        * ``_stop_heartbeat`` and ``picker.release`` are both idempotent
          (zrem/hdel no-op on a missing key; ``_beats.pop`` no-op on an
          already-cancelled task), so the task is race-safe with whichever
          side of LiteLLM's proxy eventually drives the bookkeeping.
        * For a client cancel where LiteLLM does NOT fire
          ``post_call_failure_hook`` (the CancelledError / GeneratorExit
          ``async_streaming_data_generator`` path just re-raises), the task
          is the sole cleanup -- which is what we want: a non-provider exit
          records neither a usage nor a failure.

        The task wrapper swallows every exception: a finalize-time error
        (``Redis is down``, the registry has been hot-swapped, the model
        request id no longer exists) is logged at exception level rather
        than left as an unobserved task exception that would surface as
        ``Task exception was never retrieved``.
        """
        async def _run() -> None:
            try:
                self._stop_heartbeat(ctx["request_id"])
                await self.picker.release(
                    ctx["plan"], ctx["request_id"], ctx["model"],
                )
            except Exception:
                log.exception(
                    "stream slot release failed for request_id=%s",
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
                    "stream inline slot release failed for request_id=%s",
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
            except (UnicodeDecodeError, TypeError):
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
                except (TypeError, ValueError, AttributeError):
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

        The proxy fires this hook ONCE per request (after the router has
        walked every fallback hop). ``router.py::async_function_with_fallbacks``
        writes the in-place entry stamp (``0``) on the metadata at walk
        entry, so a real proxy-side ``request_data`` lands on
        ``hop = 0``; per-hop stamps (``1 / 2 / …``) go onto rebinding
        copies that the proxy never sees. Passing ``request_data``
        through anyway costs nothing -- ``_finish_failure`` collapses
        no-hop-info to the ``hop = -1`` sentinel and routes every
        shape (int, sentinel, proxy in-place) through one
        ``_ctx_fail_booked_hops`` dedupe set. For a /v1/messages
        failure that owns the request solo, the booker is the proxy
        arm; for an OpenAI-shape request where ``async_log`` already
        booked the same int stamp, the set collapses the second
        arrival to a no-op.

        Follow-up / residual: the served-deployment attribution in
        ``_resolve_served_failure_plan`` is hop-scoped -- on the
        ``async_log`` path, ``kwargs["model"]`` is the deployment the
        router rewrote on that hop, so a sibling's error resolves
        to the sibling's plan. On this hook's path ``request_data
        ["model"]`` is the pre-call-hook's picked deployment
        (proxy-side, never rebound by the router), so for an
        exhausted ``/v1/messages`` walk the final sibling's error
        still attributes to the picked plan. Today's generated
        config (``num_retries=0``, no general fallbacks) means
        a final AUTH/QUOTA_EXHAUSTED on the sibling lands the
        1800s cooldown on the wrong plan -- pre-existing behaviour,
        not a regression of this PR, and not fixable without a
        litellm-side signal of the last failing deployment on the
        proxy path. Filed as a follow-up against #186.
        """
        meta = (request_data or {}).get("metadata") or {}
        ctx = meta.get(META_KEY)
        if isinstance(ctx, dict):
            await self._finish_failure(
                ctx, original_exception, kwargs=request_data,
            )

    async def _apply_verdict(
        self, plan, verdict, ctx: dict, *, skip_marker: bool = False,
    ) -> None:
        """Act on a classified failure: cool the plan, learn, release the lease.

        ``skip_marker=True`` skips writing the per-attempt
        ``_verdict_applied`` marker. The marker dedupes verdict
        side-effects on a request -- but a context-window-fallbacks
        walk's attempts share the single ``ctx["request_id"]`` (the
        picker minted it once, ``run_async_fallback`` does not rewrite
        it), so a marker check that says "already applied for this
        attempt" actually fires for the whole walk and would suppress a
        legitimate later hop's verdict. ``_finish_failure`` sets the
        flag for any arm where a later hop could still surface a
        verdict -- CONTEXT verdicts (whose sibling success is the
        documented case) AND mid-walk hops (hop >= 1, whose sibling
        verdict would otherwise be silently dropped). Only the
        primary-like arms (hop <= 0, the entry stamp or the
        no-router-metadata sentinel) write the marker.

        INTERNAL verdicts (plan-key ``_INTERNAL_*`` -- switchyard's
        own plumbing: ``pick_direct`` raised, settings invalid,
        heartbeat dead, etc.) carry ``cooldown_seconds=0`` already and
        ``is_our_fault`` is True for them, so they short-circuit on the
        same guard to never cool the provider and never drop the
        session lease.
        """
        # BAD_REQUEST is the only "our fault" outcome that drops out
        # without bookkeeping -- a malformed prompt is not the
        # provider's problem. CONTEXT falls through to the drop-lease
        # block below: a session leased to a too-small plan must
        # re-lease onto a sibling with a bigger window on the next
        # turn. CONTEXT verdicts carry ``cooldown_seconds=0``, so the
        # cooldown branches never fire even though they sit between
        # this guard and the drop-lease block. INTERNAL verdicts
        # share the short-circuit reason: switchyard raised, not the
        # provider, so we never cool the provider's capacity.
        if verdict.outcome is Outcome.BAD_REQUEST or verdict.outcome is Outcome.INTERNAL:
            return

        # Per-request verdict dedupe. ``ctx["request_id"]`` is shared
        # across the whole walk, so the marker fires for every
        # subsequent call within one request. ``skip_marker`` short-
        # circuits the write so a CONTEXT or mid-walk hop's verdict
        # does not silently block a legitimate later verdict in the
        # same walk.
        marker = ctx.get("_verdict_applied")
        rid = ctx.get("request_id")
        if marker and marker == rid and not skip_marker:
            return  # already applied for this attempt (legacy backstop)
        if rid and not skip_marker:
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
        # Drop the session lease for the cases that take the served plan
        # permanently out of rotation: dead quota (QUOTA_EXHAUSTED), dead
        # subscription (PLAN_DEAD), broken credentials (AUTH), and — the
        # upstream route/deployment failure case — any verdict the classifier
        # marked with drop_lease=True (today: HTTP 404). CONTEXT verdicts
        # also drop the lease: a session leased to a plan whose window is
        # too small for the prompt would otherwise keep landing on it every
        # turn until the cooldown TTL passed; dropping it lets the next pick
        # re-lease onto a sibling with a bigger window. A session leased to
        # the failing plan would otherwise keep landing on the dead capacity
        # every turn until the cooldown TTL passed; dropping it lets the
        # next pick re-lease onto a live sibling.
        if (verdict.outcome in (Outcome.QUOTA_EXHAUSTED, Outcome.PLAN_DEAD, Outcome.AUTH,
                                Outcome.CONTEXT)
                or verdict.drop_lease) and ctx.get("session"):
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
    r"""The bracketed reason in the gateway log line.

    The base string is the picker's own cap_reason ("configured",
    "quota spent", "paced 2 of 4", ...). When the lane walked a group to
    reach this pick, the group's gid (and strategy, when distinct from the
    plain per-lane shape) is appended INSIDE the brackets, so a script that
    grep'd the previous r`\[(\w+)\]` regex still extracts the FIRST token —
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


def _has_image_block(messages: Any) -> bool:
    """Does any message carry an image content block, in EITHER wire format?

    Mirrors the dual-shape stance of `_carries_tool_results`: the protocols
    name image blocks differently, and the image-routing filter must catch
    every known shape. The known shapes are:

      Anthropic:          {"type": "image",      ...}
      OpenAI Chat:        {"type": "image_url",   ...}
      OpenAI Responses:   {"type": "input_image", ...}

    `input_image` does not literally start with "image", but it is an image
    content block, so the check is a substring match on the type rather
    than a strict prefix. A custom block whose type happens to contain
    "image" (e.g. "image_metadata") would also match — that is acceptable
    here, because the picker only filters members whose model does NOT
    support images, and a non-standard block type carrying visual content
    is exactly the kind of thing the filter should treat as an image.

    A plain string content block counts as text only — providers that
    have embedded image URLs in older shapes wrap them in a list. The
    hook checks the `messages` list defensively, so a malformed message
    (non-dict, missing role) is silently skipped rather than poisoning
    the request.
    """
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if isinstance(btype, str) and "image" in btype:
                        return True
    return False


def _tool_name(tool: Any) -> str:
    """The tool's name in EITHER wire format, or "" if neither matches.

    OpenAI nests the function under a `function` key; Anthropic keeps it flat.
    Anything else (a malformed entry, a list, a bare string) yields "" so it
    never matches a blocklist entry — the blocklist is permissive-fail: a tool
    we cannot name is one we do not touch.
    """
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if isinstance(name, str):
            return name
    name = tool.get("name")
    if isinstance(name, str):
        return name
    return ""


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
            except (TypeError, ValueError, AttributeError):
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
                except (ValueError, AttributeError, OSError):
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
    except (TypeError, AttributeError):
        return None
    return _as_float(headers.get("retry-after"))


def _provider_error(exc: BaseException) -> bool:
    """True iff `exc` looks like a provider-side failure rather than ours.

    A bare `redis.exceptions.ConnectionError`, an `asyncio.CancelledError`, or
    any of SwitchYard's own bookkeeping bugs reaches ``_handle_failure``
    carrying no signal about who is at fault. Sending those through
    ``classify(...)`` would hand them to a string-and-HTTP classifier that
    cannot tell them apart from a genuine 5xx, and the plan would cool on a
    problem we caused. A provider-shaped exception either inherits from
    ``litellm.exceptions.APIError`` or ``openai.APIError`` (the SDKs vendor
    their own HTTP wrapper classes), or carries an int-coercible HTTP
    ``status_code`` attribute on the exception itself (a hand-rolled bridge
    or a small SDK that sets the attribute directly). Anything else is ours
    to log, not the provider's to cool.

    Note: ``httpx.HTTPStatusError`` and ``requests.HTTPError`` do NOT count
    here -- they expose the status on ``exc.response.status_code``, not on a
    bare ``exc.status_code``, so a 5xx raised through them is treated as
    SwitchYard's fault and lands as INTERNAL (no cooldown). Nothing in the
    current hook path raises either shape, so this gap is latent today;
    widening the gate to catch ``.response.status_code`` is a behaviour
    change reserved for a separate fix.

    Both SDK modules are imported lazily: hooks.py only loads the
    ``CustomLogger`` subclass path of litellm at module scope, and openai is
    a sidecar-only dependency. A bare ``except (ImportError, AttributeError)``
    keeps a future SDK refactor from breaking the gate -- if either import
    path stops resolving, we conservatively fall through to "not provider
    shaped" and the original (noisy, but safe) string-classifier path runs
    instead.
    """
    # litellm and openai both raise subclasses of their own APIError for
    # transport-level failures; isinstance against the base class catches
    # every concrete HTTP shape either SDK has ever shipped.
    try:
        import litellm.exceptions as _litellm_exc
        if isinstance(exc, _litellm_exc.APIError):
            return True
    except (ImportError, AttributeError):
        pass
    try:
        import openai as _openai
        if isinstance(exc, _openai.APIError):
            return True
    except (ImportError, AttributeError):
        pass
    # Bare ``status_code`` attribute on the exception itself (a hand-rolled
    # bridge or a small SDK that mirrors the OpenAI HTTP wrapper's shape
    # by setting the attribute directly). It only counts when it coerces
    # cleanly to an HTTP-range number: a free-floating ``status_code`` of
    # None or "abc" is just a misnamed attribute and proves nothing.
    sc = getattr(exc, "status_code", None)
    if sc is None:
        return False
    try:
        return 100 <= int(sc) < 600
    except (TypeError, ValueError):
        return False


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# LiteLLM 1.101.0 does NOT set ``litellm_params.response_cost`` /
# ``litellm_params.model`` / ``litellm_params.response_headers`` on the
# ``data`` dict the proxy hands to ``async_post_call_success_hook`` /
# ``async_post_call_streaming_iterator_hook`` (verified against the pinned
# 1.101.0 source: ``common_request_processing.py`` reads those fields off the
# ``ModelResponse._hidden_params`` and ``logging_obj.model_call_details``,
# not off the request ``data``). The /v1/messages success paths therefore
# must build their own kwargs-shaped dict from the response object (and,
# for the streamed case, the logging object on ``request_data``), or cost
# always lands as $0, the served-deployment re-attribution is dead, and
# provider quota headers are never absorbed.
#
# Field provenance (pinned 1.101.0):
#
#   * ``_hidden_params["model_id"]`` -- the deployment's ``model_info.id``,
#     stamped onto every response by
#     ``litellm/litellm_core_utils/llm_response_utils/response_metadata.py:
#     set_hidden_params:71-75``. ``switchyard/gen_litellm.py`` stamps this
#     field with ``Model.router_id`` (``sy.{plan}.{model}.id``, distinct
#     from the deployment string ``sy.{plan}.{model}``), so
#     ``_check_served_deployment`` can resolve it back to a Model via
#     ``Registry.model_for_router_id``. Without the stamp LiteLLM falls back
#     to a sha256 hexdigest of (model_name, litellm_params)
#     (router.py:9223-9225) -- an opaque value the SwitchYard side cannot
#     match.
#   * ``_hidden_params["response_cost"]`` -- the canonical cost figure.
#     Some providers (Azure search, Stability image-edit) write a
#     provider-reported cost under
#     ``_hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"]``
#     instead (litellm_core_utils/streaming_handler.py:1911-1914 and
#     cost_calculator.py:1787-1792) and ``cost_calculator`` consults that
#     header FIRST, so the helper cross-checks against it -- preferring the
#     header when present matches litellm's own resolution order and is
#     the only way a provider that never populates ``_hidden_params
#     ["response_cost"]`` (the streaming-handler path) can still report
#     cost on a /v1/messages request.
#   * ``_hidden_params["additional_headers"]`` -- every response header the
#     provider sent, prefixed ``llm_provider-{header}`` by
#     ``litellm_core_utils/llm_response_utils/get_headers.py:_get_llm_provider_headers``
#     so a caller that hands them back to LiteLLM can tell provider headers
#     from LiteLLM-emulated OpenAI headers. The limit-header absorption in
#     ``_absorb_limit_headers`` matches the configured ``q.headers`` keys
#     against UN-prefixed names (e.g. ``x-ratelimit-remaining-tokens``),
#     so the helper strips the ``llm_provider-`` prefix before returning.
#
# The streamed finalize runs after the response wrapper is consumed, so
# ``response_obj=None`` is the documented call shape -- the helper falls
# back to ``request_data["litellm_logging_obj"].model_call_details``, where
# litellm stashes ``model`` and ``response_cost`` for the streaming path
# (litellm_core_utils/streaming_handler.py:2386-2388).
#
# The served-deployment keys the two paths hand ``_check_served_deployment``
# are different shapes, and the field the helper picks for each path
# matches the lookup that resolves it:
#
#   * **Buffered** -- ``_hidden_params["model_id"]`` is the deployment's
#     ``model_info.id``, which ``switchyard/gen_litellm.py`` now stamps
#     with ``Model.router_id`` (``sy.{plan}.{model}.id``). That hits the
#     ``model_for_router_id`` leg of ``_check_served_deployment``'s
#     ``model_for_deployment or model_for_router_id`` chain.
#   * **Streamed** -- the response wrapper is consumed before the
#     finalize task runs, so the helper falls back to
#     ``logging_obj.litellm_params.litellm_metadata.deployment_model_name``
#     (router.py:_update_kwargs_with_deployment:3644 stamps this with
#     ``sy.{plan}.{model}`` -- the deployment string, NOT a router id).
#     That hits ``model_for_deployment`` directly.
#
# ``model_call_details["model"]`` is ``litellm_params["model"]`` after
# routing (router.py:6412) -- the provider model string (e.g.
# ``openrouter/xiaomi/mimo-v2.5``), not a deployment string -- so the
# helper treats it as a last-resort fallback and only uses it when
# ``litellm_metadata.deployment_model_name`` is absent (e.g. a future
# litellm build that stops writing the deployment_model_name key, or a
# test fixture that fabricates a bare logging object).
#
# ``model_call_details`` does NOT carry ``additional_headers`` today, so
# a streamed /v1/messages request whose only quota signal is in the
# response headers lands without limit-header absorption -- the bounded
# loss the helper is documented to accept.
def _strip_llm_provider_prefix(headers: dict[str, Any]) -> dict[str, Any]:
    """Strip the ``llm_provider-`` prefix that
    ``litellm/litellm_core_utils/llm_response_utils/get_headers.py:_get_llm_provider_headers``
    writes onto every provider header, so the limit-header lookup in
    ``_absorb_limit_headers`` matches against the un-prefixed names it
    was originally written against. Non-dict / non-string keys are
    tolerated: a key that is not a string is coerced, a non-string value
    passes through untouched (the matcher compares the lookup string
    only). An empty result is returned as an empty dict rather than the
    sentinel ``None`` so the caller can write-through the same shape.
    """
    stripped: dict[str, Any] = {}
    for k, v in headers.items():
        if not isinstance(k, str):
            stripped[str(k)] = v
            continue
        if k.startswith("llm_provider-"):
            stripped[k[len("llm_provider-"):]] = v
        else:
            stripped[k] = v
    return stripped


def _deployment_model_name_from_logging(logging_obj: Any) -> str | None:
    """Best-effort read of the served deployment string from a litellm
    logging object. The router stamps
    ``kwargs.litellm_metadata.deployment_model_name`` (router.py:_update_kwargs_with_deployment:3644)
    and ``logging_obj.litellm_params`` carries the merged dict through
    into the streaming-handler path. LiteLLM accepts ``metadata`` as an
    alias for ``litellm_metadata`` -- try both. Returns ``None`` when
    no usable string is found; never raises.
    """
    if logging_obj is None:
        return None
    litellm_params = getattr(logging_obj, "litellm_params", None)
    if not isinstance(litellm_params, dict):
        return None
    for key in ("litellm_metadata", "metadata"):
        meta = litellm_params.get(key) or {}
        if not isinstance(meta, dict):
            continue
        candidate = meta.get("deployment_model_name")
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _call_facts(
    response_obj: Any = None, request_data: Any = None,
) -> dict:
    """Kwargs-shaped facts LiteLLM would have given us on the real
    ``async_log_success_event`` kwargs.

    Returns a dict with at most three keys: ``model``, ``response_cost``,
    ``response_headers``. Missing inputs never raise -- they simply yield
    a dict without that key. The result is shaped so it can be passed
    straight to ``_finish_success(..., kwargs=...)``: ``_check_served_deployment``
    reads ``kwargs["model"]``, ``_finish_success`` reads
    ``kwargs["response_cost"]``, and ``_absorb_limit_headers`` reads
    ``kwargs["response_headers"]``.

    The cross-check prefers the provider-reported header
    ``x-litellm-response-cost`` over ``_hidden_params["response_cost"]``
    when both are present and disagree, mirroring litellm's own order in
    ``cost_calculator.py:1787-1792``.
    """
    out: dict[str, Any] = {}
    hidden: Any = None
    if response_obj is not None:
        # LiteLLM hands the hook either a ModelResponse (attribute access)
        # or a plain dict (the synthetic response the tests build, or a
        # passthrough that has already been serialised). Cover both.
        if isinstance(response_obj, dict):
            hidden = response_obj.get("_hidden_params")
        else:
            hidden = getattr(response_obj, "_hidden_params", None)
    if not isinstance(hidden, dict):
        hidden = {}

    # ---- model ----------------------------------------------------------
    model_id = hidden.get("model_id") if hidden else None
    if isinstance(model_id, str) and model_id:
        out["model"] = model_id

    # ---- response_cost + response_headers from the response --------------
    header_cost: float | None = None
    additional = hidden.get("additional_headers") if hidden else None
    if isinstance(additional, dict) and additional:
        stripped = _strip_llm_provider_prefix(additional)
        if stripped:
            out["response_headers"] = stripped
            header_cost = _as_float(stripped.get("x-litellm-response-cost"))

    cost_from_hidden = _as_float(hidden.get("response_cost")) if hidden else None
    # Prefer the provider-reported header when both are present, mirroring
    # litellm's own resolution order. Fall back to the canonical figure
    # otherwise. Either missing -> no entry.
    if header_cost is not None:
        out["response_cost"] = header_cost
    elif cost_from_hidden is not None:
        out["response_cost"] = cost_from_hidden

    # ---- fall back to logging_obj (streamed path) -------------------------
    if request_data is not None and isinstance(request_data, dict):
        logging_obj = request_data.get("litellm_logging_obj")
        details = getattr(logging_obj, "model_call_details", None) if logging_obj is not None else None
        if isinstance(details, dict):
            if "model" not in out:
                deployment_model_name = _deployment_model_name_from_logging(logging_obj)
                if deployment_model_name:
                    out["model"] = deployment_model_name
                else:
                    fallback_model = details.get("model")
                    if isinstance(fallback_model, str) and fallback_model:
                        out["model"] = fallback_model
            if "response_cost" not in out:
                fallback_cost = _as_float(details.get("response_cost"))
                if fallback_cost is not None:
                    out["response_cost"] = fallback_cost
            # ``additional_headers`` is not currently stashed on
            # ``model_call_details`` by litellm's 1.101.0 streaming path,
            # but a future release might; tolerate the key without
            # committing to it.
            if "response_headers" not in out:
                fallback_headers = details.get("additional_headers")
                if isinstance(fallback_headers, dict) and fallback_headers:
                    stripped_headers = _strip_llm_provider_prefix(fallback_headers)
                    if stripped_headers:
                        out["response_headers"] = stripped_headers

    return out


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
    field.

    Cache tokens (``cache_creation_input_tokens``,
    ``cache_read_input_tokens``) are folded into the prompt figure ONLY when
    the prompt figure came from the Anthropic-side ``input_tokens`` key.
    When ``prompt_tokens`` is present (truthy), it is already whole -- the
    gateway CLI bridge folds ``input_tokens`` + cache_read + cache_creation
    into ``prompt_tokens`` before re-exposing the cache counts at the top
    level (``sidecars/cli_bridge/server.py::to_openai``); adding them again
    here would double-count. Same shape detection applies to attribute-shaped
    (pydantic) usage objects. Cache reads stay at full weight -- the ledger
    schema does not differentiate cached from fresh input, and the planner
    weighed the alternatives and kept this raw-count fold.
    """
    src: Any = usage_override
    if src is None and response_obj is not None:
        if isinstance(response_obj, dict):
            src = response_obj.get("usage") or {}
        else:
            src = getattr(response_obj, "usage", None) or {}
    if isinstance(src, dict):
        billed_prompt = src.get("switchyard_billed_prompt_tokens")
        if billed_prompt is not None:
            billed_completion = src.get("switchyard_billed_completion_tokens")
            return int(billed_prompt or 0), int(billed_completion or 0)
    elif getattr(src, "switchyard_billed_prompt_tokens", None) is not None:
        billed_completion = getattr(src, "switchyard_billed_completion_tokens", None)
        return (
            int(src.switchyard_billed_prompt_tokens or 0),
            int(billed_completion or 0),
        )
    if isinstance(src, dict):
        if src.get("prompt_tokens"):
            prompt = int(src.get("prompt_tokens") or 0)
            cache_read = 0
            cache_creation = 0
        else:
            prompt = int(src.get("input_tokens") or 0)
            cache_read = int(src.get("cache_read_input_tokens") or 0)
            cache_creation = int(src.get("cache_creation_input_tokens") or 0)
        completion = int(src.get("completion_tokens") or src.get("output_tokens") or 0)
    else:
        if getattr(src, "prompt_tokens", None):
            prompt = int(getattr(src, "prompt_tokens", None) or 0)
            cache_read = 0
            cache_creation = 0
        else:
            prompt = int(getattr(src, "input_tokens", None) or 0)
            cache_read = int(getattr(src, "cache_read_input_tokens", None) or 0)
            cache_creation = int(getattr(src, "cache_creation_input_tokens", None) or 0)
        completion = int(
            getattr(src, "completion_tokens", None)
            or getattr(src, "output_tokens", None)
            or 0
        )
    return prompt + cache_read + cache_creation, completion


def _collect_anthropic_event_usage(event: dict, collected: dict) -> None:
    """Take the totals from a single Anthropic SSE event.

    ``message_start`` carries the initial input-token count on ``message.usage``;
    a subsequent ``message_delta`` carries the running output and cache totals
    directly on the top-level ``usage``, and some adapters (notably
    LiteLLM's Anthropic -> chat-completions bridge) also synthesise the
    input total on the final ``message_delta``. Both are *totals*, not
    deltas -- a later ``message_delta`` is the latest authoritative reading,
    not an addition. We overwrite the running figure so a fragmented stream
    (or a provider that emits a ``message_delta`` before any tokens are
    counted) cannot inflate the total.

    The ``input_tokens`` field on a ``message_delta`` is the one exception:
    it is overwritten only when strictly positive. An adapter that streams
    a placeholder ``0`` for the input count (LiteLLM's path before it has
    the real value) MUST NOT clobber a genuine ``message_start`` count, or
    every streamed /v1/messages request would book zero prompt tokens.
    The same coercion/tolerance rules apply -- non-numeric frames are
    skipped silently so a malformed delta never crashes the parser.

    The function is intentionally narrow: ``message_start`` only writes the
    input/cache fields it actually carries, ``message_delta`` only the input/
    output/cache fields it carries, and any other event type is ignored.
    Other call shapes (text completions, embeddings, etc.) reach a different
    hook path.
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
            # message_delta can carry the real input total on the final
            # frame (LiteLLM's Anthropic adapter does this so the
            # chat-completion-shaped response has the full prompt count).
            # An adapter that sends ``0`` here -- a placeholder while it
            # waits for the real number -- MUST NOT clobber a positive
            # ``message_start`` count, otherwise every streamed
            # /v1/messages request would book zero prompt tokens. Treat
            # only strictly-positive values as authoritative overwrites;
            # non-numeric payloads are tolerated (skipped) the same way
            # the other fields are.
            inp = usage.get("input_tokens")
            if inp is not None:
                try:
                    coerced = int(inp)
                except (TypeError, ValueError):
                    pass
                else:
                    if coerced > 0:
                        collected["input_tokens"] = coerced
            for key in ("output_tokens", "cache_creation_input_tokens",
                        "cache_read_input_tokens"):
                value = usage.get(key)
                if value is not None:
                    try:
                        collected[key] = int(value)
                    except (TypeError, ValueError):
                        pass


switchyard_handler = SwitchyardHandler()
