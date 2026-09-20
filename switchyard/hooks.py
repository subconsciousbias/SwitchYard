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

import hashlib
import logging
import os
import time
from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from redis.asyncio import Redis

from . import models
from .classify import Outcome, classify
from .picker import LaneSaturated, Picker
from .session import derive as derive_session
from .slots import SlotTable
from .usage import Ledger

log = logging.getLogger("switchyard")

META_KEY = "switchyard"


class SwitchyardHandler(CustomLogger):
    def __init__(self) -> None:
        self.registry = models.load()
        self._redis: Redis | None = None
        self._slots: SlotTable | None = None
        self._picker: Picker | None = None
        self._ledger: Ledger | None = None
        log.info(
            "switchyard: %d plans, lanes=%s",
            len(self.registry.plans), ",".join(self.registry.lanes),
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
    def picker(self) -> Picker:
        if self._picker is None:
            self._picker = Picker(self.registry, self.slots)
        return self._picker

    @property
    def ledger(self) -> Ledger:
        if self._ledger is None:
            self._ledger = Ledger(self.redis)
        return self._ledger

    def reload(self) -> None:
        """Pick up edits to plans.yaml without restarting the proxy."""
        self.registry = models.load()
        self._picker = Picker(self.registry, self.slots)

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

        try:
            pick = await self.picker.pick(lane, session)
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

        data["model"] = pick.plan.deployment
        meta = data.setdefault("metadata", {})
        meta[META_KEY] = {
            "lane": lane,
            "plan": pick.plan.key,
            "request_id": pick.request_id,
            "session": session,
            "sticky": pick.sticky,
            "claimed_at": time.time(),
        }
        log.info(
            "lane=%s -> %s%s%s",
            lane, pick.plan.key,
            " (sticky)" if pick.sticky else "",
            f" skipped={','.join(pick.considered)}" if pick.considered else "",
        )
        return data

    # -- outbound: release the slot, record usage --------------------------
    def _ctx(self, kwargs: dict) -> dict | None:
        meta = (kwargs.get("litellm_params", {}) or {}).get("metadata") or kwargs.get("metadata") or {}
        ctx = meta.get(META_KEY)
        return ctx if isinstance(ctx, dict) else None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        ctx = self._ctx(kwargs)
        if not ctx:
            return
        plan = self.registry.plans.get(ctx["plan"])
        await self.picker.release(ctx["plan"], ctx["request_id"])
        if not plan:
            return

        usage = getattr(response_obj, "usage", None) or {}
        get = (lambda k: usage.get(k, 0)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, 0) or 0)
        await self.ledger.record(
            plan,
            prompt_tokens=int(get("prompt_tokens") or 0),
            completion_tokens=int(get("completion_tokens") or 0),
            cost=float(kwargs.get("response_cost") or 0.0),
        )
        await self._absorb_limit_headers(plan.key, kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        ctx = self._ctx(kwargs)
        if not ctx:
            return
        await self.picker.release(ctx["plan"], ctx["request_id"])
        plan = self.registry.plans.get(ctx["plan"])
        if plan:
            await self.ledger.record(plan, failed=True)
        await self._handle_failure(ctx, kwargs.get("exception") or kwargs.get("original_exception"))

    async def async_post_call_failure_hook(
        self, request_data: dict, original_exception: Exception, user_api_key_dict: UserAPIKeyAuth, **_: Any
    ) -> None:
        meta = (request_data or {}).get("metadata") or {}
        ctx = meta.get(META_KEY)
        if isinstance(ctx, dict):
            await self.picker.release(ctx["plan"], ctx["request_id"])
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
            retry_after=retry_after,
            default_cooldown=self.registry.settings.default_cooldown_seconds,
        )
        if verdict.outcome is Outcome.QUOTA_EXHAUSTED:
            reset_at = time.time() + verdict.cooldown_seconds
            await self.ledger.note_exhaustion(plan, reset_at)
            log.warning(
                "plan=%s quota exhausted (%s) — dropping its %d slots for %ds",
                plan.key, verdict.detail, plan.max_parallel, verdict.cooldown_seconds,
            )
        if verdict.should_cool:
            await self.slots.cool_down(plan.key, verdict.cooldown_seconds, verdict.outcome.value)
        if verdict.outcome in (Outcome.QUOTA_EXHAUSTED, Outcome.AUTH) and ctx.get("session"):
            # Do not strand the session on dead capacity; let it re-lease.
            await self.slots.drop_lease(ctx["session"])

    async def _absorb_limit_headers(self, plan_key: str, kwargs: dict) -> None:
        """Prefer a provider's own remaining-quota headers over our estimate."""
        plan = self.registry.plans.get(plan_key)
        if not plan or not plan.quota.headers:
            return
        headers = {}
        for src in ("response_headers", "_response_headers"):
            h = kwargs.get(src) or (kwargs.get("additional_args") or {}).get(src)
            if isinstance(h, dict):
                headers.update({k.lower(): v for k, v in h.items()})
        if not headers:
            return
        rem_h = (plan.quota.headers.get("remaining") or "").lower()
        reset_h = (plan.quota.headers.get("reset") or "").lower()
        remaining = _as_float(headers.get(rem_h))
        reset = _as_float(headers.get(reset_h))
        if remaining is not None or reset is not None:
            await self.ledger.note_reported(plan_key, remaining, reset)


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
