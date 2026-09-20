"""Effective capacity: what a plan is *actually* allowed to run right now.

Two independent controllers, both optional, both per plan:

ConcurrencyLearner — discovers the real parallelism a provider tolerates, since
    "probably under 4 connections" is a guess. Additive-increase /
    multiplicative-decrease, the same shape as TCP congestion control: halve on
    a connection-limit rejection, creep up by one when there is unmet demand and
    no recent rejection. Learned per hour-of-day, because a provider that
    tolerates 6 connections at 04:00 may tolerate 2 at peak.

Pacer — keeps a subscription tracking to ~100% consumption at the moment its
    quota rolls over, instead of burning the allowance early and spilling to the
    tail. Two mechanisms, because concurrency alone is too coarse a knob: at
    real LLM throughput even a single busy slot can drain a monthly allowance in
    a couple of days, so "reduce concurrency" cannot slow us below one slot.

      1. A pace line — the consumption we *should* have reached by now:

             pace_line = allowance * elapsed_fraction * (1 + overshoot)

         While consumption is ahead of that line the plan is closed (0 slots)
         and reopens when the line catches up. That duty-cycles the plan, and
         the average lands on the line however fast individual requests are.

      2. A throughput-derived cap for when we are on or behind the line:

             slots = (remaining / seconds_left) / observed_rate_per_slot

    `deadline` is min(next rollover, expiry) — see periods.py. Because quota
    resets per window, the final truncated window of a cancelled plan paces
    *harder*: same allowance, less time, so a steeper line.

Only subscriptions are paced. Metered providers (OpenRouter, the Anthropic API)
and local models keep fixed caps and the ordinary spill-and-cooldown behaviour,
because there is no allowance to land exactly on.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from redis.asyncio import Redis

from .models import Plan, Settings
from .periods import deadline as window_deadline
from .periods import period_bounds
from .usage import Ledger

K_SWITCH = "sy:switch:pacing"     # runtime override of settings.pacing.enabled
K_LEARN = "sy:learn:{plan}:{bucket}"
K_PACE = "sy:pace:{plan}"
K_PRESSURE = "sy:pressure:{plan}"

EWMA_ALPHA = 0.3


@dataclass
class Capacity:
    cap: int
    reason: str
    learned: int | None = None
    paced: int | None = None
    configured: int | None = None

    @property
    def throttled(self) -> bool:
        return self.paced is not None and self.cap < (self.learned or self.cap)


def _bucket(settings: Settings, at: datetime | None = None) -> str:
    if settings.concurrency_learning.buckets != "hour_of_day":
        return "global"
    now = at or datetime.now(timezone.utc)
    return f"h{now.hour:02d}"


def _f(raw, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


class ConcurrencyLearner:
    def __init__(self, redis: Redis, settings: Settings):
        self.redis = redis
        self.settings = settings

    async def _read(self, plan_key: str, bucket: str) -> dict[str, float]:
        raw = await self.redis.hgetall(K_LEARN.format(plan=plan_key, bucket=bucket))
        out: dict[str, float] = {}
        for k, v in (raw or {}).items():
            k = k.decode() if isinstance(k, bytes) else k
            out[k] = _f(v)
        return out

    async def note_rejection(self, plan: Plan, at_concurrency: int) -> int:
        """The provider refused us on connection count: back off, hard."""
        cfg = self.settings.concurrency_learning
        floor = max(1, cfg.min_cap)
        new = max(floor, int(at_concurrency * cfg.decrease_factor))
        now = time.time()
        for bucket in {_bucket(self.settings), "global"}:
            key = K_LEARN.format(plan=plan.key, bucket=bucket)
            current = _f((await self.redis.hget(key, "cap")), new)
            await self.redis.hset(key, mapping={
                "cap": min(new, current) if current else new,
                "changed_at": now,
                "last_rejection_at": now,
            })
            await self.redis.hincrbyfloat(key, "rejections", 1)
        await self.redis.delete(K_PRESSURE.format(plan=plan.key))
        return new

    async def note_pressure(self, plan_key: str) -> None:
        """A request wanted a slot and the cap denied it. Demand exists."""
        key = K_PRESSURE.format(plan=plan_key)
        await self.redis.incr(key)
        await self.redis.expire(key, 3600)

    async def effective(self, plan: Plan) -> tuple[int, str]:
        """Learned cap for right now, probing upward when it is safe to."""
        cfg = self.settings.concurrency_learning
        ceiling = plan.max_parallel_ceiling or max(plan.configured_parallel or 1, 1) * 4
        seed = plan.configured_parallel or cfg.seed_cap

        if not cfg.enabled:
            return max(1, seed), "configured"

        bucket = _bucket(self.settings)
        local = await self._read(plan.key, bucket)
        glob = await self._read(plan.key, "global")

        # Prefer this hour's learning once it has seen enough evidence.
        if local.get("samples", 0) >= cfg.min_samples and local.get("cap"):
            cap, source = int(local["cap"]), f"learned[{bucket}]"
            state = local
        elif glob.get("cap"):
            cap, source = int(glob["cap"]), "learned[global]"
            state = glob
        else:
            cap, source = max(1, seed), "seed" if plan.configured_parallel is None else "configured"
            state = {}

        # Probe upward: only with unmet demand, a quiet period since the last
        # rejection, and a gap since the last change.
        now = time.time()
        pressure = _f(await self.redis.get(K_PRESSURE.format(plan=plan.key)))
        quiet = now - state.get("last_rejection_at", 0) > cfg.probe_cooldown_seconds
        settled = now - state.get("changed_at", 0) > cfg.probe_interval_seconds
        if pressure >= cfg.probe_pressure and quiet and settled and cap < ceiling:
            cap = min(ceiling, cap + cfg.increase_step)
            key = K_LEARN.format(plan=plan.key, bucket=bucket)
            await self.redis.hset(key, mapping={"cap": cap, "changed_at": now})
            await self.redis.hincrbyfloat(key, "samples", 1)
            await self.redis.delete(K_PRESSURE.format(plan=plan.key))
            source += "+probe"

        return max(1, min(cap, ceiling)), source


class Pacer:
    def __init__(self, redis: Redis, settings: Settings, ledger: Ledger):
        self.redis = redis
        self.settings = settings
        self.ledger = ledger

    async def note_throughput(self, plan: Plan, units: float, seconds: float) -> None:
        """One slot delivered `units` in `seconds`: that is the per-slot rate."""
        if units <= 0 or seconds <= 0:
            return
        key = K_PACE.format(plan=plan.key)
        prior = _f(await self.redis.hget(key, "rate"))
        sample = units / seconds
        rate = sample if prior <= 0 else (EWMA_ALPHA * sample + (1 - EWMA_ALPHA) * prior)
        await self.redis.hset(key, mapping={"rate": rate, "rate_at": time.time()})

    async def state(self, plan: Plan) -> dict:
        """Everything the portal needs to explain the pacing decision."""
        now = datetime.now(timezone.utc)
        used = await self.ledger.current_period(plan)
        facts = await self.ledger.quota_facts(plan.key)
        consumed = (used["cost"] if plan.quota.kind == "dollars"
                    else used["prompt_tokens"] + used["completion_tokens"])

        allowance = plan.quota.allowance
        basis = "configured"
        if allowance is None:
            observed = facts.get("observed_allowance_cost" if plan.quota.kind == "dollars"
                                 else "observed_allowance_tokens")
            if isinstance(observed, float) and observed > 0:
                allowance, basis = observed, "observed"

        dl, is_final = window_deadline(plan.quota.period, plan.expires, now)
        start, _ = period_bounds(plan.quota.period, now)
        total_s = max(1.0, (dl - start).total_seconds())
        remaining_s = max(0.0, (dl - now).total_seconds())
        elapsed_frac = 1.0 - remaining_s / total_s

        out = {
            "active": False, "reason": "", "allowance": allowance, "basis": basis,
            "consumed": consumed, "deadline": dl.timestamp(), "is_final_window": is_final,
            "elapsed_frac": round(elapsed_frac, 4), "remaining_seconds": remaining_s,
            "rate_per_slot": _f(await self.redis.hget(K_PACE.format(plan=plan.key), "rate")),
            "target_rate": None, "desired_slots": None, "consumed_frac": None,
            "pace_line": None, "ahead_by": None, "projected_end_frac": None,
        }

        if not plan.paced(self.settings):
            out["reason"] = "not a subscription" if not plan.is_subscription else "pacing off"
            return out
        if not allowance or allowance <= 0:
            out["reason"] = "no allowance known yet"
            return out

        out["consumed_frac"] = round(consumed / allowance, 4)

        # Where consumption should have reached by now, aimed slightly hot so
        # the allowance lands at ~100% rather than short of it.
        overshoot = 1.0 + self.settings.pacing.overshoot
        pace_line = allowance * min(1.0, elapsed_frac * overshoot)
        out["pace_line"] = pace_line
        out["ahead_by"] = consumed - pace_line

        remaining = allowance - consumed
        if remaining <= 0:
            out.update(active=True, reason="allowance spent", desired_slots=0, target_rate=0.0)
            return out
        if remaining_s <= 0:
            out["reason"] = "window closed"
            return out

        target_rate = (remaining / remaining_s) * overshoot
        out["target_rate"] = target_rate
        rate = out["rate_per_slot"]
        if rate > 0:
            # How far into the window this burn rate would exhaust the plan.
            out["projected_end_frac"] = round(
                elapsed_frac + (remaining / (rate * max(1, self.settings.pacing.min_slots))
                                ) / max(1.0, total_s), 4)

        # Ahead of the line: close the plan and let the line catch up. This is
        # the mechanism that actually slows us below one continuous slot.
        if consumed > pace_line:
            out.update(active=True, reason="ahead of pace, holding", desired_slots=0)
            return out

        if rate <= 0:
            out.update(active=True, reason="measuring per-slot rate", desired_slots=None)
            return out

        out.update(active=True, reason="pacing to deadline", desired_slots=target_rate / rate)
        return out

    async def desired_slots(self, plan: Plan) -> tuple[int | None, str, dict]:
        st = await self.state(plan)
        if not st["active"] or st["desired_slots"] is None:
            return None, st["reason"], st

        slots = st["desired_slots"]
        if slots <= 0:
            # Closed by the pace line (or spent). Don't smooth this: holding
            # must take effect immediately or we sail past the line.
            return 0, st["reason"], st

        # Smooth upward moves so one unusually fast request cannot open the
        # floodgates, but keep a floor so an open plan is always usable.
        key = K_PACE.format(plan=plan.key)
        prior = _f(await self.redis.hget(key, "slots"))
        smoothed = slots if prior <= 0 else EWMA_ALPHA * slots + (1 - EWMA_ALPHA) * prior
        await self.redis.hset(key, mapping={"slots": smoothed})
        return max(self.settings.pacing.min_slots, int(round(smoothed))), st["reason"], st


class CapacityPolicy:
    """Composes the two controllers into one number the picker can use."""

    def __init__(self, redis: Redis, settings: Settings, ledger: Ledger):
        self.redis = redis
        self.settings = settings
        self.learner = ConcurrencyLearner(redis, settings)
        self.pacer = Pacer(redis, settings, ledger)

    async def pacing_enabled(self) -> bool:
        """plans.yaml sets the default; the portal can flip it at runtime."""
        raw = await self.redis.get(K_SWITCH)
        if raw is None:
            return self.settings.pacing.enabled
        raw = raw.decode() if isinstance(raw, bytes) else raw
        return str(raw) == "1"

    async def set_pacing(self, enabled: bool | None) -> bool:
        """None clears the override and returns to the configured default."""
        if enabled is None:
            await self.redis.delete(K_SWITCH)
        else:
            await self.redis.set(K_SWITCH, "1" if enabled else "0")
        return await self.pacing_enabled()

    async def plan_is_paced(self, plan: Plan) -> bool:
        if not await self.pacing_enabled():
            return False
        if plan.pacing is not None:
            return bool(plan.pacing)
        return plan.is_subscription or (plan.metered and self.settings.pacing.include_metered)

    async def effective(self, plan: Plan) -> Capacity:
        learned, source = await self.learner.effective(plan)
        cap = Capacity(cap=learned, reason=source, learned=learned,
                       configured=plan.configured_parallel)

        if not await self.plan_is_paced(plan):
            return cap

        paced, reason, _ = await self.pacer.desired_slots(plan)
        if paced is None:
            cap.reason = f"{source} (pacing: {reason})"
            return cap

        cap.paced = paced
        # Pacing can only ever *reduce* concurrency below what the provider
        # tolerates; it must never talk us into exceeding a learned limit.
        cap.cap = min(learned, paced)
        cap.reason = f"paced {paced} of {learned} ({reason})"
        return cap

    async def tail_enabled(self) -> bool:
        """In pacing mode the tail is off: narrowing is the point, and falling
        back to a local model would hide the fact that we are ahead of budget."""
        if not self.settings.pacing.disable_tail:
            return True
        return not await self.pacing_enabled()
