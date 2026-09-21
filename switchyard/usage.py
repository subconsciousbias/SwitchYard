"""Usage ledger and quota headroom.

Most of these plans expose no usage API, so headroom has to be inferred from
what we ourselves sent. Two tricks make that useful rather than decorative:

  * period buckets (hour / day / period) give a burn rate, which is what
    catches a $20/hour overflow long before the credit is gone;
  * when a plan does hard-fail on quota, we record how much we had consumed
    in the current period as an *observed allowance*. After one cycle the
    board can say "you hit the wall at ~118M tokens last time", so a plan
    with `allowance: null` still gets a real headroom bar.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from redis.asyncio import Redis

from .models import Plan, Quota
from .periods import period_bounds


def _now() -> datetime:
    return datetime.now(timezone.utc)


def period_key(period: str | None, at: datetime | None = None) -> str:
    at = at or _now()
    if period == "month":
        return at.strftime("%Y-%m")
    if period == "week":
        return f"{at.isocalendar().year}-W{at.isocalendar().week:02d}"
    if period == "rolling_5h":
        # Buckets of 5h from the epoch: close enough to attribute usage to the
        # window that is actually in force, without tracking each reset.
        return f"5h-{int(at.timestamp()) // 18000}"
    if period == "day":
        return at.strftime("%Y-%m-%d")
    return "all"


K_PERIOD = "sy:usage:{plan}:p:{period}"
K_HOUR = "sy:usage:{plan}:h:{hour}"
K_DAY = "sy:usage:{plan}:d:{day}"
K_QUOTA = "sy:quota:{plan}"                 # plan-level facts
K_WINDOW = "sy:qwin:{plan}:{window}"        # per-window facts (observed allowance)

FIELDS = ("requests", "prompt_tokens", "completion_tokens", "cost", "failures")


class Ledger:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def record(
        self,
        plan: Plan,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        failed: bool = False,
    ) -> None:
        now = _now()
        # One bucket per quota window: a plan with a 5-hour *and* a weekly
        # allowance needs both counted, or pacing can only see one of them.
        keys = [
            (K_PERIOD.format(plan=plan.key, period=period_key(q.period, now)), 90 * 86400)
            for q in plan.quotas
        ]
        keys += [
            (K_HOUR.format(plan=plan.key, hour=now.strftime("%Y-%m-%dT%H")), 7 * 86400),
            (K_DAY.format(plan=plan.key, day=now.strftime("%Y-%m-%d")), 400 * 86400),
        ]
        pipe = self.redis.pipeline()
        for key, ttl in keys:
            pipe.hincrbyfloat(key, "requests", 1)
            pipe.hincrbyfloat(key, "prompt_tokens", prompt_tokens)
            pipe.hincrbyfloat(key, "completion_tokens", completion_tokens)
            pipe.hincrbyfloat(key, "cost", cost)
            if failed:
                pipe.hincrbyfloat(key, "failures", 1)
            pipe.expire(key, ttl)
        await pipe.execute()

    async def bucket(self, plan_key: str, key: str) -> dict[str, float]:
        raw = await self.redis.hgetall(key)
        out = {f: 0.0 for f in FIELDS}
        for k, v in (raw or {}).items():
            k = k.decode() if isinstance(k, bytes) else k
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
        return out

    async def window_usage(self, plan: Plan, quota: Quota,
                           at: datetime | None = None) -> dict[str, float]:
        """Consumption inside one specific quota window.

        `at` exists so the pacer can be driven from an injected clock in tests;
        in production it is always now.
        """
        return await self.bucket(
            plan.key, K_PERIOD.format(plan=plan.key, period=period_key(quota.period, at))
        )

    async def current_period(self, plan: Plan) -> dict[str, float]:
        """Consumption in the target window (the one pacing aims to fill)."""
        return await self.window_usage(plan, plan.quota)

    async def burn_rate(self, plan: Plan, hours: int = 3) -> dict[str, float]:
        """Cost and tokens per hour over the last `hours` completed buckets."""
        now = _now()
        cost = tokens = 0.0
        for i in range(hours):
            at = now - timedelta(hours=i)
            b = await self.bucket(plan.key, K_HOUR.format(plan=plan.key, hour=at.strftime("%Y-%m-%dT%H")))
            cost += b["cost"]
            tokens += b["prompt_tokens"] + b["completion_tokens"]
        return {"cost_per_hour": cost / hours, "tokens_per_hour": tokens / hours}

    async def daily_series(self, plan_key: str, days: int = 30) -> list[dict]:
        now = _now()
        out = []
        for i in range(days - 1, -1, -1):
            day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            b = await self.bucket(plan_key, K_DAY.format(plan=plan_key, day=day))
            out.append({"day": day, **b})
        return out

    # -- learning where the wall is ----------------------------------------
    async def note_exhaustion(self, plan: Plan, reset_at: float | None) -> Quota:
        """Record where a plan actually ran out, against the right window.

        A provider says "you are out of quota" without saying *which* limit you
        hit. The reset time gives it away: a couple of hours means the 5-hour
        burst window, several days means the weekly allowance. Attributing this
        correctly matters — writing a 5-hour figure into the weekly window's
        observed allowance would corrupt every pacing decision after it.
        """
        window = self.attribute_window(plan, reset_at)
        used = await self.window_usage(plan, window)
        consumed_tokens = used["prompt_tokens"] + used["completion_tokens"]
        now = time.time()
        pipe = self.redis.pipeline()
        pipe.hset(K_WINDOW.format(plan=plan.key, window=window.label), mapping={
            "last_exhausted_at": now,
            "observed_allowance_tokens": consumed_tokens,
            "observed_allowance_cost": used["cost"],
            "reset_at": reset_at or "",
        })
        pipe.hset(K_QUOTA.format(plan=plan.key), mapping={
            "last_exhausted_at": now,
            "last_exhausted_window": window.label,
            "reset_at": reset_at or "",
        })
        await pipe.execute()
        return window

    @staticmethod
    def attribute_window(plan: Plan, reset_at: float | None,
                         at: datetime | None = None) -> Quota:
        """Which window did we just hit? Pick the one whose own rollover is
        closest to the reset time the provider gave us."""
        if len(plan.quotas) == 1 or not reset_at:
            # No reset hint: blame the shortest window, which is the one you hit
            # far more often, rather than poisoning the weekly figure.
            order = {"rolling_5h": 0, "day": 1, "week": 2, "month": 3, None: 4}
            return min(plan.quotas, key=lambda q: order.get(q.period, 4))
        now = at or _now()
        target_delta = reset_at - now.timestamp()
        best, best_gap = plan.quotas[0], None
        for q in plan.quotas:
            _, end = period_bounds(q.period, now)
            gap = abs((end - now).total_seconds() - target_delta)
            if best_gap is None or gap < best_gap:
                best, best_gap = q, gap
        return best

    async def note_concurrency_rejection(self, plan: Plan) -> None:
        """A provider refusing us on connection count means our cap is wrong.

        Tracked separately from rate limiting because the fix is different:
        lower `max_parallel` in plans.yaml rather than wait it out.
        """
        key = K_QUOTA.format(plan=plan.key)
        await self.redis.hincrbyfloat(key, "concurrency_rejections", 1)
        await self.redis.hset(key, mapping={
            "concurrency_rejected_at": time.time(),
            "concurrency_rejected_at_cap": plan.max_parallel,
        })

    async def note_reported_percent(self, plan_key: str, used_percent: float | None,
                                    reset_at: float | None,
                                    window: str | None = None) -> None:
        """Record a percentage a provider stated, when it publishes no counts.

        MiniMax reports `current_weekly_used_percent` with every count set to -1,
        so there is nothing to reconcile against our own tally — the percentage
        IS the measurement. Kept in its own field so window_headroom can prefer
        it without ever mixing percent into a token or dollar total.
        """
        mapping = {}
        if used_percent is not None:
            mapping["reported_pct_used"] = max(0.0, min(100.0, float(used_percent)))
        if reset_at is not None:
            mapping["reset_at"] = reset_at
        if not mapping:
            return
        mapping["reported_at"] = time.time()
        key = (K_WINDOW.format(plan=plan_key, window=window) if window
               else K_QUOTA.format(plan=plan_key))
        await self.redis.hset(key, mapping=mapping)

    async def note_reported(self, plan_key: str, remaining: float | None,
                            reset_at: float | None, window: str | None = None,
                            limit: float | None = None) -> None:
        """Record limits a provider actually told us about (headers/sidecar).

        `limit` matters when the provider states both sides: xAI answers with
        x-ratelimit-limit-tokens next to the remaining count, and deriving the
        total from our own tally instead would understate it by everything the
        account spent outside SwitchYard.
        """
        mapping = {}
        if remaining is not None:
            mapping["reported_remaining"] = remaining
        if limit is not None:
            mapping["reported_limit"] = limit
        if reset_at is not None:
            mapping["reset_at"] = reset_at
        if not mapping:
            return
        mapping["reported_at"] = time.time()
        key = (K_WINDOW.format(plan=plan_key, window=window) if window
               else K_QUOTA.format(plan=plan_key))
        await self.redis.hset(key, mapping=mapping)

    async def window_facts(self, plan_key: str, window: str) -> dict[str, float | str]:
        return await self._facts(K_WINDOW.format(plan=plan_key, window=window))

    async def quota_facts(self, plan_key: str) -> dict[str, float | str]:
        return await self._facts(K_QUOTA.format(plan=plan_key))

    async def _facts(self, key: str) -> dict[str, float | str]:
        raw = await self.redis.hgetall(key)
        out: dict[str, float | str] = {}
        for k, v in (raw or {}).items():
            k = k.decode() if isinstance(k, bytes) else k
            v = v.decode() if isinstance(v, bytes) else v
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        return out


async def headroom(ledger: Ledger, plan: Plan) -> dict:
    """Headroom for every quota window, plus which one is closest to biting.

    The target window (weekly, usually) is what pacing fills; a constraint
    window (the 5-hour burst) can still be the one that stops you first, so the
    board needs both.
    """
    windows = [await window_headroom(ledger, plan, q) for q in plan.quotas]
    target = next((w for w in windows if w["role"] == "target"), windows[0])
    rated = [w for w in windows if w.get("pct_used") is not None]
    binding = max(rated, key=lambda w: w["pct_used"]) if rated else target
    return {**target, "windows": windows, "binding": binding,
            "binding_is_target": binding is target}


async def window_headroom(ledger: Ledger, plan: Plan, q: Quota) -> dict:
    """What the board shows in the 'quota left' column, for one window."""
    used = await ledger.window_usage(plan, q)
    facts = await ledger.window_facts(plan.key, q.label)
    tokens = used["prompt_tokens"] + used["completion_tokens"]

    basis: str | None = None
    limit: float | None = None
    consumed: float = 0.0

    meta = {"window": q.label, "role": q.role, "period": q.period}

    if q.kind == "unlimited":
        return {**meta, "kind": "unlimited", "pct_used": None, "basis": "local, unmetered",
                "used_tokens": tokens, "used_cost": used["cost"], **_reset(facts)}

    if q.kind == "dollars":
        consumed, limit, basis = used["cost"], q.allowance, "ledger ($ spent this period)"
    elif q.kind == "tokens":
        consumed, limit, basis = tokens, q.allowance, "ledger (tokens this period)"
    else:  # window / unknown
        consumed = tokens
        basis = "ledger (tokens this window)"

    if isinstance(facts.get("reported_pct_used"), float):
        # The provider gave a percentage and no counts. It is the best number
        # available, so it wins outright — but there is no limit to report, and
        # inventing one from our own tally would be a guess dressed as a fact.
        return {**meta, "kind": q.kind, "pct_used": round(float(facts["reported_pct_used"]), 1),
                "limit": None, "consumed": consumed,
                "basis": "reported by provider (% only)",
                "used_tokens": tokens, "used_cost": used["cost"], **_reset(facts)}

    if facts.get("reported_remaining") is not None and isinstance(facts.get("reported_remaining"), float):
        # A number the provider gave us always beats our own estimate.
        rem = float(facts["reported_remaining"])
        # A stated limit beats reconstructing one from our own consumption.
        total = (float(facts["reported_limit"])
                 if isinstance(facts.get("reported_limit"), float) else rem + consumed)
        return {**meta, "kind": q.kind, "pct_used": _pct(max(0.0, total - rem), total),
                "limit": total,
                "consumed": max(0.0, total - rem), "basis": "reported by provider",
                "used_tokens": tokens, "used_cost": used["cost"], **_reset(facts)}

    if limit is None and isinstance(facts.get("observed_allowance_tokens"), float):
        obs = float(facts["observed_allowance_tokens"])
        if obs > 0:
            limit, basis = obs, "observed (where it ran out last time)"

    return {**meta, "kind": q.kind, "pct_used": _pct(consumed, limit), "limit": limit,
            "consumed": consumed, "basis": basis,
            "used_tokens": tokens, "used_cost": used["cost"], **_reset(facts)}


def _pct(consumed: float, limit: float | None) -> float | None:
    if not limit or limit <= 0:
        return None
    return min(100.0, round(100.0 * consumed / limit, 1))


def _reset(facts: dict) -> dict:
    reset_at = facts.get("reset_at")
    return {
        "reset_at": reset_at if isinstance(reset_at, float) and reset_at else None,
        "last_exhausted_at": facts.get("last_exhausted_at") or None,
    }


def effective_cost_per_mtok(plan: Plan, tokens_this_month: float, metered_cost: float) -> float | None:
    """The number that decides whether a subscription is worth renewing.

    A $132 plan that delivered 40M tokens cost $3.30/Mtok; the same plan at
    400M cost $0.33. Metered providers use actual spend. Returns None when
    there is not enough traffic yet to mean anything.
    """
    if tokens_this_month < 1_000_000:
        return None
    spend = plan.monthly_cost if plan.monthly_cost else metered_cost
    if spend <= 0:
        return 0.0
    return round(spend / (tokens_this_month / 1_000_000), 4)
