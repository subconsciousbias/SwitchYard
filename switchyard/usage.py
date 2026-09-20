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

from .models import Plan


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
K_QUOTA = "sy:quota:{plan}"

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
        pk = period_key(plan.quota.period, now)
        keys = [
            (K_PERIOD.format(plan=plan.key, period=pk), 90 * 86400),
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

    async def current_period(self, plan: Plan) -> dict[str, float]:
        return await self.bucket(
            plan.key, K_PERIOD.format(plan=plan.key, period=period_key(plan.quota.period))
        )

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
    async def note_exhaustion(self, plan: Plan, reset_at: float | None) -> None:
        used = await self.current_period(plan)
        consumed_tokens = used["prompt_tokens"] + used["completion_tokens"]
        pipe = self.redis.pipeline()
        pipe.hset(K_QUOTA.format(plan=plan.key), mapping={
            "last_exhausted_at": time.time(),
            "observed_allowance_tokens": consumed_tokens,
            "observed_allowance_cost": used["cost"],
            "reset_at": reset_at or "",
        })
        await pipe.execute()

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

    async def note_reported(self, plan_key: str, remaining: float | None, reset_at: float | None) -> None:
        """Record limits a provider actually told us about (headers/sidecar)."""
        mapping = {}
        if remaining is not None:
            mapping["reported_remaining"] = remaining
        if reset_at is not None:
            mapping["reset_at"] = reset_at
        if mapping:
            mapping["reported_at"] = time.time()
            await self.redis.hset(K_QUOTA.format(plan=plan_key), mapping=mapping)

    async def quota_facts(self, plan_key: str) -> dict[str, float | str]:
        raw = await self.redis.hgetall(K_QUOTA.format(plan=plan_key))
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
    """What the board shows in the 'quota left' column."""
    used = await ledger.current_period(plan)
    facts = await ledger.quota_facts(plan.key)
    tokens = used["prompt_tokens"] + used["completion_tokens"]
    q = plan.quota

    basis: str | None = None
    limit: float | None = None
    consumed: float = 0.0

    if q.kind == "unlimited":
        return {"kind": "unlimited", "pct_used": None, "basis": "local, unmetered",
                "used_tokens": tokens, "used_cost": used["cost"], **_reset(facts)}

    if q.kind == "dollars":
        consumed, limit, basis = used["cost"], q.allowance, "ledger ($ spent this period)"
    elif q.kind == "tokens":
        consumed, limit, basis = tokens, q.allowance, "ledger (tokens this period)"
    else:  # window / unknown
        consumed = tokens
        basis = "ledger (tokens this window)"

    if facts.get("reported_remaining") is not None and isinstance(facts.get("reported_remaining"), float):
        # A number the provider gave us always beats our own estimate.
        rem = float(facts["reported_remaining"])
        total = rem + consumed
        return {"kind": q.kind, "pct_used": _pct(consumed, total), "limit": total,
                "consumed": consumed, "basis": "reported by provider",
                "used_tokens": tokens, "used_cost": used["cost"], **_reset(facts)}

    if limit is None and isinstance(facts.get("observed_allowance_tokens"), float):
        obs = float(facts["observed_allowance_tokens"])
        if obs > 0:
            limit, basis = obs, "observed (where it ran out last time)"

    return {"kind": q.kind, "pct_used": _pct(consumed, limit), "limit": limit,
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
