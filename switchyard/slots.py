"""Slot table, cooldowns, and session leases — the whole routing brain's state.

All of it lives in Redis so that N proxy workers share one view of capacity.
In-flight slots are a sorted set keyed by request id with the claim timestamp
as score, rather than a counter: a crashed worker leaks a member instead of
permanently decrementing capacity, and every claim prunes members older than
`inflight_max_age_seconds`. Capacity heals itself.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from redis.asyncio import Redis

K_INFLIGHT = "sy:inflight:{plan}"
K_COOL = "sy:cool:{plan}"
K_LEASE = "sy:lease:{session}"

# KEYS[1] inflight zset, KEYS[2] cooldown key
# ARGV[1] request id, ARGV[2] now, ARGV[3] cap, ARGV[4] stale-before, ARGV[5] ttl
# -> 1 claimed | 0 at capacity | -1 cooled down
_CLAIM = """
if redis.call('EXISTS', KEYS[2]) == 1 then return -1 end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[4])
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then return 0 end
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[5])
return 1
"""


@dataclass
class Claim:
    plan: str
    request_id: str
    session: str | None


class SlotTable:
    def __init__(self, redis: Redis, inflight_max_age: int = 900):
        self.redis = redis
        self.inflight_max_age = inflight_max_age
        self._claim = redis.register_script(_CLAIM)

    # -- capacity ----------------------------------------------------------
    async def try_claim(self, plan: str, cap: int, request_id: str) -> int:
        now = time.time()
        return int(
            await self._claim(
                keys=[K_INFLIGHT.format(plan=plan), K_COOL.format(plan=plan)],
                args=[request_id, now, cap, now - self.inflight_max_age,
                      self.inflight_max_age * 2],
            )
        )

    async def release(self, plan: str, request_id: str) -> None:
        await self.redis.zrem(K_INFLIGHT.format(plan=plan), request_id)

    async def in_flight(self, plan: str) -> int:
        key = K_INFLIGHT.format(plan=plan)
        await self.redis.zremrangebyscore(key, "-inf", time.time() - self.inflight_max_age)
        return int(await self.redis.zcard(key))

    # -- cooldown: this is how a plan's capacity *disappears* --------------
    async def cool_down(self, plan: str, seconds: int, reason: str) -> None:
        await self.redis.set(
            K_COOL.format(plan=plan), f"{reason}|{int(time.time() + seconds)}",
            ex=max(1, int(seconds)),
        )

    async def cooldown_state(self, plan: str) -> tuple[bool, int, str]:
        """(cooled, seconds_remaining, reason)"""
        key = K_COOL.format(plan=plan)
        val = await self.redis.get(key)
        if not val:
            return False, 0, ""
        val = val.decode() if isinstance(val, bytes) else val
        reason, _, until = val.partition("|")
        ttl = int(await self.redis.ttl(key) or 0)
        return True, max(0, ttl), reason

    async def clear_cooldown(self, plan: str) -> None:
        await self.redis.delete(K_COOL.format(plan=plan))

    # -- session affinity --------------------------------------------------
    async def get_lease(self, session: str) -> str | None:
        v = await self.redis.get(K_LEASE.format(session=session))
        return (v.decode() if isinstance(v, bytes) else v) if v else None

    async def set_lease(self, session: str, plan: str, ttl: int) -> None:
        await self.redis.set(K_LEASE.format(session=session), plan, ex=ttl)

    async def touch_lease(self, session: str, ttl: int) -> None:
        await self.redis.expire(K_LEASE.format(session=session), ttl)

    async def drop_lease(self, session: str) -> None:
        await self.redis.delete(K_LEASE.format(session=session))
