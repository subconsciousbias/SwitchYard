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
K_INFLIGHT_MODEL = "sy:inflight:m:{ref}"
K_COOL = "sy:cool:{plan}"
K_LEASE = "sy:lease:{session}"
# Marks a session whose first turn has had the "you are running through
# SwitchYard, your native tools are unavailable" system note appended. Sticky
# for the lease TTL so a resumed CLI session after a server restart still
# reaches the model with the note in place. Dropped together with the lease so
# a hard plan rejection (which re-leases the session) re-enables injection.
K_INJECTED = "sy:inject:{session}"
# request id -> the lane that claimed it. A slot's cost is real wherever it came
# from, but a lane's board should say which of the busy slots are *its* traffic
# and which belong to a sibling lane sharing the same model, so the two are not
# confused. Written inside the claim script so it cannot disagree with the
# counters, and dropped on release.
K_INFLIGHT_LANE = "sy:inflight:lane:{plan}"
# Consecutive transient-failure counter, per plan. Drives the escalated
# cooldown ladder (60s, 120s, 240s, …) and the portal's "failing · Nx" chip.
# TTL 7200s sliding, so a plan that goes silent long enough for its streak to
# fall out of memory resets to zero — the next failure starts at streak 1,
# not at "what it was 90 minutes ago".
K_TFAIL = "sy:tfail:{plan}"

# Two counters, claimed atomically. The PLAN's limit caps total concurrency; a
# MODEL's optional limit caps how much of that one model may take. Checking a
# model's cap against the plan's counter instead — which is what a single
# counter forces — means a plan of 2 with two models at 1 each can only ever run
# one request, because the second model sees the first model's slot.
#
# KEYS[1] plan inflight, KEYS[2] plan cooldown, KEYS[3] model inflight,
# KEYS[4] plan lane map
# ARGV[1] request id, ARGV[2] now, ARGV[3] plan cap, ARGV[4] stale-before,
# ARGV[5] ttl, ARGV[6] model cap (-1 for none), ARGV[7] lane ("" for none)
# -> 1 claimed | 0 plan at capacity | -1 cooled down | -2 model at capacity
_CLAIM = """
if redis.call('EXISTS', KEYS[2]) == 1 then return -1 end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[4])
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', ARGV[4])
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then return 0 end
local mcap = tonumber(ARGV[6])
if mcap >= 0 and redis.call('ZCARD', KEYS[3]) >= mcap then return -2 end
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[5])
redis.call('ZADD', KEYS[3], ARGV[2], ARGV[1])
redis.call('EXPIRE', KEYS[3], ARGV[5])
if ARGV[7] ~= '' then
  redis.call('HSET', KEYS[4], ARGV[1], ARGV[7])
  redis.call('EXPIRE', KEYS[4], ARGV[5])
end
return 1
"""

# Atomic INCR + EXPIRE. INCR does not set a TTL on a fresh key, so the plain
# pair left a window where a worker killed between INCR and EXPIRE would leave
# the counter without a TTL — the next quiet gap would not drop it, so a plan
# that was merely unlucky 90 minutes ago would still look like it had a streak
# today. The script closes that window.
#
# KEYS[1] streak key (K_TFAIL)
# ARGV[1] streak TTL (7200)
# -> the new streak value (post-INCR)
_BUMP_STREAK = """
-- BUMP_STREAK
local v = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], ARGV[1])
return v
"""

# Atomic INCR + EXPIRE + SET cooldown. The escalated cooldown ladder derives
# from the post-INCR streak, so the cooldown written MUST be computed against
# the same streak the caller reads back. Doing INCR and SET as two separate
# round-trips races under concurrent TRANSIENT failures: the last SET wins
# but it may have been computed against an earlier (smaller) streak than the
# final INCR value, so a plan can land at streak 6 with a cooldown that
# matches streak 5. The script keeps the streak and the cooldown in lock-step.
#
# KEYS[1] streak key (K_TFAIL), KEYS[2] cooldown key (K_COOL)
# ARGV[1] streak TTL (7200), ARGV[2] base cooldown, ARGV[3] cap (max_seconds),
# ARGV[4] reason, ARGV[5] now (epoch seconds)
# -> the new streak value (post-INCR)
_BUMP_AND_COOL = """
-- BUMP_AND_COOL
local streak = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], ARGV[1])
local base = tonumber(ARGV[2])
local cap = tonumber(ARGV[3])
local cooldown = base
if streak > 1 then
  local doubled = base * (2 ^ (streak - 1))
  if doubled > cap then
    cooldown = cap
  else
    cooldown = doubled
  end
end
local reason = ARGV[4]
local now = tonumber(ARGV[5])
local until_ts = now + cooldown
redis.call('SET', KEYS[2], reason .. '|' .. until_ts, 'EX', math.max(1, math.floor(cooldown)))
return streak
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
        self._bump_streak = redis.register_script(_BUMP_STREAK)
        self._bump_and_cool = redis.register_script(_BUMP_AND_COOL)

    # -- capacity ----------------------------------------------------------
    async def try_claim(self, plan: str, cap: int, request_id: str,
                        model_ref: str | None = None,
                        model_cap: int | None = None,
                        lane: str | None = None) -> int:
        """Claim a slot on the plan, and on the model when it has its own cap."""
        now = time.time()
        ref = model_ref or f"{plan}/*"
        return int(
            await self._claim(
                keys=[K_INFLIGHT.format(plan=plan), K_COOL.format(plan=plan),
                      K_INFLIGHT_MODEL.format(ref=ref),
                      K_INFLIGHT_LANE.format(plan=plan)],
                args=[request_id, now, cap, now - self.inflight_max_age,
                      self.inflight_max_age * 2,
                      -1 if model_cap is None else model_cap, lane or ""],
            )
        )

    async def release(self, plan: str, request_id: str, model_ref: str) -> None:
        """Release both counters. `model_ref` is required on purpose: omitting it
        leaks the model's slot until the staleness sweep notices, which shows up
        as a model that mysteriously refuses work."""
        await self.redis.zrem(K_INFLIGHT.format(plan=plan), request_id)
        await self.redis.zrem(K_INFLIGHT_MODEL.format(ref=model_ref), request_id)
        await self.redis.hdel(K_INFLIGHT_LANE.format(plan=plan), request_id)

    async def touch(self, plan: str, request_id: str, model_ref: str) -> bool:
        """Refresh a live claim's timestamp.

        A slot is only released by the completion hook, which never runs if the
        worker is killed or the client vanishes — so a dead request used to hold
        its slot until the staleness sweep, 15 minutes later. A 2-slot plan then
        looks full with nothing running.

        Heartbeating while the request is alive separates "slow" from "dead",
        which lets the sweep be aggressive without cutting off long calls (an
        OpenCode request legitimately runs for minutes). GT/XX: only update an
        existing member, never resurrect one the sweep already removed.
        """
        now = time.time()
        pipe = self.redis.pipeline()
        # CH is not optional: ZADD XX without it counts only members *added*,
        # which under XX is always zero. Reading that as "the slot is gone" made
        # every heartbeat stop after its first beat, so a request outliving
        # inflight_max_age lost its slot to the sweep while still running — and
        # a CLI call legitimately runs for minutes.
        pipe.zadd(K_INFLIGHT.format(plan=plan), {request_id: now}, xx=True, ch=True)
        pipe.zadd(K_INFLIGHT_MODEL.format(ref=model_ref), {request_id: now},
                  xx=True, ch=True)
        updated = await pipe.execute()
        return bool(updated and updated[0])

    async def in_flight_model(self, ref: str) -> int:
        key = K_INFLIGHT_MODEL.format(ref=ref)
        await self.redis.zremrangebyscore(key, "-inf", time.time() - self.inflight_max_age)
        return int(await self.redis.zcard(key))

    async def in_flight(self, plan: str) -> int:
        key = K_INFLIGHT.format(plan=plan)
        await self.redis.zremrangebyscore(key, "-inf", time.time() - self.inflight_max_age)
        return int(await self.redis.zcard(key))

    async def in_flight_by_lane(self, plan: str, model_ref: str | None = None
                                ) -> dict[str, int]:
        """Which lanes the plan's live slots belong to, {lane: count}.

        Reads the lane map against the *live* members rather than on its own, so
        an entry the sweep already dropped cannot inflate a lane's share. A slot
        claimed before lanes were recorded, or by a caller naming a deployment
        directly rather than a lane, counts under "" — shown as "direct".
        """
        key = (K_INFLIGHT_MODEL.format(ref=model_ref) if model_ref
               else K_INFLIGHT.format(plan=plan))
        await self.redis.zremrangebyscore(key, "-inf", time.time() - self.inflight_max_age)
        live = await self.redis.zrange(key, 0, -1)
        if not live:
            return {}
        mapping = await self.redis.hgetall(K_INFLIGHT_LANE.format(plan=plan))
        out: dict[str, int] = {}
        for member in live:
            rid = member.decode() if isinstance(member, bytes) else str(member)
            raw = mapping.get(rid.encode()) if mapping else None
            if raw is None and mapping:
                raw = mapping.get(rid)
            lane = (raw.decode() if isinstance(raw, bytes) else raw) or ""
            out[lane] = out.get(lane, 0) + 1
        return out

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
        # The injection marker rides on the lease: clearing it here covers the
        # three lease-drop sites (hooks.py post-call failure, picker.py held-
        # but-unservable, picker.py pinned-but-no-slot) so a hard plan rejection
        # that re-leases the session re-enables injection on the next turn.
        await self.redis.delete(K_LEASE.format(session=session))
        await self.redis.delete(K_INJECTED.format(session=session))

    async def injected(self, session: str) -> bool:
        v = await self.redis.get(K_INJECTED.format(session=session))
        return bool(v)

    async def mark_injected(self, session: str, ttl: int) -> None:
        # Plain SET EX — never SET NX. A resumed session after a stack restart
        # has nothing here today (Redis died with the stack); re-writing it on
        # every first turn keeps the gate working without races. The TTL
        # matches the lease so the two expire together. Value is opaque;
        # `injected()` only needs to see *something*.
        await self.redis.set(K_INJECTED.format(session=session), "1", ex=ttl)

    # -- transient-failure streak -----------------------------------------
    # The escalated-cooldown ladder and the board's failing chip both need to
    # know how many TRANSIENT failures in a row a plan has produced. We track
    # that here so the hook does not have to invent a counter, and so a plan
    # that was merely quiet for two hours does not look like it has been
    # failing this whole time.
    async def note_transient_failure(self, plan: str) -> int:
        """Increment the streak and refresh its TTL atomically.

        INCR does not set a TTL on a fresh key, and INCR does not refresh the
        TTL on an existing one — only EXPIRE does. A plain INCR + EXPIRE pair
        left a window where a worker killed between the two calls would leave
        the counter without a TTL, so the streak would carry forward forever
        instead of dropping after 7200s of quiet. The Lua script closes that
        window. Returns the new streak value.
        """
        return int(await self._bump_streak(
            keys=[K_TFAIL.format(plan=plan)],
            args=[7200],
        ))

    async def bump_and_cool(self, plan: str, base: int, cap: int,
                            reason: str) -> int:
        """Atomic INCR + EXPIRE + SET cooldown for the escalated ladder.

        Concurrent TRANSIENT failures on the same plan used to race: the last
        `cool_down` write wins, but it could have been computed against a
        smaller streak than the final INCR value, leaving the picker to
        observe a cooldown that undershoots what the streak warrants. Doing
        INCR, the escalated-cooldown math, and the cooldown SET in a single
        EVAL keeps the streak and the cooldown in lock-step: the cooldown
        the picker reads is always the cooldown for the streak the caller
        just observed. Returns the post-INCR streak value.
        """
        return int(await self._bump_and_cool(
            keys=[K_TFAIL.format(plan=plan), K_COOL.format(plan=plan)],
            args=[7200, int(base), int(cap), reason, int(time.time())],
        ))

    async def transient_failure_streak(self, plan: str) -> int:
        """0 when the plan is healthy or its counter has expired."""
        raw = await self.redis.get(K_TFAIL.format(plan=plan))
        if raw is None:
            return 0
        raw = raw.decode() if isinstance(raw, bytes) else raw
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    async def reset_transient_failures(self, plan: str) -> None:
        await self.redis.delete(K_TFAIL.format(plan=plan))
