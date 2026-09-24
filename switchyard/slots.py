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
# Drain flag: when set on a plan, the picker refuses to honour any lease on
# it AND refuses to claim a fresh slot on it. The body-walk gate surfaces
# the ref as "(draining)" in `considered` and falls through to the lane's
# next sibling; the affinity path drops the lease so a fresh request that
# arrives while a lease is still on the drained plan also lands on a sibling.
# apply.sh sets this flag BEFORE running the migrate helper, so the picker
# must enforce both gates itself; the flag is the contract, not apply.sh's
# ordering. A future maintainer who reorders apply.sh (or narrows the
# affinity drop) does not change that contract.
K_DRAIN = "sy:drain:{plan}"
# Reverse lease index: per-plan SET of session ids whose lease currently
# points at a ref on this plan. Backs `sessions_on_plan(plan)`, which the
# drain migration walks to find what to re-pick. Maintained by set_lease /
# touch_lease / drop_lease so a stale SET is impossible — every state change
# to a lease updates the index in the same code path. TTL matched to the
# lease so the index self-heals when a session's lease expires on its own.
K_LEASE_PLAN = "sy:lease_plan:{plan}"
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

# Refresh lease TTL + per-plan reverse-index TTL atomically. The reverse
# index is keyed by the lease's plan prefix (`sy:lease_plan:{plan}`), so the
# script reads the lease value BEFORE refreshing — and crucially, the
# previous two-step Python sequence (EXPIRE then GET) could observe the
# lease after a concurrent drop / TTL fire / failover, and the EXPIRE of
# the SET would never run, leaving the SET to age out against a still-live
# (on a stale-primary) lease. Single-script: read, conditionally refresh
# both, no race.
#
# KEYS[1] lease key
# ARGV[1] ttl
# -> 1 refreshed | 0 lease gone (nothing to refresh against)
_TOUCH_LEASE = """
-- TOUCH_LEASE
local v = redis.call('GET', KEYS[1])
if not v then return 0 end
redis.call('EXPIRE', KEYS[1], ARGV[1])
local plan = string.match(v, '^([^/]+)/')
if plan then
  redis.call('EXPIRE', 'sy:lease_plan:' .. plan, ARGV[1])
end
return 1
"""

# Drop lease + injection marker + per-plan reverse-index membership
# atomically. Reads the lease BEFORE deleting it so the SREM targets the
# right plan key; if the lease was already gone (TTL fired, prior drop,
# failover to a primary that didn't have it), the SREM is a silent no-op.
# Injection marker rides on the lease — see drop_lease's docstring for why
# clearing it here covers the three lease-drop sites.
#
# KEYS[1] lease key
# KEYS[2] injection marker key
# ARGV[1] session id (for SREM against the per-plan SET)
# -> always 1
_DROP_LEASE = """
-- DROP_LEASE
local v = redis.call('GET', KEYS[1])
redis.call('DEL', KEYS[1])
redis.call('DEL', KEYS[2])
if v then
  local plan = string.match(v, '^([^/]+)/')
  if plan then
    redis.call('SREM', 'sy:lease_plan:' .. plan, ARGV[1])
  end
end
return 1
"""

# Compare-and-drop the lease: read the lease, and only DEL + SREM + clear the
# injection marker if the current value still matches ARGV[2] (the ref the
# caller read at decision time). Same KEYS / ARGV layout as _DROP_LEASE plus
# the expected-ref check.
#
# The CAS-on-value guards two interleavings and explicitly does NOT guard a
# third:
#
#   * Lease-replaced — covered. The value differs from ARGV[2] because
#     a parallel turn (drain migration, `_visit_ref` body walk's
#     set_lease) wrote a different ref. The parallel turn's new state
#     is left alone; the caller bails out of the DEL.
#   * Lease-gone — covered. TTL fired, a prior `drop_lease` landed
#     (e.g. hooks.py:1369 on a post-call QUOTA_EXHAUSTED / PLAN_DEAD /
#     AUTH / explicit-drop verdict), or the key never existed (failover
#     to a primary without it). The CAS sees a missing key and is a
#     no-op.
#
# The CAS does NOT close the touch_lease-same-value race: _TOUCH_LEASE
# only EXPIREs the lease key, it does not re-SET the value, so a parallel
# pinned turn that touch_leases the same lease between the caller's read
# and now leaves the value unchanged. The CAS still matches and the
# drop fires — i.e. the caller does NOT get the protection a future
# reader of "compare-and-drop" might assume. Closing that race needs a
# TTL/touch-aware variant (a counter _TOUCH_LEASE bumps and _DROP_LEASE_IF
# checks); until one exists, callers that care about the touch_lease
# interleaving must be aware that this primitive is value-CAS, not
# touch-aware CAS. The picker overflow gate is the only caller today, and
# its wrapping comment in switchyard/picker.py names the limitation.
#
# KEYS[1] lease key
# KEYS[2] injection marker key
# ARGV[1] session id (for SREM)
# ARGV[2] expected ref (the ref the caller wants to drop)
# -> 1 dropped | 0 skipped (value did not match — caller no longer owns this lease)
_DROP_LEASE_IF = """
-- DROP_LEASE_IF
local v = redis.call('GET', KEYS[1])
if not v or v ~= ARGV[2] then
  return 0
end
redis.call('DEL', KEYS[1])
redis.call('DEL', KEYS[2])
local plan = string.match(v, '^([^/]+)/')
if plan then
  redis.call('SREM', 'sy:lease_plan:' .. plan, ARGV[1])
end
return 1
"""

# Set lease + per-plan reverse-index membership + both TTLs atomically.
# The previous three-step Python sequence (SET lease EX ttl, SADD plan SET,
# EXPIRE plan SET) had a single-round-trip race between the SADD and the
# EXPIRE: a network drop after SADD but before EXPIRE leaves the SET with
# the new membership but no TTL, and SADD never sets a TTL on a fresh
# key. Membership then persists forever, and `sessions_on_plan(plan)`
# reads a phantom for every plan that ever partial-failed. Single-script:
# SADD, SET, EXPIRE — no window where SET exists without TTL, and no
# caller can observe the membership without the TTL bound to it.
#
# KEYS[1] lease key
# KEYS[2] reverse-index key (sy:lease_plan:{plan})
# ARGV[1] session id (for SADD)
# ARGV[2] ref string (the value the lease holds)
# ARGV[3] ttl (lease seconds and SET EXPIRE seconds)
# -> SADD count (newly added; 1 if first, 0 if already a member)
_SET_LEASE = """
-- SET_LEASE
local added = redis.call('SADD', KEYS[2], ARGV[1])
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
redis.call('EXPIRE', KEYS[2], ARGV[3])
return added
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
        self._touch_lease = redis.register_script(_TOUCH_LEASE)
        self._drop_lease = redis.register_script(_DROP_LEASE)
        self._drop_lease_if = redis.register_script(_DROP_LEASE_IF)
        self._set_lease = redis.register_script(_SET_LEASE)

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

    async def set_lease(self, session: str, ref: str, ttl: int,
                        plan_key: str | None = None) -> None:
        # `ref` is the model ref (e.g. `claude-max/fable`); the plan key is
        # the prefix before `/`. Callers in the picker pass it explicitly so
        # the plan lookup is done once, but it is optional for tests that
        # already have the ref and don't want to thread a registry through.
        if plan_key is None:
            plan_key = ref.split("/", 1)[0]
        # Atomic: SADD, SET lease EX, EXPIRE SET — all in one Lua script.
        # The previous three-step Python sequence had a window between the
        # SADD and the EXPIRE on the SET (a single round-trip's worth) where
        # a network drop could leave the SET with new membership but no TTL;
        # `sessions_on_plan(plan)` would then read a phantom forever, and
        # the drain flow would re-pick the same session every migration.
        # Single-script: no window where the SET exists without a TTL bound
        # to it.
        await self._set_lease(
            keys=[K_LEASE.format(session=session),
                  K_LEASE_PLAN.format(plan=plan_key)],
            args=[session, ref, ttl],
        )

    async def touch_lease(self, session: str, ttl: int) -> None:
        # Atomic: the script reads the lease value first, so a concurrent
        # drop / TTL fire / failover between the read and the EXPIRE is
        # impossible. The reverse-index TTL only refreshes when the lease
        # was actually present — exactly the contract a "lost the lease"
        # caller wants (no phantom SET refresh against a dead mapping).
        await self._touch_lease(
            keys=[K_LEASE.format(session=session)],
            args=[ttl],
        )

    async def drop_lease(self, session: str) -> None:
        # Atomic: read lease, DEL lease, DEL injection marker, SREM from
        # the per-plan reverse index — all in one script. The previous
        # two-step (GET then DEL + SREM) was subject to the same race as
        # touch_lease: between the GET and the SREM, the lease could be
        # re-created with a different plan and the SREM would target the
        # wrong SET. Single-script: read once, SREM against the same
        # snapshot the read returned.
        #
        # The injection marker rides on the lease: clearing it here covers
        # the three lease-drop sites (hooks.py post-call failure, picker.py
        # held-but-unservable, picker.py pinned-but-no-slot) so a hard plan
        # rejection that re-leases the session re-enables injection on the
        # next turn.
        await self._drop_lease(
            keys=[K_LEASE.format(session=session),
                  K_INJECTED.format(session=session)],
            args=[session],
        )

    async def drop_lease_if(self, session: str, expected_ref: str) -> bool:
        # Compare-and-drop on the lease value: only DEL the lease (and the
        # matching injection marker + per-plan reverse-index entry) if the
        # lease's current value is *still* the one the caller read at
        # decision time. Returns True when the drop fired, False when the
        # lease no longer matches — it has been replaced (drain migration
        # wrote a different ref, a `_visit_ref` body walk `set_lease`d a
        # different ref), or it has gone away (TTL fired, a prior
        # `drop_lease` landed — e.g. hooks.py:1369 on a post-call
        # QUOTA_EXHAUSTED / PLAN_DEAD / AUTH / explicit-drop verdict —
        # or failover to a primary without the key), between the caller's
        # read and now. In either sub-case the caller no longer owns the
        # lease and a blind DEL would have clobbered state a concurrent
        # turn just established.
        #
        # This is a value-CAS, not a touch-aware CAS. A parallel pinned
        # follow-up that touch_leases the same lease in the window does
        # not change the lease value (_TOUCH_LEASE only EXPIREs — it does
        # not re-SET), so the CAS still matches and the drop fires. The
        # picker overflow gate (the only caller today, see
        # switchyard/picker.py) names this limitation in its wrapping
        # comment; closing the touch_lease race needs a TTL/touch-aware
        # primitive slots.py does not yet expose.
        result = await self._drop_lease_if(
            keys=[K_LEASE.format(session=session),
                  K_INJECTED.format(session=session)],
            args=[session, expected_ref],
        )
        return int(result) == 1

    # -- plan drain gate ----------------------------------------------------
    # A drained plan refuses new work without unmounting anything — apply.sh
    # is free to recreate the containers while traffic has already moved to
    # the sibling. The flag is just a sentinel; the gate lives in the picker
    # next to the cooldown gate.
    async def set_drain(self, plan: str) -> None:
        await self.redis.set(K_DRAIN.format(plan=plan), "1")

    async def clear_drain(self, plan: str) -> None:
        await self.redis.delete(K_DRAIN.format(plan=plan))

    async def is_draining(self, plan: str) -> bool:
        return bool(await self.redis.get(K_DRAIN.format(plan=plan)))

    async def drain_ttl(self, plan: str) -> int:
        """Seconds left on the drain gate, or -2 if the gate is gone.

        Real Redis semantics: -2 = key missing, -1 = key has no TTL, N = seconds
        remaining. The board clamps via `max(0, ttl)` so -2 / -1 collapse to a
        non-positive number the caller can treat as "not draining"; this method
        surfaces the raw value so a future caller that wants to distinguish
        "expired this instant" (-1 still, until the next GET) from "never had a
        TTL" (-1 from the start) can do so without a second round-trip. Mirrors
        `cooldown_state` which already returns the raw ttl alongside the
        `(cooled, remaining)` tuple.
        """
        return int(await self.redis.ttl(K_DRAIN.format(plan=plan)))

    async def sessions_on_plan(self, plan: str) -> list[str]:
        """Session ids currently leased to a ref on `plan`, in SET order.

        Used by `switchyard/drain.py` to walk every session that needs
        migrating before a drained plan is taken out of service. The reverse
        index is maintained by set_lease / touch_lease / drop_lease, so this
        is the live picture — no stale keys, no scans.
        """
        raw = await self.redis.smembers(K_LEASE_PLAN.format(plan=plan))
        return [m.decode() if isinstance(m, bytes) else m for m in raw]

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
