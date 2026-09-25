"""Just enough async Redis to exercise the routing brain in-process.

`register_script` returns a Python reference implementation of the Lua scripts
imported from `switchyard.slots` (CLAIM, BUMP_STREAK, BUMP_AND_COOL,
TOUCH_LEASE, DROP_LEASE, SET_LEASE). The two must stay in step; the integration
test against a real Redis (`tests/test_slots_lua.py`) is what catches drift.

TTL: one expiry map `_expiries: key -> absolute deadline` covers every key
type. `set(..., ex=n)` writes it; `expire` updates it for any key type;
`delete` clears it. Every accessor purges the key when its deadline has passed,
so a touched-after-expiry behaves like a miss. The clock is injectable so
tests can advance time without `asyncio.sleep`.

A handful of older tests still poke the legacy `(value, expiry)` tuple shape
directly into `fake.strings`; the string accessors tolerate that shape so the
public TTL semantics stay correct without forcing those tests to migrate.
"""
from __future__ import annotations

import re
import time

from switchyard.slots import (
    _BUMP_AND_COOL,
    _BUMP_STREAK,
    _CLAIM,
    _DROP_LEASE,
    _DROP_LEASE_IF,
    _SET_LEASE,
    _TOUCH,
    _TOUCH_LEASE,
)

# Real Redis compiles each Lua script once and matches `register_script`
# callers by exact source. The fake matches the same way: this dict pairs
# each `switchyard.slots` script constant with the `_shadow_*` method name
# the fake uses to mimic it. `register_script` looks up the constant, finds
# the method on `self`, and returns the bound method — a simple flat dispatch
# that keeps the per-script bodies in their own methods instead of as
# closures inside one giant function (which is what tripped C901 once the
# new _DROP_LEASE_IF script landed).
_SCRIPT_SOURCE_TO_SHADOW = {
    _CLAIM:           "_shadow_claim",
    _BUMP_AND_COOL:   "_shadow_bump_and_cool",
    _BUMP_STREAK:     "_shadow_bump_streak",
    _TOUCH:           "_shadow_touch",
    _TOUCH_LEASE:     "_shadow_touch_lease",
    _DROP_LEASE:      "_shadow_drop_lease",
    _DROP_LEASE_IF:   "_shadow_drop_lease_if",
    _SET_LEASE:       "_shadow_set_lease",
}


class FakePipeline:
    def __init__(self, store):
        self.store = store
        self.ops = []

    def hincrbyfloat(self, key, field, amount):
        self.ops.append(("hincrbyfloat", key, field, amount)); return self

    def hgetall(self, key):
        self.ops.append(("hgetall", key)); return self

    def hset(self, key, mapping=None, **_):
        self.ops.append(("hset", key, mapping or {})); return self

    def zadd(self, key, mapping, xx=False, ch=False, **_):
        self.ops.append(("zadd", key, mapping, xx, ch)); return self

    def incr(self, key):
        self.ops.append(("incr", key)); return self

    def pfadd(self, key, *members):
        self.ops.append(("pfadd", key, members)); return self

    def pfcount(self, *keys):
        # Real PFCOUNT accepts multiple keys and returns the union cardinality
        # in a single call; the pipeline mirrors that so model_overview can
        # queue one multi-key PFCOUNT for the plan-level n_sessions without
        # branching the API.
        self.ops.append(("pfcount", keys)); return self

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl)); return self

    async def execute(self):
        """Returns one result per queued command, in order, like a real pipeline.

        It used to return [], so a caller reading `results[0]` silently got
        nothing — which is exactly how a heartbeat whose liveness check never
        worked passed its tests.
        """
        results = []
        for op in self.ops:
            name = op[0]
            if name == "hincrbyfloat":
                _, key, field, amount = op
                # Direct store mutation: purge first so a write to an
                # expired-and-collected key recreates it instead of resurrecting
                # the old value. The TTL on an existing key survives — HSET /
                # HINCRBYFLOAT in real Redis neither refresh nor clear it.
                self.store._purge_expired(key)
                h = self.store.hashes.setdefault(key, {})
                h[field] = float(h.get(field, 0)) + float(amount)
                results.append(h[field])
            elif name == "hgetall":
                _, key = op
                self.store._purge_expired(key)
                results.append(self.store.hashes.get(key, {}))
            elif name == "hset":
                _, key, mapping = op
                self.store._purge_expired(key)
                self.store.hashes.setdefault(key, {}).update(
                    {k: str(v) for k, v in mapping.items()}
                )
                results.append(len(mapping))
            elif name == "zadd":
                _, key, mapping, xx, ch = op
                results.append(await self.store.zadd(key, mapping, xx=xx, ch=ch))
            elif name == "incr":
                _, key = op
                results.append(await self.store.incr(key))
            elif name == "pfadd":
                _, key, members = op
                results.append(await self.store.pfadd(key, *members))
            elif name == "pfcount":
                _, keys = op
                results.append(await self.store.pfcount(*keys))
            elif name == "expire":
                _, key, ttl = op
                results.append(await self.store.expire(key, ttl))
        self.ops.clear()
        return results


class FakeRedis:
    def __init__(self, clock=None):
        self.clock = clock if clock is not None else time.time
        self.strings: dict[str, str | tuple[str, float | None]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        # HLL fake: each key is a set of distinct members added via pfadd.
        # Real Redis HLLs are approximate (~1% error) and shaped differently;
        # the test surface only cares about "how many distinct sessions this
        # model saw this month" so an exact set is the right semantics to fake.
        self.hlls: dict[str, set[str]] = {}
        # Sets: exact membership (real Redis SETs use a hash table internally;
        # the test surface only needs set semantics, not encoding). Backs the
        # drain flow's reverse lease index (sy:lease_plan:{plan}) and any other
        # SMEMBERS-shaped surface that lands in tests.
        self.sets: dict[str, set[str]] = {}
        # One expiry map covers every key type. A key's presence here means
        # "has a TTL"; the value is the absolute deadline (epoch seconds). The
        # key itself is removed from its store when the deadline passes; the
        # expiry map entry is removed at the same time so a later EXPIRE on
        # the same name starts fresh rather than reporting a stale deadline.
        self._expiries: dict[str, float] = {}

    def _now(self) -> float:
        return float(self.clock())

    def _purge_expired(self, key: str) -> None:
        """Drop the key from every store if its TTL has fired.

        Centralising the check here keeps the accessors tiny: each one calls
        this before touching its store, so an expired-and-collected key reads
        as a miss instead of resurrecting the old value.
        """
        deadline = self._expiries.get(key)
        if deadline is None or deadline > self._now():
            return
        self.strings.pop(key, None)
        self.zsets.pop(key, None)
        self.hashes.pop(key, None)
        self.sets.pop(key, None)
        self.hlls.pop(key, None)
        self._expiries.pop(key, None)

    def _has_key(self, key: str) -> bool:
        """True iff the key lives in at least one store. TTL does not count."""
        return (key in self.strings or key in self.zsets
                or key in self.hashes or key in self.sets
                or key in self.hlls)

    @staticmethod
    def _string_value(stored):
        """Unwrap a string entry. New writes are plain strings; older tests
        that poke `fake.strings[k] = (value, expiry)` for setup still rely
        on this accessor returning just the value.
        """
        if isinstance(stored, tuple):
            return stored[0]
        return stored

    # -- strings -----------------------------------------------------------
    async def set(self, key, value, ex=None):
        # Strings now store the bare value; the deadline lives in the shared
        # `_expiries` map so non-string TTLs read through the same code path.
        self.strings[key] = value
        if ex:
            self._expiries[key] = self._now() + float(ex)
        else:
            self._expiries.pop(key, None)

    async def get(self, key):
        self._purge_expired(key)
        return self._string_value(self.strings.get(key))

    async def ttl(self, key):
        # Real Redis semantics: -2 = key does not exist, -1 = key exists but
        # has no TTL, N = seconds remaining. The only production caller
        # (`slots.cooldown_state`) already clamps via `max(0, ttl)`, so the
        # difference between "no TTL" and "expired this instant" is benign for
        # the existing call sites; the strict shape lets future callers
        # distinguish without surprise.
        self._purge_expired(key)
        if not self._has_key(key):
            return -2
        deadline = self._expiries.get(key)
        if deadline is None:
            return -1
        return max(0, int(deadline - self._now()))

    async def delete(self, key):
        self.strings.pop(key, None)
        self.zsets.pop(key, None)
        self.hashes.pop(key, None)
        self.sets.pop(key, None)
        self.hlls.pop(key, None)
        self._expiries.pop(key, None)

    async def expire(self, key, ttl):
        """Real EXPIRE semantics: 1 if the key exists and got a TTL, else 0.

        Works for every key type — strings, zsets, hashes, sets, HLLs — because
        the TTL lives in the shared `_expiries` map and the store lookups all
        read through it. Production callers (policy.note_pressure,
        usage pipelines) hand us TTLs in hours/days, so they were silently
        string-only before; the constraint is that those keys live in hashes,
        which means the old implementation was a no-op against real usage.
        """
        self._purge_expired(key)
        if not self._has_key(key):
            return 0
        self._expiries[key] = self._now() + float(ttl)
        return 1

    # -- zsets -------------------------------------------------------------
    async def zadd(self, key, mapping, xx=False, ch=False, **_):
        """Real ZADD counts members *added*; with CH it counts added OR updated.
        The difference is the whole liveness signal behind touch(), so the fake
        has to model it or a broken heartbeat looks fine in tests."""
        self._purge_expired(key)
        z = self.zsets.setdefault(key, {})
        changed = 0
        for member, score in mapping.items():
            if xx and member not in z:
                continue
            if member not in z:
                changed += 1
            elif ch and z[member] != float(score):
                changed += 1
            z[member] = float(score)
        return changed

    async def zrem(self, key, member):
        self._purge_expired(key)
        self.zsets.get(key, {}).pop(member, None)

    async def zcard(self, key):
        self._purge_expired(key)
        return len(self.zsets.get(key, {}))

    async def zrange(self, key, start, stop):
        self._purge_expired(key)
        members = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
        if stop == -1:
            stop = len(members)
        else:
            stop += 1
        return [m for m, _ in members[start:stop]]

    async def zremrangebyscore(self, key, lo, hi):
        self._purge_expired(key)
        z = self.zsets.get(key, {})
        for m in [m for m, s in z.items() if s <= float(hi)]:
            del z[m]

    # -- hashes ------------------------------------------------------------
    async def hgetall(self, key):
        self._purge_expired(key)
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, mapping=None, **_):
        self._purge_expired(key)
        self.hashes.setdefault(key, {}).update({k: str(v) for k, v in (mapping or {}).items()})
        # Plain HSET is not a TTL reset in real Redis; the deadline on an
        # existing key survives untouched. EXPIRE has to come from a separate
        # op if the caller wants the key to age out (or refresh).
        # (No _expiries mutation here on purpose.)

    async def hget(self, key, field):
        self._purge_expired(key)
        return self.hashes.get(key, {}).get(field)

    async def hdel(self, key, field):
        self._purge_expired(key)
        return 1 if self.hashes.get(key, {}).pop(field, None) is not None else 0

    async def incr(self, key):
        self._purge_expired(key)
        existing = self.strings.get(key)
        base = self._string_value(existing) if existing is not None else "0"
        val = int(float(base)) + 1
        self.strings[key] = str(val)
        # Real INCR does not refresh TTL on an existing key, and does not set
        # one on a fresh one. Honour that so an INCR-and-EXPIRE pair has to
        # be two ops, the same as in real Redis.
        return val

    async def hincrbyfloat(self, key, field, amount):
        self._purge_expired(key)
        h = self.hashes.setdefault(key, {})
        h[field] = str(float(h.get(field, 0)) + float(amount))
        return float(h[field])

    # -- hyperloglog (fake) ------------------------------------------------
    # Set-membership semantics: PFADD adds each member to the set, no-op if
    # already present; PFCOUNT returns len(set). Real HLLs use register arrays
    # and have bounded error; tests asserting "this many distinct sessions"
    # want exact answers.
    async def pfadd(self, key, *members):
        # Real PFADD returns 1 if at least one new member was added, 0
        # otherwise (independent of how many were passed). The fake had been
        # returning the count of newly-added members, which is *different*
        # semantics: a caller branching on `await pipe.pfadd(...) == 1` for
        # the canonical "did I add something" check would silently read a
        # count > 1 and behave wrong. Align here while only one PFADD caller
        # exists and the only test surface that cares is the cardinalities.
        self._purge_expired(key)
        s = self.hlls.setdefault(key, set())
        added = 0
        for m in members:
            if m not in s:
                s.add(m)
                added += 1
        return 1 if added else 0

    async def pfcount(self, *keys):
        # Real PFCOUNT accepts multiple keys and returns the cardinality of
        # the union of the HLLs in a single call. The fake is exact, so the
        # union is set.union over the underlying sets (with absent keys
        # treated as empty sets, same as single-key behaviour).
        for k in keys:
            self._purge_expired(k)
        if len(keys) == 1:
            return len(self.hlls.get(keys[0], set()))
        return len(set().union(*(self.hlls.get(k, set()) for k in keys)))

    # -- sets --------------------------------------------------------------
    async def sadd(self, key, *members):
        # Real SADD returns the number of members that were ADDED (i.e. were
        # not already present). Members that were already members do not
        # count. The drain flow's reverse lease index ignores the return
        # value, but keep the count right so anything branching on it later
        # (e.g. "did this session make it into the index") reads honestly.
        self._purge_expired(key)
        s = self.sets.setdefault(key, set())
        added = 0
        for m in members:
            if m not in s:
                s.add(m)
                added += 1
        return added

    async def srem(self, key, *members):
        self._purge_expired(key)
        s = self.sets.get(key, set())
        removed = 0
        for m in members:
            if m in s:
                s.discard(m)
                removed += 1
        return removed

    async def smembers(self, key):
        self._purge_expired(key)
        return list(self.sets.get(key, set()))

    async def scard(self, key):
        self._purge_expired(key)
        return len(self.sets.get(key, set()))

    async def sismember(self, key, member):
        self._purge_expired(key)
        return 1 if member in self.sets.get(key, set()) else 0

    def pipeline(self):
        return FakePipeline(self)

    async def ping(self):
        return True

    # -- the Lua shadow methods ---------------------------------------------
    # `register_script` looks each script's exact source up in
    # `_SCRIPT_SOURCE_TO_SHADOW` and returns the bound shadow method. The
    # bodies are kept as standalone methods (not closures inside
    # `register_script`) so C901 stays well below the 25 ceiling even as
    # new scripts land — adding a new script is +1 entry in the dict,
    # not +N control-flow nodes on a single function.

    async def _shadow_claim(self, keys, args):
        inflight_key, cool_key, model_key, lane_key = keys
        rid, now, plan_cap, stale_before, ttl, model_cap, lane = args
        if await self.get(cool_key) is not None:
            return -1
        await self.zremrangebyscore(inflight_key, "-inf", stale_before)
        await self.zremrangebyscore(model_key, "-inf", stale_before)
        plan_z = self.zsets.setdefault(inflight_key, {})
        if len(plan_z) >= int(plan_cap):
            return 0
        mcap = int(model_cap)
        model_z = self.zsets.setdefault(model_key, {})
        if mcap >= 0 and len(model_z) >= mcap:
            return -2
        plan_z[rid] = float(now)
        model_z[rid] = float(now)
        if lane:
            self.hashes.setdefault(lane_key, {})[rid] = str(lane)
        # Mirror the Lua's ZADD-then-EXPIRE ordering: TTL the plan zset,
        # the model zset, and the lane hash (when a lane is set) only after
        # the writes succeed. A failed claim must not leave a TTL behind,
        # and a successful one must, the same way the real Redis EVAL does
        # it.
        ttl = int(ttl)
        await self.expire(inflight_key, ttl)
        await self.expire(model_key, ttl)
        if lane:
            await self.expire(lane_key, ttl)
        return 1

    async def _shadow_bump_and_cool(self, keys, args):
        streak_key, cool_key = keys
        streak_ttl, base, cap, reason, now = args
        v = await self.incr(streak_key)
        await self.expire(streak_key, int(streak_ttl))
        # Mirror the Lua ladder: base * 2**(streak-1), capped.
        cooldown = int(base)
        if v > 1:
            doubled = int(base) * (2 ** (v - 1))
            cooldown = cap if doubled > cap else doubled
        # Match cool_down's "{reason}|{until}" shape and TTL.
        until = int(now) + cooldown
        await self.set(cool_key, f"{reason}|{until}",
                       ex=max(1, int(cooldown)))
        return v

    async def _shadow_bump_streak(self, keys, args):
        streak_key = keys[0]
        v = await self.incr(streak_key)
        await self.expire(streak_key, int(args[0]))
        return v

    async def _shadow_touch(self, keys, args):
        # Mirror the Lua's exact-source shadow for `_TOUCH` (issue #107).
        # The plan-zset ZADD result is the liveness signal: CH is not
        # optional (ZADD XX without CH always returns 0), and a dead slot
        # returns 0 here too, which is what makes touch() report the slot
        # gone so the heartbeat loop stops. Model zset ZADD is performed
        # unconditionally (the Lua does the same); the EXPIRE block is
        # gated on the plan zset saying the slot is still alive so a dead
        # slot never keeps any of the three TTLs alive.
        inflight_key, model_key, lane_key = keys
        rid, now, ttl = args
        changed = await self.zadd(inflight_key, {rid: now},
                                  xx=True, ch=True)
        await self.zadd(model_key, {rid: now}, xx=True, ch=True)
        if changed:
            # Same backstop window `_CLAIM` writes: refresh every TTL
            # only when the slot is still alive. EXPIRE on the absent
            # lane hash is already a no-op in real Redis, but the gate is
            # the documented contract — a dead slot never keeps a key alive.
            ttl = int(ttl)
            await self.expire(inflight_key, ttl)
            await self.expire(model_key, ttl)
            await self.expire(lane_key, ttl)
        return changed

    async def _shadow_touch_lease(self, keys, args):
        # Mirror the Lua: GET lease, return 0 if absent (and skip the EXPIRE
        # refresh on either key — the exact contract a "lease was just
        # dropped under us" caller wants); otherwise EXPIRE the lease AND
        # the per-plan reverse-index SET.
        lease_key = keys[0]
        ttl = int(args[0])
        v = await self.get(lease_key)
        if v is None:
            return 0
        await self.expire(lease_key, ttl)
        # Mirror Lua's `string.match(v, '^([^/]+)/')` exactly: only refresh
        # the per-plan reverse-index SET when the lease value carries a
        # `plan/` prefix. A bare lease value with no slash yields no match
        # in real Redis (no EXPIRE on the SET); `split` would have
        # refreshed a wrong SET key instead.
        m = re.match(r"^([^/]+)/", v)
        if m:
            await self.expire(f"sy:lease_plan:{m.group(1)}", ttl)
        return 1

    async def _shadow_drop_lease(self, keys, args):
        # Mirror the Lua: GET lease (to learn the plan), then DEL lease +
        # DEL injection marker + SREM from the SET.
        lease_key, inject_key = keys
        session = args[0]
        v = await self.get(lease_key)
        await self.delete(lease_key)
        await self.delete(inject_key)
        if v is not None:
            # Same Lua-vs-split drift as the touch_lease shadow: only SREM
            # the per-plan SET when the lease value carries a `plan/`
            # prefix.
            m = re.match(r"^([^/]+)/", v)
            if m:
                await self.srem(f"sy:lease_plan:{m.group(1)}", session)
        return 1

    async def _shadow_drop_lease_if(self, keys, args):
        # Mirror the Lua: GET lease, compare against ARGV[2] (the expected
        # ref the caller read at decision time), and only DEL + SREM +
        # clear the injection marker when the values match. On a mismatch
        # (or a missing lease) return 0 unchanged — same return value the
        # real script returns so `drop_lease_if`'s True/False mapping
        # stays in step. This is a value-CAS, not a touch-aware CAS — see
        # slots.py:184-213 and the picker overflow gate comment for which
        # interleavings the primitive guards and which it does not.
        lease_key, inject_key = keys
        session, expected = args
        v = await self.get(lease_key)
        if v is None or v != expected:
            return 0
        await self.delete(lease_key)
        await self.delete(inject_key)
        m = re.match(r"^([^/]+)/", v)
        if m:
            await self.srem(f"sy:lease_plan:{m.group(1)}", session)
        return 1

    async def _shadow_set_lease(self, keys, args):
        # Mirror the Lua (cross-plan hygiene, issue #109): read the old
        # lease value, parse the OLD plan prefix via `^([^/]+)/` (the same
        # regex the Lua uses — `split` would have refreshed the wrong SET
        # key on a bare lease value with no slash), and SREM the session
        # from the OLD plan's reverse-index SET when the prefix differs
        # from the NEW plan's prefix. Then SADD to the NEW SET, SET the
        # lease with EX, EXPIRE the SET. The SREM is gated on
        # `old ~= new`, so a same-plan re-set never touches the wrong SET
        # and the SADD returns 0 on an already-present member.
        #
        # Without this SREM a fake-backend test that exercises a cross-plan
        # set_lease (the matrix in `tests/test_drain.py`, picker body-walk
        # re-lease paths, etc.) silently models the exact bug this PR
        # closes — the Lua pin in `tests/test_slots_lua.py::test_set_lease_
        # cross_plan_srem` is the only place the cross-plan SREM was being
        # asserted, and the matrix suite was diverging from it.
        lease_key, set_key = keys
        session, ref, ttl = args
        old = await self.get(lease_key)
        old_plan = re.match(r"^([^/]+)/", old or "")
        new_plan = re.match(r"^([^/]+)/", ref or "")
        if old_plan and new_plan and old_plan.group(1) != new_plan.group(1):
            await self.srem(f"sy:lease_plan:{old_plan.group(1)}", session)
        added = await self.sadd(set_key, session)
        await self.set(lease_key, ref, ex=int(ttl))
        await self.expire(set_key, int(ttl))
        return added

    # -- the Lua shadows ---------------------------------------------------
    # Real Redis compiles each script once; the fake matches by exact-source
    # identity against the constants in `switchyard.slots`. The fast suite
    # already requires redis-py via slots.py, so importing the constants here
    # is safe. Anything else raises loudly rather than silently falling through
    # to the CLAIM fallback — a typo in a script constant or a renamed copy
    # should never look like a working claim.
    def register_script(self, src):
        # Dispatch via the module-level `_SCRIPT_SOURCE_TO_SHADOW` map.
        # Each shadow is a `_shadow_*` method on this class; looking up the
        # bound method on self keeps `register_script` at one branch and the
        # per-script bodies under the C901 ceiling even as new scripts land.
        method_name = _SCRIPT_SOURCE_TO_SHADOW.get(src)
        if method_name is None:
            raise ValueError(
                f"FakeRedis.register_script: unknown Lua source "
                f"({len(src)} chars). Add an exact-source shadow in "
                "tests/fake_redis.py before importing the new script "
                "into slots.py."
            )
        return getattr(self, method_name)
