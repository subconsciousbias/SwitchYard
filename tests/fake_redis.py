"""Just enough async Redis to exercise the routing brain in-process.

`register_script` returns a Python reference implementation of the Lua claim
script in switchyard/slots.py. The two must stay in step; the integration test
against a real Redis (tests/test_live_redis.py) is what catches drift.
"""
from __future__ import annotations

import time


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
        return self

    async def execute(self):
        """Returns one result per queued command, in order, like a real pipeline.

        It used to return [], so a caller reading `results[0]` silently got
        nothing — which is exactly how a heartbeat whose liveness check never
        worked passed its tests.
        """
        results = []
        for op in self.ops:
            if op[0] == "hincrbyfloat":
                _, key, field, amount = op
                h = self.store.hashes.setdefault(key, {})
                h[field] = float(h.get(field, 0)) + float(amount)
                results.append(h[field])
            elif op[0] == "hgetall":
                _, key = op
                results.append(self.store.hashes.get(key, {}))
            elif op[0] == "hset":
                _, key, mapping = op
                self.store.hashes.setdefault(key, {}).update(
                    {k: str(v) for k, v in mapping.items()}
                )
                results.append(len(mapping))
            elif op[0] == "zadd":
                _, key, mapping, xx, ch = op
                results.append(await self.store.zadd(key, mapping, xx=xx, ch=ch))
            elif op[0] == "incr":
                _, key = op
                results.append(await self.store.incr(key))
            elif op[0] == "pfadd":
                _, key, members = op
                results.append(await self.store.pfadd(key, *members))
            elif op[0] == "pfcount":
                _, keys = op
                results.append(await self.store.pfcount(*keys))
        self.ops.clear()
        return results


class FakeRedis:
    def __init__(self):
        self.strings: dict[str, tuple[str, float | None]] = {}
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

    # -- strings -----------------------------------------------------------
    async def set(self, key, value, ex=None):
        self.strings[key] = (value, time.time() + ex if ex else None)

    async def get(self, key):
        v = self.strings.get(key)
        if not v:
            return None
        value, expires = v
        if expires and expires < time.time():
            del self.strings[key]
            return None
        return value

    async def ttl(self, key):
        v = self.strings.get(key)
        if not v or not v[1]:
            return -1
        return max(0, int(v[1] - time.time()))

    async def delete(self, key):
        self.strings.pop(key, None)

    async def expire(self, key, ttl):
        if key in self.strings:
            self.strings[key] = (self.strings[key][0], time.time() + ttl)

    # -- zsets -------------------------------------------------------------
    async def zadd(self, key, mapping, xx=False, ch=False, **_):
        """Real ZADD counts members *added*; with CH it counts added OR updated.
        The difference is the whole liveness signal behind touch(), so the fake
        has to model it or a broken heartbeat looks fine in tests."""
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
        self.zsets.get(key, {}).pop(member, None)

    async def zcard(self, key):
        return len(self.zsets.get(key, {}))

    async def zrange(self, key, start, stop):
        members = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
        if stop == -1:
            stop = len(members)
        else:
            stop += 1
        return [m for m, _ in members[start:stop]]

    async def zremrangebyscore(self, key, lo, hi):
        z = self.zsets.get(key, {})
        for m in [m for m, s in z.items() if s <= float(hi)]:
            del z[m]

    # -- hashes ------------------------------------------------------------
    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, mapping=None, **_):
        self.hashes.setdefault(key, {}).update({k: str(v) for k, v in (mapping or {}).items()})

    async def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    async def hdel(self, key, field):
        return 1 if self.hashes.get(key, {}).pop(field, None) is not None else 0

    async def incr(self, key):
        val = int(float((await self.get(key)) or 0)) + 1
        await self.set(key, str(val))
        return val

    async def hincrbyfloat(self, key, field, amount):
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
        s = self.sets.setdefault(key, set())
        added = 0
        for m in members:
            if m not in s:
                s.add(m)
                added += 1
        return added

    async def srem(self, key, *members):
        s = self.sets.get(key, set())
        removed = 0
        for m in members:
            if m in s:
                s.discard(m)
                removed += 1
        return removed

    async def smembers(self, key):
        return list(self.sets.get(key, set()))

    async def scard(self, key):
        return len(self.sets.get(key, set()))

    async def sismember(self, key, member):
        return 1 if member in self.sets.get(key, set()) else 0

    def pipeline(self):
        return FakePipeline(self)

    async def ping(self):
        return True

    # -- the claim script --------------------------------------------------
    def register_script(self, src):
        # Dispatch on the Lua source so the fake can shadow both the original
        # claim script, the two transient-failure scripts added in slots.py,
        # and the two lease scripts added for the drain flow. Real Redis
        # compiles each script once; the fake has to match by hand.
        if "BUMP_AND_COOL" in src:
            async def bump_and_cool(keys, args):
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
            return bump_and_cool
        if "BUMP_STREAK" in src:
            async def bump_streak(keys, args):
                streak_key = keys[0]
                v = await self.incr(streak_key)
                await self.expire(streak_key, int(args[0]))
                return v
            return bump_streak
        if "TOUCH_LEASE" in src:
            async def touch_lease(keys, args):
                # Mirror the Lua: GET lease, return 0 if absent (and skip
                # the SET TTL refresh, the exact contract a "lease was just
                # dropped under us" caller wants); otherwise EXPIRE both
                # the lease and the per-plan SET.
                lease_key = keys[0]
                ttl = int(args[0])
                v = await self.get(lease_key)
                if v is None:
                    return 0
                await self.expire(lease_key, ttl)
                # The SET is just a Python set in this fake; the real Redis
                # SET TTL is what the Lua EXPIRE on `sy:lease_plan:{plan}`
                # would touch. Nothing for the fake to do here.
                return 1
            return touch_lease
        if "DROP_LEASE" in src:
            async def drop_lease(keys, args):
                # Mirror the Lua: GET lease (to learn the plan), then
                # DEL lease + DEL injection marker + SREM from the SET.
                lease_key, inject_key = keys
                session = args[0]
                v = await self.get(lease_key)
                await self.delete(lease_key)
                await self.delete(inject_key)
                if v is not None:
                    plan_key = v.split("/", 1)[0]
                    await self.srem(f"sy:lease_plan:{plan_key}", session)
                return 1
            return drop_lease
        if "SET_LEASE" in src:
            async def set_lease(keys, args):
                # Mirror the Lua: SADD to the SET, SET the lease with EX,
                # EXPIRE the SET. The fake's `expire` only updates string
                # keys today — same pre-existing gap the TOUCH_LEASE shadow
                # already calls out. Adding SET-key TTL tracking would let
                # a future test simulate a partial-failure mid-script; for
                # now the fake faithfully reproduces the *bug* shape (the
                # membership lands but the SET TTL is a no-op in the fake),
                # so any test that wants the "atomicity holds across
                # failures" property needs to assert the script's behaviour
                # in real Redis. The test we add here pins the membership
                # contract only.
                lease_key, set_key = keys
                session, ref, ttl = args
                added = await self.sadd(set_key, session)
                await self.set(lease_key, ref, ex=int(ttl))
                # SET EXPIRE — see comment above; the fake's expire is
                # string-only and silently does nothing for SET keys today.
                # The contract under test is the membership, not the TTL.
                return added
            return set_lease

        async def claim(keys, args):
            inflight_key, cool_key, model_key, lane_key = keys
            rid, now, plan_cap, stale_before, _ttl, model_cap, lane = args
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
            return 1
        return claim
