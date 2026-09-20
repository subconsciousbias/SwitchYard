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

    def hset(self, key, mapping=None, **_):
        self.ops.append(("hset", key, mapping or {})); return self

    def expire(self, key, ttl):
        return self

    async def execute(self):
        for op in self.ops:
            if op[0] == "hincrbyfloat":
                _, key, field, amount = op
                h = self.store.hashes.setdefault(key, {})
                h[field] = float(h.get(field, 0)) + float(amount)
            elif op[0] == "hset":
                _, key, mapping = op
                self.store.hashes.setdefault(key, {}).update(
                    {k: str(v) for k, v in mapping.items()}
                )
        self.ops.clear()
        return []


class FakeRedis:
    def __init__(self):
        self.strings: dict[str, tuple[str, float | None]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.hashes: dict[str, dict[str, str]] = {}

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
    async def zrem(self, key, member):
        self.zsets.get(key, {}).pop(member, None)

    async def zcard(self, key):
        return len(self.zsets.get(key, {}))

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

    async def incr(self, key):
        val = int(float((await self.get(key)) or 0)) + 1
        await self.set(key, str(val))
        return val

    async def hincrbyfloat(self, key, field, amount):
        h = self.hashes.setdefault(key, {})
        h[field] = str(float(h.get(field, 0)) + float(amount))
        return float(h[field])

    def pipeline(self):
        return FakePipeline(self)

    async def ping(self):
        return True

    # -- the claim script --------------------------------------------------
    def register_script(self, _src):
        async def claim(keys, args):
            inflight_key, cool_key, model_key = keys
            rid, now, plan_cap, stale_before, _ttl, model_cap = args
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
            return 1
        return claim
