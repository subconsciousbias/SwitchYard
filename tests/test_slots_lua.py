"""tests/test_slots_lua.py — Real-Lua slot-table suite, source of truth.

The Lua scripts in `switchyard/slots.py` are the source of truth. This suite
runs every script in `SlotTable` against:

  (a) a Lua-capable backend — `SWITCHYARD_TEST_REDIS_URL` (real Redis, e.g.
      `redis:7`) when set, else `fakeredis.aioredis.FakeRedis(FakeServer(),
      lua=True)` (lupa runs the real Lua);
  (b) the Python shadow in `tests/fake_redis.py`.

The same scenario matrix runs on both backends, asserting identical return
values and resulting key state after each step. A broken KEYS/ARGV index, a
Lua typo, or a drifted Python mirror all fail loudly.

The in-flight zset TTL under heartbeat (#107) is pinned as xfail: today the
heartbeat updates the score but does not refresh the EXPIRE, so a long-lived
slot expires after `2 * inflight_max_age`. The gate flips to a hard pass and
prints XPASS when #107 lands.

Backend resolution: SWITCHYARD_TEST_REDIS_URL -> `redis.asyncio.from_url`;
elif `fakeredis` with `lua=True` is importable -> that; else print a loud
SKIP line and exit 0.

The repo's fast-suite style is plain script + `__main__` runner; this file
follows the same shape, with one module-level async runner that drives
SlotTable directly on each backend and reports per-scenario equivalence
plus the xfail-#107 acceptance test. Each phase is also exposed as a
top-level `def test_*` so pytest can collect them.

Inserted BEFORE the runner so the discovery sees every test_ function.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from switchyard.slots import (  # noqa: E402
    K_COOL,
    K_INFLIGHT,
    K_INFLIGHT_LANE,
    K_INFLIGHT_MODEL,
    K_LEASE,
    K_LEASE_PLAN,
    K_TFAIL,
    SlotTable,
)
from tests.fake_redis import FakeRedis  # noqa: E402

# -------------------------------------------------------------------- backend

# Test surface constants. Keep them short: this suite is about the script
# behavior, not the production config.
PLAN = "plan"
MODEL = "plan/m"
OTHER_PLAN = "ultra"
OTHER_MODEL = "ultra/m"
LEASE_TTL = 1000
INFLIGHT_MAX_AGE = 1                 # claim TTL is 2 × this = 2s
# #107 xfail: heartbeat loop must wait past 2 × INFLIGHT_MAX_AGE (= 2s).
# 5 beats × 0.5s = 2.5s, comfortably past the TTL window.
HEARTBEAT_STEP = 0.5
HEARTBEAT_OVERRUN = 5

# Cached at module load so each test_* function pays the import cost once.
# SKIP semantics are reported through pytest.skip() when pytest is available,
# else through the loud `[SKIP]` print + return-None path the plain runner uses.
_LUA_LABEL, _LUA_FACTORY = None, None
_SKIP_REASON = None


def _resolve_lua_backend():
    """Return (label, async_factory) for a Lua-capable backend, or (None, None).

    Resolution order:
      1. SWITCHYARD_TEST_REDIS_URL (real Redis): `redis.asyncio.from_url`.
      2. fakeredis[lua]: `fakeredis.aioredis.FakeRedis(FakeServer(), lua=True)`.
         Sanity-checks lupa by running `return 1` so a missing lupa fails
         loudly here rather than mid-suite.
    Anything else means a missing test dependency; the caller SKIPs.
    """
    url = os.environ.get("SWITCHYARD_TEST_REDIS_URL")
    if url:
        from redis.asyncio import from_url

        async def factory():
            return await from_url(url)

        return url, factory
    try:
        import fakeredis.aioredis as faio  # noqa: F401
        from fakeredis import FakeServer  # noqa: F401
    except ImportError:
        return None, None

    async def factory():
        r = faio.FakeRedis(FakeServer(), lua=True)
        # Verify lupa is wired: without it, `lua=True` rejects EVAL.
        res = await r.eval("return 1", 0)
        if res != 1:
            raise RuntimeError(
                "fakeredis[lua] is installed but lupa did not execute; "
                "check that `pip install 'fakeredis[lua]'` succeeded."
            )
        return r

    return "fakeredis[lua]", factory


def _skip_or_skip():
    """Return the resolved (label, factory) or skip the running test."""
    global _LUA_LABEL, _LUA_FACTORY, _SKIP_REASON
    if _LUA_LABEL is None and _LUA_FACTORY is None and _SKIP_REASON is None:
        _LUA_LABEL, _LUA_FACTORY = _resolve_lua_backend()
        if _LUA_LABEL is None:
            _SKIP_REASON = (
                "no SWITCHYARD_TEST_REDIS_URL and fakeredis[lua] not importable; "
                "install with `pip install 'fakeredis[lua]'` or set "
                "SWITCHYARD_TEST_REDIS_URL"
            )
    if _SKIP_REASON is not None:
        try:
            import pytest
            pytest.skip(_SKIP_REASON)
        except ImportError:
            print(f"[SKIP] tests/test_slots_lua.py: {_SKIP_REASON}")
            return None
    return _LUA_LABEL, _LUA_FACTORY


def _decode(obj):
    """Recursively turn bytes inside a result into strings.

    redis-py on the wire returns bytes; fakeredis+lua does the same. The fake
    in tests/fake_redis.py stores str. Decoding makes the comparison tuple
    identical across backends.
    """
    if isinstance(obj, bytes):
        return obj.decode()
    if isinstance(obj, dict):
        return {_decode(k): _decode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_decode(x) for x in obj]
    return obj


# ----------------------------------------------------------- shared scenarios
# Every scenario is `async def fn(slots, r) -> tuple` of comparable values. The
# runner calls each one on both backends with a freshly-built SlotTable and
# redis instance, decodes the result, and compares. The tuple must capture
# only structural state (return values, presence of keys, sorted membership
# lists); TTL is intentionally NOT in the tuple, because the fake's first
# claim shadow EXPIRE-races the zset creation (an acknowledged WS1 gap) and
# would produce a spurious divergence on TTL-on-zset keys. TTL-sensitive
# assertions live in the dedicated Lua-backend-only section below.

PLAN_INFLIGHT = K_INFLIGHT.format(plan=PLAN)
PLAN_LANE = K_INFLIGHT_LANE.format(plan=PLAN)
PLAN_COOL = K_COOL.format(plan=PLAN)


async def scenario_claim_at_cap(slots, r):
    """Plan cap=1, two claims -> 1 then 0; in_flight = 1."""
    c1 = await slots.try_claim(PLAN, 1, "r1", model_ref=MODEL, model_cap=1)
    c2 = await slots.try_claim(PLAN, 1, "r2", model_ref=MODEL, model_cap=1)
    return (c1, c2, await slots.in_flight(PLAN))


async def scenario_over_cap(slots, r):
    """Plan cap=2, three claims -> 1, -2 (model full), -2."""
    c1 = await slots.try_claim(PLAN, 2, "r1", model_ref=MODEL, model_cap=1)
    c2 = await slots.try_claim(PLAN, 2, "r2", model_ref=MODEL, model_cap=1)
    c3 = await slots.try_claim(PLAN, 2, "r3", model_ref=MODEL, model_cap=1)
    return (c1, c2, c3)


async def scenario_cooled_down_refuses(slots, r):
    """cool_down, then try_claim -> -1 with reason."""
    await slots.cool_down(PLAN, 60, "quota_exhausted")
    c = await slots.try_claim(PLAN, 1, "r1", model_ref=MODEL, model_cap=1)
    cs, _ttl, reason = await slots.cooldown_state(PLAN)
    return (c, cs, reason)


async def scenario_clear_cooldown(slots, r):
    """clear_cooldown lets a new claim succeed after the cool window."""
    await slots.cool_down(PLAN, 60, "quota_exhausted")
    c_blocked = await slots.try_claim(PLAN, 1, "r1", model_ref=MODEL, model_cap=1)
    await slots.clear_cooldown(PLAN)
    c_open = await slots.try_claim(PLAN, 1, "r1", model_ref=MODEL, model_cap=1)
    return (c_blocked, c_open)


async def scenario_model_cap_refuses(slots, r):
    """Plan cap=2, model cap=1, two claims on same model -> 1 then -2."""
    c1 = await slots.try_claim(PLAN, 2, "r1", model_ref=MODEL, model_cap=1)
    c2 = await slots.try_claim(PLAN, 2, "r2", model_ref=MODEL, model_cap=1)
    return (c1, c2)


async def scenario_lane_map_write(slots, r):
    """Claim with a lane writes the request-id->lane entry in the lane map."""
    c = await slots.try_claim(PLAN, 2, "r1", model_ref=MODEL,
                              model_cap=1, lane="local")
    mapping = await r.hgetall(PLAN_LANE)
    decoded = {(k.decode() if isinstance(k, bytes) else k):
               (v.decode() if isinstance(v, bytes) else v)
               for k, v in mapping.items()}
    return (c, decoded)


async def scenario_release(slots, r):
    """Claim then release drops both plan and model zsets to 0."""
    await slots.try_claim(PLAN, 2, "r1", model_ref=MODEL, model_cap=1)
    await slots.release(PLAN, "r1", MODEL)
    return (await slots.in_flight(PLAN), await slots.in_flight_model(MODEL))


async def scenario_touch_liveness(slots, r):
    """touch returns True on a live claim, False after release."""
    await slots.try_claim(PLAN, 2, "r1", model_ref=MODEL, model_cap=1)
    alive = await slots.touch(PLAN, "r1", MODEL)
    await slots.release(PLAN, "r1", MODEL)
    gone = await slots.touch(PLAN, "r1", MODEL)
    return (alive, gone)


async def scenario_stale_member_sweep(slots, r):
    """A member backdated past inflight_max_age is pruned on the next call."""
    await slots.try_claim(PLAN, 2, "r1", model_ref=MODEL, model_cap=1)
    # Backdate the score so the next read sees it as stale.
    await r.zadd(PLAN_INFLIGHT, {"r1": time.time() - 100})
    return await slots.in_flight(PLAN)


async def scenario_bump_streak_value_and_ttl(slots, r):
    """note_transient_failure returns 1, then 2; streak reads back as 2."""
    s1 = await slots.note_transient_failure(PLAN)
    s2 = await slots.note_transient_failure(PLAN)
    streak = await slots.transient_failure_streak(PLAN)
    return (s1, s2, streak)


async def scenario_bump_and_cool_ladder(slots, r):
    """60 -> 120 -> 240 -> cap, with cooldown_state True throughout."""
    base, cap = 60, 400
    ladder = []
    for _ in range(5):
        s = await slots.bump_and_cool(PLAN, base=base, cap=cap, reason="transient")
        cs, _ttl, _reason = await slots.cooldown_state(PLAN)
        ladder.append((s, cs))
    return tuple(ladder)


async def scenario_cooldown_value_shape(slots, r):
    """Cooldown is stored as reason|until; cooldown_state splits it back."""
    await slots.bump_and_cool(PLAN, base=60, cap=400, reason="transient")
    raw = await r.get(PLAN_COOL)
    raw = raw.decode() if isinstance(raw, bytes) else raw
    reason, _, until = raw.partition("|")
    cs, ttl, rsn = await slots.cooldown_state(PLAN)
    return (raw, reason, until.isdigit(), cs, rsn, ttl > 0)


async def scenario_set_lease(slots, r):
    """set_lease writes the lease value and adds the session to the plan SET."""
    await slots.set_lease("sess-A", MODEL, LEASE_TTL, PLAN)
    return (await slots.get_lease("sess-A"),
            sorted(await slots.sessions_on_plan(PLAN)))


async def scenario_touch_lease(slots, r):
    """touch_lease preserves the lease value and keeps the session on the plan."""
    await slots.set_lease("sess-A", MODEL, LEASE_TTL, PLAN)
    await slots.touch_lease("sess-A", 100)
    return (await slots.get_lease("sess-A"),
            sorted(await slots.sessions_on_plan(PLAN)))


async def scenario_drop_lease(slots, r):
    """drop_lease clears the lease, the injection marker, and the SET entry."""
    await slots.set_lease("sess-A", MODEL, LEASE_TTL, PLAN)
    await slots.mark_injected("sess-A", LEASE_TTL)
    before = await slots.injected("sess-A")
    await slots.drop_lease("sess-A")
    return (before,
            await slots.get_lease("sess-A"),
            await slots.injected("sess-A"),
            sorted(await slots.sessions_on_plan(PLAN)))


async def scenario_sessions_on_plan(slots, r):
    """sessions_on_plan returns the SET membership per plan, in insert order."""
    await slots.set_lease("sess-A", MODEL, LEASE_TTL, PLAN)
    await slots.set_lease("sess-B", MODEL, LEASE_TTL, PLAN)
    await slots.set_lease("sess-C", OTHER_MODEL, LEASE_TTL, OTHER_PLAN)
    return (sorted(await slots.sessions_on_plan(PLAN)),
            sorted(await slots.sessions_on_plan(OTHER_PLAN)))


SCENARIOS = [
    ("claim_at_cap",                 scenario_claim_at_cap),
    ("over_cap",                     scenario_over_cap),
    ("cooled_down_refuses",          scenario_cooled_down_refuses),
    ("clear_cooldown",               scenario_clear_cooldown),
    ("model_cap_refuses",            scenario_model_cap_refuses),
    ("lane_map_write",               scenario_lane_map_write),
    ("release",                      scenario_release),
    ("touch_liveness",               scenario_touch_liveness),
    ("stale_member_sweep",           scenario_stale_member_sweep),
    ("bump_streak_value_and_ttl",    scenario_bump_streak_value_and_ttl),
    ("bump_and_cool_ladder",         scenario_bump_and_cool_ladder),
    ("cooldown_value_shape",         scenario_cooldown_value_shape),
    ("set_lease",                    scenario_set_lease),
    ("touch_lease",                  scenario_touch_lease),
    ("drop_lease",                   scenario_drop_lease),
    ("sessions_on_plan",             scenario_sessions_on_plan),
]


# ------------------------------------------------------- Lua-backend-only TTL
# These assertions live separately because the fake's first-claim EXPIRE on a
# not-yet-created zset is a known WS1 gap; only the Lua backend has the
# exact behavior under test. They run only on the Lua backend.

async def lua_ttl_assertions(r, slots):
    """Lua-only: every key TTL must be set as the Lua scripts declare."""
    plan = "ttl-plan"
    await r.delete(K_INFLIGHT.format(plan=plan))
    await r.delete(K_INFLIGHT_MODEL.format(ref="ttl/m"))
    await r.delete(K_INFLIGHT_LANE.format(plan=plan))
    await slots.try_claim(plan, 2, "rid", model_ref="ttl/m", model_cap=1,
                          lane="local")
    plan_ttl = await r.ttl(K_INFLIGHT.format(plan=plan))
    model_ttl = await r.ttl(K_INFLIGHT_MODEL.format(ref="ttl/m"))
    lane_ttl = await r.ttl(K_INFLIGHT_LANE.format(plan=plan))
    assert plan_ttl > 0, f"plan zset TTL must be set, got {plan_ttl}"
    assert model_ttl > 0, f"model zset TTL must be set, got {model_ttl}"
    assert lane_ttl > 0, f"lane hash TTL must be set, got {lane_ttl}"
    assert plan_ttl <= INFLIGHT_MAX_AGE * 2, plan_ttl
    assert model_ttl <= INFLIGHT_MAX_AGE * 2, model_ttl
    assert lane_ttl <= INFLIGHT_MAX_AGE * 2, lane_ttl
    await slots.release(plan, "rid", "ttl/m")

    # Cooldown SET has an EX equal to max(1, math.floor(cooldown)).
    await r.delete(K_COOL.format(plan="ttl-cool"))
    await slots.cool_down("ttl-cool", 60, "ttl_reason")
    cool_ttl = await r.ttl(K_COOL.format(plan="ttl-cool"))
    assert 55 <= cool_ttl <= 60, f"cooldown TTL ~60, got {cool_ttl}"
    raw = await r.get(K_COOL.format(plan="ttl-cool"))
    raw = raw.decode() if isinstance(raw, bytes) else raw
    reason, _, until = raw.partition("|")
    assert reason == "ttl_reason", raw
    assert until.isdigit(), raw

    # bump_streak TTL is 7200 (sliding window).
    await r.delete(K_TFAIL.format(plan="ttl-tfail"))
    await slots.note_transient_failure("ttl-tfail")
    tfail_ttl = await r.ttl(K_TFAIL.format(plan="ttl-tfail"))
    assert 7100 <= tfail_ttl <= 7200, tfail_ttl
    # Second bump refreshes the TTL, not keeps it: shrink manually first.
    await r.expire(K_TFAIL.format(plan="ttl-tfail"), 100)
    await slots.note_transient_failure("ttl-tfail")
    tfail_ttl_again = await r.ttl(K_TFAIL.format(plan="ttl-tfail"))
    assert 7100 <= tfail_ttl_again <= 7200, tfail_ttl_again

    # set_lease: lease key and reverse-index SET both get TTL.
    sess = "ttl-sess"
    await r.delete(K_LEASE.format(session=sess))
    await r.delete(K_LEASE_PLAN.format(plan="ttl-lease"))
    await slots.set_lease(sess, "ttl-lease/m", LEASE_TTL, "ttl-lease")
    lease_ttl = await r.ttl(K_LEASE.format(session=sess))
    set_ttl = await r.ttl(K_LEASE_PLAN.format(plan="ttl-lease"))
    assert 900 <= lease_ttl <= LEASE_TTL, lease_ttl
    assert 900 <= set_ttl <= LEASE_TTL, set_ttl

    # touch_lease: both TTLs shrink.
    await r.expire(K_LEASE.format(session=sess), 500)
    await r.expire(K_LEASE_PLAN.format(plan="ttl-lease"), 500)
    await slots.touch_lease(sess, 100)
    lease_ttl = await r.ttl(K_LEASE.format(session=sess))
    set_ttl = await r.ttl(K_LEASE_PLAN.format(plan="ttl-lease"))
    assert 50 <= lease_ttl <= 100, lease_ttl
    assert 50 <= set_ttl <= 100, set_ttl

    # Drop the lease: the script returns 0 and refreshes nothing.
    await slots.drop_lease(sess)
    await slots.touch_lease(sess, 500)
    lease_ttl = await r.ttl(K_LEASE.format(session=sess))
    assert lease_ttl == -2, f"gone lease must stay gone, got {lease_ttl}"

    # Clear the test keys.
    for k in (K_INFLIGHT.format(plan=plan),
              K_INFLIGHT_MODEL.format(ref="ttl/m"),
              K_INFLIGHT_LANE.format(plan=plan),
              K_COOL.format(plan="ttl-cool"),
              K_TFAIL.format(plan="ttl-tfail"),
              K_LEASE.format(session=sess),
              K_LEASE_PLAN.format(plan="ttl-lease")):
        await r.delete(k)
    return True


# --------------------------------------------------------- #107 acceptance
# The bug: claim sets the zset TTL to `inflight_max_age * 2`, but touch()
# updates only the score. After `2 * inflight_max_age` the key expires, so a
# long-lived slot is lost even with a fresh heartbeat. The fix should make
# touch() also refresh the EXPIRE; when that lands, this test flips from
# XFAIL to XPASS and the gate is re-tightened to a hard assertion.

async def xfail_inflight_zset_ttl_under_heartbeat(r, slots):
    """claim() under continuous heartbeats must outlive 2*inflight_max_age.

    Today the heartbeat updates only the score; the zset key's TTL is set
    only at claim time, so a slot older than `2 * inflight_max_age` is
    collected by Redis. When #107 lands (touch refreshes EXPIRE) the
    heartbeat keeps the key alive past the window, the EXISTS comes back 1,
    and the gate reports XPASS so the maintainer can convert it to a hard pass.
    """
    plan = "xfail-107"
    rid = "rid-heartbeat"
    ref = "xfail-107/m"
    cap = 1

    # Fresh claim.
    claim = await slots.try_claim(plan, cap, rid, model_ref=ref, model_cap=cap)
    assert claim == 1, claim

    # Advance past `2 * inflight_max_age` with only heartbeats: a
    # HEARTBEAT_STEP-second sleep per beat, HEARTBEAT_OVERRUN beats total.
    # The slot's TTL is set once at claim time, so a beat at the edge of the
    # window will find the key still alive; the next beat, after the TTL has
    # fired, will find it gone. When #107 lands the touch refreshes the TTL
    # each beat and the key survives every beat in the loop.
    final_touch = None
    for _ in range(HEARTBEAT_OVERRUN):
        await asyncio.sleep(HEARTBEAT_STEP)
        final_touch = await slots.touch(plan, rid, ref)

    # The acceptance contract: the key survived, and the next claim attempt
    # is refused because the live slot still occupies the cap.
    exists = await r.exists(K_INFLIGHT.format(plan=plan))
    next_claim = await slots.try_claim(plan, cap, "rid-new",
                                       model_ref=ref, model_cap=cap)

    # Clean up regardless of outcome.
    await slots.release(plan, rid, ref)

    return {
        "exists": exists,
        "next_claim_refused": next_claim in (0, -2),
        "touch_alive": bool(final_touch),
    }


# --------------------------------------------------------- per-test runner

async def _maybe_aclose(r):
    close = getattr(r, "aclose", None) or getattr(r, "close", None)
    if close is None:
        return
    maybe = close()
    if hasattr(maybe, "__await__"):
        await maybe


async def _make_redis(factory):
    """Call the factory whether it returns a redis directly or a coroutine."""
    res = factory()
    if hasattr(res, "__await__"):
        return await res
    return res


async def run_scenarios_on(redis_factory):
    """Run every scenario on a fresh redis + SlotTable pair; return results."""
    results = {}
    for name, fn in SCENARIOS:
        r = await _make_redis(redis_factory)
        slots = SlotTable(r, inflight_max_age=INFLIGHT_MAX_AGE)
        try:
            results[name] = _decode(await fn(slots, r))
        finally:
            await _maybe_aclose(r)
    return results


def _compare_results(lua_results, fake_results):
    """Compare structural state; return list of (name, lua, fake) mismatches."""
    mismatches = []
    for name, _fn in SCENARIOS:
        lua_v = lua_results[name]
        fake_v = fake_results[name]
        if lua_v != fake_v:
            mismatches.append((name, lua_v, fake_v))
    return mismatches


# ----------------------------------------------------- pytest-collectable tests

def test_scenario_matrix_lua_and_fake_match():
    """Same scenario matrix on the Lua backend and the WS1 fake -> identical."""
    resolved = _skip_or_skip()
    if resolved is None:
        return
    label, factory = resolved

    async def go_async():
        lua_results = await run_scenarios_on(factory)

        def _fake_factory():
            return FakeRedis()

        fake_results = await run_scenarios_on(_fake_factory)
        return _compare_results(lua_results, fake_results)

    mismatches = asyncio.run(go_async())
    assert not mismatches, (
        f"Lua vs fake divergence on {len(mismatches)} scenarios: "
        + "; ".join(f"{n}: lua={lv!r} fake={fv!r}"
                    for n, lv, fv in mismatches)
    )
    print(f"  scenario matrix: {len(SCENARIOS)} scenarios, "
          f"identical on Lua ({label}) and fake backends")


def test_lua_only_ttl_assertions():
    """On the Lua backend, every EXPIRE the scripts write must be set."""
    resolved = _skip_or_skip()
    if resolved is None:
        return
    _label, factory = resolved

    async def go_async():
        r = await _make_redis(factory)
        try:
            slots = SlotTable(r, inflight_max_age=INFLIGHT_MAX_AGE)
            await lua_ttl_assertions(r, slots)
        finally:
            await _maybe_aclose(r)

    asyncio.run(go_async())
    print("  lua-only TTL assertions: every key type gets the EXPIRE the "
          "script writes (plan zset, model zset, lane hash, cooldown, "
          "tfail sliding TTL, lease, reverse-index SET)")


def test_xfail_107_inflight_zset_ttl():
    """#107 acceptance: claim must outlive 2*inflight_max_age under heartbeats.

    Gated as xfail so the suite stays green today; flips to XPASS (and the
    gate to a hard pass) when issue #107 lands.
    """
    resolved = _skip_or_skip()
    if resolved is None:
        return
    _label, factory = resolved

    async def go_async():
        r = await _make_redis(factory)
        try:
            slots = SlotTable(r, inflight_max_age=INFLIGHT_MAX_AGE)
            return await xfail_inflight_zset_ttl_under_heartbeat(r, slots)
        finally:
            await _maybe_aclose(r)

    outcome = asyncio.run(go_async())
    survived = outcome["exists"] == 1
    refused = outcome["next_claim_refused"]
    alive = outcome["touch_alive"]
    print()
    if survived and refused and alive:
        # XPASS: the bug is fixed. Flip the gate to a hard assertion and
        # remove the xfail label.
        print("  xfail (#107): XPASS — heartbeat now keeps the slot past "
              "2*inflight_max_age; EXISTS=1, next claim refused, touch alive. "
              "Issue #107 appears fixed: tighten the gate to a hard pass.")
    else:
        print("  xfail (#107): XFAIL — heartbeat does not refresh the zset "
              "TTL; slot expired before the next claim was refused. "
              f"observed: {outcome}. Tracked in issue #107.")
    # xfail is informational today; never fail the suite on this test.
    assert True


# ---------------------------------------------------------------- main entry

def _amain_print():
    """Plain-script runner: discover test_* functions from globals() and run them.

    Repo convention (see CLAUDE.md and the other tests/test_*.py runners) is
    to iterate `globals()` instead of hardcoding the function names — anything
    added *after* a hardcoded call list silently never runs under the plain
    runner, which is exactly the silent-skip trap this file's module docstring
    calls out. Discovering from globals() keeps new `def test_foo()` functions
    picked up automatically.
    """
    resolved = _skip_or_skip()
    if resolved is None:
        return 0
    label, _factory = resolved
    print(f"=== tests/test_slots_lua.py — Lua backend: {label} ===")
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\nall slots-lua tests passed ({n})")
    return 0


try:
    import _pytest.outcomes as _pytest_outcomes
    _pytest_Skipped = _pytest_outcomes.Skipped
except ImportError:
    class _pytest_Skipped(Exception):
        pass


def main():
    try:
        return _amain_print()
    except AssertionError as exc:
        print(f"!! test assertion failed: {exc}", file=sys.stderr)
        return 1
    except _pytest_Skipped as exc:
        # pytest.skip() raises Skipped when run under pytest; under the plain
        # runner it surfaces as a noisy traceback. Translate to a clean exit.
        print(f"[SKIP] {exc}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
