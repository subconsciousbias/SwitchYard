"""Regression tests for SwitchyardHandler._apply_verdict.

Every coollable verdict used to raise `UnboundLocalError: cannot access local
variable 'cooldown'` because the refactor in #42 renamed the use sites but
dropped the binding. Issue #48 reintroduced it: the binding is restored by the
one-line fix at the top of the "Apply the cooldown" section.

These tests drive `_apply_verdict` end-to-end against the FakeRedis in
`tests/fake_redis.py` -- no real Redis, no network, no provider calls. They
follow the wiring pattern from `tests/test_routing.py` (registry + slots +
ledger + policy, but no Picker -- the verdict handler does not need one).
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()

from dataclasses import replace  # noqa: E402

from switchyard import models  # noqa: E402
from switchyard.classify import Outcome, Verdict  # noqa: E402
from switchyard.hooks import SwitchyardHandler  # noqa: E402
from switchyard.policy import CapacityPolicy  # noqa: E402
from switchyard.slots import SlotTable  # noqa: E402
from switchyard.usage import Ledger  # noqa: E402
from tests.fake_redis import FakeRedis  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _build(breaker_enabled: bool | None = None):
    """Wire the bits _apply_verdict reaches, with a fresh FakeRedis each call.

    `breaker_enabled` lets the TRANSIENT-with-breaker-DISABLED test stand up its
    own registry without having to mutate the loaded Settings. The default
    (None) keeps the operator's configuration intact, which is what every
    other test relies on.
    """
    reg = models.load()
    if breaker_enabled is not None:
        reg = replace(reg, settings=replace(
            reg.settings,
            transient_breaker=replace(
                reg.settings.transient_breaker, enabled=breaker_enabled)))
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    ledger = Ledger(redis)
    policy = CapacityPolicy(redis, reg.settings, ledger)

    # Skip the constructor: it starts a config-watcher thread and re-loads the
    # plan file, neither of which a unit test of _apply_verdict cares about. The
    # properties read these attributes directly, so __dict__ assignment is the
    # supported bypass.
    h = SwitchyardHandler.__new__(SwitchyardHandler)
    h.__dict__["registry"] = reg
    h.__dict__["_slots"] = slots
    h.__dict__["_ledger"] = ledger
    h.__dict__["_policy"] = policy
    h.__dict__["_redis"] = redis
    return h, reg, slots


def _first_plan(reg):
    # Deterministic victim: same fixture the other verdict-shaped tests use.
    return next(iter(reg.plans.values()))


def test_transient_with_breaker_writes_cooldown_key_and_respects_cap():
    """TRANSIENT + breaker ENABLED: bump_and_cool runs, the sy:cool: key lands,
    and the escalated cooldown honours the breaker cap exactly.

    This is the regression proof. Before the one-line fix, _apply_verdict
    raised `UnboundLocalError: cannot access local variable 'cooldown'` on the
    very first TRANSIENT call -- the bug a coollable verdict could not survive.
    """
    async def go():
        h, reg, slots = _build(breaker_enabled=True)
        plan = _first_plan(reg)
        # Per the fixture: base cooldown from the verdict, cap from the breaker.
        base = 60
        cap = reg.settings.transient_breaker.max_seconds
        assert cap == 1800, cap   # fixture sanity: cap is the shipped default
        verdict = Verdict(Outcome.TRANSIENT, base, "upstream 500")

        # Eight consecutive TRANSIENT failures. The ladder is 60, 120, 240,
        # 480, 960, 1800 (capped), 1800, 1800 -- the cap fires on the sixth.
        ctx = {"session": "sess-T", "request_id": "req-T1"}
        streaks = []
        for i in range(8):
            # Fresh request_id per call so the double-count guard does NOT
            # short-circuit the test: each apply must drive bump_and_cool.
            ctx["request_id"] = f"req-T{i+1}"
            await h._apply_verdict(plan, verdict, ctx)
            streaks.append(await slots.transient_failure_streak(plan.key))

        cooled, ttl, reason = await slots.cooldown_state(plan.key)
        return streaks, cooled, ttl, reason, cap, base

    streaks, cooled, ttl, reason, cap, base = _run(go())
    # bump_and_cool ran once per call: streak grew from 1 to 8.
    assert streaks == list(range(1, 9)), streaks
    # The sy:cool:{plan} key landed in Redis with the expected shape.
    assert cooled is True, "TRANSIENT must leave a sy:cool: key behind"
    assert reason == "transient", reason
    # Sixth failure onward the ladder is capped at breaker.max_seconds. Allow
    # a couple of seconds of slack for the TTL tick between SET and GET -- the
    # value the test reads is the remaining time, not the original SET TTL.
    assert cap - 2 <= ttl <= cap, (ttl, cap)
    # Sanity: the cap is genuinely reachable -- with base=60, doubled 5 times
    # is 1920, which would exceed 1800; without the cap the test would land
    # at 1920 instead of the cap.
    assert base * (2 ** (6 - 1)) > cap, "test's ladder must actually trip the cap"
    print(f"  TRANSIENT+breaker: streaks 1..8 = {streaks}; "
          f"final cooldown {ttl}s == cap ({cap})")


def test_transient_without_breaker_uses_cool_down_with_verdict_cooldown():
    """TRANSIENT + breaker DISABLED: bump_and_cool is skipped, the verdict's
    own cooldown lands via cool_down with no escalation."""
    async def go():
        h, reg, slots = _build(breaker_enabled=False)
        plan = _first_plan(reg)
        base = 60
        verdict = Verdict(Outcome.TRANSIENT, base, "upstream 500")

        # Drive the same call twice in a row. The breaker being off means
        # bump_and_cool is NOT taken, so cool_down runs both times and the
        # cooldown stays at the verdict's base value -- the ladder is what
        # bump_and_cool implements, so disabling the breaker collapses it.
        ctx1 = {"session": "sess-T2", "request_id": "req-T2a"}
        await h._apply_verdict(plan, verdict, ctx1)
        ctx2 = {"session": "sess-T2", "request_id": "req-T2b"}
        await h._apply_verdict(plan, verdict, ctx2)

        # No streak is maintained when the breaker is off: the hook never
        # calls bump_and_cool, so the tfail key stays absent.
        streak = await slots.transient_failure_streak(plan.key)
        cooled, ttl, reason = await slots.cooldown_state(plan.key)
        return streak, cooled, ttl, reason, base

    streak, cooled, ttl, reason, base = _run(go())
    assert streak == 0, f"breaker off must not bump the streak: {streak}"
    assert cooled is True
    assert reason == "transient"
    # Cooldown is exactly the verdict's base -- no escalation, no cap. Allow
    # a couple of seconds of slack: the TTL ticks down between SET and GET.
    assert base - 2 <= ttl <= base, (ttl, base)
    print(f"  TRANSIENT+breaker off: no streak, cooldown {ttl}s == base {base}")


def test_rate_limited_writes_cooldown_key_and_raises_no_exception():
    """RATE_LIMITED: cool_down runs, the cooldown key lands, and no exception
    escapes _apply_verdict -- the Litellm callback would swallow a bug here,
    so the test asserts explicitly that nothing is raised."""
    async def go():
        h, reg, slots = _build()
        plan = _first_plan(reg)
        verdict = Verdict(Outcome.RATE_LIMITED, 30, "rate limited")
        ctx = {"session": "sess-R", "request_id": "req-R"}

        raised = None
        try:
            await h._apply_verdict(plan, verdict, ctx)
        except BaseException as exc:                # noqa: BLE001 -- swallow on purpose
            raised = exc

        cooled, ttl, reason = await slots.cooldown_state(plan.key)
        return raised, cooled, ttl, reason

    raised, cooled, ttl, reason = _run(go())
    assert raised is None, f"_apply_verdict must not raise: {raised!r}"
    assert cooled is True
    assert reason == "rate_limited", reason
    # The cooldown lands at exactly the verdict's value -- rate_limit is not
    # part of the breaker ladder, so cool_down writes the verdict's seconds
    # verbatim. Allow a couple of seconds of slack for the TTL tick between
    # SET and GET.
    assert 28 <= ttl <= 30, ttl
    print(f"  RATE_LIMITED: cool_down -> sy:cool key ({reason}, {ttl}s); "
          f"no exception")


def test_quota_exhausted_writes_cooldown_and_drops_lease():
    """QUOTA_EXHAUSTED: cool_down runs AND the session lease is dropped --
    the lease holds the session to the dead plan, and dropping it lets the
    next pick re-lease onto a peer rather than re-feed the same dead capacity.
    Both side effects must be visible, and again no exception escapes."""
    async def go():
        h, reg, slots = _build()
        plan = _first_plan(reg)
        # 900 is the shipped default_cooldown_seconds; the verdict here
        # carries the value directly so the test does not depend on Settings.
        verdict = Verdict(Outcome.QUOTA_EXHAUSTED, 900, "insufficient balance")
        session = "sess-Q"
        ctx = {"session": session, "request_id": "req-Q"}

        # Lease the session to this plan, the way a live conversation does
        # before its quota wall trips. The hook should drop it.
        await slots.set_lease(session, plan.key, reg.settings.lease_ttl_seconds)
        lease_before = await slots.get_lease(session)
        assert lease_before == plan.key, lease_before

        raised = None
        try:
            await h._apply_verdict(plan, verdict, ctx)
        except BaseException as exc:                # noqa: BLE001
            raised = exc

        cooled, ttl, reason = await slots.cooldown_state(plan.key)
        lease_after = await slots.get_lease(session)
        # The raw sy:lease:{session} key is gone -- that's the drop_lease side
        # effect. The sy:cool:{plan} key is the cool_down side effect.
        return raised, cooled, ttl, reason, lease_before, lease_after

    raised, cooled, ttl, reason, lease_before, lease_after = _run(go())
    assert raised is None, f"_apply_verdict must not raise: {raised!r}"
    assert lease_before == "minimax-ultra", lease_before
    assert lease_after is None, (
        f"QUOTA_EXHAUSTED must drop the lease, got {lease_after!r}")
    assert cooled is True, "QUOTA_EXHAUSTED must cool the plan"
    assert reason == "quota_exhausted", reason
    # And the same K_COOL key shape -- the verdict wrote via cool_down with
    # the same key the breaker path uses. Allow a couple of seconds of slack:
    # the value was set with `ex=900`, and the test reads the TTL back, so a
    # millisecond or two has ticked off the clock in between.
    assert 895 <= ttl <= 900, ttl
    print(f"  QUOTA_EXHAUSTED: lease dropped ({lease_before!r} -> None); "
          f"sy:cool key ({reason}, {ttl}s); no exception")


def test_double_count_guard_skips_second_call_with_same_request_id():
    """Two _apply_verdict calls with the SAME ctx.request_id must apply the
    verdict once. The router reuses the kwargs dict across attempts, so
    async_log_failure_event AND async_post_call_failure_hook both fire on the
    same logical failure -- without the per-attempt marker the streak would
    double and the cooldown would jump straight to the cap.

    Same ctx object, same request_id: the second call returns early at the
    marker check. Same ctx object, DIFFERENT request_id: both calls apply,
    which is what test_transient_with_breaker_writes_cooldown_key_and_respects_cap
    relies on (each iteration bumps the streak).
    """
    async def go():
        h, reg, slots = _build(breaker_enabled=True)
        plan = _first_plan(reg)
        verdict = Verdict(Outcome.TRANSIENT, 60, "upstream 500")

        ctx = {"session": "sess-D", "request_id": "req-D"}
        await h._apply_verdict(plan, verdict, ctx)
        streak_after_first = await slots.transient_failure_streak(plan.key)
        marker_after_first = ctx.get("_verdict_applied")

        # Second call: same ctx object, same request_id. The guard short-circuits.
        await h._apply_verdict(plan, verdict, ctx)
        streak_after_second = await slots.transient_failure_streak(plan.key)

        # Third call: same ctx object, NEW request_id. The guard fires again,
        # and bump_and_cool is allowed to run -- the streak moves to 2.
        ctx["request_id"] = "req-D2"
        await h._apply_verdict(plan, verdict, ctx)
        streak_after_third = await slots.transient_failure_streak(plan.key)

        return (streak_after_first, streak_after_second, streak_after_third,
                marker_after_first)

    s1, s2, s3, marker = _run(go())
    assert s1 == 1, f"first call bumps the streak to 1, got {s1}"
    assert s2 == 1, (
        f"same request_id must NOT bump the streak again -- got {s2}; "
        "the guard is the only thing keeping a single 5xx from looking like two")
    assert s3 == 2, "a fresh request_id must be allowed to bump the streak"
    # And the guard's bookkeeping is exactly what makes (1) and (2) diverge:
    # the marker is written with the request_id it short-circuited on.
    assert marker == "req-D", marker
    print(f"  double-count guard: streak {s1} -> {s2} (same rid, skipped) "
          f"-> {s3} (new rid, applied)")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
