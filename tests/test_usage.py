"""Usage ledger tests — plan-level and model-scoped economics."""
from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from switchyard import usage as usage_module
from switchyard.models import Plan, Quota
from switchyard.usage import (
    Ledger,
    model_effective_cost_per_mtok,
    effective_cost_per_mtok,
    reported_is_current,
    window_headroom,
    K_M_HOUR,
    K_M_DAY,
    K_HOUR,
    K_DAY,
    K_WINDOW,
)

# FakeRedis must be imported after conftest patches socket.
from tests.fake_redis import FakeRedis  # noqa: E402

run = asyncio.run


def _patch_now(dt: datetime):
    """Patch _now in the usage module to return the given datetime."""
    return patch.object(usage_module, "_now", lambda: dt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fake_plan(monthly_cost: float = 0.0, metered: bool = False) -> Plan:
    """A minimal plan with one quota window, suitable for testing."""
    return Plan(
        key="test-plan",
        label="Test Plan",
        models={},
        quotas=[Quota(kind="tokens", period="month", allowance=1_000_000)],
        monthly_cost=monthly_cost,
        metered=metered,
    )


# ---------------------------------------------------------------------------
# Tests: Ledger.record with model= writes both plan and model buckets
# ---------------------------------------------------------------------------

def test_record_without_model_writes_plan_buckets_only():
    """record(model=None) behaves exactly as before: plan-level buckets only."""
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()

    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        run(ledger.record(plan, prompt_tokens=100, completion_tokens=50, cost=0.01))

    # Plan-level hour bucket must exist.
    hour_key = K_HOUR.format(plan="test-plan", hour="2025-09-21T14")
    hour = run(ledger.bucket(plan.key, hour_key))
    assert hour["prompt_tokens"] == 100
    assert hour["completion_tokens"] == 50
    assert hour["cost"] == 0.01

    # No model-scoped bucket should exist.
    m_hour_key = K_M_HOUR.format(plan="test-plan", model="test-model", hour="2025-09-21T14")
    assert m_hour_key not in redis.hashes


def test_record_with_model_writes_plan_and_model_buckets():
    """record(model=...) writes plan-level AND model-scoped buckets."""
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()

    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        run(ledger.record(
            plan,
            prompt_tokens=200,
            completion_tokens=100,
            cost=0.02,
            model="test-model",
        ))

    # Plan-level bucket unchanged.
    hour_key = K_HOUR.format(plan="test-plan", hour="2025-09-21T14")
    hour = run(ledger.bucket(plan.key, hour_key))
    assert hour["prompt_tokens"] == 200
    assert hour["completion_tokens"] == 100

    # Model-scoped bucket also written.
    m_hour_key = K_M_HOUR.format(plan="test-plan", model="test-model", hour="2025-09-21T14")
    m_hour = run(ledger.bucket(plan.key, m_hour_key))
    assert m_hour["prompt_tokens"] == 200
    assert m_hour["completion_tokens"] == 100
    assert m_hour["cost"] == 0.02

    # Day bucket too.
    m_day_key = K_M_DAY.format(plan="test-plan", model="test-model", day="2025-09-21")
    m_day = run(ledger.bucket(plan.key, m_day_key))
    assert m_day["prompt_tokens"] == 200


def test_record_with_failed_and_model():
    """Failed record with model= writes failures to both plan and model buckets."""
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()

    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        run(ledger.record(plan, failed=True, model="fail-model"))

    plan_key = K_HOUR.format(plan="test-plan", hour="2025-09-21T14")
    plan_bucket = run(ledger.bucket(plan.key, plan_key))
    assert plan_bucket["failures"] == 1

    m_key = K_M_HOUR.format(plan="test-plan", model="fail-model", hour="2025-09-21T14")
    m_bucket = run(ledger.bucket(plan.key, m_key))
    assert m_bucket["failures"] == 1


# ---------------------------------------------------------------------------
# Tests: model_overview burn math and month filtering
# ---------------------------------------------------------------------------

def test_model_overview_burn_rate():
    """model_overview returns correct cost_per_hour and tokens_per_hour."""
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan(metered=True)

    async def write_hours():
        # Hour 0: 60+40=100 tokens, $0.01
        with _patch_now(datetime(2025, 9, 21, 14, 0, tzinfo=timezone.utc)):
            await ledger.record(plan, prompt_tokens=60, completion_tokens=40, cost=0.01, model="m1")
        # Hour 1: 120+80=200 tokens, $0.02
        with _patch_now(datetime(2025, 9, 21, 13, 0, tzinfo=timezone.utc)):
            await ledger.record(plan, prompt_tokens=120, completion_tokens=80, cost=0.02, model="m1")

    run(write_hours())

    # 3-hour window: avg of (100+$0.01) + (200+$0.02) + 0 = 100 tok/hr, $0.01/hr
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        result = run(ledger.model_overview("test-plan", ["m1"], "2025-09", hours=3))
    assert "m1" in result
    assert result["m1"]["burn"]["tokens_per_hour"] == 100.0  # (100 + 200) / 3
    assert result["m1"]["burn"]["cost_per_hour"] == 0.01  # (0.01 + 0.02) / 3


def test_model_overview_month_filter():
    """Only days starting with the target month string are included.

    The August bucket is deliberately INSIDE the 31-day read window (6 days
    before the read): reading it and filtering afterwards, or not filtering at
    all, must not drag last month into "This month" — which is exactly what
    happens on September 3rd if the filter is missing.
    """
    redis = FakeRedis()
    ledger = Ledger(redis)

    async def write_days():
        plan = fake_plan(metered=True)
        # September 2025: in the month, and inside the read window
        with _patch_now(datetime(2025, 9, 1, 12, 0, tzinfo=timezone.utc)):
            await ledger.record(plan, prompt_tokens=10_000, cost=0.10, model="m1")
        # August 2025: NOT in the month, but inside the 31-day read window
        with _patch_now(datetime(2025, 8, 30, 12, 0, tzinfo=timezone.utc)):
            await ledger.record(plan, prompt_tokens=50_000, cost=0.50, model="m1")
        # October 2025: NOT in the month (future relative to the read)
        with _patch_now(datetime(2025, 10, 15, 12, 0, tzinfo=timezone.utc)):
            await ledger.record(plan, prompt_tokens=99_000, cost=0.99, model="m1")

    run(write_days())

    # Reading near the START of September is the case that catches a missing
    # filter: the 31-day walk reaches back into August from here.
    with _patch_now(datetime(2025, 9, 5, 14, 30, 0, tzinfo=timezone.utc)):
        result = run(ledger.model_overview("test-plan", ["m1"], "2025-09", hours=3))
    assert result["m1"]["month_tokens"] == 10_000
    assert result["m1"]["month_cost"] == 0.10


def test_model_overview_multiple_models():
    """model_overview returns data for all requested refs in one call."""
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan(metered=True)

    async def write_both():
        with _patch_now(datetime(2025, 9, 21, 14, 0, tzinfo=timezone.utc)):
            await ledger.record(plan, prompt_tokens=100, cost=0.01, model="model-a")
            await ledger.record(plan, prompt_tokens=200, cost=0.02, model="model-b")

    run(write_both())

    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        result = run(ledger.model_overview("test-plan", ["model-a", "model-b"], "2025-09"))
    assert "model-a" in result
    assert "model-b" in result
    assert result["model-a"]["burn"]["tokens_per_hour"] == 100.0 / 3
    assert result["model-b"]["burn"]["tokens_per_hour"] == 200.0 / 3


# ---------------------------------------------------------------------------
# Tests: model_effective_cost_per_mtok
# ---------------------------------------------------------------------------

def test_model_eff_cost_metered():
    """Metered plan: spend = model_cost (actual spend), not monthly fee."""
    plan = fake_plan(monthly_cost=100.0, metered=True)
    # 2M tokens at $0.05 each = $0.10 actual spend
    result = model_effective_cost_per_mtok(plan, 2_000_000, 0.10, 5_000_000)
    # $0.10 / 2Mtok = $0.05/Mtok
    assert result == 0.05


def test_model_eff_cost_subscription():
    """Subscription plan: allocate monthly fee pro-rata by token share."""
    plan = fake_plan(monthly_cost=132.0, metered=False)
    # Model used 40% of tokens (4M of 10M), so 40% of $132 = $52.80 for 4M tok
    result = model_effective_cost_per_mtok(plan, 4_000_000, 0.0, 10_000_000)
    # $52.80 / 4Mtok = $13.20/Mtok
    assert result == 13.20


def test_model_eff_cost_subscription_small_slice():
    """A small slice of a used subscription still shows the plan's rate.

    The pro-rata share cancels out of the division, so the number is the
    plan's own $/Mtok the moment the PLAN crosses 1M tokens — gating on the
    model's own 1M hid a real number from glm-5.3-flash at 878K.
    """
    plan = fake_plan(monthly_cost=10.0)
    # 878K of 5M plan tokens: spend = $10 * 878440/5M, rate = spend / 0.8784M
    result = model_effective_cost_per_mtok(plan, 878_440, 0.0, 5_000_000)
    assert result == round(10.0 / 5.0, 4)  # == the plan's $/Mtok


def test_model_eff_cost_subscription_plan_under_1m():
    """Subscription whose PLAN total is under 1M -> None, same as before."""
    plan = fake_plan(monthly_cost=10.0)
    assert model_effective_cost_per_mtok(plan, 400_000, 0.0, 900_000) is None


def test_model_eff_cost_no_traffic():
    """A model that saw no traffic gets no rate, on any plan kind."""
    plan = fake_plan(monthly_cost=10.0)
    assert model_effective_cost_per_mtok(plan, 0, 0.0, 5_000_000) is None
    metered = fake_plan(monthly_cost=0.0, metered=True)
    assert model_effective_cost_per_mtok(metered, 0, 0.0, 5_000_000) is None


def test_model_eff_cost_plan_tokens_zero():
    """plan_tokens == 0 guard: cannot allocate, returns None."""
    plan = fake_plan(monthly_cost=100.0)
    assert model_effective_cost_per_mtok(plan, 2_000_000, 0.0, 0) is None


def test_model_eff_cost_zero_spend():
    """spend <= 0 -> 0.0 (subscription with monthly_cost = 0)."""
    plan = fake_plan(monthly_cost=0.0)
    result = model_effective_cost_per_mtok(plan, 2_000_000, 0.0, 1_000_000)
    assert result == 0.0


def test_model_eff_cost_rounding():
    """Result is rounded to 4 decimals, matching plan version."""
    plan = fake_plan(monthly_cost=132.0, metered=False)
    # 3,333,333 tokens, $132 monthly = 132 * 3333333 / 10000000 = $44
    # 44 / 3.333333 = 13.200013... -> 13.2000
    result = model_effective_cost_per_mtok(plan, 3_333_333, 0.0, 10_000_000)
    assert result == round(132.0 * 3_333_333 / 10_000_000 / (3_333_333 / 1_000_000), 4)


# ---------------------------------------------------------------------------
# Tests: existing effective_cost_per_mtok still works
# ---------------------------------------------------------------------------

def test_effective_cost_per_mtok_unchanged():
    """Sanity-check the original function still works as before."""
    plan = fake_plan(monthly_cost=132.0)
    result = effective_cost_per_mtok(plan, 40_000_000, 0.0)
    # $132 / 40Mtok = $3.30/Mtok
    assert result == 3.30


def test_effective_cost_per_mtok_metered():
    """Metered plan with zero monthly_cost falls back to metered_cost."""
    plan = fake_plan(monthly_cost=0.0, metered=True)
    result = effective_cost_per_mtok(plan, 40_000_000, 20.0)
    # spend = 20.0 (metered_cost), since monthly_cost is 0/falsy
    # $20 / 40Mtok = $0.50/Mtok
    assert result == 0.50


# ---------------------------------------------------------------------------
# Tests: compact count formatting (the portal's `compact` filter)
# ---------------------------------------------------------------------------

def test_compact_counts():
    """K/M/B/T scaling with one decimal, trailing zeros trimmed."""
    from switchyard.portal.app import _compact
    assert _compact(4_723_058) == "4.7M"
    assert _compact(500_000) == "500K"
    assert _compact(1_230_000_000) == "1.2B"
    assert _compact(3.1e12) == "3.1T"
    assert _compact(312) == "312"
    assert _compact(0) == "0"
    assert _compact(41_090_604) == "41.1M"


def test_compact_boundary_and_trimming():
    """999,999 rounds up to 1M rather than the false-precision 1000K;
    whole values carry no trailing .0."""
    from switchyard.portal.app import _compact
    assert _compact(999_999) == "1M"
    assert _compact(1_000_000) == "1M"
    assert _compact(12_000_000) == "12M"
    assert _compact(999_499) == "999.5K"


# ---------------------------------------------------------------------------
# Tests: reported_is_current staleness gate
# ---------------------------------------------------------------------------

def test_reported_reading_expires_when_reset_passes():
    """A `reported_*` row whose `reset_at` is already in the past is stale.

    The provider told us the window resets at T; T has passed without a fresh
    reading, so the percentage is no longer about any window in force. The
    picker's `is_spent` and `window_headroom` both gate on this so a plan
    whose vendor client never writes a new row can still re-enter rotation
    after the rollover -- the asymmetry is safe because over-admission
    self-heals via the vendor's next 429.
    """
    now = time.time()
    # One hour ago: reset_at is in the past -> stale even if reported_at is fresh.
    assert reported_is_current(
        {"reset_at": now - 3600, "reported_at": now - 1.0}, "week", now=now) is False
    # "" (the note_exhaustion sentinel) is also stale: it is not a float and
    # therefore cannot be in the future.
    assert reported_is_current(
        {"reset_at": "", "reported_at": now}, "week", now=now) is False
    # Absent reset_at does not trigger rule (a): only reported_at bounds
    # freshness. With reported_at from this week, the reading is current.
    assert reported_is_current(
        {"reported_at": now}, "week", now=now) is True
    # Future reset_at + fresh reported_at is current.
    assert reported_is_current(
        {"reset_at": now + 3600, "reported_at": now - 1.0}, "week", now=now) is True


def test_reported_reading_expires_at_period_rollover():
    """A reading older than the current period bucket is stale even with no reset.

    `reset_at` may be absent (e.g. the provider's client reports only a
    percentage), so rule (b) -- `reported_at` within the current period
    bucket via `period_bounds(period)` -- is the fallback that catches a
    weekly reading from a previous week. Rule (b) is the only freshness
    check a row with no `reset_at` ever gets.
    """
    now = time.time()
    # Move `now` to midweek so the bucket start is unambiguous. Pick a Tuesday
    # so the week starts on Monday regardless of locale.
    from datetime import datetime, timedelta, timezone
    tuesday_noon = datetime(2025, 9, 16, 12, 0, 0, tzinfo=timezone.utc)  # a Tuesday
    now = tuesday_noon.timestamp()
    week_start = tuesday_noon.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start -= timedelta(days=1)  # Monday 00:00 UTC
    # One hour before the Monday bucket starts -> previous week -> stale.
    assert reported_is_current(
        {"reported_at": week_start.timestamp() - 3600}, "week", now=now) is False
    # The instant the bucket starts -> still current (>= counts).
    assert reported_is_current(
        {"reported_at": week_start.timestamp()}, "week", now=now) is True
    # Mid-week -> current.
    assert reported_is_current(
        {"reported_at": tuesday_noon.timestamp() - 60.0}, "week", now=now) is True
    # Missing reported_at (any non-float) is stale regardless of period.
    assert reported_is_current({}, "week", now=now) is False
    assert reported_is_current({"reported_at": "not-a-float"}, "week", now=now) is False
    assert reported_is_current({"reported_at": None}, "week", now=now) is False


def test_window_headroom_ignores_stale_reported_percent():
    """`window_headroom` falls back to the ledger basis when the pct is stale.

    A reported_pct_used=100 with a past reset_at would otherwise pin the bar
    at 100% forever -- exactly the bug at #45 -- so `reported_is_current`
    gates the early-return. With the gate, `window_headroom` proceeds to the
    observed-allowance / ledger branch, which is what the board wants to
    show: an estimate that may still be off, not a stale lock.
    """
    from switchyard.usage import window_headroom
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()  # month allowance = 1_000_000 tokens
    # A spent reading: 100% reported, reset 1h ago. plan.headroom would
    # normally pin pct_used to 100 here. With staleness gating, it should
    # fall through to the ledger basis (consumed=0, limit=1_000_000).
    past_reset = time.time() - 3600
    redis.hashes[K_WINDOW.format(plan=plan.key, window=plan.quota.label)] = {
        "reported_pct_used": "100.0",
        "reported_at": str(time.time()),
        "reset_at": str(past_reset),
    }
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        hr = run(window_headroom(ledger, plan, plan.quota))
    # Stale -> not 100; falls back to ledger. 0 / 1_000_000 = 0.0%.
    assert hr["pct_used"] == 0.0, hr
    assert "reported by provider" not in hr["basis"], hr
    assert hr["basis"].startswith("ledger"), hr

    # Sanity check: a CURRENT 100% reading is honoured (gating does not
    # regress the happy path).
    redis.hashes[K_WINDOW.format(plan=plan.key, window=plan.quota.label)] = {
        "reported_pct_used": "100.0",
        "reported_at": str(time.time()),
        "reset_at": str(time.time() + 3600),
    }
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        hr_current = run(window_headroom(ledger, plan, plan.quota))
    assert hr_current["pct_used"] == 100.0, hr_current
    assert hr_current["basis"] == "reported by provider (% only)", hr_current


# ---------------------------------------------------------------------------
# Tests: stale-grace for cookie-expired plans
# ---------------------------------------------------------------------------
#
# A plan whose session cookie has expired stops polling (its `due()` gate
# is False, so the writer doesn't refresh the lane-order / group-order hash).
# Without a grace window, the hash ages past `stale_after_ms` (~20 min, 2x
# the probe interval) and the picker falls back to config order -- even
# though the LAST GOOD ranking is still meaningful: every window the hash
# was computed from is still in force.
#
# The grace is capped exactly at the window boundary (`reset_at`), never
# across it. Once the window rolls over, the reader returns None and the
# picker falls back to config order exactly as it did before this work.
# ---------------------------------------------------------------------------


def test_reported_reading_with_needs_reauth_stays_current_within_same_window():
    """A `reported_*` reading on a needs_reauth plan whose window has not
    reset is still current -- the grace window is exactly the in-force
    window. The operator's last good reading drives the picker (and the
    board) until the window resets, instead of dropping to ledger estimates
    the moment the cookie dies.

    This is the existing `reported_is_current` rule (a): `reset_at` is in
    the future, so the reading is current. The test pins that
    `needs_reauth` does NOT regress the happy path: a needs_reauth plan
    with a future reset_at keeps its last good number.
    """
    now = time.time()
    future_reset = now + 3600
    facts = {"reset_at": future_reset, "reported_at": now - 60.0,
             "needs_reauth": True}
    assert reported_is_current(facts, "week", now=now) is True, facts


def test_reported_reading_with_needs_reauth_stale_after_reset_at_passes():
    """A needs_reauth plan whose last good reading's window has reset is
    back to today's behaviour: NOT eligible. The grace is capped exactly
    at the window boundary, so once `reset_at` passes the reading no
    longer describes any in-force window and the picker / board fall back
    to estimated numbers. This is the regression bar for the invariant
    "never carry a reading across a window reset into the next window".
    """
    now = time.time()
    past_reset = now - 3600
    facts = {"reset_at": past_reset, "reported_at": past_reset - 60.0,
             "needs_reauth": True}
    assert reported_is_current(facts, "week", now=now) is False, facts


def test_lane_order_hash_stays_fresh_under_needs_reauth_grace():
    """A stale lane-order hash for a needs_reauth plan whose window has not
    reset stays readable: the grace extends freshness UNTIL reset_at.

    Without grace, a hash that's older than `stale_after_ms` (here, 1000ms)
    would be dropped and `get_lane_order` would return None -- the picker
    would fall back to config order. With grace, the same hash is returned
    because the plan's last-good window hasn't reset yet.
    """
    from switchyard.usage import K_LANE_ORDER
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()
    plan_key = plan.key
    window = plan.quota.label
    # Stamp needs_reauth=1 on the probe hash; the grace looks here.
    redis.hashes[f"sy:probe:{plan_key}"] = {"needs_reauth": "1"}
    # Last good reading: reset_at well in the future.
    future_reset = time.time() + 86400
    redis.hashes[K_WINDOW.format(plan=plan_key, window=window)] = {
        "reset_at": str(future_reset),
        "reported_at": str(time.time() - 600),
    }
    # Write the lane-order hash long ago (past stale_after_ms).
    redis.hashes[K_LANE_ORDER.format(lane="forge")] = {
        "computed_at": str(time.time() - 9999),
        "stale_after_ms": "1000",
        "score_claude-max/fable": "0.9",
        "gate5h_claude-max/fable": "0",
    }
    # Without plan_windows -> still stale (no grace info to consult).
    stale = run(ledger.get_lane_order("forge"))
    assert stale is None, "without grace info, stale hash is None"
    # With plan_windows and a future reset_at -> fresh under grace.
    fresh = run(ledger.get_lane_order(
        "forge", plan_windows=[(plan_key, window)]))
    assert fresh is not None, "grace should keep the hash readable"
    assert fresh["members"][0]["ref"] == "claude-max/fable"
    assert fresh["members"][0]["score"] == 0.9


def test_lane_order_hash_grace_ends_when_reset_at_passes():
    """A stale lane-order hash whose only grace-eligible plan has a reset_at
    already in the past returns None -- the grace is capped at the window
    boundary. This is the same shape as the existing
    `test_reported_reading_expires_when_reset_passes` regression bar: the
    picker MUST drop to config order once the window rolls over.
    """
    from switchyard.usage import K_LANE_ORDER
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()
    plan_key = plan.key
    window = plan.quota.label
    redis.hashes[f"sy:probe:{plan_key}"] = {"needs_reauth": "1"}
    # Reset_at already passed -> window has rolled over.
    past_reset = time.time() - 3600
    redis.hashes[K_WINDOW.format(plan=plan_key, window=window)] = {
        "reset_at": str(past_reset),
        "reported_at": str(past_reset - 600),
    }
    redis.hashes[K_LANE_ORDER.format(lane="forge")] = {
        "computed_at": str(time.time() - 9999),
        "stale_after_ms": "1000",
        "score_claude-max/fable": "0.9",
    }
    out = run(ledger.get_lane_order(
        "forge", plan_windows=[(plan_key, window)]))
    assert out is None, "grace must end at the window boundary"


def test_lane_order_hash_grace_only_for_needs_reauth_plan():
    """A stale lane-order hash for a healthy plan is treated as stale even
    when `reset_at` is in the future: the grace is a needs_reauth-only
    extension, not a general freshness boost. Without this guard, a hash
    past its `stale_after_ms` would silently outlive its bound on every
    plan, and the perishable order would drift off real numbers rather
    than admit it is unknown.
    """
    from switchyard.usage import K_LANE_ORDER
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()
    plan_key = plan.key
    window = plan.quota.label
    # Healthy: no needs_reauth.
    redis.hashes[f"sy:probe:{plan_key}"] = {"needs_reauth": "0"}
    future_reset = time.time() + 86400
    redis.hashes[K_WINDOW.format(plan=plan_key, window=window)] = {
        "reset_at": str(future_reset),
        "reported_at": str(time.time() - 600),
    }
    redis.hashes[K_LANE_ORDER.format(lane="forge")] = {
        "computed_at": str(time.time() - 9999),
        "stale_after_ms": "1000",
        "score_claude-max/fable": "0.9",
    }
    out = run(ledger.get_lane_order(
        "forge", plan_windows=[(plan_key, window)]))
    assert out is None, "grace must not apply to non-needs_reauth plans"


def test_lane_order_hash_still_stale_when_no_plan_in_grace():
    """When the listed plans are all healthy and have no grace-eligible
    member, the hash is stale exactly as before -- the grace is purely
    opt-in via `plan_windows`. This pins that an empty / omitted plan
    list behaves bit-for-bit as the previous (no-grace) reader did.
    """
    from switchyard.usage import K_LANE_ORDER
    redis = FakeRedis()
    ledger = Ledger(redis)
    redis.hashes[K_LANE_ORDER.format(lane="forge")] = {
        "computed_at": str(time.time() - 9999),
        "stale_after_ms": "1000",
        "score_claude-max/fable": "0.9",
    }
    # Empty plan list -> no grace -> stale.
    assert run(ledger.get_lane_order(
        "forge", plan_windows=[])) is None
    # No plan_windows arg at all -> stale (back-compat).
    assert run(ledger.get_lane_order("forge")) is None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
                passed += 1
            except Exception as exc:
                print(f" FAIL {name}: {exc}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
