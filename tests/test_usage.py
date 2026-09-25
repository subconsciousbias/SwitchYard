"""Usage ledger tests — plan-level and model-scoped economics."""
from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)  # so `import conftest` resolves under plain `python3`

# Pin LiteLLM to its bundled model-cost backup BEFORE the socket guard
# installs. The hooks module imports LiteLLM at module load (via
# `models.load` -> `litellm_known_vision_models`), and LiteLLM's default
# behaviour is to fetch the live cost map from raw.githubusercontent.com
# on first import. With the guard active in plain-script mode, that
# fetch raises inside `models.load`, breaking every test that reaches
# `switchyard.hooks` -- including the four `_prompt_completion_tokens`
# tests at the bottom of this file. The bundled backup covers them.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# Install the socket guard before any switchyard import so the guard is in
# place if a switchyard module ever reaches for a non-loopback address at
# import time (FakeRedis must be imported after conftest patches socket).
import conftest  # noqa: F401  (socket guard for plain-script mode)

# ``switchyard.models`` captures SWITCHYARD_PLANS at import time, and
# ``switchyard.hooks`` instantiates ``SwitchyardHandler`` (which calls
# ``models.load``) at import time. The existing tests don't need a plans
# file, but the four ``_prompt_completion_tokens`` tests below do, so point
# the loader at the tracked example BEFORE anything imports either module.
os.environ["SWITCHYARD_PLANS"] = os.path.join(
    os.path.dirname(HERE),
    "config", "plans.example.yaml",
)

from switchyard import usage as usage_module
from switchyard.models import Plan, Quota
from switchyard.usage import (
    Ledger,
    model_effective_cost_per_mtok,
    model_effective_cost_fields,
    model_effective_cost_per_session,
    effective_cost_per_mtok,
    effective_cost_fields,
    reported_is_current,
    K_M_HOUR,
    K_M_DAY,
    K_M_SMONTH,
    K_HOUR,
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
# Tests: model_effective_cost_per_session (issue #76)
# ---------------------------------------------------------------------------

def test_model_eff_cost_session_metered():
    """Metered plan: rate = model_cost / n_sessions; gate at 100 sessions.

    A metered provider charges actual spend, so the per-session rate is the
    model's own spend divided by how many distinct sessions it served.
    Under 100 sessions the number is too noisy to read.
    """
    plan = fake_plan(monthly_cost=0.0, metered=True)
    # 200 sessions at $0.10 model spend = $0.0005/session
    result = model_effective_cost_per_session(plan, 200, 0.10, 200)
    assert result == round(0.10 / 200, 4)


def test_model_eff_cost_session_subscription():
    """Subscription plan: rate = plan.monthly_cost / plan_sessions.

    The fee allocated by session share is fee * m / p, so per-session the
    share cancels -- every model that saw traffic shows the plan's own
    rate once the PLAN clears 50 sessions. $132 across 550 plan sessions
    is $0.24/session, regardless of how the 550 split across models.
    """
    plan = fake_plan(monthly_cost=132.0, metered=False)
    # 200 of 550 plan sessions, model_cost is irrelevant for a subscription
    result = model_effective_cost_per_session(plan, 200, 0.0, 550)
    assert result == round(132.0 / 550, 4)


def test_model_eff_cost_session_subscription_small_slice():
    """A small slice of a used subscription shows the plan rate, not a
    divided-up rate.

    Mirror of `test_model_eff_cost_subscription_small_slice`: the share
    cancels in the per-session division too, so the number is the plan's
    own $/session once the PLAN clears 50 sessions, not gated on the
    model's own session count beyond the >0 check.
    """
    plan = fake_plan(monthly_cost=10.0, metered=False)
    # Model saw 22 sessions of the plan's 250; rate = $10 / 250.
    result = model_effective_cost_per_session(plan, 22, 0.0, 250)
    assert result == round(10.0 / 250, 4)


def test_model_eff_cost_session_under_thresholds():
    """Both plan kinds gate on a minimum session count.

    The subscription gate is on the PLAN's session count (50) because the
    rate is plan-level; gating on the model's own count would silently
    hide the plan rate from a small slice that is being used. The metered
    gate is on the model's own sessions (100) because the rate is the
    model's own spend.
    """
    plan = fake_plan(monthly_cost=132.0, metered=False)
    # Subscription with plan_sessions=49 -> below the threshold -> None,
    # regardless of model session count.
    assert model_effective_cost_per_session(plan, 200, 0.0, 49) is None

    metered = fake_plan(monthly_cost=0.0, metered=True)
    # Metered with n_sessions=99 -> below the model's threshold -> None,
    # even though the PLAN's count (plan_sessions) is fine.
    assert model_effective_cost_per_session(metered, 99, 0.10, 500) is None


def test_model_eff_cost_session_cold_and_zero_spend():
    """No traffic at all -> None (both kinds); zero spend -> 0.0.

    n_sessions=0 means no work to price, on either plan kind -- the same
    way model_tokens<=0 short-circuits the per-Mtok helper. A subscription
    whose monthly_cost is 0 has zero spend regardless of sessions: 0.0,
    not None (the plan still ran, it just cost nothing).
    """
    plan = fake_plan(monthly_cost=132.0, metered=False)
    assert model_effective_cost_per_session(plan, 0, 0.0, 200) is None

    metered = fake_plan(monthly_cost=0.0, metered=True)
    assert model_effective_cost_per_session(metered, 0, 0.10, 200) is None

    # Subscription with monthly_cost=0 and live sessions -> 0.0, not None.
    free = fake_plan(monthly_cost=0.0, metered=False)
    assert model_effective_cost_per_session(free, 200, 0.0, 200) == 0.0


def test_model_overview_counts_unique_sessions():
    """PFADD is idempotent per session; PFCOUNT reports the cardinality.

    Recording the SAME session id several times for model m1 (one session
    making many requests / re-leasing the model) keeps n_sessions at 1;
    a session-free record for m2 leaves it at 0. The /session cell on the
    board is the number of distinct CLI / API callers the model served,
    not the number of requests.
    """
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan(metered=True)

    with _patch_now(datetime(2025, 9, 21, 14, 0, 0, tzinfo=timezone.utc)):

        async def record_sessions():
            # Same session id recorded 5 times -> PFCOUNT = 1.
            for _ in range(5):
                await ledger.record(plan, prompt_tokens=100, cost=0.01,
                                    model="m1", session="session-A")
            # m2 sees traffic but no session at all -- a session='' record
            # would skip the PFADD (exercising the "no session, no count"
            # path), but a missing session arg exercises the same branch.
            await ledger.record(plan, prompt_tokens=100, cost=0.01, model="m2")

        run(record_sessions())

        # Spot-check the HLL key exists for m1, not for m2.
        assert K_M_SMONTH.format(
            plan="test-plan", model="m1", period="2025-09") in redis.hlls
        assert K_M_SMONTH.format(
            plan="test-plan", model="m2", period="2025-09") not in redis.hlls

        result = run(ledger.model_overview("test-plan", ["m1", "m2"], "2025-09"))
    assert result["m1"]["n_sessions"] == 1, result["m1"]
    assert result["m2"]["n_sessions"] == 0, result["m2"]


def test_record_failed_event_does_not_pfadd():
    """failed=True + session=... must NOT PFADD -- the success-only invariant.

    `record()` gates the HLL on `not failed`: a session that landed on a
    model but errored every request would otherwise pollute the rate
    denominator. Call sites today also pass no session on failure, so this
    is doubly-protected, but the helper's clause is the last line of defence
    and worth pinning: a refactor that lifts the gate (e.g. to "PFADD even
    on failure so failures show up too") would silently change what the
    board's $/session means.
    """
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan(metered=True)

    with _patch_now(datetime(2025, 9, 21, 14, 0, 0, tzinfo=timezone.utc)):
        # failed=True with both model and session set -- the exact branch
        # the gate protects.
        run(ledger.record(plan, failed=True, model="m3", session="session-A"))

    assert K_M_SMONTH.format(
        plan="test-plan", model="m3", period="2025-09") not in redis.hlls


def test_model_overview_plan_n_sessions_is_union_not_sum():
    """plan_n_sessions = |union of per-model HLLs|, not the sum.

    One session hitting two models of one plan must NOT double-count:
    union cardinality is 1 even though per-model cardinalities both read 1.
    The original implementation summed per-model counts, which over-counted
    and understated the subscription rate -- the round-2 fix is the
    multi-key PFCOUNT here, so the union semantics are the invariant.
    """
    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan(metered=True)
    with _patch_now(datetime(2025, 9, 21, 14, 0, 0, tzinfo=timezone.utc)):
        async def record():
            await ledger.record(plan, model="m1", session="shared",
                                prompt_tokens=100, cost=0.01)
            await ledger.record(plan, model="m2", session="shared",
                                prompt_tokens=100, cost=0.01)
            await ledger.record(plan, model="m2", session="unique-m2",
                                prompt_tokens=100, cost=0.01)
        run(record())
        result = run(ledger.model_overview("test-plan", ["m1", "m2"], "2025-09"))
    # Per-model: m1 saw 1 distinct, m2 saw 2 distinct.
    assert result["m1"]["n_sessions"] == 1
    assert result["m2"]["n_sessions"] == 2
    # Plan-level union: the shared session counted once, plus the unique one = 2.
    assert result["m1"]["plan_n_sessions"] == 2
    assert result["m2"]["plan_n_sessions"] == 2


# ---------------------------------------------------------------------------
# Tests: _prompt_completion_tokens honors switchyard_billed_* (issue #89)
# ---------------------------------------------------------------------------


def test_prompt_completion_tokens_books_billed_over_reported():
    """When the bridge stamps ``switchyard_billed_*``, the gateway ledger
    books those numbers (the run's true spend) instead of the provider's
    reported context-size prompt. The reported ``prompt_tokens`` /
    ``completion_tokens`` would double the burn if taken at face value on the
    mcp_bridge path; the billed keys are the authoritative count.
    """
    from switchyard.hooks import _prompt_completion_tokens
    usage = {
        # Reported shape: conversation-context size, not cumulative spend.
        "prompt_tokens": 12_345,
        "completion_tokens": 678,
        # Billed shape: the run's true spend, what the ledger must book.
        "switchyard_billed_prompt_tokens": 2_400_000,
        "switchyard_billed_completion_tokens": 1_000,
    }
    prompt, completion = _prompt_completion_tokens(None, usage)
    assert prompt == 2_400_000, prompt
    assert completion == 1_000, completion


def test_prompt_completion_tokens_plain_dict_unchanged():
    """A usage dict without the billed keys still goes through the existing
    extraction -- the new branch must be a pure override, byte-for-byte
    unchanged on the path every other provider / route takes today.
    """
    from switchyard.hooks import _prompt_completion_tokens
    usage = {
        "prompt_tokens": 12_345,
        "completion_tokens": 678,
    }
    prompt, completion = _prompt_completion_tokens(None, usage)
    assert prompt == 12_345, prompt
    assert completion == 678, completion


def test_prompt_completion_tokens_billed_attribute_shape():
    """Same override fires on the pydantic / attribute-shaped usage object
    path too, matching the function's dual-shape handling. A bare object
    carrying the billed attributes returns them; one without keeps the
    existing attribute-based extraction.
    """
    from switchyard.hooks import _prompt_completion_tokens

    class _Usage:
        # Reported fields present (so the regular path would not be empty).
        prompt_tokens = 100
        completion_tokens = 10
        # Billed attributes stamped by the bridge override the above.
        switchyard_billed_prompt_tokens = 2_400_000
        switchyard_billed_completion_tokens = 1_000

    prompt, completion = _prompt_completion_tokens(None, _Usage())
    assert prompt == 2_400_000, prompt
    assert completion == 1_000, completion


def test_prompt_completion_tokens_billed_completion_optional():
    """The billed-completion key is allowed to be absent / None -- the
    override still fires with the completion figure defaulting to 0. This
    matches the case where the bridge only has the prompt figure to stamp.
    """
    from switchyard.hooks import _prompt_completion_tokens
    usage = {
        "prompt_tokens": 12_345,
        "completion_tokens": 678,
        "switchyard_billed_prompt_tokens": 2_400_000,
    }
    prompt, completion = _prompt_completion_tokens(None, usage)
    assert prompt == 2_400_000, prompt
    assert completion == 0, completion


# ---------------------------------------------------------------------------
# Tests: windows_per_month and projection tiers
# ---------------------------------------------------------------------------
#
# Projected monthly capacity follows a tiered basis:
#   - Tier 1 (vendor): the provider gave us an absolute limit / remaining
#     count, so cap = limit × windows_per_month(q.period), basis "vendor".
#   - Tier 2 (bounded): only a percentage is current, but a previous
#     same-window reading was carried forward by note_reported_percent.
#     The pair (Δyard, Δpct) gives A_inf = Δyard × 100 / Δpct, which is
#     an UPPER bound on the true per-window allowance (bypass ≥ 0).
#     monthly_capacity_tokens stays A_cfg × wpm when A_cfg is set;
#     capacity_upper_tokens carries A_inf × wpm.
#   - Tier 3 (ledger): no prev pair. cap = A × wpm with
#     A = q.allowance else observed_allowance_tokens; basis "ledger".
#
# A dollar-kind or unlimited window contributes no token capacity.
# ---------------------------------------------------------------------------


def test_windows_per_month_mapping():
    """The 30-day-month scaling for each window kind.

    Used by `_project_capacity` to fold per-window caps into a monthly
    figure. The mapping is fixed: a 5h rolling window completes
    4.8 cycles/day, 144/month; a week fits 30/7 ≈ 4.29 times in a
    30-day month; "month" / None both stay at 1.0 (the window IS the
    month); "day" is 30 days/month; unknown periods degrade to 1.0 so
    a non-recognised window's cap stays exact rather than being multiplied
    by a wrong factor.
    """
    from switchyard.periods import windows_per_month, DAYS_PER_MONTH

    assert windows_per_month("month") == 1.0
    assert windows_per_month(None) == 1.0
    assert windows_per_month("week") == DAYS_PER_MONTH / 7
    assert windows_per_month("rolling_5h") == DAYS_PER_MONTH * 24 / 5
    assert windows_per_month("day") == float(DAYS_PER_MONTH)
    # Unknown period: degrade to 1.0 (the cap stays exact).
    assert windows_per_month("fortnight") == 1.0


def test_window_headroom_tier1_vendor_absolute_limit():
    """Tier 1: provider gave both remaining and limit -> cap = limit × wpm.

    When `reported_remaining` and `reported_limit` are both current floats,
    the vendor figure is ground truth. The window's projection is
    `vendor_total × windows_per_month`, with `basis = "vendor"` and
    `capacity_upper_tokens = None` (no ceiling needed; the vendor number
    is exact). The plan-level projection inherits the same numbers
    verbatim.
    """
    from switchyard.usage import window_headroom

    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()  # month window, allowance = 1_000_000
    reset_at = time.time() + 3600

    redis.hashes[K_WINDOW.format(plan=plan.key, window=plan.quota.label)] = {
        "reported_remaining": "300000.0",
        "reported_limit": "1000000.0",
        "reported_at": str(time.time()),
        "reset_at": str(reset_at),
    }
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        hr = run(window_headroom(ledger, plan, plan.quota))
    assert hr["pct_used"] == 70.0, hr
    assert hr["limit"] == 1_000_000.0, hr
    # Tier 1: monthly_capacity_tokens = vendor_total × 1.0 (month period).
    assert hr["monthly_capacity_tokens"] == 1_000_000.0, hr
    assert hr["capacity_basis"] == "vendor", hr
    assert hr["capacity_upper_tokens"] is None, hr
    assert hr["cfg_monthly_capacity_tokens"] == 1_000_000.0, hr
    assert hr["off_router_tokens_est"] is None, hr


def test_window_headroom_tier1_reorders_absolutes_before_pct_only():
    """A window with both reported_pct_used AND reported_remaining current
    takes the absolutes branch, not the percent-only branch.

    Before the reorder, the percent-only branch ran first; with no limit
    to derive from, it short-circuited and the absolutes were discarded.
    Putting absolutes first means a window with BOTH gets the vendor
    figure (tier 1) and the percent figure is unused. The bypass-attribution
    issue this avoids: deriving `consumed = pct/100 × remaining` would
    silently drop whatever the account spent outside SwitchYard.
    """
    from switchyard.usage import window_headroom

    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()  # month window
    reset_at = time.time() + 3600

    redis.hashes[K_WINDOW.format(plan=plan.key, window=plan.quota.label)] = {
        "reported_pct_used": "50.0",         # would imply 500_000 used
        "reported_remaining": "700000.0",    # actually 300_000 used
        "reported_limit": "1000000.0",
        "reported_at": str(time.time()),
        "reset_at": str(reset_at),
    }
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        hr = run(window_headroom(ledger, plan, plan.quota))
    # Absolutes win: pct_used derived from (limit - remaining) / limit.
    assert hr["pct_used"] == 30.0, hr
    assert hr["consumed"] == 300_000.0, hr
    assert hr["limit"] == 1_000_000.0, hr
    # Tier 1: vendor basis.
    assert hr["capacity_basis"] == "vendor", hr


def test_window_headroom_tier2_bounded_with_valid_prev_pair():
    """Tier 2: pct current, valid (prev, current) pair -> bounded basis.

    `note_reported_percent` carries the previous pct+yard pair forward
    only when the old reset_at matches the new one (same window in force)
    AND the new pct is >= the old one. With that pair available, the
    inference gives A_inf = Δyard × 100 / Δpct (an UPPER bound on the
    true per-window allowance because some traffic may have run off the
    router). The cfg allowance stays in `monthly_capacity_tokens`; the
    inferred ceiling lives in `capacity_upper_tokens`.
    """
    from switchyard.usage import window_headroom

    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()  # month, allowance 1_000_000
    reset_at = time.time() + 3600

    # First reading: 20% used, 200_000 yard tokens.
    redis.hashes[K_WINDOW.format(plan=plan.key, window=plan.quota.label)] = {
        "reported_pct_used": "20.0",
        "reported_at_yard_tokens": "200000.0",
        "reported_at": str(time.time() - 600),
        "reset_at": str(reset_at),
    }
    # Second reading: 50% used, 400_000 yard tokens. Note the carry
    # happens inside note_reported_percent -- here we simulate the
    # carried-forward state directly so the test focuses on the
    # tier-2 inference in window_headroom.
    redis.hashes[K_WINDOW.format(plan=plan.key, window=plan.quota.label)] = {
        "prev_pct_used": "20.0",
        "prev_pct_yard_tokens": "200000.0",
        "reported_pct_used": "50.0",
        "reported_at_yard_tokens": "400000.0",
        "reported_at": str(time.time()),
        "reset_at": str(reset_at),
    }
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        hr = run(window_headroom(ledger, plan, plan.quota))
    # Inference: A_inf = (400k - 200k) * 100 / (50 - 20) = 666_666.67.
    # A_inf=666_666 < A_cfg=1_000_000 ⇒ A_cfg wins; the inferred ceiling
    # is the UPPER bound (bypass ≥ 0 makes A_inf <= A_true).
    assert hr["pct_used"] == 50.0, hr
    assert hr["capacity_basis"] == "bounded", hr
    assert hr["monthly_capacity_tokens"] == 1_000_000.0, hr
    # capacity_upper_tokens = A_inf × wpm = 666_666.67 × 1.0.
    assert abs(hr["capacity_upper_tokens"] - (200_000 * 100 / 30)) < 1.0, hr
    # off_router = A_cfg * Δpct/100 − Δyard = 1_000_000 * 0.30 - 200_000 = 100_000.
    assert hr["off_router_tokens_est"] == 100_000.0, hr


def test_window_headroom_tier2_bounded_positive_off_router_augments_basis():
    """A positive `off_router_tokens_est` augments the basis string with
    "lower bound (off-router inferred)" phrasing. The board surfaces the
    bypass signal on the affected window without spending a row on a
    label. The wording is distinct from the tier-3 "observed (where it
    ran out last time)" path, so the tooltip text tells the operator
    which case fired even though both glyphs are in the `≤` family.
    """
    from switchyard.usage import window_headroom

    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()
    reset_at = time.time() + 3600

    redis.hashes[K_WINDOW.format(plan=plan.key, window=plan.quota.label)] = {
        "prev_pct_used": "10.0",
        "prev_pct_yard_tokens": "50000.0",
        "reported_pct_used": "60.0",   # Δpct = 50
        "reported_at_yard_tokens": "100000.0",  # Δyard = 50_000
        "reported_at": str(time.time()),
        "reset_at": str(reset_at),
    }
    # A_inf = 50_000 * 100 / 50 = 100_000 (well below A_cfg=1_000_000)
    # off_router = 1_000_000 * 0.50 - 50_000 = 450_000 > 0.
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        hr = run(window_headroom(ledger, plan, plan.quota))
    assert hr["off_router_tokens_est"] == 450_000.0, hr
    assert "lower bound (off-router inferred)" in hr["basis"], hr
    # And the wording is NOT the unrelated tier-3 phrase.
    assert "observed (where it ran out last time)" not in hr["basis"], hr


def test_window_headroom_tier2_degrades_to_tier3_when_no_prev_pair():
    """Tier 2 only fires when the prev pair is valid; without it the
    window degrades to tier 3 (ledger basis). The inference is NEVER
    applied on the strength of a single reading -- the bypass estimate
    needs at least one comparison point, and a single percentage is
    just a number, not an allowance.
    """
    from switchyard.usage import window_headroom

    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = fake_plan()  # allowance = 1_000_000
    reset_at = time.time() + 3600

    # Only one reading on file -- no prev pair to compare against.
    redis.hashes[K_WINDOW.format(plan=plan.key, window=plan.quota.label)] = {
        "reported_pct_used": "30.0",
        "reported_at": str(time.time()),
        "reset_at": str(reset_at),
    }
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        hr = run(window_headroom(ledger, plan, plan.quota))
    assert hr["capacity_basis"] == "ledger", hr
    # Tier 3 with A_cfg = 1_000_000: monthly_capacity = 1_000_000 × 1.0.
    assert hr["monthly_capacity_tokens"] == 1_000_000.0, hr
    assert hr["capacity_upper_tokens"] is None, hr
    assert hr["off_router_tokens_est"] is None, hr


def test_window_headroom_tier3_uses_observed_allowance_when_no_cfg():
    """Tier 3 with no q.allowance falls back to observed_allowance_tokens.

    After one cycle, `note_exhaustion` writes the tokens consumed at
    exhaustion into the window's hash as `observed_allowance_tokens`.
    A plan that never had a configured allowance (e.g. inferred from
    a hard-fail) still gets a headroom projection -- the observed
    figure IS the per-window allowance, scaled to a month via wpm.
    """
    from switchyard.usage import window_headroom

    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = Plan(
        key="obs-plan",
        label="Observed plan",
        models={},
        quotas=[Quota(kind="tokens", period="week", allowance=None)],
        monthly_cost=0.0,
        metered=False,
    )
    reset_at = time.time() + 3600
    redis.hashes[K_WINDOW.format(plan=plan.key, window=plan.quota.label)] = {
        "reported_pct_used": "10.0",
        "observed_allowance_tokens": "300000.0",
        "reported_at": str(time.time()),
        "reset_at": str(reset_at),
    }
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        hr = run(window_headroom(ledger, plan, plan.quota))
    # weekly period: wpm = 30/7 ≈ 4.29.
    expected = 300_000.0 * (30 / 7)
    assert abs(hr["monthly_capacity_tokens"] - expected) < 1.0, hr
    assert hr["capacity_basis"] == "ledger", hr
    assert hr["cfg_monthly_capacity_tokens"] is None, hr


def test_window_headroom_dollar_kind_has_no_token_projection():
    """Dollar-kind windows contribute no token capacity. A dollar
    allowance cannot be projected to tokens -- the spec keeps the
    historical rate with its caveat. The projection fields are all
    None on a dollars window.
    """
    from switchyard.usage import window_headroom

    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = Plan(
        key="dollar-plan",
        label="Dollar plan",
        models={},
        quotas=[Quota(kind="dollars", period="month", allowance=20.0)],
        monthly_cost=0.0,
        metered=True,
    )
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        hr = run(window_headroom(ledger, plan, plan.quota))
    assert hr["monthly_capacity_tokens"] is None, hr
    assert hr["capacity_basis"] is None, hr
    assert hr["capacity_upper_tokens"] is None, hr
    assert hr["cfg_monthly_capacity_tokens"] is None, hr
    assert hr["off_router_tokens_est"] is None, hr


def test_window_headroom_unlimited_has_no_token_projection():
    """Unlimited windows contribute no token capacity. Same shape as
    dollars: an unmetered plan has no cap to project."""
    from switchyard.usage import window_headroom

    redis = FakeRedis()
    ledger = Ledger(redis)
    plan = Plan(
        key="unl-plan",
        label="Unlimited plan",
        models={},
        quotas=[Quota(kind="unlimited", period="week")],
        monthly_cost=0.0,
        metered=False,
    )
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        hr = run(window_headroom(ledger, plan, plan.quota))
    assert hr["monthly_capacity_tokens"] is None, hr
    assert hr["capacity_basis"] is None, hr


def test_headroom_folds_projection_argmin():
    """Plan-level projection is the argmin of per-window monthly caps.

    A plan with two windows (a 5h and a weekly) projects to the smaller
    of the two per-window caps. The basis follows the argmin window:
    a vendor-tier 5h binding below a tier-2 weekly ceiling is exact,
    so the plan-level basis is "vendor". The argmin's window label is
    surfaced so the board can name the binding source.
    """
    from switchyard.usage import headroom

    redis = FakeRedis()
    ledger = Ledger(redis)
    reset_at = time.time() + 3600
    plan = Plan(
        key="two-win",
        label="Two windows",
        models={},
        quotas=[
            Quota(kind="tokens", period="rolling_5h", name="5h",
                  allowance=500_000),
            Quota(kind="tokens", period="week", allowance=40_000_000),
        ],
        monthly_cost=0.0,
        metered=False,
    )
    # 5h window: vendor basis, 500_000 × 144 = 72_000_000/month.
    redis.hashes[K_WINDOW.format(plan=plan.key, window="5h")] = {
        "reported_remaining": "200000.0",
        "reported_limit": "500000.0",
        "reported_at": str(time.time()),
        "reset_at": str(reset_at),
    }
    # Weekly window: no current reading -> falls to tier 3 (ledger).
    with _patch_now(datetime(2025, 9, 21, 14, 30, 0, tzinfo=timezone.utc)):
        hr = run(headroom(ledger, plan))
    assert hr["projection"]["window"] == "5h", hr["projection"]
    # 5h × 144 = 72_000_000; weekly × (30/7) ≈ 171_428_571. Argmin = 5h.
    assert hr["projection"]["capacity_tokens"] == 500_000 * 144, hr["projection"]
    assert hr["projection"]["basis"] == "vendor", hr["projection"]


def test_note_reported_percent_carries_prev_pair_same_window():
    """The first call writes (pct, yard). The second call, with the same
    reset_at and a non-decreasing pct, carries the previous pair to
    `prev_pct_used` / `prev_pct_yard_tokens`. The carry rides the SAME
    hash so `window_facts` continues to fetch everything in one round trip.
    """
    redis = FakeRedis()
    ledger = Ledger(redis)
    reset_at = time.time() + 3600

    async def go():
        await ledger.note_reported_percent(
            "p", 20.0, reset_at, window="weekly", yard_tokens=200_000.0)
        facts = await ledger.window_facts("p", "weekly")
        # First call has no prev pair (this IS the first call).
        assert "prev_pct_used" not in facts, facts
        assert facts["reported_pct_used"] == 20.0, facts
        assert facts["reported_at_yard_tokens"] == 200_000.0, facts

        await ledger.note_reported_percent(
            "p", 50.0, reset_at, window="weekly", yard_tokens=400_000.0)
        facts = await ledger.window_facts("p", "weekly")
        assert facts["reported_pct_used"] == 50.0, facts
        # Carry: prev pair is the FIRST call's values.
        assert facts["prev_pct_used"] == 20.0, facts
        assert facts["prev_pct_yard_tokens"] == 200_000.0, facts
        # New stamp alongside the new reading.
        assert facts["reported_at_yard_tokens"] == 400_000.0, facts
        return facts

    run(go())


def test_note_reported_percent_carry_skipped_on_window_rollover():
    """A new reading whose pct is LOWER than the previous one is treated as
    a window rollover -- the prior pair is dropped, NOT carried forward.
    Carrying it would attach the previous window's yard tally to the new
    window's percentage and project off the wrong bucket.
    """
    redis = FakeRedis()
    ledger = Ledger(redis)
    reset_at = time.time() + 3600

    async def go():
        await ledger.note_reported_percent(
            "p", 80.0, reset_at, window="weekly", yard_tokens=8_000_000.0)
        # A fresh reading with a LOWER pct (window rolled over): carry is skipped.
        await ledger.note_reported_percent(
            "p", 5.0, reset_at, window="weekly", yard_tokens=500_000.0)
        facts = await ledger.window_facts("p", "weekly")
        # The new reading IS on file; the previous pair was NOT carried.
        assert facts["reported_pct_used"] == 5.0, facts
        assert "prev_pct_used" not in facts, facts
        return facts

    run(go())


def test_note_reported_percent_stale_prev_pair_cleared_on_rollover():
    """A stale prev pair from a rolled-over window is cleared, not just
    skipped on the next write.

    The carry path SKIPS writing a new prev_* pair when the rollover or
    reset-mismatch gate fires, but the OLD prev_* pair can survive on
    the hash (a successful carry earlier in the same plan's lifetime
    wrote them). Without an explicit clear, a later
    `_infer_allowance_from_pair` would project arithmetic across two
    unrelated windows: Δpct spanning the boundary, Δyard stretching
    across the rollover. This test pins the explicit-clear behaviour
    (HDEL on the stale pair) so the inference degrades to tier 3
    instead of surfacing a bogus tier-2 ceiling.

    Sequence: R1+R2 build a valid carry (prev=(pct0, yard0)). R3
    triggers a reset_at mismatch (no reset_at vs the original T1) and
    clears the pair. R4 is the inference-time check: even though
    `reported_pct_used` is fresh and current, the cleared pair leaves
    the inference unable to run.
    """
    redis = FakeRedis()
    ledger = Ledger(redis)
    reset_at_1 = time.time() + 3600

    async def go():
        # R1: first reading — seeds the hash with reported_pct + yard.
        await ledger.note_reported_percent(
            "p", 5.0, reset_at_1, window="weekly", yard_tokens=500_000.0)
        # R2: same window, higher pct — carry writes prev=(5, 500K).
        await ledger.note_reported_percent(
            "p", 8.0, reset_at_1, window="weekly", yard_tokens=800_000.0)
        mid = await ledger.window_facts("p", "weekly")
        # Sanity: the carry succeeded.
        assert mid.get("prev_pct_used") == 5.0, mid
        assert mid.get("prev_pct_yard_tokens") == 500_000.0, mid
        # R3: NEW reading with NO reset_at — same_window becomes False
        # because the new reset is None (not equal to reset_at_1).
        # Carry is skipped AND the stale pair is cleared.
        await ledger.note_reported_percent(
            "p", 12.0, None, window="weekly", yard_tokens=1_200_000.0)
        facts = await ledger.window_facts("p", "weekly")
        # The prev pair must be GONE (cleared), not just absent because
        # nothing was written — this is the explicit HDEL the fix added.
        assert "prev_pct_used" not in facts, facts
        assert "prev_pct_yard_tokens" not in facts, facts
        # The current reading IS on file.
        assert facts["reported_pct_used"] == 12.0, facts
        assert facts["reported_at_yard_tokens"] == 1_200_000.0, facts
        # R4: pct-decreasing rollover — carry skipped again, pair still
        # cleared (HDEL is idempotent on absent fields).
        await ledger.note_reported_percent(
            "p", 2.0, None, window="weekly", yard_tokens=200_000.0)
        facts2 = await ledger.window_facts("p", "weekly")
        assert "prev_pct_used" not in facts2, facts2
        assert "prev_pct_yard_tokens" not in facts2, facts2
        return facts2

    run(go())


def test_infer_allowance_from_pair_returns_none_when_prev_cleared():
    """`_infer_allowance_from_pair` returns None when the prev pair has
    been cleared by a rollover / reset-mismatch.

    Regression bar for the stale-pair bug: the inference path must NOT
    promote a half-cleared hash to a tier-2 ceiling. With `prev_pct_used`
    absent, every guard in the inference fires (float(None) raises) and
    the helper returns None — `window_headroom` then falls through to
    the tier-3 (ledger) branch on the next call.
    """
    from switchyard.usage import _infer_allowance_from_pair
    # Hash with the current reading but NO prev pair — the cleared
    # state after a rollover.
    facts = {"reported_pct_used": "30.0",
             "reported_at_yard_tokens": "1500000.0"}
    used = {"prompt_tokens": 1_000_000.0, "completion_tokens": 500_000.0}
    assert _infer_allowance_from_pair(facts, used) is None


def test_note_reported_percent_carry_skipped_on_reset_change():
    """A different reset_at means a different window is in force; the
    previous pair is not about this window and must NOT be carried
    forward. Same defensive logic as the rollover check, on the other
    side of the comparison.
    """
    redis = FakeRedis()
    ledger = Ledger(redis)

    async def go():
        await ledger.note_reported_percent(
            "p", 20.0, time.time() + 3600, window="weekly", yard_tokens=200_000.0)
        # Same pct, different reset_at -> different window, no carry.
        await ledger.note_reported_percent(
            "p", 30.0, time.time() + 7200, window="weekly", yard_tokens=300_000.0)
        facts = await ledger.window_facts("p", "weekly")
        assert facts["reported_pct_used"] == 30.0, facts
        assert "prev_pct_used" not in facts, facts
        return facts

    run(go())


def test_note_reported_percent_without_yard_tokens():
    """The yard_tokens param is optional: callers that don't have the
    token tally still get the carry forward for the percentage side
    alone, but no off-router estimate can be computed (no yard delta).
    The carry gate fires only when both halves of the prev pair are
    present; if `yard_tokens` is None on the previous call, the prev
    pair is never written, so the next pct-only reading degrades to
    tier 3 (no inference possible).
    """
    redis = FakeRedis()
    ledger = Ledger(redis)
    reset_at = time.time() + 3600

    async def go():
        await ledger.note_reported_percent(
            "p", 10.0, reset_at, window="weekly")  # no yard_tokens
        await ledger.note_reported_percent(
            "p", 20.0, reset_at, window="weekly")  # also no yard_tokens
        facts = await ledger.window_facts("p", "weekly")
        # yard_tokens was None on both calls, so neither call wrote
        # reported_at_yard_tokens; with the yard side absent, the
        # carry gate cannot fire on the second call (it requires the
        # prev pair in full), so prev_pct_used / prev_pct_yard_tokens
        # are never written.
        return facts

    facts = run(go())
    # Both calls wrote `reported_pct_used`; the second one's gate is
    # not satisfied (no prev_pct_yard_tokens), so prev_pct_used is
    # NOT written. Tier 2 needs the full pair.
    assert "prev_pct_used" not in facts, facts
    assert "prev_pct_yard_tokens" not in facts, facts


# ---------------------------------------------------------------------------
# Tests: Effective $/Mtok with projection (issue #218)
# ---------------------------------------------------------------------------
#
# `effective_cost_fields` / `model_effective_cost_fields` thread a
# `projected` dict (the same `headroom()['projection']` shape) into the
# rate computation. With `projected=None` the helpers fall back to the
# historical fee/burned shape so existing tests stay meaningful; with a
# projection the subscription rate is `fee / (cap / 1e6)` independent of
# burn — the whole point of the fix.
# ---------------------------------------------------------------------------


def test_effective_cost_fields_stable_across_month():
    """A subscription's projected rate is the same on day 3 as on day 30.

    With the historical `fee / tokens_so_far` shape, the same plan's rate
    drifted downward through the month (the issue's table at $1.000 →
    $0.200 → $0.100). The projection uses the binding window's monthly
    capacity instead, which is fixed for the whole month, so two readings
    at different burn levels return the same figure. The 1M-token
    threshold that the historical branch gates on is also gone — the
    projection is independent of burn.
    """
    plan = fake_plan(monthly_cost=132.0)
    # Same projection at two different burn levels.
    proj = {"capacity_tokens": 40_000_000, "basis": "vendor",
            "upper_tokens": None, "cfg_capacity_tokens": 40_000_000}
    early = effective_cost_fields(plan, 5_000_000, 0.0, projected=proj)
    late = effective_cost_fields(plan, 60_000_000, 0.0, projected=proj)
    assert early["rate"] == late["rate"], (early, late)
    # Vendor tier: 132 / 40 = $3.30/Mtok, no upper.
    assert early["rate"] == 3.30, early
    assert early["basis"] == "vendor", early
    assert early["upper"] is None, early
    # Below-1M burn still produces a number with a projection (no gate).
    very_early = effective_cost_fields(plan, 500_000, 0.0, projected=proj)
    assert very_early["rate"] == 3.30, very_early


def test_effective_cost_fields_tier1_vendor():
    """Tier 1 (vendor): a stated absolute gives an exact rate.

    `basis="vendor"` -> rate = fee / (vendor_cap / 1e6), upper None
    (the vendor number is ground truth, no ceiling needed). The board
    reads basis="vendor" and renders the small "vendor" tag.
    """
    plan = fake_plan(monthly_cost=132.0)
    proj = {"capacity_tokens": 1_000_000, "basis": "vendor",
            "upper_tokens": None, "cfg_capacity_tokens": 1_000_000}
    fields = effective_cost_fields(plan, 5_000_000, 0.0, projected=proj)
    assert fields["rate"] == 132.0, fields  # 132 / 1 = 132
    assert fields["basis"] == "vendor", fields
    assert fields["upper"] is None, fields


def test_effective_cost_fields_tier2_bounded_with_cfg():
    """Tier 2 (bounded) WITH A_cfg: rate at cfg end, upper at inferred ceiling.

    The cfg allowance stays in `monthly_capacity_tokens` (the inference
    NEVER replaces A_cfg); `capacity_upper_tokens` carries the inferred
    ceiling. The board renders `$low–high/Mtok`. This is the case the
    plan calls "tier-2-with-A_cfg renders rate at the configured-allowance
    end and upper at the inferred ceiling".
    """
    plan = fake_plan(monthly_cost=132.0)
    # cfg cap = 1M, inferred ceiling = 666_666.67 (less than cfg)
    proj = {"capacity_tokens": 1_000_000, "basis": "bounded",
            "upper_tokens": 666_666.67, "cfg_capacity_tokens": 1_000_000}
    fields = effective_cost_fields(plan, 5_000_000, 0.0, projected=proj)
    assert fields["rate"] == 132.0, fields  # cfg end: 132 / 1
    assert fields["basis"] == "bounded", fields
    # Inferred-ceiling rate: 132 / 0.66666667 = 198.0
    assert abs(fields["upper"] - 198.0) < 0.01, fields


def test_effective_cost_fields_tier2_bounded_no_cfg():
    """Tier 2 (bounded) WITHOUT A_cfg: ceiling IS the rate, basis stays "bounded".

    With no cfg allowance the inference becomes the monthly capacity.
    The board renders the rate with a `≤` prefix (basis="bounded" tells
    it to), so the operator reads it as an upper bound on the true rate.
    `upper` is suppressed in the helper's return when `upper_cap == cap`
    (the inference equals the cap), so the template does not print the
    same number twice for the no-A_cfg path.
    """
    plan = fake_plan(monthly_cost=132.0)
    proj = {"capacity_tokens": 666_666.67, "basis": "bounded",
            "upper_tokens": 666_666.67, "cfg_capacity_tokens": None}
    fields = effective_cost_fields(plan, 5_000_000, 0.0, projected=proj)
    # Inferred allowance IS the cap: 132 / 0.66666667 = 198.0
    assert abs(fields["rate"] - 198.0) < 0.01, fields
    assert fields["basis"] == "bounded", fields
    # upper is None when the inferred ceiling equals the cap -- the
    # template would render the same number twice otherwise.
    assert fields["upper"] is None, fields


def test_effective_cost_fields_tier2_bounded_inferred_exceeds_cfg():
    """Tier 2 (bounded) with A_inf > A_cfg: template renders the range
    with the inferred-ceiling end as the low side.

    When the cfg allowance under-counted the plan, the inference gives
    A_inf > A_cfg. The rate derived from `cap` (cfg end) is higher than
    the rate derived from `upper_cap` (inferred-ceiling end). The helper
    still returns `rate = fee/cap` and `upper = fee/upper_cap`; the
    template sorts them into `low = upper` and `high = rate`, attaching
    `≤` to `high` so the upper bound always reads as the high end. This
    is the regression bar for the template's low/high ordering -- every
    prior test fixture had A_inf < A_cfg and missed the inversion.
    """
    plan = fake_plan(monthly_cost=10.0)
    # cfg allowance 1M (smaller cap ⇒ higher rate), inferred 2M
    # (larger cap ⇒ lower rate). Truth: $5/Mtok ≤ actual ≤ $10/Mtok.
    proj = {"capacity_tokens": 1_000_000, "basis": "bounded",
            "upper_tokens": 2_000_000, "cfg_capacity_tokens": 1_000_000,
            "window": "monthly"}
    fields = effective_cost_fields(plan, 500_000, 0.0, projected=proj)
    # Helper returns the cfg-end rate as `rate` and the inferred-ceiling
    # rate as `upper`. Template re-sorts to low/high on display; the
    # helper's "rate" is the higher of the two when A_inf > A_cfg.
    assert fields["rate"] == 10.0, fields  # fee / 1M
    assert fields["upper"] == 5.0, fields  # fee / 2M
    # The board's effective display is "low ≤ high" with ≤ on the
    # high end -- assert the helper's two values so the template's
    # sort logic has a stable contract to render against.
    assert fields["basis"] == "bounded", fields


def test_effective_cost_fields_tier3_ledger():
    """Tier 3 (ledger): configured allowance, no projection ceiling.

    Falls through to the historical shape: fee / cap / 1M. The board
    shows the rate with a "yard" tag (basis="ledger") — assumes all
    traffic went through SwitchYard. upper is None.
    """
    plan = fake_plan(monthly_cost=132.0)
    proj = {"capacity_tokens": 5_000_000, "basis": "ledger",
            "upper_tokens": None, "cfg_capacity_tokens": 5_000_000}
    fields = effective_cost_fields(plan, 5_000_000, 0.0, projected=proj)
    # 132 / 5 = 26.4
    assert fields["rate"] == 26.4, fields
    assert fields["basis"] == "ledger", fields
    assert fields["upper"] is None, fields


def test_effective_cost_fields_metered_ignores_projection():
    """Metered plans keep the historical rate even with a projection dict.

    The plan's economics are pay-as-you-go — there is no quota window
    to project from, so the rate is `metered_cost / tokens_so_far`,
    basis "ledger", upper None. The projection dict, if passed, is
    silently ignored on the metered branch.
    """
    plan = fake_plan(monthly_cost=0.0, metered=True)
    proj = {"capacity_tokens": 1_000_000, "basis": "vendor",
            "upper_tokens": None, "cfg_capacity_tokens": 1_000_000}
    fields = effective_cost_fields(plan, 40_000_000, 20.0, projected=proj)
    assert fields["rate"] == 0.50, fields  # 20 / 40
    assert fields["basis"] == "ledger", fields  # not "vendor"
    assert fields["upper"] is None, fields


def test_effective_cost_fields_no_projection_historical_fallback():
    """No projection: subscription falls back to the historical shape.

    Keeps the existing tests meaningful: with `projected=None` the rate
    is `fee / tokens_so_far` (subject to the 1M-token gate), so the
    original `test_effective_cost_per_mtok_unchanged` still passes.
    """
    plan = fake_plan(monthly_cost=132.0)
    fields = effective_cost_fields(plan, 40_000_000, 0.0, projected=None)
    assert fields["rate"] == 3.30, fields
    assert fields["basis"] == "ledger", fields


def test_effective_cost_fields_5pct_to_10pct_upper_bound_scenario():
    """The issue's 5%→10% scenario: 1B SwitchYard delta ⇒ 20B inferred ceiling
    against a 40B truth. The displayed rate is an UPPER bound, not a fact.

    Concrete numbers from issue #218:
      - Vendor reports 5% used; the operator then runs 1B tokens through
        SwitchYard AND 1B tokens through the vendor CLI directly.
      - Vendor truth: 5pp = 2B total ⇒ true allowance 40B.
      - Inference from SwitchYard's delta alone: 1B / 0.05 = 20B — half
        the truth, so the projected rate comes out 2× too high.

    The direction of error is fixed and provable: bypass ≥ 0 ⇒
    A_inferred ≤ A_true ⇒ the inferred $/Mtok is always an upper bound.
    This test asserts the helper returns the upper-bound rate, not a
    bare point estimate, when A_cfg exists (tier-2-with-A_cfg path).
    """
    plan = fake_plan(monthly_cost=100.0)
    # Configure the bound: A_cfg = 40B (the truth), inferred = 20B.
    # monthly_capacity_tokens = 40B × wpm; capacity_upper_tokens = 20B × wpm.
    from switchyard.periods import windows_per_month
    wpm = windows_per_month("month")  # = 1.0
    cfg_cap = 40_000_000_000 * wpm
    inferred_cap = 20_000_000_000 * wpm
    proj = {"capacity_tokens": cfg_cap, "basis": "bounded",
            "upper_tokens": inferred_cap, "cfg_capacity_tokens": cfg_cap}
    fields = effective_cost_fields(plan, 1_000_000_000, 0.0, projected=proj)
    # rate = 100 / 40000 = $0.0025/Mtok (cfg end)
    assert abs(fields["rate"] - (100.0 / 40_000.0)) < 1e-9, fields
    # upper = 100 / 20000 = $0.005/Mtok (inferred ceiling)
    assert abs(fields["upper"] - (100.0 / 20_000.0)) < 1e-9, fields
    # The ceiling IS the upper bound on the true rate. The board renders
    # this as $0.0025–0.005/Mtok so the operator sees the bound, not a
    # single fabricated figure.
    assert fields["basis"] == "bounded", fields
    assert fields["upper"] > fields["rate"], (fields["rate"], fields["upper"])


def test_model_effective_cost_fields_subscription_projects_from_binding():
    """Per-model rows inherit the plan-level projection.

    The per-model share cancels for subscriptions, so the projected rate
    is the same plan-level figure every model shows. The portal passes
    the same `projected` to every row via `_model_eff_fields`, so the
    rendered values match across the row.
    """
    plan = fake_plan(monthly_cost=132.0)
    proj = {"capacity_tokens": 40_000_000, "basis": "vendor",
            "upper_tokens": None, "cfg_capacity_tokens": 40_000_000}
    # Big model slice, small model slice — same rate.
    big = model_effective_cost_fields(
        plan, 25_000_000, 0.0, 40_000_000, projected=proj)
    small = model_effective_cost_fields(
        plan, 5_000_000, 0.0, 40_000_000, projected=proj)
    assert big["rate"] == small["rate"] == 3.30, (big, small)
    assert big["basis"] == small["basis"] == "vendor", (big, small)
    # Below-1M model tokens still produces a number with a projection —
    # no gate on the per-model share when projecting.
    tiny = model_effective_cost_fields(
        plan, 100_000, 0.0, 40_000_000, projected=proj)
    assert tiny["rate"] == 3.30, tiny


def test_model_effective_cost_fields_subscription_historical_share_cancels():
    """Without a projection, the per-model share still cancels.

    Backwards-compat path: subscription rows gate on 1M plan tokens,
    spend_m = fee·m/t, and `spend_m / (m/1M) = fee / (t/1M)` — the
    plan's own rate, identical for every model.
    """
    plan = fake_plan(monthly_cost=132.0)
    a = model_effective_cost_fields(
        plan, 20_000_000, 0.0, 40_000_000, projected=None)
    b = model_effective_cost_fields(
        plan, 5_000_000, 0.0, 40_000_000, projected=None)
    assert a["rate"] == b["rate"] == 3.30, (a, b)
    assert a["basis"] == b["basis"] == "ledger", (a, b)


def test_effective_cost_fields_below_1m_no_projection():
    """Below 1M tokens with no projection returns None — same as before."""
    plan = fake_plan(monthly_cost=132.0)
    fields = effective_cost_fields(plan, 500_000, 0.0, projected=None)
    assert fields["rate"] is None, fields
    assert fields["basis"] is None, fields
    assert fields["upper"] is None, fields


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
