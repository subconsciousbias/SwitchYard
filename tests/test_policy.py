"""Tests for the two adaptive controllers.

Concurrency learning: does a refusal actually lower the cap, per hour of day,
and does probing require real demand?
Pacing: does a plan that is ahead of budget get throttled, does one that is
behind get its full learned cap, and does a truncated final window pace harder?
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import replace
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()

from switchyard import models                        # noqa: E402
from switchyard.periods import deadline, windows_remaining  # noqa: E402
from switchyard.picker import Picker                 # noqa: E402
from switchyard.policy import CapacityPolicy         # noqa: E402
from switchyard.slots import SlotTable               # noqa: E402
from switchyard.usage import Ledger                  # noqa: E402
from tests.fake_redis import FakeRedis               # noqa: E402

run = asyncio.run


def build(pacing: bool = False, **pacing_kw):
    reg = models.load()
    settings = replace(reg.settings,
                       pacing=replace(reg.settings.pacing, enabled=pacing, **pacing_kw))
    reg = models.Registry(settings=settings, plans=reg.plans, lanes=reg.lanes)
    redis = FakeRedis()
    ledger = Ledger(redis)
    policy = CapacityPolicy(redis, settings, ledger)
    slots = SlotTable(redis, settings.inflight_max_age_seconds)
    return reg, redis, ledger, policy, Picker(reg, slots, policy)


# --------------------------------------------------------------- learning ----
def test_connection_refusal_halves_the_cap():
    async def go():
        reg, redis, _, policy, _ = build()
        plan = reg.plans["minimax-ultra"]
        before = (await policy.effective(plan)).cap
        learned = await policy.learner.note_rejection(plan, at_concurrency=before)
        after = (await policy.effective(plan)).cap
        return before, learned, after
    before, learned, after = run(go())
    assert after == learned == before // 2, (before, learned, after)
    print(f"  refused at {before} concurrent -> learned cap {after}")


def test_learning_is_per_hour_of_day():
    """A refusal at one hour must not constrain a different hour."""
    async def go():
        reg, redis, _, policy, _ = build()
        plan = reg.plans["minimax-ultra"]
        await policy.learner.note_rejection(plan, at_concurrency=4)
        this_hour = datetime.now(timezone.utc).hour
        # The bucket for another hour has no samples of its own, so it inherits
        # the global learning rather than an unrelated hour's number.
        keys = [k for k in redis.hashes if ":learn:minimax-ultra:" in k]
        return this_hour, sorted(k.split(":")[-1] for k in keys)
    hour, buckets = run(go())
    assert f"h{hour:02d}" in buckets and "global" in buckets, buckets
    print(f"  refusal at hour {hour:02d} recorded in buckets: {buckets}")


def test_probing_up_requires_unmet_demand_and_quiet():
    async def go():
        reg, redis, _, policy, picker = build()
        plan = reg.plans["minimax-ultra"]
        await policy.learner.note_rejection(plan, at_concurrency=4)   # cap -> 2
        capped = (await policy.effective(plan)).cap

        # No demand yet: must not probe, however long we wait.
        for bucket in ("global", f"h{datetime.now(timezone.utc).hour:02d}"):
            await redis.hset(f"sy:learn:{plan.key}:{bucket}",
                             mapping={"changed_at": time.time() - 10_000,
                                      "last_rejection_at": time.time() - 10_000})
        idle = (await policy.effective(plan)).cap

        # Now register unmet demand: three claims denied by the cap.
        for _ in range(3):
            await policy.learner.note_pressure(plan.key)
        probed = (await policy.effective(plan)).cap
        return capped, idle, probed
    capped, idle, probed = run(go())
    assert idle == capped, f"probed with no demand: {capped} -> {idle}"
    assert probed == capped + 1, (capped, probed)
    print(f"  cap {capped}: idle stays {idle}, under demand probes to {probed}")


def test_auto_plans_start_at_the_seed():
    """`max_parallel: auto` means "learn it", starting from the seed.

    Built here rather than read from the config, so the test survives every plan
    being given an explicit cap — it is about the feature, not today's numbers.
    """
    from dataclasses import replace
    reg = models.load()
    seed = reg.settings.concurrency_learning.seed_cap
    learned = replace(reg.plans["glm"], configured_parallel=None)
    assert learned.max_parallel == seed, learned.max_parallel
    # An explicit cap is used as-is.
    fixed = replace(reg.plans["glm"], configured_parallel=3)
    assert fixed.max_parallel == 3
    print(f"  auto -> seed {seed}; explicit -> that number")


# ----------------------------------------------------------------- pacing ----
async def _pace(plan_key: str, consumed_frac: float, allowance: float = 100_000_000,
                rate_per_slot: float = 5000.0, pacing: bool = True):
    """Put a plan at a given consumption level and ask what the pacer wants."""
    reg, redis, ledger, policy, _ = build(pacing=pacing)
    base = reg.plans[plan_key]
    plan = replace(base, quotas=(replace(base.quota, allowance=allowance),))
    from switchyard.usage import K_PERIOD, period_key
    key = K_PERIOD.format(plan=plan.key, period=period_key(plan.quota.period))
    redis.hashes[key] = {"prompt_tokens": str(allowance * consumed_frac),
                         "completion_tokens": "0", "cost": "0", "requests": "1"}
    await redis.hset(f"sy:pace:{plan.key}", mapping={"rate": rate_per_slot})
    return plan, await policy.pace_state(plan), await policy.effective(plan)


def test_ahead_of_budget_holds_the_plan_closed():
    """Well past the line -> stop, let the line catch up.

    Closing is the only way to slow below one continuously busy slot, which at
    real LLM throughput is already far too fast for a weekly allowance. The
    overspend is measured from the current elapsed fraction so the test means
    the same thing on any day of any window.
    """
    async def go():
        _, at_zero, _ = await _pace("minimax-ultra", consumed_frac=0.0)
        well_ahead = min(0.99, at_zero["elapsed_frac"] + 0.25)
        plan, st, cap = await _pace("minimax-ultra", consumed_frac=well_ahead)
        return plan, st, cap
    plan, st, cap = run(go())
    assert st["active"] and cap.cap == 0, (st["reason"], cap.cap)
    assert "ahead of pace" in st["reason"] and st["ahead_by"] > 0
    print(f"  spent to {st['elapsed_frac']*100:.0f}%+25% elapsed "
          f"(pace line {st['pace_line']/1e6:.0f}M, ahead by {st['ahead_by']/1e6:.0f}M) "
          f"-> {cap.cap} slots: {st['reason']}")


def test_behind_budget_uses_the_full_learned_cap():
    """Badly behind with little time left -> wide open, up to the learned cap."""
    async def go():
        # 1% spent with ~11 days left: the target rate is far above what one
        # slow slot can deliver, so the cap should clamp to the learned limit.
        return await _pace("minimax-ultra", consumed_frac=0.01, rate_per_slot=30.0)
    plan, st, cap = run(go())
    assert cap.cap == cap.learned, (cap.cap, cap.learned, st)
    print(f"  1% spent -> full {cap.cap} slots (pacing never exceeds the learned limit)")


def test_on_pace_uses_the_throughput_derived_cap():
    """Just behind the line -> open, but only as wide as the burn rate allows.

    The consumption is derived from how far through the window we actually are,
    not hardcoded: a fraction that sits just under a monthly line is well over a
    weekly one, so a fixed number silently means different things per plan and
    per day. This broke when minimax's target window became weekly.
    """
    async def go():
        # Where the pace line is right now, at zero spend.
        _, at_zero, _ = await _pace("minimax-ultra", consumed_frac=0.0)
        just_behind = max(0.0, at_zero["elapsed_frac"] - 0.05)
        return await _pace("minimax-ultra", consumed_frac=just_behind,
                           rate_per_slot=200.0)
    plan, st, cap = run(go())
    assert cap.cap >= 1 and st["reason"].startswith("pacing "), (st["reason"], cap)
    print(f"  60% spent vs {st['pace_line']/1e6:.0f}M line -> {cap.cap} slots "
          f"(target {st['target_rate']:.0f} units/s at {st['rate_per_slot']:.0f}/slot)")


def test_spent_allowance_closes_the_plan():
    async def go():
        _, st, cap = await _pace("minimax-ultra", consumed_frac=1.0)
        return st, cap
    st, cap = run(go())
    assert st["reason"].endswith("allowance spent") and cap.cap == 0, (st["reason"], cap)
    print("  allowance spent -> 0 slots, lane narrows instead of spilling")


def test_truncated_final_window_paces_harder():
    """A cancelled plan's last window has less time for the same allowance."""
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    normal, is_final_n = deadline("month", None, now)
    final, is_final_f = deadline("month", date(2026, 9, 27), now)
    assert not is_final_n and is_final_f
    hours_normal = (normal - now).total_seconds() / 3600
    hours_final = (final - now).total_seconds() / 3600
    assert hours_final < hours_normal
    # Same allowance over less time means a proportionally higher target rate.
    ratio = hours_normal / hours_final
    print(f"  ongoing plan has {hours_normal:.0f}h left, expiring plan {hours_final:.0f}h "
          f"-> target rate {ratio:.1f}x higher")
    assert ratio > 2


def test_intermediate_windows_are_not_merged():
    """Quota resets per window, so a cancelled plan still has whole windows."""
    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    full, tail_seconds = windows_remaining("week", date(2026, 10, 4), now)
    assert full >= 1 and tail_seconds > 0
    dl, is_final = deadline("week", date(2026, 10, 4), now)
    assert not is_final, "this week is not the final window; it rolls over first"
    print(f"  openai seat: {full} more full week(s) then a "
          f"{tail_seconds/86400:.0f}d final window — each paced separately")


def test_metered_and_local_plans_are_never_paced():
    async def go():
        reg, _, _, policy, _ = build(pacing=True)
        # Metered plans and the unmetered local box have no allowance to land on.
        for key in ("openrouter", "local-box"):
            assert not await policy.plan_is_paced(reg.plans[key]), key
        for key in ("minimax-ultra", "grok", "claude-max", "glm"):
            assert await policy.plan_is_paced(reg.plans[key]), key
    run(go())
    print("  paced: subscriptions only; metered and local keep fixed caps")


def test_pacing_can_be_switched_at_runtime():
    """The portal toggle overrides plans.yaml without a reload."""
    async def go():
        reg, _, _, policy, _ = build(pacing=False)
        assert not await policy.pacing_enabled()
        assert await policy.set_pacing(True) is True
        assert await policy.plan_is_paced(reg.plans["minimax-ultra"])
        assert not await policy.tail_enabled()
        await policy.set_pacing(None)            # back to the configured default
        return await policy.pacing_enabled()
    assert run(go()) is False
    print("  runtime toggle: off -> on -> cleared back to the configured default")


def test_pacing_mode_disables_the_tail():
    async def go():
        reg_off, *_, picker_off = build(pacing=False)
        reg_on, *_, picker_on = build(pacing=True)
        return ([m.ref for m in await picker_off._members("forge")],
                [m.ref for m in await picker_on._members("forge")])
    off, on = run(go())
    assert "local-box/qwen" in off and "local-box/qwen" not in on
    print(f"  tail present when pacing off ({off[-1]}), absent when on")




# ------------------------------------------------------- multiple windows ----
# A fixed clock, so "how far into the week are we" is not the test's problem.
# Wednesday noon: ~36% through the ISO week, ~40% through September.
NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


async def _multi(consumed_5h: float, consumed_week: float,
                 allow_5h: float = 2_000_000, allow_week: float = 40_000_000,
                 rate_per_slot: float = 500.0):
    """Put a two-window plan at given consumption in each window."""
    from switchyard.models import Quota
    from switchyard.usage import K_PERIOD, period_key
    reg, redis, ledger, policy, _ = build(pacing=True)
    base = reg.plans["claude-max"]
    plan = replace(base, configured_parallel=4,
                   max_parallel_ceiling=4, quotas=(
        Quota(name="5h", role="constraint", kind="tokens",
              period="rolling_5h", allowance=allow_5h),
        Quota(name="weekly", role="target", kind="tokens",
              period="week", allowance=allow_week),
    ))
    for q, used in ((plan.quotas[0], consumed_5h), (plan.quotas[1], consumed_week)):
        key = K_PERIOD.format(plan=plan.key, period=period_key(q.period, NOW))
        redis.hashes[key] = {"prompt_tokens": str(used), "completion_tokens": "0",
                             "cost": "0", "requests": "1"}
    await redis.hset(f"sy:pace:{plan.key}", mapping={"rate": rate_per_slot})
    return plan, await policy.pace_state(plan, NOW), await policy.effective(plan, NOW)


def test_weekly_is_the_target_and_5h_is_a_constraint():
    reg = models.load()
    plan = reg.plans["claude-max"]
    assert plan.quota.label == "weekly" and plan.quota.is_target
    assert [q.label for q in plan.constraints] == ["5h"]
    print(f"  claude-max: target={plan.quota.label}, constraints={[q.label for q in plan.constraints]}")


def test_5h_window_caps_the_rate_even_when_weekly_is_wide_open():
    """Behind on weekly, but the burst window would be blown -> 5h binds."""
    async def go():
        # Barely any of the week used, but most of this 5h window is gone.
        return await _multi(consumed_5h=1_900_000, consumed_week=1_000_000)
    plan, st, cap = run(go())
    assert st["binding"] == "5h", st
    assert "capped by 5h" in st["reason"], st["reason"]
    print(f"  weekly 2.5% used but 5h 95% used -> binding={st['binding']}, "
          f"{cap.cap} slot(s): {st['reason']}")


def test_weekly_binds_when_the_burst_window_is_fresh():
    async def go():
        return await _multi(consumed_5h=0, consumed_week=1_000_000, rate_per_slot=50.0)
    plan, st, cap = run(go())
    assert st["binding"] == "weekly", st
    print(f"  fresh 5h window -> binding={st['binding']}, {cap.cap} slot(s): {st['reason']}")


def test_spent_5h_window_holds_the_plan_without_touching_weekly():
    async def go():
        return await _multi(consumed_5h=2_000_000, consumed_week=1_000_000)
    plan, st, cap = run(go())
    assert cap.cap == 0 and st["binding"] == "5h", st
    assert st["windows"][1]["consumed_frac"] < 0.1, "weekly must be untouched"
    print(f"  5h spent -> 0 slots ({st['reason']}), weekly still at "
          f"{st['windows'][1]['consumed_frac']*100:.1f}%")


def test_holding_is_driven_by_the_target_window_only():
    """Ahead on the 5h line is fine; ahead on weekly is what stops us."""
    async def go():
        # 60% of the weekly allowance gone by Wednesday noon (~36% elapsed):
        # comfortably ahead of the line, whatever the exact hour.
        return await _multi(consumed_5h=0, consumed_week=24_000_000)
    plan, st, cap = run(go())
    assert cap.cap == 0 and "ahead of pace on weekly" in st["reason"], st["reason"]
    print(f"  60% of weekly used at {st['target']['elapsed_frac']*100:.0f}% elapsed "
          f"-> {cap.cap} slots: {st['reason']}")


def test_exhaustion_is_attributed_to_the_right_window():
    """A 2-hour reset means the 5h window; a 3-day reset means the weekly one."""
    async def go():
        from switchyard.models import Quota
        reg, redis, ledger, policy, _ = build(pacing=True)
        plan = replace(reg.plans["claude-max"], quotas=(
            Quota(name="5h", role="constraint", period="rolling_5h", kind="tokens"),
            Quota(name="weekly", role="target", period="week", kind="tokens"),
        ))
        # Fixed clock: late on a Sunday the weekly rollover is genuinely nearer
        # than the 5-hour one, so a wall-clock test here is not measuring the
        # attribution logic.
        base = NOW.timestamp()
        soon = ledger.attribute_window(plan, base + 2 * 3600, NOW)
        later = ledger.attribute_window(plan, base + 3 * 86400, NOW)
        blind = ledger.attribute_window(plan, None, NOW)
        return soon.label, later.label, blind.label
    soon, later, blind = run(go())
    assert soon == "5h" and later == "weekly" and blind == "5h", (soon, later, blind)
    print(f"  reset in 2h -> {soon}; reset in 3d -> {later}; no hint -> {blind} (safest)")


def test_partial_knowledge_still_protects_the_known_window():
    """Weekly allowance unknown, 5h known: still worth constraining."""
    async def go():
        from switchyard.models import Quota
        from switchyard.usage import K_PERIOD, period_key
        reg, redis, ledger, policy, _ = build(pacing=True)
        plan = replace(reg.plans["claude-max"], configured_parallel=4,
                       max_parallel_ceiling=4, quotas=(
            Quota(name="5h", role="constraint", kind="tokens",
                  period="rolling_5h", allowance=2_000_000),
            Quota(name="weekly", role="target", kind="tokens", period="week", allowance=None),
        ))
        key = K_PERIOD.format(plan=plan.key, period=period_key("rolling_5h", NOW))
        redis.hashes[key] = {"prompt_tokens": "1900000", "completion_tokens": "0",
                             "cost": "0", "requests": "1"}
        await redis.hset(f"sy:pace:{plan.key}", mapping={"rate": 500.0})
        return await policy.pace_state(plan, NOW), await policy.effective(plan, NOW)
    st, cap = run(go())
    assert st["active"] and st["binding"] == "5h", st
    assert cap.cap < (cap.learned or 99), (cap.cap, cap.learned)
    print(f"  weekly unknown, 5h at 95% -> still capped to {cap.cap} by {st['binding']}")


def test_learning_never_probes_above_the_configured_limit_without_a_ceiling():
    """A stated max_parallel is a stated limit, not a starting suggestion.

    The default ceiling used to be 4x the configured value, so a plan set to 2
    silently climbed to 8: the capacity board then drew more slots than
    plans.yaml declared, with nothing on it saying why. Probing past a stated
    limit needs `max_parallel_ceiling` to say how far. Backing off below it
    still happens on its own, which is the half worth having automatic.
    """
    async def go():
        reg, redis, _, policy, _ = build()
        # No ceiling declared: the configured value is the hard stop.
        plan = replace(reg.plans["glm"], max_parallel_ceiling=None)
        assert plan.configured_parallel == 2, plan.configured_parallel

        async def probe_hard(p):
            for bucket in ("global", f"h{datetime.now(timezone.utc).hour:02d}"):
                await redis.hset(f"sy:learn:{p.key}:{bucket}",
                                 mapping={"changed_at": time.time() - 10_000,
                                          "last_rejection_at": time.time() - 10_000})
            for _ in range(5):
                await policy.learner.note_pressure(p.key)
            return (await policy.effective(p)).cap

        capped = plan.configured_parallel
        for _ in range(4):                      # sustained unmet demand
            assert await probe_hard(plan) <= capped, "probed past the configured limit"

        # Declaring a ceiling is what permits it, and it stops exactly there.
        roomy = replace(plan, max_parallel_ceiling=3)
        seen = {await probe_hard(roomy) for _ in range(4)}
        assert max(seen) == 3, seen
        return capped, sorted(seen)

    capped, seen = run(go())
    print(f"  no ceiling: stays at {capped}; ceiling 3: reaches {seen[-1]}")


def test_a_plan_can_opt_out_of_learning_entirely():
    """`learning: false` pins a plan's cap to its configured value.

    local-box is the case it exists for: a local server queues requests instead
    of refusing them, so the learner sees only success, reads that as headroom,
    and probes upward while the real effect is a longer queue and worse latency.
    """
    async def go():
        reg, redis, _, policy, _ = build()
        plan = reg.plans["local-box"]
        assert plan.learning is False, "local-box should opt out in plans.yaml"
        assert not plan.learns(reg.settings)

        # Neither a refusal nor sustained demand may move it.
        await policy.learner.note_rejection(plan, at_concurrency=2)
        for _ in range(5):
            await policy.learner.note_pressure(plan.key)
        for bucket in ("global", f"h{datetime.now(timezone.utc).hour:02d}"):
            await redis.hset(f"sy:learn:{plan.key}:{bucket}",
                             mapping={"cap": 7, "changed_at": time.time() - 10_000})
        eff = await policy.effective(plan)

        # A plan that has not opted out still learns, so this is an opt-out and
        # not an accidental global off-switch.
        other = reg.plans["minimax-ultra"]
        await policy.learner.note_rejection(other, at_concurrency=4)
        moved = (await policy.effective(other)).cap
        return eff.cap, eff.reason, plan.configured_parallel, moved, other.configured_parallel

    cap, reason, configured, moved, other_configured = run(go())
    assert cap == configured, (cap, configured)
    assert reason == "configured", reason
    assert moved < other_configured, (moved, other_configured)
    print(f"  local-box pinned at {cap} ({reason}); minimax-ultra still moved to {moved}")


# ----------------------------------------------------- gate headroom ------
# The gateway-side effective cap for a CLI-backed plan sits one slot below the
# plan's physical gate (which the sidecar reads directly from plans.yaml). The
# headroom absorbs a claim/release race instead of producing a spurious 429.
# API plans are unaffected: their provider enforces the limit and the headroom
# would just sit unused.
def test_gate_headroom_applied_for_cli_backed_plan():
    """A 4-slot CLI plan yields an effective cap of 3; the sidecar still sees 4.

    Built here rather than read from the config so the test survives an
    operator deciding to bump headroom: it is about the feature, not today's
    number. The reason string must mention the headroom so the portal board
    explains why the lane draws one fewer slot than plans.yaml declares.
    """
    async def go():
        reg, _, _, policy, _ = build()
        plan = replace(reg.plans["claude-max"], configured_parallel=4,
                       max_parallel_ceiling=4)
        cap = await policy.effective(plan)
        return plan, cap
    plan, cap = run(go())
    assert plan.max_parallel == 4
    assert cap.cap == 3, cap.cap
    assert "gate headroom" in cap.reason, cap.reason
    print(f"  cli plan max_parallel={plan.max_parallel} -> cap {cap.cap} "
          f"(reason: {cap.reason})")


def test_gate_headroom_floored_at_one():
    """A single-connection CLI plan must stay usable: cap 1, never 0.

    The configured headroom is 1, but `max(1, 1 - 1) == 1`, so the application
    floor catches what arithmetic alone would collapse to zero — locking a
    one-slot plan would be worse than the race it was meant to absorb.
    """
    async def go():
        reg, _, _, policy, _ = build()
        plan = replace(reg.plans["claude-max"], configured_parallel=1,
                       max_parallel_ceiling=1)
        return plan, await policy.effective(plan)
    plan, cap = run(go())
    assert plan.max_parallel == 1
    assert cap.cap == 1, cap.cap
    print(f"  cli plan max_parallel={plan.max_parallel} -> cap {cap.cap} "
          f"(floor held against headroom=1)")


def test_gate_headroom_not_applied_for_api_plan():
    """An API plan keeps its full configured cap — the learner manages its limit.

    The provider rejects on its own, the learner backs off, and a headroom
    slot would sit unused. So the cap is exactly what plans.yaml declares, and
    the reason is the plain learner/seed one with no headroom tag.
    """
    async def go():
        reg, _, _, policy, _ = build()
        plan = reg.plans["minimax-ultra"]
        assert not plan.is_cli_backed
        cap = await policy.effective(plan)
        return plan, cap
    plan, cap = run(go())
    assert cap.cap == plan.max_parallel, (cap.cap, plan.max_parallel)
    assert "gate headroom" not in cap.reason, cap.reason
    print(f"  api plan max_parallel={plan.max_parallel} -> cap {cap.cap} "
          f"(no headroom applied)")


def test_learner_can_still_pull_cap_below_headroom():
    """A learned cap below (configured - headroom) wins: the learner is the floor.

    The learner lowers the cap on a provider-side refusal. With a CLI plan of
    4 and headroom 1, the configured-minus-headroom ceiling is 3, but the
    learner halves a rejection at concurrency 3 down to 1 — and 1 is what the
    picker should hand out. The min still wins; the reason is the learner's,
    not headroom's, because the learner was the binding constraint this time.
    """
    async def go():
        reg, _, _, policy, _ = build()
        plan = replace(reg.plans["claude-max"], configured_parallel=4,
                       max_parallel_ceiling=4)
        # Provider refused us at concurrency 3 — cap halves to floor 1.
        await policy.learner.note_rejection(plan, at_concurrency=3)
        return plan, await policy.effective(plan)
    plan, cap = run(go())
    # Learned cap is 1, headroom-applied ceiling is 3; min wins.
    assert cap.cap == 1, cap.cap
    assert cap.learned == 1, cap.learned
    # The cap_reason is the learner's, not the headroom's: the learner was the
    # binding constraint, so headroom did not further reduce the cap.
    assert "learned" in cap.reason, cap.reason
    print(f"  learner pulled cap to {cap.learned} below headroom ceiling 3 -> "
          f"effective {cap.cap} (reason: {cap.reason})")


def test_cap_reason_mentions_gate_headroom():
    """The portal board reads cap_reason to explain the gap from the declared cap.

    A 4-slot CLI plan should print "configured + gate headroom 1" so the lane
    board can show that the gap to plans.yaml is headroom, not a learner or
    pacing decision. Without this the row would look like an unexplained
    mystery of one missing slot.
    """
    async def go():
        reg, _, _, policy, _ = build()
        plan = replace(reg.plans["claude-max"], configured_parallel=4,
                       max_parallel_ceiling=4)
        return await policy.effective(plan)
    cap = run(go())
    assert "gate headroom" in cap.reason
    assert cap.reason.endswith("1") or cap.reason.endswith(" 1"), cap.reason
    print(f"  cap_reason: {cap.reason!r}")


# ----------------------------------- percent-only estimation ------------
# A subscription whose probe only publishes a percentage (`reported_pct_used`,
# no token or dollar counts) used to leave the pacer idle: `allowance: null`
# plus `consumed / observed_allowance = 0 / 0` meant "no allowance known yet"
# even after a fresh probe. The fix inverts the percentage into an allowance
# estimate (`consumed / (pct/100)`) so the rest of the pacer reads in its own
# units and the plan idles only when the gates (fresh probe, positive
# consumption, sensible percentage) say it should.
async def _pct_plan(consumed_5h: float, consumed_week: float,
                    pct_5h: float | None, pct_week: float | None,
                    *, reset_at_5h: float | None = None,
                    reset_at_week: float | None = None,
                    reported_at: float | None = None,
                    rate_per_slot: float = 500.0):
    """Build a percent-only multi-window plan with probe facts written.

    Same shape as `_multi`, but both windows have `allowance: None` and the
    caller controls the probe readings directly via Redis (`sy:qwin:...`).
    `pct_*` None means "no probe facts for this window" — the branch under
    test then stays idle. `reported_at` defaults to NOW so a fresh reading
    satisfies `reported_is_current`; tests for the stale gate pass an
    explicit past value.
    """
    from switchyard.models import Quota
    from switchyard.usage import K_PERIOD, K_WINDOW, period_key
    reg, redis, ledger, policy, _ = build(pacing=True)
    base = reg.plans["claude-max"]
    plan = replace(base, configured_parallel=4, max_parallel_ceiling=4, quotas=(
        Quota(name="5h", role="constraint", kind="tokens",
              period="rolling_5h", allowance=None),
        Quota(name="weekly", role="target", kind="tokens",
              period="week", allowance=None),
    ))
    pct_by_label = {"5h": pct_5h, "weekly": pct_week}
    reset_by_label = {"5h": reset_at_5h, "weekly": reset_at_week}
    consumed_by_label = {"5h": consumed_5h, "weekly": consumed_week}
    reported_at = reported_at if reported_at is not None else NOW.timestamp()
    for q in plan.quotas:
        used = consumed_by_label[q.label]
        key = K_PERIOD.format(plan=plan.key, period=period_key(q.period, NOW))
        redis.hashes[key] = {"prompt_tokens": str(used), "completion_tokens": "0",
                             "cost": "0", "requests": "1"}
        pct = pct_by_label[q.label]
        if pct is None:
            continue
        facts: dict[str, str] = {
            "reported_pct_used": str(pct),
            "reported_at": str(reported_at),
        }
        ra = reset_by_label[q.label]
        if ra is not None:
            facts["reset_at"] = str(ra)
        redis.hashes[K_WINDOW.format(plan=plan.key, window=q.label)] = facts
    await redis.hset(f"sy:pace:{plan.key}", mapping={"rate": rate_per_slot})
    return plan, await policy.pace_state(plan, NOW), await policy.effective(plan, NOW)


def test_pct_only_window_paces_on_probe_estimate():
    """A fresh `reported_pct_used` drives pacing on the inverted estimate.

    Both windows have `allowance: null` and the probe only publishes a
    percentage. With `consumed_5h=400_000` and `pct_5h=20`, the 5h estimate
    lands at 2_000_000 — the same number the original 5-hour allowance was
    before the percent probe replaced it. Weekly is similarly reconstructed
    at 40_000_000 from 8_000_000 consumed at 20%. The basis must read
    'estimated' (not 'unknown' / 'observed' / 'configured'), the plan must be
    active, and `consumed_frac` must round-trip to the probe's own pct — so
    the rest of the pacer (ahead-of-pace, spent, cap) reads the percentage
    as if it were the canonical fraction-of-allowance the rest of the maths
    was built around.
    """
    async def go():
        return await _pct_plan(consumed_5h=400_000, consumed_week=8_000_000,
                               pct_5h=20.0, pct_week=20.0)
    plan, st, cap = run(go())
    target = st["target"]
    constraint = st["windows"][0]
    assert target["basis"] == "estimated", target["basis"]
    assert constraint["basis"] == "estimated", constraint["basis"]
    assert st["active"] and cap.cap >= 1, (st["reason"], cap)
    # consumed_frac is consumed / estimated_allowance = consumed / (consumed / (pct/100)) = pct/100.
    assert abs(target["consumed_frac"] - 0.20) < 1e-3, target["consumed_frac"]
    assert abs(constraint["consumed_frac"] - 0.20) < 1e-3, constraint["consumed_frac"]
    print(f"  pct=20 -> estimated allowance weekly={target['allowance']/1e6:.0f}M "
          f"(consumed_frac {target['consumed_frac']:.3f}); "
          f"basis={target['basis']}, cap={cap.cap}: {st['reason']}")


def test_pct_only_window_holds_when_pct_past_fractional_line():
    """Pct past the fractional line holds it closed, even with no stated allowance.

    pct_week=50 → consumed_frac=0.50 on an estimated 40M weekly; the line at
    ~36% elapsed (Wed-noon NOW) sits at ~0.375 * 40M = ~15M. 20M spent is
    5M ahead of that, so the pacer holds. The 5h window is left well below
    its line so it does not pretend to bind. The picker reads cap=0 from
    `effective()`, the reason is the ahead-of-pace one, and the test locks
    all three: it is the regression bar for "the estimate must feed the
    ahead-of-pace check, not just the cap maths".
    """
    async def go():
        return await _pct_plan(consumed_5h=100_000, consumed_week=20_000_000,
                               pct_5h=10.0, pct_week=50.0)
    plan, st, cap = run(go())
    assert cap.cap == 0, (cap.cap, st["reason"])
    assert "ahead of pace" in st["reason"], st["reason"]
    assert st["active"] and st["desired_slots"] == 0, st
    print(f"  pct=50 at {st['target']['elapsed_frac']*100:.0f}% elapsed "
          f"-> {cap.cap} slots: {st['reason']}")


def test_pct_only_window_at_100_routes_to_spent_branch():
    """pct=100 is in-band; the estimate collapses to consumed, remaining=0, spent.

    The branch allows pct up to and including 100 — that is what the spent
    branch is FOR. Allowance = consumed / (100/100) = consumed, so `remaining`
    drops to zero and the pacer hits `w["spent"] = True`. The plan binds as
    spent regardless of how much room the 5h window reports. A pct=100
    reading that was rejected as "out of band" would have left the plan
    permanently active at 0% burned, the wrong direction.
    """
    async def go():
        return await _pct_plan(consumed_5h=200_000, consumed_week=1_000_000,
                               pct_5h=10.0, pct_week=100.0)
    plan, st, cap = run(go())
    assert cap.cap == 0, (cap.cap, st["reason"])
    assert "spent" in st["reason"], st["reason"]
    assert st["binding"] == "weekly", st["binding"]
    print(f"  pct=100 -> spent ({st['reason']}), cap {cap.cap}")


def test_pct_only_window_idles_without_probe_facts():
    """No probe reading means no estimate: pacing stays off the learned cap.

    The pacer's no-knowledge path is unchanged. With no `reported_pct_used`
    in the facts hash, the estimate branch does not fire, allowance stays
    None, the early bail in `_window` returns `allowed_rate=None`, and
    `pace_state` falls through to "no allowance known yet". `effective()`
    leaves `cap.paced` at its dataclass default (None) — the learner is
    untouched, the picker is told the plan is not in pacing mode for this
    slot, and the existing `effective()` reason prefixes `(pacing: ...)`
    explain why on the board.
    """
    async def go():
        # No probe facts written: both pct_5h and pct_week are None.
        return await _pct_plan(consumed_5h=1_000_000, consumed_week=10_000_000,
                               pct_5h=None, pct_week=None)
    plan, st, cap = run(go())
    assert not st["active"], st
    assert st["reason"] == "no allowance known yet", st["reason"]
    assert cap.paced is None, cap.paced
    print(f"  no probe -> reason {st['reason']!r}, cap.paced={cap.paced}, "
          f"cap={cap.cap} ({cap.reason})")


def test_pct_only_window_idles_on_stale_probe():
    """Stale readings lock the freshness gate: a past `reset_at` or an old
    `reported_at` keeps the plan idle, the same trap the headroom bar hit at
    #45.

    Two flavours of staleness, both routed through `reported_is_current`:
    `reset_at` already in the past (the provider's own hint says the window
    rolled over) and `reported_at` from a previous weekly bucket (the
    probe wrote a number, but for an old window). In neither case does the
    estimate fire — the idle state is identical to "no probe facts".
    """
    async def go():
        # reset_at an hour ago: the provider says the window has rolled over.
        # The reading is no longer about the window in force.
        stale_reset = await _pct_plan(consumed_5h=400_000, consumed_week=8_000_000,
                                      pct_5h=20.0, pct_week=20.0,
                                      reset_at_5h=NOW.timestamp() - 3600,
                                      reset_at_week=NOW.timestamp() - 3600)
        # reported_at 8 days back: the probe wrote for last week's window.
        old = NOW.timestamp() - 8 * 86400
        stale_reported = await _pct_plan(consumed_5h=400_000, consumed_week=8_000_000,
                                         pct_5h=20.0, pct_week=20.0,
                                         reported_at=old)
        return stale_reset, stale_reported
    sr, srp = run(go())
    assert sr[1]["reason"] == "no allowance known yet", sr[1]["reason"]
    assert sr[2].paced is None, sr[2].paced
    assert srp[1]["reason"] == "no allowance known yet", srp[1]["reason"]
    assert srp[2].paced is None, srp[2].paced
    print(f"  past reset_at -> {sr[1]['reason']!r}; "
          f"old reported_at -> {srp[1]['reason']!r}; both cap.paced=None")


def test_pct_only_window_idles_when_consumed_is_zero():
    """A positive pct with zero ledger consumption stays idle: there is
    nothing to invert.

    Without `consumed > 0`, the estimate would need to divide by zero or
    assume an infinite allowance — both wrong directions. The branch's
    `consumed > 0` gate keeps the plan idle until either our own tally or
    a `note_exhaustion` migration gives the inversion a numerator.
    """
    async def go():
        return await _pct_plan(consumed_5h=0, consumed_week=0,
                               pct_5h=20.0, pct_week=20.0)
    plan, st, cap = run(go())
    assert not st["active"], st
    assert st["reason"] == "no allowance known yet", st["reason"]
    assert cap.paced is None, cap.paced
    print(f"  pct=20 but consumed=0 -> {st['reason']!r}, cap.paced={cap.paced}")


def test_pct_only_window_clamps_against_prior_estimate():
    """A lagged / quantized probe reading cannot inflate the estimate past
    the prior value for the same window.

    The estimate `consumed / (pct/100)` is asymmetric: usage outside
    SwitchYard shrinks the denominator-driven inflation protection (safe
    direction), but a `pct` that under-reads the truth — vendor whole-
    percent quantization, a delayed update — makes the denominator too
    small and inflates `allowance` with no upper bound. The pacer persists
    the confirmed estimate next to `reported_pct_used` and re-reads it on
    the next probe: while the window in force has not rolled over, a
    concordant pair of readings promotes a prior and subsequent readings
    clamp `allowance = min(new, prior)`. Single noisy readings sit as the
    pending candidate without hardening.

    Sequence:

      * Probe #1: consumed_week=8M, pct_week=20 → estimate 40M. No prior
        yet, so the raw reading is taken at face value and stored as the
        candidate. (A single reading is not enough to clamp — that is
        the inverse-trapdoor guarantee.)
      * Probe #2: same pct_week=20 → estimate 40M again. Concordant with
        the candidate (rel_diff = 0); promoted to prior 40M.
      * Probe #3: weekly pct drops to 10 (the lagging/quantized case);
        naive estimate 80M. The prior clamps it to 40M. Without the
        clamp, `consumed_frac` would drop to 0.10, the pace line would
        land at 36% * 80M = 28.8M, and the picker would happily admit
        the plan at 8M consumed against a (false) 80M allowance. With
        the clamp, the estimate stays at 40M and `consumed_frac = 0.20`.

    The 5h window's pct is unchanged across all three probes so its
    estimate stays unclamped and identical across calls — only the
    weekly window exercises the inflation-and-clamp branch.
    """
    async def go():
        from switchyard.models import Quota
        from switchyard.usage import K_PERIOD, K_WINDOW, period_key
        reg, redis, ledger, policy, _ = build(pacing=True)
        base = reg.plans["claude-max"]
        plan = replace(base, configured_parallel=4, max_parallel_ceiling=4,
                       quotas=(
            Quota(name="5h", role="constraint", kind="tokens",
                  period="rolling_5h", allowance=None),
            Quota(name="weekly", role="target", kind="tokens",
                  period="week", allowance=None),
        ))
        # Stable ledger across all three probes — only the weekly pct
        # changes, which is the surface area of the inflation case.
        for q, used in ((plan.quotas[0], 100_000), (plan.quotas[1], 8_000_000)):
            key = K_PERIOD.format(plan=plan.key,
                                  period=period_key(q.period, NOW))
            redis.hashes[key] = {"prompt_tokens": str(used),
                                 "completion_tokens": "0",
                                 "cost": "0", "requests": "1"}
        await redis.hset(f"sy:pace:{plan.key}", mapping={"rate": 500.0})

        def _probe_facts(pct: float) -> dict[str, str]:
            return {"reported_pct_used": str(pct),
                    "reported_at": str(NOW.timestamp())}

        # Probe #1: 20% -> 40M raw estimate. No prior yet, candidate set.
        redis.hashes[K_WINDOW.format(plan=plan.key, window="5h")] = _probe_facts(20.0)
        redis.hashes[K_WINDOW.format(plan=plan.key, window="weekly")] = _probe_facts(20.0)
        st1 = await policy.pace_state(plan, NOW)

        # Probe #2: same 20% on weekly -> concordant with candidate,
        # promoted to prior 40M. 5h unchanged.
        await redis.hset(K_WINDOW.format(plan=plan.key, window="weekly"),
                         mapping=_probe_facts(20.0))
        st2 = await policy.pace_state(plan, NOW)

        # Probe #3: weekly pct drops to 10 (the lagging/quantized case),
        # 5h pct unchanged. Naive estimate for weekly is 80M; the clamp
        # must hold it at 40M because the window is the same and the
        # prior is 40M. `hset` (not dict assignment) so the persisted
        # `estimated_allowance_*` fields from probe #2 survive — the
        # clamp is the regression under test, and overwriting the hash
        # with a bare probe dict would erase the prior and turn the
        # test into a no-op (probe #3's "fresh window" would win every
        # time).
        await redis.hset(K_WINDOW.format(plan=plan.key, window="weekly"),
                         mapping=_probe_facts(10.0))
        st3 = await policy.pace_state(plan, NOW)

        weekly1 = next(w for w in st1["windows"] if w["window"] == "weekly")
        weekly2 = next(w for w in st2["windows"] if w["window"] == "weekly")
        weekly3 = next(w for w in st3["windows"] if w["window"] == "weekly")
        constraint1 = next(w for w in st1["windows"] if w["window"] == "5h")
        constraint3 = next(w for w in st3["windows"] if w["window"] == "5h")
        return weekly1, weekly2, weekly3, constraint1, constraint3

    weekly1, weekly2, weekly3, constraint1, constraint3 = run(go())
    # Probe #1: no prior yet, raw reading at face value, basis "estimated".
    assert weekly1["basis"] == "estimated", weekly1
    assert abs(weekly1["allowance"] - 40_000_000) < 1, weekly1["allowance"]
    # Probe #2: same reading again, concordant with the candidate — promoted
    # to prior 40M. Allowance still 40M (trivial min); basis unchanged.
    assert weekly2["basis"] == "estimated", weekly2
    assert abs(weekly2["allowance"] - 40_000_000) < 1, weekly2["allowance"]
    # Probe #3: pct dropped to 10 (would naively inflate to 80M), but the
    # clamp holds the estimate at the prior 40M. The window identity is
    # unchanged so the prior stays authoritative.
    assert weekly3["basis"] == "estimated", weekly3
    assert abs(weekly3["allowance"] - 40_000_000) < 1, weekly3["allowance"]
    # consumed_frac for the weekly window is now 8M / 40M = 0.20, not
    # 8M / 80M = 0.10. That is the headline property of the clamp: the
    # pace maths read the same fraction the probe said at probe #1.
    assert abs(weekly3["consumed_frac"] - 0.20) < 1e-3, weekly3["consumed_frac"]
    # The 5h window's pct was unchanged across all three probes at 20%,
    # so its estimate is unclamped — same value across calls, basis
    # "estimated".
    assert constraint1["basis"] == constraint3["basis"] == "estimated", (
        constraint1, constraint3)
    assert abs(constraint1["allowance"] - constraint3["allowance"]) < 1, (
        constraint1, constraint3)
    print(f"  probe#1 pct=20 -> weekly allowance 40M (basis={weekly1['basis']}); "
          f"probe#2 pct=20 -> concordant, prior=40M; "
          f"probe#3 pct=10 -> CLAMPED at {weekly3['allowance']/1e6:.0f}M "
          f"(naive would be 80M); weekly consumed_frac stays "
          f"{weekly3['consumed_frac']:.2f}")


def test_pct_only_window_does_not_pin_on_single_overread():
    """A single early over-read does NOT pin the estimate.

    The judge review on the prior clamp commit called out that a naive
    `min(new, prior)` ratchet is a one-way latch: an over-read on the
    very first usable reading (vendor rounding UP, a probe racing the
    ledger so consumed momentarily lags the pct) hardens the estimate
    under the true allowance for the whole window — throughput loss in
    the inverse direction. The candidate-prior state machine requires
    two concordant readings to harden a prior; a single over-read sits
    as the candidate and gets replaced on the next reading.

    Sequence:

      * Probe #1: consumed_week=8M, pct_week=10 → estimate 80M (an
        over-read of the true ~40M allowance). No prior yet; the raw
        reading is stored as the candidate.
      * Probe #2: same 8M consumed, pct_week=20 (the vendor's second
        reading corrects). Estimate 40M. |40-80|/80 = 0.5 > tolerance,
        so this is NOT concordant with the candidate — the candidate is
        replaced with 40M and no prior is set. The over-read did not
        pin anything: the next reading can still change the world.

    Without this guard, after probe #1 the prior would be 80M (the
    first reading's value under naive `min`); probe #2 would clamp to
    min(40, 80) = 40 and that's where the estimate would stay for the
    rest of the window, throttling the plan as if the allowance were
    really 40M. With the candidate-prior design, probe #1's over-read
    is discarded and the second reading is the new candidate, awaiting
    a third concordant reading to set the prior.
    """
    async def go():
        from switchyard.models import Quota
        from switchyard.usage import K_PERIOD, K_WINDOW, period_key
        reg, redis, ledger, policy, _ = build(pacing=True)
        base = reg.plans["claude-max"]
        plan = replace(base, configured_parallel=4, max_parallel_ceiling=4,
                       quotas=(
            Quota(name="5h", role="constraint", kind="tokens",
                  period="rolling_5h", allowance=None),
            Quota(name="weekly", role="target", kind="tokens",
                  period="week", allowance=None),
        ))
        for q, used in ((plan.quotas[0], 100_000), (plan.quotas[1], 8_000_000)):
            key = K_PERIOD.format(plan=plan.key,
                                  period=period_key(q.period, NOW))
            redis.hashes[key] = {"prompt_tokens": str(used),
                                 "completion_tokens": "0",
                                 "cost": "0", "requests": "1"}
        await redis.hset(f"sy:pace:{plan.key}", mapping={"rate": 500.0})

        def _probe_facts(pct: float) -> dict[str, str]:
            return {"reported_pct_used": str(pct),
                    "reported_at": str(NOW.timestamp())}

        # Probe #1: pct_week=10 (over-read, suggests 80M allowance).
        redis.hashes[K_WINDOW.format(plan=plan.key, window="5h")] = _probe_facts(10.0)
        redis.hashes[K_WINDOW.format(plan=plan.key, window="weekly")] = _probe_facts(10.0)
        st1 = await policy.pace_state(plan, NOW)

        # Probe #2: pct_week=20 (corrects to 40M). Inconsistent with the
        # candidate 80M; the candidate is replaced with 40M and no prior
        # is set. The over-read did NOT pin.
        await redis.hset(K_WINDOW.format(plan=plan.key, window="weekly"),
                         mapping=_probe_facts(20.0))
        st2 = await policy.pace_state(plan, NOW)

        weekly1 = next(w for w in st1["windows"] if w["window"] == "weekly")
        weekly2 = next(w for w in st2["windows"] if w["window"] == "weekly")
        return weekly1, weekly2

    weekly1, weekly2 = run(go())
    # Probe #1: no prior yet, raw 80M estimate at face value, basis "estimated".
    assert weekly1["basis"] == "estimated", weekly1
    assert abs(weekly1["allowance"] - 80_000_000) < 1, weekly1["allowance"]
    # Probe #2: the over-read did not pin — the candidate was replaced with
    # 40M, no prior exists, allowance is the new raw 40M (NOT clamped against
    # the over-read). Without the candidate-prior design, this would be
    # min(40, 80) = 40 with the over-read as the prior — same number, but
    # the next reading would have to climb back above 80M to escape, locking
    # the under-throttling in place.
    assert weekly2["basis"] == "estimated", weekly2
    assert abs(weekly2["allowance"] - 40_000_000) < 1, weekly2["allowance"]
    # consumed_frac reflects the new raw reading, not a pinned under-estimate.
    assert abs(weekly2["consumed_frac"] - 0.20) < 1e-3, weekly2["consumed_frac"]
    print(f"  probe#1 pct=10 over-read -> raw allowance 80M (no prior); "
          f"probe#2 pct=20 corrects -> {weekly2['allowance']/1e6:.0f}M "
          f"uncapped, consumed_frac {weekly2['consumed_frac']:.2f}")


def test_pct_only_window_clamps_release_on_window_rollover():
    """A new `reset_at` or period bucket clears the prior clamp — the fresh
    probe is taken at face value.

    The clamp protects against on-window inflation; it must not bleed
    across windows. When `reset_at` advances (the provider says the window
    rolled over) or the period bucket changes (a calendar roll-over with
    no `reset_at` hint), the prior estimate is for a different window and
    the new reading is the only signal we have for the new one. The clamp
    must fall through and let `new_estimate` stand, so a sharply larger
    real allowance on a new window does not get pinned to the old window's
    smaller one.

    This test also exercises the candidate-prior design across the
    rollover: probe #2 hardens a prior at 40M (two concordant readings),
    then probe #3 advances `reset_at` and the new reading is taken at
    face value with no clamp and no carry-over prior.
    """
    async def go():
        from switchyard.models import Quota
        from switchyard.usage import K_PERIOD, K_WINDOW, period_key
        reg, redis, ledger, policy, _ = build(pacing=True)
        base = reg.plans["claude-max"]
        plan = replace(base, configured_parallel=4, max_parallel_ceiling=4,
                       quotas=(
            Quota(name="5h", role="constraint", kind="tokens",
                  period="rolling_5h", allowance=None),
            Quota(name="weekly", role="target", kind="tokens",
                  period="week", allowance=None),
        ))
        for q, used in ((plan.quotas[0], 100_000), (plan.quotas[1], 8_000_000)):
            key = K_PERIOD.format(plan=plan.key,
                                  period=period_key(q.period, NOW))
            redis.hashes[key] = {"prompt_tokens": str(used),
                                 "completion_tokens": "0",
                                 "cost": "0", "requests": "1"}
        await redis.hset(f"sy:pace:{plan.key}", mapping={"rate": 500.0})

        def _probe_facts(pct: float, reset_at: float | None) -> dict[str, str]:
            facts = {"reported_pct_used": str(pct),
                     "reported_at": str(NOW.timestamp())}
            if reset_at is not None:
                facts["reset_at"] = str(reset_at)
            return facts

        # Window A, probe #1: reset_at=R1, pct=20 -> estimate 40M, candidate set.
        R1 = NOW.timestamp() + 7 * 86400
        redis.hashes[K_WINDOW.format(plan=plan.key, window="5h")] = (
            _probe_facts(20.0, R1))
        redis.hashes[K_WINDOW.format(plan=plan.key, window="weekly")] = (
            _probe_facts(20.0, R1))
        await policy.pace_state(plan, NOW)

        # Window A, probe #2: same pct=20 -> concordant, prior=40M hardened.
        await redis.hset(K_WINDOW.format(plan=plan.key, window="weekly"),
                         mapping=_probe_facts(20.0, R1))
        await policy.pace_state(plan, NOW)

        # Window B, probe #3: reset_at=R2 (window rolled over), weekly pct
        # reads 10 — naive estimate 80M. The prior is for R1 so the clamp
        # must NOT fire; the candidate and prior are cleared on the
        # identity mismatch and the new reading is taken at face value.
        R2 = R1 + 7 * 86400
        await redis.hset(K_WINDOW.format(plan=plan.key, window="weekly"),
                         mapping=_probe_facts(10.0, R2))
        st = await policy.pace_state(plan, NOW)
        weekly = next(w for w in st["windows"] if w["window"] == "weekly")
        return weekly
    weekly = run(go())
    # No clamp across the rollover: the new estimate (80M) is taken at
    # face value because the window identity changed and both prior and
    # candidate were cleared on the mismatch.
    assert abs(weekly["allowance"] - 80_000_000) < 1, weekly["allowance"]
    assert weekly["basis"] == "estimated", weekly["basis"]
    # consumed_frac reflects the new reading, not a carry-over clamp.
    assert abs(weekly["consumed_frac"] - 0.10) < 1e-3, weekly["consumed_frac"]
    print(f"  rollover reset_at -> clamp released; weekly allowance = "
          f"{weekly['allowance']/1e6:.0f}M (consumed_frac "
          f"{weekly['consumed_frac']:.2f}, basis {weekly['basis']})")


def test_pct_only_window_concordance_at_low_pct():
    """At low pct the flat tolerance would flip-flop the candidate forever;
    the pct-aware band concorded adjacent whole-percent readings so the
    prior still hardens exactly where inflation bites hardest.

    Concrete shape: a vendor reporting whole-percent pct values jumps
    between adjacent readings by ~1/pct in relative terms on the
    estimate. At pct=2, a 2 -> 3 jump yields estimates 50c and 33c
    (rel_diff ≈ 0.34); at pct=3, a 3 -> 4 jump yields 33c and 25c
    (rel_diff ≈ 0.25). A flat 0.10 band never concorded either pair —
    the candidate flip-flopped on every reading and the prior never
    hardened, leaving the inflation case this mechanism exists for
    unguarded exactly at low pct where quantization is most violent.

    Sequence:

      * Probe #1: pct=2, consumed_week=2M -> estimate 100M (candidate).
      * Probe #2: pct=3, consumed_week=2M -> estimate 67M. With the flat
        band: rel_diff 0.34 > 0.10, candidate flip-flops. With the
        pct-aware band: tolerance = max(0.10, 1/2) = 0.50, so the pair
        concorded -> prior = min(100M, 67M) = 67M (the more conservative
        of the two adjacent readings).
      * Probe #3: pct=1 (lagged/quantized reading, would naively inflate
        to 200M). The prior 67M clamps it to 67M; the pace maths read
        consumed_frac = 2M / 67M = 0.030 (matches pct=3, the more
        recent reading, not pct=1).

    Without the pct-aware tolerance, probe #3's 200M would land
    unclamped (no prior hardened), consumed_frac would drop to 0.010
    against a (false) 200M allowance, and the picker would happily
    admit the plan at 2M consumed against a 200M allowance.
    """
    async def go():
        from switchyard.models import Quota
        from switchyard.usage import K_PERIOD, K_WINDOW, period_key
        reg, redis, ledger, policy, _ = build(pacing=True)
        base = reg.plans["claude-max"]
        plan = replace(base, configured_parallel=4, max_parallel_ceiling=4,
                       quotas=(
            Quota(name="5h", role="constraint", kind="tokens",
                  period="rolling_5h", allowance=None),
            Quota(name="weekly", role="target", kind="tokens",
                  period="week", allowance=None),
        ))
        # 2M consumed on a window with allowance ~67M — a low-pct
        # scenario (pct 2-3 in the truth).
        for q, used in ((plan.quotas[0], 50_000), (plan.quotas[1], 2_000_000)):
            key = K_PERIOD.format(plan=plan.key,
                                  period=period_key(q.period, NOW))
            redis.hashes[key] = {"prompt_tokens": str(used),
                                 "completion_tokens": "0",
                                 "cost": "0", "requests": "1"}
        await redis.hset(f"sy:pace:{plan.key}", mapping={"rate": 500.0})

        def _probe_facts(pct: float) -> dict[str, str]:
            return {"reported_pct_used": str(pct),
                    "reported_at": str(NOW.timestamp())}

        # Probe #1: pct=2 -> estimate 100M (candidate).
        redis.hashes[K_WINDOW.format(plan=plan.key, window="5h")] = _probe_facts(2.0)
        redis.hashes[K_WINDOW.format(plan=plan.key, window="weekly")] = _probe_facts(2.0)
        st1 = await policy.pace_state(plan, NOW)

        # Probe #2: pct=3 -> estimate 67M. Adjacent whole-percent
        # reading; with pct-aware tolerance (1/pct = 0.50 at pct=2),
        # the pair concorded and prior hardens at the conservative
        # 67M. With the old flat 0.10 band the rel_diff of 0.34 would
        # have flip-flopped the candidate and no prior would harden.
        await redis.hset(K_WINDOW.format(plan=plan.key, window="weekly"),
                         mapping=_probe_facts(3.0))
        st2 = await policy.pace_state(plan, NOW)

        # Probe #3: pct=1 (a lagged/quantized spike downward). Naive
        # estimate 200M; the prior 67M clamps it. Without the prior
        # (because the flat band failed to harden it) this would land
        # unclamped at 200M.
        await redis.hset(K_WINDOW.format(plan=plan.key, window="weekly"),
                         mapping=_probe_facts(1.0))
        st3 = await policy.pace_state(plan, NOW)

        weekly1 = next(w for w in st1["windows"] if w["window"] == "weekly")
        weekly2 = next(w for w in st2["windows"] if w["window"] == "weekly")
        weekly3 = next(w for w in st3["windows"] if w["window"] == "weekly")
        return weekly1, weekly2, weekly3

    weekly1, weekly2, weekly3 = run(go())
    # Probe #1: candidate set, raw reading at face value.
    assert weekly1["basis"] == "estimated", weekly1
    assert abs(weekly1["allowance"] - 100_000_000) < 1, weekly1["allowance"]
    # Probe #2: concordant with probe #1 under the pct-aware band
    # (rel_diff 0.34 ≤ max(0.10, 1/2)=0.50). Prior hardened at
    # min(100M, 67M) = 67M.
    assert weekly2["basis"] == "estimated", weekly2
    assert abs(weekly2["allowance"] - 67_000_000) < 1.5e6, weekly2["allowance"]
    # consumed_frac = 2M / 67M ≈ 0.030 (matches pct=3, not pct=2).
    assert abs(weekly2["consumed_frac"] - 2.0 / 67.0) < 1e-3, weekly2["consumed_frac"]
    # Probe #3: pct=1 -> naive 200M, clamped by prior 67M. The prior
    # held the estimate at 67M instead of letting it inflate to 200M.
    assert weekly3["basis"] == "estimated", weekly3
    assert abs(weekly3["allowance"] - 67_000_000) < 1.5e6, weekly3["allowance"]
    # consumed_frac still reflects the clamped value, not the inflated reading.
    assert abs(weekly3["consumed_frac"] - 2.0 / 67.0) < 1e-3, weekly3["consumed_frac"]
    print(f"  probe#1 pct=2 -> candidate 100M; "
          f"probe#2 pct=3 (rel_diff {abs(67-100)/100:.2f}) -> concorded "
          f"under pct-aware band, prior={weekly2['allowance']/1e6:.0f}M; "
          f"probe#3 pct=1 -> CLAMPED at {weekly3['allowance']/1e6:.0f}M "
          f"(naive 200M); consumed_frac stays "
          f"{weekly3['consumed_frac']:.3f}")


def test_pct_only_window_rejects_multi_step_pair_at_low_pct():
    """Two-step pct jumps at very low pct must NOT concord — a pair of
    inflated readings would otherwise harden an inflated prior.

    The prior cycle-3 commit widened the concordance band to 1/pct so
    adjacent whole-percent readings concorded at low pct (where their
    natural rel_diff approaches 0.5). At pct=1 the uncapped band was 1.0,
    which concorded essentially any pair of readings and dissolved the
    'two independent readings must agree' property entirely. The judge
    review on that commit called out the concrete failure: true pct 3
    reading as 1 then 2 (estimates 200M and 100M, rel_diff 0.5) would
    concord under the uncapped band, prior would harden at min(200, 100)
    = 100M, and the clamp would hold at 1.5x truth for the rest of the
    window. `min(candidate, new)` is *within-pair* protection (smaller of
    the two readings), not absolute protection against a pair of
    inflated readings both over-reading the truth.

    Fix: cap the band at CONCORDANCE_BAND_CAP = 0.40 so multi-step
    jumps do not concord while adjacent whole-percent readings still do.
    The bad scenario rel_diff 0.5 > 0.40 (rejected). The pct=2 -> 3
    adjacent rel_diff 0.33 ≤ 0.40 (still concorded, so the prior can
    harden at the conservative end of a true-bracketing pair). Verified
    at every pct: tolerance is in [0.10, 0.40] and admits adjacent
    whole-percent rel_diffs up to 0.333 (pct=2) while rejecting any
    pair with rel_diff ≥ 0.40.

    Sequence:

      * Probe #1: pct=1, consumed=2M -> estimate 200M (candidate).
      * Probe #2: pct=2, consumed=2M -> estimate 100M. rel_diff 0.5,
        tolerance at pct=2 = max(0.10, min(0.40, 0.50)) = 0.40. 0.5 > 0.40
        -> NOT concorded. Candidate replaced with 100M. No prior.
      * Probe #3: pct=3, consumed=2M -> estimate 67M. Now compare against
        the candidate 100M: rel_diff 0.33, tolerance at pct=3 = 0.333.
        Concordant (just barely) -> prior = min(100, 67) = 67M.

    Without the cap, probe #2 would have concorded and prior would have
    hardened at 100M. With the cap, the bad pair is rejected and the
    prior tightens to the truth (67M) as soon as a concordant reading
    appears.
    """
    async def go():
        from switchyard.models import Quota
        from switchyard.usage import K_PERIOD, K_WINDOW, period_key
        reg, redis, ledger, policy, _ = build(pacing=True)
        base = reg.plans["claude-max"]
        plan = replace(base, configured_parallel=4, max_parallel_ceiling=4,
                       quotas=(
            Quota(name="5h", role="constraint", kind="tokens",
                  period="rolling_5h", allowance=None),
            Quota(name="weekly", role="target", kind="tokens",
                  period="week", allowance=None),
        ))
        for q, used in ((plan.quotas[0], 50_000), (plan.quotas[1], 2_000_000)):
            key = K_PERIOD.format(plan=plan.key,
                                  period=period_key(q.period, NOW))
            redis.hashes[key] = {"prompt_tokens": str(used),
                                 "completion_tokens": "0",
                                 "cost": "0", "requests": "1"}
        await redis.hset(f"sy:pace:{plan.key}", mapping={"rate": 500.0})

        def _probe_facts(pct: float) -> dict[str, str]:
            return {"reported_pct_used": str(pct),
                    "reported_at": str(NOW.timestamp())}

        # Probe #1: pct=1 (the under-read) -> estimate 200M, candidate set.
        redis.hashes[K_WINDOW.format(plan=plan.key, window="5h")] = _probe_facts(1.0)
        redis.hashes[K_WINDOW.format(plan=plan.key, window="weekly")] = _probe_facts(1.0)
        st1 = await policy.pace_state(plan, NOW)

        # Probe #2: pct=2 -> estimate 100M. rel_diff 0.5 vs candidate 200M.
        # Under the cap, this does NOT concord: tolerance at pct=2 is
        # max(0.10, min(0.40, 1/2)) = 0.40, and 0.5 > 0.40. The candidate
        # is replaced with 100M and no prior hardens. Under the uncapped
        # band (1/pct = 0.5 at pct=2), 0.5 ≤ 0.5 would have concorded and
        # prior = min(200, 100) = 100M would have hardened — the failure
        # mode this test locks.
        await redis.hset(K_WINDOW.format(plan=plan.key, window="weekly"),
                         mapping=_probe_facts(2.0))
        st2 = await policy.pace_state(plan, NOW)

        # Probe #3: pct=3 -> estimate 67M. rel_diff 0.33 vs candidate 100M,
        # tolerance at pct=3 = max(0.10, min(0.40, 1/3)) = 0.333. Concordant
        # (just barely) -> prior = min(100, 67) = 67M. The pair that brackets
        # truth concorded; the prior tightens to the truth.
        await redis.hset(K_WINDOW.format(plan=plan.key, window="weekly"),
                         mapping=_probe_facts(3.0))
        st3 = await policy.pace_state(plan, NOW)

        weekly1 = next(w for w in st1["windows"] if w["window"] == "weekly")
        weekly2 = next(w for w in st2["windows"] if w["window"] == "weekly")
        weekly3 = next(w for w in st3["windows"] if w["window"] == "weekly")
        return weekly1, weekly2, weekly3

    weekly1, weekly2, weekly3 = run(go())
    # Probe #1: raw 200M at face value, no prior yet.
    assert weekly1["basis"] == "estimated", weekly1
    assert abs(weekly1["allowance"] - 200_000_000) < 1, weekly1["allowance"]
    # Probe #2: the multi-step jump (1 -> 2) must NOT concord under the
    # cap. allowance stays at the raw reading (100M); no prior hardened.
    assert weekly2["basis"] == "estimated", weekly2
    assert abs(weekly2["allowance"] - 100_000_000) < 1, weekly2["allowance"]
    # consumed_frac reflects the raw reading — no clamp, no prior.
    assert abs(weekly2["consumed_frac"] - 0.02) < 1e-3, weekly2["consumed_frac"]
    # Probe #3: the truth-bracketing pair (2 -> 3, rel_diff 0.33 ≤
    # tolerance 0.333) concorded, prior = min(100, 67) = 67M. Without the
    # cap, probe #2 would have already hardened prior=100M; with the cap,
    # the prior tightens to the truth on the third reading.
    assert weekly3["basis"] == "estimated", weekly3
    assert abs(weekly3["allowance"] - 67_000_000) < 1.5e6, weekly3["allowance"]
    assert abs(weekly3["consumed_frac"] - 2.0 / 67.0) < 1e-3, weekly3["consumed_frac"]
    print(f"  probe#1 pct=1 -> candidate 200M; "
          f"probe#2 pct=2 (rel_diff {abs(100-200)/200:.2f}) -> REJECTED under cap; "
          f"probe#3 pct=3 -> concorded, prior tightens to "
          f"{weekly3['allowance']/1e6:.0f}M (truth)")


def test_pct_only_window_concordance_is_ulp_deterministic():
    """At adjacent whole-percent transitions the comparison was decided by
    one floating-point ULP — locking the new epsilon with a magnitude
    sweep at every transition.

    The judge review on the cap commit measured the precision flaw
    empirically: adjacent pairs have rel_diff mathematically identical
    to the tolerance (both are 1/pct_new), so the `<=` comparison was
    decided by 1 ULP. 3→4 failed 36/54 random-consumed trials, 6→7
    54/54, 9→10 37/54 — the candidate flip-flopped at those transitions
    and the prior never hardened, silently reintroducing the
    flip-flop regression the pct-aware band was introduced to fix.

    This test reproduces the exact arithmetic (`consumed/(pct/100)`,
    then `abs(new - cand)/max(new, cand)`) the branch uses, sweeps 54
    consumed magnitudes per adjacent pair in [pct-1 -> pct], and
    asserts every adjacent transition at pct >= 2 concorded
    deterministically. The 1→2 transition is also tested and asserted
    *rejected* (it's the one-step jump the cap exists to reject — the
    precision epsilon does not change that, because the rel_diff gap
    is 0.10, ~1e8x the epsilon).

    The sweep deliberately covers the failing-consumed cases the
    reviewer measured: small (1.0), 1e3, 1e6, 1e9, plus 50 random
    values uniformly distributed across [1.0, 1e9] (matching the
    reviewer's sweep size). Under the precision epsilon every
    adjacent pair concords; without it (verified by hand-computing the
    raw rel_diff), the failing transitions fail this test.
    """
    import random as _random
    from switchyard.policy import (
        CONCORDANCE_BAND_CAP, CONCORDANCE_EPS, ESTIMATE_TOLERANCE,
        _concordance_tolerance,
    )

    # The arithmetic under test: candidate from the prior reading,
    # new from the current reading, both with `consumed/(pct/100)`
    # exactly as `_window` computes it. Adjacent pairs (pct-1 -> pct)
    # have rel_diff mathematically identical to the tolerance (both
    # 1/pct) — the epsilon is what makes the `<=` deterministic.
    _random.seed(0)
    # Magnitudes that historically trip ULP rounding: 1.0, 1e3, 1e6,
    # 1e9, plus 50 random uniform in [1.0, 1e9] (matches the reviewer's
    # sweep size of 54 per transition).
    magnitudes = [1.0, 1e3, 1e6, 1e9] + [
        _random.uniform(1.0, 1e9) for _ in range(50)
    ]
    # Adjacent transitions at pct >= 3 must concord deterministically
    # (the precision-flavor sweep). At pct=2 the cap binds (1/2 = 0.50
    # > CONCORDANCE_BAND_CAP = 0.40), so 1->2 is rejected under the cap
    # — tested separately below.
    for pct_new in range(3, 11):  # pct=3..10 inclusive
        pct_old = pct_new - 1
        tol = _concordance_tolerance(pct_new)
        # Sanity: at pct >= 3 the cap doesn't bind, so tolerance must
        # equal 1/pct_new exactly — the boundary the reviewer flagged.
        expected_tol = 1.0 / pct_new
        assert abs(tol - expected_tol) < 1e-15, (pct_new, tol, expected_tol)
        for consumed in magnitudes:
            cand = consumed / (pct_old / 100.0)
            new = consumed / (pct_new / 100.0)
            rel_diff = abs(new - cand) / max(new, cand)
            concorded = rel_diff <= tol * (1.0 + CONCORDANCE_EPS)
            assert concorded, (
                f"adjacent {pct_old}->{pct_new} at consumed={consumed:g}: "
                f"rel_diff={rel_diff!r} tol={tol!r} (gap {rel_diff - tol:.2e}); "
                f"concordance should be deterministic but rejected")

    # The 1->2 pair is the cap-rejection case the cap exists to defend:
    # rel_diff 0.5 vs capped tolerance 0.40, gap 0.10 (about 1e8x the
    # epsilon). Every magnitude must reject.
    for consumed in magnitudes:
        cand = consumed / 0.01  # pct=1
        new = consumed / 0.02  # pct=2
        rel_diff = abs(new - cand) / max(new, cand)
        tol_at_2 = _concordance_tolerance(2.0)
        rejected = rel_diff > tol_at_2 * (1.0 + CONCORDANCE_EPS)
        assert rejected, (
            f"1->2 at consumed={consumed:g}: rel_diff={rel_diff!r} "
            f"tol={tol_at_2!r}; should be REJECTED under cap but concorded")

    # Constants sanity: epsilon is far below the cap-rel_diff gap and
    # far above ULP noise.
    assert CONCORDANCE_EPS > 1e-12, CONCORDANCE_EPS
    assert CONCORDANCE_EPS < 1e-3, CONCORDANCE_EPS
    assert ESTIMATE_TOLERANCE == 0.10
    assert CONCORDANCE_BAND_CAP == 0.40
    print(f"  swept {len(magnitudes)} magnitudes per adjacent transition "
          f"pct=2..10 + the cap-rejected 1->2 pair; "
          f"every adjacency concorded, 1->2 rejected, "
          f"CONCORDANCE_EPS={CONCORDANCE_EPS:g}")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
