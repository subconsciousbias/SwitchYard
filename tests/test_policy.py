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
os.environ.setdefault("SWITCHYARD_PLANS", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "plans.yaml"))

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
    reg = models.load()
    for key in ("glm", "opencode-go"):
        assert reg.plans[key].configured_parallel is None
        assert reg.plans[key].max_parallel == reg.settings.concurrency_learning.seed_cap
    print(f"  `max_parallel: auto` plans start at {reg.settings.concurrency_learning.seed_cap}")


# ----------------------------------------------------------------- pacing ----
async def _pace(plan_key: str, consumed_frac: float, allowance: float = 100_000_000,
                rate_per_slot: float = 5000.0, pacing: bool = True):
    """Put a plan at a given consumption level and ask what the pacer wants."""
    reg, redis, ledger, policy, _ = build(pacing=pacing)
    plan = replace(reg.plans[plan_key],
                   quota=replace(reg.plans[plan_key].quota, allowance=allowance))
    from switchyard.usage import K_PERIOD, period_key
    key = K_PERIOD.format(plan=plan.key, period=period_key(plan.quota.period))
    redis.hashes[key] = {"prompt_tokens": str(allowance * consumed_frac),
                         "completion_tokens": "0", "cost": "0", "requests": "1"}
    await redis.hset(f"sy:pace:{plan.key}", mapping={"rate": rate_per_slot})
    return plan, await policy.pacer.state(plan), await policy.effective(plan)


def test_ahead_of_budget_holds_the_plan_closed():
    """90% spent with a third of the window left -> stop, let the line catch up.

    Closing is the only way to slow below one continuously busy slot, which at
    real LLM throughput is already far too fast for a monthly allowance.
    """
    async def go():
        plan, st, cap = await _pace("minimax-ultra", consumed_frac=0.9)
        return plan, st, cap
    plan, st, cap = run(go())
    assert st["active"] and cap.cap == 0, (st["reason"], cap.cap)
    assert st["ahead_by"] > 0
    print(f"  90% spent at {st['elapsed_frac']*100:.0f}% elapsed "
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
    """Just behind the line -> open, but only as wide as the burn rate allows."""
    async def go():
        # elapsed is ~66% of September by the 20th; sit just under the line.
        return await _pace("minimax-ultra", consumed_frac=0.60, rate_per_slot=200.0)
    plan, st, cap = run(go())
    assert cap.cap >= 1 and st["reason"] == "pacing to deadline", (st, cap)
    print(f"  60% spent vs {st['pace_line']/1e6:.0f}M line -> {cap.cap} slots "
          f"(target {st['target_rate']:.0f} units/s at {st['rate_per_slot']:.0f}/slot)")


def test_spent_allowance_closes_the_plan():
    async def go():
        _, st, cap = await _pace("minimax-ultra", consumed_frac=1.0)
        return st, cap
    st, cap = run(go())
    assert st["reason"] == "allowance spent" and cap.cap == 0, (st, cap)
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
        for key in ("openrouter-mimo", "anthropic-fable", "qwen-local", "gemma-local"):
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
        return ([p.key for p in await picker_off._members("forge")],
                [p.key for p in await picker_on._members("forge")])
    off, on = run(go())
    assert "qwen-local" in off and "qwen-local" not in on
    print(f"  tail present when pacing off ({off[-1]}), absent when on")


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} policy tests passed")
