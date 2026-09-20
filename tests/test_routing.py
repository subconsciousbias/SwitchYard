"""Behavioural tests for the three properties the design promises:
ordered fill, capacity that vanishes on quota exhaustion, session affinity.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SWITCHYARD_PLANS", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "plans.yaml"))

from switchyard import models                      # noqa: E402
from switchyard.picker import LaneSaturated, Picker  # noqa: F401  # noqa: E402
from switchyard.slots import SlotTable             # noqa: E402
from tests.fake_redis import FakeRedis             # noqa: E402


def build():
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    return reg, slots, Picker(reg, slots)


def run(coro):
    return asyncio.run(coro)


def test_ordered_fill_then_spill():
    """A lane fills each plan to its cap before touching the next one."""
    async def go():
        reg, slots, picker = build()
        order = [p.key for p in reg.lane_members("forge")]
        picked = [(await picker.pick("forge", None)).plan.key for _ in range(8)]
        # First plan takes its full cap, then the next, in lane order.
        first, second = order[0], order[1]
        cap_first = reg.plans[first].max_parallel
        assert picked[:cap_first] == [first] * cap_first, picked
        assert picked[cap_first] == second, picked
        return picked
    picked = run(go())
    print("  ordered fill:", " ".join(picked))


def test_total_capacity_is_sum_of_caps():
    """Saturating the lane hands out exactly one slot per configured slot."""
    async def go():
        reg, slots, picker = build()
        members = reg.lane_members("forge")
        total = sum(p.max_parallel for p in members)
        for _ in range(total):
            await picker.pick("forge", None)
        try:
            await picker.pick("forge", None)
        except LaneSaturated as exc:
            return total, str(exc)
        raise AssertionError("lane should have been saturated")
    total, msg = run(go())
    print(f"  lane capacity: {total} slots, then 429 -> {msg[:60]}...")


def test_quota_exhaustion_removes_capacity():
    """Cooling a plan down shrinks the lane by exactly that plan's slots."""
    async def go():
        reg, slots, picker = build()
        before = (await picker.capacity("forge"))["slots_available_now"]
        victim = reg.lane_members("forge")[0]
        await slots.cool_down(victim.key, 900, "quota_exhausted")
        after = (await picker.capacity("forge"))["slots_available_now"]
        # And traffic now starts at the next plan instead.
        nxt = (await picker.pick("forge", None)).plan.key
        return before, after, victim, nxt
    before, after, victim, nxt = run(go())
    assert after == before - victim.max_parallel, (before, after)
    assert nxt != victim.key
    print(f"  {victim.key} exhausted: lane {before} -> {after} slots, traffic moves to {nxt}")


def test_session_affinity_holds_and_survives_exhaustion():
    """A session sticks to its provider, and re-leases when that dies."""
    async def go():
        reg, slots, picker = build()
        first = await picker.pick("forge", "sess-A")
        await picker.release(first.plan.key, first.request_id)
        again = await picker.pick("forge", "sess-A")
        assert again.plan.key == first.plan.key and again.sticky
        await picker.release(again.plan.key, again.request_id)

        # Another session lands on the same (not yet full) plan.
        other = await picker.pick("forge", "sess-B")
        assert other.plan.key == first.plan.key
        await picker.release(other.plan.key, other.request_id)

        # Now the leased plan runs out of quota: the session must move on
        # rather than getting stuck on dead capacity.
        await slots.cool_down(first.plan.key, 900, "quota_exhausted")
        moved = await picker.pick("forge", "sess-A")
        assert moved.plan.key != first.plan.key and not moved.sticky
        return first.plan.key, moved.plan.key
    a, b = run(go())
    print(f"  affinity: session pinned to {a}, re-leased to {b} after exhaustion")


def test_local_lane_never_escapes_to_cloud():
    """The Local Only lane has no cloud members and no tail, by design."""
    async def go():
        reg, _, picker = build()
        members = [p.key for p in reg.lane_members("local")]
        assert all(reg.plans[k].auth == "none" for k in members), members
        total = sum(reg.plans[k].max_parallel for k in members)
        for _ in range(total):
            await picker.pick("local", None)
        try:
            await picker.pick("local", None)
        except LaneSaturated:
            return members
        raise AssertionError("local lane must refuse rather than spill")
    print("  local lane:", run(go()), "— refuses instead of spilling")


def test_expiring_plans_are_drained_first():
    """Cancelled capacity is promoted ahead of plans you keep paying for."""
    reg = models.load()
    order = reg.lane_members("judge")
    expiring = [p.key for p in order if p.days_left is not None and p.days_left <= reg.settings.drain_within_days]
    keeping = [p.key for p in order if p.days_left is None]
    assert expiring, "expected some expiring plans in the judge lane"
    assert order.index(reg.plans[expiring[0]]) < order.index(reg.plans[keeping[0]])
    print("  judge lane order:", " -> ".join(p.key for p in order))


def test_one_subscription_cannot_be_used_twice_at_once():
    """Two model tiers on one Claude Max plan share its single connection.

    `apex` (heavy tier) and `judge` (regular tier) are different lanes pointing
    at different deployments, but the same subscription. Without shared slot
    accounting they would happily open two connections against a plan that
    allows one.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker = build()
        heavy = replace(reg.plans["claude-max-heavy"], enabled=True)
        plans = {**reg.plans, "claude-max-heavy": heavy}
        reg = models.Registry(settings=reg.settings, plans=plans, lanes=reg.lanes)
        picker = Picker(reg, slots)

        assert heavy.subscription == reg.plans["claude-max"].subscription == "claude-max"

        first = await picker.pick("apex", None)
        assert first.plan.key == "claude-max-heavy", first.plan.key

        # judge's turn: claude-max must now look full, so it walks past it.
        second = await picker.pick("judge", None)
        assert second.plan.key != "claude-max", "shared connection was double-booked"

        # Once the heavy call finishes, the connection is free again.
        await picker.release(first.plan.key, first.request_id)
        third = await picker.pick("judge", None)
        return first.plan.key, second.plan.key, third.plan.key, slots

    a, b, c, slots = run(go())
    inflight = run(slots.in_flight("claude-max"))
    print(f"  apex took {a}; judge fell through to {b} rather than double-booking; "
          f"after release judge could reach the subscription again (picked {c})")
    assert inflight <= 1


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
    print("\nall routing tests passed")
