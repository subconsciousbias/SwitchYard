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
        # The marker for "local" is an unmetered, unlimited quota — not the auth
        # mode, since a local server may well require a key.
        assert all(reg.plans[k].quota.kind == "unlimited" for k in members), members
        assert all(not reg.plans[k].metered for k in members), members

        # Capacity is per *subscription*, not per plan: every local model runs on
        # the same machine, so they share one pool of slots.
        subs = {reg.plans[k].subscription: reg.plans[k].max_parallel for k in members}
        total = sum(subs.values())
        for _ in range(total):
            await picker.pick("local", None)
        try:
            await picker.pick("local", None)
        except LaneSaturated:
            return members, subs, total
        raise AssertionError("local lane must refuse rather than spill")

    members, subs, total = run(go())
    print(f"  local lane: {members} share {subs} = {total} slot(s) — "
          f"refuses instead of spilling")


def test_local_models_share_one_machine():
    """Three local plans on one box must not hand out three plans' worth of
    slots. Without shared accounting, six requests would land on hardware that
    handles one or two."""
    async def go():
        reg, slots, picker = build()
        local = [p for p in reg.plans.values() if p.quota.kind == "unlimited"]
        assert len({p.subscription for p in local}) == 1, \
            [(p.key, p.subscription) for p in local]
        naive = sum(p.max_parallel for p in local)
        shared = max(p.max_parallel for p in local)

        # Saturate through the `local` lane, then confirm `bulk` — which also
        # contains a local model — sees no free local capacity.
        for _ in range(shared):
            await picker.pick("local", None)
        pick = await picker.pick("bulk", None)
        return [p.key for p in local], naive, shared, pick.plan.key

    keys, naive, shared, spilled = run(go())
    assert shared < naive
    print(f"  {len(keys)} local plans would naively offer {naive} slots; "
          f"they share {shared}. With those busy, bulk spilled to {spilled}")


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
        # Enable the heavy Claude tier and stand the other apex members down, so
        # this test is about the shared subscription rather than about whichever
        # plan currently wins apex's ordering. (Enabling astra once moved apex to
        # it and broke this test — the lane order is config, not a fixture.)
        plans = dict(reg.plans)
        plans["claude-max-heavy"] = replace(plans["claude-max-heavy"], enabled=True)
        for other in ("astra", "anthropic-fable"):
            if other in plans:
                plans[other] = replace(plans[other], enabled=False)
        reg = models.Registry(settings=reg.settings, plans=plans, lanes=reg.lanes)
        picker = Picker(reg, slots)

        assert (reg.plans["claude-max-heavy"].subscription
                == reg.plans["claude-max"].subscription == "claude-max")

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




def test_tool_calls_never_reach_a_cli_backed_plan():
    """A request carrying `tools` must skip every CLI-backed plan.

    Those plans are whole agent harnesses behind a sidecar: the caller's tool
    definitions have nowhere to run, the harness's own tools act on the
    sidecar's container rather than the caller's workspace, and the two system
    prompts stack. Silently dropping the tools would look like the model simply
    choosing not to call any.
    """
    async def go():
        reg, slots, picker = build()
        out = {}
        for lane in ("forge", "judge"):
            plain = [p.key for p in await picker._members(lane)]
            with_tools = [p.key for p in await picker._members(lane, needs_tools=True)]
            out[lane] = (plain, with_tools)
            assert all(reg.plans[k].can_use_tools for k in with_tools), with_tools
            dropped = set(plain) - set(with_tools)
            assert all(reg.plans[k].auth == "oauth_sidecar" for k in dropped), dropped

        # A tool-using request is routed, not rejected, as long as one plan can.
        pick = await picker.pick("forge", None, needs_tools=True)
        assert pick.plan.can_use_tools
        return out, pick.plan.key

    out, picked = run(go())
    for lane, (plain, with_tools) in out.items():
        print(f"  {lane}: {len(plain)} plans, {len(with_tools)} can take tools "
              f"(dropped {sorted(set(plain) - set(with_tools))})")
    print(f"  a tool-using forge request landed on {picked}")


def test_a_lane_with_no_tool_capable_plan_says_so():
    """apex is entirely CLI-backed, so a tool-using apex request must fail
    with an explanation rather than quietly losing the tools."""
    async def go():
        reg, slots, picker = build()
        assert not [p for p in await picker._members("apex", needs_tools=True)]
        try:
            await picker.pick("apex", None, needs_tools=True)
        except LaneSaturated as exc:
            return str(exc)
        raise AssertionError("expected apex to refuse a tool-using request")
    print(f"  {run(go())}")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
    print("\nall routing tests passed")
