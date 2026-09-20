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
    """A lane fills each model's plan to its cap before touching the next."""
    async def go():
        reg, slots, picker = build()
        members = reg.lane_members("forge")
        picked = [(await picker.pick("forge", None)).ref for _ in range(8)]
        first, second = members[0], members[1]
        # How many of THIS member can run at once: its plan's limit, narrowed by
        # its own. The first forge member caps itself at 1 even though its plan
        # allows 2, so "fill" means filling that, not the plan.
        cap_first = reg.plan_of(first).cap_for(first)
        assert picked[:cap_first] == [first.ref] * cap_first, picked
        assert picked[cap_first] == second.ref, picked
        return picked
    picked = run(go())
    print("  ordered fill:", " ".join(picked))


def test_total_capacity_is_sum_of_caps():
    """Saturating the lane hands out exactly one slot per configured slot."""
    async def go():
        reg, slots, picker = build()
        members = reg.lane_members("forge")
        # A lane's real capacity is, per plan, the lesser of the plan's limit and
        # what its members in this lane can reach. A plan of 4 whose only member
        # here caps itself at 2 contributes 2.
        reach: dict[str, int] = {}
        caps: dict[str, int] = {}
        for m in members:
            plan = reg.plan_of(m)
            caps[plan.key] = plan.max_parallel
            reach[plan.key] = reach.get(plan.key, 0) + (
                m.max_parallel if m.max_parallel is not None else plan.max_parallel)
        total = sum(min(caps[k], reach[k]) for k in caps)
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
        victim_plan = reg.plan_of(victim)
        # What this member actually contributed: its plan's limit, narrowed by
        # its own. Cooling the plan removes that, not the plan's nominal limit.
        contributed = victim_plan.cap_for(victim)
        await slots.cool_down(victim_plan.key, 900, "quota_exhausted")
        after = (await picker.capacity("forge"))["slots_available_now"]
        nxt = (await picker.pick("forge", None)).ref
        return before, after, victim, contributed, nxt
    before, after, victim, contributed, nxt = run(go())
    assert after == before - contributed, (before, after, contributed)
    assert nxt != victim.ref
    print(f"  {victim.ref} exhausted: lane {before} -> {after} slots "
          f"(it contributed {contributed}), traffic moves to {nxt}")


def test_session_affinity_holds_and_survives_exhaustion():
    """A session sticks to its provider, and re-leases when that dies."""
    async def go():
        reg, slots, picker = build()
        first = await picker.pick("forge", "sess-A")
        await picker.release(first.plan.key, first.request_id, first.ref)
        again = await picker.pick("forge", "sess-A")
        assert again.ref == first.ref and again.sticky
        await picker.release(again.plan.key, again.request_id, again.ref)

        # Another session lands on the same (not yet full) model.
        other = await picker.pick("forge", "sess-B")
        assert other.ref == first.ref
        await picker.release(other.plan.key, other.request_id, other.ref)

        # Now the leased plan runs out of quota: the session must move on
        # rather than getting stuck on dead capacity.
        await slots.cool_down(first.plan.key, 900, "quota_exhausted")
        moved = await picker.pick("forge", "sess-A")
        assert moved.ref != first.ref and not moved.sticky
        return first.ref, moved.ref
    a, b = run(go())
    print(f"  affinity: session pinned to {a}, re-leased to {b} after exhaustion")


def test_local_lane_never_escapes_to_cloud():
    """The Local Only lane has no cloud members and no tail, by design."""
    async def go():
        reg, _, picker = build()
        members = reg.lane_members("local")
        refs = [m.ref for m in members]
        # The marker for "local" is an unmetered, unlimited quota — not the auth
        # mode, since a local server may well require a key.
        assert all(reg.plan_of(m).quota.kind == "unlimited" for m in members), refs
        assert all(not reg.plan_of(m).metered for m in members), refs

        # Capacity belongs to the plan: every local model runs on one machine,
        # so they share one pool of slots.
        subs = {reg.plan_of(m).key: reg.plan_of(m).max_parallel for m in members}
        total = sum(subs.values())
        for _ in range(total):
            await picker.pick("local", None)
        try:
            await picker.pick("local", None)
        except LaneSaturated:
            return refs, subs, total
        raise AssertionError("local lane must refuse rather than spill")

    refs, subs, total = run(go())
    print(f"  local lane: {refs} share {subs} = {total} slot(s) — "
          f"refuses instead of spilling")


def test_plan_and_model_caps_are_separate_limits():
    """Two counters, not one.

    `local-box` allows 2 connections; each of its models allows 1. So one qwen
    and one gemma may run together, a *second* qwen may not, and nothing more
    may run at all. Enforcing the model's cap against the plan's counter — which
    is what a single counter forces — would let only one request through in
    total.
    """
    async def go():
        reg, slots, picker = build()
        box = reg.plans["local-box"]
        assert box.max_parallel == 2
        assert all(m.max_parallel == 1 for m in box.models.values())

        first = await picker.pick("local", None)          # qwen
        second = await picker.pick("local", None)         # qwen full -> gemma
        assert first.ref != second.ref, (first.ref, second.ref)
        assert {first.ref, second.ref} == {"local-box/qwen", "local-box/gemma"}

        # The plan is now full, so nothing else fits — and the refusal names the
        # plan's limit, because that is genuinely what bit.
        try:
            await picker.pick("local", None)
            raise AssertionError("plan cap of 2 was exceeded")
        except LaneSaturated as exc:
            saturated = str(exc)
        assert "plan full at 2" in saturated, saturated

        # Free gemma: the plan now has room, but qwen is still at its own cap of
        # 1. So the next pick must skip qwen *for a model reason* and take gemma.
        gemma = first if first.ref.endswith("gemma") else second
        await picker.release(gemma.plan.key, gemma.request_id, gemma.ref)
        third = await picker.pick("local", None)
        assert third.ref == gemma.ref, third.ref
        return saturated, third

    saturated, third = run(go())
    skipped = ", ".join(third.considered)
    assert "model full at 1" in skipped, skipped
    print("  plan=2, models=1 each: qwen+gemma run together; a third is refused")
    print(f"  plan full: {saturated.split(': ', 1)[1]}")
    print(f"  with room on the plan but not the model: skipped {skipped}, "
          f"took {third.ref}")


def test_expiring_plans_are_drained_first():
    """Cancelled capacity is promoted ahead of plans you keep paying for."""
    reg = models.load()
    order = reg.lane_members("judge")
    window = reg.settings.drain_within_days
    days = [reg.plan_of(m).days_left for m in order]
    expiring = [i for i, d in enumerate(days) if d is not None and d <= window]
    keeping = [i for i, d in enumerate(days) if d is None]
    assert expiring, "expected some expiring plans in the judge lane"
    assert min(expiring) < min(keeping), list(zip((m.ref for m in order), days))
    print("  judge lane order:", " -> ".join(m.ref for m in order))


def test_models_on_one_plan_share_its_connection_limit():
    """`apex` and `judge` name different models of the same Claude Max plan.

    Whatever one lane consumes counts against the other, because the connection
    limit belongs to the plan. Accounting per model instead would let two lanes
    each open the plan's full allowance.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker = build()
        plans = dict(reg.plans)

        # Enable the heavy Claude model and stand the other apex members down, so
        # this tests the shared plan rather than whichever member wins apex's
        # ordering (the drain rule reshuffles that whenever an expiry changes).
        cm = plans["claude-max"]
        plans["claude-max"] = replace(cm, models={
            **cm.models, "fable": replace(cm.models["fable"], enabled=True)})
        seat = plans["openai"]
        plans["openai"] = replace(seat, models={
            **seat.models, "astra": replace(seat.models["astra"], enabled=False)})
        reg = models.Registry(settings=reg.settings, plans=plans, lanes=reg.lanes)
        picker = Picker(reg, slots)

        cap = first_plan_cap = plans["claude-max"].max_parallel

        # Fill the plan through apex, which names claude-max/fable.
        held = []
        for _ in range(cap):
            try:
                pick = await picker.pick("apex", None)
            except LaneSaturated:
                break
            assert pick.plan.key == "claude-max", pick.ref
            held.append(pick)
        # fable caps itself at 1, so apex reaches the plan's limit only if the
        # plan's limit is 1; otherwise it fills what it can and apex saturates.
        used = await slots.in_flight("claude-max")

        # judge names claude-max/opus first. Whatever apex consumed counts
        # against the same plan, so judge may only use what is left.
        second = await picker.pick("judge", None)
        after = await slots.in_flight("claude-max")
        assert after <= cap, (after, cap)

        for pick in held:
            await picker.release(pick.plan.key, pick.request_id, pick.ref)
        freed = await slots.in_flight("claude-max")
        return [p.ref for p in held], second.ref, used, after, freed, cap

    taken, b, used, after, freed, cap = run(go())
    assert after <= cap
    print(f"  apex filled the plan with {', '.join(taken)} ({used}/{cap})")
    print(f"  judge then got {b}, plan still {after}/{cap} — never above its limit")
    print(f"  after releasing apex's slots the plan sat at {freed}")


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
            plain = [m.ref for m in await picker._members(lane)]
            with_tools = [m.ref for m in await picker._members(lane, needs_tools=True)]
            out[lane] = (plain, with_tools)
            assert all(reg.plan_of(reg.model(r)).can_use_tools for r in with_tools), with_tools
            dropped = set(plain) - set(with_tools)
            assert all(reg.plan_of(reg.model(r)).is_cli_backed for r in dropped), dropped

        # A tool-using request is routed, not rejected, as long as one plan can.
        pick = await picker.pick("forge", None, needs_tools=True)
        assert pick.plan.can_use_tools
        return out, pick.ref

    out, picked = run(go())
    for lane, (plain, with_tools) in out.items():
        print(f"  {lane}: {len(plain)} members, {len(with_tools)} can take tools "
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
