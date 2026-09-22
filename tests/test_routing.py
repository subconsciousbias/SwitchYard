"""Behavioural tests for the three properties the design promises:
ordered fill, capacity that vanishes on quota exhaustion, session affinity.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()

from switchyard import models                      # noqa: E402
from switchyard.picker import LaneSaturated, Picker  # noqa: F401  # noqa: E402
from switchyard.slots import SlotTable             # noqa: E402
from switchyard.usage import Ledger                # noqa: E402
from tests.fake_redis import FakeRedis             # noqa: E402


def build():
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    return reg, slots, Picker(reg, slots)


def build_with_policy():
    """Same as build(), but with a CapacityPolicy wired up.

    The perishable tests need a policy because the picker only reads the
    lane-order key when self.policy is not None -- that is what makes the
    default path free of Redis writes, and it is what the production code
    does (the picker always has a policy in the gateway).
    """
    from switchyard.policy import CapacityPolicy
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    ledger = Ledger(redis)
    policy = CapacityPolicy(redis, reg.settings, ledger)
    return reg, slots, Picker(reg, slots, policy), ledger


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
    """Cancelled capacity is promoted ahead of plans you keep paying for.

    The expiry is synthesised rather than read from the config: whether any
    plan happens to be expiring is a fact about one operator's billing and one
    day's date, and the shipped example config sets no dates at all. Building
    one here tests the rule on any checkout, on any day.
    """
    from dataclasses import replace
    from datetime import date, timedelta

    reg = models.load()
    lane = "judge"
    members = reg.lane_members(lane)
    # The tail is excluded: it sorts last by design whatever its expiry, so
    # using it as the victim tests the tail rule, not the drain rule.
    ordered = [m for m in members if not reg.is_tail(lane, m.ref)]
    assert len(ordered) >= 2, "need two non-tail plans to have an order at all"

    # Take a plan that is NOT already first and give it a near expiry.
    victim = reg.plan_of(ordered[-1])
    # Tomorrow, not merely "inside the window": the rule is soonest-death-first,
    # so a victim must out-expire anything the operator's own config already
    # has, or a real plan expiring sooner legitimately keeps the front spot and
    # the test fails on a working system.
    soon = date.today() + timedelta(days=1)
    plans = dict(reg.plans)
    plans[victim.key] = replace(victim, expires=soon)
    drained = models.Registry(settings=reg.settings, plans=plans, lanes=reg.lanes)

    order = [m for m in drained.lane_members(lane)
             if not drained.is_tail(lane, m.ref)]
    days = [drained.plan_of(m).days_left for m in order]
    assert order[0].plan_key == victim.key, list(zip((m.ref for m in order), days))
    # And every plan with no end date sorts after it.
    keeping = [i for i, d in enumerate(days) if d is None]
    assert not keeping or min(keeping) > 0, list(zip((m.ref for m in order), days))
    print(f"  {victim.key} expiring in {days[0]}d jumps to the front: "
          + " -> ".join(m.ref for m in order))


def test_models_on_one_plan_share_its_connection_limit():
    """`apex` and `judge` name different models of the same Claude Max plan.

    Whatever one lane consumes counts against the other, because the connection
    limit belongs to the plan. Accounting per model instead would let each lane
    open the plan's full allowance.

    Both lanes are confined to Claude Max here, so the assertion is about the
    shared plan and not about which member the drain rule happens to promote.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker = build()
        plans = dict(reg.plans)
        cm = plans["claude-max"]
        plans["claude-max"] = replace(cm, models={
            **cm.models, "fable": replace(cm.models["fable"], enabled=True)})

        lanes = dict(reg.lanes)
        lanes["apex"] = replace(lanes["apex"], order=["claude-max/fable"], tail=[])
        lanes["judge"] = replace(lanes["judge"], order=["claude-max/opus"], tail=[])

        reg = models.Registry(settings=reg.settings, plans=plans, lanes=lanes)
        picker = Picker(reg, slots)
        cap = plans["claude-max"].max_parallel
        assert cap == 2, cap

        # apex takes one (fable caps itself at 1).
        a = await picker.pick("apex", None)
        assert a.ref == "claude-max/fable", a.ref
        after_apex = await slots.in_flight("claude-max")

        # judge may take the plan's remaining slot, and no more.
        b = await picker.pick("judge", None)
        assert b.ref == "claude-max/opus", b.ref
        full = await slots.in_flight("claude-max")

        try:
            await picker.pick("judge", None)
            raise AssertionError("the plan's limit of 2 was exceeded across lanes")
        except LaneSaturated as exc:
            refusal = str(exc)

        await picker.release(a.plan.key, a.request_id, a.ref)
        freed = await slots.in_flight("claude-max")
        return after_apex, full, refusal, freed, cap

    after_apex, full, refusal, freed, cap = run(go())
    assert after_apex == 1 and full == cap and freed == 1
    assert "plan full at 2" in refusal, refusal
    print(f"  apex took claude-max/fable -> plan {after_apex}/{cap}")
    print(f"  judge took claude-max/opus -> plan {full}/{cap}, then refused: "
          f"{refusal.split(': ', 1)[1]}")
    print(f"  releasing apex's slot returned the plan to {freed}/{cap}")


def test_a_plan_marked_unsupported_is_skipped_for_tool_calls():
    """A request carrying `tools` must skip a plan explicitly marked
    `supports_tools: false`, but that plan is still a candidate for a plain
    request. Built with `dataclasses.replace` on an arbitrary member's plan —
    not by relying on which plans happen to be CLI-backed today, since that is
    no longer what determines tool capability.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker = build()
        lane = "forge"
        members = reg.lane_members(lane)
        assert len(members) >= 2, "need at least two members to make this meaningful"
        # Start from a member whose plan is tool-capable in the live config, so
        # flipping it to false is an actual change: several forge members are
        # already marked false while their bridges are unproven.
        target = next((m for m in members if reg.plan_of(m).can_use_tools), None)
        assert target is not None, "forge needs a tool-capable member"
        plan = reg.plan_of(target)

        plans = dict(reg.plans)
        plans[plan.key] = replace(plan, supports_tools=False)
        reg2 = models.Registry(settings=reg.settings, plans=plans, lanes=reg.lanes)
        picker2 = Picker(reg2, slots)

        plain = [m.ref for m in await picker2._members(lane)]
        with_tools = [m.ref for m in await picker2._members(lane, needs_tools=True)]
        assert target.ref in plain, plain
        assert target.ref not in with_tools, with_tools
        assert all(reg2.plan_of(reg2.model(r)).can_use_tools for r in with_tools), with_tools
        return plain, with_tools, target.ref

    plain, with_tools, dropped = run(go())
    print(f"  {dropped}(supports_tools: false) stays in {len(plain)} plain members, "
          f"drops out of {len(with_tools)} tool-capable members")


def test_a_lane_with_no_tool_capable_member_says_so():
    """A lane whose every member's plan is marked `supports_tools: false` must
    refuse a tool-using request with an explanation, rather than quietly
    losing the tool definitions.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker = build()
        lane = "apex"
        members = reg.lane_members(lane)
        assert members, "apex should have live members to make this meaningful"

        plans = dict(reg.plans)
        for m in members:
            plans[m.plan_key] = replace(plans[m.plan_key], supports_tools=False)
        reg2 = models.Registry(settings=reg.settings, plans=plans, lanes=reg.lanes)
        picker2 = Picker(reg2, slots)

        assert await picker2._members(lane)
        assert not await picker2._members(lane, needs_tools=True)
        try:
            await picker2.pick(lane, None, needs_tools=True)
        except LaneSaturated as exc:
            return [m.ref for m in members], str(exc)
        raise AssertionError("expected the lane to refuse a tool-using request")

    members, message = run(go())
    print(f"  apex with every member marked supports_tools: false is {members}")
    print(f"  {message}")


def test_a_lane_with_one_tool_capable_member_routes_to_it():
    """The flip side: as long as one member's plan can serve tools, a
    tool-using request is routed there rather than refused.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker = build()
        lane = "judge"
        members = reg.lane_members(lane)
        assert len(members) >= 2, "need at least two members to make this meaningful"
        capable = members[-1]

        plans = dict(reg.plans)
        for m in members:
            plans[m.plan_key] = replace(
                plans[m.plan_key], supports_tools=(m.plan_key == capable.plan_key))
        reg2 = models.Registry(settings=reg.settings, plans=plans, lanes=reg.lanes)
        picker2 = Picker(reg2, slots)

        pick = await picker2.pick(lane, None, needs_tools=True)
        assert pick.plan.can_use_tools
        assert pick.ref == capable.ref, (pick.ref, capable.ref)
        return capable.ref

    picked = run(go())
    print(f"  judge with only {picked}'s plan tool-capable routed a tool-using request there")


def test_a_pinned_followup_with_zero_wait_spills_to_a_peer():
    """A pinned follow-up is a preference, not a guarantee.

    The pin is the session lease: while it is alive the plan still holds the
    provider's prompt cache for this conversation, so a follow-up whose plan
    has a free slot honours it on the same plan. With `pin_wait_seconds: 0`
    a pinned follow-up whose plan is full, however, spills to a peer — there
    is no wait, because a wait the caller has not asked for would just turn
    into a retry that lands on the same peer anyway. Sessions follow capacity
    rather than block.

    Past the lease TTL the pin is gone with the cache, and the same follow-up
    places fresh — safe now that any bridge rebuilds a lost session from the
    caller's own request. The waiting-then-spilling case is covered by the
    test below.
    """
    async def go():
        reg, slots, picker = build()
        from dataclasses import replace
        picker.registry = replace(
            reg, settings=replace(reg.settings, pin_wait_seconds=0))
        lane = "forge"
        session = "sess-pinned"

        first = await picker.pick(lane, session)
        held = await slots.get_lease(session)
        assert held == first.ref, (held, first.ref)

        # Fill the leased plan so its next claim must fail. The pick above
        # still holds one slot of the plan itself, hence the seed.
        plan = reg.plan_of(first.model)
        taken = [first.request_id]
        for n in range(plan.max_parallel * 2):
            rid = f"filler-a-{n}"
            if await slots.try_claim(plan.key, plan.max_parallel, rid,
                                     first.ref, first.model.max_parallel) != 1:
                break
            taken.append(rid)

        # A plain request follows capacity: it re-leases onto another plan.
        spilled = await picker.pick(lane, session)
        assert spilled.ref != first.ref, "a plain request should have spilled"
        await picker.release(spilled.plan.key, spilled.request_id, spilled.ref)
        await slots.set_lease(session, first.ref, reg.settings.lease_ttl_seconds)

        # The same situation, pinned with wait=0: the follow-up also spills
        # to a peer. The pin is a preference, not a guarantee, and zero wait
        # makes that immediate — refusing instead would just bounce a retry
        # the caller would re-fire onto the peer plan anyway.
        pinned_spill = await picker.pick(lane, session, pinned=True)
        assert pinned_spill.ref != first.ref, (
            "wait=0 must let the pin give way to a peer")
        assert not pinned_spill.sticky
        await picker.release(pinned_spill.plan.key, pinned_spill.request_id,
                             pinned_spill.ref)
        await slots.set_lease(session, first.ref, reg.settings.lease_ttl_seconds)

        # With a slot free on the pinned plan, the pin IS honoured — the wait
        # loop runs once, claims the slot, and returns before the deadline
        # matters.
        for rid in taken:
            await picker.release(plan.key, rid, first.ref)
        resumed = await picker.pick(lane, session, pinned=True)
        assert resumed.ref == first.ref, (resumed.ref, first.ref)
        assert resumed.sticky
        await picker.release(resumed.plan.key, resumed.request_id, resumed.ref)

        # Past the lease TTL the pin is gone with the cache, and the follow-up
        # places fresh — safe now that any bridge rebuilds a lost session from
        # the caller's own request.
        refilled = []
        for n in range(plan.max_parallel * 2):
            rid = f"filler-b-{n}"
            if await slots.try_claim(plan.key, plan.max_parallel, rid,
                                     first.ref, first.model.max_parallel) != 1:
                break
            refilled.append(rid)
        await slots.drop_lease(session)
        late = await picker.pick(lane, session, needs_tools=True, pinned=True)
        assert late.ref != first.ref, (late.ref, first.ref)
        assert not late.sticky
        await picker.release(late.plan.key, late.request_id, late.ref)

        for rid in refilled:
            await picker.release(plan.key, rid, first.ref)
        # The spilled follow-up re-leased the session onto the peer, so the
        # loop continues there: a further follow-up pins to the NEW plan the
        # same way the original one pinned to its plan.
        resumed = await picker.pick(lane, session, pinned=True)
        assert resumed.ref == late.ref, (resumed.ref, late.ref)
        assert resumed.sticky
        return first.ref, pinned_spill.ref

    pinned, spilled_to = run(go())
    print(f"  pinned to {pinned} (wait=0): full -> {spilled_to} (peer); "
          f"free -> same plan; past the lease TTL it spills and re-pins there")


def test_a_heartbeat_keeps_a_long_request_past_the_staleness_sweep():
    """touch() must report whether the slot is still there, and keep it alive.

    ZADD XX counts only members *added*, which under XX is always zero, so the
    liveness check read every successful refresh as "the sweep already took it"
    and each heartbeat stopped after one beat. A request outliving
    inflight_max_age then lost its slot while still running, and a CLI call
    legitimately runs for minutes. Verified against real Redis: `ZADD k XX 200 a`
    returns 0 while `ZADD k XX CH 200 a` returns 1.
    """
    async def go():
        reg, slots, picker = build()
        slots.inflight_max_age = 2          # a sweep we can outlive in a test
        pick = await picker.pick("forge", None)
        plan, ref, rid = pick.plan.key, pick.ref, pick.request_id

        assert await slots.touch(plan, rid, ref) is True, \
            "a live claim must report alive, or its heartbeat gives up"

        # Beat past the sweep horizon; the slot survives because it stays fresh.
        for _ in range(3):
            await asyncio.sleep(1)
            assert await slots.touch(plan, rid, ref) is True
        assert await slots.in_flight(plan) >= 1, "sweep took a heartbeating slot"
        assert await slots.in_flight_model(ref) >= 1

        # Once released, the same call reports gone -- which is what tells the
        # heartbeat to stop rather than resurrect a finished request.
        await picker.release(plan, rid, ref)
        assert await slots.touch(plan, rid, ref) is False
        assert await slots.in_flight_model(ref) == 0, "model counter leaked"
        return plan, ref

    plan, ref = run(go())
    print(f"  {plan}/{ref.split('/')[-1]}: heartbeat held the slot past the sweep, "
          f"release freed both counters")


def test_the_generated_config_declares_no_general_fallbacks():
    """Router-level fallbacks must stay out: they route behind the picker.

    LiteLLM applies them inside the router, after the proxy's pre-call hook, so
    a fallback attempt skips the tool-capability filter, claims no slot, ignores
    the session lease and the mid-loop pin, and its tokens were booked against
    the plan the picker chose rather than the one that served it -- spending one
    subscription's quota and debiting another's. It also hid what it rescued: a
    lane listing every member meant a broken plan was retried on a healthy one
    and looked fine.

    Context-window fallbacks are the allowed exception: a prompt bigger than the
    window cannot be served where it was sent at all.

    The one router retry IS allowed: it routes through the picker via
    `async_pre_routing_hook`, which swaps in a different member of the lane on
    transient failure, so the retry delivers a real second attempt instead of
    the same broken deployment. `disable_cooldowns: true` is the only thing
    keeping the router from cooling the single-deployment group between
    attempts, where the re-pick would never fire.
    """
    from switchyard import gen_litellm

    cfg = gen_litellm.build(os.environ["SWITCHYARD_PLANS"])
    rs = cfg["router_settings"]
    assert rs["fallbacks"] == [], rs["fallbacks"]
    assert cfg["litellm_settings"]["num_retries"] == 1, cfg["litellm_settings"]["num_retries"]
    assert rs["disable_cooldowns"] is True, rs.get("disable_cooldowns")
    assert "context_window_fallbacks" in rs

    # Every deployment a context fallback names must be a real one, or LiteLLM
    # would route to a model group that does not exist.
    names = {m["model_name"] for m in cfg["model_list"]}
    for entry in rs["context_window_fallbacks"]:
        for src, targets in entry.items():
            for t in targets:
                assert t in names, (src, t, sorted(names)[:5])
    print(f"  no general fallbacks; {len(rs['context_window_fallbacks'])} "
          f"context-window entries, all naming real deployments; "
          f"num_retries=1 + disable_cooldowns=True so the router's one retry "
          f"re-routes through the picker")


def test_a_lane_board_separates_its_own_traffic_from_a_sibling_lanes():
    """A model in several lanes is busy for all of them, but the traffic belongs
    to whichever lane claimed it.

    local-box/qwen is in `local` and `bulk`. One request on `local` must read as
    `local`'s own slot there, and as somebody else's on `bulk` — otherwise a
    quiet lane looks busy and there is no way to tell which lane to throttle.
    The slot still costs real capacity in both, so slots_available_now counts it
    either way.
    """
    async def go():
        reg, slots, picker = build()
        pick = await picker.pick("local", "sess-attribution")
        assert pick.plan.key == "local-box", pick.ref

        here = await picker.capacity("local")
        there = await picker.capacity("bulk")

        assert here["slots_in_use_here"] == 1, here
        assert here["slots_in_use_elsewhere"] == 0, here
        assert there["slots_in_use_here"] == 0, there
        assert there["slots_in_use_elsewhere"] == 1, there

        def row(cap, ref):
            return next(r for r in cap["plans"] if r["ref"] == ref)

        mine, theirs = row(here, pick.ref), row(there, pick.ref)
        assert mine["model_in_flight_here"] == 1 and mine["model_in_flight_elsewhere"] == 0
        assert theirs["model_in_flight_here"] == 0 and theirs["model_in_flight_elsewhere"] == 1
        assert "local" in theirs["lanes_sharing"], theirs["lanes_sharing"]
        # The busy slot is real capacity in both views, not conjured away.
        assert mine["model_in_flight"] == theirs["model_in_flight"] == 1

        # A caller naming a deployment directly has no lane; it must not be
        # silently credited to one.
        direct = await picker.pick("bulk", None)
        await slots.release(direct.plan.key, direct.request_id, direct.ref)
        return pick.ref

    ref = run(go())
    print(f"  {ref}: 1 slot reads as 'here' on local and 'elsewhere' on bulk")


def test_a_row_draws_its_own_model_cap_not_the_plans():
    """The lane board's per-row strip must mirror what the model can actually
    claim, not the plan's nominal ceiling.

    `local-box` allows 2 connections; each of its models allows 1. Gemma's row
    used to draw two slot squares because the picker reported the plan's cap
    for the row, and the second one could never serve gemma — only its sibling
    qwen on the same plan. The fix narrows the row the same way the aggregate
    below it already did: each row caps itself at `model.max_parallel`.
    """
    async def go():
        reg, _, picker = build()
        cap = await picker.capacity("local")

        def row(ref):
            return next(r for r in cap["plans"] if r["ref"] == ref)

        gemma, qwen = row("local-box/gemma"), row("local-box/qwen")
        # Each model may claim 1 of the plan's 2 slots. The row is what this
        # model can hold, the configured is the plan's ceiling.
        assert gemma["cap"] == 1, gemma
        assert qwen["cap"] == 1, qwen
        assert gemma["cap_configured"] == 2, gemma
        assert qwen["cap_configured"] == 2, qwen
        assert gemma["model_cap"] == 1 and qwen["model_cap"] == 1
        # The reason names the binding constraint, not the plan's "configured".
        assert gemma["cap_reason"] == "model limit 1", gemma["cap_reason"]
        # Aggregate still counts the plan's reach: 1 (gemma) + 1 (qwen) = 2.
        assert cap["slots_available_now"] == 2, cap
        return gemma["cap"], cap["slots_available_now"]

    cap, total = run(go())
    print(f"  local-box/gemma row: cap={cap}, lane total still {total} "
          "(1 gemma + 1 qwen = plan's 2)")


def test_a_row_signals_when_its_narrowing_is_the_models_own():
    """The board decides how to draw a row from one boolean: was its narrowing
    the model's own choice, or something the operator did (cooldown, paced to
    zero, spent)?

    A row whose cap is narrower than the plan's width SOLELY because of the
    model's own `max_parallel` gets `cap_model_owned: True`. The template
    uses that to skip the withheld-slot loop and the "model limit N" tag, so
    a slow model on a generous plan reads as its own N slots rather than as
    a slice of broken capacity. A row whose model cap is unset, equal to the
    plan, or narrowed for any external reason gets False: the existing
    markup stays in place.

    Per the pinned example fixture:
      - local-box/gemma: model 1, plan 2 → True (narrowed by the model)
      - glm/glm-5.3-flash: model 2, plan 2 → False (model = plan, no narrowing)
      - minimax-ultra/m3: model None, plan 4 → False (no model cap at all)
    """
    async def go():
        reg, _, picker = build()
        local = await picker.capacity("local")
        forge = await picker.capacity("forge")

        def row(cap, ref):
            return next(r for r in cap["plans"] if r["ref"] == ref)

        gemma = row(local, "local-box/gemma")
        glm = row(forge, "glm/glm-5.3-flash")
        m3 = row(forge, "minimax-ultra/m3")

        # The signal tracks the picker's own narrowing step, not a reroll
        # over the plan's nominal ceiling. It is True exactly when the row's
        # effective cap is below the plan's cap AND nothing else can claim
        # credit — the model's own ceiling IS the reason.
        return gemma, glm, m3

    gemma, glm, m3 = run(go())
    assert gemma["cap_model_owned"] is True, gemma
    assert glm["cap_model_owned"] is False, glm
    assert m3["cap_model_owned"] is False, m3
    # And the cap_reason tells the same story from the other direction: the
    # model-owned row carries "model limit 1", the others carry their own
    # reasons or no reason at all.
    assert gemma["cap_reason"] == "model limit 1", gemma
    print(f"  local-box/gemma -> cap_model_owned={gemma['cap_model_owned']} "
          f"(\"{gemma['cap_reason']}\"); glm/glm-5.3-flash -> "
          f"{glm['cap_model_owned']}; minimax-ultra/m3 -> "
          f"{m3['cap_model_owned']}")


def test_a_spent_plan_is_skipped_unless_it_may_use_extra_quota():
    """100% of the target window means no capacity — and breaks affinity.

    Session affinity re-leases only when the leased plan has no free SLOT, so a
    plan that is out of quota but still accepting connections kept every turn
    of a pinned session. Observed live: a session leased to a subscription at
    100% weekly stayed there for hours, each turn quietly spending the prepaid
    credits the provider overflows into.

    `use_extra_quota: true` is the opt-in for exactly that overflow, so a plan
    that bills past its allowance can still be used on purpose.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker = build()
        from switchyard.policy import CapacityPolicy
        from switchyard.usage import Ledger
        redis = slots.redis
        policy = CapacityPolicy(redis, reg.settings, Ledger(redis))
        picker.policy = policy

        lane = "forge"
        members = reg.lane_members(lane)
        target = members[0]
        plan = reg.plan_of(target)
        assert not plan.use_extra_quota, "fixture plan should default to not overspending"

        # Lease a session to it, the way a live conversation does.
        session = "sess-spent"
        await slots.set_lease(session, target.ref, reg.settings.lease_ttl_seconds)
        first = await picker.pick(lane, session)
        assert first.ref == target.ref and first.sticky, first
        await picker.release(first.plan.key, first.request_id, first.ref)

        # The provider now says the target window is fully spent.
        await policy.ledger.note_reported_percent(
            plan.key, 100.0, None, window=plan.quota.label)

        moved = await picker.pick(lane, session)
        await picker.release(moved.plan.key, moved.request_id, moved.ref)

        # Same state, but the plan is allowed to spend past its allowance.
        plans = dict(reg.plans)
        plans[plan.key] = replace(plan, use_extra_quota=True)
        reg2 = models.Registry(settings=reg.settings, plans=plans, lanes=reg.lanes)
        picker2 = Picker(reg2, slots, policy)
        await slots.set_lease(session, target.ref, reg.settings.lease_ttl_seconds)
        allowed = await picker2.pick(lane, session)
        return target.ref, moved, allowed

    ref, moved, allowed = run(go())
    assert moved.ref != ref, f"a spent plan must not keep the lease: {moved.ref}"
    assert allowed.ref == ref, f"use_extra_quota should keep using it: {allowed.ref}"
    print(f"  {ref} spent -> moved to {moved.ref}; with use_extra_quota it stays")


def test_spent_plan_reenters_after_window_rollover():
    """A 100% reading whose `reset_at` has already passed does NOT keep the plan
    out of rotation.

    The bug at #45: a plan with no utilization probe (Claude Code / Codex)
    reports its target window at 100%, the window rolls over, the stale row
    sits in Redis forever, the picker trusts it, the plan gets zero traffic,
    and the vendor's client never writes a fresh number -- so the plan stays
    "spent" permanently even though its weekly allowance just reset. The fix
    is to gate `is_spent` on `reported_is_current`: a stale reading means
    UNKNOWN, not spent, so the plan re-enters and the vendor's next response
    (success or 429) is what settles the truth.

    Mirror of `test_a_spent_plan_is_skipped_unless_it_may_use_extra_quota`,
    but using `build_with_policy()` and supplying a `reset_at` in the past so
    the staleness gate fires. A control case with a future reset_at still
    skips -- the gate does not regress the happy path.
    """
    async def go():
        reg, slots, picker, ledger = build_with_policy()
        lane = "forge"
        members = reg.lane_members(lane)
        target = members[0]
        plan = reg.plan_of(target)
        assert not plan.use_extra_quota, "fixture plan should default to not overspending"

        # 100% reported with reset_at already in the past -> reading is stale
        # about a window that has rolled over. Pre-fix, the picker would see
        # 100% and refuse; post-fix, it re-admits the plan.
        past_reset = time.time() - 3600
        await ledger.note_reported_percent(
            plan.key, 100.0, past_reset, window=plan.quota.label)
        pick_stale = await picker.pick(lane, None)
        await picker.release(pick_stale.plan.key, pick_stale.request_id, pick_stale.ref)

        # Control: same 100% reading, but reset_at in the future. The reading
        # IS about the window in force, so the plan must still be skipped.
        future_reset = time.time() + 7 * 86400
        await ledger.note_reported_percent(
            plan.key, 100.0, future_reset, window=plan.quota.label)
        pick_future = await picker.pick(lane, None)
        await picker.release(pick_future.plan.key, pick_future.request_id, pick_future.ref)

        return target.ref, pick_stale.ref, pick_future.ref

    target_ref, stale_pick, future_pick = run(go())
    assert stale_pick == target_ref, (
        f"a stale 100% reading must not keep {target_ref} out of rotation: "
        f"picked {stale_pick}")
    assert future_pick != target_ref, (
        f"a current 100% reading must still skip {target_ref}: "
        f"picked {future_pick}")
    print(f"  {target_ref}: stale 100% re-admits ({stale_pick}); "
          f"future 100% still skips ({future_pick})")


def test_a_mid_loop_followup_finishes_on_a_spent_plan():
    """A started tool loop completes where it started, spent or not.

    Both halves of this were live failures. A session pinned to a plan that
    went to 100% was moved by the spent gate, and its next turn carried tool
    results whose ids only that plan's bridge had minted — the new plan
    answered 400 "tool results must all belong to exactly one live mcp_bridge
    session" and the caller lost the work it had already done. One extra turn
    of overflow is the cheaper mistake, and new requests still skip the plan,
    so it drains rather than being hammered.
    """
    async def go():
        from switchyard.policy import CapacityPolicy
        from switchyard.usage import Ledger

        reg, slots, picker = build()
        policy = CapacityPolicy(slots.redis, reg.settings, Ledger(slots.redis))
        picker.policy = policy

        lane = "forge"
        target = reg.lane_members(lane)[0]
        plan = reg.plan_of(target)
        session = "sess-midloop"
        await slots.set_lease(session, target.ref, reg.settings.lease_ttl_seconds)
        await policy.ledger.note_reported_percent(
            plan.key, 100.0, None, window=plan.quota.label)

        # A fresh request routes away from the spent plan...
        fresh = await picker.pick(lane, session, pinned=False)
        await picker.release(fresh.plan.key, fresh.request_id, fresh.ref)

        # ...but a follow-up carrying tool results stays on it.
        await slots.set_lease(session, target.ref, reg.settings.lease_ttl_seconds)
        followup = await picker.pick(lane, session, pinned=True)
        await picker.release(followup.plan.key, followup.request_id, followup.ref)
        return target.ref, fresh, followup

    ref, fresh, followup = run(go())
    assert fresh.ref != ref, f"a new request should avoid the spent plan: {fresh.ref}"
    assert followup.ref == ref, f"a follow-up must finish on {ref}, got {followup.ref}"
    assert followup.sticky
    print(f"  {ref} spent: new work -> {fresh.ref}, the running loop stays put")


def test_a_followup_is_recognised_in_both_wire_formats():
    """Tool results look different on each protocol, and missing one is silent.

    The Anthropic shape was not detected, so a follow-up from a Claude-protocol
    client looked like a fresh request and was free to spill to a plan whose
    bridge had never minted its ids.
    """
    from switchyard.hooks import _carries_tool_results as carries

    openai = [{"role": "tool", "tool_call_id": "call_1", "content": "{}"}]
    anthropic = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "{}"}]}]
    assert carries(openai), "OpenAI-shaped tool result missed"
    assert carries(anthropic), "Anthropic-shaped tool result missed"

    # Neither a plain turn nor the assistant's own tool CALL is a follow-up:
    # treating the call as one would pin a request that has nothing to return.
    assert not carries([{"role": "user", "content": "hello"}])
    assert not carries([{"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather"}]}])
    assert not carries(None) and not carries([])
    print("  both protocols' tool results detected; calls and plain turns are not")


def test_naming_a_deployment_still_claims_a_slot_and_honours_limits():
    """Asking for one model is legitimate; bypassing the plan's limits is not.

    A request naming `sy.claude-max.opus` used to skip SwitchYard entirely: no
    slot claimed, no usage recorded, no cooldown or quota respected, nothing on
    the board. The plan's limits are the plan's limits however the request is
    addressed. There is deliberately no spill -- a lane means "best of these",
    a deployment means "this one", so a full plan is an honest refusal.
    """
    async def go():
        reg, slots, picker = build()
        model = reg.lane_members("forge")[0]
        plan = reg.plan_of(model)

        picks = []
        for _ in range(plan.max_parallel):
            picks.append(await picker.pick_direct(model))
        assert all(p.ref == model.ref for p in picks), picks
        assert await slots.in_flight(plan.key) == plan.max_parallel

        try:
            await picker.pick_direct(model)
        except LaneSaturated as exc:
            refused = str(exc)
        else:
            raise AssertionError("a full plan must refuse, not substitute")

        for p in picks:
            await picker.release(p.plan.key, p.request_id, p.ref)
        again = await picker.pick_direct(model)
        await picker.release(again.plan.key, again.request_id, again.ref)
        return model.ref, plan.max_parallel, refused

    ref, cap, refused = run(go())
    assert ref in refused, refused
    print(f"  {ref}: {cap} concurrent, then refused ({refused.split(':')[-1].strip()})")


def test_a_pinned_followup_waits_then_spills_to_a_peer():
    """The pin waits out a short burst, then gives way to a peer.

    A pinned follow-up whose plan is full parks in the gateway for
    `pin_wait_seconds`, holding no slot, so other sessions keep being placed
    while it does. A slot freed inside the deadline is claimed; one that
    doesn't free in time gives way to a peer plan rather than blocking, the
    same way a fresh request would. A plan under an active cooldown is not
    waited on at all — cooldowns run for minutes, not seconds, so the wait
    would just be wasted, and the follow-up spills to a peer immediately.
    """
    import time as _time

    async def go():
        reg, slots, picker = build()
        from dataclasses import replace
        reg = replace(reg, settings=replace(reg.settings, pin_wait_seconds=3.0))
        picker.registry = reg
        lane = "forge"
        session = "sess-pinned-wait"

        first = await picker.pick(lane, session)
        plan = reg.plan_of(first.model)

        # Fill every remaining slot of the pinned plan, so its next claim
        # must fail — the way a burst of concurrent turns does.
        filler = None
        for n in range(plan.max_parallel * 2):
            rid = f"wait-filler-{n}"
            if await slots.try_claim(plan.key, plan.max_parallel, rid,
                                     first.ref, first.model.max_parallel) != 1:
                break
            filler = rid
        assert filler is not None, "could not fill the pinned plan"

        # The pinned follow-up waits and gets the slot the moment it frees.
        async def free_soon():
            await asyncio.sleep(0.3)
            await picker.release(plan.key, first.request_id, first.ref)

        freer = asyncio.create_task(free_soon())
        started = _time.monotonic()
        resumed = await picker.pick(lane, session, pinned=True)
        elapsed = _time.monotonic() - started
        await freer
        assert resumed.ref == first.ref, (resumed.ref, first.ref)
        assert resumed.sticky
        assert 0.3 <= elapsed < 2.5, elapsed
        await picker.release(resumed.plan.key, resumed.request_id, resumed.ref)

        # With no slot ever freeing, the wait runs to the deadline and the
        # follow-up spills to a peer. The wait happened (elapsed is past
        # the deadline) and the spill is honest — a different ref, not sticky.
        refilled = []
        for n in range(plan.max_parallel * 2):
            rid = f"wait-filler-b-{n}"
            if await slots.try_claim(plan.key, plan.max_parallel, rid,
                                     first.ref, first.model.max_parallel) != 1:
                break
            refilled.append(rid)
        started = _time.monotonic()
        spilled = await picker.pick(lane, session, pinned=True)
        elapsed = _time.monotonic() - started
        assert spilled.ref != first.ref, (
            "wait past the deadline must give way to a peer, not refuse")
        assert not spilled.sticky
        assert elapsed >= 2.5, f"the wait should have run its course: {elapsed}"
        await picker.release(spilled.plan.key, spilled.request_id, spilled.ref)
        # The spill re-leased onto the peer — put the lease back so the
        # next scenario exercises the pinned-on-the-original-plan path.
        await slots.set_lease(session, first.ref, reg.settings.lease_ttl_seconds)

        # A plan under an active cooldown is not waited on at all. Cooldowns
        # run for minutes, not seconds, so the wait would just be wasted, and
        # the follow-up gives way to a peer immediately. The failure mode the
        # old code had (silently holding the request for the whole cooldown)
        # was strictly worse: a 429 here, the caller's retry finds the same
        # cooldown, and it 429s again on its own clock.
        await slots.cool_down(plan.key, 900, "quota_exhausted")
        started = _time.monotonic()
        cooled = await picker.pick(lane, session, pinned=True)
        elapsed = _time.monotonic() - started
        assert cooled.ref != first.ref, (
            "a cooled pinned plan must give way to a peer, not wait")
        assert not cooled.sticky
        assert elapsed < 0.5, f"cooled plan must not be waited on: {elapsed}"
        await picker.release(cooled.plan.key, cooled.request_id, cooled.ref)

        for rid in refilled:
            await picker.release(plan.key, rid, first.ref)
        return first.ref, spilled.ref, cooled.ref

    pinned, spilled_to, cooled_to = run(go())
    print(f"  pinned to {pinned} (wait=3s): a free slot in <3s lands here; "
          f"a held plan spills to {spilled_to} after the deadline; a cooled "
          f"plan spills to {cooled_to} without waiting")


def test_pick_honours_exclusions():
    """A failed member is excluded from the candidate set on a re-pick.

    The pre-routing hook gets the same lane / session / tools / pin as the
    failed attempt, but skips the broken member. The remaining members are
    tried in order; the lease is NOT honoured when the held plan is the one
    excluded, so the session follows capacity rather than getting stranded on
    a broken plan.
    """
    async def go():
        reg, slots, picker = build()
        lane = "forge"
        members = reg.lane_members(lane)
        assert len(members) >= 2, "need at least two members for an exclusion"

        first = members[0]
        second = members[1]
        assert first.ref != second.ref

        # Excluding only the first member must pick the second, in order.
        picked = await picker.pick(lane, None, exclude=frozenset({first.ref}))
        assert picked.ref == second.ref, (picked.ref, second.ref)
        await picker.release(picked.plan.key, picked.request_id, picked.ref)

        # Excluding every member must surface LaneSaturated — a re-pick that
        # returns a member would be the wrong answer (the failure would just
        # happen again), and 429 is the only honest thing to send back.
        all_refs = frozenset(m.ref for m in members)
        try:
            await picker.pick(lane, None, exclude=all_refs)
        except LaneSaturated as exc:
            return first.ref, second.ref, str(exc)
        raise AssertionError("excluding every member must raise LaneSaturated")

    first, second, msg = run(go())
    print(f"  excluded {first} -> picked {second}; "
          f"excluding all -> LaneSaturated ({msg.split(': ', 1)[1]})")


def test_repick_moves_the_session_lease_and_returns_not_sticky():
    """A re-pick on a broken lease moves the session to the new plan.

    The transient-failure path must NOT drop_lease — that would strand the
    session on a peer mid-loop and break tool-call id provenance — but it
    must NOT honour the held lease either. The successful re-pick's
    `set_lease` moves the session to its new plan, and the returned Pick
    reads as not-sticky because the held plan was the one that just failed.
    """
    async def go():
        reg, slots, picker = build()
        lane = "forge"
        session = "sess-repick"
        members = reg.lane_members(lane)
        assert len(members) >= 2, "need two members to make a re-pick meaningful"
        first = members[0]
        second = members[1]
        assert first.ref != second.ref

        # Lease the session to the first member, the way a live conversation
        # would have done before the broken call.
        await slots.set_lease(session, first.ref, reg.settings.lease_ttl_seconds)
        held_before = await slots.get_lease(session)
        assert held_before == first.ref, held_before

        # Re-pick excluding the failed member. The lease must NOT be honoured
        # (the held plan is the broken one), but it must NOT be dropped
        # either — the successful pick will re-lease to its new plan.
        picked = await picker.pick(lane, session, exclude=frozenset({first.ref}))
        assert picked.ref == second.ref, (picked.ref, second.ref)
        assert not picked.sticky, "a re-pick past the held plan is not sticky"

        # The successful pick moved the session lease.
        held_after = await slots.get_lease(session)
        assert held_after == picked.ref, (held_after, picked.ref)
        assert held_after != first.ref

        await picker.release(picked.plan.key, picked.request_id, picked.ref)
        return first.ref, second.ref, picked.ref, held_before, held_after

    failed, _, new_ref, before, after = run(go())
    assert before == failed, before
    assert after == new_ref, (after, new_ref)
    print(f"  lease moved from {before} (failed) to {after} (new pick); "
          f"re-pick returned not-sticky")


def test_pick_raises_lane_saturated_when_every_member_is_excluded():
    """The narrowest exclusion — every member — must still surface a 429.

    Whatever the picked shape, the pre-routing hook has to be honest with the
    caller when the re-pick cannot place a request. `LaneSaturated` is the
    pre-call hook's signal to convert that into a 429, so this test confirms
    the same exception survives the exclude path.
    """
    async def go():
        reg, slots, picker = build()
        lane = "local"
        members = reg.lane_members(lane)
        all_refs = frozenset(m.ref for m in members)
        try:
            await picker.pick(lane, None, exclude=all_refs)
        except LaneSaturated as exc:
            return all_refs, str(exc)
        raise AssertionError("local lane must refuse an all-excluded pick")

    refs, msg = run(go())
    print(f"  excluding every member of `local` ({sorted(refs)}) -> "
          f"LaneSaturated ({msg.split(': ', 1)[1]})")


def test_bump_and_cool_atomic_streak_and_cooldown_agree():
    """Atomic INCR + EXPIRE + SET cooldown keeps streak and cooldown in lock-step.

    Concurrent TRANSIENT failures used to race: the last `cool_down` write
    won, but it could have been computed against an earlier (smaller) streak
    than the final INCR value. `bump_and_cool` does INCR + EXPIRE + the
    cooldown math + SET in one Lua, so the cooldown stored in Redis is
    always the cooldown for the streak the caller observes.
    """
    async def go():
        reg, slots, _picker = build()
        plan = next(iter(reg.plans.values())).key

        # First failure: streak 1, base cooldown, 60s.
        s1 = await slots.bump_and_cool(plan, base=60, cap=1800,
                                       reason="transient")
        assert s1 == 1
        cooled, ttl, reason = await slots.cooldown_state(plan)
        assert cooled and reason == "transient", (cooled, reason)
        # Cooldown is the base (60s) on the first failure.
        assert 55 <= ttl <= 60, ttl
        streak = await slots.transient_failure_streak(plan)
        assert streak == 1

        # Second failure: streak 2, doubled to 120s.
        s2 = await slots.bump_and_cool(plan, base=60, cap=1800,
                                       reason="transient")
        assert s2 == 2
        _, ttl2, _ = await slots.cooldown_state(plan)
        assert 115 <= ttl2 <= 120, ttl2

        # Third failure: streak 3, doubled to 240s.
        s3 = await slots.bump_and_cool(plan, base=60, cap=1800,
                                       reason="transient")
        assert s3 == 3
        _, ttl3, _ = await slots.cooldown_state(plan)
        assert 235 <= ttl3 <= 240, ttl3

        # Sixth failure: streak 6, base * 2**5 = 1920 > cap -> cap (1800s).
        for _ in range(3):
            await slots.bump_and_cool(plan, base=60, cap=1800,
                                      reason="transient")
        _, ttl6, _ = await slots.cooldown_state(plan)
        assert 1795 <= ttl6 <= 1800, ttl6

        return streak, ttl, ttl2, ttl3, ttl6

    streak, ttl1, ttl2, ttl3, ttl6 = run(go())
    print(f"  streak grew 1->3 then capped; cooldowns: "
          f"{ttl1}s, {ttl2}s, {ttl3}s, {ttl6}s (cap hit)")


def test_note_transient_failure_sets_ttl_atomically():
    """A worker killed between INCR and EXPIRE used to leave a key with no TTL,
    so the streak would carry forward past 7200s of quiet. The Lua wrapper
    closes that window: every INCR refreshes the TTL on the same call.
    """
    async def go():
        reg, slots, _picker = build()
        plan = next(iter(reg.plans.values())).key

        # First failure: the new key gets a TTL right away.
        await slots.note_transient_failure(plan)
        ttl = await slots.redis.ttl(f"sy:tfail:{plan}")
        assert 7100 <= ttl <= 7200, ttl

        # Second failure: TTL is refreshed, not kept — the property that lets
        # a quiet gap of 7200s drop the counter entirely.
        # FakeRedis stores absolute expiry, so we nudge a couple of seconds
        # off the stored value before the second bump and confirm it lands
        # back near 7200.
        await slots.redis.expire(f"sy:tfail:{plan}", 7100)
        await slots.note_transient_failure(plan)
        ttl_after = await slots.redis.ttl(f"sy:tfail:{plan}")
        assert 7100 <= ttl_after <= 7200, ttl_after

        return ttl, ttl_after

    ttl_first, ttl_second = run(go())
    print(f"  TTL after first bump: {ttl_first}s; "
          f"after refresh: {ttl_second}s — both at the 7200 ceiling")


def test_concurrent_bump_and_cool_keep_cooldown_consistent_with_streak():
    """Under concurrent failures, the cooldown the picker reads matches the
    cooldown for the FINAL streak value, not for some earlier one. Without
    atomicity the two could diverge (last SET wins, but stale streak).
    """
    async def go():
        reg, slots, _picker = build()
        plan = next(iter(reg.plans.values())).key

        # 5 concurrent bumps. The fake's Lua impl runs them sequentially in
        # asyncio scheduling order, but each call still goes through its own
        # atomic INCR-then-SET path. The final cooldown must equal the
        # cooldown for streak=5 (= 60 * 2**4 = 960).
        await asyncio.gather(*(slots.bump_and_cool(
            plan, base=60, cap=1800, reason="transient") for _ in range(5)))

        final_streak = await slots.transient_failure_streak(plan)
        _, final_ttl, _ = await slots.cooldown_state(plan)
        # 60 * 2**(5-1) = 960. Allow a second of slack for the integer now.
        assert final_streak == 5, final_streak
        assert 955 <= final_ttl <= 960, (final_streak, final_ttl)

        return final_streak, final_ttl

    streak, ttl = run(go())
    print(f"  5 concurrent bumps: streak={streak}, cooldown={ttl}s "
          f"(expected 960 for streak=5)")


def test_perishable_lane_routes_higher_room_first():
    """A perishable lane reorders the body by perishable score for new sessions.

    Two members on different plans, the same reset horizon: the emptier weekly
    allowance (90% room) ranks ahead of the half-used one (40% room), so a fresh
    request lands on the emptier plan. The lane's CONFIG order is unchanged --
    `lane_members()` is still the YAML order, and the picker reads its
    perishable rank from the ledger.
    """
    from dataclasses import replace
    from switchyard.policy import CapacityPolicy

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        lanes = dict(reg.lanes)
        lanes["perishable-test"] = replace(
            reg.lanes["apex"],
            key="perishable-test",
            order=["claude-max/fable", "openai/sol"],
            tail=[],
            strategy="perishable",
            description="",
        )
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans, lanes=lanes)
        picker2 = Picker(reg2, slots, picker.policy)
        reset = time.time() + 7 * 86400
        # Plan A empty, plan B half-used, same reset. room = 100 - pct, so:
        #   fable: 90/168 = 0.536,  sol: 40/168 = 0.238
        await ledger.note_reported_percent("claude-max", 10.0, reset, window="weekly")
        await ledger.note_reported_percent("openai", 60.0, reset, window="weekly")
        await ledger.set_lane_order(
            "perishable-test",
            {"claude-max/fable": {"score": 0.536, "gate5h": 0},
             "openai/sol": {"score": 0.238, "gate5h": 0}},
            computed_at=time.time(),
            stale_after_ms=2_000_000,
        )
        p1 = await picker2.pick("perishable-test", None)
        # No release between picks: the first fable claim sits at the model's
        # own counter (model_cap=1) so the next pick falls through to sol,
        # exactly the spill the perishable ordering produces.
        p2 = await picker2.pick("perishable-test", None)
        await picker2.release(p1.plan.key, p1.request_id, p1.ref)
        await picker2.release(p2.plan.key, p2.request_id, p2.ref)
        config_order = [m.ref for m in reg2.lane_members("perishable-test")]
        return p1.ref, p2.ref, config_order

    p1, p2, config = run(go())
    assert p1 == "claude-max/fable", p1
    assert p2 == "openai/sol", p2
    assert config == ["claude-max/fable", "openai/sol"], config
    print(f"  perishable: room 90% ({p1}) first, room 40% ({p2}) second; "
          f"config order unchanged")


def test_perishable_hysteresis_swaps_at_most_one_adjacent_pair():
    """A score below the 1.2x ratio does not swap, a large delta does exactly
    one adjacent swap; an already-aligned order is left alone.

    Hysteresis lives in the writer, so this drives `_one_adjacent_swap`
    directly with three pre-arranged states. The 51/50 case is the noise a
    probe picking up a fraction of a percent should not flip the lane over;
    the 100/50 case is the real gap a 90/30 weekly split would produce.
    """
    from switchyard.portal.app import _one_adjacent_swap

    previous = ["minimax-ultra/m3", "minimax-max/m3"]
    desired_flipped = ["minimax-max/m3", "minimax-ultra/m3"]

    # 51 vs 50: ratio 1.02, below the 1.2x threshold, no swap.
    no_swap = _one_adjacent_swap(
        previous, desired_flipped,
        {"minimax-ultra/m3": (50.0, False), "minimax-max/m3": (51.0, False)})
    assert no_swap == previous, no_swap

    # 100 vs 50: ratio 2.0, well above 1.2x, one swap to align with desired.
    one_swap = _one_adjacent_swap(
        previous, desired_flipped,
        {"minimax-ultra/m3": (50.0, False), "minimax-max/m3": (100.0, False)})
    assert one_swap == desired_flipped, one_swap

    # Already aligned: even with a big score gap, the writer does not invent
    # a swap that was never needed.
    aligned = _one_adjacent_swap(
        desired_flipped, desired_flipped,
        {"minimax-ultra/m3": (50.0, False), "minimax-max/m3": (100.0, False)})
    assert aligned == desired_flipped, aligned

    # Three-deep order, big score gap, still only ONE adjacent swap per poll.
    triple_prev = ["a/m", "b/m", "c/m"]
    triple_desired = ["c/m", "b/m", "a/m"]
    triple = _one_adjacent_swap(
        triple_prev, triple_desired,
        {"a/m": (10.0, False), "b/m": (50.0, False), "c/m": (100.0, False)})
    # b/m in front, c/m behind -- a swap of the first pair is the first
    # change that aligns anything with `desired`. The second pair is left
    # alone, which is the whole point: at most one swap per poll.
    assert triple[0] == "b/m" and triple[1] == "a/m", triple
    print("  51/50 ratio: no swap; 100/50 ratio: one swap; aligned: no swap; "
          "three-deep: only one adjacent pair moves per poll")


def test_perishable_lane_falls_back_to_config_order_when_stale():
    """A ranking whose `computed_at` is older than the writer's staleness
    bound is ignored, and the picker routes in config order.

    The stale ranking says sol comes first, but if the picker honoured it the
    pick would land on the half-used plan -- the wrong one. The staleness
    check in `get_lane_order` returns None, so the picker falls through to
    the existing config-order fill, which is fable first.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        lanes = dict(reg.lanes)
        lanes["perishable-test"] = replace(
            reg.lanes["apex"],
            key="perishable-test",
            order=["claude-max/fable", "openai/sol"],
            tail=[],
            strategy="perishable",
            description="",
        )
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans, lanes=lanes)
        picker2 = Picker(reg2, slots, picker.policy)
        reset = time.time() + 7 * 86400
        # Plans healthy so the picker does not skip them for being spent.
        await ledger.note_reported_percent("claude-max", 10.0, reset, window="weekly")
        await ledger.note_reported_percent("openai", 60.0, reset, window="weekly")
        # The stored order is "sol first" (deliberately wrong per the facts),
        # but it was written far in the past and is well past staleness.
        await ledger.set_lane_order(
            "perishable-test",
            {"openai/sol": {"score": 0.5, "gate5h": 0},
             "claude-max/fable": {"score": 0.1, "gate5h": 0}},
            computed_at=time.time() - 9999,    # 9999s ago
            stale_after_ms=1000,                # 1s -- this is way past
        )
        p1 = await picker2.pick("perishable-test", None)
        await picker2.release(p1.plan.key, p1.request_id, p1.ref)
        return p1.ref

    p1 = run(go())
    # Config order = [claude-max/fable, openai/sol], so fallback = fable first.
    assert p1 == "claude-max/fable", p1
    print(f"  stale ranking ignored, config order used -> {p1}")


def test_perishable_5h_gate_skips_a_member_for_a_new_session():
    """A member flagged `gate5h` is filtered out for a fresh request, listed
    in `considered` so the board says why.

    The gate is written by the portal's writer after reading each plan's 5h
    window facts (pct_used > 90 -> gate5h=1). The picker just trusts the
    stored flag, so this test sets the flag directly: claude-max/fable is
    gated, the ranking lists openai/sol first, and a fresh session lands on
    sol without ever seeing fable.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        lanes = dict(reg.lanes)
        lanes["perishable-test"] = replace(
            reg.lanes["apex"],
            key="perishable-test",
            order=["claude-max/fable", "openai/sol"],
            tail=[],
            strategy="perishable",
            description="",
        )
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans, lanes=lanes)
        picker2 = Picker(reg2, slots, picker.policy)
        # Make plans not look "spent" -- only the gate flag matters here.
        reset = time.time() + 7 * 86400
        await ledger.note_reported_percent("claude-max", 10.0, reset, window="weekly")
        await ledger.note_reported_percent("openai", 10.0, reset, window="weekly")
        await ledger.set_lane_order(
            "perishable-test",
            {"claude-max/fable": {"score": 0.0, "gate5h": 1},   # gated
             "openai/sol": {"score": 0.5, "gate5h": 0}},
            computed_at=time.time(),
            stale_after_ms=2_000_000,
        )
        p1 = await picker2.pick("perishable-test", None)
        return p1.ref, p1.considered

    picked, considered = run(go())
    assert picked == "openai/sol", picked
    assert any("gate5h" in c for c in considered), considered
    print(f"  gate5h skipped: picked {picked}, considered {considered}")


def test_perishable_unknown_member_sorts_after_scored_ones():
    """A member with no score slot lands after every scored member, in
    config position. The picker reorders body members by score desc, with
    unknown members filling in at their original config spot.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        lanes = dict(reg.lanes)
        lanes["perishable-test"] = replace(
            reg.lanes["apex"],
            key="perishable-test",
            order=["claude-max/fable", "openai/sol", "glm/glm-5.3"],
            tail=[],
            strategy="perishable",
            description="",
        )
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans, lanes=lanes)
        picker2 = Picker(reg2, slots, picker.policy)
        reset = time.time() + 7 * 86400
        await ledger.note_reported_percent("claude-max", 10.0, reset, window="weekly")
        await ledger.note_reported_percent("openai", 50.0, reset, window="weekly")
        # glm/glm-5.3 deliberately has no entry -- it is the unknown, and
        # must sort AFTER both scored members.
        await ledger.set_lane_order(
            "perishable-test",
            {"openai/sol": {"score": 0.5, "gate5h": 0},          # 50% room
             "claude-max/fable": {"score": 0.536, "gate5h": 0},  # 90% room
             },
            computed_at=time.time(),
            stale_after_ms=2_000_000,
        )
        picks = []
        # No release between picks: each model's own cap=1 forces a spill to
        # the next member, exposing the perishable reordering directly.
        for _ in range(3):
            p = await picker2.pick("perishable-test", None)
            picks.append(p.ref)
        return picks

    picks = run(go())
    assert picks[0] == "claude-max/fable", picks  # higher score
    assert picks[1] == "openai/sol", picks
    assert picks[2] == "glm/glm-5.3", picks       # unknown sorts last
    print(f"  scored first, unknown last: {' -> '.join(picks)}")


def test_a_strategy_absent_lane_never_touches_the_lane_order_key():
    """A lane without `strategy: perishable` is bit-for-bit unchanged: the
    picker never reads or writes the lane-order key. Verified by asserting
    the hash is absent from FakeRedis after exercising a fill-strategy lane.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        redis = slots.redis
        lanes = dict(reg.lanes)
        lanes["fill-test"] = replace(
            reg.lanes["apex"],
            key="fill-test",
            order=["claude-max/fable", "openai/sol"],
            tail=[],
            strategy="fill",
            description="",
        )
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans, lanes=lanes)
        picker2 = Picker(reg2, slots, picker.policy)
        key = "sy:lane-order:fill-test"
        before = key in redis.hashes
        # A successful pick in config order (the lane is fill, so the YAML
        # order wins).
        p1 = await picker2.pick("fill-test", None)
        # Saturate the lane to make sure even refusal paths don't write.
        try:
            for _ in range(20):
                p = await picker2.pick("fill-test", None)
                await picker2.release(p.plan.key, p.request_id, p.ref)
        except LaneSaturated:
            pass
        after = key in redis.hashes
        return before, after, p1.ref

    before, after, p1 = run(go())
    assert not before, before
    assert not after, after
    # The fill lane behaves exactly as it did before perishable landed:
    # first config member wins.
    assert p1 == "claude-max/fable", p1
    print(f"  fill-strategy lane: no lane-order key (before={before}, "
          f"after={after}); first pick still {p1} in config order")


def test_perishable_writer_keeps_both_plans_after_two_polls():
    """A perishable lane whose body mixes two plans (the common shape: every
    shipped example lane does) survives the writer running for each plan in
    turn. The blocker at bfa9ad3 was that `set_lane_order` is delete-then-write
    and `_recompute_perishable_for_plan` only scored members of THIS plan, so
    the second plan's write wiped the first plan's entries from the hash.

    The fix scores every lane member in one pass: members on THIS plan use
    fresh facts; members on OTHER plans reuse the score that was last written
    for them (and which is at most one probe interval old, well inside the
    staleness window the picker already tolerates). Driving the writer for
    both plans in sequence and asserting both refs land in the stored hash
    with non-zero scores is what catches a regression to the per-plan-only
    behaviour.
    """
    from dataclasses import replace
    from switchyard.portal.app import _recompute_perishable_for_plan

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        # apex already has one member per plan (claude-max/fable and
        # openai/astra). Flip it to perishable with no tail, so the test is
        # just about the body. The registry is rebuilt so the lane order
        # takes effect; plans are unchanged.
        lanes = dict(reg.lanes)
        lanes["perishable-test"] = replace(
            reg.lanes["apex"],
            key="perishable-test",
            tail=[],
            strategy="perishable",
            description="",
        )
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans,
                               lanes=lanes)
        reset = time.time() + 7 * 86400
        # claude-max 80% used (room 20%); openai 10% used (room 90%).
        # openai/astra should outrank claude-max/fable on score.
        await ledger.note_reported_percent("claude-max", 80.0, reset,
                                           window="weekly")
        await ledger.note_reported_percent("openai", 10.0, reset,
                                           window="weekly")
        # Drive the writer for claude-max first. With NO previous, only
        # this plan's refs get a fresh score; openai/astra has no previous
        # entry either, so it lands in `unknown` and is NOT written.
        plan_a = reg2.plans["claude-max"]
        await _recompute_perishable_for_plan(reg2, ledger, plan_a)
        after_first = await ledger.get_lane_order("perishable-test")
        # Drive the writer for openai. Now both plans have facts: openai/
        # astra is fresh, claude-max/fable is reused from `previous`. Both
        # refs must survive in the hash.
        plan_b = reg2.plans["openai"]
        await _recompute_perishable_for_plan(reg2, ledger, plan_b)
        after_second = await ledger.get_lane_order("perishable-test")
        return after_first, after_second

    after_first, after_second = run(go())
    refs_first = sorted(m["ref"] for m in after_first["members"])
    refs_second = [m["ref"] for m in after_second["members"]]
    # First poll: only the just-probed plan's ref is scored; the other
    # plan's ref has no previous and is unknown, so it is not in the hash.
    assert refs_first == ["claude-max/fable"], refs_first
    # Second poll: BOTH plans' refs are scored, in score-descending order
    # (openai/astra has higher room than claude-max/fable, so it leads).
    assert refs_second == ["openai/astra", "claude-max/fable"], refs_second
    # Every score is a real number, not the stale-zero fallback that a
    # regression would produce (a ref reused from previous carries its
    # last-written score, which is itself a fresh score from one poll ago).
    for entry in after_second["members"]:
        assert entry["score"] > 0, entry
    print(f"  first poll refs={refs_first}; "
          f"second poll refs={refs_second} (both plans present)")


def test_perishable_writer_preserves_other_plan_score_when_re_polled():
    """Driving the writer for plan B does not lose plan A's score. Plan A's
    previous score is reused, so even though the writer is invoked per-plan
    the per-lane hash stays a complete picture across the probe poll.

    `set_lane_order` deletes the key first to avoid carrying over a stale
    `score_<ref>` for a ref that left the lane -- so the test asserts that
    the surviving score comes from THIS recompute (re-reading plan A's
    facts through `previous`), not from a stale field left over by the
    delete.
    """
    from dataclasses import replace
    from switchyard.portal.app import _recompute_perishable_for_plan

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        lanes = dict(reg.lanes)
        lanes["perishable-test"] = replace(
            reg.lanes["apex"],
            key="perishable-test",
            tail=[],
            strategy="perishable",
            description="",
        )
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans,
                               lanes=lanes)
        reset = time.time() + 7 * 86400
        await ledger.note_reported_percent("claude-max", 80.0, reset,
                                           window="weekly")
        await ledger.note_reported_percent("openai", 10.0, reset,
                                           window="weekly")
        plan_a, plan_b = reg2.plans["claude-max"], reg2.plans["openai"]
        # Poll 1: claude-max. fable scored, astra unknown (no previous).
        await _recompute_perishable_for_plan(reg2, ledger, plan_a)
        # Poll 2: openai. astra fresh; fable reused from previous.
        await _recompute_perishable_for_plan(reg2, ledger, plan_b)
        score_fable_pre = next(
            m["score"] for m in
            (await ledger.get_lane_order("perishable-test"))["members"]
            if m["ref"] == "claude-max/fable")
        # Mutate claude-max's facts OUT OF BAND and re-poll openai again.
        # The writer must not pick up the new number for fable (it is on
        # the other plan); fable's score in the hash must stay what it was
        # at the previous openai poll, since we have not re-polled
        # claude-max. If the writer were to re-read claude-max's facts on
        # every recompute, fable's score would change here, which would be
        # wrong -- openai's poll has no business recomputing claude-max's
        # number.
        await ledger.note_reported_percent("claude-max", 99.0, reset,
                                           window="weekly")
        await _recompute_perishable_for_plan(reg2, ledger, plan_b)
        score_fable_post = next(
            m["score"] for m in
            (await ledger.get_lane_order("perishable-test"))["members"]
            if m["ref"] == "claude-max/fable")
        return score_fable_pre, score_fable_post

    pre, post = run(go())
    # Fable's score in the hash is unchanged across the second openai poll:
    # the writer reused `previous` rather than re-reading claude-max's
    # now-different facts.
    assert abs(pre - post) < 1e-9, (pre, post)
    print(f"  fable's score unchanged across openai's second poll "
          f"(pre={pre:.4f}, post={post:.4f})")


def test_perishable_one_adjacent_swap_reconciles_added_and_removed_refs():
    """`_one_adjacent_swap` drops refs from `previous` that have left the
    lane (a config edit, a member's plan disabled, etc.) and appends refs
    in `desired` that are not yet in `previous` (a member whose plan just
    probed for the first time, a new perishable lane warming up). The
    one-swap cap still applies on top, so the result is reachable from
    `previous` via at most one adjacent swap after reconciliation.

    The should-fix at bfa9ad3 was that the implementation never moved
    toward `desired` -- it only reordered `previous`. A newly probed ref
    stayed absent from the hash, and a removed ref came back as a stale
    `score: 0.0` entry.
    """
    from switchyard.portal.app import _one_adjacent_swap

    # Drop a removed ref (config edit removed it), add a new ref (first
    # probe of a different plan's model), with a score ratio that justifies
    # one swap.
    added_only = _one_adjacent_swap(
        ["minimax-ultra/m3"],
        ["minimax-max/m3", "minimax-ultra/m3"],
        {"minimax-ultra/m3": (50.0, False), "minimax-max/m3": (100.0, False)},
    )
    # out starts as [minimax-ultra/m3]; append minimax-max/m3 (in desired
    # order).  Loop: pair (minimax-ultra/m3, minimax-max/m3). sb/sa = 2.0
    # >= 1.2; desired.index(minimax-max/m3) < desired.index(minimax-ultra/m3);
    # swap. Result: [minimax-max/m3, minimax-ultra/m3].
    assert added_only == ["minimax-max/m3", "minimax-ultra/m3"], added_only

    # Removed ref is gone, surviving refs unchanged, no swap needed.
    dropped_only = _one_adjacent_swap(
        ["a/m", "b/m"],
        ["b/m"],
        {"a/m": (10.0, False), "b/m": (10.0, False)},
    )
    assert dropped_only == ["b/m"], dropped_only

    # Drop AND add in one call: both reconcile, one swap.
    added_and_dropped = _one_adjacent_swap(
        ["a/m", "b/m", "c/m"],
        ["c/m", "b/m", "d/m"],
        {"a/m": (10.0, False), "b/m": (50.0, False),
         "c/m": (100.0, False), "d/m": (50.0, False)},
    )
    # Step 1: drop a/m. out = [b/m, c/m].
    # Step 2: append d/m. out = [b/m, c/m, d/m].
    # Loop: pair (b/m, c/m). sb/sa = 100/50 = 2.0; desired.index(c/m) <
    # desired.index(b/m); swap. Result: [c/m, b/m, d/m].
    assert added_and_dropped == ["c/m", "b/m", "d/m"], added_and_dropped

    print(f"  added: {added_only}; dropped: {dropped_only}; "
          f"added+dropped: {added_and_dropped}")


def test_a_cached_prompt_is_booked_whole():
    """A cached Anthropic turn hides almost all of its prompt outside input_tokens.

    Measured on a live gateway: a 48,240-token Claude Code turn arrives as
    input_tokens=6 with the rest in the two cache counters, and 170 requests
    booked 472 prompt tokens between them. The capacity planner spends the
    subscription against that number, so reading input_tokens alone is not a
    small error -- it is three orders of magnitude of headroom invented.
    """
    from switchyard.hooks import _prompt_tokens

    def reader(block):
        return lambda k: block.get(k, 0)

    anthropic = {"input_tokens": 6, "cache_creation_input_tokens": 6932,
                 "cache_read_input_tokens": 41302, "output_tokens": 201}
    assert _prompt_tokens(reader(anthropic)) == 48240

    # The OpenAI pair is already whole -- cli_bridge folds both cache counters
    # into prompt_tokens before answering -- so it wins outright. Adding the
    # counters again here would double-count every bridged request.
    both = {"prompt_tokens": 48240, "input_tokens": 6,
            "cache_read_input_tokens": 41302,
            "cache_creation_input_tokens": 6932}
    assert _prompt_tokens(reader(both)) == 48240

    assert _prompt_tokens(reader({"prompt_tokens": 1200})) == 1200
    assert _prompt_tokens(reader({})) == 0
    print("  cached prompt booked whole (48,240), OpenAI pair not double-counted")


def test_a_split_stream_keeps_both_halves_of_its_usage():
    """Anthropic frames put the input and output counts on different chunks.

    message_start carries input_tokens and the cache counters; message_delta
    carries output_tokens near the end. A reader that keeps only the last
    usage block it saw keeps the second one and books the entire prompt as
    zero -- which is what a streamed turn did here for six hours straight.
    """
    from switchyard.hooks import _note_usage, _prompt_tokens

    seen: dict = {}
    _note_usage({"type": "message_start",
                 "usage": {"input_tokens": 6, "cache_creation_input_tokens": 6932,
                           "cache_read_input_tokens": 41302, "output_tokens": 1}}, seen)
    _note_usage({"type": "content_block_delta"}, seen)          # no usage at all
    _note_usage({"type": "message_delta", "usage": {"output_tokens": 201}}, seen)

    get = lambda k: seen["usage"].get(k, 0)                      # noqa: E731
    assert _prompt_tokens(get) == 48240, seen
    assert seen["usage"]["output_tokens"] == 201, seen

    # An object-shaped chunk reads the same way, and a stream that carried
    # nothing must leave the dict empty so the caller can tell it apart from
    # a turn that genuinely cost nothing.
    class Chunk:
        usage = type("U", (), {"prompt_tokens": 12, "completion_tokens": 3})()

    other: dict = {}
    _note_usage(Chunk(), other)
    assert other["usage"]["prompt_tokens"] == 12, other
    empty: dict = {}
    _note_usage({"type": "ping"}, empty)
    assert empty == {}, empty
    print("  split stream merged: prompt 48,240, completion 201")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
    print("\nall routing tests passed")
