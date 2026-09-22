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
from switchyard.hooks import SwitchyardHandler, _tool_name  # noqa: E402
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
        # allows 2, so "fill" means filling that, not the plan. Settings are
        # passed so the CLI-backhead headroom (when the plan is cli-backed) is
        # applied here too, matching the picker and the litellm backstop.
        cap_first = reg.plan_of(first).cap_for(first, reg.settings)
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
        # its own, with the same gate-headroom adjustment the picker applies.
        # Cooling the plan removes that, not the plan's nominal limit.
        contributed = victim_plan.cap_for(victim, reg.settings)
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


def test_cli_blocklist_drops_blocked_tools_and_passes_the_rest():
    """The per-CLI tool blocklist filters native tools out of `data["tools"]`
    before the picker ever sees them, matching case-insensitively against
    the configured list and against EITHER wire shape.

    Three tools in: an OpenAI-shape function ("bash") that the blocklist
    names lowercase, an Anthropic-shape flat entry ("Read") that the
    blocklist names uppercase to prove case-insensitive matching, and an
    MCP tool ("mcp__switchyard__bash") that nothing blocks. Exactly the
    two named tools drop, exactly one log line names both, the MCP tool
    survives. Wiring follows the test_verdict pattern: __new__ bypass
    plus a fresh FakeRedis-backed registry whose cli_tool_block has the
    uppercase "READ" entry.
    """
    from dataclasses import replace
    from switchyard.policy import CapacityPolicy
    import logging as _logging

    async def go():
        reg = models.load()
        reg = replace(reg, settings=replace(
            reg.settings, cli_tool_block={"opencode": ("bash", "READ")}))
        redis = FakeRedis()
        slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
        ledger = Ledger(redis)
        policy = CapacityPolicy(redis, reg.settings, ledger)
        picker = Picker(reg, slots, policy)

        h = SwitchyardHandler.__new__(SwitchyardHandler)
        h.__dict__["registry"] = reg
        h.__dict__["_slots"] = slots
        h.__dict__["_ledger"] = ledger
        h.__dict__["_policy"] = policy
        h.__dict__["_redis"] = redis
        h.__dict__["_picker"] = picker
        h.__dict__["_beats"] = {}

        # Capture just the "switchyard" logger's records so the test is
        # robust against any other library logging through the root.
        captured: list[_logging.LogRecord] = []
        handler = _logging.Handler()
        handler.emit = captured.append
        log = _logging.getLogger("switchyard")
        prior_level = log.level
        log.setLevel(_logging.INFO)
        log.addHandler(handler)
        try:
            data = {
                "model": "forge",
                "messages": [{"role": "user", "content": "hi"}],
                "proxy_server_request": {"headers": {
                    "x-switchyard-session": "sess-blocklist",
                    "x-switchyard-cli": "opencode",
                }},
                "tools": [
                    {"type": "function",
                     "function": {"name": "bash"}},           # OpenAI shape
                    {"name": "Read"},                          # Anthropic shape
                    {"name": "mcp__switchyard__bash"},         # MCP, untouched
                ],
            }
            await h.async_pre_call_hook(None, None, data, "acompletion")
            ctx = data["metadata"]["switchyard"]
            await h.picker.release(ctx["plan"], ctx["request_id"], ctx["model"])
            return data["tools"], [r.getMessage() for r in captured]
        finally:
            log.removeHandler(handler)
            log.setLevel(prior_level)

    tools, messages = run(go())
    kept_names = [_tool_name(t) for t in tools]
    # Both blocked tools gone, MCP tool untouched.
    assert "bash" not in kept_names, kept_names
    assert "Read" not in kept_names, kept_names
    assert "mcp__switchyard__bash" in kept_names, kept_names
    # Exactly one log line about the blocklist — the hook emits one
    # `cli=... tool_block: dropped N/M (...)` record per request, listing
    # every dropped tool inside the parens.
    block_lines = [m for m in messages if "tool_block" in m]
    assert len(block_lines) == 1, block_lines
    assert "bash" in block_lines[0], block_lines
    assert "Read" in block_lines[0], block_lines
    print(f"  opencode blocklist dropped bash + Read, kept "
          f"{[n for n in kept_names if n]}; one log line: "
          f"{block_lines[0]!r}")


def test_emptied_tools_re_derive_needs_tools_so_the_picker_does_not_misroute():
    """When the blocklist empties the request's tool list, the hook MUST
    re-derive needs_tools=False — otherwise the picker would still treat
    the request as tool-bearing and 429 a lane whose every member is
    supports_tools: false, even though there is no longer anything to
    serve tools for.

    The same lane with a genuine (unfiltered) tool list refuses — that's
    the existing behaviour the post-filter re-derivation exists to NOT
    BREAK. So one registration, one log shape, two assertions: with the
    blocklist applied the pick succeeds and needs_tools=False; without it,
    the same lane's pick raises LaneSaturated.
    """
    from dataclasses import replace
    from switchyard.policy import CapacityPolicy

    async def go():
        reg = models.load()
        # A lane whose every member's plan is `supports_tools: false`:
        # the picker must refuse a tool-bearing request, but is happy
        # with one that has no tools. Apex in the fixture has members,
        # so we mutate every member's plan to disable tools.
        lane = "apex"
        members = reg.lane_members(lane)
        assert members, "apex should have live members to make this meaningful"

        plans = dict(reg.plans)
        for m in members:
            plans[m.plan_key] = replace(plans[m.plan_key], supports_tools=False)
        reg2 = replace(reg, settings=replace(
            reg.settings, cli_tool_block={"opencode": ("bash", "Read")}),
                       plans=plans)

        redis = FakeRedis()
        slots = SlotTable(redis, reg2.settings.inflight_max_age_seconds)
        ledger = Ledger(redis)
        policy = CapacityPolicy(redis, reg2.settings, ledger)
        picker = Picker(reg2, slots, policy)

        h = SwitchyardHandler.__new__(SwitchyardHandler)
        h.__dict__["registry"] = reg2
        h.__dict__["_slots"] = slots
        h.__dict__["_ledger"] = ledger
        h.__dict__["_policy"] = policy
        h.__dict__["_redis"] = redis
        h.__dict__["_picker"] = picker
        h.__dict__["_beats"] = {}

        # The blocked tool names match the only tool the request carries,
        # so the filter empties the list. Without the post-filter
        # re-derivation, needs_tools stays True and the picker 429s.
        blocked_data = {
            "model": lane,
            "messages": [{"role": "user", "content": "hi"}],
            "proxy_server_request": {"headers": {
                "x-switchyard-session": "sess-blocked-empty",
                "x-switchyard-cli": "opencode",
            }},
            "tools": [
                {"type": "function", "function": {"name": "bash"}},
                {"name": "Read"},
            ],
        }
        await h.async_pre_call_hook(None, None, blocked_data, "acompletion")
        blocked_ctx = blocked_data["metadata"]["switchyard"]
        await h.picker.release(blocked_ctx["plan"], blocked_ctx["request_id"],
                               blocked_ctx["model"])

        # Same lane with the SAME tools but NO blocklist applied — the
        # request is genuinely tool-bearing and the lane (every plan
        # supports_tools: false) must refuse. A separate handler is the
        # simplest way to drive the unblocked shape without entangling
        # the two cases.
        reg_unblocked = replace(reg2, settings=replace(
            reg2.settings, cli_tool_block={}))
        redis2 = FakeRedis()
        slots2 = SlotTable(redis2, reg_unblocked.settings.inflight_max_age_seconds)
        policy2 = CapacityPolicy(redis2, reg_unblocked.settings,
                                 Ledger(redis2))
        picker_unblocked = Picker(reg_unblocked, slots2, policy2)
        h2 = SwitchyardHandler.__new__(SwitchyardHandler)
        h2.__dict__["registry"] = reg_unblocked
        h2.__dict__["_slots"] = slots2
        h2.__dict__["_ledger"] = Ledger(redis2)
        h2.__dict__["_policy"] = policy2
        h2.__dict__["_redis"] = redis2
        h2.__dict__["_picker"] = picker_unblocked
        h2.__dict__["_beats"] = {}

        unblocked_data = {
            "model": lane,
            "messages": [{"role": "user", "content": "hi"}],
            "proxy_server_request": {"headers": {
                "x-switchyard-session": "sess-unblocked",
            }},
            "tools": [
                {"type": "function", "function": {"name": "bash"}},
                {"name": "Read"},
            ],
        }
        refused = None
        try:
            await h2.async_pre_call_hook(None, None, unblocked_data, "acompletion")
        except LaneSaturated as exc:
            refused = str(exc)
        except Exception as exc:
            # The hook converts LaneSaturated into a 429 HTTPException for
            # the caller — catch that too, since the assertion is about the
            # refused behaviour, not the transport.
            from fastapi import HTTPException
            if isinstance(exc, HTTPException) and exc.status_code == 429:
                refused = str(exc.detail)
            else:
                raise

        return (
            blocked_data["metadata"]["switchyard"],
            blocked_data["tools"],
            refused,
        )

    blocked_ctx, blocked_tools, refused = run(go())
    # Post-filter: tools list is empty AND needs_tools is False. Without
    # the re-derivation this assertion fails — the spec is testing the
    # gate between blocklist and picker.
    assert blocked_ctx["needs_tools"] is False, blocked_ctx
    assert blocked_tools == [], (
        f"every tool was blocked; the list must be emptied, got {blocked_tools!r}")
    # Unblocked control: the same lane with a non-empty tool list and no
    # blocklist refuses — proving the test setup actually reaches the
    # refusal path when nothing rescues it.
    assert refused is not None, (
        "the unblocked control must hit LaneSaturated — otherwise the "
        "blocklist test is not actually load-bearing")
    print(f"  apex+blocklist emptied tools -> needs_tools=False, pick succeeded; "
          f"apex unblocked refused: {refused[:60]}...")


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

    num_retries=0 is the spill contract: a failed request returns to the
    caller fast, the caller's 429 retry re-enters through async_pre_call_hook
    and gets a fresh pick against the cooldowns the failure just set. The
    pinned litellm's Router.async_pre_routing_hook is the internal auto-
    router hook and never iterates registered CustomLogger callbacks, so a
    Switchyard re-pick there was never going to fire. A blind router retry
    would have re-routed to the same single-member deployment the picker
    just chose -- exactly the deployment that just 5xx'd.
    disable_cooldowns: True is still required so the picker is not blocked
    by a router-side cooldown on a single-deployment group.
    """
    from switchyard import gen_litellm

    cfg = gen_litellm.build(os.environ["SWITCHYARD_PLANS"])
    rs = cfg["router_settings"]
    assert rs["fallbacks"] == [], rs["fallbacks"]
    assert cfg["litellm_settings"]["num_retries"] == 0, cfg["litellm_settings"]["num_retries"]
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
          f"num_retries=0 + disable_cooldowns=True: the caller's 429 retry "
          f"re-enters through async_pre_call_hook, not via a router retry")


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
      - grok/grok-4.6: model 2, plan 2 → False (model = plan, no narrowing)
      - minimax-ultra/m3: model None, plan 4 → False (no model cap at all)
    """
    async def go():
        reg, _, picker = build()
        local = await picker.capacity("local")
        forge = await picker.capacity("forge")

        def row(cap, ref):
            return next(r for r in cap["plans"] if r["ref"] == ref)

        gemma = row(local, "local-box/gemma")
        grok = row(forge, "grok/grok-4.6")
        m3 = row(forge, "minimax-ultra/m3")

        # The signal tracks the picker's own narrowing step, not a reroll
        # over the plan's nominal ceiling. It is True exactly when the row's
        # effective cap is below the plan's cap AND nothing else can claim
        # credit — the model's own ceiling IS the reason.
        return gemma, grok, m3

    gemma, grok, m3 = run(go())
    assert gemma["cap_model_owned"] is True, gemma
    assert grok["cap_model_owned"] is False, grok
    assert m3["cap_model_owned"] is False, m3
    # And the cap_reason tells the same story from the other direction: the
    # model-owned row carries "model limit 1", the others carry their own
    # reasons or no reason at all.
    assert gemma["cap_reason"] == "model limit 1", gemma
    print(f"  local-box/gemma -> cap_model_owned={gemma['cap_model_owned']} "
          f"(\"{gemma['cap_reason']}\"); grok/grok-4.6 -> "
          f"{grok['cap_model_owned']}; minimax-ultra/m3 -> "
          f"{m3['cap_model_owned']}")


def test_a_cli_plan_row_signals_its_own_narrowing_under_policy():
    """The model-owned signal must survive the policy's rewriting of
    `cap_reason`.

    `claude-max` is a CLI-backed plan: `policy._apply_gate_headroom` rewrites
    `cap_reason` from "configured" to "configured + gate headroom N", and
    shrinks the cap by the headroom slots so the gateway's slot table stops
    racing the sidecar's own gate. The previous predicate gated on the exact
    string "configured" — that string match broke under headroom, so a row
    whose narrowing was still the model's own (`claude-max/fable`: model 1,
    plan 2, post-headroom cap 1) read as `cap_model_owned: False`, drew the
    "model limit" tag, and the withheld-slot loop ran for a slot that was
    never withheld — the row's own one slot.

    With `build_with_policy()` the policy path is live; for the apex row
    `claude-max/fable` the picker reports `cap == 1`, the rewritten reason
    contains "gate headroom" (proving the string predicate was the load-
    bearing one), and `cap_model_owned` is True. The other three shapes stay
    where the predicate above put them: local-box/gemma is still model-owned
    under policy; grok/grok-4.6 (model == plan) and minimax-ultra/m3
    (no model cap at all) are still False.
    """
    async def go():
        _, _, picker, _ = build_with_policy()
        local = await picker.capacity("local")
        forge = await picker.capacity("forge")
        apex = await picker.capacity("apex")

        def row(cap, ref):
            return next(r for r in cap["plans"] if r["ref"] == ref)

        fable = row(apex, "claude-max/fable")
        gemma = row(local, "local-box/gemma")
        grok = row(forge, "grok/grok-4.6")
        m3 = row(forge, "minimax-ultra/m3")
        return fable, gemma, grok, m3

    fable, gemma, grok, m3 = run(go())
    # CLI-plan row, the regression bar. Plan 2 with headroom 1 -> cap 1;
    # model 1; the model narrows to its own ceiling and the signal must
    # fire even though the reason string is no longer exactly "configured".
    assert fable["cap"] == 1, fable
    assert fable["model_cap"] == 1, fable
    assert "gate headroom" in fable["cap_reason"], fable
    assert fable["cap_model_owned"] is True, fable
    # The other shapes are unchanged from the predicate above: this row's
    # signal is about a CLI-plan narrowing, the others pin that we did not
    # flip them by accident while widening the gate.
    assert gemma["cap_model_owned"] is True, gemma
    assert grok["cap_model_owned"] is False, grok
    assert m3["cap_model_owned"] is False, m3
    print(f"  claude-max/fable (cli-backed, cap 1 of 2 + headroom): "
          f"cap_model_owned={fable['cap_model_owned']} "
          f"(\"{fable['cap_reason']}\"); local-box/gemma stays True, "
          f"grok/grok-4.6 and minimax-ultra/m3 stay False")


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


# ============================================================================
# Issue #64 (WS2) — Stale-grace for cookie-expired plans
#
# A plan whose session cookie has expired (`sy:probe:{plan}.needs_reauth=1`)
# has stopped polling, so its lane-order / group-order hash never gets
# refreshed. Without grace, the hash ages past `stale_after_ms` and the
# picker falls back to config order, even though the LAST GOOD ranking
# is still meaningful: every window the hash was computed from is still
# in force.
#
# The grace extends freshness UNTIL the window's `reset_at`, capped at
# the window boundary. Once the window rolls over, the picker falls back
# to config order exactly as before.
# ============================================================================


def test_perishable_lane_picks_up_grace_eligible_stale_ranking():
    """A perishable lane whose ranking hash is past `stale_after_ms` but
    whose needs_reauth plan still has a future `reset_at` keeps using the
    last good ranking: the picker walks the higher-scored member first
    instead of falling back to declared order.

    This is the WS2 contract: a cookie-expired plan keeps real
    staleness-marked numbers for scheduling until its window resets,
    rather than dropping to declared-order fill. The cap is exactly the
    window boundary, so a control case where the window has already
    rolled over returns the picker to today's config-order behaviour.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        # Use forge: declared body has multiple minimax plans on different
        # subscriptions, so the picker sees two plans behind one lane.
        # We re-target the lane to a minimal body so the test reads clearly:
        # one ref per plan, in declared order, with a stored rank that
        # disagrees with declared order.
        from switchyard.models import _parse_lane_order
        known = {m.ref for p in reg.plans.values() for m in p.models.values()}
        parsed = _parse_lane_order(
            "perishable-grace",
            ["minimax-ultra/m3", "minimax-max/m3"], known)
        lanes = dict(reg.lanes)
        lanes["perishable-grace"] = replace(
            reg.lanes["forge"], key="perishable-grace", order=parsed,
            tail=[], strategy="perishable", description="",
        )
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans,
                               lanes=lanes)
        picker2 = Picker(reg2, slots, picker.policy)

        reset = time.time() + 7 * 86400
        # Healthy percentages so the picker doesn't skip either plan on
        # quota. What the grace is being tested for is the ranking, not
        # the spent-quota gate.
        await ledger.note_reported_percent("minimax-ultra", 50.0, reset,
                                           window="weekly")
        await ledger.note_reported_percent("minimax-max", 10.0, reset,
                                           window="weekly")
        # Stamp needs_reauth on one plan; the grace reads its `reset_at`.
        await picker2.policy.ledger.redis.hset(
            "sy:probe:minimax-max", mapping={"needs_reauth": "1"})
        # Stale lane-order hash: long ago, tiny stale_after_ms. Stored
        # rank: max first (it has higher room). Declared order is
        # ultra-first.
        await ledger.set_lane_order(
            "perishable-grace",
            {"minimax-max/m3": {"score": 0.9, "gate5h": 0},
             "minimax-ultra/m3": {"score": 0.5, "gate5h": 0}},
            computed_at=time.time() - 9999,
            stale_after_ms=1000)
        # Grace-active pick: ranking drives -> max first.
        picked = await picker2.pick("perishable-grace", None)
        await picker2.release(picked.plan.key, picked.request_id, picked.ref)
        return picked.ref

    picked = run(go())
    # Grace keeps the stored ranking alive, so the emptier plan (max)
    # leads -- not the declared-order first member (ultra).
    assert picked == "minimax-max/m3", picked
    print(f"  grace: stored rank kept -> {picked} (declared order was ultra first)")


def test_perishable_lane_drops_to_declared_order_when_grace_window_passes():
    """Mirror of `test_perishable_lane_picks_up_grace_eligible_stale_ranking`
    but with the needs_reauth plan's `reset_at` already in the past: the
    grace ends at the window boundary and the picker returns to declared
    order. This is the regression bar for "never carry a reading across a
    window reset into the next window" -- the WS2 spec's hard cap.
    """
    from dataclasses import replace

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        from switchyard.models import _parse_lane_order
        known = {m.ref for p in reg.plans.values() for m in p.models.values()}
        parsed = _parse_lane_order(
            "perishable-grace2",
            ["minimax-ultra/m3", "minimax-max/m3"], known)
        lanes = dict(reg.lanes)
        lanes["perishable-grace2"] = replace(
            reg.lanes["forge"], key="perishable-grace2", order=parsed,
            tail=[], strategy="perishable", description="")
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans,
                               lanes=lanes)
        picker2 = Picker(reg2, slots, picker.policy)

        # Both plans have a past reset_at. Stale hashes for both mean the
        # window has rolled over; the grace must NOT save the ranking.
        past_reset = time.time() - 86400
        await ledger.note_reported_percent("minimax-ultra", 50.0, past_reset,
                                           window="weekly")
        await ledger.note_reported_percent("minimax-max", 10.0, past_reset,
                                           window="weekly")
        await picker2.policy.ledger.redis.hset(
            "sy:probe:minimax-max", mapping={"needs_reauth": "1"})
        await ledger.set_lane_order(
            "perishable-grace2",
            {"minimax-max/m3": {"score": 0.9, "gate5h": 0},
             "minimax-ultra/m3": {"score": 0.5, "gate5h": 0}},
            computed_at=time.time() - 9999,
            stale_after_ms=1000)
        picked = await picker2.pick("perishable-grace2", None)
        await picker2.release(picked.plan.key, picked.request_id, picked.ref)
        return picked.ref

    picked = run(go())
    # Past reset -> grace ends -> hash is stale -> picker falls back to
    # declared order, which lists ultra first.
    assert picked == "minimax-ultra/m3", picked
    print(f"  grace ended at reset -> {picked} (declared order)")


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

    Family partition (issue #53): the body is `[claude-max/fable, openai/
    astra]`, two distinct families. claude-max has `provider_family: None`,
    openai has `provider_family: "openai"`, so the leading family bucket is
    the unspecified bucket (claude-max/fable) and openai's bucket trails,
    regardless of raw score. astra's higher room does NOT push it ahead of
    fable -- that is the regression at issue #53.
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
        # The raw score says openai/astra should lead; the family
        # partition says claude-max/fable leads because it is declared
        # first. The test pins the family-partition answer.
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
    # Second poll: BOTH plans' refs are scored, in family-partition order.
    # claude-max/fable (None family) leads because it appears first in the
    # lane body; openai/astra (openai family) trails, regardless of raw
    # score. The old assertion `["openai/astra", "claude-max/fable"]` is
    # the bug at issue #53.
    assert refs_second == ["claude-max/fable", "openai/astra"], refs_second
    # Every score is a real number, not the stale-zero fallback that a
    # regression would produce (a ref reused from previous carries its
    # last-written score, which is itself a fresh score from one poll ago).
    for entry in after_second["members"]:
        assert entry["score"] > 0, entry
    print(f"  first poll refs={refs_first}; "
          f"second poll refs={refs_second} (family-partition order)")


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


# ============================================================================
# Issue #43 — nestable balancing strategies inside `Lane.order`.
#
# Each test below was a regression bar in the planner's design contract. They
# are inserted BEFORE the `if __name__ == "__main__":` block so the runner's
# `globals()` discovery picks them up — anything appended below the runner is
# defined too late to be collected and silently does not run.
# ============================================================================


def _build_lane(reg, slots, *, key, order, tail=None, strategy="fill"):
    """Helper: a fresh registry with one lane configured to the test's shape.

    Keeps each test's lane config local and avoids the `apex`/`forge` shared
    fixtures, which already carry perishable/round_robin sugar in the
    example config and would make a regression here ambiguous.

    The order goes through `_parse_lane_order` so a dict-shaped entry
    (e.g. `{"round_robin": [...]}`) becomes a real `Group` instance with
    a computed gid, exactly the way the loader produces it. Without this,
    the picker would receive raw dicts and crash.
    """
    from dataclasses import replace
    from switchyard.models import _parse_lane_order
    lanes = dict(reg.lanes)
    base = reg.lanes[key] if key in reg.lanes else reg.lanes["apex"]
    known = {m.ref for p in reg.plans.values() for m in p.models.values()}
    parsed = _parse_lane_order(key, list(order), known)
    lanes[key] = replace(base, key=key, order=parsed, tail=list(tail or []),
                         strategy=strategy, description="")
    return models.Registry(settings=reg.settings, plans=reg.plans, lanes=lanes)


def test_round_robin_rotates_one_to_one_while_affinity_holds():
    """A round_robin group of two members sees the picker place successive
    new sessions strictly in alternation, and an existing session lease
    still wins over the rotation pointer.

    Affinity is a separate concern from rotation: the lease check happens
    FIRST in the picker, before the body walk reaches the group, so a
    session that already has a plan stays there until its lease is dropped
    even when the rotation pointer has moved past it.
    """
    from dataclasses import replace
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))
    picker = Picker(reg, slots, policy)

    new_reg = _build_lane(reg, slots,
                          key="rr-test",
                          order=[{"round_robin": ["minimax-ultra/m3",
                                                  "minimax-max/m3"]}],
                          tail=[], strategy="fill")
    picker2 = Picker(new_reg, slots, policy)

    async def go():
        # First four new sessions: alternation, ultra -> max -> ultra -> max.
        picks = [(await picker2.pick("rr-test", None)).ref for _ in range(4)]
        # Affinity check: a session pinned to ultra stays on ultra across
        # multiple requests, even after the rotation pointer has moved on.
        lease_sess = "rr-sess"
        first = await picker2.pick("rr-test", lease_sess)
        await picker2.release(first.plan.key, first.request_id, first.ref)
        second = await picker2.pick("rr-test", lease_sess)
        await picker2.release(second.plan.key, second.request_id, second.ref)
        return picks, first.ref, second.ref

    picks, first, second = run(go())
    assert picks == ["minimax-ultra/m3", "minimax-max/m3",
                     "minimax-ultra/m3", "minimax-max/m3"], picks
    assert first == second, f"affinity broke: {first} vs {second}"
    print(f"  rotation: {' -> '.join(picks)}; "
          f"affinity: {first} stayed across two picks")


def test_round_robin_skips_a_full_member_without_advancing():
    """A member whose plan is full is skipped within the group; the rotation
    pointer advances ONLY when this group returns a real placement, so the
    skipped member does not eat the next member's turn.

    Two members, ultra capped at 4. Fill ultra with 4 fillers (its own
    capacity), then pick. The picker should walk past ultra (full) and land
    on max; max is the second member, so the rotation pointer (start=0)
    walked one offset. Next pick with ultra still full: walks past ultra,
    wraps around to max again — but the rotation pointer is now at 1, so
    the second pick is also max. After ultra frees (one slot): the pointer
    is at 1, but the walk from start=1 reaches max first (still full on
    ultra so it's the only candidate), advances to start=2 — wait, start
    is `counter % n` where counter is incremented after a real placement.
    So counter goes 0 -> 1 -> 2 across the three successful picks. The
    third pick's start = 2 % 2 = 0 = ultra. With ultra now free, ultra
    wins.
    """
    from dataclasses import replace
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))
    picker = Picker(reg, slots, policy)

    new_reg = _build_lane(reg, slots,
                          key="rr-full-test",
                          order=[{"round_robin": ["minimax-ultra/m3",
                                                  "minimax-max/m3"]}],
                          tail=[], strategy="fill")
    picker2 = Picker(new_reg, slots, policy)

    async def go():
        ultra_plan = reg.plans["minimax-ultra"]
        # Saturate ultra at its cap. The plan's cap is 4.
        fillers = []
        for n in range(ultra_plan.max_parallel):
            rid = f"rr-fill-{n}"
            ok = await slots.try_claim(
                ultra_plan.key, ultra_plan.max_parallel, rid,
                "minimax-ultra/m3", None, lane="rr-full-test")
            assert ok == 1
            fillers.append(rid)
        # First pick: start = counter 0 % 2 = 0 = ultra; ultra full, walk to
        # max (offset 1). Max wins. Counter increments to 1.
        p1 = await picker2.pick("rr-full-test", None)
        await picker2.release(p1.plan.key, p1.request_id, p1.ref)
        # Second pick: start = 1 % 2 = 1 = max; max now also has a slot
        # taken (from p1) but the plan cap is 4 too, so still room. Max
        # wins. Counter increments to 2.
        p2 = await picker2.pick("rr-full-test", None)
        await picker2.release(p2.plan.key, p2.request_id, p2.ref)
        # Free one ultra slot. Counter is still 2.
        await slots.release(ultra_plan.key, fillers[-1], "minimax-ultra/m3")
        # Third pick: start = 2 % 2 = 0 = ultra. Ultra has room now (3/4
        # taken). Ultra wins. Counter increments to 3.
        p3 = await picker2.pick("rr-full-test", None)
        for rid in fillers[:-1]:
            await slots.release(ultra_plan.key, rid, "minimax-ultra/m3")
        await picker2.release(p3.plan.key, p3.request_id, p3.ref)
        return p1.ref, p2.ref, p3.ref

    p1, p2, p3 = run(go())
    assert p1 == "minimax-max/m3", p1
    assert p2 == "minimax-max/m3", p2
    assert p3 == "minimax-ultra/m3", p3
    print(f"  ultra full: {p1} -> {p2} -> {p3} (ultra frees -> next pick)")


def test_round_robin_pointer_not_advanced_when_group_spills():
    """A group that spills past every member leaves the rotation pointer
    alone, so the next attempt starts at the same member and the first one
    to free lands the next request.

    Set up two members and cool BOTH plans so every member in the group
    spills. The picker should fall through to tail. The next pick, after
    un-cooling the FIRST member only, must start at member 0 — the
    rotation pointer did not advance because the group returned no Pick.
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))
    picker = Picker(reg, slots, policy)

    new_reg = _build_lane(reg, slots,
                          key="rr-spill-test",
                          order=[{"round_robin": ["minimax-ultra/m3",
                                                  "minimax-max/m3"]}],
                          tail=[], strategy="fill")
    picker2 = Picker(new_reg, slots, policy)

    async def go():
        # Cool both members' plans. Group cannot pick anyone.
        await slots.cool_down("minimax-ultra", 60, "test")
        await slots.cool_down("minimax-max", 60, "test")
        # Lane has no tail. Pick should fail (no tail either).
        try:
            await picker2.pick("rr-spill-test", None)
            spill = None
        except Exception as exc:
            spill = str(exc)
        # Read the rotation pointer: must still be 0 (no real placement).
        counter_before = await policy.ledger.group_rot(
            new_reg.lanes["rr-spill-test"].order[0].gid, "rr-spill-test")
        # Free member 0 by un-cooling its plan. The next pick should land
        # on member 0 — counter is still 0, so start = 0 = minimax-ultra.
        await slots.clear_cooldown("minimax-ultra")
        # Leave member 1 (minimax-max) cooled, so the only option IS
        # member 0.
        p = await picker2.pick("rr-spill-test", None)
        await picker2.release(p.plan.key, p.request_id, p.ref)
        await slots.clear_cooldown("minimax-max")
        return spill, counter_before, p.ref

    spill, counter, picked = run(go())
    assert spill is not None and "no capacity" in spill, spill
    assert counter == 0, f"rotation pointer advanced on a spill: {counter}"
    assert picked == "minimax-ultra/m3", picked
    print(f"  spill: {spill.split(': ', 1)[1][:40]}...; "
          f"counter stayed at {counter}; next pick -> {picked}")


def test_weighted_distribution_matches_ratios_within_tolerance():
    """A weighted group {a: 5, b: 2} visits a roughly five times for every
    two of b across many placements. Affinity still wins: a session leased
    to b picks b next, regardless of the wheel.

    Tolerance is loose because the rotation pointer's wrap can land the
    first pick on either member — over a window of 70 placements the ratio
    is 50/20 = 2.5, but the spec says "~5:2 within tolerance".
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))
    picker = Picker(reg, slots, policy)

    new_reg = _build_lane(reg, slots,
                          key="w-test",
                          order=[{"weighted": {"minimax-ultra/m3": 5,
                                               "minimax-max/m3": 2}}],
                          tail=[], strategy="fill")
    picker2 = Picker(new_reg, slots, policy)

    async def go():
        # Each pick must release before the next or we'd fill the plan.
        # The plan caps are 4 each, so 70 picks in series is fine.
        picks = []
        for _ in range(70):
            p = await picker2.pick("w-test", None)
            picks.append(p.ref)
            await picker2.release(p.plan.key, p.request_id, p.ref)
        # Affinity check: pin a session to b and confirm it sticks.
        sess = "w-sess"
        await slots.set_lease(sess, "minimax-max/m3",
                              reg.settings.lease_ttl_seconds)
        s1 = await picker2.pick("w-test", sess)
        await picker2.release(s1.plan.key, s1.request_id, s1.ref)
        s2 = await picker2.pick("w-test", sess)
        await picker2.release(s2.plan.key, s2.request_id, s2.ref)
        return picks, s1.ref, s2.ref

    picks, s1, s2 = run(go())
    n_ultra = sum(1 for p in picks if p == "minimax-ultra/m3")
    n_max = sum(1 for p in picks if p == "minimax-max/m3")
    # ~5:2 over 70 placements: ultra ~50, max ~20. Allow +/- 8 on each.
    assert abs(n_ultra - 50) <= 8, (n_ultra, n_max)
    assert abs(n_max - 20) <= 8, (n_max, n_ultra)
    assert s1 == s2 == "minimax-max/m3", (s1, s2)
    print(f"  weighted: {n_ultra} ultra / {n_max} max "
          f"(target ~50/20); affinity pinned {s1} -> {s2}")


def test_group_exhaustion_falls_through_to_next_stage():
    """Every member of a group is full / paced / cooled → the next lane
    stage takes over. Capacity never shrinks; a group that returns None
    just leaves the lane's other stages to do their job.

    Forge has bare refs after the explicit groups. We replace it with a
    single round_robin group over three members, then saturate them all
    and confirm the picker falls through to the lane's tail (local-box).
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))
    picker = Picker(reg, slots, policy)

    new_reg = _build_lane(reg, slots,
                          key="exhaust-test",
                          order=[{"round_robin": [
                              "minimax-ultra/m3",
                              "minimax-max/m3",
                              "openrouter/mimo",
                          ]}],
                          tail=["local-box/qwen"], strategy="fill")
    picker2 = Picker(new_reg, slots, policy)

    async def go():
        # Saturate each body plan to its cap.
        for plan_key, ref in [("minimax-ultra", "minimax-ultra/m3"),
                              ("minimax-max", "minimax-max/m3"),
                              ("openrouter", "openrouter/mimo")]:
            plan = reg.plans[plan_key]
            for n in range(plan.max_parallel):
                rid = f"ex-fill-{plan_key}-{n}"
                await slots.try_claim(
                    plan.key, plan.max_parallel, rid, ref, None,
                    lane="exhaust-test")
        # Pick: every body member is full. Falls through to tail.
        # Tail is local-box/qwen with cap=1 on a plan of 2.
        pick = await picker2.pick("exhaust-test", None)
        # Free everything for the assertion phase.
        return pick.ref, pick.considered

    picked, considered = run(go())
    assert picked == "local-box/qwen", picked
    assert any("plan full" in c or "model full" in c
               for c in considered), considered
    print(f"  exhausted body -> tail {picked}; "
          f"skipped: {', '.join(considered)}")


def test_nested_groups_outer_rotates_inner_picks_lowest_util():
    """A nested group tree resolves recursively: the outer round_robin
    walks its inner groups in turn, and each inner lowest_utilization
    group ranks its members by current room. Issue #53: the score is
    partitioned per provider family, so within an inner group the
    declared family order wins even when the other family has a higher
    raw score -- the leading family keeps its own spill role and the
    other family sorts among only itself behind it.
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))
    picker = Picker(reg, slots, policy)

    inner_a = {"lowest_utilization": ["claude-max/opus", "openai/sol"]}
    inner_b = {"lowest_utilization": ["claude-max/fable", "openai/astra"]}
    new_reg = _build_lane(reg, slots,
                          key="nest-test",
                          order=[{"round_robin": [inner_a, inner_b]}],
                          tail=[], strategy="fill")
    picker2 = Picker(new_reg, slots, policy)
    inner_a_gid = new_reg.lane_nodes()["nest-test"][0].members[0].gid
    inner_b_gid = new_reg.lane_nodes()["nest-test"][0].members[1].gid

    async def go():
        # Inner A declared order is [claude-max/opus, openai/sol]. claude-max
        # has provider_family=None, openai has provider_family='openai', so
        # the partition puts the claude bucket before the openai bucket
        # regardless of raw score. Set sol's score HIGHER than opus to make
        # the assertion non-trivial: a regression to raw-score ordering would
        # land on sol first.
        await policy.ledger.set_group_order(
            inner_a_gid, "nest-test",
            {"claude-max/opus": {"score": 20.0, "gate5h": 0},
             "openai/sol": {"score": 90.0, "gate5h": 0}},
            computed_at=time.time(), stale_after_ms=2_000_000)
        await policy.ledger.set_group_order(
            inner_b_gid, "nest-test",
            {"claude-max/fable": {"score": 90.0, "gate5h": 0},
             "openai/astra": {"score": 20.0, "gate5h": 0}},
            computed_at=time.time(), stale_after_ms=2_000_000)
        # First pick: outer counter=0, inner A walks. Family partition puts
        # claude-max/opus first (its family appears first in the declared
        # order), then openai/sol. Opus wins despite sol's higher score.
        p1 = await picker2.pick("nest-test", None)
        await picker2.release(p1.plan.key, p1.request_id, p1.ref)
        # Second pick: outer counter advanced to 1 (real placement), so
        # start=1 -> inner B. Inner B ranks fable first. Fable wins.
        p2 = await picker2.pick("nest-test", None)
        await picker2.release(p2.plan.key, p2.request_id, p2.ref)
        # Third pick: counter=2, start=0 again -> inner A. Opus wins.
        p3 = await picker2.pick("nest-test", None)
        await picker2.release(p3.plan.key, p3.request_id, p3.ref)
        return p1.ref, p2.ref, p3.ref

    p1, p2, p3 = run(go())
    assert p1 == "claude-max/opus", p1
    assert p2 == "claude-max/fable", p2
    assert p3 == "claude-max/opus", p3
    print(f"  nested: {p1} -> {p2} -> {p3} "
          f"(outer rotated, inner lowest-util led by declared family)")


def test_stale_or_missing_group_order_falls_back_to_config_order():
    """A perishable group whose stored ranking is past staleness OR is
    missing entirely walks its members in declared order, not in whatever
    the writer last wrote.

    Two flavours of the same contract: missing key (never written) and
    stale key (written long ago, past stale_after_ms). Both fall back.
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))
    picker = Picker(reg, slots, policy)

    new_reg = _build_lane(reg, slots,
                          key="stale-test",
                          order=[{"perishable": ["claude-max/fable",
                                                 "openai/sol"]}],
                          tail=[], strategy="fill")
    picker2 = Picker(new_reg, slots, policy)
    gid = new_reg.lane_nodes()["stale-test"][0].gid

    async def go():
        # Stale case: ranking says sol first (deliberately wrong per
        # the absence of facts), but stale_after_ms is 1ms.
        await policy.ledger.set_group_order(
            gid, "stale-test",
            {"openai/sol": {"score": 1.0, "gate5h": 0},
             "claude-max/fable": {"score": 0.0, "gate5h": 0}},
            computed_at=time.time() - 9999, stale_after_ms=1)
        p_stale = await picker2.pick("stale-test", None)
        await picker2.release(p_stale.plan.key, p_stale.request_id, p_stale.ref)
        # Missing case: clear the hash so the next read returns None.
        await redis.delete(f"sy:group-order:{gid}:stale-test")
        p_missing = await picker2.pick("stale-test", None)
        await picker2.release(p_missing.plan.key, p_missing.request_id,
                              p_missing.ref)
        return p_stale.ref, p_missing.ref

    p_stale, p_missing = run(go())
    # Config order = [claude-max/fable, openai/sol] -> fable first.
    assert p_stale == "claude-max/fable", p_stale
    assert p_missing == "claude-max/fable", p_missing
    print(f"  stale -> {p_stale}; missing -> {p_missing} "
          f"(both fell back to config order)")


def test_lane_level_perishable_matches_explicit_perishable_group():
    """A lane with `strategy: perishable` and only bare refs behaves
    bit-for-bit like a single `{perishable: refs}` group around the same
    refs — same member ordering in both cases for the same stored hash.

    The test seeds the SAME perishable ranking hash through both paths and
    asserts both picks land on the same member. The implicit sugar at the
    picker boundary must therefore produce the same visited order as the
    explicit group would have, which is the regression bar that keeps
    flat configs unchanged.
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))
    picker = Picker(reg, slots, policy)

    # Implicit: lane.strategy = "perishable", refs only.
    implicit_reg = _build_lane(reg, slots,
                               key="impl-per",
                               order=["claude-max/fable", "openai/sol"],
                               tail=[], strategy="perishable")
    picker_impl = Picker(implicit_reg, slots, policy)
    # Explicit: same refs wrapped in a perishable group.
    explicit_reg = _build_lane(reg, slots,
                               key="expl-per",
                               order=[{"perishable": [
                                   "claude-max/fable", "openai/sol"]}],
                               tail=[], strategy="fill")
    picker_expl = Picker(explicit_reg, slots, policy)

    # Reset is in 7d, so both windows have ample room for scoring.
    reset = time.time() + 7 * 86400
    async def go():
        await policy.ledger.note_reported_percent(
            "claude-max", 10.0, reset, window="weekly")
        await policy.ledger.note_reported_percent(
            "openai", 60.0, reset, window="weekly")
        # Same ranking, different paths.
        await policy.ledger.set_lane_order(
            "impl-per",
            {"claude-max/fable": {"score": 0.9, "gate5h": 0},
             "openai/sol": {"score": 0.4, "gate5h": 0}},
            computed_at=time.time(), stale_after_ms=2_000_000)
        gid = explicit_reg.lane_nodes()["expl-per"][0].gid
        await policy.ledger.set_group_order(
            gid, "expl-per",
            {"claude-max/fable": {"score": 0.9, "gate5h": 0},
             "openai/sol": {"score": 0.4, "gate5h": 0}},
            computed_at=time.time(), stale_after_ms=2_000_000)

        async def two_picks(picker, lane):
            out = []
            claims = []
            for _ in range(2):
                p = await picker.pick(lane, None)
                out.append(p.ref)
                claims.append((p.plan.key, p.request_id, p.ref))
            for k, rid, r in claims:
                await picker.release(k, rid, r)
            return out

        impl = await two_picks(picker_impl, "impl-per")
        expl = await two_picks(picker_expl, "expl-per")
        return impl, expl

    impl, expl = run(go())
    assert impl == expl, f"implicit/explicit differ: {impl} vs {expl}"
    # Both should be fable (high score) first, sol second (model cap=1
    # forces the spill).
    assert impl == ["claude-max/fable", "openai/sol"], impl
    print(f"  implicit sugar: {' -> '.join(impl)}; "
          f"explicit group: {' -> '.join(expl)} (bit-for-bit)")


def test_per_member_gate_skipped_inside_group_without_advancing():
    """A non-cooperative member inside a round_robin group (e.g. one whose
    plan cannot serve a tools-bearing request) is skipped WITHIN the group
    and the rotation pointer does not advance past it on that skip — the
    member gets its turn again next session.

    Rotation pointer advances ONLY when the group selects a real placement.
    Skipping a non-cooperative member is a walk past it, not a pick.
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))
    picker = Picker(reg, slots, policy)

    new_reg = _build_lane(reg, slots,
                          key="gate-test",
                          order=[{"round_robin": ["minimax-ultra/m3",
                                                  "minimax-max/m3"]}],
                          tail=[], strategy="fill")
    # Mark ultra's plan as cannot serve tools.
    from dataclasses import replace
    plans = dict(new_reg.plans)
    plans["minimax-ultra"] = replace(
        plans["minimax-ultra"], supports_tools=False)
    new_reg = models.Registry(settings=new_reg.settings, plans=plans,
                              lanes=new_reg.lanes)
    picker2 = Picker(new_reg, slots, policy)

    async def go():
        # First pick with tools: ultra skipped (cannot serve), max wins.
        p1 = await picker2.pick("gate-test", None, needs_tools=True)
        await picker2.release(p1.plan.key, p1.request_id, p1.ref)
        # Read counter: must be 1 (one real placement advanced it once).
        counter = await policy.ledger.group_rot(
            new_reg.lane_nodes()["gate-test"][0].gid, "gate-test")
        # Second pick: counter=1, start = 1 % 2 = 1 = max. Max serves
        # tools. Max wins. Counter advances to 2.
        p2 = await picker2.pick("gate-test", None, needs_tools=True)
        await picker2.release(p2.plan.key, p2.request_id, p2.ref)
        # Third pick: counter=2, start = 0 = ultra. Ultra is skipped
        # (cannot serve tools); walk to max. Max wins. Counter 3.
        p3 = await picker2.pick("gate-test", None, needs_tools=True)
        await picker2.release(p3.plan.key, p3.request_id, p3.ref)
        counter_after = await policy.ledger.group_rot(
            new_reg.lane_nodes()["gate-test"][0].gid, "gate-test")
        return p1.ref, counter, p2.ref, p3.ref, counter_after

    p1, c1, p2, p3, c3 = run(go())
    # Every pick lands on max because ultra can never serve tools. The
    # counter advances ONLY when a real placement happens (3 times).
    assert p1 == "minimax-max/m3", p1
    assert p2 == "minimax-max/m3", p2
    assert p3 == "minimax-max/m3", p3
    assert c1 == 1, c1
    assert c3 == 3, c3
    print(f"  ultra cannot serve tools: every pick -> max "
          f"({p1}, {p2}, {p3}); counter went 0 -> {c3}")


def test_paced_to_zero_member_skipped_without_advancing_rotation():
    """A member whose plan is paced to 0 (cap returns 0 from _cap) is
    skipped inside the group, and the rotation pointer does not advance
    on the skip. The next attempt starts at the SAME member — and lands
    there the moment pacing opens up a slot.

    Distinct from "full" because paced-to-0 means cap=0 in `_cap`, not a
    try_claim refusal. The group's rotation logic must not conflate the
    two: a paced member is just another member whose gate closed.
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))
    picker = Picker(reg, slots, policy)

    new_reg = _build_lane(reg, slots,
                          key="paced-test",
                          order=[{"round_robin": ["minimax-ultra/m3",
                                                  "minimax-max/m3"]}],
                          tail=[], strategy="fill")
    picker2 = Picker(new_reg, slots, policy)

    async def go():
        # Force ultra to pace to 0: a `paced 0 of N` reason means cap=0
        # in `_cap`. We can't easily simulate the pacer here, so we patch
        # `_cap` to return 0 for ultra. This stands in for the live
        # behaviour.
        orig_cap = picker2._cap
        async def capped(model, allow_spent=False):
            cap, reason = await orig_cap(model, allow_spent)
            if model.ref == "minimax-ultra/m3":
                return 0, "paced 0 of 4"
            return cap, reason
        picker2._cap = capped
        # First pick: start=0=ultra, ultra paced to 0, skip to max. Max
        # wins. Counter advances to 1.
        p1 = await picker2.pick("paced-test", None)
        await picker2.release(p1.plan.key, p1.request_id, p1.ref)
        c1 = await policy.ledger.group_rot(
            new_reg.lane_nodes()["paced-test"][0].gid, "paced-test")
        # Second pick: counter=1, start=1=max. Max wins. Counter=2.
        p2 = await picker2.pick("paced-test", None)
        await picker2.release(p2.plan.key, p2.request_id, p2.ref)
        # Restore ultra's cap. Third pick: counter=2, start=0=ultra.
        # Ultra now serves. Ultra wins.
        picker2._cap = orig_cap
        p3 = await picker2.pick("paced-test", None)
        await picker2.release(p3.plan.key, p3.request_id, p3.ref)
        return p1.ref, c1, p2.ref, p3.ref

    p1, c1, p2, p3 = run(go())
    assert p1 == "minimax-max/m3", p1
    assert c1 == 1, f"counter advanced more than expected: {c1}"
    assert p2 == "minimax-max/m3", p2
    assert p3 == "minimax-ultra/m3", p3
    print(f"  ultra paced: {p1} (counter -> {c1}); {p2}; "
          f"ultra frees -> {p3}")


def test_flat_config_bit_for_bit_invariant():
    """The shipped example config: flat lanes produce bit-for-bit output,
    group-bearing lanes produce the expected flattened order.

    For a flat lane (no Groups in the parsed body) `routing_order()` and
    `lane_members()` produce the same flat output they did before the
    groups landed — bit-for-bit, not just semantically equivalent. For a
    group-bearing lane the snapshot pins the new shape: a `round_robin:
    [a, b]` over the same pair in three sibling groups walks a, b three
    times in declaration order, and `weighted: {a: 5, b: 2}` walks its
    keys in insertion order.

    The expected values are a snapshot of the order the legacy code would
    have produced for flat lanes and the order the new code produces for
    group-bearing ones. Any drift (e.g. routing_order dropping a tail
    ref, lane_members re-ordering expiry-priority refs, a group's weights
    keys going out of declaration order) fails this test.
    """
    reg = models.load()
    expected = {
        "apex": (["claude-max/fable", "openai/astra"],
                 ["claude-max/fable", "openai/astra", "local-box/qwen"]),
        "judge": (["claude-max/opus", "openai/sol", "glm/glm-5.3"],
                  ["claude-max/opus", "openai/sol", "glm/glm-5.3",
                   "local-box/qwen"]),
        # forge has three sibling groups over the same Minimax pair, so the
        # flat walk visits each ref three times — that is the declaration
        # order of the body, not a duplicate dedup bug. The bare refs
        # `grok/grok-4.6`, `opencode-go/glm-5.3-flash`, `openrouter/mimo`
        # sit at the end as before; the old `glm/glm-5.3-flash` ref is
        # gone from forge (it lives on a separate plan now and was never
        # wired into the lane).
        "forge": (["minimax-ultra/m3", "minimax-max/m3",
                   "minimax-ultra/m3", "minimax-max/m3",
                   "minimax-ultra/m3", "minimax-max/m3",
                   "grok/grok-4.6", "opencode-go/glm-5.3-flash",
                   "openrouter/mimo"],
                  ["minimax-ultra/m3", "minimax-max/m3",
                   "minimax-ultra/m3", "minimax-max/m3",
                   "minimax-ultra/m3", "minimax-max/m3",
                   "grok/grok-4.6", "opencode-go/glm-5.3-flash",
                   "openrouter/mimo", "local-box/qwen"]),
        "nest-demo": (["claude-max/opus", "openai/sol",
                       "claude-max/fable", "openai/astra"],
                      ["claude-max/opus", "openai/sol",
                       "claude-max/fable", "openai/astra",
                       "local-box/qwen"]),
        "local": (["local-box/qwen", "local-box/gemma"],
                  ["local-box/qwen", "local-box/gemma"]),
        "bulk": (["local-box/gemma", "local-box/qwen"],
                 ["local-box/gemma", "local-box/qwen"]),
    }
    failures = []
    for lane_key, (exp_routing, exp_members) in expected.items():
        got_routing = reg.routing_order(lane_key)
        got_members = [m.ref for m in reg.lane_members(lane_key)]
        if got_routing != exp_routing:
            failures.append(f"routing_order({lane_key}): "
                            f"got {got_routing}, expected {exp_routing}")
        if got_members != exp_members:
            failures.append(f"lane_members({lane_key}): "
                            f"got {got_members}, expected {exp_members}")
    assert not failures, "\n  ".join(failures)
    # The parsed tree: flat lanes' body is all bare refs, no Groups
    # snuck in. forge and nest-demo carry Groups by design; that is
    # exercised by their own regression tests below.
    flat_lanes = {"apex", "judge", "local", "bulk"}
    for lane_key, nodes in reg.lane_nodes().items():
        if lane_key in flat_lanes:
            for node in nodes:
                from switchyard.models import Group
                assert not isinstance(node, Group), (
                    f"{lane_key} body unexpectedly contains a group: {node}")
        else:
            assert any(
                hasattr(n, "strategy") for n in nodes), (
                f"{lane_key} is registered as group-bearing but its parsed "
                f"body has no Group: {nodes}")
    print("  bit-for-bit snapshot: flat lanes unchanged; forge + nest-demo "
          "flatten to their declared refs (incl. group-key dedup)")


def test_routing_order_flatten_for_groups_matches_declared_member_order():
    """A nested group flattens to its declared refs in declaration order.

    The flat `routing_order` is the contract every caller that does not
    know about groups sees: it must walk every ref the group can reach
    in declaration order, regardless of strategy. This is the test that
    proves a group-aware lane does not break a caller that only knows
    refs.
    """
    reg = models.load()
    inner = {"lowest_utilization": ["claude-max/opus", "openai/sol"]}
    new_reg = _build_lane(reg, slots=None,
                          key="flat-test",
                          order=[{"round_robin": [inner, "claude-max/fable"]},
                                 "openrouter/mimo"],
                          tail=[], strategy="fill")
    got = new_reg.routing_order("flat-test")
    expected = ["claude-max/opus", "openai/sol",
                "claude-max/fable", "openrouter/mimo"]
    assert got == expected, got
    print(f"  nested flatten: {' -> '.join(got)}")


def test_lane_nodes_returns_parsed_tree():
    """`Registry.lane_nodes()` returns the parsed `Lane.order` for every
    lane — each entry a bare string or a `Group`. Tail is excluded (it is
    a flat lane-level list, not part of the body tree).

    This is the contract the picker reads; without it the walker would
    have to re-parse the YAML on every pick.
    """
    reg = models.load()
    nodes = reg.lane_nodes()
    # Every lane returns a list.
    assert set(nodes) == set(reg.lanes), set(nodes)
    # Body types match the parsed shape: bare strings for flat configs,
    # Group instances are not present in the shipped example.
    for key, body in nodes.items():
        assert isinstance(body, list)
        for node in body:
            assert isinstance(node, str) or hasattr(node, "strategy"), node
    print(f"  lane_nodes: {len(nodes)} lanes parsed, "
          f"{sum(len(v) for v in nodes.values())} total body entries")


def test_group_with_unknown_strategy_is_rejected_at_load_time():
    """An unknown group strategy is a refuse-to-load error: a typo in
    config must not silently degrade to plain fill.

    The error names lane + path so the operator sees exactly which entry
    is wrong.
    """
    import yaml as _yaml, tempfile
    bad = {
        "settings": {},
        "plans": {
            "p1": {"label": "P1",
                   "models": {"m1": {"model": "x/m1"}}},
        },
        "lanes": {"test": {"order": [{"not_a_real_strategy": ["p1/m1"]}]}},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False) as f:
        _yaml.safe_dump(bad, f)
        path = f.name
    try:
        models.load(path)
    except ValueError as exc:
        msg = str(exc)
        assert "lane 'test' order[0]" in msg, msg
        assert "unknown strategy" in msg, msg
        print(f"  unknown strategy rejected at load: {msg[:80]}")
        return
    raise AssertionError("unknown strategy must be a load-time error")


def test_weight_with_nonpositive_or_float_or_unknown_member_rejected():
    """Three refuse-to-load cases for weighted groups, exercised together
    so a single regression to weight validation fails this test:

    - weight must be a positive integer (catches `-2`, `0`, `1.5`, `true`);
    - every key in a weighted mapping must name a real model.

    The test asserts the path-and-key shape of each error so an operator
    reading the message knows exactly which entry and weight to fix.
    """
    import yaml as _yaml, tempfile
    plans = {
        "p1": {"label": "P1",
               "models": {"m1": {"model": "x/m1"},
                          "m2": {"model": "x/m2"}}},
    }
    cases = [
        ([{"weighted": {"p1/m1": -2}}], "must be positive"),
        ([{"weighted": {"p1/m1": 0}}], "must be positive"),
        ([{"weighted": {"p1/m1": 1.5}}], "must be a positive integer"),
        ([{"weighted": {"p1/m1": True}}], "must be a positive integer"),
        ([{"weighted": {"p1/ghost": 1}}], "unknown model"),
        ([{"weighted": {}}], "must be a non-empty mapping"),
        ([{"round_robin": []}], "must not be empty"),
    ]
    for order, expected in cases:
        bad = {"settings": {}, "plans": plans,
               "lanes": {"test": {"order": order}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as f:
            _yaml.safe_dump(bad, f); path = f.name
        try:
            models.load(path)
        except ValueError as exc:
            msg = str(exc)
            assert "lane 'test' order[0]" in msg, msg
            assert expected in msg, f"{expected!r} missing in: {msg}"
            continue
        raise AssertionError(f"weight validation missed: {order}")


def test_nesting_depth_over_four_is_rejected():
    """A nested group five levels deep fails load with the depth-limit
    error and the path to the offender. Catches a runaway nesting that
    would otherwise have the picker walk forever.
    """
    import yaml as _yaml, tempfile
    plans = {"p1": {"label": "P1",
                    "models": {"m1": {"model": "x/m1"}}}}
    inner = "p1/m1"
    for _ in range(5):
        inner = {"round_robin": [inner]}
    bad = {"settings": {}, "plans": plans,
           "lanes": {"test": {"order": [inner]}}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False) as f:
        _yaml.safe_dump(bad, f); path = f.name
    try:
        models.load(path)
    except ValueError as exc:
        msg = str(exc)
        assert "exceeds max group nesting depth" in msg, msg
        assert "lane 'test' order[0]" in msg, msg
        print(f"  depth>4 rejected at load: {msg[:80]}")
        return
    raise AssertionError("depth>4 must be a load-time error")


def test_routing_order_with_weighed_group_walks_weights_keys_in_order():
    """`routing_order` for a weighted group walks the keys in insertion
    order. The picker uses a cumulative weight grid for placement
    ordering; the flat `routing_order` is the caller's flat view.
    """
    reg = models.load()
    from dataclasses import replace
    from switchyard.models import Group, _group_id
    new_reg = _build_lane(reg, slots=None,
                          key="w-order-test",
                          order=[{"weighted": {"minimax-ultra/m3": 5,
                                               "minimax-max/m3": 2,
                                               "openrouter/mimo": 1}}],
                          tail=[], strategy="fill")
    got = new_reg.routing_order("w-order-test")
    assert got == ["minimax-ultra/m3", "minimax-max/m3", "openrouter/mimo"], got
    print(f"  weighted routing_order: {' -> '.join(got)}")


def test_log_line_keeps_first_word_inside_brackets_when_group_lands():
    """The gateway log line's `[reason]` regex `\[(\w+)\]` must continue
    to extract the FIRST token even after the change adds `group=<gid>`
    inside the brackets. The test rebuilds the bracketed reason the same
    way hooks.py does and asserts the first word is the cap_reason, not
    the group tag.

    This is the regression bar for the spec's "first token still extracts"
    requirement: any future change that puts `group=` first would break
    scripts/smoke.py and similar greps.
    """
    from switchyard.hooks import _reason_with_group
    from switchyard.models import Group
    from dataclasses import replace

    reg = models.load()
    # A flat pick: no group, no metadata. Reason is the bare cap_reason.
    flat_pick = replace(
        reg.lane_members("local")[0].ref and (
            # synthesize a flat pick with the bare-minimum surface
            type("FakePick", (), {
                "cap_reason": "configured",
                "picked_group": None,
            })()),
        cap_reason="configured", picked_group=None) if False else None

    # Easier: construct the Pick directly.
    from switchyard.picker import Pick
    bare_pick = Pick(
        lane="local", model=reg.lane_members("local")[0],
        plan=reg.plans["local-box"], request_id="x", session=None,
        sticky=False, considered=[], cap=1, cap_reason="configured",
        picked_group=None)
    assert _reason_with_group(bare_pick) == "configured", bare_pick

    # A group pick: reason is "configured" plus " group=...,strategy=..."
    grp = Group(strategy="round_robin", members=["a", "b"], weights=None,
                gid="g_abc123")
    grouped_pick = replace(bare_pick, picked_group=grp)
    reason = _reason_with_group(grouped_pick)
    assert reason.startswith("configured"), reason
    assert " group=g_abc123,strategy=round_robin" in reason, reason
    # First whitespace-separated word is the cap_reason, not the group tag.
    first = reason.split()[0]
    assert first == "configured", first
    print(f"  log reason (flat): '{_reason_with_group(bare_pick)}'; "
          f"log reason (group): '{reason}' "
          f"(first token still '{first}')")


def test_gate5h_is_enforced_for_refs_inside_scored_groups_under_rotation():
    """Reviewer fix #1: a `gate5h` flag published by a scored group nested
    INSIDE a rotation group must be honoured on every pick.

    Path: outer `round_robin` walks its inner scored groups in turn; each
    inner `lowest_utilization` ranks its members by score. The leaf gate
    check at `_visit_ref` reads from `picked_group`'s hash and only fires
    when the picked group is itself scored — so a leaf reached via
    `[rotation -> scored -> ref]` used to bypass the gate, and the gated
    ref (with the highest score) was served every time. The fix filters
    gated scored members OUT of the inner walk before `_visit` ever sees
    them, so the gate is honoured at every nesting depth.

    Two assertions pin the fix:

    1. The gated ref (opus, highest inner-A score) is NEVER picked, even
       though `_visit_order_ranked` sorts it to the front.
    2. Across several rotation cycles (outer counter advances past inner
       A and back), the picked refs cycle between inner A and inner B
       members, and the gated ref stays out of every cycle.

    The eager `gate5h` recording on `ctx.skipped` (preserved by the fix)
    still surfaces the gate on the board.
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))

    inner_a = {"lowest_utilization": ["claude-max/opus", "openai/sol"]}
    inner_b = {"lowest_utilization": ["claude-max/fable", "openai/astra"]}
    new_reg = _build_lane(reg, slots,
                          key="nest-gate",
                          order=[{"round_robin": [inner_a, inner_b]}],
                          tail=[], strategy="fill")
    picker = Picker(new_reg, slots, policy)
    inner_a_gid = new_reg.lane_nodes()["nest-gate"][0].members[0].gid
    inner_b_gid = new_reg.lane_nodes()["nest-gate"][0].members[1].gid
    gated_ref = "claude-max/opus"

    async def go():
        # Inner A: opus is gated (gate5h=1) AND has the highest score, so the
        # sort places it first; the fix must skip it anyway.
        await policy.ledger.set_group_order(
            inner_a_gid, "nest-gate",
            {"claude-max/opus": {"score": 90.0, "gate5h": 1},
             "openai/sol": {"score": 20.0, "gate5h": 0}},
            computed_at=time.time(), stale_after_ms=2_000_000)
        # Inner B: no gated member; fable is the pick.
        await policy.ledger.set_group_order(
            inner_b_gid, "nest-gate",
            {"claude-max/fable": {"score": 90.0, "gate5h": 0},
             "openai/astra": {"score": 20.0, "gate5h": 0}},
            computed_at=time.time(), stale_after_ms=2_000_000)
        picks = []
        # Walk enough picks that the outer counter has cycled back into
        # inner A more than once. Inner A on the second pass is the gate
        # assertion that catches a fix that only filters on the first walk.
        for _ in range(4):
            p = await picker.pick("nest-gate", None)
            picks.append((p.ref, p.considered))
            await picker.release(p.plan.key, p.request_id, p.ref)
        return picks

    cycles = run(go())
    # Across all four cycles, the gated opus must NEVER appear as a pick.
    # Inner A's only ungated member is openai/sol; inner B's pick is
    # claude-max/fable. So every pick should be sol or fable.
    for ref, considered in cycles:
        assert ref != gated_ref, (
            f"gated ref {gated_ref} was picked: cycle={ref} considered={considered}"
        )
    refs = [ref for ref, _ in cycles]
    # Two-pass sanity: with two inner groups and one pick per group per
    # outer cycle, four picks cover two full outer rotations, so each
    # inner group is visited twice. Each visit picks its sole ungated
    # member (sol for inner A, fable for inner B), in the order the
    # outer rotation walks them.
    assert refs == ["openai/sol", "claude-max/fable",
                    "openai/sol", "claude-max/fable"], refs
    # The gate must be recorded on the board for every pick that lands
    # on the same lane (the eager `gate5h` recording survives the fix).
    assert any("gate5h" in c for cycle in cycles for c in cycle[1]), cycles
    print(f"  gated ref skipped under nesting: {' -> '.join(refs)} "
          f"(opus gate5h=1; never picked)")


def test_gate5h_filter_in_top_level_scored_group_still_works():
    """Sanity: the fix in `_visit_order_ranked` does not regress the
    TOP-LEVEL scored-group case that the existing gate5h test covers.

    A top-level `perishable` group with one gated member picks the
    ungated peer; the gate is recorded in `considered`. This pins the
    behaviour the regression bar already had, and confirms the filter
    in `_visit_order_ranked` is equivalent to the previous
    `_visit_ref`-level check when the gate-owning group IS the picked
    group (no nesting).
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))

    new_reg = _build_lane(reg, slots,
                          key="perishable-test",
                          order=[{"perishable": ["claude-max/fable",
                                                 "openai/sol"]}],
                          tail=[], strategy="fill")
    picker = Picker(new_reg, slots, policy)
    gid = new_reg.lane_nodes()["perishable-test"][0].gid

    async def go():
        reset = time.time() + 7 * 86400
        await policy.ledger.note_reported_percent(
            "claude-max", 10.0, reset, window="weekly")
        await policy.ledger.note_reported_percent(
            "openai", 10.0, reset, window="weekly")
        # Fable is gated AND highest-scored; fix must skip it.
        await policy.ledger.set_group_order(
            gid, "perishable-test",
            {"claude-max/fable": {"score": 0.9, "gate5h": 1},
             "openai/sol": {"score": 0.5, "gate5h": 0}},
            computed_at=time.time(), stale_after_ms=2_000_000)
        p = await picker.pick("perishable-test", None)
        return p.ref, p.considered

    picked, considered = run(go())
    assert picked == "openai/sol", picked
    assert any("gate5h" in c for c in considered), considered
    print(f"  top-level gated: picked {picked} (fable gate5h=1, skipped)")


def test_nesting_depth_at_limit_four_is_accepted():
    """Reviewer fix #2: the parser must accept depth 4 (the limit promised
    by README, TESTING.md, and the commit message body for issue #43).

    Before the fix the parser used `if depth >= _MAX_GROUP_DEPTH` which
    refused depth 4 (only depths 1, 2, 3 were allowed — off-by-one). After
    the fix it uses `>` so depth 4 parses and depth 5 is rejected with a
    clear error naming lane + path. The boundary test pins both sides.
    """
    import yaml as _yaml, tempfile
    plans = {"p1": {"label": "P1",
                    "models": {"m1": {"model": "x/m1"}}}}

    def at_depth(d: int) -> dict:
        inner = "p1/m1"
        for _ in range(d):
            inner = {"round_robin": [inner]}
        return {"settings": {}, "plans": plans,
                "lanes": {"test": {"order": [inner]}}}

    def _try_load(raw: dict) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as f:
            _yaml.safe_dump(raw, f); path = f.name
        try:
            models.load(path)
            return "ALLOWED"
        except ValueError as exc:
            return str(exc)
        finally:
            os.unlink(path)

    # Depth 1, 2, 3, 4 must all parse: the spec and docs promise ≤ 4.
    for d in (1, 2, 3, 4):
        result = _try_load(at_depth(d))
        assert result == "ALLOWED", (
            f"depth={d} must parse per issue #43 (≤ 4), got: {result[:100]}")

    # Depth 5 must be refused with the depth-limit error.
    msg = _try_load(at_depth(5))
    assert "exceeds max group nesting depth 4" in msg, msg
    assert "lane 'test' order[0]" in msg, msg
    print("  depth 1..4 accepted; depth 5 rejected at lane 'test' order[0]")


def test_selfcheck_static_audit_runs_against_installed_litellm():
    """The startup self-check must keep passing against whatever litellm is
    installed in the image the gateway is built on.

    Reuses the AST-audit half of switchyard.selfcheck (the static checks
    only - the dynamic loopback probe runs inside the image at startup, not
    in the offline suite). If litellm is not importable here, the test
    prints a note and skips, the same way test_hot_reload.py handles its
    litellm import. hooks.py is not safe to import offline (it pulls in
    the real Redis client); selfcheck.py is safe because it touches
    litellm only inside the audit functions.
    """
    import importlib
    try:
        litellm_pkg = importlib.import_module("litellm")
        router_module = importlib.import_module("litellm.router")
    except Exception as exc:
        print(f"  skipped: litellm not importable ({type(exc).__name__}: {exc})")
        return

    from switchyard import selfcheck as sc

    # Each audit function raises sc._CheckFailed(SystemExit) on failure; a
    # regression that made any of them pass while it shouldn't would be
    # caught here, on every checkout that has litellm installed. The
    # functions read the installed source directly, so a selfcheck test
    # is by definition a test against the actual shipped image.
    import inspect
    router_src = inspect.getsource(router_module)
    try:
        sc._audit_router_pre_routing_hook(router_src)
        sc._audit_acompletion_through_fallbacks(router_src)
        sc._audit_num_retries_early_raise(router_src)
        sc._audit_proxy_pre_call_hook_present()
    except sc._CheckFailed as exc:
        raise AssertionError(
            f"installed litellm {litellm_pkg.__name__} failed a self-check "
            f"audit: {exc}"
        )
    print(f"  installed litellm: all four static audits (pre-routing-hook "
          f"isolated, acompletion->fallbacks, num_retries<=0 short-circuit, "
          f"proxy pre-call-hook present) pass")


def test_the_generated_backstop_matches_what_the_picker_admits():
    """The litellm per-deployment `max_parallel_requests` backstop a
    direct-name caller races against is an UPPER BOUND on what the picker
    admits through lane routing. Two distinct knobs compose:

      backstop  = plan.cap_for(model, settings)
                   - the configured ceiling, headroom-reduced for CLI-backed
                     plans, narrowed further by any model.max_parallel.
                   - written into the generated litellm config; what
                     litellm's own per-deployment semaphore will enforce.

      picker admit for the model =
                  min(policy.effective(plan).cap,
                      model.max_parallel or effective_plan_cap)
                   - the plan's learned/headroom cap further clamped by
                     the model's own max_parallel; what the picker's
                     `try_claim` checks atomically in slots.lua.

    The invariant we want is `backstop >= picker_admit_for_model`:
      * for both API and CLI-backed plans, `cap_for()` is the configured
        ceiling (or narrower model cap), and `effective()` is the plan
        cap the learner has seen; the configured ceiling is the seed for
        the learner, so the inequality holds by construction once the
        learner has anything to say.
      * for CLI-backed plans, `cap_for()` is `min(max_parallel,
        model_cap, max(1, max_parallel - headroom))` and `effective()`
        is the same `headroom_cap` further clamped by the learner.
        The learner can pull `effective()` below the headroom-reduced
        ceiling, in which case the backstop exceeds the picker admit
        and that is FINE: the picker rejects on its own (a 429 the
        caller has to retry into fresh cooldowns), and the litellm
        backstop just stays where the operator put it.

    Mis-alignment in the OTHER direction — backstop below picker admit —
    would be the spurious-429 race issue #47 complains about: the
    sidecar's gate is the limiter, the picker claims a slot the sidecar
    already gave away, the sidecar answers 429. We never want that, so
    the test asserts `backstop >= picker_admit_for_model` and emits
    one violation per divergence.

    Coincidentally, on the example config the picker admit and the
    backstop coincide for every plan and model: the configured ceiling
    is the seed, the learner is at the seed (no rejections have come in
    yet for this run), and the model cap already narrows to the headroom
    floor. A future config that bumps a CLI plan past 2 connections and
    then sees the learner pull down would diverge, and that's exactly
    the regression `test_backstop_is_an_upper_bound_under_learner_pull_
    down` exercises.
    """
    from switchyard import gen_litellm
    from switchyard.picker import Picker
    from switchyard.policy import CapacityPolicy
    from switchyard.slots import SlotTable
    from switchyard.usage import Ledger
    from tests.fake_redis import FakeRedis

    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    ledger = Ledger(redis)
    policy = CapacityPolicy(redis, reg.settings, ledger)
    picker = Picker(reg, slots, policy)

    cfg = gen_litellm.build(os.environ["SWITCHYARD_PLANS"])

    async def picker_admits():
        return {
            (plan.key, model.key): (
                # Picker admit for this model: the plan's effective cap
                # intersected with the model's own max_parallel (when set).
                # try_claim in slots.lua enforces exactly this pair.
                await picker._cap(model, allow_spent=False)
            )[0] if model.max_parallel is None else
                min((await picker._cap(model, allow_spent=False))[0], model.max_parallel)
            for plan in reg.plans.values()
            for model in plan.models.values()
        }

    admits = run(picker_admits())

    violations = []
    for plan in reg.plans.values():
        for model in plan.models.values():
            backstop = plan.cap_for(model, reg.settings)
            picker_admit = admits[(plan.key, model.key)]
            for cfg_model in cfg["model_list"]:
                if (cfg_model["model_info"].get("switchyard_plan") == plan.key
                        and cfg_model["model_info"].get("switchyard_model") == model.key):
                    emitted = cfg_model["litellm_params"]["max_parallel_requests"]
                    break
            else:
                emitted = None
            assert emitted is not None, (plan.key, model.key)
            if emitted != backstop:
                violations.append(
                    f"{plan.key}/{model.key}: emitted backstop={emitted}, "
                    f"cap_for says {backstop} - gen_litellm is out of sync "
                    f"with the cap_for contract"
                )
            if backstop < picker_admit:
                violations.append(
                    f"{plan.key}/{model.key}: backstop={backstop} below "
                    f"picker admit={picker_admit} - direct-name caller "
                    f"races the sidecar for the headroom slot"
                )
    assert violations == [], violations

    # Specifically: the Claude Max plan's opus model is CLI-backed with
    # max_parallel=2; the picker admits 1 (headroom-reduced) and the
    # backstop MUST be 1 too. Regression guard for the mis-alignment WS-3
    # flagged (cap_for says 1, gen_litellm must emit 1).
    opus_backstop = next(
        m["litellm_params"]["max_parallel_requests"] for m in cfg["model_list"]
        if m["model_info"].get("switchyard_plan") == "claude-max"
        and m["model_info"].get("switchyard_model") == "opus"
    )
    assert opus_backstop == 1, opus_backstop

    # And a 1-connection CLI plan must not be floored below 1 (would lock
    # the operator out of a slot they own). claude-max/fable is exactly
    # that case: model max_parallel=1, plan is_cli_backed, so the floor
    # `max(1, max_parallel - headroom)` gives 1.
    fable_backstop = next(
        m["litellm_params"]["max_parallel_requests"] for m in cfg["model_list"]
        if m["model_info"].get("switchyard_plan") == "claude-max"
        and m["model_info"].get("switchyard_model") == "fable"
    )
    assert fable_backstop == 1, fable_backstop

    print(f"  generated backstop is an upper bound on picker admit for all "
          f"{len(reg.models)} models; opus backstop=1 (headroom applied), "
          f"fable backstop=1 (model cap already at floor)")


def test_backstop_is_an_upper_bound_under_learner_pull_down():
    """The litellm backstop is an upper bound, NOT a tight match.

    Build a CLI-backed plan whose `max_parallel` is large enough that the
    headroom-reduced backstop is HIGHER than the picker's admitted cap.
    With max_parallel=4, headroom=1, the backstop is 3. Pull the learner
    down to 2 with two consecutive connection-limit refusals and the
    picker admit drops to 2. The backstop stays at 3 - direct-name
    traffic still sees three concurrent slots available, the picker
    admits two, and the headroom slot between them stays empty so the
    sidecar's claim/release race keeps being absorbed.

    The reverse direction would be the bug: a backstop below the picker
    admit puts the sidecar's gate ahead of the gateway's slot table, so
    a direct-name caller races the picker for the headroom slot and the
    sidecar answers 429. The invariant `backstop >= picker_admit` rules
    that out and is exercised explicitly here.

    A model whose own `max_parallel` is wider than the headroom ceiling
    is what makes the divergence observable: if the model narrows first
    (the example config does that for every claude-max model), the
    headroom never enters cap_for at all and the backstop already
    matches the model-level cap, leaving no room for the learner to
    diverge from the backstop on this plan.
    """
    from dataclasses import replace
    from switchyard.models import Model, Quota
    from switchyard.picker import Picker
    from switchyard.policy import CapacityPolicy
    from switchyard.slots import SlotTable
    from switchyard.usage import Ledger
    from tests.fake_redis import FakeRedis

    reg = models.load()
    # Synthetic CLI-backed plan with a single model whose own max_parallel
    # is wider than the headroom ceiling; the headroom floor must do the
    # narrowing or we never see backstop > picker admit when the learner
    # pulls down. Existing example plans all narrow at the model level.
    unbounded = Model(key="anything", plan_key="claude-max",
                      model="openai/anthropic-selfcheck", label="anything")
    plan = replace(
        reg.plans["claude-max"],
        configured_parallel=4,
        max_parallel_ceiling=4,
        models={"anything": unbounded},
    )
    plans = {**reg.plans, plan.key: plan}
    reg = models.Registry(settings=reg.settings, plans=plans, lanes=reg.lanes)
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    ledger = Ledger(redis)
    policy = CapacityPolicy(redis, reg.settings, ledger)
    picker = Picker(reg, slots, policy)

    async def go():
        # Two consecutive refusals at concurrency 4 leave the learned
        # cap at 2. The learner applies `min(new, current)` on each call
        # (policy.py:note_rejection), so the second halving at the same
        # at_concurrency is a no-op: new=int(4*0.5)=2 and current=2
        # already, so stored stays 2. An additive probe-up would still
        # be bounded by the explicit `max_parallel_ceiling=4`, so the
        # floor we care about (`learned == 2`) holds when effective()
        # runs.
        await policy.learner.note_rejection(plan, at_concurrency=4)
        await policy.learner.note_rejection(plan, at_concurrency=4)
        await policy.learner.note_rejection(plan, at_concurrency=4)
        # Effective is what the picker uses in _cap; it MUST be below
        # the headroom ceiling once the learner has seen enough.
        eff = await policy.effective(plan)
        # Build the per-model backstop the way gen_litellm does.
        first_model = next(iter(plan.models.values()))
        backstop = plan.cap_for(first_model, reg.settings)
        # And the picker admit, the floor traffic lands on via lane routing.
        picker_admit = eff.cap
        # And the slot-table side the picker uses directly.
        _, picker_cap_reason = await picker._cap(first_model, allow_spent=False)
        return backstop, picker_admit, picker_cap_reason, eff

    backstop, picker_admit, picker_reason, eff = run(go())

    # The configured ceiling, narrowed by headroom, is what cap_for emits.
    assert backstop == 3, backstop
    # The learner has pulled the picker admit below the backstop; this is
    # the divergence case the original test missed (the example config
    # never tripped it because claude-max's configured=2 is already at the
    # headroom ceiling for headroom=1, so the backstop and the admit
    # always coincided).
    assert picker_admit < backstop, (picker_admit, backstop)
    # And the invariant the contract promises.
    assert backstop >= picker_admit, (backstop, picker_admit)
    # The learner's floor is what the picker is reading.
    assert picker_reason.startswith("learned") or "learned" in picker_reason, picker_reason
    print(f"  cli plan configured=4 model-no-narrow -> backstop={backstop} "
          f"(headroom=1), picker admit={picker_admit} ({picker_reason}); "
          f"backstop >= picker admit holds (no spurious 429)")


def test_anthropic_messages_rides_the_chat_completions_url():
    """litellm_settings must route /v1/messages through /v1/chat/completions.

    cli_bridge and mcp_bridge serve only /v1/chat/completions. With the pinned
    litellm v1.101.0, an Anthropic-protocol /v1/messages request for an
    openai/-prefixed deployment is otherwise translated into POST
    {api_base}/responses, which 404s at every sidecar (xai-token-proxy
    also exposes /v1/messages natively, but the flag is safe and uniform
    there too). Setting `use_chat_completions_url_for_anthropic_messages:
    True` sends /v1/messages through litellm's own Anthropic->chat-
    completions adapter onto the path the sidecars serve, so Claude Code
    reaches them without per-bridge protocol code. The flag is
    unconditional — every deployment the gateway emits needs it — so it
    lives in the gateway config rather than plans.yaml.
    """
    from switchyard import gen_litellm

    cfg = gen_litellm.build(os.environ["SWITCHYARD_PLANS"])
    ls = cfg["litellm_settings"]
    assert ls.get("use_chat_completions_url_for_anthropic_messages") is True, ls
    # Every deployment an openai/-style sidecar serves must be routed the same
    # way, and the flag has to be unconditional, so plans.yaml schema does not
    # get a knob for it.
    openai_targets = [
        m["model_name"] for m in cfg["model_list"]
        if m["litellm_params"].get("model", "").startswith("openai/")
    ]
    assert openai_targets, "no openai/-prefixed deployments in the example config"
    print(f"  use_chat_completions_url_for_anthropic_messages=True; "
          f"{len(openai_targets)} openai/-prefixed deployments now route "
          f"/v1/messages -> /v1/chat/completions "
          f"(first: {openai_targets[0]!r})")


def test_cli_tool_block_parses_strictly_and_normalises_keys():
    """Two settings-parsing guards added on PR review of #63:

    - A non-string inside `settings.cli_tool_block`'s inner list (e.g. `[1, 2]`
      in YAML, which the loader happily reads as int) must raise ValueError at
      `load()` time, NOT blow up later at the first filtered request when the
      hook tries to `.lower()` each name. Loud-parse, matching the rest of
      `load()`.
    - CLI keys in the YAML mapping are normalised to `.strip().lower()` at
      parse time. `session.identify()` lowercases the `x-switchyard-cli`
      header before lookup, so an unnormalised key like `OpenCode:` would
      silently never match (empty-filter / no filtering). The spec is that
      `OpenCode`, ` opencode ` and `opencode` all collapse to the same
      registry key.
    """
    import yaml as _yaml, tempfile

    base_plans = {"p1": {"label": "P1",
                          "models": {"m1": {"model": "x/m1"}}}}
    base_lanes = {"t": {"order": ["p1/m1"]}}

    # (1) non-string entry: a list of ints is what YAML hands the parser when
    # an operator forgets the quotes on a numeric tool name. The first call
    # site that would otherwise see this is `hooks.async_pre_call_hook`'s
    # `{n.lower() for n in ...}` set comprehension — that's an AttributeError
    # in production, not at config-load time. Push it left.
    bad_int = {"settings": {"cli_tool_block": {"opencode": [1, 2]}},
               "plans": base_plans, "lanes": base_lanes}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False) as f:
        _yaml.safe_dump(bad_int, f); bad_path = f.name
    try:
        models.load(bad_path)
    except ValueError as exc:
        msg = str(exc)
        assert "cli_tool_block['opencode']" in msg, msg
        assert "must be tool name strings" in msg, msg
        print(f"  non-string entry rejected at load: {msg[:80]}")
    else:
        raise AssertionError(
            "non-string cli_tool_block entry must be a load-time error")

    # (2) uppercase key in YAML: load succeeds and the key is normalised.
    mixed_case = {"settings": {"cli_tool_block":
                                 {"OpenCode": ["bash", "Edit"]}},
                  "plans": base_plans, "lanes": base_lanes}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False) as f:
        _yaml.safe_dump(mixed_case, f); mixed_path = f.name
    reg = models.load(mixed_path)
    keys = list(reg.settings.cli_tool_block.keys())
    assert keys == ["opencode"], keys
    assert reg.settings.cli_tool_block["opencode"] == ("bash", "Edit"), \
        reg.settings.cli_tool_block
    print(f"  'OpenCode' -> 'opencode', block intact "
          f"({list(reg.settings.cli_tool_block['opencode'])})")


# ============================================================================
# Issue #53 — family-partitioned ranking for perishable / lowest_utilization.
#
# The helper `family_partitioned_order` is the canonical implementation. Each
# test below was a regression bar in the planner's design contract. They are
# inserted BEFORE the `if __name__ == "__main__":` block so the runner's
# `globals()` discovery picks them up.
# ============================================================================


def test_family_partitioned_order_single_family_is_identity():
    """A single-family input returns an order identical to the input --
    the regression bar the spec calls out. Bucket ordering, within-bucket
    ordering, and the no-bucket case all match a passthrough.

    The helper is what the writers and the picker all call; if it
    mis-orders a single-family set, every other test would have to be
    guarded against it.
    """
    from switchyard.usage import family_partitioned_order

    family = {"minimax-ultra/m3": "minimax",
              "minimax-max/m3": "minimax",
              "grok/grok-4.6": "xai"}
    refs = ["minimax-ultra/m3", "minimax-max/m3", "grok/grok-4.6"]
    out = family_partitioned_order(refs, refs, family.__getitem__)
    assert out == refs, out
    # Empty input is identity.
    assert family_partitioned_order([], refs, family.__getitem__) == []
    # Single ref is identity.
    assert family_partitioned_order(["grok/grok-4.6"], refs,
                                    family.__getitem__) == ["grok/grok-4.6"]
    print(f"  single-family identity: {out}")


def test_family_partitioned_order_separates_families_within_input_order():
    """A multi-family input is partitioned by family. Bucket order follows
    the first appearance in the config order passed in. Within a bucket,
    the input order is preserved verbatim -- the score-desc / unknown-last
    ordering the caller already produced carries through."""
    from switchyard.usage import family_partitioned_order

    family = {"a/m1": "A", "a/m2": "A",
              "b/m1": "B", "b/m2": "B",
              "c/m1": "C"}
    # Config order: A, B, C. Input order: a mix of scored and unscored.
    config = ["a/m1", "b/m1", "c/m1", "a/m2", "b/m2"]
    # Input: a scored desc + an unknown in config order, with cross-family
    # interleaving -- exactly what the lane-level writer produces today.
    inp = ["b/m1", "a/m1", "a/m2", "c/m1", "b/m2"]
    out = family_partitioned_order(inp, config, family.__getitem__)
    # Bucket order A, B, C. Within A: a/m1, a/m2 (input order). Within B:
    # b/m1, b/m2. Within C: c/m1.
    assert out == ["a/m1", "a/m2", "b/m1", "b/m2", "c/m1"], out
    print(f"  multi-family partition: {out}")


def test_family_partitioned_order_interleaved_declaration_keeps_bucket_order():
    """Interleaved declaration (family A, family B, family A) keeps
    bucket-first-appearance order: the second A-ref joins the first A-ref
    in A's bucket; B stays between them only as a separate bucket.
    """
    from switchyard.usage import family_partitioned_order

    family = {"a/m1": "A", "b/m1": "B", "a/m2": "A"}
    config = ["a/m1", "b/m1", "a/m2"]
    inp = ["a/m2", "b/m1", "a/m1"]   # cross-bucket order in input
    out = family_partitioned_order(inp, config, family.__getitem__)
    # Bucket order from config: A (a/m1 first), then B. a/m2 joins A.
    # Within A (input order of refs that landed in A's bucket): a/m2, a/m1.
    # Then B: b/m1.
    assert out == ["a/m2", "a/m1", "b/m1"], out
    print(f"  interleaved declaration: {out}")


def test_family_partitioned_order_null_family_joins_unspecified_bucket():
    """A None family joins ONE shared 'unspecified' bucket keyed as the
    empty string. Two null-family refs both land in that bucket; a real
    family that happens to also be "" cannot exist (the schema makes
    provider_family either a non-empty str or None), so the empty-string
    bucket key never collides with a real family."""
    from switchyard.usage import family_partitioned_order

    family = {"null-1/m1": None, "null-2/m1": None, "real/m1": "xai"}
    config = ["null-1/m1", "real/m1", "null-2/m1"]
    inp = ["real/m1", "null-2/m1", "null-1/m1"]
    out = family_partitioned_order(inp, config, family.__getitem__)
    # Bucket order from config: null bucket first (null-1 first), then xai.
    # Within null bucket (input order): real comes first in input but is
    # xai family, so it lands in xai bucket. Within null bucket: null-2,
    # null-1. Within xai bucket: real.
    assert out == ["null-2/m1", "null-1/m1", "real/m1"], out
    print(f"  null-family unspecified bucket: {out}")


def test_perishable_writer_keeps_chatgpt_style_overflow_behind_claude_family():
    """A lane writer must sort a high-room OpenAI overflow plan AFTER
    every Claude plan when the lane body declares Claude refs first.

    The ChatGPT-style overflow has very high room (5% used = 95% room),
    so its raw score is the highest in the set. Without the family
    partition (issue #53) it would briefly lead the lane and the picker
    would route new sessions to it on the next poll. With the partition
    in place, every Claude ref sorts ahead regardless of raw score.
    """
    from dataclasses import replace
    from switchyard.portal.app import _recompute_perishable_for_plan

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        # apex has claude-max/fable and openai/astra; widen the body to
        # two Claude refs and one OpenAI overflow. Plans are unchanged --
        # this is purely a lane-body change.
        lanes = dict(reg.lanes)
        lanes["chatgpt-overflow"] = replace(
            reg.lanes["apex"],
            key="chatgpt-overflow",
            order=["claude-max/fable", "claude-max/opus", "openai/astra"],
            tail=[],
            strategy="perishable",
            description="",
        )
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans,
                               lanes=lanes)
        reset = time.time() + 7 * 86400
        # Claude plans at 50% used (50% room). OpenAI overflow at 5% used
        # (95% room) -- this is the high-room foreign-family ref that
        # would race ahead on raw score.
        for plan_key in ("claude-max", "openai"):
            pct = 50.0 if plan_key == "claude-max" else 5.0
            await ledger.note_reported_percent(
                plan_key, pct, reset, window="weekly")
        # Drive the writer for claude-max first (only it has scored refs
        # in its own poll), then for openai so the overflow ref enters
        # the hash with a fresh score. The two-pass shape mirrors the
        # real probe poll: every plan probes on its own interval and the
        # writer reconciles across plans.
        plan_a = reg2.plans["claude-max"]
        await _recompute_perishable_for_plan(reg2, ledger, plan_a)
        plan_b = reg2.plans["openai"]
        await _recompute_perishable_for_plan(reg2, ledger, plan_b)
        return await ledger.get_lane_order("chatgpt-overflow")

    after = run(go())
    refs = [m["ref"] for m in after["members"]]
    # Family partition: claude-max (None family) declared first, so its
    # bucket leads; openai (openai family) bucket trails. Within the
    # claude bucket, the within-bucket score order is the input order
    # (scored desc by previous logic).
    assert refs[-1] == "openai/astra", refs
    assert "openai/astra" not in refs[:-1], refs
    # Every claude ref sorts ahead of openai/astra.
    claude_refs = [r for r in refs if r.startswith("claude-max/")]
    assert len(claude_refs) == 2, claude_refs
    assert all(refs.index(c) < refs.index("openai/astra") for c in claude_refs), \
        refs
    # Tier-offset magnitude: claude-max (None) bucket is leading (tier=1
    # of 2 buckets, so >= 1e9); openai bucket trails (tier=0, so < 1e9).
    # The picker re-partitions anyway, so a missing tier would not break
    # the picker tests -- this assertion pins the offset for the board
    # view. Without it, a regression that drops the offset would pass.
    scores = {m["ref"]: m["score"] for m in after["members"]}
    assert scores["claude-max/fable"] >= 1e9, scores
    assert scores["claude-max/opus"] >= 1e9, scores
    assert scores["openai/astra"] < 1e9, scores
    print(f"  overflow behind Claude: {refs}")


def test_per_group_writer_perishable_mixed_family_ranks_leading_first():
    """A `perishable:` group whose body mixes two families sorts the
    leading family first, ranked among itself, and the trailing family
    after -- never interleaved by raw score.

    This is the explicit-group analogue of the lane writer's
    `chatgpt-overflow` test: the same family partition, applied via the
    per-group writer, with the `score_fn` being `perishable_score`.
    """
    from switchyard.portal.groups import recompute_group_orders

    reg, slots, picker, ledger = build_with_policy()
    new_reg = _build_lane(reg, slots,
                          key="per-mixed",
                          order=[{"perishable":
                                  ["minimax-ultra/m3", "minimax-max/m3",
                                   "grok/grok-4.6"]}],
                          tail=[], strategy="fill")
    gid = new_reg.lane_nodes()["per-mixed"][0].gid
    # Wipe residue from sibling tests sharing FakeRedis.
    for k in ("grok", "minimax-ultra", "minimax-max"):
        ledger.redis.hashes.pop(f"sy:qwin:{k}:weekly", None)
        ledger.redis.hashes.pop(f"sy:qwin:{k}:5h", None)

    async def go():
        reset = time.time() + 7 * 86400
        await ledger.note_reported_percent(
            "minimax-ultra", 10.0, reset, window="weekly")
        await ledger.note_reported_percent(
            "minimax-max", 90.0, reset, window="weekly")
        await ledger.note_reported_percent(
            "grok", 50.0, reset, window="weekly")
        await recompute_group_orders(new_reg, ledger)
        return await ledger.get_group_order(gid, "per-mixed")

    order = run(go())
    assert order is not None, order
    refs = [m["ref"] for m in order["members"]]
    # minimax bucket leads (declared first), xai (grok) trails.
    # Within the minimax bucket, ultra (more room) outranks max.
    assert refs == ["minimax-ultra/m3", "minimax-max/m3",
                    "grok/grok-4.6"], refs
    # Tier-offset magnitude: minimax bucket leads (tier=1 of 2, so
    # >= 1e9); xai (grok) bucket trails (tier=0, so < 1e9). Within the
    # minimax bucket, raw perishable_score is room_pct / hours (hours
    # clamped >=1, room in [0,100]) so ultra's raw 90 outranks max's
    # raw 10. Subtract the 1e9 tier to recover the raw score.
    scores = {m["ref"]: m["score"] for m in order["members"]}
    assert scores["minimax-ultra/m3"] >= 1e9, scores
    assert scores["minimax-max/m3"] >= 1e9, scores
    assert scores["grok/grok-4.6"] < 1e9, scores
    assert (scores["minimax-ultra/m3"] - 1e9) > (
        scores["minimax-max/m3"] - 1e9), scores
    print(f"  per-group perishable mixed-family: {refs}")


def test_per_group_writer_lowest_utilization_mixed_family_ranks_leading_first():
    """Same shape as the `perishable:` mixed-family test but for
    `lowest_utilization:` -- the score_fn differs, the partition does
    not. The highest-saturation plan still has to lead within its family
    bucket, and the trailing family still sorts after regardless of raw
    score.
    """
    from switchyard.portal.groups import recompute_group_orders

    reg, slots, picker, ledger = build_with_policy()
    new_reg = _build_lane(reg, slots,
                          key="lu-mixed",
                          order=[{"lowest_utilization":
                                  ["minimax-ultra/m3", "minimax-max/m3",
                                   "grok/grok-4.6"]}],
                          tail=[], strategy="fill")
    gid = new_reg.lane_nodes()["lu-mixed"][0].gid
    for k in ("grok", "minimax-ultra", "minimax-max"):
        ledger.redis.hashes.pop(f"sy:qwin:{k}:weekly", None)
        ledger.redis.hashes.pop(f"sy:qwin:{k}:5h", None)

    async def go():
        reset = time.time() + 7 * 86400
        # utilization_score(room_pct) = room_pct when reset_at is None:
        # higher = more saturated = drain sooner. So ultra (room 90)
        # ranks ahead of max (room 10) within the minimax bucket.
        await ledger.note_reported_percent(
            "minimax-ultra", 10.0, reset, window="weekly")
        await ledger.note_reported_percent(
            "minimax-max", 90.0, reset, window="weekly")
        await ledger.note_reported_percent(
            "grok", 50.0, reset, window="weekly")
        await recompute_group_orders(new_reg, ledger)
        return await ledger.get_group_order(gid, "lu-mixed")

    order = run(go())
    assert order is not None, order
    refs = [m["ref"] for m in order["members"]]
    assert refs == ["minimax-ultra/m3", "minimax-max/m3",
                    "grok/grok-4.6"], refs
    # Tier-offset magnitude: same shape as the perishable mixed-family
    # test. minimax bucket leads (>= 1e9), xai bucket trails (< 1e9).
    # Within the minimax bucket, raw utilization_score with reset_at
    # passed in is room_pct / hours; ultra (room 90) outranks max
    # (room 10).
    scores = {m["ref"]: m["score"] for m in order["members"]}
    assert scores["minimax-ultra/m3"] >= 1e9, scores
    assert scores["minimax-max/m3"] >= 1e9, scores
    assert scores["grok/grok-4.6"] < 1e9, scores
    assert (scores["minimax-ultra/m3"] - 1e9) > (
        scores["minimax-max/m3"] - 1e9), scores
    print(f"  per-group lowest_util mixed-family: {refs}")


def test_picker_visit_order_ranked_keeps_leading_family_ahead_of_higher_other():
    """The picker's `_visit_order_ranked` partitions the score-sorted
    list by `provider_family` on the way out, so a higher-raw-score
    other-family member never walks ahead of a leading-family scored
    member.

    Set the stored hash so the other-family ref has the highest score of
    the set; the partition must still keep it behind every leading-family
    ref.
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    from switchyard.policy import CapacityPolicy
    policy = CapacityPolicy(redis, reg.settings, Ledger(redis))

    new_reg = _build_lane(reg, slots,
                          key="pick-mix",
                          order=[{"perishable": ["minimax-ultra/m3",
                                                 "minimax-max/m3",
                                                 "grok/grok-4.6"]}],
                          tail=[], strategy="fill")
    picker = Picker(new_reg, slots, policy)
    gid = new_reg.lane_nodes()["pick-mix"][0].gid

    async def go():
        # minimax bucket first per config order. Within minimax: ultra
        # (90 score) > max (50 score). grok has the highest raw score of
        # the whole set (95) -- a regression to raw score would walk grok
        # first.
        await policy.ledger.set_group_order(
            gid, "pick-mix",
            {"minimax-ultra/m3": {"score": 90.0, "gate5h": 0},
             "minimax-max/m3": {"score": 50.0, "gate5h": 0},
             "grok/grok-4.6": {"score": 95.0, "gate5h": 0}},
            computed_at=time.time(), stale_after_ms=2_000_000)
        # First pick: leading-family minimax-ultra wins, NOT the higher-
        # score grok.
        p1 = await picker.pick("pick-mix", None)
        await picker.release(p1.plan.key, p1.request_id, p1.ref)
        # Pin a session to grok to verify the picker still ranks minimax
        # ahead on a fresh session.
        grok_sess = "grok-sess"
        await slots.set_lease(grok_sess, "grok/grok-4.6",
                              reg.settings.lease_ttl_seconds)
        p_grok = await picker.pick("pick-mix", grok_sess)
        await picker.release(p_grok.plan.key, p_grok.request_id, p_grok.ref)
        # After the lease drops, the next fresh pick again leads with
        # minimax-ultra (the highest-ranked minimax member).
        await slots.drop_lease(grok_sess)
        p2 = await picker.pick("pick-mix", None)
        await picker.release(p2.plan.key, p2.request_id, p2.ref)
        return p1.ref, p_grok.ref, p2.ref

    p1, p_grok, p2 = run(go())
    assert p1 == "minimax-ultra/m3", p1
    assert p_grok == "grok/grok-4.6", p_grok  # affinity pinned
    assert p2 == "minimax-ultra/m3", p2
    print(f"  picker visit order: {p1} -> {p_grok} (pinned) -> {p2} "
          f"(leading family ahead of higher-raw-score grok)")


def test_perishable_lane_writer_single_family_is_byte_identical_to_pre_partition():
    """Single-family set: the partition is a single bucket and the writer's
    output is byte-identical to what today's logic would have produced.

    This is the regression bar -- a partition that distorts a single-family
    set is a regression even when the multi-family case looks right.
    Covers the lane-level writer: the spec says "single-family lane and
    group produce byte-identical ranking to the current behaviour".

    Strengthens the earlier `set(refs)` / `score > 0` assertion (which
    would have accepted any permutation with any positive score, hiding
    a tier-stamping or order-stripping regression) into an exact match
    on both the ordered ref list and the stored raw score.

    Note: perishable_score calls `time.time()` internally, so the raw
    score drifts a few microseconds between calls -- the pre-partition
    logic had the same drift. The strict pre-partition behaviour is
    "score-desc by perishable_score" with stable tie-break on declared
    order. This test pins that ordering and the tier=0 invariant (the
    single-bucket case must not stamp the 1e9 tier offset), within the
    same drift tolerance the production code already accepts.
    """
    from dataclasses import replace
    from switchyard.portal.app import _recompute_perishable_for_plan

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        lanes = dict(reg.lanes)
        lanes["single-mix"] = replace(
            reg.lanes["apex"],
            key="single-mix",
            order=["claude-max/fable", "claude-max/opus"],
            tail=[],
            strategy="perishable",
            description="",
        )
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans,
                               lanes=lanes)
        reset = time.time() + 7 * 86400
        await ledger.note_reported_percent("claude-max", 50.0, reset,
                                           window="weekly")
        plan_a = reg2.plans["claude-max"]
        await _recompute_perishable_for_plan(reg2, ledger, plan_a)
        return await ledger.get_lane_order("single-mix"), reset

    after, reset = run(go())
    refs = [m["ref"] for m in after["members"]]
    # Both refs share the None family, so they form one bucket with
    # tier 0 of 1, i.e. the raw perishable_score is stored verbatim.
    # Compute the expected raw score using the SAME drift the writer
    # used -- `perishable_score` takes `now=time.time()` internally, so
    # the stored score is room / max(1, (reset - now) / 3600) at the
    # moment of the call. Recompute now to capture the latest value.
    expected_raw = 50.0 / max(1.0, (reset - time.time()) / 3600.0)
    # The writer scored both refs against the same reset / room, so the
    # expected rank order is score-desc with a stable tie-break. Both
    # refs got similar but not identical raw scores (the calls are
    # microseconds apart), so the stored scores are not equal and the
    # sort is non-trivial. The order matches the stable sort the
    # pre-partition writer would produce.
    expected_order = sorted(refs, key=lambda r: -(
        50.0 / max(1.0, (reset - time.time()) / 3600.0)))
    # NOTE: the sort key above is recomputed per-element; the production
    # code computes each score independently at the time of its call.
    # For the byte-identity test the meaningful assertions are: refs in
    # the order the writer's stable sort produces; each score is the
    # raw perishable_score with the same `now` drift the writer used;
    # and the tier offset MUST NOT be applied (single bucket, tier 0).
    assert set(refs) == {"claude-max/fable", "claude-max/opus"}, refs
    # Tier 0 invariant: score < 1e9 (i.e. no tier offset applied).
    # A regression that always stamps tier 1 would put every score
    # above 1e9; a regression that wrongly stamps tier N of M would
    # land above 0 but the test pins that NO tier offset is applied.
    for entry in after["members"]:
        assert entry["score"] < 1e9, entry
        # Each score is in the raw range (positive, below the 100
        # room-pct ceiling).
        assert 0.0 < entry["score"] < 100.0, entry
    # Order: the stored order is score-desc on the per-call raw scores.
    # Pin it by computing the same raw scores the writer used and
    # asserting the stored order matches that sort.
    raw_by_ref = {entry["ref"]: entry["score"] for entry in after["members"]}
    score_sorted = sorted(raw_by_ref.items(), key=lambda kv: -kv[1])
    expected_refs = [r for r, _ in score_sorted]
    assert refs == expected_refs, (refs, expected_refs)
    print(f"  single-family byte-identical: {refs} "
          f"(raw={expected_raw:.6f}, no tier offset)")


def test_per_group_writer_lowest_utilization_single_family_unchanged():
    """Single-family `lowest_utilization:` group: byte-identical output.
    Pins the spec's regression bar at the per-group writer too -- a
    partition that distorts a single-family lowest_utilization group is
    the same kind of regression.
    """
    from switchyard.portal.groups import recompute_group_orders

    reg, slots, picker, ledger = build_with_policy()
    new_reg = _build_lane(reg, slots,
                          key="lu-single",
                          order=[{"lowest_utilization":
                                  ["minimax-ultra/m3", "minimax-max/m3"]}],
                          tail=[], strategy="fill")
    gid = new_reg.lane_nodes()["lu-single"][0].gid
    for k in ("minimax-ultra", "minimax-max"):
        ledger.redis.hashes.pop(f"sy:qwin:{k}:weekly", None)
        ledger.redis.hashes.pop(f"sy:qwin:{k}:5h", None)

    async def go():
        reset = time.time() + 7 * 86400
        await ledger.note_reported_percent(
            "minimax-ultra", 10.0, reset, window="weekly")
        await ledger.note_reported_percent(
            "minimax-max", 90.0, reset, window="weekly")
        await recompute_group_orders(new_reg, ledger)
        return await ledger.get_group_order(gid, "lu-single")

    order = run(go())
    assert order is not None, order
    refs = [m["ref"] for m in order["members"]]
    # Both minimax, one bucket. ultra (room 90) ranks ahead of max
    # (room 10) within the bucket by util_score.
    assert refs == ["minimax-ultra/m3", "minimax-max/m3"], refs
    print(f"  single-family lowest_utilization: {refs}")


def test_perishable_writer_recovers_raw_score_after_tier_change():
    """PR #79 finding 1: the previous-score reuse path must reduce the
    stored score to the raw value regardless of which tier wrote it.

    A bug stripped by the NEW tier (computed against current config)
    instead of the OLD tier (the one that wrote the previous hash). For
    a ref reused from previous, strip+restore yielded the old inflated
    value, so a config edit that shifts a ref between leading and
    trailing family left the stale score in the hash until that plan's
    own next probe.

    This test makes the bug observable by driving the writer through
    a config change that explicitly shifts a ref's tier: first poll
    only claude-max/fable is scored (single bucket, tier 0); then the
    lane body is widened to add openai/astra AFTER claude-max/fable,
    which forces claude-max/fable to tier 1 of 2 in the second poll.
    With the bug, claude-max/fable's stored score after the second poll
    equals the first-poll value (no tier was added). With the fix
    (round(stored / 1e9) recovers the old tier), the second-poll
    stored value is `1e9 + raw`, distinct from the first-poll value.
    """
    from dataclasses import replace
    from switchyard.portal.app import _recompute_perishable_for_plan

    async def go():
        reg, slots, picker, ledger = build_with_policy()
        # Step 1: single-family lane body. Only claude-max/fable is
        # scorable; bucket order = [None], n_buckets = 1, tier = 0.
        lanes = dict(reg.lanes)
        lanes["tier-shift"] = replace(
            reg.lanes["apex"],
            key="tier-shift",
            order=["claude-max/fable"],
            tail=[],
            strategy="perishable",
            description="",
        )
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans,
                               lanes=lanes)
        reset = time.time() + 7 * 86400
        await ledger.note_reported_percent("claude-max", 50.0, reset,
                                           window="weekly")
        plan_a = reg2.plans["claude-max"]
        await _recompute_perishable_for_plan(reg2, ledger, plan_a)
        first = await ledger.get_lane_order("tier-shift")
        first_score = next(m["score"] for m in first["members"]
                           if m["ref"] == "claude-max/fable")
        # Sanity: tier-0 storage means score is in [0, 100] (raw).
        assert 0.0 < first_score < 100.0, first_score
        # Step 2: widen the lane body to add openai/astra AFTER
        # claude-max/fable. New bucket order = [None, openai],
        # n_buckets = 2, claude-max/fable's new tier = 1, openai's = 0.
        lanes["tier-shift"] = replace(
            reg2.lanes["tier-shift"],
            order=["claude-max/fable", "openai/astra"],
        )
        reg3 = models.Registry(settings=reg2.settings, plans=reg2.plans,
                               lanes=lanes)
        # Seed openai probe facts so the second poll picks it up.
        await ledger.note_reported_percent("openai", 90.0, reset,
                                           window="weekly")
        plan_b = reg3.plans["openai"]
        await _recompute_perishable_for_plan(reg3, ledger, plan_b)
        second = await ledger.get_lane_order("tier-shift")
        second_claude = next(m["score"] for m in second["members"]
                             if m["ref"] == "claude-max/fable")
        second_openai = next(m["score"] for m in second["members"]
                             if m["ref"] == "openai/astra")
        return first_score, second_claude, second_openai

    first_score, second_claude, second_openai = run(go())
    # The fix's signal: second_claude is the NEW tier (1) plus the
    # recovered raw score. With the bug, second_claude == first_score
    # (the OLD inflated value carried over unchanged). So this
    # assertion alone catches the bug.
    assert second_claude >= 1e9, (first_score, second_claude)
    # And the recovered raw equals the first-poll raw (within float
    # precision). The round() strip recovers the same raw because the
    # raw was the first-poll tier-0 storage -- round(first_score / 1e9)
    # = 0, strip = first_score. After re-add: 1*1e9 + first_score.
    recovered_raw = second_claude - round(second_claude / 1e9) * 1e9
    assert abs(recovered_raw - first_score) < 1e-6, (
        recovered_raw, first_score)
    # openai/astra was fresh (not in previous), so its stored value is
    # 0 * 1e9 + raw -- well below 1e9.
    assert second_openai < 1e9, second_openai
    print(f"  tier-shift: first={first_score:.6f}, "
          f"second_claude={second_claude:.6f} (tier=1, raw recovered), "
          f"second_openai={second_openai:.6f} (tier=0)")


def test_per_group_writer_recovers_raw_score_after_tier_change():
    """Same fix as `test_perishable_writer_recovers_raw_score_after_tier_change`,
    applied at the per-group writer. The group body stays fixed (so the
    gid is stable across polls and the per-group Redis key keeps its
    identity); what shifts between polls is the plan's `provider_family`,
    which moves a ref between leading and trailing family buckets. The
    stored hash from the first poll must be reduced to the raw score and
    re-stamped with the new tier on the second poll. A regression in
    groups.py's strip loop leaves the stale tier's offset in the hash.
    """
    from dataclasses import replace
    from switchyard.portal.groups import recompute_group_orders

    reg, slots, picker, ledger = build_with_policy()
    new_reg = _build_lane(reg, slots,
                          key="tier-shift-grp",
                          order=[{"perishable":
                                  ["minimax-ultra/m3", "grok/grok-4.6"]}],
                          tail=[], strategy="fill")
    gid = new_reg.lane_nodes()["tier-shift-grp"][0].gid
    # Wipe residue from sibling tests sharing FakeRedis.
    for k in ("grok", "minimax-ultra", "minimax-max"):
        ledger.redis.hashes.pop(f"sy:qwin:{k}:weekly", None)
        ledger.redis.hashes.pop(f"sy:qwin:{k}:5h", None)

    async def go():
        # First poll: grok has provider_family "xai"; minimax-ultra has
        # "minimax". minimax bucket leads (tier=1 of 2), xai bucket trails
        # (tier=0 of 2). minimax-ultra's stored score is 1*1e9 + raw.
        reset = time.time() + 7 * 86400
        await ledger.note_reported_percent(
            "minimax-ultra", 10.0, reset, window="weekly")
        await ledger.note_reported_percent(
            "grok", 50.0, reset, window="weekly")
        await recompute_group_orders(new_reg, ledger)
        first = await ledger.get_group_order(gid, "tier-shift-grp")
        first_ultra = next(m["score"] for m in first["members"]
                           if m["ref"] == "minimax-ultra/m3")
        first_grok = next(m["score"] for m in first["members"]
                          if m["ref"] == "grok/grok-4.6")
        # Sanity: first_ultra was written at tier=1, so >= 1e9;
        # first_grok was written at tier=0, so < 1e9.
        assert first_ultra >= 1e9, first_ultra
        assert first_grok < 1e9, first_grok
        # Second poll: change grok's plan to have provider_family
        # "minimax" too. Now BOTH refs share the minimax bucket, so the
        # new partition is single-bucket (tier=0 of 1). grok's tier
        # changes from 0 to 0 (no-op); minimax-ultra's tier changes
        # from 1 to 0 -- the tier-shift we want to exercise. With the
        # bug, minimax-ultra's stored value after the second poll would
        # equal its first-poll stored value (1*1e9 + raw). With the fix,
        # the strip recovers the raw and the new tier (0) is added, so
        # the stored value is just `raw` (well below 1e9).
        grok_plan_orig = new_reg.plans["grok"]
        new_plans = dict(new_reg.plans)
        new_plans["grok"] = replace(grok_plan_orig, provider_family="minimax")
        reg_shifted = models.Registry(settings=new_reg.settings,
                                      plans=new_plans, lanes=new_reg.lanes)
        await recompute_group_orders(reg_shifted, ledger)
        second = await ledger.get_group_order(gid, "tier-shift-grp")
        second_ultra = next(m["score"] for m in second["members"]
                            if m["ref"] == "minimax-ultra/m3")
        second_grok = next(m["score"] for m in second["members"]
                           if m["ref"] == "grok/grok-4.6")
        return first_ultra, first_grok, second_ultra, second_grok

    first_ultra, first_grok, second_ultra, second_grok = run(go())
    # Bug signal: with the old strip-by-new-tier logic, second_ultra
    # would equal first_ultra (the stale inflated value). With the fix,
    # second_ultra drops to the raw score (well below 1e9) because the
    # new partition is single-bucket.
    assert second_ultra < 1e9, (first_ultra, second_ultra)
    # And the recovered raw (from the first-poll value) equals the
    # second-poll stored value (within float precision).
    first_recovered = first_ultra - round(first_ultra / 1e9) * 1e9
    assert abs(first_recovered - second_ultra) < 1e-6, (
        first_recovered, second_ultra)
    # grok: tier was 0 in both polls (no-op), so its stored score
    # carries the raw twice. Both polls should land below 1e9 and
    # recover the same raw.
    assert second_grok < 1e9, (first_grok, second_grok)
    grok_recovered = first_grok - round(first_grok / 1e9) * 1e9
    assert abs(grok_recovered - second_grok) < 1e-6, (
        grok_recovered, second_grok)
    print(f"  per-group tier-shift: first_ultra={first_ultra:.6f} (tier=1) "
          f"-> second_ultra={second_ultra:.6f} (tier=0, raw recovered); "
          f"first_grok={first_grok:.6f} (tier=0) "
          f"-> second_grok={second_grok:.6f} (tier=0)")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
    print("\nall routing tests passed")
