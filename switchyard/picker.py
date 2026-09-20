"""Lane -> plan selection: session affinity first, then ordered fill."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from .models import Plan, Registry
from .policy import CapacityPolicy
from .slots import SlotTable

log = logging.getLogger("switchyard.picker")


class LaneSaturated(Exception):
    """Every plan in the lane is at capacity or cooled down."""

    def __init__(self, lane: str, detail: str):
        self.lane = lane
        super().__init__(f"lane '{lane}' has no capacity: {detail}")


@dataclass
class Pick:
    lane: str
    plan: Plan
    request_id: str
    session: str | None
    sticky: bool          # True if we honoured an existing lease
    considered: list[str] # plans we skipped, for the capacity board
    cap: int = 0          # the effective cap this claim was made against
    cap_reason: str = ""  # configured | learned[h14] | paced 2 of 4 ...


class Picker:
    def __init__(self, registry: Registry, slots: SlotTable,
                 policy: CapacityPolicy | None = None):
        self.registry = registry
        self.slots = slots
        self.policy = policy

    # Slots, cooldowns and leases are keyed by *subscription*, not plan, so two
    # model tiers on one Claude Max plan cannot open two connections between
    # them. For most plans the subscription is the plan itself.
    async def _cap(self, plan: Plan) -> tuple[int, str]:
        """The cap to claim against: learned, then paced, else configured."""
        if self.policy is None:
            return plan.max_parallel, "configured"
        cap = await self.policy.effective(plan)
        return cap.cap, cap.reason

    async def _members(self, lane: str, needs_tools: bool = False) -> list[Plan]:
        """Lane order, minus the tail when pacing is on and minus plans that
        cannot serve this request at all."""
        members = self.registry.lane_members(lane)
        if self.policy is not None and not await self.policy.tail_enabled():
            members = [p for p in members if not self.registry.is_tail(lane, p.key)]
        if needs_tools:
            # A CLI-backed plan would silently drop the caller's tools, so it is
            # not a candidate — better to fall through to a plan that can.
            members = [p for p in members if p.can_use_tools]
        return members

    async def pick(self, lane: str, session: str | None,
                   needs_tools: bool = False) -> Pick:
        rid = uuid.uuid4().hex
        members = await self._members(lane, needs_tools)
        if not members:
            detail = ("no plan in this lane can serve tool calls — every "
                      "candidate is CLI-backed" if needs_tools
                      else "no live plans (all expired or disabled)")
            raise LaneSaturated(lane, detail)

        by_key = {p.key: p for p in members}
        skipped: list[str] = []

        # 1. Affinity. A session that already has a provider stays on it as
        #    long as that provider is still in the lane and has a free slot.
        if session:
            held = await self.slots.get_lease(session)
            if held and held in by_key:
                plan = by_key[held]
                cap, reason = await self._cap(plan)
                if cap > 0 and await self.slots.try_claim(plan.subscription, cap, rid) == 1:
                    await self.slots.touch_lease(session, self.registry.settings.lease_ttl_seconds)
                    return Pick(lane, plan, rid, session, True, [], cap, reason)
                # Its slots are full or it just got cooled down; fall through
                # and re-lease. Sessions follow capacity rather than blocking.
                skipped.append(f"{held}(lease unusable)")
            elif held:
                await self.slots.drop_lease(session)

        # 2. Ordered fill. First plan with a free slot wins, which is what
        #    makes total lane capacity the sum of the live plans' caps.
        for plan in members:
            cap, reason = await self._cap(plan)
            if cap <= 0:
                # Pacing has closed this plan for now (ahead of budget, or the
                # allowance is spent). Not an error, just no capacity.
                skipped.append(f"{plan.key}(paced to 0)")
                continue
            result = await self.slots.try_claim(plan.subscription, cap, rid)
            if result == 1:
                if session:
                    await self.slots.set_lease(
                        session, plan.key, self.registry.settings.lease_ttl_seconds
                    )
                return Pick(lane, plan, rid, session, False, skipped, cap, reason)
            if result == 0 and self.policy is not None:
                # Demand the cap refused. That is the signal the learner needs
                # before it will probe the limit upward.
                await self.policy.learner.note_pressure(plan.subscription)
            skipped.append(f"{plan.key}({'cooled' if result == -1 else f'full at {cap}'})")

        raise LaneSaturated(lane, ", ".join(skipped))

    async def release(self, plan_key: str, request_id: str) -> None:
        await self.slots.release(self.registry.subscription_of(plan_key), request_id)

    async def capacity(self, lane: str) -> dict:
        """Live picture of a lane, for the portal's capacity board."""
        rows = []
        total = live = used = 0
        tail_on = self.policy is None or await self.policy.tail_enabled()
        # Slots belong to a subscription, so two plans sharing one must be
        # counted once. Summing per plan overstated shared capacity.
        counted_live: set[str] = set()
        counted_total: set[str] = set()
        counted_used: set[str] = set()
        live_members = 0
        for plan in self.registry.lane_members(lane):
            cooled, ttl, reason = await self.slots.cooldown_state(plan.subscription)
            inflight = await self.slots.in_flight(plan.subscription)
            tail = self.registry.is_tail(lane, plan.key)
            cap, cap_reason = await self._cap(plan)
            if tail and not tail_on:
                cap = 0
                cap_reason = "tail disabled (pacing)"
            rows.append({
                "plan": plan.key,
                "label": plan.label,
                "cap": cap,
                "cap_configured": plan.max_parallel,
                "cap_reason": cap_reason,
                "in_flight": inflight,
                "cooled": cooled,
                "cooldown_remaining": ttl,
                "cooldown_reason": reason,
                "tail": tail,
                "days_left": plan.days_left,
                "shares_with": [p.key for p in self.registry.siblings(plan)],
            })
            sub = plan.subscription
            if sub not in counted_total:
                counted_total.add(sub)
                total += plan.max_parallel
            if sub not in counted_used:
                counted_used.add(sub)
                used += inflight
            # The headline number is capacity you can actually rely on now:
            # effective caps, excluding cooled plans and the emergency tail.
            if not cooled and not tail:
                live_members += 1
                if sub not in counted_live:
                    counted_live.add(sub)
                    live += cap
        return {
            "lane": lane,
            "label": self.registry.lanes[lane].label,
            "slots_configured": total,
            "slots_available_now": live,   # excludes cooled plans and the tail
            "slots_in_use": used,
            # BUG 2 was showing apex as 0/0: its only live plan is the emergency
            # tail, which is excluded from the headline. Say so rather than
            # implying the lane has no capacity at all.
            "tail_only": live_members == 0 and bool(rows),
            "plans": rows,
        }
