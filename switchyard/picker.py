"""Lane -> plan selection: session affinity first, then ordered fill."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from .models import Plan, Registry
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


class Picker:
    def __init__(self, registry: Registry, slots: SlotTable):
        self.registry = registry
        self.slots = slots

    async def pick(self, lane: str, session: str | None) -> Pick:
        rid = uuid.uuid4().hex
        members = self.registry.lane_members(lane)
        if not members:
            raise LaneSaturated(lane, "no live plans (all expired or disabled)")

        by_key = {p.key: p for p in members}
        skipped: list[str] = []

        # 1. Affinity. A session that already has a provider stays on it as
        #    long as that provider is still in the lane and has a free slot.
        if session:
            held = await self.slots.get_lease(session)
            if held and held in by_key:
                plan = by_key[held]
                if await self.slots.try_claim(plan.key, plan.max_parallel, rid) == 1:
                    await self.slots.touch_lease(session, self.registry.settings.lease_ttl_seconds)
                    return Pick(lane, plan, rid, session, True, [])
                # Its slots are full or it just got cooled down; fall through
                # and re-lease. Sessions follow capacity rather than blocking.
                skipped.append(f"{held}(lease unusable)")
            elif held:
                await self.slots.drop_lease(session)

        # 2. Ordered fill. First plan with a free slot wins, which is what
        #    makes total lane capacity the sum of the live plans' caps.
        for plan in members:
            result = await self.slots.try_claim(plan.key, plan.max_parallel, rid)
            if result == 1:
                if session:
                    await self.slots.set_lease(
                        session, plan.key, self.registry.settings.lease_ttl_seconds
                    )
                return Pick(lane, plan, rid, session, False, skipped)
            skipped.append(f"{plan.key}({'cooled' if result == -1 else 'full'})")

        raise LaneSaturated(lane, ", ".join(skipped))

    async def release(self, plan_key: str, request_id: str) -> None:
        await self.slots.release(plan_key, request_id)

    async def capacity(self, lane: str) -> dict:
        """Live picture of a lane, for the portal's capacity board."""
        rows = []
        total = live = used = 0
        for plan in self.registry.lane_members(lane):
            cooled, ttl, reason = await self.slots.cooldown_state(plan.key)
            inflight = await self.slots.in_flight(plan.key)
            tail = self.registry.is_tail(lane, plan.key)
            rows.append({
                "plan": plan.key,
                "label": plan.label,
                "cap": plan.max_parallel,
                "in_flight": inflight,
                "cooled": cooled,
                "cooldown_remaining": ttl,
                "cooldown_reason": reason,
                "tail": tail,
                "days_left": plan.days_left,
            })
            total += plan.max_parallel
            used += inflight
            if not cooled and not tail:
                live += plan.max_parallel
        return {
            "lane": lane,
            "label": self.registry.lanes[lane].label,
            "slots_configured": total,
            "slots_available_now": live,   # excludes cooled plans and the tail
            "slots_in_use": used,
            "plans": rows,
        }
