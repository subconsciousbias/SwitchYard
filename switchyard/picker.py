"""Lane -> plan selection: session affinity first, then ordered fill."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from .models import Model, Plan, Registry
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
    model: Model          # what the lane actually names
    plan: Plan            # what owns the slot, the quota and the credential
    request_id: str
    session: str | None
    sticky: bool          # True if we honoured an existing lease
    considered: list[str] # members we skipped, for the capacity board
    cap: int = 0          # the effective cap this claim was made against
    cap_reason: str = ""  # configured | learned[h14] | paced 2 of 4 ...

    @property
    def ref(self) -> str:
        return self.model.ref


def _why(result: int, plan_cap: int, model_cap: int | None) -> str:
    """Why a member was skipped, distinguishing the two limits."""
    if result == -1:
        return "cooled"
    if result == -2:
        return f"model full at {model_cap}"
    return f"plan full at {plan_cap}"


class Picker:
    def __init__(self, registry: Registry, slots: SlotTable,
                 policy: CapacityPolicy | None = None):
        self.registry = registry
        self.slots = slots
        self.policy = policy

    # Slots, cooldowns and leases are keyed by PLAN, because that is what owns
    # the connection limit. Two models on one plan therefore share it — which is
    # how `apex` and `judge` cannot between them open two connections against a
    # one-connection Claude Max plan.
    async def _cap(self, model: Model, allow_spent: bool = False) -> tuple[int, str]:
        """The PLAN's effective cap — the ceiling on total concurrency.

        A model's own `max_parallel` is a separate, narrower limit on that model
        alone, enforced by its own counter; it must not be conflated with this
        one or a plan of 2 with two models at 1 each could only run one request.
        """
        plan = self.registry.plan_of(model)
        if self.policy is None:
            return plan.max_parallel, "configured"
        # A plan the provider says is spent has no capacity, however many
        # connections it will still accept. Without this a session leased to it
        # stays leased -- affinity only re-leases when the plan has no free
        # slot -- and every turn keeps landing on a subscription with nothing
        # left, silently spending whatever overflow the provider allows.
        if not allow_spent and await self.is_spent(plan):
            return 0, "quota spent"
        cap = await self.policy.effective(plan)
        return cap.cap, cap.reason

    async def is_spent(self, plan) -> bool:
        """The provider reports the target window at 100%, and this plan is not
        allowed to spend past it (`use_extra_quota: true`)."""
        if plan.use_extra_quota or self.policy is None:
            return False
        facts = await self.policy.ledger.window_facts(plan.key, plan.quota.label)
        pct = facts.get("reported_pct_used")
        if isinstance(pct, float):
            return pct >= 100.0
        remaining = facts.get("reported_remaining")
        return isinstance(remaining, float) and remaining <= 0

    async def _members(self, lane: str, needs_tools: bool = False) -> list[Model]:
        """Lane order, minus the tail when pacing is on and minus members whose
        plan cannot serve this request at all."""
        members = self.registry.lane_members(lane)
        if self.policy is not None and not await self.policy.tail_enabled():
            members = [m for m in members if not self.registry.is_tail(lane, m.ref)]
        if needs_tools:
            # A plan marked `supports_tools: false` would silently drop the
            # caller's tools, so it is not a candidate — better to fall through
            # to one that can serve them.
            members = [m for m in members
                       if self.registry.plan_of(m).can_use_tools]
        return members

    async def pick(self, lane: str, session: str | None,
                   needs_tools: bool = False, pinned: bool = False) -> Pick:
        rid = uuid.uuid4().hex
        members = await self._members(lane, needs_tools)
        if not members:
            detail = ("nothing in this lane can serve tool calls — every "
                      "candidate plan is marked supports_tools: false" if needs_tools
                      else "no live models (all expired or disabled)")
            raise LaneSaturated(lane, detail)

        by_ref = {m.ref: m for m in members}
        skipped: list[str] = []

        # 1. Affinity. A session that already has a provider stays on it as
        #    long as that provider is still in the lane and has a free slot.
        if session:
            held = await self.slots.get_lease(session)
            if held and held in by_ref:
                model = by_ref[held]
                plan = self.registry.plan_of(model)
                # A mid-tool-loop follow-up finishes where it started, spent or
                # not. The alternative is worse than one turn of overflow: its
                # tool_call_ids exist only in that plan's bridge, so anywhere
                # else answers 400 and the caller loses work it has already
                # done. New requests still skip the plan, so it drains rather
                # than being hammered.
                cap, reason = await self._cap(model, allow_spent=pinned)
                if cap > 0 and await self.slots.try_claim(
                        plan.key, cap, rid, model.ref, model.max_parallel,
                        lane=lane) == 1:
                    await self.slots.touch_lease(session, self.registry.settings.lease_ttl_seconds)
                    return Pick(lane, model, plan, rid, session, True, [], cap, reason)
                if pinned:
                    # A mid-tool-loop follow-up cannot be spilled. Its
                    # tool_call_ids were minted by one plan's bridge and mean
                    # nothing anywhere else, so another plan would reject them
                    # outright — and even if it accepted them, it would have
                    # none of the prompt cache this conversation has been
                    # building. Better to make the caller wait for this plan.
                    raise LaneSaturated(
                        lane, f"pinned to {held} mid-tool-loop; it has no free slot")
                # Its slots are full or it just got cooled down; fall through
                # and re-lease. Sessions follow capacity rather than blocking.
                skipped.append(f"{held}(lease unusable)")
            elif held:
                await self.slots.drop_lease(session)

        # 2. Ordered fill. First plan with a free slot wins, which is what
        #    makes total lane capacity the sum of the live plans' caps.
        for model in members:
            plan = self.registry.plan_of(model)
            cap, reason = await self._cap(model)
            if cap <= 0:
                # Pacing has closed this plan for now (ahead of budget, or the
                # allowance is spent). Not an error, just no capacity.
                skipped.append(f"{model.ref}(paced to 0)")
                continue
            result = await self.slots.try_claim(
                plan.key, cap, rid, model.ref, model.max_parallel, lane=lane)
            if result == 1:
                if session:
                    await self.slots.set_lease(
                        session, model.ref, self.registry.settings.lease_ttl_seconds
                    )
                return Pick(lane, model, plan, rid, session, False, skipped, cap, reason)
            if result == 0 and self.policy is not None:
                # Demand the PLAN's cap refused. That is the signal the learner
                # needs before it will probe the limit upward. A model-level
                # refusal says nothing about the plan, so it is not pressure.
                await self.policy.learner.note_pressure(plan.key)
            skipped.append(f"{model.ref}({_why(result, cap, model.max_parallel)})")

        raise LaneSaturated(lane, ", ".join(skipped))

    async def release(self, plan_key: str, request_id: str, model_ref: str) -> None:
        await self.slots.release(plan_key, request_id, model_ref)

    async def capacity(self, lane: str) -> dict:
        """Live picture of a lane, for the portal's capacity board.

        Rows are per model, because that is what the lane names. Slots are summed
        per *plan*, because that is what owns them — two models on one plan must
        not be counted twice.
        """
        rows = []
        used = 0
        used_elsewhere = 0
        tail_on = self.policy is None or await self.policy.tail_enabled()
        counted_used: set[str] = set()
        live_members = 0
        # Per plan: how much of its limit this lane can actually reach. A plan of
        # 4 whose only member here caps itself at 2 offers 2, not 4 — summing plan
        # limits overstates a lane whose members are narrower than their plans.
        plan_caps: dict[str, int] = {}
        plan_reach: dict[str, int] = {}
        plan_total: dict[str, int] = {}

        for model in self.registry.lane_members(lane):
            plan = self.registry.plan_of(model)
            cooled, ttl, reason = await self.slots.cooldown_state(plan.key)
            inflight = await self.slots.in_flight(plan.key)
            tail = self.registry.is_tail(lane, model.ref)
            cap, cap_reason = await self._cap(model)
            model_inflight = await self.slots.in_flight_model(model.ref)
            # A model in several lanes is busy for all of them, but the traffic
            # belongs to whichever lane claimed it. Splitting the two is what
            # stops a sibling lane's work reading as this lane's consumption.
            by_lane = await self.slots.in_flight_by_lane(plan.key, model.ref)
            model_here = by_lane.get(lane, 0)
            model_elsewhere = sum(n for k, n in by_lane.items() if k and k != lane)
            model_direct = by_lane.get("", 0)
            if tail and not tail_on:
                cap = 0
                cap_reason = "tail disabled (pacing)"

            rows.append({
                "ref": model.ref,
                "model": model.key,
                "model_label": model.display,
                "plan": plan.key,
                "plan_label": plan.label,
                "cap": cap,
                "cap_configured": plan.max_parallel,
                "cap_reason": cap_reason,
                "in_flight": inflight,
                "model_in_flight": model_inflight,
                "model_in_flight_here": model_here,
                "model_in_flight_elsewhere": model_elsewhere,
                "model_in_flight_direct": model_direct,
                "lanes_sharing": [k or "direct" for k in sorted(by_lane) if k != lane],
                "model_cap": model.max_parallel,
                "cooled": cooled,
                "cooldown_remaining": ttl,
                "cooldown_reason": reason,
                "tail": tail,
                "days_left": plan.days_left,
                "shares_plan_with": [m.key for m in self.registry.siblings(model)],
                "cli_backed": plan.is_cli_backed,
            })

            plan_total[plan.key] = plan.max_parallel
            if plan.key not in counted_used:
                counted_used.add(plan.key)
                used += inflight
                plan_lanes = await self.slots.in_flight_by_lane(plan.key)
                used_elsewhere += sum(n for k, n in plan_lanes.items()
                                      if k and k != lane)
            if not cooled and not tail and cap > 0:
                live_members += 1
                plan_caps[plan.key] = cap
                reach = model.max_parallel if model.max_parallel is not None else cap
                plan_reach[plan.key] = plan_reach.get(plan.key, 0) + reach

        total = sum(plan_total.values())
        live = sum(min(plan_caps[k], plan_reach[k]) for k in plan_caps)

        return {
            "lane": lane,
            "label": self.registry.lanes[lane].label,
            "slots_configured": total,
            "slots_available_now": live,   # excludes cooled plans and the tail
            "slots_in_use": used,
            # Of the busy slots this lane can reach, how many are its own work
            # versus a sibling lane's. They still cost real capacity either way,
            # which is why slots_available_now counts both.
            "slots_in_use_here": max(0, used - used_elsewhere),
            "slots_in_use_elsewhere": used_elsewhere,
            "tail_only": live_members == 0 and bool(rows),
            "plans": rows,
        }
