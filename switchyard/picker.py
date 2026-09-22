"""Lane -> plan selection: session affinity first, then ordered fill."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from .models import Model, Plan, Registry
from .policy import CapacityPolicy
from .slots import SlotTable
from .usage import reported_is_current

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


def _perishable_visit_order(members: list, by_ref: dict,
                             order: dict | None,
                             skipped: list[str]) -> list:
    """Re-rank `members` by perishable score when the writer has a fresh one.

    Behaviour when `order` is None or stale is to leave `members` unchanged,
    so the caller's flow runs in config order -- exactly what the picker does
    for every lane without `strategy: perishable`, and bit-for-bit the same
    shape when a probe has not yet produced a ranking.

    With a fresh order the body is re-ranked: every member with a score goes
    first in score-descending order, gate5h members are filtered out (and
    listed in `skipped` so the capacity board can show why), and unscored
    members are kept in their lane_members order AFTER every scored member
    so an unknown plan never beats a known one.
    """
    if order is None:
        return members
    score_by_ref: dict[str, float] = {}
    gate5h_refs: set[str] = set()
    for entry in order["members"]:
        if entry["gate5h"]:
            gate5h_refs.add(entry["ref"])
        else:
            score_by_ref[entry["ref"]] = entry["score"]
    # Position per ref in the picker's iteration order -- which is the
    # post-`Registry.lane_members()` list (urgency-sorted body + tail), NOT
    # the YAML `order:` list. Used to stable-sort the unscored tail so an
    # unknown plan lands at its pre-perishable slot rather than drifting.
    config_pos = {m.ref: i for i, m in enumerate(members)}
    by_ref_local = {m.ref: m for m in members}
    # Drop gate5h from the iteration list first so the existing fill loop
    # below never reaches them; record the skip so the board says why.
    for ref in sorted(gate5h_refs):
        if ref in by_ref_local:
            skipped.append(f"{ref}(gate5h)")
    body = [m for m in members if m.ref not in gate5h_refs]
    scored_refs = [r for r in score_by_ref if r in by_ref_local]
    scored_refs.sort(key=lambda r: score_by_ref[r], reverse=True)
    unscored = [m for m in body if m.ref not in scored_refs]
    unscored.sort(key=lambda m: config_pos.get(m.ref, 1_000_000))
    return [by_ref_local[r] for r in scored_refs if r in by_ref_local] + unscored


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
        # Stale means UNKNOWN, not spent. Without this gate, a stale `100%`
        # from a window that has already rolled over would freeze the plan out
        # of rotation forever: the vendor's client only writes a fresh reading
        # when it actually serves a request, so a plan the picker refuses to
        # touch never gets a chance to update. Re-admitting on stale lets the
        # next attempt land, the vendor either succeed or 429, and the usual
        # classify -> quota_exhausted path self-heal from any genuine overshoot.
        if not reported_is_current(facts, plan.quota.period):
            return False
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
                   needs_tools: bool = False, pinned: bool = False,
                   exclude: frozenset[str] | None = None) -> Pick:
        rid = uuid.uuid4().hex
        members = await self._members(lane, needs_tools)
        # The `exclude` argument exists for the caller's retry re-entering
        # through async_pre_call_hook: when a previous attempt's failure
        # left a transient breaker streak or a learned cap on one member
        # of the lane, the picker can be told to skip that one member so
        # the retry lands on a peer instead of re-paying for the same
        # broken deployment. (With num_retries=0 there is no router-level
        # retry to feed this; the spill IS the caller's retry.) Refs,
        # not plans, because that is what a candidate carries and what
        # `held` below compares against. Tests also use the parameter
        # directly to drive LaneSaturated paths.
        if exclude:
            members = [m for m in members if m.ref not in exclude]
        if not members:
            detail = ("nothing in this lane can serve tool calls — every "
                      "candidate plan is marked supports_tools: false" if needs_tools
                      else "no live models (all expired or disabled)")
            raise LaneSaturated(lane, detail)

        by_ref = {m.ref: m for m in members}
        skipped: list[str] = []
        wait = (self.registry.settings.pin_wait_seconds
                if pinned and session else 0.0)

        # 1. Affinity. A session that already has a provider stays on it as
        #    long as that provider is still in the lane and has a free slot.
        #    On a re-pick the held lease may be the member that just failed;
        #    skip the pin in that case. The elif below drops the lease so
        #    the next, non-excluded pick doesn't try to honour a stale lease
        #    to the broken plan — the successful re-pick's set_lease
        #    immediately overwrites the dropped value, so the session isn't
        #    stranded on a peer (and the held plan having left the lane is
        #    handled the same way: drop, then let ordered fill land fresh).
        if session:
            held = await self.slots.get_lease(session)
            if held and held in by_ref and (not exclude or held not in exclude):
                model = by_ref[held]
                plan = self.registry.plan_of(model)
                # A mid-tool-loop follow-up inside the lease window finishes
                # where it started, spent or not: the plan still holds the
                # provider's prompt cache for this conversation and the loop's
                # quota story stays on one plan. New requests still skip the
                # plan, so it drains rather than being hammered.
                cap, reason = await self._cap(model, allow_spent=pinned)
                # When the plan's slots are all busy, a pinned follow-up waits
                # briefly for one rather than spilling to a peer. The wait
                # holds no slot — this loop parks in the gateway, and other
                # sessions keep being placed while it does — so a burst of
                # concurrent turns on the pinned plan serialises here instead
                # of bouncing retries that would all come back anyway. Past
                # the deadline the follow-up spills to a peer, the same path a
                # fresh request would take; sessions follow capacity rather
                # than blocking, and the lane only 429s when ALL of it is full.
                # No wait when there is nothing to wait FOR: a cap of 0 or an
                # active cooldown will not lift inside the deadline, so spill
                # straight away.
                cooled, _, _ = await self.slots.cooldown_state(plan.key)
                deadline = time.monotonic() + (wait if cap > 0 and not cooled else 0.0)
                while True:
                    if cap > 0 and await self.slots.try_claim(
                            plan.key, cap, rid, model.ref, model.max_parallel,
                            lane=lane) == 1:
                        await self.slots.touch_lease(
                            session, self.registry.settings.lease_ttl_seconds)
                        return Pick(lane, model, plan, rid, session, True,
                                    [], cap, reason)
                    if time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(0.5)
                # The pin is a preference with a deadline, not a guarantee.
                # The wait rides out a burst so the loop stays on the plan
                # holding the provider's prompt cache; past it the follow-up
                # places fresh down the lane, which is safe -- every bridge
                # now rebuilds a lost session from the caller's own request
                # (mcp_bridge resume_gone_session), and native plans take
                # foreign tool_call_ids as the opaque strings they are. Its
                # slots are full or it just got cooled down; re-lease wherever
                # there is room. Refusing instead does not save the cache
                # either: the caller's 429 retry lands on the peer plan anyway,
                # only after its own backoff, and while any plan in the lane
                # has a free slot the refusal is a false "no capacity". Sessions
                # follow capacity rather than blocking, and the lane only 429s
                # when ALL of it is full.
                skipped.append(f"{held}(pinned, no free slot after {wait:g}s)"
                               if pinned else f"{held}(lease unusable)")
            elif held:
                await self.slots.drop_lease(session)

        # 1b. Perishable strategy. The portal recomputes the lane order after
        # every successful probe and writes it to K_LANE_ORDER; we read it once
        # here, fall back to config order when the key is missing or stale,
        # and otherwise re-rank the body. Skipping a gate5h member here only
        # affects a fresh request -- the affinity branch above already
        # returned, so re-leasing is the only way the leased plan could land
        # back here, and that path stays unfiltered.
        lane_cfg = self.registry.lanes.get(lane)
        if lane_cfg is not None and lane_cfg.strategy == "perishable" \
                and self.policy is not None:
            order = await self.policy.ledger.get_lane_order(lane)
            members = _perishable_visit_order(members, by_ref, order, skipped)

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

    async def pick_direct(self, model: Model, session: str | None = None) -> Pick:
        """Claim a slot for a caller that named one deployment, not a lane.

        Asking for `sy.claude-max.opus` is a legitimate thing to want -- you
        want that model, not the best available one -- but it used to bypass
        SwitchYard entirely: no slot claimed, no usage recorded, no cooldown or
        quota respected, and nothing on the board. The plan's limits are the
        plan's limits however the request is addressed.

        There is no spill here by design. A lane means "the best of these"; a
        deployment means "this one", so a full or cooled plan is an honest 429
        rather than a quiet substitution the caller did not ask for.
        """
        rid = uuid.uuid4().hex
        plan = self.registry.plan_of(model)
        cap, reason = await self._cap(model)
        if cap <= 0:
            raise LaneSaturated(model.ref, f"{model.ref} unavailable: {reason}")
        result = await self.slots.try_claim(
            plan.key, cap, rid, model.ref, model.max_parallel, lane=model.ref)
        if result != 1:
            if result == 0 and self.policy is not None:
                await self.policy.learner.note_pressure(plan.key)
            raise LaneSaturated(
                model.ref, f"{model.ref}({_why(result, cap, model.max_parallel)})")
        return Pick(model.ref, model, plan, rid, session, False, [], cap, reason)

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
            # The plan's consecutive-transient-failure streak: read from Redis
            # so the board can show a 'failing' chip on its row. A streak is
            # per-plan (siblings share its quota and deserve to share the
            # warning), and one row per model on the plan reads the same value
            # — the chip is intentionally duplicated rather than reconciled.
            transient_streak = await self.slots.transient_failure_streak(plan.key)
            # The chip's alert threshold is operator-configurable, not a literal
            # — `TransientBreaker.streak_alert` is the single source of truth,
            # and the plans-table alert in collect_plans uses the same value.
            # Carrying it onto each row keeps the two surfaces in lock-step
            # when an operator raises the threshold.
            streak_alert = self.registry.settings.transient_breaker.streak_alert
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

            # Narrow the row by the model's own `max_parallel`, the same way the
            # aggregate counts this model toward its plan's reach a few lines
            # down. Without it, a model of 1 on a plan of 2 drew two slot squares
            # — the second of which it could never serve, and which the template
            # filled from the plan's in_flight counter, reading as busy capacity
            # that no traffic here could account for. The slots above this model's
            # own limit remain reachable by sibling models on the same plan, not
            # withheld from this one; the template paints them with `cap_reason`
            # so the row still says "I can use 1, the plan has 2".
            row_cap, row_cap_reason = cap, cap_reason
            if model.max_parallel is not None and model.max_parallel < cap:
                row_cap = model.max_parallel
                row_cap_reason = f"model limit {model.max_parallel}"

            # True exactly when the row's cap is narrower than the plan's width
            # solely because of this model's own max_parallel — no external
            # reason (cooldown, quota spent, pacing tail, or learner) has
            # further narrowed it. The board uses this to draw the row as its
            # own N slots with no grey "withheld" square and no "model limit"
            # tag: the row IS the plan's reach for this model, not a slice of
            # something the operator could recover.
            cap_model_owned = (
                not cooled
                and cap_reason == "configured"
                and model.max_parallel is not None
                and model.max_parallel < cap
            )

            rows.append({
                "ref": model.ref,
                "model": model.key,
                "model_label": model.display,
                "plan": plan.key,
                "plan_label": plan.label,
                "cap": row_cap,
                "cap_configured": plan.max_parallel,
                "cap_reason": row_cap_reason,
                "cap_model_owned": cap_model_owned,
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
                "transient_streak": transient_streak,
                "streak_alert": streak_alert,
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
