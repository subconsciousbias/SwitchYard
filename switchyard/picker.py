"""Lane -> plan selection: session affinity first, then ordered fill."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .models import Group, Model, Plan, Registry
from .policy import CapacityPolicy
from .slots import SlotTable
from .usage import (
    family_partitioned_order,
    perishable_score,
    reported_is_current,
    utilization_score,
)

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
    # Routing context that survived the visit — surfaced to hooks.py so the
    # log line can stamp group/strategy metadata onto the bracketed reason
    # without re-walking the tree. None when the lane is flat (no group) or
    # the lane-level `strategy: perishable` sugar picked from the legacy
    # K_LANE_ORDER hash. Group is the Group object (with .gid and .strategy);
    # for the implicit perishable sugar we synthesise a Group so the log line
    # stays uniform.
    picked_group: Group | None = None

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


# A context object threaded through every _visit call. Carries everything the
# recursion needs without polluting the picker method's signature, and lets
# `_visit_ref` (a Ref) and `_visit_group` (a Group) share state cleanly.
@dataclass
class _VisitCtx:
    lane: str
    rid: str
    session: str | None
    pinned: bool
    exclude: frozenset[str]
    skipped: list[str] = field(default_factory=list)
    needs_tools: bool = False
    wait: float = 0.0


def _member_refs(node: Any) -> list[str]:
    """Refs reachable from a node in declaration order.

    The picker uses this for a group's member list — bare refs and weighted
    keys are passed straight back; nested groups recurse depth-first.
    """
    if isinstance(node, Group):
        if node.weights is not None:
            return list(node.weights.keys())
        out: list[str] = []
        for m in node.members:
            out.extend(_member_refs(m))
        return out
    return [node]


def _first_leaf(node: Any) -> str | None:
    """The first ref reachable from `node` in declaration order.

    Used by the per-group scorer to look up a member's score in the
    group-scored hash. For a nested group, returns the first leaf ref of
    its members; for a bare ref, returns that ref; for an empty group,
    returns None.
    """
    if isinstance(node, Group):
        if node.weights is not None:
            return next(iter(node.weights), None)
        for m in node.members:
            leaf = _first_leaf(m)
            if leaf is not None:
                return leaf
        return None
    return node


def _all_leaves(node: Any) -> list[str]:
    """Every ref reachable from `node` in declaration order. Used to record
    gate5h skips for nested groups so the board sees the gate on every
    inner member, not just the first."""
    return _member_refs(node)


class Picker:
    def __init__(self, registry: Registry, slots: SlotTable,
                 policy: CapacityPolicy | None = None):
        self.registry = registry
        self.slots = slots
        self.policy = policy

    # -- shared capacity primitives (unchanged shape) ---------------------
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
            members = [m for m in members
                       if self.registry.plan_of(m).can_use_tools]
        return members

    # -- body for the picker ----------------------------------------------
    # The parsed tree `Registry.lane_nodes()[lane]` is what the picker walks.
    # For a flat config (refs only, `strategy: fill`) the body is the refs as
    # written. For a flat config with `strategy: perishable` the legacy
    # behaviour is preserved by wrapping the refs in an implicit perishable
    # group whose ranking is read from K_LANE_ORDER (the existing key). The
    # implicit wrap happens here, not in the parser, so flat configs parse
    # unchanged.
    def _effective_body(self, lane: str) -> tuple[list[Any], Group | None]:
        lane_cfg = self.registry.lanes.get(lane)
        if lane_cfg is None:
            return [], None
        nodes = self.registry.lane_nodes()[lane]
        has_groups = any(isinstance(n, Group) for n in nodes)
        if not has_groups and lane_cfg.strategy == "perishable":
            # Implicit perishable sugar. The gid is derived the same way as
            # for explicit groups so the same hash key shape (gid, lane) is
            # used if anyone later writes a per-group writer for it; today
            # the writer still uses K_LANE_ORDER and the picker reads that
            # key below when it sees the implicit gid.
            from .models import _group_id
            implicit = Group(
                strategy="perishable",
                members=list(nodes),
                weights=None,
                gid=_group_id(lane, "perishable", _member_refs_of(nodes)),
            )
            return [implicit], implicit
        return nodes, None

    # -- recursive visit ---------------------------------------------------
    async def _visit(self, node: Any, ctx: _VisitCtx,
                     picked_group: Group | None = None) -> Pick | None:
        if isinstance(node, Group):
            # The outermost group is what the log line / metadata surface.
            # Nested groups inherit the caller's `picked_group`; a top-level
            # group (the lane's body is a list of groups) IS the outermost
            # group, so when `picked_group` is None it becomes this node.
            effective = picked_group if picked_group is not None else node
            return await self._visit_group(node, ctx, picked_group=effective)
        return await self._visit_ref(node, ctx, picked_group=picked_group)

    async def _visit_ref(self, ref: str, ctx: _VisitCtx,
                         picked_group: Group | None) -> Pick | None:
        """Try to claim a slot for one ref. All per-member gates apply here:
        exclude, supports_tools (filtered upstream), cooldown, spent-quota,
        supports_tools (re-checked for safety), and (when called from inside a
        perishable / lowest_utilization group) the 5h gate."""
        if ctx.exclude and ref in ctx.exclude:
            return None
        model = self.registry.model(ref)
        if model is None:
            ctx.skipped.append(f"{ref}(unknown)")
            return None
        plan = self.registry.plan_of(model)
        if ctx.needs_tools and not plan.can_use_tools:
            return None
        # Cooldown gate: a plan under an active cooldown cannot serve. Note
        # the same way `_members` does NOT — this is intentional: a ref nested
        # inside a Group that the body iteration would have skipped anyway
        # gets the same skip reason whether we are 1 or 4 levels deep.
        cooled, _, reason = await self.slots.cooldown_state(plan.key)
        if cooled:
            ctx.skipped.append(f"{ref}(cooled: {reason})")
            return None
        # Drain gate: a plan with the drain flag set refuses to be picked.
        # The picker is safe regardless of apply.sh's order of operations;
        # the affinity path has its own is_draining check (see _affinity),
        # and this body-walk gate catches every NEW request so falling
        # through to the sibling is the only path a drained ref can take.
        if await self.slots.is_draining(plan.key):
            ctx.skipped.append(f"{ref}(draining)")
            return None
        # 5h gate: a perishable or lowest_utilization group may have gated this
        # ref on the 5h constraint window. The flag is read from the SAME hash
        # the ranking lives in, so a missing/stale hash means "no gate", which
        # is the right default — the picker falls back to config order. The
        # `picked_group` here is the OUTERMOST group, not the one currently
        # being walked: a ref reached via a nested group still wants the
        # outermost scored group's gate, because the writer publishes one
        # gate per leaf ref, not per nested member.
        if picked_group is not None and picked_group.strategy in (
                "perishable", "lowest_utilization"):
            order = await self._read_group_order(picked_group, ctx.lane)
            if order is not None:
                gate_refs = {m["ref"] for m in order["members"] if m["gate5h"]}
                if ref in gate_refs:
                    ctx.skipped.append(f"{ref}(gate5h)")
                    return None
        # The cap check is the same one `_members` does for the body — a
        # pacing-closed plan or a spent plan contributes zero slots.
        cap, cap_reason = await self._cap(model)
        if cap <= 0:
            ctx.skipped.append(f"{ref}({cap_reason})")
            return None
        result = await self.slots.try_claim(
            plan.key, cap, ctx.rid, model.ref, model.max_parallel, lane=ctx.lane)
        if result != 1:
            if result == 0 and self.policy is not None:
                await self.policy.learner.note_pressure(plan.key)
            ctx.skipped.append(f"{ref}({_why(result, cap, model.max_parallel)})")
            return None
        if ctx.session:
            await self.slots.set_lease(
                ctx.session, model.ref,
                self.registry.settings.lease_ttl_seconds, plan.key)
        return Pick(ctx.lane, model, plan, ctx.rid, ctx.session, False,
                    list(ctx.skipped), cap, cap_reason, picked_group)

    async def _visit_group(self, group: Group, ctx: _VisitCtx,
                           picked_group: Group | None) -> Pick | None:
        """Walk a Group's members per its strategy and return the first Pick
        that succeeds (or None if every member spills).

        A successful placement advances the rotation pointer (round_robin /
        weighted) so the next pick starts at the next member. A group that
        spills past every member leaves the pointer alone — successive calls
        see the same starting index, so a member whose slot frees next will
        land the next request rather than skipping it.
        """
        # Weighted uses a parallel members list (the keys) instead of the
        # nested members field. Round_robin / lowest_utilization / perishable
        # use `members`.
        if group.weights is not None:
            member_refs = list(group.weights.keys())
            weights = group.weights
        else:
            member_refs = _member_refs(group)
            weights = None

        if not member_refs:
            return None

        if group.strategy == "round_robin":
            order = await self._visit_order_round_robin(
                group, member_refs, ctx, picked_group=picked_group)
        elif group.strategy == "weighted":
            order = await self._visit_order_weighted(
                group, member_refs, weights, ctx, picked_group=picked_group)
        elif group.strategy == "lowest_utilization":
            order = await self._visit_order_ranked(
                group, member_refs, ctx, picked_group=picked_group,
                score_fn=utilization_score)
        elif group.strategy == "perishable":
            order = await self._visit_order_ranked(
                group, member_refs, ctx, picked_group=picked_group,
                score_fn=perishable_score)
        else:
            # Defensive: parser already rejects unknown strategies, but a
            # runtime-loaded registry from another path might not. Treat as
            # round_robin so we never silently lose requests.
            order = await self._visit_order_round_robin(
                group, member_refs, ctx, picked_group=picked_group)

        if order is None:
            return None
        pick = order["pick"]
        if pick is not None and self.policy is not None:
            # Advance the rotation pointer ONLY on a real placement, and
            # ONLY for strategies that USE the pointer: round_robin and
            # weighted. Perishable / lowest_utilization walk a score hash
            # and never read the pointer, so advancing it would just be
            # wasted work (and on a quiet day a phantom counter could
            # decay mid-decision). A group that spilled (every member
            # exhausted or gated) leaves the pointer where it was, so the
            # next attempt starts at the same member.
            if group.strategy in ("round_robin", "weighted"):
                await self.policy.ledger.bump_group_rot(group.gid, ctx.lane)
        return pick

    async def _visit_order_round_robin(self, group: Group, refs: list[str],
                                       ctx: _VisitCtx,
                                       picked_group: Group | None) -> dict | None:
        members = list(group.members)
        n = len(members)
        if n == 0:
            return {"pick": None}
        start = 0
        if self.policy is not None:
            start = await self.policy.ledger.group_rot(group.gid, ctx.lane) % n
        # Walk clockwise from `start`, wrapping. Members are walked via
        # `_visit` so a nested Group inside the round_robin recurses into
        # its own strategy; a bare Ref goes straight to claim. The
        # `picked_group` on the returned Pick stays the OUTERMOST group (the
        # one passed in from the lane body) so the log line and metadata
        # show the routing context the operator configured.
        for offset in range(n):
            member = members[(start + offset) % n]
            pick = await self._visit(member, ctx, picked_group=picked_group)
            if pick is not None:
                return {"pick": pick}
        return {"pick": None}

    async def _visit_order_weighted(self, group: Group, refs: list[str],
                                    weights: dict[str, int],
                                    ctx: _VisitCtx,
                                    picked_group: Group | None) -> dict | None:
        # Weighted groups hold bare refs only — nested groups are not
        # members of a weighted group, by parser rule (the weighted mapping
        # must ref-name every key). `weights.keys()` is the canonical
        # member order, in declaration order.
        members = list(weights.keys())
        n = len(members)
        if n == 0:
            return {"pick": None}
        total = sum(weights[m] for m in members)
        if total <= 0:
            return None
        # A cumulative weight grid: member i sits at [cum[i-1], cum[i]) in
        # the wheel. `start` picks a slot IN the wheel, then we find the
        # member whose range contains it and start from there, walking
        # clockwise on spills.
        cum: list[int] = []
        running = 0
        for m in members:
            running += weights[m]
            cum.append(running)
        start_slot = 0
        if self.policy is not None:
            start_slot = await self.policy.ledger.group_rot(group.gid, ctx.lane) % total
        # Member whose cumulative range contains `start_slot`.
        start_idx = next(i for i, c in enumerate(cum) if start_slot < c)
        for offset in range(n):
            member = members[(start_idx + offset) % n]
            pick = await self._visit(member, ctx, picked_group=picked_group)
            if pick is not None:
                return {"pick": pick}
        return {"pick": None}

    async def _visit_order_ranked(self, group: Group, refs: list[str],
                                  ctx: _VisitCtx, picked_group: Group | None,
                                  *, score_fn) -> dict | None:
        """Score-based ranking: perishable and lowest_utilization both read a
        group-scoped hash, sort by score descending, then walk members in
        that order. An unscored member sorts LAST so an unknown plan never
        beats a known one (matches the existing perishable contract).

        A gated member (gate5h=1) is recorded in `skipped` eagerly, BEFORE
        the walk — the legacy perishable writer did the same so the capacity
        board sees the gate even when the pick landed on a higher-ranked
        peer. The walk itself still skips gated members, so a stale hash
        that has since been refreshed cannot accidentally serve them.

        Nested groups are walked via `_visit`, not `_visit_ref`, so the
        recursion carries the same scoring contract into the inner group.
        The hash key for the score lookup is the FIRST leaf ref reachable
        from each member (refs sit at the leaves in the shipped example).

        Family partition: when the hash is present, the score-sorted list is
        re-ordered so the score never competes across `provider_family`
        boundaries. Without this, a fresh-pinned other-family member with a
        higher raw score would walk ahead of every leading-family member
        (issue #53). Bucket order is first-appearance of each family in the
        group's declared member order. A stale/missing hash skips the
        partition and walks the declared order unchanged.
        """
        order = await self._read_group_order(group, ctx.lane)
        members = list(group.members)
        if order is None:
            # Stale / missing ranking: fall back to declared order. Same
            # shape as the legacy perishable fallback.
            ranked = members
        else:
            score_by_ref = {m["ref"]: (m["score"], m["gate5h"])
                            for m in order["members"]}
            # Sort: scored first (desc), unscored after. The score lookup
            # uses the FIRST leaf ref of each member node — for a nested
            # group that is the first ref in its members list, which is
            # what the writer would have produced. Members with no leaf in
            # the hash land in `unscored` after every scored member, in
            # declared order (the legacy "unknown sorts last" rule).
            scored: list = []
            unscored: list = []
            for node in members:
                first_leaf = _first_leaf(node)
                if first_leaf in score_by_ref:
                    scored.append((node, score_by_ref[first_leaf]))
                else:
                    unscored.append(node)
            scored.sort(key=lambda rs: rs[1][0], reverse=True)
            # Filter gated scored members OUT of the walk. The gate check at
            # `_visit_ref` reads from `picked_group`'s hash and only fires when
            # the picked group is itself a scored group; for a leaf reached
            # via `[rotation -> scored -> ref]` the picked_group is the outer
            # rotation group, the gate check is bypassed, and the gated ref
            # would otherwise be served. Filtering here (where the gate is
            # actually owned) is the smallest fix that honours the contract
            # the eager `gate5h` recording below already promises the board:
            # a gated ref never lands on the wire, regardless of nesting.
            ranked_nodes = (
                [n for n, (_s, gated) in scored if not gated] + unscored)
            # Re-order by family so the score never crosses family boundaries.
            # `node_key` resolves a member node to a single ref-key (the same
            # first-leaf the score lookup uses); the family is then read off
            # that ref's plan. A node with no first leaf (empty group) is
            # passed through the partition via an empty-string key, which
            # the helper joins with the unspecified-bucket family rather
            # than crashing on a None model lookup.
            def _node_key(node):
                leaf = _first_leaf(node)
                return leaf if leaf is not None else ""
            def _family_for_key(key):
                if not key:
                    return None
                model = self.registry.model(key)
                if model is None:
                    return None
                return self.registry.plan_of(model).provider_family
            config_refs = [_node_key(m) for m in members]
            ranked_keys = [_node_key(n) for n in ranked_nodes]
            partitioned_keys = family_partitioned_order(
                ranked_keys, config_refs, _family_for_key)
            key_to_node = {_node_key(n): n for n in ranked_nodes}
            ranked = [key_to_node[k] for k in partitioned_keys]
            # Eagerly record gated refs before the walk so a higher-ranked
            # peer's success does not hide the gate on the board. The walk
            # itself now skips them (see the filter above), so this is purely
            # for the board's `considered` list.
            for node, (_score, gated) in scored:
                if gated:
                    for ref in _all_leaves(node):
                        ctx.skipped.append(f"{ref}(gate5h)")
        for member in ranked:
            pick = await self._visit(member, ctx, picked_group=picked_group)
            if pick is not None:
                return {"pick": pick}
        return {"pick": None}

    async def _read_group_order(self, group: Group, lane: str) -> dict | None:
        """The group-scoped ranking hash, falling back to the legacy lane-level
        key when the group's gid matches an implicit-perishable sugar wrap.

        Why a fallback: the existing per-lane writer
        (`_recompute_perishable_for_plan` in switchyard/portal/app.py) writes
        to K_LANE_ORDER, not K_GROUP_ORDER. The implicit perishable sugar
        (lane-level `strategy: perishable` with no explicit groups) reads
        THAT key to keep a flat config bit-for-bit identical. Explicit
        perishable groups use the group-scoped key (the producer change for
        that lives in WS2, the portal workstream).

        Stale-grace: the reader passes `(plan_key, target_window_label)`
        pairs for every member of the group, so a cookie-expired plan keeps
        its last good ranking while its window is still in force. The cap
        is exactly the window's `reset_at` — see
        `Ledger._in_grace_window`."""
        if self.policy is None:
            return None
        # Implicit-perishable sugar wraps the body in a synthesised group;
        # we know it's sugar when the gid matches the gid of the lane's
        # whole body (i.e. every member of the lane is in the group).
        lane_cfg = self.registry.lanes.get(lane)
        if lane_cfg is not None and lane_cfg.strategy == "perishable":
            nodes = self.registry.lane_nodes()[lane]
            if not any(isinstance(n, Group) for n in nodes):
                # Implicit wrap — read the legacy key.
                return await self.policy.ledger.get_lane_order(
                    lane, plan_windows=self._lane_plan_windows(lane))
        return await self.policy.ledger.get_group_order(
            group.gid, lane,
            plan_windows=self._group_plan_windows(group))

    def _group_plan_windows(
        self, group: Group
    ) -> list[tuple[str, str]]:
        """(plan_key, target_window_label) pairs for every leaf ref of `group`.

        Walks nested groups depth-first so the grace lookup matches the
        refs the picker is actually about to visit. The window label is the
        plan's primary quota (target window), the same window the perishable
        writer scores against.
        """
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for ref in _member_refs(group):
            model = self.registry.model(ref)
            if model is None:
                continue
            plan = self.registry.plan_of(model)
            if plan.key in seen:
                continue
            seen.add(plan.key)
            out.append((plan.key, plan.quota.label))
        return out

    def _lane_plan_windows(
        self, lane: str
    ) -> list[tuple[str, str]]:
        """(plan_key, target_window_label) pairs for every member of `lane`."""
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for model in self.registry.lane_members(lane):
            if self.registry.is_tail(lane, model.ref):
                continue
            plan = self.registry.plan_of(model)
            if plan.key in seen:
                continue
            seen.add(plan.key)
            out.append((plan.key, plan.quota.label))
        return out

    # -- the public pick ---------------------------------------------------
    async def pick(self, lane: str, session: str | None,
                   needs_tools: bool = False, pinned: bool = False,
                   exclude: frozenset[str] | None = None) -> Pick:
        rid = uuid.uuid4().hex
        ctx = _VisitCtx(
            lane=lane, rid=rid, session=session, pinned=pinned,
            exclude=exclude or frozenset(), needs_tools=needs_tools,
            wait=(self.registry.settings.pin_wait_seconds
                  if pinned and session else 0.0),
        )
        # The parsed body and an optional implicit-perishable wrap. Empty
        # body or a lane that has nothing on the wire at all: refuse before
        # touching Redis.
        body, implicit_group = self._effective_body(lane)
        if not body and not self.registry.lanes[lane].tail:
            # Nothing in the lane — every member is disabled or expired, or
            # the config didn't name anything. Match the existing detail
            # strings so the existing 429 messages are unchanged.
            members = await self._members(lane, needs_tools)
            detail = ("nothing in this lane can serve tool calls — every "
                      "candidate plan is marked supports_tools: false"
                      if needs_tools and not members
                      else "no live models (all expired or disabled)")
            raise LaneSaturated(lane, detail)

        # 1. Affinity — a leased plan that is still in the lane and not
        #    excluded claims the slot first, ahead of the body walk.
        affinity_pick = await self._affinity(ctx)
        if affinity_pick is not None:
            return affinity_pick

        # 2. Walk the body. Each node returns a Pick or None; first non-None
        #    wins. The body's own _visit methods handle per-member gates and
        #    group-internal ranking.
        for node in body:
            pick = await self._visit(node, ctx, picked_group=implicit_group)
            if pick is not None:
                return pick

        # 3. Tail as today — flat list, same per-member gates. A tail member
        #    that is paced-to-0 is excluded by `_members` upstream when
        #    pacing is on, but the picker re-checks at claim time too.
        tail_refs = self.registry.lanes[lane].tail
        tail_on = self.policy is None or await self.policy.tail_enabled()
        for ref in tail_refs:
            if not tail_on:
                ctx.skipped.append(f"{ref}(tail disabled (pacing))")
                continue
            pick = await self._visit_ref(ref, ctx, picked_group=None)
            if pick is not None:
                return pick

        raise LaneSaturated(lane, ", ".join(ctx.skipped))

    async def _affinity(self, ctx: _VisitCtx) -> Pick | None:
        """Honour an existing session lease before walking the body.

        The held plan must still be in the lane (after needs_tools filtering
        and exclude set) AND must accept a slot within the wait window. A
        session whose lease is alive but whose plan is now cooled or full is
        a session that has to move on — the wait rides out a burst, the
        deadline caps it, a cooled plan spills immediately.

        Returns the affinity Pick when the lease was honoured, None when the
        body walk should proceed.
        """
        if not ctx.session:
            return None
        held = await self.slots.get_lease(ctx.session)
        if not held:
            return None
        # Is the held plan still a candidate? Build the set of refs the body
        # + tail can reach, minus excludes. The held lease is a ref, so it
        # survives any tree shape.
        if held in ctx.exclude:
            await self.slots.drop_lease(ctx.session)
            return None
        # Drain gate: refuse to honour a lease on any plan that is currently
        # draining, regardless of apply.sh's order of operations. apply.sh
        # sets `sy:drain:{plan}` BEFORE it runs `python -m switchyard.drain`,
        # so the migrator walks `sessions_on_plan(plan)` while the flag is on;
        # `picker.pick(lane, session, exclude=drained_refs)` reaches this
        # path with the held lease still on the drained plan, drops it via
        # the `held in ctx.exclude` branch above, and re-leases onto a
        # sibling. The defensive check here covers the opposite direction:
        # a fresh request (no exclude set) arriving for a session whose
        # lease is still on a drained plan — which happens between the
        # flag flip and the migrate's SREM finishing — must NOT honour
        # that lease either. Drop it; the body walk below lands on a sibling.
        held_model = self.registry.model(held)
        if held_model is not None:
            held_plan = self.registry.plan_of(held_model)
            if await self.slots.is_draining(held_plan.key):
                await self.slots.drop_lease(ctx.session)
                ctx.skipped.append(f"{held}(draining)")
                return None
        reachable = set(self.registry.routing_order(ctx.lane))
        reachable.update(self.registry.lanes[ctx.lane].tail)
        if held not in reachable:
            return None
        model = self.registry.model(held)
        if model is None:
            return None
        if ctx.needs_tools and not self.registry.plan_of(model).can_use_tools:
            return None
        plan = self.registry.plan_of(model)
        cap, reason = await self._cap(model, allow_spent=ctx.pinned)
        cooled, _, _ = await self.slots.cooldown_state(plan.key)
        # No wait when there is nothing to wait FOR: a cap of 0 or an active
        # cooldown will not lift inside the deadline, so spill straight away.
        deadline = (time.monotonic() + ctx.wait
                    if cap > 0 and not cooled else time.monotonic())
        while True:
            if cap > 0 and await self.slots.try_claim(
                    plan.key, cap, ctx.rid, model.ref, model.max_parallel,
                    lane=ctx.lane) == 1:
                await self.slots.touch_lease(
                    ctx.session, self.registry.settings.lease_ttl_seconds)
                # Affinity honours the existing lease, NOT whatever group
                # the body walk would have chosen next — `picked_group`
                # stays None here on purpose so the log line shows the
                # affinity was a pin, not a group selection.
                return Pick(ctx.lane, model, plan, ctx.rid, ctx.session, True,
                            list(ctx.skipped), cap, reason)
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.5)
        # Held plan cannot serve: drop the lease (so the next pick does NOT
        # honour a stale pin to a broken plan) and note the skip for the
        # board. The body walk that follows may re-lease onto a peer.
        await self.slots.drop_lease(ctx.session)
        ctx.skipped.append(
            f"{held}(pinned, no free slot after {ctx.wait:g}s)"
            if ctx.pinned else f"{held}(lease unusable)")
        return None

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
            transient_streak = await self.slots.transient_failure_streak(plan.key)
            streak_alert = self.registry.settings.transient_breaker.streak_alert
            by_lane = await self.slots.in_flight_by_lane(plan.key, model.ref)
            model_here = by_lane.get(lane, 0)
            model_elsewhere = sum(n for k, n in by_lane.items() if k and k != lane)
            model_direct = by_lane.get("", 0)
            if tail and not tail_on:
                cap = 0
                cap_reason = "tail disabled (pacing)"

            row_cap, row_cap_reason = cap, cap_reason
            if model.max_parallel is not None and model.max_parallel < cap:
                row_cap = model.max_parallel
                row_cap_reason = f"model limit {model.max_parallel}"

            cap_model_owned = (
                not cooled
                and model.max_parallel is not None
                and model.max_parallel < plan.max_parallel
                and model.max_parallel <= cap
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
            "slots_in_use_here": max(0, used - used_elsewhere),
            "slots_in_use_elsewhere": used_elsewhere,
            "tail_only": live_members == 0 and bool(rows),
            "plans": rows,
        }


def _member_refs_of(nodes: list[Any]) -> list[str]:
    """Leaf refs reachable from a list of top-level nodes, in declaration order.

    Used only for the implicit-perishable sugar's gid: the gid must be stable
    per config, so the same sorted-leaf set produces the same gid regardless
    of how the operator nested or grouped them.
    """
    out: list[str] = []
    for n in nodes:
        out.extend(_member_refs(n))
    return out
