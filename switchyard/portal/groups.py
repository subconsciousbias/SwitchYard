"""Per-group order writers for the portal recompute.

The lane-level writer (`_recompute_perishable_for_plan` in app.py) covers the
flat `strategy: perishable` sugar. Explicit groups that need a score hash --
`perishable:` and `lowest_utilization:` -- get their own writer, keyed by
`sy:group-order:{gid}:{lane}` so the picker can read them independently for
each group occurrence. The hysteresis / 5h gate / unscored-last semantics are
shared: this file leans on the same `_one_adjacent_swap`, the same
`perishable_score` and `utilization_score` plumbing, the same `stale_after_ms`
writer. Only the key shape and the score function differ.

Why a separate file:
    * the picker reads `sy:group-order:{gid}:{lane}` for an explicit group,
      never `sy:lane-order:{lane}`, so the legacy writer would leave it empty;
    * a per-group score function (perishable vs. lowest_utilization) lives in
      the group node itself, not on the lane;
    * keeping this out of `app.py` means a regression in the writer cannot
      take down the existing flat-config path.
"""
from __future__ import annotations

import time
from typing import Any

from ..models import Group, Plan, Registry
from ..usage import Ledger, perishable_score, utilization_score


# Maps a group strategy to the score function that ranks its members. New
# strategies that need a per-group hash land here; round_robin / weighted
# are absent because their order is the rotation pointer, not a score.
_SCORE_FN = {
    "perishable": perishable_score,
    "lowest_utilization": utilization_score,
}


def _collect_group_orders(
    lane_cfg, lane_key: str, nodes: list[Any]
) -> list[Group]:
    """Walk the lane body and return the explicit groups that need a per-group
    hash. Depth-first so the inner groups of a nested tree are written in the
    same order the picker walks them; nested groups are returned too -- a
    nested `lowest_utilization` inside an outer `round_robin` still needs its
    own ranking to be useful, even though the outer group never reads it
    directly (it walks inner members via `_visit`, which does).
    """
    out: list[Group] = []
    for node in nodes:
        if isinstance(node, Group):
            if node.strategy in _SCORE_FN and node.weights is None:
                out.append(node)
            # Nested groups: walk further.
            for inner in node.members:
                if isinstance(inner, Group):
                    out.extend(_collect_group_orders(lane_cfg, lane_key, [inner]))
    return out


def _group_leaf_refs(group: Group) -> list[str]:
    """Refs reachable from a group's members in declaration order.

    A weighted group's keys are its members; nested groups recurse through
    `_member_refs` (defined in picker.py) which already handles every shape.
    Reimplemented locally to avoid a circular import.
    """
    out: list[str] = []
    def walk(node):
        if isinstance(node, Group):
            if node.weights is not None:
                out.extend(node.weights.keys())
                return
            for m in node.members:
                walk(m)
            return
        out.append(node)
    for m in group.members:
        walk(m)
    return out


async def _score_for_plan(
    plan: Plan, ledger: Ledger
) -> tuple[float | None, float | None, bool]:
    """Return (room_pct, reset_at, gate5h) for a plan's target window.

    Mirrors the per-lane writer: target window pct + reset, with the 5h
    constraint window as a gate. Returns (None, None, False) when the plan
    has no probe-shaped target window.
    """
    target_label = next(
        (q.label for q in plan.quotas if q.role == "target"),
        plan.quotas[0].label if plan.quotas else None)
    gate_label = next(
        (q.label for q in plan.quotas if q.period == "rolling_5h"), None)
    if not target_label:
        return None, None, False

    def _as_float(v) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    target_facts = await ledger.window_facts(plan.key, target_label)
    target_pct = _as_float(target_facts.get("reported_pct_used"))
    target_reset = _as_float(target_facts.get("reset_at"))
    gate_facts = (await ledger.window_facts(plan.key, gate_label)
                  if gate_label else {})
    gate_pct = _as_float(gate_facts.get("reported_pct_used"))
    gate5h = gate_pct is not None and gate_pct > 90
    if target_pct is None:
        return None, target_reset, gate5h
    return max(0.0, 100.0 - target_pct), target_reset, gate5h


def _stale_after_ms(plan: Plan | None) -> int:
    """Match the lane-level writer's staleness bound (2x probe interval, min 60s)."""
    interval = (plan.probe.interval_seconds if plan is not None and plan.probe
                is not None else 600)
    return max(60_000, int(interval) * 2000)


async def recompute_group_orders(reg: Registry, ledger: Ledger) -> int:
    """Recompute `sy:group-order:{gid}:{lane}` for every explicit
    `perishable:` and `lowest_utilization:` group in the parsed tree.

    Returns the number of group hashes written, for tests / observability.
    The lane-level writer continues to drive the legacy
    `sy:lane-order:{lane}` key for the flat `strategy: perishable` sugar; this
    function only writes the per-group keys the picker reads for explicit
    groups.

    Triggered from `_poll_probes` after the lane-level recompute -- the
    score lookup reads the same facts the lane writer already pulled, so
    the cost is one extra hash read per affected group per poll.
    """
    from .app import _one_adjacent_swap  # local import: keeps the module standalone

    written = 0
    for lane_key, nodes in reg.lane_nodes().items():
        groups = _collect_group_orders(reg.lanes[lane_key], lane_key, nodes)
        if not groups:
            continue
        for group in groups:
            await _write_group_order(
                reg, ledger, lane_key, group, _stale_after_ms(None),
                _one_adjacent_swap)
            written += 1
    return written


async def _write_group_order(
    reg: Registry, ledger: Ledger, lane_key: str, group: Group,
    stale_after_ms: int,
    swap_fn,
) -> None:
    """Score the members of a single group and write its hash.

    Members can span plans; each member's plan is scored independently using
    its own probe facts. Members whose plan has no target window land in
    `unknown` and are not written (the picker reads them as unscored and
    sorts them last in declared order).

    The hysteresis step is shared with the lane-level writer: at most one
    adjacent swap per poll, and refs that left the lane are dropped on
    write so the picker never sees a stale `score_<ref>`.
    """
    score_fn = _SCORE_FN[group.strategy]
    leaf_refs = _group_leaf_refs(group)
    if not leaf_refs:
        return

    previous = await ledger.get_group_order(group.gid, lane_key)
    previous_by_ref = ({m["ref"]: m for m in previous["members"]}
                       if previous else {})

    scored: list[tuple[str, float, bool]] = []
    unknown: list[str] = []
    for ref in leaf_refs:
        model = reg.model(ref)
        if model is None:
            unknown.append(ref)
            continue
        plan = reg.plan_of(model)
        room, reset, gate5h = await _score_for_plan(plan, ledger)
        if room is None:
            # No target window facts. Reuse the previous score for this ref
            # if we have one -- the previous poll wrote it, and one probe
            # interval of staleness is within the picker window.
            prev = previous_by_ref.get(ref)
            if prev:
                scored.append((ref, float(prev["score"]), bool(prev["gate5h"])))
            else:
                unknown.append(ref)
            continue
        score = score_fn(room, reset)
        if score is None:
            unknown.append(ref)
            continue
        scored.append((ref, score, gate5h))

    if not scored and not unknown:
        # No member produced a score: drop the key so the picker falls back
        # to config order via the missing-key path. A regression that wrote
        # an empty entries dict here would leave the picker reading stale
        # members indefinitely.
        await ledger.set_group_order(
            group.gid, lane_key, {},
            computed_at=time.time(), stale_after_ms=stale_after_ms)
        return

    scored.sort(key=lambda r: r[1], reverse=True)
    scored_map = {r[0]: (r[1], r[2]) for r in scored}
    config_position = {r: i for i, r in enumerate(leaf_refs)}
    unknown_sorted = sorted(unknown, key=lambda r: config_position.get(r, 1e9))
    desired = [r[0] for r in scored] + unknown_sorted

    if previous is None:
        next_order = desired
    else:
        previous_refs = [m["ref"] for m in previous["members"]]
        next_order = swap_fn(previous_refs, desired, scored_map)

    entries: dict[str, dict] = {}
    for ref in next_order:
        if ref in scored_map:
            s, g = scored_map[ref]
            entries[ref] = {"score": s, "gate5h": 1 if g else 0}
    await ledger.set_group_order(
        group.gid, lane_key, entries,
        computed_at=time.time(), stale_after_ms=stale_after_ms)


# -- capacity board builders --------------------------------------------------
# Functions the portal writer calls per render to materialise group structure
# into the lane dict the template reads. The writer itself does not need
# these; they exist so the board can show the same shape the picker walks.

async def build_groups(reg: Registry, ledger: Ledger, lane_key: str,
                       members_rows: list[dict]) -> list[dict]:
    """Turn `lane_nodes()` into a flat list of (kind, payload, depth) entries
    the template can iterate linearly.

    Each entry is one of:
        {"kind": "group", "strategy": ..., "gid": ..., "weight_*": ...,
         "members": [<refs>], "depth": N, "ranking": {...}|None, "next_index": int|None}
        {"kind": "ref", "row": <member_row>, "depth": N}

    The depth field is the visual indent (0 = top-level, 1 = inside one
    group, etc.); it is rendered via inline `padding-left` on the row's
    first cell so no CSS churn is needed. A flat config (no Groups) returns
    an empty list, and the template falls back to its legacy
    `lane.plans` loop -- bit-for-bit identical to today.
    """
    nodes = reg.lane_nodes().get(lane_key, [])
    rows_by_ref = {r["ref"]: r for r in members_rows}
    rot = await _rotation_pointers(reg, ledger, lane_key)
    rankings = await _ranking_snapshots(reg, ledger, lane_key)

    out: list[dict] = []
    for node in nodes:
        _flatten(node, rows_by_ref, rot, rankings, lane_key,
                 depth=0, out=out)
    return out


async def _ranking_snapshots(reg: Registry, ledger: Ledger, lane_key: str) -> dict:
    """`{(gid, lane): [<refs in score order>]}` -- current order from Redis
    for every group that uses one. The order is the published `members`
    list (already sorted by score desc), or None when missing/stale.
    """
    out: dict = {}
    nodes = reg.lane_nodes().get(lane_key, [])
    for group in _walk_groups(nodes):
        if group.strategy not in ("perishable", "lowest_utilization"):
            continue
        order = await ledger.get_group_order(group.gid, lane_key)
        if order is not None:
            out[(group.gid, lane_key)] = [m["ref"] for m in order["members"]]
    return out


def _flatten(node, rows_by_ref, rot, rankings, lane_key, *, depth, out):
    """Append one (or more) entries for `node` to `out`.

    A Group produces a `kind="group"` header followed by its members in
    declaration order; nested groups increase `depth`. A bare ref produces
    one `kind="ref"` entry with that model row.
    """
    if isinstance(node, Group):
        refs = _group_leaf_refs(node)
        header: dict[str, Any] = {
            "kind": "group",
            "strategy": node.strategy,
            "gid": node.gid,
            "members": refs,
            "depth": depth,
        }
        if node.strategy == "round_robin":
            pointer = rot.get((node.gid, lane_key), 0) or 0
            header["next_index"] = pointer % max(1, len(refs))
            header["pointer"] = pointer
        elif node.strategy == "weighted":
            header["weights"] = dict(node.weights or {})
        elif node.strategy in ("perishable", "lowest_utilization"):
            header["ranking"] = rankings.get((node.gid, lane_key))
        out.append(header)
        if node.weights is None:
            for member in node.members:
                _flatten(member, rows_by_ref, rot, rankings, lane_key,
                         depth=depth + 1, out=out)
        return
    row = rows_by_ref.get(node)
    if row is not None:
        out.append({"kind": "ref", "row": row, "depth": depth})


async def _rotation_pointers(reg: Registry, ledger: Ledger, lane_key: str) -> dict:
    """`{(gid, lane): int}` -- the current rotation pointer for every
    round_robin / weighted group on this lane. 0 when Redis has no record
    (fresh process, idle group); the template treats 0 as "start at member 0".
    """
    out: dict = {}
    nodes = reg.lane_nodes().get(lane_key, [])
    for node in _walk_groups(nodes):
        if node.strategy in ("round_robin", "weighted"):
            pointer = await ledger.group_rot(node.gid, lane_key)
            out[(node.gid, lane_key)] = pointer
    return out


def _walk_groups(nodes):
    """Depth-first yield of every Group node reachable from `nodes`."""
    for node in nodes:
        if isinstance(node, Group):
            yield node
            if node.weights is None:
                yield from _walk_groups(node.members)


async def build_group_ranking(reg: Registry, ledger: Ledger, lane_key: str,
                              group: Group) -> dict | None:
    """The per-group ranking hash, for the board's display only.

    Returns the same shape the picker reads, or None when the key is
    missing/stale. The template uses it to surface `current order: a, b, c`
    next to `lowest_utilization` / `perishable` group headers.
    """
    return await ledger.get_group_order(group.gid, lane_key)
