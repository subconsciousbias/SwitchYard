"""Drain a plan's sessions before taking it out of service.

Running `python -m switchyard.drain <plan>` migrates every session currently
leased to that plan onto a sibling, so no lease is left pointing at the
drained ref when apply.sh recreates the containers behind it. The picker
already refuses new work on a drained plan (the body-walk gate in
`_visit_ref` and the affinity drop on a draining held plan in `_affinity`),
so the migrator's contract is just "no lease points at the drained ref
after this runs" — apply.sh's order of operations is the operator's call.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from redis.asyncio import from_url as redis_from_url

from .models import load as load_models
from .picker import Picker
from .slots import SlotTable


async def migrate(plan_key: str, slots: SlotTable, picker: Picker,
                  ttl: int | None = None) -> tuple[int, list[str]]:
    """Re-pick every session leased to `plan_key` onto a sibling.

    Walks the reverse lease index (`slots.sessions_on_plan`) for the plan,
    calls `picker.pick` with the drained plan's refs excluded, and rewrites
    the lease via `set_lease(session, new_ref, ttl, plan_key)`. The pick
    is a placement check only — the slot claim it makes is released
    immediately afterwards, because no real request is in flight and there
    is no completion path to release it on its own. The lease is the
    persistent record of which ref each session now owns, and `set_lease`
    is what writes it. Prints one line per migrated session and a final
    count.

    The lane argument comes from the registry: a session is migrated through
    any lane that names a ref on the drained plan; the first lane in
    registry iteration order whose `pick` succeeds wins. A ref shared by
    several lanes (e.g. one `claude-max/fable` named in both `forge` and
    `tools`) lets a session that originated on lane A be re-leased onto a
    sibling reachable only via lane B — both legs are valid siblings, but
    the lane identity on the capacity board can change. If you need a
    stricter "same lane" contract, look up `slots.get_lease(session)` first
    and pin the candidate lane set to lanes whose `lane_members` covers the
    held ref; today that is not done.

    A session whose every candidate lane has no non-drained capacity is
    left untouched — the caller sees the SKIP line and can decide whether
    to retry or accept the loss.

    A session in `sessions_on_plan(plan)` whose lease is gone (TTL fired)
    or whose lease has moved to a different ref since the index was last
    maintained is treated as a stale reverse-index member: SKIP and
    `forget_sessions_on_plan`, no pick. Picking would write a fresh lease
    on the sibling via `_visit_ref`, leaving a phantom session on the new
    plan's SET with no lease ever pointing at the drained plan in the
    first place. The SET entry is the only stale state, so SREMing it
    brings the index back to the live picture without aborting the loop.

    A session whose release or lease re-write fails after a successful pick
    is logged and skipped rather than aborting the loop. The two failure
    cases leave different dirty state, so the log lines and the comments
    inside the two `except` blocks distinguish them: a `release` failure
    leaves one stale claim (the staleness sweep reclaims it within
    `inflight_max_age_seconds`); a `set_lease` failure here leaves no
    dirty state at all (the picker's `_visit_ref` already wrote the lease
    on the sibling, and `picker.release` already ran). In both cases
    aborting would orphan every not-yet-processed session on the plan
    `apply.sh` is taking out of service — strictly worse than the leak
    being fixed.

    The reverse-index hygiene gate at the top of the loop wraps both
    `slots.get_lease(session)` and `slots.forget_sessions_on_plan(...)`
    in the same `try/except Exception` shape as the rest of the loop:
    a transient Redis error logs and `continue`s, leaving the SET entry
    in place for the next drain cycle to retry. Aborting the loop here
    would orphan every not-yet-processed session on the plan being
    drained — the same "strictly worse" rationale the release/set_lease
    branches spell out.

    Returns (count_migrated, list_of_session_ids_migrated). The list is in
    the order sessions were processed (which is the SET iteration order).
    Stale members skipped by the reverse-index hygiene gate are NOT
    included in `migrated` (they were never migrated) and the call
    returns their count in the printed SKIP lines, but the return shape
    is unchanged so callers that pre-date this gate keep working.
    """
    registry = picker.registry
    lease_ttl = ttl if ttl is not None else registry.settings.lease_ttl_seconds
    drained_refs = frozenset(
        m.ref for p in registry.plans.values() if p.key == plan_key
        for m in p.models.values()
    )
    relevant_lanes = [
        lane for lane in registry.lanes
        if any(m.plan_key == plan_key for m in registry.lane_members(lane))
    ]

    sessions = await slots.sessions_on_plan(plan_key)
    migrated: list[str] = []
    for session in sessions:
        # Reverse-index hygiene: a session in `sessions_on_plan(plan)` is
        # expected to have a live lease whose ref belongs to this plan. Two
        # ways it can be stale at this point: (a) the lease's TTL fired and
        # the SET entry was collected alongside it (we are racing the SET's
        # own TTL), or (b) a parallel turn re-leased the session onto a
        # different plan (drain migration on a sibling, picker body-walk
        # swapping lanes, hooks re-leasing after a hard rejection) and the
        # SET entry got carried over. Either way, picking and migrating
        # this session would write a fresh lease on the sibling via the
        # picker's `_visit_ref` — a phantom: the session has no real lease
        # on this plan to migrate. SREM the SET entry and move on; the SET
        # is the only place the stale state lived.
        #
        # Both calls are wrapped in the same `try/except Exception` shape
        # as the rest of this loop's failure paths (the lane loop catches
        # "a Redis error, anything"; release and set_lease each have their
        # own `except` and a "strictly worse than the leak" comment).
        # Aborting the loop on a transient Redis error here would orphan
        # every not-yet-processed session on the plan `apply.sh` is
        # taking out of service — the exact outcome the other branches
        # document as worse than the leak being fixed. On error we leave
        # the SET entry in place; the next drain cycle (or this same call,
        # if the operator re-runs `python -m switchyard.drain`) will retry
        # the gate, which is the right idempotent recovery.
        try:
            current_lease = await slots.get_lease(session)
        except Exception as exc:
            print(f"{session}: reverse-index hygiene check failed "
                  f"({type(exc).__name__}: {exc}); SET entry left in place, "
                  f"next drain cycle will retry",
                  file=sys.stderr)
            continue
        if not current_lease or current_lease not in drained_refs:
            print(f"{session}: SKIP (stale reverse-index member on "
                  f"{plan_key}: lease "
                  f"{'missing' if not current_lease else repr(current_lease)}"
                  f"); SREM from plan SET",
                  file=sys.stderr)
            try:
                await slots.forget_sessions_on_plan(plan_key, [session])
            except Exception as exc:
                print(f"{session}: stale-member SREM failed "
                      f"({type(exc).__name__}: {exc}); SET entry left in "
                      f"place, next drain cycle will retry",
                      file=sys.stderr)
            continue
        placed = None
        for lane in relevant_lanes:
            try:
                placed = await picker.pick(
                    lane, session, exclude=drained_refs)
                break
            except Exception as exc:
                # LaneSaturated, a Redis error, anything — the next lane
                # in registry iteration order gets a turn. If every lane
                # fails, the SKIP branch below records the session as
                # left-behind without surfacing the (possibly low-value)
                # last exception.
                print(f"{session}: lane={lane} skipped ({type(exc).__name__}: {exc})",
                      file=sys.stderr)
                continue
        if placed is None:
            print(f"{session}: SKIP (no lane could place after drain)",
                  file=sys.stderr)
            continue
        try:
            await picker.release(placed.plan.key, placed.request_id, placed.ref)
        except Exception as exc:
            # release-fail: the slot claim stays on the sibling until the
            # staleness sweep reclaims it. The lease on the sibling is
            # already in place (the pick's `_visit_ref` `set_lease`'d it
            # before returning), so the session is effectively migrated —
            # only the claim counter is dirty. Aborting here would orphan
            # every not-yet-processed session on the plan being drained,
            # which is strictly worse than the leak being fixed.
            print(f"{session}: release failed ({type(exc).__name__}: {exc}); "
                  f"stale claim on {placed.ref}, sweep will reclaim",
                  file=sys.stderr)
            continue
        try:
            await slots.set_lease(session, placed.ref, lease_ttl, placed.plan.key)
        except Exception as exc:
            # set_lease-fail: `picker.release` already ran, and the pick's
            # `_visit_ref` already wrote the lease on the sibling — the
            # session is fully migrated with no dirty state left. This
            # explicit re-write is a TTL refresh (the picker's lease uses
            # the registry default TTL; this call lets the caller pass a
            # longer one via `--ttl`), so the only consequence of a
            # failure here is that the lease ages out at the registry
            # default instead of `ttl`. Aborting here would orphan the
            # remaining sessions — same logic as the release-fail case.
            print(f"{session}: lease rewrite failed ({type(exc).__name__}: {exc}); "
                  f"lease on {placed.ref} already in place from pick",
                  file=sys.stderr)
            continue
        migrated.append(session)
        print(f"{session}: {plan_key}/{placed.plan.label} -> {placed.ref}")
    return len(migrated), migrated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate sessions off a draining plan")
    parser.add_argument("plan", help="Plan key to drain")
    parser.add_argument("--ttl", type=int, default=None,
                        help="TTL for rewritten leases "
                             "(default: settings.lease_ttl_seconds)")
    args = parser.parse_args(argv)

    redis_url = os.environ.get("SWITCHYARD_REDIS_URL")
    if not redis_url:
        print("SWITCHYARD_REDIS_URL must be set", file=sys.stderr)
        return 2
    plans_path = os.environ.get("SWITCHYARD_PLANS")
    if not plans_path:
        print("SWITCHYARD_PLANS must be set", file=sys.stderr)
        return 2

    async def run() -> int:
        registry = load_models(plans_path)
        redis = redis_from_url(redis_url)
        slots = SlotTable(redis, registry.settings.inflight_max_age_seconds)
        picker = Picker(registry, slots)
        count, _ = await migrate(args.plan, slots, picker, ttl=args.ttl)
        await redis.aclose()
        print(f"migrated {count} session(s) off {args.plan}")
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
