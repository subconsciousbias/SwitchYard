"""Drain a plan's sessions before taking it out of service.

Running `python -m switchyard.drain <plan>` migrates every session currently
leased to that plan onto a sibling, so no lease is left pointing at the
drained ref when apply.sh recreates the containers behind it. The flag the
picker reads to refuse new traffic on a drained plan (`sy:drain:{plan}`) is
set AFTER migration completes; flipping it first would just make the
re-pick inside the migrator fall through to the same siblings anyway, but
keeping the order drain-after-migrate means a session that gets re-leased
during the migration lands somewhere useful rather than waiting on a
fresh pick against a now-flagged plan.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from redis.asyncio import from_url as redis_from_url

from .models import Registry, load as load_models
from .picker import Picker
from .slots import SlotTable


async def migrate(plan_key: str, slots: SlotTable, picker: Picker,
                  ttl: int | None = None) -> tuple[int, list[str]]:
    """Re-pick every session leased to `plan_key` onto a sibling.

    Walks the reverse lease index (`slots.sessions_on_plan`) for the plan,
    calls `picker.pick` with the drained plan's refs excluded, and rewrites
    the lease via `set_lease(session, new_ref, ttl, plan_key)`. Prints one
    line per migrated session and a final count.

    The lane argument comes from the registry: a session is migrated through
    the first lane that names a ref on the drained plan (the lane the
    session was reachable from before draining started). A session whose
    only lane has no non-drained capacity is left untouched — the caller
    sees the SKIP line and can decide whether to retry or accept the loss.

    Returns (count_migrated, list_of_session_ids_migrated). The list is in
    the order sessions were processed (which is the SET iteration order).
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
        placed = None
        for lane in relevant_lanes:
            try:
                placed = await picker.pick(
                    lane, session, exclude=drained_refs)
                break
            except Exception as exc:
                last_exc = exc
                continue
        if placed is None:
            print(f"{session}: SKIP (no lane could place after drain)",
                  file=sys.stderr)
            continue
        await slots.set_lease(session, placed.ref, lease_ttl, placed.plan.key)
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
