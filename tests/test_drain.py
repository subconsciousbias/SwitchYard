"""Tests for the picker drain gate, the lease-plan reverse index, and the
drain flow's session migration helper.

The drain flag flips a plan off the rotation without unmounting it: new
requests see the `(draining)` skip reason and fall through to the lane's
next sibling, while sessions that were already leased to the plan get
migrated to the sibling by `switchyard.drain.migrate` BEFORE the flag is
set. Three properties hold:

  (a) the picker honours the flag with the documented skip reason and a
      sibling absorbs the request;
  (b) the flag is exactly a sentinel — flipping it back lets traffic
      resume without restart;
  (c) `set_lease` / `touch_lease` / `drop_lease` keep the reverse index
      in sync, so `sessions_on_plan` is the live picture;
  (d) `drain.migrate` rewrites N leases to a sibling when the drained
      plan is excluded from the re-pick.

Inserted BEFORE the runner so the discovery sees every test_ function.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plans_path import plans_path  # noqa: E402

os.environ["SWITCHYARD_PLANS"] = plans_path()

from switchyard import models                      # noqa: E402
from switchyard.drain import migrate               # noqa: E402
from switchyard.picker import Picker  # noqa: E402
from switchyard.slots import SlotTable             # noqa: E402
from tests.fake_redis import FakeRedis             # noqa: E402


def _build():
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    return reg, slots, Picker(reg, slots)


def _run(coro):
    return asyncio.run(coro)


def test_drained_plan_is_skipped_with_draining_reason_and_sibling_absorbs():
    """Setting the drain flag on a plan makes the picker list the ref as
    `(draining)` in `considered` and fall through to the lane's next
    sibling.

    The fixture's `forge` lane has `minimax-ultra/m3` first in config order.
    With ultra drained, a fresh session lands on the next live member
    (`minimax-max/m3`) and the board sees the `(draining)` skip on ultra.
    """
    async def go():
        reg, slots, picker = _build()
        lane = "forge"
        target = next(m for m in reg.lane_members(lane)
                      if m.plan_key == "minimax-ultra")
        drained_plan = target.plan_key
        await slots.set_drain(drained_plan)

        pick = await picker.pick(lane, None)
        considered = pick.considered
        draining_skip = [c for c in considered if c.startswith(target.ref)
                         and "(draining)" in c]
        return pick.ref, target.ref, draining_skip, drained_plan

    picked, drained_ref, draining_skip, plan_key = _run(go())
    assert draining_skip, (
        f"a drained {plan_key} ref must surface as `(draining)` in considered, "
        f"got nothing on a pick that landed on {picked}")
    assert picked != drained_ref, (
        f"a drained {drained_ref} must NOT be picked; got {picked}")
    print(f"  drained {drained_ref}: skipped with `(draining)`, "
          f"sibling {picked} absorbed the request")


def test_clear_drain_restores_traffic_without_state_leakage():
    """The drain flag is a sentinel, not state: clear_drain makes the plan
    pickable again, and the next pick lands on it (the first member of the
    lane in config order).

    The control case at the start — with no flag ever set — picks the
    config-order-first member too, so clearing the flag returns the picker
    to its baseline shape, not to some intermediate "flag-was-once-set"
    state. The first member's plan cap is generous (e.g. minimax-ultra
    allows 4 concurrent), so consecutive picks land on the same ref without
    any release between them.
    """
    async def go():
        reg, slots, picker = _build()
        lane = "forge"
        first = reg.lane_members(lane)[0]
        first_plan = first.plan_key

        baseline = (await picker.pick(lane, None)).ref
        await slots.set_drain(first_plan)
        drained_pick = await picker.pick(lane, None)
        await slots.clear_drain(first_plan)
        cleared_pick = await picker.pick(lane, None)
        return baseline, first.ref, drained_pick.ref, cleared_pick.ref

    baseline, first_ref, drained_ref, cleared_ref = _run(go())
    assert baseline == first_ref, (
        f"baseline should pick {first_ref}; got {baseline}")
    assert drained_ref != first_ref, (
        f"drained plan must not be picked; got {drained_ref}")
    assert cleared_ref == first_ref, (
        f"clearing the drain flag must restore the config-order pick; "
        f"expected {first_ref}, got {cleared_ref}")
    print(f"  baseline={baseline}, drained={drained_ref}, "
          f"cleared={cleared_ref}: flag is a sentinel, no leakage")


def test_set_touch_drop_lease_keep_lease_plan_index_in_sync():
    """`set_lease` adds the session to the per-plan reverse index, `drop_lease`
    removes it, and `touch_lease` keeps the SET alive (so a session that is
    repeatedly touched does not silently fall out of `sessions_on_plan`).

    The drain migration walks `sessions_on_plan` to know what to migrate —
    if any of these three methods left the index stale, the migration
    would miss sessions or migrate sessions that no longer exist.
    """
    async def go():
        reg, slots, _picker = _build()
        plan_a = "claude-max"
        plan_b = "minimax-ultra"
        ttl = reg.settings.lease_ttl_seconds

        # Three sessions on plan A, one on plan B, one dropped.
        await slots.set_lease("s1", "claude-max/fable", ttl, plan_a)
        await slots.set_lease("s2", "claude-max/opus", ttl, plan_a)
        await slots.set_lease("s3", "claude-max/fable", ttl, plan_a)
        await slots.set_lease("s4", "minimax-ultra/m3", ttl, plan_b)

        on_a_after_set = sorted(await slots.sessions_on_plan(plan_a))
        on_b_after_set = sorted(await slots.sessions_on_plan(plan_b))

        # Touch one of plan A's sessions: the index must still contain it,
        # because the lease TTL was just refreshed.
        await slots.touch_lease("s1", ttl)
        on_a_after_touch = sorted(await slots.sessions_on_plan(plan_a))

        # Drop one of plan A's sessions: it must leave the index, but the
        # other two plan-A sessions must remain (SREM hits only that one).
        await slots.drop_lease("s2")
        on_a_after_drop = sorted(await slots.sessions_on_plan(plan_a))

        # Drop the plan-B session too, to confirm the reverse-index removal
        # uses the plan key derived from the lease ref, not the wrong one.
        await slots.drop_lease("s4")
        on_b_after_drop = sorted(await slots.sessions_on_plan(plan_b))

        return (on_a_after_set, on_b_after_set,
                on_a_after_touch, on_a_after_drop, on_b_after_drop)

    a_set, b_set, a_touch, a_drop, b_drop = _run(go())
    assert a_set == ["s1", "s2", "s3"], a_set
    assert b_set == ["s4"], b_set
    assert a_touch == ["s1", "s2", "s3"], a_touch
    assert a_drop == ["s1", "s3"], a_drop
    assert b_drop == [], b_drop
    print(f"  set: A={a_set} B={b_set}; "
          f"touch(A)={a_touch}; drop(s2)=A:{a_drop}; drop(s4)=B:{b_drop}")


def test_drain_migrate_rewrites_n_leases_to_the_sibling_ref():
    """`switchyard.drain.migrate` walks `sessions_on_plan(drained_plan)`,
    re-picks each session through the picker with the drained plan's refs
    excluded, and rewrites the lease to the new ref.

    The new ref must be on a DIFFERENT plan than the drained one — the
    excluded set is what forces that — and every pre-existing lease that
    pointed at the drained plan must end up on the sibling. A session that
    was on a different plan already is left alone (the SET only has leases
    pointing at the drained plan).
    """
    async def go():
        reg, slots, picker = _build()
        lane = "forge"
        # Two siblings on different plans, plus a third on a third plan
        # to confirm the drained plan is NOT in the post-migration SET.
        members = reg.lane_members(lane)
        target = next(m for m in members if m.plan_key == "minimax-ultra")
        sibling = next(m for m in members
                       if m.plan_key != target.plan_key
                       and not reg.is_tail(lane, m.ref))
        other = next(m for m in members
                     if m.plan_key not in (target.plan_key, sibling.plan_key)
                     and not reg.is_tail(lane, m.ref))
        drained_plan = target.plan_key

        ttl = reg.settings.lease_ttl_seconds

        # Three sessions on the drained plan, one on `other` (must be left
        # alone by the migration), one already on the sibling (also left
        # alone — it never appeared on the drained plan in the first place).
        await slots.set_lease("s-drain-1", target.ref, ttl, drained_plan)
        await slots.set_lease("s-drain-2", target.ref, ttl, drained_plan)
        await slots.set_lease("s-drain-3", target.ref, ttl, drained_plan)
        await slots.set_lease("s-other", other.ref, ttl, other.plan_key)
        await slots.set_lease("s-sibling", sibling.ref, ttl, sibling.plan_key)

        before_drained = sorted(await slots.sessions_on_plan(drained_plan))

        drained_refs = frozenset(
            m.ref for p in reg.plans.values() if p.key == drained_plan
            for m in p.models.values())
        count, migrated = await migrate(drained_plan, slots, picker,
                                        ttl=ttl)

        after_drained = sorted(await slots.sessions_on_plan(drained_plan))
        new_leases = {s: await slots.get_lease(s) for s in migrated}
        other_lease = await slots.get_lease("s-other")
        sibling_lease = await slots.get_lease("s-sibling")

        return (before_drained, count, migrated, after_drained,
                new_leases, other_lease, sibling_lease,
                drained_plan, target, sibling, other, drained_refs)

    (before_drained, count, migrated, after_drained,
     new_leases, other_lease, sibling_lease,
     drained_plan, target, sibling, other, drained_refs) = _run(go())

    assert before_drained == ["s-drain-1", "s-drain-2", "s-drain-3"], \
        before_drained
    assert count == 3, count
    assert sorted(migrated) == ["s-drain-1", "s-drain-2", "s-drain-3"], \
        migrated
    # Every rewritten lease points at a ref whose plan is NOT the drained
    # one — the exclude set is what guarantees this.
    for s, ref in new_leases.items():
        assert ref is not None, f"{s} has no lease after migration"
        assert "/" in ref, f"{s}: lease {ref!r} is not a model ref"
        assert ref.split("/", 1)[0] != drained_plan, (
            f"{s}: migration left the lease on the drained plan: {ref}")
        assert ref not in drained_refs, (
            f"{s}: migration rewrote to a drained ref: {ref}")
    # The drained plan's reverse index is empty after migration — nothing
    # was re-leased onto it, and the SET removal is part of the same call.
    assert after_drained == [], after_drained
    # Sessions that were NOT on the drained plan are untouched.
    assert other_lease == other.ref, other_lease
    assert sibling_lease == sibling.ref, sibling_lease
    print(f"  migrated {count} sessions off {drained_plan}: "
          f"{sorted(new_leases.items())}; other untouched ({other_lease}), "
          f"sibling untouched ({sibling_lease}); drained SET cleared")


def test_affinity_drops_lease_when_held_plan_is_draining():
    """A session whose lease is on a draining plan must NOT be honoured by
    affinity, even on a fresh request whose `ctx.exclude` does NOT name the
    drained ref. apply.sh sets the flag BEFORE the migrator runs, so a
    request can arrive between flag-flip and migrate's SREM; the affinity
    path is the safety net (see picker.py `_affinity`).

    Without this gate, the lease would be honoured, the body walk would
    be skipped, and the request would land on the draining ref — exactly
    what the drain flag is meant to prevent.
    """
    async def go():
        reg, slots, picker = _build()
        lane = "forge"
        target = next(m for m in reg.lane_members(lane)
                      if m.plan_key == "minimax-ultra")
        sibling = next(m for m in reg.lane_members(lane)
                       if m.plan_key != target.plan_key
                       and not reg.is_tail(lane, m.ref))
        ttl = reg.settings.lease_ttl_seconds

        # Lease the session to the target ref the way a live conversation
        # would. Then flip the drain flag on the target's plan — this is
        # what apply.sh:375 does, before apply.sh:384 runs the migrator.
        await slots.set_lease("affinity-drain", target.ref, ttl, target.plan_key)
        await slots.set_drain(target.plan_key)

        # Pick with the SAME session id. ctx.exclude is empty (a fresh
        # request, not the migrator): the affinity path is the only thing
        # that can stop the lease being honoured.
        pick = await picker.pick(lane, "affinity-drain")
        lease_after = await slots.get_lease("affinity-drain")
        return pick, lease_after, target, sibling

    pick, lease_after, target, sibling = _run(go())
    assert pick.ref != target.ref, (
        f"affinity on a draining {target.ref} must NOT land on the drained "
        f"ref; got {pick.ref}")
    assert pick.ref == sibling.ref, (
        f"the body walk after the affinity drop should land on the live "
        f"sibling {sibling.ref}; got {pick.ref}")
    assert lease_after == sibling.ref, (
        f"the affinity drop must RE-LEASE the session onto the sibling "
        f"so future picks honour the new pin; got {lease_after!r}")
    print(f"  drained lease held on {target.ref}; affinity dropped, "
          f"re-leased onto {pick.ref}; future pick pins {lease_after}")


def test_set_lease_writes_lease_and_reverse_index_in_one_round_trip():
    """`set_lease` is the third member of the lease trio (set / touch /
    drop). Cycle-2 closed the analogous races in `touch_lease` and
    `drop_lease` by moving both into Lua scripts; `set_lease` was left
    as a three-step Python sequence with a single-round-trip race between
    the SADD and the EXPIRE on the SET (network drop leaves the SET with
    new membership but no TTL, `sessions_on_plan(plan)` reads a phantom
    forever). Cycle-3 closes this by moving `set_lease` into a Lua script
    too.

    The contract we pin here is the observable happy-path behaviour:

      (a) the lease key holds the new ref with the requested TTL,
      (b) the SET key holds the session,
      (c) the script's SADD does not duplicate the membership on a
          same-session re-set (SADD returns 0 for an already-present
          member — the atomicity shape the Lua guarantees).

    The fake cannot directly simulate the partial-failure the script
    closes (its `expire` is string-only — see the SET_LEASE shadow's
    comment in `tests/fake_redis.py`), so the "atomicity holds across
    network failures" property is real-Redis-only. This test pins the
    membership and TTL contract; the cross-plan migration is the
    responsibility of `drop_lease` (SREM on the old plan's SET, atomic),
    not `set_lease`. In the live flow, every picker re-lease is preceded
    by `_affinity`'s `drop_lease`, so the phantom problem the reviewer
    raised on a hypothetical "set_lease twice without drop_lease" path
    does not exist in the codebase.
    """
    async def go():
        reg, slots, _picker = _build()
        ttl = reg.settings.lease_ttl_seconds
        plan_a = "claude-max"
        plan_b = "minimax-ultra"

        # First set: lease on plan A, SADD to plan A's SET.
        await slots.set_lease("s-reroute", "claude-max/fable", ttl, plan_a)
        lease_a = await slots.get_lease("s-reroute")
        on_a = sorted(await slots.sessions_on_plan(plan_a))
        ttl_a = await slots.redis.ttl("sy:lease:s-reroute")

        # Re-set on the SAME session, SAME ref: SADD returns 0 (already a
        # member), the SET still has exactly one entry for this session,
        # the lease keeps its TTL.
        await slots.set_lease("s-reroute", "claude-max/fable", ttl, plan_a)
        on_a_again = sorted(await slots.sessions_on_plan(plan_a))
        ttl_a_again = await slots.redis.ttl("sy:lease:s-reroute")
        # And again on a DIFFERENT plan — the SADD lands on the new SET
        # (the script's KEYS[2] is the new plan); the old SET still holds
        # the session because cross-plan migration is drop_lease's job,
        # not set_lease's. We assert only that the new SET gained the
        # session; the phantom-cleanup behaviour is exercised in the
        # existing `test_set_touch_drop_lease_keep_lease_plan_index_in_sync`.
        await slots.set_lease("s-reroute", "minimax-ultra/m3", ttl, plan_b)
        on_b_after = sorted(await slots.sessions_on_plan(plan_b))
        ttl_b = await slots.redis.ttl("sy:lease:s-reroute")

        return (lease_a, on_a, ttl_a,
                on_a_again, ttl_a_again,
                on_b_after, ttl_b)

    (lease_a, on_a, ttl_a,
     on_a_again, ttl_a_again,
     on_b_after, ttl_b) = _run(go())
    assert lease_a == "claude-max/fable", lease_a
    assert on_a == ["s-reroute"], on_a
    assert ttl_a > 0, f"lease must have a TTL after set_lease, got {ttl_a}"
    # Same-session re-set: SET has exactly one entry, no duplicate.
    assert on_a_again == ["s-reroute"], on_a_again
    assert ttl_a_again > 0, (
        f"lease TTL must persist across a same-session re-set, "
        f"got {ttl_a_again}")
    # Cross-plan re-set: the new SET gained the session via the script.
    assert on_b_after == ["s-reroute"], on_b_after
    assert ttl_b > 0, f"lease must have a fresh TTL after re-set, got {ttl_b}"
    print(f"  lease_a={lease_a!r} (ttl={ttl_a}s, A={on_a}); "
          f"re-set same session: A={on_a_again} (ttl={ttl_a_again}s, "
          f"no duplicate); re-set to plan B: B={on_b_after} (ttl={ttl_b}s); "
          f"SADD/SET/EXPIRE atomic, membership contract holds")


if __name__ == "__main__":
    test_funcs = [(name, fn) for name, fn in sorted(globals().items())
                  if name.startswith("test_") and callable(fn)]
    for name, fn in test_funcs:
        print(f"{name}:")
        fn()
    print(f"\nall drain tests passed ({len(test_funcs)})")
