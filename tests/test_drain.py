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

    The migration must NOT leave an in-flight slot claim on the sibling:
    `picker.pick` claims a slot the same way a real request would (the
    picker is what knows how to claim, and the migrator wants the same
    answer about which ref is reachable), but no real request is in
    flight, so the claim has to be released by the migrator. Without that
    release each migrated session leaks one inflight slot on the sibling
    plan until the staleness sweep notices — which is exactly what
    strangles the next migration that lands on the same plan. The lease
    is the persistent record; the slot claim is the temporary one and is
    gone after the migration finishes.
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

        # After migration the slot table must be clean on every plan the
        # migration touched. The drained plan was never claimed during the
        # migrate loop (its refs are excluded from the pick), so its
        # inflight count started and ends at zero. The sibling plan was
        # the destination of every migrated pick; without the explicit
        # release the migrator added, this would be 3 — one phantom slot
        # per migrated session, visible to the next picker that walks
        # this plan until the inflight staleness sweep (15 minutes)
        # noticed. The `other` plan is the lane member that was never
        # chosen, included here to pin the "no pan-planet claim" contract.
        inflight_drained = await slots.in_flight(drained_plan)
        inflight_sibling = await slots.in_flight(sibling.plan_key)
        inflight_other = await slots.in_flight(other.plan_key)

        return (before_drained, count, migrated, after_drained,
                new_leases, other_lease, sibling_lease,
                drained_plan, target, sibling, other, drained_refs,
                inflight_drained, inflight_sibling, inflight_other)

    (before_drained, count, migrated, after_drained,
     new_leases, other_lease, sibling_lease,
     drained_plan, target, sibling, other, drained_refs,
     inflight_drained, inflight_sibling, inflight_other) = _run(go())

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
    # Slot table cleanliness: no leaked inflight claim on any plan the
    # migration touched. The sibling plan is the important one — three
    # migrated sessions claim-and-release a slot each on the way past,
    # and the regression we pin is exactly that the count is back at 0
    # once the lease rewrites are done.
    assert inflight_drained == 0, (
        f"drained plan {drained_plan} must have no in-flight claims after "
        f"migration; got {inflight_drained}")
    assert inflight_sibling == 0, (
        f"sibling plan {sibling.plan_key} must have no in-flight claims "
        f"after migration (the picker claim from each migrated session "
        f"must be released by migrate itself; the lease is the persistent "
        f"record, the claim is not); got {inflight_sibling}")
    assert inflight_other == 0, (
        f"untouched plan {other.plan_key} must have no in-flight claims "
        f"after migration; got {inflight_other}")
    print(f"  migrated {count} sessions off {drained_plan}: "
          f"{sorted(new_leases.items())}; other untouched ({other_lease}), "
          f"sibling untouched ({sibling_lease}); drained SET cleared; "
          f"in_flight drained/sibling/other = "
          f"{inflight_drained}/{inflight_sibling}/{inflight_other}")


def test_drain_migrate_continues_past_release_failure():
    """A Redis error in `picker.release` after a successful pick must NOT
    abort the migrate loop — every not-yet-processed session would be
    orphaned on the plan `apply.sh` is about to take out of service, which
    is strictly worse than the leak being fixed.

    The pick itself already wrote the lease on the sibling (the picker's
    `_visit_ref` calls `set_lease` before returning the `Pick`), so the
    orphaned state is just one stale claim — the staleness sweep reclaims
    it within `inflight_max_age_seconds`. The remaining sessions still
    get migrated.

    We patch `picker.release` on the instance to throw on the first call
    only: the loop must log the failure and continue to the next two
    sessions, all three of which must have reached the `release` call (i.e.
    the loop did NOT abort partway through). SET iteration order is not
    guaranteed across runs, so the assertions check the COUNT and the
    presence/absence of the failed session in `migrated`, not specific
    session ids.
    """
    async def go():
        reg, slots, picker = _build()
        lane = "forge"
        target = next(m for m in reg.lane_members(lane)
                      if m.plan_key == "minimax-ultra")
        drained_plan = target.plan_key
        ttl = reg.settings.lease_ttl_seconds

        # Three sessions on the drained plan.
        await slots.set_lease("s-drain-1", target.ref, ttl, drained_plan)
        await slots.set_lease("s-drain-2", target.ref, ttl, drained_plan)
        await slots.set_lease("s-drain-3", target.ref, ttl, drained_plan)

        # First call to release throws; calls 2 and 3 go through to the
        # real implementation. Tracking the call count also pins the
        # "loop did not abort" contract — three picks means three release
        # attempts, so the loop saw all three sessions.
        original_release = picker.release
        calls: list[tuple[str, str, str]] = []
        async def failing_release(plan_key, request_id, model_ref):
            calls.append((plan_key, request_id, model_ref))
            if len(calls) == 1:
                raise ConnectionError("simulated redis drop")
            return await original_release(
                plan_key, request_id, model_ref)
        picker.release = failing_release

        count, migrated = await migrate(drained_plan, slots, picker,
                                        ttl=ttl)
        # The pick for the session that hit the failing release already
        # wrote its lease on the sibling — confirm it landed on a non-
        # drained ref so the only orphaned state is the stale claim.
        leases = {s: await slots.get_lease(s)
                  for s in ("s-drain-1", "s-drain-2", "s-drain-3")}
        return (count, sorted(migrated), calls, drained_plan, leases,
                target.plan_key)

    (count, migrated, calls, drained_plan, leases,
     drained_ref) = _run(go())
    assert count == 2, f"expected 2 successful migrations, got {count}"
    assert len(migrated) == 2, migrated
    # All three picks must have reached release: the loop did not abort.
    assert len(calls) == 3, (
        f"loop must NOT abort on a release failure: expected 3 release "
        f"calls (one per pick), got {len(calls)}")
    # The session whose release threw is the one whose pick happened first
    # in SET iteration order — i.e. the session NOT in `migrated`. Its
    # lease must still be on a non-drained ref (the picker's `_visit_ref`
    # wrote it before the failing release), so the only orphaned state is
    # one stale claim.
    failed = next(s for s in ("s-drain-1", "s-drain-2", "s-drain-3")
                  if s not in migrated)
    assert failed in leases and "/" in (leases[failed] or ""), (
        f"failed session {failed!r} must have a lease on a non-drained "
        f"ref (pick wrote it before release threw), got "
        f"{leases[failed]!r}")
    assert leases[failed].split("/", 1)[0] != drained_ref, (
        f"failed session {failed!r} lease must not be on the drained "
        f"plan {drained_ref!r}; got {leases[failed]!r}")
    # The two migrated sessions have leases on the sibling (drained plan
    # cleared, drain migrate re-leased them).
    for s in migrated:
        assert s in leases and leases[s] is not None, (
            f"migrated session {s!r} has no lease after migrate")
        assert leases[s].split("/", 1)[0] != drained_ref, (
            f"migrated session {s!r} lease must not be on the drained "
            f"plan {drained_ref!r}; got {leases[s]!r}")
    print(f"  drained {drained_plan}: release threw on {failed!r} "
          f"(1st pick), migrated {count}/3 sessions ({migrated}), "
          f"all 3 picks reached release (loop did not abort); {failed!r} "
          f"lease on {leases[failed]!r} — only the claim is orphaned "
          f"(sweep reclaims)")


def test_drain_migrate_continues_past_set_lease_failure():
    """A Redis error in the explicit `slots.set_lease` re-write after a
    successful pick+release must NOT abort the migrate loop — same
    contract as the release-fail case, different dirty state.

    The pick's `_visit_ref` already wrote the lease on the sibling, and
    `picker.release` already reclaimed the slot. The only consequence of
    this failure is that the lease ages out at the registry default TTL
    instead of the operator's `--ttl`. No dirty state remains.

    We patch `slots.set_lease` on the instance to throw on the first call
    only: the loop must log the failure and continue to the next two
    sessions, all three of which must have reached `set_lease` (i.e. the
    loop did NOT abort partway through). SET iteration order is not
    guaranteed across runs, so the assertions check the COUNT and the
    presence/absence of the failed session in `migrated`, not specific
    session ids. Every session still has its lease on the sibling — only
    the failed one's TTL is the registry default rather than `ttl`.
    """
    async def go():
        reg, slots, picker = _build()
        lane = "forge"
        target = next(m for m in reg.lane_members(lane)
                      if m.plan_key == "minimax-ultra")
        sibling = next(m for m in reg.lane_members(lane)
                       if m.plan_key != target.plan_key
                       and not reg.is_tail(lane, m.ref))
        drained_plan = target.plan_key
        ttl = reg.settings.lease_ttl_seconds

        # Three sessions on the drained plan.
        await slots.set_lease("s-drain-1", target.ref, ttl, drained_plan)
        await slots.set_lease("s-drain-2", target.ref, ttl, drained_plan)
        await slots.set_lease("s-drain-3", target.ref, ttl, drained_plan)

        # Fail on every even-numbered call. With one session per pick and
        # one set_lease call per pick in `_visit_ref`, the per-session
        # pattern is "picker's set_lease (odd N), migrate's explicit
        # set_lease (even N)". Failing the migrate-side calls exercises
        # the new try/except without breaking the picker's internal call
        # (which would turn this test into a `picker.pick`-failure test
        # via the per-lane SKIP branch).
        original_set_lease = slots.set_lease
        calls: list[tuple[str, str, int, str]] = []
        async def failing_set_lease(session, ref, lease_ttl, plan_key):
            calls.append((session, ref, lease_ttl, plan_key))
            if len(calls) % 2 == 0:
                raise ConnectionError("simulated redis drop")
            return await original_set_lease(
                session, ref, lease_ttl, plan_key)
        slots.set_lease = failing_set_lease

        count, migrated = await migrate(drained_plan, slots, picker,
                                        ttl=ttl)
        # All three picks reached release (succeeded) and the migrate-
        # side set_lease (failed). All three sessions are effectively
        # migrated — only the lease TTL is the registry default rather
        # than `ttl`. The sibling's in-flight count must be 0: release
        # ran for all three.
        leases = {s: await slots.get_lease(s)
                  for s in ("s-drain-1", "s-drain-2", "s-drain-3")}
        inflight_sibling = await slots.in_flight(sibling.plan_key)
        return (count, sorted(migrated), calls, drained_plan, leases,
                target.plan_key, inflight_sibling)

    (count, migrated, calls, drained_plan, leases, drained_ref,
     inflight_sibling) = _run(go())
    # All three migrate-side set_lease calls failed, so nothing landed
    # in `migrated` — but every session still has a lease on the sibling
    # (the picker's `_visit_ref` wrote it) and the loop saw all three
    # (the per-lane try/except did NOT short-circuit it).
    assert count == 0, f"expected 0 successful migrations, got {count}"
    assert migrated == [], migrated
    # The picker's _visit_ref calls set_lease once per session; the
    # migrate loop's explicit set_lease is the second call per session.
    # Three sessions → 6 set_lease calls total. The loop did not abort:
    # if it had, calls would have stopped short of 6.
    assert len(calls) == 6, (
        f"loop must NOT abort on a set_lease failure: expected 6 "
        f"set_lease calls (1 picker + 1 migrate per session × 3 "
        f"sessions), got {len(calls)}")
    # All three sessions have leases on a non-drained ref — the picker's
    # _visit_ref wrote them before migrate's explicit re-write threw.
    for s in ("s-drain-1", "s-drain-2", "s-drain-3"):
        assert s in leases and "/" in (leases[s] or ""), (
            f"session {s!r} must have a lease (pick wrote it), got "
            f"{leases[s]!r}")
        assert leases[s].split("/", 1)[0] != drained_ref, (
            f"session {s!r} lease must not be on the drained plan "
            f"{drained_ref!r}; got {leases[s]!r}")
    # Release ran for all three sessions — sibling in_flight is 0.
    assert inflight_sibling == 0, (
        f"sibling in_flight must be 0 after migrate (release ran for all "
        f"3 sessions), got {inflight_sibling}")
    print(f"  drained {drained_plan}: migrate's explicit set_lease threw "
          f"for all 3 sessions; loop saw all 3 ({len(calls)} set_lease "
          f"calls — no abort); all 3 leases on sibling from pick "
          f"({leases}); in_flight sibling={inflight_sibling} (release "
          f"ran for all 3) — no dirty state remains")


def test_drain_migrate_skips_stale_reverse_index_member():
    """A session that is still in the drained plan's reverse-index SET but
    whose lease is gone (TTL fired) or has moved to a non-drained ref
    (a parallel turn re-leased it elsewhere) must be SKIPPED + SREMmed
    from the SET, not picked/migrated.

    Two ways a SET entry can be stale at this point:

      (a) lease TTL fired without the SET entry's matching TTL being
          collected yet (the SET TTL equals the lease TTL, but the
          collection is lazy on read — the next `sessions_on_plan` call
          could observe either or both);
      (b) a parallel turn re-leased the session onto a different plan
          (drain migration on a sibling, picker body-walk swapping lanes,
          hooks re-leasing after a hard rejection) and the SET entry got
          carried over.

    Picking and migrating the stale session would call `picker._visit_ref`
    -> `set_lease` on a sibling, writing a fresh lease the session does
    NOT own (its real lease, when present, is on a different ref). The
    drained plan's SET would still hold the phantom, and a future
    `drain.migrate` would re-pick the same phantom — a phantom-loop bug.

    The fix: drain.migrate reads `slots.get_lease(session)` for every SET
    member and SREMs the stale entry via `slots.forget_sessions_on_plan`,
    printing a SKIP line to stderr. Live sessions (lease on a drained ref)
    are migrated as before. The drained plan's SET ends empty.

    The phantom SET entries are seeded directly via `redis.sadd` because
    `set_lease`'s atomic SREM/SADD would otherwise have cleaned them up
    — the test exercises the gate by deliberately bypassing the
    cross-plan hygiene that the new `_SET_LEASE` script provides.
    """
    async def go():
        reg, slots, picker = _build()
        lane = "forge"
        target = next(m for m in reg.lane_members(lane)
                      if m.plan_key == "minimax-ultra")
        sibling = next(m for m in reg.lane_members(lane)
                       if m.plan_key != target.plan_key
                       and not reg.is_tail(lane, m.ref))
        drained_plan = target.plan_key
        ttl = reg.settings.lease_ttl_seconds

        # Live member: lease on the drained plan. Will be migrated.
        await slots.set_lease("s-live", target.ref, ttl, drained_plan)

        # Stale member #1: lease TTL fired without the SET entry being
        # collected yet. We seed the SET directly via `redis.sadd` to
        # simulate the lazy-collection race window.
        await slots.redis.sadd(
            f"sy:lease_plan:{drained_plan}", "s-ghost-1")

        # Stale member #2: lease on a NON-drained ref (a parallel turn
        # re-leased it onto `sibling` via set_lease), but the SET entry
        # on the drained plan was carried over. We seed the phantom
        # directly because `set_lease`'s cross-plan SREM would otherwise
        # have cleaned it up — the test exercises the drain gate by
        # deliberately bypassing the SET_LEASE hygiene.
        await slots.set_lease("s-moved", sibling.ref, ttl, sibling.plan_key)
        await slots.redis.sadd(
            f"sy:lease_plan:{drained_plan}", "s-moved")

        before_drained = sorted(await slots.sessions_on_plan(drained_plan))

        count, migrated = await migrate(drained_plan, slots, picker, ttl=ttl)

        after_drained = sorted(await slots.sessions_on_plan(drained_plan))
        new_leases = {s: await slots.get_lease(s)
                      for s in ("s-live", "s-ghost-1", "s-moved")}
        return (before_drained, count, sorted(migrated), after_drained,
                new_leases, target, sibling)

    (before_drained, count, migrated, after_drained, new_leases,
     target, sibling) = _run(go())

    # Before migrate: 3 SET members (1 live + 2 stale).
    assert before_drained == ["s-ghost-1", "s-live", "s-moved"], \
        before_drained
    # Migrate counts ONLY the live session — stale ones are skipped before
    # the pick call, so they never enter the migration path.
    assert count == 1, f"expected 1 migration (live only), got {count}"
    assert migrated == ["s-live"], migrated
    # After migrate: drained plan's SET is empty. The live was migrated
    # off it; on FakeRedis the SREM came from `picker._affinity`'s
    # `drop_lease` (the `_shadow_drop_lease` shadow already mirrors
    # the `SREM sy:lease_plan:{plan}`), and on real Redis the
    # `_SET_LEASE` cross-plan SREM is a no-op for the live session
    # because `_affinity` already dropped the lease by then (the
    # subsequent `_visit_ref` set_lease's old-lease GET returns nil).
    # drain.migrate's explicit set_lease at the end is same-plan
    # (new ref's plan == placed.plan.key), so the cross-plan SREM is
    # closed on that call too. The stale members were SREMmed by
    # drain.migrate's hygiene gate via `slots.forget_sessions_on_plan`.
    assert after_drained == [], after_drained
    # Live session: lease moved off the drained plan.
    assert new_leases["s-live"] is not None, new_leases["s-live"]
    assert new_leases["s-live"].split("/", 1)[0] != target.plan_key, (
        f"live session s-live must have been migrated off {target.plan_key}; "
        f"got {new_leases['s-live']!r}")
    # Stale session #1: still has no lease — drain.migrate did not write one.
    assert new_leases["s-ghost-1"] is None, (
        f"stale s-ghost-1 had no lease before migrate and must not gain one; "
        f"got {new_leases['s-ghost-1']!r}")
    # Stale session #2: lease is unchanged on `sibling.ref` (drain.migrate
    # did not touch it; the parallel turn that re-leased it owns that lease).
    assert new_leases["s-moved"] == sibling.ref, (
        f"stale s-moved lease must be untouched (still on the parallel "
        f"turn's ref {sibling.ref!r}); got {new_leases['s-moved']!r}")
    print(f"  drained {target.plan_key}: stale members (s-ghost-1, s-moved) "
          f"SKIPPED + SREMmed by drain gate; live s-live migrated to "
          f"{new_leases['s-live']}; after_drained={after_drained}; "
          f"s-moved lease untouched ({new_leases['s-moved']})")


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
      (b) the SET key holds the session on the NEW plan,
      (c) the script's SADD does not duplicate the membership on a
          same-session re-set (SADD returns 0 for an already-present
          member — the atomicity shape the Lua guarantees),
      (d) on a cross-plan re-set the OLD plan's SET is empty after
          the script (cross-plan SREM fired inside the same Lua), so
          a `sessions_on_plan(old_plan)` walk returns `[]` and the
          drain migrator finds no phantoms.

    The fake cannot directly simulate the partial-failure the script
    closes (the SET-EXPIRE-after-SADD network drop — see the
    `_shadow_set_lease` comment in `tests/fake_redis.py`), so the
    "atomicity holds across network failures" property is
    real-Redis-only. The fake does mirror the cross-plan SREM (cycle-4
    uplift of `_shadow_set_lease`); the Lua-only pin in
    `tests/test_slots_lua.py::test_set_lease_cross_plan_srem` is the
    authoritative source for the real-Redis contract on (d).
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
        # And again on a DIFFERENT plan — the new `_SET_LEASE` script
        # SREMs the session from the OLD plan's reverse-index SET in the
        # same atomic script (cross-plan hygiene), then SADDs to the new
        # SET. Asserted here on FakeRedis (the matrix backend), with the
        # cross-plan SREM shadow mirroring the Lua body
        # (`tests/fake_redis.py:_shadow_set_lease`); the Lua-only pin in
        # `tests/test_slots_lua.py::test_set_lease_cross_plan_srem` is the
        # authoritative source for the real-Redis contract.
        await slots.set_lease("s-reroute", "minimax-ultra/m3", ttl, plan_b)
        on_a_after_cross = sorted(await slots.sessions_on_plan(plan_a))
        on_b_after = sorted(await slots.sessions_on_plan(plan_b))
        ttl_b = await slots.redis.ttl("sy:lease:s-reroute")

        return (lease_a, on_a, ttl_a,
                on_a_again, ttl_a_again,
                on_a_after_cross, on_b_after, ttl_b)

    (lease_a, on_a, ttl_a,
     on_a_again, ttl_a_again,
     on_a_after_cross, on_b_after, ttl_b) = _run(go())
    assert lease_a == "claude-max/fable", lease_a
    assert on_a == ["s-reroute"], on_a
    assert ttl_a > 0, f"lease must have a TTL after set_lease, got {ttl_a}"
    # Same-session re-set: SET has exactly one entry, no duplicate.
    assert on_a_again == ["s-reroute"], on_a_again
    assert ttl_a_again > 0, (
        f"lease TTL must persist across a same-session re-set, "
        f"got {ttl_a_again}")
    # Cross-plan re-set: the new SET gained the session via the script,
    # and the OLD plan's SET is empty (cross-plan SREM fired). Both
    # backends honor this — FakeRedis via `_shadow_set_lease`, real
    # Redis via the Lua body pinned in
    # `tests/test_slots_lua.py::test_set_lease_cross_plan_srem`.
    assert on_a_after_cross == [], on_a_after_cross
    assert on_b_after == ["s-reroute"], on_b_after
    assert ttl_b > 0, f"lease must have a fresh TTL after re-set, got {ttl_b}"
    print(f"  lease_a={lease_a!r} (ttl={ttl_a}s, A={on_a}); "
          f"re-set same session: A={on_a_again} (ttl={ttl_a_again}s, "
          f"no duplicate); re-set to plan B: A={on_a_after_cross}, "
          f"B={on_b_after} (ttl={ttl_b}s); cross-plan SREM atomic, "
          f"membership contract holds")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
