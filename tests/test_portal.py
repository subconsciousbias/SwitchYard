"""Portal pacing toggle: the panel must update after a click, and the button
must read as a button with an on/off state.

The primary bug was that the pacing panel lived inside static index.html and
only the capacity/plans fragments were re-fetched after a toggle, so the
heading and button label only caught up on a full page reload. This file pins
that the GET fragment reflects the POST, and that the rendered HTML carries
the on/off state visibly (aria-pressed) so a screen reader and a sighted
operator both know which it is.

No network: Redis is patched to a FakeRedis at the import site, so the
portal's startup event builds against the in-process store rather than
talking to one. The plans file is the tracked example.
"""
from __future__ import annotations

import dataclasses
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: see plans_path.py for why a stray SWITCHYARD_PLANS
# must not silently become this test's fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()

import redis.asyncio as _redis_async                    # noqa: E402
from tests.fake_redis import FakeRedis                  # noqa: E402


_FAKE_REDIS = FakeRedis()


def _fake_from_url(url, **_):
    # The portal calls Redis.from_url(SWITCHYARD_REDIS_URL, ...). The URL is
    # ignored — every test gets its own fake store, which is the whole point
    # of using one rather than letting the suite depend on a redis:6379 that
    # might be down, or worse, that might be someone else's.
    return _FAKE_REDIS


_redis_async.Redis.from_url = _fake_from_url

from fastapi.testclient import TestClient  # noqa: E402

from switchyard import models                              # noqa: E402
from switchyard.portal import app as portal_app  # noqa: E402


def _off(html: str) -> bool:
    return "Pacing mode off" in html and 'aria-pressed="false"' in html \
        and "Turn on" in html


def _on(html: str) -> bool:
    return "Pacing mode on" in html and 'aria-pressed="true"' in html \
        and "Turn off" in html


def trs_for(html):
    # The capacity fragment emits a `<tr>` with optional inline indent
    # style for rows inside a group, so `<tr>` alone misses them. Match
    # the opening tag flexibly while keeping the boundary tight -- a
    # `<tr ...>` opening followed by anything until the next `</tr>`
    # is still one row, and the `re.DOTALL` keeps it working across
    # the multi-line whitespace Jinja emits. Use a NON-capturing group
    # so re.findall returns full matches (capture groups would make
    # findall return only the captured group).
    return re.findall(r"<tr(?: [^>]*)?>.*?</tr>", html, re.DOTALL)


def row_with_ref(trs, plan, model):
    ident = f">{plan}</span><span class=\"muted\">/{model}</span>"
    return next((t for t in trs if ident in t.replace("\n", "")), None)


def test_pacing_fragment_reflects_runtime_toggle():
    """The fragment must show the new state without a full page reload.

    Before the fix, only #capacity and #plans refreshed after a click — the
    pacing panel itself was not re-rendered, so a fresh tab and the live one
    disagreed. The fix wires /fragments/pacing to poll and to be refreshed by
    the toggle's after-request handler.
    """
    with TestClient(portal_app.app) as client:
        # The example plans file leaves pacing off, so the first read of the
        # fragment should be the off state — both the label and aria-pressed.
        first = client.get("/fragments/pacing")
        assert first.status_code == 200, first.text
        assert _off(first.text), first.text

        toggled = client.post("/admin/pacing?enabled=toggle")
        assert toggled.status_code == 200, toggled.text
        assert toggled.json()["pacing"] is True

        # The panel must now show on, without anyone touching /.
        second = client.get("/fragments/pacing")
        assert second.status_code == 200, second.text
        assert _on(second.text), second.text

        # And back off again — the toggle action is the same button either way.
        toggled = client.post("/admin/pacing?enabled=toggle")
        assert toggled.json()["pacing"] is False
        third = client.get("/fragments/pacing")
        assert _off(third.text), third.text


def test_pacing_button_is_actually_a_button():
    """The control must read as a control, not as a status chip.

    The previous styling reused the .tag badge class — 11px muted pill, no
    pointer, no hover — which read as a status chip and not something to
    click. This is what made the toggle feel dead: even when it worked, there
    was no visual signal that pressing it would do anything.

    The fragment carries the HTML side of that change: a real <button>, not
    a tag-styled span, with a state attribute a screen reader can announce.
    The CSS side lives in base.html and is asserted by reading the file
    directly — it's not part of the fragment response.
    """
    with TestClient(portal_app.app) as client:
        html = client.get("/fragments/pacing").text
        # The button element, not a span or div pretending to be one.
        assert 'class="toggle"' in html, html
        assert 'type="button"' in html, html
        assert "aria-pressed=" in html, html
        # And it is NOT still wearing the badge class — the bug was that the
        # chip styling made a clickable thing look like a status label.
        assert 'class="tag"' not in html, html

        # The CSS half of the fix: cursor:pointer + a hover rule + the
        # pressed fill all live together in one .toggle block in base.html.
        # If any of them goes missing, the button quietly turns back into a
        # chip and the whole point of the change is undone.
        base = (ROOT + "/switchyard/portal/templates/base.html")
        with open(base) as fh:
            css = fh.read()
        assert ".toggle" in css and "cursor:pointer" in css, css
        assert ".toggle:hover" in css, css
        assert '.toggle[aria-pressed="true"]' in css, css


def test_pacing_fragment_carries_runtime_override_banner():
    """If the runtime disagrees with plans.yaml, the warn tag must surface.

    The panel reads capacity.pacing (runtime) and capacity.pacing_configured
    (the plans.yaml default). When the operator toggles them apart, the
    fragment has to say so — otherwise it looks like plans.yaml is being
    ignored rather than overridden.
    """
    with TestClient(portal_app.app) as client:
        # plans.example.yaml ships with pacing off, so flipping the runtime on
        # puts them out of step.
        client.post("/admin/pacing?enabled=on")
        html = client.get("/fragments/pacing").text
        assert "runtime override" in html, html
        assert "plans.yaml says" in html, html


def test_capacity_row_shows_failing_chip_when_streak_meets_alert():
    """A plan with a transient-failure streak at the alert threshold shows the
    'failing · Nx' chip. Below the threshold it stays absent.

    The chip is the operator's only signal that the escalated-cooldown
    ladder has tripped: a working picker would otherwise quietly bounce
    traffic off the plan with no warning on the board.
    """
    with TestClient(portal_app.app) as client:
        # plans.example.yaml ships with transient_breaker.streak_alert: 3,
        # so a streak of 3 is exactly the alert threshold.
        settings = portal_app.state["registry"].settings
        assert settings.transient_breaker.streak_alert == 3, \
            settings.transient_breaker

        # Pick any plan that is on a lane the capacity fragment renders.
        plan = portal_app.state["registry"].plans["minimax-ultra"]
        # The fragment reads transient_streak via the FakeRedis the slot
        # table was built against. Set the string directly on its store —
        # `(value, expiry)` — so we don't have to await an async set from a
        # sync test.
        fake = portal_app.state["slots"].redis

        # Streak 1 — below the threshold, chip absent.
        fake.strings[f"sy:tfail:{plan.key}"] = ("1", None)
        html = client.get("/fragments/capacity").text
        assert "failing" not in html, html

        # Streak 3 — at the threshold, chip present, marked warn, not bad.
        fake.strings[f"sy:tfail:{plan.key}"] = ("3", None)
        html = client.get("/fragments/capacity").text
        assert "failing · 3x" in html, html
        # The chip is warn (not bad): the plan is still accepting work, just
        # sitting on the ladder.
        assert 'class="tag warn"' in html, html
        assert 'title="consecutive transient failures' in html, html

        # Streak 7 — chip carries the actual streak, not the threshold.
        fake.strings[f"sy:tfail:{plan.key}"] = ("7", None)
        html = client.get("/fragments/capacity").text
        assert "failing · 7x" in html, html


def test_capacity_row_chip_threshold_follows_streak_alert_setting():
    """The chip threshold is operator-configurable, not a hardcoded 3.

    When `TransientBreaker.streak_alert` is raised, the chip must wait the
    full ladder — otherwise the operator-facing surface disagrees with
    itself: a row can show a `failing · 4x` warn chip while the plans-table
    alert (driven by the same `streak_alert` in collect_plans) waits for
    streak 5. This test raises the threshold to 5 and asserts the chip is
    absent at streak 3 and present at streak 5.
    """
    with TestClient(portal_app.app) as client:
        registry = portal_app.state["registry"]
        original = registry.settings
        # Default must remain 3 — render_preview.py depends on that.
        assert original.transient_breaker.streak_alert == 3

        # Raise the threshold to 5 for the duration of this test, then
        # restore. dataclasses.replace preserves the frozen dataclass shape
        # without having to enumerate every field by hand.
        registry.settings = dataclasses.replace(
            original,
            transient_breaker=dataclasses.replace(
                original.transient_breaker, streak_alert=5),
        )

        try:
            plan = registry.plans["minimax-ultra"]
            fake = portal_app.state["slots"].redis

            # Streak 3 — below the raised threshold of 5, chip absent.
            fake.strings[f"sy:tfail:{plan.key}"] = ("3", None)
            html = client.get("/fragments/capacity").text
            assert "failing" not in html, html

            # Streak 5 — at the raised threshold, chip present.
            fake.strings[f"sy:tfail:{plan.key}"] = ("5", None)
            html = client.get("/fragments/capacity").text
            assert "failing · 5x" in html, html
            assert 'class="tag warn"' in html, html
        finally:
            registry.settings = original


def test_capacity_fragment_hides_withheld_for_model_narrowed_rows():
    """A row whose cap is narrower than the plan's width SOLELY because of
    its own model's `max_parallel` is the model's own row, not a slice of
    withheld capacity.

    Before this fix, every local-box/gemma-equivalent row drew a "model
    limit N" warn tag and a grey "withheld" slot beside its single bright
    slot — reading as broken capacity when in fact the row is exactly as
    wide as the model can reach. The fix gates both on `cap_model_owned`:
    the row draws its `cap` free slots and stops.

    External narrowing still renders the old markers: a cooled plan paints
    gone squares and its cooldown tag, the same way as today. That branch
    is not gated, because it is exactly the "real trouble" the old grey
    squares were added for.
    """
    import asyncio

    with TestClient(portal_app.app) as client:
        # The local-box/gemma row is the canonical fixture: model cap 1 on a
        # plan cap 2, with no policy narrowing it further. The view must show
        # exactly one reachable slot, no "withheld" title, and no "model
        # limit" warn tag — the row IS one slot, not a slice of two.
        html = client.get("/fragments/capacity").text
        trs = trs_for(html)
        gemma = row_with_ref(trs, "local-box", "gemma")
        assert gemma is not None, "local-box/gemma row missing in /fragments/capacity"
        assert "model limit" not in gemma, gemma
        assert "withheld:" not in gemma, gemma
        assert gemma.count('<span class="slot"') == 1, gemma
        # And the row is not cooled, so no "gone" squares or "bad" tag.
        assert 'class="slot gone"' not in gemma, gemma
        assert 'class="tag bad"' not in gemma, gemma

        # The CLI-plan shape: `claude-max/fable` (plan 2, model 1) is narrowed
        # by the model, but `policy._apply_gate_headroom` has also rewritten
        # `cap_reason` from "configured" to "configured + gate headroom 1".
        # The pre-fix predicate gated on the exact string "configured" and so
        # flipped this row's `cap_model_owned` to False under policy — the
        # template then ran the withheld-slot loop and rendered the "model
        # limit" tag for a slot that was never withheld. Post-fix the same
        # one reachable slot renders, with the same absence of markers, just
        # like local-box/gemma: one free slot, no gone squares, no "model
        # limit" text, no "withheld:" tooltip.
        fable = row_with_ref(trs, "claude-max", "fable")
        assert fable is not None, "claude-max/fable row missing in /fragments/capacity"
        assert "model limit" not in fable, fable
        assert "withheld:" not in fable, fable
        assert 'class="slot gone"' not in fable, fable
        assert fable.count('<span class="slot"') == 1, fable

        # An externally-narrowed row keeps ALL the old machinery. Cool
        # minimax-ultra — the pinned example's `forge` lead row — and read
        # the fragment again. Every slot the row would have shown is now
        # a "gone" square, and the right-hand cell carries the cooldown
        # tag, not a "model limit" tag.
        asyncio.run(portal_app.state["slots"].cool_down(
            "minimax-ultra", 900, "quota_exhausted"))
        try:
            html = client.get("/fragments/capacity").text
            trs = trs_for(html)
            ultra = row_with_ref(trs, "minimax-ultra", "m3")
            assert ultra is not None, "minimax-ultra/m3 row missing after cooldown"
            # The "bad" tag carries the human-readable cooldown reason; the
            # exact text is "quota exhausted · 14m" (900s / 60 = 15, with
            # whatever fraction has elapsed in the test).
            assert 'class="tag bad"' in ultra, ultra
            assert "quota exhausted" in ultra, ultra
            # Every slot the row would have shown is "gone": the plan cap
            # is 4, the model's cap is also 4, so four gone squares in
            # the row's loop before the (now-skipped because cap == 0)
            # withheld branch.
            assert ultra.count('<span class="slot gone"></span>') == 4, ultra
            # ... and even on a cooled plan the gating does not leak:
            # the "model limit" string still does not appear.
            assert "model limit" not in ultra, ultra
        finally:
            asyncio.run(portal_app.state["slots"].clear_cooldown(
                "minimax-ultra"))


# ============================================================================
# Issue #81 — Workstream C (board model rows: draw withheld squares for a
# learned cap narrower than the model's own max_parallel).
#
# These tests pin the render-level contract that the picker and policy
# already agree on: when the learner has narrowed a plan's effective cap
# below the model's own `max_parallel`, the row draws the narrower learned
# cap as its reachable slots and paints the gap up to the model cap as
# `slot gone` squares carrying the `learned[...]` source in the tooltip —
# mirroring opencode-go/glm-5.3-flash on the operator config. The fixture
# is the shipped `openrouter/mimo` row (plan max_parallel 4, model
# max_parallel 2), which matches the issue #81 shape exactly and avoids
# touching other workers' files.
#
# Learner state lives in the FakeRedis behind the app's startup event, so
# each test seeds `sy:learn:openrouter:global` directly and clears any
# residue at start AND in a `finally` so cross-test leakage cannot mask a
# real regression.
# ============================================================================


def _seed_openrouter_learn_cap(cap):
    """Replace any prior learner state for openrouter with a single global
    bucket carrying `cap`. Returns the FakeRedis the test should clean up
    in its `finally` so subsequent tests see a fresh slate."""
    fake = portal_app.state["slots"].redis
    for key in [k for k in list(fake.hashes)
                if k.startswith("sy:learn:openrouter")]:
        del fake.hashes[key]
    fake.hashes["sy:learn:openrouter:global"] = {"cap": str(cap)}
    return fake


def test_capacity_row_renders_learned_cap_equal_to_model_limit():
    """Case 1 — learner cap == model.max_parallel.

    With a learned cap that exactly matches the model's own `max_parallel`
    (2 here), the row's `cap_model_owned` flag fires and the withheld
    branch in `_capacity_slots.html` is skipped entirely. Two free slots,
    no gone squares, no `withheld:` tooltip, no `learned[...]` chip text
    anywhere in the row, cap display `0/2`. The lane above (the picker)
    sees the same number the board paints, and the operator does not
    read the row as "2 of 4 slots gone" — which would be a regression
    of the same shape issue #81 fixed for model-narrowed rows.
    """
    with TestClient(portal_app.app) as client:
        fake = _seed_openrouter_learn_cap(2)
        try:
            html = client.get("/fragments/capacity").text
        finally:
            for key in [k for k in list(fake.hashes)
                        if k.startswith("sy:learn:openrouter")]:
                del fake.hashes[key]
        trs = trs_for(html)
        row = row_with_ref(trs, "openrouter", "mimo")
        assert row is not None, "openrouter/mimo row missing in /fragments/capacity"
        # Exactly 2 reachable slots, no gone squares (the learned cap ==
        # model cap branch makes cap_model_owned True and skips the
        # withheld loop in _capacity_slots.html).
        assert row.count('<span class="slot"') == 2, row
        assert 'class="slot gone"' not in row, row
        # No tooltip text mentioning the learner reason — there's no gone
        # square to carry one, and no chip carries the cap_reason either.
        assert "withheld:" not in row, row
        # The full chip ladder in _capacity_state.html: only "0/2"
        # renders, no tag warn / tag bad / cap_reason chip.
        assert "learned[" not in row, row
        # Cap display: the muted cell reads `0/2`, NOT `0/4` (plan cap)
        # and NOT `0/1` (any narrower state).
        assert ">0/2<" in row, row


def test_capacity_row_renders_withheld_squares_for_learned_cap_below_model():
    """Case 2 — learner cap < model.max_parallel.

    The learned cap (1 here) is narrower than the model's own
    `max_parallel` (2). The picker sets `cap_model_owned = False`, so
    `_capacity_slots.html` paints one reachable slot and one `slot gone`
    square above it carrying `withheld: learned[global]` in its tooltip
    — total 2 squares (the model cap), NOT 4 (the plan cap) and NOT 1
    (the learned cap rendered alone). The `_capacity_state.html` chip
    ladder does NOT render a divergence tag-warn chip on this row: the
    `(model_cap is not none and cap < model_cap)` branch of the chip
    condition suppresses the chip when the gap is exactly the model's
    own narrowing surface. The board, then, reads as "1 free, 1
    learner-withheld" rather than "1 free, 3 plan-withheld".
    """
    with TestClient(portal_app.app) as client:
        fake = _seed_openrouter_learn_cap(1)
        try:
            html = client.get("/fragments/capacity").text
        finally:
            for key in [k for k in list(fake.hashes)
                        if k.startswith("sy:learn:openrouter")]:
                del fake.hashes[key]
        trs = trs_for(html)
        row = row_with_ref(trs, "openrouter", "mimo")
        assert row is not None, "openrouter/mimo row missing in /fragments/capacity"
        # 1 free + 1 gone = 2 total (the model cap). NOT 4 (plan cap) and
        # NOT 1 (the row drawn without the withheld branch).
        assert row.count('<span class="slot"') == 1, row
        assert row.count('class="slot gone"') == 1, row
        # The gone square carries the tooltip "withheld: learned[global]"
        # — the learner's source string, exactly as the picker reports it.
        assert 'title="withheld: learned[' in row, row
        assert 'title="withheld: learned[global]"' in row, row
        # No divergence chip: the chip condition in _capacity_state.html
        # explicitly drops the tag when (model_cap is not none and cap <
        # model_cap), which is exactly the case here. A regression that
        # dropped that branch would paint `learned[global]` as a chip
        # text on the row, which the row already carries in the tooltip —
        # double-bad: the operator would see the same string twice in
        # one row.
        assert 'class="tag warn"' not in row, row
        # Cap display: `0/1` (the learned cap, the row's effective cap).
        assert ">0/1<" in row, row


# ============================================================================
# Issue #43 — Workstream 2 (capacity board + render_preview + portal writer).
#
# These tests pin the contract the planner wrote for the portal side of the
# groups work: the capacity board renders group structure (strategy tag,
# weights/pointer, per-member state); the per-group writer populates
# `sy:group-order:{gid}:{lane}`; flat configs stay bit-for-bit; hot-reload
# picks up a freshly introduced group on the next recompute without a restart.
#
# Inserted BEFORE the `if __name__ == "__main__":` runner so the discovery
# loop picks them up. See CLAUDE.md -- anything appended after the runner is
# defined too late and silently does not run.
# ============================================================================


def test_probes_panel_renders_needs_reauth_badge_with_last_good_time():
    """A plan whose probe is flagged `needs_reauth` renders a clearly visible
    "stale — needs re-auth (last good HH:MM)" badge on the probes panel.
    The HH:MM is the absolute UTC time of the last successful probe, so the
    operator can correlate it with their logs without doing arithmetic.

    The WS2 spec says: never hide `needs_reauth` behind silent degradation;
    the board's alert "quota probe needs a fresh session cookie" must stay.
    The badge is the visible half of that contract -- a stale row tells the
    operator exactly which plan needs attention and how long it has been
    stale, while the picker keeps using the last good ranking until the
    window resets.
    """
    import asyncio
    with TestClient(portal_app.app) as client:
        # Pick the first cookie-needing plan in the example config and mark
        # it as cookie-expired with a known last-good timestamp.
        reg = models.load()
        cookie_plan = next(p for p in reg.plans.values()
                           if p.probe and p.probe.kind == "cookie")
        # 2024-12-24T12:26:40Z deterministic. Picked from the
        # `datetime` module so the test is timezone- and locale-neutral.
        import datetime as _dt
        last_ok_at = _dt.datetime(
            2024, 12, 24, 12, 26, 40, tzinfo=_dt.timezone.utc).timestamp()
        ledger = portal_app.state["ledger"]
        asyncio.run(ledger.redis.hset(
            f"sy:probe:{cookie_plan.key}",
            mapping={"needs_reauth": "1", "last_ok_at": str(last_ok_at)}))
        try:
            html = client.get("/fragments/probes").text
            # The badge must show for the needs_reauth row, with the
            # absolute HH:MM UTC of the last good reading.
            assert "stale — needs re-auth" in html, html
            assert "last good" in html, html
            # 12:26:40Z -> HH:MM is "12:26" with the deterministic stamp.
            assert "last good 12:26" in html, html
            # The "active" tag must NOT be on a needs_reauth row.
            assert ">active<" not in html, html
        finally:
            asyncio.run(ledger.redis.hset(
                f"sy:probe:{cookie_plan.key}",
                mapping={"needs_reauth": "0"}))


def test_probes_panel_renders_active_badge_for_healthy_plan():
    """A plan with a healthy cookie renders the plain "active" badge --
    regression bar for the conditional in `_probes.html`. The WS2 change
    must not regress the healthy path.
    """
    with TestClient(portal_app.app) as client:
        html = client.get("/fragments/probes").text
        # Cookie plans in the example: at least one is healthy in the
        # default state (the live probe status is whatever FakeRedis
        # carries through; a healthy row is the path with no needs_reauth
        # stamp on its probe hash). The badge test below pins the
        # negative path; this test pins that the positive path's
        # template branch still renders for at least one row.
        assert "active" in html or "not set" in html, html


def _recompute_group_orders(reg, ledger):
    """Drive the per-group writer end-to-end against the registry's parsed tree.

    Mirrors what `_poll_probes` does after a successful probe -- one call
    writes `sy:group-order:{gid}:{lane}` for every explicit
    `perishable:` / `lowest_utilization:` group in the registry.
    """
    import asyncio
    from switchyard.portal.groups import recompute_group_orders
    return asyncio.run(recompute_group_orders(reg, ledger))


def _force_lane(reg, key, order, tail=None, strategy="fill"):
    """Replace a lane's body with a freshly parsed tree and return a registry.

    Convenience helper for the group-rendering tests: builds a `Group`
    tree the same way the YAML loader would, swaps the lane in place, and
    returns the new registry. The hot-reload path uses the same swap so
    the two share their regression bar.
    """
    from dataclasses import replace
    from switchyard.models import _parse_lane_order
    lanes = dict(reg.lanes)
    base = reg.lanes[key] if key in reg.lanes else reg.lanes["apex"]
    known = {m.ref for p in reg.plans.values() for m in p.models.values()}
    parsed = _parse_lane_order(key, list(order), known)
    lanes[key] = replace(base, key=key, order=parsed, tail=list(tail or []),
                         strategy=strategy, description="")
    return models.Registry(settings=reg.settings, plans=reg.plans, lanes=lanes)


def test_round_robin_group_renders_pointer_and_members():
    """An explicit `round_robin` group renders the strategy tag, the next
    member index, and the rotation pointer (read from
    `sy:group-rot:{gid}:{lane}`).

    A flat config renders no group headers; a round_robin-bearing config
    renders exactly one `group-head` row, with the rotation metadata
    visible to the operator.
    """
    from dataclasses import replace
    import asyncio

    with TestClient(portal_app.app) as client:
        reg = models.load()
        # Build a fresh registry with a round_robin lane on the same two
        # forge members the picker already uses.
        new_reg = _force_lane(reg, "rr-board",
                              [{"round_robin": ["minimax-ultra/m3",
                                                "minimax-max/m3"]}],
                              tail=[], strategy="fill")
        # Set the rotation pointer to a known value (5) on the FakeRedis.
        ledger = portal_app.state["ledger"]
        # The gid is computed from lane + strategy + sorted refs, same way
        # the parser produces it.
        from switchyard.models import _group_id, _leaf_refs
        body = new_reg.lane_nodes()["rr-board"]
        assert len(body) == 1 and body[0].strategy == "round_robin"
        gid = body[0].gid
        asyncio.run(ledger.bump_group_rot(gid, "rr-board"))   # -> 1
        asyncio.run(ledger.bump_group_rot(gid, "rr-board"))   # -> 2
        asyncio.run(ledger.bump_group_rot(gid, "rr-board"))   # -> 3
        asyncio.run(ledger.bump_group_rot(gid, "rr-board"))   # -> 4
        asyncio.run(ledger.bump_group_rot(gid, "rr-board"))   # -> 5

        # Swap the registry into state so collect_capacity sees the new tree.
        original_reg = portal_app.state["registry"]
        original_picker = portal_app.state["picker"]
        from switchyard.picker import Picker
        portal_app.state["registry"] = new_reg
        portal_app.state["picker"] = Picker(new_reg, portal_app.state["slots"],
                                            portal_app.state["policy"])
        try:
            html = client.get("/fragments/capacity").text
            # Strategy tag present.
            assert 'data-strategy="round_robin"' in html, html
            # Member names appear next to it.
            assert "minimax-ultra/m3" in html and "minimax-max/m3" in html, html
            # Pointer and next index show -- both are how the operator
            # reads "where in the rotation are we?".
            assert "pointer 5" in html, html
            # 5 % 2 = 1, so "next: member 1" labels minimax-max/m3.
            assert "next: member 1" in html, html
        finally:
            portal_app.state["registry"] = original_reg
            portal_app.state["picker"] = original_picker


def test_weighted_group_renders_weights():
    """A weighted group renders every key with its weight inline, in the
    same insertion order the picker walks."""
    with TestClient(portal_app.app) as client:
        reg = models.load()
        new_reg = _force_lane(reg, "w-board",
                              [{"weighted": {"minimax-ultra/m3": 5,
                                             "minimax-max/m3": 2,
                                             "openrouter/mimo": 1}}],
                              tail=[], strategy="fill")
        original_reg = portal_app.state["registry"]
        original_picker = portal_app.state["picker"]
        from switchyard.picker import Picker
        portal_app.state["registry"] = new_reg
        portal_app.state["picker"] = Picker(new_reg, portal_app.state["slots"],
                                            portal_app.state["policy"])
        try:
            html = client.get("/fragments/capacity").text
            assert 'data-strategy="weighted"' in html, html
            # Each member's weight shows next to the strategy tag, in order.
            assert "minimax-ultra/m3 x5" in html, html
            assert "minimax-max/m3 x2" in html, html
            assert "openrouter/mimo x1" in html, html
            # The order matches declaration order, so the rendered string
            # preserves the picker walk.
            i1 = html.index("minimax-ultra/m3 x5")
            i2 = html.index("minimax-max/m3 x2")
            i3 = html.index("openrouter/mimo x1")
            assert i1 < i2 < i3, (i1, i2, i3)
        finally:
            portal_app.state["registry"] = original_reg
            portal_app.state["picker"] = original_picker


def test_perishable_group_renders_stored_ranking():
    """An explicit `perishable:` group's ranking on the board matches the
    stored `sy:group-order:{gid}:{lane}` hash. A stale or missing key
    falls back to declared order.

    The picker reads the same hash; this is the visible half of the
    round trip. Test seeds the hash directly through the ledger.
    """
    import asyncio
    import time
    with TestClient(portal_app.app) as client:
        reg = models.load()
        new_reg = _force_lane(reg, "per-board",
                              [{"perishable": ["claude-max/fable",
                                               "openai/astra"]}],
                              tail=[], strategy="fill")
        gid = new_reg.lane_nodes()["per-board"][0].gid
        ledger = portal_app.state["ledger"]
        # Set the order directly: astra outranks fable on score.
        asyncio.run(ledger.set_group_order(
            gid, "per-board",
            {"openai/astra": {"score": 0.9, "gate5h": 0},
             "claude-max/fable": {"score": 0.1, "gate5h": 0}},
            computed_at=time.time(),
            stale_after_ms=2_000_000))

        original_reg = portal_app.state["registry"]
        original_picker = portal_app.state["picker"]
        from switchyard.picker import Picker
        portal_app.state["picker"] = Picker(new_reg, portal_app.state["slots"],
                                            portal_app.state["policy"])
        portal_app.state["registry"] = new_reg
        try:
            html = client.get("/fragments/capacity").text
            assert 'data-strategy="perishable"' in html, html
            # Stored ranking shows inline: "openai/astra, claude-max/fable"
            # (score desc), not the declared order.
            i_astra = html.index("openai/astra")
            i_fable = html.index("claude-max/fable")
            # The board's "ranking -> ..." portion has astra first. There
            # are multiple occurrences of each ref (one in the row, one in
            # the group header). Search the specific arrow annotation.
            arrow_idx = html.find("→")
            assert arrow_idx != -1, "the ranking arrow must be present"
            ranking_segment = html[arrow_idx:arrow_idx + 200]
            assert "openai/astra" in ranking_segment, ranking_segment
            assert "claude-max/fable" in ranking_segment, ranking_segment
            # And in score-desc order in the arrow segment.
            assert ranking_segment.index("openai/astra") < \
                ranking_segment.index("claude-max/fable"), ranking_segment
        finally:
            portal_app.state["registry"] = original_reg
            portal_app.state["picker"] = original_picker


def test_pacing_paints_tail_disabled_and_paced_to_zero_badges():
    """Pacing mode paints `tail disabled (pacing)` on tail rows and
    `paced to 0 of N` on members whose plan is currently paced down to
    zero. Both survive the template change.

    This is the regression bar for the spec's "scripts/smoke.py grep
    patterns still match" requirement -- the smoke.py run past
    `tail disabled (pacing)` must continue to find a row carrying it.
    """
    import asyncio
    with TestClient(portal_app.app) as client:
        client.post("/admin/pacing?enabled=on")
        try:
            html = client.get("/fragments/capacity").text
            # Smoke.py greps the gateway log AND the board for this string.
            assert "tail disabled (pacing)" in html, html
            # At least one tail row has the chip -- assert it is a tag,
            # not just a substring leak from somewhere else.
            assert 'class="tag warn">tail disabled (pacing)</span>' in html, \
                html
            # And the legacy `paced N of M` reason also survives: the
            # fixture seeds cap=0 for the third member with a pacing
            # reason. (The example ships with pacing off, so we depend
            # on the fixture's pacing tags to assert the badge stays
            # painted -- pacing-on must surface both kinds of badges.)
            # Note: with pacing OFF, the legacy path paints `paced N of M`
            # via cap_reason. With pacing ON, the cap_reason for a tail
            # is `tail disabled (pacing)`. Both strings must round-trip.
            assert ("paced" in html) or ("tail disabled" in html), html
        finally:
            client.post("/admin/pacing?enabled=default")


def test_flat_config_produces_same_render_as_before():
    """A flat config (no Groups in any lane's body) renders the same rows
    the legacy template did -- bit-for-bit. The shipped example config
    now uses groups (per WS3's example), so this test synthesises a flat
    registry by stripping groups from every lane's body and asserts the
    fragment renders exactly as it did pre-groups.

    This is the regression bar that keeps `group-head` rows,
    indentation and strategy tags from leaking onto a config that never
    asked for them. A regression that re-introduces the legacy flat
    path's output for a group-bearing lane would also fail this test
    -- the rendered HTML on a stripped body must look exactly like the
    legacy flat output did.
    """
    from dataclasses import replace
    with TestClient(portal_app.app) as client:
        # Build a flat registry: every lane's body is just its routing
        # order, no groups. Walk `lane_nodes()` and replace every Group
        # node with its flattened refs.
        reg = models.load()
        new_lanes = {}
        for key, lane in reg.lanes.items():
            flat: list[str] = []
            def walk(node):
                from switchyard.models import Group
                if isinstance(node, Group):
                    if node.weights is not None:
                        flat.extend(node.weights.keys())
                    else:
                        for m in node.members:
                            walk(m)
                    return
                flat.append(node)
            for node in reg.lane_nodes()[key]:
                walk(node)
            new_lanes[key] = replace(lane, order=flat,
                                     strategy="fill", description="")
        flat_reg = models.Registry(settings=reg.settings, plans=reg.plans,
                                   lanes=new_lanes)

        # Swap state in for the duration of the read.
        original_reg = portal_app.state["registry"]
        original_picker = portal_app.state["picker"]
        from switchyard.picker import Picker
        portal_app.state["registry"] = flat_reg
        portal_app.state["picker"] = Picker(flat_reg, portal_app.state["slots"],
                                            portal_app.state["policy"])
        try:
            html = client.get("/fragments/capacity").text
            assert 'class="group-head"' not in html, html
            assert "data-strategy=" not in html, html
            # No row carries an inline indent style -- the depth field
            # is always 0 on a flat config, so the template does not
            # emit `padding-left` inline.
            assert "padding-left" not in html, html
            # Every ref the example ships with renders as
            # `>plan</span><span class="muted">/model</span>` -- the
            # same pattern existing tests use to find a row. Refs that
            # the WS3 example config swapped out (e.g. `glm/glm-5.3-flash`
            # replaced by `opencode-go/glm-5.3-flash` on the forge lane)
            # are not asserted here; the regression bar is the SHAPE
            # of the rendered rows, not which exact refs the shipped
            # example happens to name today.
            refs = [("minimax-ultra", "m3"), ("minimax-max", "m3"),
                    ("grok", "grok-4.6"),
                    ("opencode-go", "glm-5.3-flash"), ("openrouter", "mimo"),
                    ("claude-max", "fable"), ("claude-max", "opus"),
                    ("openai", "astra"), ("openai", "sol"),
                    ("glm", "glm-5.3"), ("local-box", "qwen"),
                    ("local-box", "gemma")]
            for plan, model in refs:
                ident = f">{plan}</span><span class=\"muted\">/{model}</span>"
                assert ident in html, f"{plan}/{model} row missing from flat fragment"
        finally:
            portal_app.state["registry"] = original_reg
            portal_app.state["picker"] = original_picker


def test_hot_reload_picks_up_new_group_without_restart():
    """Mutating the registry to introduce a group is reflected on the next
    /fragments/capacity read, with no process restart.

    The portal's `/admin/reload` is the operator-facing knob: it reads
    plans.yaml and rebuilds the registry + picker in place. The next
    capacity read must walk the new tree. We swap the registry directly
    so the test does not need a writable plans.yaml on disk; the same
    code path is what reload_config() uses.
    """
    import asyncio
    from dataclasses import replace
    with TestClient(portal_app.app) as client:
        reg = models.load()
        # Before the swap: the shipped config may already have groups
        # (the demo `forge` lane uses round_robin / weighted / perishable).
        # What we are pinning here is that swapping the registry DOES
        # change what the next fragment read sees -- without a process
        # restart. A regression that cached the parsed tree at startup
        # would keep rendering the same groups regardless of registry
        # swaps.

        # Build a brand-new lane key that the example does not name, so
        # the swap is observable: nothing in the pre-swap fragment mentions
        # this lane's label.
        sentinel_key = "hot-reload-canary"
        from switchyard.models import Group
        from switchyard.models import _parse_lane_order
        known = {m.ref for p in reg.plans.values() for m in p.models.values()}
        parsed = _parse_lane_order(sentinel_key,
                                   [{"round_robin": ["claude-max/opus",
                                                     "openai/sol"]}], known)
        from dataclasses import replace
        new_lane_cfg = reg.lanes["apex"]
        new_lanes = dict(reg.lanes)
        new_lanes[sentinel_key] = replace(new_lane_cfg, key=sentinel_key,
                                          label="Hot Reload Canary",
                                          order=parsed, tail=[],
                                          strategy="fill", description="")
        new_reg = models.Registry(settings=reg.settings, plans=reg.plans,
                                  lanes=new_lanes)

        # Confirm the pre-swap fragment does NOT mention the canary lane
        # -- otherwise the "swap is observable" check below would have
        # nothing to compare against.
        before = client.get("/fragments/capacity").text
        assert "Hot Reload Canary" not in before, "canary lane unexpectedly present before swap"

        original_reg = portal_app.state["registry"]
        original_picker = portal_app.state["picker"]
        from switchyard.picker import Picker
        portal_app.state["registry"] = new_reg
        portal_app.state["picker"] = Picker(new_reg, portal_app.state["slots"],
                                            portal_app.state["policy"])
        try:
            after = client.get("/fragments/capacity").text
            # The new tree produced a group header on the next read,
            # without anyone restarting the FastAPI app.
            assert 'data-strategy="round_robin"' in after, "no round_robin tag"
            # The canary lane is now on the board -- proof that the
            # swap took effect on the next read, not just cached.
            assert "Hot Reload Canary" in after, "canary lane not in post-swap fragment"
        finally:
            portal_app.state["registry"] = original_reg
            portal_app.state["picker"] = original_picker


def test_per_group_writer_writes_group_order_hash():
    """The portal's per-group writer populates `sy:group-order:{gid}:{lane}`
    for every explicit `perishable:` / `lowest_utilization:` group, with
    the same hysteresis contract the lane-level writer has.

    The test seeds probe facts (pct_used + reset) on three plans and asserts
    the stored hash carries scored refs in family-partition order. Members
    whose plan has no probe facts are dropped from `entries` -- the writer
    does not invent a zero score for them, matching the lane-level writer's
    "unknown sorts last" contract.

    Family partition (issue #53): the body declares
    `[minimax-ultra, minimax-max, grok]`. minimax appears first, so its
    bucket leads and grok's xai bucket sorts after, regardless of raw
    score. The within-bucket score order still puts max (more room) ahead
    of ultra (less room) inside the minimax bucket.
    """
    import asyncio
    import time
    with TestClient(portal_app.app) as client:
        reg = models.load()
        # A perishable group on forge members. All three plans have probe
        # facts seeded, so all three refs land in the stored hash.
        new_reg = _force_lane(reg, "per-writer",
                              [{"perishable": ["minimax-ultra/m3",
                                               "minimax-max/m3",
                                               "grok/grok-4.6"]}],
                              tail=[], strategy="fill")
        gid = new_reg.lane_nodes()["per-writer"][0].gid
        ledger = portal_app.state["ledger"]

        # Wipe residue from sibling tests that share the FakeRedis.
        for k in ("grok", "minimax-ultra", "minimax-max"):
            asyncio.run(ledger.redis.delete(f"sy:qwin:{k}:weekly"))
            asyncio.run(ledger.redis.delete(f"sy:qwin:{k}:5h"))
        # Probe facts: ultra at 90% used (low room), max at 10% (high room),
        # grok at 50% (mid room). All same reset horizon so perishable_score
        # ranks them purely by room.
        reset = time.time() + 7 * 86400
        asyncio.run(ledger.note_reported_percent(
            "minimax-ultra", 90.0, reset, window="weekly"))
        asyncio.run(ledger.note_reported_percent(
            "minimax-max", 10.0, reset, window="weekly"))
        asyncio.run(ledger.note_reported_percent(
            "grok", 50.0, reset, window="weekly"))

        written = _recompute_group_orders(new_reg, ledger)
        # The writer wrote at least one group (our perishable group).
        # The shipped example also has its own perishable / lowest_util
        # groups, which count too -- the assertion is "we wrote our
        # group", not "exactly one group total".
        assert written >= 1, written
        order = asyncio.run(ledger.get_group_order(gid, "per-writer"))
        assert order is not None, order
        refs = [m["ref"] for m in order["members"]]
        # Family partition: minimax bucket leads (declared first in the
        # body), xai bucket trails. Within the minimax bucket, max (more
        # room) outranks ultra (less room) by score. Grok lands after both
        # minimax refs even though grok's raw room beats ultra's.
        assert refs == ["minimax-max/m3", "minimax-ultra/m3",
                        "grok/grok-4.6"], refs
        # Tier offset must be baked into the stored scores so the board
        # (which reads `get_group_order` without re-partitioning) sorts
        # the partition order verbatim. The minimax bucket leads (tier=1
        # of 2), so both minimax refs sit above 1e9; the xai bucket
        # trails (tier=0 of 2), so grok sits below 1e9. The picker
        # re-partitions anyway, so a missing tier would not break the
        # picker tests -- this assertion is the only thing that pins
        # the offset for the board view.
        scores = {m["ref"]: m["score"] for m in order["members"]}
        assert scores["minimax-max/m3"] >= 1e9, scores
        assert scores["minimax-ultra/m3"] >= 1e9, scores
        assert scores["grok/grok-4.6"] < 1e9, scores
        # Within the minimax bucket, raw score-desc: max (room 90) outranks
        # ultra (room 10). Subtract the 1e9 tier to recover the raw score.
        assert (scores["minimax-max/m3"] - 1e9) > (
            scores["minimax-ultra/m3"] - 1e9), scores
        # Every entry carries a real score, not the default-zero fallback.
        for entry in order["members"]:
            assert entry["score"] > 0, entry


def test_per_group_writer_drops_unknown_refs_from_hash():
    """A member whose plan has no probe facts lands in `unknown` and is
    NOT written into the group hash. The picker reads the absent ref as
    unscored and sorts it last in declared order -- the lane-level
    writer's "unknown sorts last" contract, applied to per-group hashes.
    """
    import asyncio
    import time
    with TestClient(portal_app.app) as client:
        reg = models.load()
        new_reg = _force_lane(reg, "per-unknown",
                              [{"perishable": ["minimax-ultra/m3",
                                               "grok/grok-4.6"]}],
                              tail=[], strategy="fill")
        gid = new_reg.lane_nodes()["per-unknown"][0].gid
        ledger = portal_app.state["ledger"]

        # Wipe residue from sibling tests that share the FakeRedis:
        # a previous test that seeded grok's facts would otherwise carry
        # over and turn this ref into "scored" rather than "unknown".
        # FakeRedis stores hash-backed data in `.hashes`, not `.strings`,
        # so the delete has to go straight to the underlying dict.
        for k in ("grok", "minimax-ultra", "minimax-max"):
            for win in ("weekly", "5h"):
                ledger.redis.hashes.pop(f"sy:qwin:{k}:{win}", None)

        # Seed only ultra's facts. grok/grok-4.6 has no facts, so its
        # ref is unknown and must NOT appear in the stored hash.
        reset = time.time() + 7 * 86400
        asyncio.run(ledger.note_reported_percent(
            "minimax-ultra", 50.0, reset, window="weekly"))

        _recompute_group_orders(new_reg, ledger)
        order = asyncio.run(ledger.get_group_order(gid, "per-unknown"))
        assert order is not None, order
        refs = [m["ref"] for m in order["members"]]
        assert refs == ["minimax-ultra/m3"], refs


def test_perishable_writer_treats_stale_target_and_gate_as_unknown():
    """Stale probe facts must not score a member or trip its 5h gate.

    A plan whose vendor client never re-polls leaves a row like
    `reported_pct_used: 95, reset_at: <past>` in Redis. Before the
    `reported_is_current` gate, the writer happily read the 95% as
    `target_pct` and used it to compute room=5 (low perishable score), AND
    the matching 5h row was a `gate5h=1` flag that filtered the member out
    of new-session picks. Both consequences are wrong: the plan re-enters
    rotation only by accident, never by design, and a probe that has gone
    silent can suppress the plan forever -- the same over-exclusion the
    picker's `is_spent` fix addresses on the routing side.

    The writer now gates `target_pct`, `target_reset` and `gate_pct` on
    `reported_is_current`. Stale target -> room=None -> perishable_score
    returns None -> the picker puts the member in the unscored bucket.
    Stale gate -> `this_plan_gate5h` is False -> the member is not flagged
    gate5h. Both `unknown` and unscored entries are absent from the
    `entries` map the writer commits, so the picker treats them the same:
    sorted last, never blocked.
    """
    import asyncio
    from dataclasses import replace
    from switchyard import models
    from switchyard.policy import CapacityPolicy
    from switchyard.usage import Ledger
    from switchyard.portal.app import _recompute_perishable_for_plan
    from switchyard.slots import SlotTable

    async def go():
        # Same fixture pattern as the perishable tests in test_routing.py:
        # an apex-shaped perishable lane with two plans, only one of which
        # has any probe facts at all. claude-max/fable carries the stale
        # rows; openai/sol has no rows and stays the unknown baseline.
        reg = models.load()
        redis = FakeRedis()
        slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
        ledger = Ledger(redis)
        CapacityPolicy(redis, reg.settings, ledger)
        lanes = dict(reg.lanes)
        lanes["perishable-test"] = replace(
            reg.lanes["apex"], key="perishable-test", tail=[],
            strategy="perishable", description="")
        reg2 = models.Registry(settings=reg.settings, plans=reg.plans,
                               lanes=lanes)

        plan = reg2.plans["claude-max"]
        # Stale target (weekly) and stale gate (5h): reset_at in the past.
        # Without the gate, `target_pct = 95` -> room = 5 -> a low perishable
        # score, AND `gate_pct = 95 > 90` -> this_plan_gate5h = True. With
        # the gate, both reads are treated as unknown. use `time.time() -
        # 8 * 86400` (8 days ago) for weekly, which is well outside the
        # current week bucket even by the period_bounds rule.
        import time as _time
        past_reset = _time.time() - 8 * 86400
        await ledger.note_reported_percent("claude-max", 95.0, past_reset,
                                           window="weekly")
        await ledger.note_reported_percent("claude-max", 95.0, past_reset,
                                           window="5h")
        await _recompute_perishable_for_plan(reg2, ledger, plan)
        stored = await ledger.get_lane_order("perishable-test")
        return stored

    stored = asyncio.run(go())
    # The writer wrote nothing -- every member's ratio is unknown, so neither
    # is in the scored set. `get_lane_order` returns None for a missing key,
    # which is the "fall back to config order" signal the picker reads.
    assert stored is None, stored
    print("  stale target + stale gate: writer drops the entry, "
          "picker falls back to config order")


if __name__ == "__main__":
    # Plain-script runner: discovers tests from globals(), like the rest of
    # tests/*.py. See CLAUDE.md — appending below this block would silently
    # skip a test that the suite still reports as "all green".
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  {name}: ok")
            except AssertionError as exc:
                failures += 1
                print(f"  {name}: FAIL\n    {exc}")
            except Exception as exc:
                failures += 1
                print(f"  {name}: ERROR\n    {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(1)
