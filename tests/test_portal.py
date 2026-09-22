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

from switchyard.portal import app as portal_app  # noqa: E402


def _off(html: str) -> bool:
    return "Pacing mode off" in html and 'aria-pressed="false"' in html \
        and "Turn on" in html


def _on(html: str) -> bool:
    return "Pacing mode on" in html and 'aria-pressed="true"' in html \
        and "Turn off" in html


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
    import re

    def trs_for(html):
        return re.findall(r"<tr>.*?</tr>", html, re.DOTALL)

    def row_with_ref(trs, plan, model):
        ident = f">{plan}</span><span class=\"muted\">/{model}</span>"
        return next((t for t in trs if ident in t.replace("\n", "")), None)

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
