"""Probe field-mapping and re-auth detection.

No network: these test the parsing and the failure classification, which is
where the bugs actually live. Field paths are candidate lists because vendors
rename things, so the mapper must tolerate misses and odd shapes.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()

from switchyard import models                                  # noqa: E402
from switchyard.probes import REAUTH_MARKERS, Prober, dig, first_number  # noqa: E402
from switchyard.usage import Ledger                            # noqa: E402
from tests.fake_redis import FakeRedis                         # noqa: E402

run = asyncio.run


def test_dotted_paths_survive_lists_and_missing_keys():
    doc = {"data": {"plans": [{"remains": 4200}], "total": "9000"}}
    assert dig(doc, "data.plans.0.remains") == 4200
    assert dig(doc, "data.total") == "9000"
    assert dig(doc, "data.nope.deeper") is None
    assert dig(doc, "data.plans.9.remains") is None
    print("  dotted paths handle nesting, list indexes and misses")


def test_first_matching_candidate_wins():
    doc = {"data": {"remaining_tokens": 1234}}
    paths = ["data.remains", "data.remaining_tokens", "remains"]
    assert first_number(doc, paths) == 1234.0
    # Numeric strings count; booleans must not.
    assert first_number({"a": "17.5"}, ["a"]) == 17.5
    assert first_number({"a": True}, ["a"]) is None
    assert first_number({}, ["a", "b"]) is None
    print("  candidate lists resolve in order, strings parse, booleans ignored")


def test_minimax_reauth_body_contains_prose_marker():
    # After #164, "1004" is no longer a prose marker (a real 200 reading can
    # carry it inside an unrelated field), and the 1004 reauth is detected
    # structurally out of base_resp.status_code instead. What this test
    # actually asserts is that the surviving prose markers
    # ("cookie is missing", "log in again") still match MiniMax's reauth
    # body verbatim — and that "1004" really is gone from REAUTH_MARKERS so
    # a successful reading that happens to contain "11004" can't trip it.
    body = ('{"base_resp":{"status_code":1004,'
            '"status_msg":"cookie is missing, log in again"}}').lower()
    assert "cookie is missing" in body
    assert any(m in body for m in REAUTH_MARKERS)
    assert "1004" not in REAUTH_MARKERS
    print("  MiniMax reauth body matches a surviving prose marker; "
          "'1004' is no longer a marker of its own")


def test_cookie_is_never_readable_through_status():
    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        secret = "session=abcdefghijklmnop1234; other=zz"
        await prober.set_cookie("minimax-ultra", secret)
        st = await prober.status("minimax-ultra")
        blob = repr(st)
        assert secret not in blob and "abcdefgh" not in blob, blob
        assert st["has_cookie"]
        # A length and a truncated digest — enough to tell two cookies apart,
        # and carrying no part of one. It used to end with the cookie's last
        # four characters, which is real session material on a page that
        # promises the cookie is never returned.
        fp = st["fingerprint"]
        assert fp.startswith(f"{len(secret)} chars, #"), fp
        for n in (4, 6, 8):
            assert secret[-n:] not in fp, f"fingerprint leaks the last {n} chars: {fp}"
        # Two different cookies must not collide.
        await prober.set_cookie("minimax-max", secret + "x")
        other = (await prober.status("minimax-max"))["fingerprint"]
        assert other != fp, (fp, other)
        return st
    st = run(go())
    print(f"  status exposes only a fingerprint: {st['fingerprint']!r}")


def test_a_fresh_cookie_clears_the_reauth_flag():
    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        await prober.clear_cookie("minimax-ultra")
        assert (await prober.status("minimax-ultra"))["needs_reauth"]
        await prober.set_cookie("minimax-ultra", "session=new")
        return await prober.status("minimax-ultra")
    st = run(go())
    assert not st["needs_reauth"] and st["has_cookie"]
    print("  pasting a new cookie resumes polling")


def test_polling_skips_plans_with_a_dead_or_missing_cookie():
    """An expired session must not become a request every minute forever."""
    async def go():
        reg = models.load()
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = reg.plans["minimax-ultra"]
        no_cookie = await prober.due(plan)
        await prober.set_cookie(plan.key, "session=x")
        with_cookie = await prober.due(plan)
        await prober.clear_cookie(plan.key)
        after_expiry = await prober.due(plan)
        # A plan with no probe configured is never due.
        never = await prober.due(reg.plans["local-box"])
        return no_cookie, with_cookie, after_expiry, never
    no_cookie, with_cookie, after_expiry, never = run(go())
    assert not no_cookie and with_cookie and not after_expiry and not never
    print("  due(): no cookie=False, fresh=True, expired=False, no probe=False")


def test_one_response_can_report_several_quota_windows():
    """A provider metered on a weekly allowance AND a 5-hour burst publishes both
    in one response, and both must be recorded.

    The prober used to map a single `window`, so the burst number was read and
    thrown away — leaving the constraint window that decides whether to slow
    down with no data at all. A window the response does not carry is reported
    as "no data for <name>", not as a failure, so a wrong field guess for one
    window cannot hide a good reading for another.
    """
    import http.server
    import json
    import threading

    payloads = {
        "/both": {"data": {"remains": 7_400_000, "total": 20_000_000,
                            "reset_at": 1790400000,
                            "burst_remains": 310_000, "burst_total": 800_000}},
        "/weekly": {"data": {"remains": 7_400_000, "total": 20_000_000}},
    }

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payloads.get(self.path, {})).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    async def go():
        from dataclasses import replace
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        # A probe built here, not taken from plans.yaml: this tests the
        # mechanism, so it must not break when a plan's real field paths change.
        windows = {
            "weekly": {"remaining": ["data.remains"], "total": ["data.total"],
                        "reset_at": ["data.reset_at"]},
            "5h": {"remaining": ["data.burst_remains"], "total": ["data.burst_total"]},
        }
        plan = reg_plan = models.load().plans["minimax-ultra"]
        probe = replace(reg_plan.probe, windows=windows, fields={}, scale=1.0)
        plan = replace(plan, probe=probe)
        await prober.set_cookie(plan.key, "session=x")

        both = await prober.run(
            replace(plan, probe=replace(probe, url=base + "/both")))
        only = await prober.run(
            replace(plan, probe=replace(probe, url=base + "/weekly")))
        return both, only

    both, only = asyncio.run(go())
    srv.shutdown()

    assert both.ok and both.detail == "ok", both
    weekly, burst = both.reading("weekly"), both.reading("5h")
    assert weekly and weekly.remaining == 7_400_000, weekly
    assert burst and burst.remaining == 310_000, burst
    # The flat fields stay the TARGET window's, so older callers see the weekly.
    assert both.remaining == weekly.remaining, (both.remaining, weekly.remaining)

    assert only.ok, only
    assert "no data for 5h" in only.detail, only.detail
    assert only.reading("weekly").remaining == 7_400_000
    assert only.reading("5h").remaining is None
    print(f"  both windows read ({weekly.remaining:.0f} weekly, {burst.remaining:.0f} "
          f"burst); a missing window degrades to {only.detail!r}")


def test_opencode_go_status_payload_yields_all_three_meters():
    """The Go plan's real /console/api/go/status response, parsed end to end.

    Three things about it that a token-and-headroom probe could not read, all
    captured from a live session rather than guessed: it reports *spend* against
    a limit rather than headroom, the numbers are microcents rather than tokens,
    and the reset times are ISO-8601 strings rather than epoch numbers. Getting
    any of them wrong silently loses a window.
    """
    import http.server
    import json
    import os
    import threading
    from dataclasses import replace

    payload = {
        "subscriberUserId": "acc_x", "cancelAtPeriodEnd": True,
        "access": {
            "startsAt": "2026-08-27T22:42:42.000Z",
            "endsAt": "2026-09-27T22:42:42.000Z",
            "meters": {
                "fiveHour": {"resetsAt": "2026-09-21T07:58:32.284Z",
                             "limitMicroCents": "1200000000",
                             "usedMicroCents": "383625"},
                "week": {"resetsAt": "2026-09-28T00:00:00.000Z",
                         "limitMicroCents": "3000000000",
                         "usedMicroCents": "505125"},
                "month": {"limitMicroCents": "6000000000",
                          "usedMicroCents": "10256826"},
            },
        },
    }

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            # The route really does refuse without the org header.
            if self.headers.get("x-org-id") != "wrk_TESTORG":
                body = json.dumps({"code": "org_required"}).encode()
                code = 400
            else:
                body = json.dumps(payload).encode()
                code = 200
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/console/api/go/status"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["opencode-go"]
        assert plan.probe is not None, "opencode-go should have a probe"
        plan = replace(plan, probe=replace(plan.probe, url=url))
        await prober.set_cookie(plan.key, "session=x")

        os.environ["OPENCODE_ORG_ID"] = "wrk_TESTORG"
        good = await prober.run(plan)
        os.environ.pop("OPENCODE_ORG_ID")
        unset = await prober.run(plan)
        return good, unset

    good, unset = asyncio.run(go())
    srv.shutdown()

    assert good.ok, good
    got = {w.window: w for w in good.windows}
    assert set(got) == {"5h", "weekly", "monthly"}, sorted(got)
    # microcents -> dollars, and remaining derived from limit - used.
    assert abs(got["5h"].total - 12.0) < 1e-9, got["5h"]
    assert abs(got["weekly"].total - 30.0) < 1e-9, got["weekly"]
    assert abs(got["monthly"].total - 60.0) < 1e-9, got["monthly"]
    assert abs(got["monthly"].used - 0.10256826) < 1e-9, got["monthly"]
    assert abs(got["monthly"].remaining - (60.0 - 0.10256826)) < 1e-9, got["monthly"]
    # ISO reset times parsed, not dropped.
    for name in ("5h", "weekly", "monthly"):
        assert got[name].reset_at and got[name].reset_at > 1.7e9, (name, got[name])
    assert got["weekly"].reset_at > got["5h"].reset_at

    # A missing org id is named, not sent as an empty header.
    assert not unset.ok and "OPENCODE_ORG_ID" in unset.detail, unset.detail
    print(f"  5h ${got['5h'].used:.4f}/${got['5h'].total:.0f} · "
          f"weekly ${got['weekly'].used:.4f}/${got['weekly'].total:.0f} · "
          f"monthly ${got['monthly'].used:.4f}/${got['monthly'].total:.0f}")


def test_minimax_remains_percent_payload_is_read_as_percentages():
    """MiniMax's real console payload, which publishes percentages and no counts.

    Captured live from platform.minimax.io/console/usage. Three things it forces:
    every *_count field is -1, so the only measurement is a string like "12%";
    both windows arrive in one entry (`current_interval_*` is the 5-hour window,
    `current_weekly_*` the weekly); and the entries are per model family, so the
    path has to pick `general` by name rather than by array position — MiniMax
    also returns `video`, and adding a family would otherwise shift the index.

    A percentage must never be mixed into a token total, so it takes its own
    route through the ledger and shows as "reported by provider (% only)".
    """
    import http.server
    import json
    import threading
    import time
    from dataclasses import replace
    from switchyard.usage import headroom

    # Self-healing offsets: baking static end_time ms here drifts into the past
    # across runs and the gate at #45 then sends the bar back to the ledger basis.
    now_ms = int(time.time() * 1000)
    payload = {"model_remains": [
        {"model_name": "general",
         "start_time": 1789948800000, "end_time": now_ms + 5 * 3600 * 1000,
         "current_interval_total_count": -1, "current_interval_used_count": -1,
         "current_interval_used_percent": "37.5%",
         "weekly_start_time": 1789948800000, "weekly_end_time": now_ms + 30 * 24 * 3600 * 1000,
         "current_weekly_used_count": -1, "current_weekly_used_percent": "12%"},
        {"model_name": "video", "end_time": now_ms + 5 * 3600 * 1000,
         "current_interval_used_percent": "0%"},
    ]}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if not self.headers.get("cookie"):
                body = json.dumps({"base_resp": {"status_code": 1004,
                                                  "status_msg": "cookie is missing"}}).encode()
                code = 401
            else:
                body = json.dumps(payload).encode()
                code = 200
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/backend/account/token_plan/remains_percent"

    async def go():
        redis = FakeRedis()
        ledger = Ledger(redis)
        prober = Prober(redis, ledger)
        plan = models.load().plans["minimax-ultra"]
        plan = replace(plan, probe=replace(plan.probe, url=url))

        nocookie = await prober.run(plan, record=False)
        await prober.set_cookie(plan.key, "session=x")
        good = await prober.run(plan)
        return nocookie, good, await headroom(ledger, plan)

    nocookie, good, hr = asyncio.run(go())
    srv.shutdown()

    assert not nocookie.ok and nocookie.needs_reauth, nocookie
    assert good.ok, good
    got = {w.window: w for w in good.windows}
    assert got["weekly"].used_percent == 12.0, got["weekly"]
    assert got["5h"].used_percent == 37.5, got["5h"]
    # No counts exist, so no remaining figure may be invented.
    assert got["weekly"].remaining is None and got["weekly"].total is None, got["weekly"]
    # Epoch milliseconds, not seconds.
    assert 1.79e9 < got["weekly"].reset_at < 1.8e9, got["weekly"].reset_at
    assert got["5h"].reset_at < got["weekly"].reset_at

    windows = {w["window"]: w for w in hr["windows"]}
    assert windows["weekly"]["pct_used"] == 12.0, windows["weekly"]
    assert windows["5h"]["pct_used"] == 37.5, windows["5h"]
    assert windows["5h"]["limit"] is None, "a percentage is not a limit"
    assert "% only" in windows["5h"]["basis"], windows["5h"]["basis"]
    # The burst window is what bites first, even though weekly is the target.
    assert hr["binding"]["window"] == "5h" and not hr["binding_is_target"], hr["binding"]
    print("  general family selected by name; 5h 37.5% binds over weekly 12%")


def test_polling_runs_for_a_kind_none_plan_without_a_cookie():
    """`probe.kind: none` plans carry no cookie; the poller must still run.

    Before this was fixed, `due()` unconditionally rejected plans with no stored
    cookie, which is correct for `kind: cookie` (where a missing cookie means
    we can't reach the endpoint) and wrong for everything else. The consequence
    was that four plans (glm, claude-max, openai, grok) never had `run()` ever
    called on them, so the ledger never recorded a real headroom reading, and
    `Picker.is_spent()` always returned False on them — a fully-spent seat
    kept getting offered to new sessions.

    Three things have to hold for the fix to be real:
      1. `due()` is True on a freshly-loaded plan with no cookie.
      2. After a successful `run()` against a stub server, the ledger carries
         the provider-reported percentage on the target window.
      3. `Picker.is_spent()` honours that percentage, so a 100% reading actually
         takes the plan out of the candidate pool — closing the loop that the
         bug had broken.
    """
    import http.server
    import json
    import threading
    from dataclasses import replace
    from switchyard.policy import CapacityPolicy
    from switchyard.slots import SlotTable

    payload = {"data": {"limits": [
        {"type": "TOKENS_LIMIT", "unit": 6, "number": 1, "percentage": 100,
         "nextResetTime": 1790542882984},
    ]}}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    async def go():
        redis = FakeRedis()
        ledger = Ledger(redis)
        prober = Prober(redis, ledger)
        plan = models.load().plans["glm"]
        # Point at the stub so the probe gets a real 200 with our payload.
        plan = replace(plan, probe=replace(plan.probe, url=base + "/usage"))
        # A `kind: none` plan must be due with no cookie stored — the cookie
        # gate is meaningless for it.
        no_cookie = await prober.due(plan)
        os.environ["GLM_API_KEY"] = "sk-stub"
        try:
            result = await prober.run(plan)
        finally:
            os.environ.pop("GLM_API_KEY", None)
        # The provider reported a percentage; it must land in the ledger so the
        # picker's spent gate can read it.
        facts = await ledger.window_facts(plan.key, plan.quota.label)
        pct = facts.get("reported_pct_used")
        # And the picker must agree with that percentage.
        reg = models.Registry(settings=models.load().settings,
                              plans={"glm": plan}, lanes=models.load().lanes)
        policy = CapacityPolicy(redis, reg.settings, ledger)
        slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
        from switchyard.picker import Picker
        picker = Picker(reg, slots, policy)
        spent = await picker.is_spent(plan)
        return no_cookie, result, pct, spent

    no_cookie, result, pct, spent = run(go())
    srv.shutdown()

    assert no_cookie is True, ("cookie gate fired on a kind: none plan — "
                               "the regression this test pins")
    assert result.ok, result
    assert pct == 100.0, pct
    assert spent is True, ("a 100% reading never reached the picker — the "
                            "cookie gate had starved the poller")
    print(f"  kind: none polled without a cookie; {pct:.0f}% reported; picker "
          f"sees the plan as spent")


def test_extra_usage_state_reads_the_flag_and_fails_closed():
    """on / off from a real boolean; anything else at a mapped path is unknown."""
    from switchyard.probes import extra_usage_state
    path = ("rate_limits.extra_usage.is_enabled",)
    on = {"rate_limits": {"extra_usage": {"is_enabled": True}}}
    off = {"rate_limits": {"extra_usage": {"is_enabled": False}}}
    missing = {"rate_limits": {"limits": []}}
    null = {"rate_limits": {"extra_usage": None}}
    stringy = {"rate_limits": {"extra_usage": {"is_enabled": "false"}}}
    assert extra_usage_state(on, path) == "on"
    assert extra_usage_state(off, path) == "off"
    assert extra_usage_state(missing, path) == "unknown"
    assert extra_usage_state(null, path) == "unknown"
    # A string is not a boolean: reading "false" as off is how money leaks.
    assert extra_usage_state(stringy, path) == "unknown"
    assert extra_usage_state(on, ()) == "not_checked"
    print("  true->on, false->off, missing/null/string->unknown, no path->not_checked")


def _claude_usage(extra):
    """A Claude seat /usage report, shaped like the sidecar returns it.
    `extra` is the extra_usage block; `...` leaves the block out entirely."""
    doc = {"rate_limits": {"limits": [
        {"kind": "session", "percent": 14, "resets_at": "2099-01-01T00:00:00+00:00"},
        {"kind": "weekly_all", "percent": 40, "resets_at": "2099-01-02T00:00:00+00:00"},
    ]}}
    if extra is not ...:
        doc["rate_limits"]["extra_usage"] = extra
    return doc


def _probe_claude(payload, *, use_extra_quota=False, fail_after=False):
    """Run the example claude-max probe against a stub serving `payload`, then
    return (extra-usage state, the fable model's cap reason, the ref the apex
    lane picks). With `fail_after`, a second poll gets a 500 first."""
    import http.server
    import json
    import threading
    from dataclasses import replace
    from switchyard.picker import Picker
    from switchyard.policy import CapacityPolicy
    from switchyard.slots import SlotTable

    failing = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if failing:
                self.send_response(500)
                self.send_header("content-length", "0")
                self.end_headers()
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    async def go():
        redis = FakeRedis()
        ledger = Ledger(redis)
        prober = Prober(redis, ledger)
        loaded = models.load()
        plan = loaded.plans["claude-max"]
        plan = replace(plan, use_extra_quota=use_extra_quota,
                       probe=replace(plan.probe, url=base + "/usage"))
        result = await prober.run(plan)
        assert result.ok, result
        if fail_after:
            failing.append(True)
            again = await prober.run(plan)
            assert not again.ok, again
        plans = dict(loaded.plans)
        plans["claude-max"] = plan
        reg = models.Registry(settings=loaded.settings, plans=plans,
                              lanes=loaded.lanes)
        policy = CapacityPolicy(redis, reg.settings, ledger)
        slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
        picker = Picker(reg, slots, policy)
        _, reason = await picker._cap(plan.models["fable"])
        pick = await picker.pick("apex", None)
        await picker.release(pick.plan.key, pick.request_id, pick.ref)
        return await ledger.extra_usage(plan), reason, pick.ref

    try:
        return run(go())
    finally:
        srv.shutdown()


def test_pay_as_you_go_on_takes_a_claude_seat_out_of_rotation():
    """The seat is skipped while extra usage is on, used again once it is off."""
    state, reason, ref = _probe_claude(_claude_usage({"is_enabled": True}))
    assert state == "on" and reason == "extra usage on", (state, reason)
    assert ref != "claude-max/fable", f"a billing seat must be skipped: {ref}"

    state, _, ref = _probe_claude(_claude_usage({"is_enabled": False}))
    assert state == "off", state
    assert ref == "claude-max/fable", f"a seat with it off must be used: {ref}"
    print("  extra usage on -> skipped; off -> first in the lane again")


def test_an_unreadable_extra_usage_flag_is_treated_as_on():
    """No extra_usage block in a reply that otherwise parsed: fail closed."""
    state, reason, ref = _probe_claude(_claude_usage(...))
    assert state == "unknown" and reason == "extra usage unknown", (state, reason)
    assert ref != "claude-max/fable", ref
    print("  missing flag -> unknown -> skipped")


def test_a_failed_poll_keeps_the_last_extra_usage_reading():
    """A poll that fails after an `on` reading must not make the seat look safe."""
    state, _, ref = _probe_claude(_claude_usage({"is_enabled": True}),
                                  fail_after=True)
    assert state == "on", state
    assert ref != "claude-max/fable", ref
    print("  on, then a 500 -> still on, still skipped")


def test_use_extra_quota_opts_a_billing_seat_back_in():
    """The explicit opt-in is the only way a billing seat gets traffic."""
    state, reason, ref = _probe_claude(_claude_usage({"is_enabled": True}),
                                       use_extra_quota=True)
    assert state == "on", state
    assert reason != "extra usage on", reason
    assert ref == "claude-max/fable", ref
    print("  on + use_extra_quota: true -> used on purpose")


def test_plans_whose_probe_does_not_map_extra_usage_are_untouched():
    """glm maps no extra_usage path: its state is not_checked, never blocking."""
    async def go():
        redis = FakeRedis()
        ledger = Ledger(redis)
        plan = models.load().plans["glm"]
        assert not plan.probe.extra_usage
        # Even a stray stored value must not block a plan that never opted in.
        await redis.hset(f"sy:probe:{plan.key}", mapping={"extra_usage": "on"})
        return await ledger.extra_usage(plan)
    assert run(go()) == "not_checked"
    print("  no extra_usage mapping -> not_checked")


def test_set_cookie_is_not_captured_without_a_response_header():
    """When the server does not return a Set-Cookie header, the stored
    credential is left untouched. A capture path that invented a value from
    nothing would silently overwrite a working cookie with an empty one."""
    import http.server
    import json
    import threading
    from dataclasses import replace

    payload = {"data": {"remains": 7000, "total": 10000}}
    windows = {"weekly": {"remaining": ["data.remains"], "total": ["data.total"]}}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/no-cookie"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["minimax-ultra"]
        # Opt-in capture, but the server never sets a cookie. Custom windows
        # so the probe actually parses this stub payload as a valid reading —
        # otherwise the failure-path returns first and we'd never reach the
        # capture check.
        probe = replace(plan.probe, url=url, capture_set_cookie=True,
                        windows=windows, fields={})
        plan = replace(plan, probe=probe)

        original = "session=original-cookie-value-1234567890"
        await prober.set_cookie(plan.key, original)
        before = dict(redis.hashes["sy:cred:minimax-ultra"])
        result = await prober.run(plan)
        return result, redis.hashes["sy:cred:minimax-ultra"], before

    result, cred, before = run(go())
    srv.shutdown()

    assert result.ok, result
    # Cookie, fingerprint and added_at all carry through unchanged.
    assert cred.get("cookie") == "session=original-cookie-value-1234567890", cred
    assert cred.get("fingerprint") == before.get("fingerprint"), (
        cred.get("fingerprint"), before.get("fingerprint"))
    assert cred.get("added_at") == before.get("added_at"), (
        cred.get("added_at"), before.get("added_at"))
    print("  no Set-Cookie -> credential untouched")


def test_set_cookie_is_not_captured_on_a_reauth_response():
    """A 401 or a reauth-marked body must NEVER overwrite a stored cookie, even
    if the server still returns a Set-Cookie alongside the rejection. The
    whole point of capture is to refresh a working session; writing on top of
    a known-bad session would mask the reauth signal and keep polling a dead
    plan. needs_reauth semantics stay exactly as today."""
    import http.server
    import threading
    from dataclasses import replace

    # MiniMax's exact reauth shape so REAUTH_MARKERS hits and the existing
    # 401 path fires. The server hands back a fresh-looking Set-Cookie anyway,
    # which is the trap the test is built around.
    reauth_body = ('{"base_resp":{"status_code":1004,'
                   '"status_msg":"cookie is missing, log in again"}}')
    new_cookie = "session=brand-new-cookie-from-the-server"

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = reauth_body.encode()
            self.send_response(401)
            self.send_header("content-type", "application/json")
            self.send_header("set-cookie", new_cookie)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/reauth"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["minimax-ultra"]
        probe = replace(plan.probe, url=url, capture_set_cookie=True)
        plan = replace(plan, probe=probe)

        original = "session=original-cookie-value-1234567890"
        await prober.set_cookie(plan.key, original)
        before = dict(redis.hashes["sy:cred:minimax-ultra"])
        result = await prober.run(plan)
        return (result, redis.hashes["sy:cred:minimax-ultra"],
                redis.hashes["sy:probe:minimax-ultra"], before)

    result, cred, probe_state, before = run(go())
    srv.shutdown()

    assert not result.ok and result.needs_reauth, result
    # Credential untouched despite the Set-Cookie header on the response.
    assert cred.get("cookie") == "session=original-cookie-value-1234567890", cred
    assert cred.get("fingerprint") == before.get("fingerprint"), (
        cred.get("fingerprint"), before.get("fingerprint"))
    # needs_reauth is set exactly as today — capture cannot rescue a dead
    # session.
    assert probe_state.get("needs_reauth") == "1", probe_state
    print("  401 + reauth body with Set-Cookie -> credential NOT overwritten")


def test_set_cookie_is_captured_on_a_verified_good_response():
    """A 2xx response with no reauth marker and a Set-Cookie header updates
    the stored credential: new cookie value, new added_at, new fingerprint.
    `status()` surfaces the new fingerprint but never the cookie value
    itself — the same non-leak guarantee the secret-stays-in-the-hash test
    pins for the manual paste path."""
    import http.server
    import json
    import threading
    import time
    from dataclasses import replace

    payload = {"data": {"remains": 7000, "total": 10000}}
    windows = {"weekly": {"remaining": ["data.remains"], "total": ["data.total"]}}
    new_cookie = "session=brand-new-cookie-from-the-server-session=abcdef1234567890"

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("set-cookie", new_cookie)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/fresh"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["minimax-ultra"]
        # Custom windows so the probe parses this stub as a successful reading.
        probe = replace(plan.probe, url=url, capture_set_cookie=True,
                        windows=windows, fields={})
        plan = replace(plan, probe=probe)

        original = "session=original-cookie-value-1234567890"
        await prober.set_cookie(plan.key, original)
        before = dict(redis.hashes["sy:cred:minimax-ultra"])
        # Lower bound on the new added_at: anything at-or-after this counts as
        # a fresh stamp, which is what "updated" means here.
        before_run = time.time()
        result = await prober.run(plan)
        status = await prober.status(plan.key)
        return result, redis.hashes["sy:cred:minimax-ultra"], before, before_run, status

    result, cred, before, before_run, status = run(go())
    srv.shutdown()

    assert result.ok, result
    assert cred.get("cookie") == new_cookie, cred
    # New fingerprint is distinct from the old one, even though both come from
    # the same _fingerprint function — two different cookies must read as two.
    assert cred.get("fingerprint") != before.get("fingerprint"), (
        cred.get("fingerprint"), before.get("fingerprint"))
    assert float(cred.get("added_at")) >= before_run, (
        cred.get("added_at"), before_run)
    # status() exposes the new fingerprint but never the cookie value.
    assert status["fingerprint"] == cred.get("fingerprint"), (
        status["fingerprint"], cred.get("fingerprint"))
    blob = repr(status)
    assert new_cookie not in blob, blob
    assert "abcdef1234567890" not in blob, blob
    print(f"  captured -> fingerprint {status['fingerprint']!r}, value never surfaced")


def test_minimax_200_body_with_unauthorized_count_is_not_a_reauth():
    """The repro that started this: MiniMax's real 200 response carries an
    `unauthorized_count: 0` field. The substring "unauthorized" used to fire
    the prose-marker scan on any successful payload, mark the plan as
    needs_reauth, and stop polling — even when the response was a perfectly
    good 200 with real numbers inside it.

    The fix is that the prose scan only runs when the body is *not* valid
    JSON: a body that parses cleanly is treated as data, and any
    reauth-classified status (MiniMax's 1004) is read structurally out of
    `base_resp.status_code` instead. This test serves the exact issue payload
    (a 200 with `unauthorized_count: 0` next to a `model_remains` entry that
    holds real percentages) and asserts the probe parses it, not kills it.
    """
    import http.server
    import json
    import threading
    from dataclasses import replace

    payload = {
        "base_resp": {"status_code": 0, "status_msg": "success"},
        "model_remains": [{
            "model_name": "general",
            "current_interval_usage_count": 11004,
            "current_interval_used_percent": "37.5%",
            "current_weekly_used_percent": "12%",
        }],
        "unauthorized_count": 0,
    }

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/backend/account/token_plan/remains_percent"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["minimax-ultra"]
        plan = replace(plan, probe=replace(plan.probe, url=url))
        await prober.set_cookie(plan.key, "session=x")
        return await prober.run(plan)

    result = asyncio.run(go())
    srv.shutdown()

    assert result.ok, result
    assert not result.needs_reauth, ("the substring 'unauthorized' in "
                                     "unauthorized_count flagged a 200 as reauth")
    assert result.windows, result
    got = {w.window: w for w in result.windows}
    assert got["weekly"].used_percent == 12.0, got["weekly"]
    assert got["5h"].used_percent == 37.5, got["5h"]
    print("  200 with unauthorized_count -> parsed as a real reading, not reauth")


def test_minimax_1004_in_200_body_triggers_reauth_via_base_resp():
    """MiniMax's structured 1004 reauth used to need a 401 to be caught, but
    the provider has been seen answering 200 with a `base_resp.status_code`
    of 1004. The bare substring `"1004"` is no longer in REAUTH_MARKERS (a
    legitimate response could contain it), so this must come from the
    structured `base_resp` check that runs before the prose scan and works
    on any HTTP status."""
    import http.server
    import json
    import threading
    from dataclasses import replace

    body = json.dumps({"base_resp": {"status_code": 1004,
                                      "status_msg": "cookie is missing, log in again"}})

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            data = body.encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/backend/account/token_plan/remains"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["minimax-ultra"]
        plan = replace(plan, probe=replace(plan.probe, url=url))
        await prober.set_cookie(plan.key, "session=x")
        return await prober.run(plan)

    result = asyncio.run(go())
    srv.shutdown()

    assert not result.ok, result
    assert result.needs_reauth, ("a 200 with base_resp.status_code=1004 must "
                                 "still flag the plan as needs_reauth")
    assert "1004" not in REAUTH_MARKERS, ("the bare '1004' marker would flag "
                                          "any response containing that digit")
    print("  200 with base_resp.status_code=1004 -> needs_reauth, no bare '1004' marker")


def test_non_json_body_with_unauthorized_still_triggers_reauth():
    """A non-JSON 200 with 'unauthorized' in the prose still goes through the
    prose scan — only valid-JSON bodies are exempted, because only a valid
    body could have been the structured 1004 path's target. This is the
    third leg of the contract: 401/403 always reauth, valid-JSON 2xx only
    reauth when `base_resp.status_code` says so, and prose-marker reauth is
    reserved for the case where we genuinely have no structured data."""
    import http.server
    import threading
    from dataclasses import replace

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            data = b"401 Unauthorized: please authenticate"
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/backend/account/token_plan/remains"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["minimax-ultra"]
        plan = replace(plan, probe=replace(plan.probe, url=url))
        await prober.set_cookie(plan.key, "session=x")
        return await prober.run(plan)

    result = asyncio.run(go())
    srv.shutdown()

    assert result.needs_reauth, ("non-JSON body containing 'unauthorized' must "
                                 "still flag as reauth when JSON parsing failed")
    assert not result.ok, result
    print("  non-JSON 200 'unauthorized' -> needs_reauth via prose path")


def test_capture_skipped_on_non_json_response():
    """When the response is not valid JSON, the probe fails with 'response was
    not JSON' and the capture block never runs — even though a Set-Cookie
    header is present. Capture only fires on a verified-good response."""
    import http.server
    import threading
    from dataclasses import replace

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"<!DOCTYPE html><html><body>not json</body></html>"
            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.send_header("set-cookie", "session=anonymous-victim")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/not-json"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["minimax-ultra"]
        probe = replace(plan.probe, url=url, capture_set_cookie=True)
        plan = replace(plan, probe=probe)

        original = "session=original-cookie-value-1234567890"
        await prober.set_cookie(plan.key, original)
        before = dict(redis.hashes["sy:cred:minimax-ultra"])
        result = await prober.run(plan)
        return result, redis.hashes["sy:cred:minimax-ultra"], before

    result, cred, before = run(go())
    srv.shutdown()

    assert not result.ok, result
    assert "response was not JSON" in result.detail, result.detail
    # Credential completely unchanged: capture never ran.
    assert cred.get("cookie") == "session=original-cookie-value-1234567890", cred
    assert cred.get("fingerprint") == before.get("fingerprint"), (
        cred.get("fingerprint"), before.get("fingerprint"))
    assert cred.get("added_at") == before.get("added_at"), (
        cred.get("added_at"), before.get("added_at"))
    print("  non-JSON + Set-Cookie -> capture skipped, credential untouched")


def test_capture_merges_multiple_set_cookie_headers():
    """When the server sends two Set-Cookie headers (one per cookie),
    get_list returns both verbatim and SimpleCookie parses each independently.
    The jar merge overwrites matching names and preserves the rest."""
    import http.server
    import json
    import threading
    from dataclasses import replace

    payload = {"data": {"remains": 7000, "total": 10000}}
    windows = {"weekly": {"remaining": ["data.remains"], "total": ["data.total"]}}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("set-cookie", "a=4")
            self.send_header("set-cookie", "b=3; Path=/; HttpOnly")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/multi-cookie"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["minimax-ultra"]
        probe = replace(plan.probe, url=url, capture_set_cookie=True,
                        windows=windows, fields={})
        plan = replace(plan, probe=probe)

        await prober.set_cookie(plan.key, "a=1; b=2")
        result = await prober.run(plan)
        return result, redis.hashes["sy:cred:minimax-ultra"]

    result, cred = run(go())
    srv.shutdown()

    assert result.ok, result
    # b overwritten by server (3), a merged from server (4)
    assert cred.get("cookie") == "a=4; b=3", cred
    print(f"  two Set-Cookie headers -> merged cookie {cred.get('cookie')!r}")


def test_capture_handles_comma_in_expires_attribute():
    """A Set-Cookie header with Expires=Wed, 21 Oct 2026 ... contains a comma
    that naive header joining would corrupt. get_list returns the raw header
    and SimpleCookie's internal parser correctly isolates the Expires value,
    so only the key=value pair is extracted and the attribute is dropped."""
    import http.server
    import json
    import threading
    from dataclasses import replace

    payload = {"data": {"remains": 7000, "total": 10000}}
    windows = {"weekly": {"remaining": ["data.remains"], "total": ["data.total"]}}
    new_cookie = ("login=WzqUz; Path=/; HttpOnly; "
                  "Expires=Wed, 21 Oct 2026 07:28:00 GMT")

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("set-cookie", new_cookie)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/expires-comma"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["minimax-ultra"]
        probe = replace(plan.probe, url=url, capture_set_cookie=True,
                        windows=windows, fields={})
        plan = replace(plan, probe=probe)

        await prober.set_cookie(plan.key, "auth=1; login=old; extra=k")
        result = await prober.run(plan)
        status = await prober.status(plan.key)
        return result, redis.hashes["sy:cred:minimax-ultra"], status

    result, cred, status = run(go())
    srv.shutdown()

    assert result.ok, result
    # login overwritten, auth and extra preserved
    assert cred.get("cookie") == "auth=1; login=WzqUz; extra=k", cred
    # status() never leaks the cookie value
    blob = repr(status)
    assert "WzqUz" not in blob, blob
    assert "login=WzqUz" not in blob, blob
    print(f"  Expires with comma -> parsed correctly, "
          f"cookie {cred.get('cookie')!r}")


def test_capture_skips_malformed_set_cookie_header():
    """A malformed Set-Cookie from the server (illegal name with a comma)
    raises http.cookies.CookieError on `SimpleCookie.load()`. The capture
    must skip that one header and keep the rest of the jar — a verified-good
    probe must never crash on a single bad header from the provider."""
    import http.server
    import json
    import threading
    from dataclasses import replace

    payload = {"data": {"remains": 7000, "total": 10000}}
    windows = {"weekly": {"remaining": ["data.remains"], "total": ["data.total"]}}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            # Illegal cookie name (comma is not allowed) — SimpleCookie.load
            # would raise CookieError on this string.
            self.send_header("set-cookie", "foo,bar=value")
            self.send_header("set-cookie", "session=ok")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/malformed-cookie"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["minimax-ultra"]
        probe = replace(plan.probe, url=url, capture_set_cookie=True,
                        windows=windows, fields={})
        plan = replace(plan, probe=probe)

        await prober.set_cookie(plan.key, "auth=1")
        result = await prober.run(plan)
        return result, redis.hashes["sy:cred:minimax-ultra"]

    result, cred = run(go())
    srv.shutdown()

    assert result.ok, result
    # Malformed header skipped, good header merged, existing auth preserved.
    assert cred.get("cookie") == "auth=1; session=ok", cred
    print(f"  malformed Set-Cookie -> skipped, "
          f"cookie {cred.get('cookie')!r}")


def test_capture_strips_legacy_set_cookie_attributes():
    """The pre-PR capture stored the raw Set-Cookie header verbatim, so a
    stored value with attributes looked like `session=foo; Path=/; HttpOnly`.
    On the first capture after upgrade, the attribute pairs must be dropped
    so `Path=` doesn't round-trip as a cookie name and the next request
    doesn't send `Cookie: session=newval; Path=/`."""
    import http.server
    import json
    import threading
    from dataclasses import replace

    payload = {"data": {"remains": 7000, "total": 10000}}
    windows = {"weekly": {"remaining": ["data.remains"], "total": ["data.total"]}}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("set-cookie", "session=newval; Path=/; HttpOnly")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/legacy-attrs"

    async def go():
        redis = FakeRedis()
        prober = Prober(redis, Ledger(redis))
        plan = models.load().plans["minimax-ultra"]
        probe = replace(plan.probe, url=url, capture_set_cookie=True,
                        windows=windows, fields={})
        plan = replace(plan, probe=probe)

        # Pre-PR-shaped stored value: raw Set-Cookie with attributes baked in.
        # `Path=` and `HttpOnly` must be stripped, `session` overwritten by
        # the server's `newval`.
        await prober.set_cookie(plan.key, "session=foo; Path=/; HttpOnly")
        result = await prober.run(plan)
        return result, redis.hashes["sy:cred:minimax-ultra"]

    result, cred = run(go())
    srv.shutdown()

    assert result.ok, result
    # Attribute pairs gone, session merged from server.
    assert cred.get("cookie") == "session=newval", cred
    assert "Path" not in cred.get("cookie"), cred
    assert "HttpOnly" not in cred.get("cookie"), cred
    print(f"  legacy attrs stripped -> {cred.get('cookie')!r}")


# ---------------------------------------------------------------------------
# Tests: yard_tokens threading for percent-only readings
# ---------------------------------------------------------------------------
#
# `note_reported_percent` is called with `yard_tokens` -- the window's
# current token tally from `ledger.window_usage` -- so the NEXT percent
# reading can derive an upper-bound allowance from the (prev, current)
# pair. The prober threads this from the matching Quota in `plan.quotas`
# (the Quota whose label matches the window name). Without that thread,
# successive percent readings project nothing: percentage alone is not
# a token figure.
# ---------------------------------------------------------------------------


def test_probe_threads_yard_tokens_into_percent_window():
    """A percent-only reading threads the window's current token tally
    onto `reported_at_yard_tokens` so the next reading has a comparison
    point. The probe picks the Quota whose label matches `r.window` and
    reads its `window_usage`; for a plan whose hash already has tokens
    recorded, the yard figure rides the same `sy:qwin:{plan}:{window}`
    hash as the percentage.
    """
    import http.server
    import json
    import threading
    from dataclasses import replace
    from switchyard.usage import K_WINDOW

    payload = {"model_remains": [{
        "model_name": "general",
        "current_interval_used_percent": "37.5%",
        "current_weekly_used_percent": "12%",
    }]}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/weekly_pct"

    async def go():
        redis = FakeRedis()
        ledger = Ledger(redis)
        prober = Prober(redis, ledger)
        plan = models.load().plans["minimax-ultra"]
        plan = replace(plan, probe=replace(plan.probe, url=url))
        await prober.set_cookie(plan.key, "session=x")

        # Seed the weekly period bucket with prompt + completion tokens.
        # The percent-only branch then threads this yard onto the
        # reported_at_yard_tokens field alongside the new pct reading.
        period_key = K_WINDOW.format(plan=plan.key, window="weekly")
        # window_facts reads the same hash; the percent call lands in
        # the same hash. We pre-seed with a non-zero cost figure and
        # zero tokens to confirm the prober pulls tokens (not cost).
        redis.hashes[period_key] = {}
        # The window_usage helper reads from K_PERIOD.{plan}.{period_key},
        # not from K_WINDOW; seed both for the live path's helper.
        from switchyard.usage import K_PERIOD, period_key as _period_key_fn
        # Find the weekly quota to get its period-derived bucket key.
        weekly_quota = next(q for q in plan.quotas if q.label == "weekly")
        bucket_key = K_PERIOD.format(plan=plan.key,
                                      period=_period_key_fn(weekly_quota.period))
        redis.hashes[bucket_key] = {"prompt_tokens": "1234.0",
                                     "completion_tokens": "567.0",
                                     "cost": "9.99"}
        result = await prober.run(plan)
        facts = await ledger.window_facts(plan.key, "weekly")
        return result, facts

    result, facts = asyncio.run(go())
    srv.shutdown()

    assert result.ok, result
    # Yard figure landed alongside the percent reading: tokens only,
    # not cost. 1234 + 567 = 1801.
    assert facts["reported_at_yard_tokens"] == 1801.0, facts
    assert facts["reported_pct_used"] == 12.0, facts
    print(f"  pct reading threads yard={facts['reported_at_yard_tokens']:.0f} "
          f"tokens alongside pct={facts['reported_pct_used']:.0f}%")


def test_probe_percent_reading_for_window_without_matching_quota_is_graceful():
    """A percent-only reading whose window label does not match any Quota
    in `plan.quotas` still lands; the yard_tokens default to None, the
    carry is skipped on the next call. This is the contract for a probe
    that publishes a window name the plan doesn't know about -- the
    reading is preserved, but the inference path is inert.
    """
    import http.server
    import json
    import threading
    from dataclasses import replace

    # A 'mystery' window the plan has no quota for.
    payload = {"mystery_window": {"used_percent": "50.0"}}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/mystery"

    async def go():
        redis = FakeRedis()
        ledger = Ledger(redis)
        prober = Prober(redis, ledger)
        plan = models.load().plans["minimax-ultra"]
        # Custom windows: a single 'mystery' window reading.
        probe = replace(plan.probe, url=url,
                        windows={"mystery": {"used_percent": ["mystery_window.used_percent"]}},
                        fields={})
        plan = replace(plan, probe=probe)
        await prober.set_cookie(plan.key, "session=x")
        result = await prober.run(plan)
        return result, await ledger.window_facts(plan.key, "mystery")

    result, facts = asyncio.run(go())
    srv.shutdown()

    assert result.ok, result
    assert facts["reported_pct_used"] == 50.0, facts
    # No matching quota -> no yard_tokens stamped.
    assert "reported_at_yard_tokens" not in facts, facts
    print("  unknown window -> pct recorded, yard_tokens absent")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
