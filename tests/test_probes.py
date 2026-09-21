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


def test_expired_cookie_is_detected_from_minimax_1004():
    body = ('{"base_resp":{"status_code":1004,'
            '"status_msg":"cookie is missing, log in again"}}').lower()
    assert any(m in body for m in REAUTH_MARKERS)
    print("  MiniMax 1004 'cookie is missing' reads as needs-reauth, not a generic failure")


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
    from dataclasses import replace
    from switchyard.usage import headroom

    payload = {"model_remains": [
        {"model_name": "general",
         "start_time": 1789948800000, "end_time": 1789966800000,
         "current_interval_total_count": -1, "current_interval_used_count": -1,
         "current_interval_used_percent": "37.5%",
         "weekly_start_time": 1789948800000, "weekly_end_time": 1790553600000,
         "current_weekly_used_count": -1, "current_weekly_used_percent": "12%"},
        {"model_name": "video", "end_time": 1790035200000,
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


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} probe tests passed")
