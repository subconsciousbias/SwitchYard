"""Plan promotions: bonus credits and one-off resets with an expiry.

The store is set-style (a post replaces the list), validates the whole post
before writing, and the board flags a live promotion near its expiry. No
network: Redis is a FakeRedis, the portal runs under TestClient.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from plans_path import plans_path  # noqa: E402

os.environ["SWITCHYARD_PLANS"] = plans_path()

import redis.asyncio as _redis_async                    # noqa: E402
from tests.fake_redis import FakeRedis                  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from switchyard.portal import app as portal_app  # noqa: E402
from switchyard.promotions import Promotions, parse  # noqa: E402

run = asyncio.run
ORIGIN = {"Origin": "http://localhost:4001"}


def _credit(days, remaining=250.0, pid="cloud-credit"):
    return {"id": pid, "kind": "credit", "label": "Cloud session credit",
            "applies_to": "cloud sessions", "total": 250, "remaining": remaining,
            "currency": "USD", "expires_at": time.time() + days * 86400}


def test_parse_accepts_iso_and_milliseconds_and_rejects_junk():
    iso = parse({"id": "a", "expires_at": "2026-11-05T07:59:00+00:00"})
    ms = parse({"id": "b", "expires_at": 1793865540000})
    assert abs(iso.expires_at - 1793865540.0) < 1, iso.expires_at
    assert abs(ms.expires_at - 1793865540.0) < 1, ms.expires_at
    for bad in ({}, {"id": "x", "kind": "coupon"}, {"id": "x", "total": "lots"},
                {"id": "x", "expires_at": "next tuesday"}, {"id": "x", "total": True}):
        try:
            parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    print("  ISO and ms timestamps parse; missing id / bad kind / bad numbers refused")


def test_a_post_replaces_the_list_and_a_bad_post_changes_nothing():
    async def go():
        store = Promotions(FakeRedis())
        await store.replace("p", [_credit(40), _credit(3, pid="other")], source="t")
        first = await store.view("p", alert_days=7)
        try:
            await store.replace("p", [_credit(40), {"id": ""}])
        except ValueError:
            pass
        after_bad = await store.view("p", alert_days=7)
        await store.replace("p", [])
        cleared = await store.view("p", alert_days=7)
        return first, after_bad, cleared
    first, after_bad, cleared = run(go())
    assert [p["id"] for p in first["items"]] == ["other", "cloud-credit"], first
    assert first["source"] == "t"
    assert after_bad["items"] == first["items"], "a rejected post must not write"
    assert cleared["items"] == [], cleared
    print("  soonest expiry first; rejected post leaves the list; [] clears it")


def test_expiring_soon_only_when_live_close_and_unspent():
    now = time.time()
    rows = {pid: parse(p, now).view(now, alert_days=7) for pid, p in {
        "close": _credit(3), "far": _credit(40),
        "spent": _credit(3, remaining=0), "gone": _credit(-1)}.items()}
    assert rows["close"]["expiring_soon"] is True
    assert rows["far"]["expiring_soon"] is False
    assert rows["spent"]["expiring_soon"] is False and rows["spent"]["used_up"]
    assert rows["gone"]["expiring_soon"] is False and rows["gone"]["expired"]
    print("  3d left -> flagged; 40d, used up, or expired -> not flagged")


def test_portal_endpoint_board_and_api_state():
    # Patched for this test only: other test modules install their own fake
    # at import, and a module-level swap here would hand them this one.
    original = _redis_async.Redis.from_url
    _redis_async.Redis.from_url = lambda url, **_: FakeRedis()
    try:
        _portal_round_trip()
    finally:
        _redis_async.Redis.from_url = original


def _portal_round_trip():
    with TestClient(portal_app.app, base_url="http://localhost:4001") as client:
        plan = next(iter(portal_app.state["registry"].plans))
        ok = client.post(f"/admin/plans/{plan}/promotions", headers=ORIGIN,
                         json={"promotions": [_credit(3)], "source": "test"})
        assert ok.status_code == 200 and ok.json()["promotions"] == 1, ok.text
        bad = client.post(f"/admin/plans/{plan}/promotions", headers=ORIGIN,
                          json={"promotions": [{"kind": "credit"}]})
        assert bad.status_code == 400, bad.text
        missing = client.post("/admin/plans/no-such-plan/promotions", headers=ORIGIN,
                              json={"promotions": []})
        assert missing.status_code == 404, missing.text
        foreign = client.post(f"/admin/plans/{plan}/promotions",
                              headers={"Origin": "http://evil.example"},
                              # Same list again, so the board checks below
                              # hold whether or not the guard is present.
                              json={"promotions": [_credit(3)], "source": "test"})
        assert foreign.status_code == 403, "the admin origin guard must cover it"

        html = client.get("/fragments/plans").text
        assert "Cloud session credit" in html and "3d left" in html, "board line"
        state = client.get("/api/state").json()
        row = next(p for p in state["plans"] if p["key"] == plan)
        assert row["promotions"]["items"][0]["expiring_soon"] is True, row
        assert any("expires in" in a for a in row["alerting"]), row["alerting"]
        listing = client.get("/api/promotions").json()
        assert listing[plan]["items"][0]["id"] == "cloud-credit", listing
    print(f"  {plan}: posted, shown on the board, alerting, in /api/state; "
          "bad post 400, unknown plan 404, foreign origin 403")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
