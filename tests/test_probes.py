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
os.environ.setdefault("SWITCHYARD_PLANS", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "plans.yaml"))

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
        # Only a length and the last four characters — enough to tell two
        # cookies apart, not enough to use one.
        assert st["fingerprint"].startswith(f"{len(secret)} chars ending ")
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


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} probe tests passed")
