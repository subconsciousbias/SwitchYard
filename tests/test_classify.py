"""Classifier tests built from the payloads these providers actually return.

Every case here is sourced from vendor docs or a filed bug report, because the
whole point is that HTTP status codes lie:
  * MiniMax sends insufficient balance as HTTP 500, and sometimes HTTP 200.
  * Z.AI sends rate limits, quota walls and dead subscriptions all as HTTP 429.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from switchyard.classify import (Outcome, classify,  # noqa: E402
                                 escalated_cooldown, inspect_success_payload)


def test_minimax_insufficient_balance_arrives_as_http_500():
    """Verbatim body from MiniMax-AI/MiniMax-M2 issue #62."""
    body = {"type": "error",
            "error": {"type": "api_error", "message": "insufficient balance (1008)"},
            "request_id": "05997465cd2ee4bd50b24b216c911b75"}
    v = classify(500, "insufficient balance (1008)", family="minimax", body=body)
    assert v.outcome is Outcome.QUOTA_EXHAUSTED, v
    assert v.vendor_code == 1008
    # The generic 5xx rule would have said "transient, retry in 60s" and kept
    # feeding the lane's requests into a plan with no money on it.
    assert v.cooldown_seconds >= 900


def test_minimax_exhaustion_hidden_in_an_http_200():
    """base_resp.status_code != 0 on a 200 is a failure, not a success."""
    payload = {"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"},
               "choices": [], "id": "x"}
    v = inspect_success_payload("minimax", payload)
    assert v is not None and v.outcome is Outcome.QUOTA_EXHAUSTED, v
    # And a healthy response must not be flagged.
    assert inspect_success_payload(
        "minimax", {"base_resp": {"status_code": 0}, "choices": [{"message": {}}]}) is None


def test_minimax_rate_limit_is_not_exhaustion():
    v = classify(429, "rate limit", family="minimax",
                 body={"base_resp": {"status_code": 1002, "status_msg": "rate limit"}})
    assert v.outcome is Outcome.RATE_LIMITED and v.cooldown_seconds <= 60, v


def test_minimax_connection_limit_is_its_own_thing():
    """1041 means our max_parallel is too high — a config bug, not a quota wall."""
    v = classify(400, "connection limit exceeded", family="minimax",
                 body={"base_resp": {"status_code": 1041}})
    assert v.outcome is Outcome.CONCURRENCY and v.cooldown_seconds < 60, v


def test_zai_separates_four_different_429s():
    cases = {
        1302: Outcome.RATE_LIMITED,      # request rate limit
        1305: Outcome.RATE_LIMITED,      # overloaded
        1308: Outcome.QUOTA_EXHAUSTED,   # usage limit, resets at a stated time
        1310: Outcome.QUOTA_EXHAUSTED,   # weekly/monthly quota gone
        1113: Outcome.QUOTA_EXHAUSTED,   # balance depleted
        1318: Outcome.QUOTA_EXHAUSTED,   # spend cap
        1309: Outcome.PLAN_DEAD,         # GLM Coding Plan expired
        1314: Outcome.PLAN_DEAD,         # package expired
    }
    for code, expected in cases.items():
        v = classify(429, "refused", family="zai",
                     body={"error": {"code": str(code), "message": "refused"}})
        assert v.outcome is expected, f"zai {code}: got {v.outcome}, want {expected}"


def test_zai_plan_dead_stays_out_for_a_day():
    v = classify(429, "subscription expired", family="zai",
                 body={"error": {"code": "1309"}})
    assert v.cooldown_seconds >= 86400, v


def test_zai_prompt_too_long_never_cools_the_plan():
    v = classify(400, "prompt exceeds length", family="zai",
                 body={"error": {"code": "1261"}})
    assert v.outcome is Outcome.CONTEXT and not v.should_cool and v.is_our_fault, v


def test_unknown_provider_falls_back_to_prose_and_status():
    v = classify(429, "You have hit your usage limit for this window", retry_after=3600)
    assert v.outcome is Outcome.QUOTA_EXHAUSTED and v.cooldown_seconds == 3600, v
    v = classify(429, "slow down", retry_after=10)
    assert v.outcome is Outcome.RATE_LIMITED, v
    v = classify(503, "bad gateway")
    assert v.outcome is Outcome.TRANSIENT, v


def test_a_500_that_means_exhaustion_beats_the_5xx_rule_even_without_a_family():
    v = classify(500, "insufficient balance")
    assert v.outcome is Outcome.QUOTA_EXHAUSTED, v


def test_escalated_cooldown_first_failure_keeps_base():
    """Streak 1 returns base verbatim — today's behaviour preserved."""
    assert escalated_cooldown(60, 1) == 60
    assert escalated_cooldown(60, 0) == 60    # zero counts as "no failures yet"


def test_escalated_cooldown_doubles_each_consecutive_failure():
    """60 -> 120 -> 240 -> 480 -> 960 — a broken seat stops getting re-fed every minute."""
    base = 60
    assert escalated_cooldown(base, 2) == 120
    assert escalated_cooldown(base, 3) == 240
    assert escalated_cooldown(base, 4) == 480
    assert escalated_cooldown(base, 5) == 960


def test_escalated_cooldown_caps_at_max_seconds():
    """The ladder stops doubling once the configured cap is hit.

    Without a cap a long outage would park the plan for an entire afternoon;
    1800 (30 minutes) is enough to outlast most provider blips without
    pretending a broken seat is going to recover on its own.
    """
    base = 60
    cap = 1800
    # 60 * 2**(5-1) = 960 — still under the cap
    assert escalated_cooldown(base, 5, cap=cap) == 960
    # 60 * 2**(6-1) = 1920 — first trip over the cap
    assert escalated_cooldown(base, 6, cap=cap) == 1800
    # And anything past that stays pinned at the cap.
    assert escalated_cooldown(base, 10, cap=cap) == 1800
    assert escalated_cooldown(base, 20, cap=cap) == 1800


def test_escalated_cooldown_uses_supplied_base():
    """The base is whatever classify() returned, so a TRANSIENT with a non-default
    base (e.g. an unclassified 30s) escalates from 30, not from 60."""
    assert escalated_cooldown(30, 2) == 60
    assert escalated_cooldown(15, 2) == 30


def test_sidecar_at_capacity_is_concurrency_with_twenty_second_sit_out():
    """SwitchYard's bridges return {"detail": "sidecar at capacity (N)"} when the
    gate is full — a real concurrency refusal, not a provider rate limit. The
    CONCURRENCY outcome feeds the concurrency learner (ledger + hooks) and
    gives the standard 20s sit-out, matching MiniMax code 1041.
    """
    body = {"detail": "sidecar at capacity (4)"}
    v = classify(429, "sidecar at capacity (4)", body=body)
    assert v.outcome is Outcome.CONCURRENCY, v
    assert v.cooldown_seconds == 20, v


def test_provider_rate_limit_is_unaffected_by_the_new_pattern():
    """A Z.AI-flavoured 429 carrying code 1302 — and prose with no 'sidecar'
    in it — must still classify RATE_LIMITED. The new alternative does not
    pull genuine provider rate limits into CONCURRENCY.
    """
    v = classify(429, "rate limit exceeded", family="zai",
                 body={"error": {"code": "1302", "message": "rate limit exceeded"}})
    assert v.outcome is Outcome.RATE_LIMITED, v


def test_capacity_adjacent_provider_prose_does_not_trigger_concurrency():
    """A 429 whose body merely mentions 'capacity' in provider prose must not
    match — the regex requires the exact 'sidecar at capacity' phrase that only
    our own bridges emit. A loose 'at capacity' would collide with provider
    error messages and is intentionally absent from the pattern.
    """
    body = {"error": {"message": "your account has reached its capacity limit"}}
    v = classify(429, "your account has reached its capacity limit", body=body)
    assert v.outcome is Outcome.RATE_LIMITED, v


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} classifier tests passed")
