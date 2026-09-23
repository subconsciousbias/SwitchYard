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
                                 escalated_cooldown, extract_no_text_tokens,
                                 inspect_success_payload)


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


def test_opencode_text_lost_classifies_as_text_lost_with_ten_second_cooldown():
    """WS1 changes the sidecar to emit
    "<PROVIDER> cli emitted tokens with no text (prompt_tokens=N, completion_tokens=M)"
    when the model charged tokens but the wrapper swallowed the text. The
    classifier must read this as a sidecar quirk — NOT a generic 5xx — and
    cool the plan for ten seconds. The generic 5xx rule would otherwise park
    the plan on the TRANSIENT doubling ladder (60 -> 120 -> 240 -> ...) for a
    ~20-30% instantaneous CLI failure mode that almost always self-heals on
    the next call.
    """
    msg = ("opencode-go cli emitted tokens with no text "
           "(prompt_tokens=239, completion_tokens=22)")
    v = classify(502, msg)
    assert v.outcome is Outcome.TEXT_LOST, v
    assert v.cooldown_seconds == 10, v
    assert v.detail == "cli lost the text", v
    # Cool briefly, like TRANSIENT, but it is not our fault.
    assert v.should_cool is True, v
    assert v.is_our_fault is False, v
    # The legacy 5xx rule would have said "upstream 502" with cooldown 60 —
    # the new contract phrase outranks it.
    assert v.cooldown_seconds < 60, v


def test_text_lost_cooldown_does_not_double_across_consecutive_failures():
    """Consecutive TEXT_LOST failures keep the 10s cooldown unchanged — the
    verdict never feeds into the TRANSIENT doubling ladder. classify() alone
    cannot observe streak math, but the contract is visible in two places:

      1. classify() returns Outcome.TEXT_LOST (not TRANSIENT) every time, so
         _apply_verdict's `verdict.outcome is Outcome.TRANSIENT` branch is
         skipped and bump_and_cool never runs.
      2. The cooldown is 10 every call, so even an un-wired caller cannot
         accidentally double it via Verdict.cooldown_seconds.

    Together these guarantee the plan re-admits itself within ten seconds,
    regardless of how many text-lost failures stacked up before recovery.
    """
    msg = ("opencode-go cli emitted tokens with no text "
           "(prompt_tokens=239, completion_tokens=22)")
    cooldowns = []
    for _ in range(5):
        v = classify(502, msg)
        assert v.outcome is Outcome.TEXT_LOST, v
        assert v.cooldown_seconds == 10, v
        cooldowns.append(v.cooldown_seconds)
    assert cooldowns == [10] * 5, cooldowns
    # And the same input cannot be re-classified as TRANSIENT by accident:
    # the no-text phrase outranks the generic 5xx rule.
    assert classify(502, msg).outcome is not Outcome.TRANSIENT


def test_ordinary_502_without_contract_phrase_still_classifies_transient():
    """Regression guard: a vanilla 502 with no contract phrase keeps the
    today's behaviour. TRANSIENT keeps the 60s base that the breaker ladder
    doubles, so a real upstream outage still stops getting re-fed every minute.
    """
    v = classify(502, "upstream returned an error")
    assert v.outcome is Outcome.TRANSIENT, v
    assert v.cooldown_seconds == 60, v
    # And another 5xx shape the user might mistake for text_lost — no
    # parenthesised counts, no 'no text' phrasing — stays TRANSIENT.
    v2 = classify(500, "the model emitted nothing")
    assert v2.outcome is Outcome.TRANSIENT, v2
    # And the 'no text' phrase without the parenthesised counts is not the
    # contract: it would be unsafe to assume the provider charged tokens.
    v3 = classify(500, "the model emitted no text in this turn")
    assert v3.outcome is Outcome.TRANSIENT, v3


def test_extract_no_text_tokens_returns_counts_for_contract_message():
    """The hook books the charged prompt/completion counts against the plan's
    ledger on a TEXT_LOST failure, so the effective $/Mtok denominator and
    the burn-rate timer see the real spend. extract_no_text_tokens pulls
    exactly those two counts out of the WS1 contract string.
    """
    msg = ("opencode-go cli emitted tokens with no text "
           "(prompt_tokens=239, completion_tokens=22)")
    assert extract_no_text_tokens(msg) == (239, 22), msg
    # Whitespace and case are loose; the contract says "<PROVIDER> ..." so a
    # real message may carry extra prose before the phrase and odd casing.
    assert extract_no_text_tokens(
        "OPENCODE-GO CLI emitted tokens with no text   "
        "(  prompt_tokens = 0 ,  completion_tokens = 0 )") == (0, 0)
    # LiteLLM wraps sidecar HTTP errors into exceptions like
    # `BadGatewayError: ... detail string ...` — the phrase arrives inside
    # the exception's string form, so the helper matches against full prose.
    wrapped = ("BadGatewayError: opencode-go cli emitted tokens with no text "
               "(prompt_tokens=1000, completion_tokens=500)")
    assert extract_no_text_tokens(wrapped) == (1000, 500), wrapped


def test_extract_no_text_tokens_returns_none_for_unrelated_prose():
    """Anything that does not carry the exact contract phrase returns None —
    the hook's "book charged tokens" path is gated on this, so a stray match
    would silently inflate the ledger on a real outage."""
    assert extract_no_text_tokens("insufficient balance") is None
    assert extract_no_text_tokens("") is None
    assert extract_no_text_tokens(
        "opencode-go cli emitted tokens with no text") is None  # no counts
    assert extract_no_text_tokens(
        "opencode-go cli emitted tokens with no text (prompt_tokens=239)"
    ) is None  # only one of the two counts
    assert extract_no_text_tokens(
        "opencode-go cli emitted tokens with no text "
        "(completion_tokens=22, prompt_tokens=239)") is None  # reversed order
    # And a status-only path: the upstream did not even produce tokens.
    assert extract_no_text_tokens("upstream 502 bad gateway") is None


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
