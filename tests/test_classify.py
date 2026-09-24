"""Classifier tests built from the payloads these providers actually return.

Every case here is sourced from vendor docs or a filed bug report, because the
whole point is that HTTP status codes lie:
  * MiniMax sends insufficient balance as HTTP 500, and sometimes HTTP 200.
  * Z.AI sends rate limits, quota walls and dead subscriptions all as HTTP 429.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)  # so `import conftest` resolves under plain `python3`

import conftest  # noqa: F401  (socket guard for plain-script mode)

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


def test_429_with_rate_limit_phrasing_outranks_prose():
    """A 429 carrying rate-limit phrasing must land as RATE_LIMITED even when
    the message also reads as CONTEXT ("too many tokens per minute") or
    QUOTA_EXHAUSTED ("rate limit reached for … tokens per min"). Without this
    pre-prose rule the message hits the prose block first and the per-minute
    wall becomes a 15-minute quota park (or, worse, "our fault" and never
    cooled at all).
    """
    rows = [
        # The verbatim issue: 429 "Rate limit reached for ... tokens per min"
        # was being classified as QUOTA_EXHAUSTED, parking the plan for 15
        # minutes over a per-minute wall.
        (429, "Rate limit reached for 16384 tokens per min",
         Outcome.RATE_LIMITED, lambda c: 5 <= c <= 300),
        # "too many tokens per minute" used to match _CONTEXT (it's our
        # fault, never cool). It is not: the provider is throttling by
        # tokens-per-minute, which is a rate-limit signal.
        (429, "too many tokens per minute, slow down",
         Outcome.RATE_LIMITED, lambda c: 5 <= c <= 300),
        # Same shape with a different window unit — per-second, per-hour.
        (429, "request rate limit: 100 per second, slow down",
         Outcome.RATE_LIMITED, lambda c: 5 <= c <= 300),
        (429, "you exceeded 5000 RPM for this model",
         Outcome.RATE_LIMITED, lambda c: 5 <= c <= 300),
        (429, "TPM cap reached, retry in 30s",
         Outcome.RATE_LIMITED, lambda c: 5 <= c <= 300),
    ]
    for status, message, expected, cd_pred in rows:
        v = classify(status, message)
        assert v.outcome is expected, f"{status} {message!r}: got {v.outcome}, want {expected}: {v}"
        assert cd_pred(v.cooldown_seconds), f"{status} {message!r}: cooldown {v.cooldown_seconds}: {v}"
        assert v.detail == "rate limited", v
    # The "too many tokens per minute" branch is explicitly NOT our fault.
    v = classify(429, "too many tokens per minute, slow down")
    assert v.is_our_fault is False, v
    # And retry_after must still be honoured when supplied: >0 means use it
    # (clamped to 5..300), absent/0 means fall back to 30s.
    v = classify(429, "tokens per min, slow down", retry_after=10)
    assert v.cooldown_seconds == 10, v
    v = classify(429, "tokens per min, slow down", retry_after=600)
    assert v.cooldown_seconds == 300, v  # clamped to the 300 ceiling
    v = classify(429, "tokens per min, slow down", retry_after=2)
    assert v.cooldown_seconds == 5, v    # clamped to the 5 floor


def test_500_authentication_prose_falls_through_to_transient():
    """A 5xx body that mentions 'authentication service unavailable' is an
    upstream service outage, not a credential rejection. Before this fix the
    _AUTH prose matched any 'authentication' substring regardless of status,
    so a 500/502 from an auth-subsystem outage was mis-classified as AUTH and
    the plan was cooled for 30 minutes (1800s) instead of the 60s TRANSIENT
    ladder.

    After the fix _AUTH prose only fires when status is None — the only
    case where 'unauthorized' / 'authentication' really can be read as
    'this caller did not send credentials', because there is no HTTP context
    to contradict that reading.
    """
    # 500 with AUTH prose → TRANSIENT, not AUTH.
    v = classify(500, "Internal error: authentication service unavailable")
    assert v.outcome is Outcome.TRANSIENT, v
    assert v.cooldown_seconds == 60, v
    # 502 with the same phrasing → TRANSIENT (it's an upstream bad-gateway).
    v = classify(502, "BadGateway: authentication subsystem returned 502")
    assert v.outcome is Outcome.TRANSIENT, v
    assert v.cooldown_seconds == 60, v
    # status=None with the same prose → still AUTH (no HTTP context, the
    # caller's AUTH gate is the only thing that could have produced this).
    v_none = classify(None, "authentication service unavailable")
    assert v_none.outcome is Outcome.AUTH, v_none
    assert v_none.cooldown_seconds == 1800, v_none
    # status=None with 'unauthorized' → AUTH (prose fallback for no-HTTP).
    v_none2 = classify(None, "unauthorized")
    assert v_none2.outcome is Outcome.AUTH, v_none2
    assert v_none2.cooldown_seconds == 1800, v_none2
    # And 401 with 'invalid api key' → AUTH (status-driven, unchanged).
    v_401 = classify(401, "invalid api key")
    assert v_401.outcome is Outcome.AUTH, v_401
    assert v_401.cooldown_seconds == 1800, v_401


def test_auth_prose_strict_gate_4xx_does_not_fall_back_to_prose():
    """The strict _AUTH gate (`status is None` for prose) is broader than
    "5xx" — it means *any* HTTP status code on the wire defers to the
    status branch. The PR description and the code comment in
    `classify.py` frame the change as the 5xx case, but a 4xx (non-401/403)
    body that mentions "unauthorized" / "authentication" also falls through
    to the 4xx status branch — and lands as BAD_REQUEST 0s, not AUTH 1800s.

    That is the intentional design per issue #185 (the gate is
    `status in (None, 401, 403)` — anything else, prose is ignored), but
    it is a real behavior change worth pinning: a provider that
    mis-classifies an auth failure as 400 (with "unauthorized" in the
    body) used to trigger an AUTH alert + lease drop (`hooks.py:1210`);
    now it lands as a no-cooldown BAD_REQUEST and the plan keeps getting
    re-fed against a broken credential. The other cases are unaffected
    by the strict gate because the status branch already gives them the
    correct verdict.

    The status=None case is the only one where the prose fallback really
    can be read as "this caller did not send credentials" — there is no
    HTTP context to contradict that reading.
    """
    # 400 + 'unauthorized' → BAD_REQUEST 0s (strict gate: status branch wins).
    # A provider that mis-classifies an auth failure as 400 with the word
    # 'unauthorized' in the body used to be AUTH 1800s; the strict gate
    # routes it to BAD_REQUEST instead.
    v_400 = classify(400, "unauthorized")
    assert v_400.outcome is Outcome.BAD_REQUEST, v_400
    assert v_400.cooldown_seconds == 0, v_400
    # 404 + 'unauthorized' → BAD_REQUEST 0s (same reasoning).
    v_404 = classify(404, "unauthorized")
    assert v_404.outcome is Outcome.BAD_REQUEST, v_404
    assert v_404.cooldown_seconds == 0, v_404
    # 400 + 'authentication failed' → BAD_REQUEST 0s (not AUTH).
    v_400b = classify(400, "authentication failed")
    assert v_400b.outcome is Outcome.BAD_REQUEST, v_400b
    assert v_400b.cooldown_seconds == 0, v_400b
    # 429 + 'unauthorized access' → RATE_LIMITED 30s (status 429 wins;
    # the 429 rate-limit rule does not match 'unauthorized access' and
    # the strict AUTH gate ignores the prose).
    v_429 = classify(429, "unauthorized access")
    assert v_429.outcome is Outcome.RATE_LIMITED, v_429
    # And the AUTH-prose outcomes the strict gate preserves: status=None
    # is the only case where the prose fallback fires.
    v_none = classify(None, "unauthorized")
    assert v_none.outcome is Outcome.AUTH, v_none
    assert v_none.cooldown_seconds == 1800, v_none
    # 401/403 status-driven AUTH is unchanged.
    v_401 = classify(401, "invalid api key")
    assert v_401.outcome is Outcome.AUTH, v_401
    assert v_401.cooldown_seconds == 1800, v_401
    v_403 = classify(403, "forbidden")
    assert v_403.outcome is Outcome.AUTH, v_403
    assert v_403.cooldown_seconds == 1800, v_403


def test_prose_rules_preserve_genuine_message_classification():
    """The narrowed _EXHAUSTED and _CONTEXT regexes must still cover the
    load-bearing genuine-message shapes. Each row is a real vendor message
    (or close paraphrase) — the row's expected outcome is the contract a
    broken regex would silently break.
    """
    rows = [
        # --- MiniMax ---------------------------------------------------------
        # 1008 prose, no body — hits _EXHAUSTED via 'insufficient balance'.
        (500, "insufficient balance", {"family": "minimax"},
         Outcome.QUOTA_EXHAUSTED, lambda c: c >= 900, None),
        # 1008 with the (1008) in the body — vendor-code branch fires and
        # surfaces the code so the audit ledger can attribute the spend.
        (500, "insufficient balance (1008)",
         {"family": "minimax",
          "body": {"error": {"message": "insufficient balance (1008)"}}},
         Outcome.QUOTA_EXHAUSTED, lambda c: c >= 900, 1008),
        # --- Z.AI prose without an explicit code -----------------------------
        # 1308 prose: 'usage limit; resets…' matches via 'usage limit' even
        # though the narrowed 'limit reached' no longer matches it directly.
        (429, "usage limit; resets in 24h", {},
         Outcome.QUOTA_EXHAUSTED, lambda c: c >= 900, None),
        # 1310 prose: 'weekly limit reached' matches via 'weekly limit'
        # (the narrowed 'limit reached' alternative also covers it via the
        # explicit weekly-prefix form, so the test asserts both paths).
        (429, "weekly limit reached", {},
         Outcome.QUOTA_EXHAUSTED, lambda c: c >= 900, None),
        # Z.AI 1302 'rate limit exceeded' as 429, no body — the new 429
        # rate-limit rule fires via 'rate limit' phrasing, no vendor code
        # needed. The cooldown uses the 30s fallback (no retry_after) and
        # lands in the 5..300 clamp.
        (429, "rate limit exceeded", {},
         Outcome.RATE_LIMITED, lambda c: 5 <= c <= 300, None),
        # --- OpenAI ----------------------------------------------------------
        # 'maximum context length is N tokens' on 400 — CONTEXT, our fault,
        # zero cooldown (we never park a plan over a prompt the caller
        # could fix).
        (400, "This model's maximum context length is 8192 tokens.", {},
         Outcome.CONTEXT, lambda c: c == 0, None),
        # And the OpenAI-shaped 400 with an 'invalid_request_error' code
        # in the body — also CONTEXT, still is_our_fault.
        (400, "context window exceeded: please shorten the prompt",
         {"body": {"error": {"code": "context_length_exceeded"}}},
         Outcome.CONTEXT, lambda c: c == 0, None),
    ]
    for status, message, kwargs, expected, cd_pred, expected_code in rows:
        v = classify(status, message, **kwargs)
        assert v.outcome is expected, \
            f"({status!r}, {message!r}, {kwargs!r}): got {v.outcome}, want {expected}: {v}"
        assert cd_pred(v.cooldown_seconds), \
            f"({status!r}, {message!r}, {kwargs!r}): cooldown {v.cooldown_seconds}: {v}"
        if expected_code is not None:
            assert v.vendor_code == expected_code, \
                f"({status!r}, {message!r}, {kwargs!r}): vendor_code {v.vendor_code}, want {expected_code}: {v}"
    # is_our_fault is a separate predicate the table doesn't carry: CONTEXT
    # rows must be our fault, the RATE_LIMITED/QUOTA rows must not be.
    v_ctx = classify(400, "This model's maximum context length is 8192 tokens.")
    assert v_ctx.is_our_fault is True, v_ctx
    v_rl = classify(429, "rate limit exceeded")
    assert v_rl.is_our_fault is False, v_rl


def test_429_negative_guards_remain_classified():
    """The 429 pre-prose rule must NOT swallow genuine non-rate-limit 429s.
    Each row is a real shape the system produces today, and a regression
    here would mean a CONCURRENCY signal becomes RATE_LIMITED (the
    concurrency learner stops seeing the backpressure), a quota wall
    becomes a per-minute sit-out (the plan re-admits itself too soon), or
    an AUTH signal never reaches the credential-rejection path.
    """
    rows = [
        # 'sidecar at capacity (N)' is the bridge's own backpressure phrase;
        # it must stay CONCURRENCY 20s, matching MiniMax 1041.
        (429, "sidecar at capacity (4)", {},
         Outcome.CONCURRENCY, lambda c: c == 20),
        # 'usage limit reached' (no per-minute / rate-limit / RPM / TPM
        # phrasing) — stays QUOTA_EXHAUSTED, parked for the default
        # cooldown window.
        (429, "usage limit reached", {},
         Outcome.QUOTA_EXHAUSTED, lambda c: c >= 900),
        # 401 'invalid api key' — status-driven AUTH, 1800s.
        (401, "invalid api key", {},
         Outcome.AUTH, lambda c: c == 1800),
        # status=None 'unauthorized' — prose-fallback AUTH when no HTTP
        # context is available.
        (None, "unauthorized", {},
         Outcome.AUTH, lambda c: c == 1800),
        # 'connection limit reached' on 429 — no rate-limit phrasing, so
        # the 429 rule doesn't fire. _CONCURRENCY prose catches it (the
        # bare 'limit reached' was the path the regex narrowing removed,
        # and 'connection limit' is still in the regex).
        (429, "connection limit reached", {},
         Outcome.CONCURRENCY, lambda c: c == 20),
        # 'too many tokens' on a status other than 429 with no
        # context|prompt|input pairing and no per-minute phrasing — must
        # fall through to TRANSIENT, NOT CONTEXT.
        (500, "we sent too many tokens last call", {},
         Outcome.TRANSIENT, lambda c: c == 60),
    ]
    for status, message, kwargs, expected, cd_pred in rows:
        v = classify(status, message, **kwargs)
        assert v.outcome is expected, \
            f"{status!r} {message!r} {kwargs!r}: got {v.outcome}, want {expected}: {v}"
        assert cd_pred(v.cooldown_seconds), \
            f"{status!r} {message!r} {kwargs!r}: cooldown {v.cooldown_seconds}: {v}"


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
