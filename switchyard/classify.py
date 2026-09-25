"""Turn a provider's failure into a routing decision.

The most load-bearing file here. If a real quota exhaustion is misread as a
transient blip the lane keeps feeding requests into dead capacity; if a blip is
misread as exhaustion we throw away paid capacity for fifteen minutes.

HTTP status alone is not enough, and the research says so plainly:

  * MiniMax returns insufficient balance (1008) as **HTTP 500**, and its
    OpenAI-compatible endpoint can also return **HTTP 200** with the real
    failure in `base_resp.status_code`. A status-code-only classifier scores
    that as a success and never cools the plan down.
  * Z.AI returns rate limiting, window exhaustion, monthly exhaustion, and
    "your subscription expired" all as **HTTP 429**, separable only by the
    business code in the body.

So: vendor business code first, HTTP status second, prose last.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Outcome(str, Enum):
    QUOTA_EXHAUSTED = "quota_exhausted"   # plan is spent -> drop its slots
    RATE_LIMITED = "rate_limited"         # too fast -> brief sit-out
    CONCURRENCY = "concurrency"           # too many connections -> cap is wrong
    PLAN_DEAD = "plan_dead"               # subscription over -> remove the plan
    AUTH = "auth"                         # credential broken -> alert
    CONTEXT = "context"                   # our fault, never cool the plan
    BAD_REQUEST = "bad_request"           # our fault, never cool the plan
    INTERNAL = "internal"                 # SwitchYard's own fault (session store,
                                          # hook plumbing) -- never cool the provider
    TRANSIENT = "transient"               # retry elsewhere, brief sit-out
    TEXT_LOST = "text_lost"               # model answered, CLI lost the text


LONG = -1          # substitute the caller's default_cooldown
PLAN_DEAD_COOLDOWN = 86_400


# ---------------------------------------------------------------------------
# Vendor business codes. (outcome, cooldown_seconds) — LONG means "use the
# configured default_cooldown for a real quota wall".
# ---------------------------------------------------------------------------

MINIMAX: dict[int, tuple[Outcome, int]] = {
    1000: (Outcome.TRANSIENT, 30),        # unknown error
    1001: (Outcome.TRANSIENT, 15),        # request timeout
    1002: (Outcome.RATE_LIMITED, 30),     # rate limit exceeded
    1004: (Outcome.AUTH, 1800),           # authorization failure / bad key
    1008: (Outcome.QUOTA_EXHAUSTED, LONG),  # insufficient balance  <- via HTTP 500 *or* 200
    1024: (Outcome.TRANSIENT, 60),        # internal system error
    1026: (Outcome.BAD_REQUEST, 0),       # input flagged sensitive
    1027: (Outcome.BAD_REQUEST, 0),       # output flagged sensitive
    1033: (Outcome.TRANSIENT, 60),        # system/database failure
    1039: (Outcome.QUOTA_EXHAUSTED, LONG),  # token limit reached
    1041: (Outcome.CONCURRENCY, 20),      # connection limit exceeded -> cap too high
    2013: (Outcome.BAD_REQUEST, 0),       # invalid request parameters
    2045: (Outcome.RATE_LIMITED, 120),    # rate growth spike detected
    2049: (Outcome.AUTH, 1800),           # invalid API key
    2056: (Outcome.QUOTA_EXHAUSTED, LONG),  # usage quota exhausted for window
}

ZAI: dict[int, tuple[Outcome, int]] = {
    1000: (Outcome.AUTH, 1800),           # authentication failed
    1001: (Outcome.AUTH, 1800),           # missing credentials
    1003: (Outcome.AUTH, 120),            # token expired — a sidecar may refresh
    1005: (Outcome.AUTH, 1800),           # 2FA required
    1113: (Outcome.QUOTA_EXHAUSTED, LONG),  # balance / resource package depleted
    1210: (Outcome.BAD_REQUEST, 0),
    1211: (Outcome.BAD_REQUEST, 0),
    1212: (Outcome.BAD_REQUEST, 0),
    1213: (Outcome.BAD_REQUEST, 0),
    1214: (Outcome.BAD_REQUEST, 0),
    1234: (Outcome.TRANSIENT, 60),        # network error
    1261: (Outcome.CONTEXT, 0),           # prompt exceeds length limit
    1301: (Outcome.BAD_REQUEST, 0),       # safety filter
    1302: (Outcome.RATE_LIMITED, 30),     # request rate limit
    1305: (Outcome.RATE_LIMITED, 60),     # service overloaded
    1308: (Outcome.QUOTA_EXHAUSTED, LONG),  # usage limit; resets at a stated time
    1309: (Outcome.PLAN_DEAD, PLAN_DEAD_COOLDOWN),   # GLM Coding Plan expired
    1310: (Outcome.QUOTA_EXHAUSTED, LONG),  # weekly/monthly quota exhausted
    1311: (Outcome.PLAN_DEAD, PLAN_DEAD_COOLDOWN),   # plan lacks model access
    1313: (Outcome.RATE_LIMITED, 300),    # fair-use policy throttle
    1314: (Outcome.PLAN_DEAD, PLAN_DEAD_COOLDOWN),   # enterprise package expired
    1315: (Outcome.PLAN_DEAD, PLAN_DEAD_COOLDOWN),   # key restricted
    **{c: (Outcome.QUOTA_EXHAUSTED, LONG) for c in range(1316, 1322)},  # spend caps
}

FAMILIES: dict[str, dict[int, tuple[Outcome, int]]] = {
    "minimax": MINIMAX,
    "zai": ZAI,
}

# Codes that mean "out of capacity" regardless of the HTTP status wrapping them.
# MiniMax 1008 arriving as a 500 is the case that makes this necessary.
_OVERRIDES_STATUS = {Outcome.QUOTA_EXHAUSTED, Outcome.PLAN_DEAD, Outcome.CONCURRENCY, Outcome.AUTH}


# ---------------------------------------------------------------------------
# Prose fallbacks, for providers with no business code at all.
# ---------------------------------------------------------------------------
_EXHAUSTED = re.compile(
    r"(usage limit|quota exceeded|quota exhausted|insufficient (balance|quota|credit)"
    r"|out of credit|credit limit|monthly limit|weekly limit|plan limit"
    # Bare "limit reached" used to swallow unrelated "connection limit reached"
    # / "rate limit reached" strings; the noun prefix keeps it anchored to a
    # quota wall while leaving "usage limit", "weekly limit", etc. intact.
    # The separators inside and after the group are literal spaces — `?.` /
    # bare `.` would match any single character, which is misleading for the
    # next reader and lets "usageXlimit reached" / "tokenXquota limit reached"
    # through on shape alone (no provider emits those today, but a future
    # upstream prose change could).
    r"|(?:usage|monthly|weekly|daily|plan|token quota) limit reached|no remaining|exceeded your current quota"
    r"|resource_exhausted|arrearage"
    # xAI via OpenCode says this when a SuperGrok subscription's quota is spent:
    # "personal-team-blocked:spending-limit: You have run out of credits or need a
    # Grok subscription." It reads like a dead account but the plan refills, so it
    # is a quota wall, not an expired subscription.
    r"|spending.?limit|run out of credits)",
    re.I,
)
# "too many tokens" only counts as a context-window hit when the message also
# names context|prompt|input somewhere in the same sentence — a free-floating
# "too many tokens per minute, slow down" is a per-minute rate-limit signal,
# not our prompt. OpenAI's real "maximum context length is N tokens" already
# matches via `maximum context`, so the pairing form only has to catch
# nonstandard phrasings.
_CONTEXT = re.compile(
    r"(context (length|window)"
    r"|(?:(?:context|prompt|input)[^.]{0,80}?too many tokens"
    r"|too many tokens[^.]{0,80}?(?:in|for|to).{0,40}(?:context|prompt|input))"
    r"|maximum context|prompt is too long)",
    re.I,
)
_AUTH = re.compile(r"(invalid api key|unauthorized|authentication|invalid token|expired token)", re.I)
# "sidecar at capacity" is the exact phrase SwitchYard's own bridges emit on a
# full gate (mcp_bridge/server.py, cli_bridge/server.py) — the number after it
# is the configured concurrency cap, not a vendor code, so this never collides
# with the OpenAI/MiniMax-style "(NNNN)" patterns that _CODE_IN_TEXT reads.
_CONCURRENCY = re.compile(r"(concurrenc|connection limit|too many connections|max_parallel|sidecar at capacity)", re.I)
_PLAN_DEAD = re.compile(
    r"(subscription (has )?(expired|ended)|plan expired|package expired"
    r"|no access to model)", re.I)
# WS1 changes the sidecar to emit "<PROVIDER> cli emitted tokens with no text
# (prompt_tokens=N, completion_tokens=M)" when the CLI produced tokens but the
# gateway lost the answer text. A 20-30% instantaneous CLI quirk therefore
# arrives as a 502 whose real semantics are "model charged us, the wrapper
# swallowed it" — instant, per-request, not a capacity outage. The cooldown is
# short and never doubles: the next request is almost certainly fine, and a
# long streak would just park the plan for minutes over a benign UI race.
# The two parenthesised counts are required so `extract_no_text_tokens` can
# book the charged tokens against the plan's ledger without guessing.
_NO_TEXT = re.compile(
    r"cli emitted tokens with no text"
    r"\s*\(\s*prompt_tokens\s*=\s*(\d+)\s*,\s*completion_tokens\s*=\s*(\d+)\s*\)",
    re.I,
)

# A vendor code parenthesised in the message, e.g. "insufficient balance (1008)".
_CODE_IN_TEXT = re.compile(r"\((\d{4,5})\)")


@dataclass
class Verdict:
    outcome: Outcome
    cooldown_seconds: int
    detail: str = ""
    vendor_code: int | None = None
    # True when the verdict's cooldown must also drop the caller's session
    # lease, so the next pick re-leases onto a sibling instead of being
    # pinned to the failing deployment. Used for upstream route/deployment
    # failures (HTTP 404): the sidecar URL is gone and the session lease
    # would just keep the conversation routed to a dead deployment.
    drop_lease: bool = False

    @property
    def should_cool(self) -> bool:
        return self.cooldown_seconds > 0

    @property
    def is_our_fault(self) -> bool:
        return self.outcome in (Outcome.CONTEXT, Outcome.BAD_REQUEST, Outcome.INTERNAL)


def _as_dict(body: Any) -> dict:
    if isinstance(body, dict):
        return body
    if isinstance(body, (str, bytes)):
        try:
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def extract_vendor_code(message: str, body: Any = None, family: str | None = None) -> int | None:
    """Dig the vendor's own numeric code out of wherever it put it."""
    doc = _as_dict(body)

    # MiniMax: the authoritative field, and the one that can accompany HTTP 200.
    base = doc.get("base_resp")
    if isinstance(base, dict) and base.get("status_code") not in (None, 0, "0"):
        try:
            return int(base["status_code"])
        except (TypeError, ValueError):
            pass

    # Z.AI and most OpenAI-shaped errors: error.code
    err = doc.get("error")
    if isinstance(err, dict):
        for field in ("code", "type"):
            val = err.get(field)
            if val is None:
                continue
            try:
                return int(val)
            except (TypeError, ValueError):
                m = _CODE_IN_TEXT.search(str(val))
                if m:
                    return int(m.group(1))
        m = _CODE_IN_TEXT.search(str(err.get("message", "")))
        if m:
            return int(m.group(1))

    for field in ("code", "status_code"):
        val = doc.get(field)
        if val not in (None, 0, "0"):
            try:
                return int(val)
            except (TypeError, ValueError):
                pass

    m = _CODE_IN_TEXT.search(message or "")
    return int(m.group(1)) if m else None


def extract_no_text_tokens(message: str) -> tuple[int, int] | None:
    """Pull the charged prompt/completion counts from a TEXT_LOST failure.

    The WS1 contract is "{PROVIDER} cli emitted tokens with no text
    (prompt_tokens=N, completion_tokens=M)"; the counts are what the
    provider charged the request for, even though no answer text came back.
    Returns None when the message does not carry the contract shape, so a
    hook can decide whether to book tokens without having to share a regex
    with the classifier — the same regex is used here so the two cannot
    drift apart.
    """
    m = _NO_TEXT.search(message or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def classify(
    status: int | None,
    message: str,
    *,
    family: str | None = None,
    body: Any = None,
    retry_after: float | None = None,
    default_cooldown: int = 900,
) -> Verdict:
    msg = message or ""

    # 1. Vendor business code — the only reliable signal for MiniMax and Z.AI.
    code = extract_vendor_code(msg, body, family)
    table = FAMILIES.get((family or "").lower(), {})
    if code is not None and code in table:
        outcome, cooldown = table[code]
        if cooldown == LONG:
            # Honour a stated reset time when it looks like a real window.
            cooldown = int(retry_after) if retry_after and retry_after > 60 else default_cooldown
        return Verdict(outcome, cooldown, f"{family} code {code}", code)

    # 2. status-429 with explicit rate-limit phrasing — a per-minute / per-second
    #    wall must land as RATE_LIMITED, not QUOTA_EXHAUSTED (which would park
    #    the plan for 15 minutes) and not CONTEXT (which would label the
    #    provider's throttle as our fault and never cool the plan at all).
    #    This sits before the prose block so a "429 + 'tokens per min'"
    #    message wins over _CONTEXT and _EXHAUSTED.
    if status == 429 and re.search(
        r"per.(min(ute)?|second|hour)|rate.?limit|RPM|TPM", msg, re.I,
    ):
        secs = int(retry_after) if retry_after and retry_after > 0 else 30
        return Verdict(Outcome.RATE_LIMITED, max(5, min(secs, 300)), "rate limited", code)

    # 3. Unambiguous prose, which can outrank a misleading status (a 500 that
    #    says "insufficient balance" is not a server problem we should retry).
    if _CONTEXT.search(msg):
        return Verdict(Outcome.CONTEXT, 0, "context window exceeded", code)
    if _PLAN_DEAD.search(msg):
        return Verdict(Outcome.PLAN_DEAD, PLAN_DEAD_COOLDOWN, "subscription expired", code)
    if _EXHAUSTED.search(msg):
        secs = int(retry_after) if retry_after and retry_after > 60 else default_cooldown
        return Verdict(Outcome.QUOTA_EXHAUSTED, secs, "quota exhausted (from message)", code)
    if _CONCURRENCY.search(msg):
        return Verdict(Outcome.CONCURRENCY, 20, "connection limit", code)
    if _NO_TEXT.search(msg):
        # The model answered and the CLI swallowed the text — a sidecar quirk,
        # not a capacity outage. Cool briefly (10s) so a retry lands on a
        # recovered wrapper, but never feed this into the TRANSIENT doubling
        # ladder: a 20-30% instantaneous failure rate would otherwise park the
        # plan for minutes (60 -> 120 -> 240 -> ...). _apply_verdict skips
        # bump_and_cool for any outcome that is not TRANSIENT, so consecutive
        # TEXT_LOST failures keep writing 10s and the plan re-admits itself.
        return Verdict(Outcome.TEXT_LOST, 10, "cli lost the text", code)

    # 4. HTTP status. _AUTH prose is gated behind status is None, which is
    #    broader than just the "5xx" case mentioned in issue #185: ANY HTTP
    #    status on the wire defers to the status branch — 4xx non-401/403
    #    bodies that mention "authentication" / "unauthorized" fall through
    #    to BAD_REQUEST, 5xx bodies fall through to TRANSIENT, and only the
    #    no-HTTP-context case keeps the prose-driven AUTH fallback. The
    #    strict gate is the issue #185 spec ("_AUTH prose: only when status
    #    in (None, 401, 403)") and the rationale is "trust HTTP status
    #    first" — the prose is only safe to read as "this caller did not
    #    send credentials" when there is no HTTP status to contradict it.
    if status in (401, 403):
        return Verdict(Outcome.AUTH, 1800, "credential rejected", code)
    elif status is None and _AUTH.search(msg):
        return Verdict(Outcome.AUTH, 1800, "credential rejected", code)
    if status == 402:
        return Verdict(Outcome.QUOTA_EXHAUSTED, default_cooldown, "payment required", code)
    if status == 429:
        secs = int(retry_after) if retry_after else 30
        return Verdict(Outcome.RATE_LIMITED, max(5, min(secs, 300)), "rate limited", code)
    if status in (408, 499) or status is None:
        return Verdict(Outcome.TRANSIENT, 15, "timeout", code)
    if status and 500 <= status < 600:
        return Verdict(Outcome.TRANSIENT, 60, f"upstream {status}", code)
    if status == 404:
        # Upstream route/deployment failure: the sidecar URL is gone (the
        # image was rolled, the path was renamed, the model was pulled). It
        # is NOT a caller error -- a 4xx status by itself is not proof the
        # client did anything wrong -- so cool the plan like a TRANSIENT and
        # also drop the session lease (drop_lease=True) so the next pick
        # re-leases onto a live sibling. Without the lease drop the session
        # would keep landing on the dead deployment every turn until the
        # cooldown TTL passed.
        return Verdict(
            Outcome.TRANSIENT, 60, "upstream route/deployment missing", code,
            drop_lease=True,
        )
    if status and 400 <= status < 500:
        return Verdict(Outcome.BAD_REQUEST, 0, f"client error {status}", code)
    return Verdict(Outcome.TRANSIENT, 30, "unclassified", code)


def inspect_success_payload(family: str | None, payload: Any) -> Verdict | None:
    """Catch a failure that arrived dressed as HTTP 200.

    MiniMax's OpenAI-compatible endpoint can return 200 with the real status in
    `base_resp.status_code` (0 means success). Without this check a quota-dead
    plan looks healthy, gets recorded as a success, and keeps receiving every
    request the lane can give it. Returns None when the response is genuinely OK.
    """
    doc = _as_dict(payload)
    if not doc:
        return None

    base = doc.get("base_resp")
    if isinstance(base, dict):
        raw = base.get("status_code")
        try:
            sc = int(raw) if raw is not None else 0
        except (TypeError, ValueError):
            sc = 0
        if sc not in (0,):
            return classify(
                200, str(base.get("status_msg") or f"base_resp {sc}"),
                family=family, body=doc,
            )

    # An OpenAI-shaped error object inside a 200 is equally a failure.
    err = doc.get("error")
    if isinstance(err, dict) and (err.get("message") or err.get("code")):
        return classify(200, str(err.get("message") or ""), family=family, body=doc)
    return None


def escalated_cooldown(base: int, streak: int, cap: int = 1800) -> int:
    """Double the sit-out each consecutive transient failure, capped at `cap`.

    First failure is exactly `base` (so today's behaviour is preserved: a 5xx
    still cools for 60s on its first appearance). Each subsequent failure in
    a row doubles — 60, 120, 240, 480, 960, … — until the cap is hit. The
    cap stops a long outage from parking a plan for an entire afternoon; the
    ladder is what makes a broken seat stop getting re-fed every 60s.

    The streak itself is reset by the first successful call against the plan,
    so the cooldown returns to `base` after recovery — no manual un-cool
    required.
    """
    if streak <= 1:
        return base
    return min(base * (2 ** (streak - 1)), cap)
