"""Turn a provider's failure into a routing decision.

The single most load-bearing piece of the design: if a real quota exhaustion
is misread as a transient error, the plan keeps getting picked and the lane
stalls on dead capacity. If a transient blip is misread as exhaustion, we
throw away paid capacity for 15 minutes. So: status code first, vendor error
body second, string matching only as the last resort.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Outcome(str, Enum):
    QUOTA_EXHAUSTED = "quota_exhausted"   # plan is spent -> long cooldown
    RATE_LIMITED = "rate_limited"         # too fast -> short cooldown
    AUTH = "auth"                         # broken credential -> long cooldown, alert
    CONTEXT = "context"                   # our fault, do not cool the plan
    TRANSIENT = "transient"               # retry elsewhere, brief cooldown
    BAD_REQUEST = "bad_request"           # our fault, do not cool the plan


#  Phrases that mean "this subscription is out of capacity" rather than
#  "you are sending requests too quickly". Checked only when the status code
#  alone cannot tell them apart (both arrive as 429).
_EXHAUSTED = re.compile(
    r"(usage limit|quota exceeded|quota exhausted|insufficient (balance|quota|credit)"
    r"|out of credit|credit limit|monthly limit|weekly limit|plan limit"
    r"|limit reached|no remaining|exceeded your current quota"
    r"|resource_exhausted|arrearage)",
    re.I,
)
_CONTEXT = re.compile(r"(context (length|window)|too many tokens|maximum context|prompt is too long)", re.I)
_AUTH = re.compile(r"(invalid api key|unauthorized|authentication|invalid token|expired token)", re.I)


@dataclass
class Verdict:
    outcome: Outcome
    cooldown_seconds: int
    detail: str = ""

    @property
    def should_cool(self) -> bool:
        return self.cooldown_seconds > 0


def classify(
    status: int | None,
    message: str,
    *,
    retry_after: float | None = None,
    default_cooldown: int = 900,
) -> Verdict:
    msg = message or ""

    if _CONTEXT.search(msg):
        return Verdict(Outcome.CONTEXT, 0, "context window exceeded")

    if status in (401, 403) or _AUTH.search(msg):
        # A dead credential will never fix itself; stop burning requests on it
        # but re-admit eventually in case a sidecar refreshed in the meantime.
        return Verdict(Outcome.AUTH, 1800, "credential rejected")

    if status == 402 or (status == 400 and _EXHAUSTED.search(msg)):
        # Some vendors bill-out with 402/400 rather than 429.
        return Verdict(Outcome.QUOTA_EXHAUSTED, default_cooldown, "payment/quota rejection")

    if status == 429:
        if _EXHAUSTED.search(msg):
            # Honour Retry-After if it looks like a real reset, else sit out.
            secs = int(retry_after) if retry_after and retry_after > 60 else default_cooldown
            return Verdict(Outcome.QUOTA_EXHAUSTED, secs, "plan quota exhausted")
        secs = int(retry_after) if retry_after else 30
        return Verdict(Outcome.RATE_LIMITED, max(5, min(secs, 300)), "rate limited")

    if status in (408, 499) or status is None:
        return Verdict(Outcome.TRANSIENT, 15, "timeout")

    if status and 500 <= status < 600:
        return Verdict(Outcome.TRANSIENT, 60, f"upstream {status}")

    if status and 400 <= status < 500:
        return Verdict(Outcome.BAD_REQUEST, 0, f"client error {status}")

    return Verdict(Outcome.TRANSIENT, 30, "unclassified")
