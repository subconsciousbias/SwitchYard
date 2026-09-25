"""Promotions attached to a plan: bonus credits, one-off resets, anything a
provider grants for a while and then takes back.

Providers hand these out with an expiry (a Claude seat's "$250 bonus credit
for cloud sessions", good until a date), and they are only worth anything if
someone notices before that date. None of them show up in the usage reports
the probes read, so SwitchYard cannot discover them on its own; something
outside it (a script, a person, a browser extension) posts what it sees to
`POST /admin/plans/{plan}/promotions` and the board keeps the countdown.

Routing never reads this. A promotion that applies to a surface SwitchYard
does not route (cloud sessions, say) would be a lie to act on, and one that
does apply is already visible through the plan's own quota windows. This is
bookkeeping, on purpose: the board says "use it or lose it", the operator
decides.

Storage is one Redis hash per plan, `sy:promo:{plan}`: `items` holds the
JSON list, replaced as a whole on every post (a set, not a patch, so a
promotion that disappears upstream disappears here too), plus `source` and
`updated_at`.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

K_PROMO = "sy:promo:{plan}"

KINDS = ("credit", "reset", "other")
# Posted lists are small by nature (a seat has one or two); the cap only keeps
# a buggy feeder from writing an unbounded blob into Redis.
MAX_ITEMS = 20


@dataclass(frozen=True)
class Promotion:
    id: str                         # stable per promotion, chosen by the feeder
    kind: str = "credit"            # credit | reset | other
    label: str = ""                 # what the board prints
    applies_to: str = ""            # free text: "cloud sessions", "all usage"
    total: float | None = None      # amount granted, in `currency`
    remaining: float | None = None  # amount left, in `currency`
    currency: str = ""
    expires_at: float | None = None  # unix seconds, UTC
    observed_at: float | None = None  # when the feeder last saw it

    def view(self, now: float, alert_days: float) -> dict:
        """The board's row: the stored fields plus the countdown."""
        days_left = (None if self.expires_at is None
                     else (self.expires_at - now) / 86400.0)
        expired = days_left is not None and days_left <= 0
        used_up = self.remaining is not None and self.remaining <= 0
        return {
            **asdict(self),
            "days_left": None if days_left is None else round(days_left, 1),
            "expired": expired,
            "used_up": used_up,
            # Worth flagging: live, something left, and the date is close.
            "expiring_soon": (days_left is not None and not expired
                              and not used_up and days_left <= alert_days),
        }


def _timestamp(value: Any, field: str) -> float | None:
    """Unix seconds from a number (seconds or milliseconds) or ISO-8601."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a timestamp, got a boolean")
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 1000.0 if v > 1e12 else v
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{field} is not a timestamp: {value!r}") from None
    if dt.tzinfo is None:
        # A naive time is ambiguous; UTC is the documented convention.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _amount(value: Any, field: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number, got a boolean")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number, got {value!r}") from None


def parse(raw: Any, now: float | None = None) -> Promotion:
    """One posted promotion -> Promotion, or ValueError naming the problem."""
    if not isinstance(raw, dict):
        raise ValueError("each promotion must be an object")
    pid = str(raw.get("id") or "").strip()
    if not pid:
        raise ValueError("each promotion needs an id")
    kind = str(raw.get("kind") or "credit").strip().lower()
    if kind not in KINDS:
        raise ValueError(f"promotion {pid}: kind must be one of {KINDS}, got {kind!r}")
    observed = _timestamp(raw.get("observed_at"), f"promotion {pid}: observed_at")
    return Promotion(
        id=pid[:80],
        kind=kind,
        label=str(raw.get("label") or "")[:120],
        applies_to=str(raw.get("applies_to") or "")[:80],
        total=_amount(raw.get("total"), f"promotion {pid}: total"),
        remaining=_amount(raw.get("remaining"), f"promotion {pid}: remaining"),
        currency=str(raw.get("currency") or "")[:8],
        expires_at=_timestamp(raw.get("expires_at"), f"promotion {pid}: expires_at"),
        observed_at=observed if observed is not None else (now or time.time()),
    )


class Promotions:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def replace(self, plan_key: str, items: list[Any],
                      source: str = "") -> list[Promotion]:
        """Store `items` as the plan's whole promotion list (validated first,
        so a bad post leaves the previous list in place)."""
        if not isinstance(items, list):
            raise ValueError("promotions must be a list")
        if len(items) > MAX_ITEMS:
            raise ValueError(f"at most {MAX_ITEMS} promotions per plan")
        now = time.time()
        parsed = [parse(i, now) for i in items]
        ids = [p.id for p in parsed]
        if len(set(ids)) != len(ids):
            raise ValueError("promotion ids must be unique within a plan")
        await self.redis.hset(K_PROMO.format(plan=plan_key), mapping={
            "items": json.dumps([asdict(p) for p in parsed]),
            "source": str(source or "")[:120],
            "updated_at": now,
        })
        return parsed

    async def get(self, plan_key: str) -> tuple[list[Promotion], dict]:
        """(promotions, meta). An unreadable stored list reads as empty rather
        than failing the whole board."""
        raw = await self.redis.hgetall(K_PROMO.format(plan=plan_key)) or {}
        raw = {(k.decode() if isinstance(k, bytes) else k):
               (v.decode() if isinstance(v, bytes) else v) for k, v in raw.items()}
        try:
            items = [Promotion(**d) for d in json.loads(raw.get("items") or "[]")]
        except (TypeError, ValueError):
            items = []
        try:
            updated_at = float(raw.get("updated_at")) if raw.get("updated_at") else None
        except (TypeError, ValueError):
            updated_at = None
        return items, {"source": raw.get("source") or "", "updated_at": updated_at}

    async def view(self, plan_key: str, alert_days: float,
                   now: float | None = None) -> dict:
        """What the board and /api/state show for one plan."""
        now = now or time.time()
        items, meta = await self.get(plan_key)
        rows = sorted((p.view(now, alert_days) for p in items),
                      key=lambda r: (r["expires_at"] is None, r["expires_at"] or 0))
        return {"items": rows, **meta}
