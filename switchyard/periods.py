"""Quota period arithmetic — when does the current allowance expire?

The subtlety that drives pacing: a quota **resets** every period, so unused
allowance in an intermediate period is lost forever. A cancelled plan therefore
has several full periods left plus one final period truncated by the expiry
date. Pacing is per-period burndown, never an attempt to spread the total
remaining allowance across all of them:

    |---- full month ----|---- full month ----|-- final, 9 days --|
    ^ use it all         ^ use it all         ^ use it all, faster   X expiry

Each window's deadline is therefore `min(next_rollover, expiry)`, and the final
truncated window has a higher target rate precisely because there is less time
to spend the same allowance.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

# 30-day month. Used as the canonical denominator for projecting a window's
# current rate onto a monthly figure: a window_per_month of 30 means the
# pattern is one full window per month, so multiplying a window's allowance
# or rate by 30 gives its monthly equivalent. This is a constant because the
# board / picker math must agree on it, not a knob operators tune.
DAYS_PER_MONTH = 30


def windows_per_month(period: str | None) -> float:
    """How many windows of `period` fit in one 30-day month.

    Used by the projection logic in `window_headroom` to scale a window's
    current allowance / rate up to a monthly capacity: e.g. a 5-hour rolling
    window runs 144 times per 30-day month (30 days * 24 hours / 5 hours
    per window = 144 windows), so the monthly capacity implied by a 5-hour
    window is 144x the per-window figure.

    Mapping, all keyed off the 30-day-month constant:

      - None (no period / open-ended): 1.0 — there is no recurring window, so
        treat the current allowance as already monthly.
      - "month": 1.0 — the window IS the month.
      - "week": 30/7 ≈ 4.2857 — a weekly window runs ~4.29 times per
        30-day month (52.18 weeks per year / 12 months per year × 30/30 = 4.29).
      - "rolling_5h": 30 * 24 / 5 = 144.0 — a rolling-5h bucket completes
        exactly 4.8 cycles per day, 144 per month.
      - "day": 30.0 — one window per day, 30 days per month.

    Anything else falls back to 1.0: a window with an unrecognised period
    can't be projected, so we don't pretend to know it; the projection
    then degrades gracefully (cap stays at A_true × 1 = A_true).
    """
    if period in (None, "month"):
        return 1.0
    if period == "week":
        return DAYS_PER_MONTH / 7
    if period == "rolling_5h":
        return DAYS_PER_MONTH * 24 / 5
    if period == "day":
        return float(DAYS_PER_MONTH)
    return 1.0


ROLL_5H = 5 * 3600


def _utc(at: datetime | None = None) -> datetime:
    return at or datetime.now(timezone.utc)


def period_bounds(period: str | None, at: datetime | None = None) -> tuple[datetime, datetime]:
    """(start, end) of the quota window containing `at`."""
    now = _utc(at)
    if period == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = (start + timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, end
    if period == "week":
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = midnight - timedelta(days=now.weekday())
        return start, start + timedelta(days=7)
    if period == "rolling_5h":
        epoch_bucket = int(now.timestamp()) // ROLL_5H
        start = datetime.fromtimestamp(epoch_bucket * ROLL_5H, tz=timezone.utc)
        return start, start + timedelta(seconds=ROLL_5H)
    if period == "day":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1)
    # No period: treat the window as open-ended (one long window).
    return now - timedelta(days=1), now + timedelta(days=365)


def expiry_moment(expires: date | None) -> datetime | None:
    """A plan bought through its expiry date is usable to the end of that day."""
    if expires is None:
        return None
    return datetime(expires.year, expires.month, expires.day,
                    tzinfo=timezone.utc) + timedelta(days=1)


def deadline(period: str | None, expires: date | None,
             at: datetime | None = None) -> tuple[datetime, bool]:
    """(deadline, is_final_window) for the window containing `at`.

    The deadline is the earlier of the next rollover and the plan's expiry —
    which is what makes the last window of a cancelled plan pace harder.
    """
    now = _utc(at)
    _, rollover = period_bounds(period, now)
    end = expiry_moment(expires)
    if end is not None and end <= rollover:
        return max(end, now), True
    return rollover, False


def windows_remaining(period: str | None, expires: date | None,
                      at: datetime | None = None) -> tuple[int, float]:
    """(full windows after this one, seconds in the final window).

    Display only, so you can see "2 more full months, then a 9-day window".
    """
    now = _utc(at)
    end = expiry_moment(expires)
    if end is None:
        return -1, 0.0  # no expiry: windows go on forever
    full = 0
    _, rollover = period_bounds(period, now)
    cursor = rollover
    while cursor < end:
        nxt = period_bounds(period, cursor)[1]
        if nxt >= end:
            return full, max(0.0, (end - cursor).total_seconds())
        full += 1
        cursor = nxt
    return full, 0.0
