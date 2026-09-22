"""Usage ledger and quota headroom.

Most of these plans expose no usage API, so headroom has to be inferred from
what we ourselves sent. Two tricks make that useful rather than decorative:

  * period buckets (hour / day / period) give a burn rate, which is what
    catches a $20/hour overflow long before the credit is gone;
  * when a plan does hard-fail on quota, we record how much we had consumed
    in the current period as an *observed allowance*. After one cycle the
    board can say "you hit the wall at ~118M tokens last time", so a plan
    with `allowance: null` still gets a real headroom bar.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from redis.asyncio import Redis

from .models import Plan, Quota
from .periods import period_bounds


def _now() -> datetime:
    return datetime.now(timezone.utc)


def period_key(period: str | None, at: datetime | None = None) -> str:
    at = at or _now()
    if period == "month":
        return at.strftime("%Y-%m")
    if period == "week":
        return f"{at.isocalendar().year}-W{at.isocalendar().week:02d}"
    if period == "rolling_5h":
        # Buckets of 5h from the epoch: close enough to attribute usage to the
        # window that is actually in force, without tracking each reset.
        return f"5h-{int(at.timestamp()) // 18000}"
    if period == "day":
        return at.strftime("%Y-%m-%d")
    return "all"


K_PERIOD = "sy:usage:{plan}:p:{period}"
K_HOUR = "sy:usage:{plan}:h:{hour}"
K_DAY = "sy:usage:{plan}:d:{day}"
# Model-scoped buckets mirror the plan-level ones so per-model economics can
# be queried without touching the plan-level burn rate, pacing or headroom.
K_M_HOUR = "sy:usage:{plan}:m:{model}:h:{hour}"
K_M_DAY = "sy:usage:{plan}:m:{model}:d:{day}"
# Per-model sessions-this-month: an HLL keyed by model ref so PFCOUNT returns
# the number of distinct sessions that model served this month. One session
# counted once regardless of how many requests it made -- a re-leased model
# on the same session is a free PFADD, the HLL stays at the same cardinality.
K_M_SMONTH = "sy:usage:{plan}:m:{model}:sp:{period}"
K_QUOTA = "sy:quota:{plan}"                 # plan-level facts
K_WINDOW = "sy:qwin:{plan}:{window}"        # per-window facts (observed allowance)
# Per-lane ordering written by the portal's probe poller and read on the pick
# path. Kept as the canonical key for the legacy `strategy: perishable` sugar
# — when a lane has no explicit groups and `strategy: perishable`, the picker
# boundary wraps the order in an implicit `{perishable: order}` group and reads
# THIS key (not a group-scoped one), so a flat config is bit-for-bit unchanged.
K_LANE_ORDER = "sy:lane-order:{lane}"
# Per-group ordering written by the probe poller and read on the pick path for
# groups that need a score (perishable and lowest_utilization). Keyed by the
# group's gid AND the lane that contains it, so the same group config appearing
# in two lanes does not collide. The legacy lane-level key stays in place for
# backward compatibility — the picker reads the per-group key only when an
# explicit group of these strategies is in play.
K_GROUP_ORDER = "sy:group-order:{gid}:{lane}"
# Per-group rotation pointer. Incremented atomically each time the group picks
# a real placement; round_robin / weighted use it to choose the start index,
# skipping paced-to-0 and unavailable members without burning the counter on a
# spill that returned empty. Stored under the gid so two groups in the same
# lane can rotate independently.
K_GROUP_ROT = "sy:group-rot:{gid}:{lane}"

FIELDS = ("requests", "prompt_tokens", "completion_tokens", "cost", "failures")


class Ledger:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def record(
        self,
        plan: Plan,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        failed: bool = False,
        model: str | None = None,
        session: str | None = None,
    ) -> None:
        now = _now()
        # One bucket per quota window: a plan with a 5-hour *and* a weekly
        # allowance needs both counted, or pacing can only see one of them.
        keys = [
            (K_PERIOD.format(plan=plan.key, period=period_key(q.period, now)), 90 * 86400)
            for q in plan.quotas
        ]
        keys += [
            (K_HOUR.format(plan=plan.key, hour=now.strftime("%Y-%m-%dT%H")), 7 * 86400),
            (K_DAY.format(plan=plan.key, day=now.strftime("%Y-%m-%d")), 400 * 86400),
        ]
        # When a model ref is given, also write the model-scoped buckets so
        # per-model economics can be computed without touching plan-level ones.
        if model:
            keys += [
                (K_M_HOUR.format(plan=plan.key, model=model,
                                 hour=now.strftime("%Y-%m-%dT%H")), 7 * 86400),
                (K_M_DAY.format(plan=plan.key, model=model,
                                day=now.strftime("%Y-%m-%d")), 400 * 86400),
            ]
        pipe = self.redis.pipeline()
        for key, ttl in keys:
            pipe.hincrbyfloat(key, "requests", 1)
            pipe.hincrbyfloat(key, "prompt_tokens", prompt_tokens)
            pipe.hincrbyfloat(key, "completion_tokens", completion_tokens)
            pipe.hincrbyfloat(key, "cost", cost)
            if failed:
                pipe.hincrbyfloat(key, "failures", 1)
            pipe.expire(key, ttl)
        # Sessions-this-month HLL: counted on success only (failure paths pass
        # no session), one PFADD per session -- the HLL dedupes naturally, so
        # a session that re-leases the same model across many requests still
        # shows up as exactly one session in the month's PFCOUNT.
        if model and session and not failed:
            s_month = K_M_SMONTH.format(plan=plan.key, model=model,
                                        period=now.strftime("%Y-%m"))
            pipe.pfadd(s_month, session)
            pipe.expire(s_month, 400 * 86400)
        await pipe.execute()

    @staticmethod
    def _decode_bucket(raw) -> dict[str, float]:
        """One hash reply -> a FIELDS-shaped dict. Shared by every reader so a
        real Redis (bytes keys and values) decodes exactly like the fake one —
        a decode done in one reader but not another reads as silent zeros."""
        out = {f: 0.0 for f in FIELDS}
        for k, v in (raw or {}).items():
            k = k.decode() if isinstance(k, bytes) else k
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
        return out

    async def bucket(self, plan_key: str, key: str) -> dict[str, float]:
        return self._decode_bucket(await self.redis.hgetall(key))

    async def window_usage(self, plan: Plan, quota: Quota,
                           at: datetime | None = None) -> dict[str, float]:
        """Consumption inside one specific quota window.

        `at` exists so the pacer can be driven from an injected clock in tests;
        in production it is always now.
        """
        return await self.bucket(
            plan.key, K_PERIOD.format(plan=plan.key, period=period_key(quota.period, at))
        )

    async def burn_rate(self, plan: Plan, hours: int = 3) -> dict[str, float]:
        """Cost and tokens per hour over the last `hours` completed buckets."""
        now = _now()
        cost = tokens = 0.0
        for i in range(hours):
            at = now - timedelta(hours=i)
            b = await self.bucket(plan.key, K_HOUR.format(plan=plan.key, hour=at.strftime("%Y-%m-%dT%H")))
            cost += b["cost"]
            tokens += b["prompt_tokens"] + b["completion_tokens"]
        return {"cost_per_hour": cost / hours, "tokens_per_hour": tokens / hours}

    async def model_overview(
        self, plan_key: str, model_refs: list[str], month: str,
        hours: int = 3, days: int = 31,
    ) -> dict[str, dict]:
        """Per-model economics: burn rate over the last `hours` hours and month totals.

        One pipelined round trip no matter how many refs. The month filter is
        applied when choosing day keys, not after reading: early in a month the
        31-day walk reaches back into the previous one, and "This month" that
        includes August on September 3rd is a lie.
        """
        now = _now()
        # Ordered read plan: results come back in queue order, so aggregation
        # is a zip against this list, never positional arithmetic on slices.
        reads: list[tuple[str, str, str]] = []
        for ref in model_refs:
            for i in range(hours):
                at = now - timedelta(hours=i)
                reads.append((ref, "hour", K_M_HOUR.format(
                    plan=plan_key, model=ref, hour=at.strftime("%Y-%m-%dT%H"))))
            for d in range(days):
                day = (now - timedelta(days=d)).strftime("%Y-%m-%d")
                if not day.startswith(month):
                    continue
                reads.append((ref, "day", K_M_DAY.format(
                    plan=plan_key, model=ref, day=day)))

        pipe = self.redis.pipeline()
        for _, _, key in reads:
            pipe.hgetall(key)
        # Trailing HLL reads: the per-model PFCOUNTs and one plan-level
        # union PFCOUNT. The queue order is captured in `trailing` so
        # decoding is a single zip against the slice of results after the
        # hash reads -- never an arithmetic offset that a future op
        # appended after these would silently mis-attribute.
        trailing: list[tuple[str, str | None]] = []
        for ref in model_refs:
            trailing.append(("model", ref))
            pipe.pfcount(K_M_SMONTH.format(
                plan=plan_key, model=ref, period=month))
        # Plan-level PFCOUNT: the union of every per-model HLL, returned by
        # Redis as a single integer. Sums of per-model cardinalities would
        # double-count any session that touched more than one model of this
        # plan (e.g. mid-loop spillover between two configured models), and
        # the subscription rate `monthly_cost / plan_sessions` is sensitive to
        # that denominator -- an over-counted plan_sessions understates the
        # board's $/session for every model of a subscription plan.
        trailing.append(("plan", None))
        pipe.pfcount(*[K_M_SMONTH.format(
            plan=plan_key, model=ref, period=month) for ref in model_refs])
        raw = await pipe.execute()

        out = {ref: {"burn": {"cost_per_hour": 0.0, "tokens_per_hour": 0.0},
                     "month_tokens": 0.0, "month_cost": 0.0,
                     "n_sessions": 0, "plan_n_sessions": 0}
               for ref in model_refs}
        for (ref, kind, _), r in zip(reads, raw):
            b = self._decode_bucket(r)
            row = out[ref]
            if kind == "hour":
                row["burn"]["cost_per_hour"] += b["cost"]
                row["burn"]["tokens_per_hour"] += b["prompt_tokens"] + b["completion_tokens"]
            else:
                row["month_cost"] += b["cost"]
                row["month_tokens"] += b["prompt_tokens"] + b["completion_tokens"]
        plan_n_sessions = 0
        for (kind, ref_or_none), count in zip(trailing, raw[len(reads):]):
            count = int(count or 0)
            if kind == "model":
                out[ref_or_none]["n_sessions"] = count
            else:  # "plan"
                plan_n_sessions = count
        # Surface plan_n_sessions on every row -- same value everywhere, but
        # keeps the per-model dict self-contained so callers don't need a
        # special-case lookup path.
        for row in out.values():
            row["plan_n_sessions"] = plan_n_sessions
            row["burn"]["cost_per_hour"] /= hours
            row["burn"]["tokens_per_hour"] /= hours
        return out

    async def daily_series(self, plan_key: str, days: int = 30) -> list[dict]:
        now = _now()
        out = []
        for i in range(days - 1, -1, -1):
            day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            b = await self.bucket(plan_key, K_DAY.format(plan=plan_key, day=day))
            out.append({"day": day, **b})
        return out

    # -- learning where the wall is ----------------------------------------
    async def note_exhaustion(self, plan: Plan, reset_at: float | None) -> Quota:
        """Record where a plan actually ran out, against the right window.

        A provider says "you are out of quota" without saying *which* limit you
        hit. The reset time gives it away: a couple of hours means the 5-hour
        burst window, several days means the weekly allowance. Attributing this
        correctly matters — writing a 5-hour figure into the weekly window's
        observed allowance would corrupt every pacing decision after it.
        """
        window = self.attribute_window(plan, reset_at)
        used = await self.window_usage(plan, window)
        consumed_tokens = used["prompt_tokens"] + used["completion_tokens"]
        now = time.time()
        pipe = self.redis.pipeline()
        pipe.hset(K_WINDOW.format(plan=plan.key, window=window.label), mapping={
            "last_exhausted_at": now,
            "observed_allowance_tokens": consumed_tokens,
            "observed_allowance_cost": used["cost"],
            "reset_at": reset_at or "",
        })
        pipe.hset(K_QUOTA.format(plan=plan.key), mapping={
            "last_exhausted_at": now,
            "last_exhausted_window": window.label,
            "reset_at": reset_at or "",
        })
        await pipe.execute()
        return window

    @staticmethod
    def attribute_window(plan: Plan, reset_at: float | None,
                         at: datetime | None = None) -> Quota:
        """Which window did we just hit? Pick the one whose own rollover is
        closest to the reset time the provider gave us."""
        if len(plan.quotas) == 1 or not reset_at:
            # No reset hint: blame the shortest window, which is the one you hit
            # far more often, rather than poisoning the weekly figure.
            order = {"rolling_5h": 0, "day": 1, "week": 2, "month": 3, None: 4}
            return min(plan.quotas, key=lambda q: order.get(q.period, 4))
        now = at or _now()
        target_delta = reset_at - now.timestamp()
        best, best_gap = plan.quotas[0], None
        for q in plan.quotas:
            _, end = period_bounds(q.period, now)
            gap = abs((end - now).total_seconds() - target_delta)
            if best_gap is None or gap < best_gap:
                best, best_gap = q, gap
        return best

    async def note_concurrency_rejection(self, plan: Plan) -> None:
        """A provider refusing us on connection count means our cap is wrong.

        Tracked separately from rate limiting because the fix is different:
        lower `max_parallel` in plans.yaml rather than wait it out.
        """
        key = K_QUOTA.format(plan=plan.key)
        await self.redis.hincrbyfloat(key, "concurrency_rejections", 1)
        await self.redis.hset(key, mapping={
            "concurrency_rejected_at": time.time(),
            "concurrency_rejected_at_cap": plan.max_parallel,
        })

    async def note_reported_percent(self, plan_key: str, used_percent: float | None,
                                    reset_at: float | None,
                                    window: str | None = None) -> None:
        """Record a percentage a provider stated, when it publishes no counts.

        MiniMax reports `current_weekly_used_percent` with every count set to -1,
        so there is nothing to reconcile against our own tally — the percentage
        IS the measurement. Kept in its own field so window_headroom can prefer
        it without ever mixing percent into a token or dollar total.
        """
        mapping = {}
        if used_percent is not None:
            mapping["reported_pct_used"] = max(0.0, min(100.0, float(used_percent)))
        if reset_at is not None:
            mapping["reset_at"] = reset_at
        if not mapping:
            return
        mapping["reported_at"] = time.time()
        key = (K_WINDOW.format(plan=plan_key, window=window) if window
               else K_QUOTA.format(plan=plan_key))
        await self.redis.hset(key, mapping=mapping)

    async def note_reported(self, plan_key: str, remaining: float | None,
                            reset_at: float | None, window: str | None = None,
                            limit: float | None = None) -> None:
        """Record limits a provider actually told us about (headers/sidecar).

        `limit` matters when the provider states both sides: xAI answers with
        x-ratelimit-limit-tokens next to the remaining count, and deriving the
        total from our own tally instead would understate it by everything the
        account spent outside SwitchYard.
        """
        mapping = {}
        if remaining is not None:
            mapping["reported_remaining"] = remaining
        if limit is not None:
            mapping["reported_limit"] = limit
        if reset_at is not None:
            mapping["reset_at"] = reset_at
        if not mapping:
            return
        mapping["reported_at"] = time.time()
        key = (K_WINDOW.format(plan=plan_key, window=window) if window
               else K_QUOTA.format(plan=plan_key))
        await self.redis.hset(key, mapping=mapping)

    async def window_facts(self, plan_key: str, window: str) -> dict[str, float | str]:
        return await self._facts(K_WINDOW.format(plan=plan_key, window=window))

    async def quota_facts(self, plan_key: str) -> dict[str, float | str]:
        return await self._facts(K_QUOTA.format(plan=plan_key))

    async def _facts(self, key: str) -> dict[str, float | str]:
        raw = await self.redis.hgetall(key)
        out: dict[str, float | str] = {}
        for k, v in (raw or {}).items():
            k = k.decode() if isinstance(k, bytes) else k
            v = v.decode() if isinstance(v, bytes) else v
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        return out

    # -- perishable ordering -----------------------------------------------
    async def set_lane_order(self, lane: str, entries: dict[str, float | str],
                             *, computed_at: float, stale_after_ms: int) -> None:
        """Write the per-lane perishable ordering the picker reads.

        `entries` is the *output* of the writer: each ref's score and gate flag
        plus whatever bookkeeping fields the writer chose to add. The picker
        only reads `score_<ref>`, `gate5h_<ref>`, `computed_at`, and
        `stale_after_ms`, so anything else is documentation for the board /
        debugging. Stale entries (older than `stale_after_ms`) are dropped on
        write, not just on read, so the picker can fall back to config order
        with one hash read instead of having to also timestamp-check the call.

        Not pipelined: the writer runs once per probe interval per affected
        lane (a handful of keys per minute at most), and a delete-then-write
        pair is what guarantees we never carry over an old `score_<ref>` for
        a ref that left the lane.
        """
        key = K_LANE_ORDER.format(lane=lane)
        mapping = _order_mapping(entries, computed_at, stale_after_ms)
        await self.redis.delete(key)
        await self.redis.hset(key, mapping=mapping)

    async def get_lane_order(self, lane: str,
                            plan_windows: list[tuple[str, str]] | None = None
                            ) -> dict | None:
        """The stored per-lane order, or None if the lane never produced one.

        Picker-facing shape: members is a list of refs in the order the writer
        ranked them, each entry carries `score` and `gate5h`. The freshness
        fields are read and checked here so the picker does one hash read and
        falls back to config order on a missing/stale key without further work.

        `plan_windows` is the stale-grace opt-in: when the hash would otherwise
        be stale, the reader extends freshness if any of the listed plans is
        in `needs_reauth` AND its window's `reset_at` is still in the future.
        See `_read_order` and `_in_grace_window` for the rationale.
        """
        return await self._read_order(
            K_LANE_ORDER.format(lane=lane), plan_windows=plan_windows)

    async def set_group_order(self, gid: str, lane: str,
                              entries: dict[str, float | str], *,
                              computed_at: float, stale_after_ms: int) -> None:
        """Per-group score hash. Same shape as `set_lane_order`; the key
        carries `gid` so two groups with the same member set still get
        independent rankings."""
        key = K_GROUP_ORDER.format(gid=gid, lane=lane)
        mapping = _order_mapping(entries, computed_at, stale_after_ms)
        await self.redis.delete(key)
        await self.redis.hset(key, mapping=mapping)

    async def get_group_order(self, gid: str, lane: str,
                              plan_windows: list[tuple[str, str]] | None = None
                              ) -> dict | None:
        """Per-group rank read. Same return shape as `get_lane_order`.

        Same stale-grace opt-in as `get_lane_order`: `plan_windows` lets the
        reader keep the hash fresh while a cookie-expired plan's window is
        in force, so the picker reads the last good ranking instead of
        dropping to config order.
        """
        return await self._read_order(
            K_GROUP_ORDER.format(gid=gid, lane=lane), plan_windows=plan_windows)

    async def bump_group_rot(self, gid: str, lane: str) -> int:
        """Increment and return the group's rotation pointer.

        Called once per group `_visit` that returns a real placement; a group
        that spills past every member leaves the counter alone so the next
        attempt starts at the same member. The key has a generous TTL so an
        idle group decays its counter (a fresh attempt on a quiet config
        starts at member 0, not "wherever the pointer was a week ago").
        """
        key = K_GROUP_ROT.format(gid=gid, lane=lane)
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 86400)         # a day: a quiet group's pointer decays
        results = await pipe.execute()
        return int(results[0])

    async def group_rot(self, gid: str, lane: str) -> int:
        """Current rotation pointer without advancing it."""
        raw = await self.redis.get(K_GROUP_ROT.format(gid=gid, lane=lane))
        if raw is None:
            return 0
        raw = raw.decode() if isinstance(raw, bytes) else raw
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    # -- stale-grace for cookie-expired plans ------------------------------
    # A plan with `sy:probe:{plan}.needs_reauth=1` has stopped polling
    # (its cookie is dead and the portal is asking the operator to paste
    # a fresh one). Its lane-order / group-order hash is therefore never
    # refreshed: the writer runs only on a successful probe. Without a
    # grace window, the hash ages past `stale_after_ms` (~20 min, 2x the
    # probe interval) and the picker falls back to config order, even
    # though the LAST GOOD ranking is still meaningful: every window the
    # hash was computed from is still in force.
    #
    # The grace is capped at the window boundary. The reader checks each
    # needs_reauth plan in the lane: if its window's `reset_at` is still
    # in the future, the previously stored ranking stays current for
    # scheduling. Once `reset_at` passes the window has rolled over and
    # the reading no longer describes any in-force window, so the hash
    # is treated as stale exactly as it was before this work -- the
    # fallback to config order returns. The grace cannot carry a reading
    # across a window reset into the next window; the test at
    # `test_reported_reading_expires_when_reset_passes` is the existing
    # regression bar for that invariant.
    K_PROBE = "sy:probe:{plan}"

    async def _plan_needs_reauth(self, plan_key: str) -> bool:
        """Whether the probe hash for `plan_key` carries needs_reauth=1.

        Mirrors `Prober.status`'s decode: the field is stored as a string
        "1" / "1.0" / "True" (Redis hash values are bytes-or-strings).
        """
        raw = await self.redis.hget(self.K_PROBE.format(plan=plan_key),
                                    "needs_reauth")
        if raw is None:
            return False
        if isinstance(raw, bytes):
            raw = raw.decode()
        return raw in ("1", "1.0", "True")

    async def _in_grace_window(
        self, plan_windows: list[tuple[str, str]]
    ) -> bool:
        """True iff at least one (plan, window) in the list is a needs_reauth
        plan whose window's `reset_at` is still in the future.

        Used by `_read_order` to extend freshness for the lane-order and
        group-order hashes while a cookie-expired plan's window is in
        force. Capped at the window boundary: once `reset_at` passes the
        grace ends, exactly as `reported_is_current`'s rule (a) does for
        per-window facts.
        """
        now = time.time()
        for plan_key, window_label in plan_windows:
            if not await self._plan_needs_reauth(plan_key):
                continue
            facts = await self.window_facts(plan_key, window_label)
            reset_at = facts.get("reset_at")
            if isinstance(reset_at, (int, float)) and float(reset_at) > now:
                return True
        return False

    async def _read_order(self, key: str,
                         plan_windows: list[tuple[str, str]] | None = None) -> dict | None:
        raw = await self.redis.hgetall(key)
        if not raw:
            return None
        decoded: dict[str, str] = {}
        for k, v in raw.items():
            decoded[k.decode() if isinstance(k, bytes) else k] = (
                v.decode() if isinstance(v, bytes) else v)
        try:
            computed_at = float(decoded["computed_at"])
            stale_after_ms = int(float(decoded["stale_after_ms"]))
        except (KeyError, ValueError, TypeError):
            return None
        age_ms = (time.time() - computed_at) * 1000.0
        if age_ms >= stale_after_ms:
            # Stale-grace: a plan whose session cookie has expired
            # (`sy:probe:{plan}.needs_reauth=1`) stops polling, so its
            # ranking hash never gets refreshed. While the window the
            # last good reading describes is still in force, treat the
            # hash as current -- capping exactly at the window boundary
            # (`reset_at`), never across it. The picker keeps using the
            # last good ranking instead of falling back to config order,
            # which is the difference between scheduling on the plan's
            # real numbers and on declared order with a dead cookie.
            if plan_windows and await self._in_grace_window(plan_windows):
                pass
            else:
                return None
        members: list[dict] = []
        for key_name, value in decoded.items():
            if not key_name.startswith("score_"):
                continue
            ref = key_name[len("score_"):]
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = 0.0
            gate_raw = decoded.get(f"gate5h_{ref}", "0")
            try:
                gate = int(float(gate_raw)) == 1
            except (TypeError, ValueError):
                gate = False
            members.append({"ref": ref, "score": score, "gate5h": gate})
        if not members:
            return None
        members.sort(key=lambda m: m["score"], reverse=True)
        return {"members": members, "computed_at": computed_at,
                "stale_after_ms": stale_after_ms}


def _order_mapping(entries: dict[str, float | str], computed_at: float,
                   stale_after_ms: int) -> dict[str, str]:
    mapping: dict[str, str] = {"computed_at": str(computed_at),
                               "stale_after_ms": str(stale_after_ms)}
    for ref, fields in entries.items():
        for k, v in fields.items():
            mapping[f"{k}_{ref}"] = str(v)
    return mapping


def reported_is_current(facts: dict, period: str | None,
                        now: float | None = None) -> bool:
    """Whether a `reported_*` reading still describes the window that's in force.

    A reading is current iff:

      (a) `reset_at`, when present as a float, is still in the future
          (`now < reset_at`). The provider's own reset hint is the strongest
          signal we have that the percentage is still meaningful, and it MUST be
          absent or a future float. `note_exhaustion` writes `""` into the hash
          when there is no hint -- a non-float sentinel that has to read as
          stale so a quota_exhausted cooldown does not leak into a fresh
          window.
      (b) `reported_at` is a float, and is within the current period bucket
          via `period_bounds(period)`. The bucket is the fallback when there
          is no `reset_at`: a weekly reading older than this week is not a
          reading about this week, however fresh the number looks. The pacer and
          `note_exhaustion` already use this same rollover as the source of
          truth for "which window is in force".

    `now` is injectable for tests, the same pattern as `perishable_score`;
    absent means `time.time()`.

    Anything undatable -- missing/non-float `reported_at`, `reset_at == ""`,
    or a `reset_at` already in the past -- returns False. Stale means
    UNKNOWN, not spent: callers fall back to the ledger/observed-allowance
    basis (the board shows the estimate), and the picker re-admits the plan
    so over-admission self-heals via the vendor's next 429 -> cooldown path
    rather than getting stuck in the permanent over-exclusion state the
    stale read was producing.
    """
    if now is None:
        now = time.time()
    reset_at = facts.get("reset_at")
    if reset_at is not None:
        # Present and non-empty: it must be a float that hasn't passed yet.
        # An empty string (the `note_exhaustion` sentinel) is not a float,
        # so the first clause is False -- but the empty-string check above
        # already covers the documented case. Other non-float values fall
        # through the same way: stale.
        if not isinstance(reset_at, float) or now >= reset_at:
            return False
    reported_at = facts.get("reported_at")
    if not isinstance(reported_at, float):
        return False
    start, _ = period_bounds(period, datetime.fromtimestamp(now, tz=timezone.utc))
    if reported_at < start.timestamp():
        return False
    return True


def perishable_score(room_pct: float | None, reset_at: float | None,
                     now: float | None = None) -> float | None:
    """Perishable score: how much room each hour of remaining window buys us.

    Higher = drain it faster. The choice (room% / hours_remaining) is the burn
    rate that uses the rest of the window most aggressively without overshooting
    the reset: a plan with 80% room and a reset in 4h scores 80/4 = 20, while a
    plan with 20% room and the same reset scores 20/4 = 5. With hours clamped
    to a minimum of 1 to avoid divide-by-near-zero on a window that has just
    rolled over (where reset_at is essentially "now"), the absolute number
    loses meaning -- it is only used for relative ordering.

    Returns None when either fact is missing rather than guessing, so a member
    with no facts is silently left to fall through the picker's "unknown sorts
    last" rule instead of getting an arbitrary rank.
    """
    if room_pct is None or reset_at is None:
        return None
    if now is None:
        now = time.time()
    hours = max(1.0, (float(reset_at) - float(now)) / 3600.0)
    return float(room_pct) / hours


def utilization_score(room_pct: float | None, reset_at: float | None = None,
                      now: float | None = None) -> float | None:
    """Lowest-utilization score: how much room there is, period.

    The `lowest_utilization` strategy is like `perishable` but the divisor is
    dropped — a subscription's "5h constraint full" plan is more saturated
    than its "weekly half used" plan, but a `lowest_utilization` group should
    rank the saturated one FIRST so it does not get hammered into the wall.
    That is the opposite of perishable: drain the emptiest, leave the
    fullest alone, keep every member well inside its allowance.

    Higher = more saturated (drain sooner). With reset_at=None the divisor
    drops entirely, so the score is just `room_pct` and is stable across
    members regardless of when their respective windows roll over. Returns
    None when room is missing — same contract as perishable_score, so the
    picker can reuse its "unknown sorts last" path.
    """
    if room_pct is None:
        return None
    if reset_at is None:
        return float(room_pct)
    if now is None:
        now = time.time()
    hours = max(1.0, (float(reset_at) - float(now)) / 3600.0)
    return float(room_pct) / hours


def family_partitioned_order(refs: list[str],
                             config_order: list[str],
                             family_of) -> list[str]:
    """Reorder a score-ranked list so the score never competes across families.

    The input is what the writer / picker would have produced today: scored
    members first in score-descending order, then unscored/unknown members
    in declared (config) order. Within a single provider family that order
    is already what the operator wants, so the helper leaves it alone. The
    problem only appears when a lane mixes providers from different
    families: a high-room openai/astra with score 90 and a half-used
    claude-max/fable with score 30 used to interleave by raw score, and an
    openai plan briefly ahead of every claude plan in a `forge`-shaped
    lane. That re-rank is what was costing us a quiet avalanche into a
    foreign plan every time the perishable writer fired -- the figure at
    issue #53.

    The partition puts every ref behind a leading-family bucket whose
    member order is the input order verbatim; the leading family is the
    first family to appear in `config_order` (the lane body, or the
    group's leaf refs in declaration order). Each subsequent family
    becomes its own bucket in the same first-appearance order. The output
    concatenates the buckets back-to-back. A single-family input returns
    an order identical to the input -- the regression bar.

    `family_of` is a `str | None` callable; a None family joins ONE shared
    "unspecified" bucket keyed (for bucket ordering) as the empty string.
    A non-empty string family uses the string as its bucket key directly.
    Note: this helper does NOT validate `family_of`'s return value -- if
    a caller returns the empty string `""` for a real family, that family
    silently joins the unspecified bucket. The shipped Plan schema does
    not currently enforce a non-empty family, so callers should treat
    `None` as "unspecified" and any string (including `""`) as "this
    family" accordingly.
    """
    if not refs:
        return refs
    refs_set = set(refs)
    bucket_order: list[str] = []
    for ref in config_order:
        if ref not in refs_set:
            continue
        family = family_of(ref)
        key = family if family is not None else ""
        if key not in bucket_order:
            bucket_order.append(key)
    bucket_of: dict[str, list[str]] = {key: [] for key in bucket_order}
    for ref in refs:
        family = family_of(ref)
        key = family if family is not None else ""
        bucket_of.setdefault(key, []).append(ref)
    out: list[str] = []
    for key in bucket_order:
        out.extend(bucket_of.get(key, []))
    return out


async def headroom(ledger: Ledger, plan: Plan) -> dict:
    """Headroom for every quota window, plus which one is closest to biting.

    The target window (weekly, usually) is what pacing fills; a constraint
    window (the 5-hour burst) can still be the one that stops you first, so the
    board needs both.
    """
    windows = [await window_headroom(ledger, plan, q) for q in plan.quotas]
    target = next((w for w in windows if w["role"] == "target"), windows[0])
    rated = [w for w in windows if w.get("pct_used") is not None]
    binding = max(rated, key=lambda w: w["pct_used"]) if rated else target
    return {**target, "windows": windows, "binding": binding,
            "binding_is_target": binding is target}


async def window_headroom(ledger: Ledger, plan: Plan, q: Quota) -> dict:
    """What the board shows in the 'quota left' column, for one window."""
    used = await ledger.window_usage(plan, q)
    facts = await ledger.window_facts(plan.key, q.label)
    tokens = used["prompt_tokens"] + used["completion_tokens"]

    basis: str | None = None
    limit: float | None = None
    consumed: float = 0.0

    meta = {"window": q.label, "role": q.role, "period": q.period}

    if q.kind == "unlimited":
        return {**meta, "kind": "unlimited", "pct_used": None, "basis": "local, unmetered",
                "used_tokens": tokens, "used_cost": used["cost"], **_reset(facts)}

    if q.kind == "dollars":
        consumed, limit, basis = used["cost"], q.allowance, "ledger ($ spent this period)"
    elif q.kind == "tokens":
        consumed, limit, basis = tokens, q.allowance, "ledger (tokens this period)"
    else:  # window / unknown
        consumed = tokens
        basis = "ledger (tokens this window)"

    if (isinstance(facts.get("reported_pct_used"), float)
            and reported_is_current(facts, q.period)):
        # The provider gave a percentage and no counts. It is the best number
        # available, so it wins outright — but there is no limit to report, and
        # inventing one from our own tally would be a guess dressed as a fact.
        # Gated on freshness: a stale reading (reset_at gone, reported_at from
        # a previous period bucket) would lock the bar at 100% forever once the
        # window rolls over and no probe updates it, which is the bug at #45.
        return {**meta, "kind": q.kind, "pct_used": round(float(facts["reported_pct_used"]), 1),
                "limit": None, "consumed": consumed,
                "basis": "reported by provider (% only)",
                "used_tokens": tokens, "used_cost": used["cost"], **_reset(facts)}

    if (facts.get("reported_remaining") is not None
            and isinstance(facts.get("reported_remaining"), float)
            and reported_is_current(facts, q.period)):
        # A number the provider gave us always beats our own estimate -- when it
        # is still about the window in force. The same staleness rule as above:
        # a leftover `reported_remaining: 0` from a spent window would otherwise
        # pin the bar at 100% for a fresh week.
        rem = float(facts["reported_remaining"])
        # A stated limit beats reconstructing one from our own consumption.
        total = (float(facts["reported_limit"])
                 if isinstance(facts.get("reported_limit"), float) else rem + consumed)
        return {**meta, "kind": q.kind, "pct_used": _pct(max(0.0, total - rem), total),
                "limit": total,
                "consumed": max(0.0, total - rem), "basis": "reported by provider",
                "used_tokens": tokens, "used_cost": used["cost"], **_reset(facts)}

    if limit is None and isinstance(facts.get("observed_allowance_tokens"), float):
        obs = float(facts["observed_allowance_tokens"])
        if obs > 0:
            limit, basis = obs, "observed (where it ran out last time)"

    return {**meta, "kind": q.kind, "pct_used": _pct(consumed, limit), "limit": limit,
            "consumed": consumed, "basis": basis,
            "used_tokens": tokens, "used_cost": used["cost"], **_reset(facts)}


def _pct(consumed: float, limit: float | None) -> float | None:
    if not limit or limit <= 0:
        return None
    return min(100.0, round(100.0 * consumed / limit, 1))


def _reset(facts: dict) -> dict:
    reset_at = facts.get("reset_at")
    return {
        "reset_at": reset_at if isinstance(reset_at, float) and reset_at else None,
        "last_exhausted_at": facts.get("last_exhausted_at") or None,
    }


def effective_cost_per_mtok(plan: Plan, tokens_this_month: float, metered_cost: float) -> float | None:
    """The number that decides whether a subscription is worth renewing.

    A $132 plan that delivered 40M tokens cost $3.30/Mtok; the same plan at
    400M cost $0.33. Metered providers use actual spend. Returns None when
    there is not enough traffic yet to mean anything.
    """
    if tokens_this_month < 1_000_000:
        return None
    spend = plan.monthly_cost if plan.monthly_cost else metered_cost
    if spend <= 0:
        return 0.0
    return round(spend / (tokens_this_month / 1_000_000), 4)


def model_effective_cost_per_mtok(
    plan: Plan, model_tokens: float, model_cost: float, plan_tokens: float
) -> float | None:
    """Per-model effective $/Mtok.

    The threshold differs by plan kind, because the number's meaning does:

    - metered plan: the rate is the model's own spend over its own tokens, so
      it needs 1M model tokens before it means anything;
    - subscription plan: the fee is allocated by token share, and the share
      CANCELS — spend_m = fee·m/t, so $/Mtok = fee/(t/1M), the plan's own
      rate. Every model that carried any traffic shows it as soon as the
      PLAN crosses 1M tokens; gating on the model's own share would hide a
      real number from a small slice of a subscription that is being used.

    Returns None when the model saw no traffic, or the meaningful threshold
    is not met yet; spend <= 0 -> 0.0; else rounded to 4 decimals.
    """
    if model_tokens <= 0:
        return None
    if plan.metered:
        if model_tokens < 1_000_000:
            return None
        spend = model_cost
    else:
        if plan_tokens < 1_000_000:
            return None
        spend = (plan.monthly_cost or 0.0) * model_tokens / plan_tokens
    if spend <= 0:
        return 0.0
    return round(spend / (model_tokens / 1_000_000), 4)


# Thresholds for `model_effective_cost_per_session`. Hardcoded for the same
# reason `model_effective_cost_per_mtok` is: the numbers are operator-facing
# board values, not knobs, and a missing-threshold path that returns a wrong
# rate is worse than admitting the answer is unknown.
SESSION_MIN_SUBSCRIPTION = 50
SESSION_MIN_METERED = 100


def model_effective_cost_per_session(
    plan: Plan, n_sessions: float, model_cost: float, plan_sessions: float,
) -> float | None:
    """Per-model effective $/session.

    A session here is the unit of work SwitchYard actually meters on the
    board: one CLI / API caller carrying a stable lease. A subscription
    charges one fee for the month and serves N sessions; the fee allocated
    to a model is fee * model_sessions / plan_sessions, and the share cancels
    in the per-session division -- rate = plan.monthly_cost / plan_sessions,
    the plan's own rate, shown to every model once the PLAN clears the
    subscription threshold. A metered plan charges the model's own spend
    divided by its own session count.

    Gating, in order: no sessions -> None (no work to price); subscription
    under its threshold -> None (the plan's own rate would divide by too
    few sessions to mean anything); metered under its threshold -> None;
    spend <= 0 -> 0.0; else rounded to 4 decimals.
    """
    if n_sessions <= 0:
        return None
    if plan.metered:
        if n_sessions < SESSION_MIN_METERED:
            return None
        if model_cost <= 0:
            return 0.0
        return round(model_cost / n_sessions, 4)
    # Subscription: rate is plan-level (the per-model share cancels), so
    # model_cost is unused -- fee / plan_sessions directly. The original
    # round-trip cancel (fee * n / plan_sessions, then /n) produced the
    # same number with one multiply and one divide that a reader has to
    # verify; the form here is the one the docstring already states.
    if plan_sessions < SESSION_MIN_SUBSCRIPTION:
        return None
    fee = plan.monthly_cost or 0.0
    if fee <= 0:
        return 0.0
    return round(fee / plan_sessions, 4)
