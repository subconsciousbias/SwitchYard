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
from .periods import expiry_moment, period_bounds, windows_per_month


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
                                    window: str | None = None,
                                    yard_tokens: float | None = None) -> None:
        """Record a percentage a provider stated, when it publishes no counts.

        MiniMax reports `current_weekly_used_percent` with every count set to -1,
        so there is nothing to reconcile against our own tally — the percentage
        IS the measurement. Kept in its own field so window_headroom can prefer
        it without ever mixing percent into a token or dollar total.

        `yard_tokens` threads the window's current token tally (from
        `ledger.window_usage`) onto the same hash so that the next
        percentage-only reading can derive an upper-bound allowance from
        the pair (prev_pct_used, prev_yard_tokens) -> (new_pct_used, new_yard_tokens).
        Without this carry-forward, the only comparison point between
        successive percent readings is the percentage itself, which is not
        a token figure; the carry is what makes a vendor's percent reading
        projectable into a token allowance.

        Carry rules:
          - Before overwriting the current `reported_pct_used`, the prior
            value is moved to `prev_pct_used` and its companion
            `prev_pct_yard_tokens` (read from `yard_tokens` of the prior
            call, which now sits alongside it). The carry only fires when
            the OLD `reset_at` matches the NEW one — a same-window
            in-force read. A decrease in pct between two same-window
            readings is treated as a window rollover and the carry is
            dropped (the previous reading is no longer about this window).
        """
        # Read the existing hash BEFORE writing so we can detect a
        # same-window carry and a rollover. The same key is used by the
        # write below, so this is intentionally not pipelined — the order
        # read-then-write has to be serialised against another probe
        # landing on the same plan. The prober's `_poll_probes` loop
        # debounces via `due()` so the read-then-write is usually safe,
        # but `POST /admin/probes/{plan_key}/test` in
        # `switchyard/portal/app.py` invokes `prober.run` directly with
        # no `due()` gate, so a button click can interleave with the
        # background poll on the same plan. The worst outcome is a
        # skew `prev_pct_used` / `prev_pct_yard_tokens` pair — two
        # callers each read the hash, both write a carry derived from
        # the same snapshot, and the downstream inference projects
        # arithmetic off a mismatched pair. The resulting tier-2 cap
        # is advisory (the board labels it `≤`), not a fact, so the
        # next poll replaces it within the probe interval — not worth
        # WATCH/MULTI for an advisory figure.
        key = (K_WINDOW.format(plan=plan_key, window=window) if window
               else K_QUOTA.format(plan=plan_key))
        prev_raw = await self.redis.hgetall(key) or {}

        def _get(field: str) -> str | None:
            v = prev_raw.get(field)
            if v is None:
                # Real Redis returns bytes keys + bytes values; the
                # fake returns string keys + string values. Handle both
                # so the carry works against either surface.
                v = prev_raw.get(field.encode() if isinstance(field, str) else field)
                if v is None:
                    return None
            if isinstance(v, bytes):
                v = v.decode()
            return v

        prev_reset = _get("reset_at")
        prev_pct = _get("reported_pct_used")
        prev_yard = _get("reported_at_yard_tokens")

        mapping: dict[str, str | float] = {}
        if used_percent is not None:
            mapping["reported_pct_used"] = max(0.0, min(100.0, float(used_percent)))
        if reset_at is not None:
            mapping["reset_at"] = reset_at
        if not mapping:
            return
        # Carry the previous percent+yard pair forward ONLY when the old
        # reset_at matches the new reset_at (same window in force) AND the
        # new pct is >= the old one (a decrease is the window rolling
        # over — the old pair is no longer about this window). Without
        # this guard, a vendor's "fresh percent" written across a
        # rollover would attach the previous window's token tally to
        # the new window's percentage, projecting an allowance off the
        # wrong bucket.
        try:
            prev_pct_f = float(prev_pct) if prev_pct is not None else None
            prev_reset_f = float(prev_reset) if prev_reset is not None else None
            new_reset_f = float(reset_at) if reset_at is not None else None
            new_pct_f = float(used_percent) if used_percent is not None else None
        except (TypeError, ValueError):
            prev_pct_f = prev_reset_f = new_reset_f = new_pct_f = None
        same_window = (
            prev_reset_f is not None
            and new_reset_f is not None
            and prev_reset_f == new_reset_f
        )
        no_rollover = (
            prev_pct_f is None
            or new_pct_f is None
            or new_pct_f >= prev_pct_f
        )
        prev_persisted = prev_pct is not None or prev_yard is not None
        should_carry = (
            same_window and no_rollover
            and prev_pct is not None and prev_yard is not None
        )
        if should_carry:
            mapping["prev_pct_used"] = prev_pct_f
            mapping["prev_pct_yard_tokens"] = float(prev_yard)
        if yard_tokens is not None:
            # Stamp the new yard_tokens alongside the new pct so the NEXT
            # carry can use it. The field is named to distinguish it from
            # the prev_* fields; same-hash coexistence is intentional so
            # window_facts continues to fetch everything in one round trip.
            mapping["reported_at_yard_tokens"] = float(yard_tokens)
        mapping["reported_at"] = time.time()
        await self.redis.hset(key, mapping=mapping)
        if not should_carry and prev_persisted:
            # Carry is being skipped — either a window rollover (pct
            # decreased) or a different window is in force (reset_at
            # mismatch / missing). The stale pair would otherwise sit
            # on the hash and survive into later readings, so a future
            # `_infer_allowance_from_pair` call could project arithmetic
            # across two unrelated windows (Δpct spanning the boundary,
            # Δyard stretching across the rollover). Drop it via a
            # follow-up HDEL — hset does not delete fields, and a single
            # extra round trip per skipped carry keeps the carry logic
            # local to this method. The next pct-only reading then
            # degrades to tier 3 instead of surfacing a bogus tier-2
            # ceiling.
            await self.redis.hdel(key, "prev_pct_used", "prev_pct_yard_tokens")

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

    Folds the per-window projection caps (added in `window_headroom`) into a
    plan-level `hr["projection"]` dict:

      - `capacity_tokens`: the smallest `monthly_capacity_tokens` across all
        token-kind windows (the argmin -- the binding window's figure). When
        no window yields a token capacity (all dollar / unlimited), this is
        None.
      - `basis`: the basis string of the argmin window. Argmin's basis is
        provably correct: a vendor window binding below a tier-2 window's
        lower-bound cap is exact, so the plan-level basis follows the
        smallest cap.
      - `upper_tokens`: the argmin window's `capacity_upper_tokens` (ceiling
        for tier 2, None for tier 1 / 3). The board can render this as a
        "≤ X" qualifier when the plan's own cap is a lower bound.
      - `cfg_capacity_tokens`: the argmin window's `cfg_monthly_capacity_tokens`
        -- the operator-typed plan figure if one exists.
      - `window`: the label of the argmin window, so the board can point at
        which window the projection came from.

    The argmin is `min` on a single float key (the capacity). Ties go to
    whichever window appears first in `plan.quotas`, which keeps the choice
    deterministic when two windows produce the same number.
    """
    windows = [await window_headroom(ledger, plan, q) for q in plan.quotas]
    target = next((w for w in windows if w["role"] == "target"), windows[0])
    rated = [w for w in windows if w.get("pct_used") is not None]
    binding = max(rated, key=lambda w: w["pct_used"]) if rated else target
    projection = _project_plan(windows)
    return {**target, "windows": windows, "binding": binding,
            "binding_is_target": binding is target,
            "projection": projection}


def _project_plan(windows: list[dict]) -> dict:
    """Fold the per-window projection caps into a plan-level dict.

    Argmin of `monthly_capacity_tokens` across windows that produced one.
    Returns `{"capacity_tokens": None, "basis": None, ...}` when no
    window yields a token capacity (e.g. a plan whose every window is
    dollar or unlimited).
    """
    candidates = [w for w in windows
                  if w.get("monthly_capacity_tokens") is not None
                  and w.get("monthly_capacity_tokens", 0) > 0]
    if not candidates:
        return {"capacity_tokens": None, "basis": None,
                "upper_tokens": None, "cfg_capacity_tokens": None,
                "window": None}
    # Argmin: smallest cap wins. Ties keep the first window encountered
    # in `plan.quotas` order (the order the caller built `windows` in),
    # which is the operator-declared order -- consistent with the existing
    # `target` window choice.
    argmin = min(candidates, key=lambda w: w["monthly_capacity_tokens"])
    return {"capacity_tokens": argmin["monthly_capacity_tokens"],
            "basis": argmin.get("capacity_basis"),
            "upper_tokens": argmin.get("capacity_upper_tokens"),
            "cfg_capacity_tokens": argmin.get("cfg_monthly_capacity_tokens"),
            "window": argmin.get("window")}


async def window_headroom(ledger: Ledger, plan: Plan, q: Quota) -> dict:
    """What the board shows in the 'quota left' column, for one window.

    Returns the per-window dict (basis, pct_used, limit, consumed, ...) AND,
    for `q.kind == "tokens"` windows, the projection fields used by
    `headroom()` to fold per-window caps into a plan-level monthly
    capacity: `monthly_capacity_tokens`, `capacity_basis`, `capacity_upper_tokens`,
    `cfg_monthly_capacity_tokens`, `off_router_tokens_est`.

    Projection tiers, in priority order:

      Tier 1 (vendor): absolutes current via `reported_is_current`. The
        provider gave us both remaining and a stated limit (or we can
        reconstruct one from remaining + our tally), so the per-window
        allowance is exact; cap = (limit or rem + consumed) × windows_per_month.
        Upper bound is None — the vendor number is ground truth.

      Tier 2 (bounded): pct current, no absolutes, but a previous
        same-window reading was carried forward by `note_reported_percent`.
        The pair (Δyard, Δpct) gives A_inf = Δyard × 100 / Δpct (the
        implied allowance if NO usage ran off-router). A_inf ≤ A_true
        always, because bypass ≥ 0; the derived rate is therefore an
        UPPER bound, not the truth. When q.allowance (A_cfg) exists,
        `monthly_capacity_tokens` stays A_cfg × windows_per_month (the
        inference NEVER replaces A_cfg) and `capacity_upper_tokens`
        carries the ceiling. Without A_cfg, `monthly_capacity_tokens`
        itself becomes the inferred ceiling.

      Tier 3 (ledger): no valid prev pair (only one reading on file, or a
        rollover between the two). A = q.allowance else
        observed_allowance_tokens; cap = A × windows_per_month; basis "ledger".

    `off_router_tokens_est` is computed in tier 2 whenever A_cfg is
    present and the prev pair is valid: it is the delta the inferred
    allowance suggests was spent OUTSIDE SwitchYard between the two
    readings (positive => some traffic bypassed us; negative => the
    inferred rate actually beat A_cfg, which is impossible unless the
    vendor's pct is rounding). A positive value adds "observed (where it
    ran out last time)" phrasing to the window's basis string so the
    board signals that off-router traffic is at play.

    Dollar-kind and unlimited windows contribute NO token capacity: a
    dollar allowance cannot be projected to tokens (vendor plans keep
    the historical rate with its caveat). The projection fields are
    present but None for those windows.
    """
    used = await ledger.window_usage(plan, q)
    facts = await ledger.window_facts(plan.key, q.label)
    tokens = used["prompt_tokens"] + used["completion_tokens"]

    basis: str | None = None
    limit: float | None = None
    consumed: float = 0.0

    meta = {"window": q.label, "role": q.role, "period": q.period}

    if q.kind == "unlimited":
        return {**meta, "kind": "unlimited", "pct_used": None, "basis": "local, unmetered",
                "used_tokens": tokens, "used_cost": used["cost"],
                "monthly_capacity_tokens": None,
                "capacity_basis": None, "capacity_upper_tokens": None,
                "cfg_monthly_capacity_tokens": None,
                "off_router_tokens_est": None,
                **_reset(facts)}

    if q.kind == "dollars":
        consumed, limit, basis = used["cost"], q.allowance, "ledger ($ spent this period)"
    elif q.kind == "tokens":
        consumed, limit, basis = tokens, q.allowance, "ledger (tokens this period)"
    else:  # window / unknown
        consumed = tokens
        basis = "ledger (tokens this window)"

    # Reordered: absolutes (reported_remaining / reported_limit) are checked
    # BEFORE percent-only readings. Absolutes need NO token attribution from
    # our own tally -- the provider gave us the numbers, so they are immune
    # to the bypass-attribution problem that follows the percent-only path.
    # Putting absolutes second (after pct-only) used to force every window
    # with both kinds of data through the pct-only branch, which then had to
    # re-derive a limit from `consumed / pct/100` -- a number that ignores
    # everything spent outside SwitchYard.
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
        per_window = max(0.0, total - rem)
        pct = _pct(per_window, total)
        # Projection -- tier 1 (vendor) for tokens windows.
        proj = _project_capacity(q, facts, used, tier=1,
                                 vendor_total=total, pct_used=pct)
        return {**meta, "kind": q.kind, "pct_used": pct,
                "limit": total,
                "consumed": per_window, "basis": "reported by provider",
                "used_tokens": tokens, "used_cost": used["cost"],
                **proj, **_reset(facts)}

    if (isinstance(facts.get("reported_pct_used"), float)
            and reported_is_current(facts, q.period)):
        # The provider gave a percentage and no counts. It is the best number
        # available, so it wins outright -- but the limit is a derived figure,
        # not a stated one. Gated on freshness: a stale reading (reset_at
        # gone, reported_at from a previous period bucket) would lock the bar
        # at 100% forever once the window rolls over and no probe updates it,
        # which is the bug at #45.
        pct = round(float(facts["reported_pct_used"]), 1)
        # Projection -- tier 2 or 3 for tokens windows (no absolutes to use).
        proj = _project_capacity(q, facts, used, tier=None,
                                 vendor_total=None, pct_used=pct)
        basis_str = "reported by provider (% only)"
        # If tier 2 produced a positive off-router estimate, augment the
        # basis with the lower-bound phrasing the spec calls for. Without
        # this the board would not surface the bypass signal on the
        # affected window. The wording is distinct from the unrelated
        # tier-3 "observed (where it ran out last time)" path below --
        # that one means the allowance came from observing the plan run
        # out last cycle; this one means the inferred ceiling hints at
        # off-router traffic so the figure is a lower bound on cost.
        # Same honesty family, different mechanism.
        if (q.kind == "tokens"
                and proj.get("off_router_tokens_est") is not None
                and proj["off_router_tokens_est"] > 0):
            basis_str += " · lower bound (off-router inferred)"
        return {**meta, "kind": q.kind, "pct_used": pct,
                "limit": None, "consumed": consumed,
                "basis": basis_str,
                "used_tokens": tokens, "used_cost": used["cost"],
                **proj, **_reset(facts)}

    if limit is None and isinstance(facts.get("observed_allowance_tokens"), float):
        obs = float(facts["observed_allowance_tokens"])
        if obs > 0:
            limit, basis = obs, "observed (where it ran out last time)"

    # Tier 3 (ledger) for tokens windows; dollars/unknown contribute no
    # token capacity. The fall-through path is also the one with no
    # reported reading at all -- a window whose provider has never polled.
    proj = _project_capacity(q, facts, used, tier=3,
                             vendor_total=None, pct_used=None)
    return {**meta, "kind": q.kind, "pct_used": _pct(consumed, limit), "limit": limit,
            "consumed": consumed, "basis": basis,
            "used_tokens": tokens, "used_cost": used["cost"],
            **proj, **_reset(facts)}


def _project_capacity(q: Quota, facts: dict, used: dict,
                      *, tier: int | None,
                      vendor_total: float | None,
                      pct_used: float | None) -> dict:
    """Compute the projection fields for one window's quota row.

    `tier` is the projection tier the caller picked based on which branch
    it took in `window_headroom`: 1 (vendor / absolutes current), 2
    (bounded / pct current with valid prev pair), 3 (ledger / no reading
    or invalid prev pair), or None (the pct-only branch that needs to
    decide tier 2 vs tier 3 based on the prev pair).

    Returns a dict that can be splat-merged into the window row:

      - `monthly_capacity_tokens`: the cap folded up to a 30-day month.
        For tier 1 it is the vendor limit × windows_per_month(q.period).
        For tier 2 it is A_cfg × windows_per_month when q.allowance is
        set, else the inferred A_inf × windows_per_month. For tier 3
        it is A × windows_per_month with A = q.allowance else
        observed_allowance_tokens.
      - `capacity_basis`: "vendor" | "bounded" | "ledger" -- the source
        the cap is drawn from. Tier 2 with no A_cfg degrades to "ledger"
        and the inferred A_inf becomes the monthly capacity (so the
        board can still show a number); tier 3 is always "ledger".
      - `capacity_upper_tokens`: the ceiling for tier 2 (A_inf × wpm),
        None for tier 1 (no upper bound -- vendor is exact), None for
        tier 3.
      - `cfg_monthly_capacity_tokens`: A_cfg × windows_per_month when
        A_cfg is set, None otherwise. This is the *plan's own* monthly
        cap, kept distinct from the inferred/observed one.
      - `off_router_tokens_est`: A_cfg × Δpct/100 − Δyard when tier 2
        and A_cfg is set; None otherwise. Positive => some traffic
        bypassed SwitchYard in the (prev, current) window.

    Dollar-kind and unlimited windows get all-None fields (no token
    projection possible): a dollar allowance cannot be converted to
    tokens, and the Go plans keep the historical rate with its caveat.
    """
    wpm = windows_per_month(q.period)
    none_proj = {"monthly_capacity_tokens": None,
                 "capacity_basis": None, "capacity_upper_tokens": None,
                 "cfg_monthly_capacity_tokens": None,
                 "off_router_tokens_est": None}
    if q.kind != "tokens":
        return none_proj
    a_cfg = q.allowance if isinstance(q.allowance, (int, float)) else None
    cfg_monthly = (a_cfg * wpm) if a_cfg is not None and a_cfg > 0 else None

    if tier == 1:
        # Vendor absolutes: the provider's own total is the per-window
        # allowance, no inference involved. Cap = vendor_total × wpm.
        # Upper bound is None because the vendor number is exact.
        if vendor_total is None or vendor_total <= 0:
            # No stated limit and no fallback -> degrade to tier 3.
            return _project_tier3(q, facts, wpm, a_cfg, cfg_monthly)
        return {"monthly_capacity_tokens": vendor_total * wpm,
                "capacity_basis": "vendor",
                "capacity_upper_tokens": None,
                "cfg_monthly_capacity_tokens": cfg_monthly,
                "off_router_tokens_est": None}

    # tier is None (pct-only branch) or tier == 2 explicitly: same math.
    if tier is None or tier == 2:
        inferred = _infer_allowance_from_pair(facts, used)
        if inferred is None:
            return _project_tier3(q, facts, wpm, a_cfg, cfg_monthly)
        a_inf, delta_pct, delta_yard = inferred
        cap_upper = a_inf * wpm
        # off_router_tokens_est = A_cfg × Δpct/100 − Δyard when A_cfg
        # exists and the prev pair is valid. Positive => some traffic
        # bypassed SwitchYard between the two probes (the cfg allowance
        # saw more spend than our own ledger). Negative => SwitchYard
        # saw more spend than the cfg allowance predicts (rounding, or
        # the cfg figure is too low). The board only surfaces the
        # lower-bound phrasing when this is strictly positive.
        off_router = None
        if a_cfg is not None and a_cfg > 0 and delta_pct > 0:
            off_router = a_cfg * (delta_pct / 100.0) - delta_yard
        if a_cfg is not None and a_cfg > 0:
            # Inference NEVER replaces A_cfg -- the cfg figure is what
            # the operator typed, and a too-small upper bound would only
            # tell them to lower plans.yaml. Surface the ceiling alongside.
            return {"monthly_capacity_tokens": cfg_monthly,
                    "capacity_basis": "bounded",
                    "capacity_upper_tokens": cap_upper,
                    "cfg_monthly_capacity_tokens": cfg_monthly,
                    "off_router_tokens_est": off_router}
        # No A_cfg: the inferred ceiling IS the monthly capacity. Mark
        # it "bounded" (not "ledger") so the board shows that an
        # inference produced it, not the fallback.
        return {"monthly_capacity_tokens": cap_upper,
                "capacity_basis": "bounded",
                "capacity_upper_tokens": cap_upper,
                "cfg_monthly_capacity_tokens": None,
                "off_router_tokens_est": off_router}

    # tier == 3 (or unknown -> treat as 3): ledger basis.
    return _project_tier3(q, facts, wpm, a_cfg, cfg_monthly)


def _project_tier3(q: Quota, facts: dict, wpm: float,
                   a_cfg: float | None, cfg_monthly: float | None) -> dict:
    """Tier 3: A = q.allowance else observed_allowance_tokens -> cap = A × wpm."""
    a = a_cfg
    if (a is None or a <= 0) and isinstance(
            facts.get("observed_allowance_tokens"), float):
        obs = float(facts["observed_allowance_tokens"])
        if obs > 0:
            a = obs
    if a is None or a <= 0:
        return {"monthly_capacity_tokens": None,
                "capacity_basis": "ledger",
                "capacity_upper_tokens": None,
                "cfg_monthly_capacity_tokens": cfg_monthly,
                "off_router_tokens_est": None}
    return {"monthly_capacity_tokens": a * wpm,
            "capacity_basis": "ledger",
            "capacity_upper_tokens": None,
            "cfg_monthly_capacity_tokens": cfg_monthly,
            "off_router_tokens_est": None}


def _infer_allowance_from_pair(facts: dict, used: dict
                               ) -> tuple[float, float, float] | None:
    """Infer A_inf from the (prev, current) (yard, pct) pair.

    Reads `prev_pct_used`, `prev_pct_yard_tokens`, `reported_pct_used`,
    and `reported_at_yard_tokens` from the window facts. Returns None
    when the prev pair is missing or invalid (a single reading, a stale
    prev, or a delta that would divide by zero).

    Math: the prev snapshot says (Y0, P0); the current one says (Y1, P1).
    The yard-tokens delta is Y1 - Y0, which is what SwitchYard saw pass
    through between the two probes; the pct delta is P1 - P0, which is
    what the vendor's percentage moved by. The implied per-window
    allowance (assuming ALL traffic went through SwitchYard) is:

        A_inf = (Y1 - Y0) * 100 / (P1 - P0)

    The caller has A_cfg in scope (q.allowance) and computes the bypass
    itself: bypass = A_cfg * (P1 - P0) / 100 - (Y1 - Y0). That form is
    what the spec calls out -- the inference gives A_inf, the bypass
    uses A_cfg as the reference. We surface (Δpct, Δyard) back so the
    caller can reuse the same deltas without re-decoding `facts`.

    `used` is the current window's `window_usage` dict; only its
    `prompt_tokens` + `completion_tokens` is needed (Y1). If Y1 is
    missing -- a window whose bucket TTL expired between the two probes
    -- we fall back to `reported_at_yard_tokens` for Y1 (the figure
    `note_reported_percent` stamps when the new reading lands), which
    was taken at exactly the same instant.

    Returns `(A_inf, delta_pct, delta_yard)` on success, None on any
    failure path. The caller gates "observed (where it ran out last
    time)" phrasing on a strictly positive `off_router_tokens_est`.
    """
    try:
        prev_pct_f = float(facts.get("prev_pct_used"))
        prev_yard_f = float(facts.get("prev_pct_yard_tokens"))
        curr_pct_f = float(facts.get("reported_pct_used"))
    except (TypeError, ValueError):
        return None
    # Prefer the stamp `note_reported_percent` wrote alongside the new
    # reading, falling back to the live window_usage when that stamp is
    # absent (a probe path that does not pass yard_tokens). The two
    # should agree within rounding; we trust the stamp when present
    # because it is a snapshot at the same instant as `reported_pct_used`.
    curr_yard_raw = facts.get("reported_at_yard_tokens")
    if curr_yard_raw is None:
        curr_yard_f = (used.get("prompt_tokens", 0.0)
                       + used.get("completion_tokens", 0.0))
    else:
        try:
            curr_yard_f = float(curr_yard_raw)
        except (TypeError, ValueError):
            curr_yard_f = (used.get("prompt_tokens", 0.0)
                           + used.get("completion_tokens", 0.0))
    delta_pct = curr_pct_f - prev_pct_f
    delta_yard = curr_yard_f - prev_yard_f
    if delta_pct <= 0:
        # Same-window pct delta must be positive: a non-positive means
        # the readings are the same observation, the window rolled over
        # between them (handled in the carry gate), or one of the pair
        # is missing. We treat anything here as "no inference possible"
        # -- the rollover case was already filtered by the carry rule
        # in note_reported_percent.
        return None
    if delta_yard < 0:
        # Negative yard delta means the bookkeeping rolled back between
        # probes (a window-bucket TTL expired or a manual reset); not a
        # basis for projection.
        return None
    a_inf = delta_yard * 100.0 / delta_pct
    if a_inf <= 0 or a_inf != a_inf:  # NaN check via self-inequality
        return None
    return a_inf, delta_pct, delta_yard


async def drain_risk(ledger: Ledger, plan: Plan,
                     now: datetime | None = None) -> str | None:
    """Why an expiring plan should jump its lane order right now, or None.

    An expiry date alone is not a reason: a plan with weeks left and quota
    that resets before then loses nothing by waiting its turn, and promoting
    it just starves the plans the operator ordered first. Two conditions,
    judged per quota window, must both hold:

      1. The plan expires before the window in force rolls over. This is its
         last window, so whatever is left in it is lost instead of reset.
      2. Consumption so far, extrapolated linearly to the expiry, falls short
         of the allowance -- i.e. the window is behind its pace line. At the
         rate normal routing is already achieving, it will not drain itself.

    Usage comes from `window_headroom`, the same number the board shows. A
    window whose usage is unknown (no allowance, no provider reading) never
    promotes: without a number, "at risk" would be a guess. A spent window
    has nothing left to rescue.
    """
    end = expiry_moment(plan.expires)
    if end is None:
        return None
    now = now or datetime.now(timezone.utc)
    for q in plan.quotas:
        if q.kind == "unlimited":
            continue
        h = await window_headroom(ledger, plan, q)
        start, rollover = period_bounds(q.period, now)
        reset_at = h.get("reset_at")
        if isinstance(reset_at, float) and reset_at > now.timestamp():
            # The provider's own reset beats the calendar bucket: a weekly
            # window that resets Thursdays is not the Monday-to-Monday week.
            provider_rollover = datetime.fromtimestamp(reset_at, tz=timezone.utc)
            start = provider_rollover - (rollover - start)
            rollover = provider_rollover
        if end > rollover:
            continue  # the window resets before the plan dies; nothing is lost
        pct = h.get("pct_used")
        if pct is None or pct >= 100.0:
            continue
        span = (end - start).total_seconds()
        if span <= 0:
            continue
        elapsed = min(1.0, max(0.0, (now - start).total_seconds() / span))
        if pct / 100.0 < elapsed:
            return (f"{q.label} final window {pct:.0f}% used, "
                    f"{elapsed * 100:.0f}% elapsed")
    return None


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

    Backwards-compatible wrapper for `effective_cost_fields(..., projected=None)`:
    a metered plan keeps the historical rate (spend ÷ tokens so far); a
    subscription without a projection also keeps the historical shape so the
    existing tests stay meaningful. Issue #218: a subscription WITH a
    projection now projects from the binding window instead. Use the
    `_fields` variant for board rendering and `/api/state` — that is the
    one that returns the tiered basis the board needs.
    """
    return effective_cost_fields(plan, tokens_this_month, metered_cost,
                                 projected=None)["rate"]


def effective_cost_fields(
    plan: Plan, tokens_this_month: float, metered_cost: float,
    projected: dict | None = None,
) -> dict:
    """Effective $/Mtok with provenance — used by the board's Effective column
    and `GET /api/state`.

    Returns a dict `{"rate": float|None, "basis": str|None, "upper": float|None}`:

      - `rate` — the figure to render, or None when nothing meaningful exists.
      - `basis` — one of `"vendor"`, `"bounded"`, `"ledger"`. The board uses
        this to pick the provenance glyph (vendor / ≤ / yard) and the
        tooltip text. None when no figure was produced.
      - `upper` — the inferred ceiling for `"bounded"` tiers, None otherwise.
        When set, the board renders a range `$low–high/Mtok` instead of a
        point estimate.

    Tiers (subscription with `projected`):

      - **vendor**: `projected["capacity_basis"] == "vendor"`. Cap came from
        a vendor-stated absolute limit; rate = fee / (cap / 1e6) with NO
        1M-token gate — the projection is independent of burn, which is
        the whole point of the fix.
      - **bounded**: `projected["capacity_basis"] == "bounded"` AND a
        `cfg_capacity_tokens` exists. The cfg allowance wins as the rate
        (the inference NEVER replaces A_cfg); `upper` carries the inferred
        ceiling, so the board shows `$low–high/Mtok`.
      - **bounded** (no A_cfg): the inferred ceiling IS the rate, marked
        `"bounded"` so the board knows to prefix it with `≤`.
      - **ledger**: `projected["capacity_basis"] == "ledger"`, or no
        projection at all. The plan had no vendor reading and no inferred
        allowance; we fall back to the historical `fee / tokens_so_far`
        shape so the board still shows *something* (a fact-tinged
        estimate).

    Metered plans always take the historical branch (`fee / tokens`,
    basis `"ledger"`, upper None), regardless of `projected` — the
    metered economics are different and don't have a quota window to
    project from.

    Threshold rules:
      - fee <= 0 -> rate 0.0 (the existing convention).
      - capacity_tokens <= 0 or None -> fall back to historical if it has a
        figure, else None (no projection possible).
    """
    if plan.metered:
        # Metered plans keep the historical shape — no quota window to project from.
        if tokens_this_month < 1_000_000:
            return {"rate": None, "basis": None, "upper": None}
        spend = plan.monthly_cost if plan.monthly_cost else metered_cost
        if spend <= 0:
            return {"rate": 0.0, "basis": "ledger", "upper": None}
        return {"rate": round(spend / (tokens_this_month / 1_000_000), 4),
                "basis": "ledger", "upper": None}

    # Subscription plan.
    fee = plan.monthly_cost or 0.0
    if fee <= 0:
        return {"rate": 0.0, "basis": None, "upper": None}

    cap = (projected or {}).get("capacity_tokens")
    cap_basis = (projected or {}).get("basis")
    upper_cap = (projected or {}).get("upper_tokens")

    if cap and cap > 0:
        # Projection — independent of burn, no 1M-token gate.
        rate = round(fee / (cap / 1_000_000), 4)
        if cap_basis == "bounded":
            # `upper` carries the inferred ceiling rate whenever a
            # distinct ceiling exists; it represents "if the cfg
            # allowance is wrong, the rate could be this high".
            # `upper_cap < cap` is the common case (the inferred
            # allowance is below the cfg figure), which produces a
            # HIGHER rate from `upper_cap` than from `cap`. `upper_cap
            # > cap` is also legitimate (the cfg under-counted the
            # allowance), in which case `upper_cap` produces a LOWER
            # rate — the template renders the range with `≤` on the
            # high end so the upper bound always reads as the high end
            # regardless of which side is bigger.
            #
            # Skip the `upper` field when `upper_cap == cap`: that
            # branch is the no-A_cfg tier-2 path where the inferred
            # allowance IS the cap, so a range would print the same
            # number twice (the `≤` prefix already carries the
            # "upper bound" semantics for that single figure).
            upper = None
            if upper_cap and upper_cap > 0 and upper_cap != cap:
                upper = round(fee / (upper_cap / 1_000_000), 4)
            return {"rate": rate, "basis": "bounded", "upper": upper}
        # vendor or ledger basis: a single point estimate.
        return {"rate": rate, "basis": cap_basis or "ledger", "upper": None}

    # No knowable projection — historical fallback so the board still shows
    # something (the existing fee / tokens_so_far shape).
    if tokens_this_month < 1_000_000:
        return {"rate": None, "basis": None, "upper": None}
    return {"rate": round(fee / (tokens_this_month / 1_000_000), 4),
            "basis": "ledger", "upper": None}


def model_effective_cost_per_mtok(
    plan: Plan, model_tokens: float, model_cost: float, plan_tokens: float
) -> float | None:
    """Per-model effective $/Mtok.

    Backwards-compatible wrapper for `model_effective_cost_fields(...,
    projected=None)`. Same threshold rules as before: metered plans gate on
    1M model tokens, subscription plans gate on 1M PLAN tokens, and the
    per-model share cancels for subscriptions (rate = plan monthly_cost /
    plan_tokens). With a `projected` dict, subscription rows use the binding
    window's projected capacity instead — the board passes the same
    `projected` it computed at the plan level so per-model rows inherit
    the projection verbatim.
    """
    return model_effective_cost_fields(
        plan, model_tokens, model_cost, plan_tokens, projected=None)["rate"]


def model_effective_cost_fields(
    plan: Plan, model_tokens: float, model_cost: float, plan_tokens: float,
    projected: dict | None = None,
) -> dict:
    """Per-model effective $/Mtok with provenance.

    Same return shape as `effective_cost_fields`. The per-model share
    cancels for subscriptions (spend_m = fee·m/t), so the projected rate
    is the same plan-level figure every model shows — passing the same
    `projected` to every row keeps them aligned with the plan row, and the
    spec calls for one canonical helper so board/gateway agree.

    Threshold rules:
      - metered plan: gates on 1M model tokens (the rate is meaningful
        only once the model has its own token total).
      - subscription plan: the fee is allocated by token share; the share
        cancels, so the rate is the plan's own. Gates on 1M plan tokens
        for the historical branch (the model's own share is irrelevant
        once the projection is in scope).
      - subscription with projection: NO 1M-token gate — the projection
        is independent of burn, which is the point.
    """
    if model_tokens <= 0:
        return {"rate": None, "basis": None, "upper": None}

    if plan.metered:
        if model_tokens < 1_000_000:
            return {"rate": None, "basis": None, "upper": None}
        spend = model_cost
        if spend <= 0:
            return {"rate": 0.0, "basis": "ledger", "upper": None}
        return {"rate": round(spend / (model_tokens / 1_000_000), 4),
                "basis": "ledger", "upper": None}

    # Subscription plan.
    fee = plan.monthly_cost or 0.0
    cap = (projected or {}).get("capacity_tokens")
    cap_basis = (projected or {}).get("basis")
    upper_cap = (projected or {}).get("upper_tokens")

    if cap and cap > 0:
        rate = round(fee / (cap / 1_000_000), 4)
        if cap_basis == "bounded":
            # Skip `upper` when `upper_cap == cap` (no-A_cfg tier-2:
            # the inferred allowance IS the cap, so a range would print
            # the same number twice -- the `≤` prefix already carries
            # the "upper bound" semantics for that single figure).
            upper = None
            if upper_cap and upper_cap > 0 and upper_cap != cap:
                upper = round(fee / (upper_cap / 1_000_000), 4)
            return {"rate": rate, "basis": "bounded", "upper": upper}
        return {"rate": rate, "basis": cap_basis or "ledger", "upper": None}

    if plan_tokens < 1_000_000:
        return {"rate": None, "basis": None, "upper": None}
    spend = fee * model_tokens / plan_tokens
    if spend <= 0:
        return {"rate": 0.0, "basis": "ledger", "upper": None}
    return {"rate": round(spend / (model_tokens / 1_000_000), 4),
            "basis": "ledger", "upper": None}


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
