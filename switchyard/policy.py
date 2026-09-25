"""Effective capacity: what a plan is *actually* allowed to run right now.

Two independent controllers, both optional, both per plan:

ConcurrencyLearner — discovers the real parallelism a provider tolerates, since
    "probably under 4 connections" is a guess. Additive-increase /
    multiplicative-decrease, the same shape as TCP congestion control: halve on
    a connection-limit rejection, creep up by one when there is unmet demand and
    no recent rejection. Learned per hour-of-day, because a provider that
    tolerates 6 connections at 04:00 may tolerate 2 at peak.

Pacer — keeps a subscription tracking to ~100% consumption at the moment its
    quota rolls over, instead of burning the allowance early and spilling to the
    tail. Two mechanisms, because concurrency alone is too coarse a knob: at
    real LLM throughput even a single busy slot can drain a monthly allowance in
    a couple of days, so "reduce concurrency" cannot slow us below one slot.

      1. A pace line — the consumption we *should* have reached by now:

             pace_line = allowance * elapsed_fraction * (1 + overshoot)

         While consumption is ahead of that line the plan is closed (0 slots)
         and reopens when the line catches up. That duty-cycles the plan, and
         the average lands on the line however fast individual requests are.

      2. A throughput-derived cap for when we are on or behind the line:

             slots = (remaining / seconds_left) / observed_rate_per_slot

    `deadline` is min(next rollover, expiry) — see periods.py. Because quota
    resets per window, the final truncated window of a cancelled plan paces
    *harder*: same allowance, less time, so a steeper line.

Only subscriptions are paced. Metered providers (OpenRouter, the Anthropic API)
and local models keep fixed caps and the ordinary spill-and-cooldown behaviour,
because there is no allowance to land exactly on.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from redis.asyncio import Redis

from .models import Plan, Quota, Settings
from .periods import deadline as window_deadline
from .periods import period_bounds
from .usage import K_WINDOW, Ledger, period_key, reported_is_current

K_SWITCH = "sy:switch:pacing"     # runtime override of settings.pacing.enabled
K_LEARN = "sy:learn:{plan}:{bucket}"
K_PACE = "sy:pace:{plan}"
K_PRESSURE = "sy:pressure:{plan}"

EWMA_ALPHA = 0.3

# Tolerance for promoting a candidate probe reading to a confirmed prior
# in the percent-only estimate branch. Two consecutive raw readings must
# agree within this relative band before the more conservative one is
# allowed to harden into a clamp value.
#
# Sized as a pct-aware function (`ESTIMATE_TOLERANCE` is the floor) so the
# band absorbs vendor whole-percent quantization where it bites hardest —
# at low pct. A vendor reporting whole-percent pct values gives adjacent
# readings that differ in the *estimate* by approximately 1/(pct+1) in
# relative terms (consumed flat, pct 2 -> 3 yields estimates 50c and
# 33c, rel_diff ≈ 0.33; pct 1 -> 2 yields 100c and 50c, rel_diff ≈ 0.5).
# A flat 0.10 band never concorded those adjacent readings — the candidate
# flip-flopped forever and the prior never hardened, leaving the inflation
# case this mechanism exists for unguarded exactly at low pct.
#
# The band is CAPPED at `CONCORDANCE_BAND_CAP` (0.40) so the widening does
# not become vacuous: at pct=1 the uncapped band is 1.0, which concorded
# essentially any pair of readings and dissolved the "two independent
# readings must agree" property entirely. The cap keeps adjacent
# whole-percent readings concorded (rel_diff ≤ 0.33 at pct=2, 0.25 at
# pct=3) while rejecting multi-step jumps (rel_diff 0.5 at pct=1->2,
# 0.67 at pct=1->3). The cap is the guard against hardening a pair of
# *inflated* readings whose truth sits below both — the more conservative
# of the pair is still inflated, and a permissive band would let that
# inflation harden.
#
# `prior = min(candidate, new_estimate)` on concordance is *within-pair*
# protection, not absolute protection — the prior is the smaller of the
# two readings, which is conservative only when the pair brackets truth.
# A pair of adjacent whole-percent readings straddling truth (e.g. true
# pct 3 reading as 1 then 2, both of which are one-step jumps away from
# the truth) still yields an inflated prior of min(200, 100) = 100 if
# the band is wide enough to concord them. The cap is what rejects the
# one-step jump at pct=1->2 (rel_diff 0.5, capped tolerance 0.40) and
# any multi-step jump (pct=1->3 rel_diff 0.67, etc.) — only true
# adjacent whole-percent readings at pct >= 2 (rel_diff <= 0.33) are
# allowed to concord.
ESTIMATE_TOLERANCE = 0.10
CONCORDANCE_BAND_CAP = 0.40

# Floating-point epsilon for the concordance comparison. Adjacent whole-
# percent pairs have rel_diff *mathematically* identical to the tolerance
# (both are 1/pct_new), so the `<=` comparison is decided by one ULP.
# Verified by sweep across 54 random consumed magnitudes per pair: 3->4
# failed 36/54 trials, 6->7 54/54, 9->10 37/54 — floating-point
# arithmetic made the rel_diff fraction land fractionally above the
# tolerance and rejected the pair. Widening the tolerance by ~1e-9
# relative (~2.5e-10 at tolerance 0.25) is far above ULP noise (~1e-16)
# and stays well below any rel_diff gap the cap exists to reject
# (cap rel_diff gap at pct=2 is 0.50 - 0.40 = 0.10, ~1e8x the epsilon).
CONCORDANCE_EPS = 1e-9


def _concordance_tolerance(pct: float) -> float:
    """Relative band for two adjacent whole-percent readings to concord.

    Sized to absorb a one-step vendor quantization jump (e.g. pct 2 -> 3)
    at every pct, capped at `CONCORDANCE_BAND_CAP` so the one-step jump
    at pct=1->2 (rel_diff 0.5, which would otherwise concord a pair of
    inflated readings straddling truth) and any multi-step jump are
    rejected. Floored at ESTIMATE_TOLERANCE so high-pct windows keep the
    tighter band.
    """
    return max(ESTIMATE_TOLERANCE, min(CONCORDANCE_BAND_CAP, 1.0 / max(pct, 1.0)))


@dataclass
class Capacity:
    cap: int
    reason: str
    learned: int | None = None
    paced: int | None = None
    configured: int | None = None

    @property
    def throttled(self) -> bool:
        return self.paced is not None and self.cap < (self.learned or self.cap)


def _bucket(settings: Settings, at: datetime | None = None) -> str:
    if settings.concurrency_learning.buckets != "hour_of_day":
        return "global"
    now = at or datetime.now(timezone.utc)
    return f"h{now.hour:02d}"


def _f(raw, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


class ConcurrencyLearner:
    def __init__(self, redis: Redis, settings: Settings):
        self.redis = redis
        self.settings = settings

    async def _read(self, plan_key: str, bucket: str) -> dict[str, float]:
        raw = await self.redis.hgetall(K_LEARN.format(plan=plan_key, bucket=bucket))
        out: dict[str, float] = {}
        for k, v in (raw or {}).items():
            k = k.decode() if isinstance(k, bytes) else k
            out[k] = _f(v)
        return out

    async def note_rejection(self, plan: Plan, at_concurrency: int) -> int:
        """The provider refused us on connection count: back off, hard."""
        cfg = self.settings.concurrency_learning
        floor = max(1, cfg.min_cap)
        new = max(floor, int(at_concurrency * cfg.decrease_factor))
        now = time.time()
        for bucket in {_bucket(self.settings), "global"}:
            key = K_LEARN.format(plan=plan.key, bucket=bucket)
            current = _f((await self.redis.hget(key, "cap")), new)
            await self.redis.hset(key, mapping={
                "cap": min(new, current) if current else new,
                "changed_at": now,
                "last_rejection_at": now,
            })
            await self.redis.hincrbyfloat(key, "rejections", 1)
        await self.redis.delete(K_PRESSURE.format(plan=plan.key))
        return new

    async def note_pressure(self, plan_key: str) -> None:
        """A request wanted a slot and the cap denied it. Demand exists."""
        key = K_PRESSURE.format(plan=plan_key)
        await self.redis.incr(key)
        await self.redis.expire(key, 3600)

    async def effective(self, plan: Plan) -> tuple[int, str]:
        """Learned cap for right now, probing upward when it is safe to."""
        cfg = self.settings.concurrency_learning
        # Absent an explicit ceiling, never probe above the number the config
        # states. This used to default to 4x it, which meant a plan configured
        # at 2 silently climbed to 8: the board then drew more slots than
        # plans.yaml declared, with nothing saying why. Probing *past* a stated
        # limit is a decision only the operator can make, so it needs
        # `max_parallel_ceiling`. Learning still does the valuable half by
        # itself -- backing off below the limit under pressure.
        #
        # `max_parallel: auto` is the case with nothing stated, so there the
        # seed is a guess and the ceiling has to come from cfg instead.
        if plan.max_parallel_ceiling:
            ceiling = plan.max_parallel_ceiling
        elif plan.configured_parallel is not None:
            ceiling = plan.configured_parallel
        else:
            ceiling = max(cfg.seed_cap, 1) * 4
        seed = plan.configured_parallel or cfg.seed_cap

        if not plan.learns(self.settings):
            # Either learning is off globally, or this plan opted out with
            # `learning: false`. Its cap is exactly what the config says.
            return max(1, seed), "configured"

        bucket = _bucket(self.settings)
        local = await self._read(plan.key, bucket)
        glob = await self._read(plan.key, "global")

        # Prefer this hour's learning once it has seen enough evidence.
        if local.get("samples", 0) >= cfg.min_samples and local.get("cap"):
            cap, source = int(local["cap"]), f"learned[{bucket}]"
            state = local
        elif glob.get("cap"):
            cap, source = int(glob["cap"]), "learned[global]"
            state = glob
        else:
            cap, source = max(1, seed), "seed" if plan.configured_parallel is None else "configured"
            state = {}

        # Probe upward: only with unmet demand, a quiet period since the last
        # rejection, and a gap since the last change.
        now = time.time()
        pressure = _f(await self.redis.get(K_PRESSURE.format(plan=plan.key)))
        quiet = now - state.get("last_rejection_at", 0) > cfg.probe_cooldown_seconds
        settled = now - state.get("changed_at", 0) > cfg.probe_interval_seconds
        if pressure >= cfg.probe_pressure and quiet and settled and cap < ceiling:
            cap = min(ceiling, cap + cfg.increase_step)
            key = K_LEARN.format(plan=plan.key, bucket=bucket)
            await self.redis.hset(key, mapping={"cap": cap, "changed_at": now})
            await self.redis.hincrbyfloat(key, "samples", 1)
            await self.redis.delete(K_PRESSURE.format(plan=plan.key))
            source += "+probe"

        return max(1, min(cap, ceiling)), source


class Pacer:
    def __init__(self, redis: Redis, settings: Settings, ledger: Ledger):
        self.redis = redis
        self.settings = settings
        self.ledger = ledger

    async def note_throughput(self, plan: Plan, units: float, seconds: float) -> None:
        """One slot delivered `units` in `seconds`: that is the per-slot rate."""
        if units <= 0 or seconds <= 0:
            return
        key = K_PACE.format(plan=plan.key)
        prior = _f(await self.redis.hget(key, "rate"))
        sample = units / seconds
        rate = sample if prior <= 0 else (EWMA_ALPHA * sample + (1 - EWMA_ALPHA) * prior)
        await self.redis.hset(key, mapping={"rate": rate, "rate_at": time.time()})

    async def _window(self, plan: Plan, q: Quota, now: datetime) -> dict:
        """Everything about one quota window: where we are, and the rate it allows."""
        used = await self.ledger.window_usage(plan, q, now)
        facts = await self.ledger.window_facts(plan.key, q.label)
        consumed = (used["cost"] if q.kind == "dollars"
                    else used["prompt_tokens"] + used["completion_tokens"])

        allowance, basis = q.allowance, "configured"
        if allowance is None:
            observed = facts.get("observed_allowance_cost" if q.kind == "dollars"
                                 else "observed_allowance_tokens")
            if isinstance(observed, float) and observed > 0:
                allowance, basis = observed, "observed"
        if allowance is None:
            # Percent-only fallback: invert the probe's percentage into an
            # estimate of the allowance itself. MiniMax publishes no token or
            # dollar counts, only a percentage, so without this inversion a
            # percent-only window (`source: probe`, `allowance: null`) leaves
            # the pacer with nothing to pace on and the plan idles while the
            # allowance burns. With `consumed / (pct/100)`, the inverse is
            # exactly the unit-consistent denominator the rest of the pacer
            # expects: `consumed_frac` lands at pct/100 and the ahead-of-pace
            # / spent / cap maths all read in their own native units (tokens
            # or dollars), the same way `note_throughput`'s per-slot rate
            # feeds the same unit switch.
            #
            # Three gates must all hold before we estimate:
            #
            #   (a) `reported_pct_used` is a float in (0, 100]. 0% is excluded
            #       because dividing by zero is undefined; 100% is allowed and
            #       routes the plan to the spent branch below. Anything outside
            #       the band (a typo, a negative number) is treated as missing.
            #   (b) `consumed > 0`: the inversion needs a numerator. A fresh
            #       week where nothing has yet been recorded stays idle, not
            #       "estimate an infinite allowance"; the picker's own stale
            #       reads + the vendor's first 429 self-heal a quiet plan.
            #   (c) `reported_is_current`: a stale reading (reset_at already
            #       past, or reported_at from a previous period bucket) would
            #       otherwise pin the estimate against an old window, the
            #       same trap the headroom bar hit at #45.
            #
            # `now.timestamp()` is the same injected clock `_window` is
            # driven from, so a stale reading check in the test uses the
            # fixed `NOW` rather than `time.time()`.
            #
            # Known limitations of the estimate (intentional tradeoffs):
            #
            #   * Between probe intervals consumption is pinned to the
            #     last probe's pct — the only number we have. The bound is
            #     `probe.interval_seconds` (typically 120s), so the lag is
            #     small relative to the window.
            #   * Usage outside SwitchYard shrinks the estimate: our own
            #     `consumed` undercounts, so `consumed / (pct/100)` under-
            #     estimates the real allowance and the pacer throttles
            #     harder. That is one of two directions, not the only one:
            #     a `pct` that lags the truth (vendor quantization on whole-
            #     percent readings, a delayed update) makes the *denominator*
            #     too small, which inflates `allowance` with no upper bound
            #     and loosens both the pace line and `allowed_rate` until
            #     the next probe catches up. The opposite also exists: an
            #     over-read on the very first usable reading (a vendor that
            #     rounds UP, a probe racing the ledger so consumed briefly
            #     lags the pct) would, under a naive `min(new, prior)`
            #     ratchet, lock the estimate under the true allowance for
            #     the rest of the window — throughput loss in the inverse
            #     direction. Both are guarded below: the estimate is only
            #     used as a clamp once *two* consecutive readings agree
            #     within `_concordance_tolerance(pct)` relative (pct-aware
            #     so vendor whole-percent quantization at low pct still
            #     concorded instead of flip-flopping the candidate forever,
            #     and CAPPED at `CONCORDANCE_BAND_CAP` so the widening does
            #     not become vacuous and concord a multi-step jump OR the
            #     pct=1->2 one-step jump, both of which would harden a pair
            #     of inflated readings). The candidate sits as the pending
            #     reading; a second concordant reading promotes the more
            #     conservative of the pair to the prior
            #     (within-pair protection); a divergent reading replaces
            #     the candidate without touching the prior. On-window
            #     inflation now throttles at least as hard as the last
            #     concordant pair said to; on-window over-reads never
            #     harden into a too-small allowance.
            #
            # Self-healing concurrency caveat: the read-facts → decide →
            # hset sequence below is NOT atomic across overlapping
            # `_window` callers (the portal board and the gateway picker
            # both drive it). A stale writer can clobber a freshly
            # hardened prior or the new `reset_at`/`new_bucket` window
            # identity, momentarily reverting to the previous window.
            # Every failure mode self-heals within one probe interval
            # (the next reading re-derives the candidate/prior from the
            # current state), so the worst outcome is a delayed clamp or
            # one extra unclamped reading — never a stuck value. The
            # decision was left as plain hash writes rather than moved
            # into a Lua/WATCH-MULTI pipeline because none of those
            # consequences rises above 'self-heals'; revisit if a future
            # probe-side caller needs hard atomicity.
            pct = facts.get("reported_pct_used")
            if (isinstance(pct, float) and 0.0 < pct <= 100.0
                    and consumed > 0
                    and reported_is_current(facts, q.period, now.timestamp())):
                new_estimate = consumed / (pct / 100.0)
                # Key the persistence by `q.kind`, the same pattern as
                # `observed_allowance_*`: tokens-kind windows write
                # `estimated_allowance_tokens`, dollars-kind windows write
                # `estimated_allowance_cost`. The read path mirrors it.
                # `reset_at` is stringified to compare across the float /
                # `""` boundary (`note_exhaustion` writes `""` when there
                # is no hint). Window identity (reset_at + period bucket)
                # bounds the prior and candidate to the window in force;
                # a roll-over clears both.
                est_key = "estimated_allowance_cost" if q.kind == "dollars" \
                    else "estimated_allowance_tokens"
                prior = facts.get(est_key)
                prior_reset = str(facts.get("estimated_allowance_reset_at", ""))
                prior_bucket = facts.get("estimated_allowance_bucket")
                candidate = facts.get("estimated_allowance_candidate")
                new_reset = str(facts.get("reset_at", ""))
                new_bucket = period_key(q.period, now)
                window_unchanged = (prior_reset == new_reset
                                    and prior_bucket == new_bucket)
                if not window_unchanged:
                    prior = None
                    candidate = None
                # Concordance: if we have a candidate from a prior reading,
                # compare the new raw reading against it. Concordance within
                # the pct-aware band (`_concordance_tolerance`, sized so a
                # one-step vendor quantization jump at low pct still
                # concorded) promotes the more conservative of the pair to
                # the prior; disagreement replaces the candidate without
                # touching the prior (the prior stays authoritative until
                # proven wrong by a concordant pair, not a singleton).
                #
                # Adjacent whole-percent pairs have rel_diff *mathematically*
                # equal to the tolerance (both are 1/pct_new) — the comparison
                # would otherwise be decided by a single floating-point ULP.
                # Verified by sweep at every adjacent transition: 3→4 failed
                # 36/54 random-consumed trials, 6→7 54/54, 9→10 37/54. The
                # `* (1 + CONCORDANCE_EPS)` widens the band by ~1e-9 relative,
                # far above ULP noise, so adjacency concordance is
                # deterministic. Multi-step jumps still reject because their
                # rel_diff exceeds the cap by orders of magnitude.
                if isinstance(candidate, float) and candidate > 0:
                    rel_diff = (abs(new_estimate - candidate)
                                / max(new_estimate, candidate))
                    if rel_diff <= _concordance_tolerance(pct) * (1.0 + CONCORDANCE_EPS):
                        prior = min(candidate, new_estimate)
                        candidate = None
                    else:
                        candidate = new_estimate
                else:
                    candidate = new_estimate
                # Clamp against the confirmed prior. First reading of the
                # window has no prior yet, so the raw reading is taken at
                # face value — only the *second* concordant reading can
                # harden an estimate into a clamp value.
                if isinstance(prior, float) and prior > 0:
                    allowance = min(new_estimate, prior)
                else:
                    allowance = new_estimate
                basis = "estimated"
                await self.redis.hset(
                    K_WINDOW.format(plan=plan.key, window=q.label),
                    mapping={
                        est_key: prior if isinstance(prior, float) else "",
                        "estimated_allowance_reset_at": new_reset,
                        "estimated_allowance_bucket": new_bucket,
                        "estimated_allowance_candidate": (
                            candidate if isinstance(candidate, float) else ""),
                    },
                )
            else:
                basis = "unknown"

        dl, is_final = window_deadline(q.period, plan.expires, now)
        start, _ = period_bounds(q.period, now)
        total_s = max(1.0, (dl - start).total_seconds())
        remaining_s = max(0.0, (dl - now).total_seconds())
        elapsed_frac = 1.0 - remaining_s / total_s

        w = {
            "window": q.label, "role": q.role, "period": q.period,
            "allowance": allowance, "basis": basis, "consumed": consumed,
            "deadline": dl.timestamp(), "is_final_window": is_final,
            "elapsed_frac": round(elapsed_frac, 4), "remaining_seconds": remaining_s,
            "total_seconds": total_s, "consumed_frac": None, "pace_line": None,
            "ahead_by": None, "allowed_rate": None, "spent": False,
        }
        if not allowance or allowance <= 0 or remaining_s <= 0:
            return w

        remaining = allowance - consumed
        w["consumed_frac"] = round(consumed / allowance, 4)
        if remaining <= 0:
            w.update(spent=True, allowed_rate=0.0, pace_line=allowance,
                     ahead_by=consumed - allowance)
            return w

        # The target window is aimed slightly hot so it finishes at ~100%.
        # A constraint window gets no overshoot — overshooting it is precisely
        # what we are trying to avoid.
        overshoot = (1.0 + self.settings.pacing.overshoot) if q.is_target else 1.0
        w["pace_line"] = allowance * min(1.0, elapsed_frac * overshoot)
        w["ahead_by"] = consumed - w["pace_line"]
        w["allowed_rate"] = (remaining / remaining_s) * overshoot
        return w

    async def state(self, plan: Plan, paced: bool = True,
                    now: datetime | None = None) -> dict:
        """Everything the portal needs to explain the pacing decision.

        With several windows the rule is: **spend at the slowest rate any window
        allows**, and hold entirely when the target window is ahead of its pace
        line or any window is spent. That fills the weekly allowance while never
        overshooting the 5-hour one.
        """
        now = now or datetime.now(timezone.utc)
        windows = [await self._window(plan, q, now) for q in plan.quotas]
        target = next((w for w in windows if w["role"] == "target"), windows[0])
        rate_per_slot = _f(await self.redis.hget(K_PACE.format(plan=plan.key), "rate"))

        out = {
            "active": False, "reason": "", "windows": windows, "target": target,
            "rate_per_slot": rate_per_slot, "desired_slots": None,
            "binding": None, "target_rate": None, "projected_end_frac": None,
            # Flattened target-window fields, so existing callers keep working.
            "allowance": target["allowance"], "basis": target["basis"],
            "consumed": target["consumed"], "consumed_frac": target["consumed_frac"],
            "pace_line": target["pace_line"], "ahead_by": target["ahead_by"],
            "deadline": target["deadline"], "is_final_window": target["is_final_window"],
            "elapsed_frac": target["elapsed_frac"],
            "remaining_seconds": target["remaining_seconds"],
        }

        if not paced:
            out["reason"] = "not a subscription" if not plan.is_subscription else "pacing off"
            return out

        known = [w for w in windows if w["allowed_rate"] is not None]
        if not known:
            out["reason"] = "no allowance known yet"
            return out

        spent = [w for w in known if w["spent"]]
        if spent:
            out.update(active=True, desired_slots=0, binding=spent[0]["window"],
                       reason=f"{spent[0]['window']} allowance spent")
            return out

        # Slowest window wins: respect the 5-hour limit while filling weekly.
        binding = min(known, key=lambda w: w["allowed_rate"])
        out["binding"] = binding["window"]
        out["target_rate"] = binding["allowed_rate"]

        # Holding is driven by the window we are trying to *fill*. A constraint
        # window running ahead of its own line is fine — the rate cap handles it.
        if target["ahead_by"] is not None and target["ahead_by"] > 0:
            out.update(active=True, desired_slots=0,
                       reason=f"ahead of pace on {target['window']}, holding")
            return out

        if rate_per_slot <= 0:
            out.update(active=True, reason="measuring per-slot rate")
            return out

        if target["allowance"]:
            # Where in the window this burn rate would exhaust the target.
            burn = rate_per_slot * max(1, self.settings.pacing.min_slots)
            secs_to_empty = (target["allowance"] - target["consumed"]) / burn
            out["projected_end_frac"] = round(
                target["elapsed_frac"] + secs_to_empty / target["total_seconds"], 4)

        note = "" if binding is target else f" (capped by {binding['window']})"
        out.update(active=True, desired_slots=binding["allowed_rate"] / rate_per_slot,
                   reason=f"pacing {target['window']}{note}")
        return out

    async def desired_slots(self, plan: Plan, paced: bool = True,
                            now: datetime | None = None) -> tuple[int | None, str, dict]:
        st = await self.state(plan, paced, now)
        if not st["active"] or st["desired_slots"] is None:
            return None, st["reason"], st

        slots = st["desired_slots"]
        if slots <= 0:
            # Closed by the pace line (or spent). Don't smooth this: holding
            # must take effect immediately or we sail past the line.
            return 0, st["reason"], st

        # Smooth upward moves so one unusually fast request cannot open the
        # floodgates, but keep a floor so an open plan is always usable.
        key = K_PACE.format(plan=plan.key)
        prior = _f(await self.redis.hget(key, "slots"))
        smoothed = slots if prior <= 0 else EWMA_ALPHA * slots + (1 - EWMA_ALPHA) * prior
        await self.redis.hset(key, mapping={"slots": smoothed})
        return max(self.settings.pacing.min_slots, int(round(smoothed))), st["reason"], st


class CapacityPolicy:
    """Composes the two controllers into one number the picker can use."""

    def __init__(self, redis: Redis, settings: Settings, ledger: Ledger):
        self.redis = redis
        self.settings = settings
        self.learner = ConcurrencyLearner(redis, settings)
        self.pacer = Pacer(redis, settings, ledger)
        # Held directly as well: the picker asks whether a plan's target window
        # is reported spent, which is a ledger question rather than a pacing one.
        self.ledger = ledger

    async def pacing_enabled(self) -> bool:
        """plans.yaml sets the default; the portal can flip it at runtime."""
        raw = await self.redis.get(K_SWITCH)
        if raw is None:
            return self.settings.pacing.enabled
        raw = raw.decode() if isinstance(raw, bytes) else raw
        return str(raw) == "1"

    async def set_pacing(self, enabled: bool | None) -> bool:
        """None clears the override and returns to the configured default."""
        if enabled is None:
            await self.redis.delete(K_SWITCH)
        else:
            await self.redis.set(K_SWITCH, "1" if enabled else "0")
        return await self.pacing_enabled()

    async def plan_is_paced(self, plan: Plan) -> bool:
        if not await self.pacing_enabled():
            return False
        if plan.pacing is not None:
            return bool(plan.pacing)
        return plan.is_subscription or (plan.metered and self.settings.pacing.include_metered)

    async def effective(self, plan: Plan, now: datetime | None = None) -> Capacity:
        learned, source = await self.learner.effective(plan)
        cap = Capacity(cap=learned, reason=source, learned=learned,
                       configured=plan.configured_parallel)

        if not await self.plan_is_paced(plan):
            return self._apply_gate_headroom(plan, cap)

        paced, reason, _ = await self.pacer.desired_slots(plan, True, now)
        if paced is None:
            cap.reason = f"{source} (pacing: {reason})"
            return self._apply_gate_headroom(plan, cap)

        cap.paced = paced
        # Pacing can only ever *reduce* concurrency below what the provider
        # tolerates; it must never talk us into exceeding a learned limit.
        cap.cap = min(learned, paced)
        cap.reason = f"paced {paced} of {learned} ({reason})"
        return self._apply_gate_headroom(plan, cap)

    def _apply_gate_headroom(self, plan: Plan, cap: Capacity) -> Capacity:
        """For CLI-backed plans, keep one slot of headroom below the physical gate.

        The sidecar gate reads `max_parallel` from the same plans.yaml and
        stays at the configured limit (it is the physical truth — the vendor
        CLI is the one enforcing it). The gateway's slot table, by contrast,
        is the one that decides whether SwitchYard lets a request through. If
        the two caps agree exactly, an ordinary claim/release race — or a
        release the gateway makes when the client aborts while the sidecar's
        CLI turn is still finishing — briefly sees the gateway count the slot
        as free a moment before the sidecar has, and the sidecar responds
        "sidecar at capacity" with HTTP 429. The headroom slot absorbs that
        race instead of being usable capacity, so the sidecar gate is no
        longer the limiting factor: the gateway slot table is, and it cannot
        be racing the sidecar.

        API plans are unaffected: their provider refuses on its own, the
        learner backs off, and the headroom slot would just sit unused.
        """
        if not plan.is_cli_backed:
            return cap
        headroom = max(0, self.settings.gate_headroom_slots)
        if headroom == 0:
            return cap
        physical = plan.max_parallel
        # Floor at 1 so a single-connection plan never collapses to 0 — that
        # would lock the plan while the operator still owns one usable slot.
        capped = min(cap.cap, max(1, physical - headroom))
        if capped == cap.cap:
            return cap
        cap.cap = capped
        cap.reason = f"{cap.reason} + gate headroom {headroom}"
        return cap

    async def pace_state(self, plan: Plan, now: datetime | None = None) -> dict:
        """Pacing state for the portal, with the on/off resolution applied."""
        return await self.pacer.state(plan, await self.plan_is_paced(plan), now)

    async def tail_enabled(self) -> bool:
        """In pacing mode the tail is off: narrowing is the point, and falling
        back to a local model would hide the fact that we are ahead of budget."""
        if not self.settings.pacing.disable_tail:
            return True
        return not await self.pacing_enabled()
