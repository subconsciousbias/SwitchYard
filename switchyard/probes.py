"""Quota probes for endpoints that only accept a browser session.

MiniMax exposes exact Token Plan headroom at `/coding_plan/remains`, but the
endpoint rejects API keys — with a key it answers "cookie is missing, log in
again" (1004). There is no documented API-key route, so the only way to read the
real number is to borrow a browser session.

So: the portal asks you to paste the session cookie once, stores it in Redis,
and a poller uses it to keep real headroom on the board. When the cookie
expires the probe flips to `needs_reauth` and stops trying until you paste a
fresh one — it never silently falls back to guessing.

Trust note, stated plainly because it matters: a session cookie is as powerful
as being logged in. Anyone holding it can act as you on that account, including
billing. It is stored in Redis on this host, never logged, never returned by the
API (the portal shows only a fingerprint), and revoking it is as simple as
logging out of MiniMax, which invalidates the session.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

import httpx
from redis.asyncio import Redis

from .models import Plan, Probe
from .usage import Ledger

log = logging.getLogger("switchyard.probes")

K_CRED = "sy:cred:{plan}"
K_PROBE = "sy:probe:{plan}"

# The tell that a session cookie has expired rather than something else failing.
REAUTH_MARKERS = ("cookie is missing", "log in again", "not logged in",
                  "unauthorized", "1004")


@dataclass
class WindowReading:
    """One quota window's numbers, as the provider reported them."""
    window: str
    remaining: float | None = None
    total: float | None = None
    reset_at: float | None = None
    used: float | None = None      # when the provider reports spend, not headroom
    used_percent: float | None = None   # when it reports only a percentage


@dataclass
class ProbeResult:
    ok: bool
    detail: str
    remaining: float | None = None
    total: float | None = None
    reset_at: float | None = None
    needs_reauth: bool = False
    raw: str = ""          # first 1500 chars, so you can map unknown fields
    # Every window this response carried. The flat fields above stay as the
    # target window's numbers, so existing callers and the /health shape are
    # unchanged, but a provider reporting a weekly allowance AND a 5-hour burst
    # no longer has half of it discarded.
    windows: list["WindowReading"] = field(default_factory=list)

    def reading(self, name: str) -> "WindowReading | None":
        return next((w for w in self.windows if w.window == name), None)


def _resolve_env(value: str) -> str | None:
    """`os.environ/NAME` -> its value, anything else unchanged.

    Probe headers can carry account-identifying values (OpenCode needs an
    `x-org-id`), and plans.yaml is committed, so they belong in .env like every
    other per-account setting. Returns None when the variable is unset, so the
    probe can say which one rather than sending an empty header and getting an
    opaque rejection.
    """
    text = str(value)
    if not text.startswith("os.environ/"):
        return text
    return os.environ.get(text.split("/", 1)[1]) or None


def first_timestamp(doc: Any, paths: list[str]) -> float | None:
    """A reset time as a unix float, from a number or an ISO-8601 string.

    Consoles disagree: MiniMax sends epoch seconds (or milliseconds), OpenCode
    sends "2026-09-28T00:00:00.000Z". Reading only numbers silently dropped the
    reset time for the second kind, which is what tells the pacer how long the
    window has left.
    """
    for path in paths:
        value = dig(doc, path)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            pass
        try:
            # fromisoformat handles the offset form; Z needs translating first.
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return None


def dig(doc: Any, path: str) -> Any:
    """Follow a dotted path, tolerating lists and missing keys.

    A step may be a list index, or `key=value` to pick the first element of a
    list whose `key` equals `value`. MiniMax returns one entry per model family
    (`model_remains[].model_name` of "general", "video", ...), and depending on
    array order would break the moment they add a family.

    Several conditions can be joined with `&`, because one field is not always
    enough to identify a row: z.ai returns two entries of
    `type=TOKENS_LIMIT` that differ only by `unit` (3 for the 5-hour window, 6
    for the weekly one), so `limits.type=TOKENS_LIMIT&unit=6.percentage` is the
    only way to name the right one.
    """
    cur = doc
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            if "=" in part:
                wanted = [c.partition("=") for c in part.split("&")]
                cur = next((item for item in cur
                            if isinstance(item, dict)
                            and all(str(item.get(k)) == v for k, _, v in wanted)), None)
                if cur is None:
                    return None
            else:
                try:
                    cur = cur[int(part)]
                except (ValueError, IndexError):
                    return None
        else:
            return None
    return cur


def first_percent(doc: Any, paths: list[str]) -> float | None:
    """A percentage, from 12.5 or "12.5%" or "12.5 %".

    Some consoles publish only percentages: MiniMax returns
    `current_weekly_used_percent: "0%"` and -1 for every count, so a probe that
    only reads counts learns nothing from it.
    """
    for path in paths:
        value = dig(doc, path)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().rstrip("%").strip()
        try:
            return float(text)
        except ValueError:
            continue
    return None


def first_number(doc: Any, paths: list[str]) -> float | None:
    """First path that yields something numeric. Field names differ by vendor
    and change without notice, so the config lists candidates."""
    for path in paths:
        val = dig(doc, path)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val.strip())
            except ValueError:
                continue
    return None


class Prober:
    def __init__(self, redis: Redis, ledger: Ledger):
        self.redis = redis
        self.ledger = ledger

    # -- credential storage ------------------------------------------------
    async def set_cookie(self, plan_key: str, cookie: str) -> None:
        cookie = (cookie or "").strip()
        if not cookie:
            raise ValueError("empty cookie")
        await self.redis.hset(K_CRED.format(plan=plan_key), mapping={
            "cookie": cookie, "added_at": time.time(), "fingerprint": _fingerprint(cookie),
        })
        # A fresh cookie clears the re-auth flag so polling resumes.
        await self.redis.hset(K_PROBE.format(plan=plan_key),
                              mapping={"needs_reauth": 0, "last_error": ""})

    async def clear_cookie(self, plan_key: str) -> None:
        await self.redis.delete(K_CRED.format(plan=plan_key))
        await self.redis.hset(K_PROBE.format(plan=plan_key), mapping={"needs_reauth": 1})

    async def _cookie(self, plan_key: str) -> str | None:
        raw = await self.redis.hget(K_CRED.format(plan=plan_key), "cookie")
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else raw

    async def status(self, plan_key: str) -> dict:
        """Safe to show and safe to serialise: never includes the cookie."""
        cred = await self.redis.hgetall(K_CRED.format(plan=plan_key)) or {}
        probe = await self.redis.hgetall(K_PROBE.format(plan=plan_key)) or {}
        cred = {_s(k): _s(v) for k, v in cred.items()}
        probe = {_s(k): _s(v) for k, v in probe.items()}
        return {
            "has_cookie": bool(cred.get("cookie")),
            "fingerprint": cred.get("fingerprint", ""),
            "added_at": _num(cred.get("added_at")),
            "last_ok_at": _num(probe.get("last_ok_at")),
            "last_attempt_at": _num(probe.get("last_attempt_at")),
            "last_error": probe.get("last_error", ""),
            "needs_reauth": probe.get("needs_reauth") in ("1", "1.0", "True"),
            "remaining": _num(probe.get("remaining")),
            "total": _num(probe.get("total")),
        }

    # -- the probe itself --------------------------------------------------
    async def run(self, plan: Plan, *, record: bool = True) -> ProbeResult:
        probe = plan.probe
        if probe is None:
            return ProbeResult(False, "no probe configured for this plan")

        # Configuration is checked BEFORE the cookie: a missing setting is
        # something you can fix immediately, and reporting it second means being
        # sent to fetch a cookie only to hit this straight afterwards.
        headers = {k: _resolve_env(v) for k, v in probe.headers.items()}
        # Name the *variable* to set, not just the header that wanted it: the
        # header name is not what the operator has to go and fill in.
        missing = [f"{str(raw).split('/', 1)[1]} (header {k})"
                   for k, raw in probe.headers.items()
                   if headers.get(k) is None and str(raw).startswith("os.environ/")]
        missing += [k for k, raw in probe.headers.items()
                    if headers.get(k) is None and not str(raw).startswith("os.environ/")]
        if missing:
            return await self._fail(
                plan, f"probe needs {', '.join(missing)} set in .env", True)
        headers = {k: v for k, v in headers.items() if v}

        cookie = await self._cookie(plan.key)
        if probe.kind == "cookie" and not cookie:
            return ProbeResult(False, "no session cookie stored", needs_reauth=True)
        if probe.kind == "cookie":
            headers["Cookie"] = cookie
            # Some consoles also require a matching UA/referer to answer at all.
            headers.setdefault("User-Agent", probe.user_agent)
            if probe.referer:
                headers.setdefault("Referer", probe.referer)

        await self.redis.hset(K_PROBE.format(plan=plan.key),
                              mapping={"last_attempt_at": time.time()})
        try:
            async with httpx.AsyncClient(timeout=probe.timeout_seconds) as client:
                resp = await client.request(probe.method, probe.url, headers=headers)
        except httpx.HTTPError as exc:
            return await self._fail(plan, f"request failed: {exc}", False)

        body = resp.text or ""
        lowered = body.lower()
        if resp.status_code in (401, 403) or any(m in lowered for m in REAUTH_MARKERS):
            return await self._fail(plan, f"session rejected ({resp.status_code})", True,
                                    raw=body[:1500])
        if resp.status_code >= 400:
            return await self._fail(plan, f"HTTP {resp.status_code}", False, raw=body[:1500])

        try:
            doc = resp.json()
        except ValueError:
            return await self._fail(plan, "response was not JSON", False, raw=body[:1500])

        target = probe.window or plan.quota.label
        scale = probe.scale or 1.0
        readings: list[WindowReading] = []
        for name, paths in (probe.windows or {}).items():
            name = name or target
            remaining = first_number(doc, paths.get("remaining", []))
            total = first_number(doc, paths.get("total", []))
            used = first_number(doc, paths.get("used", []))
            used_pct = first_percent(doc, paths.get("used_percent", []))
            if used_pct is None:
                remaining_pct = first_percent(doc, paths.get("remaining_percent", []))
                if remaining_pct is not None:
                    used_pct = max(0.0, 100.0 - remaining_pct)
            # Most consoles publish headroom; some publish spend against a limit
            # instead (OpenCode's Go meters give usedMicroCents/limitMicroCents),
            # so derive the one we need rather than reporting nothing.
            if remaining is None and used is not None and total is not None:
                remaining = max(0.0, total - used)
            reading = WindowReading(
                window=name,
                remaining=None if remaining is None else remaining * scale,
                total=None if total is None else total * scale,
                used=None if used is None else used * scale,
                used_percent=used_pct,
                reset_at=first_timestamp(doc, paths.get("reset_at", [])),
            )
            if reading.reset_at and reading.reset_at > 1e12:      # milliseconds
                reading.reset_at /= 1000.0
            readings.append(reading)

        found = [r for r in readings
                 if r.remaining is not None or r.used_percent is not None]
        if not found:
            # Configured field paths did not match. Hand back the raw body so
            # the portal can show it and you can map the real field names.
            return await self._fail(
                plan, "could not find a remaining value — map the fields from the raw response",
                False, raw=body[:1500])

        # A window the provider did not report is not an error: a plan may
        # publish its weekly allowance but not its burst window. Say which.
        missing = [r.window for r in readings
                   if r.remaining is None and r.used_percent is None]

        if record:
            for r in found:
                if r.remaining is not None:
                    await self.ledger.note_reported(plan.key, r.remaining, r.reset_at,
                                                    window=r.window)
                else:
                    # Percent-only: there is no count to reconcile against our
                    # own tally, so store the percentage the provider states.
                    await self.ledger.note_reported_percent(
                        plan.key, r.used_percent, r.reset_at, window=r.window)

        primary = next((r for r in found if r.window == target), found[0])

        def summarise(r: WindowReading) -> str:
            if r.remaining is not None:
                return f"{r.window}={r.remaining:.0f}"
            return f"{r.window}={r.used_percent:.0f}% used"

        await self.redis.hset(K_PROBE.format(plan=plan.key), mapping={
            "last_ok_at": time.time(), "last_error": "", "needs_reauth": 0,
            "remaining": primary.remaining if primary.remaining is not None else "",
            "total": primary.total if primary.total is not None else "",
            "windows": ",".join(summarise(r) for r in found),
        })
        log.info("probe %s: %s", plan.key, "; ".join(summarise(r) for r in found)
                 + (f" (no data for {', '.join(missing)})" if missing else ""))
        detail = "ok" if not missing else f"ok; no data for {', '.join(missing)}"
        return ProbeResult(True, detail, primary.remaining, primary.total,
                           primary.reset_at, raw=body[:1500], windows=readings)

    async def _fail(self, plan: Plan, detail: str, needs_reauth: bool,
                    raw: str = "") -> ProbeResult:
        await self.redis.hset(K_PROBE.format(plan=plan.key), mapping={
            "last_error": detail, "needs_reauth": 1 if needs_reauth else 0,
        })
        log.warning("probe %s failed: %s", plan.key, detail)
        return ProbeResult(False, detail, needs_reauth=needs_reauth, raw=raw)

    async def due(self, plan: Plan) -> bool:
        """Poll on the configured interval, and never against a dead cookie."""
        if plan.probe is None:
            return False
        st = await self.status(plan.key)
        if not st["has_cookie"] or st["needs_reauth"]:
            return False
        last = st["last_attempt_at"] or 0
        return (time.time() - last) >= plan.probe.interval_seconds


def _fingerprint(cookie: str) -> str:
    """Enough to tell two cookies apart, not enough to use one."""
    return f"{len(cookie)} chars ending {cookie[-4:]}" if len(cookie) > 8 else "short"


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
