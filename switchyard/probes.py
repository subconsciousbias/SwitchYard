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
import time
from dataclasses import dataclass
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
class ProbeResult:
    ok: bool
    detail: str
    remaining: float | None = None
    total: float | None = None
    reset_at: float | None = None
    needs_reauth: bool = False
    raw: str = ""          # first 1500 chars, so you can map unknown fields


def dig(doc: Any, path: str) -> Any:
    """Follow a dotted path, tolerating lists and missing keys."""
    cur = doc
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


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

        cookie = await self._cookie(plan.key)
        if probe.kind == "cookie" and not cookie:
            return ProbeResult(False, "no session cookie stored", needs_reauth=True)

        headers = dict(probe.headers)
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

        remaining = first_number(doc, probe.fields.get("remaining", []))
        total = first_number(doc, probe.fields.get("total", []))
        reset_at = first_number(doc, probe.fields.get("reset_at", []))
        if reset_at and reset_at > 1e12:       # milliseconds
            reset_at /= 1000.0

        if remaining is None:
            # Configured field paths did not match. Hand back the raw body so
            # the portal can show it and you can map the real field names.
            return await self._fail(
                plan, "could not find a remaining value — map the fields from the raw response",
                False, raw=body[:1500])

        if record:
            await self.ledger.note_reported(plan.key, remaining, reset_at,
                                            window=probe.window or plan.quota.label)
        await self.redis.hset(K_PROBE.format(plan=plan.key), mapping={
            "last_ok_at": time.time(), "last_error": "", "needs_reauth": 0,
            "remaining": remaining, "total": total if total is not None else "",
        })
        log.info("probe %s: remaining=%s total=%s", plan.key, remaining, total)
        return ProbeResult(True, "ok", remaining, total, reset_at, raw=body[:1500])

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
