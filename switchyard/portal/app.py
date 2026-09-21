"""SwitchYard portal — capacity, quota headroom, and subscription economics.

The point of the quota board: answer "how close am I to running out?" for
every plan on one page, without opening eight vendor dashboards.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import asyncio
import contextlib

from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from redis.asyncio import Redis

from .. import models
from ..picker import Picker
from ..policy import CapacityPolicy
from ..probes import Prober
from ..slots import SlotTable
from ..periods import windows_remaining
from ..usage import Ledger, effective_cost_per_mtok, headroom, model_effective_cost_per_mtok

import logging

BASE = os.path.dirname(__file__)
app = FastAPI(title="SwitchYard")
templates = Jinja2Templates(directory=os.path.join(BASE, "templates"))


def _connect_info(request) -> dict:
    """How to point a client at the gateway, from the portal you are looking at.

    The host comes from this request, so it is right whether you opened the
    portal on localhost or across the network. The key is deliberately NOT read
    or displayed — the page names the variable instead, because a dashboard that
    prints its own master key is one screenshot away from giving it away.
    """
    host = request.url.hostname or "localhost"
    port = os.environ.get("GATEWAY_PORT", "4000")
    reg = state.get("registry")
    return {"base_url": f"http://{host}:{port}/v1",
            "anthropic_url": f"http://{host}:{port}",
            "lanes": list(reg.lanes) if reg else []}


def _ago(ts: float | None) -> str:
    """"2m ago" for a timestamp. A reading with no age is indistinguishable from
    a fresh one, which matters most for a cookie that expires silently."""
    if not ts:
        return "never"
    delta = max(0.0, datetime.now(timezone.utc).timestamp() - float(ts))
    if delta < 90:
        return f"{int(delta)}s ago"
    if delta < 5400:
        return f"{int(delta // 60)}m ago"
    if delta < 172800:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def _interval(seconds) -> str:
    """Poll intervals under a minute rendered as "every 0m" before this."""
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "?"
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m"


def _compact(n) -> str:
    """Token counts that scale: 4.7M beats 4,723,058, and keeps beating it at
    1.2B or 3.1T. K covers the small end; bare numbers below a thousand. The
    K->M boundary is nudged so 999,999 reads 1M, not the false-precision 1000K."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "?"
    sign = "-" if n < 0 else ""
    n = abs(n)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div * 0.9995:
            return sign + f"{n / div:.1f}".rstrip("0").rstrip(".") + suffix
    return sign + f"{n:.0f}"


templates.env.filters["ago"] = _ago
templates.env.filters["interval"] = _interval
templates.env.filters["compact"] = _compact

state: dict = {}


@app.on_event("startup")
async def startup() -> None:
    registry = models.load()
    redis = Redis.from_url(os.environ.get("SWITCHYARD_REDIS_URL", "redis://redis:6379/1"))
    slots = SlotTable(redis, registry.settings.inflight_max_age_seconds)
    ledger = Ledger(redis)
    policy = CapacityPolicy(redis, registry.settings, ledger)
    state.update(
        registry=registry, redis=redis, slots=slots, ledger=ledger, policy=policy,
        picker=Picker(registry, slots, policy), prober=Prober(redis, ledger),
    )
    state["poller"] = asyncio.create_task(_poll_probes())


@app.on_event("shutdown")
async def shutdown() -> None:
    task = state.get("poller")
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _poll_probes() -> None:
    """Keep probe-backed headroom fresh.

    Deliberately quiet: a plan whose cookie has expired is skipped entirely
    (`due()` returns False) until a new one is pasted, so an expired session
    does not turn into a request every minute forever.
    """
    while True:
        try:
            reg, prober = state["registry"], state["prober"]
            for plan in reg.plans.values():
                if await prober.due(plan):
                    await prober.run(plan)
        except asyncio.CancelledError:
            raise
        except Exception:                      # never let the loop die
            logging.getLogger("switchyard.portal").exception("probe poll failed")
        await asyncio.sleep(60)


def _fmt_reset(ts: float | None) -> str:
    if not ts:
        return "—"
    delta = ts - datetime.now(timezone.utc).timestamp()
    if delta <= 0:
        return "now"
    if delta < 3600:
        return f"in {int(delta // 60)}m"
    if delta < 86400:
        return f"in {int(delta // 3600)}h {int((delta % 3600) // 60)}m"
    return f"in {int(delta // 86400)}d"


async def collect_capacity() -> dict:
    reg, picker, policy = state["registry"], state["picker"], state["policy"]
    lanes = [await picker.capacity(k) for k in reg.lanes]

    # A plan can be out of quota while its slots still look free: nothing has
    # refused a request yet, so there is no cooldown, and the capacity board
    # would happily show four idle slots on a subscription with nothing left to
    # spend. Carry each plan's binding window onto its rows so the board can say
    # so — the quota table already knows, the capacity board did not.
    quota: dict[str, dict] = {}
    for plan in reg.plans.values():
        hr = await headroom(state["ledger"], plan)
        binding = hr.get("binding") or {}
        quota[plan.key] = {"pct_used": binding.get("pct_used"),
                           "window": binding.get("window")}
    for lane in lanes:
        for row in lane["plans"]:
            row["quota"] = quota.get(row["plan"], {})
        lane["exhausted"] = sorted({
            row["plan"] for row in lane["plans"]
            if (row["quota"].get("pct_used") or 0) >= 100 and not row["tail"]})
    return {
        "lanes": lanes,
        "total_available": sum(l["slots_available_now"] for l in lanes),
        "total_in_use": sum(l["slots_in_use"] for l in lanes),
        "pacing": await policy.pacing_enabled(),
        "pacing_configured": reg.settings.pacing.enabled,
        "learning": reg.settings.concurrency_learning.enabled,
    }


async def collect_plans() -> list[dict]:
    """One row per PLAN, because quota, cost and connection limits are the
    plan's. Each row lists the models it serves, which is what lanes name."""
    reg, ledger, slots = state["registry"], state["ledger"], state["slots"]
    policy = state["policy"]
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    rows = []
    for plan in reg.plans.values():
        hr = await headroom(ledger, plan)
        # Each window has its own reset time (5h vs weekly vs monthly). The
        # template needs them pre-formatted so the per-bar caption does not
        # have to know about the `_fmt_reset` rule elsewhere on the board.
        for w in hr["windows"]:
            w["reset_human"] = _fmt_reset(w.get("reset_at"))
        burn = await ledger.burn_rate(plan)
        series = await ledger.daily_series(plan.key, days=31)
        month_tokens = sum(
            d["prompt_tokens"] + d["completion_tokens"] for d in series if d["day"].startswith(month)
        )
        month_cost = sum(d["cost"] for d in series if d["day"].startswith(month))
        cooled, ttl, reason = await slots.cooldown_state(plan.key)
        in_flight = await slots.in_flight(plan.key)
        facts = await ledger.quota_facts(plan.key)
        capacity = await policy.effective(plan)
        probe = await state["prober"].status(plan.key) if plan.probe else None
        # Models are what lanes name; the plan is what owns the limits. Named
        # `model_rows` rather than `models`, which is the imported module.
        model_refs = [m.ref for m in plan.models.values()]
        model_overview = await ledger.model_overview(
            plan.key, model_refs, month
        ) if model_refs else {}
        model_rows = []
        for m in plan.models.values():
            overview = model_overview.get(m.ref, {})
            model_rows.append({
                "key": m.key,
                "ref": m.ref,
                "label": m.display,
                "provider_model": m.model,
                "enabled": m.enabled,
                "cap": plan.cap_for(m),
                "narrowed": m.max_parallel is not None,
                "context_window": m.context_window,
                "lanes": reg.lanes_using(m),
                "burn": overview.get("burn") or {"cost_per_hour": 0.0,
                                                 "tokens_per_hour": 0.0},
                "month_tokens": overview.get("month_tokens", 0.0),
                "month_cost": overview.get("month_cost", 0.0),
                "eff_cost": model_effective_cost_per_mtok(
                    plan,
                    overview.get("month_tokens", 0.0),
                    overview.get("month_cost", 0.0),
                    month_tokens,
                ),
            })
        pace = await policy.pace_state(plan) if await policy.plan_is_paced(plan) else None

        alerting = []
        if hr.get("pct_used") is not None and hr["pct_used"] >= 80:
            alerting.append(f"{hr['pct_used']:.0f}% of quota used")
        if plan.alert_burn_rate_per_hour and burn["cost_per_hour"] >= plan.alert_burn_rate_per_hour:
            alerting.append(f"burning ${burn['cost_per_hour']:.2f}/hr")
        if plan.days_left is not None and 0 <= plan.days_left <= 21:
            alerting.append(f"expires in {plan.days_left}d — drain it")
        if cooled and reason == "quota_exhausted":
            alerting.append("out of quota")
        if cooled and reason == "auth":
            alerting.append("credential rejected")
        if cooled and reason == "plan_dead":
            alerting.append("subscription over — remove from plans.yaml")
        if pace and pace.get("basis") == "observed":
            alerting.append("pacing on an observed allowance — set it in plans.yaml")
        if pace and not pace.get("allowance"):
            alerting.append("pacing idle: no allowance known")
        if probe and probe["needs_reauth"]:
            alerting.append("quota probe needs a fresh session cookie")
        # A provider refusing us on connection count means max_parallel is set
        # higher than the plan allows. Different fix from a quota wall, so it
        # gets its own warning instead of looking like rate limiting.
        rejections = facts.get("concurrency_rejections")
        if isinstance(rejections, float) and rejections >= 1:
            at_cap = facts.get("concurrency_rejected_at_cap")
            alerting.append(
                f"refused on connection limit {int(rejections)}x"
                + (f" at cap {int(at_cap)} — lower it" if isinstance(at_cap, float) else "")
            )

        # A plan appears in a lane through its models.
        lanes_used_in = sorted({lane for m in plan.models.values()
                                for lane in reg.lanes_using(m)})

        rows.append({
            "plan": plan,
            "headroom": hr,
            "reset_human": _fmt_reset(hr.get("reset_at")),
            "in_flight": in_flight,
            "cooled": cooled,
            "cooldown_remaining": ttl,
            "cooldown_reason": reason,
            "alerting": alerting,
            "lanes": lanes_used_in,
            "series": series[-14:],
            "capacity": capacity,
            "pace": pace,
            "probe": probe,
            "windows": windows_remaining(plan.quota.period, plan.expires),
            "models": model_rows,
            "cli_backed": plan.is_cli_backed,
        })
    return rows


@app.get("/healthz")
async def healthz() -> dict:
    try:
        await state["redis"].ping()
        return {"ok": True, "plans": len(state["registry"].plans)}
    except Exception as exc:  # pragma: no cover
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


@app.get("/api/state")
async def api_state() -> dict:
    """Everything the board shows, for your own client or a CLI to consume."""
    caps = await collect_capacity()
    plans = await collect_plans()
    return {
        "capacity": caps,
        "plans": [
            {
                "key": r["plan"].key, "label": r["plan"].label,
                "models": [
                    {
                        "ref": m["ref"],
                        "burn": m.get("burn"),
                        "month_tokens": m.get("month_tokens"),
                        "month_cost": m.get("month_cost"),
                        "effective_cost_per_mtok": m.get("eff_cost"),
                    }
                    for m in r["models"] if m["enabled"]
                ],
                "monthly_cost": r["plan"].monthly_cost,
                "expires": str(r["plan"].expires) if r["plan"].expires else None,
                "days_left": r["plan"].days_left,
                "cap": r["plan"].max_parallel, "in_flight": r["in_flight"],
                "cooled": r["cooled"], "cooldown_reason": r["cooldown_reason"],
                "quota": r["headroom"],
                "effective_cap": r["capacity"].cap,
                "learned_cap": r["capacity"].learned,
                "cap_reason": r["capacity"].reason,
                "pacing": r["pace"],
                "alerting": r["alerting"],
            }
            for r in plans
        ],
    }


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html",
        {"connect": _connect_info(request),
         "capacity": await collect_capacity(),
         "plans": await collect_plans(), "probes": await collect_probes(),
         "settings": state["registry"].settings},
    )


@app.get("/fragments/capacity")
async def frag_capacity(request: Request):
    return templates.TemplateResponse(
        request, "_capacity.html", {"capacity": await collect_capacity()}
    )


@app.get("/fragments/pacing")
async def frag_pacing(request: Request):
    return templates.TemplateResponse(
        request, "_pacing.html", {"capacity": await collect_capacity()}
    )


@app.get("/fragments/probes")
async def frag_probes(request: Request):
    return templates.TemplateResponse(
        request, "_probes.html", {"probes": await collect_probes()}
    )


def _probe_window(w) -> dict:
    """One window of a probe reading, formatted for the panel.

    Providers report in different units — MiniMax publishes a percentage and no
    counts, OpenCode dollars against a limit — so the row carries a rendered
    string rather than making the template guess.
    """
    if w.used_percent is not None:
        text = f"{w.used_percent:g}% used"
    elif w.remaining is not None and w.total:
        text = f"{w.remaining:,.2f} left of {w.total:,.2f}"
    elif w.remaining is not None:
        text = f"{w.remaining:,.2f} left"
    else:
        text = "no data"
    return {"window": w.window, "text": text,
            "reset_at": w.reset_at, "reset_human": _fmt_reset(w.reset_at),
            "missing": w.remaining is None and w.used_percent is None}


async def collect_probes() -> list[dict]:
    """Plans that need a pasted browser cookie, and only those.

    This panel exists to collect a credential the operator has to go and get.
    A probe reading an API key, a CLI's own records or the OAuth proxy needs
    nothing from anyone, so listing it here is a row that can never be acted
    on — its numbers already appear in the subscription table like any other.
    """
    reg, prober = state["registry"], state["prober"]
    out = []
    for plan in reg.plans.values():
        if plan.probe is None or plan.probe.kind != "cookie":
            continue
        out.append({"plan": plan, "status": await prober.status(plan.key),
                    "last_test": state.get("probe_tests", {}).get(plan.key)})
    return out


@app.post("/admin/probes/{plan_key}/cookie")
async def save_cookie(plan_key: str, request: Request, cookie: str = Form("")):
    """Store a pasted session cookie. Never echoed back, only fingerprinted."""
    plan = state["registry"].plans.get(plan_key)
    if plan is None or plan.probe is None:
        return JSONResponse({"error": "no probe for that plan"}, status_code=404)
    try:
        await state["prober"].set_cookie(plan_key, cookie)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    result = await state["prober"].run(plan)
    state.setdefault("probe_tests", {})[plan_key] = {
        "ok": result.ok, "detail": result.detail, "remaining": result.remaining,
        "total": result.total, "raw": result.raw, "at": datetime.now(timezone.utc).timestamp(),
        # The per-window numbers are what the panel should show; the raw body is
        # only for mapping a renamed field, so it hides behind a disclosure.
        "windows": [_probe_window(w) for w in result.windows],
    }
    return templates.TemplateResponse(
        request, "_probes.html", {"probes": await collect_probes()}
    )


@app.post("/admin/probes/{plan_key}/test")
async def test_probe(plan_key: str, request: Request):
    plan = state["registry"].plans.get(plan_key)
    if plan is None or plan.probe is None:
        return JSONResponse({"error": "no probe for that plan"}, status_code=404)
    result = await state["prober"].run(plan)
    state.setdefault("probe_tests", {})[plan_key] = {
        "ok": result.ok, "detail": result.detail, "remaining": result.remaining,
        "total": result.total, "raw": result.raw, "at": datetime.now(timezone.utc).timestamp(),
        # The per-window numbers are what the panel should show; the raw body is
        # only for mapping a renamed field, so it hides behind a disclosure.
        "windows": [_probe_window(w) for w in result.windows],
    }
    return templates.TemplateResponse(
        request, "_probes.html", {"probes": await collect_probes()}
    )


@app.post("/admin/probes/{plan_key}/forget")
async def forget_cookie(plan_key: str, request: Request):
    await state["prober"].clear_cookie(plan_key)
    state.setdefault("probe_tests", {}).pop(plan_key, None)
    return templates.TemplateResponse(
        request, "_probes.html", {"probes": await collect_probes()}
    )


@app.get("/fragments/plans")
async def frag_plans(request: Request):
    return templates.TemplateResponse(
        request, "_plans.html", {"plans": await collect_plans()}
    )


@app.post("/admin/reload")
async def reload_config() -> dict:
    """Re-read plans.yaml for THIS process: the portal's copy of the registry.

    The gateway hot-swaps policy-only edits itself — it re-reads plans.yaml
    every few seconds and rebuilds its picker when the LiteLLM router surface
    (model strings, api_base, credentials, lanes) is unchanged. A restart is
    only needed when the router cannot follow, and reload.sh detects that via
    the gateway's published router signature and restarts only then.
    """
    registry = models.load()
    policy = CapacityPolicy(state["redis"], registry.settings, state["ledger"])
    state["registry"] = registry
    state["policy"] = policy
    # The policy has to be passed on, or the board silently loses pacing,
    # learned caps and the spent-quota gate the moment anyone hits reload.
    state["picker"] = Picker(registry, state["slots"], policy)
    return {"reloaded": True, "plans": len(registry.plans),
            "lanes": list(registry.lanes),
            "note": ("board reloaded; the gateway picks up policy-only edits "
                     "within seconds of the file change")}


@app.post("/admin/pacing")
async def set_pacing(enabled: str = "toggle") -> dict:
    """Turn pacing mode on or off at runtime.

    `enabled` is on | off | toggle | default (clears the override and returns
    to whatever plans.yaml says).
    """
    policy = state["policy"]
    want: bool | None
    if enabled == "default":
        want = None
    elif enabled == "toggle":
        want = not await policy.pacing_enabled()
    else:
        want = enabled in ("on", "true", "1", "yes")
    now_on = await policy.set_pacing(want)
    return {"pacing": now_on, "configured_default": state["registry"].settings.pacing.enabled}


@app.post("/admin/plans/{plan_key}/uncool")
async def uncool(plan_key: str) -> dict:
    """Re-admit a plan early — e.g. you know the window just reset."""
    await state["slots"].clear_cooldown(plan_key)
    return {"plan": plan_key, "cooled": False}
