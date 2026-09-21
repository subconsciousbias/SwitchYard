"""Switchyard portal — capacity, quota headroom, and subscription economics.

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
from ..usage import Ledger, effective_cost_per_mtok, headroom

import logging

BASE = os.path.dirname(__file__)
app = FastAPI(title="Switchyard")
templates = Jinja2Templates(directory=os.path.join(BASE, "templates"))

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
        model_rows = [{
            "key": m.key,
            "ref": m.ref,
            "label": m.display,
            "provider_model": m.model,
            "enabled": m.enabled,
            "cap": plan.cap_for(m),
            "narrowed": m.max_parallel is not None,
            "context_window": m.context_window,
            "lanes": reg.lanes_using(m),
        } for m in plan.models.values()]
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
            "burn": burn,
            "month_tokens": month_tokens,
            "month_cost": month_cost,
            "eff_cost": effective_cost_per_mtok(plan, month_tokens, month_cost),
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
    """Everything the board shows, for Paperclip or a CLI to consume."""
    caps = await collect_capacity()
    plans = await collect_plans()
    return {
        "capacity": caps,
        "plans": [
            {
                "key": r["plan"].key, "label": r["plan"].label,
                "models": [m["ref"] for m in r["models"] if m["enabled"]],
                "monthly_cost": r["plan"].monthly_cost,
                "expires": str(r["plan"].expires) if r["plan"].expires else None,
                "days_left": r["plan"].days_left,
                "cap": r["plan"].max_parallel, "in_flight": r["in_flight"],
                "cooled": r["cooled"], "cooldown_reason": r["cooldown_reason"],
                "quota": r["headroom"], "burn": r["burn"],
                "effective_cap": r["capacity"].cap,
                "learned_cap": r["capacity"].learned,
                "cap_reason": r["capacity"].reason,
                "pacing": r["pace"],
                "month_tokens": r["month_tokens"], "month_cost": r["month_cost"],
                "effective_cost_per_mtok": r["eff_cost"],
                "alerting": r["alerting"],
            }
            for r in plans
        ],
    }


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html",
        {"capacity": await collect_capacity(),
         "plans": await collect_plans(), "probes": await collect_probes(),
         "settings": state["registry"].settings},
    )


@app.get("/fragments/capacity")
async def frag_capacity(request: Request):
    return templates.TemplateResponse(
        request, "_capacity.html", {"capacity": await collect_capacity()}
    )


@app.get("/fragments/probes")
async def frag_probes(request: Request):
    return templates.TemplateResponse(
        request, "_probes.html", {"probes": await collect_probes()}
    )


async def collect_probes() -> list[dict]:
    """Plans whose real headroom comes from a console endpoint."""
    reg, prober = state["registry"], state["prober"]
    out = []
    for plan in reg.plans.values():
        if plan.probe is None:
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
    registry = models.load()
    state["registry"] = registry
    state["picker"] = Picker(registry, state["slots"])
    return {"reloaded": True, "plans": len(registry.plans), "lanes": list(registry.lanes)}


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
