"""Switchyard portal — capacity, quota headroom, and subscription economics.

The point of the quota board: answer "how close am I to running out?" for
every plan on one page, without opening eight vendor dashboards.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from redis.asyncio import Redis

from .. import models
from ..picker import Picker
from ..slots import SlotTable
from ..usage import Ledger, effective_cost_per_mtok, headroom

BASE = os.path.dirname(__file__)
app = FastAPI(title="Switchyard")
templates = Jinja2Templates(directory=os.path.join(BASE, "templates"))

state: dict = {}


@app.on_event("startup")
async def startup() -> None:
    registry = models.load()
    redis = Redis.from_url(os.environ.get("SWITCHYARD_REDIS_URL", "redis://redis:6379/1"))
    slots = SlotTable(redis, registry.settings.inflight_max_age_seconds)
    state.update(
        registry=registry, redis=redis, slots=slots,
        picker=Picker(registry, slots), ledger=Ledger(redis),
    )


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
    reg, picker = state["registry"], state["picker"]
    lanes = [await picker.capacity(k) for k in reg.lanes]
    return {
        "lanes": lanes,
        "total_available": sum(l["slots_available_now"] for l in lanes),
        "total_in_use": sum(l["slots_in_use"] for l in lanes),
    }


async def collect_plans() -> list[dict]:
    reg, ledger, slots = state["registry"], state["ledger"], state["slots"]
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

        lanes_used_in = [k for k, l in reg.lanes.items()
                         if plan.key in l.order or plan.key in l.tail]

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
                "monthly_cost": r["plan"].monthly_cost,
                "expires": str(r["plan"].expires) if r["plan"].expires else None,
                "days_left": r["plan"].days_left,
                "cap": r["plan"].max_parallel, "in_flight": r["in_flight"],
                "cooled": r["cooled"], "cooldown_reason": r["cooldown_reason"],
                "quota": r["headroom"], "burn": r["burn"],
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
        "index.html",
        {"request": request, "capacity": await collect_capacity(),
         "plans": await collect_plans(), "settings": state["registry"].settings},
    )


@app.get("/fragments/capacity")
async def frag_capacity(request: Request):
    return templates.TemplateResponse(
        "_capacity.html", {"request": request, "capacity": await collect_capacity()}
    )


@app.get("/fragments/plans")
async def frag_plans(request: Request):
    return templates.TemplateResponse(
        "_plans.html", {"request": request, "plans": await collect_plans()}
    )


@app.post("/admin/reload")
async def reload_config() -> dict:
    registry = models.load()
    state["registry"] = registry
    state["picker"] = Picker(registry, state["slots"])
    return {"reloaded": True, "plans": len(registry.plans), "lanes": list(registry.lanes)}


@app.post("/admin/plans/{plan_key}/uncool")
async def uncool(plan_key: str) -> dict:
    """Re-admit a plan early — e.g. you know the window just reset."""
    await state["slots"].clear_cooldown(plan_key)
    return {"plan": plan_key, "cooled": False}
