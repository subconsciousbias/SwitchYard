"""Render every portal template against a synthetic two-window fixture.

Catches Jinja errors and layout regressions without needing Redis, the gateway
or any provider. Writes /tmp/switchyard-preview.html for eyeballing.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SWITCHYARD_PLANS", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "plans.yaml"))

from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: E402

from switchyard import models                      # noqa: E402
from switchyard.periods import windows_remaining   # noqa: E402
from switchyard.policy import Capacity             # noqa: E402

TPL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "switchyard", "portal", "templates")


def _window(q, frac, allowance, ahead):
    return {"window": q.label, "role": q.role, "period": q.period, "allowance": allowance,
            "basis": "configured", "consumed": allowance * frac, "consumed_frac": frac,
            "pace_line": allowance * 0.5, "ahead_by": ahead, "allowed_rate": 100.0,
            "spent": False, "deadline": time.time() + 8e4, "is_final_window": False,
            "elapsed_frac": 0.5, "remaining_seconds": 8e4, "total_seconds": 6e5,
            "pct_used": frac * 100, "limit": allowance, "kind": q.kind,
            "used_tokens": allowance * frac, "used_cost": 0.0, "reset_at": None,
            "last_exhausted_at": None}


def fixture(reg):
    """Synthetic board state: one row per plan, capacity rows per model."""
    lanes = []
    for key in reg.lanes:
        rows = []
        for i, model in enumerate(reg.lane_members(key)):
            plan = reg.plan_of(model)
            cap = 0 if i == 2 else plan.cap_for(model)
            rows.append({
                "ref": model.ref, "model": model.key, "model_label": model.display,
                "plan": plan.key, "plan_label": plan.label,
                "cap": cap, "cap_configured": plan.cap_for(model),
                "cap_reason": "ahead of pace on weekly, holding" if cap == 0
                              else f"paced {cap} of {plan.max_parallel}",
                "in_flight": min(cap, 1), "cooled": i == 1,
                "cooldown_remaining": 540,
                "cooldown_reason": "quota_exhausted" if i == 1 else "",
                "tail": reg.is_tail(key, model.ref), "days_left": plan.days_left,
                "shares_plan_with": [m.key for m in reg.siblings(model)],
                "cli_backed": plan.is_cli_backed,
                # Exercise all three slot colours: the first member's busy slot
                # is this lane's, a later one's belongs to a sibling lane, so
                # the preview shows the attribution rather than only "used".
                "model_in_flight": min(cap, 1),
                "model_in_flight_here": min(cap, 1) if i == 0 else 0,
                "model_in_flight_elsewhere": min(cap, 1) if i == 3 else 0,
                "model_in_flight_direct": 0,
                "model_cap": model.max_parallel,
                "lanes_sharing": ["judge"] if i == 3 else [],
            })
        lanes.append({
            "lane": key, "label": reg.lanes[key].label,
            "slots_configured": sum(r["cap_configured"] for r in rows),
            "slots_available_now": sum(r["cap"] for r in rows
                                       if not r["cooled"] and not r["tail"]),
            "slots_in_use": sum(r["in_flight"] for r in rows),
            "slots_in_use_here": sum(r["model_in_flight_here"] for r in rows),
            "slots_in_use_elsewhere": sum(r["model_in_flight_elsewhere"] for r in rows),
            "tail_only": False, "plans": rows,
        })

    rows = []
    for plan in reg.plans.values():
        ws = [_window(q, 0.95 if q.role == "constraint" else 0.30,
                      2e6 if q.role == "constraint" else 4e7,
                      8e5 if q.role == "constraint" else -4e6) for q in plan.quotas]
        tgt = next((w for w in ws if w["role"] == "target"), ws[0])
        binding = max(ws, key=lambda w: w["pct_used"])
        rows.append({
            "plan": plan,
            "headroom": {**tgt, "windows": ws, "binding": binding,
                         "binding_is_target": binding is tgt},
            "reset_human": "in 22h",
            "burn": {"cost_per_hour": 21.4 if plan.metered else 0.0,
                     "tokens_per_hour": 5e5},
            "month_tokens": 1.2e7, "month_cost": 12.5 if plan.metered else 0.0,
            "eff_cost": 16.6 if plan.monthly_cost else None, "in_flight": 1,
            "cooled": False, "cooldown_remaining": 0, "cooldown_reason": "",
            "alerting": ["95% of quota used"] if plan.is_subscription else [],
            "lanes": sorted({l for m in plan.models.values() for l in reg.lanes_using(m)}),
            "series": [{"day": f"2026-09-{d:02d}", "prompt_tokens": d * 1e6,
                        "completion_tokens": 0, "requests": d, "cost": 0.0,
                        "failures": 0} for d in range(1, 15)],
            "capacity": Capacity(cap=1, reason="paced 1 of 4 (capped by 5h)",
                                 learned=plan.max_parallel,
                                 configured=plan.configured_parallel),
            "pace": ({"active": True, "reason": "pacing weekly (capped by 5h)",
                      "windows": ws, "target": tgt, "binding": binding["window"],
                      "ahead_by": tgt["ahead_by"], "pace_line": tgt["pace_line"],
                      "is_final_window": False, "allowance": tgt["allowance"],
                      "basis": "configured", "elapsed_frac": 0.5,
                      "target_rate": 100.0, "rate_per_slot": 200.0,
                      "consumed": tgt["consumed"], "deadline": tgt["deadline"],
                      "remaining_seconds": 8e4, "consumed_frac": 0.3,
                      "projected_end_frac": 0.7} if plan.is_subscription else None),
            "windows": windows_remaining(plan.quota.period, plan.expires),
            "models": [{"key": m.key, "ref": m.ref, "label": m.display,
                        "provider_model": m.model, "enabled": m.enabled,
                        "cap": plan.cap_for(m), "narrowed": m.max_parallel is not None,
                        "context_window": m.context_window,
                        "lanes": reg.lanes_using(m)} for m in plan.models.values()],
            "cli_backed": plan.is_cli_backed,
            "probe": None,
        })

    capacity = {"lanes": lanes,
                "total_available": sum(l["slots_available_now"] for l in lanes),
                "total_in_use": sum(l["slots_in_use"] for l in lanes),
                "pacing": True, "pacing_configured": False, "learning": True}
    return capacity, rows


def probe_fixture(reg):
    """Cookie-needing plans, one healthy and one needing re-auth.

    Mirrors collect_probes(): only `kind: cookie` probes appear on this panel,
    because it exists to collect a credential someone has to go and fetch. A
    preview listing every probe would not be a preview of the real page.
    """
    out = []
    cookie_plans = [p for p in reg.plans.values()
                    if p.probe and p.probe.kind == "cookie"]
    for i, plan in enumerate(cookie_plans):
        healthy = i == 0
        out.append({
            "plan": plan,
            "status": {"has_cookie": True, "fingerprint": "412 chars, #9f2a1c3d",
                       "windows": "weekly=12% used,5h=37% used" if healthy else "",
                       "added_at": time.time() - 3600,
                       "last_ok_at": time.time() - 120 if healthy else None,
                       "last_attempt_at": time.time() - 120,
                       "last_error": "" if healthy else "session rejected (401)",
                       "needs_reauth": not healthy,
                       "remaining": 41_200_000.0 if healthy else None,
                       "total": 100_000_000.0 if healthy else None},
            # Both outcomes are rendered: a healthy plan shows its per-window
            # readings with the raw body collapsed, a failed one leads with the
            # error. Percent-only and dollar windows both appear, since the two
            # providers report in different units.
            "last_test": {
                "ok": True, "detail": "ok",
                "remaining": None, "total": None,
                "windows": [
                    {"window": "5h", "text": "37% used", "reset_at": None,
                     "reset_human": "in 2h 10m", "missing": False},
                    {"window": "weekly", "text": "12% used", "reset_at": None,
                     "reset_human": "in 4d", "missing": False},
                ],
                "raw": '{"model_remains":[{"model_name":"general",'
                       '"current_weekly_used_percent":"12%"}]}',
                "at": time.time()} if healthy else {
                "ok": False, "detail": "session rejected (401)",
                "remaining": None, "total": None, "windows": [],
                "raw": '{"base_resp":{"status_code":1004,'
                       '"status_msg":"cookie is missing, log in again"}}',
                "at": time.time()},
        })
    return out


def main() -> int:
    reg = models.load()
    capacity, rows = fixture(reg)
    env = Environment(loader=FileSystemLoader(TPL), undefined=StrictUndefined)
    # Take the filters from the app rather than redefining them: a template
    # using a filter the preview does not know is exactly the regression this
    # is meant to catch, and duplicating them here would hide it.
    from switchyard.portal.app import _ago, _interval
    env.filters["ago"] = _ago
    env.filters["interval"] = _interval
    ctx = {"capacity": capacity, "plans": rows, "settings": reg.settings,
           "probes": probe_fixture(reg), "request": None}
    html = env.get_template("index.html").render(**ctx)
    out = "/tmp/switchyard-preview.html"
    with open(out, "w") as fh:
        fh.write(html)
    for name in ("_capacity.html", "_plans.html", "_probes.html"):
        env.get_template(name).render(**ctx)
    print(f"rendered index.html + fragments ({len(html)} bytes) -> {out}")
    print(f"  {len(rows)} plans, {sum(len(r['models']) for r in rows)} models, "
          f"{len(capacity['lanes'])} lanes, "
          f"{sum(len(r['headroom']['windows']) for r in rows)} quota windows, "
          f"{len(ctx['probes'])} probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
