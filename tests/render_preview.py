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
    lanes = []
    for k in reg.lanes:
        plans = []
        for i, p in enumerate(reg.lane_members(k)):
            plans.append({"plan": p.key, "label": p.label,
                          "cap": 0 if i == 2 else 1, "cap_configured": p.max_parallel,
                          "cap_reason": "ahead of pace on weekly, holding" if i == 2
                          else "paced 1 of 4 (capped by 5h)",
                          "in_flight": 0 if i == 2 else 1, "cooled": i == 1,
                          "cooldown_remaining": 540,
                          "cooldown_reason": "quota_exhausted" if i == 1 else "",
                          "tail": reg.is_tail(k, p.key), "days_left": p.days_left})
        lanes.append({"lane": k, "label": reg.lanes[k].label,
                      "slots_configured": sum(x["cap_configured"] for x in plans),
                      "slots_available_now": sum(x["cap"] for x in plans
                                                 if not x["cooled"] and not x["tail"]),
                      "slots_in_use": sum(x["in_flight"] for x in plans), "plans": plans})

    rows = []
    for p in reg.plans.values():
        ws = [_window(q, 0.95 if q.role == "constraint" else 0.30,
                      2e6 if q.role == "constraint" else 4e7,
                      8e5 if q.role == "constraint" else -4e6) for q in p.quotas]
        tgt = next((w for w in ws if w["role"] == "target"), ws[0])
        binding = max(ws, key=lambda w: w["pct_used"])
        rows.append({
            "plan": p,
            "headroom": {**tgt, "windows": ws, "binding": binding,
                         "binding_is_target": binding is tgt},
            "reset_human": "in 22h",
            "burn": {"cost_per_hour": 21.4 if p.metered else 0.0, "tokens_per_hour": 5e5},
            "month_tokens": 1.2e7, "month_cost": 12.5 if p.metered else 0.0,
            "eff_cost": 16.6 if p.monthly_cost else None, "in_flight": 1,
            "cooled": False, "cooldown_remaining": 0, "cooldown_reason": "",
            "alerting": ["95% of quota used"] if p.is_subscription else [],
            "lanes": [k for k, l in reg.lanes.items() if p.key in l.order or p.key in l.tail],
            "series": [{"day": f"2026-09-{d:02d}", "prompt_tokens": d * 1e6,
                        "completion_tokens": 0, "requests": d, "cost": 0.0,
                        "failures": 0} for d in range(1, 15)],
            "capacity": Capacity(cap=1, reason="paced 1 of 4 (capped by 5h)",
                                 learned=4, configured=p.configured_parallel),
            "pace": ({"active": True, "reason": "pacing weekly (capped by 5h)",
                      "windows": ws, "target": tgt, "binding": binding["window"],
                      "ahead_by": tgt["ahead_by"], "pace_line": tgt["pace_line"],
                      "is_final_window": p.key == "opencode-go",
                      "allowance": tgt["allowance"], "basis": "configured",
                      "elapsed_frac": 0.5, "target_rate": 100.0, "rate_per_slot": 200.0,
                      "consumed": tgt["consumed"], "deadline": tgt["deadline"],
                      "remaining_seconds": 8e4, "consumed_frac": 0.3,
                      "projected_end_frac": 0.7} if p.is_subscription else None),
            "windows": windows_remaining(p.quota.period, p.expires),
        })

    capacity = {"lanes": lanes, "total_available": sum(l["slots_available_now"] for l in lanes),
                "total_in_use": sum(l["slots_in_use"] for l in lanes),
                "pacing": True, "pacing_configured": False, "learning": True}
    return capacity, rows


def probe_fixture(reg):
    """Probe-capable plans, one healthy and one needing re-auth."""
    out = []
    for i, plan in enumerate(p for p in reg.plans.values() if p.probe):
        healthy = i == 0
        out.append({
            "plan": plan,
            "status": {"has_cookie": True, "fingerprint": "412 chars ending 9f2a",
                       "added_at": time.time() - 3600,
                       "last_ok_at": time.time() - 120 if healthy else None,
                       "last_attempt_at": time.time() - 120,
                       "last_error": "" if healthy else "session rejected (401)",
                       "needs_reauth": not healthy,
                       "remaining": 41_200_000.0 if healthy else None,
                       "total": 100_000_000.0 if healthy else None},
            "last_test": None if healthy else {
                "ok": False, "detail": "session rejected (401)",
                "remaining": None, "total": None,
                "raw": '{"base_resp":{"status_code":1004,'
                       '"status_msg":"cookie is missing, log in again"}}',
                "at": time.time()},
        })
    return out


def main() -> int:
    reg = models.load()
    capacity, rows = fixture(reg)
    env = Environment(loader=FileSystemLoader(TPL), undefined=StrictUndefined)
    ctx = {"capacity": capacity, "plans": rows, "settings": reg.settings,
           "probes": probe_fixture(reg), "request": None}
    html = env.get_template("index.html").render(**ctx)
    out = "/tmp/switchyard-preview.html"
    with open(out, "w") as fh:
        fh.write(html)
    for name in ("_capacity.html", "_plans.html", "_probes.html"):
        env.get_template(name).render(**ctx)
    print(f"rendered index.html + fragments ({len(html)} bytes) -> {out}")
    print(f"  {len(rows)} plans, {len(capacity['lanes'])} lanes, "
          f"{sum(len(r['headroom']['windows']) for r in rows)} quota windows, "
          f"{len(ctx['probes'])} probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
