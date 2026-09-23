"""Render every portal template against a synthetic two-window fixture.

Catches Jinja errors and layout regressions without needing Redis, the gateway
or any provider. Writes <tempdir>/switchyard-preview.html for eyeballing.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()

from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: E402

from switchyard import models                      # noqa: E402
from switchyard.periods import windows_remaining   # noqa: E402
from switchyard.policy import Capacity             # noqa: E402

TPL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "switchyard", "portal", "templates")


def _window(q, frac, allowance, ahead):
    # Two distinct reset horizons so the preview exercises per-window captions:
    # the constraint window (5h-style) is close, the target (weekly) is days
    # out. collect_plans formats this via _fmt_reset; do the same here so the
    # preview shows what the real page will.
    from switchyard.portal.app import _fmt_reset
    if q.role == "constraint":
        reset_at = time.time() + 2 * 3600
    else:
        reset_at = time.time() + 4 * 86400
    return {"window": q.label, "role": q.role, "period": q.period, "allowance": allowance,
            "basis": "configured", "consumed": allowance * frac, "consumed_frac": frac,
            "pace_line": allowance * 0.5, "ahead_by": ahead, "allowed_rate": 100.0,
            "spent": False, "deadline": time.time() + 8e4, "is_final_window": False,
            "elapsed_frac": 0.5, "remaining_seconds": 8e4, "total_seconds": 6e5,
            "pct_used": frac * 100, "limit": allowance, "kind": q.kind,
            "used_tokens": allowance * frac, "used_cost": 0.0,
            "reset_at": reset_at, "reset_human": _fmt_reset(reset_at),
            "last_exhausted_at": None}


def fixture(reg):
    """Synthetic board state: one row per plan, capacity rows per model."""
    lanes = []
    for key in reg.lanes:
        rows = []
        for i, model in enumerate(reg.lane_members(key)):
            plan = reg.plan_of(model)
            cap = 0 if i == 2 else plan.cap_for(model)
            # cap_configured is the PLAN's max_parallel, not the model-narrowed
            # one — same as picker.capacity() reports. Without the gap, no row
            # exercises the model-limit withheld square that a model of 1 on a
            # plan of 2 (e.g. local-box) draws.
            rows.append({
                "ref": model.ref, "model": model.key, "model_label": model.display,
                "plan": plan.key, "plan_label": plan.label,
                "cap": cap, "cap_configured": plan.max_parallel,
                "cap_reason": "ahead of pace on weekly, holding" if cap == 0
                              else f"paced {cap} of {plan.max_parallel}",
                "in_flight": min(cap, 1), "cooled": i == 1,
                "cooldown_remaining": 540,
                "cooldown_reason": "quota_exhausted" if i == 1 else "",
                "tail": reg.is_tail(key, model.ref), "days_left": plan.days_left,
                "shares_plan_with": [m.key for m in reg.siblings(model)],
                "cli_backed": plan.is_cli_backed,
                "quota": {"pct_used": 100 if i == 1 else 4, "window": "weekly"},
                # Exercise all three slot colours: the first member's busy slot
                # is this lane's, a later one's belongs to a sibling lane, so
                # the preview shows the attribution rather than only "used".
                "model_in_flight": min(cap, 1),
                "model_in_flight_here": min(cap, 1) if i == 0 else 0,
                "model_in_flight_elsewhere": min(cap, 1) if i == 3 else 0,
                "model_in_flight_direct": 0,
                "model_cap": model.max_parallel,
                "transient_streak": 3 if i == 0 else 0,
                # The picker now threads `streak_alert` onto each row so the
                # template's chip threshold is operator-configurable rather
                # than hardcoded. Mirror it here so the preview stays aligned
                # with the real fragment.
                "streak_alert": reg.settings.transient_breaker.streak_alert,
                "lanes_sharing": ["judge"] if i == 3 else [],
            })
        # Walk `lane_nodes()` to build the render-ready group structure
        # the same way the live `collect_capacity` does, so the shipped
        # example's group-bearing lanes (`forge`, `nest-demo`) show
        # their strategy tags / pointers / rankings in the preview.
        rows_by_ref = {r["ref"]: r for r in rows}
        groups = _build_preview_groups(reg, key, rows_by_ref)
        lanes.append({
            "lane": key, "label": reg.lanes[key].label,
            "slots_configured": sum(r["cap_configured"] for r in rows),
            "slots_available_now": sum(r["cap"] for r in rows
                                       if not r["cooled"] and not r["tail"]),
            "slots_in_use": sum(r["in_flight"] for r in rows),
            "slots_in_use_here": sum(r["model_in_flight_here"] for r in rows),
            "slots_in_use_elsewhere": sum(r["model_in_flight_elsewhere"] for r in rows),
            "tail_only": False, "plans": rows,
            "groups": groups,
            "exhausted": [r["plan"] for r in rows
                          if (r["quota"]["pct_used"] or 0) >= 100 and not r["tail"]],
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
            "in_flight": 1,
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
            "models": [{
                "key": m.key, "ref": m.ref, "label": m.display,
                "provider_model": m.model, "enabled": m.enabled,
                "cap": plan.cap_for(m), "narrowed": m.max_parallel is not None,
                "context_window": m.context_window,
                "lanes": reg.lanes_using(m),
                # Economics live at model level: a subscription allocates its fee
                # pro-rata by token share; metered plans show their own spend.
                "burn": {"cost_per_hour": 0.42 if m.enabled else 0.0,
                         "tokens_per_hour": 5e5 if m.enabled else 0.0},
                "month_tokens": 1.23e7 if m.enabled else 0.0,
                "month_cost": 8.10 if m.enabled else 0.0,
                "n_sessions": 150 if m.enabled else 0,
                "eff_cost": 0.66 if m.enabled and plan.monthly_cost else None,
                # Mirror the live shape: $/session populated on enabled rows,
                # None where the threshold or fee would gate it. The preview
                # template uses m.get('eff_cost_session') so an absent key also
                # renders cleanly.
                "eff_cost_session": 0.73 if m.enabled and plan.monthly_cost
                                    else None,
            } for m in plan.models.values()],
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

    `last_ok_hhmm` mirrors the field collect_probes() threads onto the
    status dict so the panel can render "last good HH:MM" on a stale row.
    The preview matches the real shape so the template path that uses it
    is exercised.
    """
    out = []
    cookie_plans = [p for p in reg.plans.values()
                    if p.probe and p.probe.kind == "cookie"]
    for i, plan in enumerate(cookie_plans):
        healthy = i == 0
        last_ok = time.time() - 120 if healthy else None
        last_ok_hhmm = (time.strftime("%H:%M", time.gmtime(last_ok))
                        if last_ok else "—")
        out.append({
            "plan": plan,
            "status": {"has_cookie": True, "fingerprint": "412 chars, #9f2a1c3d",
                       "windows": "weekly=12% used,5h=37% used" if healthy else "",
                       "added_at": time.time() - 3600,
                       "last_ok_at": last_ok,
                       "last_ok_hhmm": last_ok_hhmm,
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


def _build_preview_groups(reg, lane_key, rows_by_ref):
    """Walk the lane's parsed body into the render-ready group list.

    Mirrors `build_groups` from `switchyard.portal.groups`, but uses the
    fixture's rows_by_ref and skips the Redis lookup (no rotation pointer,
    no ranking snapshot -- those would require a live ledger). The shape
    matches the live template's expectations so the preview exercises the
    same rendering code paths.
    """
    from switchyard.models import Group

    def flatten(node, depth, out):
        if isinstance(node, Group):
            refs = _leaf_refs(node)
            header = {
                "kind": "group",
                "strategy": node.strategy,
                "gid": node.gid,
                "members": refs,
                "depth": depth,
            }
            if node.strategy == "round_robin":
                # No Redis in the preview; surface the pointer at a
                # deterministic value so the template exercises the path.
                header["pointer"] = 0
                header["next_index"] = 0
            elif node.strategy == "weighted":
                header["weights"] = dict(node.weights or {})
            elif node.strategy in ("perishable", "lowest_utilization"):
                # No ranking hash in the preview; show the declared order.
                header["ranking"] = None
            out.append(header)
            if node.weights is None:
                for member in node.members:
                    flatten(member, depth + 1, out)
            return
        row = rows_by_ref.get(node)
        if row is not None:
            out.append({"kind": "ref", "row": row, "depth": depth})

    out: list = []
    for node in reg.lane_nodes().get(lane_key, []):
        flatten(node, 0, out)
    return out


def _leaf_refs(group):
    """Local copy of `Group` leaf walking -- avoids the portal.groups import
    chain which requires a live Redis for the rotation-pointer reads."""
    from switchyard.models import Group
    out: list = []
    def walk(node):
        if isinstance(node, Group):
            if node.weights is not None:
                out.extend(node.weights.keys())
                return
            for m in node.members:
                walk(m)
            return
        out.append(node)
    if group.weights is not None:
        return list(group.weights.keys())
    for m in group.members:
        walk(m)
    return out


def main() -> int:
    reg = models.load()
    capacity, rows = fixture(reg)
    env = Environment(loader=FileSystemLoader(TPL), undefined=StrictUndefined)
    # Take the filters from the app rather than redefining them: a template
    # using a filter the preview does not know is exactly the regression this
    # is meant to catch, and duplicating them here would hide it.
    from switchyard.portal.app import _ago, _interval, _compact
    env.filters["ago"] = _ago
    env.filters["interval"] = _interval
    env.filters["compact"] = _compact
    ctx = {"connect": {"base_url": "http://switchyard.local:4000/v1",
                       "anthropic_url": "http://switchyard.local:4000",
                       "lanes": list(reg.lanes)},
           "capacity": capacity, "plans": rows, "settings": reg.settings,
           "probes": probe_fixture(reg), "request": None}
    html = env.get_template("index.html").render(**ctx)
    out = os.path.join(tempfile.gettempdir(), "switchyard-preview.html")
    with open(out, "w") as fh:
        fh.write(html)
    for name in ("_capacity.html", "_plans.html", "_probes.html"):
        env.get_template(name).render(**ctx)
    # Also render the capacity fragment against a SYNTHETIC group-bearing
    # lane so the preview exercises the group rendering shape (strategy tag,
    # weights/pointer, nested indentation). The shipped example has no
    # groups, so without this the template's group path would go untested.
    group_html = render_group_preview(env, capacity)
    group_out = os.path.join(tempfile.gettempdir(),
                             "switchyard-preview-groups.html")
    with open(group_out, "w") as fh:
        fh.write(group_html)
    print(f"rendered index.html + fragments ({len(html)} bytes) -> {out}")
    print(f"  flat groups preview ({len(group_html)} bytes) -> {group_out}")
    print(f"  {len(rows)} plans, {sum(len(r['models']) for r in rows)} models, "
          f"{len(capacity['lanes'])} lanes, "
          f"{sum(len(r['headroom']['windows']) for r in rows)} quota windows, "
          f"{len(ctx['probes'])} probes")
    # A flat config renders zero group headers (regression bar); a
    # group-bearing config must show at least one. Print the relevant
    # excerpt so the operator can eyeball the layout without opening the
    # generated HTML.
    group_headers = group_html.count('class="group-head"')
    print(f"  group-bearing preview: {group_headers} group header row(s)")
    return 0


def render_group_preview(env, flat_capacity):
    """Render the capacity fragment with a synthetic group-bearing lane.

    The shipped example config is flat, so we build one inline: a
    round_robin group around two existing forge members, then a nested
    weighted group inside another lane. Indentation / strategy tag /
    pointer / weights all show in the rendered HTML; tests assert the
    same strings against the live fragment, and this preview lets the
    operator eyeball the layout.
    """
    rows_by_ref = {}
    for lane in flat_capacity["lanes"]:
        rows_by_ref.update({r["ref"]: r for r in lane["plans"]})
    # Pick forge members for the round_robin (always live in the example).
    rr_members = [rows_by_ref["minimax-ultra/m3"],
                  rows_by_ref["minimax-max/m3"]]
    inner = {
        "kind": "group",
        "strategy": "lowest_utilization",
        "gid": "g_inner_preview",
        "members": ["claude-max/fable", "openai/astra"],
        "depth": 1,
        "ranking": ["claude-max/fable", "openai/astra"],
    }
    outer = {
        "kind": "group",
        "strategy": "round_robin",
        "gid": "g_outer_preview",
        "members": ["claude-max/fable", "openai/astra"],
        "depth": 0,
        "next_index": 0,
        "pointer": 0,
    }
    weighted = {
        "kind": "group",
        "strategy": "weighted",
        "gid": "g_weighted_preview",
        "members": ["grok/grok-4.6", "opencode-go/glm-5.3-flash"],
        "weights": {"grok/grok-4.6": 3, "opencode-go/glm-5.3-flash": 1},
        "depth": 0,
    }
    grouped_lane = {
        "lane": "preview-rr",
        "label": "Preview (round_robin)",
        "slots_configured": 8,
        "slots_available_now": 6,
        "slots_in_use": 1,
        "slots_in_use_here": 1,
        "slots_in_use_elsewhere": 0,
        "tail_only": False,
        "plans": rr_members,
        "groups": [outer, inner,
                   {"kind": "ref", "row": rows_by_ref["claude-max/fable"],
                    "depth": 2},
                   {"kind": "ref", "row": rows_by_ref["openai/astra"],
                    "depth": 2}],
        "exhausted": [],
    }
    weighted_lane = {
        "lane": "preview-w",
        "label": "Preview (weighted)",
        "slots_configured": 4,
        "slots_available_now": 4,
        "slots_in_use": 0,
        "slots_in_use_here": 0,
        "slots_in_use_elsewhere": 0,
        "tail_only": False,
        "plans": [rows_by_ref["grok/grok-4.6"],
                  rows_by_ref["opencode-go/glm-5.3-flash"]],
        "groups": [weighted,
                   {"kind": "ref", "row": rows_by_ref["grok/grok-4.6"],
                    "depth": 1},
                   {"kind": "ref", "row": rows_by_ref["opencode-go/glm-5.3-flash"],
                    "depth": 1}],
        "exhausted": [],
    }
    preview_capacity = {
        "lanes": [grouped_lane, weighted_lane,
                  # A flat-config lane so the regression bar is visible too.
                  flat_capacity["lanes"][0]],
        "total_available": 16,
        "total_in_use": 1,
        "pacing": False, "pacing_configured": False, "learning": True,
    }
    return env.get_template("_capacity.html").render(
        capacity=preview_capacity)


if __name__ == "__main__":
    raise SystemExit(main())
