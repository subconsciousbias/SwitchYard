"""Generate the LiteLLM proxy config from config/plans.yaml.

One source of truth: caps, lane order, credentials and expiry all come from
plans.yaml, so there is no second file to keep in sync. Run it at container
start (the entrypoint does) or by hand:

    python -m switchyard.gen_litellm config/plans.yaml config/litellm.generated.yaml
"""
from __future__ import annotations

import sys

import yaml

from .models import load


def build(plans_path: str) -> dict:
    """Emit one LiteLLM deployment per plan/model pairing.

    Credentials and the base URL come from the plan; the model string from the
    model. The per-deployment `max_parallel_requests` is a backstop only —
    Switchyard's slot table is the real gate, keyed by plan so a plan's models
    share it — but it stops a caller who names a deployment directly from
    exceeding the plan's connection limit.
    """
    reg = load(plans_path)

    model_list: list[dict] = []
    for plan in reg.plans.values():
        for model in plan.models.values():
            params: dict = {"model": model.model}
            if plan.api_base:
                params["api_base"] = plan.api_base
            if plan.api_key:
                params["api_key"] = plan.api_key
            params["max_parallel_requests"] = plan.cap_for(model)

            info = {
                "switchyard_plan": plan.key,
                "switchyard_model": model.key,
                "monthly_cost": plan.monthly_cost,
                "expires": str(plan.expires) if plan.expires else None,
                "auth": plan.auth,
                "supports_tools": plan.can_use_tools,
                "enabled": plan.enabled and model.enabled and not plan.expired,
            }
            if model.context_window:
                info["max_input_tokens"] = model.context_window
            model_list.append({
                "model_name": model.deployment,
                "litellm_params": params,
                "model_info": info,
            })

    # Lane aliases exist so /v1/models advertises the lanes and virtual-key model
    # access checks pass. The pre-call hook overwrites the target before routing,
    # so the deployment named here is only a safety default.
    for lane_key in reg.lanes:
        members = reg.lane_members(lane_key)
        if not members:
            continue
        first = members[0]
        plan = reg.plan_of(first)
        params = {"model": first.model}
        if plan.api_base:
            params["api_base"] = plan.api_base
        if plan.api_key:
            params["api_key"] = plan.api_key
        model_list.append({
            "model_name": lane_key,
            "litellm_params": params,
            "model_info": {"switchyard_lane": lane_key, "default_model": first.ref},
        })

    # Mid-call failover. Switchyard picks the entry point; if that call dies in
    # flight, LiteLLM walks the rest of the lane rather than failing the request.
    # The hook's cooldown still removes a dead plan from future picks, so this is
    # belt-and-braces, not the primary mechanism.
    fallbacks: list[dict[str, list[str]]] = []
    for lane_key in reg.lanes:
        members = reg.lane_members(lane_key)
        for i, model in enumerate(members):
            tail = [m.deployment for m in members[i + 1:]]
            if tail:
                fallbacks.append({model.deployment: tail})
        if members:
            fallbacks.append({lane_key: [m.deployment for m in members]})

    # Context-window fallbacks: when a prompt is too big for the model we picked,
    # hand it to the largest-context model we have. Only models that declare
    # `context_window` take part, so an unknown window never silently becomes a
    # wrong routing decision.
    sized = sorted(
        (m for m in reg.models.values()
         if m.context_window and m.enabled
         and reg.plan_of(m).enabled and not reg.plan_of(m).expired),
        key=lambda m: m.context_window, reverse=True)
    context_fallbacks: list[dict[str, list[str]]] = []
    for model in reg.models.values():
        plan = reg.plan_of(model)
        if not (plan.enabled and model.enabled) or plan.expired or not model.context_window:
            continue
        bigger = [m.deployment for m in sized
                  if m.ref != model.ref and m.context_window > model.context_window][:2]
        if bigger:
            context_fallbacks.append({model.deployment: bigger})
    for lane_key in reg.lanes:
        members = reg.lane_members(lane_key)
        if not members or not members[0].context_window:
            continue
        bigger = [m.deployment for m in sized
                  if m.context_window > members[0].context_window][:2]
        if bigger:
            context_fallbacks.append({lane_key: bigger})

    return {
        "model_list": model_list,
        "litellm_settings": {
            "callbacks": ["switchyard.hooks.switchyard_handler"],
            "drop_params": True,
            "request_timeout": 600,
            "num_retries": 0,   # Switchyard owns retry placement, not LiteLLM
        },
        "router_settings": {
            "enable_pre_call_checks": True,
            "routing_strategy": "simple-shuffle",
            "fallbacks": fallbacks,
            "context_window_fallbacks": context_fallbacks,
            "allowed_fails": 1,
            "cooldown_time": 60,   # short: the real cooldowns live in Redis
            "redis_host": "os.environ/REDIS_HOST",
            "redis_port": "os.environ/REDIS_PORT",
        },
        "general_settings": {
            "master_key": "os.environ/LITELLM_MASTER_KEY",
            "database_url": "os.environ/DATABASE_URL",
        },
    }


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "config/plans.yaml"
    dst = sys.argv[2] if len(sys.argv) > 2 else "config/litellm.generated.yaml"
    cfg = build(src)
    header = (
        "# GENERATED by `python -m switchyard.gen_litellm` — do not edit.\n"
        "# Edit config/plans.yaml instead, then POST /admin/reload.\n"
    )
    with open(dst, "w") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False, width=100)
    print(f"wrote {dst}: {len(cfg['model_list'])} entries, {len(cfg['router_settings']['fallbacks'])} fallback rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
