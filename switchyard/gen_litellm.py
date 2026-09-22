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
    SwitchYard's slot table is the real gate, keyed by plan so a plan's models
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

    # NO general fallbacks, on purpose. LiteLLM's fallbacks live in the router,
    # which runs *after* the proxy's pre-call hook, so every fallback attempt
    # went behind SwitchYard's back: it skipped the tool-capability filter (a
    # tool request could land on a plan whose sidecar hard-400s it), claimed no
    # slot, ignored the session lease and the mid-tool-loop pin, and -- worst --
    # the success hook booked its tokens against the plan the *picker* chose, so
    # one subscription's quota was spent and another's was debited.
    #
    # It also hid the failures it rescued: a lane listing every member meant a
    # broken plan was silently retried on a healthy one and looked fine.
    #
    # SwitchYard owns placement. A failed request now returns to the caller,
    # whose retry re-enters the picker and gets a correct pick -- against the
    # live cooldowns the failure just set, which is better placement than a
    # fixed list could give. See _check_served_deployment for the guard that
    # keeps any remaining router-level retry from hiding.
    fallbacks: list[dict[str, list[str]]] = []

    # Context-window fallbacks are kept, as the exception: a prompt bigger than
    # the model's window cannot be served where it was sent, so the alternative
    # is hard-failing it. They carry the same attribution caveat as any
    # router-level retry, which is why _check_served_deployment logs when one
    # fires rather than letting the mis-booking pass unnoticed.
    #
    # Only models that declare `context_window` take part, so an unknown window
    # never silently becomes a wrong routing decision.
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
            # One router-level retry, and that one retry is the ONLY place we
            # let litellm re-route: SwitchYard's async_pre_routing_hook swaps
            # in a different member of the lane on transient failure, so the
            # retry delivers a real second attempt instead of the same broken
            # deployment. Mid-stream failures are not retried by litellm and
            # are not affected. Two is wrong: it would silently double our
            # upstream calls without buying us anything.
            "num_retries": 1,
        },
        "router_settings": {
            "enable_pre_call_checks": True,
            "routing_strategy": "simple-shuffle",
            "fallbacks": fallbacks,
            "context_window_fallbacks": context_fallbacks,
            # `disable_cooldowns: True` is the only setting that keeps the
            # router from cooling the single-deployment group between attempts
            # (so the re-pick in async_pre_routing_hook actually fires), and
            # SwitchYard's own cooldowns — via Redis — are the only state
            # that outlives this process and the only state the portal can
            # see. With this on, cooldown_time and allowed_fails do not need
            # to be set; cooldowns live in K_COOL/{plan}, and the breaker's
            # escalating ladder lives in K_TFAIL/{plan}.
            "disable_cooldowns": True,
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
