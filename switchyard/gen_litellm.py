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
    exceeding the plan's connection limit. The number it carries has to MATCH
    what the picker would have admitted via lane routing: for CLI-backed
    plans that is `min(cap, max(1, max_parallel - gate_headroom_slots))`,
    not the raw `max_parallel`. Without the alignment, a direct-name caller
    races the physical sidecar gate for the headroom slot the picker
    deliberately leaves empty, which is exactly the spurious-429 race
    issue #47 complains about. Pass `reg.settings` so `cap_for` applies the
    headroom; API plans are unaffected.
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
            params["max_parallel_requests"] = plan.cap_for(model, reg.settings)

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
            # num_retries: 0 is the README-documented contract: a failed request
            # returns to the caller immediately, and the caller's retry re-enters
            # the proxy through async_pre_call_hook. The picker then places against
            # the fresh cooldowns the failure just set — which is strictly better
            # placement than any in-router retry could give.
            #
            # This is non-negotiable in the pinned litellm build. Router's
            # `async_pre_routing_hook` is the *internal* auto-router hook,
            # dispatched only inside `async_get_available_deployment` for
            # `routing_strategy`-based strategies; it never iterates registered
            # CustomLogger callbacks, so `SwitchyardHandler.async_pre_routing_hook`
            # has never fired. A blind router retry (num_retries >= 1) re-routes
            # to the same single-member deployment the picker just chose, which
            # is exactly the deployment that just 5xx'd. The retry cannot recover.
            #
            # Whole-lane-full already returns a clean 429 + Retry-After: 20
            # (hooks.py LaneSaturated), so the caller's well-behaved retry lands
            # against the fresh cooldowns the picker sees on re-entry.
            "num_retries": 0,
            # Route Anthropic-protocol /v1/messages onto /v1/chat/completions.
            # Without this, LiteLLM v1.101.0 translates an openai/-prefixed
            # deployment's /v1/messages request into POST {api_base}/responses,
            # which 404s at every sidecar — cli_bridge and mcp_bridge serve
            # only /v1/chat/completions. (xai-token-proxy also exposes
            # /v1/messages natively, but the flag applies uniformly and is
            # safe there too.) This flag sends /v1/messages through LiteLLM's
            # own Anthropic->chat-completions adapter onto the path the
            # sidecars serve, so Claude Code reaches them without any
            # per-bridge protocol code.
            "use_chat_completions_url_for_anthropic_messages": True,
        },
        "router_settings": {
            "enable_pre_call_checks": True,
            "routing_strategy": "simple-shuffle",
            "fallbacks": fallbacks,
            "context_window_fallbacks": context_fallbacks,
            # `disable_cooldowns: True` is still required. Without it, even at
            # num_retries=0 the router cools the deployment group between
            # calls and prevents fresh traffic from landing on the one plan
            # the picker just chose. With it on, cooldown_time and
            # allowed_fails are unused and SwitchYard's own Redis cooldowns
            # (K_COOL/{plan}, K_TFAIL/{plan}) are the only state the system
            # consults — and the only state the portal can see.
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
