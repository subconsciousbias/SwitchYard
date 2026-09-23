#!/usr/bin/env python3
"""Check a RUNNING SwitchYard deployment. This is not the test suite.

It sends real requests to real providers, so it needs the stack up, your
credentials present, and it consumes capacity — local models by default, and
paid subscription quota with --paid. Nothing here is mocked; that is the point,
and it is why it lives in scripts/ rather than tests/.

The offline suite is `scripts/test.sh`: no network, no credentials, no
Docker, and it refuses to reach anything but loopback (see tests/conftest.py).
Run that constantly; run this when you have changed the deployment.

    python3 scripts/smoke.py              # local models only, no paid quota
    python3 scripts/smoke.py --paid       # also the lanes that spend quota
    python3 scripts/smoke.py --slow       # also the CLI harness overhead (minutes)

A check either passes with evidence or fails with the reason. Nothing is
reported as "probably fine". Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The LIVE config on purpose, unlike the unit tests, which always use
# config/plans.example.yaml: smoke checks the stack that is actually running,
# so it has to read the plans that stack was started with.
os.environ.setdefault("SWITCHYARD_PLANS", str(
    Path(__file__).resolve().parent.parent / "config" / "plans.yaml"))
from switchyard import models  # noqa: E402  # read-only: which plan a member belongs to

GW = "http://localhost:4000"
# How long to wait before retrying a lane that answered "no capacity".
SATURATED_RETRY_SECONDS = 8
PORTAL = "http://localhost:4001"
SIDECARS = {"claude-max-sidecar": 8081, "codex-sidecar": 8082,
            "opencode-go-sidecar": 8084}
# Not a CLI sidecar: it forwards the caller's body to api.x.ai under our own
# OAuth grant, so it has no harness prompt to measure and no vendor CLI to log
# in. Checked for a live grant instead.
TOKEN_PROXIES = {"xai-token-proxy": 8090}

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if detail else ""))
    return ok


def sh(*args: str, timeout: int = 60) -> str:
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout).stdout.strip()


def compose(*args: str, timeout: int = 60) -> str:
    return sh("docker", "compose", *args, timeout=timeout)


def key() -> str:
    return compose("exec", "-T", "gateway", "printenv", "LITELLM_MASTER_KEY")


def post(path: str, payload: dict, api_key: str, timeout: int = 600,
         anthropic: bool = False) -> dict:
    headers = {"Content-Type": "application/json"}
    if anthropic:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    # A saturated lane is retried, once, after a pause. Several checks here
    # aim at the same 2-slot local plan back to back, and a slot is released by
    # the completion callback slightly AFTER the response reaches us — so a
    # sequential caller can genuinely race itself into "no capacity". That is
    # the lane working as designed, not a failure, and reporting it as one made
    # three checks flap. A second refusal is still reported.
    for attempt in (1, 2):
        req = urllib.request.Request(GW + path, json.dumps(payload).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()[:300]
            saturated = exc.code == 429 and "no capacity" in body
            if saturated and attempt == 1:
                time.sleep(SATURATED_RETRY_SECONDS)
                continue
            return {"__http__": exc.code, "__body__": body}


def state() -> dict:
    with urllib.request.urlopen(PORTAL + "/api/state", timeout=30) as resp:
        return json.load(resp)


def routed_to(lane: str, since: int = 25) -> str:
    """The most recent routing decision the gateway logged for this lane."""
    logs = compose("logs", f"--tail={since}", "gateway")
    for line in reversed(logs.splitlines()):
        if f"lane={lane} ->" in line:
            return line.split(f"lane={lane} -> ", 1)[1].strip()
    return ""


# --------------------------------------------------------------------- checks --
def check_services() -> None:
    out = compose("ps", "--format", "{{.Service}} {{.Status}}")
    running = [l for l in out.splitlines() if " Up " in l or l.endswith("Up")]
    expected = ({"gateway", "portal", "redis", "postgres"}
                | set(SIDECARS) | set(TOKEN_PROXIES))
    names = {l.split()[0] for l in running}
    check("all services up", expected <= names,
          f"missing: {sorted(expected - names)}" if expected - names else
          f"{len(names)} services")


def check_plugin_loaded() -> None:
    logs = compose("logs", "gateway")
    line = next((l for l in logs.splitlines() if "switchyard:" in l and "plans, lanes=" in l), "")
    check("routing plugin loaded", bool(line), line.split("switchyard: ")[-1] if line else
          "no banner — requests would bypass slot accounting entirely")


def check_sidecar_health() -> None:
    for svc, port in SIDECARS.items():
        raw = compose("exec", "-T", svc, "python3", "-c",
                      f"import json,urllib.request;"
                      f"print(json.dumps(json.load(urllib.request.urlopen("
                      f"'http://localhost:{port}/health'))))")
        try:
            d = json.loads(raw)
        except ValueError:
            check(f"{svc} health", False, raw[:120])
            continue
        # `plan` is cli_bridge-only; mcp_bridge reports `provider`. Assert on
        # what both return, or this passes for the wrong reason on a bridged
        # sidecar and prints "plan=None" while claiming success.
        ok = d.get("ok") and d.get("config_source") == "config"
        check(f"{svc} reads its plan from config", ok,
              f"plan={d.get('plan') or d.get('provider')} "
              f"models={d.get('models')} conc={d.get('concurrency')}")


def check_token_proxy_health() -> None:
    """A proxy without a grant serves nothing, and says so rather than 500ing.

    The grant is taken out from the host (`python3 -m switchyard.oauth login
    xai`) because it needs a human at a browser, so an unauthorised proxy is a
    normal state to find and worth naming precisely — the plan it fronts will
    otherwise fail every request in its lane.
    """
    for svc, port in TOKEN_PROXIES.items():
        raw = compose("exec", "-T", svc, "python3", "-c",
                      f"import json,urllib.request;"
                      f"print(json.dumps(json.load(urllib.request.urlopen("
                      f"'http://localhost:{port}/health'))))")
        try:
            d = json.loads(raw)
        except ValueError:
            check(f"{svc} health", False, raw[:120])
            continue
        if not d.get("authorised"):
            check(f"{svc} has a live grant", False,
                  f"no grant on file — run: python3 -m switchyard.oauth "
                  f"login {d.get('provider')}")
            continue
        left = d.get("expires_in")
        check(f"{svc} has a live grant", bool(d.get("ok")),
              f"{d.get('provider')} -> {d.get('upstream_base')}, "
              f"{left}s left, refresh={'yes' if d.get('has_refresh') else 'NO'}")


def check_lane(lane: str, api_key: str, expect_text: str | None = None) -> None:
    payload = {"model": lane, "max_tokens": 200,
               "messages": [{"role": "user", "content": f"Reply with exactly: {lane.upper()} OK"}]}
    d = post("/v1/chat/completions", payload, api_key)
    if "__http__" in d:
        check(f"lane {lane}", False, f"HTTP {d['__http__']}: {d['__body__'][:150]}")
        return
    content = d.get("choices", [{}])[0].get("message", {}).get("content", "")
    member = routed_to(lane)
    check(f"lane {lane} answers", bool(content),
          f"-> {member}  {content[:60]!r}")


def check_anthropic_protocol(api_key: str) -> None:
    d = post("/v1/messages", {"model": "bulk", "max_tokens": 200,
             "messages": [{"role": "user", "content": "Reply with exactly: MESSAGES OK"}]},
             api_key, anthropic=True)
    if "__http__" in d:
        check("anthropic /v1/messages", False, f"HTTP {d['__http__']}: {d['__body__'][:150]}")
        return
    blocks = d.get("content") or []
    text = next((b.get("text", "") for b in blocks if b.get("type") == "text"), "")
    check("anthropic /v1/messages", d.get("type") == "message" and bool(text),
          f"model={d.get('model')} {text[:40]!r}")


_REGISTRY = None


def registry() -> models.Registry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = models.load()
    return _REGISTRY


def check_tool_routing(lane: str, api_key: str) -> None:
    """A tool-carrying request must come back with real `tool_calls`, and the
    member that served it must belong to a plan that actually reports tool
    capability (`Plan.can_use_tools`) — not one we merely assume can, because
    that assumption is exactly what used to be wrong.

    Note: `can_use_tools` now defaults to true for every plan and is only
    false where config says `supports_tools: false`. Today no plan sets that,
    so a CLI-backed plan (claude-max, openai seat, grok, opencode-go) is a
    legal pick here even though its sidecar still hard-rejects any request
    carrying `tools` with a 400 (sidecars/cli_bridge/server.py) until the
    direct-API and MCP-bridge workstreams land. If this check lands on one of
    those plans and gets a 400, that is an accurate failure, not a bug in this
    check — it means the lane's ordered fill reached a plan whose sidecar
    can't serve tools yet.
    """
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    d = post("/v1/chat/completions",
             {"model": lane, "max_tokens": 150, "tools": tools,
              "messages": [{"role": "user", "content": "What is the weather in Oslo?"}]},
             api_key)
    name = f"lane {lane}: tool calls routed to a tool-capable plan"
    if "__http__" in d:
        check(name, False, f"HTTP {d['__http__']}: {d['__body__'][:150]}")
        return
    calls = d.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
    member = routed_to(lane)
    ref = member.split()[0] if member else ""   # routed_to() trails cap/sticky/skip notes
    model = registry().model(ref) if ref else None
    capable = bool(model) and registry().plan_of(model).can_use_tools
    check(name, bool(calls) and capable,
          f"-> {member}  tool_calls={[c['function']['name'] for c in calls]}  "
          f"plan.can_use_tools={capable}")


def check_affinity(api_key: str) -> None:
    session = f"smoke-{int(time.time())}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
               "X-Session-Id": session}
    for _ in range(3):
        req = urllib.request.Request(
            GW + "/v1/chat/completions",
            json.dumps({"model": "local", "max_tokens": 40,
                        "messages": [{"role": "user", "content": "say hi"}]}).encode(),
            headers)
        try:
            urllib.request.urlopen(req, timeout=300).read()
        except urllib.error.HTTPError as exc:
            check("session affinity", False, f"HTTP {exc.code}")
            return
    logs = compose("logs", "--tail=20", "gateway")
    sticky = [l for l in logs.splitlines() if "lane=local ->" in l and "(sticky)" in l]
    check("session affinity keeps a session on one model", len(sticky) >= 2,
          f"{len(sticky)} of the last 3 requests were sticky")


def check_cooldown_shrinks_capacity() -> None:
    lane, plan = "forge", "minimax-ultra"
    before = next(l for l in state()["capacity"]["lanes"] if l["lane"] == lane)
    compose("exec", "-T", "redis", "redis-cli", "-n", "1", "SET",
            f"sy:cool:{plan}", "quota_exhausted|0", "EX", "60")
    during = next(l for l in state()["capacity"]["lanes"] if l["lane"] == lane)
    compose("exec", "-T", "redis", "redis-cli", "-n", "1", "DEL", f"sy:cool:{plan}")
    after = next(l for l in state()["capacity"]["lanes"] if l["lane"] == lane)
    a, b, c = (x["slots_available_now"] for x in (before, during, after))
    check("cooling a plan removes its capacity", b < a and c == a,
          f"{lane}: {a} -> {b} with {plan} cooled -> {c} restored")


def check_pacing_switch() -> None:
    def toggle(value: str) -> dict:
        req = urllib.request.Request(f"{PORTAL}/admin/pacing?enabled={value}", b"",
                                     method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    on = toggle("on")
    lane = next(l for l in state()["capacity"]["lanes"] if l["lane"] == "forge")
    tail_open = [p["ref"] for p in lane["plans"] if p["tail"] and p["cap"] > 0]
    toggle("default")
    off = state()["capacity"]["pacing"]
    check("pacing mode toggles and disables the tail",
          on["pacing"] and not tail_open and not off,
          f"on={on['pacing']} tail_offered={tail_open or 'none'} cleared={not off}")


def check_quota_windows_tracked() -> None:
    keys = compose("exec", "-T", "redis", "redis-cli", "-n", "1", "--scan",
                   "--pattern", "sy:usage:*:p:*").splitlines()
    plans = {k.split(":")[2] for k in keys if k.strip()}
    multi = [p for p in plans
             if len([k for k in keys if f":usage:{p}:p:" in k]) > 1]
    check("usage booked per plan, per quota window", bool(plans),
          f"plans with usage: {sorted(plans)}" +
          (f"; multi-window: {sorted(multi)}" if multi else ""))


def check_harness_overhead() -> None:
    """The CLI's own prompt, which is quota spent on instructions you did not
    write. Slow: one real call per sidecar."""
    for svc, port, model in (("claude-max-sidecar", 8081, "claude-opus-5"),
                             ("codex-sidecar", 8082, "gpt-5.6-sol")):
        script = (
            "import json,urllib.request;"
            f"body=json.dumps({{'model':'{model}','max_tokens':60,"
            "'messages':[{'role':'user','content':'Say OK'}]}).encode();"
            f"d=json.load(urllib.request.urlopen(urllib.request.Request("
            f"'http://localhost:{port}/v1/chat/completions',body,"
            "{'Content-Type':'application/json'}),timeout=600));"
            "print(d['usage']['prompt_tokens'])")
        raw = compose("exec", "-T", svc, "python3", "-c", script, timeout=700)
        try:
            tokens = int(raw.splitlines()[-1])
        except (ValueError, IndexError):
            check(f"{svc} harness overhead", False, raw[:120])
            continue
        # Budgets sit a little above the measured figures (2 / 9,768), so a
        # regression that reintroduces the CLI's own prompt trips this.
        budget = {"claude-max-sidecar": 100, "codex-sidecar": 11000}[svc]
        check(f"{svc} harness overhead within budget", tokens <= budget,
              f"{tokens} prompt tokens (budget {budget})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paid", action="store_true",
                    help="also exercise lanes that spend subscription quota")
    ap.add_argument("--slow", action="store_true",
                    help="also measure CLI harness overhead (minutes)")
    args = ap.parse_args()

    print("infrastructure")
    check_services()
    check_plugin_loaded()
    check_sidecar_health()
    check_token_proxy_health()

    api_key = key()
    if not api_key:
        check("master key readable from the gateway", False, "empty")
        return 1

    print("\nfree lanes")
    for lane in ("local", "bulk"):
        check_lane(lane, api_key)
    check_anthropic_protocol(api_key)

    print("\nbehaviour")
    check_affinity(api_key)
    check_cooldown_shrinks_capacity()
    check_pacing_switch()
    check_quota_windows_tracked()
    for lane in ("local", "bulk"):
        check_tool_routing(lane, api_key)

    if args.paid:
        print("\npaid lanes (spending subscription quota)")
        for lane in ("forge", "judge", "apex"):
            check_lane(lane, api_key)
            check_tool_routing(lane, api_key)

    if args.slow:
        print("\nCLI harness overhead")
        check_harness_overhead()

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
