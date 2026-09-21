#!/usr/bin/env python3
"""The auth audit from scripts/apply.sh, as its own script.

Reads credentials only as EXISTENCE: file present, key non-empty, grant
authorised. Never prints a value.

    scripts/auth_audit.py               # status report
    scripts/auth_audit.py --run         # run each missing login inline, then
                                        # re-audit and report what is left

Nothing here is specific to anyone's subscriptions. Every per-subscription fact
is read from the real service config, the way the sidecars themselves read it:

  service  <-> plan   docker-compose.yml (the service whose SWITCHYARD_PLAN
                                       matches the plan key)
  CLI kind          the compose service's PROVIDER env
  credential path   the compose volume mounted at the CLI's home path, with
                    its host side resolved like compose does — ${VAR:-default}
                    -> environment, then .env, then the compose default
  opencode login id first segment of the plan's model spec after the LiteLLM
                    prefix is stripped — the same rule cli_bridge uses to
                    build `opencode run --model` (server.py)

The only hardcoded table is per-CLI facts — where each CLI stores its login
and how it logs in. Those are properties of the three CLIs the sidecar image
supports (the same set as PROFILES in sidecars/cli_bridge/server.py), so a new
subscription in docker-compose.yml is audited with no change here.

Env-file gaps (api_key plans) are never auto-filled — only the operator edits
.env. CLI logins run inline in --run because they only touch the sidecars' own
./secrets bind mounts, never the host keychain or browser session.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMPOSE_FILE = ROOT / "docker-compose.yml"

# Per-CLI facts only. Keyed by the CONTAINER-side mount path, which is where
# the CLI looks inside the container (all three CLIs are file-based there):
# OpenCode keeps auth.json in its DATA home, not its config home, so only the
# /home/node/.local/share/opencode mount carries the credential.
CONTAINER_CRED_FILES = {
    "/home/node/.claude": ".credentials.json",          # CLAUDE_CONFIG_DIR
    "/home/node/.codex": "auth.json",                   # CODEX_HOME
    "/home/node/.local/share/opencode": "auth.json",    # OpenCode data home
}

# Logins, by compose PROVIDER. {provider} is filled per plan (opencode only).
LOGIN_TEMPLATE = {
    "claude": "claude login",
    "codex": "codex login --device-auth",
    "opencode": "opencode auth login --provider {provider}",
}

_MOUNT_VAR = re.compile(r"^\$\{([A-Za-z0-9_]+):-(.*)\}$")
_MOUNT_INTERP = re.compile(r"^(\$\{[^}]*\}):(.*)$")


def env_file(path: Path = ROOT / ".env") -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def env_refs(value):
    """The os.environ/KEY references a plan field makes, if any."""
    if not value or not value.startswith("os.environ/"):
        return []
    return [value.split("/", 1)[1]]


def _svc_env(body: dict) -> dict:
    """A compose service's environment in dict form (handles list syntax)."""
    env = body.get("environment")
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    if isinstance(env, list):
        out = {}
        for item in env:
            if isinstance(item, str) and "=" in item:
                k, _, v = item.partition("=")
                out[k] = v
        return out
    return {}


def _split_volume(vol: str) -> tuple[str, str] | None:
    """Split a compose short-syntax volume into host spec and container path.

    The naive `split(":")` fails here: the host side is often `${VAR:-./path}`,
    whose `:-` IS a colon. Match the `${...}` as a unit first.
    """
    s = vol.strip()
    if s.startswith("${"):
        m = _MOUNT_INTERP.match(s)
        if not m:
            return None
        host, rest = m.group(1), m.group(2)
    elif ":" in s:
        host, rest = s.split(":", 1)
    else:
        return None
    return host, rest.split(":")[0]


def _svc_mounts(body: dict) -> list[tuple[str, str]]:
    """(host spec, container path) pairs from a service's volumes."""
    out = []
    for vol in body.get("volumes") or []:
        if isinstance(vol, dict):
            out.append((str(vol.get("source", "")), str(vol.get("target", ""))))
        elif isinstance(vol, str):
            split = _split_volume(vol)
            if split:
                out.append(split)
    return out


def resolve_host_mount(spec: str, dotenv: dict) -> str:
    """Interpolate `${VAR:-default}` like compose: shell env, then .env, then
    the default — also when the variable is set but empty, which is what the
    `:-` form means."""
    m = _MOUNT_VAR.match(spec.strip())
    if not m:
        return spec.strip()
    var, default = m.group(1), m.group(2)
    return os.environ.get(var) or (dotenv.get(var) or "") or default


def opencode_login_provider(p) -> str:
    """The plan's first enabled model's provider segment, LiteLLM prefix.

    The sidecar turns `openai/opencode-go/glm-5.3-flash` into the CLI model
    `opencode-go/glm-5.3-flash` (server.py strips one leading segment); the
    remainder's first segment is the id `opencode auth login` takes. A
    two-segment name like `openai/xai/grok-4.6` gives `xai`. The plan key is
    the fallback when the plan models nothing enabled.
    """
    for m in (p.models or {}).values():
        if getattr(m, "enabled", True) and m.model:
            parts = m.model.split("/")
            return parts[1] if len(parts) > 2 else parts[0]
    return p.key


def load_services(path: Path = COMPOSE_FILE) -> dict:
    if not path.exists():
        return {}
    import yaml
    compose = yaml.safe_load(path.read_text()) or {}
    out = {}
    for name, body in (compose.get("services") or {}).items():
        out[name] = body if isinstance(body, dict) else {}
    return out


def audit(plans_path: str = "config/plans.yaml",
          dotenv: dict | None = None,
          services: dict | None = None) -> tuple[list[str], list[str]]:
    """Returns (status_lines, login_commands_for_whatever_is_missing)."""
    if dotenv is None:
        dotenv = env_file()
    services = load_services() if services is None else services
    sys.path.insert(0, str(ROOT))
    from switchyard import models, oauth

    reg = models.load(plans_path)
    status: list[str] = []
    pending: list[str] = []

    api_plans = [p for p in reg.plans.values() if p.auth == "api_key"]
    for p in api_plans:
        needed = sorted(set(env_refs(p.api_base) + env_refs(p.api_key)))
        if not needed:
            status.append(f"  {p.key:<14} api_key: no os.environ references to check")
            continue
        absent = [k for k in needed if not dotenv.get(k)]
        if absent:
            status.append(f"  {p.key:<14} MISSING env keys: {', '.join(absent)}")
            pending.append(
                f"echo '  {p.key}: fill {', '.join(absent)} in .env, then re-run scripts/apply.sh'")
        else:
            status.append(f"  {p.key:<14} env keys present: {', '.join(needed)}")

    env_by_svc = {name: _svc_env(body) for name, body in services.items()}

    def opencode_id(p) -> str:
        return opencode_login_provider(p)

    for p in reg.plans.values():
        if not p.is_cli_backed:
            continue
        svc = next((name for name, env in env_by_svc.items()
                    if env.get("SWITCHYARD_PLAN") == p.key), None)
        if not svc:
            status.append(
                f"  {p.key:<14} cli_sidecar: no compose service sets SWITCHYARD_PLAN: {p.key}")
            continue
        provider = env_by_svc[svc].get("PROVIDER")
        if provider not in LOGIN_TEMPLATE:
            status.append(
                f"  {p.key:<14} cli_sidecar: compose service {svc!r} has no "
                f"auditable PROVIDER (set one of {sorted(LOGIN_TEMPLATE)})")
            continue
        cred_mount = next(((host, fname) for host, target in _svc_mounts(services[svc])
                           for fname in [CONTAINER_CRED_FILES.get(target)] if fname),
                          None)
        if not cred_mount:
            status.append(
                f"  {p.key:<14} cli_sidecar: {svc} mounts none of "
                f"{sorted(CONTAINER_CRED_FILES)} — no credential to audit")
            continue
        host, fname = cred_mount
        cred = Path(resolve_host_mount(host, dotenv)) / fname
        if not cred.is_absolute():
            cred = ROOT / cred
        if cred.exists() and cred.stat().st_size > 0:
            status.append(f"  {p.key:<14} cli login present ({cred})")
        else:
            where = str(cred.relative_to(ROOT)) if cred.is_relative_to(ROOT) else str(cred)
            status.append(f"  {p.key:<14} NOT signed in — {where} absent")
            pending.append(
                f"docker compose exec {svc} "
                + LOGIN_TEMPLATE[provider].format(provider=opencode_id(p)))

    for p in reg.plans.values():
        if p.auth != "oauth_proxy":
            continue
        provider = p.provider_family or "xai"
        if provider not in set(oauth.FLOWS) | set(oauth.HEADLESS_FLOWS):
            status.append(f"  {p.key:<14} oauth_proxy: no flow registered for {provider!r}")
            continue
        st = oauth.status(provider)
        if st.get("authorised"):
            status.append(f"  {p.key:<14} oauth grant present ({provider})")
        else:
            status.append(f"  {p.key:<14} oauth grant MISSING ({provider})")
            pending.append(f"python3 -m switchyard.oauth login {provider}")

    return status, pending


def main() -> int:
    args = sys.argv[1:]
    run = "--run" in args
    idx = args.index("--plans") if "--plans" in args else -1
    plans = args[idx + 1] if idx != -1 else "config/plans.yaml"

    if not os.path.exists(plans):
        print(f"no {plans} — run scripts/sync-env.sh first", file=sys.stderr)
        return 2

    status, pending = audit(plans)
    if not run:
        for line in status:
            print(line)
        if pending:
            print()
            print("ACTION REQUIRED — these need you, then run scripts/apply.sh again:")
            for cmd in pending:
                print(f"  {cmd}")
        else:
            print("  every plan's credential is in place")
        return 0

    for cmd in pending:
        print(f"==> running: {cmd}")
        if subprocess.call(cmd, shell=True) != 0:
            print(f"    command failed — fix it by hand: {cmd}", file=sys.stderr)
    status, pending = audit(plans)
    for line in status:
        print(line)
    if pending:
        print()
        print("STILL MISSING after the inline logins — these need you:")
        for cmd in pending:
            print(f"  {cmd}")
        return 1
    print("  every plan's credential is in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
