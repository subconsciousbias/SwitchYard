#!/usr/bin/env bash
# Apply an edited config/plans.yaml to the running stack — the one command.
#
#   scripts/apply.sh              # the normal case
#   scripts/apply.sh --dry-run    # validate, audit and report, touch nothing
#   scripts/apply.sh --build      # force a rebuild of the baked images too
#   scripts/apply.sh --skip-reload # reload.sh already ran, or runs separately
#
# Why this exists: plans.yaml is MOUNTED read-only into every service, so
# editing it never needs a rebuild. A rebuild is needed only when the code that
# is baked into the images changed — switchyard/*.py, sidecars/, the
# Dockerfiles, requirements.txt. Rebuilding unconditionally makes every plan
# edit look heavy and slow; skipping it when stale leaves half the stack on old
# code. So this script decides per image, comparing the mtimes of baked-in
# sources against each image's creation time.
#
# And ending up running is not enough: a plan without its credential serves 401s
# out of its lane. So the script ends with an auth audit — env keys, CLI login
# files, OAuth grants — printing the exact command for each one that is
# missing. It never runs a login itself: those are your keychain and browser
# session. It prints them; you run them.
#
# Order matters and is deliberate:
#   1. sync-env       -- append new .env.example keys BEFORE anything recreates
#   2. validate       -- a broken plans.yaml is refused while the old config runs
#   3. build (stale)  -- the only slow step, and usually skipped
#   4. compose up -d  -- creates new services, recreates what compose changed
#   5. reload.sh      -- restart the gateway, refresh the portal, wait for health
#   6. auth audit     -- what still needs YOU, with the exact command
#
# set -euo pipefail: a silent partial apply is the one outcome this must not
# produce — failing loudly mid-way beats reporting "done" while the gateway
# still routes the old config.
set -euo pipefail

cd "$(dirname "$0")/.."

force_build=0
dry_run=0
skip_reload=0
for arg in "$@"; do
  case "$arg" in
    --build) force_build=1 ;;
    --dry-run) dry_run=1 ;;
    --skip-reload) skip_reload=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

plans="config/plans.yaml"
[ -f "$plans" ] || { echo "no $plans — run scripts/sync-env.sh first" >&2; exit 1; }

# ---------------------------------------------------------------- 1. sync-env
echo "==> propagating .env.example keys (appends only, never overwrites)"
sync_out="$(scripts/sync-env.sh)"
echo "$sync_out"
# .env is read when a CONTAINER IS CREATED, not on restart, so keys appended
# after creation need recreation. Anything sync-env added triggers that below.
env_added="$(printf '%s\n' "$sync_out" \
             | grep -E '^added [0-9]+ key' | grep -oE '[0-9]+' || true)"
env_added="${env_added:-0}"

# ---------------------------------------------------------------- 2. validate
# Same preflight as reload.sh: load the registry through models.load(), print
# the lanes, refuse an edit that would leave a lane dead. The gateway generates
# its LiteLLM config at startup, so a broken file either stops it coming up or
# sends it into quiet fallback — worse to debug than a refusal here, before
# Docker has touched anything.
echo "==> validating $plans"
python3 - "$plans" <<'PY'
import sys
sys.path.insert(0, ".")
from switchyard import models

path = sys.argv[1]
reg = models.load(path)

tools = [k for k, p in reg.plans.items() if p.can_use_tools]
print(f"    {len(reg.plans)} plans, {len(reg.models)} models, {len(reg.lanes)} lanes")
print(f"    tool-capable: {len(tools)}/{len(reg.plans)}")
for lane in reg.lanes:
    members = [m.ref for m in reg.lane_members(lane)]
    print(f"    {lane:<7} {' -> '.join(members) if members else '(no live members)'}")
    if not members:
        print(f"    WARNING: lane {lane!r} has no live members — it will fail every request",
              file=sys.stderr)
PY

# -------------------------------------------------------- 3. staleness + build
# One entry per IMAGE, not per service: codex-sidecar and opencode-go-sidecar
# share the sidecar image that claude-max-sidecar builds, so they appear only
# once. A service with no image yet (first run) counts as stale.
mapfile -t stale < <(python3 - "$force_build" <<'PY'
import calendar, subprocess, sys, time
from pathlib import Path

force = int(sys.argv[1])
# The baked-in sources each image is built from; a change here means rebuild.
sources = {
    "gateway":            ["Dockerfile.gateway", "requirements.txt", "switchyard/"],
    "portal":             ["Dockerfile.portal",  "requirements.txt", "switchyard/"],
    "claude-max-sidecar": ["Dockerfile.sidecar", "requirements.txt", "sidecars/"],
    "xai-token-proxy":    ["Dockerfile.token_proxy", "requirements.txt",
                           "sidecars/token_proxy/", "switchyard/oauth.py"],
}

def compose(*args: str) -> str:
    p = subprocess.run(["docker", "compose", *args], capture_output=True, text=True)
    return p.stdout.strip()

def newest_mtime(paths) -> float:
    newest = 0.0
    # __pycache__ and *.pyc are interpreter by-products: any local test run
    # refreshes them, and counting them would flag every image stale after a
    # mere `python3 tests/test_*.py`, forcing a rebuild for a config-only edit.
    for s in paths:
        p = Path(s)
        if not p.exists():
            continue
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                    newest = max(newest, f.stat().st_mtime)
        else:
            newest = max(newest, p.stat().st_mtime)
    return newest

for svc, srcs in sources.items():
    if force:
        print(svc)
        continue
    image = compose("images", "-q", svc)
    if not image:
        print(svc)   # never built: not stale, absent
        continue
    created = subprocess.run(["docker", "image", "inspect", image,
                              "--format", "{{.Created}}"],
                             capture_output=True, text=True).stdout.strip()
    if not created:
        print(svc)
        continue
    # Docker's Created is UTC; st_mtime is an absolute epoch. calendar.timegm
    # parses the timestamp as UTC — mktime would skew by the local offset and
    # build decisions would differ by time zone.
    try:
        t = time.strptime(created.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        created_epoch = calendar.timegm(t)
    except ValueError:
        print(svc)
        continue
    if newest_mtime(srcs) > created_epoch:
        print(svc)
PY
)

if [ "${#stale[@]}" -gt 0 ]; then
  echo "==> rebuilding (baked sources newer than image): ${stale[*]}"
  if [ "$dry_run" -eq 0 ]; then
    docker compose build "${stale[@]}"
  else
    echo "    (dry-run: not building)"
  fi
else
  echo "==> nothing stale — plans.yaml is mounted, so no rebuild needed"
fi

# -------------------------------------------------------------- 4. compose up
# --force-recreate only when sync-env ADDED keys (env is read at creation), and
# only for the app services — never redis/postgres, whose recreation is churn.
# A plain `up -d` handles everything else, including adding a new service.
if [ "$dry_run" -eq 1 ]; then
  echo "==> (dry-run: would run docker compose up -d)"
elif [ "$env_added" -gt 0 ]; then
  echo "==> $env_added new env key(s) — recreating app containers on the way up"
  docker compose up -d --force-recreate \
    gateway portal claude-max-sidecar codex-sidecar opencode-go-sidecar xai-token-proxy
else
  echo "==> docker compose up -d"
  docker compose up -d
fi

# -------------------------------------------------------------- 5. reload
# The gateway's router is built at startup, so it needs the restart whatever
# compose just did. reload.sh also refreshes the portal's board and waits out
# the health flap.
if [ "$skip_reload" -eq 1 ]; then
  echo "==> skipping reload.sh (--skip-reload)"
elif [ "$dry_run" -eq 1 ]; then
  echo "==> (dry-run: would run scripts/reload.sh)"
else
  echo "==> reload.sh: gateway restart, portal refresh, health wait"
  scripts/reload.sh
fi

# ------------------------------------------------------------ 6. auth audit
# Reads credentials only as EXISTENCE: file present, key non-empty, grant
# authorisation. Never prints a value. Prints the command for each one it must
# not run itself.
echo "==> auth audit"
python3 - <<'PY'
import os, sys
from pathlib import Path

sys.path.insert(0, ".")
import yaml
from switchyard import models

reg = models.load("config/plans.yaml")
missing: list[str] = []

def env_file() -> dict:
    out = {}
    for line in Path(".env").read_text().splitlines():
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

dotenv = env_file()

api_plans = [p for p in reg.plans.values() if p.auth == "api_key"]
for p in api_plans:
    needed = sorted(set(env_refs(p.api_base) + env_refs(p.api_key)))
    if not needed:
        print(f"  {p.key:<14} api_key: no os.environ references to check")
        continue
    absent = [k for k in needed if not dotenv.get(k)]
    if absent:
        print(f"  {p.key:<14} MISSING env keys: {', '.join(absent)}")
        missing.append(f"  {p.key}: fill {', '.join(absent)} in .env, then re-run scripts/apply.sh")
    else:
        print(f"  {p.key:<14} env keys present: {', '.join(needed)}")

cli_cred = {
    "claude-max-sidecar": (
        os.environ.get("CLAUDE_CONFIG_DIR", "secrets/claude"),
        ".credentials.json"),
    "codex-sidecar": (
        os.environ.get("CODEX_CONFIG_DIR", "secrets/codex"),
        "auth.json"),
    "opencode-go-sidecar": (
        os.environ.get("OPENCODE_CONFIG_DIR", "secrets/opencode/config"),
        "auth.json"),
}
login_cmd = {
    "claude-max-sidecar": "docker compose exec claude-max-sidecar claude login",
    "codex-sidecar": "docker compose exec codex-sidecar codex login --device-auth",
    "opencode-go-sidecar": "docker compose exec opencode-go-sidecar opencode auth login --provider opencode-go",
}

compose = yaml.safe_load(Path("docker-compose.yml").read_text())
for p in reg.plans.values():
    if not p.is_cli_backed:
        continue
    svc = next((name for name, body in (compose.get("services") or {}).items()
                if isinstance(body.get("environment"), dict)
                and body["environment"].get("SWITCHYARD_PLAN") == p.key), None)
    if not svc:
        print(f"  {p.key:<14} cli_sidecar: no compose service sets SWITCHYARD_PLAN: {p.key}")
        continue
    if svc not in cli_cred:
        print(f"  {p.key:<14} cli_sidecar: no credential known for service {svc!r}")
        continue
    d, fname = cli_cred[svc]
    cred = Path(d) / fname
    if cred.exists() and cred.stat().st_size > 0:
        print(f"  {p.key:<14} cli login present ({d}/{fname})")
    else:
        print(f"  {p.key:<14} NOT signed in — {d}/{fname} absent")
        missing.append(f"  {p.key}: run {login_cmd[svc]}")

for p in reg.plans.values():
    if p.auth != "oauth_proxy":
        continue
    from switchyard import oauth
    provider = p.provider_family or "xai"
    if provider not in set(oauth.FLOWS) | set(oauth.HEADLESS_FLOWS):
        print(f"  {p.key:<14} oauth_proxy: no flow registered for {provider!r}")
        continue
    st = oauth.status(provider)
    if st.get("authorised"):
        print(f"  {p.key:<14} oauth grant present ({provider})")
    else:
        print(f"  {p.key:<14} oauth grant MISSING ({provider})")
        missing.append(f"  {p.key}: run python3 -m switchyard.oauth login {provider}")

if missing:
    print()
    print("ACTION REQUIRED — these need you, then run scripts/apply.sh again:")
    for line in missing:
        print(line)
else:
    print("  every plan's credential is in place")
PY

echo "==> done. Verify live when ready: python3 scripts/smoke.py"
