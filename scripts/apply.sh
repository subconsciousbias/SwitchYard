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
# files, OAuth grants — and runs every missing CLI login and OAuth grant INLINE
# (scripts/auth_audit.py --run), re-auditing afterwards. Only .env gaps are
# left to you: keys typed by hand stay typed by hand.
#
# Order matters and is deliberate:
#   1. sync-env       -- append new .env.example keys BEFORE anything recreates
#   2. validate       -- a broken plans.yaml is refused while the old config runs
#   3. build (stale)  -- the only slow step, and usually skipped
#   4. compose up -d  -- creates new services, recreates what compose changed
#   5. reload.sh      -- restart the gateway, refresh the portal, wait for health
#   6. auth audit     -- run missing CLI logins inline, re-audit, report rest
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
    for s in paths:
        p = Path(s)
        if not p.exists():
            continue
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
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
# See scripts/auth_audit.py: reports every plan's credential state (existence
# only, never values), runs the missing CLI logins and OAuth grants inline, and
# re-audits; only .env gaps still need you by hand.
#
# --run exits 1 when credentials are STILL missing after the inline logins —
# report that, but do not fail the apply for it: the config is live either way,
# and a fresh clone always has .env gaps only the operator can fill. Any other
# nonzero status is an audit crash and propagates.
echo "==> auth audit"
if [ "$dry_run" -eq 1 ]; then
  python3 scripts/auth_audit.py
elif python3 scripts/auth_audit.py --run; then
  :
else
  rc=$?
  if [ "$rc" -ne 1 ]; then exit "$rc"; fi
  echo "==> apply finished — the credentials listed above are still missing"
fi

echo "==> done. Verify live when ready: python3 scripts/smoke.py"
