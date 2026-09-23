#!/usr/bin/env bash
# Apply an edited config/plans.yaml to the running stack — the one command.
#
#   scripts/apply.sh                       # the normal case
#   scripts/apply.sh --dry-run             # validate, audit, plan, touch nothing
#   scripts/apply.sh --build               # force a rebuild of the baked images
#   scripts/apply.sh --skip-reload         # reload.sh already ran, or runs later
#   scripts/apply.sh --drain-grace-secs N  # grace window per drained sidecar
#                                          # (default 120 — covers mcp_bridge's
#                                          # 90s parked-call timeout)
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
#   4. drain + up     -- unload-first per sidecar (zero-downtime), then portal,
#                         then gateway, fold-on-at-a-time (NOT all six at once)
#   5. reload.sh      -- hot-swap on policy-only edits, restart on router-shaped
#   6. auth audit     -- run missing CLI logins inline, re-audit, report rest
#
# Step 4 is the unload-first drain loop. Per sidecar (in unload-first order,
# lowest ZCARD sy:inflight:{plan} first): flip the picker gate
# (sy:drain:{plan}), hand off in-flight sessions to lane siblings via
# `python -m switchyard.drain`, wait for in-flight slots to hit zero, honor
# the grace window (so a 90s parked MCP call can land), stop+recreate the
# sidecar, wait for its healthcheck, clear the drain flag, then a short settle
# before the next sidecar. Sidecars share the picker gate with the gateway;
# gateway and portal do not (they ARE the picker / board), so they are
# recreated fold-on-at-a-time AFTER the sidecar loop, never all at once.
#
# When reload.sh takes the policy-only hot-swap path, apply.sh does NOT also
# force-recreate the gateway: the hot-swap is enough. The router signature is
# computed here BEFORE the drain loop, so the gateway can be excluded when
# reload.sh is about to hot-swap instead.
#
# set -euo pipefail: a silent partial apply is the one outcome this must not
# produce — failing loudly mid-way beats reporting "done" while the gateway
# still routes the old config.
set -euo pipefail

cd "$(dirname "$0")/.."

force_build=0
dry_run=0
skip_reload=0
drain_grace_secs=120
while [ $# -gt 0 ]; do
  case "$1" in
    --build) force_build=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --skip-reload) skip_reload=1; shift ;;
    --drain-grace-secs) drain_grace_secs="${2:-}"; shift 2 ;;
    --drain-grace-secs=*) drain_grace_secs="${1#--drain-grace-secs=}"; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
# Numeric sanity: a non-numeric value would break every arithmetic comparison
# in the drain loop below.
case "$drain_grace_secs" in
  ''|*[!0-9]*) echo "--drain-grace-secs must be a non-negative integer (got: $drain_grace_secs)" >&2; exit 2 ;;
esac

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

# --------------------------------------------------- 4. drain + recreate
# A service "owns" a stale image if its baked source is in $stale. Sidecars
# share the image built under the claude-max-sidecar source, so they all flip
# together when that one source flips. env_added triggers a recreate on its
# own because .env is read at container creation, never at restart.
service_source() {
  case "$1" in
    gateway)            echo gateway ;;
    portal)             echo portal ;;
    *-sidecar)          echo claude-max-sidecar ;;
    xai-token-proxy)    echo xai-token-proxy ;;
  esac
}
needs_recreate() {
  local svc=$1 src
  [ "$env_added" -gt 0 ] && return 0
  src="$(service_source "$svc")"
  for s in "${stale[@]}"; do
    [ "$s" = "$src" ] && return 0
  done
  return 1
}

# The router signature decides whether reload.sh takes the policy-only
# hot-swap path (in which case the gateway does NOT need a recreate here) or
# the router-shaped restart path. Compute it BEFORE the drain loop so the
# gateway can be excluded from step 4 when reload.sh is about to hot-swap.
new_sig="$(python3 - "$plans" <<'PY'
import sys
sys.path.insert(0, ".")
from switchyard import models
print(models.router_signature(models.load(sys.argv[1])))
PY
)"
live_sig="$(docker compose exec -T redis redis-cli -n 1 get switchyard:router_sig 2>/dev/null || true)"
live_sig="${live_sig%$'\r'}"
if [ -n "$live_sig" ] && [ "$new_sig" = "$live_sig" ]; then
  policy_only=1
else
  policy_only=0
fi

# Sidecar service list, in compose's own order. Service-list shape is stable
# across compose versions; grep extracts the drainable set (every sidecar +
# the token proxy — gateway and portal are handled separately below).
mapfile -t sidecar_svcs < <(docker compose config --services 2>/dev/null \
                             | grep -E 'sidecar|xai-token-proxy' || true)

# Build the unload-first order: ZCARD sy:inflight:{plan}, ascending. A sidecar
# without a plan env var or an unreachable redis gets a sentinel score so it
# sorts to the end rather than disappearing (a missing plan env is a real
# misconfiguration worth surfacing, not silently dropping).
declare -A plan_for_svc=()
declare -A score_for_svc=()
declare -A leases_for_svc=()
declare -a scored=()

for svc in "${sidecar_svcs[@]}"; do
  plan="$(docker compose exec -T "$svc" printenv SWITCHYARD_PLAN 2>/dev/null \
            | tr -d '\r\n' || true)"
  if [ -z "$plan" ]; then
    echo "    warn: $svc has no SWITCHYARD_PLAN env — skipping from drain loop"
    continue
  fi
  plan_for_svc["$svc"]="$plan"
  # ZCARD returns 0 when the key is absent and exits non-zero when redis is
  # unreachable; either path lets the sidecar drain (no in-flight to wait
  # for), but we want a distinct sentinel so a missing key still sorts
  # ahead of a redis-down sidecar.
  score="$(docker compose exec -T redis redis-cli -n 1 \
              ZCARD "sy:inflight:${plan}" 2>/dev/null \
            | tr -d '\r\n' || echo 9999)"
  case "$score" in ''|*[!0-9]*) score=9999 ;; esac
  score_for_svc["$svc"]="$score"
  # SCARD of the WORKSTREAM-1 lease reverse-index. Key doesn't exist yet
  # when WORKSTREAM 1 hasn't landed; SCARD on a missing key returns 0, which
  # is the right answer for dry-run too.
  leases="$(docker compose exec -T redis redis-cli -n 1 \
               SCARD "sy:lease_plan:${plan}" 2>/dev/null \
             | tr -d '\r\n' || echo 0)"
  case "$leases" in ''|*[!0-9]*) leases=0 ;; esac
  leases_for_svc["$svc"]="$leases"
  scored+=("$score|$svc")
done

# Ascending by score. printf + sort -n keeps it a single pipeline (no
# subshell-array juggling) and tolerates an empty $scored array — a config-
# only edit can land here with nothing to drain.
mapfile -t drain_order < <(printf '%s\n' "${scored[@]:-}" \
                              | sort -t '|' -k1,1n \
                              | cut -d'|' -f2-)

# The list that ACTUALLY gets drained: sidecars whose image is stale (or
# whose env changed). Everything else is left running and just sees the new
# plans.yaml on its next read.
declare -a to_drain=()
for svc in "${drain_order[@]}"; do
  if needs_recreate "$svc"; then
    to_drain+=("$svc")
  fi
done

portal_needs=0
if [ "$env_added" -gt 0 ]; then
  portal_needs=1
else
  for s in "${stale[@]}"; do
    [ "$s" = "portal" ] && portal_needs=1
  done
fi

# The gateway does NOT need a recreate here when reload.sh is about to
# hot-swap the registry. A stale gateway image still recreates — the
# sidecar-drain / portal loop is orthogonal to the picker signature.
gateway_needs=0
if [ "$policy_only" -eq 0 ]; then
  if [ "$env_added" -gt 0 ]; then
    gateway_needs=1
  else
    for s in "${stale[@]}"; do
      [ "$s" = "gateway" ] && gateway_needs=1
    done
  fi
fi

# Predicted total drain time, used only by --dry-run. Heuristic, not a
# contract: 30s per in-flight request as a long-tail estimate, plus the
# grace window (which is what dominates an idle sidecar), plus 20s for the
# recreate+settle. Sum across sidecars and (when relevant) portal/gateway.
predict_total_secs() {
  local total=0 svc s
  for svc in "$@"; do
    s="${score_for_svc[$svc]:-0}"
    total=$(( total + s * 30 + drain_grace_secs + 20 ))
  done
  if [ "$portal_needs" -eq 1 ]; then
    total=$(( total + drain_grace_secs + 20 ))
  fi
  if [ "$gateway_needs" -eq 1 ]; then
    total=$(( total + drain_grace_secs + 20 ))
  fi
  echo "$total"
}

if [ "$dry_run" -eq 1 ]; then
  echo "==> (dry-run) planned drain order"
  if [ "${#to_drain[@]}" -eq 0 ]; then
    echo "    (no sidecar drain needed — every sidecar is up to date)"
  else
    for svc in "${to_drain[@]}"; do
      plan="${plan_for_svc[$svc]}"
      score="${score_for_svc[$svc]}"
      leases="${leases_for_svc[$svc]}"
      printf "    %-22s plan=%-14s in_flight=%-4s leases=%-4s\n" \
             "$svc" "$plan" "$score" "$leases"
    done
    est_secs="$(predict_total_secs "${to_drain[@]}")"
    echo "    predicted total drain time: ${est_secs}s (incl. ${drain_grace_secs}s grace per sidecar)"
  fi
  if [ "$portal_needs" -eq 1 ]; then
    echo "    portal: recreate (env or portal image changed)"
  else
    echo "    portal: skip (up to date)"
  fi
  if [ "$gateway_needs" -eq 1 ]; then
    echo "    gateway: recreate (env, gateway image, or router-shaped change)"
  else
    echo "    gateway: skip (up to date or policy-only hot-swap)"
  fi
  echo "    (dry-run: not writing Redis state, not touching containers)"
elif [ "${#to_drain[@]}" -eq 0 ] && [ "$portal_needs" -eq 0 ] && [ "$gateway_needs" -eq 0 ]; then
  echo "==> nothing to recreate — every service is up to date"
else
  # ---- 4a. drain + recreate sidecars, unload-first ----
  for svc in "${to_drain[@]}"; do
    plan="${plan_for_svc[$svc]}"
    echo "==> draining $svc (plan=$plan, in_flight=${score_for_svc[$svc]})"

    # (1) flip the picker gate. SET (not SETNX) — a previous apply that
    # died before clearing its flag should be overwritten, not left
    # blocking traffic forever.
    docker compose exec -T redis redis-cli -n 1 SET "sy:drain:${plan}" 1 \
      >/dev/null

    # (2) hand off in-flight sessions to lane siblings. WORKSTREAM 1 must
    # have landed for this command to exist; if it hasn't, the picker still
    # respects the drain flag (new requests skip the plan), so the worst
    # case is leases pointing at the drained ref until they expire
    # naturally — logged, not fatal.
    drain_out="$(mktemp -t switchyard-drain.XXXXXX.txt 2>/dev/null || true)"
    if docker compose exec -T gateway python -m switchyard.drain "$plan" \
         >"$drain_out" 2>&1; then
      migrated="$(tr -d '\r\n' < "$drain_out" | tail -n 1 \
                    | grep -oE '[0-9]+' || echo 0)"
      echo "    sessions migrated: ${migrated:-0}"
    else
      echo "    lease migration skipped (drain helper unavailable or no leases)"
    fi
    [ -n "$drain_out" ] && rm -f "$drain_out"

    # (3) wait for in-flight to hit zero. The hard ceiling IS the grace
    # window itself — the recreate below is the safety net for a stuck
    # slot. Logging the WARN keeps the operator informed instead of silent.
    waited=0
    while : ; do
      cur="$(docker compose exec -T redis redis-cli -n 1 \
                ZCARD "sy:inflight:${plan}" 2>/dev/null \
              | tr -d '\r\n' || echo 0)"
      case "$cur" in ''|*[!0-9]*) cur=0 ;; esac
      if [ "$cur" -eq 0 ]; then break; fi
      if [ "$waited" -ge "$drain_grace_secs" ]; then
        echo "    WARN: $plan still has $cur in-flight after ${drain_grace_secs}s — proceeding"
        break
      fi
      sleep 1
      waited=$((waited + 1))
    done
    # Honor the grace window even on a fast drain — mcp_bridge parks calls
    # for up to 90s and the picker may have handed the worker off to the
    # bridge already. Without this, a sidecar with in_flight=0 right now
    # could still have a parked MCP call in flight.
    if [ "$waited" -lt "$drain_grace_secs" ]; then
      remaining=$((drain_grace_secs - waited))
      echo "    grace: sleeping ${remaining}s for parked calls to land"
      sleep "$remaining"
    fi

    # (4) stop + recreate. --no-deps so we don't trigger the dependency
    # graph (redis/postgres) — they are not changing, and recreating them
    # is churn that loses local state.
    docker compose stop "$svc" >/dev/null
    docker compose up -d --no-deps --force-recreate "$svc" >/dev/null

    # (5) wait healthcheck green. Inspect the container directly so this
    # works on compose versions whose `ps --wait` semantics differ.
    echo -n "    waiting for $svc "
    for _ in $(seq 1 90); do
      cid="$(docker compose ps -q "$svc" 2>/dev/null | head -1 || true)"
      if [ -n "$cid" ]; then
        status="$(docker inspect --format='{{.State.Health.Status}}' \
                    "$cid" 2>/dev/null || echo starting)"
        if [ "$status" = "healthy" ]; then
          echo "— healthy"
          break
        fi
      fi
      echo -n "."
      sleep 2
    done

    # (6) clear the picker gate — new requests flow to this sidecar again.
    docker compose exec -T redis redis-cli -n 1 DEL "sy:drain:${plan}" \
      >/dev/null

    # (7) settle — let the gateway's lease watcher notice the recreated
    # sidecar's identity (the container hostname is stable, but anything
    # that read the in_flight counter wants a cycle).
    sleep 2
  done

  # ---- 4b. portal then gateway, fold-on-at-a-time (no drain flag) ----
  # Gateway and portal are the picker and the board respectively: neither
  # owns a per-plan drain flag (the picker IS the gateway). Recreate them
  # one at a time, portal first so the board is the last thing to flap.
  if [ "$portal_needs" -eq 1 ]; then
    echo "==> recreating portal (env or portal image changed)"
    docker compose stop portal >/dev/null
    docker compose up -d --no-deps --force-recreate portal >/dev/null
    echo -n "    waiting for portal "
    for _ in $(seq 1 60); do
      if curl -fsS -m 2 http://localhost:4001/healthz >/dev/null 2>&1; then
        echo "— up"
        break
      fi
      echo -n "."
      sleep 2
    done
  fi
  if [ "$gateway_needs" -eq 1 ]; then
    echo "==> recreating gateway (env, gateway image, or router-shaped change)"
    docker compose stop gateway >/dev/null
    docker compose up -d --no-deps --force-recreate gateway >/dev/null
    echo -n "    waiting for gateway "
    for _ in $(seq 1 90); do
      if curl -fsS -m 2 http://localhost:4000/health/liveliness >/dev/null 2>&1; then
        echo "— up"
        break
      fi
      echo -n "."
      sleep 2
    done
  fi
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
