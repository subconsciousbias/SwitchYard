#!/usr/bin/env bash
# Apply an edit to config/plans.yaml.
#
#   scripts/reload.sh
#
# Validates the file FIRST, because the gateway generates its LiteLLM config at
# startup: restart it with a broken plans.yaml and it either fails to come up or
# quietly falls back, which is a worse place to debug from than a refused edit.
#
# Then: ask the gateway whether its router can follow this edit. The handler
# publishes a signature of everything the LiteLLM router bakes at startup
# (model strings, api_base, credentials, lanes). If the edited file produces
# the same signature the change is policy-only — caps, quotas, lane order,
# settings — and the gateway hot-swaps its registry within ~5s, so no restart.
# Only a differing signature forces `docker compose restart gateway`.
#
# Then refresh the portal's board in place, and leave the sidecars alone —
# they re-read the file themselves within 30 seconds.
#
# Nothing is lost by the restart. Learned concurrency, pacing state, cooldowns,
# session leases and usage all live in Redis.
set -euo pipefail

# Refuse to run from a git worktree. See CLAUDE.md — same reason as apply.sh:
# docker-compose.yml pins the project name to ${SWITCHYARD_PROJECT}, so every
# worktree addresses the same compose project and `docker compose restart`
# here would bounce the LIVE gateway and portal from this worktree's branch.
# Exit 2 to match the script's usage-error convention; `2>/dev/null` makes a
# non-repo invocation fail the comparison too.
if [ "$(git rev-parse --git-dir 2>/dev/null)" != "$(git rev-parse --git-common-dir 2>/dev/null)" ]; then
  echo "refusing: this is a git worktree — run from the main checkout (see CLAUDE.md)" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

plans="config/plans.yaml"
[ -f "$plans" ] || { echo "no $plans — run scripts/sync-env.sh first" >&2; exit 1; }

portal_port="$(grep -E '^PORTAL_PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
portal_port="${portal_port:-4001}"

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

echo "==> checking whether the router can follow this edit"
# The signature is computed here from the edited file and compared with the one
# the running gateway publishes (redis db 1, same as its other state). Equal
# signatures means every change is policy-only: the gateway's own watcher swaps
# it in within ~5s. Different, absent, or an unreachable redis all restart —
# guessing "policy-only" and leaving a stale router is the worse failure.
new_sig="$(python3 - "$plans" <<'PY'
import sys
sys.path.insert(0, ".")
from switchyard import models
print(models.router_signature(models.load(sys.argv[1])))
PY
)"
live_sig="$(docker compose exec -T redis redis-cli -n 1 get switchyard:router_sig 2>/dev/null || true)"

if [ -n "$live_sig" ] && [ "$new_sig" = "$live_sig" ]; then
  echo "==> policy-only change — gateway hot-swaps within ~5s, no restart"
  # Give the watcher a cycle plus margin, then confirm the gateway actually
  # applied it; a gateway stuck on the OLD file (bad perms, broken yaml) would
  # otherwise look done. The signature alone cannot prove the swap — the
  # watcher re-publishes the running signature even when it refuses a
  # router-shaped edit — so the gateway log is the tiebreaker.
  sleep 8
  # Captured, not piped: grep -q exits at the first match, docker compose logs
  # then dies on SIGPIPE, and pipefail turns "found it" into "not found" —
  # which would restart on every successful hot-swap.
  gwlog="$(docker compose logs --since 30s gateway 2>/dev/null || true)"
  if printf '%s' "$gwlog" | grep -q "reloaded in place"; then
    echo "    gateway: hot-swapped"
  elif printf '%s' "$gwlog" | grep -q "KEEPING the current"; then
    echo "    gateway refused the swap (router-shaped) — restarting"
    restart=1
  else
    echo "    gateway did not report a swap — restarting to be safe"
    restart=1
  fi
else
  echo "==> router-shaped change (or no live signature) — restarting the gateway"
  restart=1
fi

if [ "${restart:-0}" -eq 1 ]; then
docker compose restart gateway >/dev/null

echo -n "==> waiting for the gateway "
up=0
for _ in $(seq 1 60); do
  if curl -fsS -m 2 http://localhost:4000/health/liveliness >/dev/null 2>&1; then
    up=1
    echo "— up"
    break
  fi
  echo -n "."
  sleep 2
done
# Trust the loop rather than probing again: liveliness flaps for a moment while
# workers come up, and a second probe turned a healthy restart into a failure.
if [ "$up" -ne 1 ]; then
  echo
  echo "gateway did not come back; check: docker compose logs --tail=40 gateway" >&2
  exit 1
fi
fi

echo "==> refreshing the portal board"
reload_json="$(mktemp -t switchyard-reload.XXXXXX.json)"
if curl -fsS -m 10 -X POST "http://localhost:${portal_port}/admin/reload" \
     -o "$reload_json" 2>/dev/null; then
  python3 -c "
import json
d = json.load(open('$reload_json'))
print(f\"    portal: {d['plans']} plans, lanes {', '.join(d['lanes'])}\")"
else
  echo "    portal not reachable on :${portal_port} (skipped — it reloads on restart too)"
fi
rm -f "$reload_json"

echo "==> sidecars re-read $plans themselves within 30s"
echo "done."
