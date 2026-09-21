#!/usr/bin/env bash
# Apply an edit to config/plans.yaml.
#
#   scripts/reload.sh
#
# Validates the file FIRST, because the gateway generates its LiteLLM config at
# startup: restart it with a broken plans.yaml and it either fails to come up or
# quietly falls back, which is a worse place to debug from than a refused edit.
#
# Then: restart the gateway (the only way its router picks up new models, an
# api_base or a credential), refresh the portal's board in place, and leave the
# sidecars alone — they re-read the file themselves within 30 seconds.
#
# Nothing is lost by the restart. Learned concurrency, pacing state, cooldowns,
# session leases and usage all live in Redis.
set -euo pipefail

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

echo "==> restarting the gateway (its router is built at startup)"
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
