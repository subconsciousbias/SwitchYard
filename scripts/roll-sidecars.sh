#!/usr/bin/env bash
# Recreate sidecars ONE AT A TIME without handing a request to a sidecar that
# is going away.
#
#   scripts/roll-sidecars.sh                    # every *-sidecar service
#   scripts/roll-sidecars.sh claude-max-sidecar # just these
#
# Why: waiting for in_flight=0 and then recreating still races the gateway -
# a request picked in the gap between the check and the stop lands on a dying
# container and comes back as a 500 "Server disconnected". So each plan is
# first COOLED (the same sy:cool:{plan} key a quota wall sets, with a TTL, so
# an interrupted run can never leave a seat out for more than DRAIN_TTL), new
# picks spill to its peers, the in-flight work finishes, the container is
# recreated, and the cool key is deleted once it is healthy again.
# A session parked on the old container is rebuilt from its next request by
# the bridge (resume_gone_session), the same as after any sidecar restart.
set -euo pipefail
cd "$(dirname "$0")/.."
DRAIN_TTL="${DRAIN_TTL:-300}"
IDLE_WAIT="${IDLE_WAIT:-240}"

rcli() { docker compose exec -T redis redis-cli -n 1 "$@"; }

if [ "$#" -gt 0 ]; then svcs=("$@"); else
  mapfile -t svcs < <(docker compose config --services | grep -- '-sidecar$')
fi

for svc in "${svcs[@]}"; do
  cid="$(docker compose ps -q "$svc" | head -1)"
  if [ -z "$cid" ]; then echo "==> $svc: not running, starting"; docker compose up -d --no-deps "$svc" >/dev/null; continue; fi
  plan="$(docker exec "$cid" printenv SWITCHYARD_PLAN 2>/dev/null || true)"
  port="$(docker exec "$cid" printenv SIDECAR_PORT 2>/dev/null || echo 8081)"
  echo -n "==> $svc (plan=${plan:-?}) "
  if [ -n "$plan" ] && [ -z "$(rcli GET "sy:cool:$plan" | tr -d '\r')" ]; then
    rcli SET "sy:cool:$plan" "maintenance|$(( $(date +%s) + DRAIN_TTL ))" EX "$DRAIN_TTL" >/dev/null
    cooled=1
  else
    cooled=0      # already cooled by something real: leave that alone
  fi
  f=?
  for _ in $(seq 1 $(( IDLE_WAIT / 3 ))); do
    f="$(docker exec "$cid" python3 -c "import json,urllib.request as u;print(json.load(u.urlopen('http://localhost:$port/health'))['in_flight'])" 2>/dev/null || echo 0)"
    [ "$f" = 0 ] && break
    echo -n "."; sleep 3
  done
  echo -n " in_flight=$f -> "
  docker compose up -d --no-deps --force-recreate "$svc" >/dev/null 2>&1
  s=none
  for _ in $(seq 1 60); do
    cid="$(docker compose ps -q "$svc" | head -1)"
    s="$(docker inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo none)"
    [ "$s" = healthy ] && break; sleep 2
  done
  [ "$cooled" = 1 ] && rcli DEL "sy:cool:$plan" >/dev/null
  echo "$s"
  [ "$s" = healthy ] || { echo "$svc did not come back healthy; check: docker compose logs --tail=40 $svc" >&2; exit 1; }
done
