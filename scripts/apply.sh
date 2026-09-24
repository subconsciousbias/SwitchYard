#!/usr/bin/env bash
# Apply an edited config/plans.yaml (and/or new code) to the running stack —
# the one command.
#
#   scripts/apply.sh                       # the normal case
#   scripts/apply.sh --fast                # build what changed, `up -d`, exit —
#                                          # no drain, no waits, no reload/audit
#   scripts/apply.sh --plan                # print what would be rebuilt and
#                                          # recreated (and why), then exit
#   scripts/apply.sh --dry-run             # validate, audit, plan, touch nothing
#   scripts/apply.sh --build               # force a rebuild of every baked image
#   scripts/apply.sh --no-build            # never rebuild, even when stale
#   scripts/apply.sh --skip-reload         # reload.sh already ran, or runs later
#   scripts/apply.sh --drain-grace-secs N  # grace window per drained sidecar
#                                          # (default 120 — covers mcp_bridge's
#                                          # 90s parked-call timeout)
#
# Why this exists: plans.yaml is MOUNTED read-only into every service, so
# editing it never needs a rebuild. A rebuild is needed only when the code that
# is baked into an image changed. Rebuilding unconditionally makes every plan
# edit look heavy and slow; skipping it when stale leaves half the stack on old
# code. So the rebuild decision is automatic and exact, per image, made by
# scripts/image_plan.py:
#
#   * each image's inputs are its Dockerfile plus every path its COPY/ADD
#     lines name (read from the Dockerfile, not a hard-coded list), hashed per
#     file into a manifest;
#   * the manifest is stamped on the image as the `switchyard.inputs` label at
#     build time (docker-compose.yml interpolates SWITCHYARD_INPUTS_<SERVICE>
#     into build.labels);
#   * an image whose label differs from the live manifest is rebuilt, and the
#     plan prints which files changed. An image with NO label (built before
#     this existed, or by a bare `docker compose build`) is rebuilt once.
#
# Containers are recreated when their image was rebuilt, when they run an older
# image than the tag, when they do not exist, or when their environment
# differs from what compose would create now — which is how a .env edit is
# caught (values are compared in memory, never printed).
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
#   3. plan + build   -- the only slow step, and usually skipped
#   4. recreate       -- idle services in ONE compose call; busy sidecars
#                         drained in parallel, each unload-first; then
#                         `docker compose up -d --no-recreate` so anything
#                         missing or stopped is started (#174)
#   5. reload.sh      -- hot-swap on policy-only edits, restart on router-shaped
#   6. auth audit     -- run missing CLI logins inline, re-audit, report rest
#
# --fast stops after step 3 plus a plain `docker compose up -d` of the changed
# services: no idle probe, no drain, no health wait, no reload.sh, no audit.
#
# Step 4 is the unload-first drain loop. Every service to recreate is first
# classified idle or busy with the same signal the drain has always used:
# ZCARD sy:inflight:{plan} (0 = nothing in flight at the gateway) and
# `scripts/health_idle.py`, piped into the sidecar via
# `docker compose exec -T ... python3 -`, which classifies the `/health`
# doc as `idle`, `busy`, `untracked` or `unknown`. `idle` means no model turn
# is running — a PARKED mcp_bridge session (waiting on a caller-side tool
# result, slot released) is idle, because its follow-up is rebuilt from the
# request by a recreated sidecar (see health_idle.py `classify`); `untracked`
# is the token-proxy's /health, which has no turn state, so the ZCARD is the
# whole answer. `busy` (a turn is running) and `unknown` (the probe was
# inconclusive: unreachable, or `timeout` killed the exec at 5s) are treated
# as busy.
#
# Idle sidecars get their picker gate (sy:drain:{plan}) flipped, are
# re-checked (a request may have landed between the probe and the gate), and
# every still-idle service is recreated in ONE
# `docker compose up -d --no-deps --force-recreate a b c` call — compose
# recreates them concurrently. Busy sidecars each run the drain in a
# background job, in parallel: flip the gate, hand off leased sessions to lane
# siblings via `python -m switchyard.drain`, then POLL (every
# $SWITCHYARD_APPLY_POLL_SECS, default 2s) until the plan has nothing in
# flight and /health reports idle — the next tool-call gap, usually seconds —
# with --drain-grace-secs only as the upper bound (on timeout it proceeds, as
# it always has). Then stop+recreate the sidecar, poll its healthcheck, clear
# the gate. A job that fails (healthcheck timeout, XDG preflight) leaves its
# gate SET — an unhealthy sidecar is not re-admitted — and the whole apply
# exits non-zero after every job has finished (#176).
#
# The portal has no in-flight work of its own and always joins the idle batch;
# the gateway joins it only when nothing is in flight anywhere, otherwise it is
# recreated last, after the busy sidecars (its uvicorn drains in-flight
# requests on SIGTERM within stop_grace_period).
#
# When reload.sh takes the policy-only hot-swap path, apply.sh does NOT also
# force-recreate the gateway: the hot-swap is enough. The gateway is recreated
# here only when image_plan.py says so — its image was rebuilt, its container
# runs an older image, or its environment changed; a router-shaped plans.yaml
# edit alone is left to reload.sh's restart.
#
# Portable to macOS's /bin/bash 3.2: no mapfile, no associative arrays, no
# "${empty_array[@]}" under set -u. Service lists are space-separated strings
# (compose service names never contain spaces).
#
# set -euo pipefail: a silent partial apply is the one outcome this must not
# produce — failing loudly mid-way beats reporting "done" while the gateway
# still routes the old config.
set -euo pipefail

# Refuse to run from a git worktree. docker-compose.yml pins the project name
# to ${SWITCHYARD_PROJECT}, so every worktree addresses the SAME compose
# project — building or recreating here rebuilds and recreates the LIVE
# containers and mounts this worktree's ./config into them. The guard has to
# run before the `cd` and the arg parse: --git-dir and --git-common-dir
# differ only inside a worktree, and `2>/dev/null` makes a non-repo invocation
# fail the comparison and exit 2 as well.
if [ "$(git rev-parse --git-dir 2>/dev/null)" != "$(git rev-parse --git-common-dir 2>/dev/null)" ]; then
  echo "refusing: this is a git worktree — run from the main checkout (see CLAUDE.md)" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

force_build=0
no_build=0
dry_run=0
plan_only=0
fast=0
skip_reload=0
drain_grace_secs=120
while [ $# -gt 0 ]; do
  case "$1" in
    --build) force_build=1; shift ;;
    --no-build) no_build=1; shift ;;
    --fast) fast=1; shift ;;
    --plan) plan_only=1; shift ;;
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
if [ "$force_build" -eq 1 ] && [ "$no_build" -eq 1 ]; then
  echo "--build and --no-build contradict each other" >&2
  exit 2
fi

# How long to wait for a recreated service to report healthy before failing
# the apply. An env knob so the offline tests can make a never-healthy service
# fail in seconds instead of minutes.
health_wait_secs="${SWITCHYARD_APPLY_HEALTH_WAIT_SECS:-180}"
case "$health_wait_secs" in
  ''|*[!0-9]*) echo "SWITCHYARD_APPLY_HEALTH_WAIT_SECS must be an integer" >&2; exit 2 ;;
esac
# Every wait below POLLS its real condition at this interval and proceeds the
# moment it holds; the durations (drain grace, health wait) are upper bounds
# only. Fractional values are fine (`sleep 0.2`); elapsed time is measured
# with $SECONDS, never by counting sleeps.
poll_secs="${SWITCHYARD_APPLY_POLL_SECS:-2}"
case "$poll_secs" in
  ''|*[!0-9.]*|*.*.*) echo "SWITCHYARD_APPLY_POLL_SECS must be a number" >&2; exit 2 ;;
esac

plan_flags=""
[ "$force_build" -eq 1 ] && plan_flags="--force"
[ "$no_build" -eq 1 ] && plan_flags="--no-build"

# --plan is read-only: no .env sync, no validation, no Docker mutation.
if [ "$plan_only" -eq 1 ]; then
  echo "==> image plan"
  exec python3 scripts/image_plan.py plan $plan_flags
fi

plans="config/plans.yaml"
[ -f "$plans" ] || { echo "no $plans — run scripts/sync-env.sh first" >&2; exit 1; }

# Scratch space for the machine-readable plan and per-job logs. One EXIT
# handler owns every cleanup: scratch files, background drain jobs, and any
# drain gate the idle batch flipped (a background job clears its own).
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/switchyard-apply.XXXXXX")"
plan_file="$work_dir/plan.tsv"
gated_plans=""   # plans whose sy:drain gate THIS shell set and must clear
bg_pids=""
on_exit() {
  local p pid
  for pid in $bg_pids; do
    kill -TERM "$pid" >/dev/null 2>&1 || true
  done
  # ...and let each job's own EXIT trap clear its gate before we go.
  for pid in $bg_pids; do
    wait "$pid" >/dev/null 2>&1 || true
  done
  for p in $gated_plans; do
    docker compose exec -T redis redis-cli -n 1 DEL "sy:drain:${p}" >/dev/null 2>&1 || true
  done
  rm -rf "$work_dir"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ------------------------------------------------------------ plan helpers
# The TSV scripts/image_plan.py writes:
#   build    <service>  <reason>
#   service  <service>  <image owner>  <plan or ->  <container state>
#   recreate <service>  <reason>
tsv_list() {   # tsv_list KIND -> space-separated service names
  awk -F'\t' -v k="$1" '$1==k {printf "%s ", $2}' "$plan_file"
}
svc_field() {  # svc_field SERVICE N -> field N of that service's row
  awk -F'\t' -v s="$1" -v n="$2" '$1=="service" && $2==s {print $n; exit}' "$plan_file"
}
recreate_reason() {
  awk -F'\t' -v s="$1" '$1=="recreate" && $2==s {print $3; exit}' "$plan_file"
}
in_list() {    # in_list WORD "LIST"
  case " $2 " in *" $1 "*) return 0 ;; esac
  return 1
}

# ---------------------------------------------------------------- 1. sync-env
echo "==> propagating .env.example keys (appends only, never overwrites)"
scripts/sync-env.sh
# .env is read when a CONTAINER IS CREATED, not on restart. Whether a changed
# or newly appended key actually reaches a container is decided exactly in
# step 3: image_plan.py compares each container's environment with what
# compose would create now, and marks the drifted ones for recreation.

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

# ---------------------------------------------------------- 3. plan + build
echo "==> image plan"
python3 scripts/image_plan.py plan $plan_flags --tsv "$plan_file"
build_svcs="$(tsv_list build)"
build_svcs="${build_svcs% }"
recreate_svcs="$(tsv_list recreate)"
recreate_svcs="${recreate_svcs% }"

if [ -n "$build_svcs" ]; then
  echo "==> rebuilding: $build_svcs"
  if [ "$dry_run" -eq 0 ]; then
    # shellcheck disable=SC2086  # word-splitting the service list is intended
    python3 scripts/image_plan.py build $build_svcs
  else
    echo "    (dry-run: not building)"
  fi
else
  echo "==> no image needs a rebuild (plans.yaml is mounted, so a config edit never does)"
fi

# ------------------------------------------------------------------- --fast
# Everything below the build is waiting of one kind or another. --fast skips
# all of it: recreate what the plan named in one compose call, start anything
# missing, and exit as soon as compose returns.
if [ "$fast" -eq 1 ]; then
  if [ "$dry_run" -eq 1 ]; then
    [ -n "$recreate_svcs" ] && echo "==> (dry-run) would recreate: $recreate_svcs"
    echo "==> (dry-run) would run: docker compose up -d --no-build --no-recreate"
    exit 0
  fi
  if [ -n "$recreate_svcs" ]; then
    echo "==> (fast) recreating: $recreate_svcs"
    # shellcheck disable=SC2086
    docker compose up -d --no-deps --no-build --force-recreate $recreate_svcs
  fi
  echo "==> (fast) starting anything not running"
  docker compose up -d --no-build --no-recreate
  echo "==> done (fast): no drain, no health wait, no reload.sh, no auth audit."
  echo "    a router-shaped plans.yaml edit still needs scripts/reload.sh;"
  echo "    policy-only edits hot-swap by themselves within ~5s."
  exit 0
fi

# --------------------------------------------------------------- 4. recreate
# ---- idle signal: the same two readings the drain has always used ----
inflight() {   # inflight PLAN -> ZCARD sy:inflight:{plan}; 9999 if unreadable
  local n
  n="$(docker compose exec -T redis redis-cli -n 1 \
          ZCARD "sy:inflight:${1}" 2>/dev/null \
        | tr -d '\r\n' || echo 9999)"
  case "$n" in ''|*[!0-9]*) n=9999 ;; esac
  echo "$n"
}
leases_of() {  # SCARD of the lease reverse-index; 0 when absent
  local n
  n="$(docker compose exec -T redis redis-cli -n 1 \
          SCARD "sy:lease_plan:${1}" 2>/dev/null \
        | tr -d '\r\n' || echo 0)"
  case "$n" in ''|*[!0-9]*) n=0 ;; esac
  echo "$n"
}
total_inflight() {  # every plan's sy:inflight:{plan} (not the :m:/:lane: keys)
  local keys k n total=0
  keys="$(docker compose exec -T redis redis-cli -n 1 --scan --pattern 'sy:inflight:*' \
            2>/dev/null | tr -d '\r' || echo "?")"
  [ "$keys" = "?" ] && { echo 9999; return; }
  for k in $keys; do
    case "$k" in sy:inflight:m:*|sy:inflight:lane:*) continue ;; esac
    n="$(inflight "${k#sy:inflight:}")"
    total=$((total + n))
  done
  echo "$total"
}
# `timeout` guards the OUTER `docker compose exec`, which has no built-in
# deadline. `health_idle.py` itself bounds the urllib fetch at 2s, but a
# wedged docker daemon, mid-recreate container, or stuck network namespace
# would otherwise hang the caller indefinitely. macOS has no `timeout(1)`
# (see CLAUDE.md), so probe for it and fall back to `gtimeout` (coreutils),
# then to unbounded per CLAUDE.md's documented last resort. Any non-zero exit
# routes to `unknown` — the conservative answer. Fail-open to patience.
health_probe() {  # health_probe SERVICE -> idle | busy | untracked | unknown
  local svc="$1" timeout_cmd="" health_state
  if command -v timeout >/dev/null 2>&1; then
    timeout_cmd="timeout 5"
  elif command -v gtimeout >/dev/null 2>&1; then
    timeout_cmd="gtimeout 5"
  fi
  health_state="$(
      $timeout_cmd docker compose exec -T "$svc" python3 - < scripts/health_idle.py \
        2>/dev/null || echo unknown)"
  case "$health_state" in idle|busy|untracked) echo "$health_state" ;; *) echo unknown ;; esac
}
# idle = nothing in flight at the gateway for this plan AND no model turn
# running in the sidecar. Parked mcp_bridge sessions do not count: a follow-up
# that reaches a recreated sidecar is rebuilt from its request (see
# scripts/health_idle.py `classify`). `untracked` (token-proxy) has no turn
# state of its own, so the gateway's count is the whole answer.
is_idle() {  # is_idle SERVICE -> 0 when no model turn is running
  local svc="$1" plan state
  state="$(svc_field "$svc" 5)"
  [ "$state" = "running" ] || return 0       # nothing running = nothing to drain
  case "$svc" in
    portal) return 0 ;;                        # the board holds no requests
    gateway) [ "$(total_inflight)" -eq 0 ]; return ;;
  esac
  plan="$(svc_field "$svc" 4)"
  [ -n "$plan" ] && [ "$plan" != "-" ] || return 0
  [ "$(inflight "$plan")" -eq 0 ] || return 1
  case "$(health_probe "$svc")" in idle|untracked) return 0 ;; esac
  return 1
}

# ---- readiness: what "came back" means per service ----
svc_ready() {
  local svc="$1" cid status
  case "$svc" in
    portal)  curl -fsS -m 2 http://localhost:4001/healthz >/dev/null 2>&1; return ;;
    gateway) curl -fsS -m 2 http://localhost:4000/health/liveliness >/dev/null 2>&1; return ;;
  esac
  cid="$(docker compose ps -q "$svc" 2>/dev/null | head -1 || true)"
  [ -n "$cid" ] || return 1
  # A service without a healthcheck has no .State.Health: "running" is the
  # best evidence there is.
  status="$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
              "$cid" 2>/dev/null || echo starting)"
  [ "$status" = "healthy" ] || [ "$status" = "running" ]
}
# wait_ready "SVC SVC ..." -> 0 when all are ready; else prints the laggards
# and returns 1. Polls every $poll_secs up to $health_wait_secs, returning the
# moment the last one is up. Never falls through on timeout (#176): the caller
# fails the apply. One progress line every ~10s, not one per poll.
wait_ready() {
  local pending="$1" started=$SECONDS last_note=-10 elapsed next svc
  while : ; do
    next=""
    for svc in $pending; do
      svc_ready "$svc" || next="$next $svc"
    done
    pending="${next# }"
    [ -z "$pending" ] && return 0
    elapsed=$((SECONDS - started))
    if [ "$elapsed" -ge "$health_wait_secs" ]; then
      echo "    NOT healthy after ${health_wait_secs}s: $pending" >&2
      return 1
    fi
    if [ $((elapsed - last_note)) -ge 10 ]; then
      echo "    waiting for $pending to become healthy (${elapsed}s)"
      last_note=$elapsed
    fi
    sleep "$poll_secs"
  done
}
# XDG preflight — exercise the real failure mode (`opencode --version` mkdirs
# `$XDG_STATE_HOME`; on a recreated image that forgot to bake the node-owned
# XDG tree it EACCESes and the bridge then 502s every request for the lifetime
# of this container). compose exec runs as the service's `user: node`, so a
# failure here is exactly the failure the CLI would hit.
xdg_preflight() {  # xdg_preflight SERVICE PLAN -> 0 ok, 1 failed (message printed)
  local svc="$1" plan="$2" preflight_cmd preflight_out
  # shellcheck disable=SC2016  # expanded inside the container, not here
  preflight_cmd='mkdir -p "${XDG_STATE_HOME:-$HOME/.local/state}"'
  if ! preflight_out="$(docker compose exec -T "$svc" \
                          sh -c "$preflight_cmd" 2>&1)"; then
    cat >&2 <<EOF
    preflight failed for $svc (plan=$plan):
      $preflight_out

    root cause: the recreated sidecar image is missing the node-owned
    XDG state directory. the vendor CLI's first call mkdirs
    \$XDG_STATE_HOME; on a bind-mounted parent that is root-owned
    (Docker materialises the missing path at container creation),
    user=node EACCESes and every subsequent call would 502.

    remediation: rerun from the main checkout so the sidecar image is
    rebuilt from the fixed Dockerfile.sidecar (which bakes in the
    mkdir + chown of the four XDG parents):
        scripts/apply.sh --build

    sy:drain:${plan} is still set, so this plan is NOT re-admitted
    until the next apply run lands successfully.
EOF
    return 1
  fi
  return 0
}

# ---- drain one busy sidecar (ALWAYS run as a background job: `drain_one x &`,
# so its traps and `exit`s stay inside that job's subshell) ----
# The drain flag is a TTL'd sentinel: EX (grace + 10 minutes) so an
# un-trap-able SIGKILL self-heals, and the job's EXIT trap DELs the flag on
# any other exit path so a Ctrl-C or SIGTERM mid-grace doesn't strand the
# plan. INT/TERM are routed through `exit N` so the subshell unwinds through
# the EXIT trap; `|| true` + `2>/dev/null` keep set -e from letting a failing
# DEL (redis already down) clobber the exit status. A job that fails its
# healthcheck or preflight disarms the trap on purpose: the gate stays SET
# (until its TTL) so a sidecar that would 502 is not re-admitted.
drain_one() {
  local svc="$1" plan
  plan="$(svc_field "$svc" 4)"
  # A brace group, not a subshell: the job's pid (what the parent signals and
  # waits for) must be the shell that owns the EXIT trap, or the parent can
  # exit before a killed job has cleared its gate.
  {
    # shellcheck disable=SC2329  # invoked by the EXIT trap
    clear_drain_flag() {
      trap '' INT TERM   # a second signal (parent's forward) must not cut the DEL short
      [ -n "${draining_plan:-}" ] || return 0
      docker compose exec -T redis redis-cli -n 1 DEL "sy:drain:${draining_plan}" >/dev/null 2>&1 || true
    }
    draining_plan=""
    trap clear_drain_flag EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    echo "==> draining $svc (plan=$plan, in_flight=$(inflight "$plan"))"

    # (1) flip the picker gate. SET (not SETNX) — a previous apply that
    # died before clearing its flag should be overwritten, not left
    # blocking traffic forever. The TTL covers the SIGKILL case (no
    # trap can run): the flag expires on its own within the grace
    # window plus ten minutes of slack.
    # Armed BEFORE the SET: a signal landing between the SET and the next
    # line must still DEL (a DEL of a flag not yet set is harmless).
    draining_plan="$plan"
    docker compose exec -T redis redis-cli -n 1 SET "sy:drain:${plan}" 1 EX "$((drain_grace_secs + 600))" \
      >/dev/null

    # (2) hand off in-flight sessions to lane siblings. If the gateway
    # image predates switchyard/drain.py the picker still respects the
    # drain flag (new requests skip the plan), so the worst case is
    # leases pointing at the drained ref until they expire naturally —
    # logged, not fatal.
    drain_out="$(mktemp "$work_dir/drain.XXXXXX" 2>/dev/null || true)"
    if docker compose exec -T gateway python -m switchyard.drain "$plan" \
         >"$drain_out" 2>&1; then
      # Parse ONLY the final summary line — `drain.migrate` prints per-session
      # audit lines above it ("{session}: {plan}/{label} -> {ref}"), and a
      # naive `grep -oE '[0-9]+'` would pull digits from session ids, plan
      # labels and refs (a `5g` plan's first match is the plan name, not the
      # count). The summary line has the exact shape
      #   `migrated <count> session(s) off <plan>`
      # so awk on `^migrated ` with $2 as the count is the safe shape.
      migrated="$(awk '/^migrated [0-9]+ session/ {print $2; exit}' "$drain_out")"
      [ -n "$migrated" ] || migrated=0
      echo "    sessions migrated: ${migrated}"
    else
      # Surface the helper's output so the operator can tell "no leases
      # to migrate" from "the gateway image predates switchyard/drain.py"
      # (the latter is fixable with --build; the former is a no-op).
      # set -euo pipefail is active, so every pipeline is guarded with
      # || true — the rm below is fine, the tail+grep below would
      # otherwise abort the job when $drain_out is empty.
      echo "    WARN: lease migration skipped — drain helper exited non-zero; output (tail):"
      tail -n 8 "$drain_out" 2>/dev/null \
        | sed 's/^/        /' || true
      if grep -q "No module named switchyard.drain" "$drain_out" 2>/dev/null; then
        echo "        hint: the gateway image predates switchyard/drain.py — rerun with --build to rebuild it"
      fi
    fi
    [ -n "$drain_out" ] && rm -f "$drain_out"

    # (3) wait for the first moment no model turn is running: the gateway
    # has nothing in flight for this plan (ZCARD) and the sidecar's /health
    # says idle. POLLED every $poll_secs — in an agentic loop that is the
    # next tool-call gap, usually seconds away — with the grace window only
    # as the upper bound. On timeout it proceeds anyway, as it always has:
    # the recreate is the safety net for a stuck slot.
    started=$SECONDS
    last_note=-10
    while : ; do
      cur="$(inflight "$plan")"
      [ "$cur" -eq 9999 ] && cur=0   # unreadable redis: nothing to wait on
      if [ "$cur" -eq 0 ]; then
        health_state="$(health_probe "$svc")"
      else
        health_state="in-flight"
      fi
      waited=$((SECONDS - started))
      case "$health_state" in
        idle|untracked)
          echo "    drained after ${waited}s ($svc /health reports nothing in flight)"
          break
          ;;
      esac
      if [ "$waited" -ge "$drain_grace_secs" ]; then
        echo "    WARN: $svc still busy after ${drain_grace_secs}s (in_flight=$cur, /health: $health_state) — proceeding"
        break
      fi
      if [ $((waited - last_note)) -ge 10 ]; then
        echo "    waiting for $svc to drain ($cur in flight, /health: $health_state, ${waited}s)"
        last_note=$waited
      fi
      sleep "$poll_secs"
    done

    # (4) stop + recreate. --no-deps so we don't trigger the dependency
    # graph (redis/postgres) — they are not changing, and recreating them
    # is churn that loses local state.
    docker compose stop "$svc" >/dev/null
    docker compose up -d --no-deps --no-build --force-recreate "$svc" >/dev/null

    # (5) wait healthcheck green; a timeout FAILS the job (#176) and
    # leaves the gate set.
    if ! wait_ready "$svc"; then
      echo "    $svc did not become healthy; sy:drain:${plan} left SET (not re-admitted)." >&2
      echo "    check: docker compose logs --tail=40 $svc" >&2
      draining_plan=""
      exit 1
    fi
    echo "    $svc healthy"
    # (5b) XDG preflight; on failure keep the gate set and fail.
    if ! xdg_preflight "$svc" "$plan"; then
      draining_plan=""
      exit 1
    fi

    # (6) clear the picker gate — new requests flow to this sidecar again.
    docker compose exec -T redis redis-cli -n 1 DEL "sy:drain:${plan}" \
      >/dev/null
    draining_plan=""
  }
}

# ---- partition what the plan says to recreate ----
# The gateway is excluded only when reload.sh will hot-swap AND neither its
# image nor its environment changed (then it is simply not in the plan).
idle_svcs=""
busy_svcs=""       # sidecars: drained in parallel
gateway_late=0     # busy gateway: recreated after the busy sidecars
for svc in $recreate_svcs; do
  if is_idle "$svc"; then
    idle_svcs="$idle_svcs $svc"
  elif [ "$svc" = "gateway" ]; then
    gateway_late=1
  else
    busy_svcs="$busy_svcs $svc"
  fi
done
idle_svcs="${idle_svcs# }"
busy_svcs="${busy_svcs# }"

# Predicted total drain time, used only by --dry-run. Heuristic, not a
# contract: 30s per in-flight request as a long-tail estimate, plus the
# grace window (which is what dominates a drain), plus 20s for the
# recreate+settle. Busy sidecars drain in PARALLEL, so the prediction is the
# slowest one, not the sum.
predict_total_secs() {
  local worst=0 svc plan s t
  for svc in "$@"; do
    plan="$(svc_field "$svc" 4)"
    s="$(inflight "$plan")"
    [ "$s" -eq 9999 ] && s=0
    t=$(( s * 30 + drain_grace_secs + 20 ))
    [ "$t" -gt "$worst" ] && worst=$t
  done
  echo "$worst"
}

if [ "$dry_run" -eq 1 ]; then
  echo "==> (dry-run) recreate plan"
  if [ -z "$recreate_svcs" ]; then
    echo "    (nothing to recreate — every container runs its current image and environment)"
  fi
  for svc in $idle_svcs; do
    printf "    %-22s idle -> one batched recreate   (%s)\n" "$svc" "$(recreate_reason "$svc")"
  done
  for svc in $busy_svcs; do
    plan="$(svc_field "$svc" 4)"
    printf "    %-22s busy -> drain  plan=%-14s in_flight=%-4s leases=%-4s\n" \
           "$svc" "$plan" "$(inflight "$plan")" "$(leases_of "$plan")"
  done
  if [ "$gateway_late" -eq 1 ]; then
    echo "    gateway                busy -> recreated after the drains (uvicorn drains on SIGTERM)"
  fi
  if [ -n "$busy_svcs" ]; then
    # shellcheck disable=SC2086
    echo "    predicted drain time: $(predict_total_secs $busy_svcs)s (parallel; incl. ${drain_grace_secs}s grace)"
  fi
  echo "    then: docker compose up -d --no-build --no-recreate (start anything missing)"
  echo "    (dry-run: not writing Redis state, not touching containers)"
else
  failed=""

  # ---- 4a. busy sidecars: parallel background drains ----
  # stdin from /dev/null so no job's `docker compose exec` competes for the
  # terminal; each job's output goes to its own log, printed when it ends.
  jobs_list=""
  for svc in $busy_svcs; do
    drain_one "$svc" </dev/null >"$work_dir/job.$svc.log" 2>&1 &
    bg_pids="$bg_pids $!"
    jobs_list="$jobs_list $svc:$!"
  done
  [ -n "$busy_svcs" ] && echo "==> draining in parallel (busy): $busy_svcs"

  # ---- 4b. idle services: gate, re-check, ONE compose call ----
  if [ -n "$idle_svcs" ]; then
    # Flip the gate on every idle sidecar first, then re-read the signal: a
    # request can land between the probe above and the gate. A service
    # that turned busy leaves the batch and drains like any other.
    batch=""
    for svc in $idle_svcs; do
      plan="$(svc_field "$svc" 4)"
      if [ "$(svc_field "$svc" 5)" = "running" ] && [ -n "$plan" ] && [ "$plan" != "-" ]; then
        gated_plans="$gated_plans $plan"   # registered first, as in drain_one
        docker compose exec -T redis redis-cli -n 1 SET "sy:drain:${plan}" 1 EX "$((drain_grace_secs + 600))" \
          >/dev/null
      fi
    done
    for svc in $idle_svcs; do
      if is_idle "$svc"; then
        batch="$batch $svc"
      elif [ "$svc" = "gateway" ]; then
        echo "    gateway turned busy — recreating it after the drains"
        gateway_late=1
      else
        echo "    $svc turned busy — draining it instead"
        drain_one "$svc" </dev/null >"$work_dir/job.$svc.log" 2>&1 &
        bg_pids="$bg_pids $!"
        jobs_list="$jobs_list $svc:$!"
      fi
    done
    batch="${batch# }"
    if [ -n "$batch" ]; then
      echo "==> recreating (idle, one call): $batch"
      # shellcheck disable=SC2086
      docker compose up -d --no-deps --no-build --force-recreate $batch >/dev/null
      if wait_ready "$batch"; then
        echo "    healthy: $batch"
        ok_batch="$batch"
      else
        # wait_ready printed the laggards; recompute who is ready so only
        # the healthy ones get re-admitted.
        ok_batch=""
        for svc in $batch; do
          if svc_ready "$svc"; then ok_batch="$ok_batch $svc"; else failed="$failed $svc"; fi
        done
      fi
      for svc in $ok_batch; do
        plan="$(svc_field "$svc" 4)"
        case "$svc" in gateway|portal) continue ;; esac
        [ -n "$plan" ] && [ "$plan" != "-" ] || continue
        if xdg_preflight "$svc" "$plan"; then
          docker compose exec -T redis redis-cli -n 1 DEL "sy:drain:${plan}" >/dev/null 2>&1 || true
        else
          failed="$failed $svc"
        fi
      done
      # Gates of failed services stay SET (not re-admitted); the rest are
      # cleared above. Nothing is left for on_exit to clear.
      gated_plans=""
    else
      # every idle candidate turned busy: their jobs own their gates now
      gated_plans=""
    fi
  fi

  # ---- 4c. collect the background drains ----
  for entry in $jobs_list; do
    svc="${entry%%:*}"
    pid="${entry##*:}"
    if wait "$pid"; then rc=0; else rc=$?; fi
    sed 's/^/    | /' "$work_dir/job.$svc.log" 2>/dev/null || true
    if [ "$rc" -ne 0 ]; then
      echo "    drain of $svc FAILED (exit $rc)" >&2
      failed="$failed $svc"
    fi
  done
  bg_pids=""

  # ---- 4d. a busy gateway goes last ----
  if [ "$gateway_late" -eq 1 ]; then
    echo "==> recreating gateway (in-flight requests drain on SIGTERM)"
    docker compose up -d --no-deps --no-build --force-recreate gateway >/dev/null
    wait_ready gateway || failed="$failed gateway"
  fi

  # ---- 4e. start anything missing or stopped (#174) ----
  # --no-recreate: every recreate this apply wanted is done above; this only
  # starts services with no running container (a new plan's sidecar, a
  # crashed one, a stopped redis) without touching anything that runs.
  echo "==> starting anything not running (docker compose up -d --no-recreate)"
  docker compose up -d --no-build --no-recreate

  if [ -n "$failed" ]; then
    echo "==> apply FAILED — not healthy:$failed" >&2
    echo "    each failed sidecar's sy:drain gate is left set so it takes no traffic;" >&2
    echo "    fix it and rerun scripts/apply.sh (the gate TTL is $((drain_grace_secs + 600))s)." >&2
    exit 1
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
