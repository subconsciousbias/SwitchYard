"""Classify a sidecar's /health doc as `idle`, `busy`, `untracked` or `unknown`.

Read by `scripts/apply.sh` to decide whether a sidecar can be recreated right
now (the idle batch) and, for one that cannot, to poll until it can (the
drain). `idle` means no model turn is running -- parked mcp_bridge sessions
waiting on a caller-side tool result do NOT count, see `classify` for why
that is safe. `busy` means a turn is running. `untracked` is a live /health
without turn bookkeeping (the token-proxy); apply.sh relies on the gateway's
`sy:inflight:{plan}` for it. `unknown` covers everything this script cannot
read: the endpoint is unreachable (mid-recreate, network flap), the exec
timed out, or a counter is malformed -- apply.sh treats it as `busy`, on the
principle that we would rather wait than race a recreate.

Always exits 0: the bash caller consumes the printed token and never has
to catch a non-zero exit. The print is exactly one of `idle`, `busy`,
`untracked`, `unknown`, followed by a single newline.

Bash wires it up like this (the same `docker compose exec -T` + `python3 -`
shape the compose healthchecks already use, since both sidecar and
token-proxy images ship `python3`). The outer `timeout` is conditional
because macOS has no `timeout(1)` (CLAUDE.md:179-180); the script
probes for `timeout` first, then `gtimeout`, and falls back to
unbounded per CLAUDE.md's documented last resort:

    timeout_cmd=""
    if command -v timeout >/dev/null 2>&1; then
      timeout_cmd="timeout 5"
    elif command -v gtimeout >/dev/null 2>&1; then
      timeout_cmd="gtimeout 5"
    fi
    $timeout_cmd docker compose exec -T "$svc" python3 - < scripts/health_idle.py \
        2>/dev/null || echo unknown

`SIDECAR_PORT` is set per service in docker-compose.yml (8081, 8082, 8084,
8085 for the sidecars, 8090 for the token-proxy — mirroring PROXY_PORT).
If a service is ever added without `SIDECAR_PORT`, the empty-port
fallback prints `unknown`; the missing-`in_flight` branch is what
classifies a properly-env'd token-proxy drain.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error


def _fetch(port: str) -> dict | None:
    """GET http://127.0.0.1:$port/health with a 2s timeout. Returns the
    parsed JSON doc on a 2xx, None on any failure (timeout, connection
    refused, non-2xx, non-JSON body).
    """
    url = f"http://127.0.0.1:{port}/health"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError):
        return None


def classify(doc: dict | None) -> str:
    """Map a /health doc to `idle`, `busy`, `untracked`, or `unknown`.

    The question is "is a model turn running right now?", not "does this
    sidecar hold any session at all". It is answered by `in_flight`, the
    bridge's gate-slot count:

      * cli_bridge holds a slot for exactly the length of one CLI call.
      * mcp_bridge holds a slot while a turn runs and gives it back when the
        session PARKS for a caller-side tool result (`park_session` in
        mcp_bridge/server.py), so `in_flight` counts running turns only.

    A parked session is deliberately NOT busy. In an agentic loop some session
    is almost always parked, so counting parked sessions (the old
    `in_flight + sessions + awaiting_followup` sum) kept a sidecar "busy" for
    as long as the session TTL and made every drain wait out its whole grace.
    Recreating the sidecar during a tool-call gap loses no work: the caller's
    follow-up names a session the new process does not have, and
    `handle_followup` routes it to `resume_gone_session`, which rebuilds the
    session from the request -- the request carries the whole history, tool
    results included (MCP_REBUILD_LOST, on by default; foreign ids rebuild
    even with it off). The cost is re-processing the prompt once. A follow-up
    already on its way (queued in `unpark_session`) is a gateway request in
    flight, so apply.sh's `ZCARD sy:inflight:{plan}` check catches it before
    this probe is consulted.

    `untracked`: the doc is a real /health answer but carries no `in_flight`
    -- the token-proxy, which is a pass-through with no turn or session
    state of its own. Its in-flight requests are counted by the gateway's
    `sy:inflight:{plan}`, which the caller checks separately, so there is
    nothing more to wait for here.

    `unknown`: no doc at all (unreachable, timed out, non-2xx, not JSON) or a
    malformed counter. The caller treats it as busy.
    """
    if not isinstance(doc, dict):
        return "unknown"
    if "in_flight" not in doc:
        return "untracked"
    try:
        in_flight = int(doc.get("in_flight") or 0)
    except (TypeError, ValueError):
        return "unknown"
    return "idle" if in_flight == 0 else "busy"


def main() -> int:
    port = os.environ.get("SIDECAR_PORT") or ""
    if not port:
        print("unknown")
        return 0
    print(classify(_fetch(port)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
