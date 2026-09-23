"""Classify a sidecar's /health doc as `idle`, `busy`, or `unknown`.

Read by `scripts/apply.sh`'s drain loop to gate the post-drain grace window.
The drain flag is set and the in-flight zset is drained before this probe
runs, so a `busy` reading is evidence that something is still parked (an
mcp_bridge parked call, or any session sitting in `awaiting_followup`); an
`idle` reading is the same evidence the 120-second `sleep` was
approximating. `unknown` covers the two cases this script cannot classify:

  * the endpoint is unreachable (the container is in the middle of a
    recreate, the network is mid-flap) — apply.sh treats this exactly like
    `busy` and sleeps the remaining grace, on the principle that we would
    rather wait than race a recreate.
  * the doc has no `in_flight` field (the token-proxy `/health` exposes
    OAuth state only, not session bookkeeping) — again treated as
    `unknown`.

Always exits 0: the bash caller consumes the printed token and never has
to catch a non-zero exit. The print is exactly one of `idle`, `busy`,
`unknown`, followed by a single newline.

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
    """Map a /health doc to `idle`, `busy`, or `unknown`.

    `unknown` covers a missing doc AND a doc without `in_flight` — the
    token-proxy never exposes that field, so its shape is structurally
    distinct from a sidecar that has gone idle. We don't fabricate a busy
    signal out of an absent field, because that would force every token-
    proxy drain to sleep the full grace and re-burn the wait this script
    exists to avoid.

    `idle` / `busy` are decided on
        in_flight + (sessions or 0) + (awaiting_followup or 0)
    so mcp_bridge's parked-but-released-slot session (a session sitting in
    `SESSIONS` with `awaiting_followup=True` — see mcp_bridge/server.py
    `park_session`) counts as busy. mcp_bridge releases the slot on park,
    so `in_flight` alone would miss it.
    """
    if not isinstance(doc, dict):
        return "unknown"
    if "in_flight" not in doc:
        return "unknown"
    try:
        in_flight = int(doc.get("in_flight") or 0)
    except (TypeError, ValueError):
        return "unknown"
    try:
        sessions = int(doc.get("sessions") or 0)
    except (TypeError, ValueError):
        sessions = 0
    try:
        awaiting_followup = int(doc.get("awaiting_followup") or 0)
    except (TypeError, ValueError):
        awaiting_followup = 0
    if in_flight + sessions + awaiting_followup == 0:
        return "idle"
    return "busy"


def main() -> int:
    port = os.environ.get("SIDECAR_PORT") or ""
    if not port:
        print("unknown")
        return 0
    print(classify(_fetch(port)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
