#!/usr/bin/env python3
"""A minimal MCP stdio server used to measure how long a tool call may block.

The whole inverted-bridge design rests on one property: when Claude Code calls an
MCP tool, can our handler *park* the call — long enough to hand it to the real
caller, let that caller execute it, and come back with a result — without the
harness giving up?

This server exposes one tool, `slow_echo`, which sleeps for a requested number of
seconds before answering. Driving `claude -p` against it with increasing delays
finds the ceiling. If the ceiling is generous, the bridge is viable; if it is a
few seconds, the design is dead and worth knowing early.

Deliberately dependency-free: JSON-RPC over stdio, one frame per line.
"""
from __future__ import annotations

import json
import sys
import time

PROTOCOL_VERSION = "2024-11-05"
PROGRESS_INTERVAL = 20.0

TOOLS = [
    {
        "name": "slow_echo",
        "description": ("Echo the given text back after sleeping. Use exactly the "
                        "delay the user asks for."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "text to echo back"},
                "delay": {"type": "number",
                          "description": "seconds to sleep before replying"},
                "keepalive": {"type": "boolean",
                              "description": ("send progress notifications while "
                                              "parked; false measures the client's "
                                              "raw timeout")},
            },
            "required": ["text", "delay"],
        },
    }
]


def log(message: str) -> None:
    """stderr only: stdout is the protocol channel."""
    print(f"[probe] {message}", file=sys.stderr, flush=True)


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def reply(request_id, result: dict) -> None:
    send({"jsonrpc": "2.0", "id": request_id, "result": result})


def handle(message: dict) -> None:
    method = message.get("method")
    request_id = message.get("id")

    if method == "initialize":
        reply(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "switchyard-probe", "version": "0.1.0"},
        })
        return

    if method in ("notifications/initialized", "initialized"):
        return                                  # notification: no response

    if method == "tools/list":
        reply(request_id, {"tools": TOOLS})
        return

    if method == "tools/call":
        params = message.get("params") or {}
        args = params.get("arguments") or {}
        delay = float(args.get("delay") or 0)
        text = str(args.get("text") or "")
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        token = meta.get("progressToken")
        keepalive = str(args.get("keepalive", "1")).lower() not in ("0", "false", "no")
        log(f"tools/call parked for {delay}s (progressToken="
            f"{'yes' if token is not None else 'no'}, keepalive={keepalive})")
        started = time.time()
        # The point of the probe: hold the call open. In the real bridge this is
        # where we would return the tool call to the originating HTTP caller and
        # await its result, rather than sleeping.
        #
        # While sleeping, emit notifications/progress the way tool_server.py
        # does, so the probe measures the ceiling the bridge actually has. Pass
        # keepalive=false to measure the raw ceiling without it -- that is how
        # OpenCode's ~60s client timeout was found.
        if token is not None and keepalive:
            deadline = started + delay
            progress = 0
            while time.time() < deadline:
                time.sleep(min(PROGRESS_INTERVAL, max(0.0, deadline - time.time())))
                progress += 1
                send({"jsonrpc": "2.0", "method": "notifications/progress",
                      "params": {"progressToken": token, "progress": progress,
                                 "message": "probe still parked"}})
        else:
            time.sleep(delay)
        held = time.time() - started
        log(f"answering after {held:.1f}s")
        reply(request_id, {
            "content": [{"type": "text",
                         "text": f"{text} (held {held:.1f}s)"}],
            "isError": False,
        })
        return

    if request_id is not None:
        # Unknown method with an id still needs an answer, or the client hangs.
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32601, "message": f"method not found: {method}"}})


def main() -> int:
    log("started")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            log(f"unparseable frame: {line[:120]}")
            continue
        try:
            handle(message)
        except Exception as exc:                # never die on one bad frame
            log(f"handler error: {exc!r}")
    log("stdin closed, exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
