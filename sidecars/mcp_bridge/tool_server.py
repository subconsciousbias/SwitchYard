#!/usr/bin/env python3
"""MCP stdio server spawned BY the vendor CLI -- `claude -p --mcp-config ...`
or opencode's `mcp:` config -- the other half of the inverted bridge in
server.py.

It does not execute tools itself. `tools/list` answers from a file server.py
wrote for this one request (the caller's OpenAI tools, translated to MCP);
`tools/call` is forwarded over loopback HTTP to server.py's
/internal/tools/call, which parks it until the real HTTP caller supplies a
result. That HTTP request IS the parking mechanism: holding it open keeps this
process -- and therefore the CLI's tool loop -- blocked exactly as long as
needed, with no polling on either side.

Dependency-free like probe_server.py, which proved the underlying trick:
JSON-RPC over stdio, one frame per line, stdlib urllib for the callback (run in
a thread, since urlopen blocks).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request

PROTOCOL_VERSION = "2024-11-05"

TOOLS_FILE = os.environ["SWITCHYARD_TOOLS_FILE"]
SESSION_ID = os.environ["SWITCHYARD_SESSION_ID"]
CALLBACK_URL = os.environ["SWITCHYARD_CALLBACK_URL"].rstrip("/")
# Finite callback timeout so a bridge that goes silent or dies never leaves the
# inner CLI's urlopen blocked for a day. server.py writes
# SWITCHYARD_CALLBACK_TIMEOUT into the harness env from SESSION_TTL + 60 (see
# write_claude_mcp_config / write_opencode_dir); reading it here is the safety
# net for harnesses that only pass the documented three vars. An explicit
# SWITCHYARD_CALLBACK_TIMEOUT env value still wins. The default mirrors the
# server-side default (MCP_SESSION_TTL_SECONDS, 1800 s) plus a one-minute
# grace to outlast a parked session that the reaper is about to reap.
_raw_timeout = os.environ.get("SWITCHYARD_CALLBACK_TIMEOUT", "")
if _raw_timeout:
    CALLBACK_TIMEOUT = float(_raw_timeout)
else:
    _session_ttl = float(os.environ.get("MCP_SESSION_TTL_SECONDS", "1800"))
    CALLBACK_TIMEOUT = _session_ttl + 60
# How often to send notifications/progress for a parked call.
#
# This is not cosmetic: an MCP *client* times its own requests out, and the
# bridge's whole premise is a call parked for as long as the caller needs. Every
# OpenCode tool call is made with `resetTimeoutOnProgress: true`, so each
# progress notification restarts its ~60s timer -- verified against the shipped
# binary and against a live run, where a 90s park died with "MCP error -32001:
# Request timed out" before this existed. Progress only reaches the client if it
# supplied a progressToken, which OpenCode does; Claude Code's own ceiling is
# generous enough not to need it, and sending it there is harmless.
PROGRESS_INTERVAL = float(os.environ.get("SWITCHYARD_PROGRESS_INTERVAL", "20"))


def log(message: str) -> None:
    """stderr only: stdout is the protocol channel."""
    print(f"[tool_server {SESSION_ID[:8]}] {message}", file=sys.stderr, flush=True)


def load_tools() -> list[dict]:
    with open(TOOLS_FILE) as fh:
        return json.load(fh)


TOOLS = load_tools()


async def write_reply(payload: dict, lock: asyncio.Lock) -> None:
    # Several tools/call replies can be in flight at once (parallel tool
    # calls); serialise the actual stdout writes so two never interleave.
    async with lock:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def call_back(name: str, arguments: dict) -> dict:
    """Blocking HTTP POST, run in a thread so it does not stall the stdin
    reader. Never logs the arguments or the result -- they may carry
    whatever the caller's tool handles, which is user data we have no
    business printing."""
    body = json.dumps({"session_id": SESSION_ID, "name": name,
                        "arguments": arguments}).encode()
    req = urllib.request.Request(
        f"{CALLBACK_URL}/internal/tools/call", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=CALLBACK_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:200]
        return {"content": [{"type": "text",
                              "text": f"switchyard bridge error {exc.code}: {detail}"}],
                "isError": True}
    except urllib.error.URLError as exc:
        return {"content": [{"type": "text",
                              "text": f"switchyard bridge unreachable: {exc.reason}"}],
                "isError": True}
    except OSError as exc:
        # A connect-time timeout surfaces as urllib.error.URLError(TimeoutError)
        # (caught by the URLError branch above). A read-time timeout after
        # the connection succeeded escapes as a raw TimeoutError --
        # socket.timeout is the same class on Python 3.10+, and TimeoutError
        # is a subclass of OSError, so this single except covers both. A raw
        # OSError also covers ECONNRESET/EPIPE from a bridge that died
        # mid-call. Without this branch either escape surfaces as an
        # unhandled stderr traceback in the tool thread that the CLI may
        # treat as a hard protocol error -- prefer an explicit isError reply
        # the model can act on instead, and brand it by exception class so
        # a mid-call reset does not read as a 31-minute timeout.
        if isinstance(exc, TimeoutError):
            text = (f"switchyard bridge timed out after "
                    f"{CALLBACK_TIMEOUT:.0f}s: {exc}")
        else:
            text = f"switchyard bridge connection failed: {exc}"
        return {"content": [{"type": "text", "text": text}], "isError": True}


async def keep_alive(token, lock: asyncio.Lock) -> None:
    """Restart the client's request timer while a call is parked.

    Cancelled as soon as the call resolves. `progress` only ever climbs, since
    a client may treat a decreasing value as a protocol error, and `total` is
    deliberately omitted: we do not know how long the caller will take, and
    claiming a denominator would be a lie the client might render as a bar.
    """
    progress = 0
    while True:
        await asyncio.sleep(PROGRESS_INTERVAL)
        progress += 1
        await write_reply({"jsonrpc": "2.0", "method": "notifications/progress",
                           "params": {"progressToken": token,
                                      "progress": progress,
                                      "message": "waiting for the caller's tool result"}},
                          lock)


async def handle_call(request_id, name: str, arguments: dict, lock: asyncio.Lock,
                      progress_token=None) -> None:
    log(f"tools/call {name!r} parked"
        + ("" if progress_token is not None else " (no progressToken: client "
                                                 "timeout cannot be reset)"))
    beat = (asyncio.create_task(keep_alive(progress_token, lock))
            if progress_token is not None else None)
    try:
        result = await asyncio.to_thread(call_back, name, arguments)
    finally:
        if beat is not None:
            beat.cancel()
    log(f"tools/call {name!r} resolved isError={result.get('isError', False)}")
    await write_reply({"jsonrpc": "2.0", "id": request_id, "result": result}, lock)


async def handle(message: dict, lock: asyncio.Lock) -> None:
    method = message.get("method")
    request_id = message.get("id")

    if method == "initialize":
        await write_reply({"jsonrpc": "2.0", "id": request_id, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "switchyard-mcp-bridge", "version": "0.1.0"},
        }}, lock)
        return

    if method in ("notifications/initialized", "initialized"):
        return                                  # notification: no response

    if method == "tools/list":
        await write_reply({"jsonrpc": "2.0", "id": request_id,
                            "result": {"tools": TOOLS}}, lock)
        return

    if method == "tools/call":
        params = message.get("params") or {}
        # Fire-and-forget as a task: the stdin reader keeps consuming further
        # frames while this one parks on the HTTP call, which is what lets
        # several tools/call requests be in flight at once on one stdio
        # connection -- the mechanism parallel tool calls rely on.
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        asyncio.create_task(handle_call(
            request_id, str(params.get("name") or ""),
            params.get("arguments") or {}, lock,
            progress_token=meta.get("progressToken")))
        return

    if request_id is not None:
        # Unknown method with an id still needs an answer, or the client hangs.
        await write_reply({"jsonrpc": "2.0", "id": request_id, "error": {
            "code": -32601, "message": f"method not found: {method}"}}, lock)


async def stdin_lines(queue: "asyncio.Queue[str | None]") -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            await queue.put(None)
            return
        line = line.strip()
        if line:
            await queue.put(line)


async def main_async() -> None:
    log("started")
    queue: "asyncio.Queue[str | None]" = asyncio.Queue()
    lock = asyncio.Lock()
    asyncio.create_task(stdin_lines(queue))
    while True:
        line = await queue.get()
        if line is None:
            break
        try:
            message = json.loads(line)
        except ValueError:
            log(f"unparseable frame: {line[:120]}")
            continue
        try:
            asyncio.create_task(handle(message, lock))
        except Exception as exc:                # never die on one bad frame
            log(f"handler error: {exc!r}")
    log("stdin closed, exiting")


def main() -> int:
    asyncio.run(main_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
