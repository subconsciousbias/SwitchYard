"""Host-mirror self-check (issue #264): prove, with THIS sidecar's own CLI
binary and THIS bridge's own argv, that a tool-path session shows the model
the caller's tools and nothing else -- and that a native tool the model
names anyway does not run.

Why a runtime check and not just tests. Every way the tool path has broken
was a property of a specific CLI release, invisible offline: codex 0.155.1's
catalog moved every model into a code-mode host that hides MCP tools and
advertises its own shell (issue #255); OpenCode 1.18.31 executes a built-in
it was only told to hide; Claude Code grew Agent/Skill/ToolSearch/
AskUserQuestion past a denylist (issues #195, #256). The pinned binary in
the image is the only thing that can answer "is this CLI locked down?".

How. A loopback fake model speaks the three wire protocols the CLIs use
(Anthropic /v1/messages, OpenAI /v1/responses, /v1/chat/completions),
records every request, and answers the first tool-bearing one with a call to
the CLI's NATIVE shell tool, then answers in text. The CLI is spawned with
the argv mcp_bridge.build_argv produces for a real session, redirected to
the fake by the provider's own base-URL mechanism and isolated config
directories -- no credential, no seat quota, no network. It passes when:

  * every tool the CLI offered the model is a bridged caller tool (plus the
    few read-only helpers a CLI adds for any MCP server: codex's
    list/read_mcp_resource);
  * the bridged tool is among them, at the top level;
  * the native shell call did not execute (its marker file does not exist).

mcp_bridge runs it once at startup and refuses tool-bearing requests with a
503 until it passes (MCP_HOST_MIRROR_CHECK=enforce, the default), so a plan
whose CLI is not locked down spills tool work to another plan instead of
serving it. `report` only records the result on /health; `off` skips it.

Run by hand inside a sidecar: `python3 /app/mcp_bridge/selfcheck.py`.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROBE_TOOL = {"name": "switchyard_selfcheck_probe",
              "description": "Host-mirror self-check probe. Never call it.",
              "inputSchema": {"type": "object", "properties": {}}}

# The native shell tool each CLI's model is trained to call, and its arguments.
# The command is RELATIVE: every CLI runs its shell in its own working
# directory, and an absolute path can trip a CLI's own directory checks
# (OpenCode's external_directory) and hide a shell that is in fact live.
MARKER = "native-tool-ran"
NATIVE_CALLS = {
    "claude": ("Bash", {"command": f"touch {MARKER}", "description": "selfcheck"}),
    "codex": ("exec_command", {"cmd": f"touch {MARKER}"}),
    "opencode": ("bash", {"command": f"touch {MARKER}", "description": "selfcheck"}),
}

# Tools a CLI adds for ANY configured MCP server. Read-only against that
# server (tool_server.py, which exposes no resources), so harmless.
# Named as the pinned codex sends them: inside its `functions` namespace.
ALLOWED_EXTRAS = {
    "codex": {"functions__list_mcp_resources",
              "functions__list_mcp_resource_templates",
              "functions__read_mcp_resource"},
}


# ---------------------------------------------------------------- fake model ---
class FakeModel:
    """Loopback model endpoint for the three protocols the CLIs speak."""

    def __init__(self, native: tuple[str, dict]):
        self.native = native
        self.requests: list[dict] = []
        self._native_sent = False
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _send(self, code: int, ctype: str, payload: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                self._send(200, "application/json",
                           json.dumps({"object": "list", "data": []}).encode())

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    body = {}
                path = self.path.split("?")[0]
                if path.endswith("/count_tokens"):
                    return self._send(200, "application/json", b'{"input_tokens": 1}')
                call = outer._next(path, body)
                if path.endswith("/messages"):
                    return self._send(200, "text/event-stream", _anthropic(body, call))
                if path.endswith("/responses"):
                    return self._send(200, "text/event-stream", _responses(body, call))
                if path.endswith("/chat/completions"):
                    return self._send(200, "text/event-stream", _chat(body, call))
                self._send(404, "application/json", b"{}")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _next(self, path: str, body: dict) -> tuple[str, dict] | None:
        """Record the request; the first tool-bearing one gets the native call."""
        with self._lock:
            self.requests.append({"path": path, "body": body})
            if advertised_tools(path, body) is not None and not self._native_sent:
                self._native_sent = True
                return self.native
            return None

    def __enter__(self) -> "FakeModel":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


def _sse(event: str | None, data: dict) -> str:
    return (f"event: {event}\n" if event else "") + f"data: {json.dumps(data)}\n\n"


def _anthropic(body: dict, call: tuple[str, dict] | None) -> bytes:
    usage = {"input_tokens": 1, "output_tokens": 1}
    msg = {"id": "msg_selfcheck", "type": "message", "role": "assistant",
           "model": body.get("model"), "content": [], "stop_reason": None,
           "stop_sequence": None, "usage": usage}
    out = _sse("message_start", {"type": "message_start", "message": msg})
    if call:
        block = {"type": "tool_use", "id": "toolu_selfcheck", "name": call[0], "input": {}}
        delta = {"type": "input_json_delta", "partial_json": json.dumps(call[1])}
        stop = "tool_use"
    else:
        block, delta, stop = ({"type": "text", "text": ""},
                              {"type": "text_delta", "text": "done"}, "end_turn")
    out += _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                        "content_block": block})
    out += _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                        "delta": delta})
    out += _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    out += _sse("message_delta", {"type": "message_delta", "usage": {"output_tokens": 1},
                                  "delta": {"stop_reason": stop, "stop_sequence": None}})
    out += _sse("message_stop", {"type": "message_stop"})
    return out.encode()


def _responses(body: dict, call: tuple[str, dict] | None) -> bytes:
    if call:
        item = {"type": "function_call", "id": "fc_selfcheck", "status": "completed",
                "call_id": "call_selfcheck", "name": call[0],
                "arguments": json.dumps(call[1])}
    else:
        item = {"type": "message", "id": "m_selfcheck", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}]}
    usage = {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0},
             "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0},
             "total_tokens": 2}
    resp = {"id": "resp_selfcheck", "object": "response", "status": "completed",
            "model": body.get("model"), "output": [item], "usage": usage}
    out = _sse("response.created", {"type": "response.created",
                                    "response": {**resp, "status": "in_progress", "output": []}})
    out += _sse("response.output_item.added", {"type": "response.output_item.added",
                                               "output_index": 0, "item": item})
    out += _sse("response.output_item.done", {"type": "response.output_item.done",
                                              "output_index": 0, "item": item})
    out += _sse("response.completed", {"type": "response.completed", "response": resp})
    return out.encode()


def _chat(body: dict, call: tuple[str, dict] | None) -> bytes:
    base = {"id": "chatcmpl-selfcheck", "object": "chat.completion.chunk",
            "created": 0, "model": body.get("model")}
    if call:
        delta = {"role": "assistant", "content": None, "tool_calls": [
            {"index": 0, "id": "call_selfcheck", "type": "function",
             "function": {"name": call[0], "arguments": json.dumps(call[1])}}]}
        finish = "tool_calls"
    else:
        delta, finish = {"role": "assistant", "content": "done"}, "stop"
    out = _sse(None, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
    out += _sse(None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                 "total_tokens": 2}})
    return (out + "data: [DONE]\n\n").encode()


# ---------------------------------------------------------------- evaluation ---
def _flatten(tools: list, prefix: str = "") -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            names += _flatten(tool.get("tools") or [], f"{tool.get('name')}__")
            continue
        name = tool.get("name") or (tool.get("function") or {}).get("name") \
            or tool.get("type")
        if name:
            names.append(f"{prefix}{name}")
    return names


def advertised_tools(path: str, body: dict) -> list[str] | None:
    """The tool names a request offered the model, or None if it offered none
    (a title-generation or other side request)."""
    names = _flatten(body.get("tools") or [])
    for item in body.get("input") or [] if isinstance(body.get("input"), list) else []:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            names += _flatten(item.get("tools") or [])
    return names or None


def expected_probe_name(provider: str, qualify) -> str:
    if provider == "codex":             # codex namespaces MCP tools by server
        return f"mcp__switchyard__{PROBE_TOOL['name']}"
    return qualify(PROBE_TOOL["name"])


def evaluate(provider: str, requests: list[dict], probe_name: str,
             native_executed: bool) -> dict:
    """Pure verdict over what the fake model saw and what the CLI did."""
    # Every tool offered on ANY request counts: a CLI that shows the model a
    # native tool once has exposed it, whatever a later request lists.
    offered: list[str] = []
    for req in requests:
        for name in advertised_tools(req["path"], req["body"]) or []:
            if name not in offered:
                offered.append(name)
    allowed = {probe_name} | ALLOWED_EXTRAS.get(provider, set())
    unexpected = sorted(set(offered) - allowed)
    visible = probe_name in offered
    problems = []
    if not offered:
        problems.append("the CLI never offered the model any tool")
    if unexpected:
        problems.append(f"native tools offered to the model: {', '.join(unexpected)}")
    if offered and not visible:
        problems.append(f"bridged tool {probe_name} is not visible to the model")
    if native_executed:
        problems.append(f"the native {NATIVE_CALLS[provider][0]} call executed in the sidecar")
    return {"ok": not problems, "provider": provider, "offered": offered,
            "unexpected": unexpected, "bridged_visible": visible,
            "native_executed": native_executed,
            "detail": "; ".join(problems) or "locked down"}


# ------------------------------------------------------------------- the run ---
def _redirect(provider: str, url: str, root: Path) -> tuple[list[str], dict, str | None]:
    """(extra argv, extra env, model override) sending the CLI to the fake."""
    if provider == "claude":
        return [], {"ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": "selfcheck",
                    "CLAUDE_CONFIG_DIR": str(root / "claude"),
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}, None
    if provider == "codex":
        (root / "codex").mkdir(exist_ok=True)
        return (["-c", "model_provider=switchyard_selfcheck", "-c",
                 'model_providers.switchyard_selfcheck={name="selfcheck",'
                 f'base_url="{url}/v1",env_key="SWITCHYARD_SELFCHECK_KEY",'
                 'wire_api="responses"}'],
                {"SWITCHYARD_SELFCHECK_KEY": "selfcheck",
                 "CODEX_HOME": str(root / "codex")}, None)
    config = root / "config"
    (config / "opencode").mkdir(parents=True, exist_ok=True)
    (config / "opencode" / "opencode.json").write_text(json.dumps({"provider": {
        "switchyard-selfcheck": {
            "npm": "@ai-sdk/openai-compatible", "name": "selfcheck",
            "options": {"baseURL": f"{url}/v1", "apiKey": "selfcheck"},
            "models": {"m": {"name": "m", "tool_call": True}}}}}))
    return [], {"XDG_CONFIG_HOME": str(config), "XDG_DATA_HOME": str(root / "data"),
                "XDG_STATE_HOME": str(root / "state")}, "switchyard-selfcheck/m"


async def run(bridge: Any, timeout: float = 180.0) -> dict:
    """Spawn this sidecar's CLI, exactly as a tool-path session would, against
    the fake model, and return evaluate()'s verdict."""
    provider = bridge.PROVIDER
    started = time.time()
    root = Path(tempfile.mkdtemp(prefix="sy-selfcheck-"))
    workdir = root / "session"
    workdir.mkdir()
    marker = workdir / MARKER           # where the relative `touch` lands
    result: dict
    try:
        with FakeModel(NATIVE_CALLS[provider]) as fake:
            extra_argv, extra_env, model_override = _redirect(provider, fake.url, root)
            tools_path = workdir / "tools.json"
            tools_path.write_text(json.dumps([PROBE_TOOL]))
            qualify = bridge.PROFILE["tool_qualifier"]
            model = model_override or bridge.cli_bridge.resolve_model(None)[0]
            argv, stdin_data = bridge.build_argv(
                "SwitchYard host-mirror self-check. Reply `done`.",
                "SwitchYard host-mirror self-check.", model, workdir,
                uuid.uuid4().hex, tools_path, qualify(PROBE_TOOL["name"]))
            env = {**bridge.cli_bridge.subprocess_env(), **extra_env,
                   "SWITCHYARD_SELFCHECK_URL": fake.url}
            proc = await asyncio.create_subprocess_exec(
                *argv, *extra_argv, cwd=str(workdir), env=env,
                stdin=asyncio.subprocess.PIPE if stdin_data is not None
                else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                _, err = await asyncio.wait_for(
                    proc.communicate(stdin_data.encode() if stdin_data else None), timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                err = b"timed out"
            result = evaluate(provider, fake.requests,
                              expected_probe_name(provider, qualify), marker.exists())
            result["cli_exit"] = proc.returncode
            if not result["ok"] and not fake.requests:
                result["detail"] += f"; CLI stderr: {err.decode(errors='replace')[-300:]}"
    except Exception as exc:          # a check that cannot run has not passed
        result = {"ok": False, "provider": provider,
                  "detail": f"self-check could not run: {type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(root, ignore_errors=True)
    result["checked_at"] = time.time()
    result["duration_s"] = round(time.time() - started, 2)
    return result


if __name__ == "__main__":
    import importlib.util
    import sys
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("switchyard_mcp_bridge", here / "server.py")
    server = importlib.util.module_from_spec(spec)
    sys.modules["switchyard_mcp_bridge"] = server
    spec.loader.exec_module(server)
    verdict = asyncio.run(run(server))
    print(json.dumps(verdict, indent=2))
    raise SystemExit(0 if verdict["ok"] else 1)
