"""A loopback model endpoint for the CLI lockdown tests.

Speaks the three wire protocols the sidecar CLIs use -- Anthropic
/v1/messages (claude), OpenAI /v1/responses (codex) and /v1/chat/completions
(opencode) -- streaming, records every request, and plays a script: each
request that offers the model any tool consumes the next scripted step
(`{"tool": name, "input": {...}}`, optionally `"kind": "custom"` for a
Responses custom tool, or `{"text": ...}`); tool-less side requests (title
generation) and an exhausted script get plain text.

Every tool call gets a process-unique id: a CLI that sees two calls with
the same id merges them (Claude Code does), which silently drops one.

Loopback only, so it sits inside conftest.py's socket guard.
"""
from __future__ import annotations

import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _flatten(tools, prefix: str = "") -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            names += _flatten(tool.get("tools"), f"{tool.get('name')}__")
            continue
        name = tool.get("name") or (tool.get("function") or {}).get("name") \
            or tool.get("type")
        if name:
            names.append(f"{prefix}{name}")
    return names


def advertised_tools(body: dict) -> list[str]:
    """Every tool name a request offered the model (codex's `additional_tools`
    input item and its namespaces included)."""
    names = _flatten(body.get("tools"))
    items = body.get("input") if isinstance(body.get("input"), list) else []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            names += _flatten(item.get("tools"))
    return names


def request_text(body: dict) -> str:
    """Everything a request carried, as one string, for leak checks."""
    return json.dumps(body)


_ids = itertools.count(1)


def _sse(event, data) -> str:
    return (f"event: {event}\n" if event else "") + f"data: {json.dumps(data)}\n\n"


class FakeModel:
    def __init__(self, script: list[dict] | None = None, force: bool = False):
        """`force=True` plays the script on every request, tool-bearing or
        not: a model can name a tool it was never offered, and whether the
        CLI then runs it is exactly what a text-path lockdown test checks."""
        self.script = list(script or [])
        self.force = force
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _send(self, code, ctype, payload: bytes):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                self._send(200, "application/json", b'{"object": "list", "data": []}')

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    body = {}
                path = self.path.split("?")[0]
                if path.endswith("/count_tokens"):
                    return self._send(200, "application/json", b'{"input_tokens": 1}')
                step = outer._record(path, dict(self.headers), body)
                for suffix, render in (("/messages", _anthropic),
                                       ("/responses", _responses),
                                       ("/chat/completions", _chat)):
                    if path.endswith(suffix):
                        return self._send(200, "text/event-stream", render(body, step))
                self._send(404, "application/json", b"{}")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _record(self, path, headers, body) -> dict:
        with self._lock:
            self.requests.append({"path": path, "headers": headers, "body": body})
            if (self.force or advertised_tools(body)) and self.script:
                return self.script.pop(0)
            return {"text": "done"}

    def tool_requests(self) -> list[dict]:
        return [r for r in self.requests if advertised_tools(r["body"])]

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


def _anthropic(body, step) -> bytes:
    msg = {"id": "msg_fake", "type": "message", "role": "assistant",
           "model": body.get("model"), "content": [], "stop_reason": None,
           "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}
    out = _sse("message_start", {"type": "message_start", "message": msg})
    if "tool" in step:
        block = {"type": "tool_use", "id": f"toolu_fake{next(_ids)}",
                 "name": step["tool"], "input": {}}
        delta = {"type": "input_json_delta", "partial_json": json.dumps(step.get("input", {}))}
        stop = "tool_use"
    else:
        block = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": step["text"]}
        stop = "end_turn"
    out += _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                        "content_block": block})
    out += _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                        "delta": delta})
    out += _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    out += _sse("message_delta", {"type": "message_delta", "usage": {"output_tokens": 1},
                                  "delta": {"stop_reason": stop, "stop_sequence": None}})
    out += _sse("message_stop", {"type": "message_stop"})
    return out.encode()


def _responses(body, step) -> bytes:
    n = next(_ids)
    if "tool" in step and step.get("kind") == "custom":
        item = {"type": "custom_tool_call", "id": f"ctc_{n}", "status": "completed",
                "call_id": f"call_{n}", "name": step["tool"],
                "input": step.get("input") if isinstance(step.get("input"), str)
                else json.dumps(step.get("input", {}))}
    elif "tool" in step:
        item = {"type": "function_call", "id": f"fc_{n}", "status": "completed",
                "call_id": f"call_{n}", "name": step["tool"],
                "arguments": json.dumps(step.get("input", {}))}
    else:
        item = {"type": "message", "id": f"m_{n}", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": step["text"], "annotations": []}]}
    usage = {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0},
             "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0},
             "total_tokens": 2}
    resp = {"id": f"resp_{n}", "object": "response", "status": "completed",
            "model": body.get("model"), "output": [item], "usage": usage}
    out = _sse("response.created", {"type": "response.created",
                                    "response": {**resp, "status": "in_progress", "output": []}})
    out += _sse("response.output_item.added", {"type": "response.output_item.added",
                                               "output_index": 0, "item": item})
    out += _sse("response.output_item.done", {"type": "response.output_item.done",
                                              "output_index": 0, "item": item})
    out += _sse("response.completed", {"type": "response.completed", "response": resp})
    return out.encode()


def _chat(body, step) -> bytes:
    base = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0,
            "model": body.get("model")}
    if "tool" in step:
        delta = {"role": "assistant", "content": None, "tool_calls": [
            {"index": 0, "id": f"call_fake{next(_ids)}", "type": "function",
             "function": {"name": step["tool"], "arguments": json.dumps(step.get("input", {}))}}]}
        finish = "tool_calls"
    else:
        delta, finish = {"role": "assistant", "content": step["text"]}, "stop"
    out = _sse(None, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
    out += _sse(None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
    return (out + "data: [DONE]\n\n").encode()
