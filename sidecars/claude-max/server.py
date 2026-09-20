"""OpenAI-compatible shim over the Claude Code CLI.

Why this exists: LiteLLM authenticates with static API keys, but a Claude Max
subscription is OAuth-only and its token lives in the CLI's own credential
store. So the CLI stays the client — it owns login and refresh — and this shim
just speaks HTTP on one side and `claude -p` on the other.

The one thing it must get exactly right is error mapping: a usage-limit
rejection has to leave here as **HTTP 429 with Retry-After**, because that is
the signal Switchyard uses to drop the plan's slots out of the lane. A 500
would look like a transient blip and the lane would keep picking dead capacity.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="claude-max-sidecar")

CLI = os.environ.get("CLAUDE_CLI", "claude")
MODEL = os.environ.get("CLAUDE_MODEL", "opus")
MAX_CONCURRENCY = int(os.environ.get("SIDECAR_CONCURRENCY", "1"))
TIMEOUT = int(os.environ.get("SIDECAR_TIMEOUT", "600"))

_gate = asyncio.Semaphore(MAX_CONCURRENCY)

# The CLI reports exhaustion in prose; these are the shapes worth trusting.
_LIMIT = re.compile(
    r"(usage limit reached|limit will reset|you've reached your|rate.?limit"
    r"|out of (credits|usage)|quota)", re.I,
)
_RESET_AT = re.compile(r"reset(?:s|ting)?\s+at\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)", re.I)
_AUTH = re.compile(r"(not logged in|please run .?claude login|authentication|invalid credentials)", re.I)


def flatten(messages: list[dict]) -> tuple[str, str | None]:
    """Collapse a chat array into one prompt plus a system prompt.

    A limitation worth knowing: this is stateless, so each turn re-sends the
    whole conversation and pays for it. Switchyard's session affinity keeps a
    session pinned here, which is what makes the CLI's prompt cache effective.
    """
    system: list[str] = []
    turns: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(b.get("text", "")) for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        content = (content or "").strip()
        if not content:
            continue
        role = m.get("role")
        if role == "system":
            system.append(content)
        elif role == "assistant":
            turns.append(f"Assistant: {content}")
        else:
            turns.append(f"Human: {content}")
    return "\n\n".join(turns), ("\n\n".join(system) or None)


async def run_cli(prompt: str, system: str | None) -> dict:
    cmd = [CLI, "-p", prompt, "--output-format", "json", "--model", MODEL]
    if system:
        cmd += ["--append-system-prompt", system]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=408, detail="claude cli timed out")

    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
    blob = f"{stdout}\n{stderr}"

    if proc.returncode != 0 or not stdout.strip():
        if _AUTH.search(blob):
            raise HTTPException(status_code=401, detail=f"claude cli not authenticated: {stderr[:300]}")
        if _LIMIT.search(blob):
            raise _limit_error(blob)
        raise HTTPException(status_code=502, detail=f"claude cli failed ({proc.returncode}): {stderr[:300]}")

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = {"result": stdout.strip()}

    # The CLI can exit 0 while reporting a limit inside the JSON envelope.
    if payload.get("is_error") or payload.get("subtype") in ("error_max_turns", "error_during_execution"):
        text = json.dumps(payload)
        if _LIMIT.search(text):
            raise _limit_error(text)
        raise HTTPException(status_code=502, detail=f"claude cli error: {text[:300]}")
    if _LIMIT.search(str(payload.get("result", ""))) and not payload.get("usage"):
        raise _limit_error(str(payload.get("result")))

    return payload


def _limit_error(blob: str) -> HTTPException:
    # Default to the 5-hour window; if the CLI names a reset time, trust it.
    retry_after = 5 * 3600
    m = _RESET_AT.search(blob)
    detail = "claude max usage limit reached"
    if m:
        detail = f"{detail} (resets at {m.group(1)})"
    return HTTPException(
        status_code=429,
        detail={"error": {"message": detail, "type": "usage_limit_reached"}},
        headers={"Retry-After": str(retry_after)},
    )


def to_openai(payload: dict, model: str) -> dict:
    usage = payload.get("usage") or {}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": payload.get("result", "")},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
            "completion_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0),
        },
    }


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model": MODEL, "concurrency": MAX_CONCURRENCY,
            "in_flight": MAX_CONCURRENCY - _gate._value}


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "claude-max"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    prompt, system = flatten(body.get("messages") or [])
    if not prompt:
        raise HTTPException(status_code=400, detail="no usable message content")
    model = body.get("model") or MODEL

    if _gate.locked() and _gate._value == 0:
        # Never queue: Switchyard needs to hear "full" immediately so it can
        # spill to the next plan in the lane instead of blocking a worker.
        raise HTTPException(status_code=429, detail="sidecar at capacity",
                            headers={"Retry-After": "5"})

    async with _gate:
        payload = await run_cli(prompt, system)
    result = to_openai(payload, model)

    if not body.get("stream"):
        return result

    async def one_shot():
        chunk = {
            "id": result["id"], "object": "chat.completion.chunk",
            "created": result["created"], "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant",
                         "content": result["choices"][0]["message"]["content"]},
                         "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        done = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": result["usage"]}
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(one_shot(), media_type="text/event-stream")
