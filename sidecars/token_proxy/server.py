"""OpenAI-compatible pass-through for OAuth subscriptions, using our own grant.

Why this exists, and why it is not `cli_bridge`: `cli_bridge` shells out to a
vendor CLI, and the CLI's own agent harness owns the tool loop — a caller's
`tools` never reach the model. `switchyard/oauth.py` takes out Switchyard's own
OAuth grant against the subscription instead, which authenticates an ordinary
HTTP API where tool calling is first-class. So this process does the smallest
possible thing: attach a fresh bearer token to the caller's request and forward
it **untouched**. No prompt folding, no tool stripping, no response reshaping —
that is the entire point, since it is exactly what lets `tools`/`tool_choice`
survive the trip and `tool_calls` come back.

One process per provider, selected with SWITCHYARD_PROVIDER (xai | openai).
Concurrency and routing live in plans.yaml like every other plan; this process
holds no limits of its own — see the plan's `api_base: http://…-token-proxy:PORT/v1`
in config/plans.yaml, which is unaffected by this file.

xAI is fully implemented: `api.x.ai/v1` is OpenAI-compatible, so the caller's
body goes straight to `/v1/chat/completions` with only a bearer header added.

OpenAI is NOT implemented for chat/completions or /v1/messages. The ChatGPT
seat's OAuth grant is only honoured against
`https://chatgpt.com/backend-api/codex/responses` — the **Responses** API, a
different wire shape from what a caller sends (`chat/completions`-style
`messages`). See `_translate_to_responses` for what was investigated and why a
same-day translation was not attempted. `/health` reports this honestly:
`ok` is false for the openai provider until that translation exists.

Never logs a token or an Authorization header — not even at DEBUG, not even
truncated. `switchyard.oauth.status()` is the only thing this process prints
about a grant, and it never includes the token either.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Sibling package, not installed — same layering as the gateway image, which
# sets PYTHONPATH=/switchyard rather than pip-installing switchyard into the
# base image. See this file's Dockerfile note in the deployment report.
sys.path.insert(0, os.environ.get("SWITCHYARD_ROOT", "/switchyard"))
from switchyard import oauth  # noqa: E402

log = logging.getLogger("token_proxy")

app = FastAPI(title="switchyard-token-proxy")

PROVIDER = os.environ.get("SWITCHYARD_PROVIDER", "xai").lower()
TIMEOUT = float(os.environ.get("PROXY_TIMEOUT", "600"))


def _xai_headers(provider: str) -> dict[str, str]:
    # Bearer only: api.x.ai needs nothing else, unlike the ChatGPT backend.
    return {}


def _openai_headers(provider: str) -> dict[str, str]:
    account = oauth.account_id(provider)
    if not account:
        # Loud rather than sending a request the backend will 4xx anyway —
        # the account id is extracted from the id_token at login/refresh time
        # (see switchyard/oauth.py:_chatgpt_account_id), so a missing one means
        # either no grant yet or a grant taken out before that extraction
        # existed. Either way, re-running the login fixes it; a silently
        # dropped header would surface as an opaque backend error instead.
        raise RuntimeError(
            "no ChatGPT-Account-Id on file for the openai grant; run "
            "`python -m switchyard.oauth login openai` again")
    return {"ChatGPT-Account-Id": account}


@dataclass(frozen=True)
class ProviderSpec:
    api_base: str
    # Path appended to api_base for each caller-facing route. None means "not
    # implemented for this provider" — see the module docstring for openai.
    chat_completions_path: str | None
    messages_path: str | None
    extra_headers: Callable[[str], dict[str, str]]


PROVIDERS: dict[str, ProviderSpec] = {
    "xai": ProviderSpec(
        api_base="https://api.x.ai/v1",
        chat_completions_path="/chat/completions",
        # xAI's Anthropic-shaped endpoint is forwarded to honestly rather than
        # asserted to exist: this process makes no claim about it beyond "here
        # is what api.x.ai/v1/messages said back". Untested against a real
        # grant, per this task's no-login constraint.
        messages_path="/messages",
        extra_headers=_xai_headers,
    ),
    "openai": ProviderSpec(
        api_base="https://chatgpt.com/backend-api/codex",
        chat_completions_path=None,
        messages_path=None,
        extra_headers=_openai_headers,
    ),
}


def _spec() -> ProviderSpec:
    spec = PROVIDERS.get(PROVIDER)
    if spec is None:
        raise HTTPException(
            status_code=500,
            detail=f"SWITCHYARD_PROVIDER={PROVIDER!r} is not one of {sorted(PROVIDERS)}")
    return spec


def _translate_to_responses(body: bytes) -> bytes:
    """Chat-completions body -> OpenAI Responses-API body, for the ChatGPT
    backend. NOT IMPLEMENTED — investigated, not attempted, for this reason:

    LiteLLM 1.102.0 (the version pinned in Dockerfile.gateway, confirmed with
    `docker compose exec -T gateway python3 -c "import importlib.metadata as m;
    print(m.version('litellm'))"`) DOES ship a chat/completions -> Responses
    bridge: `litellm.completion_extras.litellm_responses_transformation`,
    wired in `litellm/main.py` via `responses_api_bridge_check`. Set
    `litellm.route_all_chat_openai_to_responses = True` (or prefix the model
    with `responses/`) and `litellm.completion(custom_llm_provider="openai",
    ...)` transforms the request, POSTs it through `litellm.responses()`, and
    translates the reply back into chat.completion shape — including
    tool_calls, verified by reading
    completion_extras/litellm_responses_transformation/transformation.py.

    That bridge targets the PUBLIC `api.openai.com` Responses API. The ChatGPT
    seat's endpoint is `chatgpt.com/backend-api/codex/responses` — Codex CLI's
    own private surface, reached only through this OAuth grant, not api.openai.com.
    Nothing here confirms the two accept the same request body: Codex CLI is
    known (from public issue trackers) to send extra fields on that route —
    `originator`, a `session_id`, possibly `store:false` and prompt-cache
    fields the public Responses API does not require — and none of that is
    verifiable without either a real grant to probe against or a decompile of
    Codex CLI itself, both out of scope here (no login is permitted, and only
    OpenCode's binary was available to inspect). Bridging blind through
    LiteLLM's generic translator risks a body the backend silently
    misinterprets rather than rejects — worse than refusing outright.

    So: not faked. This raises, `/health` reports `ok: false` with
    `supports_chat_completions: false` for provider=openai, and the caller gets
    a 501 that says exactly this, instead of a response that looks like it
    worked.
    """
    raise NotImplementedError(
        "openai (ChatGPT seat) needs a chat/completions -> Responses API "
        "translation for chatgpt.com/backend-api/codex/responses, which is "
        "unverified against the real endpoint — see this function's docstring")


async def _forward(request: Request, path_attr: str, path_name: str) -> StreamingResponse | JSONResponse:
    spec = _spec()
    path = getattr(spec, path_attr)

    body = await request.body()

    if path is None:
        try:
            _translate_to_responses(body)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    try:
        token = oauth.access_token(PROVIDER)
    except RuntimeError as exc:
        # No grant, or an unrefreshable expired one — a setup problem, not a
        # transient upstream failure. 503 so Switchyard's classifier treats it
        # as "this plan cannot serve right now" rather than a hard rejection.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        extra_headers = spec.extra_headers(PROVIDER)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": request.headers.get("content-type", "application/json"),
        "Accept": request.headers.get("accept", "application/json"),
        **extra_headers,
    }
    url = spec.api_base + path

    # Only peeked at to decide streaming vs. buffered response mode — `body`
    # itself, the thing actually sent, is untouched either way.
    try:
        wants_stream = bool(json.loads(body or b"{}").get("stream"))
    except ValueError:
        wants_stream = False

    client = httpx.AsyncClient(timeout=TIMEOUT)
    if wants_stream:
        async def relay():
            async with client:
                async with client.stream("POST", url, content=body, headers=headers) as upstream:
                    if upstream.status_code >= 400:
                        # Buffer the (small) error body rather than yielding a
                        # 200 stream that then contains an error — an SSE
                        # client has no way to recover from that.
                        text = (await upstream.aread()).decode(errors="replace")
                        log.warning("%s upstream %s error: %s", PROVIDER,
                                   upstream.status_code, text[:300])
                        yield f"data: {json.dumps({'error': text[:2000]})}\n\n".encode()
                        return
                    async for chunk in upstream.aiter_raw():
                        yield chunk
        return StreamingResponse(relay(), media_type="text/event-stream")

    try:
        resp = await client.post(url, content=body, headers=headers)
    finally:
        await client.aclose()
    return JSONResponse(status_code=resp.status_code, content=_safe_json(resp))


def _safe_json(resp: httpx.Response):
    try:
        return resp.json()
    except ValueError:
        # Upstream sent something that is not JSON (an HTML error page, most
        # often) — surface it as text rather than raising here and losing the
        # real status code to a 500.
        return {"error": resp.text[:2000]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _forward(request, "chat_completions_path", "chat/completions")


@app.post("/v1/messages")
async def messages(request: Request):
    return await _forward(request, "messages_path", "messages")


@app.get("/health")
async def health() -> dict:
    spec = PROVIDERS.get(PROVIDER)
    st = oauth.status(PROVIDER)
    supports_chat = bool(spec and spec.chat_completions_path)
    # Honesty convention from sidecars/cli_bridge/server.py: `ok` is false
    # whenever this process cannot actually serve a request right now, not
    # merely whenever something is technically running.
    ok = bool(spec) and bool(st.get("authorised")) and supports_chat
    return {
        "ok": ok,
        "provider": PROVIDER,
        "authorised": st.get("authorised", False),
        "expires_in": st.get("expires_in"),
        "has_refresh": st.get("has_refresh", False),
        "account_id_on_file": bool(st.get("account_id")),
        "upstream_base": spec.api_base if spec else None,
        "supports_chat_completions": supports_chat,
        "supports_messages": bool(spec and spec.messages_path),
    }
