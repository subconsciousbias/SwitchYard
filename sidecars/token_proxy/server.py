"""OpenAI-compatible pass-through for OAuth subscriptions, using our own grant.

Why this exists, and why it is not `cli_bridge`: `cli_bridge` shells out to a
vendor CLI, and the CLI's own agent harness owns the tool loop — a caller's
`tools` never reach the model. `switchyard/oauth.py` takes out SwitchYard's own
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

OpenAI is deliberately NOT served here. A seat's grant only reaches
`chatgpt.com/backend-api/codex/responses`, a private surface that expects Codex
CLI's own identity (see `_translate_to_responses` for the captured request), and
it turned out not to be needed: `codex exec` has an MCP client, so the seat
serves tools through `sidecars/mcp_bridge` using its own official client.
`/health` still reports `ok: false` for provider=openai, because this process
genuinely cannot serve it.

Never logs a token or an Authorization header — not even at DEBUG, not even
truncated. `switchyard.oauth.status()` is the only thing this process prints
about a grant, and it never includes the token either.
"""
from __future__ import annotations

import asyncio
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
# base image. See Dockerfile.token_proxy.
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
        # is what api.x.ai/v1/messages said back". Not exercised against a
        # real grant — the chat/completions route is the one in use.
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
    """Chat-completions body -> Responses body for the ChatGPT seat. NOT USED.

    This stayed unimplemented, and then stopped being the right idea at all.

    The endpoint a seat's OAuth grant reaches is
    `chatgpt.com/backend-api/codex/responses`, not api.openai.com. A capture of
    what Codex CLI actually sends there (`-c model_providers.<id>.base_url`
    pointed at a local recorder, so nothing was forwarded and no credential was
    read) shows it is not the public Responses API with a different host:

        POST /responses
        originator: codex_exec
        session-id / thread-id / x-client-request-id: <uuid triple>
        x-codex-turn-metadata: {"installation_id":...,"window_id":...,...}
        x-codex-window-id, x-openai-internal-codex-responses-lite: true
        body: input[] (typed items), include:["reasoning.encrypted_content"],
              store:false, prompt_cache_key, reasoning{context,effort},
              text{verbosity}, client_metadata{...}

    Serving that from SwitchYard would mean presenting Codex CLI's identity to a
    private internal surface -- materially different from the xAI path, which
    uses a public OAuth client against the documented, OpenAI-compatible
    api.x.ai/v1.

    None of it is necessary. `codex exec` has an MCP client (`[mcp_servers.*]`,
    settable per invocation with -c), which the earlier note had wrong, so the
    seat serves the caller's tools through mcp_bridge like Claude and OpenCode
    do -- its own official client making its own calls. See
    sidecars/mcp_bridge/server.py's codex profile.

    Kept, not deleted, because the capture is the evidence for that choice: if
    someone later wants the direct route, this is the shape and the reason to
    think twice.
    """
    raise NotImplementedError(
        "the openai (ChatGPT seat) plan serves tools through mcp_bridge's codex "
        "profile, not through this proxy -- see this function's docstring for the "
        "captured request shape and why the direct route was not taken")


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
        # `oauth.access_token` takes a cross-process `flock` around the token
        # store, may do a refresh HTTP call, and may write the result back to
        # disk -- all blocking I/O on the event loop. Push it onto a worker
        # thread so an in-flight refresh on a slow upstream does not stall
        # other requests. RuntimeError still bubbles up unchanged and is
        # mapped to 503 below; HTTP errors raised by `_refresh` are swallowed
        # there (it returns None on httpx errors) and never reach this handler.
        token = await asyncio.to_thread(oauth.access_token, PROVIDER)
    except RuntimeError as exc:
        # No grant, or an unrefreshable expired one — a setup problem, not a
        # transient upstream failure. 503 so SwitchYard's classifier treats it
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
        # A streamed response cannot carry the upstream's headers, because they
        # are not known until the request is made; the caller gets none, and the
        # quota board falls back to whatever the last non-streamed call
        # reported. Worth knowing rather than silently assuming coverage.
        return StreamingResponse(relay(), media_type="text/event-stream")

    try:
        resp = await client.post(url, content=body, headers=headers)
    finally:
        await client.aclose()
    return JSONResponse(status_code=resp.status_code, content=_safe_json(resp),
                        headers=_quota_headers(resp))


# Headers worth passing back. Not a blanket relay: content-length and
# content-encoding describe the upstream body, not the one this process just
# re-serialised, and forwarding them produces a truncated or undecodable
# response. These are the ones that carry the provider's own headroom, which is
# otherwise lost here — xAI answers with x-ratelimit-remaining-tokens against
# x-ratelimit-limit-tokens, and Switchyard's quota board has no other source for
# this plan.
_QUOTA_HEADER_PREFIXES = ("x-ratelimit-", "ratelimit-", "x-quota-", "retry-after")


def _quota_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items()
            if k.lower().startswith(_QUOTA_HEADER_PREFIXES)}


def _safe_json(resp: httpx.Response):
    try:
        return resp.json()
    except ValueError:
        # Upstream sent something that is not JSON (an HTML error page, most
        # often) — surface it as text rather than raising here and losing the
        # real status code to a 500.
        return {"error": resp.text[:2000]}


# The subscription's real headroom, which nothing else exposes. api.x.ai's
# x-ratelimit-* headers are an advertised ceiling, not live usage — measured:
# three calls consuming ~3,000 tokens left remaining-tokens at exactly
# 53,000,000 — and /v1/usage, /v1/subscription and /v1/quota are all 404. The
# Grok CLI reads this instead, with the same bearer it uses for chat:
#
#   {"config":{"currentPeriod":{"type":"USAGE_PERIOD_TYPE_WEEKLY",
#              "start":"...","end":"2026-09-24T18:29:42+00:00"},
#              "creditUsagePercent":100.0,
#              "productUsage":[{"product":"GrokBuild","usagePercent":100.0}],
#              "prepaidBalance":{"val":995}}}
USAGE_URLS = {
    "xai": "https://cli-chat-proxy.grok.com/v1/billing?format=credits",
}


@app.get("/usage")
async def usage() -> JSONResponse:
    """The provider's own usage report, fetched with this process's grant.

    Kept here rather than in the prober because this is where the token lives:
    the portal never sees a credential, it just reads the JSON.
    """
    url = USAGE_URLS.get(PROVIDER)
    if url is None:
        raise HTTPException(status_code=501,
                            detail=f"no usage endpoint known for {PROVIDER!r}")
    try:
        # Same off-loop rationale as `_forward`: `oauth.access_token` may take
        # the cross-process store lock and do a refresh HTTP call. A blocking
        # call here would stall every other request this process is serving
        # while a refresh is in flight.
        token = await asyncio.to_thread(oauth.access_token, PROVIDER)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {token}"},
                                follow_redirects=True)
    if resp.status_code >= 400:
        raise HTTPException(status_code=502,
                            detail=f"usage endpoint returned {resp.status_code}")
    return JSONResponse(content=_safe_json(resp))


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
