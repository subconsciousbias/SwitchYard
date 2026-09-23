"""Token-proxy tests: the body reaching upstream must be byte-for-byte what
the caller sent (that is the entire point — see server.py's module docstring),
the health honesty convention, and the openai NotImplementedError path.

No network: switchyard.oauth is stubbed at the function level (access_token /
account_id / status), and httpx.AsyncClient inside the server module is
replaced with a fake that records what it was asked to send instead of
opening a socket.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sidecars", "token_proxy"))
sys.path.insert(0, HERE)

os.environ.setdefault("SWITCHYARD_PROVIDER", "xai")

from fastapi.testclient import TestClient  # noqa: E402

from _modules import load  # noqa: E402

server = load("token_proxy_server",
              os.path.join(ROOT, "sidecars", "token_proxy", "server.py"))
from switchyard import oauth  # noqa: E402


class _FakeResponse:
    """Just enough of httpx.Response for server.py's non-streaming path."""
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True}
        # Real responses carry headers, and the proxy forwards the quota ones —
        # xAI's x-ratelimit-* are the only headroom this plan has.
        self.headers = headers if headers is not None else {}

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Records the single call server.py makes, instead of hitting a socket."""
    last: dict = {}
    # Set to a _FakeResponse to control what comes back (headers, status).
    response = None

    def __init__(self, timeout=None):
        pass

    async def post(self, url, content=None, headers=None):
        _FakeAsyncClient.last = {"url": url, "content": content, "headers": headers}
        if _FakeAsyncClient.response is not None:
            return _FakeAsyncClient.response
        return _FakeResponse(200, {"id": "chatcmpl-fake",
                                   "choices": [{"message": {"role": "assistant",
                                                             "content": "ok"}}]})

    async def aclose(self):
        pass


def _install_fake_client():
    original = server.httpx.AsyncClient
    server.httpx.AsyncClient = _FakeAsyncClient
    return original


def _restore_client(original):
    server.httpx.AsyncClient = original


def _install_fake_oauth(token="fake-access-token", account=None, authorised=True):
    """Stub the three oauth entry points the proxy touches, without a store."""
    originals = (oauth.access_token, oauth.account_id, oauth.status)

    def fake_access_token(provider):
        if not authorised:
            raise RuntimeError(f"no OAuth grant for {provider!r}")
        return token

    def fake_account_id(provider):
        return account

    def fake_status(provider):
        return {"provider": provider, "authorised": authorised,
                "expires_in": 3599 if authorised else None,
                "has_refresh": authorised, "scopes": "", "account_id": account}

    oauth.access_token = fake_access_token
    oauth.account_id = fake_account_id
    oauth.status = fake_status
    return originals


def _restore_oauth(originals):
    oauth.access_token, oauth.account_id, oauth.status = originals


def test_chat_completions_forwards_tools_and_body_byte_for_byte():
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(token="secret-xai-token")
    client_original = _install_fake_client()
    try:
        client = TestClient(server.app)
        body = {
            "model": "grok-4.6",
            "messages": [{"role": "user", "content": "what's the weather"}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
            "tool_choice": "auto",
        }
        resp = client.post("/v1/chat/completions", content=json.dumps(body))
        assert resp.status_code == 200, resp.text

        sent = _FakeAsyncClient.last
        assert sent["url"] == "https://api.x.ai/v1/chat/completions", sent["url"]
        # The whole point: what reached upstream is exactly what the caller
        # sent, tools and tool_choice included — not a re-serialisation that
        # happens to be equivalent.
        assert json.loads(sent["content"]) == body, sent["content"]
        assert sent["headers"]["Authorization"] == "Bearer secret-xai-token"
        print("  tools/tool_choice reached api.x.ai untouched, bearer attached")
    finally:
        _restore_client(client_original)
        _restore_oauth(originals)


def test_missing_grant_is_503_not_an_empty_bearer_header():
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(authorised=False)
    client_original = _install_fake_client()
    _FakeAsyncClient.last = {}
    try:
        client = TestClient(server.app)
        resp = client.post("/v1/chat/completions",
                           content=json.dumps({"model": "grok-4.6", "messages": []}))
        assert resp.status_code == 503, resp.text
        assert _FakeAsyncClient.last == {}, "must not have called upstream at all"
        print(f"  no grant -> 503, no request sent: {resp.json()['detail'][:60]}")
    finally:
        _restore_client(client_original)
        _restore_oauth(originals)


def test_openai_chat_completions_is_a_clearly_marked_not_implemented():
    server.PROVIDER = "openai"
    originals = _install_fake_oauth(token="irrelevant", account="acct-1")
    client_original = _install_fake_client()
    _FakeAsyncClient.last = {}
    try:
        client = TestClient(server.app)
        resp = client.post("/v1/chat/completions",
                           content=json.dumps({"model": "gpt-5.6-sol", "messages": []}))
        assert resp.status_code == 501, resp.text
        # The message names where the seat IS served, so a 501 here reads as a
        # routing fact rather than a gap: the openai plan goes through
        # mcp_bridge's codex profile, not this proxy.
        assert "mcp_bridge" in resp.json()["detail"], resp.json()
        assert _FakeAsyncClient.last == {}, "must not silently forward to the wrong wire shape"
        print(f"  openai chat/completions -> 501: {resp.json()['detail'][:70]}...")
    finally:
        _restore_client(client_original)
        _restore_oauth(originals)
        server.PROVIDER = "xai"


def test_health_is_dishonest_never_ok_true_without_a_working_grant():
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(authorised=False)
    try:
        client = TestClient(server.app)
        health = client.get("/health").json()
        assert health["ok"] is False, health
        assert health["authorised"] is False
        print(f"  unauthorised xai: ok={health['ok']}")
    finally:
        _restore_oauth(originals)


def test_health_is_ok_for_xai_once_authorised():
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(authorised=True)
    try:
        client = TestClient(server.app)
        health = client.get("/health").json()
        assert health["ok"] is True, health
        assert health["upstream_base"] == "https://api.x.ai/v1"
        assert health["supports_chat_completions"] is True
        print(f"  authorised xai: {health}")
    finally:
        _restore_oauth(originals)


def test_health_is_never_ok_for_openai_regardless_of_grant():
    """The gap is the translation, not the grant — health must say so even
    when a real, working grant is on file."""
    server.PROVIDER = "openai"
    originals = _install_fake_oauth(authorised=True, account="acct-1")
    try:
        client = TestClient(server.app)
        health = client.get("/health").json()
        assert health["ok"] is False, health
        assert health["authorised"] is True   # the grant itself is fine
        assert health["supports_chat_completions"] is False
        print(f"  authorised-but-unsupported openai: ok={health['ok']} "
              f"authorised={health['authorised']}")
    finally:
        _restore_oauth(originals)
        server.PROVIDER = "xai"


def test_health_never_includes_a_token_or_authorization_header():
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(token="should-never-appear-anywhere")
    try:
        client = TestClient(server.app)
        blob = json.dumps(client.get("/health").json())
        assert "should-never-appear-anywhere" not in blob
        assert "Authorization" not in blob and "Bearer" not in blob
        print("  /health payload contains no token and no Authorization header")
    finally:
        _restore_oauth(originals)


def test_quota_headers_are_forwarded_but_framing_headers_are_not():
    """xAI states this plan's headroom in response headers and nowhere else.

    The proxy re-serialises the body, so relaying content-length or
    content-encoding from upstream would describe the wrong bytes and truncate
    or break decoding of the response. Only the quota headers come back.
    """
    upstream = {
        "x-ratelimit-limit-tokens": "53000000",
        "x-ratelimit-remaining-tokens": "51750000",
        "x-ratelimit-limit-requests": "8300",
        "retry-after": "12",
        "content-length": "999999",
        "content-encoding": "gzip",
        "server": "cloudflare",
    }
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(token="secret-xai-token")
    client_original = _install_fake_client()
    _FakeAsyncClient.response = _FakeResponse(payload={"ok": True}, headers=upstream)
    try:
        resp = TestClient(server.app).post(
            "/v1/chat/completions",
            content=json.dumps({"model": "grok-4.6", "messages": []}))
    finally:
        _FakeAsyncClient.response = None
        _restore_client(client_original)
        _restore_oauth(originals)
    assert resp.status_code == 200, resp.text
    got = {k.lower() for k in resp.headers}
    assert "x-ratelimit-remaining-tokens" in got, sorted(got)
    assert resp.headers["x-ratelimit-limit-tokens"] == "53000000"
    assert "retry-after" in got, sorted(got)
    # Framing and provenance headers describe the upstream body, not ours.
    assert resp.headers.get("content-encoding") != "gzip", "would break decoding"
    assert resp.headers.get("content-length") != "999999", "would truncate the body"
    assert "server" not in got or resp.headers["server"] != "cloudflare"
    print("  forwarded the x-ratelimit-* and retry-after headers, dropped the framing ones")


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} token-proxy tests passed")
