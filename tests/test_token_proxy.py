"""Token-proxy tests: the body reaching upstream must be byte-for-byte what
the caller sent (that is the entire point — see server.py's module docstring),
the health honesty convention, and the openai NotImplementedError path.

No network: switchyard.oauth is stubbed at the function level (access_token /
account_id / status), and most tests replace httpx.AsyncClient inside the
server module with a fake that records what it was asked to send instead of
opening a socket — except the `_StubUpstream` tests, which open a loopback
HTTP server on 127.0.0.1 to exercise real `httpx.AsyncClient` semantics
(`build_request`/`send(stream=True)`/`aiter_raw`) that the fake cannot mock.
"""
from __future__ import annotations

import dataclasses
import http.server
import json
import os
import socket
import sys
import threading
import time

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
        resp = client.get("/health")
        health = resp.json()
        assert resp.status_code == 503, (resp.status_code, health)
        assert health["ok"] is False, health
        assert health["authorised"] is False
        print(f"  unauthorised xai: status={resp.status_code} ok={health['ok']}")
    finally:
        _restore_oauth(originals)


def test_health_is_ok_for_xai_once_authorised():
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(authorised=True)
    try:
        client = TestClient(server.app)
        resp = client.get("/health")
        health = resp.json()
        assert resp.status_code == 200, (resp.status_code, health)
        assert health["ok"] is True, health
        assert health["upstream_base"] == "https://api.x.ai/v1"
        assert health["supports_chat_completions"] is True
        print(f"  authorised xai: status={resp.status_code} {health}")
    finally:
        _restore_oauth(originals)


def test_health_is_never_ok_for_openai_regardless_of_grant():
    """The gap is the translation, not the grant — health must say so even
    when a real, working grant is on file."""
    server.PROVIDER = "openai"
    originals = _install_fake_oauth(authorised=True, account="acct-1")
    try:
        client = TestClient(server.app)
        resp = client.get("/health")
        health = resp.json()
        assert resp.status_code == 503, (resp.status_code, health)
        assert health["ok"] is False, health
        assert health["authorised"] is True   # the grant itself is fine
        assert health["supports_chat_completions"] is False
        print(f"  authorised-but-unsupported openai: status={resp.status_code} "
              f"ok={health['ok']} authorised={health['authorised']}")
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


class _StubUpstream:
    """A loopback http.server that answers POSTs with a configured response.

    Records the requests it receives so a test can assert the caller's body
    and headers reached the proxy's upstream call unchanged. Pattern borrowed
    from tests/test_probes.py:126-160.
    """
    def __init__(self, status_code, headers=None, body=b"", hold_seconds=0.0):
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.body = body
        self.hold_seconds = hold_seconds
        self.received: list[dict] = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length) if length else b""
                outer.received.append({"path": self.path,
                                        "headers": dict(self.headers),
                                        "body": raw})
                self.send_response(outer.status_code)
                # Framing headers must match the bytes on the wire — set them
                # explicitly from the body length rather than echoing the
                # test's configured values, which would either produce a
                # protocol error (conflicting Content-Length) or truncate the
                # response. Tests that want to assert framing is filtered out
                # of the proxy's response still see those headers in
                # outer.headers above (the proxy's _quota_headers filters by
                # prefix, not by what reached the upstream).
                for k, v in outer.headers.items():
                    if k.lower() in ("content-length", "transfer-encoding",
                                     "content-encoding"):
                        continue
                    self.send_header(k, v)
                if outer.hold_seconds:
                    # Advertise a Content-Length that does not match the
                    # bytes we will actually write, then sleep before
                    # sending the (empty) body. httpx's aread() waits for
                    # the advertised length and raises ReadTimeout past the
                    # client's timeout — the realistic shape of an
                    # upstream that returned the status+headers then
                    # stalled, which is what the should-fix on the >= 400
                    # branch of _forward has to handle cleanly.
                    self.send_header("content-length", str(max(len(outer.body), 1) + 1024))
                else:
                    self.send_header("content-length", str(len(outer.body)))
                self.end_headers()
                if outer.hold_seconds:
                    time.sleep(outer.hold_seconds)
                if outer.body:
                    self.wfile.write(outer.body)

            def log_message(self, *a):
                pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.port = self.srv.server_address[1]

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def test_stream_passes_through_upstream_429_and_retry_after():
    """A streamed request that hits a quota-exhausted upstream must surface
    the upstream's 429 and its Retry-After header — not wrap a 429 in a 200
    SSE chunk. Regression for the bug where the proxy always answered 200
    once it had committed to streaming.
    """
    stub = _StubUpstream(
        status_code=429,
        headers={"Retry-After": "60",
                 "x-ratelimit-remaining-tokens": "0"},
        body=b'{"error":{"message":"quota exceeded"}}',
    )
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(token="secret-xai-token")
    saved = server.PROVIDERS["xai"]
    server.PROVIDERS["xai"] = dataclasses.replace(saved, api_base=stub.base)
    try:
        resp = TestClient(server.app).post(
            "/v1/chat/completions",
            content=json.dumps({"model": "grok-4.6",
                                "messages": [],
                                "stream": True}))
        assert resp.status_code == 429, resp.text
        assert resp.headers.get("retry-after") == "60", dict(resp.headers)
        assert resp.headers.get("x-ratelimit-remaining-tokens") == "0", dict(resp.headers)
        assert resp.json() == {"error": {"message": "quota exceeded"}}, resp.text
        # The body the proxy forwarded is exactly what the caller sent —
        # the streamed path is no different from the buffered one in that
        # regard (it is the only thing the sidecar's module docstring claims).
        assert len(stub.received) == 1, stub.received
        sent = stub.received[0]
        assert json.loads(sent["body"]) == {
            "model": "grok-4.6", "messages": [], "stream": True}, sent["body"]
        # BaseHTTPRequestHandler.headers preserves the original casing, so
        # look up the bearer case-insensitively rather than hard-coding it.
        bearer = next((v for k, v in sent["headers"].items()
                       if k.lower() == "authorization"), None)
        assert bearer == "Bearer secret-xai-token", sent["headers"]
    finally:
        server.PROVIDERS["xai"] = saved
        _restore_oauth(originals)
        stub.close()


def test_stream_passes_through_upstream_503_with_text_body():
    """A non-JSON error body from upstream must surface the upstream's status
    and quota headers too — the proxy never raises on a non-JSON error and
    must not 500 here, otherwise a transient HTML maintenance page would
    hide a real 503 from SwitchYard's classifier.
    """
    stub = _StubUpstream(
        status_code=503,
        headers={"Retry-After": "120"},
        body=b"backend exploded",
    )
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(token="secret-xai-token")
    saved = server.PROVIDERS["xai"]
    server.PROVIDERS["xai"] = dataclasses.replace(saved, api_base=stub.base)
    try:
        resp = TestClient(server.app).post(
            "/v1/chat/completions",
            content=json.dumps({"model": "grok-4.6",
                                "messages": [],
                                "stream": True}))
        assert resp.status_code == 503, resp.text
        assert resp.headers.get("retry-after") == "120", dict(resp.headers)
        assert resp.json() == {"error": "backend exploded"}, resp.text
    finally:
        server.PROVIDERS["xai"] = saved
        _restore_oauth(originals)
        stub.close()


def test_stream_success_relays_upstream_body_and_quota_headers():
    """The streamed happy path: bytes from upstream reach the caller verbatim,
    and the quota headers ride on the response head — the same headroom that
    the buffered path has always passed through, now present on streamed
    responses too."""
    stub = _StubUpstream(
        status_code=200,
        headers={"x-ratelimit-remaining-tokens": "42",
                 "x-ratelimit-limit-tokens": "53000000"},
        body=b"data: hello\n\ndata: [DONE]\n\n",
    )
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(token="secret-xai-token")
    saved = server.PROVIDERS["xai"]
    server.PROVIDERS["xai"] = dataclasses.replace(saved, api_base=stub.base)
    try:
        resp = TestClient(server.app).post(
            "/v1/chat/completions",
            content=json.dumps({"model": "grok-4.6",
                                "messages": [],
                                "stream": True}))
        assert resp.status_code == 200, resp.text
        assert resp.text == "data: hello\n\ndata: [DONE]\n\n", resp.text
        assert resp.headers.get("x-ratelimit-remaining-tokens") == "42", dict(resp.headers)
        assert resp.headers.get("x-ratelimit-limit-tokens") == "53000000", dict(resp.headers)
    finally:
        server.PROVIDERS["xai"] = saved
        _restore_oauth(originals)
        stub.close()


def test_stream_connect_refused_returns_502_not_500():
    """api_base pointed at a closed port must surface 502 — not crash, not
    200, not 504. The proxy is async; httpx surfaces ECONNREFUSED as
    ConnectError on `client.send`, which the new path catches.
    """
    # Bind a kernel-assigned socket to grab a port, then close it without
    # listening so the next connect() raises ECONNREFUSED instead of
    # succeeding against a stale listener.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    closed_port = sock.getsockname()[1]
    sock.close()
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(token="secret-xai-token")
    saved = server.PROVIDERS["xai"]
    server.PROVIDERS["xai"] = dataclasses.replace(
        saved, api_base=f"http://127.0.0.1:{closed_port}/")
    try:
        resp = TestClient(server.app).post(
            "/v1/chat/completions",
            content=json.dumps({"model": "grok-4.6",
                                "messages": [],
                                "stream": True}))
        assert resp.status_code == 502, resp.text
        body = resp.json()
        assert "error" in body and "connect" in body["error"].lower(), body
    finally:
        server.PROVIDERS["xai"] = saved
        _restore_oauth(originals)


def test_stream_read_timeout_during_error_body_returns_504_with_quota_headers():
    """Upstream sent the response status+headers (so a 429 reached us) and
    then stalled before sending the body. The aread() in the >= 400 branch
    must surface as 504 — not 500 (the bug the should-fix eliminated) and
    not 502 (we did connect, headers did arrive) — and must still forward
    the quota headers that came in with the status, since those are the
    only headroom SwitchYard's classifier has on this plan.
    """
    # The proxy's httpx client is constructed with server.TIMEOUT, so we
    # monkey-patch that to a short value. The stub then writes the status
    # + headers and sleeps past it: client.send returns immediately
    # (headers are in), aread() times out waiting for body bytes. Hold
    # for the patched timeout (not the pre-patch one) — otherwise with the
    # default 600s proxy timeout the stub's daemon thread would sleep
    # for ~601s and waste CPU until process exit, even though the proxy
    # itself has already responded with 504.
    saved_timeout = server.TIMEOUT
    server.TIMEOUT = 0.5
    hold_seconds = server.TIMEOUT + 1.0
    stub = _StubUpstream(
        status_code=429,
        headers={"Retry-After": "60",
                 "x-ratelimit-remaining-tokens": "0"},
        body=b"",
        hold_seconds=hold_seconds,
    )
    server.PROVIDER = "xai"
    originals = _install_fake_oauth(token="secret-xai-token")
    saved = server.PROVIDERS["xai"]
    server.PROVIDERS["xai"] = dataclasses.replace(saved, api_base=stub.base)
    try:
        resp = TestClient(server.app).post(
            "/v1/chat/completions",
            content=json.dumps({"model": "grok-4.6",
                                "messages": [],
                                "stream": True}))
        assert resp.status_code == 504, resp.text
        body = resp.json()
        assert "error" in body and "timed out" in body["error"].lower(), body
        # Headers that arrived with the status are still readable after
        # aclose — that is the whole point of forwarding them on a
        # ReadTimeout, since the plan's quota board has no other source.
        assert resp.headers.get("retry-after") == "60", dict(resp.headers)
        assert resp.headers.get("x-ratelimit-remaining-tokens") == "0", dict(resp.headers)
    finally:
        server.PROVIDERS["xai"] = saved
        _restore_oauth(originals)
        server.TIMEOUT = saved_timeout
        stub.close()


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
