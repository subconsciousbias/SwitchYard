"""OAuth grant tests: refresh, the no-empty-header contract, and the promise
that status() never leaks a token.

No network: every httpx.Client the module creates is redirected onto an
httpx.MockTransport, and SWITCHYARD_AUTH_STORE points at a temp file so a run
never reads or writes a real credential store.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402


def _fresh_store() -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)   # oauth.py creates it lazily; start from "does not exist"
    return path


from switchyard import oauth  # noqa: E402

oauth.STORE = _fresh_store()


def _stub_client(handler):
    """Monkeypatch oauth.httpx.Client so every `with httpx.Client(...)` in the
    module under test is answered by `handler` instead of a real socket."""
    real_client_cls = httpx.Client

    def factory(**kwargs):
        kwargs.pop("timeout", None)
        return real_client_cls(transport=httpx.MockTransport(handler), timeout=30)

    oauth.httpx.Client = factory


def _restore_client(original):
    oauth.httpx.Client = original


def _jwt(claims: dict) -> str:
    """A fabricated, unsigned JWT — good enough for _jwt_claims, which never
    checks the signature (see its docstring: the token already arrived over
    TLS from the provider by the time this process reads it)."""
    def seg(obj) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    return f"{seg({'alg': 'none'})}.{seg(claims)}.fake-signature"


def test_missing_grant_raises_rather_than_sending_an_empty_header():
    oauth.STORE = _fresh_store()
    try:
        oauth.access_token("xai")
    except RuntimeError as exc:
        assert "no OAuth grant" in str(exc), exc
        print(f"  no grant on file -> raises: {exc}")
        return
    raise AssertionError("expected access_token() to raise for an unauthorised provider")


def test_status_never_includes_the_token():
    store = _fresh_store()
    oauth.STORE = store
    secret = "xoxa-super-secret-do-not-leak-1234567890"
    oauth._store_tokens("xai", {"access_token": secret, "refresh_token": "r1",
                                "expires_in": 3600})
    st = oauth.status("xai")
    blob = json.dumps(st)
    assert secret not in blob, blob
    assert st["authorised"] is True
    assert st["expires_in"] > 3500
    print(f"  status() = {st} — no token substring present")


def test_token_refresh_when_near_expiry():
    store = _fresh_store()
    oauth.STORE = store
    # Expires in 30s, inside REFRESH_MARGIN (120s), so access_token() must
    # exchange the refresh token before handing back a value.
    oauth._store_tokens("xai", {"access_token": "old-token", "refresh_token": "r1",
                                "expires_in": 30})

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = request.read().decode()
        assert "grant_type=refresh_token" in body and "refresh_token=r1" in body
        return httpx.Response(200, json={"access_token": "new-token",
                                         "refresh_token": "r2", "expires_in": 3600})

    original = oauth.httpx.Client
    _stub_client(handler)
    try:
        token = oauth.access_token("xai")
    finally:
        _restore_client(original)

    assert token == "new-token", token
    assert len(calls) == 1, "expected exactly one refresh call"
    entry = oauth._load()["xai"]
    assert entry["access_token"] == "new-token"
    assert entry["refresh_token"] == "r2"
    assert entry["expires_at"] > time.time() + 3500
    print(f"  near-expiry token refreshed to {token!r}, expiry pushed out")


def test_a_token_with_plenty_of_life_is_not_refreshed():
    store = _fresh_store()
    oauth.STORE = store
    oauth._store_tokens("xai", {"access_token": "still-good", "refresh_token": "r1",
                                "expires_in": 3600})

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not have made a network call")

    original = oauth.httpx.Client
    _stub_client(handler)
    try:
        token = oauth.access_token("xai")
    finally:
        _restore_client(original)
    assert token == "still-good", token
    print("  token with >REFRESH_MARGIN left in it is returned as-is, no call made")


def test_a_failed_refresh_falls_back_to_the_current_token():
    """A dead refresh endpoint must not turn a still-technically-valid token
    into a hard failure — see _refresh()'s docstring: best-effort only."""
    store = _fresh_store()
    oauth.STORE = store
    oauth._store_tokens("xai", {"access_token": "old-but-not-dead",
                                "refresh_token": "r1", "expires_in": 10})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    original = oauth.httpx.Client
    _stub_client(handler)
    try:
        token = oauth.access_token("xai")
    finally:
        _restore_client(original)
    assert token == "old-but-not-dead", token
    print("  failed refresh -> caller still gets the token it already had")


def test_chatgpt_account_id_is_extracted_from_the_id_token():
    store = _fresh_store()
    oauth.STORE = store
    id_token = _jwt({"chatgpt_account_id": "acct-abc123", "sub": "user-1"})
    oauth._store_tokens("openai", {"access_token": "at", "refresh_token": "rt",
                                   "expires_in": 3600, "id_token": id_token})
    assert oauth.account_id("openai") == "acct-abc123"
    # And it never leaks into status() either, same contract as the token.
    st = oauth.status("openai")
    assert st["account_id"] == "acct-abc123"
    assert "at" != json.dumps(st)   # sanity: didn't dump the wrong field
    print(f"  account_id recovered from a fabricated id_token: {st['account_id']}")


def test_chatgpt_account_id_falls_back_to_the_nested_auth_claim():
    store = _fresh_store()
    oauth.STORE = store
    id_token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-nested"}})
    oauth._store_tokens("openai", {"access_token": "at", "expires_in": 3600,
                                   "id_token": id_token})
    assert oauth.account_id("openai") == "acct-nested"
    print("  falls back to the namespaced auth claim when the flat one is absent")


def test_account_id_is_none_without_a_grant():
    store = _fresh_store()
    oauth.STORE = store
    assert oauth.account_id("openai") is None
    print("  no grant on file -> account_id() is None, not a raise (only access_token() raises)")


def test_headless_flow_is_registered_for_openai_and_device_flow_for_xai():
    assert isinstance(oauth._flow_for("xai"), oauth.DeviceFlow)
    assert isinstance(oauth._flow_for("openai"), oauth.HeadlessFlow)
    try:
        oauth._flow_for("no-such-provider")
    except KeyError:
        print("  xai -> DeviceFlow, openai -> HeadlessFlow, unknown provider -> KeyError")
        return
    raise AssertionError("expected KeyError for an unregistered provider")


def test_status_has_one_shape_whether_or_not_a_grant_exists():
    """A provider that was never logged in must report the same keys.

    The token proxy's /health reads these straight through; a short dict for an
    unauthorised provider made it raise KeyError on the very state it exists to
    report — a container that looked crashed when it was merely waiting for a
    login.
    """
    oauth.STORE = _fresh_store()
    missing = oauth.status("xai")
    oauth._save({"xai": {"access_token": "t", "refresh_token": "r",
                         "expires_at": time.time() + 600, "scope": "a b"}})
    present = oauth.status("xai")

    assert set(missing) == set(present), (sorted(missing), sorted(present))
    assert missing["authorised"] is False and present["authorised"] is True
    assert missing["expires_in"] is None and present["expires_in"] > 0
    for shape in (missing, present):
        assert "access_token" not in str(shape), "status must never carry a token"
    print(f"  identical keys either way: {sorted(missing)}")


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} oauth tests passed")
