"""OAuth grant tests: refresh, the no-empty-header contract, and the promise
that status() never leaks a token.

No network: every httpx.Client the module creates is redirected onto an
httpx.MockTransport, and SWITCHYARD_AUTH_STORE points at a temp file so a run
never reads or writes a real credential store.
"""
from __future__ import annotations

import base64
import errno
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)  # so `import conftest` resolves under plain `python3`

import conftest  # noqa: F401  (socket guard for plain-script mode)

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


def test_default_store_moves_existing_grant_into_xai_subdir():
    """Issue #117 migration: a pre-existing secrets/oauth.json must end up at
    secrets/xai/oauth.json on first import, with the grant preserved and the
    file mode restored to 0600."""
    root = tempfile.mkdtemp(prefix="oauth_mig_")
    try:
        os.makedirs(os.path.join(root, "secrets"))
        old_path = os.path.join(root, "secrets", "oauth.json")
        payload = {"xai": {"access_token": "tok-old", "refresh_token": "r-old",
                           "expires_at": time.time() + 3600, "scope": "api"}}
        with open(old_path, "w") as fh:
            json.dump(payload, fh)
        os.chmod(old_path, 0o644)  # anything not 0600 — proves the migration re-chmods

        new_path = oauth._default_store(root=root)

        assert new_path == os.path.join(root, "secrets", "xai", "oauth.json"), new_path
        assert os.path.exists(new_path), f"migration did not create {new_path}"
        assert not os.path.exists(old_path), f"migration did not remove {old_path}"
        with open(new_path) as fh:
            assert json.load(fh) == payload, "grant bytes were not preserved"
        assert stat.S_IMODE(os.stat(new_path).st_mode) == 0o600, \
            f"mode is {oct(stat.S_IMODE(os.stat(new_path).st_mode))}, expected 0o600"
        print("  secrets/oauth.json -> secrets/xai/oauth.json, mode 0600, payload preserved")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_default_store_does_not_overwrite_existing_new_path():
    """Issue #117 cycle-3 guard: if both secrets/oauth.json and
    secrets/xai/oauth.json exist (partial migration, manual cp while
    debugging), _default_store() must leave the new file untouched. The new
    one is the valid grant; overwriting it with the (older, possibly stale)
    old one would silently revoke a working session."""
    root = tempfile.mkdtemp(prefix="oauth_both_")
    try:
        os.makedirs(os.path.join(root, "secrets", "xai"))
        old_path = os.path.join(root, "secrets", "oauth.json")
        new_path = os.path.join(root, "secrets", "xai", "oauth.json")
        new_payload = {"xai": {"access_token": "tok-NEW", "refresh_token": "r-new",
                               "expires_at": time.time() + 7200, "scope": "api"}}
        old_payload = {"xai": {"access_token": "tok-OLD", "refresh_token": "r-old",
                               "expires_at": time.time() - 60, "scope": "api"}}
        with open(new_path, "w") as fh:
            json.dump(new_payload, fh)
        with open(old_path, "w") as fh:
            json.dump(old_payload, fh)

        result = oauth._default_store(root=root)

        assert result == new_path
        # The new path must be byte-for-byte the new payload, not the old one.
        with open(new_path) as fh:
            assert json.load(fh) == new_payload, \
                "secrets/xai/oauth.json was overwritten by the older old payload"
        # The old path is also still there — we did not touch it.
        with open(old_path) as fh:
            assert json.load(fh) == old_payload, \
                "secrets/oauth.json was unexpectedly modified"
        print("  both files present -> new path preserved, old path preserved, no overwrite")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_default_store_swallows_filenotfound_from_a_raced_replace():
    """Issue #117 cycle-2 race-loser path: two importers both pass
    `os.path.exists(old_path) and not os.path.exists(new_path)` and the
    winner renames the source before the loser reaches `os.replace`. The
    loser's call must not propagate the FileNotFoundError out of
    _default_store — otherwise module import crashes and every downstream
    reader of oauth.STORE hits NameError.

    Driving the race window with unittest.mock is the only way to exercise
    this code path deterministically in a single-process test: we patch
    `os.path.exists` to claim the source is still there (so the gate
    passes) and patch `os.replace` to raise the exact exception a real race
    would produce. Without this, the second call's `os.path.exists(old_path)`
    would return False and the entire try/except would be skipped — the
    FileNotFoundError handler would never run."""
    root = tempfile.mkdtemp(prefix="oauth_race_")
    try:
        os.makedirs(os.path.join(root, "secrets"))
        new_path_expected = os.path.join(root, "secrets", "xai", "oauth.json")

        # Capture the real os.path.exists before patching, otherwise the
        # patched function would call itself recursively.
        real_exists = oauth.os.path.exists

        def fake_exists(p):
            # The migration gate checks old_path and new_path; the source is
            # still "there" at check time, the destination is not — exactly
            # the window where two concurrent callers both enter the if.
            if p.endswith(os.path.join("secrets", "oauth.json")):
                return True
            if p.endswith(os.path.join("secrets", "xai", "oauth.json")):
                return False
            # os.path.isdir for /app/secrets and os.makedirs(exist_ok=True)
            # use the real filesystem; fall through to the real exists.
            return real_exists(p)

        def race_replace(src, dst):
            # Simulate the winner having already moved the source out from
            # under us between the exists() check and the rename.
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), src)

        with mock.patch.object(oauth.os.path, "exists", side_effect=fake_exists), \
             mock.patch.object(oauth.os, "replace", side_effect=race_replace):
            new_path = oauth._default_store(root=root)

        assert new_path == new_path_expected, new_path
        # The function did not crash, and it returned the path the caller
        # asked it to default to.
        print("  FileNotFoundError on os.replace -> swallowed, new path returned")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_default_store_with_no_old_grant_still_returns_the_new_path():
    """Fresh installs have no secrets/oauth.json to migrate. _default_store()
    must still return the new path and create secrets/xai/ lazily when needed
    (the container side mounts that exact directory)."""
    root = tempfile.mkdtemp(prefix="oauth_fresh_")
    try:
        # No secrets/ at all — a clean checkout.
        new_path = oauth._default_store(root=root)
        assert new_path == os.path.join(root, "secrets", "xai", "oauth.json"), new_path
        assert not os.path.exists(new_path), "must not create the file on a read-only check"
        # A subsequent save (which is what _save() does on first login) finds
        # the directory is created on demand.
        oauth.STORE = new_path
        oauth._save({"xai": {"access_token": "tok", "refresh_token": "r"}})
        assert os.path.exists(new_path)
        print("  fresh checkout -> new path returned, secrets/xai/ created lazily on save")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_concurrent_access_token_calls_do_not_deadlock():
    """N concurrent `access_token` callers serialise through `_store_lock` and
    return without deadlock or stray exception.

    With the cross-process flock in place, the load-decide-refresh-save
    critical section in `access_token` is a single critical section. The
    first caller refreshes (the seed token is near expiry), pushes the
    expiry out by 3600 s, and every subsequent caller that re-enters the
    lock just re-reads the now-fresh entry and skips the refresh. The
    interesting properties the test pins are:

      - no thread deadlocks on the flock (a join timeout would fail first),
      - no unexpected exception escapes the critical section — in
        particular, no FileNotFoundError from a half-renamed tmp file,
        which was the symptom of the unlocked write on the old code path.

    What this test does NOT pin is a "newest refresh token survives under
    two near-simultaneous rotations" contract: with the lock in place that
    property is structurally unobservable from a single process (each caller
    is gated by `expires > time.time() - REFRESH_MARGIN`, so the second
    caller bails before calling `_refresh`). The lost-update property is
    covered by the lock's design — a sibling process that holds the flock
    during the load is the only way two writes could collide — not by
    this test.
    """
    import threading

    oauth.STORE = _fresh_store()
    # Seed with a near-expiry token so the first caller takes the refresh
    # path; subsequent callers in the same race see the fresh expiry and
    # return without re-refreshing.
    oauth._store_tokens("xai", {"access_token": "seed-access",
                                "refresh_token": "r0", "expires_in": 10})

    counter = {"n": 0}
    counter_lock = threading.Lock()
    errors: list[BaseException] = []

    def handler(request: httpx.Request) -> httpx.Response:
        with counter_lock:
            counter["n"] += 1
            seq = counter["n"]
        return httpx.Response(200, json={"access_token": f"at-{seq}",
                                         "refresh_token": f"r{seq}",
                                         "expires_in": 3600})

    original = oauth.httpx.Client
    _stub_client(handler)

    def worker():
        try:
            oauth.access_token("xai")
        except BaseException as exc:           # noqa: BLE001 — collect every error
            errors.append(exc)

    try:
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive(), "thread hung — likely a deadlock in _store_lock"
    finally:
        _restore_client(original)

    assert not errors, f"concurrent access_token raised: {errors!r}"
    # At least one refresh ran — otherwise the test would have exercised
    # only the no-refresh-needed branch and not the lock+save critical
    # section at all. With the lock in place, exactly one refresh is the
    # expected outcome; the other seven callers see the fresh token and
    # skip the refresh.
    assert counter["n"] >= 1, counter
    final = oauth._load()["xai"]
    assert final["refresh_token"].startswith("r"), final
    assert final["refresh_token"] != "r0", final
    print(f"  8 concurrent access_token calls, refresh endpoint hit "
          f"{counter['n']} time(s), store holds "
          f"r{final['refresh_token'][1:]}")


def test_save_writes_tmp_file_with_mode_0600():
    """The temp file holding the live bearer tokens must be created 0600, not
    the 0644 mkstemp defaults to on a umask-022 box. We assert that by
    patching os.replace to stat the tmp path right before swapping it in;
    the post-rename STORE must also be 0600 (belt-and-braces against a
    pre-existing 0644-era file)."""
    import stat as stat_mod

    oauth.STORE = _fresh_store()
    real_replace = oauth.os.replace
    seen_modes: list[int] = []

    def replace_with_check(src, dst):
        # Stat the temp file BEFORE the rename: this is the only window in
        # which its mode is observable at the on-disk path the rename will
        # leave behind. _save has already fchmod'd the fd to 0600; an
        # accident like `os.chmod(tmp, 0o644)` between mkstemp and replace
        # would show up here.
        seen_modes.append(stat_mod.S_IMODE(os.stat(src).st_mode))
        return real_replace(src, dst)

    oauth.os.replace = replace_with_check
    try:
        oauth._save({"xai": {"access_token": "secret", "refresh_token": "r",
                             "expires_at": time.time() + 600}})
    finally:
        oauth.os.replace = real_replace

    assert seen_modes, "test bug: os.replace was never called"
    assert all(m == 0o600 for m in seen_modes), seen_modes
    assert stat_mod.S_IMODE(os.stat(oauth.STORE).st_mode) == 0o600
    print(f"  tmp file modes seen by os.replace: {seen_modes} (== 0o600)")


def test_refresh_endpoint_failure_raises_only_when_token_is_already_expired():
    """A 500 from the refresh endpoint with the token already past expiry
    must raise -- that is the failure mode the proxy's 503 maps. The same
    500 with the token still inside REFRESH_MARGIN keeps returning the
    current token, which is the preserved fallback contract from
    `test_a_failed_refresh_falls_back_to_the_current_token`."""
    # ---- branch A: token already expired -> raise ----------------------
    oauth.STORE = _fresh_store()
    oauth._store_tokens("xai", {"access_token": "expired-but-not-yet-known",
                                "refresh_token": "r1", "expires_in": -10})

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream is on fire")

    original = oauth.httpx.Client
    _stub_client(failing)
    try:
        try:
            oauth.access_token("xai")
        except RuntimeError as exc:
            assert "expired and refresh failed" in str(exc), str(exc)
            print(f"  expired + 500 -> raises: {exc}")
        else:
            raise AssertionError("expected RuntimeError for an expired + failing refresh")
    finally:
        _restore_client(original)

    # ---- branch B: token still valid -> fall back to current ---------
    oauth.STORE = _fresh_store()
    oauth._store_tokens("xai", {"access_token": "still-good",
                                "refresh_token": "r1", "expires_in": 10})
    _stub_client(failing)
    try:
        token = oauth.access_token("xai")
    finally:
        _restore_client(original)
    assert token == "still-good", token
    print("  live + 500 -> falls back to the current token (regression preserved)")


def test_refresh_response_missing_access_token_does_not_raise_keyerror():
    """A 200 with a JSON body that lacks `access_token` would otherwise crash
    `_store_tokens` with KeyError and bubble up as a hard failure. The
    widened exception net in `_refresh` must swallow it; the only question
    for `access_token` is whether the token was already expired (raise)
    or still good (return the existing one). Never a KeyError out."""
    oauth.STORE = _fresh_store()
    oauth._store_tokens("xai", {"access_token": "current",
                                "refresh_token": "r1", "expires_in": -5})

    def missing_access_token(request: httpx.Request) -> httpx.Response:
        # HTTP 200, valid JSON, but the access_token key is not present.
        # `_store_tokens` does `body["access_token"]`, which is a KeyError.
        return httpx.Response(200, json={"expires_in": 3600,
                                         "refresh_token": "r2"})

    original = oauth.httpx.Client
    _stub_client(missing_access_token)
    try:
        try:
            oauth.access_token("xai")
        except KeyError as exc:
            raise AssertionError(f"KeyError leaked from _refresh: {exc!r}") from exc
        except RuntimeError as exc:
            # Expired-and-refresh-failed is the right outcome here -- the
            # KeyError was swallowed and turned into a None from _refresh.
            assert "expired and refresh failed" in str(exc), str(exc)
            print(f"  200 with no access_token + expired -> RuntimeError: {exc}")
        else:
            raise AssertionError("expected either the current token or a "
                                 "RuntimeError, neither happened")
    finally:
        _restore_client(original)


def test_refresh_response_non_dict_json_does_not_raise_typeerror():
    """A 200 with a JSON body that is valid JSON but not a dict (a list, a
    scalar, or `null`) makes `resp.json()` return that value, and the next
    line in `_store_tokens_locked` does `body["access_token"]`, which is
    a `TypeError` on a list/scalar (not a `KeyError`). The widened
    exception net must catch `TypeError` too -- otherwise the proxy maps
    the bubble-up to a 500 instead of the 'expired and refresh failed'
    503 the docstring at `access_token` promises.

    Pins the TypeError branch of `_refresh`'s exception net specifically
    (sibling test to `test_refresh_response_missing_access_token_does_not_raise_keyerror`,
    which pins the KeyError branch).
    """
    oauth.STORE = _fresh_store()
    oauth._store_tokens("xai", {"access_token": "expired",
                                "refresh_token": "r1", "expires_in": -10})

    def list_body(request: httpx.Request) -> httpx.Response:
        # Valid JSON, but not a dict: `_store_tokens_locked` does
        # `body["access_token"]`, which on a list raises TypeError, not
        # KeyError. A scalar (`42`, `"ok"`, `null`) hits the same path
        # -- a list is the most likely real-world shape (a buggy
        # provider returning its token as `[...]`).
        return httpx.Response(200, json=["not", "a", "dict"])

    original = oauth.httpx.Client
    _stub_client(list_body)
    try:
        try:
            oauth.access_token("xai")
        except TypeError as exc:
            raise AssertionError(
                f"TypeError leaked from _refresh: {exc!r}") from exc
        except RuntimeError as exc:
            assert "expired and refresh failed" in str(exc), str(exc)
            print(f"  200 with non-dict JSON + expired -> RuntimeError: {exc}")
        else:
            raise AssertionError(
                "expected a RuntimeError (expired + refresh-failed)")
    finally:
        _restore_client(original)


def test_refresh_swallows_json_decode_error_and_save_oserror():
    """Both `resp.json()` (a ValueError on a non-JSON body) and an OSError
    from `_save` (disk full, read-only mount, etc.) must be caught inside
    `_refresh` and returned as None, never escaping to `access_token`.

    The token here is already expired so a successful refresh would have
    raised 'expired and refresh failed' -- but the test is about the absence
    of ValueError / OSError, so we just check no other exception type leaks.
    """
    oauth.STORE = _fresh_store()
    oauth._store_tokens("xai", {"access_token": "expired",
                                "refresh_token": "r1", "expires_in": -10})

    def not_json(request: httpx.Request) -> httpx.Response:
        # Plain text body: resp.json() will raise ValueError inside _refresh.
        return httpx.Response(200, text="<html>oops</html>")

    original = oauth.httpx.Client
    real_save = oauth._save

    def save_oserror(data):
        raise OSError("simulated disk failure inside _save")

    _stub_client(not_json)
    oauth._save = save_oserror
    try:
        try:
            oauth.access_token("xai")
        except (ValueError, OSError) as exc:
            raise AssertionError(
                f"{type(exc).__name__} leaked out of _refresh: {exc!r}") from exc
        except RuntimeError as exc:
            # Both branches end up returning None from _refresh; the expired
            # token then raises the 'expired and refresh failed' message.
            assert "expired and refresh failed" in str(exc), str(exc)
            print(f"  ValueError+OSError swallowed -> RuntimeError: {exc}")
        else:
            raise AssertionError("expected a RuntimeError (expired + refresh-failed)")
    finally:
        _restore_client(original)
        oauth._save = real_save


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
