"""Switchyard's own OAuth grants for subscription providers.

Why this exists: shelling out to a vendor CLI means routing through that CLI's
agent harness, and the harness owns the tool loop. A caller's tool definitions
can therefore never reach the model, and thousands of tokens of someone else's
system prompt ride along on every request. Agentic work is mostly tool calling,
so the CLI route cannot serve the main use case at all.

The subscriptions themselves have no such limitation. They authenticate ordinary
HTTP APIs where tool calling is a first-class feature — `api.x.ai/v1` is
OpenAI-compatible, so a request body with `tools` passes straight through and
`tool_calls` come straight back. Third-party clients are plainly contemplated:
OpenCode uses exactly these flows. So Switchyard takes out **its own grant**
rather than borrowing another client's credential.

The flow parameters below are the public OAuth client registrations these
providers publish for CLI use. Tokens live under `secrets/` (gitignored, 0600),
are refreshed shortly before expiry, and are never logged — `status()` reports
whether a grant exists, never what it is.

Usage:

    python -m switchyard.oauth login xai      # prints a URL and code to approve
    python -m switchyard.oauth status         # says what is authorised
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

def _default_store() -> str:
    """Where the token store lives, host or container.

    The containers mount the repo's ./secrets at /app/secrets, so both sides
    read the same file — but only the container has /app. Defaulting to the
    container path meant `login` run from the shell (which is the only way to
    run it: the grant needs a human at a browser) completed the grant and then
    lost the tokens to a read-only /app.
    """
    override = os.environ.get("SWITCHYARD_AUTH_STORE")
    if override:
        return override
    if os.path.isdir("/app/secrets"):
        return "/app/secrets/oauth.json"
    # The repo checkout: this file is <root>/switchyard/oauth.py.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "secrets", "oauth.json")


STORE = _default_store()
REFRESH_MARGIN = 120   # refresh this many seconds before expiry


@dataclass(frozen=True)
class DeviceFlow:
    """An RFC 8628 device-authorisation grant.

    The right shape for a headless service: no loopback listener to expose and
    no browser redirect to intercept, which is exactly why the vendor CLIs' own
    browser-callback logins fail inside a container.
    """
    provider: str
    client_id: str
    device_code_url: str
    token_url: str
    scopes: str
    api_base: str
    grant_type: str = "urn:ietf:params:oauth:grant-type:device_code"
    poll_seconds: int = 5
    timeout_seconds: int = 300
    extra_headers: dict[str, str] = field(default_factory=dict)


FLOWS: dict[str, DeviceFlow] = {
    # Scopes include `api:access`, which is what makes the subscription usable
    # against the ordinary API rather than only through a vendor client.
    "xai": DeviceFlow(
        provider="xai",
        client_id="b1a00492-073a-47ea-816f-4c329264a828",
        device_code_url="https://auth.x.ai/oauth2/device/code",
        token_url="https://auth.x.ai/oauth2/token",
        scopes="openid profile email offline_access grok-cli:access api:access",
        api_base="https://api.x.ai/v1",
    ),
}


@dataclass(frozen=True)
class HeadlessFlow:
    """OpenAI's own device-style flow for a ChatGPT seat — NOT RFC 8628.

    Reverse-engineered from OpenCode's `chatgpt-headless` integration: the
    strings and the surrounding logic are both present verbatim in the binary
    at `/usr/local/lib/node_modules/opencode-ai/bin/opencode.exe` inside the
    opencode-go-sidecar container (`grep -a` for the literals below finds them;
    the file has no `strings` in that image; this was traced in the since-removed
    grok-sidecar, which ran the same image). Traced with:

        grep -a -o "id:an,.\\{0,1500\\}" opencode.exe   # the headless method
        grep -a -o "function tt(o,e,i){.\\{0,200\\}" opencode.exe   # the code exchange
        grep -a -o ".\\{0,150\\}chatgpt_account_id.\\{0,300\\}" opencode.exe

    The shape is a two-step device code exchange feeding a normal
    authorization_code+PKCE token exchange — except the *server* hands back the
    PKCE verifier instead of the client minting one, so there is no code
    challenge to generate on this end:

      1. POST {usercode_url}  {"client_id": ...}
         -> {device_auth_id, user_code, interval}
      2. Show the human `device_page_url` and the user_code.
      3. Poll POST {poll_url}  {"device_auth_id", "user_code"} every
         interval+3s. 403/404 means "not yet approved"; anything else is
         terminal. On success: {authorization_code, code_verifier}.
      4. POST {token_url} (form-encoded, same endpoint refresh uses)
         grant_type=authorization_code, code=<authorization_code>,
         redirect_uri={redirect_uri}, client_id=..., code_verifier=<verifier>
         -> the usual {access_token, refresh_token, expires_in, id_token}.

    None of this has been run — no login was performed to get here, per the
    constraint on this task. It is transcribed directly from the CLI's own
    control flow, not guessed, but treat step 3's poll semantics (which
    statuses mean "pending" vs "denied") as worth reconfirming against a real
    device_auth_id before depending on it in anger.
    """
    provider: str
    client_id: str
    usercode_url: str
    poll_url: str
    token_url: str
    redirect_uri: str
    device_page_url: str
    api_base: str
    poll_seconds: int = 5
    timeout_seconds: int = 600


HEADLESS_FLOWS: dict[str, HeadlessFlow] = {
    "openai": HeadlessFlow(
        provider="openai",
        client_id="app_EMoamEEZ73f0CkXaXp7hrann",
        usercode_url="https://auth.openai.com/api/accounts/deviceauth/usercode",
        poll_url="https://auth.openai.com/api/accounts/deviceauth/token",
        token_url="https://auth.openai.com/oauth/token",
        redirect_uri="https://auth.openai.com/deviceauth/callback",
        device_page_url="https://auth.openai.com/codex/device",
        # The ChatGPT seat is reached through Codex's own backend, not the
        # public api.openai.com — and only the Responses endpoint, not
        # chat/completions. See sidecars/token_proxy/server.py for why that
        # endpoint cannot simply be forwarded to like xAI's.
        api_base="https://chatgpt.com/backend-api/codex",
    ),
}


# ---------------------------------------------------------------- storage ----
def _load() -> dict[str, Any]:
    try:
        with open(STORE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save(data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    tmp = STORE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, STORE)
    try:
        os.chmod(STORE, 0o600)
    except OSError:
        pass


def status(provider: str) -> dict[str, Any]:
    """Whether a grant exists and how long it has left. Never the grant itself."""
    entry = _load().get(provider) or {}
    if not entry:
        # The same keys as an authorised grant, so a caller can report on a
        # provider that has never been logged in without special-casing it —
        # /health crashed on a KeyError doing exactly that.
        return {"provider": provider, "authorised": False, "expires_in": None,
                "has_refresh": False, "scopes": "", "account_id": None}
    expires = entry.get("expires_at") or 0
    return {
        "provider": provider,
        "authorised": bool(entry.get("access_token")),
        "expires_in": max(0, int(expires - time.time())) if expires else None,
        "has_refresh": bool(entry.get("refresh_token")),
        "scopes": entry.get("scope", ""),
        # Not a secret — a tenant identifier the ChatGPT backend needs in a
        # header, not a credential on its own — so unlike the token it is safe
        # to surface here for debugging.
        "account_id": entry.get("account_id"),
    }


# --------------------------------------------------------------- JWT claims --
def _jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT's payload without verifying the signature.

    Only ever used to pull a non-secret routing claim (the ChatGPT account id)
    out of a token this process itself just received over TLS from the
    provider. Never used to authorise anything — the token itself does that.
    """
    import base64
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, TypeError):
        return {}


def _chatgpt_account_id(body: dict[str, Any]) -> str | None:
    """Same claim precedence OpenCode uses: id_token first, access_token
    fallback, each checked at the same three locations a claim might be."""
    for token_key in ("id_token", "access_token"):
        token = body.get(token_key)
        if not token:
            continue
        claims = _jwt_claims(token)
        account = (claims.get("chatgpt_account_id")
                  or (claims.get("https://api.openai.com/auth") or {}).get("chatgpt_account_id")
                  or ((claims.get("organizations") or [{}])[0] or {}).get("id"))
        if account:
            return account
    return None


# ------------------------------------------------------------------ login ----
def _flow_for(provider: str) -> DeviceFlow | HeadlessFlow:
    if provider in FLOWS:
        return FLOWS[provider]
    if provider in HEADLESS_FLOWS:
        return HEADLESS_FLOWS[provider]
    raise KeyError(f"no OAuth flow registered for {provider!r}")


def begin_login(provider: str) -> dict[str, Any]:
    """Ask the provider to start a login. Returns what the human must do.

    Dispatches on flow shape: xAI is a standard RFC 8628 device grant; OpenAI's
    ChatGPT seat is not (see HeadlessFlow's docstring). Both return the same
    shape so the caller — including `finish_login` and the CLI — need not care
    which one it got.
    """
    flow = _flow_for(provider)
    if isinstance(flow, HeadlessFlow):
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                flow.usercode_url,
                json={"client_id": flow.client_id},
                headers={"Accept": "application/json"},
            )
        resp.raise_for_status()
        body = resp.json()
        return {
            "device_auth_id": body["device_auth_id"],
            "user_code": body.get("user_code"),
            "verification_uri": flow.device_page_url,
            "interval": int(body.get("interval") or flow.poll_seconds),
            "expires_in": flow.timeout_seconds,
        }

    with httpx.Client(timeout=30) as client:
        resp = client.post(
            flow.device_code_url,
            data={"client_id": flow.client_id, "scope": flow.scopes},
            headers={"Accept": "application/json"},
        )
    resp.raise_for_status()
    body = resp.json()
    return {
        "device_code": body["device_code"],
        "user_code": body.get("user_code"),
        "verification_uri": (body.get("verification_uri_complete")
                             or body.get("verification_uri")),
        "interval": int(body.get("interval") or flow.poll_seconds),
        "expires_in": int(body.get("expires_in") or flow.timeout_seconds),
    }


def finish_login(provider: str, started: dict[str, Any],
                 deadline: float) -> dict[str, Any]:
    """Poll until the human approves, then store the grant.

    Takes the whole dict `begin_login` returned rather than picking fields out
    of it, because the two flow shapes need different fields to keep polling
    (a bare device_code for xAI; a device_auth_id/user_code pair for OpenAI).
    """
    flow = _flow_for(provider)
    interval = int(started["interval"])

    if isinstance(flow, HeadlessFlow):
        device_auth_id, user_code = started["device_auth_id"], started["user_code"]
        with httpx.Client(timeout=30) as client:
            while time.time() < deadline:
                time.sleep(interval)
                resp = client.post(
                    flow.poll_url,
                    json={"device_auth_id": device_auth_id, "user_code": user_code},
                    headers={"Accept": "application/json"},
                )
                if resp.status_code == 200:
                    grant = resp.json()
                    token_resp = client.post(
                        flow.token_url,
                        data={"grant_type": "authorization_code",
                              "code": grant["authorization_code"],
                              "redirect_uri": flow.redirect_uri,
                              "client_id": flow.client_id,
                              "code_verifier": grant.get("code_verifier", "")},
                        headers={"Accept": "application/json"},
                    )
                    token_resp.raise_for_status()
                    _store_tokens(provider, token_resp.json())
                    return status(provider)
                # 403/404 is "not approved yet" in OpenCode's own polling loop;
                # anything else is treated as terminal rather than retried
                # forever against a code that will never be approved.
                if resp.status_code not in (403, 404):
                    raise RuntimeError(
                        f"device authorisation failed: {resp.status_code} {resp.text[:200]}")
        raise TimeoutError("device authorisation expired before it was approved")

    with httpx.Client(timeout=30) as client:
        while time.time() < deadline:
            time.sleep(interval)
            resp = client.post(
                flow.token_url,
                data={"client_id": flow.client_id,
                      "device_code": started["device_code"],
                      "grant_type": flow.grant_type},
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                _store_tokens(provider, resp.json())
                return status(provider)

            body: dict[str, Any] = {}
            try:
                body = resp.json()
            except ValueError:
                pass
            error = body.get("error")
            if error in ("authorization_pending", "slow_down"):
                if error == "slow_down":
                    interval += 5
                continue
            raise RuntimeError(
                f"device authorisation failed: {error or resp.text[:200]}")
    raise TimeoutError("device authorisation expired before it was approved")


def _store_tokens(provider: str, body: dict[str, Any]) -> None:
    data = _load()
    entry = data.get(provider) or {}
    entry.update({
        "access_token": body["access_token"],
        "token_type": body.get("token_type", "Bearer"),
        "scope": body.get("scope", getattr(FLOWS.get(provider), "scopes", "")),
    })
    # A refresh token is only issued once in some flows; never overwrite a good
    # one with nothing, or the grant becomes unrenewable.
    if body.get("refresh_token"):
        entry["refresh_token"] = body["refresh_token"]
    if body.get("expires_in"):
        entry["expires_at"] = time.time() + int(body["expires_in"])
    account_id = _chatgpt_account_id(body)
    if account_id:
        entry["account_id"] = account_id
    data[provider] = entry
    _save(data)


# ---------------------------------------------------------------- refresh ----
def access_token(provider: str) -> str:
    """A currently-valid access token, refreshed if close to expiry.

    Raises when there is no grant, so a misconfigured lane fails loudly instead
    of sending an empty Authorization header — which would arrive as an auth
    error and be classified as a dead credential rather than a setup mistake.
    """
    entry = _load().get(provider) or {}
    token = entry.get("access_token")
    if not token:
        raise RuntimeError(
            f"no OAuth grant for {provider!r}; run "
            f"`python -m switchyard.oauth login {provider}`")

    expires = entry.get("expires_at") or 0
    if expires and time.time() > expires - REFRESH_MARGIN:
        refreshed = _refresh(provider, entry)
        if refreshed:
            return refreshed
    return token


def account_id(provider: str) -> str | None:
    """The ChatGPT-Account-Id header value, when the provider needs one.

    Unlike access_token(), this never raises: a missing account id is only
    fatal for providers whose wire format actually needs it (OpenAI), and
    that check belongs to the caller that knows which header it is filling.
    """
    return (_load().get(provider) or {}).get("account_id")


def _refresh(provider: str, entry: dict[str, Any]) -> str | None:
    """Best-effort refresh. Returns None to let the caller use what it has:
    a failed refresh is not proof the current token has expired."""
    refresh = entry.get("refresh_token")
    if not refresh:
        return None
    flow = _flow_for(provider)
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                flow.token_url,
                data={"client_id": flow.client_id,
                      "grant_type": "refresh_token",
                      "refresh_token": refresh},
                headers={"Accept": "application/json"},
            )
        if resp.status_code != 200:
            return None
        _store_tokens(provider, resp.json())
        return (_load().get(provider) or {}).get("access_token")
    except httpx.HTTPError:
        return None


# -------------------------------------------------------------------- cli ----
def main(argv: list[str] | None = None) -> int:
    import argparse

    providers = sorted(set(FLOWS) | set(HEADLESS_FLOWS))
    ap = argparse.ArgumentParser(prog="switchyard.oauth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    login = sub.add_parser("login", help="take out a grant for a provider")
    login.add_argument("provider", choices=providers)
    st = sub.add_parser("status", help="report grants without revealing them")
    st.add_argument("provider", nargs="?", choices=providers)
    args = ap.parse_args(argv)

    if args.cmd == "status":
        for name in ([args.provider] if args.provider else providers):
            print(json.dumps(status(name)))
        return 0

    started = begin_login(args.provider)
    print(f"\n  Open: {started['verification_uri']}")
    if started.get("user_code"):
        print(f"  Code: {started['user_code']}")
    print(f"\n  Waiting up to {started['expires_in']}s for approval…\n", flush=True)
    result = finish_login(args.provider, started, time.time() + started["expires_in"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
