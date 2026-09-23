"""Just-in-time login ceremony (issue #227).

A host-side, headful-Playwright CLI that opens a plan's ``login_url`` in a
local Chromium with an ephemeral profile, lets the operator type username /
password / 2FA into the real browser (credentials never reach SwitchYard),
and on reaching the post-auth console POSTs only cookies whose domains match
the plan's allowlist to the portal's existing
``POST /admin/probes/{plan}/cookie`` endpoint (see
``switchyard/portal/app.py:1047``).

Tier-1 only: the process reads no password input. The user types into the
real browser window; the cookies are harvested from Chromium's ``context``,
filtered against the probe URL's host, and the only thing this process does
with them is POST them straight to the portal. There is no local file or
Redis write of the cookie on this side: the ephemeral profile directory is
deleted in ``finally``, the portal stores the cookie in Redis as it does for
the manual paste path, and the script never prints the cookie value.

The Playwright import is lazy on purpose. The ceremony runs on the operator's
host, never inside a SwitchYard container. The gateway / portal / sidecar /
token-proxy images ``COPY`` the whole ``switchyard/`` package, so a top-level
``import playwright`` here would force them to install Playwright too. Doing
the import inside :func:`run` keeps them unaffected, and gives a clear
install-hint error when an operator runs the ceremony without Playwright
present.

Invocation::

    python3 -m switchyard.ceremony <plan>          # actually launch Chromium
    python3 -m switchyard.ceremony <plan> --dry-run # print the plan and exit
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from urllib.parse import urlparse

import httpx

from .models import CONFIG_PATH, Plan, Probe, load as load_registry


# Surfaces in the RuntimeError when Playwright is missing. ``pip install
# playwright && playwright install chromium`` is the documented two-step; the
# second installs the bundled Chromium browser binary, which the first does
# not.
PLAYWRIGHT_INSTALL_HINT = "pip install playwright && playwright install chromium"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m switchyard.ceremony",
        description=(
            "Just-in-time login ceremony for a probe's session cookie. "
            "Opens a headful Chromium on the operator's host, lets the user "
            "log in to the plan's console in the real browser, and POSTs "
            "the resulting cookies to the portal's existing "
            "/admin/probes/{plan}/cookie endpoint."
        ),
    )
    parser.add_argument(
        "plan",
        help="Plan key whose ceremony to run (must have probe.login_ceremony: true).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan's ceremony config and exit; launch nothing.",
    )
    parser.add_argument(
        "--plans",
        default=None,
        help=f"Path to plans.yaml (default: {CONFIG_PATH}).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Maximum wait for the browser to reach the probe host (default 180).",
    )
    parser.add_argument(
        "--portal-url",
        default=None,
        help=(
            "Portal base URL (default: $SWITCHYARD_PORTAL_URL or "
            "http://localhost:4001)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan, probe = _resolve_plan(args.plan, args.plans)
    if plan is None or probe is None:
        return 2

    if args.dry_run:
        print_dry_run(plan, timeout_seconds=args.timeout_seconds,
                      portal_url=args.portal_url)
        return 0
    return run(plan, timeout_seconds=args.timeout_seconds,
               portal_url=args.portal_url)


def _resolve_plan(plan_key: str, plans_path: str | None
                  ) -> tuple[Plan | None, Probe | None]:
    """Load the registry, find the plan, gate on login_ceremony.

    Returns ``(plan, probe)`` on success or ``(None, None)`` after printing
    a refusal to stderr. The CLI refuses (rather than launching) plans
    without the opt-in flag — that gate is the single most important
    guarantee this module ships, because the alternative is a credential-
    capturing ceremony firing for any cookie probe in the file.
    """
    try:
        registry = load_registry(plans_path)
    except FileNotFoundError as exc:
        print(f"error: cannot load plans: {exc}", file=sys.stderr)
        return None, None

    plan = registry.plans.get(plan_key)
    if plan is None:
        known = sorted(registry.plans)
        print(f"error: plan {plan_key!r} not found; known plans: {known}",
              file=sys.stderr)
        return None, None
    probe = plan.probe
    if probe is None:
        print(f"error: plan {plan_key!r} has no probe configured; the "
              f"ceremony is for cookie probes only", file=sys.stderr)
        return None, None
    if not probe.login_ceremony:
        print(f"error: plan {plan_key!r} has probe.login_ceremony: false; "
              f"the ceremony CLI only runs plans that explicitly opt in. "
              f"Set probe.login_ceremony: true and probe.login_url in "
              f"config/plans.yaml first.", file=sys.stderr)
        return None, None
    if not probe.login_url:
        # The loader already raises on this; loud-fail here too so a future
        # refactor of the loader cannot silently regress this guard.
        print(f"error: plan {plan_key!r} has probe.login_ceremony: true "
              f"but no probe.login_url; refusing to launch", file=sys.stderr)
        return None, None
    return plan, probe


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def print_dry_run(plan: Plan, *, timeout_seconds: float = 180.0,
                  portal_url: str | None = None) -> None:
    """Print the plan's ceremony config and exit.

    Used by ``--dry-run`` and by the offline test that proves the CLI
    refuses non-opted-in plans. Output is line-oriented and key-aligned so
    the printed shape matches what an operator would copy/paste. The
    resolved ``timeout_seconds`` and ``portal_url`` are echoed (not the
    argparse defaults) so the operator can see what the live run will
    actually use.
    """
    probe = plan.probe
    assert probe is not None
    resolved_portal = (portal_url or os.environ.get(
        "SWITCHYARD_PORTAL_URL", "http://localhost:4001"))
    print(f"plan:             {plan.key}")
    print(f"probe.url:        {probe.url}")
    print(f"probe.host:       {urlparse(probe.url).hostname or ''}")
    print(f"login_url:        {probe.login_url}")
    print(f"timeout_seconds:  {timeout_seconds}")
    print(f"portal_url:       {resolved_portal}")
    print(f"cookie_allowlist: {sorted(cookie_allowlist(probe.url))}")
    print(f"login_ceremony:   {probe.login_ceremony}")


# ---------------------------------------------------------------------------
# Allowlist + cookie header assembly (unit-testable, no Playwright)
# ---------------------------------------------------------------------------
def cookie_allowlist(probe_url: str) -> set[str]:
    """Domains any cookie the harvest may keep.

    A cookie's ``domain`` attribute is the host it is scoped to; we accept
    the probe host itself. ``cookie_host_allowed`` lower-cases and
    ``lstrip(".")``-strips both sides before matching, so a single
    bare-host entry covers the bare form (``example.com``), the
    leading-dot form Chromium sets for site+subdomain scope
    (``.example.com``), and every deeper subdomain via suffix match.

    Bare-host matching only — no path filtering. The Network tab's
    ``name=value; ...`` form is built from these cookies alone; anything
    outside this set triggers an ``unexpected cookie host`` abort, which
    is the central safety property of the ceremony.
    """
    host = (urlparse(probe_url).hostname or "").lower()
    if not host:
        return set()
    return {host}


def cookie_host_allowed(cookie_domain: str, allowlist: set[str]) -> bool:
    """True when ``cookie_domain`` is the probe host or a subdomain of it.

    Chromium returns cookies with ``domain`` values like ``example.com`` or
    ``.example.com`` (the latter means "this site and every subdomain"). A
    probe host of ``console.example.com`` therefore accepts:

      - ``console.example.com``           (the probe host itself)
      - ``.console.example.com``          (its dot-prefixed form)
      - ``app.console.example.com``       (a deeper subdomain via suffix)
      - ``.app.console.example.com``      (dot-prefixed deeper subdomain)

    The allowlist does NOT contain ``example.com`` — the bare apex of
    ``console.example.com`` — because the allowlist is derived from
    ``probe.url``'s host and a probe host of ``console.example.com``
    means the operator has explicitly opted the probe into polling
    ``console.example.com``, not ``example.com``. Bare-apex cookies are
    only accepted when the probe host IS the apex. The dot-stripped
    comparison collapses both ``host`` and ``.host`` to the same set so
    callers may pass either form. The allowlist is lower-cased so
    mixed-case entries do not split the match.
    """
    d = (cookie_domain or "").lower().lstrip(".")
    if not d:
        return False
    bare_allowlist = {h.lower().lstrip(".") for h in allowlist}
    if d in bare_allowlist:
        return True
    return any(d.endswith("." + h) for h in bare_allowlist)


def filter_cookies_or_abort(cookies: list[dict], allowlist: set[str]) -> list[dict]:
    """Apply the allowlist; raise on any cookie outside it.

    The abort message is the literal ``unexpected cookie host: <domain>``
    shape the task spec calls for. Returns the cookies that survived, in
    Chromium's order, so :func:`assemble_cookie_header` produces the same
    ``name=value; ...`` the Network tab would.
    """
    offenders: list[str] = []
    kept: list[dict] = []
    for c in cookies:
        domain = (c.get("domain") or "")
        if not cookie_host_allowed(domain, allowlist):
            offenders.append(domain)
        else:
            kept.append(c)
    if offenders:
        # The first offender is enough — aborting on the first bad cookie is
        # the contract. The full allowlist is included so the operator can
        # see what was expected without scrolling back through config.
        raise RuntimeError(
            f"unexpected cookie host: {offenders[0]!r} "
            f"(allowlist is {sorted(allowlist)})"
        )
    return kept


def assemble_cookie_header(cookies: list[dict]) -> str:
    """Build the ``Cookie: name=value; ...`` header exactly as the Network tab
    gives it.

    Each cookie's ``name`` and ``value`` are joined with ``=``, pairs are
    joined with ``; ``, in the order Chromium returned them (storage order,
    which matches the request order Chromium would also use). Cookies with
    an empty name are skipped — the Network tab drops those silently too.
    """
    pairs: list[str] = []
    for c in cookies:
        name = c.get("name") or ""
        value = c.get("value") or ""
        if not name:
            continue
        pairs.append(f"{name}={value}")
    return "; ".join(pairs)


def cookie_fingerprint(cookie_header: str) -> str:
    """Length + short hash, never any of the value.

    Mirrors the portal's own fingerprint shape (see
    ``switchyard/probes.py`` / ``tests/test_probes.py``) so the printed line
    looks familiar: ``<N> chars, #<8 hex>``. Truncated to 8 hex chars the
    same way the portal's existing fingerprint is.
    """
    n = len(cookie_header)
    digest = hashlib.sha256(cookie_header.encode()).hexdigest()[:8]
    return f"{n} chars, #{digest}"


# ---------------------------------------------------------------------------
# Portal POST
# ---------------------------------------------------------------------------
def post_cookie_to_portal(plan_key: str, cookie_header: str, portal_url: str,
                          timeout: float = 15.0) -> tuple[int, str]:
    """POST the assembled cookie header to the portal's existing endpoint.

    The portal route is ``POST /admin/probes/{plan_key}/cookie`` (form-encoded
    ``cookie`` field — see ``switchyard/portal/app.py:1047``), the same one a
    paste from the board uses. Returns ``(status, body)``; raises on a
    transport-level failure so the caller surfaces a loud error rather than
    fall back to writing the cookie anywhere local.
    """
    url = f"{portal_url.rstrip('/')}/admin/probes/{plan_key}/cookie"
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, data={"cookie": cookie_header})
        return r.status_code, r.text


# ---------------------------------------------------------------------------
# Live run (Playwright path)
# ---------------------------------------------------------------------------
def run(plan: Plan, *, timeout_seconds: float = 180.0,
        portal_url: str | None = None) -> int:
    """Launch the headful browser, wait for the probe host, harvest cookies,
    POST them. Credentials never touch this process: the operator types them
    into the real browser window. Any cookie outside the allowlist is a hard
    error with no fallback that stores the cookie locally.
    """
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Playwright is required for the login ceremony; install it "
            f"with `{PLAYWRIGHT_INSTALL_HINT}` (import error: {exc})"
        ) from exc

    probe = plan.probe
    assert probe is not None and probe.login_url  # _resolve_plan gates this
    portal_base = portal_url or os.environ.get(
        "SWITCHYARD_PORTAL_URL", "http://localhost:4001")
    allowlist = cookie_allowlist(probe.url)
    user_data_dir = tempfile.mkdtemp(prefix="switchyard-ceremony-")

    try:
        # Import inside the function (lazy) so the gateway / portal /
        # sidecar / token-proxy images — which COPY the whole switchyard/
        # package — never pick up a Playwright import at module load.
        from playwright.sync_api import sync_playwright as _sync_playwright
        return _run_with_playwright(
            plan, probe, allowlist, portal_base, user_data_dir,
            timeout_seconds, _sync_playwright,
        )
    finally:
        # Nothing on disk, nothing in Redis. The user-data-dir holds the
        # Chromium profile, the session cookie, the cache — all gone on
        # exit. Storage state, if we ever wrote it, would live next to
        # it; we don't write one.
        shutil.rmtree(user_data_dir, ignore_errors=True)


def _run_with_playwright(plan: Plan, probe: Probe, allowlist: set[str],
                         portal_base: str, user_data_dir: str,
                         timeout_seconds: float,
                         sync_playwright) -> int:
    """Inner ceremony loop, isolated so tests can drive it with a fake
    Playwright object that returns canned cookies without launching a real
    browser (the conftest socket guard would block any network anyway)."""
    probe_host = (urlparse(probe.url).hostname or "").lower()
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir,
            headless=False,
        )
        try:
            page = (context.pages[0] if context.pages
                    else context.new_page())
            page.goto(probe.login_url)
            # Success criterion: page URL host matches the probe host. The
            # loader pins `login_url`'s host to `probe.url`'s host, so the
            # `goto` lands on the probe host and the wait passes immediately
            # on a same-host login form. A same-host OAuth flow (an IdP on
            # a sub-path of the probe host, with a same-host redirect back)
            # is the only OAuth shape that survives config load, and the
            # wait succeeds once the final same-host navigation lands.
            # Cross-host OAuth (an IdP on a different host that redirects
            # back) is rejected at config load.
            page.wait_for_url(
                lambda url: (urlparse(url).hostname or "").lower() == probe_host,
                timeout=int(timeout_seconds * 1000),
            )
            cookies = context.cookies()
            kept = filter_cookies_or_abort(cookies, allowlist)
            if not kept:
                raise RuntimeError(
                    f"no cookies harvested for {plan.key!r}; the page "
                    f"reached {probe_host!r} but no session cookie was set"
                )
            cookie_header = assemble_cookie_header(kept)
            status, body = post_cookie_to_portal(
                plan.key, cookie_header, portal_base)
            if status >= 400:
                raise RuntimeError(
                    f"portal POST {portal_base}/admin/probes/{plan.key}/cookie "
                    f"returned HTTP {status}: {body!r}"
                )
            print(f"ceremony: stored {cookie_fingerprint(cookie_header)}")
            return 0
        finally:
            try:
                context.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
