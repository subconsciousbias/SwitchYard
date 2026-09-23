"""Tests for ``switchyard.ceremony`` — the just-in-time login ceremony.

Tier-1 only: the ceremony never reads a password from the process. The
credential flows operator -> real Chromium window -> harvested cookies ->
portal POST, and nothing in this script touches it. The tests therefore
exercise the allowlist, the cookie-header assembly, the loader validation,
and the CLI's refusal gate without ever launching a browser. A single
Playwright happy-path test is gated on the import being available; the
conftest socket guard makes a real run infeasible anyway, and the test is
written so its absence is a skip, not a failure.
"""
from __future__ import annotations

import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)  # so `import conftest` resolves under plain `python3`

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
# Must run BEFORE `import switchyard.ceremony` — that module imports
# `switchyard.models`, and `models.CONFIG_PATH` captures the env var at
# import time. Pytest collects test files alphabetically, so leaving this
# unset here would let a sibling file (e.g. test_messages_hooks) import
# switchyard.models with the default `/app/config/plans.yaml` baked in.
os.environ["SWITCHYARD_PLANS"] = os.path.join(ROOT, "config", "plans.example.yaml")

import conftest  # noqa: F401  (socket guard for plain-script mode)

from switchyard import ceremony  # noqa: E402
from switchyard import models  # noqa: E402
from switchyard.ceremony import (  # noqa: E402
    assemble_cookie_header,
    build_parser,
    cookie_allowlist,
    cookie_fingerprint,
    cookie_host_allowed,
    filter_cookies_or_abort,
    main as ceremony_main,
    print_dry_run,
)
from switchyard.models import (  # noqa: E402
    Probe,
    _parse_probe,
    _url_host,
)


# ---------------------------------------------------------------------------
# Schema: Probe carries the new opt-in fields with the documented defaults.
# ---------------------------------------------------------------------------
def test_probe_carries_login_ceremony_and_login_url_with_documented_defaults():
    """Default values are False / "" so a freshly written Probe behaves like
    the pre-ceremony code: the CLI would refuse it, and no extra fields are
    required from operators who don't opt in."""
    p = Probe(url="https://example.com/health")
    assert p.login_ceremony is False
    assert p.login_url == ""


def test_probe_constructor_accepts_both_new_fields_explicitly():
    p = Probe(url="https://example.com/health",
              login_ceremony=True, login_url="https://example.com/login")
    assert p.login_ceremony is True
    assert p.login_url == "https://example.com/login"


# ---------------------------------------------------------------------------
# Loader validation: the loader rejects bad opt-in plans loudly.
# ---------------------------------------------------------------------------
def test_loader_accepts_login_ceremony_false_without_login_url():
    """`login_ceremony: false` (or unset) with no `login_url` is the default
    behaviour today; the loader must not raise on it."""
    probe = _parse_probe({
        "kind": "cookie",
        "url": "https://example.com/health",
    })
    assert probe is not None
    assert probe.login_ceremony is False
    assert probe.login_url == ""


def test_loader_rejects_login_ceremony_true_without_login_url():
    """Opting in without a target URL is the worst kind of mistake: the CLI
    would refuse at runtime, but a probe that silently broke because the
    operator forgot to fill in the URL is worse. Loud-parse at load."""
    try:
        _parse_probe({
            "kind": "cookie",
            "url": "https://example.com/health",
            "login_ceremony": True,
        })
    except ValueError as exc:
        assert "login_url" in str(exc), exc
        return
    raise AssertionError("loader accepted login_ceremony:true with no login_url")


def test_loader_rejects_login_ceremony_with_host_mismatch():
    """Host pinning: the allowlist is derived from `url`'s host, so a
    `login_url` on a different host would silently POST cookies for a
    different site than the operator logged into. The loader rejects it."""
    try:
        _parse_probe({
            "kind": "cookie",
            "url": "https://example.com/health",
            "login_ceremony": True,
            "login_url": "https://attacker.example.org/login",
        })
    except ValueError as exc:
        assert "host" in str(exc), exc
        assert "attacker.example.org" in str(exc), exc
        return
    raise AssertionError("loader accepted login_url with mismatched host")


def test_loader_accepts_login_ceremony_with_matching_host():
    """The matching-host happy path: same host on both URLs parses cleanly."""
    probe = _parse_probe({
        "kind": "cookie",
        "url": "https://platform.example.com/health",
        "login_ceremony": True,
        "login_url": "https://platform.example.com/login",
    })
    assert probe is not None
    assert probe.login_ceremony is True
    assert probe.login_url == "https://platform.example.com/login"


def test_url_host_lowercases_and_strips_port_path():
    """The host comparison is case-insensitive; scheme and path don't matter.
    A `URL:Example.COM` vs `url:example.com` mismatch would silently split
    validation, so the helper lower-cases both sides."""
    assert _url_host("https://Example.COM/Health") == "example.com"
    assert _url_host("http://example.com:8443/foo") == "example.com"
    assert _url_host("not-a-url") == ""
    assert _url_host("") == ""
    assert _url_host(None) == ""


# ---------------------------------------------------------------------------
# Allowlist + cookie host matching (the central safety property).
# ---------------------------------------------------------------------------
def test_cookie_allowlist_returns_just_the_probe_host():
    """A single bare-host entry. ``cookie_host_allowed`` strips the
    leading dot on both sides, so a bare entry already covers the
    bare form, the dot-prefixed form Chromium sets for site+subdomain
    scope, and every deeper subdomain via suffix match."""
    out = cookie_allowlist("https://platform.example.com/health")
    assert out == {"platform.example.com"}


def test_cookie_allowlist_is_empty_for_unparseable_url():
    assert cookie_allowlist("") == set()
    assert cookie_allowlist("not-a-url") == set()


def test_cookie_host_allowed_accepts_probe_host_and_subdomains():
    """The probe host and any subdomain of it are allowed; everything else
    is rejected. The dot-stripped comparison covers both the bare and
    the dot-prefixed cookie-domain form on either side."""
    allow = {"console.example.com"}
    # Probe host itself (bare and dot-prefixed cookie form).
    assert cookie_host_allowed("console.example.com", allow)
    assert cookie_host_allowed(".console.example.com", allow)
    # Deeper subdomain (bare and dot-prefixed cookie form).
    assert cookie_host_allowed("app.console.example.com", allow)
    assert cookie_host_allowed(".app.console.example.com", allow)


def test_cookie_host_allowed_accepts_apex_when_probe_host_is_apex():
    """When the probe host IS the bare apex, an apex cookie domain is
    accepted. The earlier review noted the docstring implied otherwise;
    this test ties the docstring and the behaviour together so the
    example in the docstring never drifts from the code."""
    allow = {"example.com"}
    assert cookie_host_allowed("example.com", allow)
    assert cookie_host_allowed(".example.com", allow)
    # Deeper subdomains of the apex are still accepted via suffix match.
    assert cookie_host_allowed("app.example.com", allow)


def test_cookie_host_allowed_rejects_unrelated_hosts():
    """A cookie on a different apex or a sibling subdomain is rejected —
    the abort condition the task spec calls out by name. Critically,
    a probe host of `console.example.com` does NOT accept the bare
    `example.com` apex: an operator who set the probe on `console.`
    explicitly opted into polling `console.`, not the apex."""
    allow = {"console.example.com"}
    assert not cookie_host_allowed("attacker.example.org", allow)
    assert not cookie_host_allowed("other.com", allow)
    # The bare apex of `console.example.com` is NOT on the allowlist.
    assert not cookie_host_allowed("example.com", allow)
    assert not cookie_host_allowed(".example.com", allow)
    # A lookalike (suffix match without a dot) is NOT a subdomain.
    assert not cookie_host_allowed("evilconsole.example.com", allow)
    # Empty / missing domain is always rejected.
    assert not cookie_host_allowed("", allow)


def test_cookie_host_allowed_is_case_insensitive():
    allow = {"Example.COM"}
    assert cookie_host_allowed("sub.EXAMPLE.com", allow)
    assert cookie_host_allowed(".example.com", allow)


# ---------------------------------------------------------------------------
# Cookie header assembly: the Network-tab shape.
# ---------------------------------------------------------------------------
def test_assemble_cookie_header_emits_name_value_pairs_in_order():
    """The Network tab shows `Cookie: a=1; b=2; c=3`. The ceremony must
    produce exactly that shape so the portal sees the same header Chromium
    would send."""
    cookies = [
        {"name": "session", "value": "abc123"},
        {"name": "tracking", "value": "xyz"},
    ]
    assert assemble_cookie_header(cookies) == "session=abc123; tracking=xyz"


def test_assemble_cookie_header_skips_cookies_without_a_name():
    """A cookie with an empty name (e.g. a deleted one) is silently dropped
    by the Network tab too. The ceremony must not emit `=value; ...`."""
    cookies = [
        {"name": "", "value": "ignored"},
        {"name": "kept", "value": "1"},
    ]
    assert assemble_cookie_header(cookies) == "kept=1"


def test_assemble_cookie_header_handles_empty_input():
    assert assemble_cookie_header([]) == ""


def test_assemble_cookie_header_round_trips_a_real_minimax_shape():
    """The two-cookie shape the MiniMax console uses in practice. The
    Network tab's Cookie header is exactly this string, in this order, so
    the ceremony's output must match it byte-for-byte."""
    cookies = [
        {"name": "session", "value": "eyJhbGciOiJIUzI1NiJ9.payload"},
        {"name": "acw_tc", "value": "0a1b2c3d4e5f"},
    ]
    expected = "session=eyJhbGciOiJIUzI1NiJ9.payload; acw_tc=0a1b2c3d4e5f"
    assert assemble_cookie_header(cookies) == expected


# ---------------------------------------------------------------------------
# Cookie fingerprint: never carries the value.
# ---------------------------------------------------------------------------
def test_cookie_fingerprint_never_contains_the_cookie_value():
    """The fingerprint is the operator's only visible proof the cookie was
    stored. It must not leak the value — a look at the fingerprint would
    then be enough to forge the session."""
    cookie = "session=eyJhbGciOiJIUzI1NiJ9.payload"
    fp = cookie_fingerprint(cookie)
    assert "session=" not in fp
    assert "payload" not in fp
    assert "eyJhbGciOiJIUzI1NiJ9" not in fp
    # And it carries enough to tell two cookies apart.
    assert fp.startswith(f"{len(cookie)} chars, #"), fp


def test_cookie_fingerprint_distinguishes_two_cookies():
    a = cookie_fingerprint("session=abc")
    b = cookie_fingerprint("session=abd")
    assert a != b


# ---------------------------------------------------------------------------
# "unexpected cookie host" abort: the central safety property.
# ---------------------------------------------------------------------------
def test_filter_cookies_or_abort_accepts_a_cookies_within_allowlist():
    """Happy path: every cookie's domain is the probe host or a subdomain
    of it. The function returns the cookies unchanged (in original order)."""
    cookies = [
        {"name": "a", "value": "1", "domain": "example.com"},
        {"name": "b", "value": "2", "domain": ".example.com"},
        {"name": "c", "value": "3", "domain": "sub.example.com"},
    ]
    allow = {"example.com"}
    kept = filter_cookies_or_abort(cookies, allow)
    assert [c["name"] for c in kept] == ["a", "b", "c"]


def test_filter_cookies_or_abort_raises_with_offender_domain_on_mismatch():
    """The literal `unexpected cookie host: <domain>` shape the task spec
    calls for. The allowlist is included so the operator can see what was
    expected without scrolling back through config."""
    cookies = [
        {"name": "good", "value": "1", "domain": "example.com"},
        {"name": "evil", "value": "2", "domain": "attacker.org"},
    ]
    allow = {"example.com"}
    try:
        filter_cookies_or_abort(cookies, allow)
    except RuntimeError as exc:
        msg = str(exc)
        assert msg.startswith("unexpected cookie host:"), msg
        assert "'attacker.org'" in msg, msg
        assert "'example.com'" in msg, msg
        return
    raise AssertionError("filter_cookies_or_abort accepted an off-allowlist cookie")


def test_filter_cookies_or_abort_aborts_on_first_offender():
    """The first off-allowlist cookie aborts; later offenders are not named
    in the message but they were never reached."""
    cookies = [
        {"name": "first_evil", "value": "1", "domain": "attacker.org"},
        {"name": "second_evil", "value": "2", "domain": "other.com"},
    ]
    try:
        filter_cookies_or_abort(cookies, {"example.com"})
    except RuntimeError as exc:
        assert "attacker.org" in str(exc), exc
        assert "other.com" not in str(exc), exc
        return
    raise AssertionError("filter_cookies_or_abort did not abort")


def test_filter_cookies_or_abort_returns_empty_list_when_no_cookies():
    assert filter_cookies_or_abort([], {"example.com"}) == []


# ---------------------------------------------------------------------------
# --dry-run and the CLI refusal gate.
# ---------------------------------------------------------------------------
def test_dry_run_prints_every_ceremony_relevant_field_for_an_opted_in_plan():
    """`--dry-run` must surface the plan, the probe URL, the login URL, the
    cookie allowlist, and the resolved portal URL origin + timeout so an
    operator can read it once and know exactly what would happen on a real
    run. The resolved values (not argparse defaults) are what the live path
    uses; the dry-run must show the same numbers."""
    captured = io.StringIO()
    saved = sys.stdout
    sys.stdout = captured
    try:
        probe = Probe(url="https://platform.example.com/health",
                      login_ceremony=True,
                      login_url="https://platform.example.com/console/login")
        plan = models.Plan(key="demo", label="demo",
                           models={}, probe=probe)
        print_dry_run(plan, timeout_seconds=42.0,
                      portal_url="http://ceremony.local:5555")
    finally:
        sys.stdout = saved
    out = captured.getvalue()
    assert "plan:             demo" in out, out
    assert "probe.url:        https://platform.example.com/health" in out, out
    assert "login_url:        https://platform.example.com/console/login" in out, out
    assert "platform.example.com" in out, out
    assert "login_ceremony:   True" in out, out
    # Resolved values, not argparse defaults.
    assert "timeout_seconds:  42.0" in out, out
    assert "portal_url:       http://ceremony.local:5555" in out, out


def test_dry_run_falls_back_to_default_portal_when_no_override():
    """When no `--portal-url` is passed and the env var is unset, the
    dry-run echoes the same default the live path uses (loopback portal).
    This is the path an operator gets when they run the CLI bare."""
    saved_env = os.environ.pop("SWITCHYARD_PORTAL_URL", None)
    try:
        captured = io.StringIO()
        saved = sys.stdout
        sys.stdout = captured
        try:
            probe = Probe(url="https://platform.example.com/health",
                          login_ceremony=True,
                          login_url="https://platform.example.com/console/login")
            plan = models.Plan(key="demo", label="demo",
                               models={}, probe=probe)
            print_dry_run(plan)
        finally:
            sys.stdout = saved
        out = captured.getvalue()
        assert "portal_url:       http://localhost:4001" in out, out
        assert "timeout_seconds:  180.0" in out, out
    finally:
        if saved_env is not None:
            os.environ["SWITCHYARD_PORTAL_URL"] = saved_env


def test_cli_dry_run_reflects_timeout_and_portal_url_overrides():
    """`--timeout-seconds 30 --portal-url ...` passed on the command line
    must show up in the dry-run output. Otherwise the operator gets the
    argparse default and reasonably assumes their override was dropped."""
    import tempfile
    import yaml

    from plans_path import plans_path

    with open(plans_path()) as fh:
        raw = yaml.safe_load(fh)
    raw["plans"]["minimax-ultra"]["probe"]["login_ceremony"] = True
    raw["plans"]["minimax-ultra"]["probe"]["login_url"] = (
        "https://platform.minimax.io/console/usage")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(raw, tmp)
        tmp_path = tmp.name
    try:
        captured = io.StringIO()
        saved = sys.stdout
        sys.stdout = captured
        try:
            rc = ceremony_main(["--plans", tmp_path,
                                "minimax-ultra", "--dry-run",
                                "--timeout-seconds", "30",
                                "--portal-url", "http://example.test:9999"])
        finally:
            sys.stdout = saved
        assert rc == 0, rc
        out = captured.getvalue()
        assert "timeout_seconds:  30.0" in out, out
        assert "portal_url:       http://example.test:9999" in out, out
    finally:
        os.unlink(tmp_path)


def test_cli_refuses_plan_without_login_ceremony_opt_in():
    """`python3 -m switchyard.ceremony <plan>` must refuse plans whose
    `probe.login_ceremony` is false (or unset). That gate is the single
    most important guarantee this module ships — without it the ceremony
    would silently fire for any cookie probe in the file."""
    from plans_path import plans_path
    rc = ceremony_main(["--plans", plans_path(), "minimax-ultra", "--dry-run"])
    assert rc == 2, rc


def test_cli_refuses_unknown_plan():
    """A typo in the plan name is an operator error, not a launch."""
    from plans_path import plans_path
    rc = ceremony_main(["--plans", plans_path(),
                        "not-a-plan", "--dry-run"])
    assert rc == 2, rc


def test_cli_dry_run_succeeds_for_an_opted_in_plan_from_yaml():
    """End-to-end smoke for the dry-run path: load the real example config,
    write a temporary plans file that flips one plan's `login_ceremony` on,
    and verify the CLI prints the plan and exits 0."""
    import tempfile
    import yaml

    from plans_path import plans_path

    with open(plans_path()) as fh:
        raw = yaml.safe_load(fh)
    raw["plans"]["minimax-ultra"]["probe"]["login_ceremony"] = True
    raw["plans"]["minimax-ultra"]["probe"]["login_url"] = (
        "https://platform.minimax.io/console/usage")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(raw, tmp)
        tmp_path = tmp.name
    try:
        captured = io.StringIO()
        saved = sys.stdout
        sys.stdout = captured
        try:
            rc = ceremony_main(["--plans", tmp_path,
                                "minimax-ultra", "--dry-run"])
        finally:
            sys.stdout = saved
        assert rc == 0, rc
        out = captured.getvalue()
        assert "plan:             minimax-ultra" in out, out
        assert "login_url:        https://platform.minimax.io/console/usage" in out, out
        assert "platform.minimax.io" in out, out
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Parser shape: argv parses without surprises.
# ---------------------------------------------------------------------------
def test_parser_accepts_plan_and_dry_run():
    args = build_parser().parse_args(["myplan", "--dry-run"])
    assert args.plan == "myplan"
    assert args.dry_run is True


def test_parser_default_timeout_is_documented_in_help():
    """The default timeout is documented in --help so the operator doesn't
    have to read source to find out how long the ceremony will wait."""
    parser = build_parser()
    help_text = parser.format_help()
    assert "180" in help_text, help_text


# ---------------------------------------------------------------------------
# Playwright happy path: only meaningful when Playwright is on the test
# machine. Under the conftest socket guard the real browser cannot reach
# anything, so the test is skipped (not failed) when the import is absent.
# ---------------------------------------------------------------------------
def test_playwright_happy_path_is_skipped_when_playwright_is_absent():
    try:
        import playwright  # noqa: F401
    except ImportError:
        # Expected in the offline CI environment. The skip is itself the
        # contract: the ceremony code path is exercised by reading the
        # install hint at runtime; the dedicated test of the happy path
        # needs a real browser, which the conftest socket guard forbids.
        print("  skip  test_playwright_happy_path_is_skipped_when_playwright_is_absent")
        return
    raise AssertionError(
        "Playwright is installed on this machine; the dedicated happy-path "
        "test should be enabled (the current skip is meant for offline CI). "
        "See switchyard/ceremony.py:_run_with_playwright for the live "
        "entry point — wiring a stub sync_playwright here would let the "
        "allowlist + portal POST run end-to-end without a real browser."
    )


def test_run_raises_install_hint_error_when_playwright_is_absent():
    """The install-hint error is the user-facing message when the operator
    runs the ceremony on a host that hasn't installed Playwright. The
    message must name the install command — `pip install playwright &&
    playwright install chromium` — because the first install does not
    include the browser binary."""
    probe = Probe(url="https://platform.example.com/health",
                  login_ceremony=True,
                  login_url="https://platform.example.com/login")
    plan = models.Plan(key="demo", label="demo",
                       models={}, probe=probe)
    try:
        ceremony.run(plan)
    except RuntimeError as exc:
        msg = str(exc)
        assert "pip install playwright" in msg, msg
        assert "playwright install chromium" in msg, msg
        return
    raise AssertionError("ceremony.run did not raise on missing Playwright")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
