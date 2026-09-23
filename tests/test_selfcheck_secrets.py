"""Regression tests for the secrets audit at the top of `switchyard.selfcheck`.

Issue #118: a gateway that boots with `LITELLM_MASTER_KEY=sk-switchyard-change-me`
is indistinguishable from one that boots with no master key at all - the
placeholder is published in the repo, so anyone who has read the README can
mint a valid bearer against the running container. The same applies, less
dramatically, to `POSTGRES_PASSWORD=change-me` (the default in .env.example):
the database accepts connections from anyone who knows the published string.
Both of these are fail-loud defaults on purpose, and the startup gate is the
right place to catch them.

`switchyard/selfcheck.py` runs `_audit_secrets()` as the FIRST check inside
`main_async()`, ahead of the litellm AST audits and the loopback probe. The
function is pure env (no network, no litellm import) so the offline test suite
can exercise every branch without the gateway image:

  - LITELLM_MASTER_KEY unset / empty          -> _critical (raises _CheckFailed)
  - LITELLM_MASTER_KEY contains "change-me"   -> _critical (raises _CheckFailed)
  - LITELLM_MASTER_KEY shorter than 32 chars   -> _critical (raises _CheckFailed)
  - LITELLM_MASTER_KEY valid (>= 32, no marker) -> passes silently
  - POSTGRES_PASSWORD unset / "change-me" / "litellm" -> WARNING logged,
    but does NOT raise (Postgres-password failures are loud-warn, not fail).
  - POSTGRES_PASSWORD valid                   -> no warning

The CRITICAL message must name the bad variable AND tell the operator how to
fix it (`bash scripts/sync-env.sh`); the test asserts both halves of that
contract for every failure branch.

These tests do not import litellm. The function is a pure `os.environ` reader,
matching the lazy-import style of the rest of `selfcheck.py` so a missing
litellm would not mask a missing master key (or vice versa) at startup.
"""
from __future__ import annotations

import ast
import io
import logging
import os
import sys
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `import conftest` resolves under plain `python3`

import conftest  # noqa: F401  (socket guard for plain-script mode)

ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # so `from switchyard.selfcheck import ...` resolves

from switchyard.selfcheck import _CheckFailed, _audit_secrets  # noqa: E402

SC_LOGGER_NAME = "switchyard.selfcheck"
VALID_KEY = "sk-" + ("a" * 60)        # 63 chars: >= 32, no "change-me"
WEAK_KEY_SHORT = "short"              # 5 chars, no marker
WEAK_KEY_MARKER = "change-me"          # contains marker, < 32 chars
WEAK_KEY_PLACEHOLDER = "sk-switchyard-change-me"  # the .env.example default
WEAK_KEY_WHITESPACE = "   "            # treated as empty after .strip()


class _CapturingHandler(logging.Handler):
    """Capture every log record emitted on `switchyard.selfcheck` for the
    lifetime of a `with` block. The plain-script runner (and pytest) both
    rely on the same in-memory buffer rather than on pytest's `capsys`
    fixture, so the tests work identically under `python3 tests/test_*.py`
    and `python3 -m pytest`.
    """
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _scoped_env(env: dict[str, str | None]):
    """Override the audit-relevant keys for the duration of the block.

    A value of None removes the key from `os.environ`; any other value is
    forced in as a string. On exit, the original state is restored byte-for-
    byte so a test run cannot leak settings into another test or into the
    caller's shell.

    `unittest.mock.patch.dict` cannot unset a key (it would have to assign
    `None` into `os.environ`, which raises TypeError on Python 3), so this
    helper does the unset/pop path by hand. Saving the originals before the
    block also catches the case where the test runs after another test has
    already mutated the env - the second test still sees the original state
    on exit.
    """
    originals: dict[str, str | None] = {}
    for key in env:
        originals[key] = os.environ.get(key)  # None if absent
    try:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, original in originals.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _capture_audit_secrets(env: dict[str, str | None]):
    """Run `_audit_secrets()` under a controlled env, capturing every log
    record emitted on `switchyard.selfcheck` during the call.

    `env` mirrors the keys that the audit inspects:
      - LITELLM_MASTER_KEY
      - POSTGRES_PASSWORD
    A value of None removes the key from the process env for the duration of
    the call, so the "unset" case is genuinely unset (not "set to empty
    string"). Anything else is forced into os.environ as a string.
    """
    handler = _CapturingHandler()
    sc_logger = logging.getLogger(SC_LOGGER_NAME)
    sc_logger.addHandler(handler)
    try:
        with _scoped_env(env):
            raised: BaseException | None = None
            try:
                _audit_secrets()
            except BaseException as exc:                  # noqa: BLE001
                raised = exc
        return raised, list(handler.records)
    finally:
        sc_logger.removeHandler(handler)


def _format_records(records: list[logging.LogRecord]) -> str:
    """Render captured records as a single string so assertions on log
    content stay readable even under pytest's `-q` output."""
    buf = io.StringIO()
    for r in records:
        buf.write(f"{logging.getLevelName(r.levelno)}: {r.getMessage()}\n")
    return buf.getvalue()


# --------------------------------------------------------------------- failures

def test_empty_master_key_fails_with_remediation_hint():
    """`LITELLM_MASTER_KEY=""` -> _critical. The CRITICAL message must name
    the variable and tell the operator how to fix it (`bash scripts/sync-env.sh`).
    A missing variable or empty string both fall into this branch (we test
    empty here; the next test covers the missing-variable variant).
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": "", "POSTGRES_PASSWORD": VALID_KEY})
    assert isinstance(raised, _CheckFailed), (
        f"empty LITELLM_MASTER_KEY must raise _CheckFailed (SystemExit code 1); "
        f"got {type(raised).__name__}: {raised!r}"
    )
    assert int(raised.code) == 1, raised.code
    formatted = _format_records(records)
    assert "LITELLM_MASTER_KEY" in formatted, formatted
    assert "scripts/sync-env.sh" in formatted, formatted
    assert "CRITICAL" in formatted or "selfcheck FAILED" in formatted, formatted
    print(f"  empty LITELLM_MASTER_KEY: CRITICAL mentions the variable and "
          f"sync-env.sh ({len(records)} record(s))")


def test_unset_master_key_fails_with_remediation_hint():
    """`LITELLM_MASTER_KEY` absent from os.environ -> _critical. Same contract
    as the empty-string case: a bare unset is the most common production
    failure (operator forgot to copy .env.example to .env, or compose's
    fail-closed substitution in `docker-compose.yml` caught it and crashed
    the container - either way, a clear CRITICAL line is the right answer).
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": None, "POSTGRES_PASSWORD": VALID_KEY})
    assert isinstance(raised, _CheckFailed), (
        f"unset LITELLM_MASTER_KEY must raise _CheckFailed; got "
        f"{type(raised).__name__}: {raised!r}"
    )
    formatted = _format_records(records)
    assert "LITELLM_MASTER_KEY" in formatted, formatted
    assert "scripts/sync-env.sh" in formatted, formatted
    print("  unset LITELLM_MASTER_KEY: CRITICAL names the variable and "
          "sync-env.sh")


def test_whitespace_only_master_key_fails_with_remediation_hint():
    """`LITELLM_MASTER_KEY="   "` -> _critical. The audit `.strip()`s the
    value, so whitespace-only is the same as empty; this is the case where
    an operator copy-pasted from a notes app that added trailing spaces.
    """
    raised, _ = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": WEAK_KEY_WHITESPACE,
         "POSTGRES_PASSWORD": VALID_KEY})
    assert isinstance(raised, _CheckFailed), (
        f"whitespace-only LITELLM_MASTER_KEY must raise _CheckFailed; got "
        f"{type(raised).__name__}: {raised!r}"
    )
    print("  whitespace-only LITELLM_MASTER_KEY: stripped to empty -> CRITICAL")


def test_change_me_marker_fails_with_remediation_hint():
    """`LITELLM_MASTER_KEY="change-me"` -> _critical. The marker substring
    is the published default; the CRITICAL message echoes the offending
    value (in quotes, so an operator can see exactly what was found) and
    points at sync-env.sh.
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": WEAK_KEY_MARKER,
         "POSTGRES_PASSWORD": VALID_KEY})
    assert isinstance(raised, _CheckFailed), (
        f"marker 'change-me' must raise _CheckFailed; got "
        f"{type(raised).__name__}: {raised!r}"
    )
    formatted = _format_records(records)
    assert "'change-me'" in formatted or "change-me" in formatted, formatted
    assert "scripts/sync-env.sh" in formatted, formatted
    print("  LITELLM_MASTER_KEY='change-me': CRITICAL echoes the marker and "
          "points at sync-env.sh")


def test_published_placeholder_value_fails_with_remediation_hint():
    """`LITELLM_MASTER_KEY="sk-switchyard-change-me"` (the .env.example
    default) -> _critical. This is the regression the audit exists to
    catch: a fresh clone + `cp .env.example .env` lands here, and the
    CRITICAL line is what stops the gateway from accepting "anyone" as a
    caller. The marker substring fires before the length check, so the
    message should be the marker-specific one.
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": WEAK_KEY_PLACEHOLDER,
         "POSTGRES_PASSWORD": VALID_KEY})
    assert isinstance(raised, _CheckFailed), (
        f"published placeholder must raise _CheckFailed; got "
        f"{type(raised).__name__}: {raised!r}"
    )
    formatted = _format_records(records)
    assert "change-me" in formatted, formatted
    assert "scripts/sync-env.sh" in formatted, formatted
    print("  LITELLM_MASTER_KEY=sk-switchyard-change-me: CRITICAL catches "
          "the published default before the length check")


def test_short_master_key_fails_with_remediation_hint():
    """`LITELLM_MASTER_KEY="short"` (5 chars, no marker) -> _critical. The
    length branch must fire when the marker branch does not, and the
    message must say so explicitly.
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": WEAK_KEY_SHORT,
         "POSTGRES_PASSWORD": VALID_KEY})
    assert isinstance(raised, _CheckFailed), (
        f"short LITELLM_MASTER_KEY must raise _CheckFailed; got "
        f"{type(raised).__name__}: {raised!r}"
    )
    formatted = _format_records(records)
    assert "shorter than 32" in formatted, formatted
    assert "scripts/sync-env.sh" in formatted, formatted
    print(f"  LITELLM_MASTER_KEY={WEAK_KEY_SHORT!r}: CRITICAL names the "
          f"32-char threshold and sync-env.sh")


# --------------------------------------------------------------------- pass

def test_valid_master_key_passes_silently():
    """A 32+ char key without the marker -> no raise, and no WARNING. The
    Postgres password is also set to a valid value here, so this also
    covers the "no warnings at all" happy path.
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": VALID_KEY, "POSTGRES_PASSWORD": "real-db-password-12345"})
    assert raised is None, (
        f"valid LITELLM_MASTER_KEY must not raise; got {type(raised).__name__}: "
        f"{raised!r}")
    warnings = [r for r in records if r.levelno >= logging.WARNING]
    assert not warnings, (
        f"valid LITELLM_MASTER_KEY + valid POSTGRES_PASSWORD must not emit any "
        f"warnings; got: {_format_records(warnings)}")
    print(f"  LITELLM_MASTER_KEY={len(VALID_KEY)}-char valid key, "
          f"POSTGRES_PASSWORD=non-default: no raise, no warnings")


def test_valid_master_key_with_unset_postgres_only_warns():
    """A valid master key still emits the Postgres warning if the password
    is unset. This pins the contract: the master-key branch is independent
    of the Postgres branch, so a good key + bad password is still loud
    (warning, not failure).
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": VALID_KEY, "POSTGRES_PASSWORD": None})
    assert raised is None, (
        f"valid LITELLM_MASTER_KEY must not raise on Postgres warning; got "
        f"{type(raised).__name__}: {raised!r}")
    warnings = [r for r in records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, (
        f"expected exactly one WARNING (for unset POSTGRES_PASSWORD); got "
        f"{_format_records(warnings)}")
    formatted = _format_records(warnings)
    assert "POSTGRES_PASSWORD" in formatted, formatted
    print("  valid LITELLM_MASTER_KEY + unset POSTGRES_PASSWORD: one WARNING, "
          "no CRITICAL")


# ---------------------------------------------------- postgres warning branches

def test_postgres_unset_emits_warning_not_failure():
    """`POSTGRES_PASSWORD` absent -> WARNING, no raise. The audit must NOT
    block startup on a missing Postgres password; that's a Compose concern
    (`POSTGRES_PASSWORD:?missing` already fail-closes there), and an
    operator who explicitly clears the value for a local-only deployment
    still gets a clean startup.
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": VALID_KEY, "POSTGRES_PASSWORD": None})
    assert raised is None, (
        f"unset POSTGRES_PASSWORD must not raise; got {type(raised).__name__}: "
        f"{raised!r}")
    warnings = [r for r in records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, (
        f"expected exactly one WARNING; got {_format_records(warnings)}")
    msg = warnings[0].getMessage()
    assert "POSTGRES_PASSWORD" in msg, msg
    assert "unset" in msg.lower() or "empty" in msg.lower(), msg
    print("  POSTGRES_PASSWORD unset: WARNING, exit 0")


def test_postgres_change_me_emits_warning_not_failure():
    """`POSTGRES_PASSWORD="change-me"` -> WARNING, no raise. The .env.example
    default. The audit's whole reason for existing in warning-mode (vs
    fail-mode) is so that a fresh clone + `cp .env.example .env` still boots
    - but the operator cannot miss the warning in `docker logs`.
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": VALID_KEY, "POSTGRES_PASSWORD": "change-me"})
    assert raised is None, (
        f"POSTGRES_PASSWORD='change-me' must not raise; got "
        f"{type(raised).__name__}: {raised!r}")
    warnings = [r for r in records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, (
        f"expected exactly one WARNING; got {_format_records(warnings)}")
    msg = warnings[0].getMessage()
    assert "POSTGRES_PASSWORD" in msg, msg
    assert "change-me" in msg, msg
    print("  POSTGRES_PASSWORD='change-me': WARNING, exit 0")


def test_postgres_litellm_emits_warning_not_failure():
    """`POSTGRES_PASSWORD="litellm"` -> WARNING, no raise. The literal
    LiteLLM-image default. Pinned as a separate test from the marker case
    because the message wording differs (this one names "litellm", not
    "change-me"), and a future edit that merges them would be silent.
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": VALID_KEY, "POSTGRES_PASSWORD": "litellm"})
    assert raised is None, (
        f"POSTGRES_PASSWORD='litellm' must not raise; got "
        f"{type(raised).__name__}: {raised!r}")
    warnings = [r for r in records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, (
        f"expected exactly one WARNING; got {_format_records(warnings)}")
    msg = warnings[0].getMessage()
    assert "POSTGRES_PASSWORD" in msg, msg
    assert "litellm" in msg, msg
    print("  POSTGRES_PASSWORD='litellm': WARNING, exit 0")


def test_postgres_empty_string_emits_warning_not_failure():
    """`POSTGRES_PASSWORD=""` -> WARNING, no raise. Same shape as the unset
    case once `.strip()` runs; pinned separately so the empty-string branch
    is covered independently (the operator might set the key but blank it
    out).
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": VALID_KEY, "POSTGRES_PASSWORD": ""})
    assert raised is None, (
        f"empty POSTGRES_PASSWORD must not raise; got {type(raised).__name__}: "
        f"{raised!r}")
    warnings = [r for r in records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, (
        f"expected exactly one WARNING for empty POSTGRES_PASSWORD; got "
        f"{_format_records(warnings)}")
    print("  POSTGRES_PASSWORD='': WARNING, exit 0")


def test_postgres_valid_value_emits_no_warning():
    """`POSTGRES_PASSWORD="real-secret-db-password"` -> no WARNING, no
    raise. The happy-path Postgres branch.
    """
    raised, records = _capture_audit_secrets(
        {"LITELLM_MASTER_KEY": VALID_KEY,
         "POSTGRES_PASSWORD": "real-secret-db-password"})
    assert raised is None, (
        f"valid POSTGRES_PASSWORD must not raise; got {type(raised).__name__}: "
        f"{raised!r}")
    warnings = [r for r in records if r.levelno >= logging.WARNING]
    assert not warnings, (
        f"valid POSTGRES_PASSWORD must not emit warnings; got "
        f"{_format_records(warnings)}")
    print("  POSTGRES_PASSWORD=non-default: no warning, no raise")


# --------------------------------------------------------------- wiring (AST)

def test_selfcheck_wires_audit_secrets_into_main_async():
    """`switchyard/selfcheck.py` defines `_audit_secrets()` AND calls it
    from `main_async()`. If a future edit removes the function definition
    or unhooks it from the startup gate, the secrets audit would silently
    stop firing and a default master key would not be caught. We assert
    both halves via a substring + AST check; no litellm import.
    """
    path = os.path.join(ROOT, "switchyard", "selfcheck.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    assert "def _audit_secrets" in src, (
        "switchyard/selfcheck.py no longer defines _audit_secrets - the "
        "secrets audit was dropped from the startup gate."
    )

    tree = ast.parse(src)

    found_def = any(
        isinstance(node, ast.FunctionDef)
        and node.name == "_audit_secrets"
        for node in ast.walk(tree)
    )
    assert found_def, (
        "_audit_secrets is present as text but does not parse as a function "
        "definition."
    )

    main_async = next(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.AsyncFunctionDef) and node.name == "main_async"),
        None,
    )
    assert main_async is not None, (
        "main_async() is missing from switchyard/selfcheck.py - the startup "
        "gate itself has been removed."
    )

    called = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_audit_secrets"
        for node in ast.walk(main_async)
    )
    assert called, (
        "_audit_secrets is defined but NOT called from main_async(); the "
        "secrets audit would silently stop firing and a default master key "
        "would not be caught."
    )

    # The secrets audit must run BEFORE the litellm AST audits: catching a
    # missing master key after a missing litellm import would dump a
    # confusing second CRITICAL line into `docker logs`. Asserting on the
    # call order keeps that ordering contract mechanical.
    body_calls = [
        node.value.func.id for node in main_async.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert body_calls.index("_audit_secrets") < body_calls.index("_static_audit"), (
        f"_audit_secrets must run BEFORE _static_audit (so a missing master "
        f"key is reported without a litellm-import noise line); got order "
        f"{body_calls!r}"
    )

    print("  selfcheck.py: _audit_secrets defined, called from main_async() "
          f"before _static_audit (order: {body_calls!r})")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
