"""Tests for the apply.sh drain flow's two surgical changes:

  (a) `scripts/health_idle.py` — read by the drain loop to gate the
      post-drain grace window. Four health shapes against loopback
      stubs, plus connection-refused, classify as `idle`, `busy`, or
      `unknown`; always exits 0 so bash only reads the printed token.
      Conftest allows loopback (see tests/conftest.py / CLAUDE.md), so
      a stub http.server is the right vehicle — the test exercises the
      real protocol against a fake peer.

  (b) `scripts/apply.sh` text — three surgical cuts:

      1. the drain-helper failure branch surfaces the helper's last
         ~8 lines AND prints the `--build` hint when the captured
         output carries `No module named switchyard.drain` (the
         gateway image predates WORKSTREAM 1's `switchyard/drain.py`,
         and the hint is the only thing that tells the operator which
         knob to turn).

      2. the misleading
            `lease migration skipped (drain helper unavailable or no leases)`
         message is gone — it conflated two unrelated failure modes
         (no leases is a no-op; the gateway image predating drain.py
         is fixable with --build), and the new branch names each one.

      3. the grace block consumes the health probe:
            `health_idle.py` is exec'd into the sidecar via
            `docker compose exec -T "$svc" python3 - < scripts/health_idle.py`
         and `idle` skips the sleep while `busy` and `unknown` keep
         the full `drain_grace_secs` window. This is the behavior the
         bug report's 120s every-drain cost came from, so the text
         check is on its presence here, not on a real container run
         (live verification happens at the main checkout per CLAUDE.md).

Plus a `bash -n scripts/apply.sh` syntax gate so a stray heredoc /
unbalanced quote fails before any test runs.
"""
from __future__ import annotations

import http.server
import json
import os
import socket
import subprocess
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------- health_idle.py


class _ShapeStub(http.server.BaseHTTPRequestHandler):
    """Serve the JSON doc passed in `payload` on every GET /health.

    A single instance per test, switching its `payload` between calls
    via a closure-factory (`_starter`). The factory builds the server,
    wires a payload, and yields the base URL; the test then drives
    health_idle.py against it and shuts it down.
    """

    def do_GET(self):
        body = json.dumps(self.server.payload).encode()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def _start_stub(payload):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ShapeStub)
    srv.payload = payload  # type: ignore[attr-defined]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _run_health_idle(port):
    """Invoke `scripts/health_idle.py` against 127.0.0.1:<port> in a
    subprocess, mirroring the apply.sh wrapping exactly:

        docker compose exec -T "$svc" python3 - < scripts/health_idle.py

    The container-less equivalent sets SIDECAR_PORT and runs the script
    directly, with `python3 -` substituted by `python3 <path>` (Python
    treats both as "read the script from stdin", which is what the
    in-container `python3 -` does). Exit code is always 0 by contract;
    the printed token is what carries the answer.
    """
    env = {**os.environ, "SIDECAR_PORT": str(port)}
    res = subprocess.run(
        ["python3", os.path.join(ROOT, "scripts", "health_idle.py")],
        env=env, capture_output=True, text=True,
    )
    return res.returncode, res.stdout.strip(), res.stderr


def test_health_idle_mcp_bridge_idle_shape():
    """mcp_bridge idle: every counter zero, no parked sessions.

    The mcp_bridge /health doc carries `in_flight`, `sessions`, and
    `awaiting_followup`. All three summing to zero is the only way the
    bridge is genuinely idle (a parked session still lives in SESSIONS
    with `awaiting_followup=True` — see mcp_bridge/server.py
    `park_session` — so `sessions > 0` alone is enough to keep the
    sidecar out of the `idle` bucket).
    """
    srv, port = _start_stub(
        {"in_flight": 0, "sessions": 0, "awaiting_followup": 0})
    try:
        rc, out, err = _run_health_idle(port)
    finally:
        srv.shutdown()
    assert rc == 0, (rc, err)
    assert out == "idle", (out, err)


def test_health_idle_mcp_bridge_parked_shape_is_busy():
    """mcp_bridge parked: in_flight=0 but a parked call is alive.

    `park_session` releases the slot (parked sessions don't hold a gate
    slot — see the comment block at `park_session`), so `in_flight`
    alone would miss this case. The `in_flight + sessions +
    awaiting_followup` sum is what catches it.
    """
    srv, port = _start_stub(
        {"in_flight": 0, "sessions": 2, "awaiting_followup": 1})
    try:
        rc, out, err = _run_health_idle(port)
    finally:
        srv.shutdown()
    assert rc == 0, (rc, err)
    assert out == "busy", (out, err)


def test_health_idle_cli_bridge_idle_shape():
    """cli_bridge idle: the cli_bridge doc has only `in_flight`.

    `sessions` and `awaiting_followup` are absent — they default to 0
    in the classifier (the `or 0` shape), and the sum is in_flight=0.
    """
    srv, port = _start_stub({"in_flight": 0})
    try:
        rc, out, err = _run_health_idle(port)
    finally:
        srv.shutdown()
    assert rc == 0, (rc, err)
    assert out == "idle", (out, err)


def test_health_idle_cli_bridge_busy_shape():
    """cli_bridge busy: in_flight>0 is enough on its own.

    cli_bridge doesn't park in the mcp_bridge sense, so `in_flight > 0`
    is the only signal a request is genuinely in flight. The token-proxy
    busy case below covers the "in_flight absent" shape separately.
    """
    srv, port = _start_stub({"in_flight": 3})
    try:
        rc, out, err = _run_health_idle(port)
    finally:
        srv.shutdown()
    assert rc == 0, (rc, err)
    assert out == "busy", (out, err)


def test_health_idle_token_proxy_shape_is_unknown():
    """Token-proxy shape: no `in_flight` field on the doc.

    The token-proxy /health doc exposes OAuth status only — no session
    bookkeeping. A classifier that fabricated a busy signal out of an
    absent field would force every token-proxy drain to sleep the full
    grace and re-burn the wait this whole change is meant to avoid. So
    absent `in_flight` is structurally `unknown`, and the bash caller
    treats `unknown` like `busy` — sleep the remaining grace and err on
    the side of patience.
    """
    srv, port = _start_stub(
        {"ok": True, "provider": "xai", "authorised": True,
         "supports_chat_completions": True})
    try:
        rc, out, err = _run_health_idle(port)
    finally:
        srv.shutdown()
    assert rc == 0, (rc, err)
    assert out == "unknown", (out, err)


def test_health_idle_connection_refused_is_unknown():
    """Loopback with nothing listening: connection-refused is `unknown`.

    Carried by the same `unknown` path the absent-in_flight case uses —
    the script's fetch helper returns None on URLError, classify maps
    None to `unknown`, and bash's `|| echo unknown` belt-and-braces
    a non-zero exit so a docker exec failure also lands here.
    """
    # Reserve a loopback port from the OS allocator and close it before
    # driving the probe — the kernel picks a free port number, the `with`
    # returns it as soon as we have it, and a close-then-connect race is
    # narrow enough to ignore on the loopback interface the conftest
    # guard allows. 127.0.0.1 is the only network surface the suite
    # is allowed to touch.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    rc, out, err = _run_health_idle(port)
    assert rc == 0, (rc, err)
    assert out == "unknown", (out, err)
    assert err == "", (out, err)


def test_health_idle_printable_token_exits_zero():
    """The three output tokens are the only things `print` ever emits.

    Bash only needs to read one line per drain; a stray warning on
    stdout (e.g. an unhandled KeyError) would be parsed as the state
    and steer the grace wrong. This test pins the printed surface.
    """
    cases = [
        ({"in_flight": 0}, "idle"),
        ({"in_flight": 1}, "busy"),
        ({"in_flight": 0, "sessions": 0, "awaiting_followup": 0}, "idle"),
        ({"in_flight": 0, "sessions": 1}, "busy"),
        # No in_flight at all → unknown (token-proxy shape).
        ({"ok": True}, "unknown"),
    ]
    for payload, expected in cases:
        srv, port = _start_stub(payload)
        try:
            res = subprocess.run(
                ["python3", os.path.join(ROOT, "scripts", "health_idle.py")],
                env={**os.environ, "SIDECAR_PORT": str(port)},
                capture_output=True, text=True,
            )
        finally:
            srv.shutdown()
        assert res.returncode == 0, (payload, res)
        assert res.stdout.strip() == expected, (payload, res.stdout)
        # No diagnostic noise on stderr either — the script must be
        # silent except for the single printed token.
        assert res.stderr == "", (payload, res.stderr)


# ---------------------------------------------------------------- apply.sh syntax


def test_apply_sh_bash_syntax_check():
    """`bash -n scripts/apply.sh` exits 0.

    A typo in the new heredoc or the health_state capture would
    parse-fail, and a parse-failing script is one operator run away
    from aborting mid-deploy. The shell's `-n` parse-only gate is the
    cheapest way to fail this in CI.
    """
    res = subprocess.run(
        ["bash", "-n", os.path.join(ROOT, "scripts", "apply.sh")],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, (
        f"bash -n should parse cleanly, got rc={res.returncode}; "
        f"stdout={res.stdout!r} stderr={res.stderr!r}")


# ---------------------------------------------------------------- apply.sh text


def _apply_sh_text():
    with open(os.path.join(ROOT, "scripts", "apply.sh")) as fh:
        return fh.read()


def test_apply_sh_failure_branch_surfaces_helper_output():
    """The drain-helper failure branch prints the helper's last ~8 lines.

    The original line was
        `lease migration skipped (drain helper unavailable or no leases)`
    which conflated "no leases" (a no-op) with "drain.py is missing from
    the image" (fixable with --build). The replacement branch names the
    failure mode (helper exited non-zero), tails the captured output so
    the operator can read it, AND — when the captured output carries
    the classic `No module named switchyard.drain` — adds the rebuild
    hint. The text check here pins each of those three pieces.
    """
    text = _apply_sh_text()
    assert "drain helper unavailable or no leases" not in text, (
        "the misleading 'unavailable or no leases' message must be gone")
    assert "WARN: lease migration skipped — drain helper exited non-zero" in text, (
        "failure branch must announce itself with the WARN header")
    # Tail the captured output so the operator sees something — the
    # exact byte count is a heuristic, not a contract, so we check the
    # command shape (tail -n 8 / sed / indenting) rather than a number.
    assert "tail -n 8 \"$drain_out\"" in text, (
        "failure branch must call `tail -n 8 \"$drain_out\"` to surface "
        "the helper's stderr tail")
    # And the matching `--build` hint fires when the captured output
    # names the missing module — the operator-visible part of the fix.
    assert "No module named switchyard.drain" in text
    assert "rerun with --build to rebuild it" in text, (
        "the '--build' hint must print when the helper's stderr names "
        "`No module named switchyard.drain`")


def test_apply_sh_failure_branch_pipelines_are_pipefail_safe():
    """`set -euo pipefail` is active in apply.sh; every pipeline under
    the new branch is guarded with `|| true` (or runs under `if`, which
    consumes the test command's exit code).

    A `tail` of a non-existent file or a `grep` that finds nothing
    would otherwise abort the whole script, taking the recreate loop
    down with it. The text check pins every line in the failure branch
    that runs under pipefail.
    """
    text = _apply_sh_text()
    # Find the failure-branch region (between the `else` after the
    # `if docker compose exec -T gateway ...` and the matching `fi`).
    start = text.index("    if docker compose exec -T gateway python -m switchyard.drain")
    end = text.index("    [ -n \"$drain_out\" ] && rm -f \"$drain_out\"",
                     start)
    branch = text[start:end]
    # The `tail ... | sed ...` pipeline is guarded with `|| true`. The
    # `if grep -q ...` does not need `|| true` because `if` consumes
    # the test command's exit code; only assert on the pipelines that
    # are actually exposed.
    assert "| sed 's/^/        /' || true" in branch, (
        "the tail|sed pipeline must end with `|| true` under set -euo "
        "pipefail — got the branch text:\n" + branch)
    # And the grep under `if` is the safe shape (no `|| true` needed).
    assert "if grep -q \"No module named switchyard.drain\"" in branch, (
        "the `--build` hint must fire inside an `if grep` so its "
        "non-match exit-1 doesn't trip pipefail")


def test_apply_sh_grace_block_consumes_health_probe():
    """The grace block exec's `health_idle.py` and respects the answer.

    Four things must be true:

      1. `health_idle.py` is exec'd INTO the sidecar via the same
         `docker compose exec -T "$svc" python3 - < scripts/...`
         shape the compose healthchecks already use, with a fallback
         `|| echo unknown` so a non-zero exit (timeout's exit, the
         container mid-recreate, anything else) still lands in the
         bash switch.

      2. The OUTER `docker compose exec` is guarded by a timeout that
         comes from a detected `timeout` (Linux) or `gtimeout`
         (macOS coreutils) command, with an unbounded fallback per
         CLAUDE.md:179-180 — never a hard-coded `timeout` invocation
         that would fail on a fresh macOS install.

      3. The `idle` case prints the documented message AND skips the
         sleep.

      4. The `busy` and `unknown` cases log a grace-sleeps-Ns message
         AND keep the sleep — these are the cases the original 120s
         unconditional sleep was approximating, and they must not
         silently regress to a no-op.
    """
    text = _apply_sh_text()
    # (1) the in-container exec, with timeout + fallback. The prefix
    # is `$timeout_cmd` (set just above) rather than a literal
    # `timeout 5` — that's the whole point of the cycle-2 fix.
    expected_exec = (
        '$timeout_cmd docker compose exec -T "$svc" python3 - < scripts/health_idle.py '
        '\\\n        2>/dev/null || echo unknown'
    )
    assert expected_exec in text, (
        "grace block must exec `health_idle.py` into the sidecar via "
        "`$timeout_cmd docker compose exec -T $svc python3 - < scripts/health_idle.py` "
        "with a `|| echo unknown` fallback — missed in the file")
    # (2) the timeout detection: probe for `timeout` first, fall back
    # to `gtimeout`, fall back to unbounded. The detection must be in
    # the same function/scope as the exec (not a top-level constant),
    # because a future top-level refactor would put the macOS branch
    # back in the failure path.
    assert "command -v timeout >/dev/null 2>&1" in text, (
        "grace block must probe for `timeout` (Linux) so a macOS "
        "host without GNU coreutils installed doesn't abort the "
        "drain with `command not found` mid-loop")
    assert "command -v gtimeout >/dev/null 2>&1" in text, (
        "grace block must probe for `gtimeout` (macOS coreutils) as "
        "the documented fallback per CLAUDE.md:179-180")
    assert 'timeout_cmd=""' in text, (
        "grace block must initialize `timeout_cmd` empty so the "
        "unbounded fallback (CLAUDE.md's documented last resort) "
        "fires when neither `timeout` nor `gtimeout` is on PATH")
    # (3) the idle short-circuit.
    assert "grace: skipped ($svc /health reports nothing in flight or parked)" in text, (
        "grace block must log `grace: skipped ($svc /health reports "
        "nothing in flight or parked)` on idle")
    # (4a) the busy branch keeps the sleep. Locate the `case` arm by
    # anchoring on the case statement (the only place `busy)` is the
    # left-hand label of a `case` pattern — anywhere else it's prose).
    case_idx = text.index('case "$health_state" in')
    busy_block_idx = text.index("busy)", case_idx)
    busy_end = text.index(";;", busy_block_idx)
    busy = text[busy_block_idx:busy_end]
    assert "sleep \"$remaining\"" in busy, (
        f"`busy)` branch must keep `sleep \"$remaining\"` — got: {busy!r}")
    # (4b) the wildcard (unknown) branch keeps the sleep. Same `case`
    # anchoring so a stray `*)` in a comment can't satisfy this.
    wild_block_idx = text.index("*)", case_idx)
    wild_end = text.index(";;", wild_block_idx)
    wild = text[wild_block_idx:wild_end]
    assert "sleep \"$remaining\"" in wild, (
        f"`*)` branch (unknown) must keep `sleep \"$remaining\"` — got: {wild!r}")


def test_apply_sh_predict_total_secs_does_not_exec_into_containers():
    """`predict_total_secs` must remain a heuristic, NOT exec into a
    container. The dry-run path never starts a container, and the
    function is the only thing that decides the predicted total drain
    time. If it exec'd `health_idle.py` (or any docker compose
    command), `--dry-run` would stop being a no-network helper and
    would block on a real container it has no way to talk to.

    This pins the original behavior: a heuristic, applied file-side.
    """
    text = _apply_sh_text()
    start = text.index("predict_total_secs() {")
    end = text.index("\n}\n", start) + 3
    fn = text[start:end]
    # No `docker` calls, no exec, no `health_idle.py` reference. The
    # only inputs are the score maps and `drain_grace_secs`.
    for forbidden in ("docker", "compose exec", "health_idle",
                      "SIDECAR_PORT"):
        assert forbidden not in fn, (
            f"predict_total_secs must remain a heuristic with no "
            f"container exec; found {forbidden!r} in:\n{fn}")


def test_apply_sh_step_four_header_describes_health_gated_grace():
    """The step-4 header comment in apply.sh must describe the new
    health-gated grace behavior. Three sentences carry the
    explanation: the doc carries `awaiting_followup`, the doc is
    classified as `idle`/`busy`/`unknown`, and the busy/unknown cases
    still wait. The header is a contract summary the operator reads
    before reaching for the script.
    """
    text = _apply_sh_text()
    # The block is the multi-line paragraph between "Step 4 is the
    # unload-first drain loop." and "When reload.sh takes the
    # policy-only hot-swap path," — both belong to the comment
    # structure just after the Order block.
    start = text.index("Step 4 is the unload-first drain loop.")
    end = text.index("When reload.sh takes the policy-only hot-swap path,")
    header = text[start:end]
    # health-gated grace: the script invokes scripts/health_idle.py.
    assert "scripts/health_idle.py" in header, (
        "step-4 header must name `scripts/health_idle.py` so an "
        "operator reading the contract sees the new gate")
    # idle: the no-op case.
    assert "idle" in header, (
        "step-4 header must mention the `idle` outcome")
    # busy: the parked case.
    assert "busy" in header, (
        "step-4 header must mention the `busy` outcome")
    # unknown: the case where the helper didn't return idle, e.g.
    # token-proxy /health doesn't expose in_flight.
    assert "unknown" in header, (
        "step-4 header must mention the `unknown` outcome "
        "(the /health probe returned no `in_flight` field)")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
