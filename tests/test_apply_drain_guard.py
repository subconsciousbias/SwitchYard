"""Drain-flag TTL + cleanup-trap contract for `scripts/apply.sh`.

Issue: an interrupted `apply.sh` was leaving `sy:drain:{plan}` set
forever, so the picker never routed to that plan again until someone ran
`DEL` by hand. Three failure paths converged on this:

  (a) `set -e` death (daemon hiccup, `compose up` failure) — the script
      exits with the failure code but never clears the gate.
  (b) Ctrl-C / SIGTERM during the grace sleep — same, no clear.
  (c) SIGKILL — un-trap-able, no chance to clear in-band.

The fix is two-fold, in `scripts/apply.sh` step 4a (drain loop):

  - SET now carries `EX $((drain_grace_secs + 600))` so the flag self-
    heals within (grace + 10 minutes) even when no trap can run.
  - An EXIT trap clears the flag on every other exit path. INT/TERM
    are routed through `exit N` so the shell unwinds through the EXIT
    trap; `|| true` + `2>/dev/null` keep `set -e` from letting a
    failing DEL clobber the script's exit status.

These tests stage a temp tree — a regular `git init` repo (passes the
worktree guard), the real `apply.sh` + `image_plan.py` + `health_idle.py`,
stubs for `scripts/sync-env.sh` and `scripts/auth_audit.py`, the example
`config/plans.yaml`, a symlinked `switchyard/`, and `Dockerfile.sidecar`
+ `sidecars/` (the sidecar image's build inputs). `tests/fake_docker.py`
first on PATH logs every argv and answers from a JSON state in which the
sidecar image carries no `switchyard.inputs` label (so it is rebuilt and
recreated) and its /health probe says `unknown` (so it is drained rather
than batched); no real redis, compose project, or container is touched.
The drain runs in a background job now (#262), with its own EXIT trap.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------
# Fake docker shim
# ---------------------------------------------------------------------------

# The shared stateful fake (tests/fake_docker.py). One sidecar, `test-sidecar`,
# on plan `testplan`, whose image exists WITHOUT a `switchyard.inputs` label —
# so image_plan.py rebuilds it and plans a recreate — and whose /health probe
# answers `unknown`, so apply.sh classifies it busy and runs the drain (the
# path whose SET / trap / DEL contract these tests pin).
import fake_docker  # noqa: E402


def _initial_state():
    return {
        "project": "sy",
        "services": {
            "test-sidecar": {"build": {"dockerfile": "Dockerfile.sidecar"},
                             "image": "switchyard-sidecar:latest",
                             "environment": {"SWITCHYARD_PLAN": "testplan"}},
            "redis": {"image": "redis:7-alpine", "environment": {}},
        },
        "images": {"switchyard-sidecar:latest": {"Id": "sha256:old", "Labels": {}},
                   "redis:7-alpine": {"Id": "sha256:redis", "Labels": {}}},
        "containers": {
            "test-sidecar": {"ID": "c-test-sidecar", "Image": "sha256:old",
                             "Env": ["SWITCHYARD_PLAN=testplan"],
                             "Status": "running", "Health": "healthy"},
            "redis": {"ID": "c-redis", "Image": "sha256:redis", "Env": [],
                      "Status": "running", "Health": "healthy"},
        },
        "zcard": {"testplan": [0]},
        "health": {"test-sidecar": "unknown"},
    }


# ---------------------------------------------------------------------------
# Sandbox staging
# ---------------------------------------------------------------------------


def _stage_apply_sandbox():
    """Build a fresh tempdir tree that looks enough like the real checkout
    for `apply.sh` to make it to the drain loop.

    Returns (tmp, shim_path, shim_log, http_servers, cleanup).

    - tmp         the temp root, set up as a regular git repo (so the
                  worktree guard at apply.sh:62 passes)
    - shim_path   absolute path of the fake `docker` binary; first on PATH
                  for the apply.sh subprocess
    - shim_log    path of the argv-log file; the apply.sh subprocess writes
                  one line per docker invocation
    - http_servers list of (port, ThreadingHTTPServer) — loopback HTTP
                  stubs for /healthz and /health/liveliness. Each server
                  returns 200 on every path. If a port is already in use
                  (live stack), the bind is skipped and the list omits
                  that entry — a live GET /healthz is read-only.
    - cleanup     callable that tears the whole thing down.
    """
    tmp = tempfile.mkdtemp(prefix="sy-apply-drain-")
    subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)

    # Real apply.sh — the script under test is the one this branch adds,
    # not the previous version.
    scripts_dir = os.path.join(tmp, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    for name in ("apply.sh", "image_plan.py", "health_idle.py"):
        shutil.copy2(os.path.join(ROOT, "scripts", name),
                     os.path.join(scripts_dir, name))

    # Stubs.
    with open(os.path.join(scripts_dir, "sync-env.sh"), "w") as fh:
        fh.write("#!/usr/bin/env bash\necho '(stub)'\nexit 0\n")
    os.chmod(os.path.join(scripts_dir, "sync-env.sh"), 0o755)
    with open(os.path.join(scripts_dir, "auth_audit.py"), "w") as fh:
        fh.write("#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n")
    os.chmod(os.path.join(scripts_dir, "auth_audit.py"), 0o755)

    # Real example plans.yaml — copy, don't symlink: apply.sh validates
    # the file's existence on disk.
    config_dir = os.path.join(tmp, "config")
    os.makedirs(config_dir, exist_ok=True)
    example = os.path.join(ROOT, "config", "plans.example.yaml")
    shutil.copy2(example, os.path.join(config_dir, "plans.yaml"))

    # symlinked switchyard/ — the validation + router_signature blocks
    # resolve `from switchyard import models` via `sys.path.insert(0, ".")`.
    os.symlink(os.path.join(ROOT, "switchyard"), os.path.join(tmp, "switchyard"))

    # Dockerfile.sidecar + sidecars/ — the sidecar image's build inputs, so
    # image_plan.py has a real manifest to compare with the (missing) label.
    shutil.copy2(os.path.join(ROOT, "Dockerfile.sidecar"),
                 os.path.join(tmp, "Dockerfile.sidecar"))
    shutil.copytree(os.path.join(ROOT, "sidecars"),
                    os.path.join(tmp, "sidecars"))

    # Fake docker first on PATH for the apply.sh subprocess.
    shim_path = fake_docker.install(os.path.join(tmp, "bin"))
    with open(os.path.join(tmp, "docker-state.json"), "w") as fh:
        json.dump(_initial_state(), fh)
    shim_log = os.path.join(tmp, "shim.log")
    open(shim_log, "w").close()

    # Loopback HTTP stubs. Best-effort: if the live stack already binds
    # these ports, skip — the curls will hit the real services, which
    # also serve /healthz and /health/liveliness (read-only).
    http_servers = []

    def _try_bind(port):
        srv = ThreadingHTTPServer(("127.0.0.1", port), _HealthHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        http_servers.append((port, srv))

    for port in (4001, 4000):
        try:
            _try_bind(port)
        except OSError:
            # Live stack is already on this port; skip — the curl will
            # reach the live service, which is read-only and fine.
            pass

    def cleanup():
        for _port, srv in http_servers:
            srv.shutdown()
            srv.server_close()
        shutil.rmtree(tmp, ignore_errors=True)

    return tmp, shim_path, shim_log, http_servers, cleanup


class _HealthHandler(BaseHTTPRequestHandler):
    """200 on every path — `/healthz`, `/health/liveliness`, anything."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *_args, **_kwargs):
        return


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shim_log_lines(shim_log):
    with open(shim_log) as fh:
        return [ln.rstrip("\n") for ln in fh]


def _wait_for_line(shim_log, needle, timeout=15.0, poll=0.05):
    """Poll the shim log until a line containing `needle` appears, or
    raise if it doesn't show up within `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for ln in _shim_log_lines(shim_log):
            if needle in ln:
                return ln
        time.sleep(poll)
    raise AssertionError(
        f"timed out waiting for {needle!r} in shim log; "
        f"have so far: {_shim_log_lines(shim_log)!r}")


def _index_of(shim_log, predicate):
    """Index of the first line in shim_log matching `predicate(line)`,
    or -1."""
    for i, ln in enumerate(_shim_log_lines(shim_log)):
        if predicate(ln):
            return i
    return -1


def _run_apply(tmp, shim_path, shim_log, *args, start_new_session=False):
    """Invoke `bash scripts/apply.sh <args>` inside `tmp` with the shim
    first on PATH and SHIM_LOG wired up. Returns the Popen object."""
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(shim_path) + os.pathsep + env.get("PATH", "")
    env["FAKE_DOCKER_LOG"] = shim_log
    env["FAKE_DOCKER_STATE"] = os.path.join(tmp, "docker-state.json")
    return subprocess.Popen(
        ["bash", "scripts/apply.sh", *args],
        cwd=tmp, env=env,
        start_new_session=start_new_session,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_apply_killed_mid_grace_clears_drain_flag():
    """A SIGTERM during the grace sleep must (a) leave the SET argv
    visible in the shim log and (b) run the EXIT-trap DEL after it,
    so a subsequent `docker compose exec` doesn't see a stranded drain
    flag. The script must exit nonzero."""
    tmp, shim_path, shim_log, _http, cleanup = _stage_apply_sandbox()
    try:
        # grace=30 so the sleep after ZCARD=0 is long enough that our
        # poll + kill arrive mid-grace, not after DEL has already run.
        proc = _run_apply(tmp, shim_path, shim_log,
                          "--skip-reload", "--drain-grace-secs", "30",
                          start_new_session=True)

        # Wait for the SET — that's the marker that we're inside the
        # drain loop and the kill will land during the grace sleep (the
        # explicit DEL on the happy path comes ~30s + recreate later).
        _wait_for_line(shim_log, "SET sy:drain:testplan", timeout=20)

        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise AssertionError(
                f"apply.sh did not exit within 15s of SIGTERM; "
                f"stdout={stdout!r} stderr={stderr!r}") from exc
        assert proc.returncode != 0, (
            f"SIGTERM mid-grace must exit nonzero; "
            f"got rc={proc.returncode}; "
            f"stdout={stdout!r} stderr={stderr!r}")

        lines = _shim_log_lines(shim_log)
        set_idx = _index_of(shim_log, lambda ln: "SET sy:drain:testplan" in ln)
        assert set_idx >= 0, (
            f"SET must precede the DEL; have: {lines!r}")
        del_after_set = [ln for ln in lines[set_idx + 1:]
                         if "DEL sy:drain:testplan" in ln]
        assert del_after_set, (
            f"EXIT trap must DEL sy:drain:testplan after the SET "
            f"(a stranded flag would block traffic forever). "
            f"SET was at line {set_idx}; "
            f"no DEL followed in: {lines[set_idx:]!r}")
    finally:
        cleanup()
    print("  SIGTERM mid-grace: SET logged, EXIT-trap DEL follows, "
          "exit nonzero")


def test_apply_sets_ttl_and_clears_on_happy_path():
    """Happy-path contract: the SET argv must carry `EX` with a TTL of
    (grace + 600), and the final DEL must run. grace=1 → EX 601."""
    tmp, shim_path, shim_log, _http, cleanup = _stage_apply_sandbox()
    try:
        proc = _run_apply(tmp, shim_path, shim_log,
                          "--skip-reload", "--drain-grace-secs", "1")
        try:
            stdout, stderr = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise AssertionError(
                f"apply.sh did not complete within 60s; "
                f"stdout={stdout!r} stderr={stderr!r}") from exc
        assert proc.returncode == 0, (
            f"happy-path apply.sh must exit 0; "
            f"got rc={proc.returncode}; "
            f"stdout={stdout!r} stderr={stderr!r}")

        lines = _shim_log_lines(shim_log)
        set_lines = [ln for ln in lines
                     if "SET sy:drain:testplan" in ln]
        assert set_lines, (
            f"SET sy:drain:testplan must appear in shim log; "
            f"have: {lines!r}")
        # The SET argv must carry both `EX` and `601` (grace=1 + 600).
        # If `EX` is missing, the TTL is gone and a SIGKILL strands the
        # flag forever; if `601` is missing, the TTL is wrong.
        for ln in set_lines:
            assert "EX" in ln, (
                f"SET argv must carry the EX flag (TTL); got: {ln!r}")
            assert "601" in ln, (
                f"SET argv must carry TTL=601 for grace=1; got: {ln!r}")

        del_lines = [ln for ln in lines
                     if "DEL sy:drain:testplan" in ln]
        assert del_lines, (
            f"the explicit DEL on the happy path must run; "
            f"have: {lines!r}")
    finally:
        cleanup()
    print("  happy path: SET argv carries EX 601, final DEL ran, exit 0")


def test_drain_job_signal_handlers_ignore_a_second_signal_first():
    """Under a process-group kill the drain job gets TERM directly and again
    from the parent's on_exit forward. If its handler were a bare `exit 143`,
    the second TERM could land in the EXIT trap before clear_drain_flag's own
    `trap ''` and skip the DEL (seen as a CI flake of the test above). The
    handlers must ignore INT/TERM before exiting."""
    with open(os.path.join(ROOT, "scripts", "apply.sh")) as fh:
        text = fh.read()
    body = text[text.index("drain_one() {"):]
    body = body[:body.index("\n}\n")]
    assert """trap 'trap "" INT TERM; exit 143' TERM""" in body, body[:1500]
    assert """trap 'trap "" INT TERM; exit 130' INT""" in body, body[:1500]
    assert "trap 'exit 143' TERM" not in body


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
