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
worktree guard at `apply.sh:62`), the real `apply.sh`, stubs for
`scripts/sync-env.sh` and `scripts/auth_audit.py`, the example
`config/plans.yaml`, a symlinked `switchyard/`, and `Dockerfile.sidecar`
+ `sidecars/` so `newest_mtime` is non-zero and the sidecar goes stale.
A fake `docker` shim first on PATH logs every argv to `$SHIM_LOG` and
returns canned output; no real redis, compose project, or container is
touched.
"""
from __future__ import annotations

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

DOCKER_SHIM = r"""#!/usr/bin/env python3
# Fake `docker` for tests/test_apply_drain_guard.py.
# Logs every argv to $SHIM_LOG and returns canned output. No real docker,
# no real redis, no real containers.
import os, sys

LOG = os.environ.get("SHIM_LOG")
if LOG:
    with open(LOG, "a") as fh:
        fh.write(" ".join(sys.argv[1:]) + "\n")

args = sys.argv[1:]


def _find_after(args, flag):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return None


# docker compose config --services
if args[:3] == ["compose", "config", "--services"]:
    for s in ("test-sidecar", "redis", "gateway", "portal", "xai-token-proxy"):
        print(s)
    sys.exit(0)

# docker compose exec -T <svc> ...
if args[:2] == ["compose", "exec"]:
    rest = args[2:]
    if rest and rest[0] == "-T":
        rest = rest[1:]
    svc = rest[0] if rest else ""
    rest = rest[1:]

    if svc == "redis":
        if rest and rest[0] == "redis-cli":
            rest = rest[1:]
            if rest and rest[0] == "-n":
                rest = rest[2:]
        sys.exit(0)

    if svc == "gateway":
        # python -m switchyard.drain "$plan" — return a well-formed summary
        # so apply.sh's awk parser picks up `migrated 0 session(s) off …`.
        print("migrated 0 session(s) off testplan")
        sys.exit(0)

    if svc == "test-sidecar":
        # printenv SWITCHYARD_PLAN
        if rest and rest[0] == "printenv":
            print("testplan")
        sys.exit(0)

    sys.exit(0)

# docker compose images -q <svc>
if args[:3] == ["compose", "images", "-q"]:
    svc = args[3] if len(args) > 3 else ""
    images = {
        "test-sidecar": "sha-sidecar",
        "redis": "sha-redis",
        "gateway": "sha-gateway",
        "portal": "sha-portal",
        "xai-token-proxy": "sha-xai",
    }
    img = images.get(svc, "")
    if img:
        print(img)
    sys.exit(0)

# docker compose ps -q <svc>
if args[:3] == ["compose", "ps", "-q"]:
    print("test-cid-12345")
    sys.exit(0)

# docker compose {stop,up,build}
if len(args) >= 2 and args[0] == "compose" and args[1] in ("stop", "up", "build"):
    sys.exit(0)

# docker image inspect <id> --format '{{.Created}}'
if args[:2] == ["image", "inspect"]:
    img = args[2] if len(args) > 2 else ""
    fmt = _find_after(args, "--format")
    if fmt and "Created" in fmt:
        # sidecar image is dated 2000 (stale → triggers the rebuild decision),
        # everything else is dated 2099 (fresh → kept out of the rebuild and
        # the fold-on recreate paths).
        print("2000-01-01T00:00:00.000000000Z" if img == "sha-sidecar"
              else "2099-01-01T00:00:00.000000000Z")
    sys.exit(0)

# docker inspect --format='{{.State.Health.Status}}' <cid>
if args and args[0] == "inspect":
    fmt = _find_after(args, "--format")
    if fmt and "Health" in fmt:
        print("healthy")
    sys.exit(0)

sys.exit(0)
"""


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
    shutil.copy2(os.path.join(ROOT, "scripts", "apply.sh"),
                 os.path.join(scripts_dir, "apply.sh"))

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

    # Dockerfile.sidecar + sidecars/ — without these, `newest_mtime`
    # returns 0.0 and the sidecar never goes stale, so apply.sh would
    # skip the drain loop entirely and the tests would have nothing to
    # observe. Copy rather than symlink so the mtime is concrete (any
    # value > 2000-01-01 triggers staleness against the fake image's
    # 2000-01-01 Created timestamp).
    shutil.copy2(os.path.join(ROOT, "Dockerfile.sidecar"),
                 os.path.join(tmp, "Dockerfile.sidecar"))
    shutil.copytree(os.path.join(ROOT, "sidecars"),
                    os.path.join(tmp, "sidecars"))

    # Fake docker shim. Put its dir first on PATH for the apply.sh
    # subprocess — that picks up the shim before any real `docker`.
    bin_dir = os.path.join(tmp, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    shim_path = os.path.join(bin_dir, "docker")
    with open(shim_path, "w") as fh:
        fh.write(DOCKER_SHIM)
    os.chmod(shim_path, 0o755)
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
    env["SHIM_LOG"] = shim_log
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


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
