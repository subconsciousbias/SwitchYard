"""Offline tests for apply.sh's automatic rebuild detection, --fast, the
parallel idle-batch recreate, and the polled (never blind) waits.

Everything runs in a temp tree (`git init`-ed so apply.sh's worktree guard
passes) holding copies of the real scripts, Dockerfiles, switchyard/ and
sidecars/, with `tests/fake_docker.py` installed first on PATH as `docker`.
The fake answers from a JSON state file and logs every argv, so the tests
assert on exactly which docker commands apply.sh / image_plan.py ran. No
daemon, no redis, no container is touched.

What is pinned:

  * detection is by CONTENT of each image's Dockerfile + COPY inputs, against
    the `switchyard.inputs` label the build stamped: no change -> nothing
    rebuilt; switchyard/ -> gateway+portal (+sidecar only for the modules the
    sidecar COPYs); sidecars/ -> sidecar only; a Dockerfile -> that image;
    an mtime-only touch -> nothing; a missing label -> rebuilt once;
  * a container whose environment differs from compose's resolved config is
    recreated (the .env case), and the reason names keys, never values;
  * --build forces all, --no-build never builds, --plan builds nothing;
  * --fast recreates in one compose call and never execs, stops, polls
    health, reloads or audits;
  * the normal path recreates idle services in ONE compose call, drains busy
    ones in parallel background jobs, fails non-zero when a job fails, polls
    its waits (a condition met on the 2nd poll returns in about one
    interval), and still proceeds on a drain timeout.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import fake_docker  # noqa: E402

BUILT = ["claude-max-sidecar", "gateway", "portal", "xai-token-proxy"]
ALL_APP = ["claude-max-sidecar", "codex-sidecar", "gateway", "portal", "xai-token-proxy"]


def _services():
    return {
        "gateway": {"build": {"dockerfile": "Dockerfile.gateway"},
                    "environment": {"REDIS_HOST": "redis", "GLM_API_KEY": "secret-1"}},
        "portal": {"build": {"dockerfile": "Dockerfile.portal"},
                   "environment": {"SWITCHYARD_REDIS_URL": "redis://redis:6379/1"}},
        "claude-max-sidecar": {"build": {"dockerfile": "Dockerfile.sidecar"},
                               "image": "switchyard-sidecar:latest",
                               "environment": {"SWITCHYARD_PLAN": "claude-max",
                                               "SIDECAR_PORT": "8081"}},
        "codex-sidecar": {"image": "switchyard-sidecar:latest",
                          "environment": {"SWITCHYARD_PLAN": "openai",
                                          "SIDECAR_PORT": "8082"}},
        "xai-token-proxy": {"build": {"dockerfile": "Dockerfile.token_proxy"},
                            "image": "switchyard-token-proxy:latest",
                            "environment": {"SWITCHYARD_PLAN": "grok"}},
        "redis": {"image": "redis:7-alpine", "environment": {}},
    }


class Sandbox:
    """A throwaway checkout + fake docker. Baseline: every image built by
    image_plan.py (so it carries a label) and every container current."""

    def __init__(self, baseline: bool = True):
        self.tmp = tempfile.mkdtemp(prefix="sy-apply-auto-")
        t = self.tmp
        subprocess.run(["git", "init", "-q"], cwd=t, check=True)
        os.makedirs(os.path.join(t, "scripts"))
        self.reload_real = os.path.join(t, "scripts", "reload-real.sh")
        shutil.copy2(os.path.join(ROOT, "scripts", "reload.sh"), self.reload_real)
        for name in ("apply.sh", "image_plan.py", "health_idle.py"):
            shutil.copy2(os.path.join(ROOT, "scripts", name), os.path.join(t, "scripts", name))
        self.marker = os.path.join(t, "markers.log")
        for name, body in (("sync-env.sh", "echo '(stub sync-env)'"),
                           ("reload.sh", f"echo reload >> {self.marker}"),
                           ("auth_audit.py", None)):
            p = os.path.join(t, "scripts", name)
            with open(p, "w") as fh:
                if body is None:
                    fh.write("import sys\n"
                             f"open({self.marker!r}, 'a').write('audit\\n')\n"
                             "sys.exit(0)\n")
                else:
                    fh.write(f"#!/usr/bin/env bash\n{body}\n")
            os.chmod(p, 0o755)
        os.makedirs(os.path.join(t, "config"))
        shutil.copy2(os.path.join(ROOT, "config", "plans.example.yaml"),
                     os.path.join(t, "config", "plans.yaml"))
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        for d in ("switchyard", "sidecars", "docker"):
            shutil.copytree(os.path.join(ROOT, d), os.path.join(t, d), ignore=ignore)
        for f in os.listdir(ROOT):
            if f.startswith("Dockerfile") or f == "requirements.txt":
                shutil.copy2(os.path.join(ROOT, f), os.path.join(t, f))
        self.bin = os.path.join(t, "bin")
        fake_docker.install(self.bin, curl=True)
        self.state_path = os.path.join(t, "docker-state.json")
        self.log = os.path.join(t, "docker.log")
        self.write_state({"project": "sy", "services": _services(), "images": {},
                          "containers": {}})
        if baseline:
            assert self.plan_py("build", *BUILT).returncode == 0
            self.docker("compose", "up", "-d", "--force-recreate")
        open(self.log, "w").close()

    # -- state / env ---------------------------------------------------------
    def env(self, **extra):
        env = os.environ.copy()
        env["PATH"] = self.bin + os.pathsep + env.get("PATH", "")
        env["FAKE_DOCKER_STATE"] = self.state_path
        env["FAKE_DOCKER_LOG"] = self.log
        env.setdefault("SWITCHYARD_APPLY_POLL_SECS", "0.2")
        env.update(extra)
        return env

    def state(self):
        with open(self.state_path) as fh:
            return json.load(fh)

    def write_state(self, st):
        with open(self.state_path, "w") as fh:
            json.dump(st, fh)

    def update(self, fn):
        st = self.state()
        fn(st)
        self.write_state(st)

    def log_lines(self):
        with open(self.log) as fh:
            return [ln.rstrip("\n") for ln in fh]

    def markers(self):
        if not os.path.exists(self.marker):
            return []
        with open(self.marker) as fh:
            return fh.read().split()

    def edit(self, rel, extra="\n# edited\n"):
        with open(os.path.join(self.tmp, rel), "a") as fh:
            fh.write(extra)

    # -- runners -------------------------------------------------------------
    def docker(self, *args):
        return subprocess.run([os.path.join(self.bin, "docker"), *args], cwd=self.tmp,
                              env=self.env(), capture_output=True, text=True)

    def plan_py(self, *args):
        return subprocess.run([sys.executable, "scripts/image_plan.py", *args],
                              cwd=self.tmp, env=self.env(), capture_output=True, text=True)

    def plan(self, *flags):
        tsv = os.path.join(self.tmp, "plan.tsv")
        res = self.plan_py("plan", *flags, "--tsv", tsv)
        assert res.returncode == 0, res.stderr
        rows = [ln.split("\t") for ln in open(tsv).read().splitlines()]
        build = sorted(r[1] for r in rows if r[0] == "build")
        recreate = {r[1]: r[2] for r in rows if r[0] == "recreate"}
        return build, recreate, res.stdout

    def apply(self, *args, bash="bash", timeout=90, **env):
        start = time.monotonic()
        res = subprocess.run([bash, "scripts/apply.sh", *args], cwd=self.tmp,
                             env=self.env(**env), capture_output=True, text=True,
                             timeout=timeout)
        res.elapsed = time.monotonic() - start
        return res

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def _with(fn, **kw):
    sb = Sandbox(**kw)
    try:
        fn(sb)
    finally:
        sb.close()


# ------------------------------------------------------------ image_plan.py


def test_copy_sources_are_read_from_the_dockerfile():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import image_plan
    text = ("FROM x\nCOPY --chown=node:node a/b.py /app/\n"
            "COPY [\"c d.txt\", \"/x\"]\nCOPY --from=builder /bin/z /z\n"
            "ADD e \\\n  f /dst/\n# COPY g /h\n")
    assert image_plan.copy_sources(text) == ["a/b.py", "c d.txt", "e", "f"]


def test_sidecar_inputs_match_the_dockerfile_copy_lines():
    """The sidecar manifest is exactly Dockerfile.sidecar + what it COPYs:
    only three switchyard/ modules, never picker.py."""
    def check(sb):
        res = sb.plan_py("inputs", "claude-max-sidecar")
        assert res.returncode == 0, res.stderr
        paths = [ln.split()[-1] for ln in res.stdout.splitlines()]
        assert "Dockerfile.sidecar" in paths
        assert "switchyard/caller_env.py" in paths
        assert "switchyard/models.py" in paths
        assert "switchyard/picker.py" not in paths
        assert any(p.startswith("sidecars/mcp_bridge/") for p in paths)
        assert not any(p.startswith("sidecars/token_proxy/") for p in paths)
    _with(check)


def test_no_change_rebuilds_nothing():
    def check(sb):
        build, recreate, _ = sb.plan()
        assert build == [] and recreate == {}, (build, recreate)
    _with(check)


def test_mtime_only_touch_rebuilds_nothing():
    def check(sb):
        p = os.path.join(sb.tmp, "switchyard", "picker.py")
        os.utime(p, (time.time() + 3600, time.time() + 3600))
        build, recreate, _ = sb.plan()
        assert build == [] and recreate == {}, (build, recreate)
    _with(check)


def test_switchyard_change_rebuilds_gateway_and_portal_only():
    def check(sb):
        sb.edit("switchyard/picker.py")
        build, recreate, out = sb.plan()
        assert build == ["gateway", "portal"], build
        assert sorted(recreate) == ["gateway", "portal"], recreate
        assert "changed: switchyard/picker.py" in out, out
    _with(check)


def test_caller_env_change_also_rebuilds_the_sidecar_image():
    def check(sb):
        sb.edit("switchyard/caller_env.py")
        build, recreate, _ = sb.plan()
        assert build == ["claude-max-sidecar", "gateway", "portal"], build
        # every consumer of the shared sidecar image is recreated
        assert sorted(recreate) == ["claude-max-sidecar", "codex-sidecar",
                                    "gateway", "portal"], recreate
    _with(check)


def test_sidecars_change_rebuilds_the_sidecar_image_only():
    def check(sb):
        sb.edit("sidecars/mcp_bridge/server.py")
        build, recreate, _ = sb.plan()
        assert build == ["claude-max-sidecar"], build
        assert sorted(recreate) == ["claude-max-sidecar", "codex-sidecar"], recreate
    _with(check)


def test_dockerfile_change_rebuilds_that_image():
    def check(sb):
        sb.edit("Dockerfile.portal")
        build, _, out = sb.plan()
        assert build == ["portal"], build
        assert "Dockerfile.portal" in out
    _with(check)


def test_new_file_under_a_copied_dir_is_detected():
    def check(sb):
        with open(os.path.join(sb.tmp, "sidecars", "mcp_bridge", "new_mod.py"), "w") as fh:
            fh.write("x = 1\n")
        build, _, out = sb.plan()
        assert build == ["claude-max-sidecar"], build
        assert "+sidecars/mcp_bridge/new_mod.py" in out, out
    _with(check)


def test_force_rebuilds_all_and_no_build_rebuilds_none():
    def check(sb):
        build, _, _ = sb.plan("--force")
        assert build == sorted(BUILT), build
        sb.edit("sidecars/mcp_bridge/server.py")
        build, recreate, out = sb.plan("--no-build")
        assert build == [] and recreate == {}, (build, recreate)
        assert "skipped (--no-build)" in out
    _with(check)


def test_missing_label_is_rebuilt_once():
    """First run after this lands: images exist but carry no label."""
    def check(sb):
        sb.update(lambda st: [img.update(Labels={}) for img in st["images"].values()])
        build, _, out = sb.plan()
        assert build == sorted(BUILT), build
        assert "no switchyard.inputs label" in out
        assert sb.plan_py("build", *build).returncode == 0
        sb.docker("compose", "up", "-d")
        build, recreate, _ = sb.plan()
        assert build == [] and recreate == {}, (build, recreate)
    _with(check)


def test_build_stamps_the_manifest_label():
    def check(sb):
        labels = sb.state()["images"]["switchyard-sidecar:latest"]["Labels"]
        assert labels["switchyard.inputs"].startswith("v1,Dockerfile.sidecar:"), labels
    _with(check)


def test_env_drift_recreates_only_that_service_and_never_prints_values():
    def check(sb):
        sb.update(lambda st: st["services"]["gateway"]["environment"].update(
            GLM_API_KEY="secret-2"))
        build, recreate, out = sb.plan()
        assert build == [], build
        assert list(recreate) == ["gateway"], recreate
        assert "GLM_API_KEY" in recreate["gateway"]
        assert "secret" not in out and "secret" not in recreate["gateway"], out
    _with(check)


def test_container_on_an_older_image_is_recreated_without_rebuilding():
    def check(sb):
        sb.update(lambda st: st["containers"]["codex-sidecar"].update(Image="sha256:old"))
        build, recreate, _ = sb.plan()
        assert build == [], build
        assert list(recreate) == ["codex-sidecar"], recreate
        assert "older image" in recreate["codex-sidecar"]
    _with(check)


def test_service_without_a_container_is_planned_for_start():
    def check(sb):
        sb.update(lambda st: st["containers"].pop("codex-sidecar"))
        _, recreate, _ = sb.plan()
        assert recreate == {"codex-sidecar": "no container"}, recreate
    _with(check)


# ------------------------------------------------------------------ apply.sh


def _mutating(lines):
    return [ln for ln in lines if ln.startswith(("compose build", "compose up", "compose stop"))
            or " SET " in f" {ln} " or " DEL " in f" {ln} "]


def test_apply_plan_builds_nothing():
    def check(sb):
        sb.edit("sidecars/cli_bridge/server.py")
        res = sb.apply("--plan")
        assert res.returncode == 0, res.stderr
        assert "REBUILD" in res.stdout and "claude-max-sidecar" in res.stdout
        assert _mutating(sb.log_lines()) == [], sb.log_lines()
    _with(check)


def test_apply_rejects_build_with_no_build():
    def check(sb):
        res = sb.apply("--build", "--no-build")
        assert res.returncode == 2, (res.returncode, res.stderr)
    _with(check, baseline=False)


def _fast_run(sb, bash):
    sb.edit("sidecars/mcp_bridge/server.py")
    res = sb.apply("--fast", bash=bash)
    assert res.returncode == 0, (res.stdout, res.stderr)
    lines = sb.log_lines()
    assert "compose build claude-max-sidecar" in lines, lines
    assert ("compose up -d --no-deps --no-build --force-recreate "
            "claude-max-sidecar codex-sidecar") in lines, lines
    assert "compose up -d --no-build --no-recreate" in lines, lines
    # no drain, no probe, no stop, no health poll
    for ln in lines:
        assert not ln.startswith(("compose exec", "compose stop", "inspect --format")), ln
    assert sb.markers() == [], "fast must not run reload.sh or the auth audit"
    # the rebuilt image is labelled and the containers run it
    build, recreate, _ = sb.plan()
    assert build == [] and recreate == {}, (build, recreate)


def test_apply_fast_builds_changed_and_skips_every_wait():
    _with(lambda sb: _fast_run(sb, "bash"))


def test_apply_fast_runs_under_macos_bash_3_2():
    """/bin/bash is 3.2 on macOS; on Linux it is whatever the distro ships."""
    if not os.path.exists("/bin/bash"):
        return
    _with(lambda sb: _fast_run(sb, "/bin/bash"))


def test_apply_fast_no_build_skips_the_build():
    def check(sb):
        sb.edit("sidecars/mcp_bridge/server.py")
        res = sb.apply("--fast", "--no-build")
        assert res.returncode == 0, res.stderr
        assert not any(ln.startswith("compose build") for ln in sb.log_lines())
    _with(check)


def test_apply_dry_run_touches_nothing():
    def check(sb):
        sb.edit("switchyard/caller_env.py")
        sb.update(lambda st: st["containers"].pop("xai-token-proxy"))
        res = sb.apply("--dry-run")
        assert res.returncode == 0, (res.stdout, res.stderr)
        assert _mutating(sb.log_lines()) == [], _mutating(sb.log_lines())
        assert "docker compose up -d --no-build --no-recreate" in res.stdout
    _with(check)


def test_apply_recreates_idle_services_in_one_call_and_drains_busy_in_parallel():
    def check(sb):
        sb.edit("sidecars/mcp_bridge/server.py")      # sidecar image -> 2 consumers
        sb.update(lambda st: st["containers"].pop("xai-token-proxy"))  # #174
        sb.update(lambda st: st["services"].update({
            "opencode-go-sidecar": {"image": "switchyard-sidecar:latest",
                                    "environment": {"SWITCHYARD_PLAN": "opencode-go"}},
            "opencode-go2-sidecar": {"image": "switchyard-sidecar:latest",
                                     "environment": {"SWITCHYARD_PLAN": "opencode-go2"}}}))
        sb.docker("compose", "up", "-d", "opencode-go-sidecar", "opencode-go2-sidecar")
        # claude-max + opencode-go2 idle; codex + opencode-go busy for a few polls
        sb.update(lambda st: st.update(zcard={"openai": [2, 2, 2, 2, 0],
                                              "opencode-go": [1, 1, 1, 1, 0]}))
        open(sb.log, "w").close()
        res = sb.apply("--skip-reload", "--drain-grace-secs", "30")
        assert res.returncode == 0, (res.stdout, res.stderr)
        lines = sb.log_lines()
        batch = [ln for ln in lines if ln.startswith(
            "compose up -d --no-deps --no-build --force-recreate ")
            and "claude-max-sidecar" in ln]
        assert batch == ["compose up -d --no-deps --no-build --force-recreate "
                         "claude-max-sidecar opencode-go2-sidecar xai-token-proxy"], lines
        # both busy drains started (gate SET) before either stopped: parallel
        set_idx = [i for i, ln in enumerate(lines)
                   if "SET sy:drain:openai" in ln or "SET sy:drain:opencode-go " in ln + " "]
        stop_idx = [i for i, ln in enumerate(lines) if ln.startswith("compose stop")]
        assert len(set_idx) == 2 and len(stop_idx) == 2, lines
        assert max(set_idx) < min(stop_idx), lines
        for svc in ("codex-sidecar", "opencode-go-sidecar"):
            assert f"compose up -d --no-deps --no-build --force-recreate {svc}" in lines
        for plan in ("openai", "opencode-go", "claude-max", "opencode-go2"):
            assert any(f"DEL sy:drain:{plan}" in ln for ln in lines), (plan, lines)
        assert lines[-1] == "compose up -d --no-build --no-recreate" or \
            "compose up -d --no-build --no-recreate" in lines
        assert "drained after" in res.stdout
        assert sb.markers() == ["audit"]
        build, recreate, _ = sb.plan()
        assert build == [] and recreate == {}, (build, recreate)
    _with(check)


def test_apply_normal_path_runs_under_macos_bash_3_2():
    """The drain path (background jobs, subshell traps, string lists) must
    work on /bin/bash 3.2 too, not just --fast."""
    if not os.path.exists("/bin/bash"):
        return
    def check(sb):
        sb.edit("sidecars/mcp_bridge/server.py")
        sb.update(lambda st: st.update(zcard={"openai": [2, 2, 0]}))
        res = sb.apply("--skip-reload", "--drain-grace-secs", "30", bash="/bin/bash")
        assert res.returncode == 0, (res.stdout, res.stderr)
        assert "drained after" in res.stdout, res.stdout
        lines = sb.log_lines()
        assert "compose up -d --no-deps --no-build --force-recreate claude-max-sidecar" in lines
        assert "compose up -d --no-deps --no-build --force-recreate codex-sidecar" in lines
    _with(check)


def test_apply_drain_poll_returns_on_the_second_poll_not_the_grace():
    def check(sb):
        sb.edit("sidecars/mcp_bridge/server.py")
        # codex: 3 in flight at classification, then 0 -- but its /health
        # says a turn is still running on the 1st poll and idle on the 2nd.
        sb.update(lambda st: st.update(zcard={"openai": [3, 0]},
                                       health={"codex-sidecar": ["busy", "idle"]}))
        res = sb.apply("--skip-reload", "--drain-grace-secs", "60",
                       SWITCHYARD_APPLY_POLL_SECS="0.3")
        assert res.returncode == 0, (res.stdout, res.stderr)
        m = re.search(r"drained after (\d+)s", res.stdout)
        assert m, res.stdout
        assert int(m.group(1)) <= 5, f"polled drain must not wait out the 60s grace: {m.group(0)}"
        assert "WARN" not in res.stdout
    _with(check)


def test_apply_drain_timeout_still_proceeds():
    def check(sb):
        sb.edit("sidecars/mcp_bridge/server.py")
        sb.update(lambda st: st.update(zcard={"openai": [1]}))
        res = sb.apply("--skip-reload", "--drain-grace-secs", "1")
        assert res.returncode == 0, (res.stdout, res.stderr)
        assert "still busy after 1s" in res.stdout and "proceeding" in res.stdout
        assert "compose up -d --no-deps --no-build --force-recreate codex-sidecar" in sb.log_lines()
    _with(check)


def _router_sig(sb):
    res = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); from switchyard import models; "
         "print(models.router_signature(models.load('config/plans.yaml')))"],
        cwd=sb.tmp, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def test_reload_polls_for_the_hot_swap_verdict():
    """reload.sh used to `sleep 8` before reading the gateway log once; it
    now polls and returns as soon as the swap is logged (2nd poll here)."""
    def check(sb):
        sig = _router_sig(sb)
        sb.update(lambda st: st.update(
            redis_get={"switchyard:router_sig": sig},
            logs=["", "switchyard: registry reloaded in place"]))
        res = subprocess.run(["bash", sb.reload_real], cwd=sb.tmp,
                             env=sb.env(SWITCHYARD_APPLY_POLL_SECS="0.3"),
                             capture_output=True, text=True, timeout=60)
        assert res.returncode == 0, (res.stdout, res.stderr)
        # the script's own clock around the poll loop, not wall time (which
        # also counts plans.yaml validation, slow on CI runners)
        m = re.search(r"gateway: hot-swapped after (\d+)s", res.stdout)
        assert m, res.stdout
        assert int(m.group(1)) <= 3, f"must not wait out the old fixed 8s: {m.group(0)}"
        assert "compose restart gateway" not in sb.log_lines()
    _with(check, baseline=False)


def test_reload_without_a_verdict_still_restarts_after_the_bound():
    def check(sb):
        sig = _router_sig(sb)
        sb.update(lambda st: st.update(redis_get={"switchyard:router_sig": sig},
                                       logs=[""]))
        res = subprocess.run(["bash", sb.reload_real], cwd=sb.tmp,
                             env=sb.env(SWITCHYARD_APPLY_POLL_SECS="0.5"),
                             capture_output=True, text=True, timeout=90)
        assert res.returncode == 0, (res.stdout, res.stderr)
        assert "did not report a swap within 8s" in res.stdout, res.stdout
        assert "compose restart gateway" in sb.log_lines()
    _with(check, baseline=False)


def test_parked_mcp_session_counts_as_idle():
    """health_idle.py now answers `idle` for a parked mcp_bridge doc, so a
    sidecar with only parked sessions joins the batch instead of draining."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import health_idle
    assert health_idle.classify({"in_flight": 0, "sessions": 3,
                                 "awaiting_followup": 3}) == "idle"
    assert health_idle.classify({"in_flight": 1, "sessions": 1}) == "busy"
    assert health_idle.classify({"ok": True, "provider": "xai"}) == "untracked"
    assert health_idle.classify(None) == "unknown"


def test_apply_fails_when_a_background_drain_fails_and_keeps_its_gate():
    def check(sb):
        sb.edit("sidecars/mcp_bridge/server.py")
        sb.update(lambda st: st.update(zcard={"openai": [1, 0]},
                                       health_after={"codex-sidecar": "unhealthy"}))
        res = sb.apply("--skip-reload", "--drain-grace-secs", "5",
                       SWITCHYARD_APPLY_HEALTH_WAIT_SECS="1")
        assert res.returncode == 1, (res.returncode, res.stdout, res.stderr)
        out = res.stdout + res.stderr
        assert "NOT healthy" in out and "drain of codex-sidecar FAILED" in res.stderr, out
        lines = sb.log_lines()
        assert not any("DEL sy:drain:openai" in ln for ln in lines), lines
        # the healthy sibling in the idle batch was still re-admitted
        assert any("DEL sy:drain:claude-max" in ln for ln in lines), lines
        assert sb.markers() == [], "a failed apply must not go on to reload/audit"
    _with(check)


def test_apply_fails_when_an_idle_batch_member_never_gets_healthy():
    def check(sb):
        sb.edit("sidecars/mcp_bridge/server.py")
        sb.update(lambda st: st.update(health_after={"claude-max-sidecar": "starting"}))
        res = sb.apply("--skip-reload", SWITCHYARD_APPLY_HEALTH_WAIT_SECS="1")
        assert res.returncode == 1, (res.returncode, res.stdout, res.stderr)
        lines = sb.log_lines()
        assert not any("DEL sy:drain:claude-max" in ln for ln in lines), lines
        assert any("DEL sy:drain:openai" in ln for ln in lines), lines
    _with(check)


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
