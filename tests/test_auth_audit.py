"""The audit derives every per-subscription fact from the real service config.

Issue #9 was two hardcoded holes in one map: the audit checked OpenCode's
config dir (the credential lives in the data dir) and had no entry at all for
a second Go subscription. Hardcoding per-service facts guarantees the next
subscription added to docker-compose.yml hits another hole, so the audit now
derives service↔plan, credential path (from the compose volume mounts, with
${VAR:-default} resolved) and the OpenCode login id from docker-compose.yml,
.env and plans.yaml. These tests hold synthetic compose services — including a
subscription that exists nowhere in the repo's real config — to prove that.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from plans_path import plans_path  # noqa: E402

from auth_audit import (  # noqa: E402
    CONTAINER_CRED_FILES,
    audit,
    load_services,
    opencode_login_provider,
    resolve_host_mount,
)


def run_audit(services, dotenv=None):
    return audit(plans_path(), dotenv=dotenv or {}, services=services)


def test_the_real_compose_file_parses_into_cred_mounts_and_logins():
    status, pending = run_audit(load_services())
    go = [l for l in status if l.startswith("  opencode-go ")]
    assert any("secrets/opencode/data/auth.json" in l for l in go), go
    go2 = [l for l in status if l.startswith("  opencode-go2 ")]
    assert any("secrets/opencode2/data/auth.json" in l for l in go2), go2
    print("  the real compose file yields data-dir paths for both Go sidecars")


def test_a_third_subscription_needs_no_audit_change():
    # A subscription that exists nowhere in the repo: its own mini plans file
    # plus a compose entry with distinct env var names. If this passes, adding
    # subscriptions never means editing the audit.
    import tempfile
    plans = os.path.join(tempfile.mkdtemp(prefix="sy-audit3-"), "plans.yaml")
    with open(plans, "w") as fh:
        fh.write(
            "plans:\n"
            "  opencode-go3:\n"
            "    label: Go Three\n"
            "    auth: cli_sidecar\n"
            "    monthly_cost: 10\n"
            "    max_parallel: 1\n"
            "    api_base: http://opencode-go3-sidecar:8086/v1\n"
            "    api_key: sidecar\n"
            "    models:\n"
            "      glm-4.7:\n"
            "        model: openai/opencode-go/glm-4.7\n")
    services = {
        "opencode-go3-sidecar": {"environment": {
            "PROVIDER": "opencode", "SWITCHYARD_PLAN": "opencode-go3"},
            "volumes": ["./config:/app/config:ro",
                        "${OPENCODE3_DATA_DIR:-./secrets/opencode3/data}"
                        ":/home/node/.local/share/opencode",
                        "${OPENCODE3_CONFIG_DIR:-./secrets/opencode3/config}"
                        ":/home/node/.config/opencode"]},
    }
    try:
        status, pending = audit(plans, dotenv={}, services=services)
        line = next(l for l in status if l.startswith("  opencode-go3 "))
        assert "secrets/opencode3/data/auth.json" in line, line
        cmd = next(c for c in pending if "opencode-go3-sidecar" in c)
        assert cmd.endswith("opencode auth login --provider opencode-go"), cmd
    finally:
        os.unlink(plans)
        os.rmdir(os.path.dirname(plans))
    print("  an unseen third subscription is audited from its compose entry alone")


def test_env_override_and_precedence_match_compose():
    host = "${OPENCODE_DATA_DIR:-./secrets/opencode/data}"
    # compose order: shell env beats .env beats the default; `:-` also falls
    # back on a set-but-empty value.
    assert resolve_host_mount(host, {}) == "./secrets/opencode/data"
    assert resolve_host_mount(host, {"OPENCODE_DATA_DIR": "./secrets/x"}) == "./secrets/x"
    os.environ["OPENCODE_DATA_DIR"] = "/tmp/shell-wins"
    try:
        assert resolve_host_mount(host, {"OPENCODE_DATA_DIR": "./secrets/x"}) == "/tmp/shell-wins"
        os.environ["OPENCODE_DATA_DIR"] = ""
        assert resolve_host_mount(host, {"OPENCODE_DATA_DIR": "./secrets/x"}) == "./secrets/x"
    finally:
        os.environ.pop("OPENCODE_DATA_DIR", None)
    print("  ${VAR:-default} resolves shell env, .env, default, empty->default")


def test_claude_and_codex_services_derive_their_own_paths():
    # Fixture plan keys, but synthetic services with env var names the repo
    # has never heard of.
    services = {
        "claude-seat2-sidecar": {"environment": {
            "PROVIDER": "claude", "SWITCHYARD_PLAN": "claude-max"},
            "volumes": ["${CLAUDE2_CONFIG_DIR:-./secrets/claude2}:/home/node/.claude"]},
        "codex2-sidecar": {"environment": {
            "PROVIDER": "codex", "SWITCHYARD_PLAN": "openai"},
            "volumes": ["${CODEX2_CONFIG_DIR:-./secrets/codex2}:/home/node/.codex"]},
    }
    status, pending = run_audit(services)
    cl = next(l for l in status if l.startswith("  claude-max "))
    assert "secrets/claude2/.credentials.json" in cl, cl
    cx = next(l for l in status if l.startswith("  openai "))
    assert "secrets/codex2/auth.json" in cx, cx
    # Neither synthetic store exists, so both must be queued for their own
    # login command — the per-CLI invocations, not something plan-specific.
    cmds = {c.split()[3]: c for c in pending if c.startswith("docker compose exec")}
    assert cmds["claude-seat2-sidecar"].endswith("claude login"), cmds
    assert cmds["codex2-sidecar"].endswith("codex login --device-auth"), cmds
    print("  new claude/codex sidecars derive credential path and login command")


def test_missing_mount_says_what_it_looked_for():
    services = {"opencode-go-sidecar": {"environment": {
        "PROVIDER": "opencode", "SWITCHYARD_PLAN": "opencode-go"},
        "volumes": ["./config:/app/config:ro"]}}
    status, pending = run_audit(services)
    line = next(l for l in status if l.startswith("  opencode-go "))
    assert "mounts none of" in line, line
    assert not any("docker compose exec" in c for c in pending), \
        "an unauditable service must not emit a login command"
    print("  a service without a credential mount reports the container paths it checked")


def test_opencode_login_id_comes_from_the_plan_models():
    sys.path.insert(0, ROOT)
    from switchyard import models
    reg = models.load(plans_path())
    assert opencode_login_provider(reg.plans["opencode-go"]) == "opencode-go"
    # go2's model names the same OpenCode provider id — one login product, two
    # subscriptions — so the derived id is identical for both.
    assert opencode_login_provider(reg.plans["opencode-go2"]) == "opencode-go"
    print("  the login provider id is each plan model's own provider segment")


def test_container_cred_table_only_knows_cli_homes():
    assert set(CONTAINER_CRED_FILES) == {
        "/home/node/.claude", "/home/node/.codex", "/home/node/.local/share/opencode"}
    print("  the static table holds CLI homes only — no per-subscription paths")


def test_env_bak_suffix_snapshots_are_ignored():
    # Issue #33: `.env.bak-<suffix>` is a sync-env.sh snapshot name and must
    # be ignored alongside the other `.env`-family backup patterns.
    # `git check-ignore -q` rejects multiple pathnames, so loop and assert
    # each path individually — exit 0 means that path is ignored.
    import subprocess
    import shutil
    if shutil.which("git") is None:
        # The gateway image (where the suite is usually run) ships no git;
        # a FileNotFoundError there says nothing about .gitignore.
        try:
            import pytest
            pytest.skip("git not installed")
        except ImportError:
            print("  skipped: git not installed")
            return
    snapshot_names = [
        ".env.bak-20260921", ".env.bak-", ".env.bak",
        ".env.backup.1", ".env",
    ]
    not_ignored = []
    for name in snapshot_names:
        result = subprocess.run(
            ["git", "check-ignore", "-q", name],
            cwd=ROOT,
        )
        if result.returncode != 0:
            not_ignored.append((name, result.returncode))
    assert not not_ignored, \
        f"git check-ignore expected to ignore all five names, missing: {not_ignored}"
    print("  .env.bak-<suffix> snapshots are gitignored (issue #33)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"{len(fns)} tests passed")
