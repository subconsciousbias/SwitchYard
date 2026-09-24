"""Mechanical enforcement guards: worktree refusal, .env writes, credential stores.

The guards are a layer above CLAUDE.md's prose rules: a `.claude/settings.json`
permission deny list plus a `.claude/hooks/guard.sh` PreToolUse hook for
Claude Code, mirrored in `.opencode/opencode.json` for OpenCode-driven
runtimes, and a top-of-file check in `scripts/apply.sh` / `scripts/reload.sh`
so a worktree invocation is refused before the script reaches Docker.

These tests assert the offline pieces in plain text form:

  (a) worktree refusal — `apply.sh --dry-run` and `reload.sh` exit 2 with the
      refusal message when invoked from inside a `git worktree add` checkout
      of this repo. We add a fresh detached worktree in a tempdir and run
      each script with `cwd=<that tempdir>`. The worktree's `scripts/` is
      the worktree's own copy — i.e. the modified script under test.
  (b) guard hook — `.claude/hooks/guard.sh` denies a hand-picked set of
      `.env` writes and keychain commands, and lets benign reads / commands
      through, when fed crafted stdin JSON. Includes the review-found gaps
      (rm/dd/install/rsync writes, .env.example/.env.backup.* false
      positives, `test.env` false positives), the worktree-only anchors
      (buildx/buildkit/upgrade/upload must NOT trip the worktree guard),
      and the issue's acceptance test that `git merge` (with or without
      args) is allowed from a worktree cwd while docker compose build / up
      / login / logout still exit 2.
  (c) compose fail-closed — `docker-compose.yml` declares its project name
      with the `${SWITCHYARD_PROJECT:?…}` required-substitution form so a
      worktree (no `.env`) fails at parse time, and `.env.example` carries
      the default value.
  (d) settings files — `.claude/settings.json` and `.opencode/opencode.json`
      parse as valid JSON and contain the deny strings from CLAUDE.md's
      rules. The OpenCode test also pins the broad `*` rule to `allow`
      (not `ask`) so the documented intent matches the file.

Nothing here touches the operator's real `.env`, opens a socket, or runs
Docker — the worktree test only shells out to `git worktree add` and the
two bash scripts, both of which exit at the guard before any Docker call.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from plans_path import EXAMPLE as PLANS_EXAMPLE  # noqa: E402

# Test (a): worktree refusal ----------------------------------------------

APPLY_REFUSAL = "refusing: this is a git worktree"
RELOAD_REFUSAL = "refusing: this is a git worktree"


def _spawn_worktree():
    """Create a detached worktree of the current repo in a tempdir.

    Returns (worktree_path, cleanup_callable). The worktree is rooted at
    the current repo's HEAD; the modified scripts/ files from the test
    runner's working tree are then copied into it so the guard under test
    is the one this branch actually adds (the worktree's HEAD copy would
    still be the pre-guard version otherwise).

    Detached because adding a worktree that checks out an existing branch
    is refused by git, and the test runner itself sits on a branch.
    """
    tmp = tempfile.mkdtemp(prefix="sy-guard-wt-")
    # `git worktree add` from inside a worktree is allowed when the new
    # worktree is detached — git refuses two worktrees on the same branch.
    res = subprocess.run(
        ["git", "worktree", "add", "--detach", tmp, "HEAD"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"git worktree add failed: rc={res.returncode} "
            f"stdout={res.stdout!r} stderr={res.stderr!r}")

    # The worktree checks out HEAD, which doesn't have the modifications
    # under test. Copy the modified scripts in so the guard is the one
    # this branch adds (otherwise the worktree's own apply.sh has no
    # guard and the test would silently pass for the wrong reason).
    for rel in ("scripts/apply.sh", "scripts/reload.sh"):
        src = os.path.join(ROOT, rel)
        dst = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # Read from the source so we don't depend on `cp` being allowed
        # by some future guard on `cp` itself.
        with open(src, "rb") as fh:
            data = fh.read()
        with open(dst, "wb") as fh:
            fh.write(data)
        # Preserve the executable bit so the scripts can be run as-is.
        st = os.stat(src)
        os.chmod(dst, st.st_mode)

    def cleanup():
        subprocess.run(
            ["git", "worktree", "remove", "--force", tmp],
            cwd=ROOT, capture_output=True, text=True,
        )
        subprocess.run(["rm", "-rf", tmp], capture_output=True, text=True)
    return tmp, cleanup


def _run_script(script_relpath, *args, cwd):
    """Run `bash <script_relpath> <args>` with cwd=cwd, return CompletedProcess."""
    return subprocess.run(
        ["bash", script_relpath, *args],
        cwd=cwd, capture_output=True, text=True,
    )


def test_apply_sh_refuses_to_run_from_a_worktree():
    tmp, cleanup = _spawn_worktree()
    try:
        # Sanity: the worktree's --git-dir and --git-common-dir really do
        # differ, otherwise the guard has nothing to fire on and this test
        # would silently pass on a regression that broke detection.
        gd = subprocess.run(
            ["git", "rev-parse", "--git-dir"], cwd=tmp,
            capture_output=True, text=True,
        ).stdout.strip()
        gcd = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=tmp,
            capture_output=True, text=True,
        ).stdout.strip()
        assert gd != gcd, f"worktree detection precondition failed: {gd!r} vs {gcd!r}"

        res = _run_script("scripts/apply.sh", "--dry-run", cwd=tmp)
        assert res.returncode == 2, \
            f"apply.sh from a worktree should exit 2, got {res.returncode}; " \
            f"stdout={res.stdout!r} stderr={res.stderr!r}"
        assert APPLY_REFUSAL in res.stderr, \
            f"apply.sh refused-message missing from stderr: {res.stderr!r}"
        # The guard fires BEFORE the arg parse, so an unknown flag would
        # never print its own message here.
        assert "unknown flag" not in res.stderr, res.stderr
    finally:
        cleanup()
    print("  apply.sh refuses (exit 2) when invoked from a git worktree")


def test_reload_sh_refuses_to_run_from_a_worktree():
    tmp, cleanup = _spawn_worktree()
    try:
        res = _run_script("scripts/reload.sh", cwd=tmp)
        assert res.returncode == 2, \
            f"reload.sh from a worktree should exit 2, got {res.returncode}; " \
            f"stdout={res.stdout!r} stderr={res.stderr!r}"
        assert RELOAD_REFUSAL in res.stderr, \
            f"reload.sh refused-message missing from stderr: {res.stderr!r}"
    finally:
        cleanup()
    print("  reload.sh refuses (exit 2) when invoked from a git worktree")


# Test (b): guard hook ----------------------------------------------------

GUARD = os.path.join(ROOT, ".claude", "hooks", "guard.sh")


def _feed_guard(cmd, tool_name="Bash", cwd=ROOT):
    """Pipe a JSON PreToolUse payload to guard.sh, return CompletedProcess.

    Default cwd=ROOT — the test runner itself typically sits in a worktree,
    so `is_worktree=1` is on for any worktree-only pattern test. Pass an
    explicit standalone-repo cwd to exercise the `is_worktree=0` branch.
    """
    payload = json.dumps({"tool_name": tool_name, "tool_input": {"command": cmd}})
    return subprocess.run(
        ["bash", GUARD], input=payload,
        cwd=cwd, capture_output=True, text=True,
    )


def _standalone_repo():
    """Make a fresh `git init` repo in a tempdir so `is_worktree=0`.

    Returns (path, cleanup_callable). Inside this directory
    `--git-dir == --git-common-dir`, so guard.sh treats invocations here
    as the main checkout (the worktree-only patterns short-circuit).
    """
    tmp = tempfile.mkdtemp(prefix="sy-guard-standalone-")
    init = subprocess.run(
        ["git", "init", "-q"], cwd=tmp, capture_output=True, text=True,
    )
    if init.returncode != 0:
        raise RuntimeError(
            f"git init failed in {tmp}: rc={init.returncode} "
            f"stdout={init.stdout!r} stderr={init.stderr!r}")

    def cleanup():
        subprocess.run(["rm", "-rf", tmp], capture_output=True, text=True)
    return tmp, cleanup


def test_guard_denies_dotenv_writes():
    deny_cmds = [
        # Original four — direct redirects, redirects via cp/tee/sed.
        "cp .env.example .env",
        "echo x > .env",
        "tee .env",
        "sed -i s/a/b/ .env",
        # Review-found gaps: rm/dd/install/rsync also write to .env.
        "rm .env",
        "dd if=/dev/zero of=.env",
        "install -m 600 /tmp/x .env",
        "rsync /tmp/x .env",
        "rsync -av .env.example .env",
    ]
    for cmd in deny_cmds:
        res = _feed_guard(cmd)
        assert res.returncode == 2, \
            f"guard should deny {cmd!r}, got rc={res.returncode}; " \
            f"stderr={res.stderr!r}"
        assert ".env" in res.stderr.lower(), \
            f"denial reason for {cmd!r} should mention .env, got {res.stderr!r}"
    print(f"  guard denies all {len(deny_cmds)} .env write shapes")


def test_guard_allows_unrelated_dotenv_paths():
    """False-positive-prevention from review: tokenize `.env` so the
    template and backup snapshots and unrelated paths ending in `.env`
    do NOT trip the deny.
    """
    allow_cmds = [
        "cat .env.example",
        "grep x .env.example",
        "tee .env.example",                # template, not the real one
        "mv .env.example /tmp/x",          # moving the template away
        "cp /tmp/test.env /tmp/x",         # unrelated path ending in .env
        "cp .env.example /tmp/test.env",   # template -> unrelated
        "ls /tmp/test.env/",               # directory suffix
        "echo .env.example",               # just print the literal
    ]
    for cmd in allow_cmds:
        res = _feed_guard(cmd)
        assert res.returncode == 0, \
            f"guard should allow {cmd!r}, got rc={res.returncode}; " \
            f"stderr={res.stderr!r}"
    print(f"  guard allows all {len(allow_cmds)} .env near-miss shapes")


def test_guard_denies_keychain_writes():
    deny_cmds = [
        "security add-generic-password -a me -s test -w foo",
        "security delete-keychain",
    ]
    for cmd in deny_cmds:
        res = _feed_guard(cmd)
        assert res.returncode == 2, \
            f"guard should deny {cmd!r}, got rc={res.returncode}; " \
            f"stderr={res.stderr!r}"
        assert "keychain" in res.stderr.lower() or "security" in res.stderr.lower(), \
            f"denial reason for {cmd!r} should mention keychain/security, " \
            f"got {res.stderr!r}"
    print(f"  guard denies all {len(deny_cmds)} security(1) shapes")


def test_guard_allows_benign_commands():
    allow_cmds = [
        "cat .env.example",
        "grep x .env.example",
        "scripts/sync-env.sh",
        "echo hi",
    ]
    for cmd in allow_cmds:
        res = _feed_guard(cmd)
        assert res.returncode == 0, \
            f"guard should allow {cmd!r}, got rc={res.returncode}; " \
            f"stderr={res.stderr!r}"
    print(f"  guard allows all {len(allow_cmds)} benign shapes")


def test_worktree_only_anchors_subcommand_boundary():
    """Finding 2: WORKTREE_ONLY is checked only in a worktree, and is
    anchored at end-of-token so legitimate subcommands like buildx,
    buildkit, upgrade, upload are NOT denied.

    The test runner itself is in a worktree, so `is_worktree=1` for the
    default cwd. Each must-DENY case actually fires the guard; the
    must-ALLOW cases just verify the regex doesn't false-positive.
    """
    allow_cmds = [
        "docker compose buildx create --use",     # not the build subcommand
        "docker compose buildx bake myapp",
        "docker compose buildkit inspect",
        "docker compose upgrade postgres",
        "docker compose upload nginx",
    ]
    for cmd in allow_cmds:
        res = _feed_guard(cmd)
        assert res.returncode == 0, \
            f"worktree guard should allow {cmd!r}, got rc={res.returncode}; " \
            f"stderr={res.stderr!r}"
    print(f"  worktree guard allows {len(allow_cmds)} anchored subcommand shapes")


def test_worktree_only_allows_git_merge_and_still_denies_docker_stop_list():
    """Issue acceptance test: `git merge` (bare, `--ff-only`, and `--no-ff`
    forms) must be ALLOWED from a worktree cwd, while the docker / login /
    logout stop list still fires. This guards the rule change that drops
    `git merge` from `WORKTREE_ONLY` — merging origin/<base> into the
    feature branch is a local op and the conflict-fix flow depends on it
    running inside the worktree.
    """
    for cmd in ["git merge", "git merge main --ff-only", "git merge --no-ff"]:
        res = _feed_guard(cmd)
        assert res.returncode == 0, \
            f"worktree guard must ALLOW {cmd!r} (issue: git merge is no " \
            f"longer worktree-only), got rc={res.returncode}; " \
            f"stderr={res.stderr!r}"
    for cmd in [
        "docker compose build web",
        "docker compose up -d",
        "docker login",
        "docker logout",
    ]:
        res = _feed_guard(cmd)
        assert res.returncode == 2, \
            f"worktree guard must still deny {cmd!r}, got rc={res.returncode}; " \
            f"stderr={res.stderr!r}"
        assert "refusing" in res.stderr.lower(), res.stderr
    print("  worktree guard allows git merge (with and without args) while still denying docker stop list")


def test_worktree_only_short_circuits_on_main_checkout():
    """is_worktree=0 must skip the worktree-only loop. Run the same deny
    commands from a fresh `git init` checkout and confirm rc=0.
    """
    standalone, cleanup = _standalone_repo()
    try:
        # Sanity: the standalone repo's --git-dir and --git-common-dir
        # really do agree.
        gd = subprocess.run(
            ["git", "rev-parse", "--git-dir"], cwd=standalone,
            capture_output=True, text=True,
        ).stdout.strip()
        gcd = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=standalone,
            capture_output=True, text=True,
        ).stdout.strip()
        assert gd == gcd, (
            f"standalone repo detection precondition failed: "
            f"{gd!r} vs {gcd!r}")
        for cmd in [
            "docker compose build web",
            "docker compose up -d",
        ]:
            res = _feed_guard(cmd, cwd=standalone)
            assert res.returncode == 0, (
                f"worktree-only rule {cmd!r} should NOT fire from main "
                f"checkout, got rc={res.returncode}; stderr={res.stderr!r}")
    finally:
        cleanup()
    print("  worktree-only loop short-circuits when is_worktree=0")


# Test (c): compose fail-closed ------------------------------------------

def test_compose_uses_required_project_substitution():
    with open(os.path.join(ROOT, "docker-compose.yml")) as fh:
        compose_text = fh.read()
    # The fail-closed form: docker compose errors when SWITCHYARD_PROJECT
    # is unset. The literal `:` after the var name is the required-subst
    # syntax — bash `${VAR:?msg}` substitutes VAR or, when VAR is unset
    # OR empty, exits with `msg`.
    assert "SWITCHYARD_PROJECT:?" in compose_text, \
        "docker-compose.yml should declare the project name with the " \
        "required-substitution form ${SWITCHYARD_PROJECT:?…}"
    # The hardcoded fallback name must be gone — a worktree with no .env
    # would otherwise pin to `switchyard` and address the live project.
    assert "name: switchyard\n" not in compose_text, \
        "docker-compose.yml must NOT hardcode `name: switchyard`"
    print("  docker-compose.yml fail-closed on missing SWITCHYARD_PROJECT")


def test_env_example_defines_switchyard_project():
    with open(os.path.join(ROOT, ".env.example")) as fh:
        env_text = fh.read()
    assert "SWITCHYARD_PROJECT=" in env_text, \
        ".env.example should declare SWITCHYARD_PROJECT=switchyard so " \
        "scripts/sync-env.sh appends it to the operator's .env"
    # The default must be `switchyard` — that's the compose project the
    # live stack was running under before this guard.
    assert "SWITCHYARD_PROJECT=switchyard" in env_text, \
        ".env.example should set SWITCHYARD_PROJECT=switchyard"
    print("  .env.example carries SWITCHYARD_PROJECT=switchyard")


# Test (c'): sync-env.sh newline-guard (issue #217) ----------------------
#
# Regression for the bug where `scripts/sync-env.sh` appended missing keys
# with `>>` against a target whose last line lacked a trailing newline,
# merging the new key into the existing credential
# (e.g. `LOCAL_API_KEY=sk-fooSWITCHYARD_PROJECT=switchyard`). Each test
# stages a sandbox tempdir — copy the script into `tmp/scripts/` with its
# exec bit, write `tmp/.env` and `tmp/.env.example` as bytes — so the
# script's `cd "$(dirname "$0")/.."` lands in a sandbox tree and the
# operator's real `.env` is never read or written. Belongs under (c) by
# subject (the `.env` ↔ `.env.example` hand-off) but split off so its
# regression focus reads at a glance.


def _stage_sync_env_sandbox(env_bytes, example_bytes):
    """Copy sync-env.sh into a tempdir tree and write .env + .env.example.

    Returns (sandbox, scripts_dir, env_path, cleanup). The sandbox is the
    script's effective working directory after its own `cd`; `cwd=sandbox`
    is passed to subprocess for clarity. Backup snapshots go under
    `sandbox/bu` via `SWITCHYARD_ENV_BACKUPS` (the test runner sets that
    env var on the subprocess so the real `$HOME/.switchyard/env-backups`
    is never touched).
    """
    tmp = tempfile.mkdtemp(prefix="sy-sync-env-")
    scripts_dir = os.path.join(tmp, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    src = os.path.join(ROOT, "scripts", "sync-env.sh")
    with open(src, "rb") as fh:
        data = fh.read()
    dst = os.path.join(scripts_dir, "sync-env.sh")
    with open(dst, "wb") as fh:
        fh.write(data)
    st = os.stat(src)
    os.chmod(dst, st.st_mode)
    env_path = os.path.join(tmp, ".env")
    with open(env_path, "wb") as fh:
        fh.write(env_bytes)
    with open(os.path.join(tmp, ".env.example"), "wb") as fh:
        fh.write(example_bytes)

    def cleanup():
        subprocess.run(["rm", "-rf", tmp], capture_output=True, text=True)
    return tmp, scripts_dir, env_path, cleanup


def test_sync_env_appends_key_on_a_new_line_when_target_lacks_trailing_newline():
    """Issue #217: when the target's last line has no trailing newline,
    the script must terminate the file with `\n` BEFORE its `>>` append,
    so the missing-key merge (`LOCAL_API_KEY=sk-fooSWITCHYARD_PROJECT=…`)
    cannot happen. The grep check is a real subprocess so a Python-only
    matcher cannot paper over the bug.
    """
    sandbox, scripts_dir, env_path, cleanup = _stage_sync_env_sandbox(
        env_bytes=b"LOCAL_API_KEY=sk-foo",
        example_bytes=b"LOCAL_API_KEY=sk-foo\nSWITCHYARD_PROJECT=switchyard\n",
    )
    try:
        res = subprocess.run(
            ["bash", os.path.join(scripts_dir, "sync-env.sh")],
            env={**os.environ,
                 "SWITCHYARD_ENV_BACKUPS": os.path.join(sandbox, "bu")},
            cwd=sandbox, capture_output=True, text=True,
        )
        assert res.returncode == 0, \
            f"sync-env.sh should exit 0, got {res.returncode}; " \
            f"stdout={res.stdout!r} stderr={res.stderr!r}"
        grep = subprocess.run(
            ["grep", "-q", "^SWITCHYARD_PROJECT=switchyard$", env_path],
            capture_output=True, text=True,
        )
        assert grep.returncode == 0, (
            f"appended SWITCHYARD_PROJECT must sit on its own line; "
            f"grep rc={grep.returncode}; "
            f"stdout={grep.stdout!r} stderr={grep.stderr!r}")
        with open(env_path, "rb") as fh:
            text = fh.read()
        # The credential must not have grown — i.e. the merge did NOT
        # happen. `b"LOCAL_API_KEY=sk-foo\n"` is the exact line; if the
        # merge had fired, the file would read
        # `b"LOCAL_API_KEY=sk-fooSWITCHYARD_PROJECT=switchyard\n"` and the
        # substring wouldn't be present.
        assert b"LOCAL_API_KEY=sk-foo\n" in text, (
            f"the original credential must remain on its own line, "
            f"got file: {text!r}")
        assert text.endswith(b"\n"), (
            f"the file must terminate with a newline after append, "
            f"got: {text!r}")
    finally:
        cleanup()
    print("  sync-env.sh appends missing keys on a new line (issue #217)")


def test_sync_env_does_not_double_append_across_runs_when_target_lacks_trailing_newline():
    """Idempotency contract for the corruption compounding (issue #217):

    On a target whose last line lacks a trailing newline, the script's
    `>>` append used to merge into the credential
    (`LOCAL_API_KEY=sk-fooSWITCHYARD_PROJECT=switchyard`). The duplicate
    check `grep -q "^KEY="` only matches a key at line start, so on the
    NEXT run the merged key is invisible to it and the script appends
    again — the corruption compounds on every run.

    The test runs the script TWICE in a row against a `.env` without a
    trailing newline and asserts SWITCHYARD_PROJECT appears exactly
    once on its own line at the end of the sequence:

      - With the fix: the first run appends a newline terminator and
        then `SWITCHYARD_PROJECT=switchyard` on its own line; the file
        now ends in `\\n` so the second run's `grep "^SWITCHYARD_PROJECT="`
        matches and the loop is a no-op.
      - Without the fix: the first run merges the key into the
        credential; the second run's `grep "^SWITCHYARD_PROJECT="`
        still finds nothing and appends a SECOND copy on a fresh line —
        count=2. So the assertion fires on the broken script.
    """
    sandbox, scripts_dir, env_path, cleanup = _stage_sync_env_sandbox(
        env_bytes=b"LOCAL_API_KEY=sk-foo",
        example_bytes=b"LOCAL_API_KEY=sk-foo\nSWITCHYARD_PROJECT=switchyard\n",
    )
    try:
        for _ in range(2):
            res = subprocess.run(
                ["bash", os.path.join(scripts_dir, "sync-env.sh")],
                env={**os.environ,
                     "SWITCHYARD_ENV_BACKUPS": os.path.join(sandbox, "bu")},
                cwd=sandbox, capture_output=True, text=True,
            )
            assert res.returncode == 0, \
                f"sync-env.sh should exit 0 on each run, got " \
                f"{res.returncode}; stdout={res.stdout!r} " \
                f"stderr={res.stderr!r}"
        with open(env_path, "rb") as fh:
            text = fh.read()
        assert text.count(b"SWITCHYARD_PROJECT=switchyard") == 1, (
            f"key must appear exactly once after two runs (no "
            f"double-append via the corrupted-line merge path); "
            f"got file: {text!r}")
        # And it must sit on its own line, not merged into the credential —
        # a Python-only `text.count` would still pass the count assertion
        # above if the bug merged a single copy; a real `grep` confirms
        # the structural shape.
        grep = subprocess.run(
            ["grep", "-q", "^SWITCHYARD_PROJECT=switchyard$", env_path],
            capture_output=True, text=True,
        )
        assert grep.returncode == 0, (
            f"SWITCHYARD_PROJECT must sit on its own line after two "
            f"runs; grep rc={grep.returncode}; "
            f"stdout={grep.stdout!r} stderr={grep.stderr!r}")
        # And the credential must not have grown — the side effect that
        # broke LOCAL_API_KEY authentication in the live incident.
        assert b"LOCAL_API_KEY=sk-foo\n" in text, (
            f"the original credential must remain on its own line; "
            f"got file: {text!r}")
    finally:
        cleanup()
    print("  sync-env.sh does not compound the corruption across runs (issue #217)")


# Test (d): settings files -----------------------------------------------

def test_claude_settings_json_has_deny_rules():
    with open(os.path.join(ROOT, ".claude", "settings.json")) as fh:
        settings = json.load(fh)
    deny = settings.get("permissions", {}).get("deny", [])
    expected = [
        # .env writes / reads.
        "Read(.env)",
        "Read(.env.backup.*)",
        "Edit(.env)",
        "Write(.env)",
        # Secrets dir AND the .env backup snapshots (sync-env.sh writes
        # those — they hold REAL keys; the .gitignore warns about them).
        "Read(secrets/**)",
        # Bash denies — space-glob form, not colon-form. Claude Code's
        # matcher requires the literal space; `Bash(docker compose build:*)`
        # would only match `docker compose build:foo`, not `docker compose
        # build foo`.
        "Bash(docker compose build *)",
        "Bash(docker compose up *)",
        "Bash(docker login *)",
        "Bash(docker logout *)",
        "Bash(security add-*)",
        "Bash(security delete-*)",
    ]
    missing = [e for e in expected if e not in deny]
    assert not missing, \
        f".claude/settings.json permissions.deny is missing: {missing}"
    # The colon-separator form must NOT be present — it's a footgun
    # (review finding 4).
    bad = [e for e in deny if "Bash(" in e and ":*" in e and not e.startswith("Bash(security")]
    assert not bad, \
        f".claude/settings.json permissions.deny must not use the colon " \
        f"separator form: {bad}"
    # Hook wiring — PreToolUse for Bash|Edit|Write pointing at the guard.
    pre = settings.get("hooks", {}).get("PreToolUse", [])
    assert pre, ".claude/settings.json must wire PreToolUse hooks"
    matchers = " | ".join(h.get("matcher", "") for h in pre)
    assert "Bash" in matchers and "Edit" in matchers and "Write" in matchers, \
        f"PreToolUse matcher should cover Bash|Edit|Write, got: {matchers!r}"
    cmds = []
    for h in pre:
        for hook in h.get("hooks", []):
            cmds.append(hook.get("command", ""))
    assert any(".claude/hooks/guard.sh" in c for c in cmds), \
        f"one PreToolUse hook must invoke .claude/hooks/guard.sh, got: {cmds!r}"
    print(f"  .claude/settings.json: {len(deny)} deny rules + guard hook wired")


def test_opencode_json_mirrors_deny_intent():
    with open(os.path.join(ROOT, ".opencode", "opencode.json")) as fh:
        cfg = json.load(fh)
    # Schema required — OpenCode hard-fails on invalid config.
    assert cfg.get("$schema") == "https://opencode.ai/config.json", \
        f".opencode/opencode.json must declare the schema URL, got {cfg.get('$schema')!r}"
    perm = cfg.get("permission", {})
    # read: .env, .env.backup.*, and secrets/** denied (broad allow '*' is
    # the default). The backup snapshots hold REAL keys — review finding 6.
    read_rules = perm.get("read", {})
    assert read_rules.get(".env") == "deny", \
        f"permission.read['.env'] should be 'deny', got {read_rules.get('.env')!r}"
    assert read_rules.get(".env.backup.*") == "deny", \
        f"permission.read['.env.backup.*'] should be 'deny', " \
        f"got {read_rules.get('.env.backup.*')!r}"
    assert read_rules.get("secrets/**") == "deny", \
        f"permission.read['secrets/**'] should be 'deny', got {read_rules.get('secrets/**')!r}"
    # edit: .env and .env.backup.* denied (the sync-env.sh snapshot pattern).
    edit_rules = perm.get("edit", {})
    assert edit_rules.get(".env") == "deny", \
        f"permission.edit['.env'] should be 'deny', got {edit_rules.get('.env')!r}"
    assert edit_rules.get(".env.backup.*") == "deny", \
        f"permission.edit['.env.backup.*'] should be 'deny', " \
        f"got {edit_rules.get('.env.backup.*')!r}"
    # bash: all seven deny patterns, anchored with trailing space (not
    # bare glob) so `docker compose buildx` / `buildkit` / `upgrade` /
    # `upload` are NOT denied — review finding 7. OpenCode's wildcard
    # matcher converts ` *` at pattern end to `( .*)?` so the bare
    # no-arg form (`docker compose build`) still matches.
    bash_rules = perm.get("bash", {})
    expected_bash_denies = [
        # trailing-space-anchored deny, fires when there are args
        "*docker compose build *",
        "*docker compose up *",
        "*docker login *",
        "*docker logout *",
        # no-trailing-space — matches the bare no-args form too
        "*docker compose build",
        "*docker compose up",
        "*docker login",
        "*docker logout",
        # security subcommands — already prefix-anchored on `add-`/`delete-`
        "*security add-*",
        "*security delete-*",
    ]
    for pat in expected_bash_denies:
        assert bash_rules.get(pat) == "deny", \
            f"permission.bash[{pat!r}] should be 'deny', got {bash_rules.get(pat)!r}"
    # The substring-glob form (`*docker compose build*` without the
    # trailing space) MUST be gone — it would false-positive on buildx.
    for bad in ("*docker compose build*", "*docker compose up*",
                "*docker login*", "*docker logout*"):
        if bad in bash_rules:
            raise AssertionError(
                f"permission.bash must not use the substring-glob form "
                f"{bad!r} (false-positives on buildx/buildkit/upgrade/"
                f"upload). Use the trailing-space variant instead.")
    # OpenCode evaluates the LAST matching rule — broad allow '*' must
    # come first, specific denies last, or the specific denies would
    # be shadowed. AND the broad rule must be 'allow' (not 'ask') so the
    # documented intent matches the file — review finding 5.
    bash_keys = list(bash_rules.keys())
    assert bash_keys[0] == "*", \
        f"permission.bash must declare '*' first (broad-allow), got: {bash_keys!r}"
    assert bash_rules["*"] == "allow", (
        f"permission.bash['*'] should be 'allow' so the broad rule matches "
        f"the rest of the config; review finding 5 (was 'ask', which made "
        f"every bash command trigger an operator prompt). "
        f"got {bash_rules['*']!r}")
    print("  .opencode/opencode.json: read/edit/bash deny mirrors the rules")


# Test (e): opencode wildcard runtime semantics --------------------------
#
# Cycle-2 review pointed out that the test only asserted the JSON shape
# but never exercised the runtime matcher. Both `.claude/hooks/guard.sh`
# and `.opencode/opencode.json` enforce the same rules, but with
# different engines (POSIX ERE vs opencode's Wildcard.matcher — which
# turns `*` into `.*` inside a `^...$` regex). The bare `*X` form
# (without trailing `*`) is a `^.*X$` "ends-with-X" match, NOT a
# substring match — and that's why the no-arg `docker compose build`
# is still caught while `docker compose buildx` is correctly allowed.
# Re-implementing the matcher here in Python keeps a trip wire for a
# regression that drops the trailing-space anchor.


def _oc_wildcard_match(str_, pattern):
    """Re-implementation of opencode's Wildcard.match in Python, lifted
    from packages/opencode/src/util/wildcard.ts:

      function match(str, pattern) {
        if (str) str = str.replaceAll("\\\\", "/")
        if (pattern) pattern = pattern.replaceAll("\\\\", "/")
        let escaped = pattern
          .replace(/[.+^${}()|[\\]\\\\]/g, "\\\\$&")
          .replace(/\\*/g, ".*")
          .replace(/\\?/g, ".")
        if (escaped.endsWith(" .*")) {
          escaped = escaped.slice(0, -3) + "( .*)?"
        }
        const flags = process.platform === "win32" ? "si" : "s"
        return new RegExp("^" + escaped + "$", flags).test(str)
      }

    Pure-Python equivalent — same anchored `^...$` semantics, same
    trailing-` *` -> `( .*)?` optimization. If opencode changes the
    semantics, this is the place to update.
    """
    if str_:
        str_ = str_.replace("\\", "/")
    if pattern:
        pattern = pattern.replace("\\", "/")
    # escape regex special chars
    escaped = re.sub(r"[.+^${}()|\[\]\\]", r"\\&", pattern)
    escaped = escaped.replace("*", ".*").replace("?", ".")
    if escaped.endswith(" .*"):
        escaped = escaped[:-3] + "( .*)?"
    return bool(re.fullmatch(escaped, str_, re.DOTALL))


def _oc_evaluate(rules, command):
    """Last-match-wins evaluation, like opencode's `findLast`."""
    for pat in reversed(list(rules.keys())):
        if _oc_wildcard_match(command, pat):
            return rules[pat]
    return "ask"


def test_opencode_wildcard_no_false_positives_on_buildx_buildkit():
    """Cycle-2 review concern: does the bare `*docker compose build` rule
    (kept to catch the no-args `docker compose build`) false-positive on
    `docker compose buildx` / `buildkit` / `upgrade` / `upload`?

    Empirically: NO — opencode's matcher wraps the pattern in `^...$`,
    so `*docker compose build` becomes `^.*docker compose build$` (an
    "ends-with" match), not a substring. This test reproduces the
    matcher in Python and exercises the real rule list against the
    false-positive inputs the reviewer listed.
    """
    with open(os.path.join(ROOT, ".opencode", "opencode.json")) as fh:
        cfg = json.load(fh)
    rules = cfg["permission"]["bash"]

    # MUST be allowed (the cycle-2 false-positive concern)
    allow_cmds = [
        # Cycle 1/2 reviewer-flagged false-positives:
        "docker compose buildx create --use",
        "docker compose buildx bake myapp",
        "docker compose buildkit inspect",
        "docker compose upgrade postgres",
        "docker compose upload nginx",
        # Adjacent commands that ARE real docker compose invocations but
        # don't match the build / up / login / logout stop list. These
        # would each be a regression trip wire if someone widens the
        # rules too far:
        "docker compose ps",
        "docker compose logs web",
        "docker compose down",
        "docker compose config",
        # Issue: `git merge` is no longer a stop-list command — it must
        # fall through to the broad `*` allow rule, both bare and with
        # args (the conflict-fix flow runs inside the worktree).
        "git merge",
        "git merge main --ff-only",
    ]
    for cmd in allow_cmds:
        action = _oc_evaluate(rules, cmd)
        assert action == "allow", (
            f"opencode wildcard matcher must allow {cmd!r} "
            f"(regression in the anchoring / rule shape). "
            f"got action={action!r}")

    # MUST be denied — sanity check the real stop list.
    deny_cmds = [
        "docker compose build web",
        "docker compose build",
        "docker compose up -d",
        "docker compose up",
        "docker login",
        "docker login foo",
        "docker logout",
        "security add-generic-password -a me -s test -w foo",
        "security delete-keychain",
    ]
    for cmd in deny_cmds:
        action = _oc_evaluate(rules, cmd)
        assert action == "deny", (
            f"opencode wildcard matcher must deny {cmd!r}; "
            f"got action={action!r}")

    print(f"  opencode wildcard: {len(allow_cmds)} allow + {len(deny_cmds)} deny cases all match")


def test_guard_worktree_only_message_uses_human_readable_label():
    """Cycle-2 NIT 2: the worktree-only error message should print the
    human-readable command (`docker compose build`), not the regex
    literal (`(^|[[:space:]])docker[[:space:]]+compose[[:space:]]+build([[:space:]]|$)`).
    """
    cmd = "docker compose build web"
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}})
    res = subprocess.run(
        ["bash", GUARD], input=payload,
        cwd=ROOT, capture_output=True, text=True,
    )
    assert res.returncode == 2, \
        f"worktree-only guard should refuse {cmd!r} from a worktree, " \
        f"got rc={res.returncode}; stderr={res.stderr!r}"
    # The label must be the human-readable form...
    assert "'docker compose build'" in res.stderr, (
        f"worktree-only error must print the human-readable label "
        f"'docker compose build', got: {res.stderr!r}")
    # ...and NOT print the regex literal.
    assert "[[:space:]]" not in res.stderr, (
        f"worktree-only error must NOT print the regex literal "
        f"'[[:space:]]'; got: {res.stderr!r}")
    print("  worktree-only error: human-readable label, not regex literal")


# Test (f): compose↔plans parity for the portal's probe env vars --------
#
# Regression for issue where the portal's probe code reads
# `os.environ/NAME` from plans.yaml's `probe.headers` (e.g. `Authorization:
# os.environ/GLM_API_KEY` on the glm plan) but the variable is missing
# from `services.portal.environment` in docker-compose.yml -- the probe
# fails every poll with the misleading "quota probe needs a fresh session
# cookie" wording, and the operator cannot tell the variable is missing
# from .env. Cookie-kind plans store their credential in Redis (not the
# portal's env), and plan-level `api_base`/`api_key` are gateway-only, so
# only the `os.environ/NAME` references actually under a plan's
# `probe.headers` block belong in `services.portal.environment`.
#
# A future plan that adds a new probe header mapping must surface the new
# variable here too -- this test fires on the moment the drift appears
# rather than waiting for the live probe to fail.


def _probe_header_env_vars(plans_text: str) -> set[str]:
    """Walk plans.yaml text and collect every NAME in `os.environ/NAME`
    found under any plan's `probe.headers` mapping.

    Other surfaces carry credential references but do NOT bind to the
    portal's environment:

      * `probe.headers` -- the only block the portal reads; its values
        are rendered as `Authorization: $NAME` against `os.environ`,
        so a missing `services.portal.environment.NAME` makes the probe
        fail every poll with the wrong wording.
      * `api_base` / `api_key` at plan top-level -- gateway-only.
      * `probe.cookie` storage -- Redis-backed, not an env var.

    A real YAML walk (PyYAML safe_load) keeps the assertion honest: a
    regex on raw text would false-positive on a comment that names an
    env var in prose, and miss a multi-line `headers:` block whose values
    sit on indented lines.
    """
    import yaml
    plans = yaml.safe_load(plans_text) or {}
    names: set[str] = set()
    for plan in (plans.get("plans") or {}).values():
        if not isinstance(plan, dict):
            continue
        probe = plan.get("probe")
        if not isinstance(probe, dict):
            continue
        headers = probe.get("headers")
        if not isinstance(headers, dict):
            continue
        for value in headers.values():
            if not isinstance(value, str):
                continue
            prefix = "os.environ/"
            if not value.startswith(prefix):
                continue
            name = value[len(prefix):]
            if name:
                names.add(name)
    return names


def test_compose_portal_environment_covers_every_probe_header_env_var():
    """Every `os.environ/NAME` under any plan's `probe.headers` is a key of
    `services.portal.environment` in docker-compose.yml.

    Failure message names the missing variable -- the test is the
    canonical "you added a probe header but forgot the portal env var"
    trip wire.
    """
    with open(os.path.join(ROOT, "docker-compose.yml")) as fh:
        compose_text = fh.read()
    import yaml
    compose = yaml.safe_load(compose_text) or {}
    portal_env = ((compose.get("services") or {})
                  .get("portal", {}).get("environment") or {})
    portal_keys = set(portal_env.keys())

    with open(PLANS_EXAMPLE) as fh:
        plans_text = fh.read()
    needed = _probe_header_env_vars(plans_text)

    missing = sorted(needed - portal_keys)
    assert not missing, (
        f"docker-compose.yml services.portal.environment is missing "
        f"variable(s) referenced by a plan's probe.headers in "
        f"{PLANS_EXAMPLE!r}: {missing}. Each probe header "
        f"`os.environ/NAME` reads through the portal container, so the "
        f"name must be present in services.portal.environment or the "
        f"probe fails every poll with the cookie-reauth wording "
        f"regardless of probe.kind. Cookie-kind plans store their "
        f"credential in Redis, not the environment, so they do not "
        f"appear here."
    )
    # And the assertion's direction holds: at least one variable is
    # currently required, so a future plans.yaml that drops every
    # probe.headers reference cannot silently zero out the test.
    assert needed, (
        "expected at least one probe header env var in "
        f"{PLANS_EXAMPLE!r} (e.g. GLM_API_KEY on the glm plan); "
        "if plans.yaml really has none, the test needs an explicit "
        "fixture update rather than a silent green."
    )
    print(f"  portal.environment covers {len(needed)} probe-header env var(s): "
          f"{sorted(needed)}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"{len(fns)} tests passed")
