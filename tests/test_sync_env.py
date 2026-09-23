"""scripts/sync-env.sh: the four input shapes that used to corrupt or miss.

Issue #163 reopened on top of the #217 trailing-newline fix (commit c497b4b):
the appender could still

  (1) merge into the last line when the example ended in a non-newline byte —
      already fixed by the tail -c1 / od terminator check;
  (2) skip the last example key when its line had no trailing newline, so a
      `.env.example` written on Windows or by an editor that swallows the
      final `\n` quietly lost its last key;
  (3) duplicate an existing `export KEY=real` or `KEY = real` because the
      presence check only matched `^KEY=`, so an example `KEY=` appended
      a second empty `KEY=` next to a real value;
  (4) leave a stray `\r` on the appended line for CRLF example files, and
      miss a `KEY=\r` empty value because `\r` is not in POSIX ERE.

The fixes live in `scripts/sync-env.sh`; this file pins them so a regression
shows up in the offline suite instead of in production. Every test builds a
fresh `tempfile.TemporaryDirectory()` tree, copies the real script from
`scripts/sync-env.sh`, and runs it with `SWITCHYARD_ENV_BACKUPS` and `HOME`
redirected at the tempdir so nothing touches the operator's real `.env` or
`~/.switchyard`.

Issue #118 added a fifth contract on top of the four: when the script
creates a FRESH `.env` from the example, it MUST rewrite the
`LITELLM_MASTER_KEY=sk-switchyard-change-me` line with a random `sk-…` key.
Without that rewrite the gateway's startup self-check refuses to boot with
the published placeholder, so a fresh clone + `scripts/sync-env.sh` would
land the operator on a CRITICAL exit at the very first `docker compose up
-d`. The rewrite happens only in the fresh-create branch, after `cp` and
before the script exits, so it cannot overwrite an existing value — by
definition no `.env` existed before that branch ran. The two
`test_fresh_create_…` tests pin that contract; the
`test_existing_env_with_real_master_key_is_preserved_byte_for_byte` test
pins the other half: an `.env` that already defines a real
`LITELLM_MASTER_KEY=` line must NOT be touched, even if the script's
fresh-create path would otherwise have rewritten it.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPT_SRC = ROOT / "scripts" / "sync-env.sh"


def _stage(tmp: Path) -> Path:
    """Copy the real sync-env.sh into <tmp>/scripts/ and return that path.

    Reads the source bytes (rather than `shutil.copy`) so the test does not
    depend on `cp` being allowed by any future guard. The script's own `cd
    "$(dirname "$0")/.."` then resolves `<tmp>` as its working dir.
    """
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    dst = scripts_dir / "sync-env.sh"
    dst.write_bytes(SCRIPT_SRC.read_bytes())
    dst.chmod(0o755)
    return dst


def _run(tmp: Path, script: Path):
    """Run the staged script with HOME + backups redirected at <tmp>.

    Returns the CompletedProcess so callers can assert on stdout / returncode.
    """
    env = {
        **os.environ,
        "HOME": str(tmp),
        "SWITCHYARD_ENV_BACKUPS": str(tmp / "env-backups"),
    }
    return subprocess.run(
        ["bash", str(script)],
        cwd=tmp, env=env, capture_output=True, text=True,
    )


def test_issue_217_target_without_trailing_newline_gets_one_appended():
    """Regression for issue #217: a .env that ends without '\\n' must gain a
    newline before any append, so the new key does not merge into the last
    existing line and corrupt the secret.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = _stage(tmp)
        (tmp / ".env").write_bytes(b"A_KEY=real-secret-a")           # no trailing \n
        (tmp / ".env.example").write_bytes(b"A_KEY=\nB_KEY=\n")
        before = (tmp / ".env").read_bytes()

        result = _run(tmp, script)
        assert result.returncode == 0, result.stderr

        after = (tmp / ".env").read_bytes()
        assert after == b"A_KEY=real-secret-a\nB_KEY=\n", (
            f"target was rewritten; expected the original bytes preserved "
            f"byte-for-byte plus a newline terminator and the appended key. "
            f"got {after!r}")
        assert after.startswith(before), (
            f"original {before!r} not preserved as a prefix; got {after!r}")
        assert b"B_KEY=" in after
        assert b"real-secret-a" in after
        print("  + A_KEY=real-secret-a preserved; B_KEY appended after a \\n")


def test_example_last_line_without_newline_still_contributes_its_key():
    """An example whose last line lacks a final '\\n' must still be read.

    `while read` drops the last line of a file that does not end in '\\n';
    pairing it with `|| [ -n "$line" ]` re-enters the body with the residual
    text. Without the fix, the operator's last `.env.example` key never
    makes it into `.env`.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = _stage(tmp)
        (tmp / ".env").write_bytes(b"A_KEY=val\n")
        (tmp / ".env.example").write_bytes(b"A_KEY=\nZ_KEY=")        # no final \n

        result = _run(tmp, script)
        assert result.returncode == 0, result.stderr

        after = (tmp / ".env").read_bytes()
        assert b"Z_KEY=" in after, (
            f"the last example line (no trailing newline) was dropped; "
            f"got {after!r}")
        print("  Z_KEY= present in target even without a trailing newline")


def test_export_key_in_target_is_recognised_no_empty_duplicate():
    """`export KEY=real` in .env must satisfy the presence check for example
    `KEY=`. The previous `grep -q "^KEY="` missed the `export ` prefix and
    appended a second empty `KEY=` next to the real value, leaving the
    shell to pick one and the operator to wonder which.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = _stage(tmp)
        (tmp / ".env").write_bytes(b"export KEY=real\n")
        (tmp / ".env.example").write_bytes(b"KEY=\n")

        result = _run(tmp, script)
        assert result.returncode == 0, result.stderr

        after = (tmp / ".env").read_bytes()
        assert after == b"export KEY=real\n", (
            f"export KEY=real should have been left alone; got {after!r}")
        assert after.count(b"KEY=") == 1, (
            f"empty KEY= duplicate appended; got {after!r}")
        assert b"export KEY=real" in after
        assert "nothing changed" in result.stdout, result.stdout
        print("  export KEY=real in target blocks the empty duplicate")


def test_spaced_key_in_target_is_recognised_no_empty_duplicate():
    """`KEY = real` in .env (space around the `=`) must also satisfy the
    presence check. The grep pattern's `[[:space:]]*` around `=` matches
    this; the bug it would have caused is the same empty-duplicate append
    as test 3a.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = _stage(tmp)
        (tmp / ".env").write_bytes(b"KEY = real\n")
        (tmp / ".env.example").write_bytes(b"KEY=\n")

        result = _run(tmp, script)
        assert result.returncode == 0, result.stderr

        after = (tmp / ".env").read_bytes()
        assert after == b"KEY = real\n", (
            f"KEY = real should have been left alone; got {after!r}")
        assert after.count(b"KEY") == 1, (
            f"empty KEY= duplicate appended; got {after!r}")
        assert b"KEY = real" in after
        assert "nothing changed" in result.stdout, result.stdout
        print("  KEY = real in target blocks the empty duplicate")


def test_crlf_example_line_appends_clean_and_empty_value_is_reported():
    """A CRLF-terminated example line must have its '\\r' stripped before the
    appended copy lands in .env, and a pre-existing `KEY=\\r` empty value
    must surface in the 'still empty' report (not slip through because `\\r`
    is not in POSIX ERE and GNU grep treats it as a literal `r`).
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = _stage(tmp)
        # CRLF on the example line + a pre-existing KEY=\r in the target.
        (tmp / ".env").write_bytes(b"KEY=\r\nA_KEY=val\n")
        (tmp / ".env.example").write_bytes(b"KEY=\r\nW_KEY=\r\n")

        result = _run(tmp, script)
        assert result.returncode == 0, result.stderr

        after = (tmp / ".env").read_bytes()
        # KEY was already present (matched by the widened pattern even with
        # the \r, because the tr -d '\r' pre-pass normalises it). W_KEY is
        # new and must be appended without a trailing \r.
        assert b"W_KEY=\r" not in after, (
            f"\\r leaked into the appended W_KEY= line; got {after!r}")
        assert b"W_KEY=\n" in after, (
            f"appended W_KEY= line missing or malformed; got {after!r}")
        assert b"KEY=" in after, (
            f"original KEY= line lost; got {after!r}")
        # And the empty-value report must mention KEY (and not have \r in
        # the printed name).
        assert "still empty" in result.stdout, result.stdout
        assert "KEY" in result.stdout, result.stdout
        print("  CRLF stripped on append; KEY=\\r reported as empty")


def test_same_second_no_op_run_keeps_the_first_runs_backup():
    """A no-op run's `rm -f "$backup"` must only ever delete the snapshot
    IT created that second, not another run's pre-change snapshot. The fix
    adds `$$` to the backup name so two runs in the same wall-clock second
    land on different files.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = _stage(tmp)
        (tmp / ".env").write_bytes(b"A_KEY=val\n")
        (tmp / ".env.example").write_bytes(b"A_KEY=\nB_KEY=\n")

        backups = tmp / "env-backups"
        backups.mkdir(parents=True, exist_ok=True)

        # Run 1: appends B_KEY, leaves its backup behind.
        r1 = _run(tmp, script)
        assert r1.returncode == 0, r1.stderr
        assert "added 1 key" in r1.stdout, r1.stdout
        run1_backups = sorted(p.name for p in backups.iterdir())
        assert len(run1_backups) == 1, run1_backups

        # Run 2: same second, no new keys, must NOT delete run 1's backup.
        r2 = _run(tmp, script)
        assert r2.returncode == 0, r2.stderr
        assert "nothing changed" in r2.stdout, r2.stdout
        surviving = sorted(p.name for p in backups.iterdir())
        assert surviving == run1_backups, (
            f"run 1's backup disappeared: before={run1_backups} "
            f"after={surviving}")
        print("  same-second no-op run preserved the earlier snapshot")


# ---------------------------------------------------------------- issue #118
#
# The fresh-create branch of sync-env.sh rewrites the copied
# `LITELLM_MASTER_KEY=sk-switchyard-change-me` line with a random sk- key
# so the gateway's startup self-check does not CRITICAL on the published
# placeholder. The rewrite MUST only happen on a fresh create — an existing
# `.env` that already defines `LITELLM_MASTER_KEY=` is operator territory
# and the script's "never overwrite an existing value" rule applies.

def test_fresh_create_rewrites_master_key_with_random_value():
    """When `.env` does not exist, sync-env.sh creates one from
    `.env.example` and rewrites `LITELLM_MASTER_KEY=sk-switchyard-change-me`
    with a random `sk-…` value. The published placeholder must NOT survive
    the cp — otherwise the gateway's self-check CRITICALs on the very next
    `docker compose up -d` (issue #118).

    The random key is 50 chars (`sk-` plus 24 random bytes hex-encoded to
    48 chars), well over the 32-char threshold the selfcheck enforces.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = _stage(tmp)
        # Minimal example: a single LITELLM_MASTER_KEY line, exactly like
        # the one shipped in `.env.example` before the rest of the file
        # was added.
        (tmp / ".env.example").write_bytes(
            b"LITELLM_MASTER_KEY=sk-switchyard-change-me\nA_KEY=\n")

        result = _run(tmp, script)
        assert result.returncode == 0, result.stderr

        env = (tmp / ".env").read_bytes()
        assert b"LITELLM_MASTER_KEY=sk-switchyard-change-me" not in env, (
            f"the published placeholder survived the cp; got {env!r}. "
            f"The fresh-create branch must rewrite it with a random key "
            f"so the gateway's selfcheck does not refuse to boot.")
        # Extract the rewritten value and check its shape.
        line = next(
            (b for b in env.splitlines()
             if b.startswith(b"LITELLM_MASTER_KEY=")),
            None)
        assert line is not None, f"no LITELLM_MASTER_KEY line in {env!r}"
        value = line.split(b"=", 1)[1]
        assert value.startswith(b"sk-"), (
            f"rewritten key does not start with the sk- marker: {value!r}")
        assert len(value) >= 32, (
            f"rewritten key is shorter than the 32-char selfcheck "
            f"threshold ({len(value)} chars): {value!r}")
        assert b"change-me" not in value, (
            f"rewritten key still contains the change-me marker: {value!r}")
        print(f"  fresh .env has LITELLM_MASTER_KEY=<{len(value)}-char random "
              f"sk-…> (placeholder rewritten)")


def test_fresh_create_emits_note_about_generated_key():
    """The stdout message on a fresh create must mention that the master
    key was generated, so an operator who reads the output before
    `docker compose up -d` knows the value is fresh and not the published
    one. The wording is pinned lightly (substring match) so a cosmetic
    tweak doesn't break tests.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = _stage(tmp)
        (tmp / ".env.example").write_bytes(
            b"LITELLM_MASTER_KEY=sk-switchyard-change-me\n")

        result = _run(tmp, script)
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "LITELLM_MASTER_KEY" in out, (
            f"fresh-create output does not mention the generated key; "
            f"an operator who only reads the message would not know the "
            f"value was rotated. got: {out!r}")
        assert "random" in out.lower(), (
            f"fresh-create output does not say 'random' (or equivalent) — "
            f"got: {out!r}")
        print("  fresh-create message announces the random key rotation")


def test_existing_env_with_real_master_key_is_preserved_byte_for_byte():
    """An `.env` that ALREADY exists with a real `LITELLM_MASTER_KEY=…`
    value must not be touched by the rewrite path — the rewrite is
    structurally gated to the fresh-create branch (after `cp` and before
    `exit 0`), so the operator's chosen value survives every subsequent
    run.

    The script is invoked twice: the first run creates `.env` (with a
    random key), the operator then replaces that key with their own, and
    the second run must leave the operator's key alone. The two halves
    are pinned together so a regression that, say, runs the rewrite on
    every invocation would fail this test immediately.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = _stage(tmp)
        (tmp / ".env.example").write_bytes(
            b"LITELLM_MASTER_KEY=sk-switchyard-change-me\nA_KEY=\n")

        # First run: creates .env, rewrites the placeholder with a random key.
        r1 = _run(tmp, script)
        assert r1.returncode == 0, r1.stderr
        env_path = tmp / ".env"

        # Operator replaces the random key with their own chosen value.
        operator_key = b"sk-operator-chosen-secret-very-long-12345"
        original_lines = env_path.read_bytes().splitlines(keepends=True)
        original_lines[0] = b"LITELLM_MASTER_KEY=" + operator_key + b"\n"
        env_path.write_bytes(b"".join(original_lines))
        before = env_path.read_bytes()
        assert operator_key in before

        # Second run: .env already exists, no rewrite should happen.
        r2 = _run(tmp, script)
        assert r2.returncode == 0, r2.stderr

        after = env_path.read_bytes()
        assert after == before, (
            f"operator's LITELLM_MASTER_KEY was rewritten on a no-op run: "
            f"before={before!r} after={after!r}")
        assert operator_key in after, (
            f"operator's key lost: got {after!r}")
        assert b"sk-switchyard-change-me" not in after, (
            f"the published placeholder leaked back into the operator's "
            f".env: got {after!r}")
        print("  existing .env with operator master key: preserved byte-for-byte")


def test_two_fresh_creates_produce_different_master_keys():
    """Two consecutive fresh creates must yield two DIFFERENT random keys.
    If the generator was replaced with a constant (a refactor regression
    worth catching) the second run would produce the same key as the
    first; the test would fail and the operator would notice a single
    key shared across every fresh checkout.
    """
    keys: list[bytes] = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            script = _stage(tmp)
            (tmp / ".env.example").write_bytes(
                b"LITELLM_MASTER_KEY=sk-switchyard-change-me\n")

            result = _run(tmp, script)
            assert result.returncode == 0, result.stderr

            env = (tmp / ".env").read_bytes()
            line = next(
                b for b in env.splitlines()
                if b.startswith(b"LITELLM_MASTER_KEY="))
            keys.append(line.split(b"=", 1)[1])

    assert keys[0] != keys[1], (
        f"two fresh creates produced the same master key {keys[0]!r}; "
        f"the generator is no longer random.")
    print(f"  two fresh creates produced two distinct sk-… keys "
          f"({len(keys[0])} and {len(keys[1])} chars)")


def _run_with_stripped_path(tmp: Path, script: Path, *banned: str):
    """Run the staged script with a curated PATH that contains everything
    the script needs EXCEPT the names in `banned`. Returns the
    CompletedProcess.

    Used by the no-openssl / no-python3 regression test below: the bug
    it pins only appears when neither generator is on PATH, which the
    standard _run() helper (which inherits os.environ['PATH']) cannot
    reproduce on a developer machine that has openssl installed.

    Implementation: build a fresh `<tmp>/bin/` directory that symlinks
    every name the host PATH exposes — except anything in `banned` — into
    one place, then run the script with PATH=<tmp>/bin and nothing else.
    POSIX utilities like `[`, `echo`, `cat`, `test`, `printf`, etc. that
    live under `/bin` on some distros are pulled in by walking `/bin`
    the same way (so a `/bin/openssl` is dropped alongside an
    `/usr/bin/openssl` on hosts that symlink the two together, which is
    the common Debian/Ubuntu layout).
    """
    raw_path = os.environ.get("PATH", "")
    banned_set = set(banned)
    curated_bin = tmp / "bin"
    curated_bin.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    # Walk the PATH first so user-installed tools take precedence over
    # the system /bin that we sweep afterwards; this matches the host
    # lookup order the operator's real shell uses.
    source_dirs = list(raw_path.split(os.pathsep))
    # Always also walk /bin so POSIX utilities resolve on hosts whose
    # PATH omits /bin (some macOS setups) or whose /bin is symlinked to
    # /usr/bin (Debian/Ubuntu) — both layouts need a separate sweep
    # because the banned-name filter has to apply uniformly.
    source_dirs.append("/bin")
    for d in source_dirs:
        if not d:
            continue
        try:
            entries = sorted(Path(d).iterdir())
        except (FileNotFoundError, PermissionError, NotADirectoryError):
            continue
        for entry in entries:
            name = entry.name
            if name in banned_set or name in seen:
                continue
            try:
                resolved = entry.resolve(strict=True)
            except (FileNotFoundError, RuntimeError):
                continue
            if not resolved.is_file() or not os.access(resolved, os.X_OK):
                continue
            link = curated_bin / name
            try:
                link.symlink_to(resolved)
            except FileExistsError:
                continue
            seen.add(name)
    new_path = str(curated_bin)
    env = {
        **os.environ,
        "HOME": str(tmp),
        "SWITCHYARD_ENV_BACKUPS": str(tmp / "env-backups"),
        "PATH": new_path,
    }
    return subprocess.run(
        ["bash", str(script)],
        cwd=tmp, env=env, capture_output=True, text=True,
    )


def test_fresh_create_without_openssl_or_python3_exits_nonzero_with_message():
    """PR #118 review blocker: when neither `openssl` nor `python3` is on
    PATH, the script must NOT silently write `LITELLM_MASTER_KEY=sk-` to
    the freshly-created .env (the prefix outside the subshell, with the
    subshell swallowing its own non-zero exit via `2>/dev/null`) and exit
    0. It must instead exit non-zero with the loud
    "no openssl or python3 on PATH" message so the operator knows to fix
    the host before `docker compose up -d`. This is the contract pinned
    by the PR description's "or refuses to start" half: a fresh clone
    that lands the operator on `LITELLM_MASTER_KEY=sk-` would still CRITICAL
    the gateway on the next start, so the script should fail here and
    tell the operator why, not hand them a credential the gateway will
    refuse anyway.

    The fix gates the python3 fallback on `command -v python3`, the same
    shape as the openssl gate one branch up; this test reproduces the
    pre-fix behaviour end-to-end and pins the post-fix contract.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = _stage(tmp)
        (tmp / ".env.example").write_bytes(
            b"LITELLM_MASTER_KEY=sk-switchyard-change-me\nA_KEY=\n")

        result = _run_with_stripped_path(tmp, script, "openssl", "python3")

        # The PR's loud-message contract: non-zero exit, .env is left
        # with the published placeholder (no rewrite happened), stderr
        # names both missing tools.
        assert result.returncode == 1, (
            f"fresh create with neither openssl nor python3 must exit 1 "
            f"(the loud-message branch); got exit {result.returncode}. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}")

        env_path = tmp / ".env"
        assert env_path.exists(), (
            f"the script must still create .env on the loud-message "
            f"path (cp already ran before the no-generator check), so "
            f"the operator has the file to inspect; got missing "
            f"{env_path}")
        contents = env_path.read_bytes()
        assert b"LITELLM_MASTER_KEY=sk-switchyard-change-me" in contents, (
            f".env must retain the published placeholder when the "
            f"rewrite is skipped; rewriting it to the literal 'sk-' is "
            f"the bug. got {contents!r}")
        assert b"LITELLM_MASTER_KEY=sk-\n" not in contents, (
            f"the buggy rewrite produced 'sk-' alone; that line is the "
            f"regression this test pins. got {contents!r}")
        assert b"LITELLM_MASTER_KEY=sk-\r\n" not in contents, (
            f"the buggy rewrite produced 'sk-' alone; that line is the "
            f"regression this test pins. got {contents!r}")

        assert "no openssl or python3" in result.stderr, (
            f"the loud error message must name both missing tools; "
            f"got stderr={result.stderr!r}")
        assert "sk-switchyard-change-me" in result.stderr, (
            f"the loud error message must tell the operator what to "
            f"replace; got stderr={result.stderr!r}")
        print("  no-openssl/no-python3 fresh create: exit 1, .env keeps "
              "the placeholder, stderr names both missing tools")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))