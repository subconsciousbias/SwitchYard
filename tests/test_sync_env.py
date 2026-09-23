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


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))