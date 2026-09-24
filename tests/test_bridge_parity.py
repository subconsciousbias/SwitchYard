"""Parity tests between cli_bridge (text path) and mcp_bridge (tool path).

Both bridges build argv against the same vendor CLI behind the same shared
profile (cli_bridge's PROFILE / PROFILES); mcp_bridge loads cli_bridge by file
path so a fix in one must be exercised in the other, not duplicated. This
file is the lock-step check for issue #137 -- CLI_EXTRA_ARGS tail-appended,
the codex `model_instructions_file=` always present (with the
`/app/harness/codex-instructions.md` baked default when the caller sent no
system, the workdir file when one is supplied), and the claude replace-mode
exclude flag carried on both paths.

The provider is swapped on both modules' globals (cli_bridge_server.PROFILE
and mcp_bridge_server.cli_bridge.PROFILE are kept in step within a single
test) the same way the existing codex-mcp test does in
tests/test_mcp_bridge.py:3998. Per-test save/restore keeps the suite
sequential and lets every test reason from a known starting state.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

# Two sidecars name their entry point `server.py`. Loading by file path
# under an explicit name keeps them distinct (see tests/_modules.py), and
# the mcp_bridge module ALSO loads cli_bridge by file path internally --
# `mcp_bridge_server.cli_bridge` is a process-local copy of cli_bridge, NOT
# the `cli_bridge_server` module below. We swap the provider on all three
# globals (cli_bridge_server, mcp_bridge_server.PROFILE, and
# mcp_bridge_server.cli_bridge) inside every test so the assertions see the
# same profile on every side.
os.environ["PROVIDER"] = "claude"
os.environ.setdefault("SWITCHYARD_PLAN", "claude-max")
os.environ["MCP_HOST_MIRROR_CHECK"] = "off"

from plans_path import plans_path  # noqa: E402

os.environ["SWITCHYARD_PLANS"] = plans_path()
os.environ.setdefault("SIDECAR_PORT", "8081")

from _modules import load  # noqa: E402

cli_bridge = load("cli_bridge_server_for_parity",
                  os.path.join(os.path.dirname(HERE), "sidecars", "cli_bridge", "server.py"))
mcp_bridge = load("mcp_bridge_server_for_parity",
                  os.path.join(os.path.dirname(HERE), "sidecars", "mcp_bridge", "server.py"))
# mcp_bridge loads cli_bridge by file path; that copy is reachable as
# mcp_bridge.cli_bridge (see sidecars/mcp_bridge/server.py:81-85). The two
# copies must stay in lock-step inside a test for EXTRA_ARGS / SYSTEM_MODE /
# PROFILE lookups to agree.
inner_cli = mcp_bridge.cli_bridge

PROVIDERS = ("claude", "codex", "opencode")

EXTRA = ["--flag-a", "--flag-b=42"]


@contextlib.contextmanager
def _patched_provider(provider: str):
    """Swap the provider on every module that has its own globals.

    Saves and restores `cli_bridge` / `inner_cli` (the mcp_bridge-local copy
    of cli_bridge) PROFILE/PROVIDER/CLI, plus mcp_bridge's own PROFILE and
    PROVIDER. EXTRA_ARGS is patched on both cli_bridge copies; the mcp_bridge
    build_argv reads cli_bridge.EXTRA_ARGS at call time, so the inner copy
    is what matters -- the outer copy is patched only to keep the cli_bridge
    text-path argv in step.
    """
    saved_cli_outer = (cli_bridge.PROVIDER, cli_bridge.PROFILE, cli_bridge.CLI)
    saved_inner = (inner_cli.PROVIDER, inner_cli.PROFILE, inner_cli.CLI)
    saved_mcp = (mcp_bridge.PROVIDER, mcp_bridge.PROFILE)
    saved_extra_outer = cli_bridge.EXTRA_ARGS
    saved_extra_inner = inner_cli.EXTRA_ARGS
    cli_bridge.PROVIDER = inner_cli.PROVIDER = provider
    cli_bridge.PROFILE = inner_cli.PROFILE = cli_bridge.PROFILES[provider]
    cli_bridge.CLI = inner_cli.CLI = cli_bridge.PROFILE["cli"]
    mcp_bridge.PROVIDER = provider
    mcp_bridge.PROFILE = mcp_bridge.MCP_PROFILES[provider]
    cli_bridge.EXTRA_ARGS = inner_cli.EXTRA_ARGS = list(EXTRA)
    try:
        yield
    finally:
        (cli_bridge.PROVIDER, cli_bridge.PROFILE, cli_bridge.CLI) = saved_cli_outer
        (inner_cli.PROVIDER, inner_cli.PROFILE, inner_cli.CLI) = saved_inner
        (mcp_bridge.PROVIDER, mcp_bridge.PROFILE) = saved_mcp
        cli_bridge.EXTRA_ARGS = saved_extra_outer
        inner_cli.EXTRA_ARGS = saved_extra_inner


# ---------------------------------------------------------------------------
# Helpers. Each builds the argv a test expects to inspect -- one per (provider,
# bridge) -- so the assertions stay close to the input they care about.
# ---------------------------------------------------------------------------
def _cli_argv(provider: str, *, system: str | None, image_paths=None):
    """cli_bridge.build_argv under the current test's PROVIDER/PROFILE.

    A fresh workdir is not needed: cli_bridge writes its temp files into
    tempfile.gettempdir() and reclaims them inside the with-block (the
    instructions_file helper's finally), so nothing leaks across tests.
    """
    argv, _ = cli_bridge.build_argv("hello", system, "model-x", image_paths)
    return argv


def _mcp_argv(provider: str, workdir: Path, tools_path: Path, *,
              system: str | None, image_paths=None):
    """mcp_bridge.build_argv under the current test's PROVIDER/PROFILE."""
    argv, _ = mcp_bridge.build_argv(
        "hello", system, "model-x", workdir, "sess-x", tools_path,
        "", image_paths)
    return argv


# ---------------------------------------------------------------------------
# (a) CLI_EXTRA_ARGS tail-appended on both bridges, every provider.
# ---------------------------------------------------------------------------
def test_extra_args_tail_appended_on_every_provider_and_bridge():
    """With CLI_EXTRA_ARGS patched on both cli_bridge copies, every
    provider's argv on both bridges carries the extra flags at the tail.

    The operator flags are meant to win last (cli_bridge.build_argv has
    done the same since the CLI_EXTRA_ARGS knob landed), so the assertion
    is order-sensitive: the extras ride AFTER every per-provider argv
    fragment the bridge builds (lockdown, --tools, --session-id, ...).
    A bridge that inserted the extras anywhere else would let a per-provider
    flag silently override the operator's, which is the opposite of the
    contract."""
    workdir = Path(tempfile.mkdtemp(prefix="parity-extra-args-"))
    try:
        for provider in PROVIDERS:
            with _patched_provider(provider):
                # text path: cli_bridge.build_argv returns the argv the
                # inner CLI is spawned with. image_paths=None keeps every
                # provider on the simple argv path (no stream-json stdin,
                # no -i/-f image flags) so the only difference between the
                # bridges is the EXTRA_ARGS tail.
                cli_argv = _cli_argv(provider, system=None)
                assert cli_argv[-len(EXTRA):] == EXTRA, (provider, cli_argv)
                # tool path: mcp_bridge.build_argv takes its own session
                # workdir / session-id / tools_path plumbing.
                tools_path = workdir / "tools.json"
                tools_path.write_text("[]")
                mcp_argv = _mcp_argv(provider, workdir, tools_path,
                                     system=None)
                assert mcp_argv[-len(EXTRA):] == EXTRA, (provider, mcp_argv)
                # And the inner cli_bridge's EXTRA_ARGS was the one
                # consulted (mcp_bridge imports cli_bridge, not the outer
                # cli_bridge module), so the patches stayed coherent.
                assert inner_cli.EXTRA_ARGS == EXTRA, \
                    f"{provider}: inner cli_bridge EXTRA_ARGS not patched"
        print(f"  CLI_EXTRA_ARGS tail-appended on every provider x bridge: "
              f"{', '.join(PROVIDERS)}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# (c) codex always carries model_instructions_file=. With a system prompt
# it points at workdir/instructions.md (cleanup_workdir owns the workdir);
# with none it points at the profile's instructions_default
# (/app/harness/codex-instructions.md, baked by Dockerfile.sidecar:61).
# ---------------------------------------------------------------------------
def test_codex_carries_model_instructions_file_on_both_bridges():
    """Issue #137 parity: codex's `-c model_instructions_file=...` fragment
    must be present on both bridges, with a system prompt and without.

    Dropping it on a system-less request would leave codex running with its
    full compiled-in instructions -- the same regression that used to happen
    on the text path before the helper landed.

    The cli_bridge text path adds the fragment in `run_cli`, NOT
    `build_argv` -- it goes through `instructions_file(system)` and the
    result is appended to the spawn argv downstream. To verify it without
    spawning the CLI we exercise the helper directly: with no `dir` it
    writes to a tempfile (the legacy cli_bridge text path), with
    `dir=workdir` it writes to workdir/instructions.md (the shape mcp_bridge
    uses). The trailing-newline normalisation is what keeps both files
    byte-identical regardless of which bridge wrote them.
    """
    workdir = Path(tempfile.mkdtemp(prefix="parity-codex-instr-"))
    try:
        with _patched_provider("codex"):
            # ---- cli_bridge text path: helper produces the fragment ----
            # With a system prompt: tempfile (legacy text path). The file
            # content matches the mcp_bridge case (system + "\n").
            with cli_bridge.instructions_file("CALLER SYS") as (instr_args, system):
                assert system is None, "instructions_file must consume the system"
                assert len(instr_args) == 2 and instr_args[0] == "-c", instr_args
                assert instr_args[1].startswith("model_instructions_file="), \
                    instr_args
                tmp_path = Path(instr_args[1].split("=", 1)[1])
                assert tmp_path.read_text() == "CALLER SYS\n", tmp_path.read_text()

            # With a system prompt AND `dir=workdir`: writes to
            # workdir/instructions.md, NOT self-deleted (cleanup_workdir
            # owns the workdir lifecycle).
            with cli_bridge.instructions_file(
                    "CALLER SYS", dir=workdir) as (instr_args, system):
                assert system is None
                target = workdir / "instructions.md"
                assert instr_args[1] == f"model_instructions_file={target}", \
                    instr_args
                assert target.read_text() == "CALLER SYS\n", target.read_text()

            # With no system: falls back to the baked default.
            with cli_bridge.instructions_file(None) as (instr_args, system):
                assert system is None
                assert instr_args == [
                    "-c",
                    "model_instructions_file=/app/harness/codex-instructions.md",
                ], instr_args

            # ---- mcp_bridge tool path: build_argv carries the fragment ----
            tools_path = workdir / "tools.json"
            tools_path.write_text("[]")
            with_sys, _ = mcp_bridge.build_argv(
                "hi", "CALLER SYS", "m", workdir, "sess", tools_path, "")
            instr = workdir / "instructions.md"
            assert with_sys.count(f"model_instructions_file={instr}") == 1, \
                with_sys
            assert instr.read_text() == "CALLER SYS\n", instr.read_text()

            no_sys, _ = mcp_bridge.build_argv(
                "hi", None, "m", workdir, "sess", tools_path, "")
            instr_args = [a for a in no_sys if a.startswith("model_instructions_file=")]
            assert len(instr_args) == 1, no_sys
            assert instr_args[0] == "model_instructions_file=/app/harness/codex-instructions.md", \
                instr_args
        print("  codex: model_instructions_file= always present on both "
              "bridges (workdir file with system, baked default without)")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# (d) claude replace mode carries --exclude-dynamic-system-prompt-sections
# on both bridges, alongside both --system-prompt and --system-prompt-file.
# ---------------------------------------------------------------------------
def test_claude_replace_mode_carries_exclude_flag_on_both_bridges():
    """SYSTEM_MODE=replace on either bridge passes the caller's prompt via
    `--system-prompt` and tells the CLI to drop its own injected sections
    (`--exclude-dynamic-system-prompt-sections`). Without the exclude flag
    the tool-path's replace mode would silently leave the working-dir /
    git / env blurbs in place, so the compose comment on claude-max-sidecar
    (which says the lane is in replace mode) would not actually be true on
    this bridge -- the exact parity gap this test pins.

    Two forms on each bridge:

    * inline (`--system-prompt`) -- reached when the system is under
      STDIN_PROMPT_LIMIT. Tested via build_argv on both bridges.
    * file (`--system-prompt-file`) -- reached when the system overflows
      the limit. Tested via build_argv on mcp_bridge (the inline builder
      handles the file path itself) and via the cli_bridge
      `system_prompt_file` helper on the text path (run_cli's wrapper).
    """
    saved_mode = inner_cli.SYSTEM_MODE
    saved_limit = cli_bridge.STDIN_PROMPT_LIMIT
    workdir = Path(tempfile.mkdtemp(prefix="parity-replace-"))
    try:
        with _patched_provider("claude"):
            # Force replace mode on BOTH cli_bridge copies (mcp_bridge's
            # build_argv reads cli_bridge.SYSTEM_MODE at call time, which
            # is the inner copy).
            cli_bridge.SYSTEM_MODE = inner_cli.SYSTEM_MODE = "replace"
            # ---- inline path (small system) ----
            cli_argv_inline, _ = cli_bridge.build_argv("hi", "CALLER", "m")
            assert "--system-prompt" in cli_argv_inline, cli_argv_inline
            assert "--exclude-dynamic-system-prompt-sections" in cli_argv_inline, \
                cli_argv_inline

            tools_path = workdir / "tools.json"
            tools_path.write_text("[]")
            mcp_argv_inline, _ = mcp_bridge.build_argv(
                "hi", "CALLER", "m", workdir, "sess", tools_path, "")
            assert "--system-prompt" in mcp_argv_inline, mcp_argv_inline
            assert "--exclude-dynamic-system-prompt-sections" in mcp_argv_inline, \
                mcp_argv_inline

            # ---- file path ----
            # Pin STDIN_PROMPT_LIMIT low so a 1KB system overflows into
            # the file form. mcp_bridge's build_argv has the file path
            # inline; cli_bridge's text path goes through the
            # system_prompt_file context manager inside run_cli.
            cli_bridge.STDIN_PROMPT_LIMIT = inner_cli.STDIN_PROMPT_LIMIT = 10
            huge = "S" * 1000

            mcp_argv_file, _ = mcp_bridge.build_argv(
                "hi", huge, "m", workdir, "sess", tools_path, "")
            assert "--system-prompt-file" in mcp_argv_file, mcp_argv_file
            assert "--exclude-dynamic-system-prompt-sections" in mcp_argv_file, \
                mcp_argv_file

            with cli_bridge.system_prompt_file(huge) as (cli_sys_args, system):
                assert system is None
                assert cli_sys_args[0] == "--system-prompt-file", cli_sys_args
                assert cli_sys_args[2] == "--exclude-dynamic-system-prompt-sections", \
                    cli_sys_args

            # ---- negative: append mode must NOT carry the exclude flag ----
            cli_bridge.SYSTEM_MODE = inner_cli.SYSTEM_MODE = "append"
            cli_argv_append, _ = cli_bridge.build_argv("hi", "CALLER", "m")
            assert "--exclude-dynamic-system-prompt-sections" not in cli_argv_append, \
                cli_argv_append
            mcp_argv_append, _ = mcp_bridge.build_argv(
                "hi", "CALLER", "m", workdir, "sess", tools_path, "")
            assert "--exclude-dynamic-system-prompt-sections" not in mcp_argv_append, \
                mcp_argv_append

            # ---- negative: replace mode with no system must NOT add it ----
            cli_bridge.SYSTEM_MODE = inner_cli.SYSTEM_MODE = "replace"
            cli_argv_nosys, _ = cli_bridge.build_argv("hi", None, "m")
            assert "--exclude-dynamic-system-prompt-sections" not in cli_argv_nosys, \
                cli_argv_nosys
            mcp_argv_nosys, _ = mcp_bridge.build_argv(
                "hi", None, "m", workdir, "sess", tools_path, "")
            assert "--exclude-dynamic-system-prompt-sections" not in mcp_argv_nosys, \
                mcp_argv_nosys
        print("  claude replace mode: --exclude-dynamic-system-prompt-sections "
              "present alongside both --system-prompt and --system-prompt-file "
              "on both bridges (absent in append mode / no-system)")
    finally:
        cli_bridge.SYSTEM_MODE = inner_cli.SYSTEM_MODE = saved_mode
        cli_bridge.STDIN_PROMPT_LIMIT = inner_cli.STDIN_PROMPT_LIMIT = saved_limit
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
