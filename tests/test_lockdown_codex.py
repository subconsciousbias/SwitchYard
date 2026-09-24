"""Codex lockdown, proven against the pinned CLI (issues #255, #264).

On the pinned codex every current catalog model runs in code mode: the
caller's bridged MCP tools become hidden tools inside a JavaScript `exec`
host while codex's own shell is advertised, and `sandbox_mode=read-only`
does not stop that shell under the MCP path's bypass flag.
cli_bridge.codex_lockdown_args() disables the tool features and prompt
sections and swaps in a catalog without code mode. These tests run the codex
binary Dockerfile.sidecar pins -- its real `codex debug models` feeds the
catalog -- with the bridges' own argv, against a loopback fake model that
makes the model call exec_command and apply_patch.

Skips when the pinned codex is not installed; CI's cli-lockdown job installs
it and makes a skip a failure (tests/_pinned_clis.py).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PROVIDER"] = "claude"
os.environ.setdefault("SWITCHYARD_PLAN", "claude-max")
from plans_path import plans_path  # noqa: E402

os.environ["SWITCHYARD_PLANS"] = plans_path()
os.environ.setdefault("SIDECAR_PORT", "8081")

import _fake_model  # noqa: E402
import _pinned_clis  # noqa: E402
from _modules import load  # noqa: E402

server = load("mcp_bridge_server",
              os.path.join(os.path.dirname(HERE), "sidecars", "mcp_bridge", "server.py"))
MODEL = "gpt-5.6-terra"          # a catalog model that runs in code mode unpatched
PROBE = {"name": "probe", "description": "a caller tool",
         "inputSchema": {"type": "object", "properties": {}}}
HELPERS = {"functions__list_mcp_resources", "functions__list_mcp_resource_templates",
           "functions__read_mcp_resource"}


def _fake_provider(url: str) -> list[str]:
    return ["-c", "model_provider=switchyard_test", "-c",
            'model_providers.switchyard_test={name="test",'
            f'base_url="{url}/v1",env_key="SWITCHYARD_TEST_KEY",wire_api="responses"}}']


class _RealCodex:
    """Point the lockdown's catalog generation at the real pinned binary
    (tests/_modules.py points CODEX_CLI at a fake for the offline suite)."""

    def __init__(self, codex: str, root: Path):
        self.cb, self.codex, self.root = server.cli_bridge, codex, root

    def __enter__(self):
        cb = self.cb
        self.saved = (cb.PROFILES["codex"]["cli"],
                      getattr(cb, "CODEX_CATALOG_PATH", None),
                      getattr(cb, "_codex_catalog_ready", None))
        cb.PROFILES["codex"]["cli"] = self.codex
        cb.CODEX_CATALOG_PATH = self.root / "catalog.json"
        cb._codex_catalog_ready = False
        return self

    def __exit__(self, *exc):
        cb = self.cb
        (cb.PROFILES["codex"]["cli"], cb.CODEX_CATALOG_PATH,
         cb._codex_catalog_ready) = self.saved


def _run(argv, fake, root, cwd):
    env = {**server.cli_bridge.subprocess_env(), "CODEX_HOME": str(root / "codex-home"),
           "HOME": str(root), "SWITCHYARD_TEST_KEY": "test"}
    (root / "codex-home").mkdir(exist_ok=True)
    return subprocess.run(argv + _fake_provider(fake.url), cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=180,
                          stdin=subprocess.DEVNULL)


def _script(marker: Path) -> list[dict]:
    return [{"tool": "exec_command", "input": {"cmd": f"touch {marker}"}},
            {"tool": "apply_patch", "kind": "custom",
             "input": f"*** Begin Patch\n*** Add File: {marker}.patch\n+x\n*** End Patch"}]


def _refused(fake) -> str:
    return _fake_model.request_text(fake.requests[-1]["body"])


def test_codex_tool_path_offers_only_the_callers_tools():
    try:
        codex = _pinned_clis.require("codex")
    except _pinned_clis.Skip as why:
        print(f"  skipped: {why}")
        return
    root = Path(tempfile.mkdtemp(prefix="lockdown-codex-mcp-"))
    marker = root / "native-shell-ran"
    saved = (server.PROVIDER, server.PROFILE)
    try:
        with _RealCodex(codex, root):
            server.PROVIDER = "codex"
            server.PROFILE = dict(server.MCP_PROFILES["codex"], cli=codex)
            workdir = root / "session"
            workdir.mkdir()
            tools_path = workdir / "tools.json"
            tools_path.write_text(json.dumps([PROBE]))
            argv, _ = server.build_argv("hello", "CALLER SYSTEM PROMPT", MODEL, workdir,
                                        uuid.uuid4().hex, tools_path, "probe")
            with _fake_model.FakeModel(_script(marker)) as fake:
                done = _run(argv, fake, root, workdir)
        assert done.returncode == 0, done.stderr[-500:]
        offered = set()
        for req in fake.tool_requests():
            offered |= set(_fake_model.advertised_tools(req["body"]))
        assert "mcp__switchyard__probe" in offered, offered
        assert offered - HELPERS == {"mcp__switchyard__probe"}, sorted(offered)
        refused = _refused(fake)
        assert "unsupported call: exec_command" in refused, refused[-600:]
        assert "unsupported custom tool call: apply_patch" in refused, refused[-600:]
        assert not marker.exists() and not Path(f"{marker}.patch").exists()
        system = _fake_model.request_text(fake.tool_requests()[0]["body"])
        assert "CALLER SYSTEM PROMPT" in system
        for leaked in ("You are Codex", "<environment_context>", "<skills_instructions>",
                       "<permissions instructions>"):
            assert leaked not in system, f"{leaked} reached the model"
        print(f"  codex {_pinned_clis.installed_version('codex')} tool path: caller prompt "
              "+ mcp__switchyard__probe (+ resource helpers) only; exec_command and "
              "apply_patch refused")
    finally:
        server.PROVIDER, server.PROFILE = saved
        shutil.rmtree(root, ignore_errors=True)


def test_codex_text_path_runs_no_builtin_the_model_names():
    try:
        codex = _pinned_clis.require("codex")
    except _pinned_clis.Skip as why:
        print(f"  skipped: {why}")
        return
    cb = server.cli_bridge
    root = Path(tempfile.mkdtemp(prefix="lockdown-codex-text-"))
    marker = root / "native-shell-ran"
    saved = (cb.PROVIDER, cb.PROFILE, cb.CLI)
    try:
        with _RealCodex(codex, root):
            cb.PROVIDER, cb.PROFILE, cb.CLI = "codex", cb.PROFILES["codex"], codex
            argv, _ = cb.build_argv("hello", None, MODEL)
            with _fake_model.FakeModel(_script(marker), force=True) as fake:
                done = _run(argv, fake, root, root)
        assert done.returncode == 0, done.stderr[-500:]
        assert all(not _fake_model.advertised_tools(r["body"]) for r in fake.requests), \
            [_fake_model.advertised_tools(r["body"]) for r in fake.requests]
        assert "unsupported" in _refused(fake), _refused(fake)[-600:]
        assert not marker.exists() and not Path(f"{marker}.patch").exists()
        print("  codex text path: zero tools offered; forced exec_command/apply_patch refused")
    finally:
        cb.PROVIDER, cb.PROFILE, cb.CLI = saved
        shutil.rmtree(root, ignore_errors=True)


def test_codex_output_schema_reaches_the_api_as_strict_json_schema():
    """A caller's strict response_format rides `--output-schema FILE`
    (cli_bridge.schema_args, appended after the prompt like every run_cli
    extra) and reaches the model API as `text.format` json_schema, strict:
    the API enforces it, not a prompt line."""
    try:
        codex = _pinned_clis.require("codex")
    except _pinned_clis.Skip as why:
        print(f"  skipped: {why}")
        return
    cb = server.cli_bridge
    root = Path(tempfile.mkdtemp(prefix="lockdown-codex-schema-"))
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}},
              "required": ["answer"], "additionalProperties": False}
    fmt = cb.response_schema({"response_format": {"type": "json_schema", "json_schema": {
        "name": "sum", "strict": True, "schema": schema}}})
    saved = (cb.PROVIDER, cb.PROFILE, cb.CLI)
    try:
        with _RealCodex(codex, root):
            cb.PROVIDER, cb.PROFILE, cb.CLI = "codex", cb.PROFILES["codex"], codex
            argv, _ = cb.build_argv("What is 2+3?", None, MODEL)
            with cb.schema_args(fmt) as extra, \
                    _fake_model.FakeModel([{"text": '{"answer": 5}'}]) as fake:
                done = _run(argv + extra, fake, root, root)
        assert done.returncode == 0, done.stderr[-500:]
        formats = [r["body"].get("text", {}).get("format") for r in fake.requests]
        assert {"type": "json_schema", "strict": True, "schema": schema,
                "name": "codex_output_schema"} in formats, formats
        print("  codex --output-schema: strict text.format json_schema on the wire")
    finally:
        cb.PROVIDER, cb.PROFILE, cb.CLI = saved
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
