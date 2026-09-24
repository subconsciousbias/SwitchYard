"""OpenCode lockdown, proven against the pinned CLI (issue #264).

OpenCode's `tools: {x: false}` only hides a built-in: the pinned 1.18.31
still runs one the model names. Both bridges therefore write a config that
also denies every built-in at the permission layer (cli_bridge.
opencode_config), and the text path must find that config in its spawn
directory at all -- without it `--agent switchyard` fell back to OpenCode's
`build` agent with every tool allowed. These tests run the opencode binary
Dockerfile.sidecar pins, through the bridges' own spawn/argv code, against a
loopback fake model (tests/_fake_model.py) that makes the model call `bash`.

Skips when the pinned opencode is not installed; CI's cli-lockdown job
installs it and makes a skip a failure (tests/_pinned_clis.py).
"""
from __future__ import annotations

import asyncio
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
MODEL = "switchyard-test/m"
PROBE = {"name": "probe", "description": "a caller tool",
         "inputSchema": {"type": "object", "properties": {}}}


def _xdg(root: Path, url: str) -> dict:
    """Isolated XDG dirs whose global config points a provider at the fake."""
    config = root / "config" / "opencode"
    config.mkdir(parents=True)
    (config / "opencode.json").write_text(json.dumps({"provider": {"switchyard-test": {
        "npm": "@ai-sdk/openai-compatible", "name": "switchyard-test",
        "options": {"baseURL": f"{url}/v1", "apiKey": "test"},
        "models": {"m": {"name": "m", "tool_call": True}}}}}))
    return {"XDG_CONFIG_HOME": str(root / "config"), "XDG_DATA_HOME": str(root / "data"),
            "XDG_STATE_HOME": str(root / "state"), "XDG_CACHE_HOME": str(root / "cache"),
            "HOME": str(root)}


def _no_title_call(fake) -> None:
    for req in fake.requests:
        assert "title generator" not in _fake_model.request_text(req["body"]), \
            "OpenCode made its paid title-generation call"


def test_opencode_text_path_runs_no_builtin_the_model_names():
    try:
        opencode = _pinned_clis.require("opencode")
    except _pinned_clis.Skip as why:
        print(f"  skipped: {why}")
        return
    cb = server.cli_bridge
    root = Path(tempfile.mkdtemp(prefix="lockdown-opencode-text-"))
    marker = root / "native-bash-ran"
    saved = (cb.PROVIDER, cb.PROFILE, cb.CLI, dict(os.environ))
    try:
        with _fake_model.FakeModel(
                [{"tool": "bash", "input": {"command": f"touch {marker}",
                                            "description": "x"}}], force=True) as fake:
            # run_cli is the real text-path spawn: it writes opencode_config()
            # into the fresh sy-cli-* dir, and subprocess_env() carries these
            # XDG pins through the allowlist.
            os.environ.update(_xdg(root, fake.url))
            cb.PROVIDER, cb.PROFILE, cb.CLI = "opencode", cb.PROFILES["opencode"], opencode
            payload = asyncio.run(cb.run_cli("hello", None, MODEL))
        assert payload.get("result") is not None, payload
        assert all(not _fake_model.advertised_tools(r["body"]) for r in fake.requests), \
            [_fake_model.advertised_tools(r["body"]) for r in fake.requests]
        assert not marker.exists(), "the model's bash call ran in the sidecar"
        _no_title_call(fake)
        print(f"  opencode {_pinned_clis.installed_version('opencode')} text path: "
              "switchyard agent found, zero tools, forced bash refused, no title call")
    finally:
        cb.PROVIDER, cb.PROFILE, cb.CLI = saved[:3]
        os.environ.clear()
        os.environ.update(saved[3])
        shutil.rmtree(root, ignore_errors=True)


def test_opencode_tool_path_offers_only_the_callers_tools():
    try:
        opencode = _pinned_clis.require("opencode")
    except _pinned_clis.Skip as why:
        print(f"  skipped: {why}")
        return
    root = Path(tempfile.mkdtemp(prefix="lockdown-opencode-mcp-"))
    marker = root / "native-bash-ran"
    saved = (server.PROVIDER, server.PROFILE)
    try:
        server.PROVIDER = "opencode"
        server.PROFILE = dict(server.MCP_PROFILES["opencode"], cli=opencode)
        workdir = root / "session"
        workdir.mkdir()
        tools_path = workdir / "tools.json"
        tools_path.write_text(json.dumps([PROBE]))
        argv, _ = server.build_argv("hello", "CALLER SYSTEM PROMPT", MODEL, workdir,
                                    uuid.uuid4().hex, tools_path, "switchyard_probe")
        script = [{"tool": "bash", "input": {"command": f"touch {marker}", "description": "x"}},
                  {"tool": "read", "input": {"filePath": str(tools_path)}}]
        with _fake_model.FakeModel(script) as fake:
            env = {**server.cli_bridge.subprocess_env(), **_xdg(root, fake.url)}
            done = subprocess.run(argv, cwd=workdir, env=env, capture_output=True,
                                  text=True, timeout=180, stdin=subprocess.DEVNULL)
        assert done.returncode == 0, done.stderr[-500:]
        offered = {tuple(_fake_model.advertised_tools(r["body"])) for r in fake.tool_requests()}
        assert offered == {("switchyard_probe",)}, offered
        assert not marker.exists(), "the model's bash call ran in the sidecar"
        assert done.stdout.count('"status":"error"') >= 2, done.stdout[-800:]
        assert "unavailable tool" in done.stdout, done.stdout[-800:]
        _no_title_call(fake)
        print("  opencode tool path: only switchyard_probe offered; bash/read refused; "
              "no title call")
    finally:
        server.PROVIDER, server.PROFILE = saved
        shutil.rmtree(root, ignore_errors=True)


def test_opencode_reasoning_event_reaches_the_parser_separately():
    """Lockdown: an opencode event stream with both a `reasoning` event and
    a text event must be parsed so the assistant's reasoning lives on its
    own field (`reasoning`) and the text on the result. The parser returns
    a dict with both keys intact, so to_openai lifts `reasoning_content`
    without losing either half.
    """
    cb = server.cli_bridge
    stream = "\n".join([
        json.dumps({"type": "step_start", "part": {"type": "step-start"}}),
        json.dumps({"type": "reasoning", "part": {"type": "reasoning",
                                                   "text": "the chain of thought"}}),
        json.dumps({"type": "reasoning", "part": {"type": "reasoning",
                                                   "text": " continues"}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": "the answer"}}),
        json.dumps({"type": "step_finish", "part": {
            "type": "step-finish",
            "tokens": {"input": 4, "output": 6, "reasoning": 12}}}),
    ])
    parsed = cb.parse_output(stream, "events_json")
    assert parsed["result"] == "the answer", parsed
    assert parsed["reasoning"] == "the chain of thought continues", parsed

    # to_openai lifts the reasoning onto a separate field; the answer text
    # stays on `content`. The split survives the OpenAI envelope, which is
    # what makes LiteLLM's Messages/Responses adapters able to render the
    # chain of thought without re-deriving it from the answer.
    out = cb.to_openai(parsed, "m")
    msg = out["choices"][0]["message"]
    assert msg["content"] == "the answer", msg
    assert msg["reasoning_content"] == "the chain of thought continues", msg
    print("  opencode: reasoning event + text event -> separate result + "
          "reasoning_content fields")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
