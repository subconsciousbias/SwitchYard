"""Tests for the host-mirror self-check (sidecars/mcp_bridge/selfcheck.py,
issue #264) and the gate mcp_bridge puts in front of the tool path.

The real check spawns the sidecar's pinned CLI; offline there is none, so
`run()` is driven with tests/fixtures/fake_cli_selfcheck.py, a stand-in that
speaks the Anthropic protocol to the self-check's fake model and can be
obedient, leaky (offers and runs a native tool) or hide the bridged tool.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ["PROVIDER"] = "claude"
os.environ.setdefault("SWITCHYARD_PLAN", "claude-max")
from plans_path import plans_path  # noqa: E402

os.environ["SWITCHYARD_PLANS"] = plans_path()
os.environ.setdefault("SIDECAR_PORT", "8081")
# The gate is driven explicitly below; nothing here may start a real check
# (which would spawn whatever `claude` is on this machine's PATH).
os.environ["MCP_HOST_MIRROR_CHECK"] = "off"

from _modules import load  # noqa: E402

server = load("mcp_bridge_server", os.path.join(ROOT, "sidecars", "mcp_bridge", "server.py"))
server.HOST_MIRROR_MODE = "off"
selfcheck = load("mcp_bridge_selfcheck",
                 os.path.join(ROOT, "sidecars", "mcp_bridge", "selfcheck.py"))
FAKE_CLI = os.path.join(HERE, "fixtures", "fake_cli_selfcheck.py")


def test_advertised_tools_flattens_every_protocol_shape():
    """Anthropic/OpenAI top-level `tools`, and codex's `additional_tools`
    input item with namespaces, all reduce to one list of names; a request
    offering nothing (a title call) is None."""
    assert selfcheck.advertised_tools("/v1/messages", {"tools": [{"name": "Bash"}]}) == ["Bash"]
    assert selfcheck.advertised_tools("/v1/chat/completions", {"tools": [
        {"type": "function", "function": {"name": "switchyard_x"}}]}) == ["switchyard_x"]
    codex = {"input": [{"type": "additional_tools", "tools": [
        {"type": "namespace", "name": "functions",
         "tools": [{"type": "function", "name": "read_mcp_resource"}]},
        {"type": "namespace", "name": "mcp__switchyard",
         "tools": [{"type": "function", "name": "probe"}]}]}]}
    assert selfcheck.advertised_tools("/v1/responses", codex) == [
        "functions__read_mcp_resource", "mcp__switchyard__probe"]
    assert selfcheck.advertised_tools("/v1/messages", {"messages": []}) is None
    print("  tool names read from messages / chat / responses shapes")


def test_evaluate_passes_only_a_locked_down_cli():
    probe = "mcp__switchyard__switchyard_selfcheck_probe"

    def req(names):
        return {"path": "/v1/messages", "body": {"tools": [{"name": n} for n in names]}}

    ok = selfcheck.evaluate("claude", [req([probe])], probe, native_executed=False)
    assert ok["ok"], ok
    leaky = selfcheck.evaluate("claude", [req([probe, "Bash", "Agent"])], probe, False)
    assert not leaky["ok"] and leaky["unexpected"] == ["Agent", "Bash"], leaky
    ran = selfcheck.evaluate("claude", [req([probe])], probe, native_executed=True)
    assert not ran["ok"] and "executed" in ran["detail"], ran
    hidden = selfcheck.evaluate("claude", [req(["something_else"])], probe, False)
    assert not hidden["ok"] and not hidden["bridged_visible"], hidden
    nothing = selfcheck.evaluate("claude", [], probe, False)
    assert not nothing["ok"], nothing
    # Review of #275: a native tool offered on an EARLY request fails the
    # check even when the last request lists only the bridged tool.
    early = selfcheck.evaluate("claude", [req([probe, "Bash"]), req([probe])], probe, False)
    assert not early["ok"] and early["unexpected"] == ["Bash"], early
    assert early["offered"] == [probe, "Bash"], early
    # codex's read-only MCP-resource helpers (namespaced, as the pinned CLI
    # sends them) are allowed; its shell is not.
    cprobe = "mcp__switchyard__switchyard_selfcheck_probe"
    helpers = ["functions__list_mcp_resources", "functions__read_mcp_resource"]
    codex_ok = selfcheck.evaluate("codex", [req([cprobe] + helpers)], cprobe, False)
    assert codex_ok["ok"], codex_ok
    codex_bad = selfcheck.evaluate("codex", [req([cprobe, "functions__exec_command"] + helpers)],
                                   cprobe, False)
    assert not codex_bad["ok"] and codex_bad["unexpected"] == ["functions__exec_command"], codex_bad
    print("  evaluate: pass only with bridged tool visible, nothing native, nothing run")


def test_fake_model_answers_the_first_tool_bearing_request_with_the_native_call():
    """Side requests (no tools) get text; the first request that offers
    tools gets the native call; everything after gets text."""
    with selfcheck.FakeModel(("Bash", {"command": "true"})) as fake:
        def post(path, body):
            req = urllib.request.Request(fake.url + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(req, timeout=5).read().decode()
        assert '"text_delta"' in post("/v1/messages", {"messages": []})
        assert '"tool_use"' in post("/v1/messages", {"tools": [{"name": "x"}]})
        assert '"tool_use"' not in post("/v1/messages", {"tools": [{"name": "x"}]})
        assert "function_call" not in post("/v1/responses", {"tools": [{"name": "x"}]})
        assert len(fake.requests) == 4
    print("  fake model: native call once, on the first tool-bearing request")


def _run_with_fake_cli(mode: str) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="selfcheck-cli-"))
    cli = tmp / "fake_cli_selfcheck.py"
    shutil.copy(FAKE_CLI, cli)
    cli.chmod(0o755)
    (tmp / "fake_cli_selfcheck.mode").write_text(mode)
    saved = (server.PROVIDER, server.PROFILE)
    try:
        server.PROVIDER = "claude"
        server.PROFILE = dict(server.MCP_PROFILES["claude"], cli=str(cli))
        return asyncio.run(selfcheck.run(server, timeout=30))
    finally:
        server.PROVIDER, server.PROFILE = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_passes_an_obedient_cli():
    verdict = _run_with_fake_cli("obedient")
    assert verdict["ok"], verdict
    assert verdict["offered"] == ["mcp__switchyard__switchyard_selfcheck_probe"], verdict
    print(f"  obedient CLI -> ok ({verdict['duration_s']}s)")


def test_run_fails_a_cli_that_runs_its_native_shell():
    verdict = _run_with_fake_cli("leaky")
    assert not verdict["ok"], verdict
    assert verdict["native_executed"] is True, verdict
    assert verdict["unexpected"] == ["Bash"], verdict
    print(f"  leaky CLI -> {verdict['detail']}")


def test_run_fails_a_cli_that_hides_the_bridged_tool():
    verdict = _run_with_fake_cli("hides")
    assert not verdict["ok"], verdict
    assert "never offered" in verdict["detail"], verdict
    print(f"  hiding CLI -> {verdict['detail']}")


def test_gate_refuses_tool_work_until_the_check_passes():
    """enforce: a failed verdict is a 503 (the gateway places the request on
    another plan); a passed one lets it through. report never refuses."""
    from fastapi import HTTPException
    saved = (server.HOST_MIRROR_MODE, server._HOST_MIRROR_TASK)

    async def with_verdict(verdict, mode):
        server.HOST_MIRROR_MODE = mode
        future = asyncio.get_running_loop().create_future()
        future.set_result(verdict)
        server._HOST_MIRROR_TASK = future
        await server.require_host_mirror()

    try:
        try:
            asyncio.run(with_verdict({"ok": False, "detail": "native tools offered"}, "enforce"))
            raise AssertionError("expected 503")
        except HTTPException as exc:
            assert exc.status_code == 503, exc.status_code
            assert exc.detail["error"]["type"] == "host_mirror_unverified", exc.detail
            assert exc.headers["Retry-After"] == "60", exc.headers
        asyncio.run(with_verdict({"ok": True}, "enforce"))
        asyncio.run(with_verdict({"ok": False, "detail": "x"}, "report"))
        print("  gate: enforce 503s a failed check, passes a good one; report never refuses")
    finally:
        server.HOST_MIRROR_MODE, server._HOST_MIRROR_TASK = saved


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
