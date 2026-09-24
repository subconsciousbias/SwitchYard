"""The lockdown harness itself: the pinned CLI versions it tests against
match Dockerfile.sidecar, and the fake model speaks each protocol.

The per-CLI lockdown tests (tests/test_lockdown_<cli>.py) run the real
pinned CLIs; see tests/_pinned_clis.py for how CI provides them.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import conftest  # noqa: E402,F401  (socket guard under the plain runner)
import _fake_model  # noqa: E402
import _pinned_clis  # noqa: E402


def test_dockerfile_pins_every_sidecar_cli():
    pins = _pinned_clis.pinned_versions()
    assert set(pins) == {"claude", "codex", "opencode"}, pins
    for cli, version in pins.items():
        assert version[0].isdigit(), (cli, version)
    print(f"  pinned: {pins}")


def test_installed_clis_are_the_pinned_versions():
    """In CI (SWITCHYARD_REQUIRE_CLIS=1) every pinned CLI must be installed
    at exactly the Dockerfile's version -- otherwise the lockdown tests would
    be proving something about a CLI the image does not ship."""
    checked = []
    for cli in ("claude", "codex", "opencode"):
        try:
            _pinned_clis.require(cli)
            checked.append(cli)
        except _pinned_clis.Skip as why:
            print(f"  skipped {cli}: {why}")
    print(f"  pinned and installed: {checked or 'none (local run)'}")


def test_fake_model_plays_its_script_per_protocol():
    script = [{"tool": "Bash", "input": {"command": "true"}},
              {"tool": "exec_command", "input": {"cmd": "true"}},
              {"tool": "bash", "input": {"command": "true"}}]
    with _fake_model.FakeModel(script) as fake:
        def post(path, body):
            req = urllib.request.Request(fake.url + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(req, timeout=5).read().decode()
        assert '"text_delta"' in post("/v1/messages", {"messages": []})     # side request
        assert '"tool_use"' in post("/v1/messages", {"tools": [{"name": "x"}]})
        assert '"function_call"' in post("/v1/responses", {"input": [
            {"type": "additional_tools", "tools": [{"type": "namespace", "name": "ns",
                                                    "tools": [{"name": "y"}]}]}]})
        assert '"tool_calls"' in post("/v1/chat/completions", {"tools": [
            {"type": "function", "function": {"name": "z"}}]})
        exhausted = post("/v1/chat/completions", {"tools": [
            {"type": "function", "function": {"name": "z"}}]})
        assert '"content": "done"' in exhausted and "tool_calls" not in exhausted
        assert [_fake_model.advertised_tools(r["body"]) for r in fake.tool_requests()] == [
            ["x"], ["ns__y"], ["z"], ["z"]]
    with _fake_model.FakeModel([{"tool": "bash", "input": {}}], force=True) as fake:
        req = urllib.request.Request(fake.url + "/v1/chat/completions",
                                     data=json.dumps({"messages": []}).encode(),
                                     headers={"Content-Type": "application/json"})
        assert '"tool_calls"' in urllib.request.urlopen(req, timeout=5).read().decode()
    print("  fake model: messages / responses / chat, scripted per tool-bearing request"
          " (or every request with force=True)")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
