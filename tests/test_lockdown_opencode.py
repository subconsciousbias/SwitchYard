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
import base64
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
from _modules import FakePoppler, load  # noqa: E402

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


# 1x1 magenta PNG and a minimal one-page PDF: what the media test sends.
PNG_MAGENTA = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63f8cf1f7f060006000300013fe46dabe90000000049454e44ae426082")
PDF_MIN = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
           b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
           b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 72 72]>>endobj\n"
           b"trailer<</Root 1 0 R>>\n%%EOF\n")


def _opencode_bridge_text(body: dict) -> str:
    """The text the bridge wrote into the opencode request body, stripped of
    the `-f FILE` argv opencode 1.18.31 inlines after the prompt. The
    bridge does not control opencode's wire format; what the bridge owns
    is the prompt content, which the change in this PR makes path-free
    (issue #296)."""
    msgs = body.get("messages") if isinstance(body.get("messages"), list) else []
    if not msgs:
        return ""
    last = msgs[-1]
    content = (last.get("content") or "") if isinstance(last, dict) else ""
    if not isinstance(content, str):
        content = json.dumps(content)
    # Opencode concatenates `-f FILE` per attachment after the prompt; the
    # bridge's own text is the prefix before the first `-f`.
    idx = content.find(" -f ")
    if idx == -1:
        return content
    return content[:idx]


def _stage_for_opencode_lockdown(stage: Path) -> tuple[str, list[Path]]:
    """Drive the cli_bridge the way `_handle_chat` does for a media call:
    build messages carrying the test's PNG + PDF, run stage_images under
    PROVIDER=opencode (the bridge expands the PDF via poppler and writes
    `[image N: attached]` markers), flatten, and append image_note. The
    caller passes the returned prompt + paths to build_argv so the
    request opencode actually emits carries the path-free markers +
    trailing note this PR's change is supposed to deliver.

    Both server.PROVIDER (mcp_bridge) and server.cli_bridge.PROVIDER
    (cli_bridge -- stage_images / _expand_pdf read this one) are flipped
    to "opencode"; the test's `saved` tuple restores cb.PROVIDER on the
    way out, so the next test in the file sees the module-level default
    ("claude") again."""
    server.PROVIDER = "opencode"
    server.cli_bridge.PROVIDER = "opencode"
    img_dir = stage / "img"
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "what is attached?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                      "data": base64.b64encode(PNG_MAGENTA).decode()}},
        {"type": "file", "file": {"file_data":
            "data:application/pdf;base64," + base64.b64encode(PDF_MIN).decode()}},
    ]}]
    paths = server.cli_bridge.stage_images(messages, img_dir)
    prompt, _ = server.cli_bridge.flatten(messages)
    prompt += server.cli_bridge.image_note(paths)
    return prompt, paths


def test_opencode_media_reach_the_model_not_relay_paths():
    """The bridge's image and PDF markers reach the pinned opencode as `-f`
    attaches, the bridge's own prompt text is path-free (the change in
    this PR), and the caller's tool is what the model can call on the
    tool path (issue #296).

    Opencode 1.18.31 inlines the `-f` argv into the user-message content
    on the wire, so the request body the model API sees does name the
    relay dir after opencode's own argv -- but the bridge's own prompt
    text (the prefix before the first `-f`) does not. The lockdown test
    asserts only what the bridge owns.

    `write_opencode_dir(images=True)` re-enables the native `read` tool
    in the offered set so the rebuilt-session loop can read staged
    images; bash and other built-ins stay denied.

    The bridge is driven through `stage_images` + `flatten` +
    `image_note` (the same chain `_handle_chat` runs) so the prompt
    really does carry the path-free `[image N: attached]` markers and
    the `[N attachment(s) follow this text...]` trailing note this PR
    delivers."""
    try:
        opencode = _pinned_clis.require("opencode")
    except _pinned_clis.Skip as why:
        print(f"  skipped: {why}")
        return
    cb = server.cli_bridge
    root = Path(tempfile.mkdtemp(prefix="lockdown-opencode-media-"))
    stage = root / "swimg-test" / "img"
    stage.mkdir(parents=True)
    (stage / "01.png").write_bytes(PNG_MAGENTA)
    (stage / "02.pdf").write_bytes(PDF_MIN)
    media = [stage / "01.png", stage / "02.pdf"]
    saved = (cb.PROVIDER, cb.PROFILE, cb.CLI, server.PROVIDER, server.PROFILE)
    try:
        # Tool path: opencode attaches media via -f FILE; the prompt is
        # the caller's text + image_note (path-free since #296).
        with FakePoppler():
            prompt, staged = _stage_for_opencode_lockdown(stage)
            server.PROVIDER = "opencode"
            server.PROFILE = dict(server.MCP_PROFILES["opencode"], cli=opencode)
            workdir = root / "session"
            workdir.mkdir()
            tools_path = workdir / "tools.json"
            tools_path.write_text(json.dumps([PROBE]))
            argv, _ = server.build_argv(prompt, "CALLER SYSTEM PROMPT", MODEL,
                                        workdir, uuid.uuid4().hex, tools_path,
                                        "switchyard_probe", staged)
            assert "-f" in argv, argv
            attached = [argv[i + 1] for i, a in enumerate(argv) if a == "-f"]
            assert attached == [str(p) for p in staged], argv
            with _fake_model.FakeModel([]) as fake:
                env = {**cb.subprocess_env(), **_xdg(root, fake.url)}
                done = subprocess.run(argv, cwd=workdir, env=env, capture_output=True,
                                      text=True, timeout=180, stdin=subprocess.DEVNULL)
        assert done.returncode == 0, done.stderr[-500:]
        # The bridge's own prompt text is path-free: the prompt +
        # image_note reach the model with no relay dir or staged file path
        # (opencode's own `-f FILE` argv appended after the prompt is
        # stripped -- that's opencode's wire format, not the bridge's, and
        # it is unchanged by this PR).
        bridge_text = _opencode_bridge_text(fake.requests[-1]["body"])
        assert "what is attached?" in bridge_text, bridge_text
        assert "attachment(s) follow this text" in bridge_text, bridge_text
        assert "[image 1: attached]" in bridge_text, bridge_text
        assert "[document 2: PDF," in bridge_text, bridge_text
        assert str(stage) not in bridge_text, bridge_text
        for p in media:
            assert str(p) not in bridge_text, (p, bridge_text)
        # Caller tool plus the native read (re-enabled for the rebuilt-
        # session loop on image calls -- see write_opencode_dir); bash
        # stays denied at the permission layer. Frozenset so the
        # comparison is order-independent -- opencode always renders
        # [read, switchyard_probe] in that order on the wire, and the
        # singleton tuple set would otherwise miss an element-order swap.
        offered = {frozenset(_fake_model.advertised_tools(r["body"]))
                   for r in fake.tool_requests()}
        assert offered == {frozenset({"switchyard_probe", "read"})}, sorted(offered)
        _no_title_call(fake)
        print(f"  opencode {_pinned_clis.installed_version('opencode')}: image + PDF ride "
              "-f on the tool path; bridge text path-free; "
              "switchyard_probe + read offered; no title call")
    finally:
        cb.PROVIDER, cb.PROFILE, cb.CLI, server.PROVIDER, server.PROFILE = saved
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
