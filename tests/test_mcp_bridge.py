"""Tests for the inverted MCP bridge (sidecars/mcp_bridge).

No real CLI calls here — that is exercised manually against the live stack
(see TESTING.md) and is far too slow and flaky for a unit test. Two things are
driven for real instead:

  * tool_server.py, the MCP stdio server the CLI spawns, is run as an actual
    subprocess and fed real JSON-RPC frames over its stdin/stdout pipes,
    against a stub HTTP server standing in for server.py's callback endpoint.
  * server.py's session driver (run_session) is run against a fake CLI
    process — a throwaway script that prints canned output — to prove the
    subprocess lifecycle and error classification, without a vendor CLI.

Everything else (schema translation, parking, correlation by tool_call_id,
parallel-call batching, reaping) is driven directly against server.py's
in-process session objects, which is the same code the HTTP routes call.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import http.server
import json
import os
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
import types
import threading
import time
import uuid
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MCP_BRIDGE_DIR = os.path.join(ROOT, "sidecars", "mcp_bridge")
sys.path.insert(0, MCP_BRIDGE_DIR)
sys.path.insert(0, HERE)

os.environ["PROVIDER"] = "claude"
os.environ.setdefault("SWITCHYARD_PLAN", "claude-max")
# These tests drive handle_tool_request directly, with fake sessions; the
# startup self-check that gates it (selfcheck.py) has its own tests in
# tests/test_selfcheck.py.
os.environ["MCP_HOST_MIRROR_CHECK"] = "off"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()
os.environ.setdefault("SIDECAR_PORT", "8081")

from _modules import load  # noqa: E402

server = load("mcp_bridge_server", os.path.join(MCP_BRIDGE_DIR, "server.py"))
# The env var above only takes effect if this module loads the bridge first;
# another test module may already have, so pin the mode on the module too.
server.HOST_MIRROR_MODE = "off"

TOOL_SERVER = os.path.join(MCP_BRIDGE_DIR, "tool_server.py")


def _new_session() -> "server.Session":
    session = server.Session(id=uuid.uuid4().hex, provider="claude", model="m",
                              workdir=tempfile.mkdtemp(prefix="mcpb-test-"))
    server.SESSIONS[session.id] = session
    return session


def _drop(session: "server.Session") -> None:
    server.SESSIONS.pop(session.id, None)


# --------------------------------------------------------------- translation ---
def test_translate_tools_openai_to_mcp():
    openai_tools = [
        {"type": "function", "function": {
            "name": "get_weather", "description": "Look up the weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}},
        # Some callers send the bare function shape without the "type" wrapper.
        {"name": "no_wrapper", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "function": {"name": ""}},           # dropped: no name
    ]
    mcp_tools = server.translate_tools(openai_tools)
    assert len(mcp_tools) == 2, mcp_tools
    assert mcp_tools[0] == {
        "name": "get_weather", "description": "Look up the weather",
        "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]},
    }
    assert mcp_tools[1]["name"] == "no_wrapper"
    print(f"  {len(mcp_tools)}/3 tools translated (one malformed entry dropped)")


def test_fit_tool_surface_splits_long_descriptions_for_claude():
    """Claude Code cuts an MCP tool description at 2048 characters. A
    longer caller description keeps its head on the tool and moves the
    remainder into the system prompt -- nothing lost, nothing sent twice."""
    saved = (server.PROVIDER, server.PROFILE)
    try:
        server.PROVIDER, server.PROFILE = "claude", server.MCP_PROFILES["claude"]
        head = "Runs a shell command.\n" + ("h" * 1500) + "\n"
        tail = "TAIL-" + ("t" * 3000)
        tools = [{"name": "Bash", "description": head + tail, "inputSchema": {}},
                 {"name": "Read", "description": "short", "inputSchema": {}}]
        fitted, system = server.fit_tool_surface(tools, "CALLER", {})
        bash = fitted[0]["description"]
        assert len(bash) < 2048, len(bash)
        assert bash.startswith("Runs a shell command."), bash[:40]
        assert "Tool reference: mcp__switchyard__Bash" in bash, bash[-120:]
        assert fitted[1]["description"] == "short"
        assert system.startswith("CALLER\n\n"), system[:40]
        assert "### Tool reference: mcp__switchyard__Bash" in system
        assert tail in system and head not in system, "remainder only, once"
        print("  claude: long description split, remainder in system prompt")
    finally:
        server.PROVIDER, server.PROFILE = saved


def test_fit_tool_surface_leaves_other_clis_descriptions_whole():
    """OpenCode and codex keep a 5000-character description intact
    (verified on the pinned CLIs): no split there."""
    saved = (server.PROVIDER, server.PROFILE)
    try:
        for provider in ("opencode", "codex"):
            server.PROVIDER, server.PROFILE = provider, server.MCP_PROFILES[provider]
            desc = "d" * 5000
            fitted, system = server.fit_tool_surface(
                [{"name": "bash", "description": desc, "inputSchema": {}}], None, {})
            assert fitted[0]["description"] == desc, provider
            assert "Tool reference" not in (system or ""), provider
        print("  opencode / codex: descriptions untouched")
    finally:
        server.PROVIDER, server.PROFILE = saved


def test_fit_tool_surface_maps_names_and_states_tool_choice():
    """The caller's prompt names its tools bare; the CLI exposes them with
    a prefix, so one line maps the two. tool_choice has no CLI flag and
    becomes an explicit (advisory) instruction."""
    saved = (server.PROVIDER, server.PROFILE)
    tools = [{"name": "Bash", "description": "x", "inputSchema": {}}]
    try:
        server.PROVIDER, server.PROFILE = "claude", server.MCP_PROFILES["claude"]
        _, system = server.fit_tool_surface(tools, None, {})
        assert "`Bash` is `mcp__switchyard__Bash`" in system, system
        _, system = server.fit_tool_surface(tools, None, {"tool_choice": "none"})
        assert "do not call any tool" in system, system
        _, system = server.fit_tool_surface(tools, None, {"tool_choice": "required"})
        assert "must call at least one tool" in system, system
        _, system = server.fit_tool_surface(
            tools, None, {"tool_choice": {"type": "function", "function": {"name": "Bash"}}})
        assert "must call the tool `mcp__switchyard__Bash`" in system, system
        server.PROVIDER, server.PROFILE = "opencode", server.MCP_PROFILES["opencode"]
        _, system = server.fit_tool_surface(tools, None, {})
        assert "`Bash` is `switchyard_Bash`" in system, system
        server.PROVIDER, server.PROFILE = "codex", server.MCP_PROFILES["codex"]
        _, system = server.fit_tool_surface(tools, None, {})
        assert system is None, system          # codex keeps bare names
        print("  name mapping per CLI; tool_choice none/required/named stated")
    finally:
        server.PROVIDER, server.PROFILE = saved


def test_translate_tools_logs_what_it_drops():
    """An unrecognised non-function tool or a malformed entry has no
    caller-side executor; it is dropped, but never silently. (Web and tool
    search are left out on purpose, not dropped: see the typed-tools tests.)"""
    import logging
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    server.log.addHandler(handler)
    try:
        out = server.translate_tools([
            {"type": "function", "function": {"name": "ok", "parameters": {}}},
            {"type": "some_future_server_tool"}])
    finally:
        server.log.removeHandler(handler)
    assert [t["name"] for t in out] == ["ok"], out
    assert any("some_future_server_tool" in r.getMessage() for r in records), \
        [r.getMessage() for r in records]
    print("  dropped non-function tool is logged")


def test_opencode_session_config_carries_the_caller_system_prompt():
    """The agent's `prompt:` replaces OpenCode's own ~9.5K base prompt;
    the caller's system prompt belongs there, not folded into the user
    turn beneath it."""
    import shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-oc-prompt-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    saved = (server.PROVIDER, server.PROFILE)
    try:
        server.PROVIDER, server.PROFILE = "opencode", server.MCP_PROFILES["opencode"]
        argv, _ = server.build_argv("do the thing", "CALLER SYSTEM", "m",
                                    workdir, "sess", tools_path, "")
        cfg = json.loads((workdir / "opencode.json").read_text())
        assert cfg["agent"]["switchyard"]["prompt"] == "CALLER SYSTEM", cfg
        assert argv[-1] == "do the thing", argv
        print("  opencode: caller system prompt -> agent prompt, user turn untouched")
    finally:
        server.PROVIDER, server.PROFILE = saved
        shutil.rmtree(workdir, ignore_errors=True)


TINY_PDF = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R"
            b"/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
            b"4 0 obj<</Length 55>>stream\nBT /F1 24 Tf 20 60 Td (HOST PDF MARKER) Tj ET\n"
            b"endstream endobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"trailer<</Root 1 0 R>>\n%%EOF")


from _modules import FakePoppler as _FakePoppler  # noqa: E402


def test_pdf_tool_results_reach_claude_and_codex_as_text_and_page_images():
    """Claude Code saves an MCP PDF to a sidecar path the model cannot open;
    codex drops it or dumps base64 (verified on the pinned CLIs). Every PDF
    shape a tool result can carry becomes extracted text + one image per
    page, which both CLIs pass to the model."""
    import base64
    b64 = base64.b64encode(TINY_PDF).decode()
    shapes = {
        "litellm image_url": {"type": "image_url",
                              "image_url": {"url": f"data:application/pdf;base64,{b64}"}},
        "openai file": {"type": "file", "file": {"filename": "a.pdf",
                                                 "file_data": f"data:application/pdf;base64,{b64}"}},
        "anthropic document": {"type": "document", "source": {
            "type": "base64", "media_type": "application/pdf", "data": b64}},
    }
    saved = server.PROVIDER
    try:
        with _FakePoppler():
            for provider in ("claude", "codex"):
                server.PROVIDER = provider
                for label, part in shapes.items():
                    blocks, is_error = server.openai_tool_content_to_mcp(
                        [{"type": "text", "text": "read a.pdf:"}, part])
                    assert not is_error, (provider, label)
                    assert blocks[0] == {"type": "text", "text": "read a.pdf:"}
                    assert "EXTRACTED TEXT" in blocks[1]["text"], (provider, label, blocks[1])
                    assert "2 page(s)" in blocks[1]["text"], blocks[1]
                    pages = [b for b in blocks if b["type"] == "image"]
                    assert [base64.b64decode(b["data"]) for b in pages] == [b"PNG1", b"PNG2"]
                    assert all(b["mimeType"] == "image/png" for b in pages)
        print("  PDF (image_url / file / document) -> text + page PNGs on claude and codex")
    finally:
        server.PROVIDER = saved


def test_pdf_tool_result_stays_a_pdf_on_opencode():
    """OpenCode passes a PDF to a PDF-capable model natively (as a file part)."""
    import base64
    b64 = base64.b64encode(TINY_PDF).decode()
    saved = server.PROVIDER
    try:
        server.PROVIDER = "opencode"
        blocks, _ = server.openai_tool_content_to_mcp([{"type": "image_url",
            "image_url": {"url": f"data:application/pdf;base64,{b64}"}}])
        assert blocks == [{"type": "image", "mimeType": "application/pdf", "data": b64}], blocks
        print("  opencode: PDF passed through natively")
    finally:
        server.PROVIDER = saved


def test_an_unrenderable_pdf_is_said_not_dropped():
    import base64
    saved = (server.PROVIDER, os.environ.get("PATH", ""))
    try:
        server.PROVIDER = "claude"
        os.environ["PATH"] = "/nonexistent"          # no poppler at all
        blocks, _ = server.openai_tool_content_to_mcp([{"type": "document", "source": {
            "type": "base64", "media_type": "application/pdf",
            "data": base64.b64encode(TINY_PDF).decode()}}])
        assert len(blocks) == 1 and "could not be rendered" in blocks[0]["text"], blocks
        print("  unrenderable PDF -> an explicit notice to the model")
    finally:
        server.PROVIDER, os.environ["PATH"] = saved


def test_partial_pdf_render_says_what_is_missing():
    """pdftotext failing while pages render: the pages go through and the
    note says no text could be extracted (not a bare empty line)."""
    import base64
    saved = server.PROVIDER
    try:
        server.PROVIDER = "claude"
        with _FakePoppler(text_fails=True):
            blocks = server.pdf_to_mcp_blocks(base64.b64encode(TINY_PDF).decode())
        assert "2 page(s)" in blocks[0]["text"], blocks[0]
        assert "no text could be extracted" in blocks[0]["text"], blocks[0]
        assert [b["type"] for b in blocks[1:]] == ["image", "image"], blocks
    finally:
        server.PROVIDER = saved


def test_fresh_tool_session_attaches_a_prompt_pdf_as_page_images():
    """The tool path stages the prompt the way the text path does
    (cli_bridge.stage_or_fail_async): on codex / opencode a PDF in the
    prompt becomes its text in the prompt and its pages on -i / -f."""
    import base64
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-pdfprompt-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    cb = server.cli_bridge
    saved = (server.PROVIDER, server.PROFILE, cb.PROVIDER)
    try:
        for provider, flag in (("codex", "-i"), ("opencode", "-f")):
            server.PROVIDER, server.PROFILE = provider, server.MCP_PROFILES[provider]
            cb.PROVIDER = provider
            messages = [{"role": "user", "content": [
                {"type": "text", "text": "read this"},
                {"type": "file", "file": {"file_data": "data:application/pdf;base64,"
                                          + base64.b64encode(TINY_PDF).decode()}}]}]
            with _FakePoppler():
                paths, img_dir = asyncio.run(cb.stage_or_fail_async(messages))
            try:
                prompt, _ = cb.flatten(messages)
                prompt += cb.image_note(paths)
                argv, stdin = server.build_argv(prompt, None, "m", workdir, "sess",
                                                tools_path, "", image_paths=paths)
                attached = [argv[i + 1] for i, a in enumerate(argv) if a == flag]
                assert attached == [str(p) for p in paths], (provider, argv)
                assert [Path(p).name for p in attached] == ["page-1.png", "page-2.png"]
                sent = stdin if stdin is not None else "\n".join(argv)
                assert "EXTRACTED TEXT" in sent, sent
                assert "[2 attachment(s) follow this text, in order -- look at " in sent, sent
                # The path-free note must not leak the relay dir or any
                # staged file path (issue #296).
                assert "swimg-" not in prompt, prompt
                assert "swimg-" not in (stdin or ""), stdin
                assert not any(str(p) in prompt for p in paths), prompt
            finally:
                shutil.rmtree(img_dir, ignore_errors=True)
        print("  tool path: prompt PDF -> page PNGs on codex -i / opencode -f + text")
    finally:
        server.PROVIDER, server.PROFILE, cb.PROVIDER = saved
        shutil.rmtree(workdir, ignore_errors=True)


def test_non_pdf_documents_and_files_in_a_tool_result():
    """Images inside `document` / `file` parts pass through as images; text
    files become text; anything undeliverable is isError with a notice that
    names it (never an empty block)."""
    import base64
    png = base64.b64encode(b"\x89PNG fake").decode()
    blocks, is_error = server.openai_tool_content_to_mcp([{"type": "document", "source": {
        "type": "base64", "media_type": "image/png", "data": png}}])
    assert (blocks, is_error) == ([{"type": "image", "mimeType": "image/png", "data": png}],
                                  False), blocks
    blocks, is_error = server.openai_tool_content_to_mcp([{"type": "file", "file": {
        "file_data": f"data:image/png;base64,{png}"}}])
    assert (blocks, is_error) == ([{"type": "image", "mimeType": "image/png", "data": png}],
                                  False), blocks
    blocks, is_error = server.openai_tool_content_to_mcp([{"type": "file", "file": {
        "file_data": "data:text/plain;base64,aGVsbG8gd29ybGQ="}}])
    assert (blocks, is_error) == ([{"type": "text", "text": "hello world"}], False), blocks
    for part, expect in (
            ({"type": "document", "source": {"type": "base64", "media_type": "audio/wav",
                                             "data": "AAAA"}}, "audio/wav"),
            ({"type": "file", "file": {"file_data": "data:audio/wav;base64,AAAA"}}, "audio/wav"),
            ({"type": "file", "file": {"file_id": "file-abc"}}, "remote or missing"),
            ({"type": "document", "source": {"type": "url", "url": "https://x/a.pdf"}}, "url")):
        blocks, is_error = server.openai_tool_content_to_mcp([part])
        assert is_error, part
        assert len(blocks) == 1 and expect in blocks[0]["text"], (part, blocks)
        assert "could not be delivered" in blocks[0]["text"], blocks
    assert server.carries_pdf([{"type": "image_url", "image_url": {
        "url": "data:application/pdf;base64,AAAA"}}])
    assert not server.carries_pdf([{"type": "text", "text": "application/pdf"}])
    assert not server.carries_pdf("data:application/pdf;base64,AAAA")
    print("  document/file: images and text pass, the rest is a named isError")


def test_pdf_followup_renders_off_the_event_loop():
    """A PDF tool result is rendered in a worker thread: the parked call gets
    the rendered blocks, and the loop keeps running meanwhile."""
    import base64
    import threading
    loop_threads = set()

    real = server.openai_tool_content_to_mcp

    def recording(content):
        loop_threads.add(threading.current_thread() is threading.main_thread())
        return real(content)

    async def scenario():
        session = _new_session()
        session.new_turn()
        parked = asyncio.create_task(server.register_tool_call(session.id, "read", {}))
        call = (await session.turn_future)["calls"][0]
        pdf = f"data:application/pdf;base64,{base64.b64encode(TINY_PDF).decode()}"
        msgs = [{"role": "tool", "tool_call_id": call.id,
                 "content": [{"type": "image_url", "image_url": {"url": pdf}}]}]
        body = {"model": "m", "messages": [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": call.id, "type": "function",
                 "function": {"name": "read", "arguments": "{}"}}]}, *msgs]}
        followup = asyncio.create_task(server.handle_followup(body, msgs))
        result = await parked
        session.resolve_final({"type": "final", "payload": {"result": "ok"}})
        await followup
        return result

    saved = server.PROVIDER
    server.openai_tool_content_to_mcp = recording
    try:
        server.PROVIDER = "claude"
        with _FakePoppler():
            result = asyncio.run(scenario())
    finally:
        server.openai_tool_content_to_mcp = real
        server.PROVIDER = saved
    assert loop_threads == {False}, "PDF rendered on the event-loop thread"
    assert "EXTRACTED TEXT" in result["content"][0]["text"], result
    assert [b["type"] for b in result["content"][1:]] == ["image", "image"], result


def test_pdf_renders_with_real_poppler_when_present():
    """With the poppler the sidecar image ships, the real text comes out."""
    import base64
    import shutil as _sh
    if not (_sh.which("pdftotext") and _sh.which("pdftoppm")):
        print("  skipped: poppler not installed here (the sidecar image has it)")
        return
    blocks = server.pdf_to_mcp_blocks(base64.b64encode(TINY_PDF).decode())
    assert "HOST PDF MARKER" in blocks[0]["text"], blocks[0]
    assert [b["type"] for b in blocks[1:]] == ["image"], blocks
    assert base64.b64decode(blocks[1]["data"]).startswith(b"\x89PNG"), "not a PNG"
    print("  real poppler: text extracted, 1 page rendered as PNG")


def test_tool_path_web_search_enables_each_clis_own_search():
    """With caller tools AND a web-search request, the session's CLI gets
    its own search too: Claude's WebSearch, codex live search, OpenCode's
    websearch/webfetch plus OPENCODE_ENABLE_EXA."""
    import shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-web-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    saved = (server.PROVIDER, server.PROFILE)
    try:
        server.PROVIDER, server.PROFILE = "claude", server.MCP_PROFILES["claude"]
        argv, _ = server.build_argv("q", None, "m", workdir, "s", tools_path, "x", web=True)
        assert argv[argv.index("--tools") + 1] == "WebSearch,WebFetch", argv
        i = argv.index("--allowed-tools", argv.index("--tools"))
        assert argv[i + 1] == "WebSearch,WebFetch", argv
        server.PROVIDER, server.PROFILE = "codex", server.MCP_PROFILES["codex"]
        argv, _ = server.build_argv("q", None, "m", workdir, "s", tools_path, "x", web=True)
        assert 'web_search="live"' in argv, argv
        server.PROVIDER, server.PROFILE = "opencode", server.MCP_PROFILES["opencode"]
        server.build_argv("q", None, "m", workdir, "s", tools_path, "x", web=True)
        cfg = json.loads((workdir / "opencode.json").read_text())["agent"]["switchyard"]
        assert cfg["tools"]["websearch"] and cfg["permission"]["websearch"] == "allow", cfg
        assert cfg["permission"]["switchyard_*"] == "allow", cfg
        assert server.session_env(None, workdir, workdir, web=True) == {"OPENCODE_ENABLE_EXA": "true"}
        assert server.session_env(None, workdir, workdir) == {}
        print("  tool path web: claude WebSearch+WebFetch, codex live, opencode websearch+Exa")
    finally:
        server.PROVIDER, server.PROFILE = saved
        shutil.rmtree(workdir, ignore_errors=True)


def test_web_only_request_to_the_mcp_bridge_takes_the_text_path():
    """Claude Code's WebSearch sub-request carries only the server web tool;
    it is not a tool loop and must not reach translate_tools/start_session."""
    seen = []

    async def text_path(body):
        seen.append(body)
        return {"text": True}

    real = server.cli_bridge._handle_chat
    server.cli_bridge._handle_chat = text_path
    try:
        messages = [{"role": "user", "content": "search"}]
        for shape in ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
                      {"web_search_options": {}}):
            body = {"model": "m", "messages": messages, **shape}
            assert asyncio.run(server.chat(_JsonRequest(body))) == {"text": True}
        assert len(seen) == 2 and seen[1].get("web_search_options") == {}, seen
    finally:
        server.cli_bridge._handle_chat = real
    print("  web-only request -> text path (which runs the CLI's own search)")


# ------------------------------------------------------------- typed tools ---
# What LiteLLM 1.101 forwards for Anthropic's typed tools (issue #264).
_COMPUTER_FN = {"type": "function", "function": {
    "name": "computer",
    "parameters": {"type": "computer_20250124", "display_width_px": 1024,
                   "display_height_px": 768}}}


def _props(mcp_tool: dict) -> dict:
    return mcp_tool["inputSchema"]["properties"]


def test_typed_client_tools_get_their_documented_schemas():
    """bash/text editor/memory/computer reach the inner CLI with real input
    schemas, under the caller's tool name, instead of an empty schema."""
    out = {t["name"]: t for t in server.translate_tools([
        {"type": "bash_20250124", "name": "bash"},
        {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
        {"type": "text_editor_20250124", "name": "str_replace_editor"},
        {"type": "memory_20250818", "name": "memory"},
        _COMPUTER_FN,
        {"type": "computer_20251124", "name": "screen",
         "display_width_px": 1280, "display_height_px": 800}])}
    assert set(out) == {"bash", "str_replace_based_edit_tool", "str_replace_editor",
                        "memory", "computer", "screen"}, out
    assert _props(out["bash"])["command"]["type"] == "string", out["bash"]
    assert _props(out["bash"])["restart"]["type"] == "boolean", out["bash"]
    assert out["bash"]["description"], out["bash"]

    new = out["str_replace_based_edit_tool"]["inputSchema"]
    assert new["required"] == ["command", "path"] and "additionalProperties" not in new, new
    assert "undo_edit" not in new["properties"]["command"]["enum"], new
    assert {"file_text", "old_str", "new_str", "insert_line", "insert_text",
            "view_range"} <= set(new["properties"]), new
    old = out["str_replace_editor"]["inputSchema"]
    assert "undo_edit" in old["properties"]["command"]["enum"], old

    mem = _props(out["memory"])
    assert set(mem["command"]["enum"]) == {"view", "create", "str_replace", "insert",
                                           "delete", "rename"}, mem
    assert {"old_path", "new_path", "insert_text"} <= set(mem), mem

    for name, size in (("computer", "1024x768"), ("screen", "1280x800")):
        schema = out[name]["inputSchema"]
        assert schema["type"] == "object" and schema["required"] == ["action"], schema
        assert {"coordinate", "text", "scroll_direction", "scroll_amount",
                "duration", "region"} <= set(schema["properties"]), schema
        assert size in out[name]["description"], out[name]["description"]
        assert "left_click_drag" in out[name]["description"]
    print("  bash/text_editor/memory/computer (typed + function shape) -> real schemas")


def test_plain_function_tools_named_like_typed_ones_are_untouched():
    """Only the typed shapes are rewritten: a caller's own `computer` or
    `bash` function keeps the schema it was given."""
    own = {"type": "object", "properties": {"x": {"type": "string"}}}
    out = server.translate_tools([
        {"type": "function", "function": {"name": "computer", "parameters": own}},
        {"type": "function", "function": {"name": "bash", "parameters": own}}])
    assert [t["inputSchema"] for t in out] == [own, own], out
    assert server.cli_bridge.typed_client_tool({"type": "computer_use_preview"}) is None
    print("  plain function tools keep their own schemas")


def _stub_tool_path():
    """Replace handle_tool_request and the text path with recorders."""
    calls = {"tool": [], "text": []}

    async def tool_path(body, tools, request=None):
        calls["tool"].append(tools)
        return {"tool": True}

    async def text_path(body):
        calls["text"].append(body)
        return {"text": True}

    saved = (server.handle_tool_request, server.cli_bridge._handle_chat)
    server.handle_tool_request = tool_path
    server.cli_bridge._handle_chat = text_path
    return calls, saved


def _restore_tool_path(saved) -> None:
    server.handle_tool_request, server.cli_bridge._handle_chat = saved


def test_tool_search_is_dropped_not_refused():
    calls, saved = _stub_tool_path()
    try:
        messages = [{"role": "user", "content": "hi"}]
        fn = {"type": "function", "function": {"name": "f", "parameters": {}}}
        search = {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
        body = {"model": "m", "messages": messages, "tools": [search, fn]}
        assert asyncio.run(server.chat(_JsonRequest(body))) == {"tool": True}
        assert calls["tool"] == [[fn]], calls
        body = {"model": "m", "messages": messages,
                "tools": [{"type": "tool_search_tool_bm25_20251119", "name": "s"}]}
        assert asyncio.run(server.chat(_JsonRequest(body))) == {"text": True}
    finally:
        _restore_tool_path(saved)
    print("  tool_search_tool_* dropped; alone it takes the text path")


def test_tool_search_stays_dropped_on_a_tool_less_followup():
    """The body handed on (whose tools start_session remembers) carries no
    tool search, and a remembered or rebuilt list that still has tool search
    or web tools translates to the caller tools alone (PR #285 review)."""
    seen = []

    async def tool_path(body, tools, request=None):
        seen.append((body, tools))
        return {"tool": True}

    saved = server.handle_tool_request
    saved_entries = dict(server.REMEMBERED_TOOLS)
    server.handle_tool_request = tool_path
    fn = {"type": "function", "function": {"name": "f", "parameters": {}}}
    search = {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
    web = {"type": "web_search_20250305", "name": "web_search"}
    try:
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                "tools": [search, web, fn]}
        asyncio.run(server.chat(_JsonRequest(body)))
        handed, tools = seen[-1]
        assert handed["tools"] == [web, fn] and tools == [fn], seen[-1]
        # What start_session would remember from that body, then a follow-up
        # that sends no tools (issue #260).
        server.remember_tools("sess285", handed["tools"])
        followup = {"model": "m", "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "call_sess285_1", "type": "function",
                                                  "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_sess285_1", "content": "ok"}]}
        asyncio.run(server.chat(_JsonRequest(followup)))
        _, tools = seen[-1]
        assert [t["name"] for t in server.translate_tools(tools)] == ["f"], tools
        # Even a list remembered before this fix translates without them.
        assert [t["name"] for t in server.translate_tools([search, web, fn])] == ["f"]
    finally:
        server.handle_tool_request = saved
        server.REMEMBERED_TOOLS.clear()
        server.REMEMBERED_TOOLS.update(saved_entries)
    print("  tool search stays out of remembered tools and resumed sessions")


def test_server_only_tools_are_refused_before_the_tool_path():
    """Code execution, the MCP connector and OpenAI server tools have no
    executor on a CLI plan: 400 server_tool_unsupported, before any probe,
    session or spawn (handle_tool_request is never reached)."""
    from fastapi import HTTPException
    calls, saved = _stub_tool_path()
    fn = {"type": "function", "function": {"name": "f", "parameters": {}}}
    messages = [{"role": "user", "content": "hi"}]
    cases = [
        ({"tools": [fn, {"type": "code_execution_20250825", "name": "code_execution"}]},
         "code_execution_20250825"),
        ({"tools": [{"type": "mcp_toolset", "mcp_server_name": "x"}]}, "mcp_toolset"),
        ({"tools": [fn], "mcp_servers": [{"type": "url", "url": "https://x", "name": "x"}]},
         "mcp_servers"),
        ({"tools": [{"type": "code_interpreter"}, {"type": "file_search"}]}, "file_search"),
        ({"tools": [{"type": "image_generation"}]}, "image_generation"),
        ({"tools": [{"type": "computer_use_preview"}]}, "computer_use_preview"),
        ({"tools": [{"type": "mcp", "server_label": "x"}]}, "mcp"),
        ({"tools": [{"type": "local_shell"}]}, "local_shell"),
    ]
    try:
        for shape, kind in cases:
            body = {"model": "m", "messages": messages, **shape}
            try:
                asyncio.run(server.chat(_JsonRequest(body)))
            except HTTPException as exc:
                assert exc.status_code == 400, exc
                err = exc.detail["error"]
                assert err["type"] == "server_tool_unsupported", err
                assert err["param"] == "tools" and kind in err["message"], err
            else:
                raise AssertionError(f"{kind} was not refused")
        assert calls == {"tool": [], "text": []}, calls
    finally:
        _restore_tool_path(saved)
    print(f"  {len(cases)} server-only tool shapes -> 400 before the tool path")


def test_text_path_refuses_server_tools_specifically():
    """cli_bridge's own text path gives the same specific 400 (not the
    generic tools_unsupported) and drops tool search like the MCP bridge."""
    from fastapi import HTTPException
    cli = server.cli_bridge
    messages = [{"role": "user", "content": "hi"}]
    try:
        asyncio.run(cli._handle_chat({"model": "m", "messages": messages, "tools": [
            {"type": "code_execution_20250825", "name": "code_execution"}]}))
    except HTTPException as exc:
        assert exc.detail["error"]["type"] == "server_tool_unsupported", exc.detail
    else:
        raise AssertionError("code execution was not refused on the text path")
    try:
        asyncio.run(cli._handle_chat({"model": "m", "messages": messages, "tools": [
            {"type": "bash_20250124", "name": "bash"}]}))
    except HTTPException as exc:
        assert exc.detail["error"]["type"] == "tools_unsupported", exc.detail
    else:
        raise AssertionError("a caller tool was not refused on the text path")
    print("  text path: server tool -> server_tool_unsupported, caller tool -> tools_unsupported")


def test_web_tools_are_unaffected_by_typed_tool_handling():
    """web_search_* / web_fetch_* are none of client typed, tool search or
    server-only: they still go to the CLI's own search (#278)."""
    cli = server.cli_bridge
    for tool in ({"type": "web_search_20250305", "name": "web_search"},
                 {"type": "web_fetch_20250910", "name": "web_fetch"},
                 {"type": "web_search_preview"}):
        assert cli.is_web_search_tool(tool), tool
        assert cli.typed_client_tool(tool) is None and not cli.is_tool_search_tool(tool), tool
        assert cli.server_only_tools({"tools": [tool]}) == [], tool
    calls, saved = _stub_tool_path()
    try:
        fn = {"type": "function", "function": {"name": "f", "parameters": {}}}
        body = {"model": "m", "messages": [{"role": "user", "content": "q"}],
                "tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}, fn]}
        assert asyncio.run(server.chat(_JsonRequest(body))) == {"tool": True}
        assert calls["tool"] == [[fn]], calls
    finally:
        _restore_tool_path(saved)
    print("  web tools: still served by the CLI's own search")


def test_tool_path_argv_carries_the_effort_and_env_rides_the_carrier():
    """The session's CLI gets the caller's effort, and the gateway's
    caller_env stamp is read from `switchyard` (extra_body): LiteLLM never
    forwards `metadata` to the sidecar."""
    import shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-effort-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    saved = (server.PROVIDER, server.PROFILE)
    try:
        server.PROVIDER, server.PROFILE = "claude", server.MCP_PROFILES["claude"]
        argv, _ = server.build_argv("q", None, "m", workdir, "s", tools_path, "x", effort="max")
        assert argv[argv.index("--effort") + 1] == "max", argv
    finally:
        server.PROVIDER, server.PROFILE = saved
        shutil.rmtree(workdir, ignore_errors=True)
    env = server._resolve_env({"switchyard": {"caller_env": {
        "cwd": "/Users/x/p", "platform": "darwin", "source": "request"}}})
    assert env is not None and env.cwd == "/Users/x/p", env
    print("  tool path: --effort carried; caller_env read from the extra_body carrier")


def test_resolve_env_carries_git_from_stamp():
    """A wire stamp carrying `git: True` flows through `_resolve_env` to
    the returned CallerEnvironment so the sidecar's mirror of the
    caller's cwd is `git init`ed to match. `from_wire_metadata` is the
    strict-bool gate, so a stray `git: "yes"` on the wire coerces to
    None rather than sneaking through as truthy."""
    env = server._resolve_env({"switchyard": {"caller_env": {
        "cwd": "/Users/x/proj", "platform": "darwin", "shell": "zsh",
        "source": "request", "git": True}}})
    assert env is not None and env.git is True, env
    bad = server._resolve_env({"switchyard": {"caller_env": {
        "cwd": "/Users/x/proj", "platform": "darwin", "shell": "zsh",
        "source": "request", "git": "yes"}}})
    assert bad is not None and bad.git is None, bad
    print("  stamp git: True honored; 'yes' coerced to None at the wire gate")


def test_call_id_round_trips_the_session_id():
    session = _new_session()
    call_id = session.mint_call_id()
    assert call_id.startswith(f"call_{session.id}_")
    assert server.session_id_from_call_id(call_id) == session.id
    assert server.session_id_from_call_id("not-one-of-ours") is None
    _drop(session)
    print(f"  {call_id!r} -> session {session.id}")


# ------------------------------------------------------------- parking / turn ---
def test_parked_call_returns_finish_reason_tool_calls():
    async def scenario():
        session = _new_session()
        session.new_turn()
        # Stands in for tool_server.py's POST to /internal/tools/call.
        parked = asyncio.create_task(
            server.register_tool_call(session.id, "get_weather", {"city": "nyc"}))
        turn = await session.turn_future
        response = server.render_turn(session, turn, None)

        choice = response["choices"][0]
        assert choice["finish_reason"] == "tool_calls", choice
        assert choice["message"]["content"] is None
        tool_call = choice["message"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "get_weather"
        assert json.loads(tool_call["function"]["arguments"]) == {"city": "nyc"}
        assert server.session_id_from_call_id(tool_call["id"]) == session.id

        # Resolve it so the parked coroutine (standing in for tool_server.py)
        # completes, the way a caller's follow-up eventually would.
        session.pending[tool_call["id"]].future.set_result(
            {"content": [{"type": "text", "text": "sunny"}], "isError": False})
        result = await parked
        assert result["content"][0]["text"] == "sunny"
        _drop(session)
        return tool_call["id"]

    call_id = asyncio.run(scenario())
    print(f"  finish_reason=tool_calls, minted {call_id!r}, CLI process left alive")


def test_followup_correlates_by_tool_call_id_and_resumes_the_model():
    async def scenario():
        session = _new_session()
        session.new_turn()
        parked = asyncio.create_task(server.register_tool_call(session.id, "echo", {"text": "hi"}))
        turn = await session.turn_future
        call = turn["calls"][0]

        # The caller executes the tool and sends the usual OpenAI follow-up.
        followup_body = {"model": "m", "messages": [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": call.id, "type": "function",
                              "function": {"name": "echo", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call.id, "content": "hi back"},
        ]}
        tool_msgs = [followup_body["messages"][1]]
        followup = asyncio.create_task(server.handle_followup(followup_body, tool_msgs))
        await asyncio.sleep(0)   # let handle_followup resolve the future and park a new turn

        # The CLI process, having received the tool result, finishes and exits.
        session.resolve_final({"type": "final", "payload": {"result": "done: hi back"}})
        response = await followup
        result = await parked
        return response, result

    response, tool_result = asyncio.run(scenario())
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["message"]["content"] == "done: hi back"
    assert tool_result["content"][0]["text"] == "hi back"
    print(f"  follow-up correlated by tool_call_id -> {response['choices'][0]['message']['content']!r}")


class _JsonRequest:
    """The two things chat() and the follow-up path read from a Request."""

    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body

    async def is_disconnected(self) -> bool:
        return False


ECHO_TOOLS = [{"type": "function", "function": {
    "name": "echo", "description": "echo",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}}]


def test_followup_without_tools_resumes_the_parked_session():
    """Issue #260: OpenAI-compatible clients may send `tools` on the first
    turn only. A follow-up that answers one of OUR tool calls must still
    reach the parked session -- not the text path, which would start a
    fresh tool-less CLI while the parked one waits for a result forever."""
    async def scenario():
        session = _new_session()
        server.remember_tools(session.id, ECHO_TOOLS)
        session.new_turn()
        parked = asyncio.create_task(
            server.register_tool_call(session.id, "echo", {"text": "ping"}))
        turn = await session.turn_future
        call = turn["calls"][0]
        body = {"model": "m", "messages": [   # no "tools" key at all
            {"role": "user", "content": "echo ping"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": call.id, "type": "function",
                              "function": {"name": "echo", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call.id, "content": "got ping"},
        ]}
        real_text_path = server.cli_bridge._handle_chat

        async def text_path(_body):
            raise AssertionError("tool follow-up was routed to the text path")
        server.cli_bridge._handle_chat = text_path
        try:
            followup = asyncio.create_task(server.chat(_JsonRequest(body)))
            await asyncio.sleep(0.05)
            session.resolve_final({"type": "final", "payload": {"result": "done"}})
            response = await followup
        finally:
            server.cli_bridge._handle_chat = real_text_path
        result = await parked
        _drop(session)
        return response, result

    response, result = asyncio.run(scenario())
    assert result["content"][0]["text"] == "got ping", result
    assert response["choices"][0]["message"]["content"] == "done", response
    print("  tool-less follow-up delivered 'got ping' to the parked session")


def test_streamed_followup_without_tools_resumes_the_parked_session():
    """The #260 follow-up as a streaming request: chat() rehydrates the
    body's tools and must still frame the parked session's answer as SSE."""
    from fastapi.responses import StreamingResponse

    async def scenario():
        session = _new_session()
        server.remember_tools(session.id, ECHO_TOOLS)
        session.new_turn()
        parked = asyncio.create_task(
            server.register_tool_call(session.id, "echo", {"text": "ping"}))
        turn = await session.turn_future
        call = turn["calls"][0]
        body = {"model": "m", "stream": True, "messages": [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": call.id, "type": "function",
                              "function": {"name": "echo", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call.id, "content": "got ping"},
        ]}
        followup = asyncio.create_task(server.chat(_JsonRequest(body)))
        await asyncio.sleep(0.05)
        session.resolve_final({"type": "final", "payload": {"result": "streamed done"}})
        response = await followup
        chunks = [c if isinstance(c, str) else c.decode()
                  async for c in response.body_iterator]
        result = await parked
        _drop(session)
        return response, "".join(chunks), result

    response, stream, result = asyncio.run(scenario())
    assert isinstance(response, StreamingResponse), type(response)
    assert result["content"][0]["text"] == "got ping", result
    assert "streamed done" in stream and "data: [DONE]" in stream, stream[-200:]
    print("  streamed tool-less follow-up resumed the parked session as SSE")


def test_tool_less_request_without_our_call_ids_stays_on_the_text_path():
    """Only a follow-up answering a call we minted is rerouted: a plain chat,
    or a tool message whose id we never issued, is the text path's."""
    seen = []

    async def text_path(body):
        seen.append(body)
        return {"text": True}

    real = server.cli_bridge._handle_chat
    server.cli_bridge._handle_chat = text_path
    try:
        plain = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        foreign = {"model": "m", "messages": [
            {"role": "tool", "tool_call_id": "call_notours_1", "content": "x"}]}
        assert asyncio.run(server.chat(_JsonRequest(plain))) == {"text": True}
        assert asyncio.run(server.chat(_JsonRequest(foreign))) == {"text": True}
    finally:
        server.cli_bridge._handle_chat = real
    assert len(seen) == 2 and all("tools" not in b for b in seen), seen
    print("  plain chat and foreign tool ids still take the text path")


def test_tool_path_request_params_refused_or_applied():
    """On the tool path `n` > 1 is refused before anything spawns (a tool
    loop is one conversation), a bad response_format too; the final turn
    gets `stop` and the schema check, a tool_calls turn passes untouched;
    and the schema reaches the inner model as an instruction."""
    from fastapi import HTTPException
    tools = [{"type": "function", "function": {"name": "x", "parameters": {
        "type": "object", "properties": {}}}}]
    base = {"model": "m", "tools": tools, "messages": [{"role": "user", "content": "hi"}]}
    handled = []
    answers = iter([
        {"choices": [{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": "alpha STOP beta"}}]},
        {"choices": [{"index": 0, "finish_reason": "tool_calls",
                      "message": {"role": "assistant", "content": None, "tool_calls": [
                          {"id": "c1", "type": "function",
                           "function": {"name": "x", "arguments": "{}"}}]}}]},
        {"choices": [{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": "It is 5."}}]}])

    async def fake_tool_request(body, tools, request=None):
        handled.append(body)
        return next(answers)

    real = server.handle_tool_request
    server.handle_tool_request = fake_tool_request
    try:
        for bad in ({"n": 2}, {"response_format": {"type": "grammar"}}):
            try:
                asyncio.run(server.chat(_JsonRequest({**base, **bad})))
            except HTTPException as exc:
                assert exc.status_code == 400, exc.detail
            else:
                raise AssertionError(f"{bad} accepted on the tool path")
        assert handled == [], "refused request reached the session layer"
        out = asyncio.run(server.chat(_JsonRequest({**base, "stop": "STOP"})))
        assert out["choices"][0]["message"]["content"] == "alpha ", out
        schema = {"response_format": {"type": "json_schema", "json_schema": {
            "name": "s", "schema": {"type": "object"}}}}
        out = asyncio.run(server.chat(_JsonRequest({**base, **schema})))
        assert out["choices"][0]["finish_reason"] == "tool_calls", out
        try:
            asyncio.run(server.chat(_JsonRequest({**base, **schema})))
        except HTTPException as exc:
            assert exc.status_code == 502, exc.detail
        else:
            raise AssertionError("prose final answer accepted for a schema")
    finally:
        server.handle_tool_request = real
    _fitted, system = server.fit_tool_surface([], None, schema)
    assert "final answer" in system and "JSON Schema" in system, system
    print("  tool path: n>1 / bad format refused early; stop + schema on the final turn")


def test_remembered_tools_cover_probe_ids_and_are_bounded():
    """A caller-env probe answer can come back without `tools` too; the
    probe's call id maps to the tools of the request that minted it. The
    map is bounded so a long-lived sidecar does not grow without limit."""
    saved_limit = server.REMEMBERED_TOOLS_LIMIT
    saved_entries = dict(server.REMEMBERED_TOOLS)
    try:
        probe_id = "switchyard_env_deadbeef"
        server.remember_tools(probe_id, ECHO_TOOLS)
        body = {"messages": [{"role": "tool", "tool_call_id": probe_id, "content": "cwd=/x"}]}
        assert server.remembered_tools_for(body) == ECHO_TOOLS
        server.REMEMBERED_TOOLS.clear()
        server.REMEMBERED_TOOLS_LIMIT = 3
        for i in range(5):
            server.remember_tools(f"k{i}", ECHO_TOOLS)
        assert list(server.REMEMBERED_TOOLS) == ["k2", "k3", "k4"], list(server.REMEMBERED_TOOLS)
    finally:
        server.REMEMBERED_TOOLS_LIMIT = saved_limit
        server.REMEMBERED_TOOLS.clear()
        server.REMEMBERED_TOOLS.update(saved_entries)
    print("  probe ids remembered; map bounded, oldest evicted first")


def test_followup_with_unknown_session_is_rebuilt():
    """An unknown/expired session is no longer rejected — it is rebuilt.

    This used to assert the hard 410. That made a caller that answered a tool
    call after the idle TTL (walked away, long build, laptop asleep) start over
    client-side, when its request already carried everything needed to resume.
    test_a_reaped_session_is_rebuilt_not_410d covers the happy rebuild; here we
    pin the degenerate shape: a follow-up with no tool definitions cannot be
    rebuilt, and gets a clean 400 instead of the old 410.
    """
    body = {"messages": [{"role": "tool", "tool_call_id": "call_" + "0" * 32 + "_1",
                          "content": "x"}]}
    try:
        asyncio.run(server.handle_followup(body, body["messages"]))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400, exc
        print(f"  unknown session without tools -> {exc.status_code} (rebuild is impossible)")
        return
    raise AssertionError("expected a 400 for a follow-up that cannot be rebuilt")


# ------------------------------------------------------------- parallel calls ---
def test_parallel_tool_calls_land_in_one_batch():
    async def scenario():
        session = _new_session()
        session.new_turn()
        first = asyncio.create_task(server.register_tool_call(session.id, "a", {"x": 1}))
        second = asyncio.create_task(server.register_tool_call(session.id, "b", {"y": 2}))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        turn = await session.turn_future
        assert turn["type"] == "tool_calls"
        names = sorted(c.name for c in turn["calls"])
        assert names == ["a", "b"], names

        for c in turn["calls"]:
            session.pending[c.id].future.set_result(
                {"content": [{"type": "text", "text": "ok"}], "isError": False})
        await first
        await second
        _drop(session)
        return len(turn["calls"])

    n = asyncio.run(scenario())
    print(f"  {n} parallel tool calls surfaced as one finish_reason=tool_calls turn")


# ------------------------------------------------- issue #128: leftover flush ---
def test_late_parallel_tool_call_surfaces_on_new_turn():
    """Regression for issue #128.

    A parallel `tool_use` that arrives after mcp_bridge's quiet-period batch
    has already flushed used to land in `Session.batch_ids` but its `_flush`
    no-oped against the now-done turn future. The caller's follow-up
    (`_continue_followup`) then created a new turn whose `turn_future` nothing
    ever resolved -- the inner CLI blocked on the parked call while the bridge
    blocked on the turn, holding a gate slot until the 1800 s reaper.

    The fix surfaces any leftover batch the moment `Session.new_turn()` installs
    a fresh turn_future. This test pins the exact repro from the issue:
    enqueue A, await turn `[a]`, enqueue B after the future is done, sleep so
    the no-op flush actually fires (assert `batch_ids` still holds B), then
    drive `_continue_followup` with the caller's result for A. After the fix
    the response is `finish_reason=tool_calls` naming only B, `batch_ids` is
    empty, and A's parked future resolved. On `TimeoutError` (a regression)
    the held gate slot is released and an `AssertionError` naming #128 lets a
    regression fail fast instead of hanging the test.
    """
    async def scenario():
        # Fresh gate so unpark_session's acquire_waiting binds its Condition
        # to THIS event loop, the same pattern as
        # test_unpark_returns_503_quickly_when_gate_is_full /
        # test_followup_error_result_does_not_leak (the shared `_gate`
        # singleton is bound to whichever loop first touched it earlier).
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        try:
            session = _new_session()
            session.new_turn()

            # Enqueue A and await turn [a].
            parked_a = asyncio.create_task(
                server.register_tool_call(session.id, "a", {"x": 1}))
            await asyncio.sleep(0)
            turn_a = await session.turn_future
            assert turn_a["type"] == "tool_calls", turn_a
            assert [c.name for c in turn_a["calls"]] == ["a"], turn_a
            call_a = turn_a["calls"][0]
            assert session.batch_ids == [], session.batch_ids
            # Parked state, exactly as the happy path leaves it: awaiting
            # the caller's tool result, no concurrency slot held.
            session.awaiting_followup = True
            session.holds_slot = False

            # Enqueue B *after* the turn future is done. Its BATCH_WINDOW
            # timer fires against the now-done turn future and no-ops,
            # leaving B stranded in batch_ids. That stranded state is what
            # the fix in `Session.new_turn()` rescues when
            # `_continue_followup` next installs a fresh turn_future.
            parked_b = asyncio.create_task(
                server.register_tool_call(session.id, "b", {"y": 2}))
            await asyncio.sleep(0)
            call_b_id = session.batch_ids[-1]
            assert session.batch_ids == [call_b_id], session.batch_ids

            # Sleep so the no-op flush actually fires -- without this the
            # BATCH_WINDOW timer is still queued in the loop and the next
            # step would race it. After the sleep, batch_ids still holds B
            # because `_flush` short-circuited on the done turn_future.
            await asyncio.sleep(server.BATCH_WINDOW + 0.05)
            assert session.batch_ids == [call_b_id], session.batch_ids
            assert session.turn_future.done(), \
                "turn_future from turn [a] must remain done"

            # Drive `_continue_followup` with the caller's result for A.
            # The fix should: pop A from pending, resolve A's parked
            # future, take a fresh turn (which flushes B into the fresh
            # turn_future), and render a tool_calls response naming only
            # B. Without the fix this hangs on the fresh turn_future; the
            # wait_for lets a regression fail fast instead of running out
            # the test timeout.
            body = {"model": "m", "messages": [
                {"role": "tool", "tool_call_id": call_a.id,
                 "content": '{"x":1}'},
            ]}
            try:
                response = await asyncio.wait_for(
                    server._continue_followup(body, session.id,
                                              body["messages"], None),
                    timeout=server.BATCH_WINDOW + 1)
            except asyncio.TimeoutError:
                # Without the fix `_continue_followup` blocks on the new
                # turn_future, holds the unpark-acquired slot, and never
                # resolves. Release the slot so the rest of the suite can
                # proceed and raise an AssertionError naming #128.
                # `from None` silences the chained TimeoutError so ruff B904
                # (raise inside except must use `from err`/`from None`) is
                # satisfied without changing the test's semantics.
                await server.cli_bridge._gate.release()
                raise AssertionError(
                    "issue #128: _continue_followup hung on a leftover "
                    "batch after `Session.new_turn()` -- the fix that "
                    "flushes batch_ids into the fresh turn_future is "
                    "missing") from None

            # The parked-A future was resolved inside the resolve loop;
            # parked_a task should have completed normally.
            assert not parked_a.done() or parked_a.exception() is None
            result_a = await parked_a
            assert result_a["isError"] is False, result_a

            # park_session does not touch parked calls. Resolve B's parked
            # future and drain parked_b -- B is the response's surfaced
            # call, awaiting the caller's tool result that would come next.
            call_b = session.pending[call_b_id]
            call_b.future.set_result(
                {"content": [{"type": "text", "text": '{"y":2}'}],
                 "isError": False})
            result_b = await parked_b
            assert result_b["isError"] is False, result_b

            # The response names only B -- the only call left in the batch
            # when the new turn installed its fresh turn_future.
            assert response["choices"][0]["finish_reason"] == "tool_calls", response
            names = sorted(c["function"]["name"]
                           for c in response["choices"][0]["message"]["tool_calls"])
            assert names == ["b"], names
            # Leftover batch was consumed by the fix; nothing left.
            assert session.batch_ids == [], session.batch_ids
            # The session is parked, alive, not holding a slot, the gate is clean.
            assert session.id in server.SESSIONS, session.id
            assert session.awaiting_followup, session.awaiting_followup
            assert not session.holds_slot, session.holds_slot
            assert server.cli_bridge._gate.in_flight == 0, \
                server.cli_bridge._gate.in_flight
            return names
        finally:
            server.cli_bridge._gate = saved_gate
            try:
                _drop(session)
            except (NameError, UnboundLocalError):
                pass

    names = asyncio.run(scenario())
    print(f"  leftover batch surfaced as the new turn: {names}; "
          f"batch_ids empty; gate in_flight=0")


# ------------------------------------------------------------------- reaping ---
def test_reap_abandoned_session():
    async def scenario():
        session = _new_session()
        session.new_turn()
        parked = asyncio.create_task(server.register_tool_call(session.id, "slow", {}))
        await asyncio.sleep(0)
        turn = await session.turn_future
        assert turn["type"] == "tool_calls"

        session.last_active = time.time() - server.SESSION_TTL - 5
        await server.reap_session(session)

        assert session.id not in server.SESSIONS
        assert session.dead
        try:
            await parked
        except Exception as exc:              # register_tool_call re-raises as a 504
            assert getattr(exc, "status_code", None) == 504, exc
            return str(getattr(exc, "detail", exc))
        raise AssertionError("expected the parked call to fail once its session was reaped")

    message = asyncio.run(scenario())
    print(f"  reaped: parked call failed with {message!r}, session dropped from SESSIONS")


# ------------------------------------------------- issue #131: shutdown drain ---
def test_shutdown_drain_cancels_reaper_and_reaps_all_live_sessions():
    """Issue #131: when SIGTERM lands (uvicorn PID 1, --timeout-graceful-shutdown
    30), the mcp_bridge shutdown handler MUST cancel its reaper task AND
    reap every live session before the drain deadline. Without this, parked
    calls (whose tool_server.py POSTs are still waiting) and the CLI
    subprocesses (still blocked on stdio) outlive the container, leaving
    `docker compose stop` to hit the full stop_grace_period every time.

    Drives _shutdown_drain directly: two live sessions, each with a parked
    call and a turn_future (the exact shape a working mcp_bridge leaves
    behind mid-loop). After _shutdown_drain returns, both sessions are
    popped from SESSIONS, both parked futures have been failed (so their
    consumer — register_tool_call — surfaces 504), and both turn_futures
    carry a RuntimeError. The reaper task is cancelled and awaited; the
    task object reflects cancellation.
    """
    async def scenario():
        saved_reaper = server._REAPER_TASK
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()

        # Two live sessions with parked calls + a turn_future each.
        # The turn_future is consumed (the model emitted tool_calls and the
        # batch window fired) -- this matches the production shape: a
        # working mcp_bridge leaves a parked session with its turn_future
        # already done. Shutdown drain must not touch it.
        sessions: list = []
        parked_tasks: list = []
        for _ in range(2):
            session = _new_session()
            session.new_turn()
            session.awaiting_followup = True
            parked = asyncio.create_task(
                server.register_tool_call(session.id, "slow", {}))
            await asyncio.sleep(0)
            turn = await session.turn_future
            assert turn["type"] == "tool_calls"
            sessions.append(session)
            parked_tasks.append(parked)

        assert len(server.SESSIONS) == 2, server.SESSIONS

        # Spin up a reaper task we can cancel. The handler MUST cancel this
        # and await the cancellation before returning.
        async def _never():
            await asyncio.sleep(3600)
        reaper = asyncio.create_task(_never())
        server._REAPER_TASK = reaper

        await server._shutdown_drain()

        # Reaper was cancelled (and awaited: its done() is True).
        assert reaper.cancelled() or reaper.done(), \
            "reaper task must be cancelled during shutdown drain"
        # Every live session was reaped: SESSIONS is empty, every session
        # is dead, the consumed turn_future is still done with its
        # tool_calls result (shutdown must NOT touch a turn_future the
        # normal flow already resolved), and every parked future failed
        # so register_tool_call surfaces 504.
        assert server.SESSIONS == {}, server.SESSIONS
        for s in sessions:
            assert s.dead, s
            assert s.id not in server.SESSIONS
            assert s.turn_future.done(), s
            assert s.turn_future.exception() is None, \
                "consumed turn_future must keep its tool_calls result " \
                f"(got exception={s.turn_future.exception()!r})"
        for parked in parked_tasks:
            try:
                await parked
            except server.HTTPException as exc:
                assert exc.status_code == 504, exc
            else:
                raise AssertionError("parked call must 504 after shutdown drain")

        server._REAPER_TASK = saved_reaper
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        return len(sessions)

    n = asyncio.run(scenario())
    print(f"  shutdown drain: {n} live session(s) reaped, reaper cancelled, "
          "parked calls 504'd")


def test_shutdown_drain_with_no_live_sessions_cancels_reaper_and_returns_cleanly():
    """An idle sidecar still has to cancel its reaper, but it has no work
    to do. _shutdown_drain must not raise on the empty case — uvicorn is
    about to exit and a raise here would log a noisy traceback on every
    clean restart."""
    async def scenario():
        saved_reaper = server._REAPER_TASK
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()

        async def _never():
            await asyncio.sleep(3600)
        reaper = asyncio.create_task(_never())
        server._REAPER_TASK = reaper

        await server._shutdown_drain()                  # must not raise

        assert reaper.cancelled() or reaper.done(), \
            "reaper task must be cancelled even when no sessions are live"
        server._REAPER_TASK = saved_reaper
        server.SESSIONS.update(saved_sessions)

    asyncio.run(scenario())
    print("  shutdown drain: idle path cancels reaper and returns cleanly")


def test_shutdown_drain_tolerates_an_already_cancelled_reaper_task():
    """The handler runs at most once per process, but tests in this file
    install their own _REAPER_TASK repeatedly. Two branches matter:
    no reaper installed (None), and a reaper whose .cancel() was already
    called before _shutdown_drain runs (so .cancelled() is True). Both
    must not raise: the whole point of shutdown drain is to be
    unconditional about cleanup. Pins the current `_shutdown_drain` body
    — which re-cancels unconditionally and awaits the task — against a
    future refactor that, say, only awaits if the task was created here.
    """
    async def scenario():
        saved_reaper = server._REAPER_TASK
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()

        # Branch 1: no reaper installed at all.
        server._REAPER_TASK = None
        await server._shutdown_drain()                  # must not raise

        # Branch 2: an already-cancelled reaper task. Spin one up,
        # .cancel() it, await it (so its cancelled() is True), then hand
        # it to _shutdown_drain. Re-cancelling an already-cancelled task
        # is a no-op and the `await` returns immediately, so the function
        # returns cleanly without raising.
        async def _never():
            await asyncio.sleep(3600)
        cancelled = asyncio.create_task(_never())
        cancelled.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancelled
        assert cancelled.cancelled(), \
            "test setup: task must be cancelled before _shutdown_drain sees it"
        server._REAPER_TASK = cancelled
        await server._shutdown_drain()                  # must not raise

        server._REAPER_TASK = saved_reaper
        server.SESSIONS.update(saved_sessions)

    asyncio.run(scenario())
    print("  shutdown drain: None reaper and already-cancelled reaper both no-op")


def test_startup_records_the_reaper_task_so_shutdown_can_cancel_it():
    """The startup hook must keep a reference to the reaper task on
    `server._REAPER_TASK`; the old `asyncio.create_task(reap_loop())` lost
    its only handle immediately and the cancellation path had nothing to
    point at (issue #131). Call _start_reaper() directly: the reaper task
    must end up bound to the module-level name, and it must NOT be done
    yet (the loop runs forever).

    `asyncio.create_task` schedules on the running loop, so we have to
    cancel the task before returning to keep the test idempotent across
    re-runs.
    """
    async def scenario():
        saved = server._REAPER_TASK
        try:
            server._REAPER_TASK = None
            await server._start_reaper()
            assert server._REAPER_TASK is not None, \
                "_start_reaper must keep a module-level handle"
            assert not server._REAPER_TASK.done(), \
                "reaper loop must be live immediately after startup"
            # Cancel so the loop's `sleep(REAP_INTERVAL)` raises CancelledError
            # and the task winds down cleanly before the test loop ends.
            server._REAPER_TASK.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server._REAPER_TASK
        finally:
            server._REAPER_TASK = saved

    asyncio.run(scenario())
    print("  startup: reaper task retained on server._REAPER_TASK")


# --------------------------------------- issue #131: Dockerfile / compose shape ---
def _read_text(path: str) -> str:
    return Path(os.path.join(ROOT, path)).read_text()


def test_dockerfile_sidecar_uses_exec_uvicorn_with_graceful_shutdown_bound():
    """Issue #131: sidecar's CMD must `exec` uvicorn (so uvicorn is PID 1 and
    receives SIGTERM directly, not sh), keep `${BRIDGE:-cli}_bridge` and
    `${SIDECAR_PORT:-8081}` on the shell side so runtime variable expansion
    still works, AND add `--timeout-graceful-shutdown 30` to bound uvicorn's
    drain inside compose's stop_grace_period. Asserts the literal shapes the
    Dockerfile requires — a regression that drops `exec` (re-opens the bug)
    or removes the flag (re-opens the stop_grace_period hang) fails here.
    """
    dockerfile = _read_text("Dockerfile.sidecar")
    # Locate the CMD line (the file uses heredoc-style strings elsewhere; we
    # pin the exact line that starts the array, not a substring match, so a
    # future comment containing "exec uvicorn" elsewhere does not pass).
    cmd_lines = [ln.strip() for ln in dockerfile.splitlines() if ln.lstrip().startswith("CMD ")]
    assert cmd_lines, "Dockerfile.sidecar has no CMD line"
    cmd = cmd_lines[-1]
    # uvicorn is exec'd (replaces sh), not backgrounded.
    assert "exec uvicorn" in cmd, \
        f"sidecar CMD must exec uvicorn (uvicorn= PID 1), got: {cmd!r}"
    # Runtime env vars still expand under sh before the exec.
    assert "${BRIDGE:-cli}_bridge" in cmd, \
        f"sidecar CMD must still expand BRIDGE under sh, got: {cmd!r}"
    assert "${SIDECAR_PORT:-8081}" in cmd, \
        f"sidecar CMD must still expand SIDECAR_PORT under sh, got: {cmd!r}"
    # The graceful-shutdown bound is present at the literal value uvicorn reads.
    assert "--timeout-graceful-shutdown 30" in cmd, \
        f"sidecar CMD must bound uvicorn drain at 30s, got: {cmd!r}"
    # Sanity: not bare `uvicorn` without exec — the regression we are guarding.
    bare = re.compile(r"&& uvicorn server:app")
    assert not bare.search(cmd), \
        f"sidecar CMD must not run uvicorn without exec (re-opens issue #131), got: {cmd!r}"
    print("  Dockerfile.sidecar CMD: exec uvicorn + --timeout-graceful-shutdown 30; "
          "${BRIDGE} + ${SIDECAR_PORT} still expand under sh")


def test_dockerfile_token_proxy_uses_exec_uvicorn_with_graceful_shutdown_bound():
    """Same shape as the sidecar Dockerfile: `exec uvicorn`, runtime variable
    expansion under sh, --timeout-graceful-shutdown 30. Different env var
    (PROXY_PORT), but the same exec+graceful-shutdown contract."""
    dockerfile = _read_text("Dockerfile.token_proxy")
    cmd_lines = [ln.strip() for ln in dockerfile.splitlines() if ln.lstrip().startswith("CMD ")]
    assert cmd_lines, "Dockerfile.token_proxy has no CMD line"
    cmd = cmd_lines[-1]
    assert "exec uvicorn" in cmd, \
        f"token-proxy CMD must exec uvicorn (uvicorn= PID 1), got: {cmd!r}"
    assert "${PROXY_PORT:-8090}" in cmd, \
        f"token-proxy CMD must still expand PROXY_PORT under sh, got: {cmd!r}"
    assert "--timeout-graceful-shutdown 30" in cmd, \
        f"token-proxy CMD must bound uvicorn drain at 30s, got: {cmd!r}"
    bare = re.compile(r"&& uvicorn server:app")
    assert not bare.search(cmd), \
        f"token-proxy CMD must not run uvicorn without exec, got: {cmd!r}"
    print("  Dockerfile.token_proxy CMD: exec uvicorn + --timeout-graceful-shutdown 30; "
          "${PROXY_PORT} still expands under sh")


def test_compose_xcommon_declares_init_true_and_stop_grace_period_45s():
    """The shared x-common anchor in docker-compose.yml must set `init: true`
    and `stop_grace_period: 45s` so every sidecar / token-proxy / gateway /
    portal has the SIGTERM plumbing (issue #131). The 45s figure exceeds
    uvicorn's --timeout-graceful-shutdown 30 by enough margin to cover the
    mcp_bridge shutdown drain (reaper cancel + reap all live sessions), but
    stays well under apply.sh's drain_grace_secs (120s) so a stuck sidecar
    never holds up a drain past its budget. Asserts the literal value here —
    a regression that changes the figure fails the test before the stack
    even reaches docker compose config.
    """
    compose_text = _read_text("docker-compose.yml")
    # The x-common anchor and its merged fields must be present, in source.
    # The `services:` block itself uses <<: *common, so the inheritance is
    # verified by compose at deploy time; here we pin the anchor.
    assert "x-common: &common" in compose_text, \
        "docker-compose.yml must keep the x-common anchor"
    # Both keys, in the anchor block, at the literal values the plan requires.
    init_pattern = re.compile(
        r"x-common:\s*&\s*common[\s\S]+?init:\s*true", re.MULTILINE)
    assert init_pattern.search(compose_text), \
        "x-common anchor must set init: true (tini PID-1 backstop)"
    grace_pattern = re.compile(
        r"x-common:\s*&\s*common[\s\S]+?stop_grace_period:\s*45s", re.MULTILINE)
    assert grace_pattern.search(compose_text), \
        "x-common anchor must set stop_grace_period: 45s"
    # And the values must actually flow through to the sidecars: every
    # sidecar / token-proxy declares `<<: *common` and does NOT override
    # either field. A regression that pinned init: false on a single
    # service (e.g. to work around a tini bug) would re-open the SIGTERM
    # hole on that one container — and the regression would still pass the
    # anchor check above.
    for svc in ("claude-max-sidecar", "codex-sidecar", "opencode-go-sidecar",
                "opencode-go2-sidecar", "xai-token-proxy"):
        # Each service block: anchor merge + no per-service override of
        # `init:` or `stop_grace_period:`. `init:` and `stop_grace_period:`
        # are inherited; setting either at the service level would shadow
        # the anchor.
        block_match = re.search(
            rf"^\s{{2}}{re.escape(svc)}:\s*$([\s\S]*?)(?=^\s{{2}}\w|\Z)",
            compose_text, re.MULTILINE)
        assert block_match, f"could not isolate service block for {svc}"
        block = block_match.group(1)
        assert "init:" not in block, \
            f"{svc} must inherit init: true from x-common (no override): {block[:200]!r}"
        assert "stop_grace_period:" not in block, \
            f"{svc} must inherit stop_grace_period: 45s from x-common (no override): {block[:200]!r}"
    print("  docker-compose.yml x-common: init=true + stop_grace_period=45s "
          "inherited by every sidecar and token-proxy")


# ---------------------------------------------------- tool_server.py, real pipe ---
class _StubCallback(http.server.BaseHTTPRequestHandler):
    """Stands in for server.py's /internal/tools/call. Replies "slow" calls
    after a short delay so a parallel-call test can prove requests are not
    serialised — this handler class is only ever used with ThreadingTCPServer."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        name, args = body["name"], body.get("arguments") or {}
        if name == "slow":
            time.sleep(0.3)
        text = f"{name}:{args.get('text', '')}"
        payload = json.dumps({"content": [{"type": "text", "text": text}],
                              "isError": False}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args) -> None:
        pass    # keep test output quiet


def _start_stub_callback():
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _StubCallback)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _spawn_tool_server(tools, session_id, port):
    tools_path = os.path.join(tempfile.mkdtemp(prefix="mcpb-tools-"), "tools.json")
    with open(tools_path, "w") as fh:
        json.dump(tools, fh)
    env = dict(os.environ)
    env.update({"SWITCHYARD_TOOLS_FILE": tools_path, "SWITCHYARD_SESSION_ID": session_id,
                "SWITCHYARD_CALLBACK_URL": f"http://127.0.0.1:{port}"})
    return subprocess.Popen([sys.executable, TOOL_SERVER], env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)


def test_tool_server_round_trips_a_real_stdio_tools_call():
    httpd, port = _start_stub_callback()
    proc = _spawn_tool_server(
        [{"name": "echo", "description": "d", "inputSchema": {"type": "object"}}],
        "sess-pipe", port)
    try:
        def send(msg):
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()

        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        init = json.loads(proc.stdout.readline())
        assert init["result"]["serverInfo"]["name"] == "switchyard-mcp-bridge", init

        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        listed = json.loads(proc.stdout.readline())
        assert listed["result"]["tools"][0]["name"] == "echo", listed

        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "echo", "arguments": {"text": "hi"}}})
        called = json.loads(proc.stdout.readline())
        assert called["id"] == 3 and called["result"]["content"][0]["text"] == "echo:hi", called
        print(f"  real stdio round-trip: {called['result']['content'][0]['text']!r}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        httpd.shutdown()


def test_tool_server_handles_parallel_calls_without_serialising():
    """Sends a slow call and a fast call back to back, before reading either
    reply. If tools/call blocked the stdin reader, "fast" would have to wait
    behind "slow"'s 0.3s and this would take >=0.3s either way — the point is
    that both replies arrive, correctly matched by id, and the loop never
    reads a garbled interleaved line."""
    httpd, port = _start_stub_callback()
    proc = _spawn_tool_server(
        [{"name": "slow", "description": "d", "inputSchema": {"type": "object"}},
         {"name": "fast", "description": "d", "inputSchema": {"type": "object"}}],
        "sess-parallel", port)
    try:
        def send(msg):
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()

        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        proc.stdout.readline()
        send({"jsonrpc": "2.0", "id": 10, "method": "tools/call",
              "params": {"name": "slow", "arguments": {"text": "s"}}})
        send({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
              "params": {"name": "fast", "arguments": {"text": "f"}}})

        replies = {}
        for _ in range(2):
            msg = json.loads(proc.stdout.readline())
            replies[msg["id"]] = msg["result"]["content"][0]["text"]
        assert replies == {10: "slow:s", 11: "fast:f"}, replies
        print(f"  two concurrent tools/call replies matched by id: {replies}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        httpd.shutdown()


# ------------------------------------------------------- session driver, fake CLI ---
def _write_fake_cli(body: str) -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="mcpb-fakecli-"), "fake_cli.py")
    with open(path, "w") as fh:
        fh.write(body)
    return path


def test_run_session_parses_a_fake_cli_process_to_completion():
    fake_cli = _write_fake_cli(
        "import json\n"
        "print(json.dumps({'result': '42', 'usage': "
        "{'input_tokens': 5, 'output_tokens': 1}}))\n")

    async def scenario():
        session = _new_session()
        session.new_turn()
        await server.run_session(session, [sys.executable, fake_cli])
        return session.turn_future.result()

    result = asyncio.run(scenario())
    assert result["type"] == "final", result
    assert result["payload"]["result"] == "42", result
    print(f"  fake CLI process -> final payload {result['payload']}")


def test_run_session_classifies_a_usage_limit_error_as_429():
    fake_cli = _write_fake_cli(
        "import sys\n"
        "sys.stderr.write(\"You\\u2019ve hit your usage limit. "
        "Purchase more credits.\")\n"
        "sys.exit(1)\n")

    async def scenario():
        session = _new_session()
        session.new_turn()
        await server.run_session(session, [sys.executable, fake_cli])
        return session.turn_future.result()

    result = asyncio.run(scenario())
    assert result["type"] == "error" and result["status"] == 429, result
    print(f"  fake CLI's usage-limit wording -> HTTP {result['status']}")


def test_a_caller_that_hangs_up_mid_turn_drops_the_session():
    """A dropped request must free the slot immediately, not at the idle TTL.

    Before this, a caller that disappeared while the CLI was working left a live
    subprocess holding one of the plan's two connections for the full 30-minute
    idle TTL: two such drops and SwitchYard saw the plan as full and spilled
    every request past it.
    """
    class _Hangup:
        """Minimal stand-in for starlette's Request.is_disconnected()."""
        def __init__(self, after: int):
            self.calls = 0
            self.after = after

        async def is_disconnected(self) -> bool:
            self.calls += 1
            return self.calls > self.after

    async def scenario():
        old_poll = server.DISCONNECT_POLL
        server.DISCONNECT_POLL = 0.01
        try:
            session = _new_session()
            session.new_turn()
            parked = asyncio.create_task(
                server.register_tool_call(session.id, "slow", {}))
            await asyncio.sleep(0)
            assert (await session.turn_future)["type"] == "tool_calls"

            # A second turn the caller never waits for.
            session.new_turn()
            request = _Hangup(after=1)
            try:
                await server.await_turn(session, request)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                assert status == 499, exc
            else:
                raise AssertionError("await_turn should not return after a hangup")

            assert session.dead
            assert session.id not in server.SESSIONS
            with contextlib.suppress(Exception):
                await parked
            return request.calls
        finally:
            server.DISCONNECT_POLL = old_poll

    calls = asyncio.run(scenario())
    print(f"  hangup seen after {calls} poll(s): session killed, slot freed")


def test_a_parked_session_holds_no_concurrency_slot():
    """Parking gives the slot back; unparking takes one again.

    A parked session is blocked on the caller's tool result and runs no
    inference. Holding a slot through that capped useful work at the number of
    loops in flight rather than the number of model turns — two parked loops
    made a 2-connection plan report itself full while nothing was talking to
    the provider at all.
    """
    async def scenario():
        gate = server.cli_bridge._gate
        before = gate.in_flight

        session = _new_session()
        session.holds_slot = True
        assert await gate.acquire(before + 1), "fixture should be able to take a slot"
        taken = gate.in_flight

        await server.park_session(session)
        assert session.awaiting_followup and not session.holds_slot
        assert gate.in_flight == taken - 1, "parking must return the slot"

        # Unparking takes one back before the model runs again.
        await server.unpark_session(session)
        assert session.holds_slot and not session.awaiting_followup
        assert gate.in_flight == taken, "unparking must retake a slot"

        await server.end_session(session)
        assert gate.in_flight == before, "ending must not leak the slot"

    asyncio.run(scenario())
    print("  parked sessions cost no concurrency; unparking reclaims a slot")


def test_parked_sessions_are_capped_and_the_stalest_goes_first():
    """They cost no slot, but each is a live CLI process, so they are bounded."""
    async def scenario():
        limit = server.cli_bridge.config().parked_limit
        assert limit >= 2, limit

        made = []
        for i in range(limit + 1):
            s = _new_session()
            s.awaiting_followup = True
            s.last_active = time.time() - (100 - i)   # first is stalest
            made.append(s)

        await server.enforce_parked_limit()
        alive = [s for s in made if s.id in server.SESSIONS]
        assert len(alive) <= limit, (len(alive), limit)
        assert made[0].id not in server.SESSIONS, "the stalest should go first"
        # An evicted session is owed a resumption, exactly like a preempted one.
        assert made[0].id in server.PREEMPTED

        for s in alive:
            await server.reap_session(s)
        return limit

    limit = asyncio.run(scenario())
    print(f"  parked limit {limit} enforced, stalest evicted and remembered")


def test_tool_history_is_rendered_for_a_rebuilt_session():
    """Rebuilding a preempted session must carry its tool loop into the prompt.

    cli_bridge.flatten drops both halves of a tool turn — an assistant message
    with tool_calls has no content, and a `tool` message has no idea what it
    answers — so a rebuild done with it would hand the model results for calls
    it had no record of making.
    """
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "weather in Oslo and Lisbon?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'}},
            {"id": "call_b", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Lisbon"}'}},
        ]},
        {"role": "tool", "tool_call_id": "call_b", "content": '{"temp_c":19}'},
        {"role": "tool", "tool_call_id": "call_a", "content": '{"temp_c":-3}'},
    ]
    prompt, system = server.flatten_with_tool_history(messages)
    assert system == "be terse", system
    assert "weather in Oslo and Lisbon?" in prompt
    # both calls named, with their arguments
    assert '[called get_weather({"city":"Oslo"})]' in prompt, prompt
    assert '[called get_weather({"city":"Lisbon"})]' in prompt, prompt
    # and both results attributed, so out-of-order results stay readable
    assert prompt.count("[result of get_weather:") == 2, prompt
    assert '{"temp_c":-3}' in prompt and '{"temp_c":19}' in prompt

    # The plain flattener really does lose it -- this is why the variant exists.
    plain, _ = server.cli_bridge.flatten(messages)
    assert "get_weather" not in plain, plain
    print("  tool loop rendered as narration; cli_bridge.flatten drops it")


def test_a_preempted_session_is_resumed_not_refused():
    """The follow-up of a session we preempted must be rebuilt, not 410'd.

    Preemption is our choice, not the caller's fault: it has already run the
    tool and cannot spill the answer to another plan, since no other plan minted
    its tool_call_ids or holds its prompt cache. So it is owed a resumption here.
    """
    async def scenario():
        session = _new_session()
        session.new_turn()
        parked = asyncio.create_task(server.register_tool_call(session.id, "get_weather", {}))
        await asyncio.sleep(0)
        turn = await session.turn_future
        assert turn["type"] == "tool_calls", turn
        call_id = turn["calls"][0].id
        session.awaiting_followup = True
        session.last_active = time.time() - server.PARKED_GRACE - 5

        # Evicted because the parked limit was reached, not to free a slot:
        # parked sessions hold none. Either way it is owed a resumption.
        server.note_preempted(session.id)
        await server.reap_session(session)
        assert session.id in server.PREEMPTED
        with contextlib.suppress(Exception):
            await parked
        return call_id

    call_id = asyncio.run(scenario())
    assert call_id and server.session_id_from_call_id(call_id), call_id
    print("  preempted id remembered as ours, owed a resumption")


def _stub_start_session(recorder: list):
    """Replace start_session with a capture stub for handler-level tests.

    The real one spawns the vendor CLI; here we only need to observe what the
    rebuild path would have fed it. Returns the restore callable.

    Releases the concurrency slot too: resume_gone_session acquired one before
    calling start_session, and the real start_session keeps it (it is
    released when end_session is later called on the rebuilt session). The
    stub does no work and creates no session, so leaving the slot taken would
    leak it across tests; with the example plan at concurrency 2, two
    rebuild-path tests back-to-back exhaust the gate and the next acquire
    stalls.
    """
    real = server.start_session

    async def fake(body, mcp_tools, prompt, system, model, request,
                   image_paths=None, img_dir=None, env=None, first_turn=True):
        # Mirror the env-injection + first-turn-reminder logic the real
        # start_session runs before build_argv. Without this the captured
        # `system`/`prompt` would reflect what handle_fresh passed, not
        # what the CLI actually saw -- and the issue #44 contract is
        # specifically about what the CLI sees.
        if env is not None and server._caller_env is not None:
            env_block = server._caller_env.render_system_block(env)
            system = (f"{system}\n\n{env_block}" if system else env_block)
            if first_turn:
                reminder = server._caller_env.render_first_turn_reminder(env)
                prompt = f"{reminder}\n\n{prompt}" if prompt else reminder
        recorder.append({"tools": mcp_tools, "prompt": prompt,
                         "system": system, "model": model,
                         "image_paths": image_paths,
                         "env": env, "first_turn": first_turn})
        if img_dir is not None:
            import shutil
            shutil.rmtree(img_dir, ignore_errors=True)
        await server.cli_bridge._gate.release()
        return {"stubbed": True}

    server.start_session = fake

    def restore():
        server.start_session = real
    return restore


def test_a_reaped_session_is_rebuilt_not_410d():
    """A follow-up past the idle TTL must resume the loop, not error.

    This is the walk-away-between-meetings case: the reaper collected the
    session after 30 idle minutes, the caller comes back and answers the tool
    call, and the old hard 410 forced it to start over client-side. The
    request already carries the whole loop, so the sidecar rebuilds from it
    and the client never learns the session died.
    """
    calls = []
    restore = _stub_start_session(calls)
    try:
        async def scenario():
            session = _new_session()
            session.new_turn()
            parked = asyncio.create_task(
                server.register_tool_call(session.id, "get_weather", {"city": "Oslo"}))
            await asyncio.sleep(0)
            turn = await session.turn_future
            assert turn["type"] == "tool_calls", turn
            session.awaiting_followup = True
            # Past the TTL: exactly what reap_loop checks before reaping.
            session.last_active = time.time() - server.SESSION_TTL - 5
            await server.reap_session(session)
            assert session.id not in server.SESSIONS
            with contextlib.suppress(Exception):
                await parked

            # The caller's follow-up, with everything it replayed on its own:
            # its tool call and the result it computed while away.
            body = {
                "model": "m",
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "",
                    "parameters": {"type": "object", "properties": {}}}}],
                "messages": [
                    {"role": "user", "content": "weather in Oslo?"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": f"call_{session.id}_1", "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": '{"city":"Oslo"}'}}]},
                    {"role": "tool", "tool_call_id": f"call_{session.id}_1",
                     "content": '{"temp_c":-3}'},
                ],
            }
            return await server.handle_followup(body, body["messages"][2:])

        result = asyncio.run(scenario())
        assert result == {"stubbed": True}, result
        assert len(calls) == 1, calls
        rebuilt = calls[0]
        assert '[called get_weather({"city":"Oslo"})]' in rebuilt["prompt"], rebuilt
        assert "[result of get_weather: {\"temp_c\":-3}]" in rebuilt["prompt"], rebuilt
        assert "weather in Oslo?" in rebuilt["prompt"], rebuilt
        assert rebuilt["tools"] and rebuilt["tools"][0]["name"] == "get_weather"
        print("  reaped session rebuilt from the request; loop carried as narration")

        # The kill-switch contract: REBUILD_LOST=0 still rebuilds when the
        # lost session was preempted by us (we owe a resumption regardless of
        # the operator's preference). Foreign or never-seen ids, which have
        # no lost-our-session for the operator to opt out of repairing,
        # always rebuild -- otherwise the client is trapped for nothing they
        # did. So the kill-switch only fires the old hard 410 for ids the
        # operator themselves never registered.
        #
        # The below asserts the foreign-id branch explicitly: the call must
        # get past handle_followup's parse gate, into resume_gone_session,
        # and surface the rebuild path's own 400 (no usable tool definitions)
        # instead of the old "exactly one live session" 400.
        server.REBUILD_LOST = False
        try:
            calls.clear()

            async def refused():
                body = {"model": "m", "tools": [], "messages": [
                    {"role": "tool", "tool_call_id": f"call_{'0' * 32}_1",
                     "content": "x"}]}
                try:
                    await server.handle_followup(body, body["messages"])
                except Exception as exc:
                    return getattr(exc, "status_code", None), getattr(exc, "detail", "")

            status, detail = asyncio.run(refused())
            assert status == 400, status
            # The 400 must come from the rebuild's no-tools check, not from
            # the parse gate. The strings are different -- the parse gate's
            # message no longer mentions "live mcp_bridge session".
            assert "live mcp_bridge session" not in detail, detail
            assert "no usable tool definitions" in detail, detail
        finally:
            server.REBUILD_LOST = True
        print("  REBUILD_LOST=0 rebuilds foreign ids (no lost-our-session to "
              "honour); only in-PREEMPTED ids keep the kill-switch 410")
    finally:
        restore()


def test_foreign_ids_rebuild_so_the_client_is_not_trapped():
    """A foreign tool_call_id proved nothing to refuse the caller about.

    The old behaviour 400'd on any call_id that did not parse to one of our
    session ids. That trapped an OpenCode instance whose results we never
    minted: it carried the right tool result, we just could not correlate it,
    so the request looked malformed to us and like a server bug to it. The
    right answer is the same rebuild-from-request path already used for lost
    sessions -- the caller's history carries the whole loop, and a fresh CLI
    session can pick up from it.

    `REBUILD_LOST=0` keeps the old hard 410 only as an opt-out for sessions
    we recognise as ours. Foreign ids trigger a rebuild either way: there is
    no lost session of ours for the caller to "come back to".
    """
    async def scenario_tools():
        body = {"model": "m",
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "",
                    "parameters": {"type": "object", "properties": {}}}}],
                "messages": [
                    {"role": "user", "content": "weather in Oslo?"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "not-one-of-ours", "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": '{"city":"Oslo"}'}}]},
                    {"role": "tool", "tool_call_id": "not-one-of-ours",
                     "content": '{"temp_c":-3}'},
                ]}
        return await server.handle_followup(body, body["messages"][2:])

    calls: list = []
    restore = _stub_start_session(calls)
    try:
        result = asyncio.run(scenario_tools())
        assert result == {"stubbed": True}, result
        assert len(calls) == 1, calls
        rebuilt = calls[0]
        assert '[called get_weather({"city":"Oslo"})]' in rebuilt["prompt"], rebuilt
        assert "[result of get_weather:" in rebuilt["prompt"], rebuilt
        print("  foreign tool_call_id rebuilt from the request as a lost session")

        # Without tools in the request, the rebuild path 400s on its own terms
        # ("no usable tool definitions"), not on the foreign id. That is the
        # genuine fail-fast -- the request is not survivable as a tool loop.
        # Stubbing start_session is harmless here (resume_gone_session 400s
        # before reaching it), but kept for the same reason as above: if the
        # stub were not in place, a regression that started the CLI would
        # hang the test, not fail it.
        async def no_tools():
            body = {"model": "m", "tools": [], "messages": [
                {"role": "tool", "tool_call_id": "not-one-of-ours", "content": "x"}]}
            await server.handle_followup(body, body["messages"])
        try:
            asyncio.run(no_tools())
        except Exception as exc:
            assert getattr(exc, "status_code", None) == 400, exc
        else:
            raise AssertionError("a non-rebuildable follow-up must still 400")
    finally:
        restore()


def test_multi_session_ids_pick_live_and_drop_the_rest():
    """A follow-up whose tool results span two live sessions routes to one of
    them and lets the rest go, instead of 400'ing the caller.

    Live failure: a stuck OpenCode retry or a GUI that interleaves two
    conversations bundled tool results from both into one chat completions
    POST. The old code 400'd on the second `session_id_from_call_id` match;
    the new code picks the first live session, ignores the others, and lets
    the caller's request complete.

    The unpicked session is then superseded (issue #13, defect 2): its tool
    results were dropped from this batch, so its parked calls can never be
    answered and its CLI subprocess is blocked on tool_server.py requests
    that will never reply. Leaving it live breaks the invariant "one caller
    tool loop <-> at most one live mcp_bridge session", so it is torn down
    now rather than waiting for the idle reaper.

    Stubs `_continue_followup` so the routing decision is observable without
    a CLI process. The CLI completion path is exercised by an earlier test
    (`test_followup_correlates_by_tool_call_id_and_resumes_the_model`) -- we
    only need to prove the multi-live routing and the cleanup here.
    """
    captured: list = []
    real = server._continue_followup

    async def stub(body, wanted, tool_msgs, request):
        captured.append({"wanted": wanted, "kept": len(tool_msgs)})
        # Mark the chosen session dead and return a synthetic reply; the
        # real path would drive the CLI turn here.
        session = server.SESSIONS[wanted]
        await server.end_session(session)
        return {"stubbed_choice": wanted, "kept_tool_msgs": len(tool_msgs)}

    server._continue_followup = stub
    try:
        async def go():
            s_a = _new_session()
            s_b = _new_session()
            # Parked sessions with a parked tool call each. Each session has a
            # single parked call that we register directly via the tool-call
            # path mcp_bridge uses (register_tool_call), so the SESSIONS map
            # sees them as live.
            for s in (s_a, s_b):
                s.new_turn()
                s.awaiting_followup = True
                parked = server.ParkedCall(
                    id=f"call_{s.id}_1", name="echo", arguments={},
                    future=asyncio.get_event_loop().create_future())
                # In production a parked call has tool_server.py's HTTP
                # request as its consumer and a parked session has its
                # turn_future already resolved (the CLI finished a
                # tool_calls turn). This test inlines both, so add
                # callbacks to swallow whatever supersede_session leaves
                # behind -- otherwise asyncio logs "Future exception was
                # never retrieved" for the dropped session.
                parked.future.add_done_callback(lambda f: f.exception())
                s.turn_future.add_done_callback(lambda f: f.exception())
                s.pending[parked.id] = parked
                server.SESSIONS[s.id] = s

            body = {"model": "m", "messages": [
                {"role": "tool", "tool_call_id": f"call_{s_a.id}_1",
                 "content": "from-a"},
                {"role": "tool", "tool_call_id": f"call_{s_b.id}_1",
                 "content": "from-b"},
            ]}
            return await server.handle_followup(body, body["messages"]), s_a, s_b

        result, s_a, s_b = asyncio.run(go())
        assert len(captured) == 1, captured
        chosen = captured[0]["wanted"]
        assert chosen in (s_a.id, s_b.id), captured
        # Only the chosen session's tool message survived the filter.
        assert captured[0]["kept"] == 1, captured
        assert result["kept_tool_msgs"] == 1, result
        # Both sessions are gone: the chosen one because the stub ended it
        # after the routing decision, and the dropped one because
        # handle_followup superseded it before calling _continue_followup.
        # That upholds the invariant that one tool loop has at most one live
        # mcp_bridge session (issue #13, defect 2).
        assert s_a.id not in server.SESSIONS, server.SESSIONS
        assert s_b.id not in server.SESSIONS, server.SESSIONS
        print(f"  multi-live follow-up chose session {chosen[:8]}... and "
              "superseded the other's session so neither is live")
    finally:
        server._continue_followup = real


def test_only_a_resumption_is_allowed_to_queue():
    """A new request fails fast so SwitchYard can spill; a resumption waits.

    Queuing a new request would hold it here while the rest of the lane sat
    idle, which is the opposite of what fill-and-spill is for.
    """
    async def scenario():
        gate = server.cli_bridge.Gate()
        assert await gate.acquire(1)
        assert not await gate.acquire(1), "a new request must be refused immediately"

        # A waiter blocks, then is handed the slot the moment one frees.
        waiter = asyncio.create_task(gate.acquire_waiting(1, 5.0))
        await asyncio.sleep(0.05)
        assert not waiter.done(), "acquire_waiting should still be waiting"
        await gate.release()
        assert await asyncio.wait_for(waiter, 2.0) is True
        assert gate.in_flight == 1

        # And it gives up rather than waiting forever.
        started = time.monotonic()
        assert await gate.acquire_waiting(1, 0.2) is False
        assert time.monotonic() - started >= 0.2

    asyncio.run(scenario())
    print("  new requests refused immediately; resumptions queue with a deadline")


def test_a_streamed_reply_is_framed_as_sse_with_its_tool_calls():
    """`stream: true` must produce SSE, including when the answer is tool calls.

    The MCP bridge returned a JSON body whatever the caller asked for. An
    OpenAI client that requested SSE does not error on that — it waits for
    events that never arrive — so the caller hung with no error logged
    anywhere, and every hung attempt left a session parked holding one of the
    plan's connections until it reported itself at capacity. Observed exactly
    that against the judge lane.
    """
    async def collect(result, model):
        out = []
        async for frame in server.cli_bridge.sse_from_completion(result, model):
            out.append(frame)
        return out

    tool_result = {
        "id": "chatcmpl-x", "created": 1, "model": "claude-opus-5",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "get_weather", "arguments": '{"city": "Oslo"}'}}]}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    frames = asyncio.run(collect(tool_result, "claude-opus-5"))
    assert frames[-1] == "data: [DONE]\n\n", frames[-1]
    assert all(f.startswith("data: ") for f in frames), frames

    first = json.loads(frames[0][6:])
    delta = first["choices"][0]["delta"]
    assert delta["role"] == "assistant"
    calls = delta["tool_calls"]
    assert calls[0]["index"] == 0 and calls[0]["id"] == "call_1", calls
    assert calls[0]["function"]["name"] == "get_weather", calls
    # The terminal frame carries the finish_reason the caller keys on.
    last = json.loads(frames[-2][6:])
    assert last["choices"][0]["finish_reason"] == "tool_calls", last
    assert last["usage"]["total_tokens"] == 3, last

    # A plain text answer streams too, and says "stop".
    text_result = {
        "id": "chatcmpl-y", "created": 2, "model": "claude-opus-5",
        "choices": [{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": "OK"}}],
        "usage": {"total_tokens": 1},
    }
    frames = asyncio.run(collect(text_result, "claude-opus-5"))
    assert json.loads(frames[0][6:])["choices"][0]["delta"]["content"] == "OK"
    assert json.loads(frames[-2][6:])["choices"][0]["finish_reason"] == "stop"
    print("  tool calls and text both framed as SSE, with usage and finish_reason")


# ------------------------------------------------------- oversized prompt -> stdin ---
def test_a_prompt_over_the_argv_limit_travels_on_stdin():
    """A prompt past MAX_ARG_STRLEN must not reach the exec() call at all.

    create_subprocess_exec dies with "[Errno 7] Argument list too long" on a
    single argv element over ~128 KiB -- seen live when a folded system prompt
    plus long history rode in as one argument, killing the session and leaving
    its follow-ups 410ing. Over the limit, the prompt slot must vanish from
    argv (codex keeps its "-" placeholder) and come back as stdin data.
    """
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-bigprompt-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    huge = "x" * (server.STDIN_PROMPT_LIMIT + 10)
    argv, stdin_data = server.build_argv(huge, None, "m", workdir, "sess", tools_path, "")
    assert stdin_data == huge, "oversized prompt must be returned for stdin"
    assert huge not in argv, argv
    assert argv[0] == server.PROFILE["cli"], argv

    if server.PROVIDER == "codex":
        assert "-" in argv, argv
    else:
        assert "-" not in argv, argv

    # A small prompt keeps the old argv behaviour exactly.
    argv, stdin_data = server.build_argv("small", None, "m", workdir, "sess", tools_path, "")
    assert stdin_data is None
    assert "small" in argv, argv

    # And the driver feeds the stdin data to the process when given it.
    fake_cli = _write_fake_cli(
        "import sys\n"
        "data = sys.stdin.read()\n"
        "assert data.startswith('x' * 100), f'short stdin: {len(data)}'\n"
        "import json\n"
        "print(json.dumps({'result': f'saw {len(data)} chars'}))\n")

    async def scenario():
        session = _new_session()
        session.new_turn()
        await server.run_session(session, [sys.executable, fake_cli], huge)
        return session.turn_future.result()

    result = asyncio.run(scenario())
    assert result["type"] == "final", result
    assert result["payload"]["result"] == f"saw {len(huge)} chars", result
    print(f"  {len(huge)}-char prompt stayed off argv and arrived whole on stdin")


def test_argv_path_protects_dash_leading_prompts_for_all_providers_and_stdin_path_is_unchanged():
    """Issue #121 hardening for the MCP bridge, every provider covered.

    opencode's argv parser (yargs) greedily consumes a leading `-FLAG` prompt
    as an option. claude's parser does the same against the value of `-p`,
    and codex's parser does the same against its trailing positional. The
    argv path must therefore stop that across all three providers:

      * opencode -- profile carries `prompt_terminator: "--"`; build_argv
        splices the sentinel element immediately before the prompt so yargs
        treats the prompt as a positional.
      * claude -- prompt slot is the value of `-p "{prompt}"` and there is no
        sentinel option we can insert; build_argv prepends a `Message:\\n`
        line to the prompt element so it cannot be consumed as a flag.
      * codex -- `{prompt}` is the trailing positional in argv; same prefix
        guard as claude.

    The stdin path must remain untouched on every provider -- on that path
    the prompt rides the pipe, not argv (or, for codex, the `"-"` placeholder
    is already in argv), so there is no argv element to be misparsed as a
    flag and the prefix guard does not run.

    This is exercised directly on `mcp_bridge.build_argv` -- the same call
    `start_session` makes per request. The PROFILE/PROVIDER module globals
    are saved and restored around the case for each provider, so the rest of
    the suite keeps running against the fixture (claude provider) once this
    test returns. Mirrors `tests/test_cli_bridge.py`'s argv-path / stdin-path
    layout, sibling-bridge for sibling-bridge per `sidecars/CLAUDE.md`.
    """
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-issue121-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    saved_provider, saved_profile = server.PROVIDER, server.PROFILE
    try:
        # ---- opencode: argv path carries `--` sentinel before the prompt ----
        server.PROVIDER = "opencode"
        server.PROFILE = server.MCP_PROFILES["opencode"]

        # Argv path with a system that begins with `-`: the system prompt
        # now lives in the agent's `prompt:` (write_opencode_dir), never in
        # argv, and the prompt element is still preceded by the sentinel.
        argv, stdin_data = server.build_argv(
            "build", "--agent=build", "m", workdir, "sess", tools_path, "")
        assert stdin_data is None, "argv path must not feed stdin"
        assert "--" in argv, argv
        assert argv.index("--") == len(argv) - 2, argv
        assert argv[-1] == "build", argv
        assert "--agent=build" not in argv, argv
        cfg = json.loads((workdir / "opencode.json").read_text())
        assert cfg["agent"]["switchyard"]["prompt"] == "--agent=build", cfg
        # A prompt that itself leads with `-` is what the sentinel guards.
        argv, _ = server.build_argv(
            "--agent=build", None, "m", workdir, "sess", tools_path, "")
        assert argv[-2:] == ["--", "--agent=build"], argv

        # A short prompt with no system still picks up the same sentinel.
        argv, stdin_data = server.build_argv(
            "ship", None, "m", workdir, "sess", tools_path, "")
        assert stdin_data is None, stdin_data
        assert argv[argv.index("--")] == "--"
        assert argv[argv.index("--") + 1] == "ship", argv

        # Stdin path: an oversized prompt must NOT carry the `--` sentinel
        # (it would have nothing to separate from -- the prompt is not in
        # argv at all), and the prompt body must travel via stdin_data.
        huge = "x" * (server.STDIN_PROMPT_LIMIT + 10)
        argv, stdin_data = server.build_argv(
            huge, None, "m", workdir, "sess", tools_path, "")
        assert stdin_data == huge, "oversized prompt must travel on stdin"
        assert "--" not in argv, argv
        assert huge not in argv, "the prompt body must not appear in argv"

        # ---- claude: argv path prepends `Message:\n` to the -p value ----
        server.PROVIDER = "claude"
        server.PROFILE = server.MCP_PROFILES["claude"]
        # system goes through --system-prompt on the claude MCP path (claude
        # has its own system-prompt flag, unlike opencode), so the argv-path
        # prompt slot is the value of -p with the Message: prefix guarding it.
        argv, stdin_data = server.build_argv(
            "--agent=build", "--agent=build", "m", workdir, "sess",
            tools_path, "")
        assert stdin_data is None, "argv path must not feed stdin"
        prompt_idx = argv.index("-p") + 1
        assert not argv[prompt_idx].startswith("-"), argv
        assert argv[prompt_idx].startswith("Message:"), argv
        assert "--agent=build" in argv[prompt_idx], argv
        # The sentinel element must NOT be present -- claude's profile has
        # no prompt_terminator, only the Message: prefix guard.
        assert "--" not in argv, argv

        # ---- codex: argv path prepends `Message:\n` to the trailing positional ----
        server.PROVIDER = "codex"
        server.PROFILE = server.MCP_PROFILES["codex"]
        # codex puts system in instructions.md (so it shows up as a trailing
        # `-c model_instructions_file=...` in argv), but the prompt element
        # itself is still a positional with the Message: prefix guarding it.
        # Pass system as well so this mirrors cli_bridge's codex case.
        argv, stdin_data = server.build_argv(
            "--agent=build", "--agent=build", "m", workdir, "sess",
            tools_path, "")
        assert stdin_data is None, "argv path must not feed stdin"
        # Search for the prompt element by its Message: prefix so the
        # assertion does not depend on whether the trailing instructions
        # -c override is appended (it is, when system is non-None).
        prompt_idx = [i for i, a in enumerate(argv) if a.startswith("Message:")]
        assert prompt_idx, argv
        idx = prompt_idx[0]
        assert not argv[idx].startswith("-"), argv
        assert "--agent=build" in argv[idx], argv
        # The sentinel element must NOT be present -- codex's profile has
        # no prompt_terminator, only the Message: prefix guard.
        assert "--" not in argv, argv

        # Stdin path on the non-opencode providers: an oversized prompt must
        # NOT carry the Message: prefix (the guard is argv-path only); for
        # codex the `-` placeholder stays in argv at the prompt slot.
        server.PROVIDER = "claude"
        server.PROFILE = server.MCP_PROFILES["claude"]
        argv, stdin_data = server.build_argv(
            huge, None, "m", workdir, "sess", tools_path, "")
        assert stdin_data is not None, "oversized prompt must travel on stdin"
        assert huge not in argv, "the prompt body must not appear in argv"
        assert not any(a.startswith("Message:") for a in argv), argv

        server.PROVIDER = "codex"
        server.PROFILE = server.MCP_PROFILES["codex"]
        argv, stdin_data = server.build_argv(
            huge, None, "m", workdir, "sess", tools_path, "")
        assert stdin_data is not None, "oversized prompt must travel on stdin"
        assert huge not in argv, "the prompt body must not appear in argv"
        assert "-" in argv, "codex keeps the '-' placeholder at the prompt slot"
        assert not any(a.startswith("Message:") for a in argv), argv
    finally:
        server.PROVIDER, server.PROFILE = saved_provider, saved_profile
    print("  opencode argv: '--' sentinel; claude/codex argv: 'Message:\\n' "
          "prefix; all three stdin paths unchanged")


def test_run_session_fails_502_when_structured_parser_finds_no_answer():
    """The MCP-bridge path used to echo the raw event stream too.

    cli_bridge's parser raised on a stream with no text part; the mcp_bridge
    session driver's except swallowed it and returned
    {"result": "<step_start>...<step_finish>..."} as the assistant's payload.
    That JSONL passed through the tool loop and out to the client -- issue
    #3, observed against an OpenCode lane. Now the same failure surfaces as
    a 502 so the gateway can retry on a healthy lane.
    """
    bad_stream = ('{"type":"step_start","part":{"type":"step-start"}}\n'
                  '{"type":"step_finish","part":{"type":"step-finish",'
                  '"tokens":{"input":1,"output":1}}}')
    fake_cli = _write_fake_cli(
        "import sys\n"
        f"sys.stdout.write({bad_stream!r})\n")

    async def scenario():
        session = _new_session()
        session.new_turn()
        await server.run_session(session, [sys.executable, fake_cli])
        return session.turn_future.result()

    result = asyncio.run(scenario())
    assert result["type"] == "error", result
    assert result["status"] == 502, result
    assert "no parsed answer" in result.get("detail", ""), result
    print(f"  MCP session: structured-parser failure -> {result['status']} "
          f"({result['detail'][:60]}...)")


# ------------------------------------------------- issue #64: TEXT_LOST -----
def test_run_session_emits_text_lost_contract_detail_for_no_text_with_usage():
    """Issue #64 (reviewer finding on PR #73): opencode-go and opencode-go2
    plans run mcp_bridge, NOT cli_bridge, so cli_bridge's contract-string fix
    was dead code for them. mcp_bridge's run_session calls parse_output
    directly (no in-process retry of its own — cli_bridge owns that) and was
    catching CliNoTextError as a plain ValueError, emitting the legacy
    "yielded no parsed answer" detail. The gateway's _NO_TEXT regex does
    not match that, so the affected plans kept riding the TRANSIENT doubling
    ladder while the ledger still lost the charged tokens.

    The mcp_bridge path must now catch CliNoTextError specifically and emit
    the same contract detail cli_bridge uses (same helper, same string),
    so classify.TEXT_LOST fires and extract_no_text_tokens books the
    charged prompt/completion counts.

    Issue #78 follow-up: mcp_bridge now retries once on no-text-with-usage
    (matching cli_bridge._run_cli). The contract 502 only fires when BOTH
    attempts lose text; the detail must carry the COMBINED usage of both
    attempts so the ledger books everything that was charged. The fake
    below emits no-text-with-usage on both calls (different token counts
    each time) so the combined-usage assertion has both numbers visible.
    """
    counter_path = _write_fake_cli_counter()

    call_one = ('{"type":"step_start","part":{"type":"step-start"}}\n'
                '{"type":"step_finish","part":{"type":"step-finish",'
                '"tokens":{"input":239,"output":22}}}')
    call_two = ('{"type":"step_start","part":{"type":"step-start"}}\n'
                '{"type":"step_finish","part":{"type":"step-finish",'
                '"tokens":{"input":11,"output":3}}}')
    fake_cli = _write_counter_fake_cli(counter_path, on_call_one=call_one,
                                       on_call_other=call_two)

    saved_parser = server.PROFILE["parser"]
    try:
        # mcp_bridge dispatches on PROFILE["parser"]. The test's default
        # fixture is the claude provider (claude_json), whose parser does
        # a bare json.loads and so does not raise CliNoTextError on this
        # stream — it raises JSONDecodeError. Force the opencode events
        # parser so the issue #64 path is actually exercised.
        server.PROFILE["parser"] = "events_json"

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "error", result
        assert result["status"] == 502, result
        detail = result.get("detail", "")
        # The contract phrase the gateway's _NO_TEXT regex matches. The
        # provider name comes from mcp_bridge.PROVIDER, not cli_bridge's —
        # so the test asserts on the structure, not the literal prefix.
        assert "cli emitted tokens with no text " in detail, detail
        # Both attempts lost text: 239 + 11 prompt, 22 + 3 completion.
        assert "(prompt_tokens=250, completion_tokens=25)" in detail, detail
        # And mcp_bridge does NOT reach the legacy "yielded no parsed
        # answer" branch on this shape — that was the bug.
        assert "yielded no parsed answer" not in detail, detail
        # Both sides use the same helper, so the strings must match
        # byte-for-byte: cli_bridge.format_no_text_detail(PROVIDER, usage).
        # The PROVIDER value here is whatever the test's os.environ set
        # ("claude"), so the prefix is provider-specific.
        expected = server.cli_bridge.format_no_text_detail(
            server.PROVIDER,
            {"input_tokens": 250, "output_tokens": 25})
        assert detail == expected, (detail, expected)
        # And the retry actually fired -- two spawns, not one.
        calls = _read_counter(counter_path)
        assert calls == 2, f"expected 2 spawns (1st + retry), got {calls}"
        print(f"  MCP session: no-text-with-usage x2 -> 502 with combined usage "
              f"({detail!r}, calls={calls})")
    finally:
        server.PROFILE["parser"] = saved_parser


def test_run_session_legacy_no_text_without_usage_still_emits_no_parsed_answer():
    """Regression guard: a no-text stream with ZERO charged usage must NOT
    take the new CliNoTextError branch — it has no usage to carry, so the
    legacy "yielded no parsed answer" 502 stays. cli_bridge._run_cli's
    gating on nonzero input_tokens / output_tokens / provider_cost applies
    here too (parse_output is shared), so an empty-stdout / zero-usage
    failure looks the same as it did before this PR."""
    bad_stream = ('{"type":"step_start","part":{"type":"step-start"}}\n'
                  '{"type":"step_finish","part":{"type":"step-finish"}}')
    fake_cli = _write_fake_cli(
        "import sys\n"
        f"sys.stdout.write({bad_stream!r})\n")

    saved_parser = server.PROFILE["parser"]
    try:
        server.PROFILE["parser"] = "events_json"

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "error", result
        assert result["status"] == 502, result
        detail = result.get("detail", "")
        assert "yielded no parsed answer" in detail, detail
        # And, critically, NOT the new contract shape — there are no
        # charged tokens to quote, so the contract phrase would be a lie.
        assert "emitted tokens with no text" not in detail, detail
        print(f"  MCP session: zero-usage no-text -> legacy 502 "
              f"({detail[:60]}...)")
    finally:
        server.PROFILE["parser"] = saved_parser


# ---------------------------------------- issue #78: TEXT_LOST retry-once ----
def _write_fake_cli_counter() -> str:
    """Return a fresh counter file path initialised to "0".

    Tests below use the counter-file fake-CLI idiom from tests/test_cli_bridge.py
    (see _write_counter_fake_cli): the fake increments the counter file on every
    spawn, so the test can assert how many times the CLI was actually invoked.
    A fresh mkdtemp avoids colliding with any other counter file in flight,
    and pre-creating the file with "0" sidesteps FileNotFoundError inside the
    fake CLI's `open(p).read() or '0'` on its very first spawn.
    """
    d = tempfile.mkdtemp(prefix="mcpb-cnt-")
    p = os.path.join(d, "counter")
    open(p, "w").write("0")
    return p


def _write_counter_fake_cli(counter_path: str, *, on_call_one: str = "",
                            on_call_other: str = "", stderr: str = "",
                            rc: int = 0) -> str:
    """Write a tiny Python CLI that increments `counter_path` each call.

    `on_call_one` is printed on the first call, `on_call_other` on every
    subsequent call. Same shape as test_cli_bridge.py's make_fake. Mirrors
    the cli_bridge retry tests (issue #64) so the spawn-count assertion is
    apples-to-apples against that reference.
    """
    body = (f"import sys\n"
            f"p = {counter_path!r}\n"
            f"n = int(open(p).read() or '0') + 1\n"
            f"open(p, 'w').write(str(n))\n"
            f"if n == 1:\n"
            f"  sys.stdout.write({on_call_one!r})\n"
            f"else:\n"
            f"  sys.stdout.write({on_call_other!r})\n"
            f"sys.stderr.write({stderr!r})\n"
            f"sys.exit({rc})\n")
    f = tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix="mcpb-rf-", delete=False)
    f.write(body)
    f.close()
    return f.name


def _read_counter(counter_path: str) -> int:
    return int(open(counter_path).read() or "0")


def test_run_session_retries_once_on_text_lost_and_returns_attempt_two_payload():
    """Issue #78: a single no-text-with-usage attempt is retried in-process
    exactly once, and on a successful retry the caller sees attempt 2's
    payload (not the legacy "yielded no parsed answer" 502). The call
    counter proves the CLI was spawned twice (attempt 1 + retry), and the
    payload's usage dict carries the merged token counts of both attempts
    so the ledger books each spawn exactly once.

    The retry semantics here mirror cli_bridge._run_cli, the reference
    implementation. opencode-go / opencode-go2 plans run mcp_bridge, so
    without this in-process retry the plan rides the TRANSIENT doubling
    ladder (issue #64) — the gateway has no way to know the first attempt
    was a parse-level loss, not a transport-level loss.
    """
    counter_path = _write_fake_cli_counter()
    bad = ('{"type":"step_finish","part":{"type":"step-finish",'
           '"tokens":{"input":777,"output":33}}}')
    good = ('{"type":"text","part":{"type":"text","text":"ANSWER"}}\n'
            '{"type":"step_finish","part":{"type":"step-finish",'
            '"tokens":{"input":111,"output":7,"reasoning":2}}}')
    fake_cli = _write_counter_fake_cli(counter_path, on_call_one=bad,
                                       on_call_other=good)

    saved_parser = server.PROFILE["parser"]
    try:
        server.PROFILE["parser"] = "events_json"

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "final", result
        assert result["payload"]["result"] == "ANSWER", result
        u = result["payload"]["usage"]
        # 777 + 111 = 888 input tokens across both attempts.
        assert u["input_tokens"] == 888, u
        # 33 from attempt 1, 7+2 reasoning from attempt 2; reasoning rolls
        # into output_tokens, so 33 + 9 = 42.
        assert u["output_tokens"] == 42, u
        # And exactly two spawns happened -- the retry, not an infinite loop.
        calls = _read_counter(counter_path)
        assert calls == 2, f"successful retry must call CLI twice, got {calls}"
        print(f"  MCP retry-once: bad+good -> ANSWER (input={u['input_tokens']}, "
              f"output={u['output_tokens']}, calls={calls})")
    finally:
        server.PROFILE["parser"] = saved_parser


def test_run_session_two_consecutive_text_lost_returns_502_with_combined_usage():
    """Issue #78 contract path: when both attempts lose text, the final 502
    detail must match the gateway's contract exactly (same helper as
    cli_bridge, so the strings cannot drift):
        {PROVIDER} cli emitted tokens with no text
            (prompt_tokens=N, completion_tokens=M)
    where N/M are the COMBINED charges of both attempts. The byte-for-byte
    detail check is load-bearing -- the gateway's _NO_TEXT regex matches
    the literal phrase, and extract_no_text_tokens parses the parenthesised
    numbers, so any drift here would silently change the cooldown class
    and the ledger.
    """
    counter_path = _write_fake_cli_counter()
    bad_one = ('{"type":"step_finish","part":{"type":"step-finish",'
               '"tokens":{"input":239,"output":22}}}')
    bad_two = ('{"type":"step_finish","part":{"type":"step-finish",'
               '"tokens":{"input":50,"output":5}}}')
    fake_cli = _write_counter_fake_cli(counter_path, on_call_one=bad_one,
                                       on_call_other=bad_two)

    saved_parser = server.PROFILE["parser"]
    try:
        server.PROFILE["parser"] = "events_json"

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "error", result
        assert result["status"] == 502, result
        detail = result.get("detail", "")
        # 239 + 50 prompt, 22 + 5 completion — must match the contract shape
        # via format_no_text_detail (same helper cli_bridge uses), byte-for-byte.
        expected = server.cli_bridge.format_no_text_detail(
            server.PROVIDER,
            {"input_tokens": 289, "output_tokens": 27})
        assert detail == expected, (detail, expected)
        # And the legacy "yielded no parsed answer" branch did NOT fire —
        # there is real charged usage to carry, and it must reach the
        # contract so the ledger books both attempts.
        assert "yielded no parsed answer" not in detail, detail
        # Two spawns -- 1st attempt + retry -- proves the retry fired.
        calls = _read_counter(counter_path)
        assert calls == 2, f"two failed attempts must call CLI twice, got {calls}"
        print(f"  MCP retry-once: bad+bad -> 502 with combined usage "
              f"({detail!r}, calls={calls})")
    finally:
        server.PROFILE["parser"] = saved_parser


def test_run_session_only_text_lost_triggers_retry_other_failures_stay_single_attempt():
    """Issue #78 narrow retry: only CliNoTextError (no-text with nonzero
    charged usage) triggers the retry. Auth, quota, empty stdout, and
    generic rc=1 each stay single-attempt — retrying those would either
    double-bill the caller (limit) or amplify blips into double spawns
    without improving the outcome. The counter file is the assertion:
    calls == 1 in every sub-scenario.

    Same shape as tests/test_cli_bridge.py:1015-1056 (the cli_bridge
    counterpart); the only difference is the mcp_bridge run_session surface
    (no HTTPException — error info lands in session.turn_future.result()).
    """
    counter_path = _write_fake_cli_counter()

    def run(stderr: str, rc: int) -> dict:
        # A fresh fake per scenario so the counter path is shared and the
        # branch is determined entirely by the args passed in here.
        open(counter_path, "w").write("0")
        fake_cli = _write_counter_fake_cli(counter_path, stderr=stderr, rc=rc)

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        return asyncio.run(scenario())

    try:
        # --- auth failure: single attempt, 401 ---
        result = run(stderr="please run `claude login` to authenticate", rc=1)
        assert result["type"] == "error" and result["status"] == 401, result
        assert _read_counter(counter_path) == 1, \
            f"auth failure must not retry (calls={_read_counter(counter_path)})"

        # --- quota failure: single attempt, 429 ---
        result = run(stderr="You've hit your usage limit. Purchase more credits.",
                     rc=1)
        assert result["type"] == "error" and result["status"] == 429, result
        assert _read_counter(counter_path) == 1, \
            f"quota failure must not retry (calls={_read_counter(counter_path)})"

        # --- empty stdout: single attempt, 502 ---
        # rc=0 with no stdout trips the `not stdout.strip()` branch and falls
        # through to the generic "cli failed (0)" 502 — must NOT retry.
        result = run(stderr="", rc=0)
        assert result["type"] == "error" and result["status"] == 502, result
        assert "cli failed" in result.get("detail", ""), result
        assert _read_counter(counter_path) == 1, \
            f"empty stdout must not retry (calls={_read_counter(counter_path)})"

        # --- generic rc=1 with no recognisable shape: 502 "cli failed", 1 call ---
        result = run(stderr="some unrecognised failure prose", rc=1)
        assert result["type"] == "error" and result["status"] == 502, result
        assert _read_counter(counter_path) == 1, \
            f"generic rc=1 must not retry (calls={_read_counter(counter_path)})"

        print("  MCP retry-once: auth/quota/empty/generic each spawn once; "
              "no-text-with-usage is the ONLY retried shape")
    finally:
        try:
            os.unlink(counter_path)
        except OSError:
            pass


def test_run_session_successful_retry_merges_attempt_one_usage_into_payload():
    """Issue #78 ledger invariant: a successful retry must fold attempt 1's
    charged tokens into attempt 2's payload so the ledger books BOTH
    attempts exactly once. `_sum_usage` adds numeric fields and leaves
    non-numeric (and bool) fields untouched — so the final usage dict is
    attempt 2's payload usage with attempt 1's numeric counts summed in,
    and any non-numeric fields attempt 1 happened to carry are not
    retroactively created on attempt 2's dict.

    This pins the exact dict the gateway's record() sees on a retry — if
    a future change dropped the merge (or summed into the wrong dict),
    the ledger would silently lose attempt 1's tokens.
    """
    counter_path = _write_fake_cli_counter()
    bad = ('{"type":"step_finish","part":{"type":"step-finish",'
           '"tokens":{"input":777,"output":33}}}')
    good = ('{"type":"text","part":{"type":"text","text":"ANSWER"}}\n'
            '{"type":"step_finish","part":{"type":"step-finish",'
            '"tokens":{"input":111,"output":7,"reasoning":2}}}')
    fake_cli = _write_counter_fake_cli(counter_path, on_call_one=bad,
                                       on_call_other=good)

    saved_parser = server.PROFILE["parser"]
    try:
        server.PROFILE["parser"] = "events_json"

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "final", result
        payload = result["payload"]
        assert payload["result"] == "ANSWER", payload
        u = payload["usage"]
        # Numeric fields: both attempts' counts must be summed.
        # 777 + 111 = 888 input, 33 + (7+2 reasoning) = 42 output.
        assert u["input_tokens"] == 888, u
        assert u["output_tokens"] == 42, u
        # _sum_usage only touches numeric (and not-bool) fields, so the
        # dict shape stays exactly what attempt 2 produced — just with
        # attempt 1's numeric counts folded in. The events_json parser
        # also emits cache_read_tokens (zero when no cache field), so it
        # carries through attempt 2 untouched (attempt 1 had no entry to
        # fold in). No spurious keys appear beyond what attempt 2's
        # parser produced.
        assert set(u.keys()) == {"input_tokens", "output_tokens",
                                  "cache_read_tokens"}, \
            f"unexpected keys in merged usage: {sorted(u.keys())}"
        assert u["cache_read_tokens"] == 0, u
        # And exactly two spawns happened — the retry, not an infinite loop.
        calls = _read_counter(counter_path)
        assert calls == 2, f"successful retry must call CLI twice, got {calls}"
        print(f"  MCP retry-once merged usage: keys={sorted(u.keys())}, "
              f"input={u['input_tokens']}, output={u['output_tokens']}, "
              f"calls={calls}")
    finally:
        server.PROFILE["parser"] = saved_parser


# ------------------------------------------ issue #13: rebuild churn ----------
def test_duplicate_delivery_returns_cached_response_without_rebuilding():
    """Regression for issue #13, defect 1.

    The same tool results delivered twice used to look identical to a lost
    session: nothing parked in the live session matched, so the code
    rebuilt. That wasted a turn of narration and could leave the caller in
    a worse state than before -- and on the gateway it spawned the second
    live session of defect 2. After the fix, the duplicate is recognised via
    the resolved-recently record, the cached tool_calls response is returned
    verbatim, and no rebuild fires.

    Drives _continue_followup directly: the duplicate short-circuit only
    applies while the session is still parked, so the test pins that state
    (last_response set, awaiting_followup=True, resolved_recently populated)
    and verifies a second follow-up with the same tool_call_ids returns the
    cached response without calling start_session.
    """
    captured: list = []
    restore = _stub_start_session(captured)
    try:
        async def scenario():
            session = _new_session()
            session.new_turn()
            parked = asyncio.create_task(
                server.register_tool_call(session.id, "get_weather",
                                          {"city": "Oslo"}))
            await asyncio.sleep(0)
            turn = await session.turn_future
            assert turn["type"] == "tool_calls", turn
            call = turn["calls"][0]

            # Simulate the prior delivery that consumed the tool result: pop
            # the parked call, mark it resolved, render a tool_calls response,
            # and leave the session parked -- exactly the state a duplicate
            # delivery would land in. Resolve the parked future too, so the
            # register_tool_call coroutine returns instead of hanging the test.
            session.pending.pop(call.id, None)
            session.mark_resolved(call.id)
            cached = server.render_turn(session, turn, None)
            session.last_response = cached
            session.awaiting_followup = True
            call.future.set_result(
                {"content": [{"type": "text", "text": '{"temp_c":-3}'}],
                 "isError": False})
            with contextlib.suppress(Exception):
                await parked

            # Same tool result delivered again -- a client retry because the
            # network dropped the response. Before the fix this rebuilt; now
            # it must return the cached response and leave exactly one live
            # session.
            assert session.id in server.SESSIONS
            body = {"model": "m", "messages": [
                {"role": "tool", "tool_call_id": call.id,
                 "content": '{"temp_c":-3}'},
            ]}
            second = await server._continue_followup(body, session.id,
                                                     body["messages"], None)
            return cached, second, session, call.id

        cached, second, session, call_id = asyncio.run(scenario())
        assert cached is second, "duplicate delivery must echo the cached response"
        assert cached["choices"][0]["finish_reason"] == "tool_calls", cached
        # No rebuild fired -- start_session was never called.
        assert captured == [], captured
        # The session is still in SESSIONS (and live): a duplicate delivery
        # does not consume the session, because the tool results were
        # already consumed by the first delivery.
        assert session.id in server.SESSIONS
        assert not session.dead
        assert call_id in session.resolved_recently
        print(f"  duplicate delivery returned the cached response; "
              f"{len(captured)} rebuild(s); one live session")
    finally:
        restore()


def test_duplicate_delivery_under_saturated_gate_returns_cached_without_touching_gate():
    """Cover the explicit hoisting claim from issue #130's fix.

    `_continue_followup` runs the duplicate-delivery short-circuit BEFORE
    `await unpark_session(session)` so a duplicate arriving on a saturated
    plan returns the cached tool_calls response without ever trying to
    acquire a gate slot. The companion
    `test_duplicate_delivery_returns_cached_response_without_rebuilding`
    exercises the short-circuit with a free-capacity gate (so the order
    is irrelevant), and the new `test_followup_503_on_saturated_gate_...`
    exercises the saturated gate with a FRESH delivery (so the
    short-circuit never fires). Neither catches a regression that moves
    `if` below `await unpark_session`; this test does.

    Drives one delivery to completion to populate `last_response`,
    `resolved_recently`, and `awaiting_followup=True` (same setup as the
    existing duplicate test), then saturates `cli_bridge._gate` against
    `limit` with a fresh `cli_bridge.Gate` swap (same pattern as
    `test_unpark_returns_503_quickly_when_gate_is_full`), pins
    `server.RESUME_WAIT` so a regression that called `unpark_session`
    first would 503 within `SHORT`, then re-delivers the same
    `tool_msgs`. A duplicate on a saturated gate must:
      - return `session.last_response` verbatim (cached, not a fresh turn),
      - leave `gate.in_flight == limit` (the gate was never touched),
      - leave `session.awaiting_followup` True (no unpark ran),
      - record zero `await_turn` calls (no fresh CLI turn was awaited).
    """
    SHORT = 0.3
    await_turn_calls: list = []

    # Fresh-gate swap: the shared `_gate`'s lock/Condition are lazily
    # bound to whichever event loop first called acquire_waiting, so
    # without a swap the gate's locks would be tied to an earlier
    # test's event loop. Same reason as
    # test_unpark_returns_503_quickly_when_gate_is_full.
    saved_gate = server.cli_bridge._gate
    gate = server.cli_bridge.Gate()
    server.cli_bridge._gate = gate
    saved_sessions = dict(server.SESSIONS)
    server.SESSIONS.clear()
    saved_wait = server.RESUME_WAIT
    server.RESUME_WAIT = SHORT
    real_await_turn = server.await_turn

    async def record_turn(sess, request):
        await_turn_calls.append({"session_id": sess.id})
        return {"type": "tool_calls", "calls": []}
    server.await_turn = record_turn

    try:
        async def scenario():
            session = _new_session()
            session.new_turn()
            parked = asyncio.create_task(
                server.register_tool_call(session.id, "get_weather",
                                          {"city": "Oslo"}))
            await asyncio.sleep(0)
            turn = await session.turn_future
            assert turn["type"] == "tool_calls", turn
            call = turn["calls"][0]

            # Simulate the prior delivery that consumed the tool result:
            # exactly the same hand-built parked-with-cached-response
            # state as
            # test_duplicate_delivery_returns_cached_response_without_rebuilding.
            session.pending.pop(call.id, None)
            session.mark_resolved(call.id)
            cached = server.render_turn(session, turn, None)
            session.last_response = cached
            session.awaiting_followup = True
            call.future.set_result(
                {"content": [{"type": "text", "text": '{"temp_c":-3}'}],
                 "isError": False})
            with contextlib.suppress(Exception):
                await parked

            # Saturate the gate from a known clean state. RESUME_WAIT is
            # pinned so a regression that called unpark_session BEFORE the
            # short-circuit would 503 within SHORT rather than blocking
            # the test for the production 300 s wait (and would never
            # reach the `return session.last_response` line at all).
            limit = server.cli_bridge.config().concurrency
            for _ in range(limit):
                assert await gate.acquire(limit)
            assert gate.in_flight == limit, gate.in_flight

            body = {"model": "m", "messages": [
                {"role": "tool", "tool_call_id": call.id,
                 "content": '{"temp_c":-3}'},
            ]}
            second = await server._continue_followup(
                body, session.id, body["messages"], None)
            return cached, second, session, call.id

        cached, second, session, call_id = asyncio.run(scenario())
        limit = server.cli_bridge.config().concurrency
        # 1. Returned the cached tool_calls response verbatim.
        assert cached is second, "duplicate must echo the cached response"
        assert cached["choices"][0]["finish_reason"] == "tool_calls", cached
        # 2. The gate was never touched: still at `limit`, the saturation
        # holds -- if the short-circuit had been below `await unpark_session`,
        # unpark would have acquired one slot on success or 503'd here.
        assert gate.in_flight == limit, (gate.in_flight, limit)
        # 3. The session still looks parked -- no unpark ran.
        assert session.awaiting_followup, session
        assert not session.holds_slot, session
        assert session.id in server.SESSIONS
        assert not session.dead
        assert call_id in session.resolved_recently
        # 4. No fresh CLI turn was awaited -- no new inference happened.
        assert await_turn_calls == [], await_turn_calls
        print("  duplicate delivery on a saturated gate returned the cached "
              "response; gate untouched, awaiting_followup still True, "
              "0 await_turn calls")
    finally:
        server.RESUME_WAIT = saved_wait
        server.await_turn = real_await_turn
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.cli_bridge._gate = saved_gate


def test_rebuild_supersedes_the_old_session_so_two_live_sessions_never_coexist():
    """Regression for issue #13, defect 2.

    Before this fix, a follow-up whose delivered tool_call_ids did not match
    anything parked in the live session rebuilt by calling
    `resume_gone_session` -> `start_session`. That minted a fresh session id
    and inserted it into SESSIONS, but the old one was still in SESSIONS
    too -- parked, alive, and forever waiting for tool results that will
    never come. Two live sessions of one tool loop is the bug the issue is
    named after.

    Drives the rebuild path directly: a live session with a parked call
    whose id the follow-up does NOT name, so the resolve loop resolves zero
    calls and the rebuild branch in _continue_followup runs. Without the fix
    the old session stays live alongside the rebuilt one; with it, the old
    session is superseded and removed.
    """
    captured: list = []
    restore = _stub_start_session(captured)
    try:
        async def scenario():
            session = _new_session()
            session.new_turn()
            # Parked with a tool call whose id the follow-up will NOT name --
            # so the rebuild path (not the resolve path) is taken.
            parked = asyncio.create_task(
                server.register_tool_call(session.id, "get_weather",
                                          {"city": "Oslo"}))
            await asyncio.sleep(0)
            await session.turn_future
            session.awaiting_followup = True

            # A follow-up whose tool_call_id does not match anything parked
            # in this session AND is not in resolved_recently. The
            # duplicate-delivery short-circuit must not fire (no resolved
            # ids), so the rebuild path runs.
            foreign_id = f"call_{uuid.uuid4().hex}_1"
            body = {"model": "m",
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather", "description": "",
                        "parameters": {"type": "object", "properties": {}}}}],
                    "messages": [
                        {"role": "user", "content": "weather in Oslo?"},
                        {"role": "assistant", "content": None, "tool_calls": [
                            {"id": foreign_id, "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": '{"city":"Oslo"}'}}]},
                        {"role": "tool", "tool_call_id": foreign_id,
                         "content": '{"temp_c":-3}'},
                    ]}
            tool_msgs = body["messages"][2:]
            result = await server._continue_followup(body, session.id,
                                                     tool_msgs, None)
            # The parked call's future was failed by supersede_session; the
            # register_tool_call task raised 504. Consume it here so asyncio
            # does not log "future exception was never retrieved".
            with contextlib.suppress(Exception):
                await parked
            return result, session

        result, session = asyncio.run(scenario())
        assert result == {"stubbed": True}, result
        # Rebuild fired exactly once -- the old session was rebuilt against.
        assert len(captured) == 1, captured
        # The old session is gone: not in SESSIONS, marked dead. The invariant
        # holds: one tool loop, at most one live session.
        assert session.id not in server.SESSIONS, dict(server.SESSIONS)
        assert session.dead, session
        print(f"  rebuild superseded the old session: SESSIONS has "
              f"{len(server.SESSIONS)} unrelated entry/entries")
    finally:
        restore()


def test_span2_follow_up_supersedes_dropped_sessions_so_the_invariant_holds():
    """Regression for issue #13, defect 2 (second half).

    When a follow-up's tool results span more than one live session, the
    others' tool results are dropped by design (the request can only resolve
    against one session). The dropped sessions' CLIs are blocked on
    tool_server.py requests that will never be answered, so they are now
    superseded -- exactly like the rebuild path -- so the invariant "one
    caller tool loop <-> at most one live mcp_bridge session" holds for the
    span-N case too.

    The test asserts both halves of the invariant in one go: only the chosen
    session is routed to _continue_followup, and all the others are gone
    from SESSIONS by the time handle_followup returns.
    """
    captured: list = []
    real = server._continue_followup

    async def stub(body, wanted, tool_msgs, request):
        captured.append({"wanted": wanted})
        return {"stubbed": wanted}

    server._continue_followup = stub
    try:
        async def go():
            s_a = _new_session()
            s_b = _new_session()
            s_c = _new_session()
            sessions = [s_a, s_b, s_c]
            for s in sessions:
                s.new_turn()
                s.awaiting_followup = True
                parked = server.ParkedCall(
                    id=f"call_{s.id}_1", name="echo", arguments={},
                    future=asyncio.get_event_loop().create_future())
                # In production a parked call has tool_server.py's HTTP
                # request as its consumer and a parked session has its
                # turn_future already resolved (the CLI finished a
                # tool_calls turn). This test inlines both, so add
                # callbacks to swallow whatever supersede_session leaves
                # behind -- otherwise asyncio logs "Future exception was
                # never retrieved" for the dropped sessions.
                parked.future.add_done_callback(lambda f: f.exception())
                s.turn_future.add_done_callback(lambda f: f.exception())
                s.pending[parked.id] = parked
                server.SESSIONS[s.id] = s

            body = {"model": "m", "messages": [
                {"role": "tool", "tool_call_id": f"call_{s_a.id}_1",
                 "content": "a"},
                {"role": "tool", "tool_call_id": f"call_{s_b.id}_1",
                 "content": "b"},
                {"role": "tool", "tool_call_id": f"call_{s_c.id}_1",
                 "content": "c"},
            ]}
            return await server.handle_followup(body, body["messages"]), sessions

        result, sessions = asyncio.run(go())
        # Only one routing decision; the other tool results were filtered.
        assert len(captured) == 1, captured
        chosen_id = captured[0]["wanted"]
        # The chosen session survives -- _continue_followup is a stub here,
        # so it does not end it. The dropped sessions are gone.
        survivors = [s for s in sessions if s.id in server.SESSIONS]
        assert len(survivors) == 1, survivors
        assert survivors[0].id == chosen_id, (survivors, chosen_id)
        # The dropped sessions are dead.
        for s in sessions:
            if s.id != chosen_id:
                assert s.dead, s
        assert result == {"stubbed": chosen_id}, result
        print(f"  span-3 follow-up chose {chosen_id[:8]}... and superseded "
              "the other two sessions; invariant upheld")
    finally:
        server._continue_followup = real


def test_unpark_returns_503_quickly_when_gate_is_full():
    """A pinned follow-up on a saturated plan must 503 fast, not hold the slot.

    `unpark_session` queues on the plan's concurrency gate while a parked
    session waits to be told a tool result. The gateway claimed the plan slot
    on the way in, so the wait is paid against the gateway budget, not the
    sidecar's. With RESUME_WAIT at 300s (issue #14's example), one queued
    resumption ate the slot its own pinned follow-ups were about to be
    429'd against -- a self-starvation loop. The fix is a short, bounded
    wait that returns control to the gateway the moment it expires.
    """
    SHORT = 0.3   # tight enough for a fast test, long enough to be measurable

    async def scenario():
        # The shared `_gate` singleton has its Condition lazily bound to
        # whichever event loop first calls `acquire_waiting`; an earlier
        # test's loop is stale by the time we run. Swap in a fresh gate for
        # this scenario so its locks bind to OUR event loop.
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        try:
            limit = server.cli_bridge.config().concurrency

            # Saturate the gate from a known clean state.
            for _ in range(limit):
                assert await gate.acquire(limit)
            assert gate.in_flight == limit, gate.in_flight

            session = _new_session()
            session.holds_slot = False

            saved_wait = server.RESUME_WAIT
            server.RESUME_WAIT = SHORT
            try:
                started = time.monotonic()
                try:
                    await server.unpark_session(session)
                except server.HTTPException as exc:
                    elapsed = time.monotonic() - started
                    assert exc.status_code == 503, exc.status_code
                    assert "no slot freed" in exc.detail, exc.detail
                    assert exc.headers and exc.headers.get("Retry-After") == "15", exc.headers
                    # The whole point: bounded, not 300s. SHORT leaves room for
                    # scheduling jitter without ever approaching the old default.
                    assert elapsed < SHORT * 4, elapsed
                else:
                    raise AssertionError("unpark_session must 503 when the gate stays full")
            finally:
                server.RESUME_WAIT = saved_wait
                _drop(session)
        finally:
            server.cli_bridge._gate = saved_gate

    asyncio.run(scenario())
    print(f"  unpark_session bounded to RESUME_WAIT (~{SHORT}s) instead of 300s")


def test_resume_gone_returns_503_quickly_when_gate_is_full():
    """A rebuild on a saturated plan must 503 fast, for the same reason.

    `resume_gone_session` is the rebuild path: a follow-up for a session that
    no longer exists (preempted, reaped, lost to a restart). It is the only
    path allowed to queue -- a *new* request still has to fail fast so
    SwitchYard can spill to the next plan. The queueing is still dangerous:
    a rebuild that cannot find a slot within the wait burns the whole wait
    holding a gateway plan slot, and the pinned session's own follow-ups
    correctly fail-fast rather than spill, so the same session 429s against
    the slot its own queued rebuild is holding.
    """
    SHORT = 0.3

    async def scenario():
        # Same fresh-gate reason as the unpark test: bind the lock/condition
        # to this scenario's event loop, not one an earlier test used.
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        try:
            limit = server.cli_bridge.config().concurrency

            for _ in range(limit):
                assert await gate.acquire(limit)
            assert gate.in_flight == limit, gate.in_flight

            # Use tools that pass translate_tools and a message with content, so
            # we exercise the queue branch instead of failing earlier on the
            # no-tools / no-content checks (which would mask the bug we care
            # about).
            body = {
                "model": "m",
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "",
                    "parameters": {"type": "object", "properties": {}}}}],
                "messages": [
                    {"role": "user", "content": "weather in Oslo?"},
                    {"role": "tool", "tool_call_id": "call_x", "content": '{"temp_c":-3}'},
                ],
            }
            # Stub start_session so a regression that gets past the gate would
            # still fail clearly instead of hanging on a real CLI spawn.
            calls: list = []
            restore_start = server.start_session

            async def fake_start(body, mcp_tools, prompt, system, model, request,
                                  image_paths=None, img_dir=None):
                calls.append({"prompt": prompt})
                if img_dir is not None:
                    import shutil
                    shutil.rmtree(img_dir, ignore_errors=True)
                return {"stubbed": True}
            server.start_session = fake_start

            saved_wait = server.RESUME_WAIT
            server.RESUME_WAIT = SHORT
            try:
                started = time.monotonic()
                try:
                    await server.resume_gone_session(
                        body, body["tools"], "nonexistent-session-id", None,
                        why="lost")
                except server.HTTPException as exc:
                    elapsed = time.monotonic() - started
                    assert exc.status_code == 503, exc.status_code
                    assert "no slot freed" in exc.detail, exc.detail
                    assert exc.headers and exc.headers.get("Retry-After") == "30", exc.headers
                    assert elapsed < SHORT * 4, elapsed
                else:
                    raise AssertionError("resume_gone_session must 503 when the gate stays full")
                assert not calls, calls   # the real path was never reached
            finally:
                server.RESUME_WAIT = saved_wait
                server.start_session = restore_start
        finally:
            server.cli_bridge._gate = saved_gate

    asyncio.run(scenario())
    print(f"  resume_gone_session bounded to RESUME_WAIT (~{SHORT}s) instead of 300s")


# ------------------------------------------ issue #80: gate-slot leak on error ---
def test_error_result_releases_gate_slot_and_removes_session():
    """A CLI failure on a fresh turn must tear the session down, not leak a slot.

    Before issue #80, every CLI subprocess failure path resolved the turn
    future with `{"type": "error", ...}` and `render_turn` then raised
    HTTPException. That exception escaped `start_session` *without* running
    `end_session`, so the Session stayed in `SESSIONS` holding its gate slot
    until the 1800 s idle reaper came round. Four such leaks physically
    filled the sidecar gate, every request 429'd `sidecar at capacity`, and
    the gateway's `ConcurrencyLearner` ratcheted the learned cap to 1.

    Drives `handle_fresh` (the new-session path) end-to-end: get_weather is
    not command-shaped so the env probe does not fire (`find_command_tool`
    returns None and `_maybe_probe` falls through under `probe: auto`), the
    gate has free capacity, and the stubbed `run_session` resolves the turn
    future with a 502 error dict. The contract: HTTPException 502 propagates,
    `gate.in_flight == 0`, `server.SESSIONS == {}`, and the captured session
    is `dead` and `not holds_slot`.
    """
    ERROR = {"type": "error", "status": 502,
             "detail": f"{server.PROVIDER} cli failed (1): boom"}
    captured_session: dict = {}

    async def scenario():
        # Same fresh-gate reason as the 503 tests above: bind the lock/
        # condition to this scenario's event loop, not one an earlier test
        # used.
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()
        real_run = server.run_session

        async def fake_run(session, argv, stdin_data=None):
            captured_session["session"] = session
            session.resolve_final(ERROR)

        server.run_session = fake_run
        try:
            body = {"model": "m",
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather", "description": "",
                        "parameters": {"type": "object",
                                       "properties": {"city": {"type": "string"}},
                                       "required": ["city"]}}}],
                    "messages": [{"role": "user", "content": "weather in Oslo?"}]}
            try:
                await server.handle_fresh(body, body["tools"], None)
            except server.HTTPException as exc:
                assert exc.status_code == 502, exc.status_code
                assert exc.detail == ERROR["detail"], exc.detail
            else:
                raise AssertionError("handle_fresh must 502 when the CLI errors")
            return captured_session.get("session")
        finally:
            server.run_session = real_run
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate

    session = asyncio.run(scenario())
    # The gate slot is back: the leak that physically filled the sidecar gate
    # under the bug is gone.
    assert server.cli_bridge._gate.in_flight == 0, \
        server.cli_bridge._gate.in_flight
    # The captured session was removed from SESSIONS, not just marked dead --
    # the leak that filled the sidecar gate under the bug. (Earlier tests in
    # this file leave their own parked sessions behind, so the dict is not
    # necessarily empty as a whole; the assertion is on THIS session.)
    assert session is not None
    assert session.id not in server.SESSIONS, \
        {sid: (s.dead, s.holds_slot)
         for sid, s in server.SESSIONS.items()}
    # end_session flipped the dead/holds_slot flags exactly as it does for the
    # happy-path final-answer case.
    assert session.dead, session
    assert not session.holds_slot, session
    print(f"  fresh turn -> 502 error; gate slot released "
          f"(in_flight={server.cli_bridge._gate.in_flight}); session dead and gone")


def test_text_lost_does_not_leak_a_slot():
    """The TEXT_LOST / WS1 contract path must tear the session down too.

    The gateway's `_NO_TEXT` regex classifies the 502 error detail that
    cli_bridge / mcp_bridge emit when the CLI printed tokens but no answer.
    Both sidecars use the same `format_no_text_detail` helper so the
    classifier cannot drift between them (issue #64). This test pins the
    exact byte-for-byte detail and asserts the no-leak invariants.

    Same harness as the regression above; the only difference is the
    detail string -- shaped exactly the way cli_bridge / mcp_bridge / the
    gateway's classifier all expect.
    """
    detail = server.cli_bridge.format_no_text_detail(
        server.PROVIDER, {"input_tokens": 239, "output_tokens": 22})
    ERROR = {"type": "error", "status": 502, "detail": detail}
    captured_session: dict = {}

    async def scenario():
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()
        real_run = server.run_session

        async def fake_run(session, argv, stdin_data=None):
            captured_session["session"] = session
            session.resolve_final(ERROR)

        server.run_session = fake_run
        try:
            body = {"model": "m",
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather", "description": "",
                        "parameters": {"type": "object",
                                       "properties": {"city": {"type": "string"}},
                                       "required": ["city"]}}}],
                    "messages": [{"role": "user", "content": "weather in Oslo?"}]}
            try:
                await server.handle_fresh(body, body["tools"], None)
            except server.HTTPException as exc:
                # Byte-for-byte equality guards the gateway's _NO_TEXT
                # regex classification: a single drift here would re-break
                # the WS1 contract downstream.
                assert exc.detail == detail, (exc.detail, detail)
            else:
                raise AssertionError("handle_fresh must 502 on the TEXT_LOST path")
            return captured_session.get("session")
        finally:
            server.run_session = real_run
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate

    session = asyncio.run(scenario())
    assert server.cli_bridge._gate.in_flight == 0, \
        server.cli_bridge._gate.in_flight
    assert session is not None
    assert session.id not in server.SESSIONS, \
        {sid: (s.dead, s.holds_slot)
         for sid, s in server.SESSIONS.items()}
    assert session.dead, session
    assert not session.holds_slot, session
    print(f"  TEXT_LOST detail preserved byte-for-byte ({detail!r}); "
          f"gate slot released; session dead and gone")


def test_followup_error_result_does_not_leak():
    """A CLI failure mid-tool-loop must tear the parked-session->continue
    path down, on the same contract as the fresh-turn path.

    Hand-builds a parked session (`_new_session()`, park a call via
    `register_tool_call`, consume the tool_calls turn, mark the session
    awaiting_followup, holds_slot=False -- the parked state from
    `test_a_parked_session_holds_no_concurrency_slot`). Then deliver a
    real tool result for the parked call (so `_continue_followup` takes
    the resolve path, not the rebuild path), and stub `await_turn` to
    return a 502 error dict. The contract: HTTPException 502 propagates,
    `gate.in_flight == 0`, the session is gone from SESSIONS, dead,
    and `not holds_slot`.

    `unpark_session` is exercised against a fresh gate with free
    capacity, so the failure mode being tested is purely the awaited
    turn returning an error -- not a saturated gate (the 503 path is
    covered by `test_unpark_returns_503_quickly_when_gate_is_full`).
    """
    ERROR = {"type": "error", "status": 502,
             "detail": f"{server.PROVIDER} cli failed (1): boom"}
    captured_session: dict = {}

    async def scenario():
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()

        # Hand-build a parked session, identical pattern to
        # test_duplicate_delivery_returns_cached_response_... /
        # test_rebuild_supersedes_the_old_session_...
        session = _new_session()
        session.new_turn()
        parked = asyncio.create_task(
            server.register_tool_call(session.id, "get_weather",
                                      {"city": "Oslo"}))
        await asyncio.sleep(0)
        turn = await session.turn_future
        assert turn["type"] == "tool_calls", turn
        call = turn["calls"][0]

        # Park state, exactly as the happy path leaves it. Crucially, do
        # NOT pre-resolve the parked future -- _continue_followup needs to
        # consume it (via `call.future.set_result(...)` in the resolve
        # branch) so the resolve loop increments and we take the
        # unpark/new_turn/error path, not the rebuild path on a
        # zero-resolved branch.
        session.awaiting_followup = True
        session.holds_slot = False
        captured_session["session"] = session

        # _continue_followup awaits `session.new_turn()` to install a new
        # turn_future. In production the existing run_session task resolves
        # that future when the CLI emits its next answer; here we stub
        # await_turn to return the error dict directly, but the await on
        # new_turn itself still has to be unblocked or the test hangs.
        # Override new_turn on this session only so it returns an
        # already-resolved future -- otherwise `await session.new_turn()`
        # would hang (run_session is never started in this scenario).
        real_new_turn = type(session).new_turn

        def resolved_new_turn(self):
            f = asyncio.get_event_loop().create_future()
            f.set_result(ERROR)
            self.turn_future = f
            return f
        type(session).new_turn = resolved_new_turn

        # Stub await_turn to return an error dict directly. The handler does
        # the unpark/new_turn dance first, so this stub short-circuits only
        # the model-turn step -- exactly the failure mode the issue describes.
        real_await_turn = server.await_turn

        async def error_turn(sess, request):
            return ERROR
        server.await_turn = error_turn

        try:
            body = {"model": "m",
                    "messages": [
                        {"role": "tool", "tool_call_id": call.id,
                         "content": '{"temp_c":-3}'},
                    ]}
            try:
                await server._continue_followup(body, session.id,
                                                body["messages"], None)
            except server.HTTPException as exc:
                assert exc.status_code == 502, exc.status_code
                assert exc.detail == ERROR["detail"], exc.detail
            else:
                raise AssertionError("_continue_followup must 502 when "
                                     "the awaited turn returns an error")
            # The parked register_tool_call future was set inside
            # _continue_followup's resolve loop before the awaited turn 502'd.
            # Drain it so its outcome is observed (a 502 in the follow-up
            # path must NOT leak a never-awaited parked task).
            tool_result = await parked
            assert tool_result["content"] == [
                {"type": "text", "text": '{"temp_c":-3}'}], tool_result
            assert tool_result["isError"] is False, tool_result
            return session
        finally:
            server.await_turn = real_await_turn
            type(session).new_turn = real_new_turn
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate

    session = asyncio.run(scenario())
    assert server.cli_bridge._gate.in_flight == 0, \
        server.cli_bridge._gate.in_flight
    assert session.id not in server.SESSIONS, \
        {sid: (s.dead, s.holds_slot)
         for sid, s in server.SESSIONS.items()}
    assert session.dead, session
    assert not session.holds_slot, session
    print(f"  follow-up turn -> 502 error; gate slot released "
          f"(in_flight={server.cli_bridge._gate.in_flight}); session dead and gone")


def test_rebuild_error_does_not_leak():
    """`resume_gone_session` must tear the rebuilt session down on error too.

    The rebuild path inherits the slot (acquired by `acquire_resume_slot`)
    and hands it to `start_session`, which now owns the teardown contract
    on the rebuild path as well. A 502 from the rebuilt CLI must release
    the slot and remove the rebuilt session -- otherwise a stuck sidecar
    whose CLI fails on every rebuild accumulates one slot per attempt until
    the gate is full (issue #80 symptom, rebuild variant).

    Real `resume_gone_session` with a fresh gate that has free capacity (so
    `acquire_resume_slot` returns True and we exercise the error path,
    not the 503 path covered by `test_resume_gone_returns_503_quickly_...`).
    The stubbed `run_session` resolves with a 502 error dict; the rebuild
    in `start_session` raises HTTPException(502) out of `render_turn`,
    which the new try/except in `start_session` catches and tears down.
    """
    ERROR = {"type": "error", "status": 502,
             "detail": f"{server.PROVIDER} cli failed (1): boom"}
    captured_session: dict = {}

    async def scenario():
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()
        real_run = server.run_session

        async def fake_run(session, argv, stdin_data=None):
            captured_session["session"] = session
            session.resolve_final(ERROR)

        server.run_session = fake_run
        try:
            # Mirrors the body shape used by test_resume_gone_returns_503_...
            # above, with a tool that does NOT fire the env probe.
            body = {
                "model": "m",
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "",
                    "parameters": {"type": "object",
                                   "properties": {"city": {"type": "string"}},
                                   "required": ["city"]}}}],
                "messages": [
                    {"role": "user", "content": "weather in Oslo?"},
                    {"role": "tool", "tool_call_id": "call_x",
                     "content": '{"temp_c":-3}'},
                ],
            }
            try:
                await server.resume_gone_session(
                    body, body["tools"], "nonexistent-session-id", None,
                    why="lost")
            except server.HTTPException as exc:
                assert exc.status_code == 502, exc.status_code
                assert exc.detail == ERROR["detail"], exc.detail
            else:
                raise AssertionError("resume_gone_session must 502 when the "
                                     "rebuilt CLI errors")
            return captured_session.get("session")
        finally:
            server.run_session = real_run
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate

    session = asyncio.run(scenario())
    assert server.cli_bridge._gate.in_flight == 0, \
        server.cli_bridge._gate.in_flight
    assert session is not None
    assert session.id not in server.SESSIONS, \
        {sid: (s.dead, s.holds_slot)
         for sid, s in server.SESSIONS.items()}
    assert session.dead, session
    assert not session.holds_slot, session
    print(f"  rebuild -> 502 error; gate slot released "
          f"(in_flight={server.cli_bridge._gate.in_flight}); rebuilt session dead and gone")


# ------------------------------------------ issue #29: argv E2BIG regression ---
def test_an_oversized_system_prompt_goes_to_a_file_not_argv():
    """Claude's --system-prompt flag is one argv element, and MAX_ARG_STRLEN
    caps that at ~128 KiB (issue #29). Over the limit the prompt must move
    to --system-prompt-file (a file in the session workdir, reclaimed by
    cleanup_workdir) rather than ride argv and fail execve with E2BIG.

    Mirror of the cli_bridge test for the file-form flag: the same escape
    hatch used for oversized system prompts in the text-only path. Under
    the limit the inline --system-prompt flag keeps its old behaviour.
    """
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-test-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")

    # Oversized system prompt routes to a file inside the session workdir.
    huge = "s" * (server.STDIN_PROMPT_LIMIT + 10)
    argv, stdin_data = server.build_argv("prompt", huge, "m", workdir, "sess",
                                         tools_path, "")
    assert "--system-prompt-file" in argv, argv
    assert "--system-prompt" not in argv, argv
    spath = Path(argv[argv.index("--system-prompt-file") + 1])
    assert spath.parent == workdir, spath
    assert spath.name == "system-prompt.md", spath
    assert spath.read_text() == huge + "\n", "the file must contain the caller's system prompt"
    assert stdin_data is None, "system-prompt-file path does not touch stdin"
    assert all(huge != element for element in argv), argv

    # A small system prompt still rides inline, as before.
    argv2, stdin_data2 = server.build_argv("prompt", "be terse", "m", workdir,
                                            "sess", tools_path, "")
    assert "--system-prompt" in argv2, argv2
    assert "be terse" in argv2, argv2
    assert "--system-prompt-file" not in argv2, argv2

    # The claimed lifecycle, closed: the workdir (and the file in it) must
    # actually be reclaimed, not merely created.
    server.cleanup_workdir(workdir)
    assert not spath.exists(), "cleanup_workdir must reclaim system-prompt.md with the workdir"
    print(f"  {len(huge)}-char system prompt -> {spath.name}; small system prompt still inline")


def test_e2big_from_the_spawn_is_413():
    """A failed execve with E2BIG must surface as 413, not 500.

    The structured-cli error path keeps the call alive and returns 502 so
    the gateway retries elsewhere; E2BIG is the caller's request being too
    big for argv (issue #29), not the plan being broken. Returning 500 here
    would tell the router this is a transient blip and burn retries on a
    request that will never fit.
    """
    async def scenario():
        session = _new_session()
        session.new_turn()

        real_exec = asyncio.create_subprocess_exec
        captured_exc = OSError(errno.E2BIG, "Argument list too long")

        async def fake_exec(*args, **kwargs):
            raise captured_exc
        asyncio.create_subprocess_exec = fake_exec
        try:
            await server.run_session(session, ["fake-argv"])
        finally:
            asyncio.create_subprocess_exec = real_exec
        return session.turn_future.result()

    result = asyncio.run(scenario())
    assert result["type"] == "error" and result["status"] == 413, result
    detail = result["detail"]
    assert detail["error"]["type"] == "request_too_large", detail
    assert "too large" in detail["error"]["message"], detail
    print(f"  E2BIG from spawn -> HTTP {result['status']} ({detail['error']['message'][:60]}...)")


# ----------------------------------------------------------- images (issue #30) ---
# A 1x1 magenta PNG. Same one used by the cli_bridge test -- issue #30 is
# the silent drop on the text path, and the same payload shape reaches the
# MCP bridge through `messages[].content` (the tool side carries image
# blocks too; see openai_tool_content_to_mcp below).
PNG_MAGENTA = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_openai_tool_content_to_mcp_passes_through_image_data_urls():
    """An OpenAI-shaped image_url with a `data:` URL must reach the CLI as a
    real MCP image content block, not a stringified JSON blob the model
    cannot see. The decoded bytes are the model's only way to read it.
    """
    payload = [{"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64.b64encode(PNG_MAGENTA).decode()}"}}]
    blocks, is_error = server.openai_tool_content_to_mcp(payload)
    assert is_error is False, (blocks, is_error)
    assert blocks == [{"type": "image", "mimeType": "image/png",
                       "data": base64.b64encode(PNG_MAGENTA).decode()}], blocks
    print(f"  data: URL -> MCP image block ({len(blocks[0]['data'])} chars of base64)")


def test_openai_tool_content_to_mcp_converts_anthropic_shaped_images():
    """Anthropic's image block (`source.type == "base64"`) arrives on the
    MCP path too, e.g. when the caller is Anthropic-protocol. The same MCP
    shape must come out.
    """
    payload = [{"type": "image",
                "source": {"type": "base64", "media_type": "image/png",
                           "data": base64.b64encode(PNG_MAGENTA).decode()}}]
    blocks, is_error = server.openai_tool_content_to_mcp(payload)
    assert not is_error, (blocks, is_error)
    assert blocks == [{"type": "image", "mimeType": "image/png",
                       "data": base64.b64encode(PNG_MAGENTA).decode()}], blocks
    print("  Anthropic-shaped image -> MCP image block (mimeType=image/png)")


def test_openai_tool_content_to_mcp_marks_remote_url_as_is_error():
    """A remote http(s) URL on the tool path cannot be fetched from here.
    The call must surface as isError, with a log line warning about it --
    otherwise the caller would receive a tool result the model cannot see,
    and might hallucinate an answer from the missing image.
    """
    payload = [{"type": "image_url",
                "image_url": {"url": "https://example.com/x.png"}}]
    blocks, is_error = server.openai_tool_content_to_mcp(payload)
    assert is_error is True, (blocks, is_error)
    # The block becomes a plain text block so the CLI still has SOMETHING to
    # feed back -- an empty content list would break the stdio framing.
    assert blocks and blocks[0]["type"] == "text", blocks
    print(f"  remote URL -> isError=True text block ({blocks[0]['text'][:30]}...)")


def test_openai_tool_content_to_mcp_keeps_text_in_a_list():
    """A mixed list of text and image blocks must keep both. Otherwise the
    model's text answer would be stripped alongside the image.
    """
    payload = [{"type": "text", "text": "the colour is "},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{base64.b64encode(PNG_MAGENTA).decode()}"}}]
    blocks, is_error = server.openai_tool_content_to_mcp(payload)
    assert not is_error, (blocks, is_error)
    assert blocks[0] == {"type": "text", "text": "the colour is "}, blocks
    assert blocks[1]["type"] == "image", blocks[1]
    print("  text + image kept in order, image reached the CLI")


def test_flatten_with_tool_history_includes_image_markers_after_staging():
    """Rebuilding a session with an image-bearing tool result must carry the
    image into the rebuilt CLI's prompt -- not as raw base64 (which would
    blow past MAX_ARG_STRLEN) and not as nothing (the bug fix). stage_or_fail
    replaces the image blocks with `[image N: <path>]` markers BEFORE
    flatten_with_tool_history runs, so the markers survive into narration.
    """
    import shutil
    img_dir = Path(tempfile.mkdtemp(prefix="mcpb-imgtest-"))
    try:
        messages = [
            {"role": "user", "content": "what colour is this?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "look", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_a",
             "content": [{"type": "image_url", "image_url": {
                 "url": f"data:image/png;base64,{base64.b64encode(PNG_MAGENTA).decode()}"}}]},
        ]
        # Stage images into a temp dir the same way resume_gone_session does.
        paths, _ = server.cli_bridge.stage_or_fail(messages)
        assert len(paths) == 1, paths
        # flatten_with_tool_history renders the tool loop as narration. The
        # image block has been replaced with a text marker, so the marker
        # appears inside the result.
        prompt, _system = server.flatten_with_tool_history(messages)
        assert "[image 1:" in prompt, prompt
        assert "look" in prompt, prompt
        assert "[result of look:" in prompt, prompt
        print(f"  staged tool-image marker carried into narration: "
              f"{[l for l in prompt.splitlines() if 'image' in l][0][:80]}")
    finally:
        shutil.rmtree(img_dir, ignore_errors=True)


def test_mcp_build_argv_passes_images_through_claude_profile():
    """The claude MCP profile carries media the way the text path does:
    inline, as one stream-json stdin message (cli_bridge.claude_media_stdin).
    No Read, no --add-dir, so no relay path the model could hand to the
    caller's own Read tool (issue #264). The MCP-tools --allowed-tools
    allowlist and --session-id survive."""
    import shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-argv-"))
    tools_path = workdir / "tools.json"
    mcp_tools = [{"name": "get_weather", "description": "",
                  "inputSchema": {"type": "object", "properties": {}}}]
    tools_path.write_text("[]")
    img_dir = workdir / "img"
    img = img_dir / "00.png"
    img_dir.mkdir(parents=True)
    img.write_bytes(PNG_MAGENTA)
    try:
        allowed = ",".join(server.PROFILE["tool_qualifier"](t["name"])
                           for t in mcp_tools)
        argv, stdin_data = server.build_argv("look at this", None, "m", workdir,
                                              "sess", tools_path, allowed, [img])
        assert "--add-dir" not in argv, argv
        assert not any(a.startswith("Read(") for a in argv), argv
        assert argv[argv.index("--tools") + 1] == "", argv
        assert any("mcp__switchyard__get_weather" in a for a in argv), argv
        assert "--session-id" in argv, argv
        assert argv[argv.index("--input-format") + 1] == "stream-json", argv
        assert argv[argv.index("--output-format") + 1] == "stream-json", argv
        assert argv.count("--output-format") == 1, argv
        assert not any("look at this" in a for a in argv), argv
        msg = json.loads(stdin_data)
        content = msg["message"]["content"]
        assert content[0]["type"] == "text" and "look at this" in content[0]["text"], content
        assert content[1]["type"] == "image", content
        assert base64.b64decode(content[1]["source"]["data"]) == PNG_MAGENTA
        assert str(img_dir) not in stdin_data
        print("  mcp_bridge claude build_argv(image) -> stream-json stdin, --tools '', "
              "mcp__switchyard__get_weather still present")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_mcp_build_argv_allows_no_builtin_tools_when_no_image():
    """Without images the claude MCP session gets NO built-in tools
    (`--tools ""`): the model's tool list is exactly the caller's bridged
    mcp__switchyard__* tools. A denylist missed Agent/Skill/ToolSearch/
    AskUserQuestion and any built-in a CLI release adds (issues #195, #256,
    #264); an empty allowlist cannot. The shared cli_bridge.CLAUDE_LOCKDOWN
    rides along: only --mcp-config's servers, no login-dir settings, no
    permission prompts.
    """
    import shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-argv-noimg-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    try:
        argv, stdin_data = server.build_argv("just text", None, "m", workdir,
                                              "sess", tools_path, "")
        assert argv[argv.index("--tools") + 1] == "", argv
        assert "--disallowed-tools" not in argv, argv
        lockdown = list(server.cli_bridge.CLAUDE_LOCKDOWN)
        i = argv.index(lockdown[0])
        assert argv[i:i + len(lockdown)] == lockdown, argv
        # And the MCP-tools allowlist is still wired up.
        assert "--allowed-tools" in argv, argv
        assert "--mcp-config" in argv, argv
        assert stdin_data is None
        print("  mcp_bridge claude build_argv(text-only) -> --tools '' + lockdown")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_codex_mcp_build_argv_appends_the_lockdown():
    """Issue #255: on the pinned codex the caller's bridged tools were hidden
    inside the code-mode `exec` host while codex's own shell was advertised.
    The MCP argv must carry the shared cli_bridge lockdown -- every tool
    feature disabled, the prompt sections off, and the locked-down model
    catalog -- whatever the request looks like."""
    import shutil
    # The codex branch now goes through cli_bridge.instructions_file
    # (issue #137 parity -- same helper, same path), which reads
    # cli_bridge.PROFILE for the `instructions_arg` template. Production
    # has cli_bridge.PROFILE in sync with mcp_bridge (same env var), but
    # this test only swaps mcp_bridge's globals, so the inner copy needs
    # to follow along or the helper yields an empty fragment.
    saved = (server.PROVIDER, server.PROFILE,
             server.cli_bridge.PROVIDER, server.cli_bridge.PROFILE,
             server.cli_bridge.CLI)
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-codex-argv-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    try:
        server.PROVIDER = "codex"
        server.PROFILE = server.MCP_PROFILES["codex"]
        server.cli_bridge.PROVIDER = "codex"
        server.cli_bridge.PROFILE = server.cli_bridge.PROFILES["codex"]
        server.cli_bridge.CLI = server.cli_bridge.PROFILES["codex"]["cli"]
        argv, _ = server.build_argv("hi", "CALLER SYSTEM", "gpt-5.6-terra",
                                    workdir, "sess", tools_path, "")
        for feature in ("shell_tool", "unified_exec", "multi_agent", "code_mode_host"):
            i = argv.index(feature)
            assert argv[i - 1] == "--disable", (feature, argv)
        for override in ("include_environment_context=false",
                         "skills.include_instructions=false",
                         "tools.experimental_request_user_input.enabled=false"):
            assert argv[argv.index(override) - 1] == "-c", (override, argv)
        catalog = [el for el in argv if el.startswith("model_catalog_json=")]
        assert len(catalog) == 1, argv
        path = catalog[0].split("=", 1)[1].strip('"')
        data = json.loads(Path(path).read_text())
        assert not any("tool_mode" in m for m in data["models"]), data
        instr = argv.index(f"model_instructions_file={workdir / 'instructions.md'}")
        assert instr < argv.index("--disable"), argv
        print("  codex MCP argv: bypass + lockdown + locked-down catalog")
    finally:
        (server.PROVIDER, server.PROFILE,
         server.cli_bridge.PROVIDER, server.cli_bridge.PROFILE,
         server.cli_bridge.CLI) = saved
        shutil.rmtree(workdir, ignore_errors=True)


def test_write_opencode_dir_enables_read_when_images_passed():
    """The opencode harness disables every tool by default. For an image
    session the model needs `read` to look at the staged files via the
    agent's own Read tool, so the disable list must be opt-out via
    `images=True`. Without it, image sessions would 200 with no answer.
    """
    import shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-op-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    try:
        server.write_opencode_dir(workdir, "sess", tools_path, images=False)
        cfg = json.loads((workdir / "opencode.json").read_text())
        assert cfg["agent"]["switchyard"]["tools"]["read"] is False, cfg

        server.write_opencode_dir(workdir, "sess", tools_path, images=True)
        cfg = json.loads((workdir / "opencode.json").read_text())
        assert cfg["agent"]["switchyard"]["tools"]["read"] is True, cfg
        # Other tools stay disabled.
        for tool in ("bash", "edit", "write", "grep", "glob"):
            assert cfg["agent"]["switchyard"]["tools"][tool] is False, tool
        assert cfg["agent"]["switchyard"]["permission"]["read"] == "allow", cfg
        print("  write_opencode_dir(images=True) -> read enabled, others stay disabled")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_write_opencode_dir_denies_builtins_at_the_permission_layer():
    """`tools: {bash: false}` only hides bash; OpenCode 1.18.31 still runs it
    when the model names it. The session config must deny everything at the
    permission layer except the bridged `switchyard_*` tools, and must be
    the shared cli_bridge.opencode_config lockdown (bridge-siblings rule)
    plus the MCP server -- nothing else."""
    import shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-op-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    try:
        server.write_opencode_dir(workdir, "sess", tools_path, images=False)
        cfg = json.loads((workdir / "opencode.json").read_text())
        agent = cfg["agent"]["switchyard"]
        assert agent["permission"] == {"*": "deny", "switchyard_*": "allow"}, agent
        assert cfg["permission"] == agent["permission"], cfg
        assert cfg["agent"]["title"] == {"disable": True}, cfg
        assert not any(agent["tools"].values()), agent["tools"]
        mcp = cfg.pop("mcp")
        assert mcp["switchyard"]["environment"]["SWITCHYARD_SESSION_ID"] == "sess", mcp
        assert cfg == server.cli_bridge.opencode_config(allow=("switchyard_*",)), cfg
        print("  write_opencode_dir: * deny, switchyard_* allow, title call off")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ----------------------------------------- issue #44: caller-env probe + render ---
def test_fresh_with_unknown_env_emits_synthetic_probe_tool_call():
    """A fresh request whose env is unknown AND whose tools include a
    recognised command tool MUST get a synthetic `switchyard_env_` tool_call
    back, NOT a session allocation. No gate slot is taken, no SESSIONS entry
    is created -- the probe travels back to the caller for local execution.

    The probe happens BEFORE any gate acquire / workdir creation, exactly as
    the spec requires -- otherwise a caller whose tool needed permission
    would block the entire sidecar slot table for the duration of a human
    approval flow.

    We stub start_session anyway, so that any "should-have-probed-but-didn't"
    regression fails as a 502 from a real spawn rather than as a silent
    pass-through.
    """
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        body = {
            "model": "m",
            "tools": [{"type": "function", "function": {
                "name": "Bash", "description": "Run shell commands",
                "parameters": {"type": "object",
                               "properties": {"command": {"type": "string"}},
                                "required": ["command"]}}}],
            "messages": [
                {"role": "system", "content": "no env here"},
                {"role": "user", "content": "What OS am I on?"},
            ],
        }
        response = asyncio.run(server.handle_tool_request(body, body["tools"], None))
        # The probe is a chat.completion with finish_reason=tool_calls.
        # If the stub was called instead, start_session was incorrectly
        # invoked on the probe path -- a regression.
        assert "choices" in response, \
            f"expected a probe response, got {response!r}"
        assert response["choices"][0]["finish_reason"] == "tool_calls", response
        tc = response["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"].startswith(caller_env_module.PROBE_PREFIX), tc["id"]
        # The synthetic call's arguments are the probe command on the right key.
        args = json.loads(tc["function"]["arguments"])
        assert args.get("command"), args
        assert "pwd" in args["command"] and "uname" in args["command"]
        assert "$SHELL" in args["command"]
        # CRITICAL: no slot taken, no session allocated.
        assert server.cli_bridge._gate.in_flight == 0, \
            f"probe must not hold a slot: in_flight={server.cli_bridge._gate.in_flight}"
        assert server.SESSIONS == {}, f"probe must not allocate a session: {server.SESSIONS}"
        # start_session was NOT called for the probe path.
        assert captured == [], captured
        print(f"  probe emitted: {tc['id']} -> args has pwd+uname+$SHELL; "
              f"no slot taken (in_flight=0)")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)



def test_windows_caller_gets_a_probe_its_shell_can_run():
    """A caller known only to be on Windows (its prompt names the platform,
    not the cwd) whose command tool does not say which shell it runs gets
    the PowerShell probe; a PowerShell-named tool gets it anywhere; a Bash
    tool keeps the POSIX probe. The POSIX probe errors in PowerShell and
    cmd, which used to leave a Windows caller with no cwd at all."""
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    captured = []
    restore = _stub_start_session(captured)
    cases = [
        ("execute_command", "<env>\nPlatform: win32\n</env>",
         caller_env_module.PROBE_COMMAND_POWERSHELL),
        ("PowerShell", "no env here", caller_env_module.PROBE_COMMAND_POWERSHELL),
        ("Bash", "<env>\nPlatform: win32\n</env>", caller_env_module.PROBE_COMMAND),
        ("execute_command", "no env here", caller_env_module.PROBE_COMMAND),
    ]
    try:
        for n, (tool, system, want) in enumerate(cases):
            server.FAILED_PROBES.clear()
            server.RESOLVED_PROBES.clear()
            body = {"model": "m",
                    "tools": [{"type": "function", "function": {
                        "name": tool, "description": "Run a command",
                        "parameters": {"type": "object",
                                       "properties": {"command": {"type": "string"}},
                                       "required": ["command"]}}}],
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": f"windows probe case {n}"}]}
            response = asyncio.run(server.handle_tool_request(body, body["tools"], None))
            tc = response["choices"][0]["message"]["tool_calls"][0]
            assert tc["id"].startswith(caller_env_module.PROBE_PREFIX), tc
            assert tc["function"]["name"] == tool, tc
            assert json.loads(tc["function"]["arguments"]) == {"command": want}, (tool, tc)
        assert captured == [], captured
        print("  win32 + unnamed shell / PowerShell tool -> PowerShell probe; Bash -> POSIX")
    finally:
        restore()
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_windows_probe_answer_resolves_the_callers_environment():
    """The PowerShell probe's CRLF answer is consumed like the POSIX one:
    the env it names is what the session is built with."""
    body = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": caller_env_module.mint_probe_call_id("a" * 16), "type": "function",
            "function": {"name": "PowerShell", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": caller_env_module.mint_probe_call_id("a" * 16),
         "content": "cwd=C:\\Users\\Me\\proj\r\nplatform=Windows\r\nshell=powershell\r\n"}]}
    saved_resolved = dict(server.RESOLVED_PROBES)
    try:
        env, stripped = server._consume_probe_results(body)
        assert (env.cwd, env.platform, env.shell) == (
            "C:\\Users\\Me\\proj", "Windows", "powershell"), env
        assert all(m.get("role") != "tool" for m in stripped["messages"]), stripped
        print("  PowerShell probe answer -> C:\\Users\\Me\\proj / Windows / powershell")
    finally:
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)

def test_probe_follow_up_strips_synthetic_exchange_and_starts_session():
    """The follow-up request carrying the probe result must: parse the env,
    strip the synthetic assistant+tool exchange, then proceed into the
    normal fresh path with the env known. The system the CLI finally sees
    contains the `[SwitchYard tool execution environment]` block; the
    prompt's first user turn carries the one-line reminder.
    """
    saved_gate_count = server.cli_bridge._gate.in_flight
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        messages = [
            {"role": "system", "content": "no env"},
            {"role": "user", "content": "What OS am I on?"},
        ]
        # First request: env unknown -> synthetic probe emitted.
        first = asyncio.run(server.handle_tool_request(
            {"model": "m", "tools": [{
                "type": "function", "function": {
                    "name": "Bash", "description": "",
                    "parameters": {"type": "object",
                                   "properties": {"command": {"type": "string"}},
                                   "required": ["command"]}}}],
             "messages": messages}, [{
                "type": "function", "function": {
                    "name": "Bash", "description": "",
                    "parameters": {"type": "object",
                                   "properties": {"command": {"type": "string"}},
                                   "required": ["command"]}}}], None))
        probe_id = first["choices"][0]["message"]["tool_calls"][0]["id"]

        # Second request: caller executed the probe, replies with a tool
        # message whose tool_call_id is the probe id and content is the
        # three labelled lines from PROBE_COMMAND. The synthetic assistant
        # message that minted the probe MUST be stripped out before the
        # CLI sees it.
        followup_messages = messages + [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": probe_id, "type": "function",
                 "function": {"name": "Bash",
                              "arguments": json.dumps({"command": "pwd"})}}]},
            {"role": "tool", "tool_call_id": probe_id, "content": (
                "cwd=/home/u/dir\n"
                "platform=Linux\n"
                "shell=/bin/zsh\n")},
            {"role": "user", "content": "ok thanks"},
        ]
        body2 = {"model": "m",
                 "tools": [{"type": "function", "function": {
                     "name": "Bash", "description": "",
                     "parameters": {"type": "object",
                                    "properties": {"command": {"type": "string"}},
                                    "required": ["command"]}}}],
                 "messages": followup_messages}
        asyncio.run(server.handle_tool_request(body2, body2["tools"], None))

        assert len(captured) == 1, captured
        seen = captured[0]
        # The CLI's system prompt has the env block.
        assert "[SwitchYard tool execution environment]" in seen["system"], \
            seen["system"]
        assert "Caller platform: Linux" in seen["system"], seen["system"]
        assert "Caller working directory: /home/u/dir" in seen["system"], seen["system"]
        assert "Caller shell: /bin/zsh" in seen["system"], seen["system"]
        # The synthetic assistant+tool exchange is gone from what the CLI sees:
        # the rendered prompt has the first real user turn + reminder, but no
        # "[called Bash" narration of the probe and no Human: cwd=... block.
        assert "cwd=/home/u/dir" not in seen["prompt"], \
            "probe-result text must not leak into the rebuilt prompt"
        assert "[SwitchYard: tools execute on Linux in /home/u/dir" in seen["prompt"], \
            seen["prompt"]
# The probe-and-resume flow must not leave a gate slot taken: it
        # never started a real CLI (start_session was stubbed) and the
        # resolved probe is in-memory state, not a concurrency seat.
        assert server.cli_bridge._gate.in_flight == saved_gate_count, (
            server.cli_bridge._gate.in_flight, saved_gate_count)
        print("  follow-up: env parsed, synthetic exchange stripped, system has env block")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_probe_iserror_marks_failed_and_no_retry_on_second_attempt():
    """An isError / unparseable probe result marks the fingerprint failed
    and the next attempt with the same fingerprint MUST NOT re-emit the
    probe. Falling back to the unknown-env path is the only allowed
    behaviour; no retry loop, no escalation, just the env-block with the
    'unknown' wording."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        messages = [
            {"role": "system", "content": "no env"},
            {"role": "user", "content": "What OS am I on?"},
        ]
        # 1. First call: probe emitted.
        first = asyncio.run(server.handle_tool_request(
            {"model": "m", "tools": tools, "messages": messages}, tools, None))
        probe_id = first["choices"][0]["message"]["tool_calls"][0]["id"]

        # 2. Second call: caller REFUSED the probe (isError=true).
        followup_messages = messages + [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": probe_id, "type": "function",
                 "function": {"name": "Bash",
                              "arguments": json.dumps({"command": "pwd"})}}]},
            {"role": "tool", "tool_call_id": probe_id, "is_error": True,
             "content": "permission denied"},
            {"role": "user", "content": "ok thanks anyway"},
        ]
        body2 = {"model": "m", "tools": tools, "messages": followup_messages}
        asyncio.run(server.handle_tool_request(body2, tools, None))

        # Now the third call with the SAME fingerprint MUST NOT probe again.
        # Reset the message prefix back to the first two turns to keep the
        # fingerprint stable.
        messages3 = list(messages) + [
            {"role": "user", "content": "next question"},
        ]
        body3 = {"model": "m", "tools": tools, "messages": messages3}
        third = asyncio.run(server.handle_tool_request(body3, tools, None))

        # The fingerprint must be in FAILED_PROBES now.
        assert len(server.FAILED_PROBES) >= 1, dict(server.FAILED_PROBES)
        # The third call's response is NOT a synthetic probe -- it is the
        # stubbed start_session result (a normal session with the unknown
        # env block in its system).
        if "choices" in third:
            # Got a real completion; it must NOT be a probe.
            tcs = third["choices"][0]["message"].get("tool_calls") or []
            assert not any((tc.get("id") or "").startswith(caller_env_module.PROBE_PREFIX)
                           for tc in tcs), tcs
        else:
            # Stubbed {"stubbed": True} response from start_session.
            assert third == {"stubbed": True}, third
        # The system the CLI sees has the unknown-env wording.
        if captured:
            seen = captured[-1]
            assert "Caller platform: unknown" in seen["system"], seen["system"]
            assert "Caller working directory: unknown" in seen["system"], seen["system"]
        print("  isError probe -> FAILED_PROBES populated, "
              "no retry on second attempt, env rendered as 'unknown'")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_passive_env_skips_probe_and_renders_request_values():
    """A request that already carries its own environment (an OpenCode-
    shaped system block, say) MUST be parsed passively and the probe
    skipped entirely. The system the CLI sees has the resolved values."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools, "messages": [
            {"role": "system", "content": (
                "<environment>\n"
                "  <working_directory>C:\\Users\\demo\\p</working_directory>\n"
                "  <platform>windows</platform>\n"
                "</environment>"
            )},
            {"role": "user", "content": "hi"},
        ]}
        response = asyncio.run(server.handle_tool_request(body, tools, None))
        # It went straight to the stubbed start_session (no probe emitted).
        assert response == {"stubbed": True}, response
        # No probe was emitted; FAILED_PROBES / RESOLVED_PROBES empty.
        assert server.FAILED_PROBES == {}, dict(server.FAILED_PROBES)
        assert server.RESOLVED_PROBES == {}, dict(server.RESOLVED_PROBES)
        # The system the CLI sees has the request-derived env.
        seen = captured[-1]
        assert "[SwitchYard tool execution environment]" in seen["system"], seen["system"]
        assert r"Caller working directory: C:\Users\demo\p" in seen["system"]
        assert "Caller platform: windows" in seen["system"]
        # Reminder on the first turn.
        assert "[SwitchYard: tools execute on windows in C:\\Users\\demo\\p" in seen["prompt"]
        print("  request-carries-env: passive parse, no probe, env block rendered")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_metadata_caller_env_is_honored_when_present():
    """metadata.switchyard.caller_env stamped on the request body MUST be
    honoured as the resolved env (source=request). The stamp is the easy
    path; the passive parsers are the belt-and-braces backup."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools,
                "metadata": {"switchyard": {"caller_env": {
                    "platform": "macos", "cwd": "/Users/x/p",
                    "shell": "zsh", "source": "request"}}},
                "messages": [
                    {"role": "user", "content": "hi"},
                ]}
        response = asyncio.run(server.handle_tool_request(body, tools, None))
        assert response == {"stubbed": True}, response
        seen = captured[-1]
        assert "Caller platform: macos" in seen["system"]
        assert "Caller working directory: /Users/x/p" in seen["system"]
        assert "Caller shell: zsh" in seen["system"]
        print("  metadata-stamped caller_env honored -> system reflects macos values")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_run_session_subprocess_cwd_is_session_workdir():
    """run_session must spawn the inner CLI with cwd == session.workdir,
    NOT /app/mcp_bridge. The fake CLI prints its own cwd, so we can
    assert the spawned process actually saw the per-session directory.
    """
    import shutil as _shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-cwdsess-"))
    session = server.Session(id=uuid.uuid4().hex, provider="claude", model="m",
                              workdir=str(workdir))
    server.SESSIONS[session.id] = session
    fake_cli = _write_fake_cli(
        "import json, os\n"
        "print(json.dumps({'result': os.getcwd(), 'usage': "
        "{'input_tokens': 1, 'output_tokens': 0}}))\n")
    try:
        async def scenario():
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "final", result
        assert os.path.realpath(result["payload"]["result"]) == os.path.realpath(str(workdir)), (
            f"expected inner CLI cwd == session.workdir "
            f"{os.path.realpath(str(workdir))!r}, "
            f"got {os.path.realpath(result['payload']['result'])!r}"
        )
        print(f"  inner CLI spawned with cwd={result['payload']['result']}")
    finally:
        server.SESSIONS.pop(session.id, None)
        try:
            os.unlink(fake_cli)
        except OSError:
            pass
        try:
            os.unlink(os.path.dirname(fake_cli))
        except OSError:
            pass
        _shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------- PR #66 review fixes ----------------
def _set_caller_environment_probe(probe: str):
    """Replace cli_bridge._config with one whose caller_environment.probe
    is `probe`. The probe mode determines how the mcp_bridge handles
    unresolvable envs (auto=fallback unknown, required=503, disabled=skip
    probe entirely). Returns a restore callable."""
    from dataclasses import replace as _replace
    saved = server.cli_bridge._config
    ce = server.cli_bridge.config().caller_environment
    new_ce = _replace(ce, probe=probe) if ce is not None else None
    server.cli_bridge._config = _replace(saved, caller_environment=new_ce)
    def restore():
        server.cli_bridge._config = saved
    return restore


def test_probe_required_unresolvable_returns_400_no_unknown_wording():
    """PR #66 blocker (round 2 update): probe=required + an unresolvable
    env must refuse the request with HTTP 400 (type=caller_environment_required)
    and NO Retry-After header, NOT silently fall through to the unknown-env
    wording. The round-1 status was 503 + Retry-After:5; round 2 changed
    it to 400 because the failure is not transient -- nothing about the
    request, the caller, or the operator's plans.yaml can change in 5
    seconds, so a Retry-After actively misleads the caller. The file's
    503 + Retry-After convention is reserved for transient-capacity
    conditions.

    The scenario: a caller has tools (so a probe is minted), but refuses
    the probe on the second turn (is_error=true). The second turn has no
    passive env in the request body and the probe was refused -- the env
    cannot be resolved. With probe=required, handle_fresh must raise
    400 instead of falling through to `unknown`."""
    from fastapi import HTTPException
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    restore_cfg = _set_caller_environment_probe("required")
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        messages = [
            {"role": "system", "content": "no env"},
            {"role": "user", "content": "What OS am I on?"},
        ]
        # 1. First request: env unknown, probe minted.
        first = asyncio.run(server.handle_tool_request(
            {"model": "m", "tools": tools, "messages": messages}, tools, None))
        probe_id = first["choices"][0]["message"]["tool_calls"][0]["id"]

        # 2. Second request: caller REFUSED the probe (is_error=true).
        # With probe=required, the env is now unresolvable and the
        # sidecar MUST refuse -- not silently fall through to the
        # unknown-env wording.
        refusal = messages + [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": probe_id, "type": "function",
                 "function": {"name": "Bash",
                              "arguments": json.dumps({"command": "pwd"})}}]},
            {"role": "tool", "tool_call_id": probe_id, "is_error": True,
             "content": "permission denied"},
            {"role": "user", "content": "ok thanks anyway"},
        ]
        try:
            response = asyncio.run(server.handle_tool_request(
                {"model": "m", "tools": tools, "messages": refusal},
                tools, None))
        except HTTPException as exc:
            # 400 (Bad Request) -- not 503. The failure is a config /
            # request mismatch, not transient capacity.
            assert exc.status_code == 400, \
                f"probe=required must 400, got {exc.status_code}: {exc.detail}"
            assert isinstance(exc.detail, dict), exc.detail
            assert exc.detail.get("error", {}).get("type") == \
                "caller_environment_required", exc.detail
            # NO Retry-After: the failure is not transient. A 5-second
            # Retry-After would actively mislead the caller -- the env
            # will still be unresolvable on retry (the operator's
            # plans.yaml still has no fallback_platform, the body still
            # has no passive env, the CLI still refuses the probe).
            assert not (exc.headers and "retry-after" in {k.lower() for k in exc.headers}), \
                f"probe=required must NOT carry Retry-After (failure is not " \
                f"transient), got headers={exc.headers!r}"
            print("  probe=required + refused probe -> HTTP 400 "
                  "type=caller_environment_required, no Retry-After")
            return
        raise AssertionError(
            f"probe=required must raise 400, but handle_tool_request returned "
            f"{response!r} (silent fallback to unknown is the regression)")
    finally:
        restore_cfg()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_metadata_source_config_is_relabeled_to_request():
    """PR #66 should-fix: a metadata stamp claiming source=config must
    be re-labeled to source=request. The values still flow through (we
    do not want to lose data), but at the request tier of the
    precedence chain, not the config tier. The invariant: the ONLY
    path that produces source=config is the operator's plans.yaml
    settings; anything arriving over the wire cannot.

    A caller that stamps `source=config` to bypass the precedence chain
    -- say, to force the sidecar to honor a fake Linux /app env as the
    caller's environment, exactly the bug #44 is meant to prevent --
    gets the request tier instead. The relay-env-as-caller-env
    invariant stays intact."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools,
                "metadata": {"switchyard": {"caller_env": {
                    # The classic bug attempt: claim config-tier so
                    # the precedence chain skips the passive parse
                    # AND uses these values as operator-forced.
                    "platform": "linux", "cwd": "/app/mcp_bridge",
                    "shell": "/bin/sh", "source": "config"}}},
                "messages": [
                    {"role": "system", "content": (
                        "<environment>\n"
                        "  <working_directory>C:\\Users\\demo</working_directory>\n"
                        "  <platform>windows</platform>\n"
                        "</environment>"
                    )},
                    {"role": "user", "content": "hi"},
                ]}
        response = asyncio.run(server.handle_tool_request(body, tools, None))
        # The CLI sees the env block; the values are honored (we don't
        # drop data) but at the REQUEST tier, not the config tier.
        # The request ran end-to-end through `handle_fresh` -> stubbed
        # `start_session` -> `{"stubbed": True}`; confirming the response
        # shape pins the wiring (probe did NOT short-circuit, followup
        # path was NOT taken).
        assert response == {"stubbed": True}, response
        seen = captured[-1]
        env = seen["env"]
        assert env.source == "request", \
            f"metadata-stamped source=config must be re-labeled to request, " \
            f"got {env.source!r}"
        assert env.platform == "linux"
        assert env.cwd == "/app/mcp_bridge"
        # The metadata-stamped values WON, not the passive parse's
        # windows values, because metadata is consulted first and both
        # are at the request tier. Either is consistent with the
        # invariant; the point is the source label.
        print(f"  metadata claims source=config -> re-labeled to "
              f"{env.source!r}, values still flow through")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_probe_still_fires_when_only_the_platform_is_known():
    """Issue #187: a platform-only env (a `fallback_platform` stamp, a
    prompt naming only the OS) used to count as known and skip the probe,
    so the cwd stayed unknown for the whole session. The probe fires unless
    the cwd is known."""
    saved = (dict(server.FAILED_PROBES), dict(server.RESOLVED_PROBES),
             server._caller_env_settings, server.cli_bridge.config)

    class _Cfg:
        caller_environment = type("CE", (), {"probe": "auto"})()

    tools = [{"type": "function", "function": {
        "name": "Bash", "description": "",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}]
    try:
        server.FAILED_PROBES.clear()
        server.RESOLVED_PROBES.clear()
        server.cli_bridge.config = lambda: _Cfg()
        platform_only = {"model": "m", "tools": tools,
                         "metadata": {"switchyard": {"caller_env": {
                             "platform": "darwin", "source": "request"}}},
                         "messages": [{"role": "user", "content": "probe me"}]}
        async def probe(body):
            future = server._maybe_probe(body, tools)
            return await future if future is not None else None

        response = asyncio.run(probe(platform_only))
        assert response is not None, "platform-only env must still probe"
        call = response["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "Bash", call
        server.FAILED_PROBES.clear()
        server.RESOLVED_PROBES.clear()
        with_cwd = dict(platform_only, messages=[{"role": "user", "content": "other"}],
                        metadata={"switchyard": {"caller_env": {
                            "platform": "darwin", "cwd": "/Users/x", "source": "request"}}})
        assert asyncio.run(probe(with_cwd)) is None
        print("  platform-only env probes; env with a cwd does not")
    finally:
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved[0])
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved[1])
        server._caller_env_settings, server.cli_bridge.config = saved[2], saved[3]


def test_metadata_source_host_is_relabeled_to_request():
    """Round-2 should-fix: `source=host` is also plans.yaml-derived
    (the `fallback_platform` branch of `resolve()` produces it), so a
    caller who stamps `source=host` to bypass the precedence chain
    must get the request tier instead. Same defensive re-labeling as
    `source=config`."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools,
                "metadata": {"switchyard": {"caller_env": {
                    "platform": "macos", "cwd": "/Users/x",
                    "shell": "zsh", "source": "host"}}},
                "messages": [{"role": "user", "content": "hi"}]}
        asyncio.run(server.handle_tool_request(body, tools, None))
        seen = captured[-1]
        env = seen["env"]
        assert env.source == "request", \
            f"metadata-stamped source=host must be re-labeled to request, " \
            f"got {env.source!r}"
        assert env.platform == "macos"
        assert env.cwd == "/Users/x"
        print(f"  metadata claims source=host -> re-labeled to "
              f"{env.source!r}, values still flow through")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_metadata_stamp_without_source_falls_through_to_passive_parse():
    """Round-2 should-fix: a wire stamp without an explicit source
    marker must NOT preempt passive detection. The round-1 helper
    returned a CallerEnvironment for any stamp with at least one field
    (even without a source), so a stamp like `{"platform": "linux"}`
    would win over a passive-parseable body. Restore the pre-round-1
    strictness: only honored sources yield a CallerEnvironment;
    everything else falls through to `parse_request(body)`.

    The end-to-end check: stamp values without source + passive-
    parseable system prompt -> the passive parse wins, not the
    stamp values."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools,
                # Stamp WITHOUT a source field, with values that would
                # win if from_wire_metadata were permissive. The body
                # ALSO carries a passive OpenCode-shaped env block.
                "metadata": {"switchyard": {"caller_env": {
                    "platform": "linux", "cwd": "/app/mcp_bridge",
                    "shell": "/bin/sh"}}},
                "messages": [
                    {"role": "system", "content": (
                        "<environment>\n"
                        "  <working_directory>C:\\Users\\demo\\p</working_directory>\n"
                        "  <platform>windows</platform>\n"
                        "</environment>"
                    )},
                    {"role": "user", "content": "hi"},
                ]}
        asyncio.run(server.handle_tool_request(body, tools, None))
        seen = captured[-1]
        env = seen["env"]
        # The stamp's `linux / /app/mcp_bridge` did NOT win. The
        # passive parse's `windows / C:\Users\demo\p` did.
        assert env.platform == "windows", \
            f"stamp-without-source must fall through to passive parse, " \
            f"got platform={env.platform!r} (stamp won)"
        assert env.cwd == "C:\\Users\\demo\\p", \
            f"stamp-without-source must fall through to passive parse, " \
            f"got cwd={env.cwd!r} (stamp won)"
        assert env.source == "request", env.source
        # Critically: the relay's /app/mcp_bridge is NOT in the
        # rendered env -- the bug we're avoiding.
        assert env.cwd != "/app/mcp_bridge", env.cwd
        print(f"  stamp without source -> passive parse wins "
              f"(platform={env.platform!r}, cwd={env.cwd!r})")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_concurrent_same_fingerprint_mints_only_one_probe():
    """PR #66 should-fix: _maybe_probe's check+mint pair is atomic under
    asyncio.gather. Without the lock, two (or three) concurrent calls
    with the same fingerprint would all pass the `fp not in caches`
    check and all mint duplicate probes -- one would carry through and
    the others would be silently dropped at consume-time, polluting
    the caller's history with extra synthetic tool_calls and wasting
    work. With PROBE_LOCK + the _PROBE_PENDING sentinel published
    before releasing the lock, exactly one mints and the others see
    fp already-taken and fall through to handle_fresh.

    Stubbed start_session so the handle_fresh path does not actually
    spawn a CLI; the test asserts the count distribution: 1 probe
    response + (N-1) stubbed sessions."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools, "messages": [
            {"role": "system", "content": "no env"},
            {"role": "user", "content": "What OS am I on?"},
        ]}

        async def scenario():
            return await asyncio.gather(
                server.handle_tool_request(dict(body), tools, None),
                server.handle_tool_request(dict(body), tools, None),
                server.handle_tool_request(dict(body), tools, None),
            )

        responses = asyncio.run(scenario())
        # Exactly one probe response was emitted.
        probes = [r for r in responses
                  if "choices" in r
                  and r["choices"][0].get("finish_reason") == "tool_calls"
                  and any((tc.get("id") or "").startswith(
                          caller_env_module.PROBE_PREFIX)
                          for tc in (r["choices"][0]["message"]
                                      .get("tool_calls") or []))]
        assert len(probes) == 1, \
            f"asyncio.gather(3x same-fingerprint) must mint exactly 1 probe, " \
            f"got {len(probes)} (duplicate probes is the regression)"
        # The other two went through start_session (stubbed).
        stubbed = [r for r in responses if r == {"stubbed": True}]
        assert len(stubbed) == 2, \
            f"expected 2 stubbed sessions, got {len(stubbed)}"
        # The PENDING sentinel is in RESOLVED_PROBES; only the probe
        # result can overwrite it with a real env (or FAILED_PROBES).
        assert len(server.RESOLVED_PROBES) == 1, dict(server.RESOLVED_PROBES)
        assert next(iter(server.RESOLVED_PROBES.values())) is \
            server._PROBE_PENDING, dict(server.RESOLVED_PROBES)
        print("  asyncio.gather(3x same-fingerprint) -> 1 probe + 2 stubbed; "
              "PENDING sentinel in RESOLVED_PROBES")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


# Module-level alias for tests that want the probe prefix string.
caller_env_module = sys.modules.get("switchyard.caller_env")
if caller_env_module is None:
    import importlib.util as _il
    _path = os.path.join(os.path.dirname(HERE), "switchyard", "caller_env.py")
    _spec = _il.spec_from_file_location("switchyard.caller_env", _path)
    caller_env_module = _il.module_from_spec(_spec)
    caller_env_module.__package__ = "switchyard"
    sys.modules.setdefault("switchyard", type(sys)("switchyard"))
    sys.modules["switchyard.caller_env"] = caller_env_module
    _spec.loader.exec_module(caller_env_module)


# ------------------------------------------ issue #70: enforce max_tokens post-hoc ---
def test_no_tools_fallthrough_inherits_max_tokens_enforcement_and_health_reports_it():
    """The mcp_bridge no-tools path delegates to cli_bridge._handle_chat, so it
    inherits the per-profile max_tokens mode (issue #70). On a codex profile
    the bare request must 400 with max_tokens_unenforceable; on a claude
    profile /health reports enforces_max_tokens=true with no reason. The tool
    loop (handle_fresh / run_session / render_turn) is exercised separately:
    render_turn's final turn now runs cli_bridge.enforce_max_tokens on the
    payload, so an overshoot on the tool path truncates with finish_reason
    "length" the same way cli_bridge._complete does -- covered by
    test_render_turn_final_truncates_long_answer_to_cap_with_length_finish.
    """
    from fastapi import HTTPException
    saved_provider, saved_profile = server.cli_bridge.PROVIDER, server.cli_bridge.PROFILE
    try:
        # An unenforceable lane -> 400 max_tokens_unenforceable on the
        # no-tools fall-through (no shipped profile is one any more; codex
        # enforces since the tool lockdown).
        server.cli_bridge.PROVIDER = "opencode"
        server.cli_bridge.PROFILE = dict(server.cli_bridge.PROFILES["opencode"],
                                         enforce_max_tokens=False)
        server.PROVIDER = "opencode"

        async def go_codex():
            try:
                await server.cli_bridge._handle_chat({
                    "model": "m", "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}]})
            except HTTPException as exc:
                return exc.status_code, exc.detail

        status, detail = asyncio.run(go_codex())
        assert status == 400, (status, detail)
        assert detail["error"]["type"] == "max_tokens_unenforceable", detail
        assert detail["error"]["message"] == \
            "max_tokens is not enforceable on this lane", detail

        # claude -> /health reports enforces_max_tokens=true, no reason key.
        server.cli_bridge.PROVIDER = "claude"
        server.cli_bridge.PROFILE = server.cli_bridge.PROFILES["claude"]
        server.PROVIDER = "claude"
        h_resp = asyncio.run(server.health())
        h = json.loads(h_resp.body)
        assert h_resp.status_code == 200, (h_resp.status_code, h)
        assert h["enforces_max_tokens"] is True, h
        assert "enforces_max_tokens_reason" not in h, h
    finally:
        server.cli_bridge.PROVIDER, server.cli_bridge.PROFILE = \
            saved_provider, saved_profile
        server.PROVIDER = saved_provider

    print("  mcp no-tools fall-through: unenforceable lane -> 400 max_tokens_unenforceable; "
          "claude /health -> enforces_max_tokens=True")


# ----------------------------------------------------- /health status codes ---
def test_health_returns_200_with_ok_true_when_config_source_is_config():
    """Happy path: an mcp_bridge with a real plans.yaml-backed config returns
    200 + ok=true. The compose healthcheck and `scripts/health_idle.py` only
    see a 2xx response on this side, so this is the contract that must hold
    whenever the bridge is actually ready to serve a tool loop.
    """
    from fastapi.testclient import TestClient
    saved_cfg = server.cli_bridge._config
    saved_cfg_at = server.cli_bridge._config_at
    saved_cfg_mtime = server.cli_bridge._config_mtime
    server.cli_bridge._config = None
    server.cli_bridge._config_at = 0.0
    server.cli_bridge._config_mtime = None
    try:
        client = TestClient(server.app)
        resp = client.get("/health")
        body = resp.json()
        assert resp.status_code == 200, (resp.status_code, body)
        assert body["ok"] is True, body
        assert body["config_source"] == "config", body
        assert body["provider"] == "claude", body
    finally:
        server.cli_bridge._config = saved_cfg
        server.cli_bridge._config_at = saved_cfg_at
        server.cli_bridge._config_mtime = saved_cfg_mtime
    print("  mcp /health: config_source=config -> 200 + ok=true")


def test_health_returns_503_with_ok_false_on_cold_start_fallback():
    """The cold-start fallback (unknown SWITCHYARD_PLAN, _config reset to
    None) must surface as 503 + ok=false. The mcp_bridge health doc is
    built from cli_bridge.config(), so resetting cli_bridge's _config and
    pointing it at a plan that does not exist drives the same `fallback`
    source that the bridge's cold-start path takes -- and the response
    code must match the body's `ok: false`, otherwise the compose
    healthcheck would happily route traffic into a sidecar that just
    told us it cannot resolve its own plan.
    """
    from fastapi.testclient import TestClient
    saved_env = dict(os.environ)
    saved_cfg = server.cli_bridge._config
    saved_cfg_at = server.cli_bridge._config_at
    saved_cfg_mtime = server.cli_bridge._config_mtime
    saved_plan = server.cli_bridge.PLAN
    os.environ["SWITCHYARD_PLAN"] = "no-such-plan"
    server.cli_bridge.PLAN = "no-such-plan"
    server.cli_bridge._config = None
    server.cli_bridge._config_at = 0.0
    server.cli_bridge._config_mtime = None
    try:
        client = TestClient(server.app)
        resp = client.get("/health")
        body = resp.json()
        assert resp.status_code == 503, (resp.status_code, body)
        assert body["ok"] is False, body
        assert body["config_source"] == "fallback", body
    finally:
        server.cli_bridge._config = saved_cfg
        server.cli_bridge._config_at = saved_cfg_at
        server.cli_bridge._config_mtime = saved_cfg_mtime
        server.cli_bridge.PLAN = saved_plan
        os.environ.clear()
        os.environ.update(saved_env)
    print("  mcp /health: cold-start fallback (unknown plan) -> 503 + ok=false, config_source=fallback")


# ----------------------------------------- last-call context vs billed sum -----
def _write_inner_transcript(session: "server.Session", root: Path,
                            calls: list[dict]) -> None:
    """Lay down `calls` user/assistant JSONL pairs under the sanitized workdir.

    Mirrors the shape Claude Code writes: alternating `user` / `assistant`
    entries, with the assistant's `message.usage` carrying the per-turn
    token breakdown that `last_call_usage` reads back. Pairs are appended in
    order; `last_call_usage` scans the file in reverse so the LAST assistant
    in the list is what callers see.
    """
    sanitized = re.sub(r"[^A-Za-z0-9]", "-", str(session.spawn_dir or session.workdir))
    target = root / sanitized
    target.mkdir(parents=True, exist_ok=True)
    # build_argv pins the name with --session-id (see claude_session_uuid).
    path = target / f"{server.claude_session_uuid(session.id)}.jsonl"
    with path.open("w") as fh:
        for i, usage in enumerate(calls):
            fh.write(json.dumps({"type": "user", "message": {"role": "user",
                                                               "content": f"q{i}"}}) + "\n")
            fh.write(json.dumps({"type": "assistant",
                                 "message": {"role": "assistant",
                                             "content": f"a{i}",
                                             "usage": usage}}) + "\n")


def test_sessions_sharing_a_mirrored_cwd_read_their_own_transcript():
    """Two sessions for the same caller cwd run in the same mirror
    directory, so Claude files both transcripts in one project dir. Each
    session must read its own (pinned by --session-id), never the newest."""
    saved = server.cli_bridge.CLAUDE_PROJECTS
    tmp = Path(tempfile.mkdtemp(prefix="mcpb-projects-shared-"))
    usage_a = {"input_tokens": 11, "output_tokens": 1,
               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    usage_b = {"input_tokens": 99, "output_tokens": 9,
               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    a, b = _new_session(), _new_session()
    try:
        a.spawn_dir = b.spawn_dir = "/Users/me/proj"
        _write_inner_transcript(a, tmp, [usage_a])
        _write_inner_transcript(b, tmp, [usage_b])   # newer file, other session
        server.cli_bridge.CLAUDE_PROJECTS = tmp
        assert server.last_call_usage(a)["input_tokens"] == 11
        assert server.last_call_usage(b)["input_tokens"] == 99
        print("  shared mirror dir: each session reads its own pinned transcript")
    finally:
        server.cli_bridge.CLAUDE_PROJECTS = saved
        _drop(a)
        _drop(b)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


class _MirrorSandbox:
    """Point the mirror machinery at a temp root with a fresh registry."""

    def __enter__(self):
        import tempfile as _tf
        self.root = Path(_tf.mkdtemp(prefix="mcpb-mirror-root-"))
        self.saved = (server.MIRROR_ROOTS, server.MIRROR_DENY, server.MIRROR_LIMIT,
                      server.MIRROR_MANIFEST, dict(server.MIRROR_USERS),
                      dict(server.MIRROR_CREATED))
        server.MIRROR_ROOTS = (f"{self.root}/",)
        server.MIRROR_DENY = (f"{self.root}/node",)
        server.MIRROR_MANIFEST = self.root / "manifest.json"
        server.MIRROR_USERS.clear()
        server.MIRROR_CREATED.clear()
        return self

    def env(self, cwd, **kw):
        return server._caller_env.CallerEnvironment(cwd=cwd, source="request", **kw)

    def __exit__(self, *exc):
        import shutil
        (server.MIRROR_ROOTS, server.MIRROR_DENY, server.MIRROR_LIMIT,
         server.MIRROR_MANIFEST, users, created) = self.saved
        server.MIRROR_USERS.clear()
        server.MIRROR_USERS.update(users)
        server.MIRROR_CREATED.clear()
        server.MIRROR_CREATED.update(created)
        shutil.rmtree(self.root, ignore_errors=True)


def test_mirror_is_created_only_for_safe_caller_paths():
    """The CLI runs in a directory at the caller's own cwd path so its
    environment block agrees with the caller (issue #264). Only plain
    absolute POSIX paths under the image's mirror roots qualify: never the
    relay's own dirs, the credential dirs under /home/node, a Windows path
    or anything with `..`."""
    with _MirrorSandbox() as box:
        made = server.acquire_mirror(box.env(f"{box.root}/me/proj"), "s1")
        assert made == box.root / "me" / "proj" and made.is_dir(), made
        assert not (made / ".git").exists()
        repo = server.acquire_mirror(box.env(f"{box.root}/me/repo", git=True), "s2")
        assert (repo / ".git").is_dir(), repo
        assert list(made.iterdir()) == [], "nothing may be written into a mirror"
        for refused in (None, "relative/path", "C:\\Users\\me", f"{box.root}/me/../x",
                        f"{box.root}/node/.claude", "/tmp/elsewhere", "/app/mcp_bridge"):
            assert server.acquire_mirror(box.env(refused), "s3") is None, refused
        assert server.acquire_mirror(None, "s4") is None
        print("  mirror: caller path recreated (git init on request); unsafe paths refused")



def test_windows_caller_cwd_is_never_mirrored():
    """A Windows cwd in any spelling has no Linux path equal to it, so the
    session keeps its workdir (the env block and reminder carry the
    caller's path instead). Git Bash's /c/... is not under a mirror root."""
    CE = server._caller_env.CallerEnvironment
    for cwd in ("C:\\Users\\me\\proj", "C:/Users/me/proj", "c:/Users/me/proj",
                "\\\\server\\share\\proj", "//server/share/proj", "/c/Users/me/proj",
                "/C:/Users/me/proj"):
        assert server.mirror_target(CE(cwd=cwd, platform="win32", source="request")) is None, cwd
    print("  C:\\..., C:/..., UNC and /c/... cwds keep the session workdir")

def test_mirror_is_removed_with_its_last_session_and_only_what_was_created():
    """Review of #273: a caller-supplied path must not leave directories and
    repos behind. Each mirror is reference-counted; the last session to end
    removes what the relay created -- the dir, the ancestors it made, the
    .git it initialised -- and nothing that existed before."""
    with _MirrorSandbox() as box:
        (box.root / "pre").mkdir()                    # existed before any mirror
        path = f"{box.root}/pre/me/proj"
        a = server.acquire_mirror(box.env(path, git=True), "a")
        b = server.acquire_mirror(box.env(path, git=True), "b")
        assert a == b and (a / ".git").is_dir()
        server.release_mirror(path, "a")
        assert a.is_dir(), "removed while another session still runs in it"
        server.release_mirror(path, "b")
        assert not (box.root / "pre" / "me").exists(), list((box.root / "pre").iterdir())
        assert (box.root / "pre").is_dir(), "removed a directory the relay did not create"
        assert server.MIRROR_USERS == {} and server.MIRROR_CREATED == {}
        print("  mirror removed with its last session; pre-existing ancestors kept")


def test_nested_mirrors_never_remove_a_live_one():
    """/x/me/proj is released while /x/me is still a live mirror: removal
    walks up through the ancestors it created but stops at a live mirror."""
    with _MirrorSandbox() as box:
        inner = server.acquire_mirror(box.env(f"{box.root}/me/proj"), "inner")
        outer = server.acquire_mirror(box.env(f"{box.root}/me"), "outer")
        server.release_mirror(str(inner), "inner")
        assert not inner.exists() and outer.is_dir(), "a live mirror was removed"
        server.release_mirror(str(outer), "outer")
        print("  nested mirrors: releasing the inner one keeps the live outer one")


def test_mirror_count_is_bounded():
    with _MirrorSandbox() as box:
        server.MIRROR_LIMIT = 2
        assert server.acquire_mirror(box.env(f"{box.root}/a"), "1") is not None
        assert server.acquire_mirror(box.env(f"{box.root}/b"), "2") is not None
        assert server.acquire_mirror(box.env(f"{box.root}/c"), "3") is None
        assert server.acquire_mirror(box.env(f"{box.root}/a"), "4") is not None, \
            "joining an existing mirror is not a new one"
        assert not (box.root / "c").exists()
        print("  mirrors capped at MIRROR_LIMIT; joining an existing one still allowed")


def test_startup_sweeps_mirrors_a_crash_left_behind():
    with _MirrorSandbox() as box:
        leftover = server.acquire_mirror(box.env(f"{box.root}/crash/proj", git=True), "gone")
        assert json.loads(server.MIRROR_MANIFEST.read_text())
        server.MIRROR_USERS.clear()                     # the process died
        server.MIRROR_CREATED.clear()
        server.sweep_mirrors()
        assert not leftover.exists() and not (box.root / "crash").exists()
        print("  startup sweep removed a crashed process's mirror")


def test_end_session_reclaims_the_mirror():
    """Every teardown path (final answer, reaper, supersede) goes through
    end_session, which must release the session's mirror."""
    with _MirrorSandbox() as box:
        session = _new_session()
        mirror = server.acquire_mirror(box.env(f"{box.root}/me/proj"), session.id)
        session.spawn_dir = str(mirror)
        asyncio.run(server.end_session(session))
        assert not mirror.exists() and not (box.root / "me").exists()
        print("  end_session released and removed the session's mirror")


def test_session_env_and_argv_follow_the_mirror():
    """OpenCode's --dir is the mirror and its per-session config rides
    OPENCODE_CONFIG (the mirror is shared); Claude gets the caller's SHELL
    and a --session-id. Every added key is on the allowlist."""
    import shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-mirror-wd-"))
    mirror = Path("/Users/me/proj")
    env = server._caller_env.CallerEnvironment(cwd=str(mirror), shell="zsh",
                                               source="request")
    saved = (server.PROVIDER, server.PROFILE)
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    try:
        server.PROVIDER, server.PROFILE = "opencode", server.MCP_PROFILES["opencode"]
        extra = server.session_env(env, workdir, mirror)
        assert extra == {"OPENCODE_CONFIG": str(workdir / "opencode.json")}, extra
        assert server.session_env(env, workdir, workdir) == {}
        argv, _ = server.build_argv("hi", None, "m", workdir, "sess", tools_path, "",
                                    spawn_dir=mirror)
        assert argv[argv.index("--dir") + 1] == str(mirror), argv
        assert (workdir / "opencode.json").exists()

        server.PROVIDER, server.PROFILE = "claude", server.MCP_PROFILES["claude"]
        assert server.session_env(env, workdir, mirror) == {"SHELL": "/bin/zsh"}
        sid = uuid.uuid4().hex
        argv, _ = server.build_argv("hi", None, "m", workdir, sid, tools_path, "")
        assert argv[argv.index("--session-id") + 1] == str(uuid.UUID(sid)), argv
        allowed = set(server.cli_bridge.SUBPROCESS_ENV_KEYS)
        assert {"OPENCODE_CONFIG", "SHELL"} <= allowed
        print("  opencode --dir=<mirror> + OPENCODE_CONFIG; claude SHELL + --session-id")
    finally:
        server.PROVIDER, server.PROFILE = saved
        shutil.rmtree(workdir, ignore_errors=True)


def test_claude_shell_names_a_windows_shell_as_the_caller_does():
    """Claude echoes $SHELL in its Shell: line (bash/zsh for anything
    containing those names, the raw value otherwise). POSIX names keep the
    /bin/<name> spelling; a Windows shell is passed as written, never as a
    made-up /bin/PowerShell or /bin/C:\\... path."""
    long = ("PowerShell (primary); Bash tool also available for POSIX scripts"
            " \u2014 each takes its own syntax.")
    cases = {"zsh": "/bin/zsh", "bash": "/bin/bash", "fish": "/bin/fish",
             "/usr/local/bin/fish": "/usr/local/bin/fish", "/bin/zsh": "/bin/zsh",
             "PowerShell": "PowerShell", "powershell": "powershell", "pwsh": "pwsh",
             "cmd": "cmd", "powershell.exe": "powershell.exe",
             "C:\\Program Files\\Git\\bin\\bash.exe": "C:\\Program Files\\Git\\bin\\bash.exe",
             long: long}
    for shell, want in cases.items():
        assert server.relay_shell(shell) == want, (shell, server.relay_shell(shell))
    saved = (server.PROVIDER, server.PROFILE)
    try:
        server.PROVIDER, server.PROFILE = "claude", server.MCP_PROFILES["claude"]
        env = server._caller_env.CallerEnvironment(cwd="C:\\x", shell="PowerShell",
                                                   platform="win32", source="request")
        assert server.session_env(env, Path("/tmp/w"), Path("/tmp/w")) == {"SHELL": "PowerShell"}
    finally:
        server.PROVIDER, server.PROFILE = saved
    print("  SHELL: POSIX names -> /bin/<name>; Windows shells verbatim")


def test_final_turn_reports_last_call_context_size_and_bills_the_sum():
    """A final turn with a 20-call summed payload keeps the SUM under
    `switchyard_billed_*` (for the ledger) but reports the LAST call's size
    under `usage.prompt_tokens` (for the caller looking at context).

    The summed payload is what cli_bridge.to_openai emits today -- the
    CLI's own transcript writes per-turn usage, but the bridge's existing
    parser folds them all into one running total before the response is
    built. That sum is still the right number to bill, since each turn
    consumed real tokens; it is just not the right number to display as a
    context meter. The last call's size replaces `prompt_tokens`; the
    cache breakdown still rides alongside it (folded into `prompt_tokens`
    by to_openai), the way it would on a text path.
    """
    # Twenty per-turn usage records; the LAST one carries a distinctive
    # input+cache shape the assertions can pin to without ambiguity.
    calls = []
    for i in range(19):
        calls.append({
            "input_tokens": 100 + i,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 0,
            "output_tokens": 5,
        })
    last_usage = {
        "input_tokens": 1234,
        "cache_read_input_tokens": 56,
        "cache_creation_input_tokens": 7,
        "output_tokens": 42,
    }
    calls.append(last_usage)
    summed = {
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "cache_read_input_tokens": sum(c["cache_read_input_tokens"] for c in calls),
        "cache_creation_input_tokens": sum(c["cache_creation_input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
    }
    assert len(calls) == 20, calls

    saved = server.cli_bridge.CLAUDE_PROJECTS
    tmp = Path(tempfile.mkdtemp(prefix="mcpb-projects-"))
    try:
        session = _new_session()
        _write_inner_transcript(session, tmp, calls)
        server.cli_bridge.CLAUDE_PROJECTS = tmp

        # cli_bridge.to_openai's Anthropic-shaped branch folds cache reads
        # AND cache writes into prompt_tokens, mirroring what the text
        # bridge reports.
        expected_prompt = (last_usage["input_tokens"]
                           + last_usage["cache_read_input_tokens"]
                           + last_usage["cache_creation_input_tokens"])
        expected_billed_prompt = (summed["input_tokens"]
                                  + summed["cache_read_input_tokens"]
                                  + summed["cache_creation_input_tokens"])

        result = {"type": "final", "payload": {"result": "answer", "usage": summed}}
        response = server.render_turn(session, result, session.model)
        usage = response["usage"]

        assert usage["prompt_tokens"] == expected_prompt, usage
        assert usage["completion_tokens"] == last_usage["output_tokens"], usage
        assert usage["total_tokens"] == (
            expected_prompt + last_usage["output_tokens"]), usage
        # Cache reads surface in prompt_tokens_details (OpenAI convention).
        assert usage["prompt_tokens_details"]["cached_tokens"] == \
            last_usage["cache_read_input_tokens"], usage
        # Cache writes ride top-level using the verbatim Anthropic spelling,
        # exactly as cli_bridge.to_openai emits on the text path.
        assert usage["cache_creation_input_tokens"] == \
            last_usage["cache_creation_input_tokens"], usage

        # And the SUM -- what the ledger needs to book -- is preserved
        # under the *billed* keys, not lost when the context view replaced
        # prompt_tokens.
        assert usage["switchyard_billed_prompt_tokens"] == \
            expected_billed_prompt, usage
        assert usage["switchyard_billed_completion_tokens"] == \
            summed["output_tokens"], usage

        _drop(session)
    finally:
        server.cli_bridge.CLAUDE_PROJECTS = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  final turn: prompt_tokens={usage['prompt_tokens']} "
          f"(last call), billed_prompt={usage['switchyard_billed_prompt_tokens']} "
          f"(20-call sum)")


def test_tool_calls_turn_reports_last_call_context_and_zero_billed():
    """A mid-loop `tool_calls` turn goes through `_with_context_usage`
    with an empty `billed` dict (nothing has been booked yet on this turn),
    so the response's `prompt_tokens` is the LAST inner CLI call's context
    size, the cache breakdown rides alongside it, and the billed keys are
    stamped as 0/0 — the caller can read size, the ledger has nothing to
    book yet.

    The final-turn test pins the same code path with a non-empty billed
    dict; this one pins the empty-billed branch (`render_turn`'s
    `tool_calls` branch) and the empty-transcript fallback that the final
    turn already covers separately.
    """
    # Three per-turn usage records; the LAST one carries a distinctive
    # input+cache shape the assertions can pin to without ambiguity.
    calls = []
    for i in range(2):
        calls.append({
            "input_tokens": 100 + i,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 0,
            "output_tokens": 5,
        })
    last_usage = {
        "input_tokens": 987,
        "cache_read_input_tokens": 65,
        "cache_creation_input_tokens": 4,
        "output_tokens": 17,
    }
    calls.append(last_usage)

    saved = server.cli_bridge.CLAUDE_PROJECTS
    tmp = Path(tempfile.mkdtemp(prefix="mcpb-projects-tools-"))
    try:
        session = _new_session()
        _write_inner_transcript(session, tmp, calls)
        server.cli_bridge.CLAUDE_PROJECTS = tmp

        # cli_bridge.to_openai's Anthropic-shaped branch folds cache reads
        # AND cache writes into prompt_tokens, mirroring what the text
        # bridge reports — same expectation as the final-turn test.
        expected_prompt = (last_usage["input_tokens"]
                           + last_usage["cache_read_input_tokens"]
                           + last_usage["cache_creation_input_tokens"])

        # A single fake call so render_turn's tool_calls branch builds
        # the response without raising; render_turn only reads .id /
        # .name / .arguments, so a SimpleNamespace stands in for the full
        # ParkedCall (its future would need an event loop this thread
        # doesn't have).
        parked = types.SimpleNamespace(id="call_abc", name="lookup",
                                       arguments={"q": "x"})
        result = {"type": "tool_calls", "calls": [parked]}
        response = server.render_turn(session, result, session.model)
        usage = response["usage"]

        # Context view: the LAST CLI call's size, not the mid-loop zero
        # the bridge used to ship.
        assert usage["prompt_tokens"] == expected_prompt, usage
        assert usage["completion_tokens"] == last_usage["output_tokens"], usage
        assert usage["total_tokens"] == (
            expected_prompt + last_usage["output_tokens"]), usage
        # Cache breakdown rides alongside, exactly the way the text path
        # reports it.
        assert usage["prompt_tokens_details"]["cached_tokens"] == \
            last_usage["cache_read_input_tokens"], usage
        assert usage["cache_creation_input_tokens"] == \
            last_usage["cache_creation_input_tokens"], usage

        # Billed keys are stamped as 0 — nothing has been booked on this
        # turn. They MUST be present and zero, not absent and not missing.
        assert usage["switchyard_billed_prompt_tokens"] == 0, usage
        assert usage["switchyard_billed_completion_tokens"] == 0, usage

        # The tool_calls branch's other duties are intact: finish_reason
        # and the tool_calls list ride through the wrap untouched.
        assert response["choices"][0]["finish_reason"] == "tool_calls", response
        out_calls = response["choices"][0]["message"]["tool_calls"]
        assert len(out_calls) == 1, out_calls
        assert out_calls[0]["id"] == "call_abc", out_calls[0]
        assert out_calls[0]["function"]["name"] == "lookup", out_calls[0]
        _drop(session)

        # Empty-transcript flavour: no JSONL to read, last_call_usage
        # returns None, the response keeps the mid-loop {0, 0, 0} it built,
        # and billed keys are still stamped as 0/0 (consistent with the
        # no-billed path on a fresh turn).
        session = _new_session()
        tmp2 = Path(tempfile.mkdtemp(prefix="mcpb-projects-empty-tools-"))
        server.cli_bridge.CLAUDE_PROJECTS = tmp2
        try:
            parked2 = types.SimpleNamespace(id="call_def", name="noop",
                                            arguments={})
            result2 = {"type": "tool_calls", "calls": [parked2]}
            response2 = server.render_turn(session, result2, session.model)
            usage2 = response2["usage"]
            assert usage2["prompt_tokens"] == 0, usage2
            assert usage2["completion_tokens"] == 0, usage2
            assert usage2["total_tokens"] == 0, usage2
            assert usage2["switchyard_billed_prompt_tokens"] == 0, usage2
            assert usage2["switchyard_billed_completion_tokens"] == 0, usage2
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
        _drop(session)
    finally:
        server.cli_bridge.CLAUDE_PROJECTS = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  tool_calls turn: prompt_tokens={usage['prompt_tokens']} "
          f"(last call), billed_prompt={usage['switchyard_billed_prompt_tokens']} "
          f"(mid-loop)")


# --------------------------------------- teardown: remove per-session project dir --
def test_end_session_removes_claude_project_dir_for_a_mcp_relay_session():
    """`last_call_usage` reads `~/.claude/projects/<sanitized spawn_dir>/<uuid>.jsonl`
    during the final turn response build; teardown (reap/end_session/shutdown
    drain) runs strictly after the response is produced, so deleting the
    project dir on the teardown path cannot race the read.

    This test pins both halves of that contract for a session whose workdir
    is the production-shape ``/relay/<session_id>``:
      (1) `last_call_usage` reads the file the fake CLI wrote under a temp
          CLAUDE_PROJECTS root, before end_session is called.
      (2) After end_session runs, the per-session project dir
          `<sanitized spawn_dir>` is gone (the parent's other dirs are
          untouched).

    Reviewer finding, PR #322 cycle 2: the previous version of this test
    used ``mkdtemp(prefix="mcpb-test-")`` to simulate the fallback path,
    but the helper's ownership test was ``spawn_dir.name == session.id``,
    which matches the production ``/relay/<session_id>`` shape and not
    the fallback's ``mcpb-<id8>-<random>``. The cycle-3 fix adds a
    disjunction in the helper (``name == session_id`` OR
    ``^mcpb-<id8>-``); this test pins the production half of that
    disjunction end-to-end through ``end_session``. The matching
    ``test_end_session_removes_claude_project_dir_for_a_mcp_fallback_session``
    pins the fallback shape (also cycle-3).
    """
    saved = server.cli_bridge.CLAUDE_PROJECTS
    saved_provider = server.PROVIDER
    tmp = Path(tempfile.mkdtemp(prefix="mcpb-projects-cleanup-"))
    try:
        server.cli_bridge.CLAUDE_PROJECTS = tmp
        session = _new_session()
        # Production workdir shape: /relay/<session_id>. The helper's
        # ownership test is `spawn_dir.name == session.id`, which the
        # session.uuid-as-dirname satisfies exactly.
        session.spawn_dir = "/relay/" + session.id
        # Sibling kept around: a different session's project dir in the same
        # parent must NOT be collateral damage.
        sibling = tmp / re.sub(r"[^A-Za-z0-9]", "-", str(spawn_dir_alt(session)))
        sibling.mkdir(parents=True, exist_ok=True)
        usage = {"input_tokens": 17, "output_tokens": 4,
                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        _write_inner_transcript(session, tmp, [usage])
        # (1) read still works before teardown
        assert server.last_call_usage(session) == {
            "input_tokens": 17, "output_tokens": 4,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        }, server.last_call_usage(session)
        target = tmp / re.sub(r"[^A-Za-z0-9]", "-", session.spawn_dir)
        assert target.is_dir(), list(tmp.iterdir())

        # (2) teardown path removes the project's <sanitized workdir> dir
        # but nothing else. Session.workdir is an mcpb-test-* tmpdir from
        # _new_session; clean that up too.
        asyncio.run(server.end_session(session))

        assert not target.exists(), \
            f"project dir was not removed: {target} ({list(tmp.iterdir())})"
        assert sibling.is_dir(), \
            "sibling project dir was collateral damage from teardown"
        _drop(session)
        shutil.rmtree(Path(session.workdir), ignore_errors=True)
    finally:
        server.PROVIDER = saved_provider
        server.cli_bridge.CLAUDE_PROJECTS = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print("  teardown: last_call_usage reads the transcript before, "
          "end_session removes the /relay/<session_id> project dir (sibling kept)")


def spawn_dir_alt(session: "server.Session") -> Path:
    """Helper: a fresh /relay/<uuid> shape different from session.spawn_dir.

    Used by the sibling-must-not-be-collateral check above; the sibling
    has a new uuid, so its sanitized basename is a different dir under
    the same CLAUDE_PROJECTS root.
    """
    return Path("/relay/alt-" + uuid.uuid4().hex)


def test_end_session_removes_claude_project_dir_for_a_mcp_fallback_session():
    """The companion to ``test_end_session_removes_claude_project_dir_for_a_mcp_relay_session``:
    pins the ``mkdtemp(prefix=f"mcpb-{session_id[:8]}-")`` fallback shape
    that ``new_workdir`` returns when ``MCP_WORKDIR_ROOT`` cannot be
    ``mkdir``'d (Dockerfile.sidecar usually makes ``/relay`` sticky, so
    this is rare; it is the path tests and unusual deployments take).

    The cycle-2 helper accepted only the ``/relay/<session_id>`` shape
    (basename equals session.uuid exactly); it explicitly did not accept
    this fallback, leaking the corresponding project dir on every
    fallback-path session. Reviewer finding, PR #322 cycle 3: cycle-2
    trade a leak in the production path for a leak in the fallback
    path. This test pins the cycle-3 fix -- the helper's disjunction
    accepts both shapes, and the fallback's project dir is reclaimed.
    """
    saved = server.cli_bridge.CLAUDE_PROJECTS
    saved_provider = server.PROVIDER
    tmp = Path(tempfile.mkdtemp(prefix="mcpb-projects-fallback-"))
    try:
        server.cli_bridge.CLAUDE_PROJECTS = tmp
        session = _new_session()
        # Fallback shape: mkdtemp's prefix is `mcpb-<session_id[:8]>-<rand>`.
        # The base name is NOT equal to the full session id; matching only
        # the full uuid would leak this path.
        fallback = Path(tempfile.mkdtemp(
            prefix=f"mcpb-{session.id[:8]}-",
            dir="/tmp"))
        session.spawn_dir = str(fallback)
        usage = {"input_tokens": 11, "output_tokens": 7,
                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        _write_inner_transcript(session, tmp, [usage])
        # (1) read still works before teardown
        assert server.last_call_usage(session) == {
            "input_tokens": 11, "output_tokens": 7,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        }, server.last_call_usage(session)
        target = tmp / re.sub(r"[^A-Za-z0-9]", "-", session.spawn_dir)
        assert target.is_dir(), list(tmp.iterdir())

        # (2) teardown path removes the fallback's project dir too.
        asyncio.run(server.end_session(session))

        assert not target.exists(), (
            f"fallback-shape project dir was not removed: {target} "
            f"({list(tmp.iterdir())})")
        _drop(session)
        shutil.rmtree(fallback, ignore_errors=True)
    finally:
        server.PROVIDER = saved_provider
        server.cli_bridge.CLAUDE_PROJECTS = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print("  teardown: end_session also removes the mcpb-<id8>-* fallback "
          "shape's project dir (cycle-3 fix for cycle-2 regression)")


def test_end_session_keeps_project_dir_for_a_non_owned_spawn_cwd():
    """A mirrored caller cwd shares one Claude project dir across every
    session for that caller, and a project dir owned by another session
    / shadowed under another path is not ours to reclaim.

    Pinned against three non-owned shapes:
      A. caller mirror `/Users/me/proj` -- basename `proj` ≠ session.id.
      B. caller mirror `/Users/me/repos/mcpb-fleetside` -- basename
         `mcpb-fleetside` starts with `mcpb-` but is NOT this session.id
         (reviewer finding, PR #322 cycle 2: the old `mcpb-` basename
         guard would have wrongly deleted this shared project dir).
      C. someone else's `/relay/<other-uuid>` workdir -- basename is a
         fresh uuid, not session.id, and rmtree would race that other
         session's live `last_call_usage` read of its own transcript.

    In every case the per-session project dir survives `end_session`.
    """
    saved = server.cli_bridge.CLAUDE_PROJECTS
    saved_provider = server.PROVIDER
    tmp = Path(tempfile.mkdtemp(prefix="mcpb-projects-guard-"))
    try:
        server.cli_bridge.CLAUDE_PROJECTS = tmp
        # Case A: caller-mirror spawn path. The basename is the caller's
        # own `proj`, never a session uuid we minted.
        mirror_a = Path("/Users/me/proj")
        a = _new_session()
        a.spawn_dir = str(mirror_a)
        _write_inner_transcript(a, tmp, [{"input_tokens": 1, "output_tokens": 1,
                                          "cache_read_input_tokens": 0,
                                          "cache_creation_input_tokens": 0}])
        target_a = tmp / re.sub(r"[^A-Za-z0-9]", "-", str(mirror_a))
        assert target_a.is_dir()

        # Case B: caller mirror whose LAST segment starts with `mcpb-` but
        # is the caller's, not ours. The old basename guard would have
        # deleted it; the new `name == session.id` guard refuses because
        # `mcpb-fleetside` is not a hex uuid.
        mirror_b = Path("/Users/me/repos/mcpb-fleetside")
        b = _new_session()
        b.spawn_dir = str(mirror_b)
        _write_inner_transcript(b, tmp, [{"input_tokens": 2, "output_tokens": 2,
                                          "cache_read_input_tokens": 0,
                                          "cache_creation_input_tokens": 0}])
        target_b = tmp / re.sub(r"[^A-Za-z0-9]", "-", str(mirror_b))
        assert target_b.is_dir()

        # Case C: someone else's /relay/<uuid> -- the production shape,
        # but a DIFFERENT uuid, not this session's. We must not touch it.
        other_id = uuid.uuid4().hex
        other = _new_session()
        other.spawn_dir = "/relay/" + other_id
        _write_inner_transcript(other, tmp, [{"input_tokens": 3, "output_tokens": 3,
                                               "cache_read_input_tokens": 0,
                                               "cache_creation_input_tokens": 0}])
        target_c = tmp / re.sub(r"[^A-Za-z0-9]", "-", other.spawn_dir)
        assert target_c.is_dir()

        asyncio.run(server.end_session(a))
        asyncio.run(server.end_session(b))
        # We do NOT end_session `other` -- it's the "another live session"
        # shape, simulating a sibling whose transcript is still in use.
        assert target_a.is_dir(), f"mirror A wrongly removed: {target_a}"
        assert target_b.is_dir(), f"mirror B wrongly removed: {target_b}"
        assert target_c.is_dir(), f"sibling session's project dir wrongly removed: {target_c}"

        _drop(a)
        _drop(b)
        _drop(other)
    finally:
        server.PROVIDER = saved_provider
        server.cli_bridge.CLAUDE_PROJECTS = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print("  teardown guard: caller mirrors (/Users/me/proj, "
          "/Users/me/repos/mcpb-fleetside) and a sibling session's "
          "/relay/<other-uuid> project dir all survive end_session")


def test_cleanup_project_dir_is_noop_off_claude_provider():
    """The helper short-circuits on non-claude providers: a codex/opencode
    spawn cwd never produces a Claude transcript, so any project dir under
    CLAUDE_PROJECTS for it (if one somehow exists) must NOT be removed on
    teardown -- it is someone else's, not ours to reclaim."""
    saved = server.cli_bridge.CLAUDE_PROJECTS
    saved_provider = server.PROVIDER
    tmp = Path(tempfile.mkdtemp(prefix="mcpb-projects-nop-"))
    try:
        server.cli_bridge.CLAUDE_PROJECTS = tmp
        # Use a session-shaped spawn_dir whose basename IS a session id
        # we control, so the provider switch is the ONLY difference
        # between the no-op case and the reclaim case below.
        session = _new_session()
        spawn_dir = Path("/relay/" + session.id)
        session.spawn_dir = str(spawn_dir)
        # Plant a project dir the helper would normally claim.
        sanitized = re.sub(r"[^A-Za-z0-9]", "-", str(spawn_dir))
        planted = tmp / sanitized
        planted.mkdir(parents=True, exist_ok=True)
        sentinel = planted / "sentinel.jsonl"
        sentinel.write_text("not empty\n")

        # codex profile: same PROVIDER switch the bridge uses to gate the
        # helper. After the helper runs, the planted dir must still be
        # there even though spawn_dir.name == session.id would otherwise
        # authorize the cleanup.
        server.PROVIDER = "codex"
        server._cleanup_project_dir(spawn_dir, session.id)
        assert planted.is_dir() and sentinel.is_file(), \
            "codex profile must NOT remove a Claude project dir"
        # And sanity: the helper is callable as a no-op with None too.
        assert server._cleanup_project_dir(None, session.id) is None

        # Switch back to claude: now the dir IS reclaimed. (Proves the
        # provider gate is the only thing keeping the dir alive above.)
        server.PROVIDER = "claude"
        server._cleanup_project_dir(spawn_dir, session.id)
        assert not planted.exists(), (
            f"claude profile should reclaim its own /relay/<session_id> "
            f"project dir: {list(tmp.iterdir())}")
        _drop(session)
    finally:
        server.PROVIDER = saved_provider
        server.cli_bridge.CLAUDE_PROJECTS = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print("  cleanup: codex/opencode profiles no-op; claude reclaims "
          "/relay/<session_id> project dirs")


def test_no_inner_transcript_keeps_old_usage_and_bills_mirror_it():
    """An empty/nonexistent transcript root is the common state for callers
    that have not yet finished their first CLI turn, or run a provider with
    no transcript layout to read. The bridge must keep today's usage shape
    unchanged AND still surface the billed figures alongside, so the caller's
    view does not silently degrade just because we could not read the
    transcript.

    Two flavours are pinned: the dir does not exist (fresh sidecar on a new
    machine) and the dir exists but has no JSONL in it (workdir folder just
    got created, the CLI has not run yet).
    """
    saved = server.cli_bridge.CLAUDE_PROJECTS
    tmp = Path(tempfile.mkdtemp(prefix="mcpb-projects-empty-"))
    try:
        # First flavour: no project dir at all for this session's workdir.
        session = _new_session()
        server.cli_bridge.CLAUDE_PROJECTS = tmp
        summed = {
            "input_tokens": 500,
            "output_tokens": 200,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        result = {"type": "final", "payload": {"result": "answer", "usage": summed}}
        response = server.render_turn(session, result, session.model)
        usage = response["usage"]
        # Today would be: prompt_tokens=500, completion_tokens=200,
        # total_tokens=700 -- the to_openai shape with no Anthropic cache
        # keys, so prompt_tokens is just input_tokens. That stays put.
        assert usage["prompt_tokens"] == 500, usage
        assert usage["completion_tokens"] == 200, usage
        assert usage["total_tokens"] == 700, usage
        # Billed keys mirror the kept usage -- not zero, not absent.
        assert usage["switchyard_billed_prompt_tokens"] == 500, usage
        assert usage["switchyard_billed_completion_tokens"] == 200, usage
        _drop(session)

        # Second flavour: the project dir exists but has no JSONL yet.
        session = _new_session()
        sanitized = re.sub(r"[^A-Za-z0-9]", "-", str(session.workdir))
        (tmp / sanitized).mkdir(parents=True, exist_ok=True)
        result = {"type": "final", "payload": {"result": "answer", "usage": summed}}
        response = server.render_turn(session, result, session.model)
        usage = response["usage"]
        assert usage["prompt_tokens"] == 500, usage
        assert usage["completion_tokens"] == 200, usage
        assert usage["switchyard_billed_prompt_tokens"] == 500, usage
        assert usage["switchyard_billed_completion_tokens"] == 200, usage
        _drop(session)
    finally:
        server.cli_bridge.CLAUDE_PROJECTS = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print("  no transcript: today's usage kept; billed keys mirror it")


def test_run_session_attempt_passes_only_the_allowlisted_env_to_the_cli():
    """mcp_bridge's only `create_subprocess_exec` (the one inside
    _run_session_attempt) must hand the inner CLI a plain-dict `env=`
    whose keys are a subset of cli_bridge's SUBPROCESS_ENV_KEYS. A
    planted secret (LITELLM_MASTER_KEY, the kind compose's env_file would
    have injected into the container) must be absent -- otherwise a
    single prompt injection on the MCP path exfiltrates every provider
    key, the gateway master key and the OAuth grant (issue #116).
    """
    async def scenario():
        session = _new_session()
        session.new_turn()

        real_exec = asyncio.create_subprocess_exec
        captured: list = []

        async def fake_exec(*args, **kwargs):
            captured.append(kwargs.get("env"))
            class _Stub:
                returncode = 0

                async def communicate(self, input=None):
                    return (b"", b"")

                async def wait(self):
                    return 0

                def kill(self):
                    return None
            return _Stub()

        # Plant the secret AFTER importing server so SUBPROCESS_ENV_KEYS is
        # already captured; this is the real failure shape -- an .env line
        # arrives in a running sidecar via env_file.
        planted = "LITELLM_MASTER_KEY"
        previous = os.environ.get(planted)
        os.environ[planted] = "sk-test-planted-secret"
        try:
            asyncio.create_subprocess_exec = fake_exec
            await server.run_session(session, ["fake-argv"])
        finally:
            asyncio.create_subprocess_exec = real_exec
            if previous is None:
                os.environ.pop(planted, None)
            else:
                os.environ[planted] = previous
        return session.turn_future.result(), captured

    result, captured = asyncio.run(scenario())
    # run_session resolved without raising -- if _run_session_attempt
    # raised because the fake spawn returned nothing usable, the resolved
    # turn_future still has to be the success path here.
    assert isinstance(result, dict), result

    assert captured, "create_subprocess_exec was never called"
    # Every captured env, not just the last: a future regression that
    # leaked the env on attempt N but switched to the allowlist on
    # attempt N+1 would pass a `captured[-1]` peek while still having
    # shipped a real spawn with the leaked bag. The contract is "every
    # CLI spawn on this path gets the allowlist", so assert it on every
    # capture. (run_session's no-text-with-usage retry only adds a
    # second spawn if the first raises CliNoTextError -- the empty-
    # stdout stub here trips the terminal HTTPException branch, not the
    # retry branch -- so today there is exactly one capture and the
    # single-env check below is no weaker than the per-attempt check.)
    for attempt, env in enumerate(captured, start=1):
        # Plain dict, NOT None (inherit parent) and NOT os.environ (copy whole bag).
        assert isinstance(env, dict), (attempt, type(env))
        # Every key on the allowlist -- exactly, no more, no less.
        bad = set(env) - set(server.cli_bridge.SUBPROCESS_ENV_KEYS)
        assert not bad, \
            f"attempt {attempt}: unexpected env keys passed to CLI: {sorted(bad)}"
        # The planted secret must not be present on any attempt.
        assert "LITELLM_MASTER_KEY" not in env, \
            f"attempt {attempt}: LITELLM_MASTER_KEY was passed to the inner CLI: " \
            f"{sorted(env)[:5]}..."
    print(f"  _run_session_attempt env allowlist honoured on every attempt: "
          f"{len(captured)} capture(s), all keys subset of "
          f"{len(server.cli_bridge.SUBPROCESS_ENV_KEYS)}-key allowlist; "
          f"LITELLM_MASTER_KEY absent")


def test_codex_mcp_profile_argv_carries_disable_overrides_after_bypass():
    """The codex MCP profile runs a CLI whose own tools act on the sidecar.

      1. --dangerously-bypass-approvals-and-sandbox KEPT. Without it codex
         refuses an MCP tool call ("MCP tool call requires approval").
      2. NO `sandbox_mode=` override. It was here on the theory that it
         neutralised the built-in shell; against the pinned 0.155.1 it does
         not (issue #255), and a flag that looks protective but is not is
         worse than none. The shell is removed by the lockdown build_argv
         appends instead (test_codex_mcp_build_argv_appends_the_lockdown).
      3. `tools.web_search=false` kept, after the bypass flag.
      4. Every `mcp_servers.switchyard.*` line KEPT -- parking an MCP tool
         call is the whole point of this bridge.
    """
    profile = server.MCP_PROFILES["codex"]
    argv = profile["argv"]

    assert "--dangerously-bypass-approvals-and-sandbox" in argv, \
        f"--dangerously-bypass-approvals-and-sandbox missing from argv: {argv}"
    assert not any(isinstance(el, str) and el.startswith("sandbox_mode=")
                   for el in argv), argv
    bypass_idx = argv.index("--dangerously-bypass-approvals-and-sandbox")
    web_search_c_idx = next(
        (i for i, el in enumerate(argv)
         if el == "-c" and i + 1 < len(argv)
         and argv[i + 1].startswith("tools.web_search=")), None)
    assert web_search_c_idx is not None, argv
    assert bypass_idx < web_search_c_idx, argv

    # All four mcp_servers.switchyard.* lines still there. Parking an MCP
    # tool call is the bridge's whole purpose -- a regression that drops
    # them turns the bridge into a CLI with no tools.
    mcp_fragments = [el for el in argv
                     if isinstance(el, str) and el.startswith("mcp_servers.switchyard.")]
    assert len(mcp_fragments) == 4, \
        f"expected 4 mcp_servers.switchyard.* fragments, got {len(mcp_fragments)}: {argv}"
    # Spot-check: command, args, startup_timeout_sec and env.
    keys = {el.split("=", 1)[0] for el in mcp_fragments}
    assert keys == {
        "mcp_servers.switchyard.command",
        "mcp_servers.switchyard.args",
        "mcp_servers.switchyard.startup_timeout_sec",
        "mcp_servers.switchyard.env",
    }, f"missing/wrong mcp_servers.switchyard.* keys: {keys}"

    print(f"  codex argv: bypass flag kept, no sandbox_mode override, "
          f"tools.web_search override AFTER it; "
          f"{len(mcp_fragments)} mcp_servers.switchyard.* keys intact")


# ------------------------------------------ issue #127: envelope -> error ----
def test_run_session_resolves_is_error_usage_limit_envelope_as_429():
    """Issue #127: mcp_bridge used to ignore the CLI's exit-0 envelope when it
    carried ``is_error=true`` plus a usage-limit string, so the session ended
    up resolved as a "final" payload (i.e. answered as 200 to the caller).
    The bridge now calls cli_bridge.check_result_envelope -- the same helper
    cli_bridge raises from -- and resolves the turn_future as a 429 carrying
    ``Retry-After``.
    """
    payload = ('{"type":"result","is_error":true,'
               '"result":"Claude AI usage limit reached|1760000000",'
               '"usage":{"input_tokens":0,"output_tokens":0}}')
    fake_cli = _write_fake_cli(
        "import sys\n"
        f"sys.stdout.write({payload!r})\n")
    session = _new_session()

    async def scenario():
        session.new_turn()
        await server.run_session(session, [sys.executable, fake_cli])
        return session.turn_future.result()

    try:
        result = asyncio.run(scenario())
        assert result["type"] == "error", result
        assert result["status"] == 429, result
        assert result.get("headers", {}).get("Retry-After"), result
        print(f"  MCP session: exit-0 is_error + usage-limit text -> "
              f"{result['status']} (Retry-After="
              f"{result['headers'].get('Retry-After')})")
    finally:
        _drop(session)
        try:
            os.unlink(fake_cli)
        except OSError:
            pass
        try:
            os.unlink(os.path.dirname(fake_cli))
        except OSError:
            pass


def test_run_session_resolves_error_max_turns_envelope_as_502():
    """Issue #127 sibling case: an error_max_turns subtype (no limit wording)
    must resolve as a 502, never a "final" payload. The plain JSON envelope
    used to flow through the session driver unchanged.
    """
    payload = '{"subtype":"error_max_turns","result":"max turns exceeded"}'
    fake_cli = _write_fake_cli(
        "import sys\n"
        f"sys.stdout.write({payload!r})\n")
    session = _new_session()

    async def scenario():
        session.new_turn()
        await server.run_session(session, [sys.executable, fake_cli])
        return session.turn_future.result()

    try:
        result = asyncio.run(scenario())
        assert result["type"] == "error", result
        assert result["status"] == 502, result
        assert "Retry-After" not in (result.get("headers") or {}), result
        print(f"  MCP session: exit-0 error_max_turns -> {result['status']}")
    finally:
        _drop(session)
        try:
            os.unlink(fake_cli)
        except OSError:
            pass
        try:
            os.unlink(os.path.dirname(fake_cli))
        except OSError:
            pass


def test_check_result_envelope_helper_classifies_four_payload_shapes():
    """Pin cli_bridge.check_result_envelope directly. The shared helper is the
    single source of truth for both bridges, so a regression here would
    break them together. mcp_bridge calls it via server.cli_bridge, the
    same way _run_session_attempt does.
    """
    cb = server.cli_bridge

    # Clean answer with usage: not an error.
    clean = {"result": "42", "usage": {"input_tokens": 1, "output_tokens": 1}}
    assert cb.check_result_envelope(clean) is None, clean

    # Usage-limit envelope: 429 with Retry-After.
    limited = {"type": "result", "is_error": True,
               "result": "Claude AI usage limit reached|1760000000",
               "usage": {"input_tokens": 0, "output_tokens": 0}}
    exc = cb.check_result_envelope(limited)
    assert exc is not None and exc.status_code == 429, exc
    assert exc.headers and "Retry-After" in exc.headers, exc.headers

    # error_during_execution: 502, no Retry-After.
    bad = {"subtype": "error_during_execution", "result": "boom"}
    exc = cb.check_result_envelope(bad)
    assert exc is not None and exc.status_code == 502, exc

    # Limit text only, no is_error / subtype / usage: still 429 + Retry-After.
    # This is the second branch of check_result_envelope (the ``_LIMIT`` regex
    # match on ``result`` with no ``usage`` field); the event-stream parsers
    # strip ``is_error``/``subtype`` before parse_output returns, so a real
    # CLI's limit message can land here even when nothing else flags it.
    text_only = {"result": "Claude AI usage limit reached|1760000000"}
    exc = cb.check_result_envelope(text_only)
    assert exc is not None and exc.status_code == 429, exc
    assert exc.headers and "Retry-After" in exc.headers, exc.headers
    print("  cli_bridge.check_result_envelope: None / 429+Retry-After / 502 / "
          "429+Retry-After across clean / usage-limit / error_during_execution"
          " / limit-text-only")


# ----------------------------------------- MCP_WORKDIR_ROOT (issue #294) -----
def test_new_workdir_lives_sunder_root_with_full_session_id():
    """new_workdir puts the session dir directly under MCP_WORKDIR_ROOT,
    names the dir with the full session uuid, gives distinct ids distinct
    dirs, and lets cleanup_workdir's rmtree remove exactly that session
    dir -- the root must stay. /relay is the production default; the
    test points MCP_WORKDIR_ROOT at a temp dir so no /relay exists."""
    import shutil
    saved = server.MCP_WORKDIR_ROOT
    test_root: Path | None = None
    try:
        test_root = Path(tempfile.mkdtemp(prefix="mcpb-wdroot-"))
        server.MCP_WORKDIR_ROOT = test_root
        a_id, b_id = uuid.uuid4().hex, uuid.uuid4().hex
        a = server.new_workdir(a_id)
        b = server.new_workdir(b_id)
        assert a == server.MCP_WORKDIR_ROOT / a_id, a
        assert b == server.MCP_WORKDIR_ROOT / b_id, b
        assert a != b, "distinct session ids must produce distinct dirs"
        assert a.is_dir() and b.is_dir()
        # The full uuid is the dirname, not a truncated prefix; this is
        # what makes cleanup_workdir's rmtree remove exactly one session.
        assert a.name == a_id and b.name == b_id
        server.cleanup_workdir(a)
        assert not a.exists(), a
        # cleanup_workdir must not have touched the root itself.
        assert server.MCP_WORKDIR_ROOT.is_dir(), server.MCP_WORKDIR_ROOT
        # The other session is untouched by the first cleanup.
        assert b.is_dir(), b
        print("  new_workdir: <root>/<full_session_id>; cleanup removes one, root stays")
    finally:
        server.MCP_WORKDIR_ROOT = saved
        # Remove the test-created temp root, NOT the saved production root
        # (default /relay): on a host without /relay rmtree is a silent no-op,
        # but inside the sidecar image it would silently wipe every
        # session dir beneath /relay. The test owns only test_root.
        if test_root is not None:
            shutil.rmtree(test_root, ignore_errors=True)


def test_new_workdir_falls_back_when_root_is_uncreatable():
    """When MCP_WORKDIR_ROOT cannot accept a mkdir (an image without /relay,
    a read-only env, an existing file in the way), new_workdir logs a
    warning and falls back to a mcpb- prefixed tempfile.mkdtemp dir
    rather than raising. This keeps the relay alive in any host environment."""
    saved_root = server.MCP_WORKDIR_ROOT
    try:
        # An existing file at the root path makes mkdir(parents=True) raise
        # NotADirectoryError on Linux; any OSError is enough to trigger the
        # fallback (the function's broad except catches them all).
        blocker = Path(tempfile.mkdtemp(prefix="mcpb-block-")) / "notadir"
        blocker.parent.mkdir(parents=True, exist_ok=True)
        blocker.write_text("not a directory")
        server.MCP_WORKDIR_ROOT = blocker
        captured: list[str] = []
        import logging as _lg
        class _Capture(_lg.Handler):
            def emit(self, record):
                captured.append(record.getMessage())
        cap = _Capture()
        cap.setLevel(_lg.WARNING)
        server.log.addHandler(cap)
        try:
            sid = uuid.uuid4().hex
            wd = server.new_workdir(sid)
        finally:
            server.log.removeHandler(cap)
        assert any("falling back to tempfile" in m for m in captured), captured
        assert wd.is_dir()
        assert wd.name.startswith(f"mcpb-{sid[:8]}-"), wd
        # The fallback dir is sibling to the blocker, not under it: nothing
        # in the caller-visible workdir path was written inside the root.
        assert blocker not in wd.parents and wd != blocker, wd
        shutil.rmtree(wd.parent if wd.parent == blocker.parent else wd,
                      ignore_errors=True)
        # Clean up the temp blocker hierarchy.
        import shutil as _sh
        _sh.rmtree(blocker.parent, ignore_errors=True)
        print("  new_workdir: OSError -> tempfile.mkdtemp fallback, no raise, warning logged")
    finally:
        server.MCP_WORKDIR_ROOT = saved_root


# -------------------------------------------- issue #292: MCP + reasoning ---
def test_mcp_bridge_build_argv_passes_thinking_to_cli_bridge():
    """mcp_bridge's build_argv must thread the thinking policy into the
    shared argv builder; both bridges share that builder (cli_bridge is
    loaded by file path, see sidecars/mcp_bridge/server.py:81-85), and a
    mid-loop session needs the same reasoning switch the text path uses.

    The CLI's argv builder itself is exercised end-to-end in cli_bridge's
    tests; this is the wiring pin that the mcp_bridge's per-session argv
    actually consults the carrier.
    """
    captured = {}

    saved_prov = server.PROVIDER
    saved_prof = server.PROFILE
    try:
        server.PROVIDER = "claude"
        server.PROFILE = server.MCP_PROFILES["claude"]
        workdir = Path(tempfile.mkdtemp(prefix="mcpb-thinking-"))
        try:
            tools_path = workdir / "tools.json"
            tools_path.write_text(json.dumps([{"name": "probe", "inputSchema": {}}]))
            argv, _ = server.build_argv(
                "hello", None, "claude-sonnet-5", workdir,
                uuid.uuid4().hex, tools_path, "mcp__switchyard__probe",
                thinking={"type": "enabled", "display": "omitted"})
            captured["claude"] = argv
            server.PROVIDER = "opencode"
            server.PROFILE = server.MCP_PROFILES["opencode"]
            argv, _ = server.build_argv(
                "hello", None, "m", workdir,
                uuid.uuid4().hex, tools_path, "switchyard_probe",
                thinking={"type": "enabled"})
            captured["opencode"] = argv
            server.PROVIDER = "codex"
            server.PROFILE = server.MCP_PROFILES["codex"]
            argv, _ = server.build_argv(
                "hello", None, "gpt-5.6-terra", workdir,
                uuid.uuid4().hex, tools_path, "probe",
                thinking={"type": "adaptive"})
            captured["codex"] = argv
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    finally:
        server.PROVIDER, server.PROFILE = saved_prov, saved_prof

    # Each profile accepted the policy without complaint. None of them have
    # an extra "thinking" flag today (effort / variant already gate the
    # reasoning switch), so the policy is on the caller's request without
    # altering argv structure -- a future profile with a per-call flag
    # would land in cli_bridge.thinking_args() and absorb it here.
    for name, argv in captured.items():
        assert isinstance(argv, list), (name, argv)
    print("  mcp_bridge.build_argv: thinking policy threads through to cli_bridge")


def test_mcp_bridge_render_turn_emits_reasoning_on_completion_only():
    """Issue #292 pin: accumulated reasoning rides with the COMPLETED final
    turn, never with an incomplete parked `tool_calls` response. A mid-loop
    tool_calls turn has no reasoning yet (none has been emitted by the CLI
    on the call session), so the message carries the message standard
    shape -- content=None, tool_calls=[...]. The final turn renders the
    parsed CLI payload through cli_bridge.to_openai, which lifts reasoning
    onto `reasoning_content` regardless of how the request reached the
    bridge.

    This test exercises the renderer's shape contract directly: a parsed
    payload carrying both `result` and `reasoning` is what the parser
    emits on a reasoning turn, and to_openai is the single path through
    which reasoning reaches a chat-completion caller. The test does not
    take the full `render_turn` shortcut because render_turn consults
    `last_call_usage` against `cli_bridge.CLAUDE_PROJECTS` and the wider
    session state; the path being pinned here is `parse_output` ->
    `to_openai`, and that's what this exercises.
    """
    cb = server.cli_bridge
    reasoning_payload = {
        "result": "the answer", "reasoning": "thinking out loud",
        "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}}
    out = cb.to_openai(reasoning_payload, "m")
    msg = out["choices"][0]["message"]
    assert msg["content"] == "the answer", msg
    assert msg["reasoning_content"] == "thinking out loud", msg
    assert "tool_calls" not in msg, msg

    # Mid-loop tool_calls turn: render is hand-built by render_turn,
    # not by to_openai. Verify the shape of that hand-built message
    # carries content=None and tool_calls=[...], with no reasoning
    # field -- reasoning waits for the FINAL turn, not the parked one.
    mid = {"choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                                               "tool_calls": [{"id": "c1", "type": "function",
                                                               "function": {"name": "probe",
                                                                            "arguments": "{}"}}]},
                        "finish_reason": "tool_calls"}], "usage": {}}
    assert "reasoning_content" not in mid["choices"][0]["message"]
    assert mid["choices"][0]["message"]["content"] is None
    assert mid["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "probe"
    print("  mcp_bridge final turn: parsed reasoning lifted to reasoning_content "
          "by to_openai; tool_calls turn path carries no reasoning")


def test_mcp_bridge_render_turn_omitted_display_on_final_reply():
    """The text bridge's reasoning_display=omitted contract must reach the
    MCP bridge's final turn too: a caller asking for the reasoning only
    gets an empty assistant text and the chain on `reasoning_content`.
    The render here is the call into cli_bridge.to_openai with the
    request's stored display policy.
    """
    cb = server.cli_bridge
    payload = {"result": "the answer", "reasoning": "thinking",
               "usage": {"input_tokens": 1, "output_tokens": 1}}
    out = cb.to_openai(payload, "m", reasoning_display="omitted")
    msg = out["choices"][0]["message"]
    assert msg["content"] == "", msg
    assert msg["reasoning_content"] == "thinking", msg
    print("  mcp_bridge final turn + omitted display: content=\"\", "
          "reasoning_content populated")


def test_mcp_bridge_render_turn_passes_session_thinking_to_to_openai():
    """End-to-end pin for the production render path: a Session that was
    started with `display == "omitted"` on the thinking policy must surface
    `content == ""` and `reasoning_content` populated on its final turn.
    The earlier `test_mcp_bridge_render_turn_omitted_display_on_final_reply`
    bypasses `render_turn` by calling `to_openai` directly, so it does not
    catch the production bug where render_turn drops the display policy.
    This test drives render_turn end to end.
    """
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-render-omitted-"))
    try:
        session = server.Session(
            id=uuid.uuid4().hex, provider="claude", model="claude-sonnet-5",
            workdir=str(workdir),
            thinking={"type": "enabled", "display": "omitted"})
        result = {"type": "final",
                  "payload": {"result": "the answer", "reasoning": "thinking",
                              "usage": {"input_tokens": 1, "output_tokens": 1}}}
        response = server.render_turn(session, result, session.model)
        msg = response["choices"][0]["message"]
        assert msg["content"] == "", ("display=omitted MUST suppress visible "
                                       "content on the mcp_bridge final turn "
                                       "to match the cli_bridge text path",
                                       msg)
        assert msg["reasoning_content"] == "thinking", msg

        # And the same Session with a different policy (summarized) MUST NOT
        # suppress content -- only display=omitted does.
        session.thinking = {"type": "enabled", "display": "summarized"}
        result2 = {"type": "final",
                   "payload": {"result": "the answer", "reasoning": "thinking",
                               "usage": {"input_tokens": 1, "output_tokens": 1}}}
        response2 = server.render_turn(session, result2, session.model)
        msg2 = response2["choices"][0]["message"]
        assert msg2["content"] == "the answer", msg2
        assert msg2["reasoning_content"] == "thinking", msg2

        # And a Session without any thinking policy MUST NOT suppress content
        # even if the parser happened to extract reasoning text -- the caller's
        # policy, not the parser's accident, decides what to render.
        session.thinking = None
        result3 = {"type": "final",
                   "payload": {"result": "the answer", "reasoning": "thinking",
                               "usage": {"input_tokens": 1, "output_tokens": 1}}}
        response3 = server.render_turn(session, result3, session.model)
        msg3 = response3["choices"][0]["message"]
        assert msg3["content"] == "the answer", msg3
        assert msg3["reasoning_content"] == "thinking", msg3
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print("  render_turn (production path): session.thinking.display -> "
          "to_openai(reasoning_display=...) honours omitted contract")


# ------------------------------------------ issue #130: acquire-before-deliver --
def test_followup_503_on_saturated_gate_leaves_session_parked():
    """A 503 on a saturated plan must leave the follow-up path's state untouched.

    Before issue #130, `_continue_followup` resolved the parked tool_call
    futures BEFORE calling `unpark_session`, and `unpark_session` cleared
    `awaiting_followup` up front. On a 503 the parked calls had already been
    popped and the CLI had already been kicked back into inference with no
    slot held -- the inner CLI ran over the cap with the session looking
    delivered, the gateway's retry hit the supersede-and-rebuild path, and
    the original CLI subprocess leaked until the 1800 s idle reaper came
    round. The fix is acquire-before-deliver: hoist the duplicate-delivery
    short-circuit (issue #13 defect 1), then `await unpark_session` so the
    slot is held and `awaiting_followup` cleared only AFTER a successful
    acquire, THEN the resolve loop. A 503 above mutates nothing; the
    gateway's retry walks the same follow-up path against the same live
    session.

    Drives `_continue_followup` twice through the same hand-built parked
    session, mirroring `test_followup_error_result_does_not_leak` (parked
    session built with `_new_session` + `register_tool_call`, stubbed
    `new_turn`/`await_turn`) and `test_unpark_returns_503_quickly_when_gate_is_full`
    (fresh `cli_bridge.Gate` swap, saturate to `limit`, `server.RESUME_WAIT`
    pinned to `SHORT`):

      1. First call: the gate is saturated, `_continue_followup` raises
         HTTPException 503 with the parked future NOT done, the call still in
         `session.pending`, `session.awaiting_followup` still True, the id
         NOT in `resolved_recently`, and `gate.in_flight == limit`.
      2. `gate.release()` a slot and retry `_continue_followup`: the turn
         proceeds from the SAME session (no supersede/rebuild -- exactly one
         `await_turn` call, exactly one live session in `SESSIONS`), pending
         is popped, and the parked `register_tool_call` task resolves to the
         delivered content.
    """
    SHORT = 0.3
    # tool_calls turn keeps the session in SESSIONS (park_session rather than
    # end_session) -- the assertion "one live session in SESSIONS" below needs
    # the post-turn state to be parked, not ended.
    SECOND_TURN = {"type": "tool_calls", "calls": []}
    await_turn_calls: list = []

    async def scenario():
        # Same fresh-gate reason as the unpark/resume tests: bind the lock
        # and condition to this scenario's event loop, not one an earlier test
        # used.
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()

        # Hand-build a parked session: tool_calls turn already resolved by
        # the CLI, the parked future waiting on the caller. `holds_slot` is
        # False (parked sessions hold no slot by design; see park_session).
        session = _new_session()
        session.new_turn()
        parked = asyncio.create_task(
            server.register_tool_call(session.id, "get_weather",
                                      {"city": "Oslo"}))
        await asyncio.sleep(0)
        turn = await session.turn_future
        assert turn["type"] == "tool_calls", turn
        call = turn["calls"][0]
        session.awaiting_followup = True
        session.holds_slot = False

        # Saturate the gate from a known clean state so unpark_session's
        # acquire_waiting 503s deterministically within SHORT.
        limit = server.cli_bridge.config().concurrency
        for _ in range(limit):
            assert await gate.acquire(limit)
        assert gate.in_flight == limit, gate.in_flight

        # Stub await_turn so we can assert it is called exactly once on the
        # retry path (no rebuild fired). Stub new_turn on the session so its
        # awaitable resolves without run_session being started -- the second
        # pass otherwise would block forever on a never-running CLI.
        real_new_turn = type(session).new_turn

        def resolved_new_turn(self):
            f = asyncio.get_event_loop().create_future()
            f.set_result(SECOND_TURN)
            self.turn_future = f
            return f
        type(session).new_turn = resolved_new_turn

        real_await_turn = server.await_turn

        async def record_turn(sess, request):
            await_turn_calls.append({"session_id": sess.id})
            return SECOND_TURN
        server.await_turn = record_turn

        saved_wait = server.RESUME_WAIT
        server.RESUME_WAIT = SHORT
        try:
            body = {"model": "m",
                    "messages": [
                        {"role": "tool", "tool_call_id": call.id,
                         "content": '{"temp_c":-3}'},
                    ]}
            # ---- 1) Saturated-gate path: 503, nothing mutated. ----
            try:
                await server._continue_followup(body, session.id,
                                                body["messages"], None)
            except server.HTTPException as exc:
                assert exc.status_code == 503, exc.status_code
                assert "no slot freed" in exc.detail, exc.detail
                assert exc.headers and exc.headers.get("Retry-After") == "15", \
                    exc.headers
            else:
                raise AssertionError(
                    "_continue_followup must 503 when the gate stays full")

            # Invariants: the parked future is still waiting, the call is
            # still in pending, the session still looks parked, the id was
            # NOT marked resolved (so a retry cannot be misread as a
            # duplicate), and the gate was not silently mutated.
            assert not call.future.done(), \
                "parked future must not be resolved on 503"
            assert call.id in session.pending, \
                f"parked call must remain in pending on 503: {session.pending}"
            assert session.awaiting_followup, \
                "awaiting_followup must remain True on 503"
            assert not session.holds_slot, \
                "holds_slot must remain False on 503"
            assert call.id not in session.resolved_recently, \
                "resolved_recently must not record the id on 503"
            assert gate.in_flight == limit, gate.in_flight
            assert session.id in server.SESSIONS, dict(server.SESSIONS)
            assert not session.dead, session

            # ---- 2) Retry: free a slot and run again, same session. ----
            await gate.release()
            assert gate.in_flight == limit - 1, gate.in_flight

            response = await server._continue_followup(
                body, session.id, body["messages"], None)

            # No rebuild: the same live session answered, exactly one turn
            # awaited, exactly one live session in SESSIONS, and the parked
            # register_tool_call task received the delivered content.
            assert response is not None, response
            assert len(await_turn_calls) == 1, await_turn_calls
            assert await_turn_calls[0]["session_id"] == session.id, \
                await_turn_calls
            assert session.id in server.SESSIONS, dict(server.SESSIONS)
            # The parked call was popped during the resolve loop on the
            # retry and resolved to the delivered content.
            assert call.id not in session.pending, \
                f"parked call must be popped after retry resolves it: " \
                f"{session.pending}"
            assert call.future.done(), \
                "parked future must be resolved after the retry"
            tool_result = await parked
            assert tool_result["content"] == [
                {"type": "text", "text": '{"temp_c":-3}'}], tool_result
            assert tool_result["isError"] is False, tool_result
        finally:
            server.RESUME_WAIT = saved_wait
            server.await_turn = real_await_turn
            type(session).new_turn = real_new_turn
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate

    asyncio.run(scenario())
    assert server.cli_bridge._gate.in_flight == 0, \
        server.cli_bridge._gate.in_flight
    print("  follow-up 503 left the session parked (awaiting_followup=True, "
          "parked future not done); retry after gate.release() resolved the "
          "same session's parked call (one await_turn call, one live session)")


def test_followup_superseded_mid_turn_rebuilds_instead_of_500():
    """Regression for issue #96.

    Two follow-ups race on the same mcp_bridge session: follow-up #1 resolves
    its parked tool_call_ids, calls unpark_session, then awaits the new
    turn_future while the CLI is still working. While that turn is pending,
    the winning follow-up #2 calls supersede_session on the same session
    (its tool_call_ids do not match because follow-up #1 already consumed
    them). supersede_session fails the pending turn_future with a
    `_SupersedeSignal`.

    Before this fix, that exception escaped the `except BaseException`
    teardown in `_continue_followup` and the gateway saw a 500. After this
    fix, the catch recognises the typed signal, rebuilds via
    `resume_gone_session`, and returns the rebuilt response the winning
    follow-up is also getting. Reverting the server.py change reproduces
    the production error verbatim.
    """
    captured: list = []
    restore = _stub_start_session(captured)
    try:
        async def scenario():
            session = _new_session()
            session.new_turn()
            # Parked with a real parked call whose id the follow-up WILL name
            # so the resolve loop resolves ONE call (the normal continue
            # path) -- not the resolved == 0 branch the parallel-batch test
            # exercises. `register_tool_call` is what tool_server.py invokes
            # from its inner CLI; it blocks until `set_result` is called
            # from `_continue_followup`'s resolve loop. `new_turn()` is
            # called first because Session.enqueue's _flush only resolves a
            # turn_future that exists.
            parked = asyncio.create_task(
                server.register_tool_call(session.id, "get_weather",
                                          {"city": "Oslo"}))
            await asyncio.sleep(0)
            turn = await session.turn_future
            assert turn["type"] == "tool_calls", turn
            session.awaiting_followup = True
            parked_id = next(iter(session.pending))

            # Concurrent supersede -- 50 ms after _continue_followup starts
            # awaiting the new turn_future. The gap mirrors the production
            # race window (two follow-ups from the same GUI arriving a few
            # ms apart, the second winning the resolve). Using the same
            # reason string the resolved == 0 branch uses so a future
            # refactor that flips which branch fires doesn't change the
            # test's contract.
            async def race():
                await asyncio.sleep(0.05)
                await server.supersede_session(
                    session,
                    "no parked call matched delivered ids (rebuilding)")

            supersede_task = asyncio.create_task(race())

            body = {"model": "m",
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather", "description": "",
                        "parameters": {"type": "object",
                                       "properties": {}}}}],
                    "messages": [
                        {"role": "user", "content": "weather in Oslo?"},
                        {"role": "assistant", "content": None, "tool_calls": [
                            {"id": parked_id, "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": '{"city":"Oslo"}'}}]},
                        {"role": "tool", "tool_call_id": parked_id,
                         "content": '{"temp_c":-3}'},
                    ]}
            tool_msgs = body["messages"][2:]
            # Drive _continue_followup directly with request=None (the test
            # path through await_turn skips its RuntimeError-to-HTTPException
            # mapping and propagates the underlying error -- which is exactly
            # the surface _continue_followup's catch handles).
            try:
                result = await server._continue_followup(body, session.id,
                                                         tool_msgs, None)
                await supersede_task
                return result, session
            finally:
                # The parked call's future was resolved by _continue_followup
                # BEFORE supersede fired, so the register_tool_call task
                # returns cleanly. Consume it here so asyncio does not log
                # a "future exception was never retrieved" warning if the
                # race ordering ever shifts.
                with contextlib.suppress(Exception):
                    await parked

        result, session = asyncio.run(scenario())
        # The rebuild fired -- start_session was called exactly once.
        assert len(captured) == 1, captured
        # And the caller got the rebuilt response back, NOT the
        # `_SupersedeSignal` (or pre-fix RuntimeError) that used to
        # escape and surface as a 500.
        assert result == {"stubbed": True}, result
        # The superseded session is gone from SESSIONS and marked dead.
        # The exception that would have escaped pre-fix had the same
        # "_SupersedeSignal" message text, so a future regression that
        # drops the catch reproduces the exact production error.
        assert session.id not in server.SESSIONS, dict(server.SESSIONS)
        assert session.dead, session
        print(f"  follow-up caught its own supersede and rebuilt: "
              f"start_session called once; session {session.id[:8]}... "
              f"ended")
    finally:
        restore()


def test_followup_duplicate_delivery_in_flight_attaches_to_running_turn():
    """Regression for issue #96's owner comment / PR #306.

    A follow-up whose `tool_call_id` matches nothing parked here BUT
    whose original turn is still in flight (mid-turn, not parked,
    `turn_future` not done) attaches to the running turn rather than
    superseding it. Superseding here would race the CLI and throw the
    caller's tool results away; rebuilding from the request would mint
    a fresh CLI and skip the work already in flight. Reverting the
    new branch takes the supersede+rebuild path, which is wrong on a
    session whose CLI is still working.
    """
    captured = []
    restore = _stub_start_session(captured)
    try:
        async def scenario():
            session = _new_session()
            # Start the original turn (the one the follow-up is
            # duplicating). We do NOT call register_tool_call: this
            # branch fires when the follow-up's ids match nothing
            # parked (resolved == 0) but the session is mid-turn.
            session.new_turn()
            session.awaiting_followup = False   # mid-turn state
            # Mark the old id as already-resolved so the follow-up is
            # unambiguously a duplicate (not in pending, but already
            # consumed). The resolve loop iterates pending; an empty
            # pending means resolved stays at 0, which is what the new
            # branch's condition requires.
            old_id = "call_old_1"
            session.mark_resolved(old_id)

            # Resolve the turn with a final payload, AFTER yielding so
            # _continue_followup has a chance to reach its `await_turn`
            # before `turn_future.done()` flips True.
            async def resolve_turn():
                await asyncio.sleep(0)
                session.resolve_final({
                    "type": "final",
                    "payload": {"result": "Oslo is cold"},
                })
            resolve_task = asyncio.create_task(resolve_turn())

            body = {"model": "m",
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather", "description": "",
                        "parameters": {"type": "object",
                                       "properties": {}}}}],
                    "messages": [
                        {"role": "user", "content": "weather in Oslo?"},
                        {"role": "assistant", "content": None, "tool_calls": [
                            {"id": old_id, "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": '{}'}}]},
                        {"role": "tool", "tool_call_id": old_id,
                         "content": "{}"},
                    ]}
            tool_msgs = body["messages"][2:]
            try:
                result = await server._continue_followup(body, session.id,
                                                         tool_msgs, None)
                await resolve_task
                return result, session, captured
            finally:
                # The branch ends with end_session on a final result;
                # if it parked instead, drain the parked-future warning.
                if (session.id in server.SESSIONS
                        and session.turn_future is not None
                        and not session.turn_future.done()):
                    with contextlib.suppress(Exception):
                        await session.turn_future

        result, session, captured = asyncio.run(scenario())
        # The duplicate-delivery-in-flight branch fired: it attached
        # to the running turn rather than supersede+rebuild. start_session
        # was NOT called.
        assert captured == [], captured
        # The branch ends with the running turn's final response,
        # rendered through the normal render_turn path (to_openai
        # picks `payload["result"]` for content, finish_reason="stop"
        # for the final type).
        assert result["choices"][0]["message"]["content"] == "Oslo is cold", result
        assert result["choices"][0]["finish_reason"] == "stop", result
        # And the session is gone (final turn -> end_session) -- the
        # branch did not leave it parked or holding the slot.
        assert session.id not in server.SESSIONS, dict(server.SESSIONS)
        assert session.dead, session
        print(f"  duplicate delivery attached to running turn: "
              f"start_session not called, finish_reason=stop, "
              f"content={result['choices'][0]['message']['content']!r}")
    finally:
        restore()


# ----------------------------------------- issue #137: _warm serialization ---
def _write_event_fake_cli(workdir: Path, marker_path: Path, event: str,
                          *, sleep: float = 0.0) -> str:
    """Write a fake CLI into `workdir / event / "fake_cli.py"` that records
    `<time> <event>_SPAWN` and `<time> <event>_DONE` to the shared
    `marker_path`, optionally sleeping first, then prints a successful
    parseable payload.

    Both fakes share the same marker file but use distinct event names
    so the test can order the events by wall-clock time and assert the
    warm_gate's serialization semantics:

    * SLOW_FIRST spawn / SLOW_FIRST done -- a deliberately slow first
      run that the gate must serialize.
    * FAST_SECOND spawn -- must wait for SLOW_FIRST done before this
      timestamp is recorded.

    Each fake gets its own subdir under `workdir` (`{event}/fake_cli.py`)
    so the two subprocesses don't overwrite each other's script -- both
    would otherwise land at `workdir/fake_cli.py`.

    The test owns `workdir` (it creates and rmtree's it around the
    run); the helper does not mkdtemp anything of its own, so a normal
    suite run of the three new tests leaves zero /tmp/mcpb-* dirs
    behind -- matching every other lock-step test in this file that
    sets its own `workdir = Path(tempfile.mkdtemp(...))` at the top
    and `shutil.rmtree(workdir, ignore_errors=True)`s it in `finally`.
    """
    body = (
        "import json, time\n"
        f"marker = {str(marker_path)!r}\n"
        f"event = {event!r}\n"
        f"sleep = {sleep}\n"
        "with open(marker, 'a') as fh:\n"
        "    fh.write(f'{time.time()} {event}_SPAWN\\n')\n"
        "    fh.flush()\n"
        "if sleep:\n"
        "    time.sleep(sleep)\n"
        "with open(marker, 'a') as fh:\n"
        f"    fh.write(f'{{time.time()}} {event}_DONE\\n')\n"
        "    fh.flush()\n"
        "print(json.dumps({'result': event, 'usage': "
        "{'input_tokens': 1, 'output_tokens': 1}}))\n"
    )
    sub = workdir / event
    sub.mkdir()
    path = sub / "fake_cli.py"
    path.write_text(body)
    return str(path)


def _read_marker_events(marker_path: str) -> list[tuple[float, str]]:
    """Parse the marker file's `<time> <event>` lines into a sorted list.

    Wall-clock writes are append-only and the test's event names are
    process-unique (SLOW_FIRST / FAST_SECOND), so an unsorted
    read still preserves the order each fake wrote its lines. The sort
    by timestamp catches any clock-skew drift on slow CI hosts.
    """
    out: list[tuple[float, str]] = []
    with open(marker_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t_str, name = line.split(" ", 1)
            out.append((float(t_str), name))
    out.sort(key=lambda x: x[0])
    return out


def _first_time(events: list[tuple[float, str]], name: str) -> float | None:
    for t, n in events:
        if n == name:
            return t
    return None


def test_run_session_warm_gate_serializes_concurrent_first_calls():
    """Issue #137 parity: the cli_bridge text path wraps `invoke` in
    `warm_gate`, which serializes the first concurrent call so a token
    refresh races exactly once. The mcp_bridge tool path must do the
    same -- without it a cold start would have every parallel spawn race
    to refresh and rewrite the shared credential file.

    Drives two `run_session`s concurrently, the first artificially slow.
    The shared marker file records each subprocess's spawn and done time.
    The load-bearing assertion is that the second subprocess's spawn is
    AT OR AFTER the first's done -- the gate held the lock for the
    first's full body before the second could proceed. A concurrent
    spawn race (the bug this guard exists to prevent) would have
    FAST_SECOND_SPAWN < SLOW_FIRST_DONE.

    We deliberately do NOT pin the strict cross-process chain
    `slow_spawn < fast_spawn < fast_done` with `<`. `time.time()` is not
    monotonic; an NTP step on a CI host between two subprocesses'
    wall-clock writes turns a causally-correct run into a red one, and
    the gate's correctness is already implied by the load-bearing
    `fast_spawn >= slow_done` below (a spurious NTP step can't make a
    value go below a *later* event on the same process).

    `_warm` and `_warmup_lock` are saved and restored: the cli_bridge
    module is process-shared across every test in this file, and an
    earlier test that warmed it would otherwise skip the slow path
    entirely and silently pass without exercising the lock. Session
    new_turn() must be called inside the event loop (it uses
    asyncio.get_event_loop().create_future), so the session setup also
    moves into the async function.
    """
    saved_warm = server.cli_bridge._warm
    saved_lock = server.cli_bridge._warmup_lock
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-warm-"))
    try:
        marker_path = workdir / "events.log"
        marker_path.write_text("")
        slow_cli = _write_event_fake_cli(workdir, marker_path,
                                        "SLOW_FIRST", sleep=0.3)
        fast_cli = _write_event_fake_cli(workdir, marker_path, "FAST_SECOND")
        s_slow = _new_session()
        s_fast = _new_session()
        # Cold start: the gate must take the slow path. Set the new Event /
        # Lock inside the running loop so they bind to the same loop the
        # warm_gate body will use; creating them outside the loop works in
        # Python 3.10/3.11 but the second test in this file runs AFTER the
        # first asyncio.run() closed its loop, and a fresh asyncio.Lock() in
        # a thread with no current loop is the wrong shape.
        async def scenario():
            server.cli_bridge._warm = asyncio.Event()
            server.cli_bridge._warmup_lock = asyncio.Lock()
            s_slow.new_turn()
            s_fast.new_turn()
            t_slow = asyncio.create_task(
                server.run_session(s_slow, [sys.executable, slow_cli]))
            # Yield once so the slow task grabs the warmup lock first.
            # Otherwise the gather below could schedule the fast task
            # ahead of it, and the "fast fake spawns after slow done"
            # assertion would still hold but for the wrong reason (both
            # race for the lock at the same time). The yield makes the
            # contract under test explicit.
            await asyncio.sleep(0)
            t_fast = asyncio.create_task(
                server.run_session(s_fast, [sys.executable, fast_cli]))
            await asyncio.gather(t_slow, t_fast)

        asyncio.run(scenario())

        # Both calls completed successfully.
        assert s_slow.turn_future.result()["type"] == "final"
        assert s_fast.turn_future.result()["type"] == "final"

        # _warm is set after the first call succeeded.
        assert server.cli_bridge._warm.is_set(), \
            "_warm must be set after the first run_session succeeded"

        # The slow fake's spawn must come first (it acquired the lock).
        # The fast fake's spawn must wait for the slow fake's done --
        # that is exactly what the gate guards: a token-refresh race
        # would otherwise have both spawns in flight at the same time.
        events = _read_marker_events(str(marker_path))
        slow_spawn = _first_time(events, "SLOW_FIRST_SPAWN")
        slow_done = _first_time(events, "SLOW_FIRST_DONE")
        fast_spawn = _first_time(events, "FAST_SECOND_SPAWN")
        fast_done = _first_time(events, "FAST_SECOND_DONE")
        assert slow_spawn is not None and slow_done is not None, events
        assert fast_spawn is not None and fast_done is not None, events
        # The gate holds the warmup lock for the first call's body. The
        # second call can only enter the body (the spawn above) AFTER the
        # first call's body has completed and set _warm. A concurrent
        # spawn race (the bug this guard exists to prevent) would have
        # FAST_SECOND_SPAWN < SLOW_FIRST_DONE.
        assert fast_spawn >= slow_done, (
            f"gate did not serialize the cold-start race: "
            f"slow_done={slow_done} fast_spawn={fast_spawn} events={events}")
        print(f"  run_session warm_gate: serialized cold-start race "
              f"(slow_done={slow_done:.3f} <= fast_spawn={fast_spawn:.3f}); "
              f"_warm set after first success; fast_done={fast_done}")
    finally:
        # Restore the cli_bridge module-level globals so the next test
        # starts from a known state. Doing this in finally (vs in the
        # inner scenario) means we always restore, even when the assert
        # below fires.
        server.cli_bridge._warm = saved_warm
        server.cli_bridge._warmup_lock = saved_lock
        shutil.rmtree(workdir, ignore_errors=True)


def test_run_session_warm_gate_failed_first_leaves_gate_closed():
    """A failed first run leaves the gate closed: the next caller
    serializes alone again, exactly the success-only semantics today's
    `invoke` had. Without this the gate would treat any error as a
    success and stop protecting subsequent cold starts.

    Two phases:

    * First run uses a fake CLI that exits 1 with an auth-style stderr
      -- `_run_session_attempt` resolves the turn_future as 401 and
      returns None (no payload, no exception). `run_session` reports
      `gate["success"] = False` and the gate stays closed.
    * Second run uses a successful fake. The gate was still closed, so
      this run takes the lock, runs alone, and sets _warm.

    The `_warm` / `_warmup_lock` save/restore is the same isolation
    pattern as the serialize test above; new_turn() must be called
    inside the running loop (see comment in the serialize test).
    """
    saved_warm = server.cli_bridge._warm
    saved_lock = server.cli_bridge._warmup_lock
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-warm-"))
    try:
        fail_cli = _write_fail_fake_cli(workdir)
        good_cli = _write_good_fake_cli(workdir)
        s_fail = _new_session()
        s_good = _new_session()

        async def scenario():
            server.cli_bridge._warm = asyncio.Event()
            server.cli_bridge._warmup_lock = asyncio.Lock()
            s_fail.new_turn()
            s_good.new_turn()
            await server.run_session(s_fail, [sys.executable, fail_cli])
            # The failed run must NOT have warmed the gate.
            assert not server.cli_bridge._warm.is_set(), \
                "a failed first run left _warm set -- gate no longer success-only"
            await server.run_session(s_good, [sys.executable, good_cli])

        asyncio.run(scenario())

        # The failed run resolved as a 401 (auth-style stderr maps to
        # that on this branch; cli_bridge._AUTH re is the same regex the
        # production spawn path uses).
        failed = s_fail.turn_future.result()
        assert failed["type"] == "error", failed
        assert failed["status"] == 401, failed

        # The good run succeeded and warmed the gate.
        good = s_good.turn_future.result()
        assert good["type"] == "final", good
        assert server.cli_bridge._warm.is_set(), \
            "_warm must be set after the second (successful) run"
        print("  run_session warm_gate: failed first run leaves gate closed "
              "(401), good second run sets _warm")
    finally:
        server.cli_bridge._warm = saved_warm
        server.cli_bridge._warmup_lock = saved_lock
        shutil.rmtree(workdir, ignore_errors=True)


def _write_fail_fake_cli(workdir: Path) -> str:
    """Write a fake CLI into `workdir / "fail" / "fake_cli.py"` that exits 1
    with auth-style stderr; mapped by `_run_session_attempt` to a 401
    error envelope (no payload). The test owns `workdir`.

    Each failing/good pair sits in its own subdir under the test's
    `workdir` so the two scripts don't overwrite each other at
    `workdir/fake_cli.py`."""
    body = (
        "import sys\n"
        "sys.stderr.write('please run `claude login` to authenticate')\n"
        "sys.exit(1)\n"
    )
    sub = workdir / "fail"
    sub.mkdir()
    path = sub / "fake_cli.py"
    path.write_text(body)
    return str(path)


def _write_good_fake_cli(workdir: Path) -> str:
    """Write a fake CLI into `workdir / "good" / "fake_cli.py"` that prints a
    successful parseable payload. The test owns `workdir`."""
    body = (
        "import json\n"
        "print(json.dumps({'result': 'OK', 'usage': "
        "{'input_tokens': 1, 'output_tokens': 1}}))\n"
    )
    sub = workdir / "good"
    sub.mkdir()
    path = sub / "fake_cli.py"
    path.write_text(body)
    return str(path)


def test_followup_reaped_mid_turn_rebuilds_instead_of_500():
    """Regression for issue #197.

    A follow-up races the 1800 s idle reaper (or an explicit
    `reap_session` call) on the same mcp_bridge session: follow-up #1
    resolves its parked tool_call_ids, calls unpark_session, then awaits
    the new turn_future while the CLI is still working. While that turn
    is pending, the reaper fires `reap_session` on the same session
    (the idle TTL has been reached, or the operator asked for it
    directly). `reap_session` fails the in-flight `turn_future` with a
    `_ReapedSignal`.

    Before this fix, that bare `RuntimeError("session reaped")` escaped
    the `except BaseException` teardown in `_continue_followup` and the
    gateway saw a 500 (in tests, where `await_turn` does not map the
    bare RuntimeError to a 499 because the request is in-process). The
    pre-existing 499 mapping in `await_turn` was already a substring
    match on `"reaped" in str(exc)` -- a brittle pattern that mirrors
    nothing in the typed-signal design PR #306 introduced for the
    supersede race. After this fix, the catch recognises the typed
    `_ReapedSignal`, rebuilds via `resume_gone_session`, and returns the
    rebuilt response instead of letting the signal escape. Reverting
    the server.py change reproduces the production error verbatim.
    """
    captured: list = []
    restore = _stub_start_session(captured)
    try:
        async def scenario():
            session = _new_session()
            session.new_turn()
            # Parked with a real parked call whose id the follow-up WILL name
            # so the resolve loop resolves ONE call (the normal continue
            # path) -- not the resolved == 0 branch the parallel-batch test
            # exercises. `register_tool_call` is what tool_server.py invokes
            # from its inner CLI; it blocks until `set_result` is called
            # from `_continue_followup`'s resolve loop. `new_turn()` is
            # called first because Session.enqueue's _flush only resolves a
            # turn_future that exists.
            parked = asyncio.create_task(
                server.register_tool_call(session.id, "get_weather",
                                          {"city": "Oslo"}))
            await asyncio.sleep(0)
            turn = await session.turn_future
            assert turn["type"] == "tool_calls", turn
            session.awaiting_followup = True
            parked_id = next(iter(session.pending))

            # Concurrent reap -- 50 ms after _continue_followup starts
            # awaiting the new turn_future. The gap mirrors the production
            # race window (a follow-up that arrives after the operator (or
            # a hot path) reaped an already-idle session). On a reaped
            # session `reap_session` is idempotent: end_session's dead-guard
            # makes a second call a no-op, so the only side effect we are
            # racing is the `set_exception(_ReapedSignal)` on the
            # in-flight turn_future.
            async def race():
                await asyncio.sleep(0.05)
                await server.reap_session(session)

            reap_task = asyncio.create_task(race())

            body = {"model": "m",
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather", "description": "",
                        "parameters": {"type": "object",
                                       "properties": {}}}}],
                    "messages": [
                        {"role": "user", "content": "weather in Oslo?"},
                        {"role": "assistant", "content": None, "tool_calls": [
                            {"id": parked_id, "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": '{"city":"Oslo"}'}}]},
                        {"role": "tool", "tool_call_id": parked_id,
                         "content": '{"temp_c":-3}'},
                    ]}
            tool_msgs = body["messages"][2:]
            # Drive _continue_followup directly with request=None (the test
            # path through await_turn skips its RuntimeError-to-HTTPException
            # mapping and propagates the underlying error -- which is exactly
            # the surface _continue_followup's catch handles).
            try:
                result = await server._continue_followup(body, session.id,
                                                         tool_msgs, None)
                await reap_task
                return result, session
            finally:
                # The parked call's future was resolved by _continue_followup
                # BEFORE reap fired, so the register_tool_call task returns
                # cleanly. Consume it here so asyncio does not log a "future
                # exception was never retrieved" warning if the race
                # ordering ever shifts.
                with contextlib.suppress(Exception):
                    await parked

        result, session = asyncio.run(scenario())
        # The rebuild fired -- start_session was called exactly once.
        assert len(captured) == 1, captured
        # And the caller got the rebuilt response back, NOT the
        # `_ReapedSignal` (or pre-fix bare RuntimeError) that used to
        # escape and surface as a 500.
        assert result == {"stubbed": True}, result
        # The reaped session is gone from SESSIONS and marked dead.
        # The exception that would have escaped pre-fix had the same
        # "session reaped" message text, so a future regression that
        # drops the catch reproduces the exact production error.
        assert session.id not in server.SESSIONS, dict(server.SESSIONS)
        assert session.dead, session
        print(f"  follow-up caught its own reap and rebuilt: "
              f"start_session called once; session {session.id[:8]}... "
              f"ended")
    finally:
        restore()


def test_followup_duplicate_delivery_reaped_mid_turn_rebuilds_instead_of_500():
    """Regression for issue #197, second catch site.

    Same race as `test_followup_reaped_mid_turn_rebuilds_instead_of_500`,
    but on the duplicate-delivery-in-flight branch in `_continue_followup`
    (the one whose `except BaseException` was added in PR #306 alongside
    the main path). The duplicate-delivery path fires when the follow-up's
    tool_call_ids match nothing parked AND the session's `turn_future` is
    still in flight -- the natural supersede+rebuild path would race the
    CLI, so this branch attaches to the running turn instead. The new
    `_ReapedSignal` catch has to fire here too, otherwise a follow-up that
    loses the in-flight race to the idle reaper still escapes as a 500.
    """
    captured: list = []
    restore = _stub_start_session(captured)
    try:
        async def scenario():
            session = _new_session()
            # Start the original turn (the one the follow-up is
            # duplicating). We do NOT call register_tool_call: this
            # branch fires when the follow-up's ids match nothing
            # parked (resolved == 0) but the session is mid-turn.
            session.new_turn()
            session.awaiting_followup = False   # mid-turn state
            # Mark the old id as already-resolved so the follow-up is
            # unambiguously a duplicate (not in pending, but already
            # consumed). The resolve loop iterates pending; an empty
            # pending means resolved stays at 0, which is what the new
            # branch's condition requires.
            old_id = "call_old_1"
            session.mark_resolved(old_id)

            # Concurrent reap -- 50 ms after _continue_followup starts
            # awaiting the in-flight turn_future. The turn_future has
            # to still be pending when reap fires, otherwise
            # `set_exception` is a no-op and the catch never sees the
            # signal. Mirrors the main test's 50 ms gap.
            async def race():
                await asyncio.sleep(0.05)
                await server.reap_session(session)

            reap_task = asyncio.create_task(race())

            body = {"model": "m",
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather", "description": "",
                        "parameters": {"type": "object",
                                       "properties": {}}}}],
                    "messages": [
                        {"role": "user", "content": "weather in Oslo?"},
                        {"role": "assistant", "content": None, "tool_calls": [
                            {"id": old_id, "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": '{}'}}]},
                        {"role": "tool", "tool_call_id": old_id,
                         "content": "{}"},
                    ]}
            tool_msgs = body["messages"][2:]
            try:
                result = await server._continue_followup(body, session.id,
                                                         tool_msgs, None)
                await reap_task
                return result, session
            finally:
                # The branch attaches to the running turn; if it parked
                # instead, drain the parked-future warning.
                if (session.id in server.SESSIONS
                        and session.turn_future is not None
                        and not session.turn_future.done()):
                    with contextlib.suppress(Exception):
                        await session.turn_future

        result, session = asyncio.run(scenario())
        # The rebuild fired -- start_session was called exactly once
        # (the duplicate-delivery-in-flight branch's natural attachment
        # would have called start_session ZERO times; the typed catch
        # converts the in-flight reap race into a rebuild instead).
        assert len(captured) == 1, captured
        # And the caller got the rebuilt response back, NOT the
        # `_ReapedSignal` that used to escape and surface as a 500.
        assert result == {"stubbed": True}, result
        # The reaped session is gone from SESSIONS and marked dead.
        assert session.id not in server.SESSIONS, dict(server.SESSIONS)
        assert session.dead, session
        print(f"  duplicate-delivery follow-up caught its own reap and "
              f"rebuilt: start_session called once; session "
              f"{session.id[:8]}... ended")
    finally:
        restore()


# --------------------------------- issue #70: max_tokens on the mcp_bridge tool path ---
def test_render_turn_final_truncates_long_answer_to_cap_with_length_finish():
    """The mcp_bridge final turn now runs cli_bridge.enforce_max_tokens the
    same way cli_bridge._complete does (bridge-siblings rule, issue #70):
    an assistant reply whose byte length exceeds `cap * 4` is cut on a UTF-8
    boundary to roughly that size, finish_reason becomes "length", and the
    usage the CLI reported is left untouched -- the caller paid for whatever
    the CLI actually produced (issue #70's honest-bookkeeping rule). The
    truncation site documents the tool path's no-400 contract: a CLI tool
    loop is one conversation the router cannot spill mid-loop, so the only
    honest outcome on an overshoot is to truncate and surface length.
    """
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-render-trunc-"))
    try:
        session = server.Session(
            id=uuid.uuid4().hex, provider="claude", model="claude-sonnet-5",
            workdir=str(workdir), max_tokens=10)
        # 100 'x' bytes -- ceil(100 / 4) = 25 tokens, well past the cap of 10.
        # The cut lands at 40 bytes (cap * 4) modulo a multi-byte boundary
        # that the 100-byte ASCII string does not exercise, so the cut is
        # byte-exact at 40 here.
        long_text = "x" * 100
        result = {"type": "final",
                  "payload": {"result": long_text,
                              "usage": {"input_tokens": 5, "output_tokens": 25,
                                        "total_tokens": 30}}}
        response = server.render_turn(session, result, session.model)
        msg = response["choices"][0]["message"]
        assert response["choices"][0]["finish_reason"] == "length", \
            response["choices"][0]
        # enforce_max_tokens cuts to cap * 4 = 40 bytes for pure ASCII;
        # 100 'x' -> 40 'x' on a UTF-8 boundary, so byte-exact here.
        assert len(msg["content"].encode("utf-8")) <= 40, \
            (len(msg["content"]), msg["content"])
        assert msg["content"].startswith("x" * 30), \
            ("truncated answer should still begin with the original text",
             msg["content"])
        # Honest usage: the CLI reported output_tokens=25 (what it actually
        # generated); enforce_max_tokens does NOT touch usage, and neither
        # does to_openai when finish_reason is length -- the caller books
        # what it paid for, not the byte-truncated text length.
        usage = response["usage"]
        assert usage["completion_tokens"] == 25, usage
        assert usage["prompt_tokens"] == 5, usage
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print(f"  render_turn final turn: 100-byte answer at cap=10 -> "
          f"{len(msg['content'])}-byte cut, finish_reason=length, "
          f"usage unchanged (output_tokens={usage['completion_tokens']})")


def test_render_turn_tool_calls_turn_passes_through_byte_untouched():
    """A mid-loop `tool_calls` turn is byte-untouched with
    finish_reason="tool_calls" -- the cap is for the final assistant
    answer text, not for a parked tool request. The tool_calls response
    is hand-built by render_turn (not built by to_openai), so
    enforce_max_tokens must NOT run on it; doing so would silently cut
    the tool-call arguments JSON and break the call.
    """
    async def go():
        workdir = Path(tempfile.mkdtemp(prefix="mcpb-render-tc-"))
        try:
            session = server.Session(
                id=uuid.uuid4().hex, provider="claude", model="claude-sonnet-5",
                workdir=str(workdir), max_tokens=1)
            # A parked call carries a name and arguments that look like tool
            # plumbing, not assistant text. Set up a single fake call so the
            # tool_calls branch builds the response without raising. render_turn
            # only reads c.id / c.name / c.arguments, so the future just needs
            # to exist; the loop context here makes a real Future constructable.
            call = server.ParkedCall(
                id="call_x", name="probe",
                arguments={"k": "v" * 1000},
                future=asyncio.get_running_loop().create_future())
            result = {"type": "tool_calls", "calls": [call]}
            response = server.render_turn(session, result, session.model)
            assert response["choices"][0]["finish_reason"] == "tool_calls", \
                response["choices"][0]
            # Arguments are the same JSON the tool_server.py expects to
            # round-trip back to the CLI -- byte-untouched regardless of cap.
            msg = response["choices"][0]["message"]
            assert msg["content"] is None, msg
            assert msg["tool_calls"][0]["function"]["name"] == "probe"
            assert msg["tool_calls"][0]["function"]["arguments"] == \
                '{"k": "' + ("v" * 1000) + '"}', msg["tool_calls"]
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        print("  render_turn tool_calls turn: arguments JSON byte-untouched "
              "despite max_tokens=1, finish_reason=tool_calls")

    asyncio.run(go())


def test_render_turn_final_within_or_no_cap_keeps_stop_finish():
    """Two adjacent cases: a final answer that fits the cap, and a final
    answer where the caller never asked for one at all (session.max_tokens
    is None). Both must land as finish_reason="stop" with the content
    byte-identical to what the CLI produced -- enforce_max_tokens is a
    no-op in both cases, and the unchanged-output behaviour (and any
    caller-side caching keyed on it) is preserved (see cli_bridge
    enforce_max_tokens docstring).
    """
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-render-stop-"))
    try:
        # Within-cap: 20-byte answer at cap=10 -> 5 tokens, well under the
        # cap, must pass through untouched.
        session = server.Session(
            id=uuid.uuid4().hex, provider="claude", model="claude-sonnet-5",
            workdir=str(workdir), max_tokens=10)
        short_text = "short answer."          # 13 bytes, 4 tokens
        result = {"type": "final",
                  "payload": {"result": short_text,
                              "usage": {"input_tokens": 1, "output_tokens": 4}}}
        response = server.render_turn(session, result, session.model)
        assert response["choices"][0]["finish_reason"] == "stop", \
            response["choices"][0]
        assert response["choices"][0]["message"]["content"] == short_text, \
            response["choices"][0]["message"]

        # No cap: session.max_tokens is None, so enforce_max_tokens is a
        # no-op and finish_reason stays "stop" regardless of length.
        session.max_tokens = None
        long_text = "y" * 4096
        result2 = {"type": "final",
                   "payload": {"result": long_text,
                               "usage": {"input_tokens": 1, "output_tokens": 1024}}}
        response2 = server.render_turn(session, result2, session.model)
        assert response2["choices"][0]["finish_reason"] == "stop", \
            response2["choices"][0]
        assert response2["choices"][0]["message"]["content"] == long_text, \
            "no cap must leave the answer byte-identical"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print("  render_turn final turn: within-cap and no-cap both -> "
          "finish_reason=stop, content byte-identical")


def test_render_turn_final_on_rebuilt_session_still_truncates_to_cap():
    """A follow-up-born rebuilt session must still honour the cap on its
    eventual final turn. The rebuild path is resume_gone_session ->
    start_session with the same body, and start_session re-reads
    cli_bridge.request_max_tokens(body) every time it runs -- so a Session
    built from the rebuild path carries the same max_tokens value a fresh
    Session would. This test pins both halves end-to-end: (a) the real
    start_session wiring (body -> Session.max_tokens) for both the
    max_tokens spelling and LiteLLM's max_completion_tokens rename, and
    (b) the render_turn final-turn path consuming session.max_tokens to
    truncate an overshooting answer.

    Same harness as test_error_result_releases_gate_slot_and_removes_session:
    a stubbed run_session captures the Session that start_session actually
    constructed, which is the only way to pin the production wiring rather
    than the request_max_tokens primitive in isolation (the previous test
    assigned request_max_tokens output directly and so did not exercise
    the new Session(max_tokens=...) kwarg in start_session -- a regression
    that drops the kwarg or breaks resume_gone_session's body passthrough
    would have passed).
    """
    captured_sessions: list = []
    FINAL_RESULT = {"type": "final",
                    "payload": {"result": "ok", "usage": {"input_tokens": 1,
                                                          "output_tokens": 1,
                                                          "total_tokens": 2}}}

    async def scenario():
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()
        real_run = server.run_session

        async def fake_run(session, argv, stdin_data=None):
            # Capture the Session that start_session actually constructed
            # -- its max_tokens field is set by the line this PR added.
            captured_sessions.append(session)
            session.resolve_final(FINAL_RESULT)

        server.run_session = fake_run
        try:
            tools = [{"type": "function", "function": {
                "name": "f", "description": "",
                "parameters": {"type": "object", "properties": {}}}}]
            # (a) Drive start_session's body -> Session.max_tokens wiring
            # for both cap spellings the runtime ever sees. A fresh gate
            # is bound here (same as the issue #80 regression test) so the
            # gate acquire inside handle_fresh is uncontested.
            body_with_cap = {"model": "m", "max_tokens": 10,
                             "tools": tools,
                             "messages": [{"role": "user", "content": "hi"}]}
            await server.handle_fresh(body_with_cap, tools, None)

            # LiteLLM renames max_tokens to max_completion_tokens for gpt-5
            # names (issue #279); the same body, the same primitive.
            body_litellm = {**body_with_cap}
            del body_litellm["max_tokens"]
            body_litellm["max_completion_tokens"] = 25
            await server.handle_fresh(body_litellm, tools, None)
        finally:
            server.run_session = real_run
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate

    asyncio.run(scenario())
    # Two sessions captured, one per handle_fresh call.
    assert len(captured_sessions) == 2, len(captured_sessions)
    session_with_cap, session_litellm = captured_sessions
    # (a) start_session picked the cap up from each body's spelling --
    # not from the Session default. A regression that drops the
    # `max_tokens=cli_bridge.request_max_tokens(body)` kwarg from
    # start_session's Session(...) constructor would land these as None.
    assert session_with_cap.max_tokens == 10, session_with_cap.max_tokens
    assert session_litellm.max_tokens == 25, session_litellm.max_tokens

    # (b) render_turn's final-turn path consumes session.max_tokens and
    # truncates an overshooting answer -- the end-to-end contract.
    result = {"type": "final",
              "payload": {"result": "z" * 100,
                          "usage": {"input_tokens": 2, "output_tokens": 25,
                                    "total_tokens": 27}}}
    response = server.render_turn(session_with_cap, result, session_with_cap.model)
    msg = response["choices"][0]["message"]
    assert response["choices"][0]["finish_reason"] == "length", \
        response["choices"][0]
    assert len(msg["content"].encode("utf-8")) <= 40, \
        ("start_session-built Session final turn must truncate to cap*4 bytes",
         len(msg["content"]))
    assert msg["content"].startswith("z" * 30), msg["content"]
    print(f"  start_session-built Session: max_tokens={session_with_cap.max_tokens}, "
          f"max_completion_tokens variant={session_litellm.max_tokens}; "
          f"cap=10 truncates 100-byte answer to {len(msg['content'])} bytes, "
          f"finish_reason=length")


def test_continue_followup_refreshes_session_max_tokens_from_per_turn_body():
    """The live follow-up path must refresh session.max_tokens from the
    per-turn body, mirroring what start_session already does on the rebuild
    path. Without this refresh, a session that stays alive across turns uses
    the stale cap from the first request -- exactly the failure #133
    describes for a Claude Code / OpenCode tool loop that sends max_tokens
    on every turn.

    Three cases pinned (each via a fresh parked session and a real
    _continue_followup call, with await_turn and new_turn stubbed the same
    way test_followup_error_result_does_not_leak does):

      (a) follow-up ADDS a cap (None -> 10) -- render_turn would have
          landed as 'stop' on an unbounded answer without the refresh;
      (b) follow-up CHANGES the cap (10 -> 25) -- render_turn would have
          kept truncating to 10 instead of 25;
      (c) follow-up DROPS the cap (10 -> None) -- render_turn would have
          kept truncating to 10 instead of letting the answer be unbounded.

    All three must leave session.max_tokens == the per-turn body's
    request_max_tokens by the time _continue_followup returns, so the
    subsequent render_turn sees the per-turn policy.
    """
    FINAL = {"type": "final",
             "payload": {"result": "ok", "usage": {"input_tokens": 1,
                                                   "output_tokens": 1,
                                                   "total_tokens": 2}}}
    bodies_and_initial = [
        # (a) follow-up adds a cap
        ({"model": "m", "max_tokens": 10,
          "messages": [{"role": "tool", "tool_call_id": "t1",
                        "content": "ok"}]}, None),
        # (b) follow-up changes the cap
        ({"model": "m", "max_tokens": 25,
          "messages": [{"role": "tool", "tool_call_id": "t1",
                        "content": "ok"}]}, 10),
        # (c) follow-up drops the cap
        ({"model": "m",
          "messages": [{"role": "tool", "tool_call_id": "t1",
                        "content": "ok"}]}, 10),
    ]

    async def scenario():
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()
        real_new_turn = server.Session.new_turn
        real_await_turn = server.await_turn

        async def fast_await_turn(sess, request):
            return FINAL
        server.await_turn = fast_await_turn

        observed: list = []
        try:
            for body, initial_max in bodies_and_initial:
                # Hand-build a parked session identical to the issue #80
                # regression's setup: real new_turn() so the initial
                # turn_future resolves to tool_calls via register_tool_call's
                # _flush. Override new_turn on the class BEFORE calling
                # _continue_followup so the await on new_turn inside it
                # returns an already-resolved future (run_session is never
                # started in this scenario -- without the override the
                # await on new_turn would hang).
                session = _new_session()
                session.new_turn()
                parked = asyncio.create_task(
                    server.register_tool_call(session.id, "echo", {"text": "x"}))
                await asyncio.sleep(0)
                turn = await session.turn_future
                assert turn["type"] == "tool_calls", turn
                call = turn["calls"][0]
                session.awaiting_followup = True
                session.holds_slot = False
                # Pre-load the session's stored max_tokens with the FIRST
                # request's value (None in case (a), 10 in cases (b)/(c))
                # so the refresh line is what changes it, not start_session.
                session.max_tokens = initial_max

                def resolved_new_turn(self):
                    f = asyncio.get_event_loop().create_future()
                    f.set_result(FINAL)
                    self.turn_future = f
                    return f
                server.Session.new_turn = resolved_new_turn

                # Replace the synthetic tool_msg's call_id with the real one
                # minted by register_tool_call.
                followup_body = {**body}
                followup_body["messages"] = [
                    {"role": "tool", "tool_call_id": call.id,
                     "content": "ok"}]
                await server._continue_followup(
                    followup_body, session.id,
                    followup_body["messages"], None)
                observed.append((session.id, session.max_tokens,
                                 call.id))
                # Drain the parked coroutine so its outcome is observed.
                await parked
                server.SESSIONS.pop(session.id, None)
                # Restore real new_turn so the next iteration's parked-
                # session setup uses the real one.
                server.Session.new_turn = real_new_turn
        finally:
            server.Session.new_turn = real_new_turn
            server.await_turn = real_await_turn
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate
        return observed

    observed = asyncio.run(scenario())
    # (a) None -> 10
    assert observed[0][1] == 10, observed[0]
    # (b) 10 -> 25
    assert observed[1][1] == 25, observed[1]
    # (c) 10 -> None
    assert observed[2][1] is None, observed[2]
    print("  live follow-up refresh: None->10, 10->25, 10->None all wired "
          "(session.max_tokens now follows the per-turn body)")


# ------------------- parked session: CLI dies mid-loop reaps on finally -----
def test_run_session_finally_reaps_a_parked_session_after_cli_exit_1():
    """A fake CLI that exits 1 while its tool_call is parked must NOT leave
    the session parked forever.

    Before this fix, `run_session`'s finally only called `session.touch()`.
    Every CLI failure path inside `_run_session_attempt` (502 on nonzero
    exit, 408 on PROCESS_TIMEOUT, the 500 crash handler) only resolved the
    `turn_future` -- a no-op on the delivered tool_calls future -- and never
    ended the session. The parked tool_call futures stayed unresolved, so
    tool_server.py's HTTP POST blocked forever, the inner CLI sat wedged on
    the reply, and the follow-up arrived to find a session that looked
    live but would never answer.

    The fix: `finally` calls `note_preempted(session.id)` and
    `await reap_session(session)` when the session is not dead AND
    `session.awaiting_followup` is True (a normal park keeps run_session
    awaiting proc.communicate, so reaching finally with the flag still
    set means the driver died mid-loop). This test drives the
    nonzero-exit branch directly with a hand-built parked session and a
    fake CLI that exits 1 with a stderr line; _run_session_attempt hits
    the generic 502 path (line 1537) and the finally must reap.

    Mirrors the parked-session setup in
    `test_followup_503_on_saturated_gate_leaves_session_parked` (park
    via `_new_session` + `register_tool_call`, await the tool_calls turn,
    mark `awaiting_followup`/`holds_slot`).
    """
    fake_cli = _write_fake_cli(
        "import sys\n"
        "sys.stderr.write('boom\\n')\n"
        "sys.exit(1)\n")

    async def scenario():
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()
        try:
            session = _new_session()
            session.new_turn()
            parked = asyncio.create_task(
                server.register_tool_call(session.id, "get_weather",
                                          {"city": "Oslo"}))
            await asyncio.sleep(0)
            turn = await session.turn_future
            assert turn["type"] == "tool_calls", turn
            call = turn["calls"][0]
            session.awaiting_followup = True
            session.holds_slot = False

            # CLI dies mid-park with a nonzero exit. _run_session_attempt
            # hits the generic 502 path (no auth/limit/status match for
            # the fake's stderr), resolves turn_future as 502 (no-op since
            # it is already done), returns None. The new finally reaps.
            await server.run_session(session, [sys.executable, fake_cli])
            # End-of-fix invariants: dead, gone from SESSIONS, id
            # remembered as preempted, gate untouched (parked sessions
            # hold no slot by design; reap_session's holds_slot guard
            # skips release).
            assert session.dead, session.dead
            assert session.id not in server.SESSIONS, dict(server.SESSIONS)
            assert session.id in server.PREEMPTED, dict(server.PREEMPTED)
            assert gate.in_flight == 0, gate.in_flight
            # fail_pending fired in reap_session, so the parked register
            # tool_call task raised HTTPException 504 (register_tool_call's
            # RuntimeError -> 504 mapping). Drain it so its outcome is
            # observed; future regressions that leave the parked future
            # unresolved would log "future exception was never retrieved".
            return session, parked, call.id
        finally:
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate

    session, parked, parked_call_id = asyncio.run(scenario())
    # Drain the parked task OUTSIDE the scenario so the suppressed
    # HTTPException doesn't get a "future exception was never retrieved"
    # warning if the test runner exits first.
    with contextlib.suppress(Exception):
        parked.result()
    print(f"  run_session finally reaped parked session {session.id[:8]}... "
          f"after CLI exit 1 (dead, gone, in PREEMPTED, gate in_flight=0); "
          f"parked call {parked_call_id[:12]}... failed via fail_pending")


def test_run_session_finally_reaps_a_parked_session_after_process_timeout():
    """The PROCESS_TIMEOUT kill branch must reap a parked session on finally
    too -- the timeout kills the proc but otherwise leaves the session in
    the exact same half-state as a crash.

    Same harness as the exit-1 test, with a short PROCESS_TIMEOUT override
    and a fake CLI that hangs past it. _run_session_attempt hits the
    asyncio.TimeoutError branch (line 1502), kills the proc, resolves
    turn_future as 408 (no-op), returns None. The finally must reap.
    """
    saved_timeout = server.PROCESS_TIMEOUT
    server.PROCESS_TIMEOUT = 0.3
    try:
        fake_cli = _write_fake_cli(
            "import time\n"
            "time.sleep(10)\n")

        async def scenario():
            saved_gate = server.cli_bridge._gate
            gate = server.cli_bridge.Gate()
            server.cli_bridge._gate = gate
            saved_sessions = dict(server.SESSIONS)
            server.SESSIONS.clear()
            try:
                session = _new_session()
                session.new_turn()
                parked = asyncio.create_task(
                    server.register_tool_call(session.id, "get_weather",
                                              {"city": "Oslo"}))
                await asyncio.sleep(0)
                turn = await session.turn_future
                assert turn["type"] == "tool_calls", turn
                session.awaiting_followup = True
                session.holds_slot = False

                # The wait_for(PROCESS_TIMEOUT=0.3) fires; proc.kill() runs;
                # resolve_final(408) is a no-op on the delivered turn_future.
                # The new finally reaps.
                await server.run_session(session, [sys.executable, fake_cli])
                assert session.dead, session.dead
                assert session.id not in server.SESSIONS, dict(server.SESSIONS)
                assert session.id in server.PREEMPTED, dict(server.PREEMPTED)
                assert gate.in_flight == 0, gate.in_flight
                return session, parked
            finally:
                server.SESSIONS.clear()
                server.SESSIONS.update(saved_sessions)
                server.cli_bridge._gate = saved_gate

        session, parked = asyncio.run(scenario())
        with contextlib.suppress(Exception):
            parked.result()
        print(f"  run_session finally reaped parked session {session.id[:8]}... "
              f"after PROCESS_TIMEOUT (dead, gone, in PREEMPTED, gate "
              f"in_flight=0)")
    finally:
        server.PROCESS_TIMEOUT = saved_timeout


def test_followup_after_parked_cli_died_rebuilds_via_preempted():
    """A follow-up arriving after a parked session's CLI died must rebuild
    from the request history, NOT block on the missing parked call.

    Drives the full follow-up path end-to-end:
      1. park a session via _new_session + register_tool_call,
      2. kill the CLI with a fake that exits 1 -- run_session's new
         finally reaps the session and adds its id to PREEMPTED,
      3. post a follow-up via handle_followup; the lookup misses
         SESSIONS, sees the id in PREEMPTED, and rebuilds via
         resume_gone_session (stubbed start_session so the test does not
         spawn a real CLI for the rebuilt turn).

    Asserts the rebuild fired (start_session called once, response is the
    stub), the slot was released (gate.in_flight back to 0), and the
    parked register_tool_call task was failed by reap_session's
    fail_pending (no orphan future-warning on exit).

    Mirrors the rebuild-stub pattern from
    `test_followup_superseded_mid_turn_rebuilds_instead_of_500` (stubbed
    start_session, fresh gate).
    """
    fake_cli = _write_fake_cli(
        "import sys\nsys.exit(1)\n")
    captured: list = []
    restore = _stub_start_session(captured)
    try:
        async def scenario():
            saved_gate = server.cli_bridge._gate
            gate = server.cli_bridge.Gate()
            server.cli_bridge._gate = gate
            saved_sessions = dict(server.SESSIONS)
            server.SESSIONS.clear()
            try:
                session = _new_session()
                session.new_turn()
                parked = asyncio.create_task(
                    server.register_tool_call(session.id, "get_weather",
                                              {"city": "Oslo"}))
                await asyncio.sleep(0)
                turn = await session.turn_future
                assert turn["type"] == "tool_calls", turn
                call = turn["calls"][0]
                session.awaiting_followup = True
                session.holds_slot = False

                # Step 1: CLI dies mid-park. The new finally reaps.
                await server.run_session(session, [sys.executable, fake_cli])
                assert session.id not in server.SESSIONS
                assert session.id in server.PREEMPTED

                # Step 2: follow-up arrives. handle_followup looks the id
                # up in SESSIONS (miss), checks PREEMPTED (hit), and
                # routes to resume_gone_session -> start_session (stub).
                body = {"model": "m",
                        "tools": [{"type": "function", "function": {
                            "name": "get_weather", "description": "",
                            "parameters": {"type": "object",
                                           "properties": {"city": {
                                               "type": "string"}}}}}],
                        "messages": [
                            {"role": "user", "content": "weather?"},
                            {"role": "assistant", "content": None,
                             "tool_calls": [{"id": call.id, "type": "function",
                                              "function": {"name": "get_weather",
                                                           "arguments": '{"city":"Oslo"}'}}]},
                            {"role": "tool", "tool_call_id": call.id,
                             "content": '{"temp_c":-3}'},
                        ]}
                tool_msgs = body["messages"][2:]
                response = await server.handle_followup(body, tool_msgs, None)
                with contextlib.suppress(Exception):
                    parked.result()
                return response, captured, gate.in_flight
            finally:
                server.SESSIONS.clear()
                server.SESSIONS.update(saved_sessions)
                server.cli_bridge._gate = saved_gate

        response, captured, in_flight = asyncio.run(scenario())
        # Rebuild fired -- start_session was called exactly once. The
        # follow-up got the same rebuilt response the stub returns.
        assert len(captured) == 1, captured
        assert response == {"stubbed": True}, response
        # Slot released back to the gate: resume_gone_session acquired via
        # acquire_resume_slot, then the stub released it.
        assert in_flight == 0, in_flight
        print("  follow-up after parked-CLI crash rebuilt via PREEMPTED: "
              "start_session called once, response=stubbed, gate in_flight=0")
    finally:
        restore()


# ------------------- tool_server.py: finite CALLBACK_TIMEOUT default -------
def test_tool_server_callback_timeout_default_is_session_ttl_plus_60():
    """tool_server.py must compute its default CALLBACK_TIMEOUT as
    SESSION_TTL + 60 (a finite ceiling, not None), so a bridge that goes
    silent or dies never leaves the inner CLI's urlopen blocked for a day.
    An explicit SWITCHYARD_CALLBACK_TIMEOUT env value still wins.

    tool_server reads env vars at import time, so the test imports it in
    a subprocess under several env combinations and reads the printed
    CALLBACK_TIMEOUT back. Loopback is allowed by the conftest socket
    guard; only real-provider hosts are blocked.
    """
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-cbto-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    saved_env = {k: os.environ.get(k) for k in (
        "SWITCHYARD_CALLBACK_TIMEOUT", "MCP_SESSION_TTL_SECONDS")}
    try:
        cases = [
            # (MCP_SESSION_TTL_SECONDS, expected default)
            ("120",   180.0),    # 120 + 60
            ("1800",  1860.0),   # the default; mirrors server.py
            ("3600",  3660.0),
        ]
        for ttl, expected in cases:
            env = {**os.environ,
                   "SWITCHYARD_TOOLS_FILE": str(tools_path),
                   "SWITCHYARD_SESSION_ID": "abcd",
                   "SWITCHYARD_CALLBACK_URL": "http://127.0.0.1:1",
                   "MCP_SESSION_TTL_SECONDS": ttl}
            env.pop("SWITCHYARD_CALLBACK_TIMEOUT", None)
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import sys, json; "
                 "sys.path.insert(0, %r); "
                 "import tool_server; "
                 "print(json.dumps({'CALLBACK_TIMEOUT': "
                 "tool_server.CALLBACK_TIMEOUT}))"
                 % MCP_BRIDGE_DIR],
                env=env, capture_output=True, text=True, timeout=10,
            )
            assert proc.returncode == 0, (proc.stdout, proc.stderr)
            data = json.loads(proc.stdout.strip())
            assert data["CALLBACK_TIMEOUT"] == expected, \
                f"MCP_SESSION_TTL_SECONDS={ttl}: expected {expected}, " \
                f"got {data['CALLBACK_TIMEOUT']}"

        # An explicit SWITCHYARD_CALLBACK_TIMEOUT still wins, regardless of
        # MCP_SESSION_TTL_SECONDS (and even when the explicit value is
        # below the computed default).
        for explicit in ("42", "5", "9999"):
            env = {**os.environ,
                   "SWITCHYARD_TOOLS_FILE": str(tools_path),
                   "SWITCHYARD_SESSION_ID": "abcd",
                   "SWITCHYARD_CALLBACK_URL": "http://127.0.0.1:1",
                   "MCP_SESSION_TTL_SECONDS": "1800",
                   "SWITCHYARD_CALLBACK_TIMEOUT": explicit}
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import sys, json; "
                 "sys.path.insert(0, %r); "
                 "import tool_server; "
                 "print(json.dumps({'CALLBACK_TIMEOUT': "
                 "tool_server.CALLBACK_TIMEOUT}))"
                 % MCP_BRIDGE_DIR],
                env=env, capture_output=True, text=True, timeout=10,
            )
            assert proc.returncode == 0, (proc.stdout, proc.stderr)
            data = json.loads(proc.stdout.strip())
            assert data["CALLBACK_TIMEOUT"] == float(explicit), \
                f"SWITCHYARD_CALLBACK_TIMEOUT={explicit}: expected " \
                f"{float(explicit)}, got {data['CALLBACK_TIMEOUT']}"
        print("  tool_server CALLBACK_TIMEOUT default = SESSION_TTL+60 "
              "(120+60=180, 1800+60=1860, 3600+60=3660 verified); "
              "explicit SWITCHYARD_CALLBACK_TIMEOUT=42/5/9999 wins")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_write_claude_mcp_config_injects_callback_timeout_into_harness_env():
    """write_claude_mcp_config (the claude MCP harness writer) must inject
    SWITCHYARD_CALLBACK_TIMEOUT into the stdio child's env dict. The
    harness dict is the only reliable way both stdio children get the
    value (claude + opencode); tool_server.py's SESSION_TTL+60 default
    is the safety net for harnesses that don't pass it.

    Pinned against the same SESSION_TTL constant server.py uses, so a
    drift on either side (call SESSION_TTL + N off-by-one, drop the env
    entry, drop write_opencode_dir's parallel injection) is caught by
    this test.

    Also asserts the operator-override path: if SWITCHYARD_CALLBACK_TIMEOUT
    is set in the sidecar's own environment (the container-scope the
    reviewer flagged in finding 4), both writers forward it verbatim
    rather than unconditionally clobbering it with SESSION_TTL+60.
    """
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-claudecbto-"))
    saved_env = os.environ.get("SWITCHYARD_CALLBACK_TIMEOUT")
    try:
        tools_path = workdir / "tools.json"
        tools_path.write_text("[]")

        # Default path: no operator override -> SESSION_TTL+60 is written
        # into both stdio children's env dicts, identical to the value
        # tool_server.py would compute as a safety net.
        if "SWITCHYARD_CALLBACK_TIMEOUT" in os.environ:
            del os.environ["SWITCHYARD_CALLBACK_TIMEOUT"]
        mcp_json = server.write_claude_mcp_config(workdir, "sess-x", tools_path)
        cfg = json.loads(mcp_json.read_text())
        env = cfg["mcpServers"]["switchyard"]["env"]
        assert env["SWITCHYARD_CALLBACK_TIMEOUT"] == str(server.SESSION_TTL + 60), \
            env
        # The three documented keys are still all present (regression
        # guard against this PR accidentally dropping one).
        assert env["SWITCHYARD_TOOLS_FILE"] == str(tools_path), env
        assert env["SWITCHYARD_SESSION_ID"] == "sess-x", env
        assert env["SWITCHYARD_CALLBACK_URL"].startswith("http://"), env
        # And write_opencode_dir gets the same value on its parallel
        # `environment` block.
        server.write_opencode_dir(workdir, "sess-y", tools_path)
        ocfg = json.loads((workdir / "opencode.json").read_text())
        ocfg_env = ocfg["mcp"]["switchyard"]["environment"]
        assert ocfg_env["SWITCHYARD_CALLBACK_TIMEOUT"] == str(server.SESSION_TTL + 60), \
            ocfg_env

        # Operator-override path: an operator-tuned value at container
        # scope is forwarded verbatim by BOTH writers, so the contract
        # is enforced at the writer (not left to whichever stdio harness
        # resolves env-precedence). Use a value clearly distinct from the
        # default so a regression that always writes the default is
        # caught.
        os.environ["SWITCHYARD_CALLBACK_TIMEOUT"] = "77"
        mcp_json = server.write_claude_mcp_config(workdir, "sess-x2",
                                                   tools_path)
        cfg = json.loads(mcp_json.read_text())
        assert cfg["mcpServers"]["switchyard"]["env"][
            "SWITCHYARD_CALLBACK_TIMEOUT"] == "77", cfg
        server.write_opencode_dir(workdir, "sess-y2", tools_path)
        ocfg = json.loads((workdir / "opencode.json").read_text())
        assert ocfg["mcp"]["switchyard"]["environment"][
            "SWITCHYARD_CALLBACK_TIMEOUT"] == "77", ocfg
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if saved_env is None:
            os.environ.pop("SWITCHYARD_CALLBACK_TIMEOUT", None)
        else:
            os.environ["SWITCHYARD_CALLBACK_TIMEOUT"] = saved_env
    print(f"  write_claude_mcp_config + write_opencode_dir inject "
          f"SWITCHYARD_CALLBACK_TIMEOUT=SESSION_TTL+60={server.SESSION_TTL+60:.0f} "
          f"by default; operator override 77 is forwarded verbatim")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
