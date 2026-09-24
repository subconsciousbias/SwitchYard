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


def _codex_bridge_text(body: dict) -> str:
    """The text the bridge wrote into the codex request body, stripped of
    the `<image name="..." path="...">` blocks codex 0.155.1 wraps each
    `-i` arg with on the wire. The bridge does not control codex's wire
    format; what the bridge owns is the user-text content, which the change
    in this PR makes path-free (issue #296).

    Codex 0.155.1's Responses-API wire shape puts top-level items of
    `type: "message"` (and `additional_tools` for tools) on `body["input"]`,
    with `input_text` blocks nested inside `item["content"]`. Descend one
    level for message items; keep the `<image ...>` filter on each text
    block (codex wraps each `-i` arg in its own wire-format wrapper that
    names the staged file path -- that is codex's, not the bridge's)."""
    items = body.get("input") if isinstance(body.get("input"), list) else []
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        blocks = item.get("content")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "input_text":
                continue
            text = block.get("text") or ""
            if not isinstance(text, str):
                text = json.dumps(text)
            # Codex's own wire format around each `-i` arg is an
            # `<image name=... path=...>` input_text block -- the bridge
            # never emits that shape.
            if text.lstrip().startswith("<image "):
                continue
            parts.append(text)
    return "\n".join(parts)


def _stage_for_lockdown(stage: Path, provider: str) -> tuple[str, list[Path]]:
    """Drive the cli_bridge the way `_handle_chat` does for a media call:
    build messages carrying the test's PNG + PDF, run stage_images (the
    bridge writes `[image N: attached]` markers, expands the PDF via
    poppler on codex/opencode), flatten the messages into a prompt, and
    append image_note. The caller passes the returned prompt + paths to
    build_argv so the request codex/opencode actually emits carries the
    path-free markers + trailing note this PR's change is supposed to
    deliver.

    Both server.PROVIDER (mcp_bridge) and server.cli_bridge.PROVIDER
    (cli_bridge -- stage_images / _expand_pdf read this one) are flipped
    to `provider` so stage_images takes the right branch for the PDF
    (expansion on codex/opencode; single document block on claude). The
    test's `saved` tuple restores cb.PROVIDER on the way out. The caller
    must already be inside a FakePoppler() block when this runs on
    codex/opencode."""
    server.PROVIDER = provider
    server.cli_bridge.PROVIDER = provider
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


def test_codex_media_reach_the_model_not_relay_paths():
    """The bridge's image and PDF markers reach the pinned codex as `-i`
    attaches, the bridge's own prompt text is path-free (the change in
    this PR), and the caller's tool (only) is what the model can call on
    the tool path (issue #296).

    Codex 0.155.1 wraps every `-i` arg in its own `<image name="..."
    path="...">` block on the wire, so the request body the model API sees
    does name the relay dir inside codex's wrapper -- but the bridge's
    own prompt text (the chat content the bridge wrote, stripped of
    codex's wire-format wrappers) does not. The lockdown tests assert
    only what the bridge owns.

    The bridge is driven through `stage_images` + `flatten` +
    `image_note` (the same chain `_handle_chat` runs) so the prompt
    really does carry the path-free `[image N: attached]` markers and
    the `[N attachment(s) follow this text...]` trailing note this PR
    delivers."""
    try:
        codex = _pinned_clis.require("codex")
    except _pinned_clis.Skip as why:
        print(f"  skipped: {why}")
        return
    cb = server.cli_bridge
    root = Path(tempfile.mkdtemp(prefix="lockdown-codex-media-"))
    _fake_login(root)
    stage = root / "swimg-test" / "img"
    stage.mkdir(parents=True)
    (stage / "01.png").write_bytes(PNG_MAGENTA)
    (stage / "02.pdf").write_bytes(PDF_MIN)
    media = [stage / "01.png", stage / "02.pdf"]
    saved = (cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE, server.PROVIDER, server.PROFILE)
    try:
        # Text path: the bridge hands codex `-i` per file; the prompt is the
        # usual flattened chat + image_note (path-free since #296).
        with FakePoppler(), _RealCodex(codex, root):
            server.PROVIDER = "codex"
            cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE = ("codex", cb.PROFILES["codex"], codex, True)
            prompt, staged = _stage_for_lockdown(stage, "codex")
            argv, _ = cb.build_argv(prompt, None, MODEL, image_paths=staged)
            assert "-i" in argv, argv
            attached = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
            assert attached == [str(p) for p in staged], argv
            with _fake_model.FakeModel([]) as fake:
                done = _run(argv, fake, root, root)
        assert done.returncode == 0, done.stderr[-500:]
        # Text path offers zero tools.
        assert all(not _fake_model.advertised_tools(r["body"]) for r in fake.requests), \
            [_fake_model.advertised_tools(r["body"]) for r in fake.requests]
        # The bridge's own prompt text is path-free: the prompt +
        # image_note reach the model with no relay dir or staged file path
        # (codex's own `<image name=... path=...>` wrappers around each
        # `-i` arg are filtered out -- they're codex's wire format, not the
        # bridge's, and they are unchanged by this PR).
        text = _codex_bridge_text(fake.requests[-1]["body"])
        assert "what is attached?" in text, text
        assert "attachment(s) follow this text" in text, text
        assert "[image 1: attached]" in text, text
        assert "[document 2: PDF," in text, text
        assert str(stage) not in text, text
        for p in media:
            assert str(p) not in text, (p, text)

        # Tool path: same attachment, plus the caller's MCP tool only.
        with FakePoppler(), _RealCodex(codex, root):
            server.PROVIDER = "codex"
            cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE = ("codex", cb.PROFILES["codex"], codex, True)
            prompt, staged = _stage_for_lockdown(stage, "codex")
            server.PROFILE = dict(server.MCP_PROFILES["codex"], cli=codex)
            workdir = root / "session"
            workdir.mkdir()
            tools_path = workdir / "tools.json"
            tools_path.write_text(json.dumps([PROBE]))
            argv, _ = server.build_argv(prompt, "CALLER SYSTEM PROMPT", MODEL,
                                        workdir, uuid.uuid4().hex, tools_path,
                                        "mcp__switchyard__probe", staged)
            assert "-i" in argv, argv
            attached = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
            assert attached == [str(p) for p in staged], argv
            with _fake_model.FakeModel([{"tool": "exec_command",
                                          "input": {"cmd": "touch /tmp/native"}}]) as fake:
                done = _run(argv, fake, root, workdir)
        assert done.returncode == 0, done.stderr[-500:]
        # Caller tool only (plus the three read-only MCP-resource helpers);
        # exec_command still refused -- the same codex_lockdown_args() argv
        # is applied whether or not image_paths is set, so the lockdown
        # holds with media attached too.
        offered = set()
        for req in fake.tool_requests():
            offered |= set(_fake_model.advertised_tools(req["body"]))
        assert "mcp__switchyard__probe" in offered, offered
        assert offered - HELPERS == {"mcp__switchyard__probe"}, sorted(offered)
        text = _fake_model.request_text(fake.requests[-1]["body"])
        assert "unsupported call: exec_command" in text, text[-600:]
        bridge_text = _codex_bridge_text(fake.requests[-1]["body"])
        assert "what is attached?" in bridge_text, bridge_text
        assert "attachment(s) follow this text" in bridge_text, bridge_text
        assert "[image 1: attached]" in bridge_text, bridge_text
        assert "[document 2: PDF," in bridge_text, bridge_text
        assert str(stage) not in bridge_text, bridge_text
        for p in media:
            assert str(p) not in bridge_text, (p, bridge_text)
        print(f"  codex {_pinned_clis.installed_version('codex')}: image + PDF ride -i "
              "on both paths; bridge text path-free; caller tool only; "
              "exec_command still refused")
    finally:
        cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE, server.PROVIDER, server.PROFILE = saved
        shutil.rmtree(root, ignore_errors=True)


# 1x1 magenta PNG and a minimal one-page PDF: what the media test sends.
PNG_MAGENTA = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63f8cf1f7f060006000300013fe46dabe90000000049454e44ae426082")
PDF_MIN = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
           b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
           b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 72 72]>>endobj\n"
           b"trailer<</Root 1 0 R>>\n%%EOF\n")


def _fake_login(root: Path) -> Path:
    """Codex needs only CODEX_HOME; a tiny placeholder keeps tests fast."""
    home = root / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    return home


def test_codex_reasoning_event_parsed_separately_from_answer():
    """Codex reports reasoning as a `reasoning` item on the same event stream
    the agent_message item lives on, with usage flags `reasoning_output_tokens`
    on the turn.completed usage object. The parser collects the chain onto
    `payload.reasoning` (separate from `result`) and rolls the reasoning
    output tokens into output_tokens (already there) -- the two have to live
    on different keys so to_openai can lift reasoning onto a separate
    message field without paying the user's bill twice.
    """
    cb = server.cli_bridge
    stream = "\n".join([
        json.dumps({"type": "item.completed",
                    "item": {"type": "reasoning", "text": "thinking step"}}),
        json.dumps({"type": "item.completed",
                    "item": {"type": "agent_message", "text": "the answer"}}),
        json.dumps({"type": "turn.completed",
                    "usage": {"input_tokens": 4, "output_tokens": 2,
                              "reasoning_output_tokens": 8}}),
    ])
    parsed = cb.parse_output(stream, "codex_jsonl")
    assert parsed["result"] == "the answer", parsed
    assert parsed.get("reasoning") == "thinking step", parsed
    # Reasoning tokens stay rolled into output_tokens (the parser's contract),
    # so the ledger books the full billed output.
    assert parsed["usage"]["output_tokens"] == 10, parsed["usage"]

    # to_openai lifts the reasoning onto a separate field; the result stays
    # on `content`. This is the only path the adapter uses to surface the
    # chain of thought on a codex turn -- downstream codex never sees the
    # raw item.
    out = cb.to_openai(parsed, "gpt-5.6-terra")
    msg = out["choices"][0]["message"]
    assert msg["content"] == "the answer", msg
    assert msg["reasoning_content"] == "thinking step", msg
    print("  codex: reasoning item -> separate `reasoning` field; tokens "
          "rolled into output; reasoning_content lifted by to_openai")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
