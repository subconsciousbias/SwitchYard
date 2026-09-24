"""Claude Code lockdown, proven against the pinned CLI (issue #264; #269 review).

Both bridges build Claude's argv so the model sees the caller's tools and
nothing of the relay: no built-in tools (`--tools ""`), no account MCP
connectors (`--strict-mcp-config`), none of the login directory's CLAUDE.md,
hooks or skills (`--setting-sources ""`), no permission prompt nobody can
answer. The unit tests check the argv; these run the claude binary
Dockerfile.sidecar pins, with exactly that argv, against a loopback fake
model (tests/_fake_model.py), from a login directory with planted
CLAUDE.md / hook / skill markers and an OAuth credentials file, and make the
model call Agent, AskUserQuestion and Bash anyway.

Skips when the pinned claude is not installed; CI's cli-lockdown job
installs it and makes a skip a failure (tests/_pinned_clis.py).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
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

MARKERS = ("RELAY_CLAUDE_MD_MARKER", "RELAY_HOOK_MARKER", "RELAY_SKILL_MARKER")
PROBE = {"name": "probe", "description": "a caller tool",
         "inputSchema": {"type": "object", "properties": {}}}


def _login_dir(root: Path) -> Path:
    """A Claude login dir as a sidecar mounts one: OAuth credentials, plus
    the CLAUDE.md, hook and skill an operator's login may carry."""
    home = root / "claude"
    (home / "skills" / "relayskill").mkdir(parents=True)
    (home / "CLAUDE.md").write_text(f"{MARKERS[0]}\n")
    (home / "settings.json").write_text(json.dumps({"hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": f"echo {MARKERS[1]}"}]}]}}))
    (home / "skills" / "relayskill" / "SKILL.md").write_text(
        f"---\nname: relayskill\ndescription: {MARKERS[2]}\n---\nbody\n")
    (home / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {
        "accessToken": "sk-ant-oat01-lockdown-test", "refreshToken": "sk-ant-ort01-x",
        "expiresAt": int(time.time() * 1000) + 86_400_000,
        "scopes": ["user:inference", "user:profile"], "subscriptionType": "max"}}))
    os.chmod(home / ".credentials.json", 0o600)
    return home


def _run(argv: list[str], fake: _fake_model.FakeModel, root: Path, cwd: Path,
         stdin_data: str | None = None):
    env = {**server.cli_bridge.subprocess_env(),
           "HOME": str(root), "CLAUDE_CONFIG_DIR": str(root / "claude"),
           "ANTHROPIC_BASE_URL": fake.url,
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    env.pop("ANTHROPIC_API_KEY", None)
    if stdin_data is not None:
        return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,
                              timeout=180, input=stdin_data)
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=180, stdin=subprocess.DEVNULL)


def _assert_relay_stays_out(fake: _fake_model.FakeModel) -> None:
    assert fake.requests, "the CLI never reached the model"
    for req in fake.requests:
        text = _fake_model.request_text(req["body"])
        for marker in MARKERS:
            assert marker not in text, f"login-dir {marker} reached the model"
        assert "authorization" in {k.lower() for k in req["headers"]}, \
            "OAuth bearer missing: the lockdown broke the subscription login"


def test_claude_tool_path_offers_only_the_callers_tools():
    try:
        claude = _pinned_clis.require("claude")
    except _pinned_clis.Skip as why:
        print(f"  skipped: {why}")
        return
    root = Path(tempfile.mkdtemp(prefix="lockdown-claude-"))
    _login_dir(root)
    marker = root / "native-bash-ran"
    script = [{"tool": "Agent", "input": {"description": "x", "prompt": "hi"}},
              {"tool": "AskUserQuestion", "input": {"questions": [{
                  "question": "q?", "header": "h", "multiSelect": False,
                  "options": [{"label": "a", "description": "a"},
                              {"label": "b", "description": "b"}]}]}},
              {"tool": "Bash", "input": {"command": f"touch {marker}"}}]
    saved = (server.PROVIDER, server.PROFILE)
    try:
        server.PROVIDER = "claude"
        server.PROFILE = dict(server.MCP_PROFILES["claude"], cli=claude)
        workdir = root / "session"
        workdir.mkdir()
        tools_path = workdir / "tools.json"
        tools_path.write_text(json.dumps([PROBE]))
        argv, _ = server.build_argv("hello", "CALLER SYSTEM PROMPT", "claude-sonnet-5",
                                    workdir, uuid.uuid4().hex, tools_path,
                                    "mcp__switchyard__probe")
        with _fake_model.FakeModel(script) as fake:
            done = _run(argv, fake, root, workdir)
        assert done.returncode == 0, done.stderr[-500:]
        tool_requests = fake.tool_requests()
        offered = {tuple(_fake_model.advertised_tools(r["body"])) for r in tool_requests}
        assert offered == {("mcp__switchyard__probe",)}, offered
        results = _fake_model.request_text(tool_requests[-1]["body"])
        for name in ("Agent", "AskUserQuestion", "Bash"):
            assert f"No such tool available: {name}" in results, (name, results[-800:])
        assert not marker.exists(), "the native Bash call ran in the sidecar"
        _assert_relay_stays_out(fake)
        print(f"  claude {_pinned_clis.installed_version('claude')} tool path: only "
              "mcp__switchyard__probe offered; Agent/AskUserQuestion/Bash refused; "
              "no login-dir content; OAuth intact")
    finally:
        server.PROVIDER, server.PROFILE = saved
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def test_claude_text_path_offers_no_tools():
    try:
        claude = _pinned_clis.require("claude")
    except _pinned_clis.Skip as why:
        print(f"  skipped: {why}")
        return
    cb = server.cli_bridge
    root = Path(tempfile.mkdtemp(prefix="lockdown-claude-text-"))
    _login_dir(root)
    saved = (cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE)
    try:
        cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE = (
            "claude", cb.PROFILES["claude"], claude, True)
        argv, _ = cb.build_argv("hello", None, "claude-sonnet-5")
        with _fake_model.FakeModel([{"tool": "Agent", "input": {}}]) as fake:
            done = _run(argv, fake, root, root)
        assert done.returncode == 0, done.stderr[-500:]
        assert fake.tool_requests() == [], [
            _fake_model.advertised_tools(r["body"]) for r in fake.tool_requests()]
        _assert_relay_stays_out(fake)
        print("  claude text path: zero tools offered, no login-dir content, OAuth intact")
    finally:
        cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE = saved
        import shutil
        shutil.rmtree(root, ignore_errors=True)


# 1x1 magenta PNG and a minimal one-page PDF: what the media test sends.
PNG_MAGENTA = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63f8cf1f7f060006000300013fe46dabe90000000049454e44ae426082")
PDF_MIN = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
           b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
           b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 72 72]>>endobj\n"
           b"trailer<</Root 1 0 R>>\n%%EOF\n")


def _media_blocks(fake: _fake_model.FakeModel) -> list[tuple[str, str]]:
    return [(b.get("type"), (b.get("source") or {}).get("media_type"))
            for req in fake.requests for m in req["body"].get("messages") or []
            if isinstance(m.get("content"), list)
            for b in m["content"] if b.get("type") in ("image", "document")]


def test_claude_media_reach_the_model_inline():
    """Both bridges hand Claude its images and PDFs as stream-json stdin blocks;
    the pinned CLI must pass them to the model as image/document blocks, offer
    no Read, and put no relay path in front of the model (issue #264)."""
    try:
        claude = _pinned_clis.require("claude")
    except _pinned_clis.Skip as why:
        print(f"  skipped: {why}")
        return
    import shutil
    cb = server.cli_bridge
    root = Path(tempfile.mkdtemp(prefix="lockdown-claude-media-"))
    _login_dir(root)
    stage = root / "swimg-test" / "img"
    stage.mkdir(parents=True)
    (stage / "01.png").write_bytes(PNG_MAGENTA)
    (stage / "02.pdf").write_bytes(PDF_MIN)
    media = [stage / "01.png", stage / "02.pdf"]
    saved = (cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE, server.PROVIDER, server.PROFILE)
    try:
        cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE = ("claude", cb.PROFILES["claude"], claude, True)
        argv, stdin_data = cb.build_argv("what is attached?", None, "claude-sonnet-5",
                                         image_paths=media)
        with _fake_model.FakeModel([]) as fake:
            done = _run(argv, fake, root, root, stdin_data)
        assert done.returncode == 0, done.stderr[-500:]
        assert cb.parse_output(done.stdout, "claude_json").get("type") == "result", done.stdout[-500:]
        assert _media_blocks(fake) == [("image", "image/png"), ("document", "application/pdf")], \
            _media_blocks(fake)
        assert fake.tool_requests() == [], "text path offered tools"
        assert str(stage) not in _fake_model.request_text(fake.requests[-1]["body"])
        _assert_relay_stays_out(fake)

        server.PROVIDER = "claude"
        server.PROFILE = dict(server.MCP_PROFILES["claude"], cli=claude)
        workdir = root / "session"
        workdir.mkdir()
        tools_path = workdir / "tools.json"
        tools_path.write_text(json.dumps([PROBE]))
        argv, stdin_data = server.build_argv("what is attached?", None, "claude-sonnet-5",
                                             workdir, uuid.uuid4().hex, tools_path,
                                             "mcp__switchyard__probe", media)
        with _fake_model.FakeModel([]) as fake:
            done = _run(argv, fake, root, workdir, stdin_data)
        assert done.returncode == 0, done.stderr[-500:]
        assert _media_blocks(fake) == [("image", "image/png"), ("document", "application/pdf")], \
            _media_blocks(fake)
        offered = {tuple(_fake_model.advertised_tools(r["body"])) for r in fake.tool_requests()}
        assert offered == {("mcp__switchyard__probe",)}, offered
        assert str(stage) not in _fake_model.request_text(fake.requests[-1]["body"])
        _assert_relay_stays_out(fake)
        print(f"  claude {_pinned_clis.installed_version('claude')}: image + PDF reach the "
              "model inline on both paths; no Read, no relay path")
    finally:
        cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE, server.PROVIDER, server.PROFILE = saved
        shutil.rmtree(root, ignore_errors=True)


def test_claude_web_request_offers_only_websearch_and_webfetch():
    """A caller's server-side web_search / web_fetch request adds Claude
    Code's own WebSearch and WebFetch (cli_bridge.claude_web_args), and
    nothing else: no Bash, no Read, no Agent."""
    try:
        claude = _pinned_clis.require("claude")
    except _pinned_clis.Skip as why:
        print(f"  skipped: {why}")
        return
    cb = server.cli_bridge
    root = Path(tempfile.mkdtemp(prefix="lockdown-claude-web-"))
    _login_dir(root)
    saved = (cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE)
    try:
        cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE = ("claude", cb.PROFILES["claude"], claude, True)
        argv, _ = cb.build_argv("latest python release?", None, "claude-sonnet-5", web=True)
        with _fake_model.FakeModel([{"text": "3.14"}]) as fake:
            done = _run(argv, fake, root, root)
        assert done.returncode == 0, done.stderr[-500:]
        offered = {name for r in fake.tool_requests()
                   for name in _fake_model.advertised_tools(r["body"])}
        assert offered == {"WebSearch", "WebFetch"}, offered
        _assert_relay_stays_out(fake)
        print("  claude web request: exactly WebSearch + WebFetch offered")
    finally:
        cb.PROVIDER, cb.PROFILE, cb.CLI, cb.BARE = saved
        import shutil
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
