"""Tests for switchyard/caller_env.py (issue #44).

The unit-tested surface covers parsers, the probe-id round-trip, the
command-tool recogniser, the renderers and the resolve() precedence
chain. The LiteLLM transport for `metadata.switchyard.caller_env` is
covered manually in TESTING.md section 8 -- the offline suite cannot
prove it.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)  # so `import conftest` resolves under plain `python3`

import conftest  # noqa: F401  (socket guard for plain-script mode)

from switchyard import caller_env  # noqa: E402


# ----------------------------------------------------------- dataclass shape ---
def test_caller_environment_unknown_is_completely_empty():
    """unknown() yields a CallerEnvironment with no fields and source=unknown.

    The whole point of the sentinel is that callers can check `env.source
    == "unknown"` without poking at each field. None / None / None / unknown.
    """
    env = caller_env.CallerEnvironment.unknown()
    assert env.cwd is None and env.platform is None and env.shell is None
    assert env.source == "unknown"
    print("  unknown(): cwd=None platform=None shell=None source=unknown")


# ------------------------------------------------------------- OpenCode parser ---
def test_parse_request_opencode_environment_block():
    """OpenCode puts <environment> in the system prompt with cwd/platform."""
    data = {"messages": [
        {"role": "system", "content": (
            "<environment>\n"
            "  <working_directory>/home/alice/proj</working_directory>\n"
            "  <workspace_root>/home/alice/proj</workspace_root>\n"
            "  <platform>linux</platform>\n"
            "</environment>\n"
            "You are a coding assistant."
        )},
    ]}
    env = caller_env.parse_request(data)
    assert env is not None, env
    assert env.cwd == "/home/alice/proj", env.cwd
    assert env.platform == "linux", env.platform
    assert env.shell is None
    assert env.source == "request", env.source
    print(f"  opencode prompt -> {env}")


# --------------------------------------------------------------- Codex parser ---
def test_parse_request_codex_environment_context():
    """Codex puts <environment_context> with cwd / shell / platform."""
    data = {"messages": [
        {"role": "user", "content": (
            "<environment_context>\n"
            "  <cwd>/Users/bob/codex-proj</cwd>\n"
            "  <shell>zsh</shell>\n"
            "  <platform>darwin</platform>\n"
            "</environment_context>\n"
            "Help me refactor this file."
        )},
    ]}
    env = caller_env.parse_request(data)
    assert env is not None
    assert env.cwd == "/Users/bob/codex-proj"
    assert env.shell == "zsh"
    assert env.platform == "darwin"
    assert env.source == "request"
    print(f"  codex first-turn -> {env}")


# ----------------------------------- Anthropic-protocol / generic env section ---
def test_parse_request_anthropic_style_env_block():
    """Anthropic-protocol requests may carry an <env> block; recognised as
    request source so it takes precedence over host and probe."""
    data = {"messages": [
        {"role": "system", "content": (
            "<env>\n"
            "  cwd=C:\\Users\\someone\\proj\n"
            "  platform=Windows\n"
            "  shell=powershell\n"
            "</env>"
        )},
    ]}
    env = caller_env.parse_request(data)
    assert env is not None
    assert env.cwd == r"C:\Users\someone\proj"
    assert env.platform == "Windows"
    assert env.shell == "powershell"
    assert env.source == "request"
    print(f"  anthropic <env> -> {env}")


def test_parse_request_generic_first_turn_env_section():
    """A first-turn labelled block without a known tag is still parsed."""
    data = {"messages": [
        {"role": "user", "content": (
            "Hi, here's my context:\n"
            "cwd: /tmp/work\n"
            "platform: freebsd\n"
            "shell: tcsh\n"
            "now please help me with..."
        )},
    ]}
    env = caller_env.parse_request(data)
    assert env is not None
    assert env.cwd == "/tmp/work"
    assert env.platform == "freebsd"
    assert env.shell == "tcsh"
    assert env.source == "request"
    print(f"  generic first-turn -> {env}")


# ------------------------------------------- Claude Code sends no environment ---
def test_parse_request_claude_code_no_env_returns_none():
    """A Claude Code request carries no environment block at all -- parse_request
    returns None so the resolver falls through to probe or unknown."""
    data = {"messages": [
        {"role": "system", "content": (
            "You are Claude Code. Help the user with their code.\n"
            "Tools are available; use them when relevant."
        )},
        {"role": "user", "content": "Hi"},
    ]}
    env = caller_env.parse_request(data)
    assert env is None, env
    print("  Claude Code (no env block) -> parse_request returns None")


# ----------------------------------------------------------- relay-path shield ---
def test_parse_request_drops_relay_looking_cwd():
    """A parsed cwd that names the relay container is the bug we are avoiding
    -- not the answer. The OpenCode parser strips it; the env reports the
    rest of the values (if any) and the cwd is dropped."""
    data = {"messages": [
        {"role": "system", "content": (
            "<environment>\n"
            "  <working_directory>/app/mcp_bridge</working_directory>\n"
            "  <platform>linux</platform>\n"
            "</environment>"
        )},
    ]}
    env = caller_env.parse_request(data)
    assert env is not None
    assert env.cwd is None, env.cwd                # /app/... is the relay, dropped
    assert env.platform == "linux"
    print(f"  relay-path cwd dropped: cwd=None platform={env.platform!r}")


# ------------------------------------------------------- precedence matrix ---
class _StubCfg:
    """A duck-typed stand-in for CallerEnvironmentSettings (resolve() only
    reads the attributes)."""
    def __init__(self, probe="auto", platform=None, cwd=None,
                 shell=None, fallback_platform=None):
        self.probe = probe
        self.platform = platform
        self.cwd = cwd
        self.shell = shell
        self.fallback_platform = fallback_platform


def test_resolve_config_beats_all():
    """Forced config values win over request and probe and host."""
    cfg = _StubCfg(platform="windows", cwd="C:\\x", shell="powershell")
    request_data = {"messages": [
        {"role": "system", "content": (
            "<environment>\n  <working_directory>/home/eve</working_directory>\n"
            "  <platform>linux</platform>\n</environment>"
        )},
    ]}
    env = caller_env.resolve(
        request_data, cfg,
        probe_result=caller_env.CallerEnvironment(platform="macos", source="probe"),
    )
    assert env.source == "config"
    assert env.platform == "windows"
    assert env.cwd == r"C:\x"
    assert env.shell == "powershell"
    print(f"  config beats probe+request -> {env}")


def test_resolve_request_beats_probe():
    """A request that names its own environment wins over a probe result."""
    cfg = _StubCfg()      # no overrides
    request_data = {"messages": [
        {"role": "system", "content": (
            "<environment>\n  <working_directory>/u/alice</working_directory>\n"
            "  <platform>linux</platform>\n</environment>"
        )},
    ]}
    probe_env = caller_env.CallerEnvironment(
        platform="windows", cwd="C:\\Users\\Alice", source="probe")
    env = caller_env.resolve(request_data, cfg, probe_result=probe_env)
    assert env.source == "request"
    assert env.platform == "linux"
    assert env.cwd == "/u/alice"
    print(f"  request beats probe -> {env}")


def test_resolve_probe_beats_host():
    """When neither config nor request yields a value, probe wins over host."""
    cfg = _StubCfg(fallback_platform="macos")
    probe_env = caller_env.CallerEnvironment(platform="windows", source="probe")
    env = caller_env.resolve({}, cfg, probe_result=probe_env)
    assert env.source == "probe"
    assert env.platform == "windows"
    print(f"  probe beats host -> {env}")


def test_resolve_host_fallback_only_platform_never_cwd():
    """Host fallback is platform ONLY. The docker host's working directory
    has no reliable relationship to the caller's, so the fallback NEVER
    carries a cwd or a shell -- only the platform itself. (The `host_hint`
    parameter this test replaced was dead code; the cwd-suppression
    property is enforced by the fallback branch hard-coding cwd=None,
    not by anything threading the host through.)"""
    cfg = _StubCfg(fallback_platform="macos")
    env = caller_env.resolve({}, cfg)
    assert env.source == "host"
    assert env.platform == "macos"
    assert env.cwd is None, env.cwd
    assert env.shell is None, env.shell
    print(f"  host fallback: platform only, cwd+shell always None -> {env}")


def test_resolve_drops_relay_looking_fallback_platform():
    """A `fallback_platform` that itself names the relay container cannot
    be honored -- otherwise a Linux /app host-side misconfiguration would
    silently become the caller's environment, the exact bug the change
    exists to prevent. The relay-path guard `_is_relay_path` catches
    `/app/mcp_bridge`, `/app/cli_bridge`, `/tmp/mcpb-*` and similar.
    Without a valid fallback_platform AND without a request body env,
    resolution falls through to `unknown`."""
    cfg = _StubCfg(fallback_platform="/app/mcp_bridge")    # relay path
    env = caller_env.resolve({}, cfg)
    assert env.source == "unknown"
    assert env.platform is None
    assert env.cwd is None
    print(f"  relay-path fallback_platform rejected -> {env}")


def test_resolve_required_raises_when_unresolvable():
    """`probe: required` must RAISE CallerEnvironmentRequired instead of
    falling through to `unknown`. The operator opted in to loud refusal;
    returning `unknown` would defeat the whole point of the option.

    The bridges translate the exception to an HTTPException. The text
    path downgrades `required` to `auto` BEFORE calling resolve, so
    this exception only fires on the mcp path -- but the resolver's
    contract is the same regardless of caller."""
    cfg = _StubCfg(probe="required")
    try:
        caller_env.resolve({}, cfg)
    except caller_env.CallerEnvironmentRequired as exc:
        # The message names the operator's chosen behaviour and tells
        # them how to unblock: forced values, passive env, or auto mode.
        assert "probe=required" in str(exc), exc
        assert "refuse" in str(exc), exc
        print(f"  probe=required + unresolvable -> raised: {str(exc)[:60]}...")
        return
    raise AssertionError("probe=required must raise, not return unknown")


def test_resolve_required_does_not_raise_when_forced_config_present():
    """`probe: required` is about LOUDNESS when unresolvable; it does not
    change the precedence chain. A forced config value (the very thing
    required-mode is asking for) still wins and returns source=config
    without raising."""
    cfg = _StubCfg(probe="required", platform="windows", cwd=r"C:\x",
                   shell="powershell")
    env = caller_env.resolve({}, cfg)
    assert env.source == "config"
    assert env.platform == "windows"
    assert env.cwd == r"C:\x"
    assert env.shell == "powershell"
    print("  probe=required + forced config -> no raise, source=config")


def test_resolve_required_does_not_raise_when_request_has_env():
    """`probe: required` + a request that already carries its env (a
    passive OpenCode-style block, say) is fully resolved. Required-mode
    only refuses when NOTHING yields an env."""
    cfg = _StubCfg(probe="required")
    request_data = {"messages": [
        {"role": "system", "content": (
            "<environment>\n"
            "  <working_directory>/u/alice/proj</working_directory>\n"
            "  <platform>linux</platform>\n"
            "</environment>"
        )},
    ]}
    env = caller_env.resolve(request_data, cfg)
    assert env.source == "request"
    assert env.platform == "linux"
    assert env.cwd == "/u/alice/proj"
    print("  probe=required + passive env -> no raise, source=request")


def test_from_wire_metadata_rejects_config_claim():
    """The wire is caller-controlled; only the operator's plans.yaml
    may produce `source=config`. A metadata stamp claiming `config` is
    re-labeled to `request` so the precedence chain is honest: the
    values still flow through, but at the request tier, not the config
    tier. Without this, a caller who reached the sidecar directly
    could bypass the precedence chain by stamping `source=config`
    on a fake Linux /app env -- the relay-env-as-caller-env bug
    the whole change exists to prevent."""
    rejected = caller_env.from_wire_metadata({
        "platform": "linux", "cwd": "/app/mcp_bridge", "shell": "/bin/sh",
        "source": "config"})
    assert rejected is not None
    assert rejected.source == "request"   # re-labeled, never honored as-is
    # The values still flow through (we are not losing data, just
    # preventing the precedence bypass).
    assert rejected.platform == "linux"
    assert rejected.cwd == "/app/mcp_bridge"
    assert rejected.shell == "/bin/sh"
    print(f"  metadata claims source=config -> re-labeled to {rejected.source}")


def test_from_wire_metadata_passes_through_non_config_sources():
    """Non-`config` claims (`request`, `probe`, `host`) flow through as-is.
    The re-labeling is targeted, not blanket."""
    passthrough = caller_env.from_wire_metadata({
        "platform": "windows", "cwd": r"C:\Users\x", "shell": "powershell",
        "source": "request"})
    assert passthrough is not None
    assert passthrough.source == "request"
    assert passthrough.platform == "windows"
    print(f"  metadata source=request -> honored as {passthrough.source}")


def test_from_wire_metadata_rejects_malformed_input():
    """Non-dicts, empty dicts, dicts with no usable fields, and dicts with
    no usable source marker all return None -- the caller falls back to
    `parse_request(body)`. This matches the pre-round-1 strictness: a
    stamp without an explicit source label does NOT preempt passive
    detection."""
    assert caller_env.from_wire_metadata(None) is None
    assert caller_env.from_wire_metadata("not a dict") is None
    assert caller_env.from_wire_metadata([]) is None
    assert caller_env.from_wire_metadata({}) is None
    assert caller_env.from_wire_metadata({"source": "request"}) is None
    # No source field at all: falls through to passive parse.
    assert caller_env.from_wire_metadata({"platform": "linux", "cwd": "/x"}) \
        is None
    # Explicit source="unknown" is a sentinel for "I don't know"; treat
    # it as absent -- the caller falls through to passive parse.
    assert caller_env.from_wire_metadata(
        {"source": "unknown", "platform": "linux"}) is None
    # Unknown source label: also treated as absent.
    assert caller_env.from_wire_metadata(
        {"source": "made_up", "platform": "linux"}) is None
    print("  malformed/no-source metadata -> None; passive parse picks up")


def test_from_wire_metadata_relabels_host_to_request():
    """Round-2 should-fix: `source=host` is also plans.yaml-derived (the
    `fallback_platform` branch of `resolve()` produces it), so a caller
    who stamps `source=host` to bypass the precedence chain gets the
    request tier instead -- same defensive re-labeling as `source=config`.
    """
    rejected = caller_env.from_wire_metadata({
        "platform": "macos", "cwd": "/Users/x", "shell": "zsh",
        "source": "host"})
    assert rejected is not None
    assert rejected.source == "request", \
        f"metadata-stamped source=host must be re-labeled to request, " \
        f"got {rejected.source!r}"
    # Values still flow through (we don't drop data, we just prevent
    # the precedence bypass).
    assert rejected.platform == "macos"
    assert rejected.cwd == "/Users/x"
    assert rejected.shell == "zsh"
    print(f"  metadata claims source=host -> re-labeled to "
          f"{rejected.source!r}, values still flow through")


def test_from_wire_metadata_honors_probe_source():
    """`source=probe` is intentionally NOT re-labeled -- the only
    legitimate minter is the sidecar's own `_consume_probe_results`,
    which stamps source=probe on a freshly parsed env. A wire forgery
    of probe is technically possible but not useful (the attacker
    would have to make a real tool round-trip succeed to fake the
    parsed env). Leave probe alone."""
    parsed = caller_env.from_wire_metadata({
        "platform": "linux", "cwd": "/home/u/proj", "shell": "/bin/zsh",
        "source": "probe"})
    assert parsed is not None
    assert parsed.source == "probe"
    assert parsed.platform == "linux"
    print(f"  metadata source=probe -> honored as {parsed.source!r}")


def test_resolve_unknown_when_nothing_yields():
    """No config, no request, no probe, no host -> unknown."""
    cfg = _StubCfg()
    env = caller_env.resolve({}, cfg)
    assert env.source == "unknown"
    assert env.cwd is None and env.platform is None and env.shell is None
    print(f"  unknown path -> {env}")


# ----------------------------------------------------------- probe id mechanics ---
def test_probe_id_mint_and_parse_round_trip():
    """mint_probe_call_id + parse_probe_call_id are perfect inverses."""
    fp = caller_env.fingerprint([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ])
    cid = caller_env.mint_probe_call_id(fp)
    assert cid.startswith(caller_env.PROBE_PREFIX)
    assert caller_env.parse_probe_call_id(cid) == fp
    assert len(fp) == 16
    print(f"  {cid!r} -> fp={fp}")


def test_probe_id_round_trip_is_restart_safe():
    """parse_probe_call_id needs no in-memory state. A new instance with
    nothing else loaded still recognises an id minted by a previous run."""
    cid = caller_env.mint_probe_call_id("0123456789abcdef")
    # Wipe caller_env's module state (none, but the point is we read
    # nothing internal). Verify the parse with a fresh function call.
    assert caller_env.parse_probe_call_id(cid) == "0123456789abcdef"
    # Idempotent on the parse: calling it twice returns the same value.
    assert caller_env.parse_probe_call_id(cid) == caller_env.parse_probe_call_id(cid)
    print("  restart-safe probe id parse (no in-memory state needed)")


def test_probe_id_parse_rejects_non_probe_ids():
    """Anything that is not a switchyard_env_<16hex> is rejected."""
    assert caller_env.parse_probe_call_id(None) is None
    assert caller_env.parse_probe_call_id("") is None
    assert caller_env.parse_probe_call_id("call_abc_1") is None
    assert caller_env.parse_probe_call_id("switchyard_env_short") is None
    assert caller_env.parse_probe_call_id("switchyard_env_zzzzzzzzzzzzzzzz") is None
    print("  non-probe ids / wrong-length / non-hex all rejected")


def test_probe_id_fingerprint_is_stable_across_turns():
    """Two requests that share the same first two messages get the same fp.

    A Claude Code session sends the same system prompt and roughly the
    same first user turn across retries; the probe id MUST be the same
    so a retry of the same probe is correlated, not correlated with a
    different session.
    """
    a = caller_env.fingerprint([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ])
    b = caller_env.fingerprint([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ])
    c = caller_env.fingerprint([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "different"},
    ])
    assert a == b
    assert a != c
    print(f"  same prefix -> same fp ({a}); different prefix -> different ({c})")


def test_probe_command_contains_no_env_enumeration():
    """The probe command MUST NOT call env / printenv / set -- both dump
    the caller's environment which carries whatever secrets the caller's
    terminal session holds. Allow only pwd, uname -s, and a single
    allowlisted variable ($SHELL via echo).
    """
    banned = ["env", "printenv", "set", "export", "declare"]
    for word in banned:
        # Word-boundary check, not substring: 'environment' must not match.
        assert f" {word} " not in caller_env.PROBE_COMMAND and \
               not caller_env.PROBE_COMMAND.startswith(word + " ") and \
               not caller_env.PROBE_COMMAND.endswith(" " + word), \
            f"PROBE_COMMAND must not call {word!r}: {caller_env.PROBE_COMMAND!r}"
    assert "pwd" in caller_env.PROBE_COMMAND
    assert "uname" in caller_env.PROBE_COMMAND
    assert "$SHELL" in caller_env.PROBE_COMMAND
    print("  PROBE_COMMAND only enumerates: pwd / uname / $SHELL")


def test_parse_probe_result_tolerates_shell_quoting_and_extra_lines():
    """Real shells wrap values in quotes and add stray lines -- the parser
    must still find the labels and recover cwd / platform / shell."""
    content = (
        "bash: warning: could not find any active jobs\n"
        "cwd=/home/user/proj\n"
        "platform=Linux\n"
        "shell=/bin/bash\n"
    )
    env = caller_env.parse_probe_result(content)
    assert env is not None
    assert env.cwd == "/home/user/proj"
    assert env.platform == "Linux"
    assert env.shell == "/bin/bash"
    print(f"  probe stdout -> {env}")


def test_parse_probe_result_drops_relay_looking_cwd():
    """A probe that returns the relay container's own cwd is the bug we
    are avoiding; the parser drops it the same way parse_request does."""
    content = (
        "cwd=/app/mcp_bridge/something\n"
        "platform=Linux\n"
        "shell=/bin/sh\n"
    )
    env = caller_env.parse_probe_result(content)
    assert env is not None
    assert env.cwd is None
    assert env.platform == "Linux"
    print(f"  relay-cwd probe -> cwd dropped: {env}")


# -------------------------------------------------- command-tool recognition ---
def test_find_command_tool_claude_bash_like():
    """A Claude-style `Bash` tool with a required `command` parameter."""
    tools = [{"type": "function", "function": {
        "name": "Bash", "description": "Run shell commands",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}]
    hit = caller_env.find_command_tool(tools)
    assert hit is not None
    assert hit == ("Bash", "command"), hit
    print(f"  claude Bash -> {hit}")


def test_find_command_tool_opencode_bash():
    """OpenCode shell uses `command` too -- the same schema."""
    tools = [{"type": "function", "function": {
        "name": "shell", "description": "",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}]
    hit = caller_env.find_command_tool(tools)
    assert hit == ("shell", "command"), hit
    print(f"  opencode shell -> {hit}")


def test_find_command_tool_generic_command_name():
    """Any name works as long as the schema has a string command field."""
    tools = [{"type": "function", "function": {
        "name": "run_command", "description": "",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}]
    hit = caller_env.find_command_tool(tools)
    assert hit == ("run_command", "command"), hit
    print(f"  generic run_command -> {hit}")


def test_find_command_tool_powershell_compatible():
    """A PowerShell-compatible tool with `script` or `cmd` arg is also
    recognised; the schema-driven matcher is provider-agnostic."""
    tools = [{"type": "function", "function": {
        "name": "pwsh", "description": "",
        "parameters": {"type": "object",
                       "properties": {"script": {"type": "string"}},
                       "required": ["script"]}}}]
    hit = caller_env.find_command_tool(tools)
    assert hit is not None
    assert hit[1] in ("command", "cmd", "script", "shell_command"), hit
    print(f"  pwsh-style -> {hit}")


def test_find_command_tool_rejects_get_weather_with_bad_schema():
    """`get_weather` with no command-shaped field is not a shell tool, no
    matter what name it has. The schema-driven matcher is the whole point
    of the rec -- a name-based one would have to special-case `Bash`
    against every other possible name, and that always breaks the moment
    someone adds a new tool."""
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Look up the weather",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    hit = caller_env.find_command_tool(tools)
    assert hit is None, hit
    print("  get_weather (no command-shaped field) -> None")


def test_find_command_tool_rejects_bash_with_bad_schema():
    """The bug a name-based matcher would fall for: a tool NAMED `Bash`
    but with a non-command-shaped parameter (e.g. only a `pattern`
    string). It is not a shell tool -- do not recognise it."""
    tools = [{"type": "function", "function": {
        "name": "Bash", "description": "Not actually a shell tool",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"}},
                       "required": ["pattern"]}}}]
    hit = caller_env.find_command_tool(tools)
    assert hit is None, hit
    print("  Bash with non-command schema -> None (no name-based match)")


def test_find_command_tool_handles_bare_function_shape():
    """Some clients send the bare function shape without the `type: function`
    wrapper. The match still works."""
    tools = [
        {
            "name": "run",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        }
    ]
    hit = caller_env.find_command_tool(tools)
    assert hit is not None, hit
    assert hit == ("run", "command"), hit
    print(f"  bare-function-shape -> {hit}")





# ----------------------------------------------------------- render blocks ---
def test_render_system_block_known_env():
    """A resolved env renders with the literal field values."""
    env = caller_env.CallerEnvironment(
        cwd=r"C:\Users\demo", platform="windows", shell="powershell",
        source="request")
    block = caller_env.render_system_block(env)
    assert "Caller platform: windows" in block
    assert r"Caller working directory: C:\Users\demo" in block
    assert "Caller shell: powershell" in block
    assert "[SwitchYard tool execution environment]" in block
    assert "relay only" in block
    assert "authoritative" in block
    print(f"  render known env: {block.splitlines()[0]!r}")


def test_render_system_block_unknown_env_uses_unknown_wording():
    """Unknown fields render as the literal string 'unknown' so the model
    sees 'Caller platform: unknown' rather than 'Caller platform: None'."""
    env = caller_env.CallerEnvironment(cwd=None, platform=None,
                                        shell=None, source="unknown")
    block = caller_env.render_system_block(env)
    assert "Caller platform: unknown" in block
    assert "Caller working directory: unknown" in block
    assert "Caller shell: unknown" in block
    # NEVER substitutes the relay's environment as a fallback -- that's the
    # whole bug we are avoiding.
    assert "Linux" not in block
    assert "/app" not in block
    print("  render unknown env: all fields 'unknown', no relay leakage")


def test_render_first_turn_reminder_known():
    """A known env produces the owner-template one-liner with platform+cwd."""
    env = caller_env.CallerEnvironment(
        cwd=r"C:\src\proj", platform="Windows", shell="powershell",
        source="request")
    reminder = caller_env.render_first_turn_reminder(env)
    assert "[SwitchYard: tools execute on Windows in C:\\src\\proj" in reminder
    assert "/app" in reminder            # relay path warning is part of the line
    assert "must not be used for tool paths" in reminder
    print(f"  reminder known: {reminder[:60]}...")


def test_render_first_turn_reminder_unknown_uses_own_wording():
    """An unknown env produces the explicit 'do not assume the relay is
    the caller' wording from the OWNER's template."""
    env = caller_env.CallerEnvironment(cwd=None, platform=None,
                                        shell=None, source="unknown")
    reminder = caller_env.render_first_turn_reminder(env)
    assert "caller environment unknown" in reminder
    assert "do not assume the relay container is the caller machine" in reminder
    # Issue #256: a headless relay has no one to answer a question, so the
    # reminder must not tell the model to ask one.
    assert "ask if needed" not in reminder
    assert "instead of asking" in reminder
    print(f"  reminder unknown: {reminder[:60]}...")


def test_render_functions_are_pure_no_environment_substitution():
    """The renderers must NEVER substitute the running process's environment
    for an unknown field. Linux / /app leakage would be the bug."""
    env = caller_env.CallerEnvironment.unknown()
    block = caller_env.render_system_block(env)
    reminder = caller_env.render_first_turn_reminder(env)
    # Linux / /app would only appear in the reminder if we substituted the
    # host env. The system block says "unknown", the reminder says
    # "caller environment unknown".
    assert "/app/" not in block
    assert "/app/" not in reminder
    print("  render functions are pure (no host-env leakage)")



# ---------------------------------- issue #187: real caller prompt formats ---
CLAUDE_CODE_ENV_SECTION = (
    "You are Claude Code.\n\n# Environment\n"
    "You have been invoked in the following environment: \n"
    " - Primary working directory: /Users/me/Documents/GitHub/proj\n"
    " - Is a git repository: true\n"
    " - Platform: darwin\n"
    " - Shell: zsh\n"
    " - OS Version: Darwin 27.0.0\n")

OPENCODE_ENV_BLOCK = (
    "You are opencode.\n"
    "Here is some useful information about the environment you are running in:\n"
    "<env>\n"
    "  Working directory: /home/oranode/orca/workspaces/SwitchYard/x\n"
    "  Workspace root folder: /home/oranode/orca/workspaces/SwitchYard\n"
    "  Is directory a git repo: yes\n"
    "  Platform: linux\n"
    "  Today's date: Thu Sep 24 2026\n"
    "</env>\n")


def test_parse_claude_code_environment_section():
    """Claude Code's system prompt lists its environment as bulleted
    `- Label: value` lines under `# Environment`. All three fields must
    come through -- the old parser found none of them."""
    env = caller_env.parse_request({"messages": [
        {"role": "system", "content": CLAUDE_CODE_ENV_SECTION},
        {"role": "user", "content": "hi"}]})
    assert env is not None
    assert env.cwd == "/Users/me/Documents/GitHub/proj", env
    assert env.platform == "darwin", env
    assert env.shell == "zsh", env
    print(f"  Claude Code # Environment -> {env}")


def test_parse_top_level_anthropic_system():
    """On /v1/messages the gateway sees Claude Code's system prompt as the
    top-level `system` (string or block list), not a message."""
    for system in (CLAUDE_CODE_ENV_SECTION,
                   [{"type": "text", "text": "preamble"},
                    {"type": "text", "text": CLAUDE_CODE_ENV_SECTION}]):
        env = caller_env.parse_request({
            "system": system, "messages": [{"role": "user", "content": "hi"}]})
        assert env is not None and env.cwd == "/Users/me/Documents/GitHub/proj", env
    print("  top-level system (string and block list) parsed")


def test_parse_opencode_env_block_keeps_cwd():
    """OpenCode's `<env>` block says `Working directory:`. An earlier parser
    matched its platform and returned before anything read the cwd, so the
    tool path ran "in unknown" (issue #255's vacuous reminder)."""
    env = caller_env.parse_request({"messages": [
        {"role": "system", "content": OPENCODE_ENV_BLOCK},
        {"role": "user", "content": "hi"}]})
    assert env is not None
    assert env.cwd == "/home/oranode/orca/workspaces/SwitchYard/x", env
    assert env.platform == "linux", env
    print(f"  OpenCode <env> -> {env}")


def test_parsers_merge_field_by_field():
    """One parser may know the platform and another the cwd; the result
    carries both instead of whichever parser matched first."""
    # The OpenCode-style parser matches `<environment>` and finds only the
    # platform; the cwd is in a labelled line only a later parser reads.
    text = ("<environment>\n  <platform>windows</platform>\n</environment>\n"
            "Working directory: C:/Users/me/proj\nShell: pwsh\n")
    env = caller_env.parse_request({"messages": [{"role": "system", "content": text}]})
    assert (env.cwd, env.platform, env.shell) == ("C:/Users/me/proj", "windows", "pwsh"), env
    print(f"  merged: {env}")


def test_real_paths_containing_app_are_not_relay_paths():
    """Relay paths are the relay's own directories, matched as prefixes.
    A substring match on "/app/" discarded real caller paths."""
    for real in ("/home/u/myapp/app/src", "/Users/me/app/web", "/srv/app/x"):
        assert not caller_env._is_relay_path(real), real
    for relay in ("/app/mcp_bridge", "/app/cli_bridge/x", "/tmp/mcpb-1234-ab",
                  "/tmp/sy-cli-xyz", "/tmp/switchyard-relay/a"):
        assert caller_env._is_relay_path(relay), relay
    env = caller_env.parse_request({"messages": [{"role": "system", "content":
        "# Environment\n - Primary working directory: /Users/me/app/web\n"}]})
    assert env is not None and env.cwd == "/Users/me/app/web", env
    print("  /app/ inside a real path kept; relay prefixes still rejected")


def test_relay_cwd_from_one_parser_is_rescued_by_a_later_one():
    """The merge strips a relay-looking cwd per parser and keeps going, so a
    later parser can still supply the caller's real cwd. Previously the
    first parser won outright and its stripped cwd was final."""
    text = ("<environment>\n  <working_directory>/app/mcp_bridge</working_directory>\n"
            "  <platform>linux</platform>\n</environment>\n"
            "<environment_context>\n  <cwd>/home/u/proj</cwd>\n  <shell>zsh</shell>\n"
            "</environment_context>\n")
    env = caller_env.parse_request({"messages": [{"role": "system", "content": text}]})
    assert (env.cwd, env.platform, env.shell) == ("/home/u/proj", "linux", "zsh"), env
    print(f"  relay cwd stripped, real cwd rescued: {env}")
def test_parse_git_flag_from_claude_code_and_opencode():
    """The caller's prompt says whether its cwd is a repository; the relay's
    mirror of that directory is `git init`ed to match."""
    cc = caller_env.parse_request({"messages": [
        {"role": "system", "content": CLAUDE_CODE_ENV_SECTION}]})
    oc = caller_env.parse_request({"messages": [
        {"role": "system", "content": OPENCODE_ENV_BLOCK}]})
    no = caller_env.parse_request({"messages": [{"role": "system", "content":
        "<env>\nWorking directory: /x\nIs directory a git repo: no\n</env>"}]})
    bare = caller_env.parse_request({"messages": [{"role": "system", "content":
        "# Environment\n - Primary working directory: /x\n"}]})
    assert (cc.git, oc.git, no.git, bare.git) == (True, True, False, None)
    print("  git flag: true / yes / no / absent")


# ------------------------------------------------------ Windows callers ---
# Claude Code 2.1.278's environment section as it renders on Windows
# (process.platform win32, cwd in Windows form, the shell description it
# gives when its PowerShell tool is on), with CRLF line ends.
CLAUDE_CODE_WINDOWS_SECTION = (
    "# Environment\r\nYou have been invoked in the following environment: \r\n"
    " - Primary working directory: C:\\Users\\Me\\proj\r\n"
    " - Is a git repository: true\r\n"
    " - Platform: win32\r\n"
    " - Shell: PowerShell (primary); Bash tool also available for POSIX scripts"
    " \u2014 each takes its own syntax.\r\n"
    " - OS Version: Windows 11 Pro 10.0.22631\r\n")


def _env_of(text: str):
    return caller_env.parse_request({"messages": [{"role": "system", "content": text},
                                                  {"role": "user", "content": "hi"}]})


def test_parse_request_windows_claude_code_section():
    env = _env_of(CLAUDE_CODE_WINDOWS_SECTION)
    assert env.cwd == "C:\\Users\\Me\\proj", env
    assert env.platform == "win32" and env.git is True, env
    assert env.shell.startswith("PowerShell (primary)"), env
    print(f"  Claude Code on Windows (CRLF): {env.cwd} / {env.platform} / git")


def test_parse_request_windows_opencode_codex_and_path_forms():
    """OpenCode's <env> (forward-slash drive path), codex's
    environment_context (powershell), a UNC share and a Git Bash path all
    parse with the path exactly as the caller wrote it."""
    oc = _env_of("<env>\n  Working directory: C:/Users/Me/proj\n"
                 "  Workspace root folder: C:/Users/Me/proj\n"
                 "  Is directory a git repo: yes\n  Platform: win32\n</env>")
    assert (oc.cwd, oc.platform, oc.git) == ("C:/Users/Me/proj", "win32", True), oc
    cx = _env_of("<environment_context>\n  <cwd>C:\\Users\\Me\\proj</cwd>\n"
                 "  <shell>powershell</shell>\n</environment_context>")
    assert (cx.cwd, cx.shell) == ("C:\\Users\\Me\\proj", "powershell"), cx
    unc = _env_of("<env>\nWorking directory: \\\\server\\share\\proj\nPlatform: win32\n</env>")
    assert unc.cwd == "\\\\server\\share\\proj", unc
    gb = _env_of("# Environment\n - Primary working directory: /c/Users/Me/proj\n"
                 " - Platform: win32\n - Shell: bash\n")
    assert (gb.cwd, gb.shell) == ("/c/Users/Me/proj", "bash"), gb
    print("  OpenCode C:/..., codex powershell, UNC and /c/... all parse verbatim")


def test_parse_request_cline_system_information():
    """Cline-family agents (OpenAI-compatible, PowerShell on Windows) state
    their environment in a SYSTEM INFORMATION section; it is read passively
    instead of costing a probe."""
    text = ("You are an agent.\n\n====\n\nSYSTEM INFORMATION\n\n"
            "Operating System: Windows 11\n"
            "Default Shell: C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\n"
            "Home Directory: C:/Users/me\n"
            "Current Working Directory: c:/Users/me/proj\n\n====\n")
    env = _env_of(text)
    assert env.cwd == "c:/Users/me/proj", env
    assert env.platform == "Windows 11", env
    assert env.shell.endswith("powershell.exe"), env
    roo = _env_of("SYSTEM INFORMATION\n\nOperating System: macOS Sonoma\n"
                  "Default Shell: /bin/zsh\nCurrent Workspace Directory: /Users/me/proj\n")
    assert (roo.cwd, roo.shell) == ("/Users/me/proj", "/bin/zsh"), roo
    print("  Cline/Roo SYSTEM INFORMATION: cwd, OS and default shell read")


def test_is_windows_platform():
    for yes in ("win32", "windows", "Windows", "Windows_NT", "Windows 11 Pro 10.0"):
        assert caller_env.is_windows_platform(yes), yes
    for no in (None, "", "darwin", "linux", "Darwin", "MINGW64_NT-10.0-22631",
               "MSYS_NT-10.0", "CYGWIN_NT-10.0"):
        assert not caller_env.is_windows_platform(no), no
    print("  win32/windows/Windows_NT are Windows; MINGW/MSYS/Cygwin are not")


def _cmd_tool(name, description="", key="command"):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {key: {"type": "string"}},
                       "required": [key]}}}


def test_find_probe_tool_writes_the_probe_for_the_tools_shell():
    """The probe must be one the caller's shell can run: PowerShell and cmd
    have no printf/uname and cmd has no $(...)."""
    posix, ps, cmd = (caller_env.PROBE_COMMAND, caller_env.PROBE_COMMAND_POWERSHELL,
                      caller_env.PROBE_COMMAND_CMD)
    cases = [
        # (tool, platform hint, expected probe)
        (_cmd_tool("Bash"), None, posix),
        (_cmd_tool("Bash", "Runs in Git Bash. Prefer the PowerShell tool."), "win32", posix),
        (_cmd_tool("run_bash"), "windows", posix),
        (_cmd_tool("PowerShell"), None, ps),
        (_cmd_tool("pwsh", key="script"), None, ps),
        (_cmd_tool("execute_command", "Runs the command in PowerShell."), None, ps),
        (_cmd_tool("execute_command", "Runs the command with cmd.exe."), None, cmd),
        (_cmd_tool("cmd"), "win32", cmd),
        (_cmd_tool("cmd"), None, cmd),
        (_cmd_tool("CMD", "Run a command"), "win32", cmd),
        (_cmd_tool("cmd_exe"), None, cmd),
        (_cmd_tool("command_prompt"), None, cmd),
        # `cmd` as shorthand for "command" names no shell.
        (_cmd_tool("run_cmd"), "win32", ps),
        (_cmd_tool("exec_cmd"), None, posix),
        (_cmd_tool("execute_command", "Runs the command in bash."), "win32", posix),
        (_cmd_tool("execute_command"), "win32", ps),
        (_cmd_tool("execute_command"), "Windows 11", ps),
        (_cmd_tool("execute_command"), None, posix),
        (_cmd_tool("execute_command"), "darwin", posix),
        (_cmd_tool("execute_command"), "MINGW64_NT-10.0", posix),
        # Names more than one shell: the platform decides.
        (_cmd_tool("shell", "bash on Linux, PowerShell on Windows"), "win32", ps),
        (_cmd_tool("shell", "bash on Linux, PowerShell on Windows"), "linux", posix),
    ]
    for tool, platform, want in cases:
        name, key, command = caller_env.find_probe_tool([tool], platform)
        assert name == tool["function"]["name"], (name, tool)
        assert key in ("command", "script"), key
        assert command == want, (tool["function"], platform, command)
    assert caller_env.find_probe_tool([], "win32") is None
    assert caller_env.find_probe_tool([{"type": "function", "function": {
        "name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}}}}], "win32") is None
    print(f"  {len(cases)} tool/platform combinations -> posix/powershell/cmd probe")


def test_posix_probe_is_byte_identical():
    """The POSIX probe every non-Windows caller gets is unchanged."""
    assert caller_env.PROBE_COMMAND == (
        "printf 'cwd=%s\\nplatform=%s\\nshell=%s\\n' \"$(pwd)\" \"$(uname -s)\" \"$SHELL\"")
    assert caller_env.find_probe_tool([_cmd_tool("Bash")])[2] == caller_env.PROBE_COMMAND
    for command in caller_env.PROBE_COMMANDS.values():
        for word in ("env", "printenv", "set", "Get-ChildItem", "gci", "dir"):
            assert f" {word} " not in f" {command} ", (word, command)
    print("  POSIX probe unchanged; no probe enumerates the environment")


def test_parse_probe_result_windows_outputs():
    """PowerShell and cmd answers (CRLF) parse; cmd's UNC fallback to
    C:\\Windows is not taken for the caller's cwd."""
    ps = caller_env.parse_probe_result(
        "cwd=C:\\Users\\Me\\proj\r\nplatform=Windows\r\nshell=powershell\r\n")
    assert (ps.cwd, ps.platform, ps.shell) == ("C:\\Users\\Me\\proj", "Windows",
                                               "powershell"), ps
    cmd = caller_env.parse_probe_result("shell=cmd\r\nplatform=Windows\r\ncwd=C:\\work\r\n")
    assert (cmd.cwd, cmd.platform, cmd.shell) == ("C:\\work", "Windows", "cmd"), cmd
    unc = caller_env.parse_probe_result(
        "'\\\\server\\share\\proj'\r\nCMD.EXE was started with the above path as the "
        "current directory.\r\nUNC paths are not supported.  Defaulting to Windows "
        "directory.\r\nshell=cmd\r\nplatform=Windows\r\ncwd=C:\\Windows\r\n")
    assert unc.cwd is None and unc.platform == "Windows", unc
    gb = caller_env.parse_probe_result(
        "cwd=/c/Users/Me/proj\nplatform=MINGW64_NT-10.0-22631\nshell=/usr/bin/bash\n")
    assert gb.cwd == "/c/Users/Me/proj", gb
    print("  PowerShell / cmd (CRLF) / Git Bash answers parse; UNC fallback cwd dropped")


def test_parse_probe_result_rejects_an_echoed_probe():
    """A probe that another shell ECHOED instead of running (the cmd probe
    under bash prints `cwd=%CD%` and a made-up `shell=cmd`) is no answer."""
    for text in ("shell=cmd\nplatform=Windows\ncwd=%CD%\n",
                 "cwd=%CD%\n",
                 "cwd=$(pwd)\nplatform=$(uname -s)\nshell=\n",
                 "cwd=$PWD\n"):
        assert caller_env.parse_probe_result(text) is None, text
    assert caller_env.parse_probe_result("cwd=C:\\$Recycle.Bin\\x\n").cwd == \
        "C:\\$Recycle.Bin\\x"
    print("  literal %CD% / $(pwd) / $PWD answers rejected; a real $ in a path kept")


def test_probes_run_in_their_own_shell_and_stay_silent_in_others():
    """Each probe, executed for real: it answers in its own shell and yields
    NO answer in a shell it was not written for (a wrong guess must never
    produce a made-up environment). pwsh is used when installed."""
    import shutil
    import subprocess

    def run(argv):
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60,
                             stdin=subprocess.DEVNULL)
        return caller_env.parse_probe_result(out.stdout + out.stderr)

    here = os.path.realpath(os.getcwd())
    posix_shells = [sh for sh in ("sh", "bash", "zsh", "dash") if shutil.which(sh)]
    assert posix_shells, "no POSIX shell to run the probes in"
    for sh in posix_shells:
        env = run([sh, "-c", caller_env.PROBE_COMMAND])
        assert env is not None and os.path.realpath(env.cwd) == here, (sh, env)
        for other in (caller_env.PROBE_COMMAND_POWERSHELL, caller_env.PROBE_COMMAND_CMD):
            assert run([sh, "-c", other]) is None, (sh, other)
    ran = f"posix probe in {', '.join(posix_shells)}"
    pwsh = shutil.which("pwsh")
    if pwsh:
        env = run([pwsh, "-NoProfile", "-NonInteractive", "-Command",
                   caller_env.PROBE_COMMAND_POWERSHELL])
        assert env is not None and os.path.realpath(env.cwd) == here, env
        assert env.shell == "pwsh" and env.platform in ("Darwin", "Linux", "Windows"), env
        assert run([pwsh, "-NoProfile", "-NonInteractive", "-Command",
                    caller_env.PROBE_COMMAND_CMD]) is None
        ran += "; powershell probe in pwsh"
    print(f"  {ran}; the other probes stay silent there")


def test_render_system_block_tells_a_windows_model_about_paths_and_shell():
    """On a Windows caller the block adds one note (paths as the caller
    writes them, commands in the caller's shell); every other block is
    exactly what it was."""
    win = caller_env.render_system_block(caller_env.CallerEnvironment(
        cwd="C:\\Users\\Me\\proj", platform="win32", shell="PowerShell", source="request"))
    assert "Caller working directory: C:\\Users\\Me\\proj" in win, win
    assert "The caller is on Windows." in win and "PowerShell and\ncmd are not POSIX" in win
    for platform in ("windows", "Windows_NT"):
        assert "The caller is on Windows." in caller_env.render_system_block(
            caller_env.CallerEnvironment(platform=platform, source="request"))
    for platform in (None, "darwin", "linux", "MINGW64_NT-10.0"):
        block = caller_env.render_system_block(caller_env.CallerEnvironment(
            cwd="/x", platform=platform, shell="zsh", source="request"))
        assert "Windows" not in block, block
        assert block.endswith("the caller\nenvironment above is authoritative."), block
    print("  Windows note on win32/windows/Windows_NT only; POSIX block unchanged")

if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
