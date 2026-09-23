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


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} caller-env tests passed")
