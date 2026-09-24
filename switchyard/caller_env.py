"""The caller's tool-execution environment, resolved per request.

The bug this module exists to fix. A Claude Code tab pointed at the gateway
says it is on Linux in `/app/mcp_bridge`, even though the model is talking
back to a Windows host. Two root causes (issue #44):

  1. The client sends no environment at all -- the inner CLI's own
     `# Environment` block, listing `/app/mcp_bridge` and the relay's Linux
     kernel, is the only "where am I" the model ever sees.
  2. The inner CLI adds that block even with `--system-prompt` set; the CLI
     ignores `--exclude-dynamic-system-prompt-sections` in that mode.

The fix is a first-class `CallerEnvironment`: where the tools actually run,
not where the inner CLI runs. It is resolved per request, and the inner
CLI is told about it explicitly. The relay's own environment -- the one
the inner CLI sees by default -- is recognised as the relay's and only
the relay's, never the caller's.

RESOLUTION PRECEDENCE (matches the owner comment on issue #44 exactly):

  1. config forced platform/cwd/shell           -> source=config
  2. environment already present in the request -> source=request
  3. probe result (a tool round-trip)            -> source=probe
  4. config fallback_platform (PLATFORM ONLY)    -> source=host
  5. otherwise                                   -> source=unknown

Host working directory is never used as caller cwd. A docker host's
working directory has no reliable relationship to the directory a caller
later opens in Claude Code or OpenCode. Host platform is a reasonable
last-resort hint only.

PASSIVE ALWAYS BEATS PROBE. A request that names its own environment is
telling us the truth; a probe is a best-guess round-trip that we run only
when we have no other answer. Probing over a known request value would
both cost a tool call and silently overwrite a correct value with a wrong
one.

CONFIG BEATS ALL. When a deployment knows the caller's environment --
a kiosk with one fixed caller, a mobile shell that always points at one
device -- configuration is the source of truth, and the request and the
probe are both politely ignored.

CORRELATION BY ID. A probe's tool_call_id embeds its session fingerprint,
so the mcp_bridge can recover the request from the id alone without a
pending-dictionary dependency. A gateway or sidecar restart, or a duplicate
sidecar instance, cannot strand an in-flight probe.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

# The prefix that tags every synthetic probe tool_call_id, the only public
# surface a caller may match on. The rest of the id is the request's
# fingerprint, recoverable by parse_probe_call_id.
PROBE_PREFIX = "switchyard_env_"

# The probe command. Read-only, narrow, emits exactly three labelled lines.
# NEVER replaces this with `env` or `printenv` -- both dump the caller's
# environment, which carries whatever secrets the caller's terminal
# session holds (AWS keys, GitHub PATs, etc.). The shell name comes from
# the allowlisted `$SHELL` echo, not from any other variable.
#
# This is the POSIX probe: sh, bash, zsh, fish on Linux and macOS, and Git
# Bash / MSYS on Windows (where `uname -s` says MINGW64_NT-...). It does NOT
# run in PowerShell or cmd -- neither has printf/uname, and cmd has no
# `$(...)` -- so a caller whose command tool is one of those gets its own
# probe below (probe_command_for / find_probe_tool). The labelled
# `cwd=...` / `platform=...` / `shell=...` form is what parse_probe_result
# looks for: a regex anchored to those labels tolerates shell quoting,
# whitespace, CRLF line ends and the trailing newline without parsing a
# structured output the various shells don't have a common syntax for.
PROBE_COMMAND = (
    "printf 'cwd=%s\\nplatform=%s\\nshell=%s\\n' \"$(pwd)\" \"$(uname -s)\" \"$SHELL\""
)

# The same three lines from Windows PowerShell 5.1 and PowerShell 7 (pwsh).
# ProviderPath, not Path: on a UNC location Path is prefixed with
# `Microsoft.PowerShell.Core\FileSystem::`. `$IsWindows` does not exist on
# 5.1 (which only runs on Windows), hence `-eq $false` rather than `-not`.
# No variables are assigned, so a persistent caller session is left as it
# was. In a POSIX shell this is a syntax error that prints no labelled line.
PROBE_COMMAND_POWERSHELL = (
    "\"cwd=$((Get-Location).ProviderPath)\"; "
    "\"platform=$(if ($IsWindows -eq $false) { if ($IsMacOS) { 'Darwin' } "
    "else { 'Linux' } } else { 'Windows' })\"; "
    "\"shell=$(if ($PSVersionTable.PSEdition -eq 'Core') { 'pwsh' } "
    "else { 'powershell' })\""
)

# The same from cmd.exe. `cwd` comes LAST on purpose: in a shell that is not
# cmd, `&` backgrounds (bash) or starts a job (pwsh) and the echo prints the
# literal `%CD%` -- parse_probe_result drops a result whose cwd is still an
# unexpanded placeholder, so a mis-aimed cmd probe yields nothing rather
# than a made-up `shell=cmd`.
PROBE_COMMAND_CMD = "echo shell=cmd& echo platform=Windows& echo cwd=%CD%"

PROBE_COMMANDS = {"posix": PROBE_COMMAND, "powershell": PROBE_COMMAND_POWERSHELL,
                  "cmd": PROBE_COMMAND_CMD}

# A path under the relay container is the bug we are explicitly avoiding.
# If a parsed cwd looks like the relay itself, the value is the relay's
# environment, not the caller's -- the inner CLI in /app/mcp_bridge will
# happily tell us so on its own. Reject any value that names the relay
# process's own filesystem.
#
# Matched as PREFIXES of the relay's own directories (issue #187): a
# substring match on "/app/" threw away real caller paths such as
# /home/u/myapp/app/src or /Users/me/app/web.
_RELAY_PATH_PREFIXES = ("/app/mcp_bridge", "/app/cli_bridge", "/relay/",
                        "/tmp/mcpb-", "/tmp/sy-cli-", "/tmp/switchyard-relay/")


def _is_relay_path(value: str | None) -> bool:
    if not value:
        return False
    v = value.replace("\\", "/")
    return any(v.startswith(prefix) for prefix in _RELAY_PATH_PREFIXES)


@dataclass(frozen=True)
class CallerEnvironment:
    """Where the caller's tools execute. `source` says how we know.

    `cwd`, `platform`, `shell` are independently optional -- a request may
    carry a platform but no cwd, a probe may yield cwd and platform but
    no shell (cmd / powershell often resolve the shell later). Unknown
    fields stay None rather than guessing.

    `source` is a coarse provenance label, not a confidence score: it says
    who told us, not how sure we are. `unknown` means nobody did.
    """
    cwd: str | None = None
    platform: str | None = None
    shell: str | None = None
    source: str = "unknown"   # config | request | probe | host | unknown
    # Whether the caller's cwd is a git repository, when its prompt says so.
    # Only used to make the relay's mirror of that directory agree
    # (mcp_bridge.mirror_dir_for); None means nobody said.
    git: bool | None = None

    @classmethod
    def unknown(cls) -> "CallerEnvironment":
        return cls(cwd=None, platform=None, shell=None, source="unknown")


class CallerEnvironmentRequired(Exception):
    """Raised by `resolve()` when `cfg.probe == "required"` and the env
    could not be resolved.

    The operator opted in to loud refusal -- the alternative, rendering
    "Caller platform: unknown" into the inner CLI's system prompt, is
    exactly the relay-env-as-caller-env bug the change exists to
    prevent. The bridges translate this to an HTTPException (503 on
    the mcp path; the text path downgrades `required` to `auto` before
    calling resolve, since there is no probe possible there and
    refusing every request that lacks a passive env would be the wrong
    default for a permissive caller).
    """


def from_wire_metadata(meta: dict | None) -> CallerEnvironment | None:
    """Parse a `metadata.switchyard.caller_env` stamped onto the wire.

    The metadata field is caller-controlled. It MAY have been minted by
    the gateway, which DID read the operator's plans.yaml -- but at this
    layer we have no way to verify the stamp was minted by the gateway
    versus by a caller that reached the sidecar directly. The safe
    answer is two-fold:

      1. NEVER honor plans.yaml-derived labels (`source=config` or
         `source=host`) from the wire. The ONLY path that produces
         `source=config` is the operator's plans.yaml forced values,
         resolved through `CallerEnvironmentSettings` -> `resolve()`.
         The ONLY path that produces `source=host` is
         `cfg.fallback_platform` in the same resolver. Both are
         re-labeled to `source=request` so values still flow through
         at the request tier of the precedence chain, not at the
         privileged plans.yaml tiers. A caller who reaches the sidecar
         directly cannot use either label to bypass the precedence
         chain.
      2. Strict source-marker gate: only stamps with an explicit,
         sensible `source` label are honored. `None`, `"unknown"`,
         and unknown labels all return None so the caller falls back
         to `parse_request(body)` -- matching the pre-round-1
         behaviour. This closes a small attack-surface expansion in
         the round-1 helper, which returned parsed for any stamp with
         at least one field, letting values without a source marker
         preempt passive detection.

    The `source=probe` case is intentionally honored as-is: the only
    legitimate minter is the sidecar's own `_consume_probe_results`,
    which stamps source=probe on a freshly parsed env. A wire forgery
    of probe is technically possible but not useful -- the attacker
    would have to make the tool round-trip succeed to fake the parsed
    env, which they could do anyway. Leave probe alone.

    Returns None for malformed input, when no field carries data, or
    when the source marker is absent / unknown / `unknown`. The caller
    falls back to `parse_request(body)` in those cases.
    """
    if not isinstance(meta, dict):
        return None
    cwd = meta.get("cwd")
    platform = meta.get("platform")
    shell = meta.get("shell")
    # The `git` field is caller-controlled (it is the same bool the
    # caller's prompt advertised about its cwd). Accept only a real
    # boolean -- anything else from the wire (`"yes"`, `1`, an
    # accident of JSON, a legacy stamp that never carried the field)
    # coerces to None so a misbehaving caller cannot stamp a truthy
    # git flag that did not come from the prompt parse.
    git = meta.get("git")
    git = git if isinstance(git, bool) else None
    if not (cwd or platform or shell):
        return None
    raw_source = meta.get("source")
    # Strict source-marker gate. Re-label plans.yaml-derived labels to
    # `request` (values still flow through, but at the request tier);
    # honor `request` and `probe` as-is; everything else (None,
    # `unknown`, unknown labels) returns None so the caller falls
    # through to `parse_request`.
    if raw_source == "config" or raw_source == "host":
        source = "request"       # re-labeled, never honored as-is
    elif raw_source == "request":
        source = "request"
    elif raw_source == "probe":
        source = "probe"          # sidecar-minted only; leave alone
    else:
        return None              # None, "unknown", unknown labels -> passive parse
    return CallerEnvironment(cwd=cwd, platform=platform, shell=shell,
                             source=source, git=git)


# ---------------------------------------------------------------------------
# Passive parsers
# ---------------------------------------------------------------------------
# Unknown callers may put their environment in any of several places. Each
# parser is best-effort and returns CallerEnvironment | None. They never
# raise on a malformed request: the cheap shape-checks happen first, and
# anything weird falls through to the next parser or to `unknown`.


def _blocks_text(content: Any) -> str:
    """Pull text out of either a string content or an OpenAI/Anthropic list-of-blocks
    content. Empty string for anything we cannot read; callers treat that as a
    "no information here" signal and fall through to the next parser."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(str(blk.get("text", "")))
            elif isinstance(blk, dict) and isinstance(blk.get("text"), str):
                parts.append(blk["text"])
        return "\n".join(parts)
    return ""


def _label_value(text: str, *labels: str) -> str | None:
    """First non-empty value among `labels`, matched case-insensitively.

    Two syntaxes are accepted, on a single line:

      `label: value` or `label = value`      (yaml / labelled-line style,
                                              optionally a `- ` / `* ` list
                                              item, as Claude Code's
                                              `# Environment` section is)
      `<label>value</label>`                 (xml-tag style, used by codex
                                              and OpenCode)

    Returns the first non-empty value across all labels, in order.
    """
    if not text:
        return None
    for label in labels:
        labelled = re.compile(
            rf"(?im)^\s*(?:[-*]\s+)?{re.escape(label)}\s*[:=]\s*(.+?)\s*$")
        m = labelled.search(text)
        if m:
            return m.group(1).strip().strip('"').strip("'")
        xml = re.compile(rf"(?is)<{re.escape(label)}\s*>(.+?)</{re.escape(label)}\s*>")
        m = xml.search(text)
        if m:
            return m.group(1).strip()
    return None


def _parse_opencode_env(text: str) -> CallerEnvironment | None:
    """OpenCode currently puts an `<environment>` block in the system
    prompt with `<working_directory>`, `<workspace_root>` and `<platform>`
    lines. Both cwd and platform are common; shell is usually absent.

    Example (verbatim from a real OpenCode 1.18.31 prompt):
        <environment>
          <working_directory>/home/user/project</working_directory>
          <workspace_root>/home/user/project</workspace_root>
          <platform>linux</platform>
        </environment>
    """
    # Distinct from codex's `<environment_context>`: only match the
    # OpenCode form `<environment>` or `<environment ...>` -- never the
    # codex prefix. Same for `working_directory`: that key is uniquely
    # OpenCode's, codex uses `<cwd>` directly.
    if ("<environment>" not in text and "<environment " not in text
            and "working_directory" not in text):
        return None
    cwd = _label_value(text, "working_directory", "workspace_root")
    platform = _label_value(text, "platform")
    if cwd is None and platform is None:
        return None
    return CallerEnvironment(cwd=cwd, platform=platform, shell=None, source="request")


def _parse_codex_env(text: str) -> CallerEnvironment | None:
    """Codex puts a `<environment_context>` block in its first user message
    carrying `cwd`, `shell` and `platform`. Recognised by its tag.

    Example (verbatim from a real codex 0.155.1 request):
        <environment_context>
          <cwd>/home/user/project</cwd>
          <shell>zsh</shell>
          <platform>linux</platform>
        </environment_context>
    """
    # The full `<environment_context>` tag is uniquely codex's; do not
    # match the OpenCode `<environment>` (no `_context` suffix) which is
    # what `_parse_opencode_env` handles above.
    if "<environment_context" not in text:
        return None
    cwd = _label_value(text, "cwd", "working_directory")
    shell = _label_value(text, "shell")
    platform = _label_value(text, "platform")
    if cwd is None and shell is None and platform is None:
        return None
    return CallerEnvironment(cwd=cwd, platform=platform, shell=shell, source="request")


def _parse_anthropic_system(text: str) -> CallerEnvironment | None:
    """Anthropic-protocol requests carry their environment in the system
    prompt: an `<env>` block (older Claude Code, and OpenCode's prompt), a
    `<runtime>` block, or Claude Code's `# Environment` section of bulleted
    lines. All are fair game as a passive source.

    Examples:
        <env>
          cwd=C:\\Users\\someone\\proj
          shell=PowerShell
          platform=Windows
        </env>

        <env>
          Working directory: /home/user/project
          Platform: linux
        </env>

        # Environment
         - Primary working directory: /Users/me/project
         - Platform: darwin
         - Shell: zsh
    """
    if ("<env" not in text and "<runtime" not in text
            and "# Environment" not in text):
        return None
    cwd = _label_value(text, "cwd", "working_directory", "primary working directory",
                       "working directory", "directory")
    platform = _label_value(text, "platform", "os")
    shell = _label_value(text, "shell")
    if cwd is None and platform is None and shell is None:
        return None
    return CallerEnvironment(cwd=cwd, platform=platform, shell=shell, source="request")


def _parse_generic_env_section(text: str) -> CallerEnvironment | None:
    """Generic fallback: scan the first system + first user text for an
    explicit environment section. We deliberately accept labelled lines
    that mention cwd, platform and shell in any casing, but only on the
    first system and first user turns -- a long later user turn that
    happens to mention `platform:` is not a passive environment signal.

    Recognised: `cwd:`, `working directory:`, `platform:`, `os:`,
    `shell:`, and the `SYSTEM INFORMATION` section of Cline-family agents
    (`Operating System:`, `Default Shell:`, `Current Working Directory:` /
    `Current Workspace Directory:`) -- the usual prompt of an
    OpenAI-compatible agent on Windows, whose shell tool runs PowerShell.
    The scan stops at the first match to avoid picking up a body paragraph
    that mentions one of those words.
    """
    if not text:
        return None
    cwd = _label_value(text, "cwd", "working_directory", "primary working directory",
                       "working directory", "current working directory",
                       "current workspace directory")
    platform = _label_value(text, "platform", "os", "operating system")
    shell = _label_value(text, "shell", "default shell")
    if cwd is None and platform is None and shell is None:
        return None
    return CallerEnvironment(cwd=cwd, platform=platform, shell=shell, source="request")


def parse_request(data: dict[str, Any]) -> CallerEnvironment | None:
    """Best-effort passive parse of an OpenAI-shaped request body.

    Reads the top-level `system` (an Anthropic-shaped body, which is what
    the gateway sees for Claude Code's /v1/messages) plus the first two
    messages (system + first user turn), then runs every known parser and
    merges the results FIELD BY FIELD: each of cwd / platform / shell comes
    from the first parser that found it. Returning the first parser that
    found anything lost Claude Code's cwd whenever an earlier parser matched
    only its platform (issue #187). Any parsed cwd that names the relay is
    dropped (the inner CLI sees its own filesystem; that is the bug we are
    avoiding).
    """
    if not isinstance(data, dict):
        return None
    messages = data.get("messages") or []
    if not isinstance(messages, list):
        return None
    blocks: list[str] = []
    top_system = _blocks_text(data.get("system"))
    if top_system:
        blocks.append(top_system)
    for msg in messages[:2]:                # system + first user turn
        if not isinstance(msg, dict):
            continue
        text = _blocks_text(msg.get("content"))
        if text:
            blocks.append(text)
    if not blocks:
        return None
    text = "\n".join(blocks)
    fields: dict[str, str | None] = {"cwd": None, "platform": None, "shell": None}
    for parser in (_parse_opencode_env, _parse_codex_env,
                   _parse_anthropic_system, _parse_generic_env_section):
        env = parser(text)
        if env is None:
            continue
        for name in fields:
            value = getattr(env, name)
            # Strip relay-looking values. A request that names a relay path
            # is telling us the relay's environment, not the caller's, and
            # the inner CLI already gave us that -- the bug is using it.
            if name == "cwd" and _is_relay_path(value):
                value = None
            if fields[name] is None and value:
                fields[name] = value
    if not any(fields.values()):
        return None
    return CallerEnvironment(source="request", git=_parse_git_flag(text), **fields)


def _parse_git_flag(text: str) -> bool | None:
    """Claude Code's `Is a git repository: true`, OpenCode's
    `Is directory a git repo: yes`. None when neither is present."""
    value = _label_value(text, "is a git repository", "is directory a git repo")
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("true", "yes"):
        return True
    if value in ("false", "no"):
        return False
    return None


# ---------------------------------------------------------------------------
# Command-tool recognition
# ---------------------------------------------------------------------------
# A tool that LOOKS like a shell tool is not the same as a tool NAMED like
# one. `get_weather` named `Bash` is still not a command tool; an
# OpenCode shell named `run_command` is. We inspect the JSON schema: a
# string property called `command`, `cmd`, or `script`, ideally required.
# Anything else is treated as non-command.

_COMMAND_ARG_KEYS = ("command", "cmd", "script", "shell_command")


def _tool_schema(tool: dict[str, Any]) -> tuple[str | None, dict]:
    """Return (name, parameters-dict) for an OpenAI tool in either shape.

    OpenAI shape: {"type": "function", "function": {"name", "parameters"}}
    Bare shape:   {"name", "parameters"}  (some clients omit the wrapper)
    """
    if not isinstance(tool, dict):
        return None, {}
    fn = tool.get("function") if tool.get("type") == "function" else tool
    if not isinstance(fn, dict):
        return None, {}
    return fn.get("name"), fn.get("parameters") or {}


def _schema_has_command(parameters: dict) -> str | None:
    """The matching arg key in `parameters`, or None.

    A schema with a required string `command` is the canonical fit; a
    schema that merely has `command` listed as a property also fits when
    every listed property is a string. An object of `command` plus
    unrelated fields still fits -- the actual recognition is "is there a
    command-shaped parameter," not "is this tool a thin wrapper."
    """
    if not isinstance(parameters, dict):
        return None
    props = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    for key in _COMMAND_ARG_KEYS:
        spec = props.get(key)
        if not isinstance(spec, dict):
            continue
        if spec.get("type") != "string":
            continue
        # A `command` that is BOTH required and the only required field is
        # the strongest signal; required-alone is weaker (the caller might
        # invoke without it via tooling quirks); absent-required-but-listed
        # is a hint but not a match on its own. Accept any of the three:
        # the worst case is recognising a tool that takes a command in
        # one of its fields, which is exactly what we want to find.
        if key in required or spec.get("type") == "string":
            return key
    return None


def find_command_tool(tools: list[dict[str, Any]] | None) -> tuple[str, str] | None:
    """Recognise a caller-supplied command-execution tool by schema, not
    by name. Returns (tool_name, arg_key) on a hit, None otherwise.

    Adapters in priority order:
      1. Claude-style: parameter key is `command`.
      2. OpenCode shell / bash: parameter key is `command` or `cmd`.
      3. Generic command / shell / bash / powershell: any of the known
         arg keys, with the tool name matching a recognised hint.
      4. Generic: any tool with a required string command-shaped field,
         regardless of name.

    A tool named `Bash` with no command-shaped schema must NOT match; a
    tool named anything with a required string `command` SHOULD. The
    `Bash` with a bad schema is exactly the bug a name-based matcher
    would fall for.
    """
    if not tools:
        return None
    for tool in tools:
        name, params = _tool_schema(tool)
        if not name:
            continue
        arg_key = _schema_has_command(params)
        if arg_key is None:
            continue
        return name, arg_key
    return None


# Which shell a caller's command tool runs, so the probe is written in a
# syntax that shell can execute. The tool's NAME is the strongest hint (a
# `Bash` tool is bash even on Windows, where Claude Code runs it in Git
# Bash; a `PowerShell` tool is PowerShell), then its description when it
# names exactly one shell family, then the caller's platform: a Windows
# caller with an unnamed shell gets the PowerShell probe (the Windows
# default for interactive agents, and harmless -- a syntax error with no
# labelled output -- in cmd and in POSIX shells). Everything else keeps the
# POSIX probe, byte for byte what it always was.
_POWERSHELL_HINT = re.compile(r"powershell|pwsh", re.I)
_POSIX_HINT = re.compile(r"\b(?:bash|zsh|fish|dash|ksh|sh|posix|wsl)\b", re.I)
_CMD_HINT = re.compile(r"\bcmd\.exe\b|command prompt", re.I)


def is_windows_platform(platform: str | None) -> bool:
    """`win32` (Claude Code, OpenCode), `windows` (codex), `Windows_NT`,
    `Windows 11 Pro ...`. Not MINGW/MSYS/Cygwin: those are POSIX shells."""
    return bool(platform) and str(platform).strip().lower().startswith("win")


def tool_shell_kind(tool: dict[str, Any], platform: str | None = None) -> str:
    """`posix`, `powershell` or `cmd` for a command tool (see above)."""
    name, _ = _tool_schema(tool)
    fn = tool.get("function") if isinstance(tool, dict) and tool.get("type") == "function" \
        else tool
    description = str((fn or {}).get("description") or "") if isinstance(fn, dict) else ""
    name = str(name or "").replace("_", " ").replace("-", " ")
    if _POWERSHELL_HINT.search(name):
        return "powershell"
    if _POSIX_HINT.search(name):
        return "posix"
    # A tool NAMED cmd is cmd.exe; `run_cmd` / `exec_cmd` use "cmd" for
    # "command" and say nothing about the shell, so only the whole name counts.
    if _CMD_HINT.search(name) or name.strip().lower() in ("cmd", "cmd exe"):
        return "cmd"
    hints = {kind for kind, rx in (("powershell", _POWERSHELL_HINT),
                                   ("posix", _POSIX_HINT), ("cmd", _CMD_HINT))
             if rx.search(description)}
    if len(hints) == 1:
        return hints.pop()
    return "powershell" if is_windows_platform(platform) else "posix"


def find_probe_tool(tools: list[dict[str, Any]] | None,
                    platform: str | None = None) -> tuple[str, str, str] | None:
    """(tool_name, arg_key, probe command) for the command tool
    find_command_tool picks, with the probe written for that tool's shell.
    `platform` is whatever is already known of the caller's platform (a
    platform-only prompt, fallback_platform); it only matters when the tool
    itself does not say which shell it runs."""
    hit = find_command_tool(tools)
    if hit is None:
        return None
    name, arg_key = hit
    tool = next(t for t in tools or [] if _tool_schema(t)[0] == name)
    return name, arg_key, PROBE_COMMANDS[tool_shell_kind(tool, platform)]


# ---------------------------------------------------------------------------
# Probe primitives
# ---------------------------------------------------------------------------
def fingerprint(messages: list[dict[str, Any]] | None) -> str:
    """16-hex of the request's prefix. Same idea as
    switchyard.session._prefix_fingerprint: stable across the turns of one
    session, distinct across sessions, recoverable without any
    in-memory state.

    Used as the probe id's suffix -- two probes from different sessions
    look different, so a sidecar can never correlate the wrong tool
    result against the wrong session.
    """
    if not messages:
        return "0" * 16
    parts: list[str] = []
    for msg in messages[:2]:
        if not isinstance(msg, dict):
            continue
        text = _blocks_text(msg.get("content"))
        if text:
            parts.append(text[:2000])
    if not parts:
        return "0" * 16
    blob = "\x00".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()[:16]


def mint_probe_call_id(fp: str) -> str:
    """Self-describing id: PROBE_PREFIX + fingerprint.

    Anything minted by mint_probe_call_id can be split back apart with
    parse_probe_call_id, regardless of how many sidecar instances have
    restarted since it was minted.
    """
    return f"{PROBE_PREFIX}{fp}"


def parse_probe_call_id(call_id: str | None) -> str | None:
    """Inverse of mint_probe_call_id. None for anything that is not a
    probe id; the 16-hex fingerprint otherwise.

    Empty in-memory state does NOT prevent this from working: the
    fingerprint is recoverable from the id itself, which is what makes
    a gateway or sidecar restart safe.
    """
    if not isinstance(call_id, str) or not call_id.startswith(PROBE_PREFIX):
        return None
    rest = call_id[len(PROBE_PREFIX):]
    if len(rest) != 16:
        return None
    try:
        int(rest, 16)
    except ValueError:
        return None
    return rest


# A probe variable that no shell expanded: cmd's %CD%, POSIX $(pwd) / $PWD,
# PowerShell's $((Get-Location)...).
_UNEXPANDED = re.compile(r"%CD%|\$\(|\$PWD\b", re.I)


def parse_probe_result(content: Any) -> CallerEnvironment | None:
    """The three labelled lines from a probe (PROBE_COMMANDS) -> CallerEnvironment.

    Tolerates shell quoting, surrounding text and any whitespace. Each
    line is matched on its own, so a stray `echo` or warning line that
    precedes or follows the labels is ignored. A path that names the
    relay container is dropped with the rest of its field -- the same
    invariant as parse_request: relay paths are the bug, not the answer.
    """
    text = _blocks_text(content)
    if not text:
        return None
    cwd = _label_value(text, "cwd")
    platform = _label_value(text, "platform")
    shell = _label_value(text, "shell")
    if cwd is None and platform is None and shell is None:
        return None
    if cwd is not None and _UNEXPANDED.search(cwd):
        # The probe's text came back echoed, not executed: a shell other
        # than the one the probe was written for printed it literally
        # (e.g. the cmd probe under bash). Its other lines are just as
        # literal, so none of it describes the caller.
        return None
    if _is_relay_path(cwd) or "UNC paths are not supported" in text:
        # cmd.exe refuses a UNC current directory and silently runs in
        # C:\Windows instead, so a cwd it reports then is not the caller's.
        cwd = None
    return CallerEnvironment(cwd=cwd, platform=platform, shell=shell, source="probe")



# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
# The block is appended to the inner CLI's system prompt. The wording
# follows the OWNER's template (issue #44 comment of 2026-09-22 17:10:52)
# almost verbatim -- a model that has read similar blocks in the wild
# recognises it faster than a custom one.
_SYSTEM_BLOCK_TEMPLATE = """[SwitchYard tool execution environment]

The tools available to you execute on the caller's machine.

Caller platform: {platform}
Caller working directory: {cwd}
Caller shell: {shell}

Your CLI process itself is running inside a SwitchYard relay container.
Environment information reported by that relay process describes the
relay only and is not the filesystem/platform where your tools execute.

For filesystem paths, commands, and tool operations, the caller
environment above is authoritative."""


def _render_field(value: str | None) -> str:
    return value if value else "unknown"


def render_system_block(env: CallerEnvironment) -> str:
    """The `[SwitchYard tool execution environment]` block for `env`.

    Unknown fields render as the literal string "unknown", so the model
    sees "Caller platform: unknown" rather than "Caller platform: None"
    -- and the surrounding prose explicitly says unknown means unknown,
    not missing. The relay's environment is NEVER substituted as a
    fallback: the relay is the relay, and saying otherwise is the bug.
    """
    block = _SYSTEM_BLOCK_TEMPLATE.format(
        platform=_render_field(env.platform),
        cwd=_render_field(env.cwd),
        shell=_render_field(env.shell),
    )
    if is_windows_platform(env.platform):
        # The relay CLI's own environment says Linux and names a Linux
        # directory; on a Windows caller nothing of that shape is usable.
        block += _WINDOWS_NOTE
    return block


_WINDOWS_NOTE = """

The caller is on Windows. Give its tools paths in the form the caller
uses (like the working directory above), never a Linux path of the relay,
and write commands in the syntax of the caller shell above: PowerShell and
cmd are not POSIX shells."""


def render_first_turn_reminder(env: CallerEnvironment) -> str:
    """One short line per the OWNER's example, prepended to the first
    real user turn. Counter-anchors the relay CLI's own appended
    `# Environment` block, which the model ranks above the system prompt
    on this exact turn.

    Unknown-env wording is explicit: the model is told not to assume the
    relay container is the caller machine, and to state its assumptions
    instead of asking -- a headless relay has no one to ask (issue #256).
    The known-env line keeps the OWNER-template anchor -- "/app/..." as
    one example of a relay path -- so a model that has seen the prior
    line still recognises it as authoritative; the /relay and /tmp
    additions (issue #294) are a deliberate extension -- they name the
    two other shapes a relay CLI now reports (the per-session workdir
    lives at /relay/<session_id>, cli_bridge's spawned CLI cwd is
    /tmp/sy-cli-*) so the model's "trust the relay env" mistake does
    not reappear with a different root.
    """
    if env.platform or env.cwd:
        return (
            f"[SwitchYard: tools execute on {env.platform or 'unknown'}"
            f" in {env.cwd or 'unknown'}. Any Linux environment the "
            "relay CLI reports for itself (like /app/..., /relay/..., or "
            "/tmp/...) describes only the relay container and must not "
            "be used for tool paths.]"
        )
    return (
        "[SwitchYard: caller environment unknown -- do not assume the "
        "relay container is the caller machine; your tools run on the "
        "caller, so use them to find out, and state any assumption you "
        "make instead of asking.]"
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def resolve(data: dict[str, Any], cfg: Any,
            probe_result: CallerEnvironment | None = None) -> CallerEnvironment:
    """Apply the precedence rules to produce one CallerEnvironment.

    `cfg` is a CallerEnvironmentSettings (or anything with the four
    fields: probe, platform, cwd, shell, fallback_platform).  The
    attribute names match the dataclass in switchyard.models, and the
    only thing we read is the four fields, so a stub is enough for a
    test.

    Precedence, exactly:

      1. cfg forced platform/cwd/shell -> source=config
      2. parse_request -> source=request
      3. probe_result -> source=probe
      4. cfg.fallback_platform (platform ONLY) -> source=host
      5. otherwise -> unknown

    Host working directory is never used as caller cwd. Probe never
    overwrites a known value. Config beats all.

    When `cfg.probe == "required"` and resolution falls through to (5),
    raises CallerEnvironmentRequired instead of returning unknown --
    the operator asked us to refuse unresolvable envs, so we do. The
    bridges translate this to an HTTPException; the text path
    downgrades "required" to "auto" before calling because it cannot
    probe and refusing every request that lacks a passive env would
    be the wrong default for a permissive caller.
    """
    def _set_source(env: CallerEnvironment | None, source: str) -> CallerEnvironment | None:
        if env is None:
            return None
        # Carry `git` through so the flag parsed off the caller's prompt
        # (Claude Code's `Is a git repository: true` / OpenCode's
        # `Is directory a git repo: yes`) survives the resolve() round-trip
        # at the request and probe tiers. The forced-config branch above
        # correctly keeps git=None (plans.yaml never forces git), and the
        # host fallback below hard-codes git=None as well.
        return CallerEnvironment(cwd=env.cwd, platform=env.platform,
                                 shell=env.shell, source=source,
                                 git=env.git)

    forced = CallerEnvironment(
        cwd=getattr(cfg, "cwd", None),
        platform=getattr(cfg, "platform", None),
        shell=getattr(cfg, "shell", None),
    )
    if forced.cwd or forced.platform or forced.shell:
        return _set_source(forced, "config")    # type: ignore[return-value]

    request_env = parse_request(data) if data else None
    if request_env is not None:
        return _set_source(request_env, "request")

    if probe_result is not None:
        return _set_source(probe_result, "probe")

    fb_platform = getattr(cfg, "fallback_platform", None)
    if fb_platform and not _is_relay_path(fb_platform):
        return CallerEnvironment(cwd=None, platform=fb_platform,
                                 shell=None, source="host")

    if getattr(cfg, "probe", "auto") == "required":
        raise CallerEnvironmentRequired(
            "caller_environment.probe=required but the caller's environment "
            "could not be resolved: no forced override, no passive parse, "
            "no probe result, and no fallback_platform. The sidecar will "
            "refuse the request rather than render 'unknown' -- an operator "
            "who set probe=required wants loud refusal, not silent fallback.")

    return CallerEnvironment.unknown()
