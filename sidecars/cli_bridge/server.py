"""OpenAI-compatible shim over a vendor's own CLI, for OAuth-only plans.

Why this exists: LiteLLM authenticates with static API keys, but a Claude Max
subscription and a ChatGPT seat are both OAuth-only, and their tokens live in
the respective CLI's credential store. So the CLI stays the client — it owns
login and refresh — and this shim just speaks HTTP on one side and the CLI on
the other. No OAuth token ever reaches LiteLLM.

One process per plan, selected with PROVIDER:
    PROVIDER=claude    -> `claude -p`       (Claude Max)
    PROVIDER=codex     -> `codex exec`      (ChatGPT seat)
    PROVIDER=opencode  -> `opencode run`    (SuperGrok via OpenCode; OpenCode Go)

PROMPT STACKING, and why these lanes refuse tool calls. Each of these CLIs is a
whole agent harness: it has its own system prompt, its own tools and its own
loop. Pointing another harness (OpenCode, Cursor, Claude Code) at a lane backed
by one of them stacks two agent prompts, and the caller's tool definitions have
nowhere to go — the inner harness's tools act on *this container*, not the
caller's workspace, and their results never reach the caller.

Two mitigations, neither of which makes these lanes agentic:
  * SYSTEM_MODE=replace passes the caller's system prompt with the CLI's
    override flag (`--system-prompt`) instead of appending to the built-in one,
    so only one agent prompt is in play. In that mode the Claude CLI also gets
    `--exclude-dynamic-system-prompt-sections`, which drops the working
    directory, git state and environment blurbs it would otherwise inject — noise
    the caller pays for on every request. Verify both flags exist on your CLI
    version (`claude --help | grep system-prompt`); a wrong flag is a hard error,
    which is why the code default stays `append`.
  * BARE=1 strips the inner harness's tools and caps it at one turn, which is
    as close to a plain completion as a CLI gets.
SwitchYard additionally refuses to route a request containing `tools` to any
CLI-backed plan, and this process rejects one outright rather than dropping the
definitions silently. Text in, text out is the contract.

One CLI can front several subscriptions — OpenCode logged into both xAI and
OpenCode Zen, for instance. Those are separate quotas, so each gets its own
process with its own SWITCHYARD_SUBSCRIPTION; sharing one process would conflate
two connection limits into one gate.

Concurrency, the default model and the allowed model aliases all come from
`config/plans.yaml` — the same file SwitchYard routes from — so there is exactly
one place to change a connection limit. SWITCHYARD_SUBSCRIPTION names which
subscription this process serves; every plan sharing it contributes its model
alias, and the tightest `max_parallel` among them is the connection limit.

The thing it must get exactly right is error mapping: a usage-limit rejection
has to leave here as **HTTP 429 with Retry-After**, because that is the signal
SwitchYard uses to drop the plan's slots out of the lane. A 500 would look like
a transient blip and the lane would keep feeding requests to dead capacity.
"""
from __future__ import annotations

from typing import Any

import asyncio
import base64
import binascii
import collections
import contextlib
import errno
import importlib.util as _il
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import jsonschema

log = logging.getLogger("cli_bridge")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# switchyard/caller_env.py: per-request resolution of the caller's tool-
# execution environment (issue #44). Same import strategy as the mcp_bridge:
# production has PYTHONPATH=/app + the package on disk; tests fall back to
# the file path under the repo root.
try:
    import switchyard.caller_env as _caller_env
except ImportError:
    _ce_path = Path(__file__).resolve().parent.parent.parent / "switchyard" / "caller_env.py"
    if _ce_path.exists():
        _spec = _il.spec_from_file_location("switchyard.caller_env", _ce_path)
        _caller_env = _il.module_from_spec(_spec)
        _caller_env.__package__ = "switchyard"
        sys.modules.setdefault("switchyard", type(sys)("switchyard"))
        sys.modules["switchyard.caller_env"] = _caller_env
        _spec.loader.exec_module(_caller_env)
    else:
        _caller_env = None
# switchyard/models.py: CallerEnvironmentSettings lives here (the typed
# surface plans.yaml's settings.caller_environment is parsed into). Same
# package / path-load idiom as the caller_env block above: production has
# Dockerfile.sidecar's COPY of models.py + PYTHONPATH=/app, so the package
# branch wins in the container; tests that load this module by file path
# without the repo root on sys.path take the path-load branch.
# A None here is a deploy bug: read_config() must log.error and drop the
# caller_environment config rather than silently construct None, so the
# regression is loud in `docker compose logs`, not a quiet half-feature.
try:
    import switchyard.models as _models
except ImportError:
    _models_path = Path(__file__).resolve().parent.parent.parent / "switchyard" / "models.py"
    if _models_path.exists():
        _mspec = _il.spec_from_file_location("switchyard.models", _models_path)
        _models = _il.module_from_spec(_mspec)
        _models.__package__ = "switchyard"
        sys.modules.setdefault("switchyard", type(sys)("switchyard"))
        sys.modules["switchyard.models"] = _models
        _mspec.loader.exec_module(_models)
    else:
        _models = None

app = FastAPI(title="switchyard-cli-bridge")

PROVIDER = os.environ.get("PROVIDER", "claude").lower()
# The plan this process fronts. SWITCHYARD_SUBSCRIPTION is the old name, kept so
# an existing compose file keeps working.
PLAN = (os.environ.get("SWITCHYARD_PLAN")
        or os.environ.get("SWITCHYARD_SUBSCRIPTION", ""))
PLANS_PATH = os.environ.get("SWITCHYARD_PLANS", "/app/config/plans.yaml")
CONFIG_TTL = 30.0          # re-read plans.yaml this often, so edits land live
TIMEOUT = int(os.environ.get("SIDECAR_TIMEOUT", "600"))
# Escape hatch for CLI flags that differ by version, e.g. "--full-auto".
EXTRA_ARGS = [a for a in os.environ.get("CLI_EXTRA_ARGS", "").split() if a]
# replace = override the CLI's own agent prompt; append = stack on top of it.
SYSTEM_MODE = os.environ.get("SYSTEM_MODE", "append").lower()
# Strip the inner harness's tools and cap it at a single turn.
BARE = os.environ.get("BARE", "1") not in ("0", "false", "no")
# One argv element may not exceed the kernel's MAX_ARG_STRLEN (128 KiB on
# Linux); a longer one makes create_subprocess_exec fail with
# "[Errno 7] Argument list too long" before the CLI even starts. Same limit
# and same stdin escape hatch as mcp_bridge's STDIN_PROMPT_LIMIT -- keep the
# two in step. All three profile CLIs take the prompt on stdin: `claude -p`
# and `opencode run` read a piped prompt, `codex exec -` reads stdin when its
# PROMPT argument is `-`.
#
# The limit is measured in UTF-8 BYTES, because that is what MAX_ARG_STRLEN
# caps -- len() counts characters, and a CJK prompt of 40k characters is
# 120k bytes, easily past the kernel's line while looking comfortably under
# a character count (issue #29).
STDIN_PROMPT_LIMIT = int(os.environ.get("MCP_STDIN_PROMPT_LIMIT", "100000"))


def over_argv_limit(text: str) -> bool:
    """True when `text` is too big to ride argv as a single element.

    The kernel counts bytes, so the check does too: UTF-8 is the encoding
    communicate() writes to the pipe and execve assumes for argv.
    """
    return len(text.encode("utf-8")) > STDIN_PROMPT_LIMIT


# --------------------------------------------------- subprocess env (issue #116) ---
# Subprocess env contract. Every `asyncio.create_subprocess_exec` that spawns a
# vendor CLI passes `env=subprocess_env()` -- never `env=os.environ` and never
# `env=None`. The container runs as root with the operator's `.env` mounted
# via compose's env_file, so an inner CLI inheriting the parent environment
# could exfiltrate every provider key, the gateway master key and the OAuth
# grant on a single prompt injection. The allowlist below is what the inner
# CLIs actually need: enough to look up their own binaries (PATH), pick up
# HOME and the XDG_* vars Dockerfile.sidecar pins, and find Claude/Codex's
# credential stores (CLAUDE_CONFIG_DIR, CODEX_HOME).
#
# Values come from os.environ ONLY when set by name -- widening to
# `os.environ.copy()` is the regression we are guarding against. A future CLI
# auth/config failure caused by a missing var is fixed by adding that var to
# SUBPROCESS_ENV_KEYS deliberately (and only that var), so the allowlist
# stays the canonical list.
#
# Bridge-siblings rule: mcp_bridge imports `subprocess_env` from this module
# rather than duplicating it (see sidecars/CLAUDE.md).
SUBPROCESS_ENV_KEYS = (
    "PATH", "HOME", "LANG", "LC_ALL", "TERM",
    "USER", "LOGNAME", "SHELL", "TMPDIR",
    "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
    "CLAUDE_CONFIG_DIR", "CODEX_HOME",
    # Never inherited from the sidecar in practice: mcp_bridge sets it per
    # session to that session's own config when OpenCode runs in a mirror of
    # the caller's cwd (mcp_bridge.session_env). Listed so the allowlist
    # stays the one canonical set of keys a CLI can receive.
    "OPENCODE_CONFIG",
    # Likewise set per spawn, for a web-search request only, to switch on
    # OpenCode's Exa-backed websearch tool (see wants_web_search).
    "OPENCODE_ENABLE_EXA",
)


# --------------------------------------------------- opencode lockdown ---
# OpenCode's built-in tools. `agent.tools: {name: false}` only HIDES a tool
# from the model's tool list: verified on the pinned 1.18.31, a model that
# names `bash` anyway still has it executed, in the sidecar, as `node`. The
# `permission` block is what actually refuses the call ("Model tried to call
# unavailable tool"), and a refusal there does not end the run. So every
# OpenCode config the bridges write carries both, and the deny is `*` --
# a built-in added by a future OpenCode is refused before anyone lists it.
OPENCODE_BUILTIN_TOOLS = (
    "bash", "edit", "write", "read", "grep", "glob", "list", "patch",
    "todowrite", "todoread", "webfetch", "websearch", "task", "multiedit",
    "skill",
)


def opencode_config(allow: tuple[str, ...] = (), prompt: str | None = None) -> dict:
    """The OpenCode project config both bridges run under.

    `allow` names permission patterns to let through: mcp_bridge passes
    `switchyard_*` (its bridged tools) and, for image sessions, `read`.
    Everything else is denied at the permission layer and hidden from the
    tool list. `agent.title.disable` stops the extra title-generation model
    call OpenCode otherwise makes on every run -- a paid request whose
    output nobody reads. sidecars/cli_bridge/harness/opencode.json is this
    function's output for `allow=()`; tests hold the two in step.

    `prompt`, the caller's system prompt, becomes the agent's `prompt:`,
    which REPLACES OpenCode's own base prompt (thousands of tokens of "You
    are opencode..." for most models) rather than sitting underneath the
    caller's instructions in the user turn (issue #264). Both bridges pass
    it: bridge-siblings rule.
    """
    # `"*": "deny"` refuses everything a future OpenCode adds -- but on the
    # pinned 1.18.31 it also drops EVERY built-in from the tool list, even one
    # explicitly allowed (verified against a fake model). So when a built-in
    # is allowed, the other built-ins are denied by name instead.
    allowed_builtins = [tool for tool in allow if tool in OPENCODE_BUILTIN_TOOLS]
    permission = {} if allowed_builtins else {"*": "deny"}
    if allowed_builtins:
        permission.update({tool: "deny" for tool in OPENCODE_BUILTIN_TOOLS
                           if tool not in allow})
    permission.update({pattern: "allow" for pattern in allow})
    config = {
        "$schema": "https://opencode.ai/config.json",
        "permission": permission,
        "agent": {
            "switchyard": {
                "description": "Plain completion relay for Switchyard",
                "mode": "primary",
                "tools": {tool: tool in allow for tool in OPENCODE_BUILTIN_TOOLS},
                "permission": permission,
            },
            "title": {"disable": True},
        },
    }
    if prompt:
        config["agent"]["switchyard"]["prompt"] = prompt
    return config


# ------------------------------------------------------------ web search ---
# A caller that asks for SERVER-SIDE web search -- Claude Code's WebSearch
# makes a sub-request carrying Anthropic's `web_search_20250305` server tool,
# which LiteLLM turns into OpenAI's `web_search_options` -- used to get the
# model's memory presented as "web search results" (issue #264). Such a
# request now switches on the CLI's OWN provider-side search for that request
# only: Claude Code's WebSearch (Anthropic-side), codex's `web_search="live"`
# (OpenAI-side), OpenCode's `websearch`/`webfetch` (Exa). None of them touch
# the sidecar's filesystem. Without the request, all three stay off.
# Anthropic's `web_fetch_*` server tool (fetch one URL) reaches the sidecar
# verbatim through LiteLLM and is served the same way: each CLI's web
# surface fetches as well as searches (Claude Code's WebFetch, codex's
# web.run `open`, OpenCode's webfetch).
WEB_SEARCH_TOOL_PREFIXES = ("web_search",   # web_search, web_search_preview, web_search_20250305
                            "web_fetch")    # web_fetch_20250910


def is_web_search_tool(tool) -> bool:
    """A server-side web search or web fetch tool."""
    return isinstance(tool, dict) and str(tool.get("type", "")).startswith(WEB_SEARCH_TOOL_PREFIXES)


def wants_web_search(body: dict) -> bool:
    """True when the caller asked for server-side web search or fetch."""
    return body.get("web_search_options") is not None or any(
        is_web_search_tool(tool) for tool in body.get("tools") or [])


# ----------------------------------------------------------- typed tools ---
# LiteLLM forwards Anthropic's versioned tool types to the sidecar verbatim,
# as {"type": "bash_20250124", "name": "bash"} -- no input schema at all --
# and a computer tool as a function whose `parameters` is the tool's own
# config ({"type": "computer_20250124", "display_width_px": ...}), which is
# not a JSON Schema (issue #264). Three kinds, three outcomes:
#
#   * CLIENT-executed typed tools (bash, text editor, memory, computer): the
#     Anthropic client runs them, so they are caller tools like any other.
#     They get their documented input schema here, under the caller's own
#     tool name, so the inner CLI can call them and the parked call goes back
#     under that name (LiteLLM maps it back to a tool_use).
#   * Tool search (tool_search_tool_*): dropped. Every tool is shown to the
#     inner CLI anyway, and `defer_loading` on the others is ignored.
#   * SERVER-executed tools with no executor here (Anthropic code execution,
#     MCP connector; OpenAI file search, code interpreter, ...): refused with
#     400 server_tool_unsupported so the router spills to a plan that runs
#     them. Handing such a call back to the client, or answering without it,
#     is the silent degraded 200 the issue is about.
# Web search / web fetch are server tools too, but the CLIs serve them with
# their own search (wants_web_search above); they are none of the above.
_TYPED_TOOL_RE = re.compile(r"^(bash|text_editor|memory|computer)_\d{8}$")

_TEXT_EDITOR_PROPS = {
    "command": {"type": "string",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
    "path": {"type": "string", "description": "Absolute path to the file or directory."},
    "file_text": {"type": "string", "description": "create: the whole content of the new file."},
    "old_str": {"type": "string",
                "description": "str_replace: the exact text to replace; must match exactly once."},
    "new_str": {"type": "string", "description": "str_replace: the replacement text."},
    "insert_line": {"type": "integer",
                    "description": "insert: the line number after which to insert (0 = top of file)."},
    "insert_text": {"type": "string", "description": "insert: the text to insert."},
    "view_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                   "description": "view: [start_line, end_line], 1-indexed; -1 as end means end of file."},
}


def _text_editor_tool(version: str) -> tuple[str, dict]:
    props = dict(_TEXT_EDITOR_PROPS)
    if version == "20250124":
        # The first version has undo_edit and takes insert's text in new_str.
        props["new_str"] = {"type": "string",
                            "description": "str_replace: the replacement text; insert: the text to insert."}
        del props["insert_text"]
        commands = "view, create, str_replace, insert, undo_edit"
    else:
        props["command"] = {**props["command"],
                            "enum": ["view", "create", "str_replace", "insert"]}
        commands = "view, create, str_replace, insert"
    return (f"View, create and edit text files. Commands: {commands}. view shows a file "
            "with line numbers (or lists a directory); create writes file_text to path; "
            "str_replace replaces old_str (which must occur exactly once) with new_str; "
            "insert adds text after insert_line.",
            {"type": "object", "properties": props, "required": ["command", "path"]})


def _computer_tool(config: dict) -> tuple[str, dict]:
    width, height = config.get("display_width_px"), config.get("display_height_px")
    size = f" The display is {width}x{height} pixels." if width and height else ""
    return (("Control the computer's mouse and keyboard and take screenshots." + size
             + " Actions: key (press a key or combination, e.g. ctrl+s, in text), "
             "type (type text), mouse_move (to coordinate), left_click, right_click, "
             "middle_click, double_click, triple_click (at coordinate, text may hold "
             "modifier keys), left_click_drag (to coordinate), left_mouse_down, "
             "left_mouse_up, scroll (at coordinate, scroll_direction up/down/left/right, "
             "scroll_amount), hold_key (text for duration seconds), wait (duration "
             "seconds), screenshot, cursor_position, zoom (a region "
             "[x1, y1, x2, y2] at full resolution)."),
            {"type": "object",
             "properties": {
                 "action": {"type": "string", "description": "The action to perform."},
                 "coordinate": {"type": "array", "items": {"type": "integer"},
                                "minItems": 2, "maxItems": 2,
                                "description": "[x, y] in screen pixels."},
                 "text": {"type": "string",
                          "description": "Text to type, key(s) to press, or modifier keys."},
                 "scroll_direction": {"type": "string",
                                      "enum": ["up", "down", "left", "right"]},
                 "scroll_amount": {"type": "integer"},
                 "duration": {"type": "number", "description": "Seconds (hold_key, wait)."},
                 "region": {"type": "array", "items": {"type": "integer"},
                            "minItems": 4, "maxItems": 4,
                            "description": "zoom: [x1, y1, x2, y2]."},
             },
             "required": ["action"]})


_TYPED_TOOLS = {
    "bash": lambda version, tool: (
        "Run a command in a persistent bash shell session. State (working "
        "directory, environment variables) persists between calls. Set restart "
        "to true to restart the shell.",
        {"type": "object",
         "properties": {"command": {"type": "string", "description": "The bash command to run."},
                        "restart": {"type": "boolean",
                                    "description": "Restart the shell session."}}}),
    "text_editor": lambda version, tool: _text_editor_tool(version),
    "memory": lambda version, tool: (
        "Read and write a persistent memory directory (/memories) that survives "
        "across conversations. Commands: view (a directory or a file, optional "
        "view_range), create (path, file_text), str_replace (path, old_str, "
        "new_str), insert (path, insert_line, insert_text), delete (path), "
        "rename (old_path, new_path).",
        {"type": "object",
         "properties": {
             "command": {"type": "string", "enum": ["view", "create", "str_replace",
                                                    "insert", "delete", "rename"]},
             "path": {"type": "string", "description": "A path under /memories."},
             "file_text": {"type": "string"},
             "old_str": {"type": "string"},
             "new_str": {"type": "string"},
             "insert_line": {"type": "integer"},
             "insert_text": {"type": "string"},
             "old_path": {"type": "string"},
             "new_path": {"type": "string"},
             "view_range": {"type": "array", "items": {"type": "integer"},
                            "minItems": 2, "maxItems": 2},
         },
         "required": ["command"]}),
    "computer": lambda version, tool: _computer_tool(tool),
}


def typed_client_tool(tool) -> dict | None:
    """{name, description, parameters} for an Anthropic client-executed typed
    tool (bash/text editor/memory/computer), in either shape LiteLLM sends;
    None for anything else. The caller's tool name is kept."""
    if not isinstance(tool, dict):
        return None
    kind, config = str(tool.get("type") or ""), tool
    if kind == "function":
        # LiteLLM's computer shape: the tool config sits in `parameters`.
        fn = tool.get("function") or {}
        config = fn.get("parameters") if isinstance(fn, dict) else None
        if not isinstance(config, dict):
            return None
        kind = str(config.get("type") or "")
        if not kind.startswith("computer_"):
            return None
        name = fn.get("name")
    else:
        name = tool.get("name")
    m = _TYPED_TOOL_RE.match(kind)
    if not m:
        return None
    family, version = m.group(1), kind.rsplit("_", 1)[1]
    description, parameters = _TYPED_TOOLS[family](version, config)
    default_name = {"bash": "bash", "text_editor": "str_replace_based_edit_tool",
                    "memory": "memory", "computer": "computer"}[family]
    return {"name": name or default_name, "description": description,
            "parameters": parameters}


def is_tool_search_tool(tool) -> bool:
    """Anthropic's tool search (regex/bm25): nothing to run -- the inner CLI
    sees every tool anyway."""
    return isinstance(tool, dict) and str(tool.get("type", "")).startswith("tool_search_tool")


# Server-executed tools no CLI plan can run. OpenAI types match exactly
# (`computer_use_preview` must not be taken for Anthropic's computer_*).
OPENAI_SERVER_TOOL_TYPES = frozenset({
    "file_search", "code_interpreter", "image_generation", "computer_use_preview",
    "computer_use", "mcp", "local_shell"})
ANTHROPIC_SERVER_TOOL_PREFIXES = ("code_execution", "mcp_toolset")


def server_only_tools(body: dict) -> list[str]:
    """The server-executed tools in this request that no CLI plan has an
    executor for -- sorted type names, plus `mcp_servers` for Anthropic's
    MCP connector body parameter. Empty when there are none."""
    found = set()
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        kind = str(tool.get("type") or "")
        if kind in OPENAI_SERVER_TOOL_TYPES or kind.startswith(ANTHROPIC_SERVER_TOOL_PREFIXES):
            found.add(kind)
    if body.get("mcp_servers"):
        found.add("mcp_servers")
    return sorted(found)


def server_tool_refusal(kinds: list[str]) -> HTTPException:
    """400 server_tool_unsupported: a 4xx the router spills to another plan
    (e.g. an API plan that runs code execution) instead of a degraded 200."""
    return HTTPException(
        status_code=400,
        detail={"error": {
            "message": (f"{PROVIDER} is a CLI-backed plan and has no executor for the "
                        f"server-side tool(s) {', '.join(kinds)}; route this request "
                        "to a plan whose API runs them."),
            "type": "server_tool_unsupported",
            "param": "tools"}})


# -------------------------------------------- effort and output cap ---
# The caller's reasoning effort, in any of the shapes it reaches a sidecar:
# `reasoning_effort` (OpenAI chat), `reasoning.effort` (Responses),
# `output_config.effort` (Anthropic), or `switchyard.reasoning_effort` -- the
# gateway moves it there for CLI plans (switchyard.hooks.carry_to_cli_sidecar),
# because left in place it made LiteLLM send codex tool turns to a Responses
# endpoint no sidecar serves. Each CLI gets it through its own switch.
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
CODEX_EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra")
OPENCODE_VARIANTS = ("minimal", "low", "medium", "high", "max")

# Issue #292: explicit request/display policy, lifted onto the carrier by
# `carry_to_cli_sidecar`. The CLI's thinking mode is only enabled when the
# caller asked for it -- `reasoning_effort` (already mapped above) or
# `thinking.type` in {enabled, adaptive}, never for `disabled` -- and the
# display policy (`display: omitted`) suppresses the assistant's visible
# content while preserving the accumulated reasoning on the response, so the
# adapter path (LiteLLM -> chat / Messages / Responses) sees both halves
# and can adapt the one the caller wanted.
THINKING_DISPLAY_VALUES = ("summarized", "full", "omitted")
THINKING_REQUESTED_TYPES = ("enabled", "adaptive")


def request_thinking(body: dict) -> dict | None:
    """The caller's reasoning-request/display policy.

    Reads from `switchyard.thinking` (the carrier the gateway writes) first,
    then falls back to a top-level `thinking` (Claude Code's spelling) when
    the request arrived from a non-routed caller. An OpenAI `reasoning`
    field that carries only a `display` marker (the gateway already popped
    the effort-typed reason but kept the display marker on the body) is
    also accepted; the sidecar sees the display policy without re-deriving
    the effort. Returns a normalised dict carrying only the fields a CLI
    sidecar acts on -- `type` (the enable signal) and `display` (the
    visibility policy). Returns None when no policy was requested; an
    explicit `thinking.type == "disabled"` is also passed through as `None`
    to match the "off unless asked" rule.
    """
    def _from(d: dict | None) -> dict | None:
        if not isinstance(d, dict):
            return None
        kind = d.get("type")
        display = d.get("display")
        if kind == "disabled":
            return None
        if not kind and not display:
            return None
        out: dict = {}
        if kind in THINKING_REQUESTED_TYPES:
            out["type"] = kind
        if display in THINKING_DISPLAY_VALUES:
            out["display"] = display
        return out or None

    def _from_reasoning(carrier_body: dict) -> dict | None:
        prior = carrier_body.get("reasoning")
        if isinstance(prior, dict):
            display = prior.get("display")
            if isinstance(display, str) and display in THINKING_DISPLAY_VALUES:
                return {"display": display}
        return None

    carried = body.get("switchyard") if isinstance(body.get("switchyard"), dict) else {}
    return (_from(carried.get("thinking"))
            or _from(body.get("thinking"))
            or _from_reasoning(carried)
            or _from_reasoning(body))


def request_effort(body: dict) -> str | None:
    carried = body.get("switchyard") if isinstance(body.get("switchyard"), dict) else {}
    reasoning = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}
    output_config = body.get("output_config") if isinstance(body.get("output_config"), dict) else {}
    effort = (body.get("reasoning_effort") or carried.get("reasoning_effort")
              or reasoning.get("effort") or output_config.get("effort"))
    return str(effort).strip().lower() if effort else None


def effort_args(effort: str | None) -> list[str]:
    """The CLI flag for `effort`, or nothing when this CLI has no such level
    (an unknown level must never break the request)."""
    if not effort:
        return []
    if PROVIDER == "claude":
        level = {"none": "low", "minimal": "low", "ultra": "max"}.get(effort, effort)
        return ["--effort", level] if level in CLAUDE_EFFORTS else []
    if PROVIDER == "codex":
        level = {"none": "low", "minimal": "low"}.get(effort, effort)
        return ["-c", f'model_reasoning_effort="{level}"'] if level in CODEX_EFFORTS else []
    level = {"none": "minimal", "xhigh": "max", "ultra": "max"}.get(effort, effort)
    return ["--variant", level] if level in OPENCODE_VARIANTS else []


def thinking_display(body: dict) -> str | None:
    """The caller's `thinking.display` policy -- `summarized`, `full`, or
    `omitted`. `omitted` means the caller wants only the reasoning, not the
    assistant text; the CLI's argv still runs with the reasoning enabled, the
    final answer's `result` is left empty (or matches the reasoning so the
    gateway's `text`-only consumers see *something*), and the adapter path
    gets a separate `reasoning_content` to render.
    """
    policy = request_thinking(body) or {}
    return policy.get("display")


def thinking_args(thinking: dict | None) -> list[str]:
    """The argv fragment that turns this CLI's reasoning on.

    Only added when the caller actually asked for it -- any policy other than
    the explicit `enabled`/`adaptive` request leaves reasoning off, so the
    CLI's default cost / latency shape is preserved. Unknown / missing policy
    returns `[]` rather than breaking the request.
    """
    if not thinking or "type" not in thinking:
        return []
    if PROVIDER == "claude":
        # claude stream-json already surfaces assistant reasoning through the
        # `assistant` event's content blocks; no CLI flag is required -- it is
        # the model's own behaviour at a given effort level. (Keeping the
        # switch explicit nonetheless is the contract callers asked for.)
        return []
    if PROVIDER == "codex":
        # codex has no per-call "thinking on/off" beyond the effort override
        # the effort branch already added; reasoning shown via the
        # `reasoning` item the parser already extracts.
        return []
    # opencode: --variant <low|medium|high|max|minimal> doubles as the
    # reasoning level; effort_args() above has already mapped the caller's
    # effort to that family. No extra flags are required to enable it, so
    # nothing further goes here.
    return []


def reasoning_event(opencode_part: dict, claude_message_content: list,
                    codex_item: dict) -> str | None:
    """Extract the reasoning text from one CLI event.

    Three shapes, all verified against real output (issue #292):

      OpenCode (`part.type == "reasoning"`):
        {"type":"reasoning","part":{"type":"reasoning","text":"..."}}
      Claude (assistant message content block):
        {"type":"thinking","thinking":"..."}
      Codex (`item.type == "reasoning"`):
        {"type":"item.completed","item":{"type":"reasoning","text":"..."}}

    Returns the text or None for anything else.
    """
    if opencode_part and isinstance(opencode_part, dict):
        kind = opencode_part.get("type")
        if kind == "reasoning":
            text = opencode_part.get("text")
            return text if isinstance(text, str) else None
    if claude_message_content and isinstance(claude_message_content, list):
        for block in claude_message_content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("thinking", "redacted_thinking"):
                text = block.get("thinking")
                if isinstance(text, str):
                    return text
    if codex_item and isinstance(codex_item, dict):
        if codex_item.get("type") == "reasoning":
            text = codex_item.get("text")
            return text if isinstance(text, str) else None
    return None


def request_max_tokens(body: dict) -> int | None:
    """The caller's output cap under any of its spellings. LiteLLM renames
    max_tokens to max_completion_tokens for gpt-5 names (issue #133), and a
    Responses-shaped body says max_output_tokens."""
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        value = body.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def subprocess_env() -> dict:
    """A minimal env dict for spawning a vendor CLI subprocess.

    Only the keys in SUBPROCESS_ENV_KEYS are carried forward, and only when
    they are set in the sidecar's own environment. A missing var the CLI
    actually needs (a credential-store fallback, a locale the chat tool
    depends on) is fixed by adding it to SUBPROCESS_ENV_KEYS above -- not by
    widening this function to copy `os.environ`.
    """
    return {k: os.environ[k] for k in SUBPROCESS_ENV_KEYS if k in os.environ}


def kill_process_group(proc) -> None:
    """Kill `proc` and its whole process group, with fallbacks.

    All vendor CLI spawns in this bridge (and in mcp_bridge, which imports
    this helper) use `start_new_session=True`, so the CLI is its own
    session leader and `os.killpg(os.getpgid(pid), SIGKILL)` reaches every
    grandchild the CLI spawned -- `tool_server.py`, node workers,
    whatever else the inner harness forked. Without the group kill, a
    `proc.kill()` only reaches the CLI itself and leaves the grandchildren
    alive and re-attaching to the plan's connection limit.

    ProcessLookupError is suppressed: the process or its group can already
    be gone (a CLI that just exited on its own, a session we lost track of).
    `proc.kill()` is the fallback when the group kill raises for any other
    reason (no pgid on this platform, e.g.).
    """
    pid = getattr(proc, "pid", None)
    if pid is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


# ------------------------------------------------ claude tool lockdown ---
# Claude Code's built-ins are removed by ALLOWLIST (`--tools ""`, or
# `--tools Read` for an image request), never by a --disallowed-tools
# denylist: a denylist has to name every built-in and silently misses each
# one a CLI release adds -- which is how Agent/Task/Skill/ToolSearch
# (issue #195) and AskUserQuestion (issue #256) stayed live. Verified on the
# pinned 2.1.278 against a fake model: with `--tools ""` the tool list is
# empty (or only the bridged mcp__switchyard__* tools), and a model that
# names Bash/Read/Agent/AskUserQuestion anyway gets "No such tool available".
#
# The rest of the lockdown, shared by both bridges (bridge-siblings rule):
#   --strict-mcp-config          only MCP servers from --mcp-config; no
#                                connectors attached to the logged-in account
#   --setting-sources ""         no user/project settings: the login dir's
#                                CLAUDE.md, hooks and skills stay out of the
#                                prompt (OAuth still reads .credentials.json)
#   --permission-prompts none    anything that would prompt is denied -- there
#                                is no human on a headless sidecar to answer
CLAUDE_LOCKDOWN = ("--strict-mcp-config", "--setting-sources", "",
                   "--permission-prompts", "none")


# ------------------------------------------------------------------ images ---
# flatten() used to drop every non-text content block, so a request with a
# screenshot was accepted, answered confidently, and wrong -- `#EDF6EC` for a
# magenta swatch (issue #30). Not a refusal the caller could see: an answer.
#
# The fix stages the bytes to disk and lets each CLI carry them the way that
# CLI actually can (verified against the installed versions):
#   claude 2.1.278 `-p`   -- no image flag, but `--input-format stream-json`
#                            takes image AND document (PDF) blocks inline
#                            (claude_media_stdin). Staging for a native Read
#                            leaked relay paths the model then handed to the
#                            caller's Read tool (issue #264).
#   codex 0.153.4 `exec`  -- native `-i FILE`, repeatable. Images only.
#   opencode 1.18.32 `run`-- native `-f FILE(s)`, repeatable. Its attachment
#                            docs list PNG/JPEG/GIF/WebP as image media and
#                            reject PDF.
# So a PDF in the prompt reaches codex and opencode as what they DO take:
# its extracted text at the PDF's place in the prompt plus one PNG per page
# attached with -i / -f (render_pdf, poppler -- the same conversion the tool
# path does for a PDF in a tool result). Only a PDF poppler can make nothing
# of is refused.
# Any image block that cannot be carried that way -- a plain http(s) URL, a
# corrupt payload -- is an error, never a silent drop: a wrong answer is the
# one failure a caller cannot distinguish from a real reading.
MEDIA_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif",
               "image/webp": "webp", "application/pdf": "pdf"}
EXT_MEDIA = {ext: media for media, ext in MEDIA_TYPES.items()}


class ImageUnsupportedError(Exception):
    """A request carried an image this plan has no way to deliver.

    Raised instead of quietly continuing without the image, which is what
    made vision through the bridge look like confident hallucination.
    Callers turn this into HTTP 400 images_unsupported via .http().
    """

    def http(self) -> HTTPException:
        return HTTPException(
            status_code=400,
            detail={"error": {"message": str(self), "type": "images_unsupported"}})


def _decode_data_url(url: str) -> tuple[str, bytes] | None:
    """(media_type, bytes) from a `data:<mt>;base64,<blob>` URL, or None."""
    m = re.match(r"data:([^;,]+);base64,(.+)", url or "", re.S)
    if not m:
        return None
    try:
        return m.group(1), base64.b64decode(m.group(2))
    except (binascii.Error, ValueError):
        return None


def _stage_image(block: dict, img_dir: Path, n: int) -> Path:
    """Write one image content block to disk; return its path.

    Handles the Anthropic shape ({"type": "image", "source": {"type":
    "base64", ...}}), the OpenAI one ({"type": "image_url", "image_url":
    {"url": "data:...;base64,..."}}), and the MCP tool-result one
    ({"type": "image", "mimeType": ..., "data": ...}). Anything else -- a
    remote URL most commonly -- cannot be fetched from here.
    """
    data = None
    media = None
    if block.get("type") == "image":
        source = block.get("source") if isinstance(block.get("source"), dict) else {}
        if source.get("type") == "base64":
            media = source.get("media_type")
            try:
                data = base64.b64decode(source.get("data") or "")
            except (binascii.Error, ValueError):
                data = None
        elif block.get("data") and block.get("mimeType"):
            media = block.get("mimeType")
            try:
                data = base64.b64decode(block["data"])
            except (binascii.Error, ValueError):
                data = None
    elif block.get("type") == "image_url":
        url = (block.get("image_url") or {}).get("url") \
            if isinstance(block.get("image_url"), dict) else None
        decoded = _decode_data_url(url)
        if decoded:
            media, data = decoded
    elif block.get("type") == "document":
        # Anthropic document block -- a PDF from Claude Code's Read, typically.
        source = block.get("source") if isinstance(block.get("source"), dict) else {}
        if source.get("type") == "base64" and source.get("media_type") == "application/pdf":
            media = "application/pdf"
            try:
                data = base64.b64decode(source.get("data") or "")
            except (binascii.Error, ValueError):
                data = None
    elif block.get("type") == "file":
        # OpenAI file part: {"file": {"file_data": "data:application/pdf;base64,..."}}
        file_data = (block.get("file") or {}).get("file_data") \
            if isinstance(block.get("file"), dict) else None
        decoded = _decode_data_url(file_data)
        if decoded and decoded[0] == "application/pdf":
            media, data = decoded
    if not data:
        raise ImageUnsupportedError(
            f"image block {n} carries no inline base64 data (a remote URL "
            "cannot be fetched here); this plan refuses to drop it silently")
    ext = MEDIA_TYPES.get(media or "", (media or "img").split("/")[-1] or "img")
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / f"{n:02d}.{ext}"
    path.write_bytes(data)
    return path


# A PDF becomes extracted text + page PNGs for a CLI that cannot take one
# (codex/opencode prompts, claude/codex tool results). Env names keep the
# MCP_ prefix they were introduced under (#280); both bridges read them.
PDF_PAGE_LIMIT = int(os.environ.get("MCP_PDF_PAGE_LIMIT", "20"))
PDF_RENDER_DPI = int(os.environ.get("MCP_PDF_RENDER_DPI", "100"))


def render_pdf(pdf: Path, out_dir: Path, page_limit: int | None = None,
               dpi: int | None = None) -> tuple[str, list[Path]]:
    """(extracted text, [page PNG paths]) for a PDF on disk, via poppler.

    Pages land in `out_dir` as page-<n>.png, in page order, at most
    `page_limit` of them. Either half may be empty -- a scanned PDF has no
    text, a broken renderer no pages -- but not both: that raises
    RuntimeError (and a missing poppler raises OSError), so the caller can
    say the PDF was not delivered instead of sending nothing. Blocking:
    run it off the event loop.
    """
    page_limit = PDF_PAGE_LIMIT if page_limit is None else page_limit
    dpi = PDF_RENDER_DPI if dpi is None else dpi
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-l", str(page_limit),
         str(pdf), str(out_dir / "page")], capture_output=True, text=True, timeout=60)
    images = sorted(out_dir.glob("page-*.png"),
                    key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    text = subprocess.run(["pdftotext", "-layout", "-l", str(page_limit), str(pdf), "-"],
                          capture_output=True, text=True, timeout=30)
    extracted = text.stdout.strip() if text.returncode == 0 else ""
    if text.returncode != 0:
        log.warning("pdftotext failed on a PDF: %s",
                    text.stderr.strip()[:200] or f"exit {text.returncode}")
    if not images and not extracted:
        raise RuntimeError((pages.stderr or text.stderr).strip()[:200]
                           or f"exit {pages.returncode}")
    return extracted, images


def pdf_pages_note(n_pages: int, page_limit: int | None = None) -> str:
    """How many page images went along, and whether the PDF was cut."""
    page_limit = PDF_PAGE_LIMIT if page_limit is None else page_limit
    if not n_pages:
        return "no page images could be rendered"
    return (f"{n_pages} page(s)"
            f"{f' (first {page_limit} only)' if n_pages >= page_limit else ''}")


def _expand_pdf(pdf: Path, n: int) -> tuple[str, list[Path]]:
    """A staged PDF for a CLI that takes images only: (marker, page PNGs).

    The marker carries the extracted text in the PDF's place in the prompt;
    the pages are attached with the CLI's image flag. Nothing usable ->
    ImageUnsupportedError, a 400, never a PDF dropped unseen.

    The marker names the count of attached pages but NOT their paths: every
    CLI now attaches media through its own mechanism (-i on codex, -f on
    opencode, inline on claude), so the path is noise the model could hand
    to the caller's Read tool (issue #296)."""
    try:
        text, pages = render_pdf(pdf, pdf.parent / f"{pdf.stem}-pages")
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise ImageUnsupportedError(
            f"{PROVIDER} cannot take a PDF in the prompt, and document {n} "
            f"could not be rendered to text or page images here: {exc}") from exc
    shown = (f"{pdf_pages_note(len(pages))} attached as images"
             if pages else pdf_pages_note(0))
    told = "extracted text follows" if text else "no text could be extracted"
    return f"[document {n}: PDF, {shown}; {told}]\n{text}".rstrip(), pages


def stage_images(messages: list[dict], img_dir: Path) -> list[Path]:
    """Decode every image content block into `img_dir`, in place.

    Each image block is replaced with a text block carrying
    `[image N: attached]`, so the text collapser keeps a pointer instead of
    nothing, and the model is told bytes are coming without naming a path.
    Works on user messages, system, and `tool` messages alike -- the rebuild
    path in mcp_bridge relies on the last of those. Blocks that cannot be
    decoded raise ImageUnsupportedError rather than vanishing.

    On codex/opencode a PDF is expanded (_expand_pdf): its marker carries the
    extracted text and the returned paths are its page PNGs, not the PDF.

    No relay path appears in the marker on any provider: every CLI attaches
    media through its own mechanism (claude: inline via claude_media_stdin;
    codex: -i on argv; opencode: -f on argv), and a path in front of the
    model is something a caller's tool could Read (issue #296)."""
    staged: list[Path] = []
    blocks = 0
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        replaced: list = []
        for block in content:
            if is_media_block(block):
                blocks += 1
                n = blocks
                path = _stage_image(block, img_dir, n)
                if path.suffix == ".pdf" and PROVIDER != "claude":
                    marker, pages = _expand_pdf(path, n)
                    staged += pages
                    replaced.append({"type": "text", "text": marker})
                    continue
                staged.append(path)
                kind = "document" if path.suffix == ".pdf" else "image"
                replaced.append({"type": "text", "text": f"[{kind} {n}: attached]"})
            else:
                replaced.append(block)
        m["content"] = replaced
    return staged


def image_note(paths: list[Path]) -> str:
    """The line appended to the prompt that makes the model look instead of
    guess. Without it the path markers were just decoration: the model answered
    from its prior about what such a screenshot usually shows.

    Wording is provider-agnostic on purpose: every CLI attaches the media
    itself (claude inline, codex -i, opencode -f), so the prompt needs no
    per-CLI shape and -- more importantly -- names no path the model could
    hand to one of the caller's tools (issue #296)."""
    return (f"\n\n[{len(paths)} attachment(s) follow this text, in order -- look at "
            "them before answering; do not guess at their contents.]")


def is_media_block(block) -> bool:
    """An image (any shape) or a PDF (Anthropic document, OpenAI file part)."""
    if not isinstance(block, dict):
        return False
    kind = block.get("type")
    if kind in ("image", "image_url"):
        return True
    if kind == "document":
        source = block.get("source") if isinstance(block.get("source"), dict) else {}
        return source.get("media_type") == "application/pdf"
    if kind == "file":
        data = (block.get("file") or {}).get("file_data") \
            if isinstance(block.get("file"), dict) else ""
        return str(data or "").startswith("data:application/pdf")
    return False


def has_image_blocks(messages: list[dict]) -> bool:
    """Whether any message carries an image or PDF content block."""
    for m in messages or []:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if is_media_block(b):
                return True
    return False


def claude_media_stdin(prompt: str, paths: list[Path]) -> str:
    """The stream-json user message that gives Claude the prompt and its media
    inline: `claude -p --input-format stream-json` passes image and document
    blocks straight to the model (verified on the pinned 2.1.278), so nothing
    is staged for a native Read and no relay path reaches the model -- which
    used to hand those paths to the CALLER's Read tool (issue #264)."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for path in paths:
        media = EXT_MEDIA.get(path.suffix.lstrip("."), "image/png")
        kind = "document" if media == "application/pdf" else "image"
        content.append({"type": kind, "source": {
            "type": "base64", "media_type": media,
            "data": base64.b64encode(path.read_bytes()).decode()}})
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


CLAUDE_STREAM_ARGS = ("--input-format", "stream-json", "--verbose")


def claude_stream_argv(argv: list[str]) -> list[str]:
    """Switch a claude argv to stream-json in and out (in requires out)."""
    argv = list(argv)
    if "--output-format" in argv:
        argv[argv.index("--output-format") + 1] = "stream-json"
    else:
        argv += ["--output-format", "stream-json"]
    return argv + list(CLAUDE_STREAM_ARGS)


def stage_or_fail(messages: list[dict]) -> tuple[list[Path], Path | None]:
    """Stage a request's images into a fresh temp dir, or 400 loudly.

    Returns (paths, dir); the caller owns the dir and must remove it. Shared
    by cli_bridge and mcp_bridge so the loud-failure contract cannot drift
    between the two.
    """
    if not has_image_blocks(messages):
        return [], None
    img_dir = Path(tempfile.mkdtemp(prefix="swimg-"))
    # except BaseException, not just ImageUnsupportedError: an OSError from
    # write_bytes (full disk, permission flip mid-request) is otherwise a
    # silent `swimg-*` dir leak on every retry until the temp partition fills.
    try:
        paths = stage_images(messages, img_dir / "img")
        if not paths:
            # Only PDFs that rendered to text alone: nothing to attach.
            shutil.rmtree(img_dir, ignore_errors=True)
            return [], None
        return paths, img_dir
    except BaseException:
        shutil.rmtree(img_dir, ignore_errors=True)
        raise


async def stage_or_fail_async(messages: list[dict]) -> tuple[list[Path], Path | None]:
    """stage_or_fail off the event loop when there is media to stage: a PDF
    on codex/opencode is rendered with poppler (blocking, up to seconds), and
    other sessions must keep being served meanwhile."""
    if not has_image_blocks(messages):
        return [], None
    return await asyncio.to_thread(stage_or_fail, messages)

# Per-provider invocation. `prompt` and `system` are substituted; `system` is
# dropped entirely when the CLI has no equivalent flag.
PROFILES: dict[str, dict] = {
    "claude": {
        "cli": os.environ.get("CLAUDE_CLI", "claude"),
        "model": os.environ.get("CLAUDE_MODEL", "opus"),
        "args": ["-p", "{prompt}", "--output-format", "json", "--model", "{model}"],
        # SYSTEM_MODE=replace swaps these for --system-prompt, which overrides
        # the CLI's own agent prompt instead of stacking on top of it. See the
        # prompt-stacking note in the module docstring.
        "system_args": ["--append-system-prompt", "{system}"],
        "system_args_replace": ["--system-prompt", "{system}"],
        # The same two flags in their -file forms, used when the caller's
        # system prompt is itself over the argv limit (issue #29): an inline
        # --append-system-prompt is one argv element, and the kernel rejects
        # execve with E2BIG long before the total argument block is anywhere
        # near ARG_MAX. Verified on the pinned CLI with
        # `claude --help | grep system-prompt` (2.1.278 documents
        # --system-prompt[-file] / --append-system-prompt[-file]); like the
        # inline flags, a wrong flag is a hard error, so the file form is
        # reached only on the oversized path -- ordinary requests keep the
        # verified inline behaviour.
        "system_file_args": ["--append-system-prompt-file", "{path}"],
        "system_file_args_replace": ["--system-prompt-file", "{path}"],
        # Strip the inner harness's own tools: they would act on the sidecar's
        # container, not the caller's workspace, and the caller never sees them.
        # `--tools ""` is an allowlist of zero built-ins (see CLAUDE_LOCKDOWN).
        "bare_args": ["--max-turns", "1", "--tools", "", *CLAUDE_LOCKDOWN],
        # No image variant: media ride inline on stdin (claude_media_stdin),
        # so an image request needs no Read and no extra turn.
        # Only valid alongside --system-prompt. Drops the CLI's dynamically
        # injected sections (working directory, git state, environment), which
        # are pure noise when the caller supplies its own prompt — and which the
        # caller is paying for on every single request.
        "replace_extra_args": ["--exclude-dynamic-system-prompt-sections"],
        "parser": "claude_json",
        # Reset-window hint the CLI prints when the 5h limit is hit.
        "default_retry_after": 5 * 3600,
        # max_tokens enforcement. Claude Code has no token cap flag, so the
        # sidecar truncates the answer to roughly max_tokens*4 bytes
        # post-hoc (see enforce_max_tokens). finish_reason becomes "length"
        # in that case; usage still reports the real billed tokens.
        "enforce_max_tokens": True,
    },
    # OpenCode, logged in to the provider whose subscription this process
    # fronts. Headless is `opencode run`, models are named provider/model, and
    # `--format json` emits event objects. This is how a SuperGrok subscription
    # is reached without Grok Build (which needs SuperGrok Heavy specifically),
    # and it serves OpenCode Zen plans the same way.
    "opencode": {
        "cli": os.environ.get("OPENCODE_CLI", "opencode"),
        "model": os.environ.get("OPENCODE_MODEL", "xai/grok-4.6"),
        # --agent switchyard selects the minimal agent opencode_config() writes
        # into the spawn directory (harness/opencode.json is the same config):
        # every tool hidden and denied. Measured on a trivial call, 7,239 -> 423
        # tokens, a 94% cut — the difference between a subscription being
        # usable for volume and not.
        #
        # The caller's system prompt becomes the agent's `prompt:`, which
        # replaces OpenCode's base prompt (issue #264) instead of being folded
        # into the user turn beneath it; with no caller system prompt the
        # agent has none. (`--pure` changes nothing; no plugins are installed.)
        "args": os.environ.get(
            "OPENCODE_ARGS",
            "run --model {model} --format json --agent switchyard {prompt}").split(),
        # Issue #121: OpenCode uses yargs (verified on 1.18.31), and yargs
        # parses any argv element starting with '-' as a flag. A caller
        # prompt (or system block folded onto it) like "--agent=build" would
        # silently override the `--agent switchyard` the bridge pinned,
        # replacing the no-tools harness with one that has every tool --
        # and the difference is invisible from the answer, because the
        # override happens at flag parse, not at the model layer.
        #
        # The fix is a `--` sentinel spliced in by build_argv immediately
        # before the prompt element; yargs treats `--` as "stop parsing
        # flags", so the element that follows is a positional no matter what
        # it starts with. The sentinel lives in the profile (not in the
        # args template) so OPENCODE_ARGS env overrides keep working --
        # overrides only need to carry the {prompt} slot, not the terminator.
        "prompt_terminator": "--",
        # No system-prompt FLAG at all, verified by testing: --prompt, --system
        # and --system-prompt each exit 1 as unknown options.
        #
        # An agent's `prompt:` field does work — it replaces the base prompt
        # rather than adding to it. An earlier note here said it had no effect,
        # on the strength of an agent told to ignore the user and answer a fixed
        # token; the model answered the user instead. That proved nothing: an
        # instruction to disregard the user is injection-shaped, and refusing it
        # is correct behaviour, not evidence the field was ignored. A neutral
        # marker in the same field was obeyed immediately.
        #
        # It is still omitted, for a different reason: no base prompt file
        # matches `xai/grok-*`, so there is nothing to replace and the field is
        # pure cost. The agent's `tools:` config is where the token cut comes
        # from. The caller's prompt is folded into the message; see fold_system.
        "system_args": [],
        "system_via_agent_prompt": True,
        "parser": "events_json",
        "default_retry_after": 3600,
        "enforce_max_tokens": True,
    },
    # NOTE: the Codex CLI's flags move between releases. Verify against the
    # installed version with `codex exec --help`; override with CLI_EXTRA_ARGS
    # or CODEX_ARGS rather than editing this file.
    "codex": {
        "cli": os.environ.get("CODEX_CLI", "codex"),
        "model": os.environ.get("CODEX_MODEL", "gpt-5"),
        # --skip-git-repo-check: the sidecar's working directory is not a git
        #   repo, and codex otherwise refuses with "Not inside a trusted directory".
        # --model: the ids follow an unobvious scheme — not
        #   gpt-5/gpt-5-codex but gpt-5.6-sol / -terra / -luna and gpt-6-astra.
        #   An unknown id is rejected with "The '<id>' model is not supported when
        #   using Codex with a ChatGPT account", so confirm before setting one.
        # model_instructions_file replaces the compiled-in base instructions.
        #   Measured: 14,255 -> 10,015 tokens on a trivial call. The residue is
        #   codex's own tool schema; include_plan_tool / include_apply_patch_tool /
        #   tools.web_search all had no effect, and experimental_instructions_file
        #   barely moved it (14,154), so this is the key that works.
        "args": os.environ.get(
            "CODEX_ARGS",
            "exec --json --skip-git-repo-check --model {model} {prompt}").split(),
        # Written per request from the caller's system prompt, which makes this a
        # real override rather than an append — the same contract as Claude's
        # --system-prompt.
        "instructions_arg": "-c model_instructions_file={path}",
        "instructions_default": "/app/harness/codex-instructions.md",
        "system_args": [],
        "parser": "codex_jsonl",
        "default_retry_after": 3600,
        # Issue #70 refused max_tokens here because codex exec ran its own
        # multi-turn agent loop, and truncating narration mid-episode is not
        # an honest cap. The tool lockdown (codex_lockdown_args, issue #264)
        # removed every codex tool: the text path is one answer, so the same
        # post-hoc truncation claude and opencode use is an honest cap now.
        # Refusing it rejected every Claude Code / OpenCode request, all of
        # which carry a max_tokens.
        "enforce_max_tokens": True,
    },
}

if PROVIDER not in PROFILES:
    raise SystemExit(f"PROVIDER must be one of {sorted(PROFILES)}, got {PROVIDER!r}")

PROFILE = PROFILES[PROVIDER]
CLI = PROFILE["cli"]


# ---------------------------------------------------------------------------
# Configuration comes from plans.yaml, not from the compose file. Duplicating a
# connection limit in two places means one of them is wrong the moment you
# change the other.
# ---------------------------------------------------------------------------
@dataclass
class Config:
    concurrency: int
    model: str
    models: set   # aliases a request may ask for
    source: str = "config"   # config | fallback — reported by /health
    # How many sessions may sit parked awaiting a tool result. A parked session
    # runs no inference, so it does not occupy a concurrency slot -- but it is
    # still a live CLI process holding memory and a connection pool, so it needs
    # a limit of its own. Defaults to twice the concurrency.
    max_parked: int = 0
    # Per-deployment caller-environment resolution policy, lifted from
    # settings.caller_environment in plans.yaml. Drives the inner CLI's
    # `[SwitchYard tool execution environment]` block (issue #44).
    caller_environment: Any = None

    @property
    def parked_limit(self) -> int:
        return self.max_parked if self.max_parked > 0 else self.concurrency * 2


_config: Config | None = None
_config_at = 0.0
# Last plans.yaml mtime we already logged a load error for. Mirrors the
# `self._plans_mtime` advance pattern in switchyard/hooks.py:_maybe_reload
# (~269-279): each new failure is reported once, not on every subsequent
# config() call while the file is still broken. The next edit gets a new
# mtime and is tried again.
_config_mtime: float | None = None


def read_config() -> Config:
    """Concurrency and model aliases for this plan, read from plans.yaml.

    This delegates to ``switchyard.models.load`` so the sidecar and the
    gateway share one parser, one error vocabulary, and one model-shape.
    An earlier version walked plans.yaml by hand with
    ``except (OSError, ValueError, TypeError)`` — which silently missed
    ``yaml.YAMLError`` (issue #141), so a stray tab made every request
    500 for as long as the file stayed broken. With ``models.load`` we
    raise on every parse / value / shape error and let ``config()``
    decide whether to keep the last-good config or serve a cold-start
    fallback.

    A missing plan, ``_models is None`` (deploy bug — Dockerfile.sidecar
    must COPY switchyard/models.py), and any parse/value error all
    propagate out of this function: the caller (``config()``) is the
    one that knows whether there is a last-good to keep or whether this
    is a cold start.
    """
    if _models is None:
        # Deploy bug: production has Dockerfile.sidecar's COPY of models.py
        # + PYTHONPATH=/app, so reaching this branch means the image was
        # built without models.py or with the PYTHONPATH line dropped.
        # Caller_environment config (and the model alias allowlist) would
        # silently degrade, so raise loudly instead.
        raise RuntimeError(
            "switchyard.models is not importable here. Dockerfile.sidecar "
            "must COPY switchyard/models.py to /app/switchyard/models.py "
            "and set PYTHONPATH=/app."
        )

    registry = _models.load(PLANS_PATH)
    target = PLAN or PROVIDER
    plan = registry.plans.get(target)
    if plan is None:
        raise KeyError(f"no plan {target!r} in {PLANS_PATH}")

    # Concurrency: the configured cap when set, otherwise the seed from
    # settings.concurrency_learning.seed_cap. Same formula as
    # switchyard/policy.py:142 — the sidecar has no learner, so the seed
    # IS the cap when the plan omits an explicit max_parallel. The plan's
    # own value wins when both are present; settings is the fallback.
    # Deliberately NOT applied: gate_headroom_slots, max_parallel_ceiling.
    # The physical gate stays at the configured cap by design
    # (Settings.gate_headroom_slots docstring, models.py ~267-277) and
    # the ceiling only bounds the gateway's learner, which the sidecar
    # does not run.
    cap = plan.configured_parallel or registry.settings.concurrency_learning.seed_cap
    parked = plan.max_parked_sessions
    caller_environment = registry.settings.caller_environment

    # Aliases/default from the plan's live models (enabled and on an
    # enabled, unexpired plan). Strip the leading LiteLLM provider prefix
    # only: `openai/claude-opus-5` -> `claude-opus-5`, but
    # `openai/opencode-go/glm-5.3-flash` keeps the inner provider/model
    # shape OpenCode actually answers to.
    models: set[str] = set()
    default: str | None = None
    for m in plan.live_models:
        spec = m.model
        alias = spec.split("/", 1)[1] if "/" in spec else spec
        models.add(alias)
        if default is None:
            default = alias

    env_conc = os.environ.get("SIDECAR_CONCURRENCY")
    fallback_model = os.environ.get(f"{PROVIDER.upper()}_MODEL") or PROFILE["model"]
    concurrency = int(env_conc) if env_conc else cap
    model = default or fallback_model
    source = "config" if models else "fallback"
    if source == "fallback":
        # Loud, because this is how a sidecar ends up serving a model
        # nobody configured: a disabled/expired plan reads no models,
        # quietly uses the profile default, and looks healthy while
        # doing it.
        log.error("read no models for plan %r from %s — falling back to %r. "
                  "Check SWITCHYARD_PLAN and the plan's `models:` block.",
                  target, PLANS_PATH, model)
    return Config(concurrency=max(1, concurrency), model=model,
                  models=models | {model}, source=source,
                  max_parked=max(0, parked),
                  caller_environment=caller_environment)


def _fallback_config() -> Config:
    """Env/profile fallback used on cold start when plans.yaml is unreadable.

    The ``SIDECAR_CONCURRENCY`` / ``<PROVIDER>_MODEL`` escape hatches still
    apply so an operator can bring a sidecar up by hand. ``source=fallback``
    keeps /health honest about what is being served: a sidecar that could
    not read its config is visibly broken, not silently running on the
    profile default at concurrency 1.
    """
    fallback_model = os.environ.get(f"{PROVIDER.upper()}_MODEL") or PROFILE["model"]
    env_conc = os.environ.get("SIDECAR_CONCURRENCY")
    concurrency = int(env_conc) if env_conc else 1
    return Config(concurrency=max(1, concurrency), model=fallback_model,
                  models={fallback_model}, source="fallback")


def config() -> Config:
    global _config, _config_at, _config_mtime
    if _config is None or (time.time() - _config_at) > CONFIG_TTL:
        try:
            _config = read_config()
            _config_at = time.time()
        except Exception as exc:
            # Advance the mtime stamp so a broken file is reported once,
            # not on every subsequent config() call while the file is
            # still broken. The next edit gets a new mtime and is tried
            # again — same shape as switchyard/hooks.py:_maybe_reload
            # (~269-279).
            try:
                mtime: float | None = os.stat(PLANS_PATH).st_mtime
            except OSError:
                mtime = None
            if mtime != _config_mtime:
                _config_mtime = mtime
                if _config is not None:
                    log.warning("could not reload %s (%s: %s); "
                                "keeping last-good config",
                                PLANS_PATH, type(exc).__name__, exc)
                else:
                    log.warning("could not load %s (%s: %s); "
                                "no last-good, serving env/profile fallback",
                                PLANS_PATH, type(exc).__name__, exc)
            # A warm sidecar must never serve a model it has not already
            # vetted, so keep the last-good `_config` when one exists.
            # Cold start (no `_config` yet) gets the fallback Config so
            # /health still reports `source=fallback` rather than crashing
            # every request with a load failure.
            if _config is None:
                _config = _fallback_config()
                _config_at = time.time()
    return _config


class Gate:
    """A concurrency gate whose limit can change between requests.

    A *new* request never queues: a full gate answers immediately so SwitchYard
    can spill to the next plan in the lane instead of holding a worker open.
    The one exception is acquire_waiting, used only to resume a tool-calling
    session that was already committed to this plan — see mcp_bridge.

    FIFO transfer semantics: a slot freed by `release()` is transferred
    directly to the oldest queued waiter (re-incrementing `_in_flight` to
    cancel the decrement, and resolving the waiter's future), so a brand-new
    `acquire()` can never steal a slot from a woken waiter, and the
    arrival-order contract of `acquire_waiting` is the real one.
    """

    def __init__(self) -> None:
        self._in_flight = 0
        self._lock = asyncio.Lock()
        # Waiters, oldest first. A future is set with True when release()
        # transfers a slot directly to it (no Condition, no notify, no race).
        self._waiters: collections.deque[asyncio.Future] = collections.deque()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def acquire(self, limit: int) -> bool:
        """Try to take a slot. Never queues: returns False on a full gate."""
        async with self._lock:
            if self._in_flight >= limit:
                return False
            self._in_flight += 1
            return True

    async def acquire_waiting(self, limit: int, timeout: float) -> bool:
        """Acquire, waiting up to `timeout` for a slot to come free.

        Callers are queued in arrival order, and a slot is transferred
        directly to the oldest queued waiter by `release()` -- no Condition
        notify, no luck of scheduling. On timeout / cancellation the
        bookkeeping (remove-or-keep the future, return-or-decrement the
        transferred slot) is done under the lock so a release() that
        transfers a slot to us in the same tick is always owned by us,
        and a release() that did not transfer a slot is never silently
        consumed by us. The previous shape re-checked the future outside
        the lock; a release() that fired in that window leaked a slot.
        """
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._lock:
            if self._in_flight < limit:
                self._in_flight += 1
                return True
            self._waiters.append(fut)
        try:
            try:
                await asyncio.wait_for(asyncio.shield(fut), timeout)
            except asyncio.TimeoutError:
                # Resolve the timeout under the lock: a release() in the
                # same tick may have transferred a slot to us, in which
                # case the future is done with True and the slot is ours
                # to keep (return True); otherwise the future is still
                # pending (or already cancelled by a sibling caller) and
                # we owe nothing.
                async with self._lock:
                    if fut.done() and not fut.cancelled() \
                            and fut.result() is True:
                        return True
                    with contextlib.suppress(ValueError):
                        self._waiters.remove(fut)
                return False
            return bool(fut.result())
        except BaseException:
            # Caller cancellation or unexpected exit. Do the abandon
            # bookkeeping under the lock for the same reason as the
            # timeout arm: a release() that transferred to us in the
            # same tick owns the slot -- if the future is done with
            # True we have to release it back, otherwise the gate
            # permanently shrinks capacity.
            async with self._lock:
                if fut.done() and not fut.cancelled() \
                        and fut.result() is True:
                    # We were transferred a slot we no longer want; hand
                    # it back. Re-enter release() semantics: prefer a
                    # live waiter, otherwise decrement freely.
                    while self._waiters:
                        other = self._waiters.popleft()
                        if other.done():
                            continue
                        other.set_result(True)
                        return False
                    self._in_flight = max(0, self._in_flight - 1)
                else:
                    with contextlib.suppress(ValueError):
                        self._waiters.remove(fut)
            if not fut.done():
                fut.cancel()
            raise

    async def release(self) -> None:
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            # Hand the slot directly to the oldest waiter: re-increment to
            # cancel the decrement and resolve the future. `fut.set_result`
            # is synchronous, so doing it inside the lock is safe -- the
            # waiter's `wait_for` will wake and see `_in_flight` already
            # reflecting the transfer.
            while self._waiters:
                fut = self._waiters.popleft()
                if fut.done():
                    continue            # stale (already cancelled by its caller)
                self._in_flight += 1
                fut.set_result(True)
                return
            # No waiter -- the slot is simply freed.


_gate = Gate()

# Concurrency is N CLI subprocesses inside THIS one container, all reading the
# same credential directory — not N containers, so one login covers every
# concurrent run. But a cold start with a token due for refresh would have every
# subprocess racing to refresh and rewrite that shared credential file at once.
# So the first request runs alone; once one has succeeded, the token is fresh and
# the rest can proceed at full concurrency.
_warm = asyncio.Event()
_warmup_lock = asyncio.Lock()


@contextlib.asynccontextmanager
async def warm_gate():
    """Serialize the first call so a token refresh races exactly once.

    Yields a dict kept for backward compatibility with the prior contract;
    no caller writes to it and the gate no longer reads it (mcp_bridge's
    `gate["success"] = False` assignments were removed in the same
    revision that flipped `_warm` to "set on completion, not on success").
    Three paths:

    * Fast path -- `_warm` is already set: the lock is never taken; the body
      runs in parallel with every other concurrent caller.
    * Slow path under the lock -- `_warm` was just set by a concurrent caller
      that beat us to the first run: the body runs again, in parallel from
      then on.
    * Solo path -- we are the first caller under the lock: the body runs
      alone. `_warm` is set once the body completes -- whether it returned
      normally or raised. Serialising every later call behind `_warmup_lock`
      while holding gate slots on a failed first attempt (e.g. a 429) is
      worse than letting a second concurrent caller race; one failed refresh
      is enough information to release the lock.

    Bridge-siblings rule: mcp_bridge imports this context manager and wraps
    `run_session` with it -- the cold-start token-refresh race is a property
    of the inner CLI, not the bridge that fronts it.
    """
    state: dict = {}
    if _warm.is_set():
        yield state
        return
    async with _warmup_lock:
        if _warm.is_set():
            yield state
            return
        try:
            yield state
        finally:
            # Set _warm once the first attempt has run, regardless of
            # outcome: a failed first call should not serialise every later
            # call behind _warmup_lock while holding gate slots.
            _warm.set()
            log.info("%s: first call completed, releasing full concurrency", PROVIDER)


async def invoke(prompt: str, system: str | None, model: str | None,
                image_paths: list[Path] | None = None,
                fmt: dict | None = None, *, web: bool = False,
                effort: str | None = None,
                thinking: dict | None = None) -> dict:
    async with warm_gate():
        return await run_cli(prompt, system, model, image_paths, fmt, web=web,
                             effort=effort, thinking=thinking)

# The CLI reports exhaustion in prose; these are the shapes worth trusting.
def normalise(text: str) -> str:
    """Fold typographic punctuation before matching.

    Codex writes "You\u2019ve hit your usage limit" with a curly apostrophe, so a
    pattern containing a straight quote silently fails to match — and an exhausted
    subscription then looks like a transient error, which is the one mistake that
    makes the router hammer a dead plan.
    """
    return (text or "").replace("\u2019", "'").replace("\u2018", "'") \
                       .replace("\u201c", '"').replace("\u201d", '"')


# Deliberately loose. Vendors reword these constantly, and the cost of a miss is
# asymmetric: a missed limit means a 502 that the router treats as transient and
# retries, while a false positive merely rests a healthy plan for a while.
_LIMIT = re.compile(
    r"(usage limit"                       # "hit your usage limit", "usage limit reached"
    r"|(hit|reached|exceeded) your"       # "you've hit your ...", "reached your ..."
    r"|limit (will )?reset"
    r"|purchase more credits"
    r"|out of (credits|usage|quota)"
    r"|insufficient (balance|credit|quota)"
    r"|rate.?limit"
    r"|too many requests"
    r"|quota)", re.I,
)
_RESET_AT = re.compile(r"reset(?:s|ting)?\s+at\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)", re.I)
# Codex states an absolute reset: "try again at Sep 22nd, 2026 4:37 AM".
_TRY_AGAIN_AT = re.compile(
    r"try again at\s+([A-Z][a-z]{2,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"(?:\s+\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)?)", re.I)


def seconds_until(text: str) -> int | None:
    """Seconds until an absolute reset time stated in a CLI's error message.

    Worth the effort: without it a 31-hour lockout gets the default one-hour
    cooldown, so the lane retries a dead plan thirty more times. The stated time
    carries no timezone, so it is read as UTC; a result in the past or absurdly
    far out is discarded rather than trusted.
    """
    m = _TRY_AGAIN_AT.search(normalise(text))
    if not m:
        return None
    stamp = re.sub(r"(\d)(st|nd|rd|th)", r"\1", m.group(1), flags=re.I).replace(".", "")
    stamp = re.sub(r"\s+", " ", stamp).strip().rstrip(",")
    for fmt in ("%b %d, %Y %I:%M %p", "%b %d %Y %I:%M %p", "%B %d, %Y %I:%M %p",
                "%B %d %Y %I:%M %p", "%b %d, %Y %H:%M", "%b %d, %Y", "%B %d, %Y"):
        try:
            when = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        delta = (when - datetime.now(timezone.utc)).total_seconds()
        if 60 <= delta <= 14 * 86400:
            return int(delta)
        return None
    return None
_AUTH = re.compile(r"(not logged in|please run .?claude login|authentication|invalid credentials)", re.I)


def flatten(messages: list[dict]) -> tuple[str, str | None]:
    """Collapse a chat array into one prompt plus a system prompt.

    A limitation worth knowing: this is stateless, so each turn re-sends the
    whole conversation and pays for it. SwitchYard's session affinity keeps a
    session pinned here, which is what makes the CLI's prompt cache effective.
    """
    system: list[str] = []
    turns: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(b.get("text", "")) for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        content = (content or "").strip()
        if not content:
            continue
        role = m.get("role")
        if role == "system":
            system.append(content)
        elif role == "assistant":
            turns.append(f"Assistant: {content}")
        else:
            turns.append(f"Human: {content}")
    return "\n\n".join(turns), ("\n\n".join(system) or None)


def resolve_model(requested: str | None) -> tuple[str, str | None]:
    """(model to run, warning). An unlisted request falls back to the default.

    A caller cannot make the CLI run an arbitrary model string: the alias has to
    belong to an enabled plan on this subscription. Falling back is logged
    loudly, because an `apex` escalation quietly served by the `judge` model is
    the kind of bug nobody notices.
    """
    cfg = config()
    if not requested or requested == cfg.model:
        return cfg.model, None
    if requested in cfg.models:
        return requested, None
    return cfg.model, (f"model {requested!r} is not an enabled plan on "
                       f"subscription {PLAN or PROVIDER!r} "
                       f"({sorted(cfg.models)}); ran {cfg.model} instead")


# ------------------------------------------------- codex tool lockdown ---
# Codex's own tools act on the sidecar, never on the caller, and the caller
# never sees them run. Verified on the pinned 0.155.1 against a fake model
# (issue #264): `-c sandbox_mode="read-only"` does NOT neutralise the shell --
# the prompt still reads `sandbox_mode` is `danger-full-access` under the
# bypass flag -- and the current catalog entries (gpt-5.6-*, gpt-6-*) run in
# `tool_mode: code_mode_only`, where every tool, the caller's bridged MCP
# tools included, is a hidden nested tool inside a JavaScript `exec` host
# while `exec_command` (the shell) is advertised. That is issue #255: the
# model never saw `switchyard_bash`.
#
# The lockdown, applied on both bridges (bridge-siblings rule):
#   * `--disable <feature>` for every feature that adds a tool or prompt
#     section. An unknown name is a hard error (exit 1), so a CLI upgrade
#     that renames one fails loudly instead of quietly reopening a tool.
#   * `-c include_*=false` etc. drops the environment/permissions/
#     collaboration/apps/skills sections and the question tool, and
#     `project_doc_max_bytes=0` stops AGENTS.md discovery in the relay dir.
#   * `-c model_catalog_json=<file>`: the CLI's own catalog (`codex debug
#     models`) with the code-mode, multi-agent, apply_patch and web-search
#     fields removed, so tools are plain top-level functions again.
# Result, verified: the model sees the caller's prompt and the caller's
# tools (plus codex's three read-only MCP-resource helpers), and a forced
# exec_command / apply_patch call is refused ("unsupported call").
CODEX_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "shell_snapshot", "multi_agent", "goals",
    "apps", "plugins", "remote_plugin", "browser_use", "in_app_browser",
    "computer_use", "image_generation", "view_image", "skill_search",
    "skill_mcp_dependency_install", "tool_suggest", "sleep_tool",
    "code_mode_host", "hooks", "memories", "personality",
    "workspace_dependencies",
)
CODEX_CONFIG_OVERRIDES = (
    # A ChatGPT login gets a live, server-side `web.run` search tool that the
    # tool-feature switches and the catalog do not remove -- found by
    # scripts/live_host_mirror_check.py; invisible to any fake-model check.
    # The only mode that takes it away is `disabled` (verified live).
    'web_search="disabled"',
    "include_environment_context=false",
    "include_permissions_instructions=false",
    "include_collaboration_mode_instructions=false",
    "include_apps_instructions=false",
    "skills.include_instructions=false",
    "tools.experimental_request_user_input.enabled=false",
    "project_doc_max_bytes=0",
)
# Catalog fields that put a model into code mode / multi-agent mode or give it
# the built-in patch and web-search tools.
CODEX_CATALOG_DROP = ("tool_mode", "multi_agent_version",
                      "apply_patch_tool_type", "web_search_tool_type")
CODEX_CATALOG_PATH = Path(os.environ.get(
    "CODEX_CATALOG_PATH",
    os.path.join(tempfile.gettempdir(), "switchyard-codex-catalog.json")))
_codex_catalog_ready = False
# Serialises generate-and-publish: two concurrent first requests must not both
# write and rename the catalog (the loser's rename used to 502 the request).
_codex_catalog_lock = threading.Lock()


def patch_codex_catalog(catalog: dict) -> dict:
    """`codex debug models` output with every tool-adding field removed."""
    models = catalog["models"] if isinstance(catalog, dict) else catalog
    for model in models:
        for key in CODEX_CATALOG_DROP:
            model.pop(key, None)
        model["supports_search_tool"] = False
        model["experimental_supported_tools"] = []
    return catalog


def codex_catalog_path() -> Path:
    """Write the locked-down catalog once per process and return its path.

    Fails closed: if the catalog cannot be built, codex would run with code
    mode and its own tools, so the request is refused with a 502 (the router
    cools the plan) rather than served on an unlocked CLI.
    """
    global _codex_catalog_ready
    if _codex_catalog_ready and CODEX_CATALOG_PATH.exists():
        return CODEX_CATALOG_PATH
    with _codex_catalog_lock:
        # Re-check under the lock: a concurrent first request may have just
        # published it.
        if _codex_catalog_ready and CODEX_CATALOG_PATH.exists():
            return CODEX_CATALOG_PATH
        tmp_name = None
        try:
            done = subprocess.run(
                [PROFILES["codex"]["cli"], "debug", "models"],
                capture_output=True, text=True, timeout=60,
                env=subprocess_env(), cwd=tempfile.gettempdir())
            if done.returncode != 0:
                raise RuntimeError(f"exit {done.returncode}: {done.stderr[:300]}")
            catalog = patch_codex_catalog(json.loads(done.stdout))
            # A unique temp file beside the target, then an atomic replace:
            # a reader never sees a half-written catalog.
            with tempfile.NamedTemporaryFile(
                    "w", dir=CODEX_CATALOG_PATH.parent, prefix=".codex-catalog-",
                    suffix=".tmp", delete=False) as fh:
                tmp_name = fh.name
                fh.write(json.dumps(catalog))
            os.replace(tmp_name, CODEX_CATALOG_PATH)
            tmp_name = None
        except (OSError, ValueError, KeyError, TypeError, RuntimeError,
                subprocess.SubprocessError) as exc:
            log.error("codex tool lockdown unavailable: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"codex tool lockdown unavailable: {exc}") from exc
        finally:
            if tmp_name:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
        _codex_catalog_ready = True
    return CODEX_CATALOG_PATH


def codex_lockdown_args(web: bool = False) -> list[str]:
    """The argv fragment that strips codex down to the caller's tools.

    `web=True` (the caller asked for web search) swaps the web-search
    override for `web_search="live"`: codex's own server-side search."""
    argv: list[str] = []
    for feature in CODEX_DISABLED_FEATURES:
        argv += ["--disable", feature]
    for override in CODEX_CONFIG_OVERRIDES:
        if web and override.startswith("web_search="):
            override = 'web_search="live"'
        argv += ["-c", override]
    argv += ["-c", f'model_catalog_json="{codex_catalog_path()}"']
    return argv


def claude_web_args(argv: list[str]) -> list[str]:
    """Add Claude Code's own WebSearch and WebFetch to an argv (mutating its
    `--tools` and `--max-turns` in place) and return the extra flags to append.

    WebSearch runs Anthropic-side; WebFetch fetches from the sidecar, which
    touches nothing of the caller's. A search and an answer are two turns."""
    if "--tools" in argv:
        i = argv.index("--tools")
        tools = [*argv[i + 1].split(","), "WebSearch", "WebFetch"]
        argv[i + 1] = ",".join(dict.fromkeys(t for t in tools if t))
    if "--max-turns" in argv:
        j = argv.index("--max-turns")
        argv[j + 1] = str(max(int(argv[j + 1]), 4))
    return ["--allowed-tools", "WebSearch,WebFetch"]


def build_argv(prompt: str, system: str | None,
               model: str | None = None,
               image_paths: list[Path] | None = None,
               *, web: bool = False, effort: str | None = None,
               thinking: dict | None = None) -> tuple[list[str], str | None]:
    """Build the CLI argv, plus the prompt to feed it on stdin (or None).

    Returns a pair so an oversized prompt can travel on stdin instead of argv
    (see STDIN_PROMPT_LIMIT for why). The {prompt} slot is handled outside
    fill(): on the argv path it is substituted directly, on the stdin path
    codex's "-" placeholder stays in the template and claude/opencode drop the
    element entirely.

    `image_paths` are staged image files (see stage_images); each profile
    carries them by its own mechanism. No CLI gets them as prompt text: a
    base64 blob in the prompt would blow past MAX_ARG_STRLEN and the model
    cannot read raw bytes anyway.

    `thinking` (issue #292) carries the explicit request/display policy. It
    is only consulted when a request/display policy is present -- a request
    with no policy gets nothing on the argv, leaving reasoning off by
    default. OpenCode already takes the caller's effort via `effort_args`;
    claude and codex stream reasoning through their existing event channels.
    """
    model = model or config().model
    image_paths = image_paths or []
    # claude with media: the prompt and its media ride stdin together as one
    # stream-json message (claude_media_stdin), whatever the prompt's size.
    use_stdin = (PROVIDER == "claude" and bool(image_paths)) or over_argv_limit(prompt)

    def fill(tpl: str) -> str:
        return tpl.replace("{model}", model).replace("{system}", system or "")

    argv = [CLI]
    for element in PROFILE["args"]:
        if element != "{prompt}":
            argv.append(fill(element))
        elif use_stdin and PROVIDER == "codex":
            argv.append("-")
        elif not use_stdin:
            # Issue #121: a prompt element that starts with "-" can be parsed
            # as a CLI flag by yargs (OpenCode) or clap (Codex), or sit in the
            # value slot of an option (claude's `-p "{prompt}"`) and still be
            # misinterpreted by tools that lookahead past option boundaries.
            # Two profiles, two strategies, applied only on the argv path:
            #   * opencode carries a profile-level "prompt_terminator" ("--");
            #     spliced in immediately before the prompt so yargs sees the
            #     element that follows as a positional.
            #   * claude and codex have no sentinel option (the prompt slot is
            #     an option value or a positional whose arg shape we do not
            #     own); for them, prefix the prompt with a fixed non-dash line
            #     when it starts with "-", so the element can never look like
            #     a flag.
            # On the stdin path neither fix is needed: the prompt rides the
            # pipe, not argv, and the flag-vs-positional question never
            # comes up.
            terminator = PROFILE.get("prompt_terminator")
            if terminator:
                argv.append(terminator)
                argv.append(prompt)
            elif prompt.startswith("-"):
                argv.append(f"Message:\n{prompt}")
            else:
                argv.append(prompt)

    key = ("system_args_replace" if SYSTEM_MODE == "replace"
           and PROFILE.get("system_args_replace") else "system_args")
    if system and PROFILE.get(key):
        argv += [fill(a) for a in PROFILE[key]]

    if BARE and PROFILE.get("bare_args"):
        argv += [fill(a) for a in PROFILE["bare_args"]]
    if web and PROVIDER == "claude":
        # Adds WebSearch/WebFetch to --tools and raises --max-turns in place;
        # independent of media, which only switch the input to stream-json.
        argv += claude_web_args(argv)

    if key == "system_args_replace" and system and PROFILE.get("replace_extra_args"):
        argv += [fill(a) for a in PROFILE["replace_extra_args"]]

    if image_paths and PROVIDER != "claude":   # claude: inline, see the return
        if PROVIDER == "codex":
            for path in image_paths:          # repeatable `-i` per file
                argv += ["-i", str(path)]
        else:   # opencode: `-f FILE(s)` attaches to the message
            for path in image_paths:
                argv += ["-f", str(path)]

    if PROVIDER == "codex":
        # The text path's codex has the same shell, patch tool and code-mode
        # host as the tool path's; see CODEX_DISABLED_FEATURES.
        argv += codex_lockdown_args(web=web)

    argv += effort_args(effort)
    argv += thinking_args(thinking)
    if PROVIDER == "claude" and image_paths:
        # Media ride inline on stdin as stream-json, with the prompt.
        return claude_stream_argv(argv) + EXTRA_ARGS, claude_media_stdin(prompt, image_paths)
    return argv + EXTRA_ARGS, (prompt if use_stdin else None)


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def iter_json_objects(blob: str):
    """Yield every JSON object in a stream, whether JSONL or concatenated.

    OpenCode emits one object per line, but pretty-printed or run-together output
    would silently yield nothing under a line-based parser — and "no text found"
    then falls back to returning the raw stream as the answer, which is exactly
    the bug this replaces.
    """
    decoder = json.JSONDecoder()
    index = 0
    length = len(blob)
    while index < length:
        brace = blob.find("{", index)
        if brace < 0:
            return
        try:
            obj, end = decoder.raw_decode(blob, brace)
        except ValueError:
            index = brace + 1
            continue
        index = end
        if isinstance(obj, dict):
            yield obj


class CliNoTextError(ValueError):
    """parse_output found no assistant text but did find charged usage.

    Carries the parsed usage so the caller can act on it: _run_cli uses this
    to drive one in-process retry (issue #64: opencode-ai@1.18.31 occasionally
    emits a step_finish with nonzero output tokens but no text event, which
    is roughly 15-30% of glm-5.3-flash calls). On a ValueError parent so the
    structured-parser catch in mcp_bridge still routes it to its existing
    502 path without further changes there.
    """

    def __init__(self, usage: dict) -> None:
        super().__init__("no assistant text in CLI output (charged tokens present)")
        self.usage = dict(usage or {})


def _sum_usage(target: dict, extra: dict) -> dict:
    """Add every numeric field of `extra` into `target`; mutate target.

    Two attempts' token counts add exactly: 100 input + 50 input = 150 input.
    Returns target for chaining. Non-numeric (and bool) fields are left
    untouched, so an empty extra contributes nothing.
    """
    for key, value in (extra or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value
    return target


def format_no_text_detail(provider: str, usage: dict) -> str:
    """Render the WS1/TEXT_LOST contract detail string.

    The gateway's classifier (switchyard.classify._NO_TEXT regex) and the
    ledger hook (extract_no_text_tokens) match this exact shape: the
    literal phrase `cli emitted tokens with no text` followed by
    `prompt_tokens=N, completion_tokens=M`. Both cli_bridge._run_cli and
    mcp_bridge.run_session call this helper so the contract cannot drift
    between the two sidecars (issue #64): opencode-go / opencode-go2 plans
    use mcp_bridge, so the helper is imported by both rather than
    inlined.
    """
    u = usage or {}
    return (f"{provider} cli emitted tokens with no text "
            f"(prompt_tokens={int(u.get('input_tokens', 0))}, "
            f"completion_tokens={int(u.get('output_tokens', 0))})")


def _parse_claude_json(stdout: str) -> dict:
    """Parse a Claude CLI stdout, which is either `claude --output-format json`
    (one JSON object) or `claude --output-format stream-json` (one event per
    line). Returns the result object -- a dict that already looks like the
    OpenAI-shape the rest of the pipeline consumes (with `result`, `usage`,
    `is_error`, ...). Reasoning text the assistant emitted on prior events is
    collected onto a `reasoning` field so `to_openai` can lift it onto
    `message.reasoning_content`. Returns the dict, or raises JSONDecodeError
    when the stream-json form has no `result` event to anchor on.
    """
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        # --output-format stream-json (one event per line): reasoning block
        # content lands on preceding assistant events, gather those first,
        # then return the trailing result event with the reasoning field
        # attached.
        reasoning_parts: list[str] = []
        for evt in iter_json_objects(stdout):
            if evt.get("type") != "assistant":
                continue
            content = (evt.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("thinking", "redacted_thinking"):
                    text = block.get("thinking")
                    if isinstance(text, str):
                        reasoning_parts.append(text)
        for evt in reversed(list(iter_json_objects(stdout))):
            if evt.get("type") == "result":
                if reasoning_parts:
                    evt = dict(evt)
                    evt["reasoning"] = "".join(reasoning_parts)
                return evt
        raise
    # --output-format json (single object): reasoning text lives in
    # `parsed["content"]` as `thinking` / `redacted_thinking` blocks, the same
    # shape the stream-json assistant events carry. Walk the result's content
    # so the buffered path matches the stream-json contract.
    if isinstance(parsed, dict):
        content = parsed.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("thinking", "redacted_thinking"):
                    text = block.get("thinking")
                    if isinstance(text, str):
                        parts.append(text)
            if parts and "reasoning" not in parsed:
                parsed = {**parsed, "reasoning": "".join(parts)}
    return parsed


def _parse_events_json(stdout: str) -> dict:
    """Parse `opencode --format json` (events_json) and `codex exec --json`
    (codex_jsonl) stdout. Both write one JSON object per line; the two wire
    shapes differ only in what they wrap each piece under -- `part` for
    OpenCode, `item` for Codex. So they share a loop here rather than
    duplicating half the parser twice. Returns the standard
    `{result, usage, reasoning?}` dict the rest of the pipeline consumes;
    raises `CliNoTextError` when the stream had billable output but no text
    event (issue #64), or `JSONDecodeError` for a clean zero-text stream.
    """
    # Two different event shapes, both verified against real output.
    #
    # OpenCode nests everything under `part`:
    #   {"type":"text","part":{"type":"text","text":"OK"}}
    #   {"type":"reasoning","part":{"type":"reasoning","text":"..."}}
    #   {"type":"step_finish","part":{"type":"step-finish",
    #      "tokens":{"input":6194,"output":18,"reasoning":0,
    #                "cache":{"read":1280}}}}
    #
    # Codex uses `item` plus a top-level usage object:
    #   {"type":"item.completed","item":{"type":"agent_message","text":"OK"}}
    #   {"type":"item.completed","item":{"type":"reasoning","text":"..."}}
    #   {"type":"turn.completed","usage":{"input_tokens":14159,
    #      "cached_input_tokens":12160,"output_tokens":7,
    #      "reasoning_output_tokens":0}}
    #
    # Reasoning tokens are counted into output because they are billed, but
    # reasoning *text* is never part of the answer.
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict = {}

    def add(field: str, value) -> None:
        usage[field] = usage.get(field, 0) + _int(value)

    for evt in iter_json_objects(stdout):
        part = evt.get("part") if isinstance(evt.get("part"), dict) else {}
        item = evt.get("item") if isinstance(evt.get("item"), dict) else {}

        # --- OpenCode ------------------------------------------------
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        if part.get("type") == "reasoning":
            rtext = reasoning_event(part, None, None)
            if rtext:
                reasoning_parts.append(rtext)
        tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else None
        if tokens:
            cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
            add("input_tokens", tokens.get("input"))
            add("output_tokens", tokens.get("output"))
            add("output_tokens", tokens.get("reasoning"))
            add("cache_read_tokens", cache.get("read"))
        if isinstance(part.get("cost"), (int, float)):
            # The provider's notional API cost. Recorded for visibility only:
            # on a prepaid subscription the marginal cost of a request is zero.
            usage["provider_cost"] = usage.get("provider_cost", 0.0) + float(part["cost"])

        # --- Codex ---------------------------------------------------
        if item.get("type") in ("agent_message", "message") and isinstance(item.get("text"), str):
            text_parts.append(item["text"])
        if item.get("type") == "reasoning":
            rtext = reasoning_event(None, None, item)
            if rtext:
                reasoning_parts.append(rtext)
        top = evt.get("usage") if isinstance(evt.get("usage"), dict) else None
        if top:
            add("input_tokens", top.get("input_tokens") or top.get("prompt_tokens"))
            add("output_tokens", top.get("output_tokens") or top.get("completion_tokens"))
            add("output_tokens", top.get("reasoning_output_tokens"))
            add("cache_read_tokens", top.get("cached_input_tokens"))

        # --- generic fallbacks for shapes neither of the above covers --
        if not part and not item:
            msg = evt.get("message") or evt.get("text") or evt.get("delta")
            if isinstance(msg, dict):
                msg = msg.get("content") or msg.get("text")
            if isinstance(msg, list):
                msg = "".join(str(b.get("text", "")) for b in msg
                              if isinstance(b, dict))
            if isinstance(msg, str) and msg.strip() and evt.get("type") in (
                    None, "message", "assistant", "agent_message",
                    "response.output_text.delta"):
                text_parts.append(msg)
            # Fallback for a reasoning item the generic path is asked
            # about (a future CLI may emit ``{"type":"reasoning","text":"..."}``
            # without the part/item wrapper). Walk the event for any
            # reason-shaped payload so the contract stays "every reasoning
            # text the CLI said flows onto reasoning_content".
            rtext = reasoning_event(evt.get("reasoning") if isinstance(evt.get("reasoning"), dict) else {},
                                    None, None)
            if rtext:
                reasoning_parts.append(rtext)

    if not text_parts:
        # Issue #64: opencode-ai@1.18.31 sometimes emits a step_finish with
        # nonzero output tokens but no text event. The provider bills those
        # tokens regardless, so the parse failure must carry the usage —
        # not discard it as the original JSONDecodeError did. A zero-usage
        # no-text stream stays on the old path (no retry, original message).
        if _int(usage.get("input_tokens")) or _int(usage.get("output_tokens")) \
                or (isinstance(usage.get("provider_cost"), (int, float))
                    and float(usage["provider_cost"]) > 0):
            raise CliNoTextError(usage)
        raise json.JSONDecodeError("no assistant text in CLI output", stdout, 0)
    out: dict = {"result": "".join(text_parts), "usage": usage}
    reasoning_text = "".join(reasoning_parts)
    if reasoning_text:
        out["reasoning"] = reasoning_text
    return out


def parse_output(stdout: str, kind: str | None = None) -> dict:
    """Normalise a CLI's output into {result, reasoning, usage}.

    `kind` defaults to this process's own PROFILE, but mcp_bridge calls this
    with its own provider's parser kind explicitly — the event shapes
    ("claude_json", "events_json", "codex_jsonl") are the same regardless of
    which sidecar is asking, so there is no reason to duplicate this parser.

    Issue #292: reasoning text is collected separately from the answer so the
    adapter path can lift it into `reasoning_content` for chat / Messages /
    Responses callers. `result` stays as the assistant's answer text only;
    `reasoning` (when present) is the accumulated chain-of-thought. Both are
    stringy, both can be empty, and the absence of `reasoning` is itself the
    "omitted" signal in the OpenAI chat-completion shape.
    """
    kind = kind or PROFILE["parser"]
    if kind == "claude_json":
        return _parse_claude_json(stdout)
    if kind in ("events_json", "codex_jsonl"):
        return _parse_events_json(stdout)
    return {"result": stdout.strip()}


# Parser kinds whose output format is fully specified above; for them the
# raw-output fallback at the _run_cli call site must never fire.
STRUCTURED_PARSERS = {"claude_json", "events_json", "codex_jsonl"}


def estimate_tokens(text: str) -> int:
    """Rough token count from UTF-8 bytes, for post-hoc max_tokens enforcement.

    No tokenizer is available in this image, and the sidecar lives behind the
    CLI anyway -- a precise count would still have to assume a tokeniser that
    never sees the model's output. Four bytes per token is the commonly used
    heuristic; rounding up via ceil() makes a too-short cap still cut
    something rather than silently letting the original through.
    """
    if not text:
        return 0
    return -(-len(text.encode("utf-8")) // 4)   # ceil(bytes / 4)


def enforce_max_tokens(payload: dict, max_tokens: int | None) -> tuple[dict, str]:
    """Truncate the assistant text to honor `max_tokens`; report the finish_reason.

    Returns `(payload, finish_reason)`: `"length"` when content was cut,
    `"stop"` when it already fit. `payload["result"]` is rewritten only on
    truncation -- the byte-identical case is left alone so the unchanged-
    output behaviour (and any caching keyed on it) is preserved. `usage`
    is intentionally NOT touched: the caller paid for whatever the CLI
    actually produced, and `completion_tokens` must stay honest so the
    ledger books the real charge (issue #70).
    """
    if not max_tokens or max_tokens <= 0:
        return payload, "stop"
    text = payload.get("result", "") or ""
    if estimate_tokens(text) <= max_tokens:
        return payload, "stop"
    payload = dict(payload)
    # Cut on a byte boundary and decode `errors="ignore"` so a multi-byte
    # sequence split across the cut does not raise UnicodeDecodeError;
    # the model already paid for what we threw away.
    cut_bytes = max_tokens * 4
    payload["result"] = text.encode("utf-8")[:cut_bytes].decode("utf-8", errors="ignore")
    return payload, "length"


# ---------------------------------------------------------- request params ---
# OpenAI parameters no CLI has a flag for used to be dropped without a word:
# `stop` came back unapplied, `n: 2` came back as one choice, and a
# `response_format` schema came back as prose (the #264 conformance matrix).
# A 200 that ignored what the caller asked for is the one failure the router
# cannot spill on, so each is now either executed or refused:
#   stop            -- applied to the finished text (the CLI generates the
#                      whole answer; the caller sees the same prefix it
#                      would have got, and pays for what the CLI produced).
#   n               -- fanned out as n independent runs on the text path;
#                      refused on the tool path, where a tool loop is one
#                      conversation that cannot fork.
#   response_format -- native where the CLI has it (claude --json-schema,
#                      codex --output-schema for strict schemas), an explicit
#                      instruction elsewhere, and validated either way: an
#                      answer that does not match is a 502, never a 200.
MAX_N = 8


def request_n(body: dict) -> int:
    n = body.get("n")
    if n is None:
        return 1
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_N:
        raise HTTPException(status_code=400, detail={"error": {
            "message": f"n must be an integer from 1 to {MAX_N} on this lane",
            "type": "invalid_request_error", "param": "n"}})
    return n


def response_schema(body: dict) -> dict | None:
    """The caller's `response_format` as {"schema", "strict", "json_object"},
    or None for free text. Anything else -- an unknown type, a typeless or
    non-object value -- is refused, not ignored."""
    fmt = body.get("response_format")
    if fmt is None or (isinstance(fmt, dict) and fmt.get("type") == "text"):
        return None
    if not isinstance(fmt, dict):
        fmt = {"type": type(fmt).__name__}
    if fmt.get("type") == "json_object":
        return {"schema": {"type": "object"}, "strict": False, "json_object": True}
    spec = fmt.get("json_schema") if isinstance(fmt.get("json_schema"), dict) else {}
    if fmt.get("type") == "json_schema" and isinstance(spec.get("schema"), dict):
        try:
            jsonschema.validators.validator_for(spec["schema"]).check_schema(spec["schema"])
        except jsonschema.SchemaError as exc:
            raise HTTPException(status_code=400, detail={"error": {
                "message": f"response_format schema is not a valid JSON Schema: {exc.message}",
                "type": "invalid_request_error", "param": "response_format"}}) from exc
        return {"schema": spec["schema"], "strict": bool(spec.get("strict")),
                "json_object": False}
    raise HTTPException(status_code=400, detail={"error": {
        "message": f"response_format {fmt.get('type')!r} is not supported on this lane",
        "type": "invalid_request_error", "param": "response_format"}})


def native_schema(fmt: dict | None, images: bool = False) -> bool:
    """Whether this CLI enforces the schema itself (see schema_args).

    `images` no longer matters: claude's media ride stream-json stdin on the
    ordinary text-path argv, and --json-schema alongside them is proven on
    the pinned CLI (test_lockdown_claude). Kept for callers' signatures."""
    if not fmt or fmt["json_object"]:
        return False
    if PROVIDER == "claude":
        return True
    return PROVIDER == "codex" and fmt["strict"]


def validate_stop(body: dict) -> None:
    """`stop` is a string or a list of strings; anything else is a 400
    before the CLI runs, not a TypeError after the quota is spent."""
    stop = body.get("stop")
    if stop is None or isinstance(stop, str) or (
            isinstance(stop, list) and all(isinstance(s, str) for s in stop)):
        return
    raise HTTPException(status_code=400, detail={"error": {
        "message": "stop must be a string or a list of strings",
        "type": "invalid_request_error", "param": "stop"}})


def schema_instruction(fmt: dict | None, native: bool) -> str:
    """The prompt line for a schema the CLI is not enforcing itself."""
    if not fmt or native:
        return ""
    if fmt["json_object"]:
        return ("Respond with only a single JSON object: no prose before or "
                "after it, no code fences.")
    return ("Respond with only a JSON value that validates against this JSON "
            "Schema: no prose before or after it, no code fences.\n"
            + json.dumps(fmt["schema"]))


@contextlib.contextmanager
def schema_args(fmt: dict | None, web: bool = False):
    """The argv fragment that makes the CLI enforce the schema natively.

    claude `--json-schema` gives the model a StructuredOutput tool and
    returns its input as the result (verified on the pinned 2.1.278); a
    model that answers in text first is sent back once, so it needs a turn
    more than the text path's one. codex `--output-schema FILE` becomes the
    API's strict `text.format` json_schema, which the API rejects unless the
    schema is strict-shaped -- so only a caller-declared strict schema goes
    that way; anything else gets the instruction and the validation.

    The later `--max-turns` wins, so with `web` it must keep claude_web_args'
    search-and-answer turns and still add the StructuredOutput retry."""
    if not native_schema(fmt):
        yield []
        return
    if PROVIDER == "claude":
        yield ["--json-schema", json.dumps(fmt["schema"]), "--max-turns", "5" if web else "3"]
        return
    fd, path = tempfile.mkstemp(prefix="sy-schema-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(fmt["schema"], fh)
        yield ["--output-schema", path]
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def conform_structured(text: str, fmt: dict) -> str:
    """The answer as schema-valid JSON text, or a 502 the router can spill on."""
    body = text.strip()
    fenced = re.fullmatch(r"```[\w+-]*\s*(.*?)\s*```", body, re.S)
    if fenced:
        body = fenced.group(1)
    try:
        value = json.loads(body)
        if fmt["json_object"] and not isinstance(value, dict):
            raise ValueError("not a JSON object")
        jsonschema.validate(value, fmt["schema"])
    except (ValueError, jsonschema.ValidationError) as exc:
        reason = exc.message if isinstance(exc, jsonschema.ValidationError) else str(exc)
        raise HTTPException(status_code=502, detail={"error": {
            "message": f"{PROVIDER} answer does not match response_format: {reason[:200]}",
            "type": "structured_output_invalid"}}) from exc
    return body


def apply_stop(text: str, stop) -> tuple[str, bool]:
    """Cut the text at the first stop sequence; report whether one hit."""
    stops = [stop] if isinstance(stop, str) else \
        [s for s in stop or [] if isinstance(s, str)]
    hits = [i for i in (text.find(s) for s in stops if s) if i >= 0]
    return (text[:min(hits)], True) if hits else (text, False)


def finish_completion(result: dict, body: dict) -> dict:
    """Apply `stop` and `response_format` to a finished completion (both
    bridges, every choice). A tool-call turn passes untouched. An answer cut
    short -- by the token cap or by the caller's own stop sequence -- is
    not expected to validate, so only a complete answer is checked. A stop
    hit inside a `length` cut reports "stop": the model reached the stop
    sequence before the cap, which is where the real API would have ended."""
    fmt = response_schema(body)
    for choice in result.get("choices") or []:
        message = choice.get("message") or {}
        if choice.get("finish_reason") == "tool_calls" or message.get("tool_calls"):
            continue
        cut = choice.get("finish_reason") == "length"
        text, stopped = apply_stop(message.get("content") or "", body.get("stop"))
        if stopped:
            choice["finish_reason"] = "stop"
        if fmt and not (cut or stopped):
            text = conform_structured(text, fmt)
        message["content"] = text
    return result


def merge_completions(results: list[dict]) -> dict:
    """n independent runs as one n-choice completion, usage summed."""
    merged = dict(results[0])
    merged["choices"] = [{**r["choices"][0], "index": i} for i, r in enumerate(results)]
    usage: dict = {}
    for r in results:
        for key, value in (r.get("usage") or {}).items():
            if isinstance(value, dict):
                inner = usage.setdefault(key, {})
                for k, v in value.items():
                    inner[k] = inner.get(k, 0) + (v or 0)
            elif isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    merged["usage"] = usage
    return merged


@contextlib.contextmanager
def instructions_file(system: str | None, dir: "Path | None" = None):
    """Yield the argv fragment that overrides this CLI's base instructions.

    Codex takes its system prompt as a *file path* in config rather than a flag,
    so the caller's prompt is written to disk per request. That makes it a
    real replacement of the built-in instructions — the same contract as Claude's
    --system-prompt — instead of yet another layer stacked on top.

    With a system prompt and `dir` set (the mcp_bridge path, where the
    session workdir has its own lifecycle), the prompt is written to
    `dir/instructions.md` and NOT self-deleted: `cleanup_workdir` owns the
    session workdir and reclaims it together with everything else. Without
    `dir`, the legacy tempfile path is kept for callers that have no other
    home for the file. With no system prompt, the profile's
    `instructions_default` (`/app/harness/codex-instructions.md`, baked by
    Dockerfile.sidecar:61) is used verbatim.

    Yields ([], system) for CLIs with no such mechanism, leaving the caller's
    text to be folded into the prompt instead.
    """
    # `dir` is the mcp_bridge tool path's signature -- only mcp_bridge's
    # codex branch sets it (cleanup_workdir owns the workdir lifecycle).
    # When called from there the inner cli_bridge module's PROVIDER may
    # not be codex (test fixtures and concurrent-call paths can leave it
    # on whatever the module happened to load with), so read the codex
    # profile directly instead of trusting PROFILE -- the OLD lookup
    # silently no-oped on a non-codex caller PROFILE, dropping the
    # `-c model_instructions_file=...` fragment on the tool path and
    # leaving codex to load its full compiled-in base instructions (the
    # regression #137 closed on the text path).
    if dir is not None:
        profile = PROFILES["codex"]
    else:
        profile = PROFILE
    template = profile.get("instructions_arg")
    if not template:
        yield [], system
        return

    path = profile.get("instructions_default")
    tmp = None
    if system:
        if dir is not None:
            # Caller owns the cleanup (cleanup_workdir reclaims the session
            # workdir with everything else). NOT self-deleted.
            target = dir / "instructions.md"
            target.write_text(system if system.endswith("\n") else system + "\n")
            path = str(target)
        else:
            tmp = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
            tmp.write(system if system.endswith("\n") else system + "\n")
            tmp.close()
            path = tmp.name
    try:
        # The template is one token like `-c key={path}`; split so the value is
        # passed as a single argv element even when the path contains spaces.
        parts = [p.replace("{path}", path) for p in template.split(" ")]
        yield parts, None
    finally:
        if tmp:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


def fold_system(prompt: str, system: str | None) -> tuple[str, str | None]:
    """When a CLI has no system-prompt flag, put the caller's instructions at
    the top of the prompt rather than discarding them silently."""
    if PROFILE.get("instructions_arg"):
        # Handled by instructions_file(), which overrides rather than appends.
        return prompt, system
    if PROFILE.get("system_via_agent_prompt"):
        # OpenCode: _run_cli_attempt writes it as the agent's `prompt:`,
        # which replaces the CLI's base prompt (opencode_config).
        return prompt, system
    key = ("system_args_replace" if SYSTEM_MODE == "replace"
           and PROFILE.get("system_args_replace") else "system_args")
    if system and not PROFILE.get(key):
        return f"{system}\n\n{prompt}", None
    return prompt, system


@contextlib.contextmanager
def system_prompt_file(system: str | None):
    """Move an oversized system prompt off argv and into a temp file.

    The prompt has a stdin escape hatch (build_argv); the system prompt is
    its own argv element, and --append-system-prompt carries it inline, so
    a large system block fails execve with E2BIG exactly as the prompt
    once did (issue #29). Past the limit the text is written to a temp
    file and the -file form of the same flag is passed instead — the
    pinned CLI (2.1.278) documents both. Yields ([], system) unchanged
    when there is nothing to do: no system, a system under the limit, or
    a profile with no file flag (opencode folds the system prompt into the
    stdin prompt, codex overrides via instructions_file, so neither ever
    needs this).
    """
    key: str | None
    if SYSTEM_MODE == "replace":
        # Replace mode must replace or do nothing: falling back to the append
        # form here would quietly turn an override into a stack-up. A profile
        # with inline system_args_replace but no file form gets build_argv's
        # inline path (and, oversized, the spawn guard's 413) instead. That
        # inline path keeps its own looser fallback -- pre-existing behaviour
        # this deliberately does not change.
        key = ("system_file_args_replace"
               if PROFILE.get("system_file_args_replace") else None)
    else:
        key = "system_file_args" if PROFILE.get("system_file_args") else None
    if not system or key is None or not over_argv_limit(system):
        yield [], system
        return
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        tmp.write(system if system.endswith("\n") else system + "\n")
        tmp.close()
        parts = [a.replace("{path}", tmp.name) for a in PROFILE[key]]
        # --exclude-dynamic-system-prompt-sections is only valid alongside
        # the system-prompt flag; build_argv adds it for the inline replace
        # path, so the file path has to carry it too or SYSTEM_MODE=replace
        # would silently stop stripping the CLI's injected sections.
        if key == "system_file_args_replace" and PROFILE.get("replace_extra_args"):
            parts += PROFILE["replace_extra_args"]
        yield parts, None
    finally:
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp.name)


async def run_cli(prompt: str, system: str | None, model: str | None = None,
                image_paths: list[Path] | None = None,
                fmt: dict | None = None, *, web: bool = False,
                effort: str | None = None,
                thinking: dict | None = None) -> dict:
    prompt, system = fold_system(prompt, system)
    with instructions_file(system) as (extra_args, system), \
            system_prompt_file(system) as (sys_args, system), \
            schema_args(fmt, web=web) as fmt_args:
        # When system_prompt_file wrote a temp file it None's `system`,
        # which short-circuits build_argv's inline --system-prompt branch
        # and the matching replace_extra_args one — the file path now
        # carries the same content, with the same extras appended above.
        return await _run_cli(prompt, system, model,
                              extra_args + sys_args + fmt_args, image_paths,
                              web=web, effort=effort, thinking=thinking)


async def _run_cli(prompt: str, system: str | None, model: str | None,
                   extra_args: list,
                   image_paths: list[Path] | None = None, *, web: bool = False,
                   effort: str | None = None,
                   thinking: dict | None = None) -> dict:
    """Spawn the CLI, parse its output, and surface errors as HTTPExceptions.

    Issue #64: opencode-ai@1.18.31 occasionally emits a step_finish with
    nonzero output tokens but no text event (glm-5.3-flash, ~15-30% of calls).
    When that happens parse_output raises CliNoTextError, and the caller's
    retried here exactly once. A successful retry's payload has the first
    attempt's usage folded into it, so charged tokens are booked exactly
    once per attempt. A second no-text-with-usage failure raises the
    contract 502 carrying the COMBINED usage of both attempts.
    """
    try:
        return await _run_cli_attempt(prompt, system, model, extra_args,
                                      image_paths, web=web, effort=effort,
                                      thinking=thinking)
    except CliNoTextError as exc:
        try:
            payload = await _run_cli_attempt(prompt, system, model, extra_args,
                                              image_paths, web=web, effort=effort,
                                              thinking=thinking)
        except CliNoTextError as exc2:
            combined = _sum_usage(dict(exc.usage), exc2.usage)
            raise HTTPException(
                status_code=502,
                detail=format_no_text_detail(PROVIDER, combined),
            ) from exc2
        # Successful retry: the first attempt's charged tokens are still real,
        # so fold them into the payload's usage rather than dropping them.
        payload.setdefault("usage", {})
        _sum_usage(payload["usage"], exc.usage)
        return payload


async def _run_cli_attempt(prompt: str, system: str | None, model: str | None,
                           extra_args: list,
                           image_paths: list[Path] | None = None, *,
                           web: bool = False, effort: str | None = None,
                           thinking: dict | None = None) -> dict:
    """One end-to-end spawn+parse+post-parse check, without the no-text retry.

    Raises HTTPException for terminal errors (spawn failure, timeout, auth,
    quota, upstream client errors, structured-parser failure with no usage).
    Raises CliNoTextError when the structured parser finds no text but does
    find charged usage; that signals the retriable no-text-with-usage shape
    to _run_cli, which then performs exactly one in-process retry.
    """
    cmd, stdin_data = build_argv(prompt, system, model, image_paths, web=web,
                                 effort=effort, thinking=thinking)
    cmd = cmd + list(extra_args)
    spawn_env = subprocess_env()
    if web and PROVIDER == "opencode":
        spawn_env["OPENCODE_ENABLE_EXA"] = "true"

    # stdin must be closed explicitly: codex reads "additional input from stdin"
    # and would block forever on an inherited descriptor that never closes.
    # communicate() closes the pipe after writing, and the stdin path is only
    # ever taken when an oversized prompt rides there (see build_argv).
    #
    # cwd is a fresh per-call temp dir (issue #44): keeps the inner CLI's
    # session out of /app/cli_bridge, where paths beneath it would look like
    # a meaningful project root the model could safely use. Cleaned up at
    # the bottom of this function (success AND failure paths).
    spawn_cwd = Path(tempfile.mkdtemp(prefix="sy-cli-"))
    try:
        # OpenCode reads its project config from the working directory, so
        # the `switchyard` agent has to live HERE. Spawning in a fresh temp
        # dir without it made `--agent switchyard` fall back to OpenCode's
        # `build` agent -- every built-in tool, `permission: * allow`, full
        # base prompt. Inside the try: a failed write is a spawn that never
        # happened, cleaned up and classified by the OSError handler below.
        if PROVIDER == "opencode":
            (spawn_cwd / "opencode.json").write_text(
                json.dumps(opencode_config(
                    allow=("websearch", "webfetch") if web else (), prompt=system)))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=spawn_cwd,
            # env=allowlist, NOT os.environ.copy(): the operator's .env is
            # mounted into this container, so inheriting it wholesale would
            # hand every provider key, the gateway master key and the OAuth
            # grant to the inner CLI -- a single prompt injection exfiltrates
            # all of it (issue #116).
            env=spawn_env,
            # The CLI is its own session leader so a timeout / supersede /
            # reap can SIGKILL the whole group -- the CLI plus tool_server.py
            # plus any node workers the inner harness forked. Without this
            # a bare `proc.kill()` only reaches the CLI itself and leaves
            # the grandchildren alive and re-attaching to the plan's
            # connection limit. kill_process_group is the matching helper.
            start_new_session=True,
        )
    except OSError as exc:
        shutil.rmtree(spawn_cwd, ignore_errors=True)
        _cleanup_project_dir(spawn_cwd)
        # A spawn that never happened is not a broken gateway: E2BIG is the
        # caller's request being too big for argv even after the stdin and
        # file escape hatches (issue #29 asks for 413, so a client can tell
        # "too large" from "down"), and any other spawn failure is this
        # plan's capacity being broken, which must read as 502 so the router
        # cools the plan instead of retrying it as a transient blip.
        if exc.errno == errno.E2BIG:
            raise HTTPException(
                status_code=413,
                detail={"error": {
                    "message": (f"request too large for {PROVIDER} cli "
                                f"({exc.strerror}); reduce its size"),
                    "type": "request_too_large"}}) from exc
        raise HTTPException(
            status_code=502,
            detail=f"{PROVIDER} cli could not be spawned: {exc}") from exc
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(input=stdin_data.encode() if stdin_data is not None
                             else None),
            timeout=TIMEOUT)
    except asyncio.TimeoutError:
        kill_process_group(proc)
        shutil.rmtree(spawn_cwd, ignore_errors=True)
        _cleanup_project_dir(spawn_cwd)
        raise HTTPException(status_code=408, detail=f"{PROVIDER} cli timed out") from None

    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
    blob = normalise(f"{stdout}\n{stderr}")
    try:
        if proc.returncode != 0 or not stdout.strip():
            if _AUTH.search(blob):
                raise HTTPException(status_code=401,
                                    detail=f"{PROVIDER} cli not authenticated: {stderr[:300]}")
            up_status, up_message, up_retryable = upstream_error(stdout)

            # A quota message wins over the upstream status, even a non-retryable 403.
            # xAI answers an exhausted SuperGrok subscription with
            # "personal-team-blocked:spending-limit: You have run out of credits or
            # need a Grok subscription..." / 403 / isRetryable:false — which reads
            # terminal but is a plan that refills. Calling it a dead credential would
            # sideline a healthy subscription.
            if _LIMIT.search(blob) or seconds_until(blob):
                raise _limit_error(f"{up_message}\n{blob}" if up_message else blob)

            # Otherwise an explicit upstream client error is the truth: a bad model
            # id, a blocked account, a rejected credential. Surface it as-is so the
            # router does not cool a plan down over something waiting cannot fix.
            if up_status in (401, 402, 403) and up_retryable is not True:
                raise HTTPException(
                    status_code=up_status,
                    detail={"error": {"message": up_message or blob[:300],
                                      "type": "entitlement_or_credentials"}})
            status, detail = error_from_events(stdout)
            if status and 400 <= status < 500 and status != 429:
                # A client error is our fault, not the provider's — surface it as-is
                # so SwitchYard does not cool the plan down over a bad request.
                raise HTTPException(status_code=status,
                                    detail={"error": {"message": detail or blob[:300],
                                                      "type": "upstream_client_error"}})
            raise HTTPException(
                status_code=502,
                detail=f"{PROVIDER} cli failed ({proc.returncode}): "
                       f"{detail or stderr[:300] or stdout[:300]}")

        try:
            payload = parse_output(stdout)
        except CliNoTextError:
            # Issue #64 retriable shape -- propagate to _run_cli's retry without
            # wrapping, so the combined-usage 502 above can name real numbers.
            raise
        except (json.JSONDecodeError, ValueError) as exc:
            # A structured parser knowing the output shape has nothing left to
            # guess: echoing the raw stream here is how step_start / step_finish
            # JSONL leaked through to clients as the "answer" (issue #3). Surface
            # the failure as a real error so the gateway retries instead.
            if PROFILE["parser"] in STRUCTURED_PARSERS:
                raise HTTPException(
                    status_code=502,
                    detail=f"{PROVIDER} cli yielded no parsed answer: {exc}") from exc
            payload = {"result": stdout.strip()}

        # The CLI can exit 0 while reporting a limit inside the JSON envelope.
        # check_result_envelope is shared with mcp_bridge (same logic; see
        # that module's call site in _run_session_attempt).
        if (exc := check_result_envelope(payload)) is not None:
            raise exc

        return payload
    finally:
        # Reclaim the per-call cwd regardless of how the function exits.
        shutil.rmtree(spawn_cwd, ignore_errors=True)
        _cleanup_project_dir(spawn_cwd)


def upstream_error(stdout: str) -> tuple[int | None, str, bool | None]:
    """(status, message, retryable) from an OpenCode APIError envelope.

    OpenCode reports the provider's own answer:
      {"type":"error","error":{"name":"APIError","data":{
         "message":"personal-team-blocked:spending-limit: You have run out of
                    credits or need a Grok subscription...",
         "statusCode":403,"isRetryable":false}}}
    That is far better signal than pattern-matching the prose: 403 with
    isRetryable false means no amount of waiting will help, so it must not be
    dressed up as a usage limit that resets in an hour.
    """
    for evt in iter_json_objects(stdout):
        err = evt.get("error") if isinstance(evt.get("error"), dict) else {}
        data = err.get("data") if isinstance(err.get("data"), dict) else {}
        if not data:
            continue
        status = data.get("statusCode")
        retryable = data.get("isRetryable")
        message = data.get("message")
        if isinstance(status, int) or isinstance(message, str):
            return (status if isinstance(status, int) else None,
                    str(message or "")[:500],
                    retryable if isinstance(retryable, bool) else None)
    return None, "", None


def error_from_events(stdout: str) -> tuple[int | None, str]:
    """(upstream status, message) from a CLI's error events.

    Codex reports failures as `{"type":"error","message":"{...nested json...}"}`
    on **stdout**, leaving stderr with an unrelated informational line. Reading
    only stderr turned a clear "model is not supported" into an opaque 502.
    """
    status: int | None = None
    messages: list[str] = []
    for evt in iter_json_objects(stdout):
        item = evt.get("item") if isinstance(evt.get("item"), dict) else {}
        for candidate in (evt, item, evt.get("error") if isinstance(evt.get("error"), dict) else {}):
            if not isinstance(candidate, dict):
                continue
            if candidate.get("type") in ("error", "turn.failed") or candidate.get("message"):
                msg = candidate.get("message")
                if isinstance(msg, str) and msg.strip():
                    messages.append(msg.strip())
        # The message is often itself JSON carrying the HTTP status.
        for msg in list(messages):
            if msg.startswith("{"):
                try:
                    inner = json.loads(msg)
                except ValueError:
                    continue
                if isinstance(inner, dict):
                    if isinstance(inner.get("status"), int):
                        status = inner["status"]
                    err = inner.get("error") if isinstance(inner.get("error"), dict) else {}
                    if isinstance(err.get("message"), str):
                        messages.append(err["message"])
    unique = list(dict.fromkeys(m for m in messages if not m.startswith("{")))
    return status, " | ".join(unique)[:500]


def check_result_envelope(payload: dict, default_retry_after: int | None = None) -> HTTPException | None:
    """Classify a CLI's exit-0 JSON envelope as a real answer or a terminal error.

    Some CLI versions exit 0 even when the underlying run failed, putting the
    error inside the JSON body: ``is_error=true``,
    ``subtype in {"error_max_turns","error_during_execution"}``, or a
    usage-limit string in ``result``. The cli_bridge call site raises the
    returned exception directly; mcp_bridge reads it field-by-field
    (``status_code`` / ``detail`` / ``headers``) and resolves the session's
    ``turn_future`` instead. Returns ``None`` when the payload is a genuine
    answer.

    The ``default_retry_after`` mirrors ``_limit_error``'s: limit errors must
    carry ``Retry-After`` headers, and mcp_bridge passes its own provider
    profile's window because its PROFILE may front a different provider than
    this module was imported under (see ``_limit_error`` for the same caveat).
    """
    # The CLI can exit 0 while reporting a limit inside the JSON envelope.
    if payload.get("is_error") or payload.get("subtype") in ("error_max_turns", "error_during_execution"):
        text = json.dumps(payload)
        if _LIMIT.search(text):
            return _limit_error(text, default_retry_after)
        return HTTPException(status_code=502, detail=f"{PROVIDER} cli error: {text[:300]}")
    if _LIMIT.search(str(payload.get("result", ""))) and not payload.get("usage"):
        return _limit_error(str(payload.get("result")), default_retry_after)
    return None


def _limit_error(blob: str, default_retry_after: int | None = None) -> HTTPException:
    # Default to the plan's window; if the CLI names a reset time, trust it.
    # mcp_bridge passes its own provider profile's window explicitly, since it
    # may be running as a different PROVIDER than this module was imported
    # under (see the note on read_config for how the two stay independent).
    retry_after = default_retry_after if default_retry_after is not None else PROFILE["default_retry_after"]
    detail = f"{PROVIDER} usage limit reached"

    blob = normalise(blob)
    absolute = seconds_until(blob)
    if absolute:
        retry_after = absolute
        hours = absolute / 3600
        detail = f"{detail} (resets in {hours:.1f}h)"
    else:
        m = _RESET_AT.search(blob)
        if m:
            detail = f"{detail} (resets at {m.group(1)})"
    return HTTPException(
        status_code=429,
        detail={"error": {"message": detail, "type": "usage_limit_reached"}},
        headers={"Retry-After": str(retry_after)},
    )


def to_openai(payload: dict, model: str, finish_reason: str = "stop",
              reasoning_display: str | None = None) -> dict:
    """Render a CLI's completed payload as an OpenAI chat-completion response.

    Issue #292: the reasoning text the parser extracted (`payload["reasoning"]`)
    rides onto `choices[0].message.reasoning_content`, leaving `content` for the
    assistant's visible answer alone. `reasoning_content` is only emitted
    when the parser found reasoning text; its absence on a regular turn is
    itself the "no reasoning requested" signal, which is what LiteLLM and
    Anthropic-protocol adapters use to omit the field on the wire.

    `reasoning_display="omitted"` means the caller asked for the assistant
    text NOT to be rendered -- the reasoning IS returned, on
    `reasoning_content`, and `content` is left empty so a `text`-only consumer
    (a downstream chat client that ignores the field) sees nothing rather
    than the duplicated answer.
    """
    usage = payload.get("usage") or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    # Anthropic counts cache tokens separately from input_tokens:
    #   cache_read_input_tokens     -- served from the prompt cache
    #   cache_creation_input_tokens -- written to the cache this turn
    # Per OpenAI convention (and how the OpenCode/Codex parser above feeds
    # cache reads into input_tokens already) cached tokens are a *subset*
    # of prompt_tokens, so we only add them when the payload is clearly
    # Anthropic-shaped. Otherwise the OpenCode/Codex cache_read_tokens would
    # be double-counted into prompt_tokens.
    anthropic_shape = ("cache_read_input_tokens" in usage
                       or "cache_creation_input_tokens" in usage)
    if anthropic_shape:
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        prompt_tokens = input_tokens + cache_read + cache_creation
    else:
        prompt_tokens = input_tokens
    out_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
    }
    if anthropic_shape:
        # Expose the cache breakdown so LiteLLM and callers can see it. The
        # OpenAI-shaped `prompt_tokens_details.cached_tokens` is what the
        # gateway maps back to `cache_read_input_tokens` for Anthropic-protocol
        # clients (Claude Code via /v1/messages); there is no official OpenAI
        # slot for cache *creation*, so we carry it top-level using the
        # verbatim Anthropic spelling — LiteLLM round-trips it if it
        # recognises the key, and worst case drops it while totals stay
        # correct. Both are emitted even when zero so the shape is stable.
        out_usage["prompt_tokens_details"] = {"cached_tokens": cache_read}
        out_usage["cache_creation_input_tokens"] = cache_creation
    message: dict = {"role": "assistant"}
    if reasoning_display == "omitted":
        message["content"] = ""
    else:
        message["content"] = payload.get("result", "")
    reasoning_text = payload.get("reasoning")
    if isinstance(reasoning_text, str) and reasoning_text:
        message["reasoning_content"] = reasoning_text
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": out_usage,
    }


# --------------------------------------------------------------- plan usage ---
# Where the CLI writes its session transcripts. `claude -p "/usage"` runs the
# same slash command the TUI does and records the report it fetched, which is
# the only way to read plan headroom without calling Anthropic's API with the
# subscription's own token -- the thing a subscription's terms do not allow.
CLAUDE_PROJECTS = Path(os.environ.get(
    "CLAUDE_PROJECTS_DIR", str(Path.home() / ".claude" / "projects")))
USAGE_TIMEOUT = float(os.environ.get("USAGE_TIMEOUT_SECONDS", "120"))

# How often the streaming reply yields a SSE comment frame (`: keepalive`)
# while a turn is still running. The frame is comment-only, no `data:`, so
# the gateway's streaming collector (switchyard/hooks.py:1228-1240) skips
# it the same way it skips any other event without a payload. Long enough
# not to spam the wire (a 15 s chat on Claude is rare); short enough that
# a dropped connection is observed promptly. Patchable in tests via
# `cli_bridge.SSE_KEEPALIVE_SECONDS = 0.05`.
SSE_KEEPALIVE_SECONDS = float(os.environ.get("SSE_KEEPALIVE_SECONDS", "15"))


def _mangled_project_name(cwd: Path) -> str:
    """The Claude transcript-dir name for a given cwd.

    The CLI mangles any non-alphanumeric character in the absolute cwd to a
    hyphen when it picks the project dir. Same formula the mcp bridge uses
    for its session workdir (mcp_bridge.last_call_usage), so a CLI run and
    an mcp session that share a cwd share a project dir -- the cleanup helper
    below has to match that name exactly, or it would delete the wrong dir.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def _cleanup_project_dir(spawn_cwd: Path) -> None:
    """Best-effort: remove the Claude project dir for a per-call temp cwd.

    The Claude CLI writes `<CLAUDE_PROJECTS>/<mangled spawn_cwd>/...`
    alongside the per-call cwd that `--agent` saw as its working directory.
    Nothing on the CLI side reclaims those dirs, so without this helper each
    text-path spawn leaves one behind (issue #132). It is only safe to delete
    a directory that:

      * we derived from a spawn cwd this process just created via
        ``tempfile.mkdtemp`` -- not an arbitrary operator path, and
      * sits directly under ``CLAUDE_PROJECTS`` -- otherwise the helper is
        not the one that owns it.

    ``shutil.rmtree(..., ignore_errors=True)`` swallows the missing-dir case
    too, so a re-entrant or partial-run teardown is harmless.
    """
    if PROVIDER != "claude":
        return
    try:
        target = CLAUDE_PROJECTS / _mangled_project_name(spawn_cwd)
    except (OSError, ValueError):
        return
    if target.parent != CLAUDE_PROJECTS:
        return
    if not target.exists():
        return
    shutil.rmtree(target, ignore_errors=True)


def _find_usage_report(since: float) -> dict | None:
    """The newest `usageReport` written to a transcript after `since`.

    Transcripts are JSONL, one object per line, and the report is nested
    somewhere inside the record for the command that produced it — the exact
    depth has moved between versions, so it is searched for by key rather than
    by a fixed path.

    Bounded: iterates transcripts newest-first by mtime and returns on the
    first one that contains a valid `usageReport`. The `/usage` slash
    command produces exactly one such record per invocation, so reading more
    than the first hit was wasted I/O -- issue #132 boards-polling path
    spent tens of ms rglobbing before this bound landed.

    Sync, blocking I/O. The async callers (``usage_report``) wrap this in
    ``asyncio.to_thread`` so the rglob does not stall the event loop.
    """
    if not CLAUDE_PROJECTS.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    try:
        for root, _dirs, files in os.walk(CLAUDE_PROJECTS):
            for name in files:
                if not name.endswith(".jsonl"):
                    continue
                p = Path(root) / name
                try:
                    stamp = p.stat().st_mtime
                except OSError:
                    continue
                if stamp < since - 5:
                    continue
                candidates.append((stamp, p))
    except OSError:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    for _stamp, path in candidates:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "usageReport" not in text:
            continue
        for line in text.splitlines():
            if "usageReport" not in line:
                continue
            try:
                found = _dig_key(json.loads(line), "usageReport")
            except ValueError:
                continue
            if isinstance(found, dict) and found.get("rate_limits"):
                return found
    return None


def _dig_key(node, key: str):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _dig_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _dig_key(value, key)
            if found is not None:
                return found
    return None


CODEX_SESSIONS = Path(os.environ.get(
    "CODEX_SESSIONS_DIR", str(Path.home() / ".codex" / "sessions")))
# Codex rollouts are append-only JSONL the CLI writes on every call. The
# `rate_limits` block we mine is only as fresh as the last request, so the
# scan is the same on a board poll and on a real /usage request -- a tiny
# TTL cache amortises the rglob across the board's 1-Hz polling without
# making /usage stale.
CODEX_RATELIMITS_TTL_SECONDS = float(os.environ.get(
    "CODEX_RATELIMITS_TTL_SECONDS", "30"))
# Retention: rollouts older than this are deleted in the same to_thread pass
# that scans for rate_limits, so a busy seat doesn't accumulate years of
# JSONL. Default 48h mirrors the typical Codex weekly window plus a buffer;
# override via env when an operator wants a tighter or looser cut.
CODEX_ROLLOUT_RETENTION_HOURS = float(os.environ.get(
    "CODEX_ROLLOUT_RETENTION_HOURS", "48"))
_CODEx_RATELIMITS_CACHE: dict[str, tuple[float, dict | None]] = {}


def _prune_codex_rollouts() -> None:
    """Delete Codex session rollouts older than the retention window.

    Failures are logged at debug and swallowed -- a transient ENOSPC /
    EACCES on one file must not take the rate-limit scan with it, and the
    next call will retry the file anyway.
    """
    if not CODEX_SESSIONS.is_dir():
        return
    cutoff = time.time() - CODEX_ROLLOUT_RETENTION_HOURS * 3600.0
    try:
        for path in CODEX_SESSIONS.rglob("*.jsonl"):
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                log.debug("codex rollout prune skipped %s: %s", path, exc)
    except OSError as exc:
        log.debug("codex rollout prune walk failed under %s: %s",
                  CODEX_SESSIONS, exc)


def _codex_rate_limits() -> dict | None:
    """The newest `rate_limits` block Codex wrote to a session rollout.

    Codex records it on ordinary calls, so unlike Claude nothing has to be run
    to produce it — the cost is that it is only as fresh as the last request
    this sidecar made. `resets_at` says which window it describes, so a stale
    reading is still interpretable rather than silently wrong.

    Sync, blocking I/O. The async callers (``usage_report``) wrap this in
    ``asyncio.to_thread`` so the rglob does not stall the event loop. A tiny
    TTL cache amortises the scan across board polling -- a fresh request
    passes through, but the board's 1-Hz poll does not re-rglob on every hit.
    Keying on ``str(CODEX_SESSIONS)`` alone keeps the cache bounded: the
    TTL itself decides whether a hit is fresh, so including it would silently
    leak the old entry every time an operator flips the env var (reviewer
    finding, PR #322 cycle 2). ``CODEX_SESSIONS`` is captured at import, so
    it does not change for the life of the process.
    """
    key = str(CODEX_SESSIONS)
    now = time.monotonic()
    cached = _CODEx_RATELIMITS_CACHE.get(key)
    if cached is not None:
        expires, value = cached
        if expires > now:
            return value
    _prune_codex_rollouts()
    value = _scan_codex_rate_limits()
    _CODEx_RATELIMITS_CACHE[key] = (now + CODEX_RATELIMITS_TTL_SECONDS, value)
    return value


def _scan_codex_rate_limits() -> dict | None:
    """The non-cached core of `_codex_rate_limits` -- see that wrapper."""
    if not CODEX_SESSIONS.is_dir():
        return None
    files: list[tuple[float, Path]] = []
    try:
        for path in CODEX_SESSIONS.rglob("*.jsonl"):
            try:
                files.append((path.stat().st_mtime, path))
            except OSError:
                continue
    except OSError:
        return None
    files.sort(key=lambda pair: pair[0], reverse=True)
    for _stamp, path in files[:25]:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "rate_limits" not in text:
            continue
        for line in reversed(text.splitlines()):
            if "rate_limits" not in line:
                continue
            try:
                found = _dig_key(json.loads(line), "rate_limits")
            except ValueError:
                continue
            if isinstance(found, dict) and found.get("primary"):
                return {"rate_limits": found,
                        "observed_at": _stamp}
    return None


async def sse_from_completion(result: dict, model: str):
    """Re-emit a finished completion as a one-shot SSE stream.

    Neither CLI streams, so there is nothing to relay incrementally: the answer
    is complete before the first byte goes out. A caller that asked for
    `stream: true` still needs SSE framing, though — given a JSON body instead,
    an OpenAI client waits for events that never arrive and simply hangs.

    Tool calls are carried too. Without them a tool-using client streaming
    against a bridged plan sees an empty message and no finish_reason it
    recognises.

    Issue #292: a reasoning-enabled turn emits two deltas in order —
    `reasoning_content` first, then `content` — because the chat-completion
    streaming schema deltas the two fields on the same `choices[0]` and
    clients expect them in the same order on the wire. The terminal chunk
    carries the same fields as one frame so a client that reads `delta`
    rather than accumulating gets both halves.
    """
    choices = result.get("choices") or [{}]
    base = {"id": result.get("id"), "object": "chat.completion.chunk",
            "created": result.get("created"), "model": model}

    for choice in choices:
        message = choice.get("message") or {}
        first = {"role": "assistant"}
        reasoning_text = message.get("reasoning_content")
        if reasoning_text:
            # Reasoning first; the wire layout matches what LiteLLM emits for a
            # native reasoning turn on the chat-completion delta path, and
            # what's documented for the `reasoning_content` OpenAI extension.
            first["reasoning_content"] = reasoning_text
        if message.get("content"):
            first["content"] = message["content"]
        if message.get("tool_calls"):
            first["tool_calls"] = [
                {"index": i, "id": call.get("id"), "type": call.get("type", "function"),
                 "function": call.get("function", {})}
                for i, call in enumerate(message["tool_calls"])]
        index = choice.get("index", 0)
        yield f"data: {json.dumps({**base, 'choices': [{'index': index, 'delta': first, 'finish_reason': None}]})}\n\n"

    done = {**base,
            "choices": [{"index": choice.get("index", 0), "delta": {},
                          "finish_reason": choice.get("finish_reason") or "stop"}
                         for choice in choices],
            "usage": result.get("usage")}
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


async def sse_keepalive_stream(run_turn, model: str, *,
                              release_on_abandon=None):
    """An SSE stream that yields the assistant role delta immediately, then
    `: keepalive` comment frames every `SSE_KEEPALIVE_SECONDS` while `run_turn`
    runs, then re-emits the finished completion through `sse_from_completion`.

    Used by the streaming reply path on both bridges: the role delta is the
    first byte the caller sees, so a slow CLI no longer holds an
    unresponsive stream open. The `: keepalive` frames are SSE comments --
    no `data:` line -- which the gateway's streaming collector skips
    (switchyard/hooks.py:1228-1240: frames without `data:` are not
    payloads).

    Mid-turn HTTPException becomes a terminal `data: {"error": ...}` frame
    followed by `[DONE]`: once the first byte is out the response status
    code is fixed, so 4xx cannot be raised any more and the only honest
    signal to the caller is an error frame. Pre-stream validation must
    stay in the caller (before this generator runs) so known-bad requests
    still get real 4xx.

    On GeneratorExit (client disconnect) the underlying task is cancelled
    and awaited so the gate slot the caller was holding is released
    promptly rather than waiting on the CLI's idle TTL.

    `release_on_abandon` (optional) is an awaitable the caller passes when
    it acquired a gate slot pre-stream: the wrapper awaits it if the
    generator is closed BEFORE the turn task is created -- the realistic
    case is a client disconnect between `StreamingResponse` being returned
    and the first `body_iterator.__anext__()` (Starlette cancels the
    stream task before anything in the generator ever runs beyond the
    role-delta yield). Without this, the endpoint's pre-stream acquire
    leaks the slot: every release path lives downstream of
    `task = asyncio.create_task(run_turn())`, and a generator that never
    gets there has no `finally` to release. The task's own `finally`
    (when it does get to run) keeps doing its existing release on
    normal / cancelled / failed exit; this callback only fires for the
    pre-task-create window.

    Loop shape: the consumer pulls chunks via the body_iterator; between
    pulls we `asyncio.wait({task}, timeout=SSE_KEEPALIVE_SECONDS)` and yield
    a `: keepalive` comment only when the task is still running at the
    deadline. `asyncio.wait` wakes immediately when the task completes --
    a turn finishing at t=T emits its completion frames at t=T, not at the
    next interval boundary, so a 3-second turn no longer waits 15 s for the
    next keepalive slot (pre-PR shape: sleep-first loop delayed every
    streamed turn by up to `SSE_KEEPALIVE_SECONDS` of tail latency).
    The wait yields control to the event loop the same way `asyncio.sleep`
    does, so the task and the consumer both make progress between chunks
    (a fixture's `resolve_final`, the CLI's stdout writes, etc.). A fast
    turn therefore emits no keepalives -- one role delta, then the
    completion frames.

    `run_turn` is an async callable returning the completion dict that
    `sse_from_completion` understands -- the cli_bridge caller passes a
    closure that runs `_complete(body)` (with `merge_completions` and
    `finish_completion`); the mcp_bridge caller passes a closure that
    runs `handle_tool_request(...)` + `finish_completion(...)`.
    """
    base = {"id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model}
    # `task` starts unset so the generator's `finally` can tell whether
    # `run_turn` was ever scheduled. If the generator is closed at the
    # role-delta yield below (before `asyncio.create_task(run_turn())`
    # runs), the task is still None and the abandon-callback fires --
    # without that, the endpoint's pre-stream acquire leaks the slot.
    task = None
    try:
        # Role delta first -- a single byte that establishes the stream and
        # tells the caller "yes, this is a chat completion". A client that
        # times out waiting for this sees a 200 + first-byte-they-cannot-read,
        # the same shape it would see on any other successful streamed call;
        # the keepalives below keep the connection alive while we work.
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        task = asyncio.create_task(run_turn())
        while True:
            # Wait for the task with a deadline -- `asyncio.wait` wakes
            # immediately on completion, so a turn that finishes at t=T
            # emits its completion frames at t=T, not at the next
            # interval boundary (the previous sleep-first loop added up
            # to SSE_KEEPALIVE_SECONDS of tail latency to every streamed
            # turn). Yield a `: keepalive` comment only when the task is
            # still running at the deadline. The wait also yields control
            # to the event loop, so the turn task AND the consumer both
            # make progress between chunks (a fixture's `resolve_final`,
            # the CLI's stdout writes, etc.).
            done, _ = await asyncio.wait({task}, timeout=SSE_KEEPALIVE_SECONDS)
            if done:
                break
            yield ": keepalive\n\n"
        result = task.result()
    except asyncio.CancelledError:
        # Client disconnected (GeneratorExit path). Cancel the task so the
        # gate slot is released, then re-raise so the generator protocol
        # closes cleanly.
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        raise
    except HTTPException as exc:
        # Mid-turn failure: surface as an SSE error frame and [DONE].
        # The status code cannot be changed any more.
        # Unwrap one level if the detail is the in-bridge
        # `detail={"error": {...}}` envelope (image unsupported, 502
        # upstream, usage-limit, caller_environment_required, ...) -- the
        # repo's convention is `detail={"error": {...}}`, not
        # `detail={"error": {"error": {...}}}`. LiteLLM 1.102.1's
        # `_extract_error_from_chunk` reads `error.get("message")` and
        # `error.get("code")` off the OUTER dict; double-nesting makes
        # it find neither, raises `OpenAIError(status_code=500)` and
        # burns a TRANSIENT cooldown for a caller-fault 4xx. Pin the
        # status code on the frame so the parser maps it correctly.
        detail = exc.detail
        if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            detail = detail["error"]
        if not isinstance(detail, dict):
            detail = {"message": str(detail), "type": "http_error"}
        if "code" not in detail:
            detail = {**detail, "code": exc.status_code}
        yield f"data: {json.dumps({'error': detail})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as exc:
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': type(exc).__name__}})}\n\n"
        yield "data: [DONE]\n\n"
        return
    finally:
        # Pre-task-create abandon window: the generator was closed (a
        # client disconnect, a streaming middleware cancellation) at or
        # before the role-delta yield, so `run_turn` never started. The
        # endpoint acquired the slot pre-stream and is waiting for us
        # to release it -- the abandon callback is the only place that
        # can. When `task` is set, the task's own `finally` (and the
        # downstream release chain -- end_session in cli_bridge,
        # end_session / park_session in mcp_bridge) owns the release;
        # we do NOT fire the abandon callback or we would double-release.
        if task is None and release_on_abandon is not None:
            with contextlib.suppress(BaseException):
                await release_on_abandon()
        # Belt and braces for the post-task-create path: if we exited
        # without completing the task (client cancelled between the
        # wait_for timing out and our processing of the result), cancel
        # and await it now so its gate slot is released.
        elif task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task

    # Re-emit the finished completion through the unchanged sse_from_completion.
    # tool_calls, reasoning_content, and finish_reason all ride the same frames
    # they always did -- the only difference on the wire is the keepalives
    # before them.
    async for frame in sse_from_completion(result, model):
        yield frame


async def usage_report() -> dict:
    """What the vendor's own client last reported about this plan's headroom.

    Deliberately never a direct call to the vendor's usage API — Anthropic's
    api/oauth/usage or ChatGPT's backend-api/codex/usage — because reaching
    those from here means using the subscription's own OAuth token, which is
    what its terms do not allow. The CLI is asked instead, or its own records
    are read.
    """
    if PROVIDER == "codex":
        # Read CODEX_HOME at call time, not import time: `codex login --device-auth`
        # is run inside a running sidecar (TESTING.md:256), and the auth check
        # should not need a restart to notice. Compose pins CODEX_HOME for the
        # codex seat (secrets/codex), so this looks inside the mounted auth store.
        codex_auth = Path(os.environ.get(
            "CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
        # An empty auth.json is not a real login — its presence alone is a
        # stale file left behind by an interrupted `codex login`, and the
        # 503 path would otherwise mis-claim the seat is logged in. Lifted
        # from PR #208's `codex_logged_in()` body (PR #208 was closed as
        # superseded by #320 in the cycle-2 review). The PR body was kept
        # in step with the four-test supersession in cycle 3.
        try:
            logged_in = codex_auth.is_file() and codex_auth.stat().st_size > 0
        except OSError:
            logged_in = False
        if not logged_in:
            raise HTTPException(
                status_code=401,
                detail="codex cli is not logged in for this plan; "
                       "run `codex login --device-auth` in this sidecar")
        # Off-thread so the rglob + retention walk does not stall the event
        # loop on the board's 1-Hz /usage poll. `_codex_rate_limits` itself
        # wraps a TTL cache, so repeat calls inside the window are O(1).
        report = await asyncio.to_thread(_codex_rate_limits)
        if report is None:
            raise HTTPException(
                status_code=503,
                detail="no rate_limits recorded yet — codex writes them on a "
                       "real request, so send one through this plan first")
        return report
    if PROVIDER != "claude":
        raise HTTPException(status_code=501,
                            detail=f"no usage report implemented for {PROVIDER!r}")
    started = time.time()
# Spawn `claude -p /usage` in a fresh per-call temp cwd (same `sy-cli-`
    # prefix as the text path). Without this the CLI would write its
    # transcript to whatever the container's cwd is at the time -- usually
    # `/app`, which is also where the live code is -- and leave a project
    # dir behind under `~/.claude/projects/-app` on every /usage poll.
    # The matching project dir is cleaned up after the scan returns (or
    # raises), via `_cleanup_project_dir`. The try/finally wraps BOTH
    # `create_subprocess_exec` and the scan: a spawn that fails
    # (missing binary, wrong arch, renames mid-deploy) leaves a sy-cli-*
    # cwd behind if the cleanup only followed the spawn, so the finally
    # starts before the await (reviewer finding, PR #322 cycle 2).
    # Match the spawn shape used by `_run_cli_attempt`: the CLI is its own
    # session leader, so a timeout SIGKILLs the whole group (CLI plus any
    # grandchildren the inner harness forked). kill_process_group is the
    # matching helper.
    spawn_cwd = Path(tempfile.mkdtemp(prefix="sy-cli-"))
    try:
        proc = await asyncio.create_subprocess_exec(
            PROFILE["cli"], "-p", "/usage",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=spawn_cwd,
            env=subprocess_env(),
            start_new_session=True)
        try:
            _, err = await asyncio.wait_for(proc.communicate(), USAGE_TIMEOUT)
        except asyncio.TimeoutError:
            kill_process_group(proc)
            raise HTTPException(
                status_code=504,
                detail=f"/usage did not finish within {USAGE_TIMEOUT:.0f}s") from None
        if proc.returncode != 0:
            raise HTTPException(
                status_code=502,
                detail=f"/usage exited {proc.returncode}: "
                       f"{err.decode(errors='replace')[:300]}")
        # Scan off-thread so the rglob never stalls the event loop -- the
        # portal's board polls /usage at ~1 Hz, and a transcript tree that
        # grew for hours was enough to drop chunk frames.
        report = await asyncio.to_thread(_find_usage_report, started)
        if report is None:
            # The command ran but wrote nothing we recognise — a version change in
            # the transcript shape is the likely cause, and saying so beats
            # returning an empty report that reads as "no usage".
            raise HTTPException(
                status_code=502,
                detail="/usage produced no usageReport in the CLI's transcripts; the "
                       "transcript shape may have changed in this CLI version")
        return report
    finally:
        shutil.rmtree(spawn_cwd, ignore_errors=True)
        _cleanup_project_dir(spawn_cwd)


@app.get("/usage")
async def usage() -> dict:
    return await usage_report()


@app.get("/health")
async def health() -> JSONResponse:
    cfg = config()
    health_doc = {"ok": cfg.source == "config", "provider": PROVIDER,
                  "config_source": cfg.source, "supports_tools": False,
                  # All three profiles carry images in some form (see stage_images /
                  # build_argv in this file). A "false" here would mean: callers may
                  # still send image blocks, but the sidecar will drop them. Better
                  # honest than optimistic.
                  "supports_images": PROVIDER in ("claude", "codex", "opencode"),
                  "home": os.environ.get("HOME", ""), "warm": _warm.is_set(),
                  # Profile-driven: claude/opencode honor max_tokens post-hoc
                  # (truncate, finish_reason "length"); codex refuses with 400
                  # because its multi-turn agent loop makes truncation a lie.
                  # The reason key surfaces only on the refusing lane, so a
                  # caller can tell a future "streaming-only" refusal apart
                  # from the current one.
                  "enforces_max_tokens": bool(PROFILE.get("enforce_max_tokens")),
                  "system_mode": SYSTEM_MODE, "bare": BARE,
                  "plan": PLAN or PROVIDER, "model": cfg.model,
                  "models": sorted(cfg.models), "concurrency": cfg.concurrency,
                  "in_flight": _gate.in_flight, "config": PLANS_PATH}
    if not health_doc["enforces_max_tokens"]:
        reason = PROFILE.get("enforce_max_tokens_reason")
        if reason:
            health_doc["enforces_max_tokens_reason"] = reason
    return JSONResponse(health_doc, status_code=200 if health_doc["ok"] else 503)


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list",
            "data": [{"id": m, "object": "model", "owned_by": f"switchyard-{PROVIDER}"}
                     for m in sorted(config().models)]}


async def _handle_chat(body: dict):
    """The text-only completion path, factored out of the route so mcp_bridge
    can call it directly for a request with no `tools` — same gate, same
    config, same error classification, not a re-implementation of any of it.
    That identity is what "no regression on the text path" means here: there
    is only one code path for it, whichever sidecar is asking.

    `n` > 1 runs `_complete` n times one after another -- each takes and
    releases its own gate slot, so a lane with concurrency 1 serves it too
    and it never 429s against itself; `stop` and `response_format` are
    applied to the finished answer (finish_completion). Both are checked
    before anything spawns.

    Streaming path uses `sse_keepalive_stream`: it yields the assistant
    role delta immediately and a `: keepalive` comment frame every
    `SSE_KEEPALIVE_SECONDS` while the turn runs, then re-emits the finished
    completion through `sse_from_completion`. Pre-stream validation
    (request_n, validate_stop, response_schema) stays exactly where it is
    above -- known-bad requests still get real 4xx; only failures that
    surface AFTER the role delta has been sent become SSE error frames.

    The fail-fast gate acquire for the stream case happens HERE, before
    `StreamingResponse` is returned, so a saturated plan answers 429 with
    `Retry-After: 5` before any byte of the role delta lands on the wire
    -- the gate's documented "never queue, spill to the next plan"
    contract (`Gate.acquire`'s docstring + the comment above the
    `if not await _gate.acquire(limit): raise HTTPException(...)` lines)
    depends on that 429 reaching LiteLLM pre-stream. Occupancy is
    unchanged: the slot is held for the turn either way; the streaming
    closure just inherits it and releases in its own cleanup. The
    `_complete` call below receives `gate_already_held=True` so it
    doesn't try to acquire a second slot.
    """
    n = request_n(body)
    validate_stop(body)
    response_schema(body)
    if not body.get("stream"):
        result = merge_completions([await _complete(body) for _ in range(n)])
        result = finish_completion(result, body)
        return result

    limit = config().concurrency
    if not await _gate.acquire(limit):
        raise HTTPException(status_code=429,
                            detail=f"sidecar at capacity ({limit})",
                            headers={"Retry-After": "5"})

    async def _release_slot() -> None:
        # Belt-and-braces release: `_complete`'s `finally` releases
        # on every path where the task actually ran, but if the SSE
        # generator is abandoned BEFORE `run_turn`'s task is created
        # (a client disconnect between StreamingResponse being
        # returned and the first body_iterator __anext__) there is no
        # task and therefore no `_complete` finally. The wrapper's
        # abandon-callback fires this; without it the pre-stream
        # acquire leaks. Idempotent: `gate.release` max-clamps to 0,
        # so a double-release from a downstream path is harmless.
        await _gate.release()

    async def _run_turn() -> dict:
        # Same composition as the non-streaming path: n completions,
        # finish_completion. `_complete` skips its own acquire/release
        # because this closure owns the slot -- ownership was acquired
        # above so a saturated plan answers 429 pre-stream. The slot
        # is released in `finally` so a mid-turn HTTPException, a
        # generator exit, or a task crash all give the slot back.
        try:
            result = merge_completions(
                [await _complete(body, gate_already_held=True)
                 for _ in range(n)])
            return finish_completion(result, body)
        finally:
            await _gate.release()

    return StreamingResponse(
        sse_keepalive_stream(_run_turn, body.get("model") or "",
                             release_on_abandon=_release_slot),
        media_type="text/event-stream")


async def _complete(body: dict, *, gate_already_held: bool = False) -> dict:
    """One CLI run for a text-only request, as a non-streamed completion.

    `gate_already_held=True` is used by the streaming path: the endpoint
    acquired the slot before returning `StreamingResponse` so a saturated
    plan answers 429 pre-stream, and the streaming closure is responsible
    for releasing. In that mode this function neither acquires nor
    releases, and a mid-turn HTTPException does not unwind a slot it
    never took. Default (`False`) keeps the historical non-streamed
    behaviour intact (acquire here, release in the local `finally`).
    """
    # Refuse max_tokens on lanes that cannot honor it honestly (issue #70).
    # Placed BEFORE the tools_refusal block: a request carrying both max_tokens
    # AND tools should report the more specific max_tokens reason rather than a
    # generic tools-unsupported 400. Placement also guarantees the 400 is
    # raised before any stage_or_fail / flatten / gate / spawn, so no SSE
    # header is ever sent for a request we already know to refuse.
    if request_max_tokens(body) and not PROFILE.get("enforce_max_tokens", True):
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": "max_tokens is not enforceable on this lane",
                "type": "max_tokens_unenforceable",
                "plan": PLAN or PROVIDER,
                "reason": PROFILE.get("enforce_max_tokens_reason")}})

    # Refuse tool calls loudly. SwitchYard already routes these away from
    # CLI-backed plans; if one arrives anyway, dropping the definitions silently
    # would look like the model simply choosing not to call anything.
    # A server-side web-search tool is not a caller tool: it asks for the
    # CLI's own provider-side search (wants_web_search), so it is taken out
    # of `tools` rather than refused.
    # A server-side tool no CLI can run gets the more specific refusal, so
    # the router knows why it is spilling. Tool search is dropped outright.
    server_tools = server_only_tools(body)
    if server_tools:
        raise server_tool_refusal(server_tools)
    web = wants_web_search(body)
    if body.get("tools"):
        body = {**body, "tools": [t for t in body["tools"]
                                  if not (is_web_search_tool(t) or is_tool_search_tool(t))]}
    if body.get("tools"):
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": (f"{PROVIDER} is a CLI-backed plan and cannot serve tool "
                            "calls: the caller's tools would have nowhere to run. "
                            "Route tool-using requests to an API-keyed plan."),
                "type": "tools_unsupported"}})

    messages = body.get("messages") or []

    # Stage images BEFORE flatten, so the markers `[image N: <path>]` survive
    # into the prompt text and the per-CLI argv flags can carry the files.
    # A request with an unsupported image (a remote URL we cannot fetch) is
    # an error, never a silent drop — that is the bug fix in #30.
    try:
        image_paths, img_dir = await stage_or_fail_async(messages)
    except ImageUnsupportedError as exc:
        raise exc.http() from exc
    try:
        prompt, system = flatten(messages)
        if not prompt:
            raise HTTPException(status_code=400, detail="no usable message content")
        if image_paths:
            # The prompt is what makes the model LOOK at the staged files: the
            # markers on their own are just decoration.
            prompt += image_note(image_paths)
        fmt = response_schema(body)
        native = native_schema(fmt, images=bool(image_paths))
        if schema_instruction(fmt, native):
            prompt += "\n\n" + schema_instruction(fmt, native)
        model, warning = resolve_model(body.get("model"))
        if warning:
            # Loud, because silently running a weaker model than the lane asked for
            # would make an `apex` escalation quietly indistinguishable from `judge`.
            log.warning("%s", warning)

        # Resolve the caller's tool-execution environment. NO probe on the
        # text path: there is no tool round-trip, so the only signal is
        # whatever is in the request itself (stamped metadata or passive
        # parse), and the configured fallback_platform -- which is platform
        # only, NEVER a cwd.
        #
        # NOTE on `probe: required`: the text path cannot probe (no tool
        # round-trip), so a `required` config here would refuse every
        # request that does not already carry a passive environment. That
        # is not the right default for a permissive caller -- the operator
        # who set `required` wants loud refusal only on the mcp path,
        # where probing is possible. The text path treats `required` as
        # `auto`: render "unknown" wording if passive parse fails, never
        # 503. The mcp_bridge (which can probe) honours the full contract.
        if _caller_env is not None:
            ce_cfg = config().caller_environment
            if ce_cfg is None:
                if _models is None:
                    log.error("no caller_environment config AND switchyard.models "
                              "is not importable; CallerEnvironmentSettings "
                              "default cannot be built. Dockerfile.sidecar "
                              "must COPY switchyard/models.py.")
                else:
                    try:
                        ce_cfg = _models.CallerEnvironmentSettings()
                    except Exception as exc:
                        log.warning("could not build default CallerEnvironmentSettings: %s",
                                    exc, exc_info=True)
                        ce_cfg = None
            # Downgrade `required` -> `auto` for the text path only. Done
            # via a tiny shim rather than mutating the dataclass so the
            # other consumer (mcp_bridge) still sees the operator's value.
            resolve_cfg = ce_cfg
            if ce_cfg is not None and getattr(ce_cfg, "probe", "auto") == "required":
                from dataclasses import replace
                resolve_cfg = replace(ce_cfg, probe="auto")
            meta = (((body or {}).get("metadata") or {}).get("switchyard") or {}).get("caller_env")
            # NEVER trust a wire-stamped `source=config`: the only path
            # that produces source=config is the operator's plans.yaml
            # settings. See switchyard/caller_env.py:from_wire_metadata.
            env = _caller_env.from_wire_metadata(meta)
            if env is None and resolve_cfg is not None:
                try:
                    env = _caller_env.resolve(body, resolve_cfg)
                except _caller_env.CallerEnvironmentRequired:
                    # Cannot happen on the text path -- resolve_cfg.probe
                    # was just downgraded to `auto`. Kept as a defence
                    # in depth: if a future refactor changes the
                    # downgrade, the worst case here is still rendering
                    # the "unknown" wording rather than a 503 on text.
                    env = None
            if env is None:
                env = _caller_env.CallerEnvironment.unknown()
            env_block = _caller_env.render_system_block(env)
            # Append to the system string BEFORE profile flags so SYSTEM_MODE=replace
            # semantics are untouched: the block rides inside the caller's system
            # content, not on top of it. See fold_system and the profile's
            # system_args_replace in build_argv for why replace mode is the
            # only way this can fail loudly, not silently.
            system = (f"{system}\n\n{env_block}" if system else env_block)
            # Reminder on the first user turn, claude profile only --
            # the other profiles have no system-block + first-turn-mismatch
            # failure mode (their CLI's own Environment block lands at the
            # end of the prompt, not in the middle of a `# Environment`
            # trailer). Same first-turn rule as the mcp_bridge rebuild.
            if PROVIDER == "claude":
                reminder = _caller_env.render_first_turn_reminder(env)
                prompt = f"{reminder}\n\n{prompt}" if prompt else reminder

        limit = config().concurrency
        # `gate_already_held` is the streaming path's handoff: the
        # endpoint acquired the slot pre-stream so a saturated plan
        # answers 429 before any byte lands on the wire. In that mode
        # neither this acquire nor the matching release fires here --
        # the streaming closure inherits ownership and releases in
        # its own cleanup.
        if gate_already_held:
            thinking_policy = request_thinking(body)
            payload = await invoke(prompt, system, model, image_paths or None,
                                   fmt if native else None, web=web,
                                   effort=request_effort(body),
                                   thinking=thinking_policy)
        else:
            if not await _gate.acquire(limit):
                # Never queue: SwitchYard needs to hear "full" immediately so it can
                # spill to the next plan in the lane instead of holding a worker open.
                raise HTTPException(status_code=429,
                                    detail=f"sidecar at capacity ({limit})",
                                    headers={"Retry-After": "5"})
            thinking_policy = request_thinking(body)
            try:
                payload = await invoke(prompt, system, model, image_paths or None,
                                       fmt if native else None, web=web,
                                       effort=request_effort(body),
                                       thinking=thinking_policy)
            finally:
                await _gate.release()
        payload, finish_reason = enforce_max_tokens(
            payload, request_max_tokens(body))
        # `display` rides on the same policy we just read into `thinking_policy`
        # -- resolve it once here rather than calling `thinking_display(body)`,
        # which would re-walk every spelling of the carrier.
        display = (thinking_policy or {}).get("display")
        result = to_openai(payload, model, finish_reason=finish_reason,
                           reasoning_display=display)
    finally:
        # Image dir is owned here; the workdir on the mcp_bridge path is owned
        # by the session and outlives the request, so this only cleans up
        # fresh-request dirs. Stage failed before we get here in that case.
        if img_dir is not None:
            shutil.rmtree(img_dir, ignore_errors=True)
    return result


@app.post("/v1/chat/completions")
async def chat(request: Request):
    return await _handle_chat(await request.json())
