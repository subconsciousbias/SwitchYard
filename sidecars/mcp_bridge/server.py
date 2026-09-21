"""Inverted MCP bridge: lets a stateless OpenAI-compatible caller run its own
tool loop through a vendor CLI that is itself an agent harness.

Why this exists. README.md ("Tool capability is a per-plan property") and
cli_bridge/server.py's module docstring cover the underlying problem: `claude
-p` and `opencode run` are whole agent harnesses, not model endpoints, so they
own the tool loop and a caller's `tools` normally have nowhere to run.
Agentic work is mostly tool calling, so without this bridge a CLI-backed lane
(the only way to reach an OAuth subscription like Claude Max) is useless for
it.

THE TRICK, proven against a real `claude -p --mcp-config ...` round-trip
(mcp_bridge/probe_server.py, see TESTING.md): an MCP tool call can be *parked*
-- held open indefinitely -- without the CLI giving up, because from the CLI's
point of view the tool is just slow. So instead of executing a tool
ourselves:

  1. translate the caller's OpenAI `tools` into MCP tool definitions and start
     the CLI as a background task. Its `--mcp-config` points at
     tool_server.py, which the CLI spawns as its own child, wired back to
     THIS process over a loopback HTTP callback (see CALLBACK_BASE).
  2. when the model calls a tool, tool_server.py POSTs it to
     /internal/tools/call and blocks. We mint a tool_call_id (which embeds the
     session id -- see Session.mint_call_id), park a Future keyed by it, and
     answer the ORIGINAL http caller right away with finish_reason:
     tool_calls. The CLI subprocess and tool_server.py's POST are both still
     alive, waiting; nothing about them is torn down.
  3. the caller executes the tool(s) and sends the usual OpenAI follow-up
     (the assistant's tool_calls message plus one `tool` message per result).
     We find the parked session by the tool_call_id, resolve the matching
     Future(s), tool_server.py replies to the CLI over stdio, and the CLI
     continues -- possibly straight into another round of tool calls.
  4. repeat until the model emits text instead of a tool call, at which point
     the CLI process exits and we return an ordinary completion.

A request with no `tools` never touches any of this: chat() delegates it to
cli_bridge's own _handle_chat, unchanged. That identity -- not a parallel
implementation -- is what keeps the text path from regressing.

Provider coverage. `codex exec` has no MCP client, so this only runs as
PROVIDER=claude or PROVIDER=opencode; codex-backed tool calls still refuse the
way cli_bridge already refuses them. The claude profile below is the one
verified end to end (see TESTING.md); the opencode profile follows OpenCode's
documented `mcp:`/`agent.tools` config shape but was not exercised against a
live login while building this -- confirm it before trusting it in
production.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("mcp_bridge")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="switchyard-mcp-bridge")

# ---------------------------------------------------------------------------
# Reuse cli_bridge/server.py's argv/parsing/error-classification logic rather
# than duplicate it -- loaded by file path, not `import server`, because both
# sidecars ship a file named server.py and a plain import from two directories
# on sys.path would collide (and shadow whichever loaded second).
# ---------------------------------------------------------------------------
_CLI_BRIDGE_FILE = Path(__file__).resolve().parent.parent / "cli_bridge" / "server.py"
_spec = importlib.util.spec_from_file_location("switchyard_cli_bridge", _CLI_BRIDGE_FILE)
cli_bridge = importlib.util.module_from_spec(_spec)
sys.modules["switchyard_cli_bridge"] = cli_bridge
_spec.loader.exec_module(cli_bridge)   # builds its own (unmounted) FastAPI app too; harmless

PROVIDER = cli_bridge.PROVIDER          # same env var, read once, same value
HERE = Path(__file__).resolve().parent
TOOL_SERVER = HERE / "tool_server.py"
# The port THIS process listens on, so tool_server.py's child process (spawned
# by the CLI, possibly with a different cwd/user) can reach us at all times.
# Same env var cli_bridge uses, so compose does not need a second port number.
CALLBACK_BASE = f"http://127.0.0.1:{os.environ.get('SIDECAR_PORT', '8081')}"

# How long a session with no activity (no new tool call, no process exit) is
# tolerated before it is reaped. A caller that abandons mid-loop would
# otherwise hold a live CLI subprocess -- and its plan's concurrency slot --
# forever. Generous by design: a caller may be minutes into a build or test
# before it answers a tool call.
SESSION_TTL = float(os.environ.get("MCP_SESSION_TTL_SECONDS", "1800"))
REAP_INTERVAL = float(os.environ.get("MCP_REAP_INTERVAL_SECONDS", "30"))
# How often to check whether the http caller of the *current* turn has hung up.
# See await_turn: a caller that disappears mid-turn is the one abandonment we
# can actually observe, and it must not wait for the idle reaper.
DISCONNECT_POLL = float(os.environ.get("MCP_DISCONNECT_POLL_SECONDS", "2"))
# How long a session parked awaiting a follow-up is protected from preemption.
# Past it, a new request may take its slot rather than be told the sidecar is
# full. Parked sessions no longer hold a slot, so this only bounds how long a
# session may sit parked before enforce_parked_limit will consider it stale.
PARKED_GRACE = float(os.environ.get("MCP_PARKED_GRACE_SECONDS", "60"))
# How long the follow-up of a preempted session may wait for a slot on THIS
# plan before giving up. Only a resumption ever waits: see resume_gone_session.
#
# Bounded, not generous. A queued resumption holds a gateway plan slot for the
# full wait while doing no model work, and a queued rebuild of a pinned session
# starves that session's own follow-ups against the same plan (issue #14).
# 15s lets one in-flight turn make progress; anything longer just hands the
# caller time to retry into a slot we are still holding. Both queue sites share
# the knob; the 503 + Retry-After path on each returns control to SwitchYard
# immediately when the wait expires.
RESUME_WAIT = float(os.environ.get("MCP_RESUME_WAIT_SECONDS", "15"))
# A follow-up naming a session that is gone for any OTHER reason -- reaped
# after the idle TTL (a caller that walked away mid-loop), dropped when its
# caller hung up, crashed, hit the process timeout, or lost to a sidecar
# restart -- rebuilds exactly like a preempted one: the request itself carries
# the whole history, tool results included, so the loop can continue as if
# nothing happened and the client never learns the session died. Off switches
# the behaviour back to the old hard 410.
REBUILD_LOST = os.environ.get("MCP_REBUILD_LOST", "1") not in ("0", "false", "no")
# How many preempted session ids to remember, so their follow-ups can be
# recognised and resumed rather than refused. Ids are cheap; this only needs to
# outlive the callers' tool execution, not the process.
PREEMPTED_MEMORY = int(os.environ.get("MCP_PREEMPTED_MEMORY", "512"))
# How long a queued resumption sleeps between attempts to reclaim a slot from a
# parked session that has gone stale while it waited.
RECLAIM_POLL = float(os.environ.get("MCP_RECLAIM_POLL_SECONDS", "5"))
# A hard ceiling on the CLI subprocess's total lifetime, independent of the
# idle-based SESSION_TTL above -- belt and braces against a CLI that wedges
# without ever going idle (still POSTing keepalives, say).
PROCESS_TIMEOUT = float(os.environ.get("MCP_PROCESS_TIMEOUT_SECONDS", str(6 * 3600)))
# How long to wait for a second, third, ... parallel tool call to arrive
# before answering the http caller with whatever has been collected so far.
# The model can emit several tool_use blocks in one turn; MCP delivers them as
# separate tools/call frames, and nothing tells us in advance how many are
# coming, so we wait for the burst to go quiet.
BATCH_WINDOW = float(os.environ.get("MCP_BATCH_WINDOW_SECONDS", "0.25"))
# A single argv element may not exceed the kernel's MAX_ARG_STRLEN (128 KiB on
# Linux); a longer one makes create_subprocess_exec fail with
# "[Errno 7] Argument list too long" before the CLI even starts -- observed
# live when a folded system prompt + long history rode along as one argument.
# At or over this threshold the prompt is handed to the CLI on stdin instead.
# All three profile CLIs take it there: `claude -p` and `opencode run` read a
# piped prompt, `codex exec -` reads stdin when its PROMPT argument is `-`.
STDIN_PROMPT_LIMIT = int(os.environ.get("MCP_STDIN_PROMPT_LIMIT", "100000"))

MCP_PROFILES: dict[str, dict] = {
    "claude": {
        "cli": os.environ.get("CLAUDE_CLI", "claude"),
        # --allowed-tools pre-approves the named tools without needing
        # --permission-mode bypassPermissions, which matters twice over: it is
        # refused outright when running as root (every sidecar container
        # does), and an explicit allowlist is tighter anyway. Verified against
        # a real round-trip -- see TESTING.md for the transcript.
        "argv": ["-p", "{prompt}", "--model", "{model}",
                 "--mcp-config", "{mcp_config}",
                 "--allowed-tools", "{allowed_tools}",
                 "--disallowed-tools",
                 "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,NotebookEdit",
                 "--output-format", "json"],
        # Claude Code names an MCP server's tools mcp__<server>__<tool>; the
        # server name here (`switchyard`) must match the key used in
        # write_claude_mcp_config below.
        "tool_qualifier": lambda name: f"mcp__switchyard__{name}",
        "parser": "claude_json",
        "default_retry_after": 5 * 3600,
    },
    # Verified against a live `opencode run` on OpenCode **v1** (1.18.31, pinned
    # in Dockerfile.sidecar): the --dir project config below loads the local MCP
    # server, its tools arrive named switchyard_<name>, nothing prompts for
    # permission, and a 90s parked call completes.
    #
    # That last one only holds because tool_server.py sends
    # notifications/progress: OpenCode calls every tool with
    # `resetTimeoutOnProgress: true`, and without those notifications a 90s park
    # died with "MCP error -32001: Request timed out" at about 60s.
    #
    # OpenCode v2 renames all of this, so a version bump is not a drop-in:
    # `mcp` -> `mcp.servers`, `enabled: true` -> `disabled: false`,
    # `agent` -> `agents`, `prompt` -> `system`, `disable` -> `disabled`, and
    # `timeout` becomes `{catalog, execution}` -- which would replace the
    # progress keepalive with a plain execution timeout. v2 is not on npm as
    # `opencode-ai` (still 1.18.31 there; only @opencode/client is 2.x), so it
    # arrives by its own installer. Re-run the checks above before moving.
    "opencode": {
        "cli": os.environ.get("OPENCODE_CLI", "opencode"),
        "argv": ["run", "--model", "{model}", "--format", "json",
                 "--dir", "{workdir}", "--agent", "switchyard", "{prompt}"],
        # OpenCode's tool ids are <server>_<tool>; see write_opencode_dir.
        "tool_qualifier": lambda name: f"switchyard_{name}",
        "parser": "events_json",
        "default_retry_after": 3600,
    },
    # Codex DOES have an MCP client, contrary to what this file used to say:
    # `[mcp_servers.<name>]` in config.toml, settable per invocation with -c,
    # plus `codex mcp add/list/get`. Verified against a live `codex exec` on a
    # ChatGPT seat (0.155.1): the server starts, the model calls the tool, and a
    # 90s parked call completes in 101s wall clock.
    #
    # That wrong assumption is why the OpenAI seat was going to be reached
    # instead through `chatgpt.com/backend-api/codex/responses`, which a capture
    # of the real request shows requires presenting Codex CLI's own identity
    # (`originator: codex_exec`, `x-openai-internal-codex-responses-lite`, a
    # session/thread id pair and an `x-codex-turn-metadata` blob) to a private
    # internal endpoint. This path needs none of that: the seat's own official
    # client makes the call, exactly as it does on the text path.
    "codex": {
        "cli": os.environ.get("CODEX_CLI", "codex"),
        # --dangerously-bypass-approvals-and-sandbox is load-bearing and not
        # gratuitous: codex refuses an MCP tool call under any headless approval
        # policy ("MCP tool call requires approval, but approval policy is
        # never"), and its own docs scope this flag to externally sandboxed
        # environments -- which a sidecar container is. The alternative,
        # --approve-for-me, routes every call through an extra model review,
        # which is latency and quota per tool call on a high-volume path.
        #
        # The tool surface here is one MCP server that executes nothing locally:
        # it parks an HTTP call back to this process. Codex's own shell tools are
        # what the container boundary contains, the same as on the text path.
        "argv": ["exec", "--json", "--skip-git-repo-check",
                 "--dangerously-bypass-approvals-and-sandbox",
                 "--model", "{model}",
                 "-c", "mcp_servers.switchyard.command=python3",
                 "-c", 'mcp_servers.switchyard.args=["{tool_server}"]',
                 "-c", "mcp_servers.switchyard.startup_timeout_sec=20",
                 "-c", 'mcp_servers.switchyard.env={SWITCHYARD_TOOLS_FILE="{tools_file}",SWITCHYARD_SESSION_ID="{session_id}",SWITCHYARD_CALLBACK_URL="{callback}"}',
                 "{prompt}"],
        # Codex reports the server and the tool separately in its event stream
        # (`{"server":"switchyard","tool":"get_weather"}`) and takes no allowlist
        # argument, so nothing needs qualifying -- but the key must exist for
        # build_argv's shared allowlist construction.
        "tool_qualifier": lambda name: name,
        "parser": "codex_jsonl",
        "default_retry_after": 3600,
    },
}

if PROVIDER not in MCP_PROFILES:
    raise SystemExit(
        f"mcp_bridge supports PROVIDER in {sorted(MCP_PROFILES)}, got {PROVIDER!r}")
PROFILE = MCP_PROFILES[PROVIDER]


# ---------------------------------------------------------------------------
# Session bookkeeping. One Session per live CLI subprocess, spanning however
# many HTTP round-trips the caller's tool loop takes. It is intentionally the
# only piece of state mcp_bridge keeps: the conversation itself lives inside
# the still-running CLI process, which remembers its own turns, so a
# follow-up request only ever needs to deliver tool results, never replay text.
# ---------------------------------------------------------------------------
@dataclass
class ParkedCall:
    id: str
    name: str
    arguments: dict
    future: "asyncio.Future"
    parked_at: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    provider: str
    model: str
    workdir: str
    proc: "asyncio.subprocess.Process | None" = None
    pending: dict = field(default_factory=dict)     # call_id -> ParkedCall, awaiting the caller
    batch_ids: list = field(default_factory=list)    # calls parked but not yet surfaced
    turn_future: "asyncio.Future | None" = None
    seq: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    dead: bool = False
    # True between answering the caller with tool_calls and its follow-up
    # arriving. Nothing enforces a deadline on it -- the idle reaper does that --
    # but /health reports it, and a reap log says which state was abandoned.
    awaiting_followup: bool = False
    # Whether this session currently holds a concurrency slot. A parked session
    # does NOT: it is blocked on the caller's tool result and runs no inference,
    # so holding a slot would cap useful work at the number of loops in flight
    # rather than the number of model turns. The slot is taken again when the
    # follow-up arrives, waiting if the plan is busy.
    holds_slot: bool = False
    _flush_handle: object = None

    def touch(self) -> None:
        self.last_active = time.time()

    def mint_call_id(self) -> str:
        """call_<session_id>_<n> -- opaque to the caller, but round-trips the
        session id so a follow-up needs no other state to be correlated. See
        session_id_from_call_id for the inverse."""
        self.seq += 1
        return f"call_{self.id}_{self.seq}"

    def new_turn(self) -> "asyncio.Future":
        self.turn_future = asyncio.get_event_loop().create_future()
        return self.turn_future

    def enqueue(self, call: ParkedCall) -> None:
        """Add a parked call to the current batch and (re)start the quiet-
        period timer. Each new arrival pushes the timer back, so a burst of
        parallel tool calls all land in one batch instead of trickling out as
        several single-call turns."""
        self.pending[call.id] = call
        self.batch_ids.append(call.id)
        loop = asyncio.get_event_loop()
        if self._flush_handle is not None:
            self._flush_handle.cancel()
        self._flush_handle = loop.call_later(BATCH_WINDOW, self._flush)

    def _flush(self) -> None:
        if not self.batch_ids or self.turn_future is None or self.turn_future.done():
            return
        calls = [self.pending[cid] for cid in self.batch_ids]
        self.batch_ids = []
        self.turn_future.set_result({"type": "tool_calls", "calls": calls})

    def resolve_final(self, result: dict) -> None:
        if self.turn_future is not None and not self.turn_future.done():
            self.turn_future.set_result(result)

    def fail_pending(self, message: str) -> None:
        for call in list(self.pending.values()):
            if not call.future.done():
                call.future.set_exception(RuntimeError(message))
        self.pending.clear()
        self.batch_ids = []


SESSIONS: dict[str, Session] = {}
# session id -> when it was preempted. A follow-up naming one of these is not a
# caller error: we took its slot, so we owe it a resumption rather than a 410.
# Kept even though a lost session rebuilds regardless now (REBUILD_LOST): it
# keeps the log honest about WHICH loss the caller is resuming from.
PREEMPTED: "collections.OrderedDict[str, float]" = collections.OrderedDict()


def note_preempted(session_id: str) -> None:
    PREEMPTED[session_id] = time.time()
    while len(PREEMPTED) > PREEMPTED_MEMORY:
        PREEMPTED.popitem(last=False)


def session_id_from_call_id(call_id: str) -> str | None:
    if not call_id.startswith("call_"):
        return None
    rest = call_id[len("call_"):]
    session_id, sep, _seq = rest.rpartition("_")
    return session_id if sep else None


def translate_tool(openai_tool: dict) -> dict | None:
    """One OpenAI tool -> one MCP tool. `inputSchema` and `parameters` are
    near-identical JSON Schema, so this is close to a rename."""
    fn = openai_tool.get("function") if openai_tool.get("type") == "function" else openai_tool
    if not isinstance(fn, dict) or not fn.get("name"):
        return None
    return {
        "name": fn["name"],
        "description": fn.get("description") or "",
        "inputSchema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def translate_tools(openai_tools: list[dict]) -> list[dict]:
    return [t for t in (translate_tool(x) for x in (openai_tools or [])) if t]


def stringify_tool_content(content) -> str:
    """A `tool` message's content per the OpenAI schema: a string, or a list
    of content parts. Collapse either into the plain text tool_server.py hands
    back to the CLI as the tool's result."""
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content
                          if isinstance(b, dict) and b.get("type") == "text")
    return str(content) if content is not None else ""


def cleanup_workdir(workdir: Path) -> None:
    shutil.rmtree(workdir, ignore_errors=True)


def write_claude_mcp_config(workdir: Path, session_id: str, tools_path: Path) -> Path:
    cfg = {
        "mcpServers": {
            "switchyard": {
                "command": sys.executable,
                "args": [str(TOOL_SERVER)],
                "env": {
                    "SWITCHYARD_TOOLS_FILE": str(tools_path),
                    "SWITCHYARD_SESSION_ID": session_id,
                    "SWITCHYARD_CALLBACK_URL": CALLBACK_BASE,
                },
            }
        }
    }
    path = workdir / "mcp.json"
    path.write_text(json.dumps(cfg))
    return path


def write_opencode_dir(workdir: Path, session_id: str, tools_path: Path) -> None:
    """`opencode run --dir <workdir>` reads project config from that
    directory. This adds one local MCP server and an agent that exposes
    only its tools -- the same pattern cli_bridge/harness/opencode.json uses
    to gate the built-in ones. Verified against a live run on OpenCode v1; the
    keys are all renamed in v2 -- see MCP_PROFILES for the mapping."""
    cfg = {
        "mcp": {
            "switchyard": {
                "type": "local",
                "command": [sys.executable, str(TOOL_SERVER)],
                "environment": {
                    "SWITCHYARD_TOOLS_FILE": str(tools_path),
                    "SWITCHYARD_SESSION_ID": session_id,
                    "SWITCHYARD_CALLBACK_URL": CALLBACK_BASE,
                },
                "enabled": True,
            }
        },
        # Same explicit per-builtin disable list as
        # cli_bridge/harness/opencode.json's `switchyard` agent -- confirmed
        # against a real config there, unlike a wildcard key, which is not a
        # documented part of this schema. The MCP server's own tools are not
        # in this list, so they are not disabled by it.
        "agent": {"switchyard": {"tools": {
            "bash": False, "edit": False, "write": False, "read": False,
            "grep": False, "glob": False, "list": False, "patch": False,
            "todowrite": False, "todoread": False, "webfetch": False,
            "websearch": False, "task": False, "multiedit": False,
        }}},
    }
    (workdir / "opencode.json").write_text(json.dumps(cfg))


def build_argv(prompt: str, system: str | None, model: str, workdir: Path,
                session_id: str, tools_path: Path,
                allowed_tools: str) -> tuple[list[str], str | None]:
    """Build the CLI argv, plus the prompt to feed it on stdin (or None).

    Returns a pair so an oversized prompt can travel on stdin instead of argv
    (see STDIN_PROMPT_LIMIT for why). The caller passes the second element
    straight into run_session.
    """
    instructions: Path | None = None
    if PROVIDER == "claude":
        mcp_config = write_claude_mcp_config(workdir, session_id, tools_path)
        effective_prompt = prompt
    elif PROVIDER == "codex":
        # Codex takes its MCP server entirely on the command line (-c
        # mcp_servers.*), so there is no project config to write. The system
        # prompt goes through model_instructions_file, the same override
        # cli_bridge uses -- a real replacement, not an append, and the only key
        # that measurably cuts codex's own prompt.
        mcp_config = None
        effective_prompt = prompt
        if system:
            instructions = workdir / "instructions.md"
            instructions.write_text(system)
    else:
        write_opencode_dir(workdir, session_id, tools_path)
        mcp_config = None
        # No system-prompt mechanism in `opencode run` (see cli_bridge's
        # fold_system for the same fact on the text path) -- fold it into the
        # prompt rather than drop it.
        effective_prompt = f"{system}\n\n{prompt}" if system else prompt

    def fill(tpl: str) -> str:
        return (tpl.replace("{model}", model)
                   .replace("{workdir}", str(workdir))
                   .replace("{mcp_config}", str(mcp_config) if mcp_config else "")
                   .replace("{tool_server}", str(TOOL_SERVER))
                   .replace("{tools_file}", str(tools_path))
                   .replace("{session_id}", session_id)
                   .replace("{callback}", CALLBACK_BASE)
                   .replace("{allowed_tools}", allowed_tools))

    # The {prompt} slot is handled outside fill(): on the argv path it is
    # substituted directly, on the stdin path codex's "-" placeholder stays in
    # the template and claude/opencode drop the element entirely.
    use_stdin = len(effective_prompt) > STDIN_PROMPT_LIMIT
    argv = [PROFILE["cli"]]
    for element in PROFILE["argv"]:
        if element != "{prompt}":
            argv.append(fill(element))
        elif use_stdin and PROVIDER == "codex":
            argv.append("-")
        elif not use_stdin:
            argv.append(effective_prompt)
    if PROVIDER == "claude" and system:
        argv += ["--system-prompt", system]
    if instructions is not None:
        argv += ["-c", f"model_instructions_file={instructions}"]
    return argv, (effective_prompt if use_stdin else None)


# ---------------------------------------------------------------------------
# The background driver: one asyncio task per session, running the CLI
# subprocess for its whole lifetime. It never touches turn_future itself
# except at the very end (or on error/timeout) -- every tool-call turn is
# resolved by Session._flush, triggered from register_tool_call below.
# ---------------------------------------------------------------------------
async def run_session(session: Session, argv: list[str],
                      stdin_data: str | None = None) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        session.proc = proc
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(input=stdin_data.encode() if stdin_data is not None
                                 else None),
                timeout=PROCESS_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            session.resolve_final({"type": "error", "status": 408,
                                    "detail": f"{PROVIDER} cli timed out after "
                                              f"{PROCESS_TIMEOUT:.0f}s"})
            return

        stdout = out.decode(errors="replace")
        stderr = err.decode(errors="replace")
        blob = cli_bridge.normalise(f"{stdout}\n{stderr}")

        if proc.returncode != 0 or not stdout.strip():
            # Same classification cli_bridge._run_cli applies to the text
            # path, reused rather than re-derived -- see its comments for why
            # each check comes in this order.
            if cli_bridge._AUTH.search(blob):
                session.resolve_final({"type": "error", "status": 401,
                                        "detail": f"{PROVIDER} cli not authenticated"})
                return
            up_status, up_message, up_retryable = cli_bridge.upstream_error(stdout)
            if cli_bridge._LIMIT.search(blob) or cli_bridge.seconds_until(blob):
                exc = cli_bridge._limit_error(
                    f"{up_message}\n{blob}" if up_message else blob,
                    PROFILE["default_retry_after"])
                session.resolve_final({"type": "error", "status": exc.status_code,
                                        "detail": exc.detail, "headers": exc.headers})
                return
            if up_status in (401, 402, 403) and up_retryable is not True:
                session.resolve_final({"type": "error", "status": up_status,
                                        "detail": up_message or blob[:300]})
                return
            status, detail = cli_bridge.error_from_events(stdout)
            if status and 400 <= status < 500 and status != 429:
                session.resolve_final({"type": "error", "status": status, "detail": detail})
                return
            session.resolve_final({"type": "error", "status": 502,
                                    "detail": f"{PROVIDER} cli failed ({proc.returncode}): "
                                              f"{(detail or stderr[:300] or stdout[:300])}"})
            return

        try:
            payload = cli_bridge.parse_output(stdout, PROFILE["parser"])
        except (json.JSONDecodeError, ValueError) as exc:
            # Same contract as cli_bridge._run_cli: a structured parser that
            # cannot find the answer must fail the call, not echo its raw
            # event stream back as the answer (issue #3).
            if PROFILE["parser"] in cli_bridge.STRUCTURED_PARSERS:
                session.resolve_final({"type": "error", "status": 502,
                                        "detail": f"{PROVIDER} cli yielded no "
                                                  f"parsed answer: {exc}"})
                return
            payload = {"result": stdout.strip()}
        session.resolve_final({"type": "final", "payload": payload})
    except Exception as exc:                        # never leave a session parked forever
        log.exception("mcp session %s crashed", session.id)
        session.resolve_final({"type": "error", "status": 500, "detail": str(exc)})
    finally:
        session.touch()


async def end_session(session: Session) -> None:
    if session.dead:
        return
    session.dead = True
    SESSIONS.pop(session.id, None)
    cleanup_workdir(Path(session.workdir))
    if session.holds_slot:
        session.holds_slot = False
        await cli_bridge._gate.release()


async def park_session(session: Session) -> None:
    """Hand the ball to the caller and give the concurrency slot back.

    A parked session is blocked on a tool result and runs no inference, so
    holding a slot would cap useful work at the number of loops in flight
    rather than the number of model turns. It is still a live CLI process,
    though, so the number of parked sessions is capped separately.
    """
    session.awaiting_followup = True
    session.touch()                   # the clock starts when the caller gets the ball
    if session.holds_slot:
        session.holds_slot = False
        await cli_bridge._gate.release()
    await enforce_parked_limit(exclude=session.id)


async def unpark_session(session: Session) -> None:
    """Take a slot back before the CLI runs again, waiting if the plan is busy.

    This is the only place that queues. A follow-up has nowhere else to go --
    its tool_call_ids exist in this process alone -- so it waits rather than
    being refused, which is the opposite of a NEW request's contract.
    """
    session.awaiting_followup = False
    if session.holds_slot:
        return
    limit = cli_bridge.config().concurrency
    waited = time.time()
    if not await cli_bridge._gate.acquire_waiting(limit, RESUME_WAIT):
        raise HTTPException(
            status_code=503,
            detail=f"no slot freed within {RESUME_WAIT:.0f}s to resume this "
                   f"tool call; the plan is busy with other turns",
            headers={"Retry-After": "15"})
    session.holds_slot = True
    held = time.time() - waited
    if held > 1:
        log.info("session %s waited %.1fs for a slot to resume", session.id, held)


async def enforce_parked_limit(exclude: str | None = None) -> None:
    """Keep the number of parked sessions under the plan's limit.

    Parked sessions cost no concurrency but each is a CLI process holding
    memory and a connection pool, so they cannot accumulate without bound. The
    stalest goes first, and it is owed a resumption exactly like a preempted
    one -- its id is remembered and its follow-up rebuilds it.
    """
    limit = cli_bridge.config().parked_limit
    while True:
        parked = [x for x in SESSIONS.values()
                  if x.awaiting_followup and not x.dead and x.id != exclude]
        if len(parked) < limit:
            return
        victim = min(parked, key=lambda x: x.last_active)
        log.warning("parked sessions at the limit (%d); dropping the stalest, "
                    "%s (idle %.0fs)", limit, victim.id,
                    time.time() - victim.last_active)
        note_preempted(victim.id)
        await reap_session(victim)


async def await_turn(session: Session, request: "Request | None") -> dict:
    """Await this turn's result, abandoning the session if the caller hangs up.

    The parked state *between* turns has no open connection to watch: we
    answered with `tool_calls` and the caller will come back on a new request,
    or never. The one drop we can actually see is a caller that disappears
    while the CLI is still working — and such a session is worthless, since
    nobody will ever answer its tool calls. Left alone it would hold a live CLI
    subprocess and one of the plan's connection slots until the idle reaper came
    round half an hour later, which on a 2-connection plan is most of its
    capacity. So it is killed as soon as the disconnect is seen.
    """
    turn = session.turn_future
    if request is None:                       # direct in-process callers (tests)
        return await turn

    async def watch() -> None:
        while not turn.done():
            if await request.is_disconnected():
                log.warning("caller hung up mid-turn; dropping mcp session %s",
                            session.id)
                await reap_session(session)
                return
            await asyncio.sleep(DISCONNECT_POLL)

    watcher = asyncio.create_task(watch())
    try:
        return await turn
    except RuntimeError as exc:
        if session.dead and "reaped" in str(exc):
            # 499: nginx's "client closed request". Nobody is listening for it,
            # but it keeps the log honest about why the turn ended.
            raise HTTPException(status_code=499,
                                 detail="caller disconnected; session dropped") from exc
        raise
    finally:
        watcher.cancel()


async def reap_session(session: Session) -> None:
    log.warning("reaping abandoned mcp session %s (idle %.0fs%s, %d pending call(s))",
                session.id, time.time() - session.last_active,
                ", awaiting follow-up" if session.awaiting_followup else "",
                len(session.pending))
    session.fail_pending(f"session {session.id} reaped after {SESSION_TTL:.0f}s idle")
    if session.turn_future is not None and not session.turn_future.done():
        session.turn_future.set_exception(RuntimeError("session reaped"))
    if session.proc is not None and session.proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            session.proc.kill()
    await end_session(session)


async def reap_loop() -> None:
    while True:
        await asyncio.sleep(REAP_INTERVAL)
        now = time.time()
        for session in list(SESSIONS.values()):
            if not session.dead and now - session.last_active > SESSION_TTL:
                await reap_session(session)


@app.on_event("startup")
async def _start_reaper() -> None:
    asyncio.create_task(reap_loop())


# ---------------------------------------------------------------------------
# The internal callback tool_server.py posts to. register_tool_call is the
# actual parking primitive; the route is a thin JSON wrapper so tests can
# drive it without a real HTTP round-trip.
# ---------------------------------------------------------------------------
async def register_tool_call(session_id: str, name: str, arguments: dict) -> dict:
    session = SESSIONS.get(session_id)
    if session is None or session.dead:
        raise HTTPException(status_code=404, detail="unknown or reaped mcp_bridge session")
    call = ParkedCall(id=session.mint_call_id(), name=name, arguments=arguments,
                       future=asyncio.get_running_loop().create_future())
    session.touch()
    session.enqueue(call)
    # Never log arguments or results at info level -- they can carry
    # whatever the caller's tool handles, which is user data.
    log.debug("session %s parked tool call %s (%s)", session_id, call.id, name)
    try:
        return await call.future
    except RuntimeError as exc:
        raise HTTPException(status_code=504, detail=str(exc))


@app.post("/internal/tools/call")
async def internal_tools_call(request: Request):
    payload = await request.json()
    return await register_tool_call(payload.get("session_id"), payload.get("name") or "",
                                     payload.get("arguments") or {})


# ---------------------------------------------------------------------------
# The public OpenAI-compatible surface.
# ---------------------------------------------------------------------------
def render_turn(session: Session, result: dict, requested_model: str | None) -> dict:
    if result["type"] == "tool_calls":
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in result["calls"]]}
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
            "created": int(time.time()), "model": requested_model or session.model,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
            # Token accounting for the CLI-backed path is already approximate
            # (see cli_bridge); mid-loop it is not available at all.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    if result["type"] == "final":
        return cli_bridge.to_openai(result["payload"], requested_model or session.model)
    if result["type"] == "error":
        raise HTTPException(status_code=result["status"], detail=result["detail"],
                             headers=result.get("headers"))
    raise HTTPException(status_code=500,
                         detail=f"unexpected mcp_bridge turn result: {result['type']!r}")


async def handle_fresh(body: dict, tools: list[dict],
                       request: "Request | None" = None) -> dict:
    mcp_tools = translate_tools(tools)
    if not mcp_tools:
        raise HTTPException(status_code=400, detail="no usable tool definitions")

    prompt, system = cli_bridge.flatten(body.get("messages") or [])
    if not prompt:
        raise HTTPException(status_code=400, detail="no usable message content")
    model, warning = cli_bridge.resolve_model(body.get("model"))
    if warning:
        log.warning("%s", warning)

    limit = cli_bridge.config().concurrency
    if not await cli_bridge._gate.acquire(limit):
        # Never queue, same contract as cli_bridge: SwitchYard needs "full"
        # immediately so it can spill to the next plan in the lane. Only a
        # follow-up waits, because it has nowhere else to go.
        raise HTTPException(status_code=429, detail=f"sidecar at capacity ({limit})",
                             headers={"Retry-After": "5"})

    return await start_session(body, mcp_tools, prompt, system, model, request)


async def start_session(body: dict, mcp_tools: list[dict], prompt: str,
                        system: str | None, model: str,
                        request: "Request | None") -> dict:
    """Spawn a CLI session and run one turn. The gate slot is already held.

    Shared by a fresh request and a resumption, so the two cannot drift: the
    only difference between them is how the prompt was built and how the slot
    was obtained.
    """
    session_id = uuid.uuid4().hex
    workdir = Path(tempfile.mkdtemp(prefix=f"mcpb-{session_id[:8]}-"))
    # The slot was acquired by the caller before getting here; the session owns
    # it from now on, and gives it back when it parks or ends.
    session = Session(id=session_id, provider=PROVIDER, model=model,
                      workdir=str(workdir), holds_slot=True)
    try:
        tools_path = workdir / "tools.json"
        tools_path.write_text(json.dumps(mcp_tools))
        allowed = ",".join(PROFILE["tool_qualifier"](t["name"]) for t in mcp_tools)
        argv, stdin_data = build_argv(prompt, system, model, workdir, session_id,
                                      tools_path, allowed)
    except Exception:
        await cli_bridge._gate.release()
        cleanup_workdir(workdir)
        raise

    SESSIONS[session_id] = session
    session.new_turn()
    asyncio.create_task(run_session(session, argv, stdin_data))
    result = await await_turn(session, request)
    response = render_turn(session, result, body.get("model"))
    if result["type"] != "tool_calls":
        await end_session(session)
    else:
        await park_session(session)
    return response


def flatten_with_tool_history(messages: list[dict]) -> tuple[str, str | None]:
    """Collapse a conversation *including its tool loop* into one prompt.

    cli_bridge.flatten drops both halves of a tool turn: an assistant message
    carrying tool_calls has no content, and a `tool` message becomes a bare
    "Human: {json}" with nothing saying what it answers. That is fine for the
    plain text path, which never sees a tool loop, but useless for rebuilding a
    preempted session — the model would be handed results for calls it has no
    record of making.

    So the loop is rendered as narration the CLI can read: what was called, with
    what arguments, and what came back. The model is then in the same position
    as if it had made the calls itself, without needing the harness to replay
    native tool_use blocks it has no way to inject.
    """
    system: list[str] = []
    turns: list[str] = []
    # name calls by id so a result can say which call it belongs to, which
    # matters as soon as a turn made more than one.
    called: dict[str, str] = {}

    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(str(b.get("text", "")) for b in content
                                 if isinstance(b, dict) and b.get("type") == "text")
        content = (content or "").strip()

        if role == "system":
            if content:
                system.append(content)
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            parts = [content] if content else []
            for call in calls:
                fn = (call.get("function") or {})
                name = str(fn.get("name") or "tool")
                called[str(call.get("id"))] = name
                parts.append(f"[called {name}({fn.get('arguments') or '{}'})]")
            if parts:
                turns.append("Assistant: " + " ".join(parts))
        elif role == "tool":
            name = called.get(str(m.get("tool_call_id")), "tool")
            turns.append(f"[result of {name}: {stringify_tool_content(m.get('content'))}]")
        elif content:
            turns.append(f"Human: {content}")

    return "\n\n".join(turns), ("\n\n".join(system) or None)


async def acquire_resume_slot(limit: int, timeout: float) -> bool:
    """Wait for a slot to resume a rebuilt session.

    Nothing to reclaim here any more: parked sessions hold no slot, so the only
    thing worth waiting for is a model turn finishing.
    """
    return await cli_bridge._gate.acquire_waiting(limit, timeout)


async def resume_gone_session(body: dict, tools: list[dict], session_id: str,
                              request: "Request | None", why: str) -> dict:
    """Rebuild a lost session on the same plan, once a slot frees.

    This is the only path allowed to queue. A *new* request must still fail fast
    so SwitchYard can spill it to the next plan in the lane, but a session that
    is already mid-loop has nowhere to spill to: its tool results belong to a
    conversation this plan has been building a prompt cache for, and starting it
    over elsewhere would both lose that cache and re-run work the caller has
    already paid for. So it waits its turn here instead.

    "Lost" covers more than preemption: the idle reaper (a caller that walked
    away between meetings and answered the tool call past the TTL), a crashed
    CLI, the process timeout, a caller we saw hang up, and a sidecar restart
    that took every session with it. The rebuild is possible in every case for
    the same reason: the caller's request carries the whole history, tool
    results included, so `flatten_with_tool_history` can render the loop as
    narration for a fresh CLI session. What is genuinely lost is the CLI's own
    context -- the replayed history is longer than the tool results alone and
    the provider sees a new prompt prefix -- which is a quota cost, not a
    correctness one, and a fair price for the caller never seeing a 410.
    """
    mcp_tools = translate_tools(tools)
    if not mcp_tools:
        raise HTTPException(status_code=400, detail="no usable tool definitions")
    prompt, system = flatten_with_tool_history(body.get("messages") or [])
    if not prompt:
        raise HTTPException(status_code=400, detail="no usable message content")
    model, warning = cli_bridge.resolve_model(body.get("model"))
    if warning:
        log.warning("%s", warning)

    limit = cli_bridge.config().concurrency
    waited = time.time()
    if not await acquire_resume_slot(limit, RESUME_WAIT):
        raise HTTPException(
            status_code=503,
            detail=(f"no slot freed on this plan within {RESUME_WAIT:.0f}s to resume "
                    f"the {why} session"),
            headers={"Retry-After": "30"})
    log.info("resuming %s session %s after waiting %.1fs for a slot",
             why, session_id, time.time() - waited)
    return await start_session(body, mcp_tools, prompt, system, model, request)


async def handle_followup(body: dict, tool_msgs: list[dict],
                          request: "Request | None" = None) -> dict:
    """Deliver the caller's tool results to the parked session that minted the
    tool_call_ids; rebuild the session when it is gone, and never 400 the caller
    for a parse shape that is not their fault.

    Two non-1 cases used to raise 400 here: zero live sessions (the GUI's ids
    are foreign -- from a different MCP host -- or the session was lost), and
    several live sessions (a stuck retry, a multi-conversation GUI, a result
    batch replayed across two of our sessions). Both are recoverable the same
    way the file already recovers a lost session: the caller's request carries
    the whole tool loop, so it can be re-rendered as narration and replayed
    against a fresh CLI. The "client never sees an error" invariant this file
    is built on means doing exactly that, every time.

    `REBUILD_LOST=0` remains a kill-switch for ids we recognise as ours, so an
    operator can opt back into the old hard 410. For genuinely foreign ids the
    kill-switch does not apply -- there is no lost session of ours for the
    caller to recover from, only their own request -- so a rebuild fires either
    way and a warning is logged so it is visible rather than silent.
    """
    parsed = [(m, session_id_from_call_id(m["tool_call_id"])) for m in tool_msgs]
    session_ids = {sid for _, sid in parsed if sid}
    live_ids = [sid for sid in session_ids
                if (s := SESSIONS.get(sid)) is not None and not s.dead]

    if live_ids:
        wanted = live_ids[0]
        if len(live_ids) > 1:
            log.warning(
                "follow-up tool results span %d live mcp_bridge sessions (%s); "
                "resolving against %s and dropping the others' tool results",
                len(live_ids), live_ids, wanted)
        session_msgs = [m for m, sid in parsed if sid == wanted]
        # Dead-but-still-present ids (matched a session that has been reaped
        # since the GUI started assembling this batch) are treated like lost
        # ones here too: the rebuild from the request is the same and the
        # caller sees one answer either way.
        dead_ids = [sid for sid in session_ids
                    if (s := SESSIONS.get(sid)) is not None and s.dead]
        for sid in dead_ids:
            log.info("follow-up mentions a dead session %s alongside a live "
                     "one; replaying the dead session's tool results as part "
                     "of the live session's next turn", sid)
        return await _continue_followup(body, wanted, session_msgs, request)

    # No live session. Pick the most useful diagnostic id we have for the log
    # line and rebuild from the caller's request. `resume_gone_session` only
    # uses it for tracing -- the rebuilt session gets a fresh `uuid4`.
    diagnostic = next(iter(session_ids)) if session_ids else "unknown"
    if not session_ids:
        log.info(
            "follow-up carried no parsable mcp_bridge session ids; rebuilding "
            "from the request as a lost session rather than refusing")
    elif diagnostic in PREEMPTED or REBUILD_LOST:
        pass                    # the normal rebuild path -- silent on purpose
    else:
        # We did not mint these ids, but the operator opted out of rebuilds.
        # Refusing a foreign batch traps the client for nothing they did; a
        # rebuild from their request lets them continue, with a loud log.
        log.warning(
            "REBUILD_LOST=0 and follow-up carries foreign session id %s; "
            "rebuilding anyway so the client is not trapped", diagnostic)
    return await resume_gone_session(
        body, body.get("tools") or [], diagnostic, request,
        why=("preempted" if diagnostic in PREEMPTED
             else "reaped, foreign, or otherwise lost"))


async def _continue_followup(body: dict, wanted: str, tool_msgs: list[dict],
                             request: "Request | None") -> dict:
    """Resolve the tool results against a known-live session and continue the
    CLI turn. Pulled out of handle_followup so the live-vs-rebuild choice above
    does not nest two copies of the resolve/park dance.

    The set of `tool_msgs` here is already filtered to those whose embedded
    session id matches `wanted` -- any tool results for other live sessions or
    for foreign sessions were dropped by the caller, by design.
    """
    session = SESSIONS[wanted]
    resolved = 0
    for m in tool_msgs:
        call = session.pending.pop(m["tool_call_id"], None)
        if call is None:
            continue          # already resolved, or a stale id -- tolerate rather than fail
        if call.future.done():
            continue
        call.future.set_result({
            "content": [{"type": "text", "text": stringify_tool_content(m.get("content"))}],
            "isError": bool(m.get("is_error")),
        })
        resolved += 1
    if resolved == 0:
        # Nothing parked in this live session matched. Could be the GUI
        # replaying an already-delivered result, ids from a parallel batch we
        # already finished, or just garbled input. Rebuilding from the request
        # is at most one wasted turn of narration and keeps the caller
        # moving instead of throwing a 400 they cannot recover from.
        log.warning(
            "live mcp_bridge session %s had no parked tool call matching the "
            "delivered ids; rebuilding from the request so the caller is not "
            "trapped", wanted)
        return await resume_gone_session(
            body, body.get("tools") or [], wanted, request,
            why="no parked call matched delivered ids")

    # The results are in; the model is about to run again, so take a slot back.
    await unpark_session(session)
    session.touch()
    await session.new_turn()
    result = await await_turn(session, request)
    response = render_turn(session, result, body.get("model"))
    if result["type"] != "tool_calls":
        await end_session(session)
    else:
        await park_session(session)
    return response


async def handle_tool_request(body: dict, tools: list[dict],
                              request: "Request | None" = None) -> dict:
    tool_msgs = [m for m in (body.get("messages") or [])
                 if m.get("role") == "tool" and m.get("tool_call_id")]
    if tool_msgs:
        return await handle_followup(body, tool_msgs, request)
    return await handle_fresh(body, tools, request)


@app.get("/health")
async def health() -> dict:
    cfg = cli_bridge.config()
    return {"ok": cfg.source == "config", "provider": PROVIDER, "supports_tools": True,
            "config_source": cfg.source, "model": cfg.model, "models": sorted(cfg.models),
            "concurrency": cfg.concurrency, "in_flight": cli_bridge._gate.in_flight,
            "sessions": len(SESSIONS),
            "awaiting_followup": sum(1 for x in SESSIONS.values()
                                     if x.awaiting_followup and not x.dead),
            "parked_limit": cfg.parked_limit,
            "session_ttl_seconds": SESSION_TTL}


@app.get("/usage")
async def usage() -> dict:
    """Same report as the text bridge: the CLI's own /usage, via its transcript.
    Registered on both apps because a plan on BRIDGE=mcp serves from this one."""
    return await cli_bridge.usage_report()


@app.get("/v1/models")
async def models() -> dict:
    return await cli_bridge.models()


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    tools = body.get("tools")
    if not tools:
        # Identical to the plain CLI shim: this literally calls its function,
        # not a re-implementation of it, so there is nothing here to regress.
        # That path already frames a streamed reply.
        return await cli_bridge._handle_chat(body)

    result = await handle_tool_request(body, tools, request)
    if not body.get("stream"):
        return result
    # A caller that asked for SSE and got a JSON body does not error -- it waits
    # for events that never arrive. That is a client hanging with no log line
    # anywhere, and each hung attempt parks a session holding a connection until
    # the plan reports itself full.
    return StreamingResponse(
        cli_bridge.sse_from_completion(result, result.get("model") or ""),
        media_type="text/event-stream")
