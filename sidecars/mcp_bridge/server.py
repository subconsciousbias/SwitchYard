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

from typing import Any

import asyncio
import collections
import contextlib
import errno
import importlib.util
import json
import logging
import os
import re
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

# switchyard/caller_env.py: per-request resolution of the caller's tool-
# execution environment (issue #44). Imported as a module so a test can run
# WITHOUT the package being on sys.path -- see /app/switchyard in production
# (PYTHONPATH=/app) and tests/_caller_env_loader below for the same fallback
# as the bridges themselves.
try:
    import switchyard.caller_env as _caller_env
except ImportError:
    _caller_env_file = Path(__file__).resolve().parent.parent.parent / "switchyard" / "caller_env.py"
    if _caller_env_file.exists():
        _ce_spec = importlib.util.spec_from_file_location(
            "switchyard.caller_env", _caller_env_file)
        _caller_env = importlib.util.module_from_spec(_ce_spec)
        _caller_env.__package__ = "switchyard"
        sys.modules.setdefault("switchyard", type(sys)("switchyard"))
        sys.modules["switchyard.caller_env"] = _caller_env
        _ce_spec.loader.exec_module(_caller_env)
    else:
        _caller_env = None

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
# Issue #78: one warning per no-text event -- one when the retry is scheduled,
# one when the contract 502 fires -- so retry-rate vs contract-502-rate is
# visible in docker logs without touching the gateway. Opt out with
# LOG_TEXT_LOST=0/false/no (same shape as MCP_REBUILD_LOST above).
LOG_TEXT_LOST = os.environ.get("LOG_TEXT_LOST", "1") not in ("0", "false", "no")
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
# Same argv-limit knob and byte check as cli_bridge, re-exported rather
# than duplicated: both bridges must agree on what "too big for argv"
# means (MAX_ARG_STRLEN, ~128 KiB per element -- issue #29), and two
# copies of either the number or the logic would drift. cli_bridge owns
# both; the MCP_STDIN_PROMPT_LIMIT env var is documented there.
STDIN_PROMPT_LIMIT = cli_bridge.STDIN_PROMPT_LIMIT
over_argv_limit = cli_bridge.over_argv_limit

MCP_PROFILES: dict[str, dict] = {
    "claude": {
        "cli": os.environ.get("CLAUDE_CLI", "claude"),
        # --allowed-tools pre-approves the named tools without needing
        # --permission-mode bypassPermissions, which matters twice over: it is
        # refused outright when running as root (every sidecar container
        # does), and an explicit allowlist is tighter anyway. Verified against
        # a real round-trip -- see TESTING.md for the transcript.
        #
        # --disallowed-tools is NOT in the template: Claude Code concatenates
        # flag values across repeated flags rather than overriding the first,
        # so a Read-allowing image request would still be denied Read. The list
        # is built per request by build_argv (Read dropped when images are
        # present, exactly like cli_bridge's bare_args_images pattern).
        "argv": ["-p", "{prompt}", "--model", "{model}",
                 "--mcp-config", "{mcp_config}",
                 "--allowed-tools", "{allowed_tools}",
                 "--output-format", "json"],
        # Shared with build_argv when it has to emit --disallowed-tools. Read
        # is removed in the image variant; everything else is the same.
        "disallowed_tools_default":
            "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,NotebookEdit",
        "disallowed_tools_images":
            "Bash,Edit,Write,Glob,Grep,WebFetch,WebSearch,NotebookEdit",
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
        # Issue #121: opencode's argv parser greedily consumes a `-FLAG`
        # prompt as an option. build_argv reads this key and inserts the
        # sentinel on the argv path immediately before the prompt element,
        # so the next element is unambiguously a positional. Skipped on
        # the stdin path -- there the prompt is fed via stdin (or, for
        # codex, the `"-"` placeholder already in argv), and there is no
        # argv element to be parsed as a flag. Adding `"--"` directly to
        # the argv template is not enough: build_argv substitutes `{prompt}`
        # outside `fill()`, and only this codepath runs the insertion.
        "prompt_terminator": "--",
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
        # The flag alone is not enough any more (issue #116): codex 0.155.1
        # has a built-in shell tool that, even inside this container, can run
        # `id`, `env`, and read ~/.codex/auth.json -- any one of which would
        # exfiltrate the OAuth grant mounted for the ChatGPT seat. So the
        # flag is paired with explicit -c overrides:
        #   tools.web_search=false           strips the web-search tool
        #   -c sandbox_mode="read-only"      neutralises the built-in shell
        # -c fragments are applied in argv order, so they sit AFTER the
        # bypass flag and WIN over its implicit `danger-full-access`. The
        # four `mcp_servers.switchyard.*` lines below follow.
        #
        # FALLBACK (issue #116 No-Go path): if a live MCP-path request still
        # parks an MCP tool call but built-in `id` / `env` / cat
        # ~/.codex/auth.json attempts are NOT refused (i.e. the sandbox_mode
        # override loses to the bypass flag on this codex version), the
        # codex MCP path is unsafe to keep. The operator action is to flip
        # codex-sidecar's `BRIDGE: mcp` -> `BRIDGE: cli` in docker-compose.yml
        # (owned by the workstream-2 docker-compose change); the text path's
        # built-in shell is the same codex, the same container, but it is
        # already blocked from running tools by the cli_bridge gate, so it is
        # safe.
        "argv": ["exec", "--json", "--skip-git-repo-check",
                 "--dangerously-bypass-approvals-and-sandbox",
                 "-c", "sandbox_mode=\"read-only\"",
                 "-c", "tools.web_search=false",
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
    # tool_call_ids the session has delivered to tool_server.py (i.e. consumed
    # from `pending` and handed back to the CLI), keyed by resolution time.
    # Used to tell a *duplicate* delivery ("caller retried a request whose
    # response was lost") apart from a *lost* session ("nothing parked here
    # matches those ids") -- see _continue_followup. Bounded so a long-running
    # session does not accumulate forever; 256 covers a generous tool-loop
    # burst and is well under the size of any plausible retry window.
    resolved_recently: dict = field(default_factory=dict)
    # The most recent chat-completion response rendered for this session.
    # Returned verbatim when a duplicate follow-up arrives: the caller is
    # retrying because the previous response never arrived, not asking for a
    # new turn. None until the first render.
    last_response: dict | None = None
    _flush_handle: object = None

    def touch(self) -> None:
        self.last_active = time.time()

    def mark_resolved(self, call_id: str) -> None:
        """Record that `call_id` has been delivered to tool_server.py.

        A subsequent follow-up that names this id again is a duplicate
        delivery -- the caller's retry of a request whose response never
        arrived, or a GUI replaying an already-consumed batch. Without this
        record, the duplicate would look like a lost session and trigger a
        rebuild (issue #13, defect 1).
        """
        self.resolved_recently[call_id] = time.time()
        while len(self.resolved_recently) > 256:
            oldest = min(self.resolved_recently, key=self.resolved_recently.get)
            self.resolved_recently.pop(oldest, None)

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


# Caller-environment probe bookkeeping. See switchyard/caller_env.py for the
# protocol; this file just remembers enough to be idempotent across retries
# without depending on a sidecar-local "pending" dict that a restart would
# lose (the id itself is the primary key -- the fingerprint embedded in it is
# recoverable with empty in-memory state).
#
# RESOLVED_PROBES:  fingerprint -> CallerEnvironment. Set when a probe result
#                   came back successfully and the env was parsed.
# FAILED_PROBES:    fingerprint -> "failed" sentinel. Set when a probe result
#                   came back isError / unparseable, or the caller visibly
#                   refused to run the synthetic call. Both are bounded.
# Both maps are bounded so a long-running sidecar cannot grow them forever;
# a probe's relevance is short (it is the first session of a fingerprint),
# not the lifetime of the process.
RESOLVED_PROBES: dict[str, Any] = {}
FAILED_PROBES: dict[str, str] = {}
PROBE_CACHE_LIMIT = int(os.environ.get("MCP_PROBE_CACHE_LIMIT", "512"))

# Sentinel published into RESOLVED_PROBES the moment we mint a probe,
# before releasing PROBE_LOCK. A concurrent same-fingerprint caller
# arriving after the lock is released sees fp in RESOLVED_PROBES and
# skips the probe path entirely, so we mint exactly one probe per
# fingerprint even under asyncio.gather. The sentinel is overwritten
# with the real CallerEnvironment when the probe result comes back
# through `_consume_probe_results`. Without the lock, two concurrent
# requests both pass the `fp in {FAILED,RESOLVED}_PROBES` checks and
# both mint duplicate probes for the same session.
_PROBE_PENDING = object()
PROBE_LOCK = asyncio.Lock()


def remember_probe_env(fp: str, env: Any) -> None:
    RESOLVED_PROBES[fp] = env
    while len(RESOLVED_PROBES) > PROBE_CACHE_LIMIT:
        RESOLVED_PROBES.pop(next(iter(RESOLVED_PROBES)))


def mark_probe_failed(fp: str, reason: str = "failed") -> None:
    FAILED_PROBES[fp] = reason
    while len(FAILED_PROBES) > PROBE_CACHE_LIMIT:
        FAILED_PROBES.pop(next(iter(FAILED_PROBES)))


def session_id_from_call_id(call_id: str) -> str | None:
    if not call_id.startswith("call_"):
        return None
    rest = call_id[len("call_"):]
    session_id, sep, _seq = rest.rpartition("_")
    return session_id if sep else None


# ---------------------------------------------------------------------------
# Caller-environment resolution at the mcp_bridge layer. The gateway already
# tries to stamp this on the request's metadata (`metadata.switchyard.caller_env`),
# but the gateway->sidecar transport is not yet verified to forward every
# metadata field at runtime -- passive re-resolution from the request body is
# the primary source, the stamp is belt-and-braces. See hooks.py where the
# stamp is added.
# ---------------------------------------------------------------------------
def _caller_env_settings() -> Any:
    """The active CallerEnvironmentSettings, or a sensible default.

    Reads the field off the cli_bridge Config when present (plans.yaml
    loaded successfully); otherwise returns a CallerEnvironmentSettings
    with `probe=auto` -- the field is always defined, the question is
    only what the operator set.

    `_caller_env.CallerEnvironmentSettings` does NOT exist (caller_env.py
    owns runtime resolution, not the typed settings class -- the class
    lives in switchyard.models.py). Build the default from
    cli_bridge._models instead, which cli_bridge loads via the same
    package / path-load idiom this module already uses for _caller_env.
    """
    if _caller_env is None:
        return None
    cfg = cli_bridge.config()
    settings = getattr(cfg, "caller_environment", None)
    if settings is not None:
        return settings
    models = getattr(cli_bridge, "_models", None)
    if models is None:
        log.error("no caller_environment config AND switchyard.models is not "
                  "importable via cli_bridge; CallerEnvironmentSettings default "
                  "cannot be built. Dockerfile.sidecar must COPY "
                  "switchyard/models.py.")
        return None
    try:
        return models.CallerEnvironmentSettings()
    except Exception as exc:
        log.warning("could not build default CallerEnvironmentSettings: %s",
                    exc, exc_info=True)
        return None


def _resolve_env(body: dict, *, known: Any = None) -> Any:
    """Stamped env first (if any of the trusted sources), else passive parse,
    else unknown. Caller may pass `known` (e.g. from probe or cache) to skip
    parsing and trust the value -- the contract is "I already know this"."""
    if known is not None:
        return known
    if _caller_env is None:
        return None
    meta = (((body or {}).get("metadata") or {}).get("switchyard") or {}).get("caller_env")
    # NEVER trust a wire-stamped `source=config`: the only path that
    # produces source=config is the operator's plans.yaml settings.
    # Values still flow through at the request tier. See
    # switchyard/caller_env.py:from_wire_metadata.
    parsed = _caller_env.from_wire_metadata(meta)
    if parsed is not None:
        return parsed
    return _caller_env.parse_request(body)


def _consume_probe_results(body: dict) -> tuple[Any, dict]:
    """Pop synthetic probe exchanges out of `body['messages']`.

    Returns (parsed_env_or_None, body_without_synthetic). Idempotent on a
    re-delivery of the same probe id: a probe result that was already
    consumed is just absent from the returned body, and the cached env
    is returned again. A failed / isError result marks the fingerprint
    failed and returns None, so the caller falls back to the unknown-env
    path with no retry.
    """
    if _caller_env is None:
        return None, body
    new_messages: list[dict] = []
    parsed_env: Any = None
    for msg in (body.get("messages") or []):
        if msg.get("role") != "tool":
            new_messages.append(msg)
            continue
        call_id = msg.get("tool_call_id") or ""
        fp = _caller_env.parse_probe_call_id(call_id)
        if fp is None:
            new_messages.append(msg)
            continue
        # isError from the caller is a refusal; cache the failure and
        # drop both halves of the synthetic exchange.
        is_err = bool(msg.get("is_error"))
        parsed = None if is_err else _caller_env.parse_probe_result(msg.get("content"))
        if parsed is None:
            mark_probe_failed(fp, "iserror_or_unparseable" if is_err else "unparseable")
            log.info("mcp_bridge probe %s -> %s; falling back without retry",
                     call_id, "iserror" if is_err else "no recognised cwd=/platform=/shell= lines")
        else:
            remember_probe_env(fp, parsed)
            log.info("mcp_bridge probe %s -> env: platform=%r cwd=%r shell=%r",
                     call_id, parsed.platform, parsed.cwd, parsed.shell)
            parsed_env = parsed
    body = dict(body)
    body["messages"] = new_messages
    return parsed_env, body


def _strip_synthetic_assistant(messages: list[dict]) -> list[dict]:
    """Drop the assistant tool_call message we minted on the previous turn.

    The probe mints an assistant message carrying one tool_call with a
    `switchyard_env_` id. The CLI must never see it -- it is a relay
    bookkeeping artefact, not model output -- so it is removed here.
    """
    if _caller_env is None:
        return list(messages)
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            tcs = msg.get("tool_calls") or []
            if tcs and all((tc.get("id") or "").startswith(_caller_env.PROBE_PREFIX)
                           for tc in tcs):
                continue
        out.append(msg)
    return out


def _maybe_probe(body: dict, tools: list[dict]) -> "asyncio.Future | None":
    """If the env is still unknown AND a probe is allowed AND we have a
    recognisable command tool, mint a synthetic tool_calls response.

    Returns a Future that resolves to the rendered chat.completion dict
    on a hit, None otherwise. The caller `await`s it. The coroutine
    shape is required for the lock: the check-and-mint pair is held
    under PROBE_LOCK so two concurrent same-fingerprint requests
    cannot both mint duplicate probes (issue #66 review).

    Does NOT take a gate slot or a session slot: the probe is a tiny
    JSON response that travels back to the caller, who executes it
    locally and answers on the next request. The next request is what
    builds a real session.
    """
    async def _run() -> dict | None:
        if _caller_env is None:
            return None
        cfg = cli_bridge.config()
        ce_settings = getattr(cfg, "caller_environment", None)
        if ce_settings is None:
            return None
        if getattr(ce_settings, "probe", "auto") == "disabled":
            return None
        messages = body.get("messages") or []
        env = _resolve_env(body)
        if env is not None and env.source != "unknown":
            return None
        fp = _caller_env.fingerprint(messages)
        # Atomic check+mint. Without the lock, two concurrent
        # same-fingerprint callers both pass the `fp not in caches`
        # checks below and both mint duplicate probes; the second
        # is silently dropped at consume-time but still consumed a
        # slot on the way out and polluted the caller's history.
        # With the lock, the first caller publishes a PENDING
        # sentinel into RESOLVED_PROBES before releasing; the
        # second caller sees fp already-taken and returns None.
        async with PROBE_LOCK:
            if fp in FAILED_PROBES:
                return None
            if fp in RESOLVED_PROBES:
                # Cache hit but the env was set on the request --
                # nothing to do.
                return None
            tool = _caller_env.find_command_tool(tools)
            if tool is None:
                return None
            name, arg_key = tool
            call_id = _caller_env.mint_probe_call_id(fp)
            # Publish PENDING so a concurrent caller arriving after
            # we release the lock sees fp as already-taken. The real
            # env overwrites this when _consume_probe_results runs.
            remember_probe_env(fp, _PROBE_PENDING)
        # Build the response OUTSIDE the lock: nothing in it touches
        # shared state, and no other caller is blocked behind us.
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": name,
                          "arguments": json.dumps({arg_key: _caller_env.PROBE_COMMAND})}}
        ]}
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or "",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return asyncio.ensure_future(_run())


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


def openai_tool_content_to_mcp(content) -> tuple[list[dict], bool]:
    """OpenAI tool message content -> MCP tool result content blocks.

    Three shapes arrive here: a plain string (kept as one text block), an
    Anthropic-style image block with `source.type == "base64"`, and an
    OpenAI image_url block whose `url` is a `data:` URL. Anything we cannot
    decode into bytes (a remote http(s) URL the sidecar has no way to fetch)
    becomes a loud isError text block, never a silent drop — the caller's
    model would otherwise happily hallucinate a reading of it.

    Returns (content_blocks, is_error). The contract matches what
    tool_server.py hands back to the CLI over stdio.
    """
    is_error = False
    if isinstance(content, list):
        blocks: list[dict] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                blocks.append({"type": "text", "text": str(part.get("text") or "")})
            elif ptype == "image":
                # Anthropic-style: source.base64 / source.media_type.
                src = part.get("source") if isinstance(part.get("source"), dict) else {}
                if src.get("type") == "base64" and src.get("data"):
                    blocks.append({"type": "image",
                                    "mimeType": src.get("media_type") or "image/png",
                                    "data": src["data"]})
                else:
                    log.warning("tool result image could not be passed through "
                                "(no base64 source); surfacing as isError")
                    is_error = True
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url") \
                    if isinstance(part.get("image_url"), dict) else None
                media, _b = cli_bridge._decode_data_url(url or "") or (None, None)
                if media and url and url.startswith("data:") and "," in url:
                    blocks.append({"type": "image", "mimeType": media,
                                    "data": url.split(",", 1)[1]})
                else:
                    log.warning("tool result image could not be passed through "
                                "(remote URL or unrecognised shape); surfacing as isError")
                    is_error = True
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        return blocks, is_error
    text = str(content) if content is not None else ""
    return [{"type": "text", "text": text}], False


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


def write_opencode_dir(workdir: Path, session_id: str, tools_path: Path,
                       images: bool = False) -> None:
    """`opencode run --dir <workdir>` reads project config from that
    directory. This adds one local MCP server and an agent that exposes
    only its tools -- the same pattern cli_bridge/harness/opencode.json uses
    to gate the built-in ones. Verified against a live run on OpenCode v1; the
    keys are all renamed in v2 -- see MCP_PROFILES for the mapping.

    `images=True` re-enables the `read` tool: staged image files must be
    readable for the model to see them (the alternative `-f` flag attaches
    them to the prompt, but Read still needs to exist for the loop on a
    rebuilt session). The MCP server's own tools are not affected either
    way.
    """
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
        # in this list, so they are not disabled by it. `read` is the one
        # exception: a session that has to read its own staged images needs
        # it back. That is opt-in via `images`, since most sessions are
        # text-only and the Read tool would be unnecessary surface area.
        "agent": {"switchyard": {"tools": {
            "bash": False, "edit": False, "write": False, "read": bool(images),
            "grep": False, "glob": False, "list": False, "patch": False,
            "todowrite": False, "todoread": False, "webfetch": False,
            "websearch": False, "task": False, "multiedit": False,
        }}},
    }
    (workdir / "opencode.json").write_text(json.dumps(cfg))


def build_argv(prompt: str, system: str | None, model: str, workdir: Path,
                session_id: str, tools_path: Path,
                allowed_tools: str,
                image_paths: list | None = None) -> tuple[list[str], str | None]:
    """Build the CLI argv, plus the prompt to feed it on stdin (or None).

    Returns a pair so an oversized prompt can travel on stdin instead of argv
    (see STDIN_PROMPT_LIMIT for why). The caller passes the second element
    straight into run_session.

    `image_paths`, when present, are added per-profile: claude gets
    `--add-dir` + an explicit Read allowlist; codex gets one `-i FILE` per
    path; opencode gets one `-f FILE` per path. The text bridge's
    build_argv does the same work for non-tool sessions, and the per-CLI
    mechanism is the same one (see cli_bridge/server.py's build_argv for
    why each profile wants it this way).

    For the claude profile only, `--disallowed-tools` is also emitted here
    (the MCP-tools `--allowed-tools` allowlist stays in the argv template
    so every code path keeps the switchyard tool qualifiers). When images
    are present, Read is dropped from the disallowed list -- otherwise the
    model cannot Read its own staged image, the same bug cli_bridge's
    bare_args_images variant already avoids.
    """
    instructions: Path | None = None
    image_paths = image_paths or []
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
        write_opencode_dir(workdir, session_id, tools_path,
                           images=bool(image_paths))
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
    use_stdin = over_argv_limit(effective_prompt)
    argv = [PROFILE["cli"]]
    terminator = PROFILE.get("prompt_terminator")
    for element in PROFILE["argv"]:
        if element != "{prompt}":
            argv.append(fill(element))
        elif use_stdin and PROVIDER == "codex":
            argv.append("-")
        elif not use_stdin:
            # Issue #121 lock-step: opencode carries a `--` sentinel so the
            # argv parser (yargs) stops at option lookup and the prompt
            # becomes an unambiguous positional. claude and codex have no
            # sentinel; for them, a prompt that still starts with `-` after
            # the system-prompt fold would be argv-parsed as a flag, so we
            # prepend a fixed non-dash line (`Message:`) to the element
            # itself. Both bridges (cli_bridge and mcp_bridge) apply the
            # same fix in the same place so behaviour matches.
            prompt_element = effective_prompt
            if not terminator and prompt_element.startswith("-"):
                prompt_element = "Message:\n" + prompt_element
            if terminator:
                argv.append(terminator)
            argv.append(prompt_element)
    if PROVIDER == "claude" and system:
        if over_argv_limit(system):
            # Same E2BIG trap as the prompt (issue #29): --system-prompt is
            # one argv element. The -file form is verified on the pinned CLI
            # (2.1.278), and the file lives in the session workdir so
            # cleanup_workdir reclaims it with everything else -- the same
            # lifecycle codex's instructions.md already has here. Trailing
            # newline matches cli_bridge's system_prompt_file so both files
            # have identical content shape regardless of which sidecar wrote.
            spath = workdir / "system-prompt.md"
            spath.write_text(system if system.endswith("\n") else system + "\n")
            argv += ["--system-prompt-file", str(spath)]
        else:
            argv += ["--system-prompt", system]
    if instructions is not None:
        argv += ["-c", f"model_instructions_file={instructions}"]

    # claude only: --disallowed-tools is conditional because the list omits
    # Read when images are present. Appended at the end so the two-list
    # question (template vs build_argv) is obvious in the code rather than
    # split across two layers. Concatenation semantics means even an extra
    # empty list would be safer than getting Read wrong here.
    if PROVIDER == "claude":
        key = "disallowed_tools_images" if image_paths else "disallowed_tools_default"
        argv += ["--disallowed-tools", PROFILE[key]]

    if image_paths:
        if PROVIDER == "claude":
            img_dir = Path(image_paths[0]).parent
            argv += ["--add-dir", str(img_dir),
                     "--allowed-tools", f"Read({img_dir}/**)"]
        elif PROVIDER == "codex":
            for path in image_paths:
                argv += ["-i", str(path)]
        else:   # opencode: `-f FILE(s)` attaches to the message
            for path in image_paths:
                argv += ["-f", str(path)]

    return argv, (effective_prompt if use_stdin else None)


# ---------------------------------------------------------------------------
# The background driver: one asyncio task per session, running the CLI
# subprocess for its whole lifetime. It never touches turn_future itself
# except at the very end (or on error/timeout) -- every tool-call turn is
# resolved by Session._flush, triggered from register_tool_call below.
# ---------------------------------------------------------------------------
async def _run_session_attempt(session: Session, argv: list[str],
                               stdin_data: str | None = None) -> dict | None:
    """One spawn+classify+parse for run_session, without the no-text retry.

    Mirrors cli_bridge._run_cli_attempt, but every terminal branch resolves the
    session's `turn_future` itself and returns None -- mcp_bridge has no
    HTTPException layer, the session IS the response channel. Raises
    cli_bridge.CliNoTextError unwrapped for the retriable no-text-with-usage
    shape (issue #78), and returns the parsed payload dict on success.
    """
    try:
        # cwd is the per-session workdir so the inner CLI never starts in
        # /app/mcp_bridge. The path still belongs to the relay -- this is
        # NOT the caller's cwd, and the env-block / first-turn reminder
        # make that clear to the model -- but a session-shaped directory
        # keeps paths beneath it from looking like a meaningful project
        # root the model could safely use (issue #44).
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=session.workdir,
            # env=allowlist, NOT os.environ.copy(): same rationale as
            # cli_bridge._run_cli_attempt -- the operator's .env is mounted
            # into this container, and we cannot hand every provider key and
            # the OAuth grant to a child CLI whose built-in tools we cannot
            # fully neutralise. The helper lives on cli_bridge because the
            # allowlist is the one being shared (bridge-siblings rule).
            env=cli_bridge.subprocess_env(),
        )
    except OSError as exc:
        # Same mapping cli_bridge._run_cli applies to a failed spawn (issue
        # #29): E2BIG is the request, not the plan, so it must read as 413;
        # any other spawn failure is this plan's capacity being broken.
        if exc.errno == errno.E2BIG:
            session.resolve_final({
                "type": "error", "status": 413,
                "detail": {"error": {
                    "message": (f"request too large for {PROVIDER} cli "
                                f"({exc.strerror}); reduce its size"),
                    "type": "request_too_large"}}})
        else:
            session.resolve_final({
                "type": "error", "status": 502,
                "detail": f"{PROVIDER} cli could not be spawned: {exc}"})
        return None
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
        return None

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
            return None
        up_status, up_message, up_retryable = cli_bridge.upstream_error(stdout)
        if cli_bridge._LIMIT.search(blob) or cli_bridge.seconds_until(blob):
            exc = cli_bridge._limit_error(
                f"{up_message}\n{blob}" if up_message else blob,
                PROFILE["default_retry_after"])
            session.resolve_final({"type": "error", "status": exc.status_code,
                                    "detail": exc.detail, "headers": exc.headers})
            return None
        if up_status in (401, 402, 403) and up_retryable is not True:
            session.resolve_final({"type": "error", "status": up_status,
                                    "detail": up_message or blob[:300]})
            return None
        status, detail = cli_bridge.error_from_events(stdout)
        if status and 400 <= status < 500 and status != 429:
            session.resolve_final({"type": "error", "status": status, "detail": detail})
            return None
        session.resolve_final({"type": "error", "status": 502,
                                "detail": f"{PROVIDER} cli failed ({proc.returncode}): "
                                          f"{(detail or stderr[:300] or stdout[:300])}"})
        return None

    try:
        payload = cli_bridge.parse_output(stdout, PROFILE["parser"])
    except cli_bridge.CliNoTextError:
        # Issue #64 retriable shape -- propagate unwrapped to run_session's
        # retry driver so the contract 502 can name real numbers, and a
        # successful retry can fold attempt 1's charged usage into attempt 2.
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        # Same contract as cli_bridge._run_cli: a structured parser that
        # cannot find the answer must fail the call, not echo its raw
        # event stream back as the answer (issue #3).
        if PROFILE["parser"] in cli_bridge.STRUCTURED_PARSERS:
            session.resolve_final({"type": "error", "status": 502,
                                    "detail": f"{PROVIDER} cli yielded no "
                                              f"parsed answer: {exc}"})
            return None
        payload = {"result": stdout.strip()}

    # Some CLIs exit 0 while reporting a limit or error inside the JSON
    # envelope (is_error=true / error_max_turns / usage-limit text). cli_bridge
    # raises the same shape in _run_cli_attempt; here we have no HTTPException
    # layer, so we read the exception's fields and resolve the turn_future as
    # an error -- never as a "final" payload (issue #127).
    if (exc := cli_bridge.check_result_envelope(
            payload, PROFILE["default_retry_after"])) is not None:
        session.resolve_final({"type": "error", "status": exc.status_code,
                                "detail": exc.detail, "headers": exc.headers})
        return None
    return payload


async def run_session(session: Session, argv: list[str],
                      stdin_data: str | None = None) -> None:
    """Retry-once driver around _run_session_attempt (issue #78).

    opencode-ai@1.18.31 occasionally emits a step_finish with nonzero output
    tokens but no text event (glm-5.3-flash, ~15-30% of calls on some plans).
    When parse_output raises CliNoTextError, attempt 1 is re-spawned exactly
    once with identical argv/stdin. On a successful retry, attempt 1's
    charged tokens are folded into the payload so the ledger books each
    attempt exactly once; on a second CliNoTextError the contract 502 names
    the COMBINED usage of both attempts. Every other failure mode stays
    single-attempt.
    """
    try:
        try:
            payload = await _run_session_attempt(session, argv, stdin_data)
        except cli_bridge.CliNoTextError as exc:
            if LOG_TEXT_LOST:
                log.warning("mcp session %s: no text with usage %s; retrying once",
                            session.id, exc.usage)
            try:
                payload = await _run_session_attempt(session, argv, stdin_data)
            except cli_bridge.CliNoTextError as exc2:
                combined = cli_bridge._sum_usage(dict(exc.usage), exc2.usage)
                if LOG_TEXT_LOST:
                    log.warning("mcp session %s: text lost on both attempts "
                                "(combined usage=%s); contract 502",
                                session.id, combined)
                session.resolve_final({
                    "type": "error", "status": 502,
                    "detail": cli_bridge.format_no_text_detail(PROVIDER, combined)})
                return
            # Successful retry: attempt 1's tokens are still real, fold them
            # into the payload so the ledger books both attempts exactly once.
            payload.setdefault("usage", {})
            cli_bridge._sum_usage(payload["usage"], exc.usage)
        if payload is None:
            return                      # a terminal error was already resolved
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


async def supersede_session(session: Session, reason: str) -> None:
    """Tear down a session that is about to be replaced by a rebuilt one.

    Distinct from end_session: the natural termination path (final answer or
    final tool_calls delivered) has no parked calls and a finished subprocess.
    A session being superseded may have parked tool calls whose tool_server.py
    requests will never receive a response, and a CLI subprocess blocked on a
    follow-up that will never come -- both must be torn down here, otherwise
    the rebuild path leaves the superseded session live (issue #13, defect 2:
    one tool loop spanning two live CLI sessions).
    """
    log.info("superseding mcp_bridge session %s (%s); %d pending call(s)",
             session.id, reason, len(session.pending))
    session.fail_pending(f"session {session.id} superseded ({reason})")
    if session.turn_future is not None and not session.turn_future.done():
        session.turn_future.set_exception(
            RuntimeError(f"session {session.id} superseded ({reason})"))
    if session.proc is not None and session.proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            session.proc.kill()
    await end_session(session)


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
        raise HTTPException(status_code=504, detail=str(exc)) from exc


@app.post("/internal/tools/call")
async def internal_tools_call(request: Request):
    payload = await request.json()
    return await register_tool_call(payload.get("session_id"), payload.get("name") or "",
                                     payload.get("arguments") or {})


# ---------------------------------------------------------------------------
# The public OpenAI-compatible surface.
# ---------------------------------------------------------------------------
def last_call_usage(session: Session) -> dict | None:
    """The most recent `assistant` usage record written by the inner CLI.

    The CLI's own transcript (`~/.claude/projects/<sanitized-workdir>/*.jsonl`)
    is the most reliable place to read the *current* context size: the CLI
    itself counts it per turn, and the file is written synchronously before
    exit. The OpenAI-shaped `usage.prompt_tokens` cli_bridge.to_openai emits is
    the SUM of every turn the CLI ran -- which is what the ledger needs to
    book -- but it is also wildly inflated for context-meter purposes: a
    caller asking for context size mid-loop wants the size of the LAST turn,
    not the running total. The summed figure is still kept on the response,
    just under `switchyard_billed_*`, so the caller's view is honest.

    Returns None when there is nothing meaningful to read: a non-claude
    provider (no transcript shape to lean on), a missing project dir (the
    CLI never wrote here), an OSError (transient), or a transcript with no
    assistant usage line yet (mid-run, before the first call finished). All
    four are caller-neutral -- the response keeps today's usage unchanged.
    """
    if PROVIDER != "claude":
        return None
    sanitized = re.sub(r"[^A-Za-z0-9]", "-", str(session.workdir))
    project_dir = cli_bridge.CLAUDE_PROJECTS / sanitized
    try:
        if not project_dir.is_dir():
            return None
        newest: tuple[float, Path] | None = None
        for path in project_dir.glob("*.jsonl"):
            try:
                stamp = path.stat().st_mtime
            except OSError:
                continue
            if newest is None or stamp > newest[0]:
                newest = (stamp, path)
        if newest is None:
            return None
        path = newest[1]
        # Read the whole file rather than tailing: transcripts are small
        # (kilobytes) and JSONL's "scan in reverse" is what callers expect
        # -- the latest assistant usage is what matters, but a partial
        # tail could swallow the newline that splits entries.
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            return None
    except OSError:
        return None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        return {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "cache_read_input_tokens":
                int(usage.get("cache_read_input_tokens", 0) or 0),
            "cache_creation_input_tokens":
                int(usage.get("cache_creation_input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
        }
    return None


def _with_context_usage(response: dict, session: Session, billed: dict) -> dict:
    """Replace `response["usage"]` with the last-call context, keep the sum as billed.

    `billed` is whatever the caller wants surfaced as the run-total so far --
    an empty dict on a mid-loop tool_calls (nothing booked yet, the caller
    just wants a real context meter) or `dict(response["usage"])` on the
    final turn (the OpenAI-shaped sum the CLI's payload carried). The
    OpenAI-shaped last-call view comes from cli_bridge.to_openai, which folds
    cache reads / cache writes into `prompt_tokens` the same way it does on
    the text path; the billed figures sit alongside as separate keys so the
    ledger has the running total it needs.
    """
    usage = response.get("usage") or {}
    last = last_call_usage(session)
    if last is not None:
        usage = cli_bridge.to_openai({"usage": last}, "")["usage"]
    usage["switchyard_billed_prompt_tokens"] = int(billed.get("prompt_tokens", 0) or 0)
    usage["switchyard_billed_completion_tokens"] = int(billed.get("completion_tokens", 0) or 0)
    response["usage"] = usage
    return response


def render_turn(session: Session, result: dict, requested_model: str | None) -> dict:
    if result["type"] == "tool_calls":
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in result["calls"]]}
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
            "created": int(time.time()), "model": requested_model or session.model,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
            # Token accounting for the CLI-backed path is already approximate
            # (see cli_bridge); mid-loop it is not available at all.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        # No usage has been booked yet on a mid-loop tool_calls turn, but
        # the caller can still want a real context meter — the CLI wrote
        # one to its transcript before exiting this turn.
        return _with_context_usage(response, session, {})
    if result["type"] == "final":
        response = cli_bridge.to_openai(result["payload"], requested_model or session.model)
        # On the final turn, the OpenAI-shaped sum the CLI's payload carries
        # is what should be billed; the last-call context replaces
        # `prompt_tokens` so the caller sees a context meter, not the run total.
        return _with_context_usage(response, session, dict(response["usage"]))
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

    messages = body.get("messages") or []
    # Stage images BEFORE flatten so the markers `[image N: <path>]` survive
    # into the prompt text. An unsupported image (a remote URL we cannot
    # fetch) is an error here, not a silent drop -- the bug fix in #30.
    try:
        image_paths, img_dir = cli_bridge.stage_or_fail(messages)
    except cli_bridge.ImageUnsupportedError as exc:
        raise exc.http() from exc
    # Mirror cli_bridge's _handle_chat: the img_dir is owned here until
    # start_session takes it. Any exception that escapes between now and
    # then (e.g. an unexpected failure in flatten / resolve_model / the
    # gate) leaks the dir without this finally -- start_session has its own
    # cleanup, but it never gets the dir if we never reach it.
    try:
        prompt, system = cli_bridge.flatten(messages)
        if image_paths:
            prompt += cli_bridge.image_note(image_paths)
        if not prompt:
            raise HTTPException(status_code=400, detail="no usable message content")
        model, warning = cli_bridge.resolve_model(body.get("model"))
        if warning:
            log.warning("%s", warning)

        # Resolve the caller's tool-execution environment once per fresh
        # request. Re-resolved here -- the gateway's stamp is best-effort
        # and the request body is the primary source.
        env = _resolve_env(body)
        if env is None or env.source == "unknown":
            try:
                env = _caller_env.resolve(body, _caller_env_settings()) \
                    if _caller_env else None
            except _caller_env.CallerEnvironmentRequired as exc:
                # The operator set probe=required and we cannot resolve
                # the env. Refuse the request instead of rendering
                # "Caller platform: unknown" -- the whole point of the
                # option is loud refusal. 400 (Bad Request), NOT 503 +
                # Retry-After: nothing about the request, the caller,
                # or the operator's plans.yaml can change in 5 seconds
                # (the env will still be unresolvable on retry), so a
                # Retry-After would actively mislead the caller. The
                # file's 503 + Retry-After convention is reserved for
                # transient-capacity conditions.
                log.warning("probe=required but env unresolvable; refusing "
                            "request: %s", exc)
                raise HTTPException(
                    status_code=400,
                    detail={"error": {
                        "type": "caller_environment_required",
                        "message": (str(exc)
                            + " Configure caller_environment.platform/cwd/shell "
                              "in plans.yaml, supply a passive environment in the "
                              "request body, or set probe=auto to fall back to "
                              "the unknown-env wording.")}}) from exc
        if env is None:
            env = _caller_env.CallerEnvironment.unknown() if _caller_env else None

        limit = cli_bridge.config().concurrency
        if not await cli_bridge._gate.acquire(limit):
            # Never queue, same contract as cli_bridge: SwitchYard needs "full"
            # immediately so it can spill to the next plan in the lane. Only a
            # follow-up waits, because it has nowhere else to go.
            raise HTTPException(status_code=429, detail=f"sidecar at capacity ({limit})",
                                 headers={"Retry-After": "5"})

        return await start_session(body, mcp_tools, prompt, system, model, request,
                                   image_paths, img_dir, env=env)
    except Exception as exc:
        log.warning("handle_fresh failed before start_session: %s", exc, exc_info=True)
        if img_dir is not None:
            shutil.rmtree(img_dir, ignore_errors=True)
        raise


async def start_session(body: dict, mcp_tools: list[dict], prompt: str,
                        system: str | None, model: str,
                        request: "Request | None",
                        image_paths: list | None = None,
                        img_dir: Path | None = None,
                        env: Any = None,
                        first_turn: bool = True) -> dict:
    """Spawn a CLI session and run one turn. The gate slot is already held.

    Shared by a fresh request and a resumption, so the two cannot drift: the
    only difference between them is how the prompt was built and how the slot
    was obtained.

    `image_paths` are appended to the CLI's argv per profile (see build_argv);
    they are the only thing build_argv needs. `img_dir`, when set, is the
    caller's separate temp staging dir (cli_bridge.stage_or_fail always mints
    one -- even on a fresh request, which has its own workdir): this function
    owns its cleanup, since the workdir is unrelated. `workdir` is cleaned
    separately by end_session (called above for the final case, park_session
    for the parked case); on an exception it is cleaned below.

    `env` is the resolved CallerEnvironment for this session; when present
    its `[SwitchYard tool execution environment]` block is appended to
    `system` (it rides inside the caller's system content for
    SYSTEM_MODE=replace semantics to be untouched) and a one-line reminder
    is prepended to the first user turn of the prompt.

    `first_turn=True` triggers the reminder line. A fresh session always
    has it; a resumption rebuild builds a NEW CLI session and so gets it
    again -- the rebuild is a new first turn by construction.
    """
    session_id = uuid.uuid4().hex
    workdir = Path(tempfile.mkdtemp(prefix=f"mcpb-{session_id[:8]}-"))
    # The slot was acquired by the caller before getting here; the session owns
    # it from now on, and gives it back when it parks or ends.
    session = Session(id=session_id, provider=PROVIDER, model=model,
                      workdir=str(workdir), holds_slot=True)
    # Inject the env block (system) and the first-turn reminder (prompt).
    # Both are pure functions of (prompt, system, env, first_turn) so a
    # rebuild of the same request produces the same CLI argv.
    if env is not None and _caller_env is not None:
        env_block = _caller_env.render_system_block(env)
        system = (f"{system}\n\n{env_block}" if system else env_block)
        if first_turn:
            reminder = _caller_env.render_first_turn_reminder(env)
            prompt = f"{reminder}\n\n{prompt}" if prompt else reminder
    try:
        tools_path = workdir / "tools.json"
        tools_path.write_text(json.dumps(mcp_tools))
        allowed = ",".join(PROFILE["tool_qualifier"](t["name"]) for t in mcp_tools)
        argv, stdin_data = build_argv(prompt, system, model, workdir, session_id,
                                      tools_path, allowed, image_paths)
    except Exception as exc:
        log.warning("start_session failed to build argv: %s", exc, exc_info=True)
        await cli_bridge._gate.release()
        cleanup_workdir(workdir)
        if img_dir is not None:
            shutil.rmtree(img_dir, ignore_errors=True)
        raise

    SESSIONS[session_id] = session
    session.new_turn()
    try:
        asyncio.create_task(run_session(session, argv, stdin_data))
        result = await await_turn(session, request)
        response = render_turn(session, result, body.get("model"))
        # Cache the rendered response so a duplicate follow-up can be answered
        # with the same tool_calls response instead of being misread as a lost
        # session (issue #13, defect 1). For a brand-new session, no follow-up
        # has been minted yet so this is defensive -- _continue_followup also
        # caches, and supersedes on rebuild.
        session.last_response = response
        if result["type"] != "tool_calls":
            await end_session(session)
        else:
            await park_session(session)
    except BaseException:
        # render_turn raises HTTPException on every CLI failure path (502,
        # TEXT_LOST, 413, 429, ...), and a cancelled handler is also possible.
        # Both must tear the session down: otherwise the session stays in
        # SESSIONS holding its gate slot until the 1800 s idle reaper comes
        # round, and four such leaks physically fill the sidecar gate (issue
        # #80). BaseException (not Exception) catches the cancel too;
        # end_session's `if session.dead: return` makes the call idempotent
        # against the 499-already-reaped path that await_turn can raise when
        # the caller has hung up. NOT a bare `finally: end_session` -- parked
        # sessions hold no slot by design and must survive the tool_calls
        # success path. The cleanup itself awaits cli_bridge._gate.release(),
        # which can be interrupted by a *second* CancelledError (server-shutdown
        # re-cancel, FastAPI lifespan teardown, ...) -- swallow anything the
        # cleanup raises so the ORIGINAL exception still propagates and the
        # 1800 s reaper stays the last-resort backstop if even that didn't run
        # to completion. (PR #82 review round 1.)
        with contextlib.suppress(BaseException):
            await end_session(session)
        raise
    if img_dir is not None:
        # Clean up the staging dir we own. end_session above already cleaned
        # the session workdir, which is a separate tree.
        shutil.rmtree(img_dir, ignore_errors=True)
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
    messages = body.get("messages") or []
    # Stage images BEFORE flatten_with_tool_history, so the markers survive
    # into the rendered narration and the rebuilt CLI can be told the paths.
    # The new workdir does not exist yet (start_session creates it), so the
    # staging dir is a separate temp dir the call owns and cleans up.
    try:
        image_paths, img_dir = cli_bridge.stage_or_fail(messages)
    except cli_bridge.ImageUnsupportedError as exc:
        raise exc.http() from exc
    # Same broadened cleanup as handle_fresh: anything that escapes between
    # here and start_session (a gate acquire failure past RESUME_WAIT, an
    # unexpected exception in flatten_with_tool_history) would otherwise
    # leak the swimg-* dir.
    try:
        prompt, system = flatten_with_tool_history(messages)
        if image_paths:
            prompt += cli_bridge.image_note(image_paths)
        if not prompt:
            raise HTTPException(status_code=400, detail="no usable message content")
        model, warning = cli_bridge.resolve_model(body.get("model"))
        if warning:
            log.warning("%s", warning)

        # Re-resolve the env on the rebuild path too -- the rebuilt session
        # is a fresh CLI turn, the caller's environment may have changed
        # (or been newly detected on this very request), and rebuilds are
        # pure functions of (body, env) so the same env produces the same
        # argv. Pass `first_turn=True`: a rebuild is a new first turn.
        env = _resolve_env(body)
        if env is None or env.source == "unknown":
            try:
                env = _caller_env.resolve(body, _caller_env_settings()) \
                    if _caller_env else None
            except _caller_env.CallerEnvironmentRequired as exc:
                # Same refusal semantics as handle_fresh: probe=required +
                # unresolvable env = 400 (Bad Request), NOT 503 +
                # Retry-After. The failure is not transient.
                log.warning("probe=required but env unresolvable on rebuild; "
                            "refusing request: %s", exc)
                raise HTTPException(
                    status_code=400,
                    detail={"error": {
                        "type": "caller_environment_required",
                        "message": (str(exc)
                            + " Configure caller_environment.platform/cwd/shell "
                              "in plans.yaml, supply a passive environment in the "
                              "request body, or set probe=auto to fall back to "
                              "the unknown-env wording.")}}) from exc
        if env is None:
            env = _caller_env.CallerEnvironment.unknown() if _caller_env else None

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
        # Slot ownership passes to start_session here; it owns the teardown
        # contract (issue #80) on this rebuild path too.
        return await start_session(body, mcp_tools, prompt, system, model, request,
                                    image_paths, img_dir, env=env, first_turn=True)
    except Exception as exc:
        log.warning("resume_gone_session failed before start_session: %s",
                    exc, exc_info=True)
        if img_dir is not None:
            shutil.rmtree(img_dir, ignore_errors=True)
        raise


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
            # A session whose results were dropped can never be continued
            # coherently: its CLI is blocked on tool_server.py requests whose
            # replies will not arrive. Supersede each one now so it stops
            # looking live -- otherwise the invariant "one caller tool loop
            # <-> at most one live mcp_bridge session" is broken (issue #13,
            # defect 2).
            for sid in live_ids[1:]:
                dropped = SESSIONS.get(sid)
                if dropped is not None and not dropped.dead:
                    await supersede_session(
                        dropped, f"dropped from span-{len(live_ids)} follow-up "
                                 f"routed to {wanted[:8]}")
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
        # Always record the id as resolved: whether we just consumed it or it
        # was already popped on a prior delivery, the call is no longer
        # parked in this session. Without this, a retry of an already-delivered
        # batch looks identical to a lost session and triggers a rebuild
        # (issue #13, defect 1).
        session.mark_resolved(m["tool_call_id"])
        if call is None:
            continue          # already resolved, or a stale id -- tolerate rather than fail
        if call.future.done():
            continue
        mcp_content, content_is_error = openai_tool_content_to_mcp(m.get("content"))
        call.future.set_result({
            "content": mcp_content,
            # isError is OR'd, not replaced: the caller's explicit flag still wins
            # (e.g. "I already know this tool failed"), but we surface what we
            # could not deliver ourselves too.
            "isError": bool(m.get("is_error")) or content_is_error,
        })
        resolved += 1
    if resolved == 0:
        delivered_ids = [m["tool_call_id"] for m in tool_msgs]
        # Defect 1: every delivered id is already resolved and the session is
        # still parked, so this is a duplicate delivery of a batch we already
        # turned into a tool_calls response. Return the cached response rather
        # than rebuilding -- the caller is retrying because the original
        # response never arrived, not asking for a new turn. (We only return
        # the cached response while the session is parked: once the CLI has
        # continued, last_response is from a stale turn and the right answer
        # is whatever the new turn produces.)
        if (session.last_response is not None
                and session.awaiting_followup
                and all(cid in session.resolved_recently for cid in delivered_ids)):
            log.info(
                "live mcp_bridge session %s received a duplicate delivery of "
                "%d already-resolved tool_call_id(s); returning the cached "
                "tool_calls response rather than rebuilding",
                session.id, len(delivered_ids))
            return session.last_response
        # Could be a lost session, ids from a parallel batch we already
        # finished mid-turn, or just garbled input. Rebuilding from the
        # request keeps the caller moving instead of throwing a 400 they
        # cannot recover from -- but the superseded session must be torn down
        # first, otherwise the rebuild leaves two live sessions on one tool
        # loop (issue #13, defect 2).
        log.warning(
            "live mcp_bridge session %s had no parked tool call matching the "
            "delivered ids; superseding it and rebuilding from the request "
            "so the caller is not trapped", session.id)
        await supersede_session(
            session, "no parked call matched delivered ids (rebuilding)")
        return await resume_gone_session(
            body, body.get("tools") or [], session.id, request,
            why="no parked call matched delivered ids")

    # The results are in; the model is about to run again, so take a slot back.
    # unpark_session is deliberately OUTSIDE the try below: its 503-on-saturated-
    # gate failure intentionally leaves the session in place for the gateway to
    # retry (issue #13, follow-up path must queue rather than fail fast).
    await unpark_session(session)
    try:
        session.touch()
        await session.new_turn()
        result = await await_turn(session, request)
        response = render_turn(session, result, body.get("model"))
        # Cache the rendered response so a later duplicate delivery of THIS
        # batch's tool_call_ids can be answered with the same tool_calls response
        # -- without this, every retry looks like a lost session and rebuilds.
        session.last_response = response
        if result["type"] != "tool_calls":
            await end_session(session)
        else:
            await park_session(session)
    except BaseException:
        # Same teardown contract as start_session: every CLI failure mode
        # makes render_turn raise HTTPException, which used to escape here
        # without running end_session -- the session stayed in SESSIONS
        # holding its gate slot until the 1800 s idle reaper came round, and
        # four such leaks physically filled the sidecar gate (issue #80).
        # end_session's dead-guard makes the call idempotent against the
        # 499-already-reaped path await_turn can raise when the caller
        # disconnects mid-turn. NOT a bare `finally: end_session` -- parked
        # sessions must survive the tool_calls success path. Same second-
        # cancel hardening as start_session: swallow anything the cleanup
        # raises so the ORIGINAL exception still propagates and the 1800 s
        # reaper stays the last-resort backstop. (PR #82 review round 1.)
        with contextlib.suppress(BaseException):
            await end_session(session)
        raise
    return response


async def handle_tool_request(body: dict, tools: list[dict],
                              request: "Request | None" = None) -> dict:
    # Probe results travel as ordinary `tool` messages whose tool_call_id
    # carries `switchyard_env_`. Consume them FIRST (before any session
    # lookup): the synthetic assistant message that minted the probe is
    # stripped out so the inner CLI never sees the relay's bookkeeping,
    # the parsed env is stashed in the probe cache, and we fall through
    # into the normal fresh path with the env known.
    probe_env, body = _consume_probe_results(body)
    body = dict(body)
    body["messages"] = _strip_synthetic_assistant(body.get("messages") or [])
    if probe_env is not None:
        # Stash the result on the body so handle_fresh can pick it up
        # without re-parsing. Pure function-of-body, so rebuilds stay
        # deterministic.
        body.setdefault("metadata", {}).setdefault("switchyard", {})["caller_env"] = {
            "cwd": probe_env.cwd, "platform": probe_env.platform,
            "shell": probe_env.shell, "source": "probe",
        }
    tool_msgs = [m for m in (body.get("messages") or [])
                 if m.get("role") == "tool" and m.get("tool_call_id")]
    if tool_msgs:
        return await handle_followup(body, tool_msgs, request)
    # Fresh path: a probe goes BEFORE any gate acquire, session allocation
    # or workdir creation -- the spec is explicit about that. The probe
    # response is a one-shot synthetic tool_calls that travels back to
    # the caller immediately; no slot is held while waiting for the
    # caller to execute the command and answer.
    probe_future = _maybe_probe(body, tools)
    if probe_future is not None:
        probe_response = await probe_future
        if probe_response is not None:
            return probe_response
    return await handle_fresh(body, tools, request)


@app.get("/health")
async def health() -> dict:
    cfg = cli_bridge.config()
    health_doc = {"ok": cfg.source == "config", "provider": PROVIDER, "supports_tools": True,
                 # All three MCP_PROFILES carry images via the same per-profile
                 # flags as cli_bridge (see build_argv / write_opencode_dir).
                 # Expressed as a set-membership check rather than a constant so
                 # adding a future provider here matches the cli_bridge side.
                 "supports_images": PROVIDER in ("claude", "opencode", "codex"),
                 "config_source": cfg.source, "model": cfg.model, "models": sorted(cfg.models),
                 "concurrency": cfg.concurrency, "in_flight": cli_bridge._gate.in_flight,
                 "sessions": len(SESSIONS),
                 "awaiting_followup": sum(1 for x in SESSIONS.values()
                                          if x.awaiting_followup and not x.dead),
                 "parked_limit": cfg.parked_limit,
                 "session_ttl_seconds": SESSION_TTL,
                 # Profile-driven (mirrors cli_bridge /health). The MCP path's
                 # tool loop ignores max_tokens regardless (see handle_fresh /
                 # render_turn), so this field is reported but only the
                 # no-tools fall-through actually applies it -- a CLI tool
                 # request is not the contract the cap was designed for.
                 "enforces_max_tokens": bool(cli_bridge.PROFILE.get("enforce_max_tokens"))}
    if not health_doc["enforces_max_tokens"]:
        reason = cli_bridge.PROFILE.get("enforce_max_tokens_reason")
        if reason:
            health_doc["enforces_max_tokens_reason"] = reason
    return health_doc


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
