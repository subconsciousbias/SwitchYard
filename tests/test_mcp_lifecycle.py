"""Lifecycle hardening tests for the mcp_bridge (issue #264 #137 final slice).

The pins in this file cover the lifecycle work that lands in
sidecars/mcp_bridge/server.py alongside the other #264 #137 final-slice
changes (see `.orca-auto-plan.md` for the workstream). They are kept in a
dedicated file so a regression in the session machinery is a single
file-level signal in `bash scripts/test.sh` rather than buried in a 9000-
line test_mcp_bridge.py.

The fixtures reuse `_new_session` / `_drop` from the same patterns the
existing tests use (Session objects parked via register_tool_call, then
exercised end-to-end through the asyncio path). Every test below is plain
async; no HTTP client, no subprocess -- the things under test are
in-process state machines and the helpers they call.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
MCP_BRIDGE_DIR = os.path.join(os.path.dirname(HERE), "sidecars", "mcp_bridge")
sys.path.insert(0, MCP_BRIDGE_DIR)
sys.path.insert(0, HERE)

os.environ["PROVIDER"] = "claude"
os.environ.setdefault("SWITCHYARD_PLAN", "claude-max")
os.environ["MCP_HOST_MIRROR_CHECK"] = "off"

from plans_path import plans_path  # noqa: E402

os.environ["SWITCHYARD_PLANS"] = plans_path()
os.environ.setdefault("SIDECAR_PORT", "8081")

from _modules import load  # noqa: E402

server = load("mcp_bridge_server", os.path.join(MCP_BRIDGE_DIR, "server.py"))
server.HOST_MIRROR_MODE = "off"


def _new_session() -> "server.Session":
    session = server.Session(id=uuid.uuid4().hex, provider="claude",
                              model="m",
                              workdir=tempfile.mkdtemp(prefix="mcpb-lifecycle-"))
    server.SESSIONS[session.id] = session
    return session


def _drop(session: "server.Session") -> None:
    server.SESSIONS.pop(session.id, None)


def _write_fake_transcript(session, entries: list[dict]) -> None:
    """Write a Claude transcript JSONL with the supplied assistant entries.

    `_cumulative_transcript_usage` reads from
    `~/.claude/projects/<sanitized-spawn-dir>/<session-uuid>.jsonl` (the same
    path the inner CLI writes to). A test that wants to drive the cumulative
    usage helper writes a fake transcript there and asks `_book_unbooked_usage`
    to read it back. `spawn_dir` defaults to the session's workdir so the
    sanitization function lands on the same project dir the real CLI used.
    """
    sanitized = __import__("re").sub(
        r"[^A-Za-z0-9]", "-",
        str(session.spawn_dir or session.workdir))
    project_dir = server.cli_bridge.CLAUDE_PROJECTS / sanitized
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{server.claude_session_uuid(session.id)}.jsonl"
    with open(path, "w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


# ---------------- #135: cumulative-usage booking at teardown -----------------
def test_book_unbooked_usage_keeps_silent_when_nothing_to_book():
    """A session whose last_response already includes the cumulative usage
    must NOT log a spurious un-booked warning -- the helper is silent on
    the no-delta case, and a regression that always warns would flood the
    logs of every healthy session teardown.
    """
    session = _new_session()
    session.billed_prompt_tokens = 100
    session.billed_completion_tokens = 25
    _write_fake_transcript(session, [{
        "type": "assistant", "message": {"usage": {
            "input_tokens": 100, "output_tokens": 25,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0}}}])

    captured: list[tuple] = []
    real_log = server.log.warning
    server.log.warning = lambda *a, **kw: captured.append((a, kw))
    try:
        server._book_unbooked_usage(session)
    finally:
        server.log.warning = real_log
    _drop(session)
    assert not captured, captured
    print("  _book_unbooked_usage: silent when cumulative == billed (no log spam)")


def test_book_unbooked_usage_logs_delta_when_session_was_preempted():
    """A session that ended mid-park with billed_*=0 must report the entire
    cumulative transcript as the un-booked delta. The warning text names
    the session id, the prompt / completion deltas, and the cumulative and
    billed totals, so an operator reconciling the ledger can write off the
    difference on the next cycle.
    """
    session = _new_session()
    # Preempted mid-loop: nothing ever billed; the cumulative transcript
    # is what the ledger is missing.
    session.billed_prompt_tokens = 0
    session.billed_completion_tokens = 0
    _write_fake_transcript(session, [
        {"type": "assistant", "message": {"usage": {
            "input_tokens": 50, "output_tokens": 10,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0}}},
        {"type": "assistant", "message": {"usage": {
            "input_tokens": 80, "output_tokens": 15,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0}}},
    ])
    captured: list[tuple] = []
    real_log = server.log.warning
    server.log.warning = lambda *a, **kw: captured.append((a, kw))
    try:
        server._book_unbooked_usage(session)
    finally:
        server.log.warning = real_log
    _drop(session)
    assert len(captured) == 1, captured
    args = captured[0][0]
    assert "un-booked usage" in args[0], args
    # Positional args after the format string: session.id, prompt_delta,
    # completion_delta, cumulative_input, cumulative_output,
    # billed_prompt, billed_completion.
    assert args[1] == session.id, args
    assert args[2] == 130, args          # 50 + 80
    assert args[3] == 25, args           # 10 + 15
    assert args[4] == 130, args          # cumulative input
    assert args[5] == 25, args           # cumulative output
    assert args[6] == 0, args            # billed prompt
    assert args[7] == 0, args            # billed completion
    print(f"  _book_unbooked_usage: preempted mid-park session "
          f"{session.id[:8]}... -> warning names prompt=130 completion=25")


def test_reap_session_books_unbooked_usage_before_destroying_session():
    """`reap_session` (the idle-TTL collector) MUST call
    `_book_unbooked_usage` before end_session strips the transcript-
    recoverable fields. Driving this end to end via a fake transcript
    proves the helper sees the session intact.
    """
    session = _new_session()
    session.billed_prompt_tokens = 0
    session.billed_completion_tokens = 0
    _write_fake_transcript(session, [{
        "type": "assistant", "message": {"usage": {
            "input_tokens": 7, "output_tokens": 3,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0}}}])
    captured: list[tuple] = []
    real_log = server.log.warning
    server.log.warning = lambda *a, **kw: captured.append((a, kw))
    try:
        asyncio.run(server.reap_session(session))
    finally:
        server.log.warning = real_log
    # After reap, session is gone from SESSIONS, marked dead, and the
    # booking warning is logged. Two warnings fire: the "reaping ..." line
    # from reap_session itself, then the "_book_unbooked_usage" warning
    # the helper emits. We only assert on the booking warning here -- the
    # reap line is its own concern.
    assert session.dead, session.dead
    assert session.id not in server.SESSIONS, dict(server.SESSIONS)
    booking = [c for c in captured if "un-booked usage" in c[0][0]]
    assert len(booking) == 1, captured
    print(f"  reap_session: idle-TTL reaper logs un-booked warning before "
          f"stripping transcript; session {session.id[:8]}... marked dead")


# ---------------- #264: tools-free follow-up routes by fingerprint -----------
import contextlib


class _StubFingerprinter:
    """16-hex of (system + first user), stable across turns of one
    session and distinct across sessions. The properties
    `parked_session_by_fingerprint` actually relies on -- matches when
    the system + first user round-trip the same way, distinct when they
    differ. Installed on `server._caller_env` only for the lifetime of
    the surrounding test, so the stub never leaks into other suites
    that share the cached mcp_bridge_server module."""

    def fingerprint(self, messages: list[dict]) -> str:
        import hashlib
        keys = []
        first_user = ""
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "system":
                keys.append(("sys", str(m.get("content") or "")))
            elif m.get("role") == "user" and not first_user:
                first_user = str(m.get("content") or "")
        blob = "\x00".join(k[1] for k in keys) + "\x00" + first_user
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@contextlib.contextmanager
def _fingerprint_helper_installed():
    """Make sure server._caller_env has a working `fingerprint` for the
    fingerprint tests, even when switchyard/caller_env.py is not
    importable in this environment -- and restore the original on exit
    so the stub does not leak into tests that share the cached
    mcp_bridge_server module (test_mcp_bridge.py, the lockdown suites).
    """
    saved = server._caller_env
    if saved is None or not hasattr(saved, "fingerprint"):
        server._caller_env = _StubFingerprinter()
    try:
        yield server._caller_env
    finally:
        server._caller_env = saved


def test_parked_session_by_fingerprint_finds_a_matching_parked_session():
    """The fingerprint-based nudge continuity routing matches the single
    parked session whose `session.fingerprint` equals the request's
    fingerprint (system + first user).
    """
    with _fingerprint_helper_installed() as fp_helper:

        older = _new_session()
        older.awaiting_followup = True
        older.last_active = time.time() - 30
        newer = _new_session()
        newer.awaiting_followup = True
        newer.last_active = time.time() - 5     # more recently parked

        messages = [
            {"role": "system", "content": "you are a helper"},
            {"role": "user", "content": "first nudge"},
        ]
        fp = fp_helper.fingerprint(messages)
        older.fingerprint = fp
        newer.fingerprint = fp

        body = {"messages": messages}
        match = server.parked_session_by_fingerprint(body)
        try:
            # AMBIGUITY (review comment on #264): two parked sessions
            # share the same fingerprint -> the helper MUST return None
            # so the call falls through to the text path, not silently
            # cross-wire the nudge to the most recently parked session.
            assert match is None, (
                f"expected None on ambiguous fingerprint match, got "
                f"match.id={getattr(match, 'id', None)}")
        finally:
            _drop(older)
            _drop(newer)
    print("  parked_session_by_fingerprint: ambiguous (2+ parked share "
          "fingerprint) -> None (no silent cross-wire)")


def test_parked_session_by_fingerprint_returns_none_when_nothing_matches():
    """A request whose fingerprint does not match any parked session
    returns None -- the caller falls through to the text path, exactly
    like the pre-#264 routing. The feature never blocks the call.
    """
    with _fingerprint_helper_installed():
        session = _new_session()
        session.awaiting_followup = True
        session.fingerprint = "abc"             # doesn't match anything else
        body = {"messages": [
            {"role": "user", "content": "completely unrelated conversation"}]}
        try:
            match = server.parked_session_by_fingerprint(body)
            assert match is None, match
        finally:
            _drop(session)
    print("  parked_session_by_fingerprint: no match -> None (caller falls "
          "through to text path)")


def test_parked_session_by_fingerprint_resolves_unique_match():
    """Exactly one parked session matches the fingerprint -> the helper
    returns it (the unique-match happy path the chat() routing depends on).
    """
    with _fingerprint_helper_installed() as fp_helper:
        only = _new_session()
        only.awaiting_followup = True
        only.last_active = time.time() - 5

        messages = [
            {"role": "system", "content": "you are a helper"},
            {"role": "user", "content": "first nudge"},
        ]
        fp = fp_helper.fingerprint(messages)
        only.fingerprint = fp

        body = {"messages": messages}
        try:
            match = server.parked_session_by_fingerprint(body)
            assert match is not None, "no parked session matched the fingerprint"
            assert match.id == only.id, (match.id, only.id)
        finally:
            _drop(only)
    print("  parked_session_by_fingerprint: unique parked match -> the "
          "session the chat() routing should resume")


# ---------------- #255: literal-text tool call detection --------------------
def test_detect_literal_tool_calls_promotes_xml_form_to_real_call():
    """An Anthropic-style `<tool_use name="bash">{"cmd":"ls"}</tool_use>`
    block embedded in assistant text becomes a real tool_call dict when
    `bash` is in the known tool set. The pattern is removed from the
    cleaned text and a synthetic call_<sid>_<n> id is minted.
    """
    known = {"bash", "read"}
    text = 'Sure, let me list it.\n<tool_use name="bash">{"cmd":"ls"}</tool_use>\nDone.'
    cleaned, tool_calls, dropped = server._detect_literal_tool_calls(
        text, known)
    assert "<tool_use" not in cleaned, cleaned
    # The literal pattern is gone; the surrounding narrative ("Sure, let me
    # list it." / "Done.") is preserved.
    assert "let me list" in cleaned, cleaned
    assert "Done." in cleaned, cleaned
    assert len(tool_calls) == 1, tool_calls
    assert tool_calls[0]["function"]["name"] == "bash", tool_calls
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args == {"cmd": "ls"}, args
    assert tool_calls[0]["id"].startswith("call_lit_"), tool_calls[0]
    assert dropped == [], dropped
    print("  _detect_literal_tool_calls: <tool_use> XML form -> real "
          "tool_call; cleaned text keeps the surrounding narrative")


def test_detect_literal_tool_calls_promotes_bracket_form_to_real_call():
    """A `[called bash({"cmd":"ls"})]` bracket form -- the same shape
    `flatten_with_tool_history` emits as rebuild narration -- also gets
    promoted when the tool name is known. A bracket form with EMPTY args
    (which matches the regex but has no JSON body) lands on the dropped
    list, not the tool_calls list.
    """
    known = {"bash"}
    text = '[called bash({"cmd":"ls"})] and then [called bash()]'
    cleaned, tool_calls, dropped = server._detect_literal_tool_calls(
        text, known)
    # The JSON-bearing bracket became a real call; the empty-args bracket
    # is dropped with a note (no JSON body to lift).
    assert len(tool_calls) == 1, tool_calls
    assert tool_calls[0]["function"]["name"] == "bash", tool_calls
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"cmd": "ls"}, tool_calls
    assert len(dropped) == 1, dropped
    assert "literal tool call" in dropped[0], dropped
    print("  _detect_literal_tool_calls: bracket form with JSON -> real call; "
          "empty-args bracket -> dropped note")


def test_detect_literal_tool_calls_drops_unknown_tool_names():
    """A literal call whose tool name is not in the known set must be
    dropped with a note, never silently renamed to a different tool.
    Forwarding it as a real tool_call would land on a 404 in the CLI.
    """
    known = {"bash"}
    text = 'let me look: <tool_use name="webfetch">{"url":"x"}</tool_use>'
    cleaned, tool_calls, dropped = server._detect_literal_tool_calls(
        text, known)
    assert tool_calls == [], tool_calls
    assert len(dropped) == 1, dropped
    assert "unknown tool" in dropped[0], dropped
    assert "webfetch" in dropped[0], dropped
    print("  _detect_literal_tool_calls: unknown tool name -> dropped note "
          "(no silent rename)")


# ---------------- #63: rebuild note when history mentions native tool use ----
def test_history_mentions_native_tool_use_on_assistant_tool_call():
    """A history whose assistant message has a tool_call to a known
    Claude Code native tool (`Bash`, `Read`, etc.) returns True, so the
    rebuild note gets prepended on resume. The check keys on the un-
    qualified tool name the model emits in its native session, not on
    `mcp__switchyard__*` (which is the relay's own wire shape).
    """
    messages = [
        {"role": "system", "content": "you have tools"},
        {"role": "user", "content": "show me /etc/passwd"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "x", "type": "function",
                          "function": {"name": "Bash",
                                       "arguments": "{}"}}]},
    ]
    assert server._history_mentions_native_tool_use(messages), messages
    print("  _history_mentions_native_tool_use: assistant Bash tool_call "
          "-> True (rebuild note gets prepended)")


def test_history_mentions_native_tool_use_ignores_bridged_tool_names():
    """A history whose only tool calls are bridged (mcp__switchyard__Bash
    etc.) returns False -- the rebuilt session is still serving the
    bridge, and the "tool surface changed" note would be wrong there.
    """
    messages = [
        {"role": "system", "content": "you have tools"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "x", "type": "function",
                          "function": {"name": "mcp__switchyard__Bash",
                                       "arguments": "{}"}}]},
    ]
    assert not server._history_mentions_native_tool_use(messages), messages
    print("  _history_mentions_native_tool_use: mcp__switchyard__Bash only "
          "-> False (no rebuild note)")


def test_history_mentions_native_tool_use_ignores_pure_chat():
    """A history with no assistant tool calls at all returns False. Pure
    text conversations do not need the rebuild note -- the inner CLI sees
    the same tool surface it always did, just rebuilt from the caller's
    full history.
    """
    messages = [
        {"role": "system", "content": "you are a helper"},
        {"role": "user", "content": "explain rebuilds"},
        {"role": "assistant", "content": "rebuilds are ..."},
    ]
    assert not server._history_mentions_native_tool_use(messages), messages
    print("  _history_mentions_native_tool_use: pure chat history -> False "
          "(no rebuild note)")


def test_history_mentions_native_tool_use_picks_up_bracket_narration():
    """Bracket-form narration `[called Bash(...)]` in an assistant text
    block -- the shape `flatten_with_tool_history` emits on a rebuild --
    ALSO triggers the rebuild note. The rebuilt prompt re-renders the
    loop as narration; if that narration names a native tool, the model
    will try to call it again unless reminded the surface changed.
    """
    messages = [
        {"role": "system", "content": "you have tools"},
        {"role": "assistant", "content":
         'Here is what I ran: [called Bash({"cmd":"ls"})]'},
    ]
    assert server._history_mentions_native_tool_use(messages), messages
    print("  _history_mentions_native_tool_use: [called Bash(...)] narration "
          "-> True (rebuild note prepended)")


# ---------------- render_turn integration: literal tool call lands parked ----
def test_render_turn_lifts_literal_tool_call_to_parked_call():
    """The render_turn final branch's literal-tool-call detection must
    actually turn a matched pattern into a parked `ParkedCall` (issue #255,
    end-to-end pin). Driving render_turn with a final payload whose result
    is an XML `<tool_use>` block verifies that:

      * the assistant content is the cleaned text (pattern stripped);
      * a new `call_<sid>_<n>` id is minted on `session.pending`;
      * the response shape carries a tool_calls turn with finish_reason
        "tool_calls" so the caller treats it like a real tool round;
      * the dropped notes surface on `__literal_dropped` for the next
        follow-up's rebuild (issue #255's "system note on the next turn").

    The session is hand-built (no spawn) so the test stays in-process.
    """
    async def go():
        workdir = tempfile.mkdtemp(prefix="mcpb-lit-render-")
        try:
            session = server.Session(
                id=uuid.uuid4().hex, provider="claude", model="m",
                workdir=workdir, max_tokens=None)
            server.SESSIONS[session.id] = session
            # Stash the tool list so REMEMBERED_TOOLS[name] returns it.
            server.remember_tools(session.id, [{"type": "function",
                "function": {"name": "bash",
                              "parameters": {"type": "object",
                                             "properties": {}}}}])
            payload = {"result":
                'Sure, calling ls.\n<tool_use name="bash">'
                '{"cmd":"ls -la"}</tool_use>\nDone.',
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "total_tokens": 2}}
            result = {"type": "final", "payload": payload}
            response = server.render_turn(session, result, "m")
            # tool_calls response shape, not a text reply.
            choice = response["choices"][0]
            assert choice["finish_reason"] == "tool_calls", response
            msg = choice["message"]
            assert msg["tool_calls"], msg
            tc = msg["tool_calls"][0]
            assert tc["function"]["name"] == "bash", tc
            args = json.loads(tc["function"]["arguments"])
            assert args == {"cmd": "ls -la"}, args
            # The literal pattern is gone from the assistant content.
            content = msg.get("content") or ""
            assert "<tool_use" not in content, content
            assert "calling ls" in content, content
            # The session now has a parked call keyed by the minted id.
            parked_ids = list(session.pending.keys())
            assert parked_ids, session.pending
            call = session.pending[parked_ids[0]]
            assert call.name == "bash", call
            assert call.arguments == {"cmd": "ls -la"}, call
        finally:
            server.SESSIONS.pop(session.id, None)
            shutil.rmtree(workdir, ignore_errors=True)
        print("  render_turn: <tool_use> block in final payload becomes a "
              "parked tool_call; cleaned content rides alongside; "
              "session.pending carries the new call")

    asyncio.run(go())


# ---------------- #264: chat() routing for fingerprint-matched nudges ---------
# The chat()-level tests below patch the LOWER-LEVEL helpers
# (`supersede_session`, `resume_gone_session`, `cli_bridge._handle_chat`),
# not `_continue_parked_nudge` itself, so the helper's body is
# exercised end-to-end (cycle-2 review: stubbing the helper at the
# chat() layer left the helper's own logic -- the vanish-rebuild branch,
# the parked-session supersede, the body["tools"] fall-through -- with
# zero coverage). The helper's own test
# (`test_continue_parked_nudge_supersedes_and_rebuilds_promptly`) drives
# the real helper against a stalled parked session and asserts it
# returns quickly rather than hanging for SESSION_TTL.
def _patch_resume_layer(parked):
    """Wire up the lower-level helpers so chat()'s fingerprint routing
    can be observed end-to-end without the real supersede / rebuild
    paths running (they would spawn a fresh CLI). The fingerprint
    helper itself runs for real, so the chat() routing decision
    is genuine."""
    superseded: list[tuple] = []
    resumed: list[tuple] = []
    real_supersede = server.supersede_session
    real_resume = server.resume_gone_session
    real_handle_chat = server.cli_bridge._handle_chat
    handle_chat_called: list[dict] = []

    async def fake_supersede(session, reason):
        superseded.append((session.id, reason))

    async def fake_resume(body, tools, session_id, request, why):
        resumed.append((session_id, list(tools), why))
        return {"id": "fake-resume", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": "rebuilt"}}]}

    async def fake_handle_chat(body):
        handle_chat_called.append(body)
        return {"id": "fake-fresh", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": "fresh"}}]}

    server.supersede_session = fake_supersede
    server.resume_gone_session = fake_resume
    server.cli_bridge._handle_chat = fake_handle_chat
    return superseded, resumed, handle_chat_called, (real_supersede,
            real_resume, real_handle_chat)


def _restore_resume_layer(saved):
    real_supersede, real_resume, real_handle_chat = saved
    server.supersede_session = real_supersede
    server.resume_gone_session = real_resume
    server.cli_bridge._handle_chat = real_handle_chat


def test_chat_routes_fingerprint_nudge_to_parked_session():
    """Issue #264 nudge continuity: a tools-free body whose fingerprint
    matches a parked session is recognised by chat(); the new
    `_continue_parked_nudge` helper then supersedes the parked session
    (the CLI cannot ingest user text mid-loop) and rebuilds via
    `resume_gone_session`. The end-to-end check is at the chat()
    layer (the level the actual routing bug lives at): drive chat()
    with a body carrying no tools and no tool messages, and assert
    that (a) `_continue_parked_nudge` was reached, (b) the parked
    session's supersede was logged, (c) the rebuild returned the
    expected response.

    This is the test that would have caught the dead routing in PR
    #339 -- the fingerprint helper test alone left the chat() layer
    unobserved.
    """
    with _fingerprint_helper_installed() as fp_helper:
        parked = _new_session()
        parked.awaiting_followup = True
        parked.last_active = time.time() - 5
        parked_tools = [{"type": "function",
                         "function": {"name": "bash",
                                      "parameters": {"type": "object",
                                                     "properties": {}}}}]
        server.remember_tools(parked.id, parked_tools)

        messages = [
            {"role": "system", "content": "you are a helper"},
            {"role": "user", "content": "first nudge"},
        ]
        parked.fingerprint = fp_helper.fingerprint(messages)

        superseded, resumed, handle_chat_called, saved = _patch_resume_layer(
            parked)
        try:
            from fastapi.testclient import TestClient
            http = TestClient(server.app)
            body = {"model": "m", "messages": messages}
            resp = http.post("/v1/chat/completions", json=body)
            assert resp.status_code == 200, resp.text
            data = resp.json()
            # Rebuild returned the expected response -- the parked CLI
            # was superseded and a fresh CLI was rebuilt from history.
            assert data["choices"][0]["message"]["content"] == "rebuilt", data
            assert len(superseded) == 1, superseded
            assert superseded[0][0] == parked.id, superseded
            assert len(resumed) == 1, resumed
            assert resumed[0][0] == parked.id, resumed
            # The remembered tools were passed to the rebuild.
            assert resumed[0][1] == parked_tools, resumed
            # Fresh text path was NOT taken.
            assert not handle_chat_called, handle_chat_called
        finally:
            _restore_resume_layer(saved)
            _drop(parked)

    print("  chat(): tools-free body fingerprint-matching parked session "
          "-> supersede + resume_gone_session (rebuilt), not fresh text "
          "path; parked CLI cannot ingest user text mid-loop")


def test_chat_skips_routing_when_parked_session_lacks_remembered_tools():
    """When the fingerprint-matched parked session has no entry in
    REMEMBERED_TOOLS (the dict is bounded; a long-running process can
    evict a session's tools), chat() must NOT crash and the parked
    session is superseded, with the rebuild falling through to the
    text path rather than driving an empty tool list into the harness.
    The caller still gets an answer.
    """
    with _fingerprint_helper_installed() as fp_helper:
        parked = _new_session()
        parked.awaiting_followup = True
        parked.last_active = time.time() - 5
        # Intentionally NOT remembering tools for this session.
        messages = [
            {"role": "system", "content": "you are a helper"},
            {"role": "user", "content": "first nudge"},
        ]
        parked.fingerprint = fp_helper.fingerprint(messages)

        superseded, resumed, handle_chat_called, saved = _patch_resume_layer(
            parked)
        try:
            from fastapi.testclient import TestClient
            http = TestClient(server.app)
            body = {"model": "m", "messages": messages}
            resp = http.post("/v1/chat/completions", json=body)
            assert resp.status_code == 200, resp.text
            # REMEMBERED_TOOLS eviction -> no tools on body ->
            # supersede_session fires but resume_gone_session is NOT
            # called (no tools to rebuild with), the text path is taken.
            assert len(superseded) == 1, superseded
            assert not resumed, resumed
            assert len(handle_chat_called) == 1, handle_chat_called
        finally:
            _restore_resume_layer(saved)
            _drop(parked)

    print("  chat(): fingerprint-matched parked session without "
          "remembered tools -> supersede + cli_bridge._handle_chat "
          "(text path; no empty-tool inference)")


def test_chat_skips_routing_on_ambiguous_fingerprint_match():
    """Two parked sessions share the same (system + first user)
    fingerprint. The chat() layer must NOT pick one and route to it
    (silent cross-wire). The call falls through to the fresh text path
    -- the "one tool loop == one conversation" invariant wins.
    """
    with _fingerprint_helper_installed() as fp_helper:
        older = _new_session()
        older.awaiting_followup = True
        older.last_active = time.time() - 30
        server.remember_tools(older.id, [{"type": "function",
                                          "function": {"name": "older",
                                                       "parameters": {"type": "object",
                                                                      "properties": {}}}}])
        newer = _new_session()
        newer.awaiting_followup = True
        newer.last_active = time.time() - 5
        server.remember_tools(newer.id, [{"type": "function",
                                          "function": {"name": "newer",
                                                       "parameters": {"type": "object",
                                                                      "properties": {}}}}])
        messages = [
            {"role": "system", "content": "you are a helper"},
            {"role": "user", "content": "first nudge"},
        ]
        fp = fp_helper.fingerprint(messages)
        older.fingerprint = fp
        newer.fingerprint = fp

        superseded, resumed, handle_chat_called, saved = _patch_resume_layer(
            older)
        try:
            from fastapi.testclient import TestClient
            http = TestClient(server.app)
            body = {"model": "m", "messages": messages}
            resp = http.post("/v1/chat/completions", json=body)
            assert resp.status_code == 200, resp.text
            # Ambiguous match -> no session picked, no supersede, no
            # rebuild, just the text path.
            assert not superseded, (
                f"chat() must not pick one of two ambiguous matches, but "
                f"superseded {superseded}")
            assert not resumed, resumed
            assert len(handle_chat_called) == 1, handle_chat_called
        finally:
            _restore_resume_layer(saved)
            _drop(older)
            _drop(newer)

    print("  chat(): ambiguous fingerprint (2+ parked sessions) -> "
          "fresh text path (no silent cross-wire)")


# ---------------- #264: _continue_parked_nudge body coverage -----------------
def test_continue_parked_nudge_supersedes_and_rebuilds_promptly():
    """The cycle-2 review blocker noted the helper installs a fresh
    `turn_future` but never delivers the nudge to the parked CLI --
    proc.stdin was closed at spawn, the CLI is blocked on tool_server
    waiting for tool results, and `await_turn` therefore hangs until
    the idle reaper rebuilds at SESSION_TTL.

    The fix is to supersede the parked session and rebuild from the
    request's history. Drive the REAL helper against a parked session
    that has a `last_response` (so the cached-response path doesn't
    short-circuit) and assert:

      * the parked session was superseded,
      * `resume_gone_session` was called with the remembered tools,
      * the helper returned promptly (no hang).

    `resume_gone_session` is mocked to return a known response; the
    real helper drives the supersede + rebuild decision and returns
    the rebuilt body without ever touching `await_turn`.
    """
    with _fingerprint_helper_installed() as fp_helper:
        parked = _new_session()
        parked.awaiting_followup = True
        parked.last_active = time.time() - 5
        parked.last_response = {"id": "prior", "object": "chat.completion",
                                "choices": [{"message": {"role": "assistant",
                                                          "content": "tool call"}}]}
        parked_tools = [{"type": "function",
                         "function": {"name": "bash",
                                      "parameters": {"type": "object",
                                                     "properties": {}}}}]
        server.remember_tools(parked.id, parked_tools)

        messages = [
            {"role": "system", "content": "you are a helper"},
            {"role": "user", "content": "first nudge"},
        ]
        parked.fingerprint = fp_helper.fingerprint(messages)

        superseded: list[tuple] = []
        resumed: list[tuple] = []
        real_supersede = server.supersede_session
        real_resume = server.resume_gone_session

        async def fake_supersede(session, reason):
            superseded.append((session.id, reason))
            # Mark dead so a subsequent test would see this session as gone.
            session.dead = True
            server.SESSIONS.pop(session.id, None)

        async def fake_resume(body, tools, session_id, request, why):
            resumed.append((session_id, list(tools), why))
            return {"id": "fake-rebuild", "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": "rebuilt"}}]}

        server.supersede_session = fake_supersede
        server.resume_gone_session = fake_resume
        try:
            body = {"model": "m", "messages": messages,
                    "tools": parked_tools}

            async def go():
                # If the helper hung, this would block until the test
                # timeout (a clear regression signal). The bounded
                # asyncio.wait_for below turns "the helper returned
                # promptly" into a hard assertion.
                return await asyncio.wait_for(
                    server._continue_parked_nudge(parked, body, None),
                    timeout=5.0)

            result = asyncio.run(go())
            # Supersede fired for the parked session with the documented
            # reason (mid-loop / stdin closed) -- not a vague reason.
            assert len(superseded) == 1, superseded
            assert superseded[0][0] == parked.id, superseded
            assert "stdin" in superseded[0][1] or "mid-loop" in superseded[0][1], \
                superseded
            # Rebuild was called with the parked session's remembered
            # tools, NOT an empty list (the helper did the body stamping
            # before delegating).
            assert len(resumed) == 1, resumed
            assert resumed[0][0] == parked.id, resumed
            assert resumed[0][1] == parked_tools, resumed
            # The helper returned the rebuilt response, NOT a hung
            # turn_future. This is the regression net for the cycle-2
            # blocker.
            assert result["choices"][0]["message"]["content"] == "rebuilt", \
                result
            # The parked session was superseded -- it should no longer
            # be in SESSIONS or be considered live.
            assert parked.dead, parked.dead
            assert parked.id not in server.SESSIONS, parked.id
        finally:
            server.supersede_session = real_supersede
            server.resume_gone_session = real_resume
            _drop(parked)

    print("  _continue_parked_nudge: real helper, stalled parked CLI "
          "-> supersede(parked) + resume_gone_session(rebuilt), "
          "returns promptly (no SESSION_TTL hang)")


def test_continue_parked_nudge_falls_through_when_no_remembered_tools():
    """A fingerprint-matched nudge for a session whose REMEMBERED_TOOLS
    entry has been evicted: the helper supersedes the parked session
    and falls through to `cli_bridge._handle_chat` (the text path).
    `resume_gone_session` is NOT called because the helper has no tools
    to rebuild with (resume_gone_session would 400 on an empty tool
    list). The caller still gets an answer.
    """
    with _fingerprint_helper_installed() as fp_helper:
        parked = _new_session()
        parked.awaiting_followup = True
        parked.last_active = time.time() - 5
        # Intentionally NOT remembering tools for this session.
        messages = [
            {"role": "system", "content": "you are a helper"},
            {"role": "user", "content": "first nudge"},
        ]
        parked.fingerprint = fp_helper.fingerprint(messages)

        superseded: list[tuple] = []
        resumed: list[tuple] = []
        handle_chat_called: list[dict] = []
        real_supersede = server.supersede_session
        real_resume = server.resume_gone_session
        real_handle_chat = server.cli_bridge._handle_chat

        async def fake_supersede(session, reason):
            superseded.append((session.id, reason))
            session.dead = True
            server.SESSIONS.pop(session.id, None)

        async def fake_resume(body, tools, session_id, request, why):
            resumed.append((session_id, list(tools), why))
            return {"id": "should-not-fire"}

        async def fake_handle_chat(body):
            handle_chat_called.append(body)
            return {"id": "fake-fresh", "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": "text path"}}]}

        server.supersede_session = fake_supersede
        server.resume_gone_session = fake_resume
        server.cli_bridge._handle_chat = fake_handle_chat
        try:
            body = {"model": "m", "messages": messages}

            async def go():
                return await asyncio.wait_for(
                    server._continue_parked_nudge(parked, body, None),
                    timeout=5.0)

            result = asyncio.run(go())
            # Supersede fired (parked session is still dead regardless
            # of whether we have tools to rebuild with).
            assert len(superseded) == 1, superseded
            assert superseded[0][0] == parked.id, superseded
            # But resume_gone_session was NOT called -- empty tools
            # would 400. The text path is taken.
            assert not resumed, resumed
            assert len(handle_chat_called) == 1, handle_chat_called
            assert result["choices"][0]["message"]["content"] == "text path", \
                result
        finally:
            server.supersede_session = real_supersede
            server.resume_gone_session = real_resume
            server.cli_bridge._handle_chat = real_handle_chat
            _drop(parked)

    print("  _continue_parked_nudge: REMEMBERED_TOOLS evicted -> "
          "supersede(parked) + cli_bridge._handle_chat (text path), "
          "no resume_gone_session call (would 400 on empty tools)")


def test_continue_parked_nudge_rebuilds_when_session_vanished():
    """Race: the fingerprint match returned `parked` but between then
    and the helper's `SESSIONS.get(parked.id)` the session was reaped
    (or superseded by a parallel follow-up). The helper must rebuild
    from the request without trying to supersede a session that is no
    longer live, and the rebuild must succeed.
    """
    with _fingerprint_helper_installed() as fp_helper:
        parked = _new_session()
        parked.awaiting_followup = True
        parked.last_active = time.time() - 5
        # REMEMBERED_TOOLS is empty -- simulate a session that vanished
        # AND lost its remembered tools, so the helper takes the text
        # path through cli_bridge._handle_chat.
        messages = [
            {"role": "system", "content": "you are a helper"},
            {"role": "user", "content": "first nudge"},
        ]
        parked.fingerprint = fp_helper.fingerprint(messages)

        superseded: list[tuple] = []
        resumed: list[tuple] = []
        handle_chat_called: list[dict] = []
        real_supersede = server.supersede_session
        real_resume = server.resume_gone_session
        real_handle_chat = server.cli_bridge._handle_chat

        async def fake_supersede(session, reason):
            superseded.append((session.id, reason))

        async def fake_resume(body, tools, session_id, request, why):
            resumed.append((session_id, list(tools), why))
            return {"id": "fake-rebuild", "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": "rebuilt-vanish"}}]}

        async def fake_handle_chat(body):
            handle_chat_called.append(body)
            return {"id": "fake-fresh", "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": "text path"}}]}

        server.supersede_session = fake_supersede
        server.resume_gone_session = fake_resume
        server.cli_bridge._handle_chat = fake_handle_chat
        # Now race: reaper collects parked between fingerprint-match and
        # helper entry. Without remembered tools, the helper takes the
        # text path.
        parked_tools = []
        server.REMEMBERED_TOOLS.pop(parked.id, None)
        # Mark the parked session as gone BEFORE the helper runs.
        parked.dead = True
        server.SESSIONS.pop(parked.id, None)
        try:
            body = {"model": "m", "messages": messages,
                    "tools": parked_tools}

            async def go():
                return await asyncio.wait_for(
                    server._continue_parked_nudge(parked, body, None),
                    timeout=5.0)

            result = asyncio.run(go())
            # Helper saw the session was gone -> no supersede (we never
            # try to supersede a dead session), and took the text path
            # because the body has no tools.
            assert not superseded, (
                f"helper must not supersede a vanished session, but "
                f"superseded {superseded}")
            assert not resumed, resumed
            assert len(handle_chat_called) == 1, handle_chat_called
            assert result["choices"][0]["message"]["content"] == "text path", \
                result
        finally:
            server.supersede_session = real_supersede
            server.resume_gone_session = real_resume
            server.cli_bridge._handle_chat = real_handle_chat

    print("  _continue_parked_nudge: parked session vanished between "
          "fingerprint match and helper entry -> no supersede attempt, "
          "no resume_gone_session call, falls through to text path")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
