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
import contextlib
import errno
import http.server
import json
import os
import socketserver
import subprocess
import sys
import tempfile
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()
os.environ.setdefault("SIDECAR_PORT", "8081")

from _modules import load  # noqa: E402

server = load("mcp_bridge_server", os.path.join(MCP_BRIDGE_DIR, "server.py"))

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

    async def fake(body, mcp_tools, prompt, system, model, request):
        recorder.append({"tools": mcp_tools, "prompt": prompt,
                         "system": system, "model": model})
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

            async def fake_start(body, mcp_tools, prompt, system, model, request):
                calls.append({"prompt": prompt})
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


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} mcp-bridge tests passed")
