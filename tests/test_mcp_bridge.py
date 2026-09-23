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
import base64
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

    async def fake(body, mcp_tools, prompt, system, model, request,
                   image_paths=None, img_dir=None, env=None, first_turn=True):
        # Mirror the env-injection + first-turn-reminder logic the real
        # start_session runs before build_argv. Without this the captured
        # `system`/`prompt` would reflect what handle_fresh passed, not
        # what the CLI actually saw -- and the issue #44 contract is
        # specifically about what the CLI sees.
        if env is not None and server._caller_env is not None:
            env_block = server._caller_env.render_system_block(env)
            system = (f"{system}\n\n{env_block}" if system else env_block)
            if first_turn:
                reminder = server._caller_env.render_first_turn_reminder(env)
                prompt = f"{reminder}\n\n{prompt}" if prompt else reminder
        recorder.append({"tools": mcp_tools, "prompt": prompt,
                         "system": system, "model": model,
                         "image_paths": image_paths,
                         "env": env, "first_turn": first_turn})
        if img_dir is not None:
            import shutil
            shutil.rmtree(img_dir, ignore_errors=True)
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


# ------------------------------------------------- issue #64: TEXT_LOST -----
def test_run_session_emits_text_lost_contract_detail_for_no_text_with_usage():
    """Issue #64 (reviewer finding on PR #73): opencode-go and opencode-go2
    plans run mcp_bridge, NOT cli_bridge, so cli_bridge's contract-string fix
    was dead code for them. mcp_bridge's run_session calls parse_output
    directly (no in-process retry of its own — cli_bridge owns that) and was
    catching CliNoTextError as a plain ValueError, emitting the legacy
    "yielded no parsed answer" detail. The gateway's _NO_TEXT regex does
    not match that, so the affected plans kept riding the TRANSIENT doubling
    ladder while the ledger still lost the charged tokens.

    The mcp_bridge path must now catch CliNoTextError specifically and emit
    the same contract detail cli_bridge uses (same helper, same string),
    so classify.TEXT_LOST fires and extract_no_text_tokens books the
    charged prompt/completion counts.

    Issue #78 follow-up: mcp_bridge now retries once on no-text-with-usage
    (matching cli_bridge._run_cli). The contract 502 only fires when BOTH
    attempts lose text; the detail must carry the COMBINED usage of both
    attempts so the ledger books everything that was charged. The fake
    below emits no-text-with-usage on both calls (different token counts
    each time) so the combined-usage assertion has both numbers visible.
    """
    counter_path = _write_fake_cli_counter()

    call_one = ('{"type":"step_start","part":{"type":"step-start"}}\n'
                '{"type":"step_finish","part":{"type":"step-finish",'
                '"tokens":{"input":239,"output":22}}}')
    call_two = ('{"type":"step_start","part":{"type":"step-start"}}\n'
                '{"type":"step_finish","part":{"type":"step-finish",'
                '"tokens":{"input":11,"output":3}}}')
    fake_cli = _write_counter_fake_cli(counter_path, on_call_one=call_one,
                                       on_call_other=call_two)

    saved_parser = server.PROFILE["parser"]
    try:
        # mcp_bridge dispatches on PROFILE["parser"]. The test's default
        # fixture is the claude provider (claude_json), whose parser does
        # a bare json.loads and so does not raise CliNoTextError on this
        # stream — it raises JSONDecodeError. Force the opencode events
        # parser so the issue #64 path is actually exercised.
        server.PROFILE["parser"] = "events_json"

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "error", result
        assert result["status"] == 502, result
        detail = result.get("detail", "")
        # The contract phrase the gateway's _NO_TEXT regex matches. The
        # provider name comes from mcp_bridge.PROVIDER, not cli_bridge's —
        # so the test asserts on the structure, not the literal prefix.
        assert "cli emitted tokens with no text " in detail, detail
        # Both attempts lost text: 239 + 11 prompt, 22 + 3 completion.
        assert "(prompt_tokens=250, completion_tokens=25)" in detail, detail
        # And mcp_bridge does NOT reach the legacy "yielded no parsed
        # answer" branch on this shape — that was the bug.
        assert "yielded no parsed answer" not in detail, detail
        # Both sides use the same helper, so the strings must match
        # byte-for-byte: cli_bridge.format_no_text_detail(PROVIDER, usage).
        # The PROVIDER value here is whatever the test's os.environ set
        # ("claude"), so the prefix is provider-specific.
        expected = server.cli_bridge.format_no_text_detail(
            server.PROVIDER,
            {"input_tokens": 250, "output_tokens": 25})
        assert detail == expected, (detail, expected)
        # And the retry actually fired -- two spawns, not one.
        calls = _read_counter(counter_path)
        assert calls == 2, f"expected 2 spawns (1st + retry), got {calls}"
        print(f"  MCP session: no-text-with-usage x2 -> 502 with combined usage "
              f"({detail!r}, calls={calls})")
    finally:
        server.PROFILE["parser"] = saved_parser


def test_run_session_legacy_no_text_without_usage_still_emits_no_parsed_answer():
    """Regression guard: a no-text stream with ZERO charged usage must NOT
    take the new CliNoTextError branch — it has no usage to carry, so the
    legacy "yielded no parsed answer" 502 stays. cli_bridge._run_cli's
    gating on nonzero input_tokens / output_tokens / provider_cost applies
    here too (parse_output is shared), so an empty-stdout / zero-usage
    failure looks the same as it did before this PR."""
    bad_stream = ('{"type":"step_start","part":{"type":"step-start"}}\n'
                  '{"type":"step_finish","part":{"type":"step-finish"}}')
    fake_cli = _write_fake_cli(
        "import sys\n"
        f"sys.stdout.write({bad_stream!r})\n")

    saved_parser = server.PROFILE["parser"]
    try:
        server.PROFILE["parser"] = "events_json"

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "error", result
        assert result["status"] == 502, result
        detail = result.get("detail", "")
        assert "yielded no parsed answer" in detail, detail
        # And, critically, NOT the new contract shape — there are no
        # charged tokens to quote, so the contract phrase would be a lie.
        assert "emitted tokens with no text" not in detail, detail
        print(f"  MCP session: zero-usage no-text -> legacy 502 "
              f"({detail[:60]}...)")
    finally:
        server.PROFILE["parser"] = saved_parser


# ---------------------------------------- issue #78: TEXT_LOST retry-once ----
def _write_fake_cli_counter() -> str:
    """Return a fresh counter file path initialised to "0".

    Tests below use the counter-file fake-CLI idiom from tests/test_cli_bridge.py
    (see _write_counter_fake_cli): the fake increments the counter file on every
    spawn, so the test can assert how many times the CLI was actually invoked.
    A fresh mkdtemp avoids colliding with any other counter file in flight,
    and pre-creating the file with "0" sidesteps FileNotFoundError inside the
    fake CLI's `open(p).read() or '0'` on its very first spawn.
    """
    d = tempfile.mkdtemp(prefix="mcpb-cnt-")
    p = os.path.join(d, "counter")
    open(p, "w").write("0")
    return p


def _write_counter_fake_cli(counter_path: str, *, on_call_one: str = "",
                            on_call_other: str = "", stderr: str = "",
                            rc: int = 0) -> str:
    """Write a tiny Python CLI that increments `counter_path` each call.

    `on_call_one` is printed on the first call, `on_call_other` on every
    subsequent call. Same shape as test_cli_bridge.py's make_fake. Mirrors
    the cli_bridge retry tests (issue #64) so the spawn-count assertion is
    apples-to-apples against that reference.
    """
    body = (f"import sys\n"
            f"p = {counter_path!r}\n"
            f"n = int(open(p).read() or '0') + 1\n"
            f"open(p, 'w').write(str(n))\n"
            f"if n == 1:\n"
            f"  sys.stdout.write({on_call_one!r})\n"
            f"else:\n"
            f"  sys.stdout.write({on_call_other!r})\n"
            f"sys.stderr.write({stderr!r})\n"
            f"sys.exit({rc})\n")
    f = tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix="mcpb-rf-", delete=False)
    f.write(body)
    f.close()
    return f.name


def _read_counter(counter_path: str) -> int:
    return int(open(counter_path).read() or "0")


def test_run_session_retries_once_on_text_lost_and_returns_attempt_two_payload():
    """Issue #78: a single no-text-with-usage attempt is retried in-process
    exactly once, and on a successful retry the caller sees attempt 2's
    payload (not the legacy "yielded no parsed answer" 502). The call
    counter proves the CLI was spawned twice (attempt 1 + retry), and the
    payload's usage dict carries the merged token counts of both attempts
    so the ledger books each spawn exactly once.

    The retry semantics here mirror cli_bridge._run_cli, the reference
    implementation. opencode-go / opencode-go2 plans run mcp_bridge, so
    without this in-process retry the plan rides the TRANSIENT doubling
    ladder (issue #64) — the gateway has no way to know the first attempt
    was a parse-level loss, not a transport-level loss.
    """
    counter_path = _write_fake_cli_counter()
    bad = ('{"type":"step_finish","part":{"type":"step-finish",'
           '"tokens":{"input":777,"output":33}}}')
    good = ('{"type":"text","part":{"type":"text","text":"ANSWER"}}\n'
            '{"type":"step_finish","part":{"type":"step-finish",'
            '"tokens":{"input":111,"output":7,"reasoning":2}}}')
    fake_cli = _write_counter_fake_cli(counter_path, on_call_one=bad,
                                       on_call_other=good)

    saved_parser = server.PROFILE["parser"]
    try:
        server.PROFILE["parser"] = "events_json"

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "final", result
        assert result["payload"]["result"] == "ANSWER", result
        u = result["payload"]["usage"]
        # 777 + 111 = 888 input tokens across both attempts.
        assert u["input_tokens"] == 888, u
        # 33 from attempt 1, 7+2 reasoning from attempt 2; reasoning rolls
        # into output_tokens, so 33 + 9 = 42.
        assert u["output_tokens"] == 42, u
        # And exactly two spawns happened -- the retry, not an infinite loop.
        calls = _read_counter(counter_path)
        assert calls == 2, f"successful retry must call CLI twice, got {calls}"
        print(f"  MCP retry-once: bad+good -> ANSWER (input={u['input_tokens']}, "
              f"output={u['output_tokens']}, calls={calls})")
    finally:
        server.PROFILE["parser"] = saved_parser


def test_run_session_two_consecutive_text_lost_returns_502_with_combined_usage():
    """Issue #78 contract path: when both attempts lose text, the final 502
    detail must match the gateway's contract exactly (same helper as
    cli_bridge, so the strings cannot drift):
        {PROVIDER} cli emitted tokens with no text
            (prompt_tokens=N, completion_tokens=M)
    where N/M are the COMBINED charges of both attempts. The byte-for-byte
    detail check is load-bearing -- the gateway's _NO_TEXT regex matches
    the literal phrase, and extract_no_text_tokens parses the parenthesised
    numbers, so any drift here would silently change the cooldown class
    and the ledger.
    """
    counter_path = _write_fake_cli_counter()
    bad_one = ('{"type":"step_finish","part":{"type":"step-finish",'
               '"tokens":{"input":239,"output":22}}}')
    bad_two = ('{"type":"step_finish","part":{"type":"step-finish",'
               '"tokens":{"input":50,"output":5}}}')
    fake_cli = _write_counter_fake_cli(counter_path, on_call_one=bad_one,
                                       on_call_other=bad_two)

    saved_parser = server.PROFILE["parser"]
    try:
        server.PROFILE["parser"] = "events_json"

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "error", result
        assert result["status"] == 502, result
        detail = result.get("detail", "")
        # 239 + 50 prompt, 22 + 5 completion — must match the contract shape
        # via format_no_text_detail (same helper cli_bridge uses), byte-for-byte.
        expected = server.cli_bridge.format_no_text_detail(
            server.PROVIDER,
            {"input_tokens": 289, "output_tokens": 27})
        assert detail == expected, (detail, expected)
        # And the legacy "yielded no parsed answer" branch did NOT fire —
        # there is real charged usage to carry, and it must reach the
        # contract so the ledger books both attempts.
        assert "yielded no parsed answer" not in detail, detail
        # Two spawns -- 1st attempt + retry -- proves the retry fired.
        calls = _read_counter(counter_path)
        assert calls == 2, f"two failed attempts must call CLI twice, got {calls}"
        print(f"  MCP retry-once: bad+bad -> 502 with combined usage "
              f"({detail!r}, calls={calls})")
    finally:
        server.PROFILE["parser"] = saved_parser


def test_run_session_only_text_lost_triggers_retry_other_failures_stay_single_attempt():
    """Issue #78 narrow retry: only CliNoTextError (no-text with nonzero
    charged usage) triggers the retry. Auth, quota, empty stdout, and
    generic rc=1 each stay single-attempt — retrying those would either
    double-bill the caller (limit) or amplify blips into double spawns
    without improving the outcome. The counter file is the assertion:
    calls == 1 in every sub-scenario.

    Same shape as tests/test_cli_bridge.py:1015-1056 (the cli_bridge
    counterpart); the only difference is the mcp_bridge run_session surface
    (no HTTPException — error info lands in session.turn_future.result()).
    """
    counter_path = _write_fake_cli_counter()

    def run(stderr: str, rc: int) -> dict:
        # A fresh fake per scenario so the counter path is shared and the
        # branch is determined entirely by the args passed in here.
        open(counter_path, "w").write("0")
        fake_cli = _write_counter_fake_cli(counter_path, stderr=stderr, rc=rc)

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        return asyncio.run(scenario())

    try:
        # --- auth failure: single attempt, 401 ---
        result = run(stderr="please run `claude login` to authenticate", rc=1)
        assert result["type"] == "error" and result["status"] == 401, result
        assert _read_counter(counter_path) == 1, \
            f"auth failure must not retry (calls={_read_counter(counter_path)})"

        # --- quota failure: single attempt, 429 ---
        result = run(stderr="You've hit your usage limit. Purchase more credits.",
                     rc=1)
        assert result["type"] == "error" and result["status"] == 429, result
        assert _read_counter(counter_path) == 1, \
            f"quota failure must not retry (calls={_read_counter(counter_path)})"

        # --- empty stdout: single attempt, 502 ---
        # rc=0 with no stdout trips the `not stdout.strip()` branch and falls
        # through to the generic "cli failed (0)" 502 — must NOT retry.
        result = run(stderr="", rc=0)
        assert result["type"] == "error" and result["status"] == 502, result
        assert "cli failed" in result.get("detail", ""), result
        assert _read_counter(counter_path) == 1, \
            f"empty stdout must not retry (calls={_read_counter(counter_path)})"

        # --- generic rc=1 with no recognisable shape: 502 "cli failed", 1 call ---
        result = run(stderr="some unrecognised failure prose", rc=1)
        assert result["type"] == "error" and result["status"] == 502, result
        assert _read_counter(counter_path) == 1, \
            f"generic rc=1 must not retry (calls={_read_counter(counter_path)})"

        print(f"  MCP retry-once: auth/quota/empty/generic each spawn once; "
              f"no-text-with-usage is the ONLY retried shape")
    finally:
        try:
            os.unlink(counter_path)
        except OSError:
            pass


def test_run_session_successful_retry_merges_attempt_one_usage_into_payload():
    """Issue #78 ledger invariant: a successful retry must fold attempt 1's
    charged tokens into attempt 2's payload so the ledger books BOTH
    attempts exactly once. `_sum_usage` adds numeric fields and leaves
    non-numeric (and bool) fields untouched — so the final usage dict is
    attempt 2's payload usage with attempt 1's numeric counts summed in,
    and any non-numeric fields attempt 1 happened to carry are not
    retroactively created on attempt 2's dict.

    This pins the exact dict the gateway's record() sees on a retry — if
    a future change dropped the merge (or summed into the wrong dict),
    the ledger would silently lose attempt 1's tokens.
    """
    counter_path = _write_fake_cli_counter()
    bad = ('{"type":"step_finish","part":{"type":"step-finish",'
           '"tokens":{"input":777,"output":33}}}')
    good = ('{"type":"text","part":{"type":"text","text":"ANSWER"}}\n'
            '{"type":"step_finish","part":{"type":"step-finish",'
            '"tokens":{"input":111,"output":7,"reasoning":2}}}')
    fake_cli = _write_counter_fake_cli(counter_path, on_call_one=bad,
                                       on_call_other=good)

    saved_parser = server.PROFILE["parser"]
    try:
        server.PROFILE["parser"] = "events_json"

        async def scenario():
            session = _new_session()
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "final", result
        payload = result["payload"]
        assert payload["result"] == "ANSWER", payload
        u = payload["usage"]
        # Numeric fields: both attempts' counts must be summed.
        # 777 + 111 = 888 input, 33 + (7+2 reasoning) = 42 output.
        assert u["input_tokens"] == 888, u
        assert u["output_tokens"] == 42, u
        # _sum_usage only touches numeric (and not-bool) fields, so the
        # dict shape stays exactly what attempt 2 produced — just with
        # attempt 1's numeric counts folded in. The events_json parser
        # also emits cache_read_tokens (zero when no cache field), so it
        # carries through attempt 2 untouched (attempt 1 had no entry to
        # fold in). No spurious keys appear beyond what attempt 2's
        # parser produced.
        assert set(u.keys()) == {"input_tokens", "output_tokens",
                                  "cache_read_tokens"}, \
            f"unexpected keys in merged usage: {sorted(u.keys())}"
        assert u["cache_read_tokens"] == 0, u
        # And exactly two spawns happened — the retry, not an infinite loop.
        calls = _read_counter(counter_path)
        assert calls == 2, f"successful retry must call CLI twice, got {calls}"
        print(f"  MCP retry-once merged usage: keys={sorted(u.keys())}, "
              f"input={u['input_tokens']}, output={u['output_tokens']}, "
              f"calls={calls}")
    finally:
        server.PROFILE["parser"] = saved_parser


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

            async def fake_start(body, mcp_tools, prompt, system, model, request,
                                  image_paths=None, img_dir=None):
                calls.append({"prompt": prompt})
                if img_dir is not None:
                    import shutil
                    shutil.rmtree(img_dir, ignore_errors=True)
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


# ------------------------------------------ issue #80: gate-slot leak on error ---
def test_error_result_releases_gate_slot_and_removes_session():
    """A CLI failure on a fresh turn must tear the session down, not leak a slot.

    Before issue #80, every CLI subprocess failure path resolved the turn
    future with `{"type": "error", ...}` and `render_turn` then raised
    HTTPException. That exception escaped `start_session` *without* running
    `end_session`, so the Session stayed in `SESSIONS` holding its gate slot
    until the 1800 s idle reaper came round. Four such leaks physically
    filled the sidecar gate, every request 429'd `sidecar at capacity`, and
    the gateway's `ConcurrencyLearner` ratcheted the learned cap to 1.

    Drives `handle_fresh` (the new-session path) end-to-end: get_weather is
    not command-shaped so the env probe does not fire (`find_command_tool`
    returns None and `_maybe_probe` falls through under `probe: auto`), the
    gate has free capacity, and the stubbed `run_session` resolves the turn
    future with a 502 error dict. The contract: HTTPException 502 propagates,
    `gate.in_flight == 0`, `server.SESSIONS == {}`, and the captured session
    is `dead` and `not holds_slot`.
    """
    ERROR = {"type": "error", "status": 502,
             "detail": f"{server.PROVIDER} cli failed (1): boom"}
    captured_session: dict = {}

    async def scenario():
        # Same fresh-gate reason as the 503 tests above: bind the lock/
        # condition to this scenario's event loop, not one an earlier test
        # used.
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()
        real_run = server.run_session

        async def fake_run(session, argv, stdin_data=None):
            captured_session["session"] = session
            session.resolve_final(ERROR)

        server.run_session = fake_run
        try:
            body = {"model": "m",
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather", "description": "",
                        "parameters": {"type": "object",
                                       "properties": {"city": {"type": "string"}},
                                       "required": ["city"]}}}],
                    "messages": [{"role": "user", "content": "weather in Oslo?"}]}
            try:
                await server.handle_fresh(body, body["tools"], None)
            except server.HTTPException as exc:
                assert exc.status_code == 502, exc.status_code
                assert exc.detail == ERROR["detail"], exc.detail
            else:
                raise AssertionError("handle_fresh must 502 when the CLI errors")
            return captured_session.get("session")
        finally:
            server.run_session = real_run
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate

    session = asyncio.run(scenario())
    # The gate slot is back: the leak that physically filled the sidecar gate
    # under the bug is gone.
    assert server.cli_bridge._gate.in_flight == 0, \
        server.cli_bridge._gate.in_flight
    # The captured session was removed from SESSIONS, not just marked dead --
    # the leak that filled the sidecar gate under the bug. (Earlier tests in
    # this file leave their own parked sessions behind, so the dict is not
    # necessarily empty as a whole; the assertion is on THIS session.)
    assert session is not None
    assert session.id not in server.SESSIONS, \
        {sid: (s.dead, s.holds_slot)
         for sid, s in server.SESSIONS.items()}
    # end_session flipped the dead/holds_slot flags exactly as it does for the
    # happy-path final-answer case.
    assert session.dead, session
    assert not session.holds_slot, session
    print(f"  fresh turn -> 502 error; gate slot released "
          f"(in_flight={server.cli_bridge._gate.in_flight}); session dead and gone")


def test_text_lost_does_not_leak_a_slot():
    """The TEXT_LOST / WS1 contract path must tear the session down too.

    The gateway's `_NO_TEXT` regex classifies the 502 error detail that
    cli_bridge / mcp_bridge emit when the CLI printed tokens but no answer.
    Both sidecars use the same `format_no_text_detail` helper so the
    classifier cannot drift between them (issue #64). This test pins the
    exact byte-for-byte detail and asserts the no-leak invariants.

    Same harness as the regression above; the only difference is the
    detail string -- shaped exactly the way cli_bridge / mcp_bridge / the
    gateway's classifier all expect.
    """
    detail = server.cli_bridge.format_no_text_detail(
        server.PROVIDER, {"input_tokens": 239, "output_tokens": 22})
    ERROR = {"type": "error", "status": 502, "detail": detail}
    captured_session: dict = {}

    async def scenario():
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()
        real_run = server.run_session

        async def fake_run(session, argv, stdin_data=None):
            captured_session["session"] = session
            session.resolve_final(ERROR)

        server.run_session = fake_run
        try:
            body = {"model": "m",
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather", "description": "",
                        "parameters": {"type": "object",
                                       "properties": {"city": {"type": "string"}},
                                       "required": ["city"]}}}],
                    "messages": [{"role": "user", "content": "weather in Oslo?"}]}
            try:
                await server.handle_fresh(body, body["tools"], None)
            except server.HTTPException as exc:
                # Byte-for-byte equality guards the gateway's _NO_TEXT
                # regex classification: a single drift here would re-break
                # the WS1 contract downstream.
                assert exc.detail == detail, (exc.detail, detail)
            else:
                raise AssertionError("handle_fresh must 502 on the TEXT_LOST path")
            return captured_session.get("session")
        finally:
            server.run_session = real_run
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate

    session = asyncio.run(scenario())
    assert server.cli_bridge._gate.in_flight == 0, \
        server.cli_bridge._gate.in_flight
    assert session is not None
    assert session.id not in server.SESSIONS, \
        {sid: (s.dead, s.holds_slot)
         for sid, s in server.SESSIONS.items()}
    assert session.dead, session
    assert not session.holds_slot, session
    print(f"  TEXT_LOST detail preserved byte-for-byte ({detail!r}); "
          f"gate slot released; session dead and gone")


def test_followup_error_result_does_not_leak():
    """A CLI failure mid-tool-loop must tear the parked-session->continue
    path down, on the same contract as the fresh-turn path.

    Hand-builds a parked session (`_new_session()`, park a call via
    `register_tool_call`, consume the tool_calls turn, mark the session
    awaiting_followup, holds_slot=False -- the parked state from
    `test_a_parked_session_holds_no_concurrency_slot`). Then deliver a
    real tool result for the parked call (so `_continue_followup` takes
    the resolve path, not the rebuild path), and stub `await_turn` to
    return a 502 error dict. The contract: HTTPException 502 propagates,
    `gate.in_flight == 0`, the session is gone from SESSIONS, dead,
    and `not holds_slot`.

    `unpark_session` is exercised against a fresh gate with free
    capacity, so the failure mode being tested is purely the awaited
    turn returning an error -- not a saturated gate (the 503 path is
    covered by `test_unpark_returns_503_quickly_when_gate_is_full`).
    """
    ERROR = {"type": "error", "status": 502,
             "detail": f"{server.PROVIDER} cli failed (1): boom"}
    captured_session: dict = {}

    async def scenario():
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()

        # Hand-build a parked session, identical pattern to
        # test_duplicate_delivery_returns_cached_response_... /
        # test_rebuild_supersedes_the_old_session_...
        session = _new_session()
        session.new_turn()
        parked = asyncio.create_task(
            server.register_tool_call(session.id, "get_weather",
                                      {"city": "Oslo"}))
        await asyncio.sleep(0)
        turn = await session.turn_future
        assert turn["type"] == "tool_calls", turn
        call = turn["calls"][0]

        # Park state, exactly as the happy path leaves it. Crucially, do
        # NOT pre-resolve the parked future -- _continue_followup needs to
        # consume it (via `call.future.set_result(...)` in the resolve
        # branch) so the resolve loop increments and we take the
        # unpark/new_turn/error path, not the rebuild path on a
        # zero-resolved branch.
        session.awaiting_followup = True
        session.holds_slot = False
        captured_session["session"] = session

        # _continue_followup awaits `session.new_turn()` to install a new
        # turn_future. In production the existing run_session task resolves
        # that future when the CLI emits its next answer; here we stub
        # await_turn to return the error dict directly, but the await on
        # new_turn itself still has to be unblocked or the test hangs.
        # Override new_turn on this session only so it returns an
        # already-resolved future -- otherwise `await session.new_turn()`
        # would hang (run_session is never started in this scenario).
        real_new_turn = type(session).new_turn

        def resolved_new_turn(self):
            f = asyncio.get_event_loop().create_future()
            f.set_result(ERROR)
            self.turn_future = f
            return f
        type(session).new_turn = resolved_new_turn

        # Stub await_turn to return an error dict directly. The handler does
        # the unpark/new_turn dance first, so this stub short-circuits only
        # the model-turn step -- exactly the failure mode the issue describes.
        real_await_turn = server.await_turn

        async def error_turn(sess, request):
            return ERROR
        server.await_turn = error_turn

        try:
            body = {"model": "m",
                    "messages": [
                        {"role": "tool", "tool_call_id": call.id,
                         "content": '{"temp_c":-3}'},
                    ]}
            try:
                await server._continue_followup(body, session.id,
                                                body["messages"], None)
            except server.HTTPException as exc:
                assert exc.status_code == 502, exc.status_code
                assert exc.detail == ERROR["detail"], exc.detail
            else:
                raise AssertionError("_continue_followup must 502 when "
                                     "the awaited turn returns an error")
            return session
        finally:
            server.await_turn = real_await_turn
            type(session).new_turn = real_new_turn
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate

    session = asyncio.run(scenario())
    assert server.cli_bridge._gate.in_flight == 0, \
        server.cli_bridge._gate.in_flight
    assert session.id not in server.SESSIONS, \
        {sid: (s.dead, s.holds_slot)
         for sid, s in server.SESSIONS.items()}
    assert session.dead, session
    assert not session.holds_slot, session
    print(f"  follow-up turn -> 502 error; gate slot released "
          f"(in_flight={server.cli_bridge._gate.in_flight}); session dead and gone")


def test_rebuild_error_does_not_leak():
    """`resume_gone_session` must tear the rebuilt session down on error too.

    The rebuild path inherits the slot (acquired by `acquire_resume_slot`)
    and hands it to `start_session`, which now owns the teardown contract
    on the rebuild path as well. A 502 from the rebuilt CLI must release
    the slot and remove the rebuilt session -- otherwise a stuck sidecar
    whose CLI fails on every rebuild accumulates one slot per attempt until
    the gate is full (issue #80 symptom, rebuild variant).

    Real `resume_gone_session` with a fresh gate that has free capacity (so
    `acquire_resume_slot` returns True and we exercise the error path,
    not the 503 path covered by `test_resume_gone_returns_503_quickly_...`).
    The stubbed `run_session` resolves with a 502 error dict; the rebuild
    in `start_session` raises HTTPException(502) out of `render_turn`,
    which the new try/except in `start_session` catches and tears down.
    """
    ERROR = {"type": "error", "status": 502,
             "detail": f"{server.PROVIDER} cli failed (1): boom"}
    captured_session: dict = {}

    async def scenario():
        saved_gate = server.cli_bridge._gate
        gate = server.cli_bridge.Gate()
        server.cli_bridge._gate = gate
        saved_sessions = dict(server.SESSIONS)
        server.SESSIONS.clear()
        real_run = server.run_session

        async def fake_run(session, argv, stdin_data=None):
            captured_session["session"] = session
            session.resolve_final(ERROR)

        server.run_session = fake_run
        try:
            # Mirrors the body shape used by test_resume_gone_returns_503_...
            # above, with a tool that does NOT fire the env probe.
            body = {
                "model": "m",
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "",
                    "parameters": {"type": "object",
                                   "properties": {"city": {"type": "string"}},
                                   "required": ["city"]}}}],
                "messages": [
                    {"role": "user", "content": "weather in Oslo?"},
                    {"role": "tool", "tool_call_id": "call_x",
                     "content": '{"temp_c":-3}'},
                ],
            }
            try:
                await server.resume_gone_session(
                    body, body["tools"], "nonexistent-session-id", None,
                    why="lost")
            except server.HTTPException as exc:
                assert exc.status_code == 502, exc.status_code
                assert exc.detail == ERROR["detail"], exc.detail
            else:
                raise AssertionError("resume_gone_session must 502 when the "
                                     "rebuilt CLI errors")
            return captured_session.get("session")
        finally:
            server.run_session = real_run
            server.SESSIONS.clear()
            server.SESSIONS.update(saved_sessions)
            server.cli_bridge._gate = saved_gate

    session = asyncio.run(scenario())
    assert server.cli_bridge._gate.in_flight == 0, \
        server.cli_bridge._gate.in_flight
    assert session is not None
    assert session.id not in server.SESSIONS, \
        {sid: (s.dead, s.holds_slot)
         for sid, s in server.SESSIONS.items()}
    assert session.dead, session
    assert not session.holds_slot, session
    print(f"  rebuild -> 502 error; gate slot released "
          f"(in_flight={server.cli_bridge._gate.in_flight}); rebuilt session dead and gone")


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


# ----------------------------------------------------------- images (issue #30) ---
# A 1x1 magenta PNG. Same one used by the cli_bridge test -- issue #30 is
# the silent drop on the text path, and the same payload shape reaches the
# MCP bridge through `messages[].content` (the tool side carries image
# blocks too; see openai_tool_content_to_mcp below).
PNG_MAGENTA = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_openai_tool_content_to_mcp_passes_through_image_data_urls():
    """An OpenAI-shaped image_url with a `data:` URL must reach the CLI as a
    real MCP image content block, not a stringified JSON blob the model
    cannot see. The decoded bytes are the model's only way to read it.
    """
    payload = [{"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64.b64encode(PNG_MAGENTA).decode()}"}}]
    blocks, is_error = server.openai_tool_content_to_mcp(payload)
    assert is_error is False, (blocks, is_error)
    assert blocks == [{"type": "image", "mimeType": "image/png",
                       "data": base64.b64encode(PNG_MAGENTA).decode()}], blocks
    print(f"  data: URL -> MCP image block ({len(blocks[0]['data'])} chars of base64)")


def test_openai_tool_content_to_mcp_converts_anthropic_shaped_images():
    """Anthropic's image block (`source.type == "base64"`) arrives on the
    MCP path too, e.g. when the caller is Anthropic-protocol. The same MCP
    shape must come out.
    """
    payload = [{"type": "image",
                "source": {"type": "base64", "media_type": "image/png",
                           "data": base64.b64encode(PNG_MAGENTA).decode()}}]
    blocks, is_error = server.openai_tool_content_to_mcp(payload)
    assert not is_error, (blocks, is_error)
    assert blocks == [{"type": "image", "mimeType": "image/png",
                       "data": base64.b64encode(PNG_MAGENTA).decode()}], blocks
    print(f"  Anthropic-shaped image -> MCP image block (mimeType=image/png)")


def test_openai_tool_content_to_mcp_marks_remote_url_as_is_error():
    """A remote http(s) URL on the tool path cannot be fetched from here.
    The call must surface as isError, with a log line warning about it --
    otherwise the caller would receive a tool result the model cannot see,
    and might hallucinate an answer from the missing image.
    """
    payload = [{"type": "image_url",
                "image_url": {"url": "https://example.com/x.png"}}]
    blocks, is_error = server.openai_tool_content_to_mcp(payload)
    assert is_error is True, (blocks, is_error)
    # The block becomes a plain text block so the CLI still has SOMETHING to
    # feed back -- an empty content list would break the stdio framing.
    assert blocks and blocks[0]["type"] == "text", blocks
    print(f"  remote URL -> isError=True text block ({blocks[0]['text'][:30]}...)")


def test_openai_tool_content_to_mcp_keeps_text_in_a_list():
    """A mixed list of text and image blocks must keep both. Otherwise the
    model's text answer would be stripped alongside the image.
    """
    payload = [{"type": "text", "text": "the colour is "},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{base64.b64encode(PNG_MAGENTA).decode()}"}}]
    blocks, is_error = server.openai_tool_content_to_mcp(payload)
    assert not is_error, (blocks, is_error)
    assert blocks[0] == {"type": "text", "text": "the colour is "}, blocks
    assert blocks[1]["type"] == "image", blocks[1]
    print(f"  text + image kept in order, image reached the CLI")


def test_flatten_with_tool_history_includes_image_markers_after_staging():
    """Rebuilding a session with an image-bearing tool result must carry the
    image into the rebuilt CLI's prompt -- not as raw base64 (which would
    blow past MAX_ARG_STRLEN) and not as nothing (the bug fix). stage_or_fail
    replaces the image blocks with `[image N: <path>]` markers BEFORE
    flatten_with_tool_history runs, so the markers survive into narration.
    """
    import shutil
    img_dir = Path(tempfile.mkdtemp(prefix="mcpb-imgtest-"))
    try:
        messages = [
            {"role": "user", "content": "what colour is this?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "look", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_a",
             "content": [{"type": "image_url", "image_url": {
                 "url": f"data:image/png;base64,{base64.b64encode(PNG_MAGENTA).decode()}"}}]},
        ]
        # Stage images into a temp dir the same way resume_gone_session does.
        paths, _ = server.cli_bridge.stage_or_fail(messages)
        assert len(paths) == 1, paths
        # flatten_with_tool_history renders the tool loop as narration. The
        # image block has been replaced with a text marker, so the marker
        # appears inside the result.
        prompt, _system = server.flatten_with_tool_history(messages)
        assert "[image 1:" in prompt, prompt
        assert "look" in prompt, prompt
        assert "[result of look:" in prompt, prompt
        print(f"  staged tool-image marker carried into narration: "
              f"{[l for l in prompt.splitlines() if 'image' in l][0][:80]}")
    finally:
        shutil.rmtree(img_dir, ignore_errors=True)


def test_mcp_build_argv_passes_images_through_claude_profile():
    """The claude MCP profile must carry image files the same way the
    text-path build_argv does: --add-dir <imgdir> + --allowed-tools
    Read(<imgdir>/**). Without these flags the staged files are
    unreachable to the CLI's harness, and the model never sees them.

    Images also lift Read out of --disallowed-tools: Claude Code concatenates
    repeated flag values, so leaving Read in the disallowed list would
    silently block the model from Reading the staged file (the bug fix in
    PR #32 review). The MCP-tools --allowed-tools allowlist survives in
    either path.
    """
    import shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-argv-"))
    tools_path = workdir / "tools.json"
    # Use a real tool list (not "[]") so the {allowed_tools} substitution has
    # something to qualify -- the regression we care about is that the MCP
    # tool qualifiers stay on the argv when images are present.
    mcp_tools = [{"name": "get_weather", "description": "",
                  "inputSchema": {"type": "object", "properties": {}}}]
    tools_path.write_text("[]")
    img_dir = workdir / "img"
    img = img_dir / "00.png"
    img_dir.mkdir(parents=True)
    img.write_bytes(PNG_MAGENTA)
    try:
        allowed = ",".join(server.PROFILE["tool_qualifier"](t["name"])
                           for t in mcp_tools)
        argv, stdin_data = server.build_argv("look at this", None, "m", workdir,
                                              "sess", tools_path, allowed, [img])
        assert "--add-dir" in argv, argv
        assert str(img_dir) in argv, argv
        # The MCP template already uses --allowed-tools for the MCP tools,
        # then a second one is appended for Read(<imgdir>/**). Match the
        # second by pattern, not by position.
        allowed_pairs = [a for a in argv
                         if a.startswith("Read(") and a.endswith("/**)")]
        assert allowed_pairs, argv
        assert allowed_pairs[0] == f"Read({img_dir}/**)", allowed_pairs[0]
        # The MCP-tools allowlist survives: the {allowed_tools} slot must
        # still be substituted into argv even when an image request also
        # appends its own Read(...) entry.
        assert any("mcp__switchyard__get_weather" in a for a in argv), argv
        # Read is OUT of --disallowed-tools in the image variant. The
        # value of the flag lives at argv[idx + 1]; split on "," to assert
        # the list itself, since the order is not contractual.
        idx = argv.index("--disallowed-tools")
        disallowed = argv[idx + 1]
        for tool in disallowed.split(","):
            assert tool != "Read", (disallowed, tool)
        assert stdin_data is None
        print(f"  mcp_bridge claude build_argv(image) -> --add-dir={img_dir}, "
              f"--allowed-tools={allowed_pairs[0]}, Read dropped from "
              f"--disallowed-tools, mcp__switchyard__get_weather still present")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_mcp_build_argv_keeps_full_disallowed_list_when_no_image():
    """Regression guard for the non-image path: Read stays in
    --disallowed-tools when the request carries no image. The fix that
    lifted Read for image sessions must not regress text-only sessions,
    which still need Read (and Bash/Edit/Write/etc.) blocked.
    """
    import shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-argv-noimg-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    try:
        argv, stdin_data = server.build_argv("just text", None, "m", workdir,
                                              "sess", tools_path, "")
        idx = argv.index("--disallowed-tools")
        disallowed = argv[idx + 1]
        # The original full list is preserved: Read, Bash, Edit, Write, Glob,
        # Grep, WebFetch, WebSearch, NotebookEdit -- nine names. The exact
        # ordering matches MCP_PROFILES["claude"]["disallowed_tools_default"];
        # asserted as a sorted set so a future re-order in the profile does
        # not silently fail this guard.
        assert sorted(disallowed.split(",")) == sorted(
            "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,NotebookEdit".split(",")
        ), disallowed
        # And the MCP-tools allowlist is still wired up.
        assert "--allowed-tools" in argv, argv
        assert stdin_data is None
        print(f"  mcp_bridge claude build_argv(text-only) -> "
              f"--disallowed-tools includes Read; original 9 names preserved")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_write_opencode_dir_enables_read_when_images_passed():
    """The opencode harness disables every tool by default. For an image
    session the model needs `read` to look at the staged files via the
    agent's own Read tool, so the disable list must be opt-out via
    `images=True`. Without it, image sessions would 200 with no answer.
    """
    import shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-op-"))
    tools_path = workdir / "tools.json"
    tools_path.write_text("[]")
    try:
        server.write_opencode_dir(workdir, "sess", tools_path, images=False)
        cfg = json.loads((workdir / "opencode.json").read_text())
        assert cfg["agent"]["switchyard"]["tools"]["read"] is False, cfg

        server.write_opencode_dir(workdir, "sess", tools_path, images=True)
        cfg = json.loads((workdir / "opencode.json").read_text())
        assert cfg["agent"]["switchyard"]["tools"]["read"] is True, cfg
        # Other tools stay disabled.
        for tool in ("bash", "edit", "write", "grep", "glob"):
            assert cfg["agent"]["switchyard"]["tools"][tool] is False, tool
        print("  write_opencode_dir(images=True) -> read enabled, others stay disabled")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ----------------------------------------- issue #44: caller-env probe + render ---
def test_fresh_with_unknown_env_emits_synthetic_probe_tool_call():
    """A fresh request whose env is unknown AND whose tools include a
    recognised command tool MUST get a synthetic `switchyard_env_` tool_call
    back, NOT a session allocation. No gate slot is taken, no SESSIONS entry
    is created -- the probe travels back to the caller for local execution.

    The probe happens BEFORE any gate acquire / workdir creation, exactly as
    the spec requires -- otherwise a caller whose tool needed permission
    would block the entire sidecar slot table for the duration of a human
    approval flow.

    We stub start_session anyway, so that any "should-have-probed-but-didn't"
    regression fails as a 502 from a real spawn rather than as a silent
    pass-through.
    """
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        body = {
            "model": "m",
            "tools": [{"type": "function", "function": {
                "name": "Bash", "description": "Run shell commands",
                "parameters": {"type": "object",
                               "properties": {"command": {"type": "string"}},
                                "required": ["command"]}}}],
            "messages": [
                {"role": "system", "content": "no env here"},
                {"role": "user", "content": "What OS am I on?"},
            ],
        }
        response = asyncio.run(server.handle_tool_request(body, body["tools"], None))
        # The probe is a chat.completion with finish_reason=tool_calls.
        # If the stub was called instead, start_session was incorrectly
        # invoked on the probe path -- a regression.
        assert "choices" in response, \
            f"expected a probe response, got {response!r}"
        assert response["choices"][0]["finish_reason"] == "tool_calls", response
        tc = response["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"].startswith(caller_env_module.PROBE_PREFIX), tc["id"]
        # The synthetic call's arguments are the probe command on the right key.
        args = json.loads(tc["function"]["arguments"])
        assert args.get("command"), args
        assert "pwd" in args["command"] and "uname" in args["command"]
        assert "$SHELL" in args["command"]
        # CRITICAL: no slot taken, no session allocated.
        assert server.cli_bridge._gate.in_flight == 0, \
            f"probe must not hold a slot: in_flight={server.cli_bridge._gate.in_flight}"
        assert server.SESSIONS == {}, f"probe must not allocate a session: {server.SESSIONS}"
        # start_session was NOT called for the probe path.
        assert captured == [], captured
        print(f"  probe emitted: {tc['id']} -> args has pwd+uname+$SHELL; "
              f"no slot taken (in_flight=0)")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_probe_follow_up_strips_synthetic_exchange_and_starts_session():
    """The follow-up request carrying the probe result must: parse the env,
    strip the synthetic assistant+tool exchange, then proceed into the
    normal fresh path with the env known. The system the CLI finally sees
    contains the `[SwitchYard tool execution environment]` block; the
    prompt's first user turn carries the one-line reminder.
    """
    saved_gate_count = server.cli_bridge._gate.in_flight
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        messages = [
            {"role": "system", "content": "no env"},
            {"role": "user", "content": "What OS am I on?"},
        ]
        # First request: env unknown -> synthetic probe emitted.
        first = asyncio.run(server.handle_tool_request(
            {"model": "m", "tools": [{
                "type": "function", "function": {
                    "name": "Bash", "description": "",
                    "parameters": {"type": "object",
                                   "properties": {"command": {"type": "string"}},
                                   "required": ["command"]}}}],
             "messages": messages}, [{
                "type": "function", "function": {
                    "name": "Bash", "description": "",
                    "parameters": {"type": "object",
                                   "properties": {"command": {"type": "string"}},
                                   "required": ["command"]}}}], None))
        probe_id = first["choices"][0]["message"]["tool_calls"][0]["id"]

        # Second request: caller executed the probe, replies with a tool
        # message whose tool_call_id is the probe id and content is the
        # three labelled lines from PROBE_COMMAND. The synthetic assistant
        # message that minted the probe MUST be stripped out before the
        # CLI sees it.
        followup_messages = messages + [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": probe_id, "type": "function",
                 "function": {"name": "Bash",
                              "arguments": json.dumps({"command": "pwd"})}}]},
            {"role": "tool", "tool_call_id": probe_id, "content": (
                "cwd=/home/u/dir\n"
                "platform=Linux\n"
                "shell=/bin/zsh\n")},
            {"role": "user", "content": "ok thanks"},
        ]
        body2 = {"model": "m",
                 "tools": [{"type": "function", "function": {
                     "name": "Bash", "description": "",
                     "parameters": {"type": "object",
                                    "properties": {"command": {"type": "string"}},
                                    "required": ["command"]}}}],
                 "messages": followup_messages}
        asyncio.run(server.handle_tool_request(body2, body2["tools"], None))

        assert len(captured) == 1, captured
        seen = captured[0]
        # The CLI's system prompt has the env block.
        assert "[SwitchYard tool execution environment]" in seen["system"], \
            seen["system"]
        assert "Caller platform: Linux" in seen["system"], seen["system"]
        assert "Caller working directory: /home/u/dir" in seen["system"], seen["system"]
        assert "Caller shell: /bin/zsh" in seen["system"], seen["system"]
        # The synthetic assistant+tool exchange is gone from what the CLI sees:
        # the rendered prompt has the first real user turn + reminder, but no
        # "[called Bash" narration of the probe and no Human: cwd=... block.
        assert "cwd=/home/u/dir" not in seen["prompt"], \
            "probe-result text must not leak into the rebuilt prompt"
        assert "[SwitchYard: tools execute on Linux in /home/u/dir" in seen["prompt"], \
            seen["prompt"]
        print(f"  follow-up: env parsed, synthetic exchange stripped, system has env block")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_probe_iserror_marks_failed_and_no_retry_on_second_attempt():
    """An isError / unparseable probe result marks the fingerprint failed
    and the next attempt with the same fingerprint MUST NOT re-emit the
    probe. Falling back to the unknown-env path is the only allowed
    behaviour; no retry loop, no escalation, just the env-block with the
    'unknown' wording."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        messages = [
            {"role": "system", "content": "no env"},
            {"role": "user", "content": "What OS am I on?"},
        ]
        # 1. First call: probe emitted.
        first = asyncio.run(server.handle_tool_request(
            {"model": "m", "tools": tools, "messages": messages}, tools, None))
        probe_id = first["choices"][0]["message"]["tool_calls"][0]["id"]

        # 2. Second call: caller REFUSED the probe (isError=true).
        followup_messages = messages + [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": probe_id, "type": "function",
                 "function": {"name": "Bash",
                              "arguments": json.dumps({"command": "pwd"})}}]},
            {"role": "tool", "tool_call_id": probe_id, "is_error": True,
             "content": "permission denied"},
            {"role": "user", "content": "ok thanks anyway"},
        ]
        body2 = {"model": "m", "tools": tools, "messages": followup_messages}
        asyncio.run(server.handle_tool_request(body2, tools, None))

        # Now the third call with the SAME fingerprint MUST NOT probe again.
        # Reset the message prefix back to the first two turns to keep the
        # fingerprint stable.
        messages3 = list(messages) + [
            {"role": "user", "content": "next question"},
        ]
        body3 = {"model": "m", "tools": tools, "messages": messages3}
        third = asyncio.run(server.handle_tool_request(body3, tools, None))

        # The fingerprint must be in FAILED_PROBES now.
        assert len(server.FAILED_PROBES) >= 1, dict(server.FAILED_PROBES)
        # The third call's response is NOT a synthetic probe -- it is the
        # stubbed start_session result (a normal session with the unknown
        # env block in its system).
        if "choices" in third:
            # Got a real completion; it must NOT be a probe.
            tcs = third["choices"][0]["message"].get("tool_calls") or []
            assert not any((tc.get("id") or "").startswith(caller_env_module.PROBE_PREFIX)
                           for tc in tcs), tcs
        else:
            # Stubbed {"stubbed": True} response from start_session.
            assert third == {"stubbed": True}, third
        # The system the CLI sees has the unknown-env wording.
        if captured:
            seen = captured[-1]
            assert "Caller platform: unknown" in seen["system"], seen["system"]
            assert "Caller working directory: unknown" in seen["system"], seen["system"]
        print(f"  isError probe -> FAILED_PROBES populated, "
              f"no retry on second attempt, env rendered as 'unknown'")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_passive_env_skips_probe_and_renders_request_values():
    """A request that already carries its own environment (an OpenCode-
    shaped system block, say) MUST be parsed passively and the probe
    skipped entirely. The system the CLI sees has the resolved values."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools, "messages": [
            {"role": "system", "content": (
                "<environment>\n"
                "  <working_directory>C:\\Users\\demo\\p</working_directory>\n"
                "  <platform>windows</platform>\n"
                "</environment>"
            )},
            {"role": "user", "content": "hi"},
        ]}
        response = asyncio.run(server.handle_tool_request(body, tools, None))
        # It went straight to the stubbed start_session (no probe emitted).
        assert response == {"stubbed": True}, response
        # No probe was emitted; FAILED_PROBES / RESOLVED_PROBES empty.
        assert server.FAILED_PROBES == {}, dict(server.FAILED_PROBES)
        assert server.RESOLVED_PROBES == {}, dict(server.RESOLVED_PROBES)
        # The system the CLI sees has the request-derived env.
        seen = captured[-1]
        assert "[SwitchYard tool execution environment]" in seen["system"], seen["system"]
        assert r"Caller working directory: C:\Users\demo\p" in seen["system"]
        assert "Caller platform: windows" in seen["system"]
        # Reminder on the first turn.
        assert "[SwitchYard: tools execute on windows in C:\\Users\\demo\\p" in seen["prompt"]
        print("  request-carries-env: passive parse, no probe, env block rendered")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_metadata_caller_env_is_honored_when_present():
    """metadata.switchyard.caller_env stamped on the request body MUST be
    honoured as the resolved env (source=request). The stamp is the easy
    path; the passive parsers are the belt-and-braces backup."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools,
                "metadata": {"switchyard": {"caller_env": {
                    "platform": "macos", "cwd": "/Users/x/p",
                    "shell": "zsh", "source": "request"}}},
                "messages": [
                    {"role": "user", "content": "hi"},
                ]}
        response = asyncio.run(server.handle_tool_request(body, tools, None))
        assert response == {"stubbed": True}, response
        seen = captured[-1]
        assert "Caller platform: macos" in seen["system"]
        assert "Caller working directory: /Users/x/p" in seen["system"]
        assert "Caller shell: zsh" in seen["system"]
        print("  metadata-stamped caller_env honored -> system reflects macos values")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_run_session_subprocess_cwd_is_session_workdir():
    """run_session must spawn the inner CLI with cwd == session.workdir,
    NOT /app/mcp_bridge. The fake CLI prints its own cwd, so we can
    assert the spawned process actually saw the per-session directory.
    """
    import shutil as _shutil
    workdir = Path(tempfile.mkdtemp(prefix="mcpb-cwdsess-"))
    session = server.Session(id=uuid.uuid4().hex, provider="claude", model="m",
                              workdir=str(workdir))
    server.SESSIONS[session.id] = session
    fake_cli = _write_fake_cli(
        "import json, os\n"
        "print(json.dumps({'result': os.getcwd(), 'usage': "
        "{'input_tokens': 1, 'output_tokens': 0}}))\n")
    try:
        async def scenario():
            session.new_turn()
            await server.run_session(session, [sys.executable, fake_cli])
            return session.turn_future.result()

        result = asyncio.run(scenario())
        assert result["type"] == "final", result
        assert result["payload"]["result"] == str(workdir), (
            f"expected inner CLI cwd == session.workdir {workdir!r}, "
            f"got {result['payload']['result']!r}"
        )
        print(f"  inner CLI spawned with cwd={result['payload']['result']}")
    finally:
        server.SESSIONS.pop(session.id, None)
        try:
            os.unlink(fake_cli)
        except OSError:
            pass
        try:
            os.unlink(os.path.dirname(fake_cli))
        except OSError:
            pass
        _shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------- PR #66 review fixes ----------------
def _set_caller_environment_probe(probe: str):
    """Replace cli_bridge._config with one whose caller_environment.probe
    is `probe`. The probe mode determines how the mcp_bridge handles
    unresolvable envs (auto=fallback unknown, required=503, disabled=skip
    probe entirely). Returns a restore callable."""
    from dataclasses import replace as _replace
    saved = server.cli_bridge._config
    ce = server.cli_bridge.config().caller_environment
    new_ce = _replace(ce, probe=probe) if ce is not None else None
    server.cli_bridge._config = _replace(saved, caller_environment=new_ce)
    def restore():
        server.cli_bridge._config = saved
    return restore


def test_probe_required_unresolvable_returns_400_no_unknown_wording():
    """PR #66 blocker (round 2 update): probe=required + an unresolvable
    env must refuse the request with HTTP 400 (type=caller_environment_required)
    and NO Retry-After header, NOT silently fall through to the unknown-env
    wording. The round-1 status was 503 + Retry-After:5; round 2 changed
    it to 400 because the failure is not transient -- nothing about the
    request, the caller, or the operator's plans.yaml can change in 5
    seconds, so a Retry-After actively misleads the caller. The file's
    503 + Retry-After convention is reserved for transient-capacity
    conditions.

    The scenario: a caller has tools (so a probe is minted), but refuses
    the probe on the second turn (is_error=true). The second turn has no
    passive env in the request body and the probe was refused -- the env
    cannot be resolved. With probe=required, handle_fresh must raise
    400 instead of falling through to `unknown`."""
    from fastapi import HTTPException
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    restore_cfg = _set_caller_environment_probe("required")
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        messages = [
            {"role": "system", "content": "no env"},
            {"role": "user", "content": "What OS am I on?"},
        ]
        # 1. First request: env unknown, probe minted.
        first = asyncio.run(server.handle_tool_request(
            {"model": "m", "tools": tools, "messages": messages}, tools, None))
        probe_id = first["choices"][0]["message"]["tool_calls"][0]["id"]

        # 2. Second request: caller REFUSED the probe (is_error=true).
        # With probe=required, the env is now unresolvable and the
        # sidecar MUST refuse -- not silently fall through to the
        # unknown-env wording.
        refusal = messages + [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": probe_id, "type": "function",
                 "function": {"name": "Bash",
                              "arguments": json.dumps({"command": "pwd"})}}]},
            {"role": "tool", "tool_call_id": probe_id, "is_error": True,
             "content": "permission denied"},
            {"role": "user", "content": "ok thanks anyway"},
        ]
        try:
            response = asyncio.run(server.handle_tool_request(
                {"model": "m", "tools": tools, "messages": refusal},
                tools, None))
        except HTTPException as exc:
            # 400 (Bad Request) -- not 503. The failure is a config /
            # request mismatch, not transient capacity.
            assert exc.status_code == 400, \
                f"probe=required must 400, got {exc.status_code}: {exc.detail}"
            assert isinstance(exc.detail, dict), exc.detail
            assert exc.detail.get("error", {}).get("type") == \
                "caller_environment_required", exc.detail
            # NO Retry-After: the failure is not transient. A 5-second
            # Retry-After would actively mislead the caller -- the env
            # will still be unresolvable on retry (the operator's
            # plans.yaml still has no fallback_platform, the body still
            # has no passive env, the CLI still refuses the probe).
            assert not (exc.headers and "retry-after" in {k.lower() for k in exc.headers}), \
                f"probe=required must NOT carry Retry-After (failure is not " \
                f"transient), got headers={exc.headers!r}"
            print(f"  probe=required + refused probe -> HTTP 400 "
                  f"type=caller_environment_required, no Retry-After")
            return
        raise AssertionError(
            f"probe=required must raise 400, but handle_tool_request returned "
            f"{response!r} (silent fallback to unknown is the regression)")
    finally:
        restore_cfg()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_metadata_source_config_is_relabeled_to_request():
    """PR #66 should-fix: a metadata stamp claiming source=config must
    be re-labeled to source=request. The values still flow through (we
    do not want to lose data), but at the request tier of the
    precedence chain, not the config tier. The invariant: the ONLY
    path that produces source=config is the operator's plans.yaml
    settings; anything arriving over the wire cannot.

    A caller that stamps `source=config` to bypass the precedence chain
    -- say, to force the sidecar to honor a fake Linux /app env as the
    caller's environment, exactly the bug #44 is meant to prevent --
    gets the request tier instead. The relay-env-as-caller-env
    invariant stays intact."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools,
                "metadata": {"switchyard": {"caller_env": {
                    # The classic bug attempt: claim config-tier so
                    # the precedence chain skips the passive parse
                    # AND uses these values as operator-forced.
                    "platform": "linux", "cwd": "/app/mcp_bridge",
                    "shell": "/bin/sh", "source": "config"}}},
                "messages": [
                    {"role": "system", "content": (
                        "<environment>\n"
                        "  <working_directory>C:\\Users\\demo</working_directory>\n"
                        "  <platform>windows</platform>\n"
                        "</environment>"
                    )},
                    {"role": "user", "content": "hi"},
                ]}
        response = asyncio.run(server.handle_tool_request(body, tools, None))
        # The CLI sees the env block; the values are honored (we don't
        # drop data) but at the REQUEST tier, not the config tier.
        seen = captured[-1]
        env = seen["env"]
        assert env.source == "request", \
            f"metadata-stamped source=config must be re-labeled to request, " \
            f"got {env.source!r}"
        assert env.platform == "linux"
        assert env.cwd == "/app/mcp_bridge"
        # The metadata-stamped values WON, not the passive parse's
        # windows values, because metadata is consulted first and both
        # are at the request tier. Either is consistent with the
        # invariant; the point is the source label.
        print(f"  metadata claims source=config -> re-labeled to "
              f"{env.source!r}, values still flow through")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_metadata_source_host_is_relabeled_to_request():
    """Round-2 should-fix: `source=host` is also plans.yaml-derived
    (the `fallback_platform` branch of `resolve()` produces it), so a
    caller who stamps `source=host` to bypass the precedence chain
    must get the request tier instead. Same defensive re-labeling as
    `source=config`."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools,
                "metadata": {"switchyard": {"caller_env": {
                    "platform": "macos", "cwd": "/Users/x",
                    "shell": "zsh", "source": "host"}}},
                "messages": [{"role": "user", "content": "hi"}]}
        asyncio.run(server.handle_tool_request(body, tools, None))
        seen = captured[-1]
        env = seen["env"]
        assert env.source == "request", \
            f"metadata-stamped source=host must be re-labeled to request, " \
            f"got {env.source!r}"
        assert env.platform == "macos"
        assert env.cwd == "/Users/x"
        print(f"  metadata claims source=host -> re-labeled to "
              f"{env.source!r}, values still flow through")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_metadata_stamp_without_source_falls_through_to_passive_parse():
    """Round-2 should-fix: a wire stamp without an explicit source
    marker must NOT preempt passive detection. The round-1 helper
    returned a CallerEnvironment for any stamp with at least one field
    (even without a source), so a stamp like `{"platform": "linux"}`
    would win over a passive-parseable body. Restore the pre-round-1
    strictness: only honored sources yield a CallerEnvironment;
    everything else falls through to `parse_request(body)`.

    The end-to-end check: stamp values without source + passive-
    parseable system prompt -> the passive parse wins, not the
    stamp values."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools,
                # Stamp WITHOUT a source field, with values that would
                # win if from_wire_metadata were permissive. The body
                # ALSO carries a passive OpenCode-shaped env block.
                "metadata": {"switchyard": {"caller_env": {
                    "platform": "linux", "cwd": "/app/mcp_bridge",
                    "shell": "/bin/sh"}}},
                "messages": [
                    {"role": "system", "content": (
                        "<environment>\n"
                        "  <working_directory>C:\\Users\\demo\\p</working_directory>\n"
                        "  <platform>windows</platform>\n"
                        "</environment>"
                    )},
                    {"role": "user", "content": "hi"},
                ]}
        asyncio.run(server.handle_tool_request(body, tools, None))
        seen = captured[-1]
        env = seen["env"]
        # The stamp's `linux / /app/mcp_bridge` did NOT win. The
        # passive parse's `windows / C:\Users\demo\p` did.
        assert env.platform == "windows", \
            f"stamp-without-source must fall through to passive parse, " \
            f"got platform={env.platform!r} (stamp won)"
        assert env.cwd == "C:\\Users\\demo\\p", \
            f"stamp-without-source must fall through to passive parse, " \
            f"got cwd={env.cwd!r} (stamp won)"
        assert env.source == "request", env.source
        # Critically: the relay's /app/mcp_bridge is NOT in the
        # rendered env -- the bug we're avoiding.
        assert env.cwd != "/app/mcp_bridge", env.cwd
        print(f"  stamp without source -> passive parse wins "
              f"(platform={env.platform!r}, cwd={env.cwd!r})")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


def test_concurrent_same_fingerprint_mints_only_one_probe():
    """PR #66 should-fix: _maybe_probe's check+mint pair is atomic under
    asyncio.gather. Without the lock, two (or three) concurrent calls
    with the same fingerprint would all pass the `fp not in caches`
    check and all mint duplicate probes -- one would carry through and
    the others would be silently dropped at consume-time, polluting
    the caller's history with extra synthetic tool_calls and wasting
    work. With PROBE_LOCK + the _PROBE_PENDING sentinel published
    before releasing the lock, exactly one mints and the others see
    fp already-taken and fall through to handle_fresh.

    Stubbed start_session so the handle_fresh path does not actually
    spawn a CLI; the test asserts the count distribution: 1 probe
    response + (N-1) stubbed sessions."""
    saved_sessions = dict(server.SESSIONS)
    saved_failed = dict(server.FAILED_PROBES)
    saved_resolved = dict(server.RESOLVED_PROBES)
    server.SESSIONS.clear()
    server.FAILED_PROBES.clear()
    server.RESOLVED_PROBES.clear()
    captured = []
    restore = _stub_start_session(captured)
    try:
        tools = [{"type": "function", "function": {
            "name": "Bash", "description": "",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        body = {"model": "m", "tools": tools, "messages": [
            {"role": "system", "content": "no env"},
            {"role": "user", "content": "What OS am I on?"},
        ]}

        async def scenario():
            return await asyncio.gather(
                server.handle_tool_request(dict(body), tools, None),
                server.handle_tool_request(dict(body), tools, None),
                server.handle_tool_request(dict(body), tools, None),
            )

        responses = asyncio.run(scenario())
        # Exactly one probe response was emitted.
        probes = [r for r in responses
                  if "choices" in r
                  and r["choices"][0].get("finish_reason") == "tool_calls"
                  and any((tc.get("id") or "").startswith(
                          caller_env_module.PROBE_PREFIX)
                          for tc in (r["choices"][0]["message"]
                                      .get("tool_calls") or []))]
        assert len(probes) == 1, \
            f"asyncio.gather(3x same-fingerprint) must mint exactly 1 probe, " \
            f"got {len(probes)} (duplicate probes is the regression)"
        # The other two went through start_session (stubbed).
        stubbed = [r for r in responses if r == {"stubbed": True}]
        assert len(stubbed) == 2, \
            f"expected 2 stubbed sessions, got {len(stubbed)}"
        # The PENDING sentinel is in RESOLVED_PROBES; only the probe
        # result can overwrite it with a real env (or FAILED_PROBES).
        assert len(server.RESOLVED_PROBES) == 1, dict(server.RESOLVED_PROBES)
        assert next(iter(server.RESOLVED_PROBES.values())) is \
            server._PROBE_PENDING, dict(server.RESOLVED_PROBES)
        print(f"  asyncio.gather(3x same-fingerprint) -> 1 probe + 2 stubbed; "
              f"PENDING sentinel in RESOLVED_PROBES")
    finally:
        restore()
        server.SESSIONS.clear()
        server.SESSIONS.update(saved_sessions)
        server.FAILED_PROBES.clear()
        server.FAILED_PROBES.update(saved_failed)
        server.RESOLVED_PROBES.clear()
        server.RESOLVED_PROBES.update(saved_resolved)


# Module-level alias for tests that want the probe prefix string.
caller_env_module = sys.modules.get("switchyard.caller_env")
if caller_env_module is None:
    import importlib.util as _il
    _path = os.path.join(os.path.dirname(HERE), "switchyard", "caller_env.py")
    _spec = _il.spec_from_file_location("switchyard.caller_env", _path)
    caller_env_module = _il.module_from_spec(_spec)
    caller_env_module.__package__ = "switchyard"
    sys.modules.setdefault("switchyard", type(sys)("switchyard"))
    sys.modules["switchyard.caller_env"] = caller_env_module
    _spec.loader.exec_module(caller_env_module)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} mcp-bridge tests passed")


# ------------------------------------------------------------- context usage ---
def _write_inner_transcript(session: "server.Session", root: Path, calls: list[dict]) -> None:
    """Lay down what the inner claude CLI writes: one assistant entry per model call."""
    import re
    folder = root / re.sub(r"[^A-Za-z0-9]", "-", str(session.workdir))
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{uuid.uuid4()}.jsonl", "w", encoding="utf-8") as fh:
        for usage in calls:
            fh.write(json.dumps({"type": "user", "message": {"content": "x"}}) + "\n")
            fh.write(json.dumps({"type": "assistant",
                                 "message": {"content": [], "usage": usage}}) + "\n")


def test_final_turn_reports_the_last_calls_size_not_the_runs_sum():
    """A tool loop is ONE inner CLI run, so its final usage is the sum of every
    model call in it. Claude Code reads a reply's usage as its context size and
    auto-compacts past the window, so reporting the sum compacted the caller's
    session after every tool-using answer. The caller must see the last call's
    size; the ledger must still book what the run spent."""
    session = _new_session()
    root = Path(tempfile.mkdtemp(prefix="projects-"))
    calls = [{"input_tokens": 2, "cache_read_input_tokens": 100_000 + i * 1000,
              "cache_creation_input_tokens": 500, "output_tokens": 50}
             for i in range(20)]
    _write_inner_transcript(session, root, calls)
    summed = {k: sum(c[k] for c in calls) for k in calls[0]}
    old_root = server.cli_bridge.CLAUDE_PROJECTS
    server.cli_bridge.CLAUDE_PROJECTS = root
    try:
        response = server.render_turn(
            session, {"type": "final", "payload": {"result": "done", "usage": summed}}, None)
    finally:
        server.cli_bridge.CLAUDE_PROJECTS = old_root
        _drop(session)
    usage = response["usage"]
    assert usage["prompt_tokens"] == 2 + 119_000 + 500, usage
    assert usage["switchyard_billed_prompt_tokens"] == (
        summed["input_tokens"] + summed["cache_read_input_tokens"]
        + summed["cache_creation_input_tokens"]), usage
    assert usage["switchyard_billed_completion_tokens"] == summed["output_tokens"]


def test_no_inner_transcript_keeps_the_old_usage():
    session = _new_session()
    old_root = server.cli_bridge.CLAUDE_PROJECTS
    server.cli_bridge.CLAUDE_PROJECTS = Path(tempfile.mkdtemp(prefix="projects-empty-"))
    try:
        response = server.render_turn(
            session, {"type": "final", "payload": {"result": "done", "usage": {
                "input_tokens": 7, "output_tokens": 3}}}, None)
    finally:
        server.cli_bridge.CLAUDE_PROJECTS = old_root
        _drop(session)
    assert response["usage"]["prompt_tokens"] == 7
    assert response["usage"]["switchyard_billed_prompt_tokens"] == 7
