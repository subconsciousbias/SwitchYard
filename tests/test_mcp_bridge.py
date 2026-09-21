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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MCP_BRIDGE_DIR = os.path.join(ROOT, "sidecars", "mcp_bridge")
sys.path.insert(0, MCP_BRIDGE_DIR)
sys.path.insert(0, HERE)

os.environ["PROVIDER"] = "claude"
os.environ.setdefault("SWITCHYARD_PLAN", "claude-max")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans_path import plans_path  # noqa: E402

os.environ.setdefault("SWITCHYARD_PLANS", plans_path())
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


def test_followup_with_unknown_session_is_rejected():
    body = {"messages": [{"role": "tool", "tool_call_id": "call_" + "0" * 32 + "_1",
                          "content": "x"}]}
    try:
        asyncio.run(server.handle_followup(body, body["messages"]))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 410, exc
        print(f"  unknown/expired session -> {exc.status_code}")
        return
    raise AssertionError("expected a 410 for a session that is not live")


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


def test_a_parked_session_is_preempted_rather_than_refusing_a_new_request():
    """A stale parked session must yield its slot, and a fresh one must not.

    Observed live: two lane probes each left a session parked awaiting a
    follow-up that never came, the gate stayed full, and the next request got
    "sidecar at capacity (2)" — which SwitchYard read as concurrency pressure
    and used to cool a perfectly healthy plan. Nothing frees those slots on its
    own, because through a proxy the caller's disconnect is invisible.
    """
    async def scenario():
        old_grace = server.PARKED_GRACE
        server.PARKED_GRACE = 60.0
        try:
            fresh = _new_session()
            fresh.awaiting_followup = True          # parked just now
            assert not await server.reclaim_parked_slot(), \
                "a session parked inside the grace period must be left alone"

            stale = _new_session()
            stale.awaiting_followup = True
            stale.last_active = time.time() - server.PARKED_GRACE - 5
            stalest = _new_session()
            stalest.awaiting_followup = True
            stalest.last_active = time.time() - server.PARKED_GRACE - 500

            assert await server.reclaim_parked_slot()
            # The stalest goes first, and only one goes per call.
            assert stalest.id not in server.SESSIONS
            assert stale.id in server.SESSIONS
            assert fresh.id in server.SESSIONS

            # A session mid-turn (not parked) is never a candidate.
            busy = _new_session()
            for victim in (stale, fresh):
                await server.reap_session(victim)
            assert not await server.reclaim_parked_slot()
            await server.reap_session(busy)
        finally:
            server.PARKED_GRACE = old_grace

    asyncio.run(scenario())
    print("  stalest parked session preempted; fresh and mid-turn ones protected")


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

        assert await server.reclaim_parked_slot()
        assert session.id in server.PREEMPTED
        with contextlib.suppress(Exception):
            await parked

        # An id from a session nobody preempted still gets the honest 410.
        unknown = f"call_{'0' * 32}_1"
        body = {"model": "m", "tools": [], "messages": [
            {"role": "tool", "tool_call_id": unknown, "content": "x"}]}
        try:
            await server.handle_followup(body, body["messages"])
        except Exception as exc:
            assert getattr(exc, "status_code", None) == 410, exc
        else:
            raise AssertionError("an unknown session must not be resumed")

        return call_id

    call_id = asyncio.run(scenario())
    assert call_id and server.session_id_from_call_id(call_id), call_id
    print("  preempted id remembered and resumable; unknown id still 410s")


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


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} mcp-bridge tests passed")
