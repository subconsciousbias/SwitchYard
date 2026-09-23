"""Regression tests for SwitchyardHandler.async_pre_call_hook's two new
side effects on the first turn of a CLI-backed session:

  - the per-CLI tool blocklist filters native tools out of `data["tools"]`
    so they do not silently land in the sidecar's container;
  - the per-CLI first-turn system note ("you are running through
    SwitchYard, your native tools are unavailable") is injected after the
    caller's leading system block, exactly once per session.

The session identity check (`identify`) and the per-session idempotency
marker (`sy:inject:{session}` in the slot table) are the load-bearing
pieces. Drop_lease on the slot table clears the marker so a hard plan
rejection that re-leases the session re-enables injection on the next turn.

Same wiring pattern as `tests/test_verdict.py` — registry + slots + ledger +
policy + picker, FakeRedis for everything Redis-shaped, no Picker needed
for the verdict tests but required here because async_pre_call_hook goes
through `picker.pick`. The `__new__` bypass skips __init__, so any
attribute the hook reads has to be wired by hand; `_beats` is the one the
verdict suite did not need.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()


from switchyard import models                          # noqa: E402
from switchyard.hooks import SwitchyardHandler, INJECTION_NOTE  # noqa: E402
from switchyard.picker import Picker                   # noqa: E402
from switchyard.policy import CapacityPolicy           # noqa: E402
from switchyard.session import (                       # noqa: E402
    CLI_HEADER, CLI_META_FIELD, derive as derive_session,
)
from switchyard.slots import SlotTable, K_INJECTED     # noqa: E402
from switchyard.usage import Ledger                    # noqa: E402
from tests.fake_redis import FakeRedis                 # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _build():
    """Wire the bits async_pre_call_hook reaches, with a fresh FakeRedis.

    Mirror of test_verdict._build, with two additions: a real Picker (the hook
    calls picker.pick before the injection block, so the test needs one), and
    `_beats={}` (the constructor is bypassed, but _start_heartbeat — which the
    hook always calls — writes to self._beats, and an AttributeError there
    would surface as a confusing crash inside the block we are trying to test).
    """
    reg = models.load()
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    ledger = Ledger(redis)
    policy = CapacityPolicy(redis, reg.settings, ledger)
    picker = Picker(reg, slots, policy)

    h = SwitchyardHandler.__new__(SwitchyardHandler)
    h.__dict__["registry"] = reg
    h.__dict__["_slots"] = slots
    h.__dict__["_ledger"] = ledger
    h.__dict__["_policy"] = policy
    h.__dict__["_redis"] = redis
    h.__dict__["_picker"] = picker
    # _start_heartbeat runs unconditionally; without this dict the hook
    # raises AttributeError on the very first request — masking the test's
    # real assertion.
    h.__dict__["_beats"] = {}
    return h, reg, redis


def _request(session=None, cli=None, messages=None):
    """Build a request dict shaped the way the LiteLLM proxy hands it to us.

    `proxy_server_request.headers` is where the operator's CLI harness sets
    the two governance headers (`x-switchyard-session`, `x-switchyard-cli`).
    `messages=None` is also accepted: the spec covers that explicitly.
    """
    data: dict = {"model": "forge"}
    if messages is not None:
        data["messages"] = messages
    headers: dict = {}
    if session is not None:
        headers["x-switchyard-session"] = session
    if cli is not None:
        headers[CLI_HEADER] = cli
    if headers:
        data["proxy_server_request"] = {"headers": headers}
    return data


def _system_count(messages):
    """How many leading (contiguous) system messages are at the head of the
    conversation. The injection block inserts the new system note right
    after this prefix, so any assertion about *which* index gets the note
    needs this number up front."""
    n = 0
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            n += 1
        else:
            break
    return n


async def _release(h, data):
    """Release the slot the hook claimed for this request.

    Without this, a second request that targets the same plan/model would
    see the slot still held and pick differently, so the second-turn test
    would not actually exercise the same leased plan. The pick context
    sits at data["metadata"]["switchyard"], which the hook itself writes.
    """
    ctx = data["metadata"]["switchyard"]
    await h.picker.release(ctx["plan"], ctx["request_id"], ctx["model"])


def test_first_turn_after_pickup_gets_the_system_message():
    """cli=opencode + session header: exactly one system message containing
    "SwitchYard" sits in the leading system block, and the sy:inject:…
    marker key for that session is present in Redis.

    The injection is conditional on cli ∈ INJECT_CLIS — opencode is one of
    them — and on the marker being absent, which it is on the first call.
    """
    async def go():
        h, reg, redis = _build()
        session = "sess-inj-1"
        before = [
            {"role": "system", "content": "You are Claude Code."},
            {"role": "user", "content": "list the repo"},
        ]
        data = _request(session=session, cli="opencode", messages=list(before))

        await h.async_pre_call_hook(None, None, data, "acompletion")
        await _release(h, data)

        # The session id the hook actually stored carries derive_session's
        # "h:" prefix — derive_session's own return value is the source of
        # truth, not the raw header string.
        actual = derive_session(data, None)
        marker = redis.strings.get(K_INJECTED.format(session=actual))
        return data["messages"], marker, _system_count(before)

    messages, marker, sys_prefix = _run(go())
    # The marker key landed in Redis with the session in it.
    assert marker is not None, f"expected {K_INJECTED!r} to be set"
    assert marker[0] == "1", f"marker value should be '1', got {marker[0]!r}"
    # The note was inserted right after the caller's one leading system
    # message — the test's leading system is at index 0, the note at 1.
    assert messages[sys_prefix]["role"] == "system"
    assert "SwitchYard" in messages[sys_prefix]["content"], messages[sys_prefix]
    assert messages[sys_prefix]["content"] == INJECTION_NOTE
    # And nothing else changed shape: no extra system message, no duplicated
    # note, the original first user turn is still in place.
    assert messages[sys_prefix + 1] == {"role": "user", "content": "list the repo"}
    system_count = sum(1 for m in messages if m.get("role") == "system")
    assert system_count == 2, system_count
    print(f"  opencode first turn: 1 -> 2 system messages, "
          f"note at index {sys_prefix}, marker key set")


def test_second_turn_same_session_does_not_re_inject():
    """Two fresh request dicts, same session id: the injected note appears
    exactly once across both turns.

    The marker is the gate — written on the first call, read on the second,
    so the second turn is a no-op for the insertion block. Without the
    marker, both turns would prepend the note, and a 60-token instruction
    would appear twice in the resumed session's message log.
    """
    async def go():
        h, reg, redis = _build()
        session = "sess-inj-2"
        first_msgs = [{"role": "user", "content": "first turn"}]
        second_msgs = [{"role": "user", "content": "second turn"}]

        d1 = _request(session=session, cli="claude-code", messages=list(first_msgs))
        await h.async_pre_call_hook(None, None, d1, "acompletion")
        await _release(h, d1)

        d2 = _request(session=session, cli="claude-code", messages=list(second_msgs))
        await h.async_pre_call_hook(None, None, d2, "acompletion")
        await _release(h, d2)

        return d1["messages"], d2["messages"]

    first, second = _run(go())
    # First turn: leading user message got a system note prepended.
    assert len(first) == 2, first
    assert first[0]["role"] == "system"
    assert "SwitchYard" in first[0]["content"]
    assert first[1] == {"role": "user", "content": "first turn"}
    # Second turn: the original user message is the only thing in the
    # message list — no duplicate system note was appended.
    assert len(second) == 1, second
    assert second[0] == {"role": "user", "content": "second turn"}
    # And nothing about the second turn snuck through the marker: no
    # additional system role anywhere in second.
    assert all(m.get("role") != "system" for m in second)
    print(f"  same-session second turn: no re-injection "
          f"(first len={len(first)}, second len={len(second)})")


def test_non_proxied_cli_gets_no_message():
    """No CLI header, or an unknown CLI: messages are untouched and no
    marker key is written. Unknown CLIs are the silent majority — most
    real-world callers don't set the header at all — so this is the
    default no-op path."""
    async def go():
        h, reg, redis = _build()
        session = "sess-inj-3"
        # Unknown CLI name: the blocklist lookup falls through silently
        # (cli_tool_block.get("openai-direct", ()) is ()) and the
        # INJECT_CLIS gate also misses.
        before = [{"role": "user", "content": "hi"}]
        data = _request(session=session, cli="openai-direct",
                        messages=list(before))
        await h.async_pre_call_hook(None, None, data, "acompletion")
        await _release(h, data)

        actual = derive_session(data, None)
        marker = redis.strings.get(K_INJECTED.format(session=actual))
        return data["messages"], marker

    messages, marker = _run(go())
    assert messages == [{"role": "user", "content": "hi"}], messages
    assert marker is None, (
        f"unknown cli must NOT write a {K_INJECTED!r} key, got {marker!r}")
    print("  cli=openai-direct: messages untouched, no marker key")


def test_no_session_identity_no_injection():
    """Request with no `messages` key, no headers, no metadata: derive_session
    returns None (no header, no fingerprint without messages, no metadata,
    no trace id), and the injection block short-circuits on the
    `if cli and session and ...` guard.

    The 'message-less request' path the spec calls out: a tool-only call,
    or a malformed request — neither should ever get the note appended.
    """
    async def go():
        h, reg, redis = _build()
        data = _request()  # no session, no cli, no messages
        await h.async_pre_call_hook(None, None, data, "acompletion")
        # Release whatever the hook claimed (it picks on "forge" regardless
        # of session/cli/messages presence).
        await _release(h, data)

        # No messages key, nothing got injected, nothing was stashed under
        # it. No marker either (the block gates on session not None).
        return "messages" in data, list(redis.strings.keys())

    has_messages_key, marker_keys = _run(go())
    assert has_messages_key is False, (
        "the spec says this request has no `messages` key — it must stay gone")
    # No session id was derived — and even if it had been, no cli matched
    # INJECT_CLIS — so no {K_INJECTED!r} key should have been written.
    assert not any(k.startswith(K_INJECTED.format(session=""))
                   for k in marker_keys), (
        f"no session means no {K_INJECTED!r}* key — found {marker_keys!r}")
    print(f"  message-less request: no messages key, no {K_INJECTED!r}* key")


def test_lease_drop_re_enables_injection():
    """Inject once, then `await h.picker.slots.drop_lease(session)`, run
    again: the note is present again.

    Regression for the drop_lease contract in slots.py: clearing the
    lease also clears the injected marker, so a hard plan rejection that
    re-leases the session re-enables injection on the next turn. Without
    it, a session that lost its plan to a 429 (lease dropped, fresh plan
    granted, marker survives) would lose the instruction that explains
    why its native tools are stripped in this environment.
    """
    async def go():
        h, reg, redis = _build()
        session = "sess-inj-5"
        first_msgs = [{"role": "user", "content": "first turn"}]
        d1 = _request(session=session, cli="codex", messages=list(first_msgs))
        await h.async_pre_call_hook(None, None, d1, "acompletion")
        await _release(h, d1)

        actual = derive_session(d1, None)
        marker_after_first = redis.strings.get(K_INJECTED.format(session=actual))

        # The drop happens the way _apply_verdict does it for QUOTA_EXHAUSTED.
        await h.picker.slots.drop_lease(actual)
        marker_after_drop = redis.strings.get(K_INJECTED.format(session=actual))

        # Fresh request, same session id: without the drop the marker
        # would still gate the second turn. The drop is what makes the
        # second turn a re-injection rather than a no-op.
        second_msgs = [{"role": "user", "content": "second turn"}]
        d2 = _request(session=session, cli="codex", messages=list(second_msgs))
        await h.async_pre_call_hook(None, None, d2, "acompletion")
        await _release(h, d2)

        return (marker_after_first, marker_after_drop,
                d1["messages"], d2["messages"])

    m1, m_drop, first, second = _run(go())
    assert m1 is not None, "first call must set the marker"
    assert m_drop is None, (
        f"drop_lease must clear the marker, got {m_drop!r}")
    # First call's leading user turn now has the note prepended.
    assert first[0]["role"] == "system" and "SwitchYard" in first[0]["content"]
    assert first[1] == {"role": "user", "content": "first turn"}
    # Second call: marker was cleared by drop_lease, so injection fires
    # again. Without the drop, the second turn would be a single user
    # message — this is the regression we are catching.
    assert len(second) == 2, second
    assert second[0]["role"] == "system" and "SwitchYard" in second[0]["content"]
    assert second[1] == {"role": "user", "content": "second turn"}
    print(f"  drop_lease cleared marker ({m_drop is None}); "
          f"second turn re-injected the system note")


def test_cli_identity_via_metadata_fallback():
    """CLI supplied as `metadata.switchyard_cli` instead of the header:
    injection still happens. Symmetry with the session-identity fallback
    in session.derive — same body-shaped path, same INJECT_CLIS gate,
    same `sy:inject:{session}` marker."""
    async def go():
        h, reg, redis = _build()
        session = "sess-inj-6"
        before = [{"role": "user", "content": "metadata cli"}]
        data: dict = {
            "model": "forge",
            "messages": list(before),
            "proxy_server_request": {"headers": {
                "x-switchyard-session": session,
                # Deliberately no x-switchyard-cli here — the metadata
                # fallback is what carries the CLI identity.
            }},
            "metadata": {CLI_META_FIELD: "opencode"},
        }
        await h.async_pre_call_hook(None, None, data, "acompletion")
        await _release(h, data)

        actual = derive_session(data, None)
        marker = redis.strings.get(K_INJECTED.format(session=actual))
        return data["messages"], marker

    messages, marker = _run(go())
    assert marker is not None, (
        f"metadata.{CLI_META_FIELD} must drive injection, got {marker!r}")
    assert len(messages) == 2, messages
    assert messages[0]["role"] == "system" and "SwitchYard" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "metadata cli"}
    print(f"  metadata.{CLI_META_FIELD}=opencode: header-less injection fired, "
          f"marker key set")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
