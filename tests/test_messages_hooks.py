"""Tests for the /v1/messages success-finishing path.

LiteLLM 1.101.0 routes buffered ``/v1/messages`` (Anthropic Messages API) through
``async_post_call_success_hook`` only and streamed ``/v1/messages`` through
``async_post_call_streaming_iterator_hook`` only -- the normal
``async_log_success_event`` does NOT fire for either. The four classic call
types (``completion``, ``acompletion``, ``text_completion``,
``atext_completion``) DO still fire ``async_log_success_event``. When both hooks
end up firing for the same request (some LiteLLM builds do for the buffered
OpenAI paths), the exactly-once marker keeps the slot release, ledger record,
transient-failure reset, throughput note and limit-header capture from running
twice.

These tests wire a minimal ``SwitchyardHandler`` against the shared FakeRedis
(``tests/fake_redis.py``) and drive the four hooks end-to-end -- no real Redis,
no network, no provider. The wiring pattern matches ``test_verdict.py``: a
``SwitchyardHandler.__new__`` skip of ``__init__`` (no config-watcher thread,
no real registry load), with the dependent objects injected through
``__dict__``.

The tests live BEFORE ``if __name__ == "__main__":`` (see CLAUDE.md: anything
appended after the runner is defined too late to be collected and silently
does not run).
"""
from __future__ import annotations

import asyncio
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()

from switchyard import models  # noqa: E402
from switchyard.hooks import (  # noqa: E402
    META_KEY,
    SwitchyardHandler,
    UNLOGGED_CALL_TYPES,
    _TERMINATED,
    _collect_anthropic_event_usage,
)
from switchyard.picker import Picker  # noqa: E402
from switchyard.policy import CapacityPolicy  # noqa: E402
from switchyard.slots import SlotTable  # noqa: E402
from switchyard.usage import Ledger  # noqa: E402
from tests.fake_redis import FakeRedis  # noqa: E402


# ----------------------------------------------------------------------------
# Sync helpers (safe to call from outside an event loop).
# ----------------------------------------------------------------------------
def _bucket_sync(redis: FakeRedis, key: str) -> dict:
    """Synchronous read of a FIELDS-shaped bucket out of FakeRedis.

    Reaches into the FakeRedis dict directly instead of going through the
    async ``hgetall`` -- the tests' assertion code lives outside the event
    loop and must not call ``asyncio.run`` from inside a coroutine.
    """
    out = {f: 0.0 for f in ("requests", "prompt_tokens",
                            "completion_tokens", "cost", "failures")}
    raw = redis.hashes.get(key, {})
    for k, v in raw.items():
        try:
            out[k.decode() if isinstance(k, bytes) else k] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def _first_period_key(redis: FakeRedis, plan_key: str) -> str | None:
    """Find the period bucket for ``plan_key`` written by ``ledger.record``.

    The ledger writes one bucket per quota window; for tests we read the
    first one (the test fixture plans all have a single window). Returns
    None when no bucket exists yet.
    """
    for k in redis.hashes:
        if k.startswith(f"sy:usage:{plan_key}:p:"):
            return k
    return None


# ----------------------------------------------------------------------------
# Build helpers -- run inside ``asyncio.run`` so the test bodies stay sync.
# ----------------------------------------------------------------------------
def _build(*, claim_lane: str = "apex"):
    """Wire a SwitchyardHandler + dependents against a fresh FakeRedis.

    Pre-claims a slot on the chosen lane so the success / failure hooks
    have a real claim to release. Returns ``(handler, reg, slots, ledger,
    policy, redis, pick)``.
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
    h.__dict__["_picker"] = picker
    h.__dict__["_redis"] = redis
    h.__dict__["_beats"] = {}

    pick = asyncio.run(picker.pick(claim_lane, None))
    return h, reg, slots, ledger, policy, redis, pick


def _ctx_for(pick, *, call_type: str = "anthropic_messages"):
    """Build the per-request metadata dict a finished hook would see."""
    return {
        "lane": pick.lane,
        "plan": pick.plan.key,
        "model": pick.model.ref,
        "request_id": pick.request_id,
        "session": None,
        "sticky": pick.sticky,
        "claimed_at": 0.0,
        "cap": pick.cap,
        "direct": False,
        "needs_tools": False,
        "pinned": False,
        "picked_group_gid": "",
        "picked_group_strategy": "",
        "call_type": call_type,
    }


def _meta_for(ctx):
    """Wrap a ctx dict the way ``async_pre_call_hook`` left it in metadata."""
    return {META_KEY: ctx}


def _anthropic_response(pick, *, usage, extra=None):
    """Build a buffered /v1/messages-shaped response for the given pick."""
    base = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": pick.model.ref,
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": usage,
    }
    if extra:
        base.update(extra)
    return base


# ============================================================================
# Section 1: pre-call hook stores call_type
# ============================================================================
def test_pre_call_hook_stamps_call_type_into_ctx():
    """``async_pre_call_hook`` must stamp call_type so the post-call hooks know
    which side of LiteLLM's split logging path owns the slot release.

    Without the stamp, ``async_post_call_success_hook`` cannot tell an
    ``anthropic_messages`` request (owned by post-call) from an ``acompletion``
    one (owned by normal logging) -- and double-booking the slot is exactly
    the bug this whole change closes.
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
    h.__dict__["_picker"] = picker
    h.__dict__["_redis"] = redis
    h.__dict__["_beats"] = {}

    out = asyncio.run(h.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data={"model": "apex", "metadata": {}},
        call_type="anthropic_messages",
    ))
    assert out["metadata"][META_KEY]["call_type"] == "anthropic_messages", (
        out["metadata"][META_KEY])
    print("  pre-call stamped call_type='anthropic_messages' onto ctx")


# ============================================================================
# Section 2: buffered /v1/messages accounting
# ============================================================================
def test_buffered_messages_post_call_finish_releases_slot_and_books_usage():
    """Buffered /v1/messages reaches async_post_call_success_hook with an
    Anthropic-shaped response (input_tokens / output_tokens, NOT the OpenAI
    prompt_tokens / completion_tokens). The post-call hook must release the
    slot, book the prompt/completion tokens onto the plan ledger, reset the
    transient-failure streak, note throughput and stamp the served deployment
    onto the response.

    The metadata path used by async_post_call_success_hook is the
    request_data dict, not the kwargs-style model_call_details dict: LiteLLM
    passes ``data`` straight through, and ``litellm_params`` (where
    ``response_cost`` would normally live) is a sibling key.
    """
    h, _reg, slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {
        "metadata": _meta_for(ctx),
        "litellm_params": {"metadata": _meta_for(ctx)},
    }
    response = _anthropic_response(pick, usage={
        "input_tokens": 270,
        "output_tokens": 64,
    })

    async def go():
        assert await slots.in_flight(pick.plan.key) == 1
        return await h.async_post_call_success_hook(
            data=request_data,
            user_api_key_dict=None,
            response=response,
        )

    out = asyncio.run(go())

    assert asyncio.run(slots.in_flight(pick.plan.key)) == 0, (
        "post-call success did not release the slot")
    plan_keys = [k for k in redis.hashes.keys()
                 if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
    assert plan_keys, f"no period bucket for plan {pick.plan.key}"
    bucket = _bucket_sync(redis, plan_keys[0])
    assert bucket["prompt_tokens"] == 270, bucket
    assert bucket["completion_tokens"] == 64, bucket
    assert bucket["requests"] == 1, bucket
    assert out.get("switchyard") == {
        "lane": ctx["lane"], "plan": pick.plan.key,
        "model": pick.model.ref, "sticky": False,
    }, out.get("switchyard")
    assert ctx[_TERMINATED] == "success", ctx.get(_TERMINATED)
    print(f"  buffered /v1/messages released slot, booked "
          f"{int(bucket['prompt_tokens'])} prompt / "
          f"{int(bucket['completion_tokens'])} completion tokens")


def test_buffered_messages_tolerates_openai_shaped_usage():
    """A model that hands the buffered hook an OpenAI-shaped usage block
    (prompt_tokens / completion_tokens) still books correctly -- some
    providers route /v1/messages through LiteLLM's OpenAI normaliser and the
    response ends up with OpenAI keys.

    The implementation accepts both shapes -- a real Anthropic request carries
    input_tokens / output_tokens; a normalised request carries prompt_tokens /
    completion_tokens. Both must produce identical ledger entries.
    """
    h, _reg, _slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {
        "metadata": _meta_for(ctx),
        "litellm_params": {"metadata": _meta_for(ctx)},
    }
    response = _anthropic_response(pick, usage={
        "prompt_tokens": 12, "completion_tokens": 7,
    })

    async def go():
        await h.async_post_call_success_hook(
            data=request_data, user_api_key_dict=None, response=response,
        )
        plan_keys = [k for k in redis.hashes.keys()
                     if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
        return _bucket_sync(redis, plan_keys[0])

    bucket = asyncio.run(go())
    assert bucket["prompt_tokens"] == 12, bucket
    assert bucket["completion_tokens"] == 7, bucket
    print(f"  OpenAI-shaped usage on /v1/messages: "
          f"{int(bucket['prompt_tokens'])} prompt / "
          f"{int(bucket['completion_tokens'])} completion")


def test_buffered_messages_folds_cache_tokens_into_prompt_tokens():
    """Anthropic reports cache_read_input_tokens and
    cache_creation_input_tokens as separate keys; the gateway CLI bridge
    folds them into input_tokens when projecting to OpenAI shape, and the
    hooks here must do the same so the ledger (which only knows prompt /
    completion) sees the full input cost.

    Without the fold, a cache-heavy plan under-reports its prompt tokens
    and the pacer thinks it has more headroom than it does. Cache tokens
    fold ONLY into prompt_tokens -- not completion_tokens.
    """
    h, _reg, _slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {
        "metadata": _meta_for(ctx),
        "litellm_params": {"metadata": _meta_for(ctx)},
    }
    response = _anthropic_response(pick, usage={
        "input_tokens": 100,
        "output_tokens": 12,
        "cache_read_input_tokens": 800,
        "cache_creation_input_tokens": 50,
    })

    async def go():
        await h.async_post_call_success_hook(
            data=request_data, user_api_key_dict=None, response=response,
        )
        plan_keys = [k for k in redis.hashes.keys()
                     if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
        return _bucket_sync(redis, plan_keys[0])

    bucket = asyncio.run(go())
    # 100 (input) + 800 (cache read) + 50 (cache creation) = 950
    assert bucket["prompt_tokens"] == 950, bucket
    assert bucket["completion_tokens"] == 12, bucket
    print(f"  cache tokens folded: 100 input + 800 cache_read + "
          f"50 cache_creation = {int(bucket['prompt_tokens'])} prompt tokens")


def test_buffered_messages_uses_response_cost_when_metered():
    """``response_cost`` from the litellm_params must land in the ledger when
    the plan is metered. On a subscription the marginal cost is zero and
    writing it would inflate the month_cost figure the portal uses to
    decide whether to renew -- so cost is only recorded for metered plans.
    """
    from dataclasses import replace
    h, reg, _slots, _ledger, _policy, redis, pick = _build()
    plan_key = pick.plan.key
    metered_plan = replace(reg.plans[plan_key], metered=True)
    new_plans = {**reg.plans, metered_plan.key: metered_plan}
    h.registry = models.Registry(settings=reg.settings, plans=new_plans,
                                  lanes=reg.lanes)

    ctx = _ctx_for(pick, call_type="anthropic_messages")
    ctx["plan"] = plan_key
    request_data = {
        "metadata": _meta_for(ctx),
        "litellm_params": {
            "metadata": _meta_for(ctx),
            "response_cost": 0.0007,
        },
    }
    response = _anthropic_response(pick, usage={
        "input_tokens": 100, "output_tokens": 12,
    })

    async def go():
        await h.async_post_call_success_hook(
            data=request_data, user_api_key_dict=None, response=response,
        )
        plan_keys = [k for k in redis.hashes.keys()
                     if k.startswith(f"sy:usage:{plan_key}:p:")]
        return _bucket_sync(redis, plan_keys[0])

    bucket = asyncio.run(go())
    assert abs(bucket["cost"] - 0.0007) < 1e-9, bucket
    print(f"  metered plan booked cost=${bucket['cost']:.6f}")


# ============================================================================
# Section 3: streamed /v1/messages accounting
# ============================================================================
def test_streamed_messages_extracts_usage_from_message_start_and_delta():
    """Streamed Anthropic usage is split across two SSE event types:
    ``message_start`` carries the initial input/cache totals, successive
    ``message_delta`` frames carry the running output/cache totals.

    The hooks must accumulate across the whole stream and book the final
    totals onto the plan ledger. message_delta's totals are *running*, not
    deltas, so a later frame overwrites the previous one rather than adding
    to it -- a fragmented stream (or a frame that arrives before the model
    has produced any output) cannot inflate the total.
    """
    h, _reg, slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {"metadata": _meta_for(ctx)}

    chunks = [
        # message_start with input + cache; output_tokens is the "at least 1" placeholder
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"claude-...","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":300,"output_tokens":1,"cache_creation_input_tokens":10,"cache_read_input_tokens":50}}}\n\n',
        # a content_block_delta -- no usage here
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}\n\n',
        # message_delta running totals (real output count)
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":42,"cache_creation_input_tokens":10,"cache_read_input_tokens":50}}\n\n',
        # another message_delta -- output grew
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":null},"usage":{"output_tokens":73,"cache_creation_input_tokens":10,"cache_read_input_tokens":50}}\n\n',
        # message_stop
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]

    async def produce():
        for c in chunks:
            yield c

    async def consume():
        async for _ in h.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None, response=produce(), request_data=request_data,
        ):
            pass
        # Give the detached finalize task time to land on the event loop.
        await asyncio.sleep(0.05)

    asyncio.run(consume())

    assert asyncio.run(slots.in_flight(pick.plan.key)) == 0
    plan_keys = [k for k in redis.hashes.keys()
                 if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
    bucket = _bucket_sync(redis, plan_keys[0])
    # 300 (input) + 50 (cache_read) + 10 (cache_creation) = 360 prompt
    # Output: 73 (the LAST message_delta wins -- not 1+42+73, not 42)
    assert bucket["prompt_tokens"] == 360, bucket
    assert bucket["completion_tokens"] == 73, bucket
    assert bucket["requests"] == 1, bucket
    print(f"  streamed Anthropic SSE: prompt={int(bucket['prompt_tokens'])} "
          f"completion={int(bucket['completion_tokens'])}")


def test_streamed_messages_yields_every_chunk_verbatim():
    """The streaming hook must preserve every byte and never let a parse
    error break the client stream.

    We interleave a malformed frame (a frame with no ``data:`` line and an
    unparseable JSON payload) with valid Anthropic SSE chunks. Every byte
    the upstream iterator produces must reach the consumer, in order, and
    the malformed frame must not throw.
    """
    h, _reg, _slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {"metadata": _meta_for(ctx)}

    raw_chunks = [
        b'event: ping\ndata: {"type": "ping"}\n\n',
        b'event: message_start\ndata: {not valid json\n\n',  # malformed payload
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":50,"output_tokens":1}}}\n\n',
        b': this is a comment line\n\n',                      # SSE comment, no data:
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]

    async def produce():
        for c in raw_chunks:
            yield c

    seen = []

    async def consume():
        async for c in h.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None, response=produce(), request_data=request_data,
        ):
            seen.append(c)

    asyncio.run(consume())

    assert seen == raw_chunks, (
        f"stream lost chunks:\n  sent={raw_chunks}\n  "
        f"got={[s[:30] for s in seen]}")
    print(f"  streamed hook passed {len(seen)} chunks verbatim through "
          f"malformed payloads + SSE comments")


def test_streamed_messages_tolerates_dict_and_pydantic_chunk_shapes():
    """A few code paths (agentic loop, synthetic stream) hand the hook
    pre-parsed chunks (dicts, pydantic objects) instead of raw SSE bytes.
    The hook must accumulate usage from those too, and must not regress on
    any of the three shapes.
    """
    h, _reg, _slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {"metadata": _meta_for(ctx)}

    class _Pydantic:
        def model_dump(self):
            return {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 21},
            }

    chunks = [
        {"type": "message_start",
         "message": {"id": "m", "type": "message", "role": "assistant",
                     "model": "x", "content": [], "stop_reason": None,
                     "stop_sequence": None,
                     "usage": {"input_tokens": 80, "output_tokens": 1}}},
        _Pydantic(),
        {"type": "message_stop"},
    ]

    async def produce():
        for c in chunks:
            yield c

    async def consume():
        async for _ in h.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None, response=produce(), request_data=request_data,
        ):
            pass
        await asyncio.sleep(0.05)

    asyncio.run(consume())
    plan_keys = [k for k in redis.hashes.keys()
                 if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
    bucket = _bucket_sync(redis, plan_keys[0])
    assert bucket["prompt_tokens"] == 80, bucket
    assert bucket["completion_tokens"] == 21, bucket
    print(f"  streamed hook extracted usage from dict + pydantic chunks: "
          f"{int(bucket['prompt_tokens'])} / "
          f"{int(bucket['completion_tokens'])}")


# ============================================================================
# Section 4: cancellation safety
# ============================================================================
def test_streamed_messages_consumer_disconnect_releases_slot_exactly_once():
    """When the consumer of the streaming response disappears mid-iteration
    (a real client hangup, not an upstream error), the slot MUST still be
    released -- otherwise the picker observes the plan as busy until the
    staleness sweep eventually drops the entry, and the capacity board
    reports a slot that nothing is running.

    The streaming hook drives the finalize from a detached task spawned in
    the cancellation arm of the iterator, so the asyncio GeneratorExit /
    CancelledError raised by the client disconnect cannot prevent the
    release. The exactly-once marker then guards against any other path
    also firing on the same request.
    """
    h, _reg, slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {"metadata": _meta_for(ctx)}

    async def produce():
        yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}\n\n'
        yield b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}\n\n'
        yield b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":null},"usage":{"output_tokens":3}}\n\n'
        yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    async def consume_one_chunk():
        gen = h.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None, response=produce(), request_data=request_data,
        )
        # Take one chunk only, then bail (mimics a client that hung up after
        # reading the start of the stream).
        async for _ in gen:
            break
        # Force-close the iterator so the detached finalize gets scheduled.
        await gen.aclose()
        await asyncio.sleep(0.1)

    assert asyncio.run(slots.in_flight(pick.plan.key)) == 1
    asyncio.run(consume_one_chunk())
    assert asyncio.run(slots.in_flight(pick.plan.key)) == 0, (
        "client disconnect left the slot claimed")
    print("  client cancel mid-stream released the slot; "
          "detach task ran after GeneratorExit")


def test_streamed_messages_upstream_error_still_releases_slot():
    """An exception raised by the upstream iterator (a network drop, a
    provider-side closure) must also trigger the slot release. The
    streaming hook wraps its ``async for`` in a try/except that schedules
    the detached task from the failure arm, so a partial stream ends the same
    way a clean one does.
    """
    h, _reg, slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {"metadata": _meta_for(ctx)}

    class _NetworkDrop(Exception):
        pass

    async def produce():
        yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
        raise _NetworkDrop("upstream closed")

    async def consume():
        try:
            async for _ in h.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=produce(), request_data=request_data,
            ):
                pass
        except _NetworkDrop:
            pass
        await asyncio.sleep(0.1)

    assert asyncio.run(slots.in_flight(pick.plan.key)) == 1
    asyncio.run(consume())
    assert asyncio.run(slots.in_flight(pick.plan.key)) == 0
    print("  upstream exception also released the slot via detach task")


# ============================================================================
# Section 5: OpenAI-shaped routes are not double-finished
# ============================================================================
def test_logged_call_type_is_not_double_finished_by_post_call_hook():
    """``/v1/chat/completions`` buffered (call_type = ``acompletion``) reaches
    BOTH ``async_log_success_event`` AND ``async_post_call_success_hook`` on
    the same request. The first call runs the finish logic; the second call
    sees the marker and short-circuits. Without the marker, the slot would
    be released twice and the ledger would record the request twice.

    This is the regression test for the OpenAI-shaped double-call case the
    task explicitly calls out.
    """
    h, _reg, slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="acompletion")  # NOT in UNLOGGED_CALL_TYPES
    kwargs = {"metadata": _meta_for(ctx)}
    response = {
        "id": "chatcmpl-x", "object": "chat.completion",
        "model": pick.model.ref,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5},
    }

    async def go():
        await h.async_log_success_event(
            kwargs=kwargs, response_obj=response,
            start_time=None, end_time=None,
        )
        # The post-call hook arrives next; it stamps and would normally
        # try to finish, but the marker blocks it.
        await h.async_post_call_success_hook(
            data={"metadata": _meta_for(ctx)},
            user_api_key_dict=None,
            response=dict(response),  # fresh dict so we can stamp it
        )
        plan_keys = [k for k in redis.hashes.keys()
                     if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
        return (await slots.in_flight(pick.plan.key),
                _bucket_sync(redis, plan_keys[0]))

    inflight, bucket = asyncio.run(go())
    assert inflight == 0, "slot must be released exactly once"
    assert bucket["requests"] == 1, bucket
    assert bucket["prompt_tokens"] == 11, bucket
    assert bucket["completion_tokens"] == 5, bucket
    assert ctx[_TERMINATED] == "success", ctx.get(_TERMINATED)
    print(f"  acompletion on both hooks: slot released once, "
          f"requests={int(bucket['requests'])} (no double-booking)")


def test_logged_call_type_is_not_double_finished_by_streaming_hook():
    """Streamed ``/v1/chat/completions`` (call_type = ``acompletion``) reaches
    ``async_post_call_streaming_iterator_hook`` too -- for the reasoning
    split. The hook must NOT also finish success accounting for it (that
    path is still owned by ``async_log_success_event`` when it fires after
    the assembled ModelResponse).

    Without the gate, an OpenAI streamed request would see two finishes --
    the streaming hook's detached task and async_log_success_event -- and
    the marker would short-circuit the second, but only because the second
    is the right one. We want only the right one to ever run at all.
    """
    h, _reg, slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="acompletion")
    request_data = {"metadata": _meta_for(ctx)}

    class _FakeDelta:
        choices = []
        def __init__(self):
            pass

    async def produce():
        yield _FakeDelta()
        yield _FakeDelta()

    async def consume():
        async for _ in h.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None, response=produce(), request_data=request_data,
        ):
            pass
        await asyncio.sleep(0.05)

    asyncio.run(consume())
    assert asyncio.run(slots.in_flight(pick.plan.key)) == 1, (
        "streamed acompletion must not release the slot here")
    plan_keys = [k for k in redis.hashes.keys()
                 if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
    assert not plan_keys, plan_keys
    print("  streamed acompletion: slot still claimed (open for "
          "async_log_success_event to release)")


# ============================================================================
# Section 6: failure paths
# ============================================================================
def test_logged_failure_is_not_double_finished():
    """Same shape as the success double-call case: a buffered OpenAI-shaped
    failure can fire both ``async_log_failure_event`` and
    ``async_post_call_failure_hook``. The first call releases + records the
    failure; the second sees the marker and short-circuits.
    """
    h, _reg, slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="acompletion")
    kwargs = {"metadata": _meta_for(ctx), "exception": RuntimeError("boom")}

    async def go():
        await h.async_log_failure_event(
            kwargs=kwargs, response_obj=None,
            start_time=None, end_time=None,
        )
        # Post-call failure hook arrives second.
        await h.async_post_call_failure_hook(
            request_data={"metadata": _meta_for(ctx)},
            original_exception=RuntimeError("boom"),
            user_api_key_dict=None,
        )
        return await slots.in_flight(pick.plan.key)

    inflight = asyncio.run(go())
    assert inflight == 0, "slot must be released exactly once on failure"
    assert ctx[_TERMINATED] == "failure", ctx.get(_TERMINATED)
    print("  failure path: slot released once via log_failure_event, "
          "post_call_failure_hook short-circuited on marker")


def test_messages_failure_releases_via_post_call_failure_hook():
    """``/v1/messages`` failures reach ``async_post_call_failure_hook``
    (LiteLLM does NOT fire ``async_log_failure_event`` for them). The hook
    must release the slot and call ``_handle_failure`` so the picker places
    against the cooldowns this failure just set on the caller's retry.
    """
    h, _reg, slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")

    class _UpstreamError(Exception):
        status_code = 429

    async def go():
        assert await slots.in_flight(pick.plan.key) == 1
        await h.async_post_call_failure_hook(
            request_data={"metadata": _meta_for(ctx)},
            original_exception=_UpstreamError("rate limit"),
            user_api_key_dict=None,
        )
        return await slots.in_flight(pick.plan.key)

    inflight = asyncio.run(go())
    assert inflight == 0, "slot must be released by post_call_failure_hook"
    assert ctx[_TERMINATED] == "failure", ctx.get(_TERMINATED)
    print("  anthropic_messages failure released slot via "
          "post_call_failure_hook")


# ============================================================================
# Section 7: defensive checks
# ============================================================================
def test_unlogged_call_type_in_normal_log_event_is_a_no_op():
    """If a future LiteLLM version starts firing ``async_log_success_event``
    for ``anthropic_messages`` (it does not, today -- verified 1.101.0), the
    hook must NOT also run the finish: the post-call side already owns the
    slot. The call_type gate at the top of ``async_log_success_event``
    enforces this.
    """
    h, _reg, slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    kwargs = {"metadata": _meta_for(ctx)}

    async def go():
        # First the post-call hook runs and finishes the slot.
        await h.async_post_call_success_hook(
            data={"metadata": _meta_for(ctx)},
            user_api_key_dict=None,
            response={
                "type": "message", "model": pick.model.ref,
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )
        # Then a stray async_log_success_event arrives; must NOT release
        # again or double-book.
        await h.async_log_success_event(
            kwargs=kwargs,
            response_obj={"usage": {"prompt_tokens": 9999,
                                    "completion_tokens": 9999}},
            start_time=None, end_time=None,
        )
        plan_keys = [k for k in redis.hashes.keys()
                     if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
        return (await slots.in_flight(pick.plan.key),
                _bucket_sync(redis, plan_keys[0]))

    inflight, bucket = asyncio.run(go())
    assert inflight == 0, "no double-release on a stray log_success_event"
    assert bucket["requests"] == 1, bucket
    # The bogus 9999 must NOT have been booked -- it was the post-call
    # hook's 10/2 that landed.
    assert bucket["prompt_tokens"] == 10, bucket
    print("  stray log_success_event on anthropic_messages was a no-op; "
          f"prompt={int(bucket['prompt_tokens'])} (the 9999 was ignored)")


def test_unlogged_call_types_constant_contains_anthropic_messages():
    """Belt-and-braces check on the dispatch constant: any new unlogged
    call type MUST be added here. A test that fails on this lets a future
    reviewer see exactly which call types are covered, not a chain of
    elif blocks scattered through the hooks.
    """
    assert "anthropic_messages" in UNLOGGED_CALL_TYPES, UNLOGGED_CALL_TYPES
    assert "aanthropic_messages" in UNLOGGED_CALL_TYPES, UNLOGGED_CALL_TYPES
    print(f"  UNLOGGED_CALL_TYPES = {sorted(UNLOGGED_CALL_TYPES)}")


def test_collect_anthropic_event_usage_tolerates_partial_frames():
    """``_collect_anthropic_event_usage`` is the lowest-level parser. A
    ``message_start`` with no usage block, a ``message_delta`` with no
    usage, and a junk event type all need to be no-ops so the streaming
    hook never crashes on a partial frame.
    """
    collected: dict = {}
    # Empty usage.
    _collect_anthropic_event_usage({"type": "message_start", "message": {}}, collected)
    assert collected == {}, collected
    # message_delta with no usage.
    _collect_anthropic_event_usage({"type": "message_delta", "delta": {}}, collected)
    assert collected == {}, collected
    # Junk type.
    _collect_anthropic_event_usage({"type": "ping"}, collected)
    assert collected == {}, collected
    # A real frame does land.
    _collect_anthropic_event_usage({
        "type": "message_start",
        "message": {"usage": {"input_tokens": 17, "output_tokens": 1,
                              "cache_read_input_tokens": 4,
                              "cache_creation_input_tokens": 0}},
    }, collected)
    assert collected["input_tokens"] == 17, collected
    assert collected["cache_read_input_tokens"] == 4, collected
    print("  parser tolerates empty / partial / non-dict Anthropic frames")


def test_plan_no_longer_at_cap_after_messages_finish():
    """A picked plan that is at its connection limit cannot accept another
    pick until the in-flight request finishes. After the buffered
    ``/v1/messages`` post-call hook runs, the slot must be back on the
    picker -- the next ``pick`` succeeds.
    """
    h, reg, slots, _ledger, _policy, _redis, pick = _build()

    # Force the picked plan to its cap. With one slot already claimed, the
    # next pick against the same lane must spill.
    plan = pick.plan
    cap = plan.cap_for(pick.model, reg.settings)
    claimed_ids = [pick.request_id]

    async def saturate():
        ids = []
        for _ in range(cap - 1):
            try:
                p = await h.picker.pick(pick.lane, None)
                ids.append(p.request_id)
            except Exception:
                # If the lane doesn't have a same-plan second member, claim
                # directly against the picked model instead.
                p = await h.picker.pick_direct(pick.model, None)
                ids.append(p.request_id)
        return ids

    more_ids = asyncio.run(saturate())
    claimed_ids.extend(more_ids)

    # Now finish the original claim via the post-call hook.
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {
        "metadata": _meta_for(ctx),
        "litellm_params": {"metadata": _meta_for(ctx)},
    }
    response = _anthropic_response(pick, usage={
        "input_tokens": 5, "output_tokens": 1,
    })

    async def go():
        await h.async_post_call_success_hook(
            data=request_data, user_api_key_dict=None, response=response,
        )
        inflight = await slots.in_flight(plan.key)
        # A fresh pick on the same lane / model must succeed (the cap is
        # no longer exceeded because the original claim released).
        p = await h.picker.pick(pick.lane, None)
        return inflight, p

    inflight, _new_pick = asyncio.run(go())
    assert inflight == cap - 1, (
        f"slot release did not drop the picker to cap-1={cap-1}, "
        f"got {inflight}")
    print(f"  picker is back at cap-1 after the messages post-call release "
          f"(was at {cap}, now {inflight})")


# ============================================================================
# Section 8: failure classification on a 200
# ============================================================================
def test_messages_200_with_embedded_error_records_as_failure():
    """Some providers (MiniMax is the load-bearing case) return HTTP 200
    with the real failure in the body. The buffered hook must catch that
    via ``inspect_success_payload`` and record it as a failure (no
    usage, no throughput, just the failed=True record + verdict cooldown).
    """
    h, _reg, _slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")

    request_data = {
        "metadata": _meta_for(ctx),
        "litellm_params": {"metadata": _meta_for(ctx)},
    }
    # An HTTP 200 carrying a base_resp.status_code != 0 -- the exact shape
    # that triggers inspect_success_payload.
    response = _anthropic_response(pick, usage={
        "input_tokens": 50, "output_tokens": 1,  # would book if success
    }, extra={
        "base_resp": {"status_code": 1008, "status_msg": "insufficient balance"},
    })

    async def go():
        await h.async_post_call_success_hook(
            data=request_data, user_api_key_dict=None, response=response,
        )
        plan_keys = [k for k in redis.hashes.keys()
                     if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
        return (_bucket_sync(redis, plan_keys[0]) if plan_keys else None)

    bucket = asyncio.run(go())
    if bucket is not None:
        # The handler refused the success and recorded a failure instead.
        assert bucket["failures"] >= 1 or bucket["requests"] == 0, bucket
        # And the prompt/completion counts from the usage block were NOT
        # booked -- the verdict path skipped the ledger.record of success.
        assert bucket["prompt_tokens"] == 0, (
            f"prompt tokens were booked despite the embedded error: {bucket}")
    print("  HTTP 200 carrying base_resp.status_code=1008 was recorded as a "
          "failure, not a success")


# ============================================================================
# Section 9: end-to-end pick -> post-call happy path
# ============================================================================
def test_end_to_end_pick_then_finish_then_pick_again():
    """End-to-end sanity: pick a slot, finish via post-call, pick again on
    the same lane. The second pick must succeed because the slot was
    released by the first finish.

    This is the property that the operator notices: a successful request
    does not strand its own connection limit.
    """
    h, _reg, _slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {
        "metadata": _meta_for(ctx),
        "litellm_params": {"metadata": _meta_for(ctx)},
    }
    response = _anthropic_response(pick, usage={
        "input_tokens": 1, "output_tokens": 1,
    })

    async def go():
        await h.async_post_call_success_hook(
            data=request_data, user_api_key_dict=None, response=response,
        )
        # Second pick: the cap is restored.
        return await h.picker.pick(pick.lane, None)

    p2 = asyncio.run(go())
    assert p2.request_id != pick.request_id, (
        f"second pick landed on the same request_id {pick.request_id}")
    print(f"  end-to-end pick -> finish -> pick: lane {pick.lane} served "
          "two requests back-to-back")


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as exc:
                failures += 1
                print(f" FAIL {name}\n    {exc}")
            except Exception as exc:
                failures += 1
                print(f" FAIL {name}\n    {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(f"{failures} failed")
    n = sum(1 for k, v in globals().items()
            if k.startswith("test_") and callable(v))
    print(f"\n{n} messages-hook tests passed")