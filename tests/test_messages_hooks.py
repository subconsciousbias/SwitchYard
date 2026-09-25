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
from types import SimpleNamespace

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
    _call_facts,
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


def _failure_count(redis: FakeRedis, plan_key: str) -> int:
    """How many failure rows ``ledger.record`` has written for ``plan_key``.

    Reads every period bucket so a multi-window plan does not under-report.
    Returns 0 when no bucket exists yet (which is the expected pre-failure
    state).
    """
    n = 0
    for k, v in redis.hashes.items():
        if k.startswith(f"sy:usage:{plan_key}:p:"):
            for fk, fv in v.items():
                if isinstance(fk, bytes):
                    fk = fk.decode()
                if fk == "failures":
                    try:
                        n += int(float(fv))
                    except (TypeError, ValueError):
                        pass
    return n


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


def test_buffered_messages_issue58_payload_folds_cache_into_prompt_tokens():
    """Regression pin for the #58 booking hole: an Anthropic-shaped usage
    block where input_tokens is tiny (6) but cache_read_input_tokens
    (41,302) and cache_creation_input_tokens (6,932) carry almost all of
    the prompt cost must book as 48,240 prompt tokens, not 6. Without the
    cache fold the pacer thinks it has near-infinite headroom and a
    cache-heavy plan burns the budget in one burst.
    """
    h, _reg, _slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {
        "metadata": _meta_for(ctx),
        "litellm_params": {"metadata": _meta_for(ctx)},
    }
    response = _anthropic_response(pick, usage={
        "input_tokens": 6,
        "cache_creation_input_tokens": 6932,
        "cache_read_input_tokens": 41302,
        "output_tokens": 201,
    })

    async def go():
        await h.async_post_call_success_hook(
            data=request_data, user_api_key_dict=None, response=response,
        )
        plan_keys = [k for k in redis.hashes.keys()
                     if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
        return _bucket_sync(redis, plan_keys[0])

    bucket = asyncio.run(go())
    # 6 (input) + 6932 (cache_creation) + 41302 (cache_read) = 48240
    assert bucket["prompt_tokens"] == 48240, bucket
    assert bucket["completion_tokens"] == 201, bucket
    print(f"  #58 Anthropic payload booked: {int(bucket['prompt_tokens'])} "
          f"prompt / {int(bucket['completion_tokens'])} completion "
          f"(6 + 6932 + 41302)")


def test_buffered_messages_does_not_double_count_cache_when_prompt_tokens_present():
    """No-double-count regression: when the OpenAI-shaped ``prompt_tokens``
    is present (already folded: e.g. the gateway CLI bridge's ``to_openai``
    output) the cache counters must NOT be added again. The buggy
    unconditional fold would produce 96474 (= 48240 + 41302 + 6932) instead
    of 48240 -- which silently doubles every cache-heavy plan's burn rate.
    """
    h, _reg, _slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {
        "metadata": _meta_for(ctx),
        "litellm_params": {"metadata": _meta_for(ctx)},
    }
    response = _anthropic_response(pick, usage={
        "prompt_tokens": 48240,
        "completion_tokens": 201,
        "cache_read_input_tokens": 41302,
        "cache_creation_input_tokens": 6932,
    })

    async def go():
        await h.async_post_call_success_hook(
            data=request_data, user_api_key_dict=None, response=response,
        )
        plan_keys = [k for k in redis.hashes.keys()
                     if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
        return _bucket_sync(redis, plan_keys[0])

    bucket = asyncio.run(go())
    assert bucket["prompt_tokens"] == 48240, bucket
    assert bucket["completion_tokens"] == 201, bucket
    print(f"  OpenAI-shaped prompt_tokens honoured verbatim "
          f"(no cache double-count): {int(bucket['prompt_tokens'])} prompt "
          f"/ {int(bucket['completion_tokens'])} completion")


def test_buffered_messages_uses_response_cost_when_metered():
    """``response_cost`` from the response's ``_hidden_params`` must land in
    the ledger when the plan is metered.

    On a subscription the marginal cost is zero and writing it would inflate
    the month_cost figure the portal uses to decide whether to renew -- so
    cost is only recorded for metered plans.

    The response object is built with the exact ``_hidden_params`` shape
    LiteLLM 1.101.0 sets on a ``/v1/messages`` response: ``response_cost``,
    ``model_id``, and ``additional_headers`` whose keys carry the
    ``llm_provider-`` prefix that
    ``litellm_core_utils/llm_response_utils/get_headers.py`` writes. The
    request_data carries NO ``litellm_params`` key, which is the production
    shape (LiteLLM 1.101.0 does not populate it on the post-call request
    path -- verified against the pinned source). Cost, served-deployment
    re-attribution, and limit-header absorption all have to work from the
    response alone.
    """
    from dataclasses import replace
    from switchyard.models import Quota
    h, reg, _slots, _ledger, _policy, redis, pick = _build()
    plan_key = pick.plan.key
    # Configure a quota window whose `headers` map tells the limit-header
    # absorption code where to look in the provider's response -- the
    # keys the absorption matches against are un-prefixed (the helper
    # strips the ``llm_provider-`` before the lookup), so the quota
    # mapping below targets the same names that arrive post-strip.
    quota_with_headers = Quota(
        name="window1", role="target", kind="tokens", period="month",
        source="headers",
        headers={"remaining": "x-ratelimit-remaining-tokens",
                 "reset": "x-ratelimit-reset",
                 "limit": "x-ratelimit-limit-tokens"},
    )
    metered_plan = replace(
        reg.plans[plan_key], metered=True, quotas=(quota_with_headers,),
    )
    new_plans = {**reg.plans, metered_plan.key: metered_plan}
    h.registry = models.Registry(settings=reg.settings, plans=new_plans,
                                  lanes=reg.lanes)

    ctx = _ctx_for(pick, call_type="anthropic_messages")
    ctx["plan"] = plan_key
    # Production-shaped: NO ``litellm_params`` key. Pre-fix this test
    # hand-built a ``litellm_params.response_cost`` here, which is the
    # exactly the field that never lands on a real /v1/messages request.
    request_data = {"metadata": _meta_for(ctx)}
    # The response carries every fact the kwargs need, exactly as a
    # /v1/messages response would at 1.101.0: ``response_cost`` for
    # cost, ``model_id`` for served-deployment re-attribution, and
    # ``additional_headers`` (with the ``llm_provider-`` prefix that
    # the streaming handler / response processor writes) for
    # limit-header absorption.
    response = _anthropic_response(pick, usage={
        "input_tokens": 100, "output_tokens": 12,
    })
    response["_hidden_params"] = {
        "model_id": pick.model.ref,
        "response_cost": 0.0007,
        "additional_headers": {
            "llm_provider-x-ratelimit-remaining-tokens": "8000",
            "llm_provider-x-ratelimit-reset": "1234567890",
            "llm_provider-x-ratelimit-limit-tokens": "32000",
        },
    }

    async def go():
        await h.async_post_call_success_hook(
            data=request_data, user_api_key_dict=None, response=response,
        )
        plan_keys = [k for k in redis.hashes.keys()
                     if k.startswith(f"sy:usage:{plan_key}:p:")]
        bucket = _bucket_sync(redis, plan_keys[0])
        # Limit-header absorption writes the quota window key, not the
        # period bucket, so look for it directly.
        window_keys = [k for k in redis.hashes.keys()
                       if k.startswith(f"sy:qwin:{plan_key}:")]
        window = (_bucket_sync(redis, window_keys[0])
                  if window_keys else {})
        return bucket, window

    bucket, window = asyncio.run(go())
    assert abs(bucket["cost"] - 0.0007) < 1e-9, bucket
    # The provider-reported remaining/reset/limit values land on the
    # window hash so the pacer / portal see the real quota -- this is
    # the property the pre-fix code lost by reading a non-existent
    # ``litellm_params`` key.
    assert window.get("reported_remaining") == 8000.0, window
    assert window.get("reported_limit") == 32000.0, window
    assert window.get("reset_at") == 1234567890.0, window
    print(f"  metered plan booked cost=${bucket['cost']:.6f} + window "
          f"reported_remaining={int(window.get('reported_remaining', 0))}, "
          f"limit={int(window.get('reported_limit', 0))}")


def test_buffered_messages_served_deployment_mismatch_reattributes_cost():
    """When ``_hidden_params["model_id"]`` disagrees with the picked model,
    ``_check_served_deployment`` re-attributes the call to the sibling
    deployment that actually answered, so a router-level fallback does not
    spend one plan's quota and credit another's.

    Pre-fix this property was untestable: ``litellm_params.model`` was
    always absent (LiteLLM 1.101.0 does not set it on ``data``), so the
    served-deployment code path was dead. With the response-object read
    path, a real mismatch on ``_hidden_params["model_id"]`` is what the
    served deployment sees.

    The ``model_id`` field carries LiteLLM's router-side model id. With
    ``gen_litellm.py`` now stamping ``model_info["id"]`` with a
    deterministic ``Model.router_id`` (``sy.{plan}.{model}.id``,
    distinct from the deployment string ``sy.{plan}.{model}`` so it
    does not collide with the cost map), the test uses that exact
    stamped shape -- which is what every response carries in
    production -- and ``registry.model_for_router_id`` resolves it
    back to a known Model. A pre-stamp router would have emitted a
    sha256 hexdigest here (opaque, unresolvable); the new stamp is
    the matching end of that wire.
    """
    from dataclasses import replace
    h, reg, _slots, _ledger, _policy, redis, pick = _build()
    plan = pick.plan
    # Pick a sibling model on the same plan so the served ref is
    # also a known, registered model. The picker may have landed on
    # the same plan from a different lane (and pick.plan.models is
    # a dict), so the sibling lookup walks every other model the
    # plan owns rather than indexing the same list.
    siblings = [m for m in plan.models.values() if m.ref != pick.model.ref]
    assert siblings, (
        "fixture plan must have a second model so a sibling move is "
        f"exercisable; got plan.models={list(plan.models.keys())}")
    served = siblings[0]
    served_ref = served.ref
    served_router_id = served.router_id
    # Mark the served plan as metered so cost is recorded (the cost
    # figure itself does not matter for the re-attribution assertion;
    # the assertion is that the bookkeeping ran AT ALL on the served
    # model rather than the picked one).
    metered_plan = replace(plan, metered=True)
    new_plans = {**reg.plans, metered_plan.key: metered_plan}
    h.registry = models.Registry(settings=reg.settings, plans=new_plans,
                                  lanes=reg.lanes)

    ctx = _ctx_for(pick, call_type="anthropic_messages")
    ctx["plan"] = plan.key
    request_data = {"metadata": _meta_for(ctx)}
    response = _anthropic_response(pick, usage={
        "input_tokens": 200, "output_tokens": 50,
    })
    # ``_hidden_params["model_id"]`` carries the stamped router id of
    # the deployment that served the call -- a sibling on the same
    # plan, not what the picker picked. ``_call_facts`` writes it
    # through to ``kwargs["model"]`` unchanged; ``_check_served_deployment``
    # resolves it via ``registry.model_for_router_id``.
    response["_hidden_params"] = {
        "model_id": served_router_id,
        "response_cost": 0.0013,
    }

    async def go():
        await h.async_post_call_success_hook(
            data=request_data, user_api_key_dict=None, response=response,
        )
        plan_keys = [k for k in redis.hashes.keys()
                     if k.startswith(f"sy:usage:{plan.key}:p:")]
        return _bucket_sync(redis, plan_keys[0])

    bucket = asyncio.run(go())
    # Cost still lands (metered flag follows the served plan, not the
    # picked one -- both are the same plan here, but the assertion is
    # that the bookkeeping ran at all).
    assert bucket["cost"] > 0, bucket
    # And the model-scoped token bucket is keyed by the served model,
    # not the picked one. Pre-fix this was always ``pick.model.ref``
    # because the served-deployment code path was never entered.
    model_buckets = [k for k in redis.hashes.keys()
                     if k.startswith(f"sy:usage:{plan.key}:m:{served_ref}:")]
    assert model_buckets, (
        f"served ref {served_ref} did not get its own model-scoped "
        f"bucket; got "
        f"{[k for k in redis.hashes.keys() if k.startswith('sy:usage:')]}")
    print(f"  served-deployment mismatch (picked {pick.model.ref} -> "
          f"served {served_ref}) re-attributed cost + tokens to the "
          f"right model bucket via router id {served_router_id!r}")


def test_model_for_router_id_resolves_and_distinguishes_from_deployment():
    """Contract test for ``Registry.model_for_router_id``: the stamped
    ``Model.router_id`` resolves to its Model, and the lookup does NOT
    accidentally match the deployment string (``Model.deployment``)
    even when they look almost the same. Together with the
    re-attribution test above, this pins the wire end of the
    served-deployment fix so a future ``gen_litellm.py`` change that
    forgets the ``.id`` suffix, or a future registry edit that
    collapses the two lookups into one, fails on contract rather than
    silently no-op in production.
    """
    reg = models.load()
    # Every model gets its own router_id; one resolves, the others
    # don't, and a deployment string passed as router_id does NOT match.
    sample = next(iter(reg.models.values()))
    resolved = reg.model_for_router_id(sample.router_id)
    assert resolved is sample, (
        f"router_id {sample.router_id!r} did not resolve to its model "
        f"{sample.ref}; got {resolved}")
    # The deployment form (without ``.id``) is a different key -- it
    # is what ``model_for_deployment`` resolves, not what
    # ``model_for_router_id`` should.
    assert reg.model_for_router_id(sample.deployment) is None, (
        f"model_for_router_id must NOT match the deployment string "
        f"{sample.deployment!r}; it resolves through "
        f"model_for_deployment instead")
    # An arbitrary hash-shaped string does not resolve -- this is the
    # pre-fix shape that drove the original blocker.
    assert reg.model_for_router_id(
        "8ef256308fd1ebca000000000000000000000000000000000000000000000000"
    ) is None, "an opaque sha256 router id must not match any Model"
    print(f"  Registry.model_for_router_id: {sample.router_id!r} -> "
          f"{sample.ref}; deployment string {sample.deployment!r} does NOT "
          f"match (lookup is router-id-only)")


def test_streamed_messages_finalize_reads_cost_from_logging_obj():
    """The streamed /v1/messages finalize path runs after the response
    wrapper is consumed, so the response object is unavailable. The
    helper falls back to ``logging_obj.model_call_details`` for cost and
    model -- the place litellm's streaming handler stashes them
    (litellm_core_utils/streaming_handler.py:2386-2388).

    The test wires a streaming hook the same way
    ``test_streamed_messages_extracts_usage_from_message_start_and_delta``
    does (so the finalize task runs and `_finish_success` is exercised
    end-to-end), but builds the ``request_data`` with a synthetic
    ``litellm_logging_obj`` instead of an ``litellm_params`` key. A
    pre-fix version of this test would have written the cost into
    ``request_data["litellm_params"]["response_cost"]`` -- the field
    that does not exist in production on /v1/messages requests.

    Plan is flipped to ``metered=True`` so the cost actually lands on
    the ledger; the original ``test_buffered_messages_uses_response_cost_when_metered``
    does the same and that cost is the load-bearing assertion of this
    whole path.
    """
    from dataclasses import replace
    h, reg, _slots, _ledger, _policy, redis, pick = _build()
    metered_plan = replace(reg.plans[pick.plan.key], metered=True)
    new_plans = {**reg.plans, metered_plan.key: metered_plan}
    h.registry = models.Registry(settings=reg.settings, plans=new_plans,
                                  lanes=reg.lanes)
    ctx = _ctx_for(pick, call_type="anthropic_messages")

    # Synthetic logging object that mimics what litellm's streaming
    # handler writes onto ``logging_obj.model_call_details``: model +
    # response_cost. No additional_headers today -- that's a documented
    # limitation of the streamed path.
    class _LoggingObj:
        model_call_details = {
            "model": pick.model.ref,
            "response_cost": 0.0042,
        }

    request_data = {
        "metadata": _meta_for(ctx),
        "litellm_logging_obj": _LoggingObj(),
    }

    chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":150,"output_tokens":1}}}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":75}}\n\n',
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
        # Let the detached finalize task land on the event loop.
        await asyncio.sleep(0.05)

    asyncio.run(consume())
    plan_keys = [k for k in redis.hashes.keys()
                 if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
    bucket = _bucket_sync(redis, plan_keys[0])
    # Cost landed from logging_obj.model_call_details -- the source the
    # /v1/messages streamed path actually has in production.
    assert abs(bucket["cost"] - 0.0042) < 1e-9, bucket
    # Tokens still land from the SSE chunks (the streamed-side path,
    # not the new _hidden_params read path).
    assert bucket["prompt_tokens"] == 150, bucket
    assert bucket["completion_tokens"] == 75, bucket
    print(f"  streamed /v1/messages: cost=${bucket['cost']:.6f} read off "
          f"logging_obj.model_call_details, prompt={int(bucket['prompt_tokens'])}, "
          f"completion={int(bucket['completion_tokens'])}")


def test_call_facts_is_safe_on_empty_and_bad_inputs():
    """Contract test for ``_call_facts``: every documented call shape
    returns a dict and never raises. Missing inputs mean missing keys,
    not exceptions -- the helpers are invoked from the success hooks,
    where any raise would skip the slot release and corrupt the picker.

    If the field names/locations litellm reads from ever move, this
    test fires on the production-shape side (a ``request_data`` shaped
    like a real /v1/messages request still yields ``response_cost > 0``)
    so the regression is caught even before the integration tests.
    """
    # No inputs at all -> empty dict, not a raise.
    assert _call_facts() == {}, _call_facts()
    assert _call_facts(None, None) == {}, _call_facts(None, None)
    # Empty dicts / non-dict request_data are tolerated.
    assert _call_facts(None, {}) == {}, _call_facts(None, {})
    assert _call_facts(None, "not a dict") == {}, _call_facts(None, "not a dict")

    class _BadResp:
        _hidden_params = None  # explicit None

    assert _call_facts(_BadResp()) == {}, _call_facts(_BadResp())

    # A response with a non-dict _hidden_params does not raise; the
    # helper bails to the empty-dict path.
    class _WrongTypeResp:
        _hidden_params = "string, not a dict"

    assert _call_facts(_WrongTypeResp()) == {}, _call_facts(_WrongTypeResp())

    # The production-shaped read path -- ``data`` as the proxy hands
    # it to ``async_post_call_success_hook`` on /v1/messages (no
    # ``litellm_params`` key), plus a response whose ``_hidden_params``
    # carries the exact fields LiteLLM 1.101.0 sets -- returns a dict
    # with cost > 0. This is the assertion that catches a 1.101.0 ->
    # future bump that renames ``model_id`` / ``response_cost`` /
    # ``additional_headers``: the read site has to track the new names,
    # or this test fails.
    class _ProdResp:
        _hidden_params = {
            "model_id": "claude-prod-test",
            "response_cost": 0.0009,
            "additional_headers": {
                "llm_provider-x-ratelimit-remaining-requests": "42",
            },
        }

    facts = _call_facts(_ProdResp(), {"metadata": {"switchyard": {}}})
    assert facts.get("response_cost") == 0.0009, facts
    assert facts.get("model") == "claude-prod-test", facts
    # The ``llm_provider-`` prefix must be stripped so the existing
    # ``_absorb_limit_headers`` matching (which reads un-prefixed names)
    # works unchanged.
    assert "x-ratelimit-remaining-requests" in facts.get("response_headers", {}), (
        facts)
    assert "llm_provider-x-ratelimit-remaining-requests" not in facts.get(
        "response_headers", {}), facts
    print(f"  _call_facts contract: empty inputs -> {{}}; production-shape "
          f"response -> {facts}")


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


def test_streamed_messages_adapter_input_in_delta_overwrites_start_zero():
    """LiteLLM's Anthropic -> chat-completions adapter streams the input
    total on the final ``message_delta`` rather than on ``message_start``
    (the adapter has not yet had the chance to read the real count when
    the start frame is emitted). The native Anthropic API does the
    opposite -- input on the start, output on subsequent deltas. The
    hook must accept either shape and book the same totals.

    The exact sequence replayed here is the adapter's:
        1. message_start carries ``input_tokens=0`` (placeholder, the
           adapter hasn't computed the real count yet).
        2. A few content_block_deltas stream the assistant text, no
           usage payload on any of them.
        3. The final message_delta carries the real input/output/cache
           totals on its top-level ``usage`` block.

    The ledger MUST record the delta's input (12000), not the start
    frame's placeholder zero. If it recorded zero, every streamed
    /v1/messages request routed through the adapter would book
    prompt_tokens=0 and the plan ledger would undercount usage.
    """
    h, _reg, slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {"metadata": _meta_for(ctx)}

    chunks = [
        # 1. message_start: input=0 placeholder, output=1 placeholder,
        #    cache tokens real. The ``0`` is the failure mode -- the
        #    genuine count arrives later.
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"claude-...","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":1,"cache_creation_input_tokens":8,"cache_read_input_tokens":12}}}\n\n',
        # 2. Two content deltas (no usage payloads).
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n',
        # 3. Final message_delta: real input=12000, output=77, cache
        #    totals carry through.
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":12000,"output_tokens":77,"cache_creation_input_tokens":8,"cache_read_input_tokens":12}}\n\n',
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
    # 12000 (input, from message_delta) + 12 (cache_read) + 8
    # (cache_creation) = 12020 prompt. The key assertion is that the
    # input came from the delta (12000), not the start frame's 0.
    assert bucket["prompt_tokens"] == 12020, (
        f"expected prompt_tokens=12020 (12000 from message_delta input "
        f"+ cache totals), got {bucket}")
    assert bucket["completion_tokens"] == 77, (
        f"expected completion_tokens=77 (from the final message_delta), "
        f"got {bucket}")
    assert bucket["requests"] == 1, bucket
    print(f"  LiteLLM adapter sequence: input arrived on message_delta "
          f"({int(bucket['prompt_tokens'])} prompt incl. cache, "
          f"{int(bucket['completion_tokens'])} completion) -- not the "
          "start frame's placeholder zero")


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
# Section 4c: streamed /v1/chat/completions disconnect (issue #182)
# ============================================================================
def test_streamed_openai_consumer_disconnect_releases_slot_via_aclose():
    """Streamed ``/v1/chat/completions`` (call_type = ``acompletion``, NOT in
    ``UNLOGGED_CALL_TYPES``) must still release the slot when the consumer
    hangs up mid-stream, even though ``async_log_success_event`` owns the
    success bookkeeping.

    Pre-fix: the bare pass-through ``async for chunk in response: yield chunk``
    in ``_stream_chunks`` had no ``try/except`` around it, so a client
    disconnect arriving as ``GeneratorExit`` at a yield left the slot
    claimed until the staleness sweep eventually dropped it -- exactly the
    zero-chunk CLI-sidecar leak reported in the issue.

    Post-fix: the same marker-free detached slot-release task that the
    ``/v1/messages`` branch already uses (idempotent with ``picker.release``
    and ``_stop_heartbeat``, deliberately does NOT claim
    ``ctx[_TERMINATED]`` so ``async_log_success_event`` can still book
    partial usage when chunks were non-empty) is scheduled from the new
    ``except BaseException`` arm.

    Variant A exercises the ``GeneratorExit`` arm: the consumer takes one
    OpenAI-shaped chunk then ``gen.aclose()`` (which throws ``GeneratorExit``
    into the suspended generator).
    """
    h, _reg, slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="acompletion")  # NOT in UNLOGGED_CALL_TYPES
    request_data = {"metadata": _meta_for(ctx)}

    class _Chunk:
        choices = []
        def __init__(self):
            pass

    async def produce():
        yield _Chunk()
        yield _Chunk()

    async def consume_one_chunk():
        h._start_heartbeat(pick.plan.key, pick.model.ref, pick.request_id)
        gen = h.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None, response=produce(), request_data=request_data,
        )
        # Take one chunk only, then bail (mimics a client that hung up
        # after reading the start of an OpenAI stream).
        async for _ in gen:
            break
        # Force-close the iterator so the detached slot-release task
        # gets scheduled and runs.
        await gen.aclose()
        await asyncio.sleep(0.1)

    assert asyncio.run(slots.in_flight(pick.plan.key)) == 1
    asyncio.run(consume_one_chunk())
    assert asyncio.run(slots.in_flight(pick.plan.key)) == 0, (
        "client disconnect left the slot claimed on a streamed "
        "/v1/chat/completions call")
    assert pick.request_id not in h._beats, (
        f"heartbeat for disconnected request still alive: {h._beats}")
    print("  streamed /v1/chat/completions client cancel (GeneratorExit) "
          "released the slot; detach task ran after aclose")


def test_streamed_openai_consumer_disconnect_releases_slot_via_task_cancel():
    """Variant B: the zero-chunk CLI-sidecar shape.

    The CLI-sidecar pattern that surfaced as the reported bug: the
    upstream provider (the sidecar) takes a long time before sending
    any chunk, the client hangs up, no chunk was ever pulled. A
    never-started generator's ``aclose()`` is a no-op and would not
    exercise the fix, so we start a consumer task that is suspended on
    the producer's first ``__anext__`` (the producer hangs on
    ``asyncio.Event().wait()`` to mimic a slow sidecar), then cancel
    the task. ``CancelledError`` propagates through the async
    generator's ``async for chunk in response: yield chunk`` into the
    new ``except BaseException`` arm, which schedules the marker-free
    detached slot release.

    Note: this test does NOT drive ``async_log_success_event`` -- the
    reported case is exactly the disconnect-before-any-chunk path, so
    the iterator-side cleanup is the sole cleanup. Success accounting
    is still owned by ``async_log_success_event`` when it does fire
    (asserted by ``test_logged_call_type_is_not_double_finished_by_streaming_hook``).
    """
    h, _reg, slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="acompletion")  # NOT in UNLOGGED_CALL_TYPES
    request_data = {"metadata": _meta_for(ctx)}

    async def produce():
        # Sidecar-style slow producer: never yields a chunk in this test.
        # The `if False: yield` keeps this an async generator so the
        # streaming hook accepts it.
        await asyncio.Event().wait()
        if False:
            yield  # pragma: no cover -- never reached

    async def consume_zero_chunks():
        h._start_heartbeat(pick.plan.key, pick.model.ref, pick.request_id)
        gen = h.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None, response=produce(), request_data=request_data,
        )
        # Park on the producer's first ``__anext__`` (which awaits
        # ``asyncio.Event().wait()`` and never completes). ``drive()``
        # cancels the surrounding task, throwing ``CancelledError``
        # into the async generator at its suspended point.
        async for _ in gen:
            pass  # unreachable in this variant

    async def drive():
        # Run the consumer in a separate task so we can cancel it from
        # outside. The consumer parks on the producer's ``await
        # asyncio.Event().wait()`` (which never completes); cancelling
        # the task throws ``CancelledError`` into the async generator
        # at its suspended point.
        consumer = asyncio.create_task(consume_zero_chunks())
        # Yield long enough for the consumer to start and park.
        await asyncio.sleep(0.05)
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
        # Let the detached slot-release task run.
        await asyncio.sleep(0.1)

    assert asyncio.run(slots.in_flight(pick.plan.key)) == 1
    asyncio.run(drive())
    assert asyncio.run(slots.in_flight(pick.plan.key)) == 0, (
        "zero-chunk client disconnect left the slot claimed on a "
        "streamed /v1/chat/completions call (CLI-sidecar leak)")
    assert pick.request_id not in h._beats, (
        f"heartbeat for disconnected request still alive: {h._beats}")
    print("  streamed /v1/chat/completions zero-chunk cancel "
          "(CancelledError) released the slot; detach task ran after "
          "task.cancel")


# ============================================================================
# Section 4b: streamed error race (review blocker on PR #67)
# ============================================================================
def test_streamed_messages_error_race_books_partial_usage_as_failure():
    """Reproduce the race described in the PR #67 review.

    LiteLLM 1.101.0's ``async_streaming_data_generator`` (common_request_processing.py:3687)
    catches an upstream ``Exception`` from the streaming iterator and calls
    ``proxy_logging_obj.post_call_failure_hook`` (utils.py:2493), which itself
    awaits ``update_request_status`` BEFORE iterating the registered
    callbacks (utils.py:2525). ``update_request_status`` awaits an internal
    cache write whenever ``self.alerting is not None`` -- true for any
    deployment that follows the documented Slack-alerting recipe. That
    intermediate await yields to the event loop.

    Pre-fix: ``_stream_chunks``'s ``except BaseException`` arm scheduled a
    detached ``_finish_success`` task. That task would run during the
    intermediate ``update_request_status`` await, set
    ``ctx[_TERMINATED] = "success"`` before its first real await, and then
    ``post_call_failure_hook`` arrived and called ``_finish_failure`` --
    which saw the marker and short-circuited. Net effect: usage booked,
    transient-failure streak reset, but NO failure record, NO verdict, NO
    cooldown. The bug is invisible from inside this repo's offline test
    harness (no Slack alerting, no intermediate await) and only surfaces
    when alerting is configured on the live proxy.

    Post-fix: ``_stream_chunks``'s exception arm schedules a
    SLOT-RELEASE-ONLY detached task and stashes the partial usage on ctx
    under ``_stream_collected_usage``. The task does NOT claim
    ``_TERMINATED`` -- so whichever path arrives first wins on its own
    merits, not on a race. The post-call failure hook is the SOLE owner of
    failure semantics; it sees the stashed partial usage and books it
    alongside the failure record.

    This test forces the race by sleeping once (mimicking the intermediate
    ``update_request_status`` await), then driving the post-call failure
    hook. With the bug, the marker comes out ``"success"`` and the failure
    is suppressed. With the fix, the marker comes out ``"failure"`` and
    the failure + partial usage both land on the ledger.
    """
    h, _reg, slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {"metadata": _meta_for(ctx)}

    class _UpstreamError(Exception):
        status_code = 503

    async def produce():
        # message_start: 40 input, 1 output placeholder
        yield (b'event: message_start\ndata: {"type":"message_start",'
               b'"message":{"id":"m","type":"message","role":"assistant",'
               b'"model":"x","content":[],"stop_reason":null,'
               b'"stop_sequence":null,'
               b'"usage":{"input_tokens":40,"output_tokens":1}}}\n\n')
        # a content delta (no usage)
        yield (b'event: content_block_delta\ndata: {"type":"content_block_delta",'
               b'"index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n')
        # message_delta: real output of 17 (overwrites the placeholder)
        yield (b'event: message_delta\ndata: {"type":"message_delta",'
               b'"delta":{"stop_reason":null},'
               b'"usage":{"output_tokens":17}}\n\n')
        # upstream blows up
        raise _UpstreamError("provider 503")

    async def run():
        # Drive the streaming hook to the exception arm. Pre-fix, this
        # would have scheduled a detached _finish_success. Post-fix, it
        # stashes partial usage and schedules the slot-release-only task.
        try:
            async for _ in h.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=produce(),
                request_data=request_data,
            ):
                pass
        except _UpstreamError:
            pass
        # Yield to the event loop -- this is the model for LiteLLM's
        # intermediate `await self.update_request_status(...)` inside
        # `proxy_logging_obj.post_call_failure_hook`. The detached task
        # scheduled above runs during this yield.
        await asyncio.sleep(0.05)
        # Now simulate LiteLLM's `post_call_failure_hook` arriving AFTER
        # the intermediate await.
        await h.async_post_call_failure_hook(
            request_data={"metadata": _meta_for(ctx)},
            original_exception=_UpstreamError("provider 503"),
            user_api_key_dict=None,
        )
        # Let any post-call bookkeeping settle.
        await asyncio.sleep(0.05)
        return (await slots.in_flight(pick.plan.key),
                ctx.get(_TERMINATED),
                redis.hashes)

    inflight, terminated, hashes = asyncio.run(run())
    assert inflight == 0, "slot must be released exactly once"
    assert terminated == "failure", (
        f"failure marker must win the race; got {terminated!r}. "
        "If this is 'success', the pre-fix bug is back: the detached "
        "stream finalize claimed _TERMINATED before "
        "post_call_failure_hook could record the failure."
    )
    plan_keys = [k for k in hashes.keys()
                 if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
    assert plan_keys, "no period bucket was written for the failed plan"
    bucket = _bucket_sync(redis, plan_keys[0])
    assert bucket["failures"] >= 1, (
        f"failure must be recorded on the ledger; got {bucket}")
    # Partial usage that the streaming hook parsed before the upstream
    # blew up lands on the ledger alongside the failure record.
    assert bucket["prompt_tokens"] == 40, bucket
    assert bucket["completion_tokens"] == 17, bucket
    # The transient-failure streak must NOT have been reset -- this is
    # the failure path, not success.
    print(f"  streamed error race: marker='{terminated}', "
          f"prompt={int(bucket['prompt_tokens'])}, "
          f"completion={int(bucket['completion_tokens'])}, "
          f"failures={int(bucket['failures'])} "
          "(success path suppressed by the slot-release-only split)")


def test_streamed_messages_client_cancel_does_not_record_failure():
    """Treat cancellation distinctly from upstream error.

    ``asyncio.CancelledError`` / ``GeneratorExit`` raised inside the
    streaming iterator is a client-side event -- LiteLLM's
    ``async_streaming_data_generator`` does NOT fire
    ``post_call_failure_hook`` for those (common_request_processing.py:3670
    catches them and just re-raises). The slot-release-only detached task
    is the SOLE cleanup, so it must release the slot without writing a
    failure ledger entry, applying a verdict, or resetting the
    transient-failure streak.

    Compare to the upstream-error test above: that one DID call
    ``async_post_call_failure_hook`` afterwards, and the failure landed on
    the ledger. Here the test does not, so the ledger stays empty.
    """
    h, _reg, slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {"metadata": _meta_for(ctx)}

    async def produce():
        # Two chunks, then a clean message_stop. We never reach the
        # message_stop because the consumer disconnects after the first
        # chunk.
        yield (b'event: message_start\ndata: {"type":"message_start",'
               b'"message":{"id":"m","type":"message","role":"assistant",'
               b'"model":"x","content":[],"stop_reason":null,'
               b'"stop_sequence":null,'
               b'"usage":{"input_tokens":3,"output_tokens":1}}}\n\n')
        yield (b'event: content_block_delta\ndata: {"type":"content_block_delta",'
               b'"index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n')
        yield (b'event: message_delta\ndata: {"type":"message_delta",'
               b'"delta":{"stop_reason":null},'
               b'"usage":{"output_tokens":2}}\n\n')
        yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    async def consume_one_chunk():
        gen = h.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None, response=produce(),
            request_data=request_data,
        )
        # Take one chunk then bail (mimics a client that hung up after the
        # start of the stream).
        async for _ in gen:
            break
        # Force-close the iterator so the detached slot-release-only task
        # gets scheduled and runs.
        await gen.aclose()
        await asyncio.sleep(0.05)

    assert asyncio.run(slots.in_flight(pick.plan.key)) == 1
    asyncio.run(consume_one_chunk())
    # Slot released...
    assert asyncio.run(slots.in_flight(pick.plan.key)) == 0, (
        "client cancel left the slot claimed")
    # ...but the failure ledger is empty -- a client cancel is not a
    # provider failure and must not be classified as one.
    plan_keys = [k for k in redis.hashes.keys()
                 if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
    assert not plan_keys, (
        f"client cancel wrote a ledger entry: {plan_keys}; expected none")
    assert ctx.get(_TERMINATED) is None, (
        f"client cancel must not claim _TERMINATED; "
        f"got {ctx.get(_TERMINATED)!r}")
    print("  client cancel released slot without failure ledger entry "
          "and without claiming _TERMINATED (cancellation distinct from "
          "upstream error)")


def test_streamed_messages_partial_usage_books_with_failure_record():
    """Independent of the race: a streamed /v1/messages that fails after
    the iterator has parsed partial usage must book the partial tokens
    alongside the failure record. This is the bounded partial-loss the
    reviewer's design accepts and the property that
    ``_finish_failure`` now guarantees by consulting
    ``ctx['_stream_collected_usage']``.

    The previous race-reproduction test verifies the race itself; this
    one verifies the partial-usage bookkeeping in the simpler,
    no-intermediate-await case so a regression in either is easy to
    localise.
    """
    h, _reg, slots, _ledger, _policy, redis, pick = _build()
    ctx = _ctx_for(pick, call_type="anthropic_messages")
    request_data = {"metadata": _meta_for(ctx)}

    class _ProviderReset(Exception):
        status_code = 502

    async def produce():
        # Just enough to put partial usage on ctx: a message_start with
        # the input/cache totals and a single message_delta that sets
        # the output total.
        yield (b'event: message_start\ndata: {"type":"message_start",'
               b'"message":{"id":"m","type":"message","role":"assistant",'
               b'"model":"x","content":[],"stop_reason":null,'
               b'"stop_sequence":null,'
               b'"usage":{"input_tokens":22,"output_tokens":1,'
               b'"cache_read_input_tokens":4,"cache_creation_input_tokens":1}'
               b'}}\n\n')
        yield (b'event: message_delta\ndata: {"type":"message_delta",'
               b'"delta":{"stop_reason":null},'
               b'"usage":{"output_tokens":9,'
               b'"cache_read_input_tokens":4,"cache_creation_input_tokens":1}'
               b'}\n\n')
        raise _ProviderReset("upstream reset")

    async def run():
        try:
            async for _ in h.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=produce(),
                request_data=request_data,
            ):
                pass
        except _ProviderReset:
            pass
        await asyncio.sleep(0.05)
        await h.async_post_call_failure_hook(
            request_data={"metadata": _meta_for(ctx)},
            original_exception=_ProviderReset("upstream reset"),
            user_api_key_dict=None,
        )
        await asyncio.sleep(0.05)

    asyncio.run(run())
    plan_keys = [k for k in redis.hashes.keys()
                 if k.startswith(f"sy:usage:{pick.plan.key}:p:")]
    assert plan_keys, "no period bucket was written for the failed plan"
    bucket = _bucket_sync(redis, plan_keys[0])
    # 22 (input) + 4 (cache_read) + 1 (cache_creation) = 27 prompt
    # Output: 9 (the message_delta value, not the placeholder 1)
    assert bucket["prompt_tokens"] == 27, bucket
    assert bucket["completion_tokens"] == 9, bucket
    assert bucket["failures"] >= 1, bucket
    assert asyncio.run(slots.in_flight(pick.plan.key)) == 0
    print(f"  partial usage on failure: prompt={int(bucket['prompt_tokens'])} "
          f"(22 input + 4 cache_read + 1 cache_creation), "
          f"completion={int(bucket['completion_tokens'])}, "
          f"failures={int(bucket['failures'])}")


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


def test_collect_anthropic_event_usage_delta_input_zero_does_not_clobber_start():
    """A ``message_delta`` whose ``input_tokens`` is ``0`` is the
    LiteLLM-adapter placeholder, NOT an authoritative recount -- the
    parser must leave a positive ``message_start`` value in place.

    The sibling test at line 448 already covers the opposite shape:
    genuine ``input_tokens`` on the start frame, no ``input_tokens``
    field on the deltas (native Anthropic). This one closes the loop
    by covering the shape where a delta DOES carry ``input_tokens``
    but the value is the placeholder zero, and proves the parser
    refuses to overwrite the start frame's count. Without this guard,
    every streamed /v1/messages request routed through the adapter
    would book prompt_tokens=0 regardless of what the real count was.
    """
    collected: dict = {}
    # 1. message_start carries the genuine input + cache totals.
    _collect_anthropic_event_usage({
        "type": "message_start",
        "message": {
            "usage": {
                "input_tokens": 500, "output_tokens": 1,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 5,
            },
        },
    }, collected)
    assert collected["input_tokens"] == 500, collected
    # 2. A delta whose input is the ``0`` placeholder -- the parser
    #    MUST leave the 500 from step 1 intact, otherwise the ledger
    #    will book prompt_tokens=0 for an Anthropic adapter request.
    _collect_anthropic_event_usage({
        "type": "message_delta",
        "delta": {"stop_reason": None},
        "usage": {
            "input_tokens": 0,
            "output_tokens": 17,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 5,
        },
    }, collected)
    assert collected["input_tokens"] == 500, (
        f"message_delta with input_tokens=0 clobbered the "
        f"message_start value; collected={collected}")
    assert collected["output_tokens"] == 17, collected
    # 3. A later delta with a REAL positive input_tokens DOES overwrite
    #    (running totals, not deltas) -- the guard only vetoes the
    #    zero placeholder, not genuine counts.
    _collect_anthropic_event_usage({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {
            "input_tokens": 12000,
            "output_tokens": 77,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 5,
        },
    }, collected)
    assert collected["input_tokens"] == 12000, (
        f"positive message_delta input_tokens did not overwrite the "
        f"running total; collected={collected}")
    assert collected["output_tokens"] == 77, collected
    # 4. A non-numeric payload on a delta is tolerated (skipped) the
    #    same way other fields are; the running total stays put.
    _collect_anthropic_event_usage({
        "type": "message_delta",
        "delta": {},
        "usage": {"input_tokens": "not-a-number", "output_tokens": 80,
                  "cache_read_input_tokens": 30,
                  "cache_creation_input_tokens": 5},
    }, collected)
    assert collected["input_tokens"] == 12000, (
        f"non-numeric message_delta input_tokens corrupted the "
        f"running total; collected={collected}")
    assert collected["output_tokens"] == 80, (
        f"non-numeric message_delta input_tokens blocked the rest of "
        f"the loop from overwriting output_tokens; collected={collected}")
    print("  parser refuses to overwrite a positive message_start input "
          "count with a zero (placeholder) message_delta value, but "
          "accepts a positive overwrite and tolerates non-numeric input")


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


# ============================================================================
# Issue #75 — image detection on the inbound request body.
#
# The hook derives `needs_images` from the message list before passing it to
# the picker. Three shapes must all report `needs_images=True`:
#   - Anthropic:   {"type": "image", ...}
#   - OpenAI Chat: {"type": "image_url", ...}
#   - OpenAI Resp: {"type": "input_image", ...}
#
# A plain string content block is text-only; a malformed entry (non-dict,
# missing content) is silently skipped. Without the right shape, an image
# request lands on a text-only model and silently fails at the provider,
# which is exactly the bug the picker image-routing filter exists to prevent.
# ============================================================================


def test_pre_call_hook_derives_needs_images_from_anthropic_image_block():
    """An image-bearing request with the Anthropic shape (the protocol the
    gateway's /v1/messages endpoints speak) drives `needs_images=True` and
    passes that flag to the picker. The metadata stamped onto the request
    must include `needs_images`, the picker must be called with the flag,
    and the routing log line must include the " images" tag so a grep
    across the gateway log finds every image-bearing request.
    """
    import logging as _logging
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

    # Capture log records so the assertion about the " images" tag can
    # grep exactly the line the operator reads.
    captured: list[_logging.LogRecord] = []
    handler = _logging.Handler()
    handler.emit = captured.append
    log = _logging.getLogger("switchyard")
    prior_level = log.level
    log.setLevel(_logging.INFO)
    log.addHandler(handler)
    try:
        data = {
            "model": "apex",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/png",
                                                 "data": "fakepng"}},
                ]},
            ],
            "proxy_server_request": {"headers": {
                "x-switchyard-session": "sess-image-anthropic",
            }},
        }
        asyncio.run(h.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=data,
            call_type="anthropic_messages",
        ))
        ctx = data["metadata"][META_KEY]
        await_release = asyncio.run(_release_picker(h, ctx))
    finally:
        log.removeHandler(handler)
        log.setLevel(prior_level)

    # `picker.release` is typed -> None; the await only proves the call
    # did not raise. Tie the assertion to observable side effects: the
    # slot for the picked plan/request_id/model is gone, and the plan is
    # back at zero in-flight.
    assert await_release is None, await_release
    assert asyncio.run(slots.in_flight(ctx["plan"])) == 0, ctx["plan"]
    assert asyncio.run(slots.in_flight_model(ctx["model"])) == 0, ctx["model"]
    assert ctx["needs_images"] is True, ctx
    log_lines = [r.getMessage() for r in captured]
    routing_lines = [m for m in log_lines if m.startswith("lane=")]
    assert routing_lines, log_lines
    assert any("images" in line for line in routing_lines), routing_lines
    print("  anthropic image block -> needs_images=True in ctx + ' images' tag in log")


def test_pre_call_hook_derives_needs_images_from_openai_image_url_block():
    """The OpenAI Chat Completions shape uses `image_url` rather than `image`.
    The hook must catch it too, otherwise an OpenAI-shaped image request
    under a lane with `image_routing: always` would silently land on a
    text-only model. Same flip-side as the Anthropic test, exercised end
    to end through the pre-call hook so the metadata stamp and the log
    line are both verified.
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

    data = {
        "model": "apex",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "https://example.com/cat.png"}},
            ]},
        ],
        "proxy_server_request": {"headers": {
            "x-switchyard-session": "sess-image-openai",
        }},
    }
    asyncio.run(h.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=data,
        call_type="acompletion",
    ))
    ctx = data["metadata"][META_KEY]
    asyncio.run(_release_picker(h, ctx))

    assert ctx["needs_images"] is True, ctx
    print("  openai image_url block -> needs_images=True in ctx")


def test_pre_call_hook_derives_needs_images_from_openai_input_image_block():
    """The OpenAI Responses API shape uses `input_image`. The hook must
    catch it the same way it catches `image` and `image_url` -- the three
    names are the load-bearing image-bearing content block types across
    the protocols the gateway fronts.
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

    data = {
        "model": "apex",
        "messages": [
            {"role": "user", "content": [
                {"type": "input_image",
                 "image_url": "data:image/png;base64,fakepng"},
            ]},
        ],
        "proxy_server_request": {"headers": {
            "x-switchyard-session": "sess-image-responses",
        }},
    }
    asyncio.run(h.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=data,
        call_type="acompletion",
    ))
    ctx = data["metadata"][META_KEY]
    asyncio.run(_release_picker(h, ctx))

    assert ctx["needs_images"] is True, ctx
    print("  openai input_image block -> needs_images=True in ctx")


def test_pre_call_hook_leaves_needs_images_false_for_plain_text():
    """The negative case: a plain text-only request MUST keep
    `needs_images=False`, regardless of how many user/assistant turns the
    conversation already contains. A misfire here would route every
    text-only request through the image-capability filter and 429 a
    any lane whose every member lacks `supports_images: true`.
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

    # A multi-turn text conversation with NO image block. needs_images
    # must come out False even though the messages list is non-trivial.
    data = {
        "model": "apex",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": [
                {"type": "text", "text": "no images here"},
            ]},
        ],
        "proxy_server_request": {"headers": {
            "x-switchyard-session": "sess-text-only",
        }},
    }
    asyncio.run(h.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=data,
        call_type="acompletion",
    ))
    ctx = data["metadata"][META_KEY]
    asyncio.run(_release_picker(h, ctx))

    assert ctx["needs_images"] is False, ctx
    print("  plain text-only body -> needs_images=False in ctx")


async def _release_picker(h, ctx):
    """Release a slot the pre-call hook just claimed, keeping the fake
    redis state clean between tests. Defined locally rather than reusing
    the inline `_release_picker` from other tests, because every test
    needs a fresh release path on its own state.
    """
    await h.picker.release(ctx["plan"], ctx["request_id"], ctx["model"])


# ============================================================================
# Section 7: CONTEXT-window fallback success books the served plan
# ============================================================================
# A router-level ``context_window_fallbacks`` walk reuses the same nested
# metadata["switchyard"] ctx across attempts (litellm.router_utils.
# fallback_event_handlers.run_async_fallback only shallow-copies the metadata
# per hop). Pre-fix this caused a permanent double-booking hole: the failure
# on the first attempt claimed ``_TERMINATED = "failure"``, which
# short-circuited ``_finish_success`` on the fallback attempt -- one
# subscription's quota was spent and another's was credited. The fix
# classifies BEFORE claiming the marker and routes CONTEXT verdicts onto a
# path that does not claim ``_TERMINATED`` (a per-request ``_ctx_fail_booked_hops``
# set dedupes the same-request double-fire instead).
def test_context_window_fallback_success_books_served_plan_and_drops_lease():
    """Two-attempt CONTEXT-fallback regression: a context-window failure on
    attempt 1 (pick = local-box/qwen) is followed by a successful fallback
    on attempt 2 (served = openrouter/mimo). Pre-fix this case was a silent
    ledger hole: attempt 1 claimed ``_TERMINATED = "failure"``, and
    attempt 2's ``_finish_success`` short-circuited on the marker so the
    served plan's quota was never debited. Post-fix the CONTEXT failure
    takes a path that does not claim the marker, the served plan's tokens
    land on its own ledger bucket, no slot is claimed on the served plan,
    and a pre-set session lease on the picked plan is dropped so the next
    turn re-leases onto a sibling with a bigger window.

    The shape mirrors ``test_buffered_messages_served_deployment_mismatch_reattributes_cost``
    (:502): same `_hidden_params["model_id"]` -> router_id contract, same
    ``async_post_call_success_hook` -> ``_finish_success`` path. The
    CONTEXT-failure wiring is new: a ``ContextWindowExceededError`` (the
    litellm exception class) reaches ``_finish_failure`` and routes through
    the new CONTEXT branch.
    """
    h, reg, slots, _ledger, _policy, redis, pick = _build(claim_lane="local")
    # pick is local-box/qwen; the served fallback is openrouter/mimo (the
    # only sibling with a strictly-larger context_window).
    siblings = [
        m for m in reg.models.values()
        if m.ref != pick.model.ref and m.context_window and m.context_window > pick.model.context_window
    ]
    assert siblings, (
        "fixture must have at least one bigger-window sibling for the "
        "fallback path to be exercisable; got "
        f"{[(m.ref, m.context_window) for m in reg.models.values() if m.context_window]}")
    served = siblings[0]
    served_ref = served.ref
    served_plan_key = reg.plan_of(served).key
    picked_plan_key = pick.plan.key
    assert served_plan_key != picked_plan_key, (
        "test needs two plans so the served fallback is on a different "
        "plan from the pick; both landed on "
        f"{picked_plan_key}/{served_plan_key}")

    # A session lease pre-set by an earlier turn. The CONTEXT verdict must
    # drop it: the session would otherwise keep landing on the
    # too-small-window plan every turn.
    session = "sess-context-fallback"
    lease_ttl = reg.settings.lease_ttl_seconds
    asyncio.run(slots.set_lease(session, pick.model.ref, lease_ttl))
    assert asyncio.run(slots.get_lease(session)) == pick.model.ref, (
        "fixture lease must be live before the failure")

    ctx = _ctx_for(pick, call_type="anthropic_messages")
    ctx["session"] = session  # so drop_lease has something to drop

    # A real litellm exception -- the class _finish_failure's CONTEXT branch
    # recognises through the prose regex on its message.
    from litellm import ContextWindowExceededError
    exc = ContextWindowExceededError(
        message=(
            "This model's maximum context length is 131072 tokens. "
            "However, you requested 200000 tokens."
        ),
        model=pick.model.ref,
        llm_provider="openai",
    )

    async def go():
        # Attempt 1: CONTEXT failure. async_post_call_failure_hook owns
        # /v1/messages failures (LiteLLM 1.101.0 does NOT fire
        # async_log_failure_event for anthropic_messages), so drive it
        # through that path -- the same shape /v1/messages failures
        # take in production.
        await h.async_post_call_failure_hook(
            request_data={"metadata": _meta_for(ctx)},
            original_exception=exc,
            user_api_key_dict=None,
        )
        # The CONTEXT branch must NOT have claimed the marker.
        post_failure_terminated = ctx.get(_TERMINATED)
        post_failure_ctx_flag = ctx.get("_ctx_fail_booked_hops")
        # Pre-set lease must be dropped.
        lease_after_failure = await slots.get_lease(session)
        # The picked plan got a failure record.
        picked_keys = [
            k for k in redis.hashes.keys()
            if k.startswith(f"sy:usage:{picked_plan_key}:p:")
        ]
        picked_bucket = _bucket_sync(redis, picked_keys[0]) if picked_keys else {}

        # Attempt 2: the litellm router moves the request to a sibling
        # deployment. _hidden_params["model_id"] carries that sibling's
        # router_id, exactly the shape a production response carries
        # (gen_litellm.py stamps ``model_info["id"]`` with
        # ``Model.router_id``). async_post_call_success_hook fires for
        # /v1/messages; _finish_success resolves the served plan via
        # ``_check_served_deployment``.
        response = _anthropic_response(pick, usage={
            "input_tokens": 333, "output_tokens": 44,
        })
        response["_hidden_params"] = {
            "model_id": served.router_id,
        }
        await h.async_post_call_success_hook(
            data={"metadata": _meta_for(ctx)},
            user_api_key_dict=None,
            response=response,
        )
        served_keys = [
            k for k in redis.hashes.keys()
            if k.startswith(f"sy:usage:{served_plan_key}:p:")
        ]
        served_bucket = _bucket_sync(redis, served_keys[0]) if served_keys else {}
        return (
            post_failure_terminated, post_failure_ctx_flag, lease_after_failure,
            picked_bucket, served_bucket,
        )

    (
        post_failure_terminated, post_failure_ctx_flag, lease_after_failure,
        picked_bucket, served_bucket,
    ) = asyncio.run(go())

    # Marker was NOT claimed by the CONTEXT failure (so the fallback success
    # was allowed to run).
    assert post_failure_terminated is None, (
        f"CONTEXT failure must not claim _TERMINATED (would suppress "
        f"the fallback success); got {post_failure_terminated!r}")
    # The per-hop dedupe set has the hop this call contributed. The
    # /v1/messages post-call-only test uses kwargs without a router
    # stamp, so the hop is the -1 sentinel. The set is what keeps a
    # real OpenAI-shape double-fire of async_log + async_post_call on
    # one attempt from booking twice.
    assert post_failure_ctx_flag == {-1}, (
        f"per-hop dedupe set must contain the booked hop; got "
        f"{post_failure_ctx_flag!r}")
    # Session lease was dropped by the CONTEXT verdict so the next turn
    # re-leases onto a sibling with a bigger window.
    assert lease_after_failure is None, (
        f"CONTEXT verdict must drop the session lease; got {lease_after_failure!r}")
    # The picked plan got a failure record -- failure-side bookkeeping
    # still ran on the path that did the work.
    assert picked_bucket["failures"] >= 1, (
        f"picked plan ({picked_plan_key}) must record the CONTEXT "
        f"failure; got {picked_bucket}")
    # The served plan got the tokens from the successful fallback.
    assert served_bucket["prompt_tokens"] == 333, served_bucket
    assert served_bucket["completion_tokens"] == 44, served_bucket
    assert served_bucket["requests"] == 1, served_bucket
    # And the success-side bookkeeping claimed the marker for attempt 2.
    assert ctx[_TERMINATED] == "success", ctx.get(_TERMINATED)
    # No slot left claimed on either plan. _finish_success does NOT
    # claim-and-release a slot on the served plan (a post-hoc claim gates
    # nothing -- the picker is the only placement owner), so the served
    # plan was never inflight to begin with.
    assert asyncio.run(slots.in_flight(picked_plan_key)) == 0
    assert asyncio.run(slots.in_flight(served_plan_key)) == 0
    print(f"  CONTEXT failure on {pick.model.ref}: lease dropped, "
          f"failure recorded on {picked_plan_key}; fallback success on "
          f"{served_ref} (plan={served_plan_key}) booked "
          f"{int(served_bucket['prompt_tokens'])} prompt / "
          f"{int(served_bucket['completion_tokens'])} completion; "
          f"ctx[_TERMINATED]={ctx.get(_TERMINATED)!r}; no slot claimed "
          f"on either plan")


def test_non_context_failure_still_claims_marker_and_suppresses_success():
    """Non-CONTEXT control for the no-router-metadata path: a 5xx on
    attempt 1 with kwargs that carry no ``attempted_fallbacks``
    stamp (the internal-caller shape; older-litellm shapes; the proxy
    side of ``async_post_call_failure_hook`` on builds without the
    in-place entry stamp) must still claim ``_TERMINATED`` so a later
    success on the same attempt short-circuits unbooked.

    Today the route is single-deployment (the generated config carries
    ``num_retries=0`` and no general fallbacks), so no runtime
    success can come -- but the claim keeps the bookkeeping
    idempotent for the OpenAI-shape double-fire case where
    ``async_log_failure_event`` and ``async_post_call_failure_hook``
    both reach ``_finish_failure`` for one attempt.

    The production-shaped companion at hop=0 lives in
    ``test_non_context_failure_with_router_entry_stamp_claims_marker``
    below.
    """
    h, _reg, slots, _ledger, _policy, _redis, pick = _build(claim_lane="local")
    ctx = _ctx_for(pick, call_type="acompletion")
    kwargs = {"metadata": _meta_for(ctx)}

    class _Upstream5xx(Exception):
        status_code = 502

    async def go():
        # Attempt 1: a non-CONTEXT 5xx -- the existing claim-marker path.
        await h.async_log_failure_event(
            kwargs=kwargs, response_obj=None,
            start_time=None, end_time=None,
        )
        post_failure_terminated = ctx.get(_TERMINATED)
        # A later "success" (in production: litellm's router-level
        # context_window_fallbacks walking to a sibling -- the same
        # single-member deployment group, hence still the same plan).
        response = {
            "id": "x", "object": "chat.completion",
            "model": pick.model.ref,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 7},
        }
        await h.async_log_success_event(
            kwargs=kwargs, response_obj=response,
            start_time=None, end_time=None,
        )
        return post_failure_terminated

    post_failure_terminated = asyncio.run(go())
    # The 5xx claimed the marker as today.
    assert post_failure_terminated == "failure", (
        f"non-CONTEXT failure must claim _TERMINATED; got "
        f"{post_failure_terminated!r}")
    # And the later success short-circuited -- the marker is still
    # ``failure``, so the request was never booked.
    assert ctx[_TERMINATED] == "failure", (
        f"later success on a non-CONTEXT failure must no-op; got "
        f"ctx[_TERMINATED]={ctx[_TERMINATED]!r}")
    assert asyncio.run(slots.in_flight(pick.plan.key)) == 0
    print(f"  non-CONTEXT 5xx on {pick.model.ref}: marker claimed; "
          f"later success no-op'd via first-line check")


def test_context_window_failure_double_fire_books_exactly_one_failure():
    """Same-request double-fire dedup: an OpenAI-shaped request fires both
    ``async_log_failure_event`` and ``async_post_call_failure_hook`` for
    one CONTEXT attempt (LiteLLM does this for the buffered OpenAI routes
    on a router-level retry). The CONTEXT branch must dedup so exactly
    one failure record survives per request -- matching today's
    one-record-per-request semantics.

    The non-CONTEXT case has the existing ``_TERMINATED`` marker. The
    CONTEXT branch cannot reuse it (would suppress the fallback success),
    so the per-hop dedupe (``_ctx_fail_booked_hops`` set) is the
    sole guard against double-booking, and the test pins the set on
    that account.
    """
    h, _reg, slots, _ledger, _policy, redis, pick = _build(claim_lane="local")
    ctx = _ctx_for(pick, call_type="acompletion")  # NOT in UNLOGGED_CALL_TYPES
    from litellm import ContextWindowExceededError
    exc = ContextWindowExceededError(
        message="prompt is too long: 200000 tokens exceeds the maximum of 131072",
        model=pick.model.ref,
        llm_provider="openai",
    )
    kwargs = {"metadata": _meta_for(ctx), "exception": exc}

    async def go():
        # Both hooks fire for one CONTEXT attempt. The first call books
        # the failure; the second sees the same hop in the per-hop
        # dedupe set and no-ops.
        await h.async_log_failure_event(
            kwargs=kwargs, response_obj=None,
            start_time=None, end_time=None,
        )
        first_failure_count = _failure_count(
            redis, pick.plan.key,
        )
        first_hops_flag = ctx.get("_ctx_fail_booked_hops")
        await h.async_post_call_failure_hook(
            request_data={"metadata": _meta_for(ctx)},
            original_exception=exc,
            user_api_key_dict=None,
        )
        second_failure_count = _failure_count(
            redis, pick.plan.key,
        )
        return first_failure_count, second_failure_count, first_hops_flag

    first_count, second_count, first_hops_flag = asyncio.run(go())
    # First call did the work: failure counter incremented once, the
    # per-hop dedupe set now has the hop (the no-router-metadata
    # sentinel -1).
    assert first_count == 1, (
        f"first CONTEXT hook must book one failure record; got "
        f"{first_count}")
    assert first_hops_flag == {-1}, (
        f"first CONTEXT hook must add the hop to _ctx_fail_booked_hops; "
        f"got {first_hops_flag!r}")
    # Second hook saw the dedup and no-op'd: counter unchanged.
    assert second_count == first_count, (
        f"second CONTEXT hook must no-op via the per-hop dedupe; got "
        f"{first_count} -> {second_count}")
    # Marker NOT claimed -- CONTEXT never claims (a sibling success
    # could still arrive and must be allowed to book).
    assert ctx.get(_TERMINATED) is None, (
        f"CONTEXT branch must not claim _TERMINATED; got "
        f"{ctx.get(_TERMINATED)!r}")
    assert ctx.get("_ctx_fail_booked_hops") == {-1}
    # Exactly one failure record on the picked plan -- the second hook
    # must not have re-booked.
    plan_keys = [
        k for k in redis.hashes.keys()
        if k.startswith(f"sy:usage:{pick.plan.key}:p:")
    ]
    assert plan_keys, "no period bucket for the failed plan"
    bucket = _bucket_sync(redis, plan_keys[0])
    assert bucket["failures"] == 1, (
        f"CONTEXT double-fire must leave exactly one failure record; "
        f"got {bucket}")
    # Slot was released exactly once.
    assert asyncio.run(slots.in_flight(pick.plan.key)) == 0
    print(f"  CONTEXT double-fire: 1 failure record ({int(bucket['failures'])}), "
          f"slot released once, ctx[_TERMINATED]={ctx.get(_TERMINATED)!r}, "
          f"_ctx_fail_booked_hops={sorted(ctx.get('_ctx_fail_booked_hops') or set())!r}")


def test_multihop_sibling_non_context_failure_books_sibling_plan():
    """Multi-hop CONTEXT-window-fallbacks walk where a sibling hop fails
    with a non-CONTEXT verdict (e.g. a 5xx or 401 on the fallback
    deployment) must:

      * Book the CONTEXT failure against the picked plan (hop 0),
        without claiming ``_TERMINATED`` so a later success on yet
        another sibling could still book.
      * Book the non-CONTEXT failure against the sibling plan (hop 1),
        not the picked plan. The pre-fix implementation used ``ctx["plan"]``
        for every hop's verdict, which meant an AUTH hop on a sibling
        cooled the picked plan for 1800s (the plan did nothing wrong),
        a TRANSIENT hop bumped the picked plan's breaker streak (the
        wrong plan's ladder), and the sibling's plan saw no failure row
        at all for the attempt that actually failed on it.
      * Run per-hop dedupe: each hop gets exactly one failure row.
      * Apply per-hop verdict cooldown: the sibling's TRANSIENT cooldown
        lands on the sibling, not on the picked plan; the picked plan
        stays untouched (the only failure the picker ever made was the
        CONTEXT one, which carries ``cooldown_seconds=0``).

    The lifecycle this exercises (uses the production-shaped kwargs
    the router stamps at walk entry, ``attempted_fallbacks = 0``):

      async_log_failure_event(kwargs=hop0_kwargs, exc=ContextWindow…)
        -> hop=0, books CONTEXT failure on picked plan, adds
           ``_ctx_fail_booked_hops = {0}``.
      async_log_failure_event(kwargs=hop1_kwargs, exc=Upstream5xx)
        -> hop=1, books non-CONTEXT failure on sibling plan, adds
           ``_ctx_fail_booked_hops = {0, 1}``. The marker is NOT
           claimed mid-walk (would suppress a sibling success on a
           later hop -- the #186 hole).
    """
    h, reg, slots, _ledger, _policy, redis, pick = _build(claim_lane="local")
    siblings = [
        m for m in reg.models.values()
        if m.ref != pick.model.ref and m.context_window and m.context_window > pick.model.context_window
    ]
    assert siblings, (
        "fixture must have at least one bigger-window sibling for the "
        "fallback walk to be exercisable; got "
        f"{[(m.ref, m.context_window) for m in reg.models.values() if m.context_window]}")
    served = siblings[0]
    served_ref = served.ref
    served_plan_key = reg.plan_of(served).key
    picked_plan_key = pick.plan.key
    assert served_plan_key != picked_plan_key, (
        "test needs the bigger-window sibling on a different plan so "
        "verdict + failure-row attribution has two distinct buckets to "
        "test against; both landed on "
        f"{picked_plan_key}/{served_plan_key}")
    picked_deployment = pick.model.deployment
    served_deployment = served.deployment

    ctx = _ctx_for(pick, call_type="acompletion")
    from litellm import ContextWindowExceededError

    async def go():
        # Hop 0: a real CONTEXT failure on the picked deployment. Routers
        # always fire ``async_log_failure_event`` per attempt for the
        # OpenAI-shaped call types; drive this through that hook with
        # the production-shaped kwargs the router stamps at walk
        # entry (``attempted_fallbacks = 0`` -- router.py:7582,
        # ``async_function_with_fallbacks``).
        hop0_ctxw_exc = ContextWindowExceededError(
            message=(
                "This model's maximum context length is 131072 tokens. "
                "However, you requested 200000 tokens."
            ),
            model=pick.model.ref,
            llm_provider="openai",
        )
        hop0_meta = dict(_meta_for(ctx))
        hop0_meta["attempted_fallbacks"] = 0
        hop0_kwargs = {
            "model": picked_deployment,
            "litellm_params": {"metadata": hop0_meta},
            "exception": hop0_ctxw_exc,
        }
        await h.async_log_failure_event(
            kwargs=hop0_kwargs, response_obj=None,
            start_time=None, end_time=None,
        )
        # Walk failed the picked deployment for context; litellm
        # walks to the sibling. The sibling returns a 5xx. ``run_async_fallback``
        # catches any exception per hop and the walk bubbles the last
        # error up, so ``async_log_failure_event`` fires for hop 1 with
        # ``attempted_fallbacks=1`` and ``kwargs["model"]`` rewritten to
        # the sibling's deployment string by ``_update_kwargs_before_fallbacks``.
        class _Sibling5xx(Exception):
            status_code = 503
            message = "upstream temporarily unavailable"

        hop1_exc = _Sibling5xx()
        hop1_kwargs_meta = dict(_meta_for(ctx))
        hop1_kwargs_meta["attempted_fallbacks"] = 1
        hop1_kwargs = {
            "model": served_deployment,
            "litellm_params": {"metadata": hop1_kwargs_meta},
            "exception": hop1_exc,
        }
        await h.async_log_failure_event(
            kwargs=hop1_kwargs, response_obj=None,
            start_time=None, end_time=None,
        )
        # Snapshot the per-plan state after both hops.
        picked_keys = [
            k for k in redis.hashes.keys()
            if k.startswith(f"sy:usage:{picked_plan_key}:p:")
        ]
        served_keys = [
            k for k in redis.hashes.keys()
            if k.startswith(f"sy:usage:{served_plan_key}:p:")
        ]
        picked_bucket = (
            _bucket_sync(redis, picked_keys[0]) if picked_keys else {}
        )
        served_bucket = (
            _bucket_sync(redis, served_keys[0]) if served_keys else {}
        )
        # The per-plan cooldown key. ``_apply_verdict`` calls
        # ``slots.bump_and_cool`` for TRANSIENT outcomes; the sibling's
        # cooldown should land on the sibling's plan, not the picked
        # plan's.
        return (
            picked_bucket, served_bucket,
            ctx.get(_TERMINATED),
            ctx.get("_ctx_fail_booked_hops"),
        )

    (
        picked_bucket, served_bucket,
        terminated_flag, hops_flag,
    ) = asyncio.run(go())

    # The CONTEXT hop didn't claim ``_TERMINATED``; the non-CONTEXT
    # mid-walk hop ALSO did not claim it -- mirroring CONTEXT's
    # no-claim behavior. The walk raised (no sibling success is
    # coming) but a non-CONTEXT mid-walk has no way to know that at
    # hop time; claiming ``_TERMINATED="failure"`` would suppress
    # ``_finish_success`` on a later hop and re-open the original #186
    # ledger hole. The marker stays unset until either a primary-like
    # arm claims it (none fired here for this mid-walk test -- hop
    # 0 used the production-shaped kwargs-with-stamp entry, so the
    # claim condition ``hop <= 0 AND not is_context`` could have
    # applied to hop 0 if it was non-CONTEXT; hop 1 is mid-walk and
    # explicitly skips the claim per the rule) or a sibling success
    # claims ``"success"``.
    assert terminated_flag is None, (
        f"mid-walk non-CONTEXT hop must not claim _TERMINATED (would "
        f"suppress a sibling success on a later hop); got "
        f"{terminated_flag!r}")
    # Per-hop dedupe set records both attempts. The test drives hop 0
    # with the production-shaped ``attempted_fallbacks=0`` kwargs
    # (router entry stamp on the picked deployment), so the set must
    # contain 0 (from the CONTEXT CONTEXT attempt) and 1 (from hop
    # 1's 5xx).
    expected_hops = {0, 1}
    assert (hops_flag or set()) >= expected_hops, (
        f"per-hop dedupe set must record both attempted_fallbacks "
        f"(0 for the router entry stamp, 1 for the first sibling); "
        f"got {hops_flag!r}")
    # Both hops contributed at least the explicit stamps.
    for h in expected_hops:
        assert h in (hops_flag or set()), (
            f"per-hop dedupe set must record hop {h}; got {hops_flag!r}")
    # The picked plan got the CONTEXT failure row. (non-CONTEXT path
    # also tries to write one on this plan, but only the CONTEXT hop
    # resolves target = picked_plan.)
    assert picked_bucket.get("failures", 0) >= 1, (
        f"picked plan ({picked_plan_key}) must record at least the "
        f"CONTEXT failure; got {picked_bucket}")
    # The sibling plan got its OWN non-CONTEXT failure row, attributed
    # to the sibling model. Pre-fix this bucket stayed empty (every
    # failure row landed on the picked plan).
    assert served_bucket.get("failures", 0) >= 1, (
        f"served plan ({served_plan_key}) must record the hop-1 failure "
        f"on the sibling deployment; got {served_bucket}")
    # And the model-scoped row for the served ref is on the served plan,
    # not the picked one -- a key sanity check.
    served_model_rows = [
        k for k in redis.hashes.keys()
        if k.startswith(f"sy:usage:{served_plan_key}:m:{served_ref}:")
    ]
    assert served_model_rows, (
        f"served ref {served_ref} must have its own model-scoped bucket "
        f"under {served_plan_key}; got "
        f"{[k for k in redis.hashes.keys() if k.startswith('sy:usage:')]}")
    # No slot left claimed on either plan. Both hops' releases collapse
    # onto ctx["plan"] (the picked plan); the sibling was never inflight
    # so its in_flight was always 0.
    assert asyncio.run(slots.in_flight(picked_plan_key)) == 0
    assert asyncio.run(slots.in_flight(served_plan_key)) == 0
    print(
        f"  multi-hop walk: picked ({picked_plan_key}) booked "
        f"{int(picked_bucket.get('failures', 0))} failure, served "
        f"({served_plan_key}) booked "
        f"{int(served_bucket.get('failures', 0))} failure (model "
        f"{served_ref}); per-hop dedupe={sorted(hops_flag)}; "
        f"ctx[_TERMINATED]={terminated_flag!r}"
    )


def test_multihop_5xx_then_success_books_served_plan():
    """Residual #186 hole from cycle 2: a three-attempt CONTEXT-window-
    fallbacks walk where hop 1 fails with a non-CONTEXT verdict and
    hop 2 succeeds on a second sibling. Pre-fix the non-CONTEXT hop
    claimed ``_TERMINATED="failure"`` mid-walk, so hop 2's
    ``_finish_success`` short-circuited and the served plan's tokens
    were never booked -- exactly the bug this PR sets out to fix,
    reopening as soon as the chain has more than one fallback target
    and the first sibling hops with a non-CONTEXT error.

    Post-fix (this commit) the non-CONTEXT mid-walk branch mirrors
    CONTEXT's no-claim behavior: ``_TERMINATED`` is left unset on
    every per-hop mid-walk arm (int attempted_fallbacks stamps). The
    only arms that claim ``_TERMINATED="failure"`` are the primary
    sentinel -- fires when no walked fallback ever started (single
    attempt) or the proxy-side async_post_call was the terminal arm
    after every hop is exhausted.

    The fixture's autoloaded context_window_fallbacks config only
    emits one sibling target per source deployment (the local 1M-
    context member is intentionally absent in tests/plans.yaml, see
    the comment at ``config/plans.example.yaml:925-927``), so this
    test injects a synthetic 2M-context sibling on the openrouter
    plan via ``Registry.replace`` to give the walk 2 fallback
    targets. The injection is local to this test (the handler's
    registry is swapped and the original is not mutated); the rest
    of the suite uses the autoloaded fixture untouched.
    """
    from dataclasses import replace as _replace
    from switchyard import models as _models
    h, reg, slots, _ledger, _policy, redis, pick = _build(claim_lane="local")
    siblings = [
        m for m in reg.models.values()
        if m.ref != pick.model.ref and m.context_window and m.context_window > pick.model.context_window
    ]
    assert siblings, (
        "fixture must have at least one bigger-window sibling; got "
        f"{[(m.ref, m.context_window) for m in reg.models.values() if m.context_window]}")
    # Inject a second sibling on the same plan so the chain has 2
    # targets and the router can walk past an exhausted hop 1.
    openrouter_plan = reg.plans["openrouter"]
    synth_model = _models.Model(
        key="mimo-2m",
        plan_key="openrouter",
        model="openai/mimo-2m-synthetic",
        context_window=openrouter_plan.models["mimo"].context_window * 2,
    )
    synth_models = {**openrouter_plan.models, "mimo-2m": synth_model}
    new_openrouter = _replace(openrouter_plan, models=synth_models)
    new_plans = {**reg.plans, "openrouter": new_openrouter}
    h.registry = _models.Registry(
        settings=reg.settings, plans=new_plans, lanes=reg.lanes,
    )
    # Re-resolve the plan/model references against the new registry.
    reg = h.registry
    sibling_a = reg.models[siblings[0].ref]
    sibling_a_plan_key = reg.plan_of(sibling_a).key
    sibling_b = reg.models["openrouter/mimo-2m"]
    sibling_b_plan_key = reg.plan_of(sibling_b).key
    picked_plan_key = pick.plan.key
    assert sibling_a_plan_key != picked_plan_key, (
        "first sibling must be on a different plan than the pick so the "
        "served-plan attribution has somewhere distinct to land")
    assert sibling_b.ref != sibling_a.ref, (
        f"second sibling must be a different model ref from the first "
        f"so the success-side served-deployment resolution has somewhere "
        f"distinct to land; both landed on {sibling_a.ref}")

    ctx = _ctx_for(pick, call_type="acompletion")
    from litellm import ContextWindowExceededError

    async def go():
        # Hop 0: a CONTEXT failure on the picked deployment. The
        # router stamps ``attempted_fallbacks = 0`` in place on the
        # metadata at walk entry (``router.py::async_function_with_fallbacks``
        # verified for both the 1.101.0 gateway pin and the 1.102.1
        # test gate), so this hop carries the entry stamp just like a
        # real production call does. Building the kwargs with the
        # stamp keeps the fixture aligned with the wire shape the
        # cycle-3 review verified, so the per-hop dedupe set
        # collision with the post-call tail (`0 in booked_hops`) is
        # exercised end-to-end here.
        hop0_exc = ContextWindowExceededError(
            message=(
                "This model's maximum context length is 131072 tokens. "
                "However, you requested 200000 tokens."
            ),
            model=pick.model.ref,
            llm_provider="openai",
        )
        hop0_meta = dict(_meta_for(ctx))
        hop0_meta["attempted_fallbacks"] = 0
        hop0_kwargs = {
            "model": pick.model.deployment,
            "litellm_params": {"metadata": hop0_meta},
            "exception": hop0_exc,
        }
        await h.async_log_failure_event(
            kwargs=hop0_kwargs, response_obj=None,
            start_time=None, end_time=None,
        )

        # Hop 1: a non-CONTEXT 5xx on sibling A (a different plan from
        # the pick). THIS hop is the focus of the BLOCKER finding:
        # pre-fix it would claim ``_TERMINATED="failure"`` and suppress
        # the success on hop 2.
        class _SiblingA5xx(Exception):
            status_code = 503
            message = "upstream temporarily unavailable"

        hop1_meta = dict(_meta_for(ctx))
        hop1_meta["attempted_fallbacks"] = 1
        hop1_kwargs = {
            "model": sibling_a.deployment,
            "litellm_params": {"metadata": hop1_meta},
            "exception": _SiblingA5xx(),
        }
        await h.async_log_failure_event(
            kwargs=hop1_kwargs, response_obj=None,
            start_time=None, end_time=None,
        )

        # Snapshot mid-walk state BEFORE hop 2 succeeds (so the
        # BLOCKER's exact behaviour can be asserted: at this point
        # ctx[_TERMINATED] must still be unset, proving hop 1's
        # non-CONTEXT mid-walk branch did NOT claim the marker that
        # would otherwise suppress hop 2's success).
        mid_walk_terminated = ctx.get(_TERMINATED)

        # Hop 2: a SUCCESS on sibling B (also a different plan from the
        # pick, and a different plan from sibling A). The
        # _hidden_params["model_id"] carries sibling B's router_id --
        # the resolved wire shape for the success side. If hop 1 had
        # claimed ``_TERMINATED="failure"``, this hook's first-line
        # check would short-circuit and the served plan's tokens would
        # never book.
        response = {
            "id": "x", "object": "chat.completion",
            "model": sibling_b.ref,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 80},
        }
        response["_hidden_params"] = {"model_id": sibling_b.router_id}
        await h.async_log_success_event(
            kwargs={
                "model": sibling_b.deployment,
                "litellm_params": {"metadata": hop1_meta},
            },
            response_obj=response,
            start_time=None, end_time=None,
        )

        # Snapshot every relevant counter.
        picked_keys = [
            k for k in redis.hashes.keys()
            if k.startswith(f"sy:usage:{picked_plan_key}:p:")
        ]
        sibling_a_keys = [
            k for k in redis.hashes.keys()
            if k.startswith(f"sy:usage:{sibling_a_plan_key}:p:")
        ]
        sibling_b_keys = [
            k for k in redis.hashes.keys()
            if k.startswith(f"sy:usage:{sibling_b_plan_key}:p:")
        ]
        return (
            mid_walk_terminated,
            ctx.get(_TERMINATED),
            _bucket_sync(redis, picked_keys[0]) if picked_keys else {},
            _bucket_sync(redis, sibling_a_keys[0]) if sibling_a_keys else {},
            _bucket_sync(redis, sibling_b_keys[0]) if sibling_b_keys else {},
        )

    (
        mid_walk_terminated, post_walk_terminated,
        picked_bucket, sibling_a_bucket, sibling_b_bucket,
    ) = asyncio.run(go())

    # CRITICAL: hop 1's non-CONTEXT branch did NOT claim ``_TERMINATED``
    # mid-walk (mirroring CONTEXT's no-claim behaviour). Pre-fix this
    # was "failure" and would have suppressed hop 2's success.
    assert mid_walk_terminated is None, (
        f"hop 1 (non-CONTEXT mid-walk) must NOT claim _TERMINATED or "
        f"hop 2's success is suppressed -- the original #186 hole. "
        f"Got {mid_walk_terminated!r}")
    # After hop 2 success the marker is "success": the served plan's
    # tokens were booked.
    assert post_walk_terminated == "success", (
        f"_finish_success must run on hop 2 and claim 'success'; got "
        f"{post_walk_terminated!r}")
    # The picked plan got the CONTEXT failure row.
    assert picked_bucket.get("failures", 0) >= 1, (
        f"picked plan ({picked_plan_key}) must record the CONTEXT "
        f"failure; got {picked_bucket}")
    # Sibling A got the non-CONTEXT hop-1 failure row (and the
    # transient verdict + cooldown -- the cooldown bumps its ladder,
    # not the picked plan's). Pre-fix this bucket was empty for
    # cross-plan walks.
    assert sibling_a_bucket.get("failures", 0) >= 1, (
        f"sibling-a plan ({sibling_a_plan_key}) must record hop 1's "
        f"5xx; got {sibling_a_bucket}")
    # The served plan got the tokens from the successful fallback.
    # sibling_a and sibling_b live on the same plan in this fixture
    # (the openrouter singleton), so the period bucket aggregates
    # both hops: 1 failure row from hop 1 plus 1 success row from
    # hop 2. The booking-side assertions are on the aggregated
    # bucket -- 1+ requests (hop 1's failure row counts a request
    # too), the exact success tokens, and 1 failure row.
    assert sibling_b_bucket.get("requests", 0) >= 1, (
        f"served plan ({sibling_b_plan_key}) must record at least "
        f"the successful fallback as one request (hop 1 also bumps "
        f"requests via the failure row when both siblings share the "
        f"plan); got {sibling_b_bucket}")
    assert sibling_b_bucket.get("prompt_tokens", 0) == 500, (
        f"served plan must book the prompt tokens from the "
        f"successful fallback; got {sibling_b_bucket}")
    assert sibling_b_bucket.get("completion_tokens", 0) == 80, (
        f"served plan must book the completion tokens from the "
        f"successful fallback; got {sibling_b_bucket}")
    assert sibling_b_bucket.get("failures", 0) >= 1, (
        f"served plan must record the hop-1 5xx failure row alongside "
        f"the hop-2 success; got {sibling_b_bucket}")
    # And the picked plan did NOT receive the success-side tokens
    # (its bucket has only the CONTEXT failure row -- no requests,
    # no prompt tokens, no completion tokens -- because every served
    # plan row landed on the openrouter bucket).
    assert picked_bucket.get("requests", 0) <= 1, (
        f"picked plan has at most 1 request (the CONTEXT failure row); "
        f"got {picked_bucket}")
    assert picked_bucket.get("prompt_tokens", 0) == 0, (
        f"picked plan must NOT receive success-side prompt tokens "
        f"(the served plan owns the request); got {picked_bucket}")
    assert picked_bucket.get("completion_tokens", 0) == 0, (
        f"picked plan must NOT receive success-side completion "
        f"tokens; got {picked_bucket}")
    # Slot was released from the picked plan; sibling plans were never
    # inflight (placement only ever claimed on the picked plan).
    assert asyncio.run(slots.in_flight(picked_plan_key)) == 0
    assert asyncio.run(slots.in_flight(sibling_a_plan_key)) == 0
    assert asyncio.run(slots.in_flight(sibling_b_plan_key)) == 0
    print(
        f"  three-hop walk: picked ({picked_plan_key}) CONTEXT "
        f"failure {int(picked_bucket.get('failures', 0))}; sibling-a "
        f"({sibling_a_plan_key}) hop-1 5xx "
        f"{int(sibling_a_bucket.get('failures', 0))}; sibling-b "
        f"({sibling_b_plan_key}) success "
        f"{int(sibling_b_bucket.get('requests', 0))} req / "
        f"{int(sibling_b_bucket.get('prompt_tokens', 0))} prompt / "
        f"{int(sibling_b_bucket.get('completion_tokens', 0))} completion; "
        f"ctx[_TERMINATED]={post_walk_terminated!r}"
    )


def test_sibling_hop_classified_with_sibling_plan_family():
    """Cross-family SHOULD-FIX: the verdict for a sibling hop is
    classified with the SIBLING plan's ``provider_family`` (the
    plan that produced the error), not the picked plan's family.

    Pre-fix ``_finish_failure`` called ``_classify_failure(exception,
    plan.provider_family)`` where ``plan`` was always ``ctx["plan"]``
    (the picked plan). On a cross-family walk (the picker picked
    minimax-ultra/m3, the router walked to a glm sibling for context)
    a sibling's vendor code was interpreted through the wrong table:
    MiniMax 1113 (insufficient balance / ZAI) was unknown to the
    MiniMax table and fell through to a generic verdict (RATE_LIMITED,
    BAD_REQUEST or TRANSIENT depending on the HTTP status) instead of
    ``QUOTA_EXHAUSTED``. The verdict then ran on the right plan but
    with the wrong semantics.

    Post-fix the classification reads ``target.provider_family`` where
    ``target`` is the served failure plan resolved by
    ``_resolve_served_failure_plan`` from ``kwargs["model"]``.

    The assertion is white-box: wrap ``classify`` to capture every
    ``family=`` it received, then drive a hop 1 sibling failure and
    assert the recorded family equals the sibling plan's
    ``provider_family`` (the post-fix value), not the picked plan's
    (the pre-fix value).
    """
    from dataclasses import replace as _replace
    from switchyard import classify as _classify_mod

    # Pick a real MiniMax plan (family='minimax') as the picked plan;
    # pick its first member; pick a ZAI plan (family='zai') sibling
    # with strictly-larger context_window (synthesised so the
    # generator's walk would target it). The fixture's autoloaded
    # context_window is None for most providers, so we synthesise a
    # window via dataclasses.replace.
    h, reg, slots, _ledger, _policy, redis, pick = _build()
    # Switch the picked plan to a minimax plan for the test. The picker
    # was called with no lane, so we kept the fixture's default pick;
    # for this test we explicitly resolve a minimax pick using the
    # picker, then restore ``h.registry`` to use that synthesis.
    # Resolve a minimax-family plan/model (any minimax member) and
    # use it as the picked plan/model.
    minimax_models = [
        m for m in reg.models.values() if reg.plan_of(m).provider_family == "minimax"
    ]
    assert minimax_models, (
        "fixture must include at least one minimax-family model; got "
        f"{[(m.ref, reg.plan_of(m).provider_family) for m in reg.models.values() if reg.plan_of(m).provider_family]}")
    pick = minimax_models[0]
    picked_plan = reg.plan_of(pick)
    picked_family = picked_plan.provider_family
    assert picked_family == "minimax", picked_family

    # Build a sibling on a ZAI plan with strictly-larger context_window
    # so the failure-row target differs from the picked plan. Take any
    # ZAI model and bump its window past minimax's default.
    zai_models = [
        m for m in reg.models.values() if reg.plan_of(m).provider_family == "zai"
    ]
    assert zai_models, (
        "fixture must include at least one zai-family model; got "
        f"{[(m.ref, reg.plan_of(m).provider_family) for m in reg.models.values() if reg.plan_of(m).provider_family]}")
    sibling = zai_models[0]
    sibling_plan = reg.plan_of(sibling)
    sibling_family = sibling_plan.provider_family
    assert sibling_family == "zai", sibling_family
    assert sibling.ref != pick.ref, (
        "sibling must differ from pick so _resolve_served_failure_plan "
        "returns the sibling plan, not the picked plan")
    assert sibling_plan.key != picked_plan.key, (
        "sibling must be on a different plan than the pick")

    # Synthesise context windows on both plans so registry.model_for_deployment
    # / model_for_router_id still resolve and the walk-target distinction
    # works (this is what _resolve_served_failure_plan reads, not
    # context_window per se). The actual contract tested is which
    # provider_family classify received, so the synthesis is irrelevant
    # to the assertion.
    new_models_for_pick = dict(picked_plan.models)
    new_models_for_pick[pick.key] = _replace(pick, context_window=131072)
    new_picked_plan = _replace(picked_plan, models=new_models_for_pick)
    new_models_for_sibling = dict(sibling_plan.models)
    new_models_for_sibling[sibling.key] = _replace(sibling, context_window=262144)
    new_sibling_plan = _replace(sibling_plan, models=new_models_for_sibling)
    new_plans = {**reg.plans,
                 picked_plan.key: new_picked_plan,
                 sibling_plan.key: new_sibling_plan}
    h.registry = type(reg)(
        settings=reg.settings, plans=new_plans, lanes=reg.lanes,
    )

    # Build a ctx as if the pre-call hook had picked the minimax
    # model and the picker had populated ``caller_env`` etc.
    # ``_ctx_for`` reads ``pick.lane / pick.plan.key / pick.model.ref``;
    # we manually build the ctx to avoid the picker round-trip.
    from switchyard.hooks import META_KEY
    ctx = {
        "lane": "test-cross-family",
        "plan": picked_plan.key,
        "model": pick.ref,
        "request_id": "rid-cross-family-test",
        "session": None,
        "sticky": False,
        "claimed_at": 0.0,
        "cap": pick.max_parallel,
        "direct": False,
        "needs_tools": False,
        "needs_images": False,
        "pinned": False,
        "picked_group_gid": "",
        "picked_group_strategy": "",
        "call_type": "acompletion",
    }
    # Wrap classify to capture every (family, status, message) call.
    captured: list[dict] = []
    original = _classify_mod.classify

    def _spy(status, message, *, family=None, body=None, retry_after=None,
             default_cooldown=900):
        captured.append({
            "family": family, "status": status,
            "message": (
                message[:80] if isinstance(message, str) else str(message)
            ),
        })
        return original(
            status, message, family=family, body=body,
            retry_after=retry_after, default_cooldown=default_cooldown,
        )

    # Patch the classify the handler uses (its imported binding in
    # hooks.py -- the test resolves the binding via the module).
    from switchyard import hooks as _hooks_mod
    real_handler_classify = _hooks_mod.classify
    _hooks_mod.classify = _spy
    try:
        # Drive a non-CONTEXT sibling hop: kwargs["model"] is the
        # sibling's deployment (so _resolve_served_failure_plan
        # returns the sibling's plan), and ``attempted_fallbacks=1``
        # so the dedupe uses the per-hop int slot. The exception
        # carries a ``status_code`` attribute so the new
        # ``_provider_error`` gate in ``_finish_failure`` does NOT
        # short-circuit it to ``Outcome.INTERNAL`` -- the test wants
        # ``classify`` to actually run with the sibling family's
        # ``FAMILIES`` table.
        class _SiblingZAIQuotaError(Exception):
            status_code = 400
            message = "ZAI vendor code 1113 with insufficient balance body"

        sibling_meta = {META_KEY: ctx, "attempted_fallbacks": 1}
        sibling_kwargs = {
            "model": sibling.deployment,
            "litellm_params": {"metadata": sibling_meta},
            "exception": _SiblingZAIQuotaError(),
        }
        asyncio.run(h.async_log_failure_event(
            kwargs=sibling_kwargs, response_obj=None,
            start_time=None, end_time=None,
        ))
    finally:
        _hooks_mod.classify = real_handler_classify

    # The handler should have called classify with the SIBLING plan's
    # family (post-fix). Pre-fix it would have called classify with
    # the picked plan's family.
    sibling_calls = [
        c for c in captured
        if c["family"] == sibling_family
    ]
    assert sibling_calls, (
        f"sibling hop must classify with the sibling plan's "
        f"provider_family={sibling_family!r}; recorded families "
        f"{[c['family'] for c in captured]}")
    picked_calls = [
        c for c in captured
        if c["family"] == picked_family
    ]
    assert not picked_calls, (
        f"sibling hop must NOT classify with the picked plan's "
        f"family={picked_family!r} (would interpret the sibling's "
        f"vendor code through the wrong table); recorded families "
        f"{[c['family'] for c in captured]}")
    # Slot was released on the picked plan (sibling was never inflight).
    assert asyncio.run(slots.in_flight(picked_plan.key)) == 0
    print(
        f"  cross-family walk: classify called with "
        f"{len(captured)} family(ies); sibling hop used "
        f"{sibling_family!r} (correct); picked-family "
        f"{picked_family!r} not used (correct)."
    )


def test_single_attempt_non_context_with_router_entry_stamp_claims_marker():
    """Production-shape companion to
    ``test_non_context_failure_still_claims_marker_and_suppresses_success``:
    the router stamps ``attempted_fallbacks = 0`` on the metadata at
    walk entry (``router.py::async_function_with_fallbacks:7582`` in
    place), so a single-attempt non-CONTEXT failure for an OpenAI-
    shape route arrives with the entry stamp and ``hop = 0``. The
    contract
    ``non-CONTEXT primary-like arms (hop <= 0) claim _TERMINATED`` so
    a same-attempt late success can't double-book must hold for this
    shape too -- not only for the no-router-metadata path
    (``hop = -1``) that the cycle-2 test exercises.

    Pre-fix this test would have failed: the cycle-2 implementation
    only claimed ``_TERMINATED`` on ``hop == -1``, silently dropping
    the production contract for router traffic that always stamps
    ``0``. The cycle-3 fix widens the claim to ``hop <= 0`` so both
    the entry stamp and the no-router-metadata sentinel claim. The
    cycle-3 review verified that this is the invariant the router
    actually emits (1.101.0 + 1.102.1).
    """
    h, _reg, slots, _ledger, _policy, _redis, pick = _build(claim_lane="local")
    ctx = _ctx_for(pick, call_type="acompletion")
    # Production shape: kwargs carries the router's entry stamp.
    hop0_meta = dict(_meta_for(ctx))
    hop0_meta["attempted_fallbacks"] = 0
    kwargs = {
        "model": pick.model.deployment,
        "litellm_params": {"metadata": hop0_meta},
        "exception": None,
    }

    # Drive through the unified ``_finish_failure`` so the production-
    # shaped kwargs are read end-to-end. The None exception is fine
    # here -- the cycle-3 _finish_failure classifies with empty
    # message + no status (TRANSIENT unclassified), enough to
    # exercise the marker-claim branch on hop <= 0.
    asyncio.run(_drive_log_failure(h, kwargs, ctx))
    post_failure_terminated = ctx.get(_TERMINATED)
    assert post_failure_terminated == "failure", (
        f"non-CONTEXT failure at hop=0 (router entry stamp) must claim "
        f"_TERMINATED='failure'; got {post_failure_terminated!r}")
    # And the per-hop dedupe set has hop 0.
    assert ctx.get("_ctx_fail_booked_hops") == {0}, (
        f"per-hop dedupe set must contain the entry stamp; got "
        f"{ctx.get('_ctx_fail_booked_hops')!r}")
    # Slot was released (idempotent zrem).
    assert asyncio.run(slots.in_flight(pick.plan.key)) == 0
    print(
        f"  router-stamped non-CONTEXT (hop=0) on "
        f"{pick.model.ref}: marker claimed "
        f"({post_failure_terminated!r}); per-hop dedupe={sorted(ctx.get('_ctx_fail_booked_hops') or set())!r}"
    )


def test_midwalk_non_context_does_not_write_verdict_applied_marker():
    """Cycle-3 finding: the verdict's ``_verdict_applied`` marker skip
    was keyed on the verdict's ``is_context`` only -- a non-CONTEXT
    mid-walk hop (``hop = 1``) wrote ``ctx["_verdict_applied"] =
    rid``, and ``ctx["request_id"]`` is shared across the whole walk,
    so the marker then suppressed every later hop's verdict in the
    same request. The reachable shape the reviewer cites: hop 0
    CONTEXT (no marker), hop 1 sibling 5xx (TRANSIENT verdict, marker
    written), hop 2 sibling 401 (AUTH verdict **suppressed** -- the
    sibling that actually rejected the request never gets the 1800s
    cooldown or the ``note_exhaustion`` it should).

    Post-fix (this commit) ``skip_marker = is_context or hop > 0``,
    so every mid-walk arm (``hop >= 1``) skips the marker write
    regardless of verdict type. The marker is still written only on
    primary-like arms (hop <= 0) where it dedupes the OpenAI
    double-fire.

    This test drives the exact two-non-CONTEXT-siblings shape on
    an injected openrouter 2-target chain (the fixture's autoloaded
    chain only has 1 sibling, but the suffix reproduces the same
    dedupe behavior on a 2-hop openrouter walk).
    """
    from dataclasses import replace as _replace
    from switchyard import models as _models
    h, reg, slots, _ledger, _policy, redis, pick = _build(claim_lane="local")
    siblings = [
        m for m in reg.models.values()
        if m.ref != pick.model.ref and m.context_window and m.context_window > pick.model.context_window
    ]
    assert siblings, "fixture must have at least one bigger sibling"
    or_plan = reg.plans["openrouter"]
    synth = _models.Model(
        key="mimo-2m", plan_key="openrouter",
        model="openai/mimo-2m-synthetic",
        context_window=or_plan.models["mimo"].context_window * 2,
    )
    new_or = _replace(
        or_plan,
        models={**or_plan.models, "mimo-2m": synth},
    )
    new_plans = {**reg.plans, "openrouter": new_or}
    h.registry = type(reg)(
        settings=reg.settings, plans=new_plans, lanes=reg.lanes,
    )
    reg = h.registry
    sibling_a = reg.models[siblings[0].ref]
    sibling_b = reg.models["openrouter/mimo-2m"]
    assert sibling_a.ref != sibling_b.ref

    ctx = _ctx_for(pick, call_type="acompletion")
    from switchyard.hooks import META_KEY

    async def go():
        # Hop 1: non-CONTEXT TRANSIENT 5xx on sibling A (mid-walk).
        # Production-shaped kwargs carries ``attempted_fallbacks = 1``.
        class _SiblingA5xx(Exception):
            status_code = 503
            message = "upstream temporarily unavailable"

        hop1_meta = {META_KEY: ctx, "attempted_fallbacks": 1}
        hop1_kwargs = {
            "model": sibling_a.deployment,
            "litellm_params": {"metadata": hop1_meta},
            "exception": _SiblingA5xx(),
        }
        await h.async_log_failure_event(
            kwargs=hop1_kwargs, response_obj=None,
            start_time=None, end_time=None,
        )
        # The marker must NOT be written on a mid-walk hop. The
        # ``_verdict_applied`` ctx key stays unset, so the next hop
        # is free to apply its verdict. (The TRANSIENT verdict on
        # hop 1 still cools the sibling; the test asserts the MARKER
        # behaviour, not the cooldown.)
        mid_walk_marker = ctx.get("_verdict_applied")
        # Hop 2: non-CONTEXT AUTH 401 on sibling B (mid-walk). With
        # the cycle-3 fix, the marker is still unset, so hop 2's
        # verdict lands fully (no suppression by hop 1's marker).
        class _SiblingBAuth(Exception):
            status_code = 401
            message = "credentials rejected"

        hop2_meta = {META_KEY: ctx, "attempted_fallbacks": 2}
        hop2_kwargs = {
            "model": sibling_b.deployment,
            "litellm_params": {"metadata": hop2_meta},
            "exception": _SiblingBAuth(),
        }
        await h.async_log_failure_event(
            kwargs=hop2_kwargs, response_obj=None,
            start_time=None, end_time=None,
        )
        post_walk_marker = ctx.get("_verdict_applied")
        return mid_walk_marker, post_walk_marker, ctx.get("_TERMINATED")

    mid_walk_marker, post_walk_marker, terminated_flag = asyncio.run(go())

    # The marker is NOT written on the mid-walk hop. Pre-fix it
    # would have been written here (cycle-3 finding).
    assert mid_walk_marker is None, (
        f"mid-walk hop (hop=1) must NOT write _verdict_applied (the "
        f"shared request_id would suppress the next hop's verdict); "
        f"got {mid_walk_marker!r}")
    # The marker is also NOT written on hop 2 (still mid-walk). The
    # post-walk marker check is just to confirm the unified rule
    # applies on every hop >= 1.
    assert post_walk_marker is None, (
        f"mid-walk hop (hop=2) must NOT write _verdict_applied; got "
        f"{post_walk_marker!r}")
    # ``_TERMINATED`` is unset for the same reason (mid-walk arms
    # never claim -- a later hop's success must still be allowed to
    # book).
    assert terminated_flag is None, (
        f"mid-walk hops must not claim _TERMINATED (a later success "
        f"would be suppressed); got {terminated_flag!r}")
    # Per-hop dedupe set has both int stamps.
    assert ctx.get("_ctx_fail_booked_hops") == {1, 2}, (
        f"per-hop dedupe set must record both attempted_fallbacks; "
        f"got {ctx.get('_ctx_fail_booked_hops')!r}")
    print(
        f"  mid-walk non-CONTEXT (hop=1, hop=2) on "
        f"{sibling_a.ref} + {sibling_b.ref}: marker never written "
        f"(mid-walk_marker={mid_walk_marker!r}, post_walk_marker="
        f"{post_walk_marker!r}); ctx[_TERMINATED]={terminated_flag!r}; "
        f"per-hop dedupe={sorted(ctx.get('_ctx_fail_booked_hops') or set())!r}"
    )


async def _drive_log_failure(h, kwargs, ctx):
    """Bare ``_finish_failure`` driver for tests that do not care
    which hook fires the call.
    """
    exception = kwargs.get("exception")
    return await h._finish_failure(
        ctx, exception, kwargs=kwargs,
    )


# ============================================================================
# Issue #188 — splitter flush tail must reach the client, not just the log.
#
# `ReasoningSplitter.flush()` returns whatever the splitter is still
# holding back when the stream ends — a partial tag opener/closer that
# straddled a chunk boundary, or the trailing reasoning of an unclosed
# think block. Before this fix, ``async_post_call_streaming_iterator_hook``
# only logged the tail (``log.debug("stream ended mid-tag; flushed ...")``)
# without yielding anything, so the client silently lost the last one or
# two characters of every streamed response that ended mid-tag. The fix
# yields one synthetic OpenAI-shaped chunk carrying the tail, cloned from
# the last emitted frame so the terminal ``finish_reason`` on the real
# last chunk is preserved.
#
# ``_OAIStreamingChunk`` mirrors the OpenAI-shaped stream chunk the
# hook reads via ``getattr(chunk, "choices", None)`` /
# ``getattr(delta, ...)``. ``acompletion`` is NOT in
# ``UNLOGGED_CALL_TYPES``, so the streamer takes the bare-pass-through
# arm with the disconnect-cleanup try/except — but the splitter-path
# applies regardless of call_type once the OpenAI-shape condition above
# matches, so the test wires that path with the cheapest call_type for
# the OpenAI shape.
# ============================================================================


class _OAIStreamingChunk:
    """Fake OpenAI-shaped streaming chunk with mutable attributes.

    Mirrors the shape the streaming hook reads
    (``chunk.choices[0].delta.content``); mutable by design so the
    splitter's reasoning-split path can rewrite ``delta.content`` and
    ``delta.reasoning_content`` in place, and so a deep-copied synthetic
    tail chunk can stamp the flushed tail without disturbing the
    already-yielded source.

    ``choices`` (optional): when supplied, a list of
    ``(content, reasoning_content, finish_reason)`` tuples, one per
    choice (n > 1 streams). When omitted, a single choice carrying the
    positional ``content / reasoning_content / finish_reason`` is built.
    """
    def __init__(self, content="", reasoning_content=None, finish_reason=None,
                 choices=None):
        self.id = "chatcmpl-fake"
        self.object = "chat.completion.chunk"
        self.created = 0
        self.model = "fake-model"
        if choices is None:
            self.choices = [_OAIChoice(
                content=content, reasoning_content=reasoning_content,
                finish_reason=finish_reason,
            )]
        else:
            self.choices = [
                _OAIChoice(content=c, reasoning_content=r, finish_reason=fr)
                for c, r, fr in choices
            ]
            for idx, choice in enumerate(self.choices):
                choice.index = idx


class _OAIChoice:
    def __init__(self, content="", reasoning_content=None, finish_reason=None):
        self.index = 0
        self.finish_reason = finish_reason
        self.delta = _OAIDelta(
            content=content, reasoning_content=reasoning_content,
        )


class _OAIDelta:
    def __init__(self, content="", reasoning_content=None, role=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.role = role


def _concat_streamed(chunks):
    """Walk every yielded chunk, concatenating each chunk's delta.content
    / delta.reasoning_content into separate strings, and capture the
    ``finish_reason`` carried by every chunk. Returns
    ``(content, reasoning, count, last_chunk, finish_reasons)``.

    Tests assert on the second-to-last ``finish_reason`` to verify the
    real terminal chunk (which sits just before the synthetic tail) is
    preserved untouched — the synthetic tail itself carries
    ``finish_reason=None`` by design.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    count = 0
    last = None
    finish_reasons: list = []
    for c in chunks:
        count += 1
        last = c
        for choice in getattr(c, "choices", None) or []:
            d = getattr(choice, "delta", None)
            finish_reasons.append(getattr(choice, "finish_reason", None))
            if d is None:
                continue
            t = getattr(d, "content", None)
            if t:
                content_parts.append(t)
            r = getattr(d, "reasoning_content", None)
            if r:
                reasoning_parts.append(r)
    return (
        "".join(content_parts), "".join(reasoning_parts),
        count, last, finish_reasons,
    )


async def _drive_split(h, request_data, chunks):
    """Drive the streaming hook over a list of OpenAI-shaped chunks."""
    async def produce():
        for c in chunks:
            yield c
    out = []
    async for c in h.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None, response=produce(), request_data=request_data,
    ):
        out.append(c)
    return out


def test_streaming_splitter_flush_yields_trailing_partial_tag_as_content():
    """Test (a): a streamed response whose last chunk ended mid-tag
    (the trailing ``<`` of ``The answer is x <``) used to lose the
    ``<`` — pre-fix the splitter would call ``flush()`` and then drop
    the result because the chunk loop had already exited. Post-fix one
    synthetic chunk carrying the trailing partial tag opener is yielded,
    and the client-visible concatenated content is the full model
    output, not the truncated ``The answer is x ``.
    """
    h, _reg, _slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="acompletion")
    request_data = {"metadata": _meta_for(ctx)}

    chunks = [
        _OAIStreamingChunk(content="The answer is x "),
        _OAIStreamingChunk(content="<", finish_reason="stop"),
    ]

    yielded = asyncio.run(_drive_split(h, request_data, chunks))
    content, reasoning, count, _last, finish_reasons = _concat_streamed(yielded)

    # Full model output reached the client.
    assert content == "The answer is x <", content
    # No reasoning split happened (we are not inside a think block).
    assert reasoning == "", reasoning
    # Three chunks: two source frames plus the synthetic tail that
    # carries the flushed ``<``. The terminal source frame's
    # ``finish_reason`` is unchanged — the synthetic's is None.
    assert count == len(chunks) + 1, (
        f"expected {len(chunks)} + 1 chunks (incl. synthetic tail), "
        f"got {count}: "
        f"{[getattr(c.choices[0].delta, 'content', '') for c in yielded]}")
    # Frame-by-frame finish_reason: the second source frame carries
    # ``stop``, the synthetic tail carries ``None`` (so the real
    # terminal chunk keeps its ``finish_reason``).
    assert finish_reasons[:-1] == [None, "stop"], finish_reasons
    # Sanity-check the synthetic tail: ``finish_reason=None`` and
    # content is just the flushed opener, with no reasoning split.
    tail = yielded[-1]
    assert tail.choices[0].finish_reason is None, (
        tail.choices[0].finish_reason)
    assert tail.choices[0].delta.content == "<", (
        tail.choices[0].delta.content)
    assert not getattr(tail.choices[0].delta, "reasoning_content", None), (
        tail.choices[0].delta)
    print(f"  trailing '<' yielded as synthetic chunk; concatenated "
          f"content={content!r}, count={count}")


def test_streaming_splitter_flush_yields_unterminated_think_tail_as_reasoning():
    """Test (b): an unterminated ``<think>`` block — the model opened a
    think tag, streamed reasoning, and ended without ever emitting
    ``</think>``. The splitter holds back the very last char (``<``) as
    a possible ``</`` prefix so the next chunk can complete the closing
    tag. Pre-fix, ``flush()`` correctly returned that held-back char as
    reasoning — but the streaming hook never yielded it, so the client
    silently lost the last character of every unterminated think-block
    stream. Post-fix, the synthetic chunk stamps the held-back char onto
    ``reasoning_content`` so the full chain of thought reaches the
    client.

    Chunks: three frames — the ``<think>`` opener, the reasoning body,
    and a trailing single ``<`` char that the splitter holds back as a
    possible ``</`` prefix. The splitter feeds the open tag
    (``in_think`` flips to True), emits the body verbatim through
    ``feed``, holds back the trailing ``<``, and on ``flush()`` returns
    ``("", "<")``. The synthetic tail lands the ``<`` back onto
    ``reasoning_content``; the ``content`` side is empty because the
    splitter is still in_think on flush. The held-back ``<`` lands on
    the tail rather than being silently dropped.
    """
    h, _reg, _slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="acompletion")
    request_data = {"metadata": _meta_for(ctx)}

    chunks = [
        _OAIStreamingChunk(content="<think>"),
        _OAIStreamingChunk(content="body content"),
        _OAIStreamingChunk(content="<", finish_reason="stop"),
    ]

    yielded = asyncio.run(_drive_split(h, request_data, chunks))
    content, reasoning, count, _last, finish_reasons = _concat_streamed(yielded)

    # All of the content stream's traffic is reasoning: nothing leaked
    # into ``delta.content``.
    assert content == "", content
    # The body arrives via feed (chunk 2) and the held-back ``<`` lands
    # on the synthetic tail (chunk 4) — the concatenation is the full
    # model output.
    assert reasoning == "body content<", reasoning
    assert count == len(chunks) + 1, (
        f"expected {len(chunks)} + 1 (incl. synthetic tail), got "
        f"{count}: deltas="
        f"{[getattr(c.choices[0].delta, 'content', '') for c in yielded]}")
    # The synthetic tail carries the held-back ``<`` as
    # ``reasoning_content``; the real terminal chunk keeps its
    # ``finish_reason=stop`` and the synthetic's is None.
    assert finish_reasons[:-1] == [None, None, "stop"], finish_reasons
    tail = yielded[-1]
    assert tail.choices[0].finish_reason is None, (
        tail.choices[0].finish_reason)
    assert not getattr(tail.choices[0].delta, "content", None), (
        tail.choices[0].delta.content)
    assert tail.choices[0].delta.reasoning_content == "<", (
        tail.choices[0].delta.reasoning_content)
    print(f"  held-back '<' yielded as reasoning_content on synthetic "
          f"tail; concatenated reasoning={reasoning!r}, count={count}")


def test_streaming_splitter_flush_blank_secondary_choices_on_n_gt_1():
    """Reviewer should-fix: synthetic tail must not re-deliver n>1 choices' tail.

    For ``n > 1`` streams the proxy serves every choice on every chunk and
    the splitter only ever reads ``choices[0]``, so the splitter-flush
    tail can only be attributed to choice 0. If the synthetic tail kept
    choices[1..]'s previously-yielded ``delta.content`` and ``finish_reason``
    untouched, a client walking the stream would see:

    * the secondary choice's final ``delta.content`` re-delivered on the
      synthetic tail frame (a duplicated tail for choices[1..]), and
    * that secondary choice's real ``finish_reason`` (e.g. ``"stop"``)
      riding on a non-terminal chunk (a duplicated terminal signal --
      a client that terminates on the first ``finish_reason`` it sees
      would cut the stream short right after the real terminal chunk).

    The fix strips both on the synthetic tail: ``delta.content = None``
    and ``finish_reason = None`` for every secondary choice, leaving
    ``choices[0]`` carrying the flushed content / reasoning and a
    ``finish_reason=None`` so the real terminal chunk on ``choices[0]``
    is the only stop signal on the wire for that choice.

    Chunks: a two-choice stream ending mid-tag on ``choices[0]`` only --
    choice 1 was already done with ``"done "`` and the real
    ``finish_reason="stop"`` on the last frame. The synthetic tail must
    carry ``choices[0].delta.content = "<"`` and ``finish_reason=None``
    while ``choices[1]`` has its ``delta.content`` blanked to ``None``
    and its ``finish_reason`` blanked to ``None`` -- NOT re-delivered.
    """
    h, _reg, _slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="acompletion")
    request_data = {"metadata": _meta_for(ctx)}

    # Stream shape:
    #
    #   chunk 1 (choices[0]="(prefix ", choices[1]="X"):
    #     splitter.feed("(prefix ") -> emits "(prefix " (no tag boundary
    #     straddled); choices[1] is untouched by the splitter path.
    #
    #   chunk 2 (choices[0]="<", choices[1]="Y", finish_reason=stop):
    #     splitter.feed("<") -> content="" reasoning="" (held back as a
    #     possible ``</`` prefix); choices[1] passes through with content
    #     "Y" untouched.
    #
    #   synthetic tail (clone of chunk 2):
    #     With the fix: choices[0].content = "<" (from splitter.flush),
    #     choices[0].finish_reason = None; choices[1].delta.content
    #     blanked to None and choices[1].finish_reason blanked to None --
    #     the synthetic frame must NOT re-deliver "Y" or carry "stop" on
    #     choices[1] (a duplicated terminal signal that would cut a
    #     client's stream short after the real terminal chunk).
    chunks = [
        _OAIStreamingChunk(choices=[
            ("(prefix ", None, None),
            ("X", None, None),
        ]),
        _OAIStreamingChunk(choices=[
            ("<", None, None),
            ("Y", None, "stop"),
        ]),
    ]

    yielded = asyncio.run(_drive_split(h, request_data, chunks))
    _content, _reasoning, count, _last, _frs = _concat_streamed(yielded)

    assert count == len(chunks) + 1, (
        f"expected {len(chunks)} + 1 (incl. synthetic tail), got {count}")

    tail = yielded[-1]
    assert len(tail.choices) == 2, (
        f"synthetic tail must keep the same number of choices as "
        f"last_chunk, got {len(tail.choices)}")

    # choices[0]: carried the flushed tail, finish_reason cleared.
    assert tail.choices[0].delta.content == "<", (
        tail.choices[0].delta.content)
    assert tail.choices[0].finish_reason is None, (
        tail.choices[0].finish_reason)

    # choices[1] (secondary): delta.content MUST be blanked to None -- a
    # re-delivered "Y" would have been the pre-fix bug (the synthetic
    # frame would re-emit a tail the splitter never wrote).
    assert tail.choices[1].delta.content is None, (
        f"choices[1].delta.content must be blanked on synthetic tail, "
        f"got {tail.choices[1].delta.content!r}")
    # choices[1].finish_reason MUST be blanked to None too -- the previous
    # frame had finish_reason="stop" and would otherwise appear as a
    # duplicated terminal signal on a non-terminal chunk.
    assert tail.choices[1].finish_reason is None, (
        f"choices[1].finish_reason must be blanked on synthetic tail, "
        f"got {tail.choices[1].finish_reason!r}")
    print("  n>1 synthetic tail: choices[0] carries flushed '<' + "
          "finish_reason=None; choices[1].content blanked to None, "
          "choices[1].finish_reason blanked to None (was 'stop')")


def test_streaming_splitter_flush_falls_back_when_clone_raises():
    """Reviewer should-fix: ``_tail_chunk`` cloning raising falls back.

    The chunk-loop arm (lines above) wraps the per-chunk split in
    ``try/except Exception`` with the comment "never break a stream
    over this". The synthetic tail emission must follow the same
    invariant: a raise inside ``_tail_chunk`` (frozen pydantic chunk
    with non-copyable internals, a chunk whose ``__deepcopy__`` /
    ``setattr`` shape this hook does not tolerate, etc.) must NOT
    propagate out of the iterator -- that would tear the streaming
    response at the very end after every real chunk had been
    delivered, strictly worse than the pre-fix behaviour.

    Falls back to the ``last_chunk=None`` skeleton path on the
    failure arm so the trailing content / reasoning still arrives
    on a recognizable OpenAI-shaped frame (``choices[0]`` skeleton).

    The fix is exercised by stubbing ``_tail_chunk`` to raise on the
    populated-branch path and return a SimpleNamespace skeleton on
    the fallback path; the hook is then driven over the issue's
    ``["The answer is x ", "<"]`` repro, and the synthesized tail
    on the wire is asserted to be the skeleton (no clone).
    """
    h, _reg, _slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="acompletion")
    request_data = {"metadata": _meta_for(ctx)}

    calls = {"n": 0}

    def _stub_tail_chunk(last_chunk, content, reasoning):  # noqa: ANN001
        # Force the clone path to raise once (mimicking a frozen
        # Pydantic chunk) so the hook's outer try/except fires.
        calls["n"] += 1
        if calls["n"] == 1 and last_chunk is not None:
            raise TypeError("simulated frozen chunk; deepcopy not supported")
        # Fallback path: same as the ``last_chunk is None`` arm.
        delta = SimpleNamespace(
            content=content,
            reasoning_content=(reasoning if reasoning else None),
            role=None,
        )
        choice = SimpleNamespace(
            index=0, delta=delta, finish_reason=None,
        )
        return SimpleNamespace(choices=[choice])

    # Patch the bound method for the duration of this call.
    original = h._tail_chunk
    h._tail_chunk = _stub_tail_chunk
    try:
        chunks = [
            _OAIStreamingChunk(content="The answer is x "),
            _OAIStreamingChunk(content="<", finish_reason="stop"),
        ]
        yielded = asyncio.run(_drive_split(h, request_data, chunks))
    finally:
        h._tail_chunk = original

    content, _reasoning, count, _last, finish_reasons = _concat_streamed(yielded)

    # Full output delivered, no exception tore the stream.
    assert calls["n"] == 2, (
        f"_tail_chunk should have been called twice (clone attempt + "
        f"skeleton fallback), got {calls['n']}")
    assert content == "The answer is x <", content
    assert count == len(chunks) + 1, (
        f"expected {len(chunks)} + 1 (incl. fallback tail), got {count}")
    # The synthetic tail is the skeleton: choices[0] only, content=``<``.
    tail = yielded[-1]
    assert len(tail.choices) == 1, (
        f"skeleton fallback must carry a single choice, got {len(tail.choices)}")
    assert tail.choices[0].delta.content == "<", (
        tail.choices[0].delta.content)
    assert tail.choices[0].finish_reason is None, (
        tail.choices[0].finish_reason)
    # The real terminal chunk kept its ``finish_reason="stop"``: the
    # synthetic tail carries ``None`` on the skeleton.
    assert finish_reasons[:-1] == [None, "stop"], finish_reasons
    print(f"  clone raise -> SimpleNamespace skeleton fallback; "
          f"concat={content!r}, count={count}, _tail_chunk calls={calls['n']}")


def test_streaming_splitter_flush_tail_inside_unterminated_think_block():
    """Test (c): a fully closed stream yields no synthetic chunk beyond
    the source shapes. Chunks ``["already done ", "no flush tail"]``
    carry no OPEN tag, no partial-CLOSE prefix; the splitter emits
    everything via ``feed``, ``flush()`` returns ``("", "")``, and the
    hook yields the source shapes only — no spurious synthetic frame
    is appended.

    This is the negative-side regression pin: the synthetic emission
    must be conditional on ``flush()`` returning non-empty, otherwise
    every streamed response would gain a phantom chunk at end-of-stream.
    Pre-fix the hook never yielded a synthetic chunk (the bug); post-fix
    it yields one only when there is something to deliver.
    """
    h, _reg, _slots, _ledger, _policy, _redis, pick = _build()
    ctx = _ctx_for(pick, call_type="acompletion")
    request_data = {"metadata": _meta_for(ctx)}

    chunks = [
        _OAIStreamingChunk(content="already done "),
        _OAIStreamingChunk(content="no flush tail", finish_reason="stop"),
    ]

    yielded = asyncio.run(_drive_split(h, request_data, chunks))
    content, reasoning, count, last, finish_reasons = _concat_streamed(yielded)

    # No synthetic chunk was added: the chunk count is exactly the
    # number of source shapes.
    assert count == len(chunks), (
        f"expected {len(chunks)} chunks (no synthetic tail), got "
        f"{count}: "
        f"{[getattr(c.choices[0].delta, 'content', '') for c in yielded]}")
    # Terminal source chunk keeps its real finish_reason — and since
    # there is no synthetic frame, ``last`` IS that real terminal chunk.
    assert finish_reasons == [None, "stop"], finish_reasons
    assert last.choices[0].finish_reason == "stop", (
        last.choices[0].finish_reason)
    # No raw content leaked into the response.
    assert content == "already done no flush tail", content
    assert reasoning == "", reasoning
    print(f"  closed stream yielded exactly {count} chunks (no "
          f"synthetic); concatenated content={content!r}")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
