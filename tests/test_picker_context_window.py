"""The context-window awareness introduced for issue #104.

Two concerns land here:

  1. The `load()` pre-fill: a model whose `context_window` is unset should pick
     up `litellm.model_cost.max_input_tokens` automatically, but only when the
     operator did not set a value of their own. The same shape as the existing
     `supports_images` pre-fill: litellm is a hint, the operator is the source
     of truth.

  2. The picker gate: with `settings.enforce_context_window = true`, an
     oversized request must skip a peer whose `context_window` cannot hold it
     and land on a fitting peer. Default off (the LiteLLM-side
     `context_window_fallbacks` already handles the common case), so the
     behaviour-preservation test pins today's "no gate" outcome too.

The pre-fill test is written defensively so a checkout without litellm
still exercises the contract: we inject a fake `litellm` module into
`sys.modules` (mirroring how test fixtures can avoid the ambient
litellm) and verify both the pre-fill and the operator-wins paths. The
gate tests build a registry from a fixture YAML that already sets
`context_window` per model — no litellm needed at all.

The conftest socket guard stays inert for these tests: nothing reaches a
provider, the picker never gets a real slot claim, and FakeRedis backs
the slot table. The `_runner` plain-script convention is followed: all
test_* functions live ABOVE the `if __name__ == "__main__":` block.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # so `from switchyard...` works
sys.path.insert(0, HERE)                    # so `import conftest` works under plain `python3`

import conftest  # noqa: F401  (socket guard for plain-script mode)

from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()

from switchyard import models  # noqa: E402
from switchyard.picker import (  # noqa: E402
    LaneSaturated,
    Picker,
    _estimate_input_tokens,
)
from switchyard.slots import SlotTable  # noqa: E402
from tests.fake_redis import FakeRedis  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def _build(reg):
    redis = FakeRedis()
    slots = SlotTable(redis, reg.settings.inflight_max_age_seconds)
    return reg, slots, Picker(reg, slots)


# ---------------------------------------------------------------------------
# Pre-fill: litellm.model_cost.max_input_tokens lands on unset models,
# operator-set values are never overwritten.
# ---------------------------------------------------------------------------


class _FakeModelCost:
    """In-memory stand-in for litellm.model_cost. Mapping-like iteration
    surface is all `litellm_known_context_windows` reads, plus a few entries
    shaped to exercise the bool-subclass and negative-value rejects."""

    def __init__(self, entries: dict[str, dict]):
        self._entries = entries

    def items(self):
        return self._entries.items()


class _FakeLitellmModule(types.ModuleType):
    """The minimum surface `litellm_known_context_windows` imports: a module
    object whose `model_cost` attribute is a Mapping with .items(). Both
    `import litellm` and `from litellm import model_cost` resolve to the same
    module, so a sys.modules['litellm'] = ... stub satisfies both.
    """

    def __init__(self, model_cost):
        super().__init__("litellm")
        self.model_cost = model_cost


def _stub_litellm(monkey):
    """Inject a fake `litellm` module into sys.modules so
    `litellm_known_context_windows` finds a non-empty catalog. Returns a
    restore() callable that drops the stub; the caller MUST call restore()
    even on the failure path so a later test (or this test's later
    assertion) sees the original `litellm` import behaviour.
    """
    saved = sys.modules.get("litellm")
    entries = {
        # The case under test: a positive integer pre-fills `context_window`.
        "fake/prefill-model": {"max_input_tokens": 12345},
        # Negative and zero values must be skipped (a 0-token window is not
        # useful for a gate), so the pre-fill leaves the model unset.
        "fake/zero-window": {"max_input_tokens": 0},
        "fake/neg-window": {"max_input_tokens": -1},
        # `bool` is an int subclass in Python; this entry exists so the
        # helper's explicit `isinstance(value, bool)` guard has something to
        # refuse. Without that branch the stub silently treats True as 1
        # and False as 0.
        "fake/bool-window": {"max_input_tokens": True},
        # Non-int values must also be skipped.
        "fake/str-window": {"max_input_tokens": "4096"},
    }
    module = _FakeLitellmModule(_FakeModelCost(entries))
    sys.modules["litellm"] = module
    monkey["litellm"] = saved
    return module


def _restore_litellm(monkey):
    saved = monkey.pop("litellm", None)
    if saved is None:
        sys.modules.pop("litellm", None)
    else:
        sys.modules["litellm"] = saved


def _write_fixture_yaml(tmp,
                        plan_models: dict[str, dict],
                        lane_order: list[str] | None = None) -> str:
    """Materialise a minimal plans.yaml with one plan and the requested model
    bodies. The shape mirrors config/plans.example.yaml closely enough that
    `load()`'s parser accepts it; only the bits the picker gate cares about
    need to be real. Returns the file path.

    `lane_order` defaults to the model keys in declaration order, so the
    common case (one model per test) does not require a separate lane shape.
    Pass an explicit list when the caller wants a specific order — the gate
    tests do, because the assertion depends on which peer overflows first.
    """
    if lane_order is None:
        lane_order = [f"gateplan/{k}" for k in plan_models]
    lines: list[str] = []
    lines.append("plans:")
    lines.append("  gateplan:")
    lines.append("    label: Gate Plan")
    lines.append("    max_parallel: 4")
    lines.append("    api_base: http://example.invalid/v1")
    lines.append("    api_key: fake")
    lines.append("    quotas:")
    lines.append("      - name: weekly")
    lines.append("        role: target")
    lines.append("        kind: tokens")
    lines.append("        period: week")
    lines.append("        source: none")
    lines.append("    models:")
    for key, body in plan_models.items():
        lines.append(f"      {key}:")
        lines.append(f"        model: {body['model']}")
        for k, v in body.items():
            if k == "model":
                continue
            if k == "context_window":
                lines.append(f"        context_window: {v}")
            elif k == "max_parallel":
                lines.append(f"        max_parallel: {v}")
            elif k == "supports_images":
                lines.append(
                    f"        supports_images: {'true' if v else 'false'}")
    lines.append("lanes:")
    lines.append("  saturate:")
    lines.append("    label: Saturate")
    lines.append("    order:")
    for ref in lane_order:
        lines.append(f"      - {ref}")
    lines.append("    tail: []")
    lines.append("")
    body = "\n".join(lines)
    path = os.path.join(str(tmp), "plans.yaml")
    with open(path, "w") as fh:
        fh.write(body)
    return path


def test_model_context_window_prefilled_from_litellm():
    """A model with no operator `context_window` picks up
    `litellm.model_cost.max_input_tokens`; an operator-set value wins.

    Written defensively: the helper may legitimately return `{}` on a
    checkout without litellm (and we have no litellm in this test
    environment), so we inject a fake `litellm` module into `sys.modules`
    so the helper has a non-empty catalog to read from. The same loader
    code path runs either way — lazy import + broad except — so the
    contract under test (operator wins, litellm hints the rest) is
    what travels, not the catalog itself.
    """
    monkey: dict = {}
    try:
        _stub_litellm(monkey)

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture_yaml(tmp, {
                "unset-model": {"model": "fake/prefill-model"},
                "operator-wins": {
                    "model": "fake/prefill-model",
                    "context_window": 999,
                },
                "no-match": {"model": "fake/unknown-model"},
                "negative": {"model": "fake/neg-window"},
                "zero": {"model": "fake/zero-window"},
                "bool": {"model": "fake/bool-window"},
                "string": {"model": "fake/str-window"},
            })
            reg = models.load(path)

        unset = reg.plans["gateplan"].models["unset-model"]
        operator = reg.plans["gateplan"].models["operator-wins"]
        no_match = reg.plans["gateplan"].models["no-match"]
        negative = reg.plans["gateplan"].models["negative"]
        zero = reg.plans["gateplan"].models["zero"]
        bool_entry = reg.plans["gateplan"].models["bool"]
        string_entry = reg.plans["gateplan"].models["string"]

        # The pre-fill case under test: litellm max_input_tokens lands.
        assert unset.context_window == 12345, unset
        # Operator-set values are never overwritten.
        assert operator.context_window == 999, operator
        # A model whose key+model string do not match any litellm entry
        # stays None (no spurious 0, no exception).
        assert no_match.context_window is None, no_match
        # Helper refuses negative / zero / bool / non-int values, so the
        # corresponding model stays None rather than landing on something
        # the gate could compare against.
        assert negative.context_window is None, negative
        assert zero.context_window is None, zero
        assert bool_entry.context_window is None, bool_entry
        assert string_entry.context_window is None, string_entry

        # Sanity: the helper still reads `litellm.model_cost.max_input_tokens`
        # as documented.
        from switchyard.models import litellm_known_context_windows
        catalog = litellm_known_context_windows()
        assert catalog.get("fake/prefill-model") == 12345, catalog
        assert "fake/neg-window" not in catalog, catalog
        assert "fake/zero-window" not in catalog, catalog
        assert "fake/bool-window" not in catalog, catalog
        assert "fake/str-window" not in catalog, catalog

        print(f"  prefill ok: unset={unset.context_window}, operator="
              f"{operator.context_window}, no_match={no_match.context_window}; "
              f"refused negatives/zero/bool/str entries as expected")
    finally:
        _restore_litellm(monkey)


# ---------------------------------------------------------------------------
# The picker gate: when `enforce_context_window` is true and a peer cannot
# hold the request, the body walk spills to the next peer (or saturates if
# no peer fits). The default `enforce_context_window=False` must preserve
# today's behaviour bit-for-bit.
# ---------------------------------------------------------------------------


def _enable_gate(path: str) -> None:
    """Patch `enforce_context_window: true` into the fixture's `settings`
    block. Done by string substitution rather than a re-parse so the test
    stays a single-pass fixture with one source of truth for its shape.
    """
    with open(path) as fh:
        body = fh.read()
    body = body.replace(
        "plans:",
        "settings:\n  enforce_context_window: true\nplans:",
        1,
    )
    with open(path, "w") as fh:
        fh.write(body)


def test_gate_returns_none_when_no_peer_fits():
    """Single-member lane, tiny context window, oversized messages — the
    picker cannot land the request anywhere and `pick()` raises
    `LaneSaturated`. The skip reason on the raised exception names the
    context-window gate explicitly so the board can show why the lane
    refused.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture_yaml(tmp, {
            "small": {"model": "fake/small", "context_window": 4,
                      "max_parallel": 1},
            # Second member still has a tiny window; the body walk visits
            # it after `small` and overflows there too, then raises.
            "other": {"model": "fake/other", "context_window": 4,
                      "max_parallel": 1},
        })
        _enable_gate(path)
        reg = models.load(path)
        assert reg.settings.enforce_context_window is True, reg.settings

        async def go():
            reg_, slots, picker = _build(reg)
            # ~24k bytes of message payload: well above 4-token windows on
            # both members. The estimate is bytes // 4, so 24000 -> 6000
            # estimated tokens, which exceeds both `context_window: 4`.
            big = "x" * 24000
            data = {"messages": [{"role": "user", "content": big}]}
            try:
                await picker.pick("saturate", None, data=data)
            except LaneSaturated as exc:
                return str(exc)

        refused = run(go())

    assert refused is not None, "pick() should have raised LaneSaturated"
    assert "context_overflow" in refused, refused
    print(f"  oversized request refused: {refused}")


def test_gate_skips_overflow_lands_on_fitting_peer():
    """Two peers; the first has a tiny `context_window`, the second is
    large. An oversized request overflows the first peer (recorded in
    `ctx.skipped`), lands on the second. The body walk continues to the
    next peer exactly like every other gate — no wholesale filtering, so
    a single-member lane would still saturate instead of misrouting.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture_yaml(tmp, {
            "small": {"model": "fake/small", "context_window": 4,
                      "max_parallel": 1},
            "large": {"model": "fake/large", "context_window": 1_000_000,
                      "max_parallel": 1},
        })
        _enable_gate(path)
        reg = models.load(path)
        assert reg.settings.enforce_context_window is True

        async def go():
            reg_, slots, picker = _build(reg)
            big = "x" * 24000   # ~6000 estimated tokens — fits in the large
                                # peer (1M-token window) but not the small
                                # one (4-token window).
            data = {"messages": [{"role": "user", "content": big}]}
            pick = await picker.pick("saturate", None, data=data)
            return pick.ref, pick.considered

        ref, considered = run(go())

    assert ref == "gateplan/large", (ref, considered)
    assert any("gateplan/small(context_overflow:" in s for s in considered), (
        ref, considered)
    print(f"  oversized request: {ref} (skipped "
          f"{[s for s in considered if 'context_overflow' in s]})")


def test_gate_disabled_by_default():
    """`enforce_context_window` defaults to False. With the gate off, an
    oversized request lands on whichever peer the body walk reaches first
    — the same outcome as before the gate existed. Behaviour-preservation
    pin: a regression that flips the default to True breaks every existing
    operator's capacity picture.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture_yaml(tmp, {
            "small": {"model": "fake/small", "context_window": 4,
                      "max_parallel": 1},
            "large": {"model": "fake/large", "context_window": 1_000_000,
                      "max_parallel": 1},
        })
        reg = models.load(path)
        # The shipped default — confirmed.
        assert reg.settings.enforce_context_window is False, reg.settings

        async def go():
            reg_, slots, picker = _build(reg)
            big = "x" * 24000
            data = {"messages": [{"role": "user", "content": big}]}
            # Body walk: lands on the first peer (gateplan/small) because
            # the gate is off and that peer's slot is free. The point of
            # this test is that the gate did NOT skip past it.
            pick = await picker.pick("saturate", None, data=data)
            return pick.ref, pick.considered

        ref, considered = run(go())

    assert ref == "gateplan/small", (ref, considered)
    # No `context_overflow` skip recorded: the gate did not fire.
    assert not any("context_overflow" in s for s in considered), considered
    print(f"  gate off (default): picked {ref}, no context_overflow skip "
          f"recorded")


def test_estimate_skipped_when_gate_off():
    """A default-off operator must pay zero `json.dumps` cost on every
    request. PR #330 cycle-3: the eager `_estimate_input_tokens(data)` at
    `pick()` ctx construction was unconditional, so the bytes->tokens
    heuristic ran on every /v1/messages call regardless of opt-in. The
    fix computes lazily via `_ensure_estimated` only inside the gate
    branches (`_visit_ref` / `_affinity`), and only when the gate would
    actually fire (`enforce_context_window` on AND `model.context_window`
    set). This regression test spies on `_estimate_input_tokens` and
    asserts it never runs for a default-off operator.

    The spy is bound by replacing the module attribute
    `picker_mod._estimate_input_tokens`; that's what makes the spy
    effective, since `_ensure_estimated` resolves the helper through
    `switchyard.picker`'s module global (not a bound closure), and the
    restore in `finally` leaves the module clean for sibling tests.
    """
    import switchyard.picker as picker_mod
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture_yaml(tmp, {
            "small": {"model": "fake/small", "context_window": 4,
                      "max_parallel": 1},
            "large": {"model": "fake/large", "context_window": 1_000_000,
                      "max_parallel": 1},
        })
        reg = models.load(path)
        assert reg.settings.enforce_context_window is False, reg.settings
        _, _, picker_inst = _build(reg)

        original = picker_mod._estimate_input_tokens
        calls: list[dict | None] = []

        def spy(data):
            calls.append(data)
            return original(data)

        try:
            picker_mod._estimate_input_tokens = spy
            big = "x" * 24000
            data = {"messages": [{"role": "user", "content": big}]}
            pick = run(picker_inst.pick("saturate", None, data=data))
            assert pick.ref in ("gateplan/small", "gateplan/large"), pick
        finally:
            picker_mod._estimate_input_tokens = original

    assert calls == [], (
        f"default-off operator must not trigger _estimate_input_tokens; "
        f"got {len(calls)} call(s): the gate is off, every peer carries "
        f"a context_window, but `enforce_context_window` is the operator "
        f"opt-in (default False), so the heuristic must not run. Each "
        f"call is one synchronous json.dumps of the full message list.")
    print(f"  lazy estimate: 0 json.dumps for default-off operator "
          f"(gate never fires); spy recorded {len(calls)} call(s)")


def test_context_window_fallbacks_still_in_generated_config():
    """Sanity: the new gate does NOT regress `context_window_fallbacks` in
    the generated litellm config — the field the existing
    `tests/test_routing.py:1194-1203` regression locks. Reading the
    registry the loader produces is enough; we do not need a full
    `gen_litellm.build` round-trip to confirm the registry is unchanged.
    """
    reg = models.load()
    # `gen_litellm.py` reads `Registry.lane_members(...)` and the per-model
    # `context_window`; that field is still on every Model dataclass and
    # still defaults to None, and the loader's pre-fill only fills it when
    # litellm reports a max_input_tokens. The presence of a `local-box/qwen`
    # context_window in the example config confirms the field path is
    # intact end-to-end.
    qwen = reg.plans["local-box"].models["qwen"]
    assert qwen.context_window == 262144, qwen
    # And `gen_litellm` continues to read the field — verify by building
    # the generated config and asserting the key is present (matching
    # `tests/test_routing.py:1194-1203`'s pin).
    from switchyard import gen_litellm
    cfg = gen_litellm.build(os.environ["SWITCHYARD_PLANS"])
    rs = cfg["router_settings"]
    assert "context_window_fallbacks" in rs, (
        "tests/test_routing.py:1194-1203 regressed: "
        f"context_window_fallbacks missing from {sorted(rs)}")
    # And every named fallback target is a real deployment — same contract
    # the routing test pins, repeated here so a regression in either file
    # surfaces in either suite.
    names = {m["model_name"] for m in cfg["model_list"]}
    for entry in rs["context_window_fallbacks"]:
        for src, targets in entry.items():
            for t in targets:
                assert t in names, (src, t, sorted(names)[:5])
    print(f"  registry intact: local-box/qwen.context_window="
          f"{qwen.context_window}; generated config carries "
          f"{len(rs['context_window_fallbacks'])} context_window_fallbacks "
          f"entries (every target a real deployment)")


def test_estimate_input_tokens_returns_zero_for_missing_messages():
    """`_estimate_input_tokens` returns 0 when there are no messages to
    estimate, so the gate never fires spuriously for callers that pass
    `data=None` or `data={}`. The four branches: None, non-dict, dict
    without messages, dict with messages=non-list.
    """
    assert _estimate_input_tokens(None) == 0
    assert _estimate_input_tokens("not a dict") == 0
    assert _estimate_input_tokens({}) == 0
    assert _estimate_input_tokens({"messages": "not a list"}) == 0
    # A real messages list returns a positive integer (bytes // 4).
    n = _estimate_input_tokens({"messages": [{"role": "user", "content": "hi"}]})
    assert isinstance(n, int) and n > 0, n
    print(f"  estimate: None/empty -> 0; 1 short message -> {n} tokens")


# ---------------------------------------------------------------------------
# End-to-end hooks.async_pre_call_hook integration tests for issue #104.
#
# PR #330 review caught a wiring gap: the new context-window gate never
# fired from `/v1/messages` because switchyard/hooks.py:465-467 called
# picker.pick / pick_direct positionally without forwarding `data`. The
# tests below drive the *production* request path through
# SwitchyardHandler.async_pre_call_hook so a future regression that drops
# `data=data` from the call sites fails here instead of in production.
#
# SwitchyardHandler is constructed via __new__ + __dict__ injection so we
# avoid the real __init__'s redis/registry/watcher wiring (no real Redis,
# no plans-file mtime stat, no background threads). The pattern mirrors
# tests/test_routing.py:493-501.
# ---------------------------------------------------------------------------


def _build_hook(reg):
    """Build a SwitchyardHandler with a custom registry and FakeRedis.

    Bypasses `__init__` (which would re-load plans.yaml, claim a Redis
    client, start the file watcher, and launch the heartbeat loop). Used
    only by the integration tests below; the picker is the same object
    the production gateway holds.
    """
    from switchyard.hooks import SwitchyardHandler
    from switchyard.policy import CapacityPolicy
    from switchyard.usage import Ledger
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
    h.__dict__["_beats"] = {}
    return h


def _capture_saturated(exc) -> str | None:
    """The hook converts `LaneSaturated` into a 429 `HTTPException`. Return
    the refused message either way so the assertion can check for
    `context_overflow` without caring about the transport.
    """
    if isinstance(exc, LaneSaturated):
        return str(exc)
    from fastapi import HTTPException
    if isinstance(exc, HTTPException) and exc.status_code == 429:
        return str(exc.detail)
    return None


def _hook_log_lines(h, level: int, key: str):
    """Capture every `switchyard` log line during `body()`. The hook emits
    the gate's skip reason on its pick log line (hooks.py:564-565 emits
    `skipped=...` after the picked model), which is the path the
    integration tests assert against.
    """
    import logging as _logging
    captured: list[_logging.LogRecord] = []
    handler = _logging.Handler()
    handler.emit = captured.append
    log = _logging.getLogger("switchyard")
    prior_level = log.level
    log.setLevel(level)
    log.addHandler(handler)

    async def runner(coro):
        # `coro` is already a coroutine object (the test's `async def go`
        # body) — await it directly. Wrap the log handler so the test
        # does not see records from the next test even on failure.
        try:
            return await coro
        finally:
            log.removeHandler(handler)
            log.setLevel(prior_level)

    async def run_body(coro):
        result = await runner(coro)
        messages = [r.getMessage() for r in captured]
        return result, messages

    return run_body


def test_async_pre_call_hook_lane_skips_overflow_picks_fitting_peer():
    """End-to-end: the picker must consult the request body via the hook.

    Drives SwitchyardHandler.async_pre_call_hook with a lane name and an
    oversized `messages` payload. The small peer should be skipped (the
    picker logs `context_overflow` for it and the request lands on the
    larger peer instead). Without the `data=data` wiring at
    switchyard/hooks.py:472-475, `ctx.data` would be None, the estimate
    would be 0, and the picker would land on the small peer -- the exact
    regression PR #330 review caught.

    The hook stamps `pick.considered` onto its own log line
    (hooks.py:564-565 emits `skipped=...` after the picked model), so the
    picker's gate reason is observable through the logs without needing
    direct access to the Pick object.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture_yaml(tmp, {
            "small": {"model": "fake/small", "context_window": 4,
                      "max_parallel": 1},
            "large": {"model": "fake/large", "context_window": 1_000_000,
                      "max_parallel": 1},
        })
        _enable_gate(path)
        reg = models.load(path)
        h = _build_hook(reg)

        async def go():
            data = {
                "model": "saturate",
                "messages": [{"role": "user", "content": "x" * 24000}],
            }
            return await h.async_pre_call_hook(None, None, data, "acompletion")

        run_body = _hook_log_lines(h, 20, "skipped=")
        data, messages = run(run_body(go()))
    ctx = data["metadata"]["switchyard"]
    # Release the slot so the suite does not strand a fake inflight.
    run(h.picker.release(ctx["plan"], ctx["request_id"], ctx["model"]))
    # Pick log line names the picked model AND the skipped reasons. The
    # overflow reason on the small peer is the load-bearing assertion.
    overflow_lines = [m for m in messages
                      if "gateplan/small(context_overflow:" in m]
    assert any("gateplan/large" in m for m in overflow_lines), overflow_lines
    print(f"  hooks path (lane): picked {ctx['model']}; overflow lines="
          f"{overflow_lines}")


def test_async_pre_call_hook_single_peer_lane_returns_429_with_overflow():
    """End-to-end refusal: every lane member's context_window cannot hold
    the request body, so the hook surfaces the gate's reason through
    async_pre_call_hook as a 429.

    Mirrors the existing `tests/test_routing.py:925-944` shape: catch
    both the `LaneSaturated` form and its `HTTPException` wrapper, take
    the refused string, and assert on the reason.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture_yaml(tmp, {
            "small": {"model": "fake/small", "context_window": 4,
                      "max_parallel": 1},
            "other": {"model": "fake/other", "context_window": 4,
                      "max_parallel": 1},
        })
        _enable_gate(path)
        reg = models.load(path)
        h = _build_hook(reg)

        async def go():
            data = {
                "model": "saturate",
                "messages": [{"role": "user", "content": "x" * 24000}],
            }
            return await h.async_pre_call_hook(
                None, None, data, "acompletion")

        refused: str | None = None
        try:
            run(go())
        except Exception as exc:
            refused = _capture_saturated(exc)
    assert refused is not None, "hook should have refused the oversized request"
    assert "context_overflow" in refused, refused
    print(f"  hooks path (lane, no peer fits): {refused}")


def test_async_pre_call_hook_deployment_path_routes_data_into_pick_direct():
    """End-to-end for the direct-routed branch: a caller names a
    deployment, not a lane. The hook resolves the model name and calls
    `picker.pick_direct` (which has no body walk). The same `data=data`
    wiring at switchyard/hooks.py:472 must reach that call site too --
    the unit tests above only cover `_visit_ref`, not `pick_direct`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture_yaml(tmp, {
            "small": {"model": "fake/small", "context_window": 4,
                      "max_parallel": 1},
        })
        _enable_gate(path)
        reg = models.load(path)
        # The deployment shape is `sy.<plan_key>.<model_key>` per
        # `switchyard/models.py:159`. The fixture plan/model keys are
        # `gateplan`/`small`, so the deployment name is `sy.gateplan.small`.
        deployment = "sy.gateplan.small"
        assert reg.model_for_deployment(deployment) is not None, deployment
        h = _build_hook(reg)

        async def go():
            data = {
                "model": deployment,
                "messages": [{"role": "user", "content": "x" * 24000}],
            }
            return await h.async_pre_call_hook(
                None, None, data, "acompletion")

        refused: str | None = None
        try:
            run(go())
        except Exception as exc:
            refused = _capture_saturated(exc)
    assert refused is not None, "hook should have refused via pick_direct"
    assert "context_overflow" in refused, refused
    print(f"  hooks path (deployment): {refused}")


def test_async_pre_call_hook_drops_sticky_lease_when_body_outgrows_peer():
    """The `_affinity` gate (PR #330 should-fix #2). First call carries a
    short body and lands on a peer with a small context_window. The
    hook sets the lease via `_visit_ref`. Second call on the same
    session sends an oversized body; the affinity gate at
    `switchyard/picker.py:973-985` must drop the lease and let the body
    walk re-place the session on a larger-context peer, mirroring the
    image spill precedent at `switchyard/picker.py:957-967`.

    This is the path that PR #330's six unit tests could not exercise --
    `_affinity` returns before `_visit_ref` so the body-walk gate never
    sees the request. Driving it through the hook exercises the full
    request path: a real lease, a real slot claim, a real re-pick.

    The session ID is whatever `hooks.derive_session` returns for a
    header-only `x-switchyard-session: sess-grow`; that's `h:sess-grow`
    (the `h:` prefix the session module adds to header-derived IDs).
    """
    with tempfile.TemporaryDirectory() as tmp:
        # First plan member has a tiny context_window; second is generous.
        # The first turn's tiny body fits in the tiny peer, so the lease
        # lands there. The second turn's oversized body triggers the
        # affinity gate, which drops the lease and re-places on the
        # generous peer.
        path = _write_fixture_yaml(tmp, {
            "small": {"model": "fake/small", "context_window": 8,
                      "max_parallel": 1},
            "large": {"model": "fake/large", "context_window": 1_000_000,
                      "max_parallel": 1},
        })
        _enable_gate(path)
        reg = models.load(path)
        h = _build_hook(reg)
        session_id = "h:sess-grow"

        async def go():
            # Turn 1: short body -> hook lands on the small peer
            # (`gateplan/small` is the lane's declared order), the lease
            # is set for the session ID.
            data1 = {
                "model": "saturate",
                "messages": [{"role": "user", "content": "hi"}],
                "proxy_server_request": {
                    "headers": {"x-switchyard-session": "sess-grow"},
                },
            }
            await h.async_pre_call_hook(None, None, data1, "acompletion")
            ctx1 = data1["metadata"]["switchyard"]
            held = ctx1["model"]
            lease_before = await h.slots.get_lease(session_id)
            # Stop the heartbeat before the second turn fires so it does
            # not race the affinity-drop test's own set_lease call.
            beat = h._beats.pop(ctx1["request_id"], None)
            if beat is not None:
                beat.cancel()
                try:
                    await beat
                except BaseException:
                    pass
            # Release turn 1's slot too so the suite does not strand a
            # fake inflight on `gateplan/small`. Mirrors the sibling
            # `test_async_pre_call_hook_lane_skips_overflow_picks_fitting_peer`'s
            # hygiene that the cycle-2 review called out.
            await h.picker.release(
                ctx1["plan"], ctx1["request_id"], ctx1["model"])

            # Turn 2: same session, oversized body. The affinity gate
            # must see `context_overflow` on the held peer, drop the
            # lease, and fall through to the body walk, which lands on
            # the OTHER peer (the only one whose window fits).
            data2 = {
                "model": "saturate",
                "messages": [{"role": "user", "content": "x" * 24000}],
                "proxy_server_request": {
                    "headers": {"x-switchyard-session": "sess-grow"},
                },
            }
            await h.async_pre_call_hook(None, None, data2, "acompletion")
            ctx2 = data2["metadata"]["switchyard"]
            new_held = ctx2["model"]

            # Release the second slot so the suite does not strand a
            # fake inflight.
            await h.picker.release(ctx2["plan"], ctx2["request_id"], ctx2["model"])
            return held, lease_before, new_held

        held, lease_before, new_held = run(go())
    assert held == "gateplan/small", held
    assert lease_before == "gateplan/small", (lease_before, held)
    assert new_held == "gateplan/large", new_held
    print(f"  affine path: turn 1 leased {held}, turn 2 spilled to "
          f"{new_held} (lease dropped on overflow)")


def test_async_pre_call_hook_affinity_drop_no_op_when_lease_already_moved():
    """Regression for PR #330 cycle-2 (and cycle-3) reviewer findings.

    The affinity gate at `switchyard/picker.py:1041-1049` must use
    `drop_lease_if` (CAS) on the gate, not blind `drop_lease`. A
    parallel same-session turn (drain migration, body walk on a
    different request) can `set_lease` a new value between the
    picker's `held = get_lease(...)` read and the gate's drop; a
    blind DEL would clobber that turn's healthy lease including its
    injection marker and per-plan reverse-index entry (slots.py:514-527).

    Cycle-3 noted that the previous cycle-2 test exercised the
    `drop_lease_if` primitive directly, bypassing the gate entirely --
    a regression to blind `drop_lease` at the gate site would still
    pass it. This test fixes that by DRIVING the gate: turn 1 sticks
    the session to the small peer (via `async_pre_call_hook`, the
    production entry point), then we wrap `slots.get_lease` to also
    `set_lease(large)` as a side effect (simulating the parallel
    writer winning between `_affinity`'s read and the gate's drop).
    Turn 2 then runs through `Picker.pick` directly -- the hook's
    exact picker call with the session pre-resolved (the hook would
    just resolve the same session id and call this), so the affinity
    gate path it tests is bit-for-bit the production one. The
    earlier "via async_pre_call_hook" wording would have implied the
    hook surface was under test here when only turn 1 exercises it;
    this rewrite is behaviour-equivalent and clearer about what the
    test actually drives.

    Disambiguator: `pick.considered` records skipped refs. In the
    clean (non-racy) affinity path the `_affinity` gate's CAS succeeds,
    its skip appends, AND the body walk's `_visit_ref` overflow gate
    ALSO appends -- so `considered` lists the `context_overflow` skip
    for `gateplan/small` TWICE. In the racy path the affinity CAS
    no-ops (correct: do not advertise a skip that did not happen),
    `_affinity` does NOT append, and only the body walk appends --
    so `considered` lists it ONCE. A blind-DEL regression would
    unconditionally append in `_affinity`, producing the duplicate
    count and failing this assertion.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture_yaml(tmp, {
            "small": {"model": "fake/small", "context_window": 8,
                      "max_parallel": 1},
            "large": {"model": "fake/large", "context_window": 1_000_000,
                      "max_parallel": 1},
        })
        _enable_gate(path)
        reg = models.load(path)
        h = _build_hook(reg)
        session_id = "h:sess-par"

        async def go():
            # Turn 1: short body sticks the session to the small peer.
            data1 = {
                "model": "saturate",
                "messages": [{"role": "user", "content": "hi"}],
                "proxy_server_request": {
                    "headers": {"x-switchyard-session": "sess-par"},
                },
            }
            await h.async_pre_call_hook(None, None, data1, "acompletion")
            ctx1 = data1["metadata"]["switchyard"]
            assert ctx1["model"] == "gateplan/small", ctx1["model"]
            assert (await h.slots.get_lease(session_id)) == "gateplan/small"

            # Stop the heartbeat before the simulated parallel turn
            # touches the lease, then release turn 1's slot so a
            # parallel `set_lease` is not racing a fresh claim.
            beat = h._beats.pop(ctx1["request_id"], None)
            if beat is not None:
                beat.cancel()
                try:
                    await beat
                except BaseException:
                    pass
            await h.picker.release(
                ctx1["plan"], ctx1["request_id"], ctx1["model"])

            # Wrap `slots.get_lease` so the picker reads `small` (its
            # decision-time value) but the live lease in Redis is
            # flipped to `large` by the parallel writer. This is the
            # exact race the gate's CAS discipline must survive: by
            # the time `_affinity` reaches `drop_lease_if`, the live
            # value is no longer the one it read.
            real_get_lease = h.slots.get_lease
            racy_done = False

            async def racy_get_lease(session):
                nonlocal racy_done
                held = await real_get_lease(session)
                if not racy_done and held == "gateplan/small":
                    await h.slots.set_lease(
                        session, "gateplan/large",
                        reg.settings.lease_ttl_seconds, "gateplan")
                    racy_done = True
                return held

            h.slots.get_lease = racy_get_lease
            try:
                # Turn 2: oversized body through the hook. `_affinity`
                # reads `held = small` (our wrapper), proceeds through
                # the liveness / draining checks, reaches the gate,
                # and tries `drop_lease_if(session, "small")` -- but the
                # live value is `large` now, so the CAS no-ops.
                data2 = {
                    "model": "saturate",
                    "messages": [{"role": "user", "content": "x" * 24000}],
                    "proxy_server_request": {
                        "headers": {"x-switchyard-session": "sess-par"},
                    },
                }
                pick = await h.picker.pick(
                    "saturate", session_id, data=data2)
                lease_after = await h.slots.get_lease(session_id)
                await h.picker.release(
                    pick.plan.key, pick.request_id, pick.model.ref)
                return pick.ref, list(pick.considered), lease_after
            finally:
                h.slots.get_lease = real_get_lease

        ref, considered, lease_after = run(go())
    # The body walk has to land on the fitting peer (no lease to honor
    # after the CAS no-op, so `_visit_ref` re-runs the body walk and
    # its own overflow gate records the skip).
    assert ref == "gateplan/large", (ref, considered, lease_after)
    # Critical: lease survives on large. A blind-DEL regression would
    # have wiped it (the parallel writer's injection marker and
    # reverse-index entry); the body walk's re-lease would put it
    # back, but the test would not catch that. The check below IS the
    # disambiguator:
    overflow_skips = [s for s in considered if "context_overflow" in s]
    assert len(overflow_skips) == 1, (
        f"racy affinity path should record the overflow skip ONCE "
        f"(the CAS no-op path in _affinity does not append, only the "
        f"body walk's _visit_ref gate appends). A blind-DEL regression "
        f"would record it TWICE -- one from _affinity, one from "
        f"_visit_ref. got considered={considered}")
    assert overflow_skips[0] == (
        "gateplan/small(context_overflow:6008>8)"
    ), overflow_skips
    # And the parallel writer's lease on large survives the gate. Body
    # walk overwrote it with a fresh value during re-place, but the
    # result is the live lease sitting on the large peer (the body
    # walk re-leased on its own claim).
    assert lease_after == "gateplan/large", lease_after
    print(f"  affinity CAS regression: racy parallel turn -> "
          f"`_affinity` no-op'd (no blind DEL), single overflow skip "
          f"recorded by body walk; pick={ref}, considered={considered}, "
          f"lease={lease_after}")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
