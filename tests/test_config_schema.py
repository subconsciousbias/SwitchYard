"""Load-time validation of `max_parallel` and `max_parallel_ceiling`.

The loader used to coerce these with `int(body[k]) if body.get(k) else None`
and friends: a `0` silently dropped to `None`, a non-int string like
`max_parallel: "two"` raised a bare `ValueError` from `int()` without the
plan key, and a `max_parallel_ceiling: 0` quietly disabled a ceiling that
was meant to be set. The cost was real: an `asyncio.Semaphore(-3)` crashes
the gateway on the first request against the affected plan, the watcher
keeps the previous registry so the bad value never gets a chance to
re-render, and the operator has no idea which plan shipped the typo.

These tests pin the loud-parse contract: every invalid value (plan
`max_parallel: -3 / 0 / "two"`, model `max_parallel: 0 / -1 / "two"`,
`max_parallel_ceiling: "two"`, `max_parallel_ceiling: 2` with
`max_parallel: 4`) is a `ValueError` naming the offending plan or
`plan/model`, and every valid shape (`auto`, integer >= 1, the model's
own narrowing) parses cleanly and lands on the right Plan/Model field.
The positive cases also pin that `Plan.max_parallel` (the property that
returns `configured_parallel or SEED_CAP`) can never observe a leaked
zero/negative — the configured_parallel path either holds a positive int
or `None` (for `auto`), and anything else fails at parse time.

Each negative case writes a one-plan temp plans.yaml and calls
`models.load(path)`; the plain-script `__main__` block at the bottom
delegates to `tests/_runner.run(globals())` so the file is both runnable
under pytest and as `python3 tests/test_config_schema.py`, matching the
dual-use convention in tests/test_routing.py.
"""
from __future__ import annotations

import os
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)  # so `import conftest` resolves under plain `python3`

import conftest  # noqa: F401  (socket guard for plain-script mode)

from switchyard import models  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_plans(plans: dict, *, lanes: dict | None = None) -> str:
    """Write a temp plans.yaml with the given `plans` mapping and return
    its path. Caller is responsible for unlinking the file (most tests
    do this in a `finally` block).
    """
    body = {"settings": {}, "plans": plans, "lanes": lanes or {}}
    fd = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(body, fd)
    fd.close()
    return fd.name


def _default_plan(**overrides) -> dict:
    """One-plan dict, ready to be passed to `_write_plans`.

    Default shape: `foo` with `max_parallel: 4`, one model `m1`.
    Overrides land on the `foo` plan (or, if `models` is in overrides,
    replaces the whole model list).
    """
    plan = {"label": "Foo",
            "max_parallel": 4,
            "models": {"m1": {"model": "x/m1"}}}
    plan.update(overrides)
    return {"foo": plan}


def _expect_load_error(plans: dict, *, expected_substring: str,
                       lanes: dict | None = None) -> str:
    """Write a temp plans.yaml, call `models.load`, return the raised
    message.

    Asserts that `ValueError` fires and that `expected_substring` is
    somewhere in the message. Test functions pin additional name checks
    on top of this.
    """
    path = _write_plans(plans, lanes=lanes)
    try:
        try:
            models.load(path)
        except ValueError as exc:
            msg = str(exc)
            assert expected_substring in msg, (
                f"expected {expected_substring!r} in error, got: {msg!r}")
            return msg
    finally:
        os.unlink(path)
    raise AssertionError(
        f"expected ValueError for plans={plans!r}, "
        f"substring={expected_substring!r}; load() succeeded")


# ---------------------------------------------------------------------------
# Negative cases: plan-level max_parallel
# ---------------------------------------------------------------------------

def test_plan_max_parallel_negative_int_is_rejected():
    """A negative `max_parallel` would crash `asyncio.Semaphore(-N)`.

    The loader must refuse this at load time, naming the offending plan
    so the operator can fix it without hunting through a stack trace.
    """
    msg = _expect_load_error(
        _default_plan(max_parallel=-3),
        expected_substring="plan 'foo' max_parallel")
    assert "got -3" in msg, msg
    assert "'foo'" in msg, msg
    assert ">= 1" in msg, msg
    print(f"  plan max_parallel: -3 -> {msg[:80]}...")


def test_plan_max_parallel_zero_is_rejected():
    """`max_parallel: 0` used to be silently coerced to `None` (via the
    truthiness check), which made the learner take over a plan that was
    meant to be hard-stopped. The fix rejects 0 and points at
    `enabled: false` so disabling a plan keeps a single spelling.
    """
    msg = _expect_load_error(
        _default_plan(max_parallel=0),
        expected_substring="plan 'foo' max_parallel")
    assert "got 0" in msg, msg
    assert "enabled: false" in msg, msg
    print(f"  plan max_parallel: 0 -> {msg[:80]}...")


def test_plan_max_parallel_string_is_rejected():
    """`max_parallel: "two"` -- a word in the operator's head, a typo at
    load time. The bare `int()` coercion used to raise `ValueError("int()
    can't convert non-string with explicit base")` with no plan key. The
    loud-parse version names the plan.
    """
    msg = _expect_load_error(
        _default_plan(max_parallel="two"),
        expected_substring="plan 'foo' max_parallel")
    assert "'two'" in msg, msg
    assert "'foo'" in msg, msg
    print(f"  plan max_parallel: \"two\" -> {msg[:80]}...")


def test_plan_max_parallel_bool_is_rejected():
    """`True` / `False` are technically `int` subclasses in Python; the
    helper rejects them explicitly so an operator who typed `max_parallel:
    yes` (which YAML 1.1 reads as `True`) gets a clear message rather
    than `max_parallel: 1`.
    """
    msg = _expect_load_error(
        _default_plan(max_parallel=True),
        expected_substring="plan 'foo' max_parallel")
    assert "bool" in msg, msg
    print(f"  plan max_parallel: True -> {msg[:80]}...")


# ---------------------------------------------------------------------------
# Negative cases: model-level max_parallel
# ---------------------------------------------------------------------------

def test_model_max_parallel_zero_is_rejected():
    """`max_parallel: 0` on a model used to silently drop to `None` (the
    truthiness check `int(...) if body.get(...) else None`), so the
    model kept the plan's full cap instead of being throttled to zero.
    The fix rejects 0 with a pointer to `enabled: false`.
    """
    msg = _expect_load_error(
        _default_plan(models={"m1": {"model": "x/m1", "max_parallel": 0}}),
        expected_substring="model foo/m1 max_parallel")
    assert "got 0" in msg, msg
    assert "foo/m1" in msg, msg
    assert "enabled: false" in msg, msg
    print(f"  model max_parallel: 0 -> {msg[:80]}...")


def test_model_max_parallel_negative_is_rejected():
    """A negative model cap is just as bad as a negative plan cap -- the
    picker would crash on the first claim. Loader names the model so the
    fix is one line away.
    """
    msg = _expect_load_error(
        _default_plan(models={"m1": {"model": "x/m1", "max_parallel": -1}}),
        expected_substring="model foo/m1 max_parallel")
    assert "got -1" in msg, msg
    assert "foo/m1" in msg, msg
    assert ">= 1" in msg, msg
    print(f"  model max_parallel: -1 -> {msg[:80]}...")


def test_model_max_parallel_string_is_rejected():
    """A non-numeric string on a model's `max_parallel` raises with the
    model key in the message -- mirroring the plan-level convention so
    an operator hitting either case can find the fix without grep.
    """
    msg = _expect_load_error(
        _default_plan(models={"m1": {"model": "x/m1", "max_parallel": "two"}}),
        expected_substring="model foo/m1 max_parallel")
    assert "'two'" in msg, msg
    assert "foo/m1" in msg, msg
    print(f"  model max_parallel: \"two\" -> {msg[:80]}...")


# ---------------------------------------------------------------------------
# Negative cases: float truncation
# ---------------------------------------------------------------------------

def test_plan_max_parallel_float_is_rejected():
    """`max_parallel: 3.0` and `3.5` used to silently truncate to `3`
    via `int()`, and `max_parallel: -0.5` raised the misleading
    `got 0` from the `< 1` check. The fix rejects floats outright so the
    operator sees the value they actually typed rather than a hidden
    floor.
    """
    for value in (3.0, 3.5, -0.5):
        msg = _expect_load_error(
            _default_plan(max_parallel=value),
            expected_substring="plan 'foo' max_parallel")
        assert "float" in msg, (value, msg)
        assert repr(value) in msg, (value, msg)
    print("  plan max_parallel: 3.0 / 3.5 / -0.5 -> all rejected as float")


def test_model_max_parallel_float_is_rejected():
    """Same gap as the plan-level helper. PyYAML preserves the float type
    on round-trip, so `model.max_parallel: 2.5` lands in the helper as
    a Python float and must be rejected before `int()` would silently
    floor it.
    """
    for value in (2.0, 2.5, -1.5):
        msg = _expect_load_error(
            _default_plan(models={"m1": {"model": "x/m1", "max_parallel": value}}),
            expected_substring="model foo/m1 max_parallel")
        assert "float" in msg, (value, msg)
        assert repr(value) in msg, (value, msg)
    print("  model max_parallel: 2.0 / 2.5 / -1.5 -> all rejected as float")


def test_plan_max_parallel_ceiling_float_is_rejected():
    """`max_parallel_ceiling: 3.0` used to land as `3` and then either
    pass the `< 1` check silently or trigger the misleading `got 0`
    for negative floats. Reject before `int()` so the operator sees
    what they typed.
    """
    for value in (3.0, 3.5, -0.5):
        msg = _expect_load_error(
            _default_plan(max_parallel_ceiling=value),
            expected_substring="plan 'foo' max_parallel_ceiling")
        assert "float" in msg, (value, msg)
        assert repr(value) in msg, (value, msg)
    print("  plan max_parallel_ceiling: 3.0 / 3.5 / -0.5 -> all rejected as float")


# ---------------------------------------------------------------------------
# Negative cases: max_parallel_ceiling
# ---------------------------------------------------------------------------

def test_plan_max_parallel_ceiling_string_is_rejected():
    """`max_parallel_ceiling: "two"` raises the same loud-parse shape:
    plan key named, offending value in the message.
    """
    msg = _expect_load_error(
        _default_plan(max_parallel_ceiling="two"),
        expected_substring="plan 'foo' max_parallel_ceiling")
    assert "'two'" in msg, msg
    print(f"  plan max_parallel_ceiling: \"two\" -> {msg[:80]}...")


def test_plan_max_parallel_ceiling_below_configured_is_rejected():
    """`max_parallel_ceiling: 2` on a plan with `max_parallel: 4` is a
    contradiction: the learner can never satisfy a ceiling below the
    configured cap. Reject it explicitly with both numbers in the
    message so the operator can see the conflict.
    """
    msg = _expect_load_error(
        _default_plan(max_parallel=4, max_parallel_ceiling=2),
        expected_substring="plan 'foo' max_parallel_ceiling")
    assert "max_parallel (4)" in msg, msg
    assert "got 2" in msg, msg
    print(f"  plan ceiling 2 with max_parallel 4 -> {msg[:80]}...")


def test_plan_max_parallel_ceiling_zero_is_rejected():
    """`max_parallel_ceiling: 0` used to drop to `None` via the truthiness
    check -- turning "no ceiling" into "no ceiling" silently, while the
    operator meant the learner should never probe above zero (i.e. the
    plan is capped for good). The fix rejects 0 with the same shape as
    a plan-level `max_parallel: 0`.
    """
    msg = _expect_load_error(
        _default_plan(max_parallel_ceiling=0),
        expected_substring="plan 'foo' max_parallel_ceiling")
    assert "got 0" in msg, msg
    assert ">= 1" in msg, msg
    print(f"  plan max_parallel_ceiling: 0 -> {msg[:80]}...")


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------

def test_plan_max_parallel_auto_is_accepted():
    """`max_parallel: auto` (and its case variants) is the learner-managed
    shape: the Plan gets `configured_parallel=None`, so `Plan.max_parallel`
    falls back to `SEED_CAP` until the learner takes over.
    """
    path = _write_plans(_default_plan(max_parallel="auto"))
    try:
        reg = models.load(path)
        plan = reg.plans["foo"]
        assert plan.configured_parallel is None, plan.configured_parallel
        # The Plan.max_parallel property falls back to SEED_CAP when
        # configured_parallel is None.
        assert plan.max_parallel == models.SEED_CAP, plan.max_parallel
        print(f"  plan max_parallel: auto -> configured_parallel=None, "
              f"max_parallel falls back to SEED_CAP ({models.SEED_CAP})")
    finally:
        os.unlink(path)


def test_plan_max_parallel_auto_case_insensitive():
    """`Auto`, `AUTO`, `auto` are all the same plan-level setting. Pin
    this so a future refactor that casefolds only the lowercase form
    cannot regress it.
    """
    for variant in ("auto", "Auto", "AUTO", "  AUTO  "):
        path = _write_plans(_default_plan(max_parallel=variant))
        try:
            reg = models.load(path)
            assert reg.plans["foo"].configured_parallel is None, (
                variant, reg.plans["foo"].configured_parallel)
        finally:
            os.unlink(path)
    print("  plan max_parallel: 'auto' / 'Auto' / 'AUTO' / '  AUTO  ' all parse as auto")


def test_plan_max_parallel_positive_int_is_accepted():
    """A plain positive int lands on `configured_parallel` and propagates
    through `Plan.max_parallel` without falling back to SEED_CAP. Pin
    a few values around the fixture's typical range to catch any
    off-by-one in the helper's `< 1` check.
    """
    for value in (1, 2, 4, 8, 16):
        path = _write_plans(_default_plan(max_parallel=value))
        try:
            reg = models.load(path)
            plan = reg.plans["foo"]
            assert plan.configured_parallel == value, (
                value, plan.configured_parallel)
            assert plan.max_parallel == value, (value, plan.max_parallel)
        finally:
            os.unlink(path)
    print("  plan max_parallel: 1, 2, 4, 8, 16 -> configured_parallel and "
          "Plan.max_parallel match exactly")


def test_model_max_parallel_narrows_plan_cap():
    """`model.max_parallel: 2` on a `max_parallel: 4` plan is the
    canonical narrowing shape. `Plan.cap_for(model)` returns the model's
    own ceiling, not the plan's. The lane-board relies on this for the
    per-row strip; a regression here would silently widen the row.
    """
    path = _write_plans(_default_plan(
        max_parallel=4,
        models={"m1": {"model": "x/m1", "max_parallel": 2}},
    ))
    try:
        reg = models.load(path)
        plan = reg.plans["foo"]
        model = plan.models["m1"]
        assert plan.max_parallel == 4, plan.max_parallel
        assert model.max_parallel == 2, model.max_parallel
        assert plan.cap_for(model, reg.settings) == 2, (
            plan.cap_for(model, reg.settings))
        print("  plan=4, model=2: cap_for(model) == 2 (model narrows plan)")
    finally:
        os.unlink(path)


def test_model_max_parallel_absent_means_no_narrowing():
    """A model with no `max_parallel` keeps the plan's cap. The picker
    reads `model.max_parallel is None` to skip the narrowing branch.
    """
    path = _write_plans(_default_plan(max_parallel=4))
    try:
        reg = models.load(path)
        model = reg.plans["foo"].models["m1"]
        assert model.max_parallel is None, model.max_parallel
        assert reg.plans["foo"].cap_for(model, reg.settings) == 4, (
            reg.plans["foo"].cap_for(model, reg.settings))
        print("  model with no max_parallel -> cap_for stays at plan's 4")
    finally:
        os.unlink(path)


def test_plan_max_parallel_ceiling_above_configured_is_accepted():
    """`max_parallel_ceiling: 8` on a `max_parallel: 4` plan is the
    learner-headroom shape: the configured cap is fixed, the ceiling
    caps the learner. Both fields land on the Plan dataclass.
    """
    path = _write_plans(_default_plan(max_parallel=4, max_parallel_ceiling=8))
    try:
        reg = models.load(path)
        plan = reg.plans["foo"]
        assert plan.configured_parallel == 4, plan.configured_parallel
        assert plan.max_parallel_ceiling == 8, plan.max_parallel_ceiling
        print("  plan max_parallel=4, ceiling=8: both land on the dataclass")
    finally:
        os.unlink(path)


def test_plan_max_parallel_ceiling_omitted_means_none():
    """A plan that doesn't set `max_parallel_ceiling` parses as
    `None` -- "no ceiling" -- via the explicit `is not None` check,
    not the truthiness coercion that used to drop `0` to `None` too.
    """
    path = _write_plans(_default_plan(max_parallel=4))
    try:
        reg = models.load(path)
        plan = reg.plans["foo"]
        assert plan.max_parallel_ceiling is None, plan.max_parallel_ceiling
        print("  plan with no max_parallel_ceiling -> None (no ceiling)")
    finally:
        os.unlink(path)


def test_plan_max_parallel_auto_with_ceiling_is_accepted():
    """`max_parallel: auto` + `max_parallel_ceiling: 6` is the learner
    shape: the ceiling is the upper bound, the configured cap is None
    because the learner drives it. The ceiling-below-configured check
    must NOT fire when configured is None (no comparison to make).
    """
    path = _write_plans(_default_plan(max_parallel="auto",
                                       max_parallel_ceiling=6))
    try:
        reg = models.load(path)
        plan = reg.plans["foo"]
        assert plan.configured_parallel is None, plan.configured_parallel
        assert plan.max_parallel_ceiling == 6, plan.max_parallel_ceiling
        print("  plan max_parallel: auto, ceiling 6 -> auto stored as None, "
              "ceiling stored as 6")
    finally:
        os.unlink(path)


def test_plan_max_parallel_never_leaks_zero_or_negative_into_plan():
    """The full enumeration of bad plan-level values raises BEFORE the
    Plan dataclass is constructed. Pinning this on the property's
    return value (`Plan.max_parallel`) is the user-facing version of
    the check: a loaded registry's plan never observes `0` or a
    negative number from the configured path.
    """
    for v in (-3, 0, "two", True):
        path = _write_plans(_default_plan(max_parallel=v))
        try:
            try:
                reg = models.load(path)
            except ValueError:
                continue
            # If load() didn't raise, the loaded plan must not expose a
            # zero/negative cap through any path.
            plan = reg.plans["foo"]
            assert plan.configured_parallel != 0, (v, plan)
            if plan.configured_parallel is not None:
                assert plan.configured_parallel >= 1, (v, plan)
            assert plan.max_parallel >= 1, (v, plan.max_parallel)
        finally:
            os.unlink(path)
    print("  bad plan-level values never leak into Plan.max_parallel: "
          "all raise or land >= 1")


def test_router_signature_does_not_change_with_max_parallel():
    """`router_signature` deliberately excludes caps: it fingerprints
    what the generated LiteLLM config bakes in (model strings, api_base,
    api_key, context windows, lane keys), not policy. Now that bad
    values cannot load, the policy-only carve-out is safe -- a bad cap
    surfaces as a load-time ValueError rather than a router rebuild
    mismatch. This test pins both halves: a cap edit on an otherwise
    unchanged config keeps the signature, and a bad cap is rejected
    before the registry is built.
    """
    good = _write_plans(_default_plan(max_parallel=4))
    try:
        good_reg = models.load(good)
        good_sig = models.router_signature(good_reg)
    finally:
        os.unlink(good)
    edited = _write_plans(_default_plan(max_parallel=5))
    try:
        edited_reg = models.load(edited)
        edited_sig = models.router_signature(edited_reg)
    finally:
        os.unlink(edited)
    assert good_sig == edited_sig, (good_sig, edited_sig)
    # And a bad cap raises before the registry exists -- nothing for
    # router_signature to fingerprint, so this is the side of the contract
    # that protects the watcher from a broken edit.
    bad = _write_plans(_default_plan(max_parallel=-1))
    try:
        try:
            models.load(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on -1")
    finally:
        os.unlink(bad)
    print(f"  router_signature: cap edit 4->5 keeps signature {good_sig[:12]}..., "
          "bad -1 rejected at load")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
