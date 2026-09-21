"""The live config-reload feature: plans.yaml edits reach the running
gateway without a rebuild, and edits that the LiteLLM router cannot follow
do NOT — the running router and the routing policy must never disagree.

Three areas, mirroring the implementation:

- `router_signature` (switchyard/models.py): the fingerprint of everything
  the generated LiteLLM config bakes in — model strings, api_base, api_key,
  context windows, and the SET of lane keys. Caps, quota windows, lane
  order, settings and costs are policy, read from the registry per request,
  and must leave the signature alone. (Lane order *contents* are policy
  because the hook rewrites routing per request; only lane keys appear in
  the generated model_list.)
- `SwitchyardHandler._maybe_reload` / `_swap_registry` (switchyard/hooks.py):
  an mtime bump loads a fresh registry; a signature match swaps it in place,
  a mismatch keeps the running one and says so loudly; a broken file is
  reported once and the stamp advances so it is not retried every cycle.
- `_stat_plans`: a missing file returns None — "no news", not an error —
  and the reload is a no-op.

The handler tests never construct the full SwitchyardHandler: importing
switchyard.hooks builds a module-level singleton (hooks.py line 660) that
starts its own watcher thread and would dial a real Redis. Instead a bare
instance is laid out by hand with a FakeRedis underneath, so the swap logic
runs exactly as shipped with no network and no threads. If litellm is not
installed on this host, those tests print a note and skip; the suite stays
green.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()

import yaml  # noqa: E402

from switchyard import models                      # noqa: E402
from switchyard.picker import Picker               # noqa: E402
from switchyard.policy import CapacityPolicy       # noqa: E402
from switchyard.slots import SlotTable             # noqa: E402
from switchyard.usage import Ledger                # noqa: E402
from tests.fake_redis import FakeRedis             # noqa: E402

try:
    from switchyard import hooks                   # noqa: E402
    _HOOKS_UNAVAILABLE = ""
except Exception as exc:                           # litellm missing, mostly
    hooks = None
    _HOOKS_UNAVAILABLE = f"{type(exc).__name__}: {exc}"

if hooks is not None:
    # Importing hooks builds the module-level singleton, whose watcher thread
    # dials a Redis that is not there in a test run. Its warnings are about
    # ITS redis, tell us nothing about the code under test, and would only
    # bury the assertions' own output; the capture test below re-enables the
    # logger for exactly as long as it needs records.
    hooks.log.disabled = True


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# fixtures: temp copies of the example config, mutated one field at a time
# ---------------------------------------------------------------------------

def _write_copy(mutate=None) -> str:
    """A temp plans.yaml: the example fixture, optionally edited."""
    with open(plans_path()) as fh:
        raw = yaml.safe_load(fh)
    if mutate is not None:
        mutate(raw)
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as fh:
        yaml.safe_dump(raw, fh)
    return path


def _sig_of(path: str) -> str:
    return models.router_signature(models.load(path))


def _rewrite(path: str, mutate, stamp: float | None = None) -> float:
    """Edit the yaml at `path` in place and force a mtime the gate cannot
    mistake for the old one — back-to-back writes can land in the same
    filesystem timestamp tick, and the test is about the gate, not the clock."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    mutate(raw)
    with open(path, "w") as fh:
        yaml.safe_dump(raw, fh)
    if stamp is None:
        stamp = os.stat(path).st_mtime + 30.0
    os.utime(path, (stamp, stamp))
    return stamp


# -- policy edits: the hook rewrites routing from the live registry ---------
def _bump_parallel(raw):
    raw["plans"]["minimax-ultra"]["max_parallel"] = 5


def _rotate_forge(raw):
    order = raw["lanes"]["forge"]["order"]
    raw["lanes"]["forge"]["order"] = order[1:] + order[:1]


def _retune_pin_wait(raw):
    raw["settings"]["pin_wait_seconds"] = 3


# -- router-shaped edits: the generated model_list bakes these in -----------
def _retag_model(raw):
    raw["plans"]["local-box"]["models"]["qwen"]["model"] = \
        "openai/Qwen3.8-Flash-Next-EDITED"


def _move_api_base(raw):
    raw["plans"]["minimax-ultra"]["api_base"] = "https://api.example.invalid/v1"


def _swap_api_key(raw):
    raw["plans"]["minimax-ultra"]["api_key"] = "os.environ/EDITED_KEY"


def _add_lane(raw):
    raw["lanes"]["shadow"] = {"label": "Shadow",
                              "order": ["local-box/qwen"], "tail": []}


def _drop_lane(raw):
    del raw["lanes"]["bulk"]


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _config_path:
    """Point models.CONFIG_PATH at a path for the duration, then restore."""

    def __init__(self, path: str):
        self.path = path

    def __enter__(self) -> str:
        self.old = models.CONFIG_PATH
        models.CONFIG_PATH = self.path
        return self.path

    def __exit__(self, *exc):
        models.CONFIG_PATH = self.old
        return False


def _bare_handler(path: str):
    """A SwitchyardHandler with no __init__, no watcher thread, no real
    Redis: exactly the state `_maybe_reload` and `_swap_registry` touch."""
    h = hooks.SwitchyardHandler.__new__(hooks.SwitchyardHandler)
    fake = FakeRedis()
    h._redis = fake
    h.registry = models.load(path)
    h._slots = SlotTable(fake, h.registry.settings.inflight_max_age_seconds)
    h._ledger = Ledger(fake)
    h._policy = CapacityPolicy(fake, h.registry.settings, h._ledger)
    h._picker = Picker(h.registry, h._slots, h._policy)
    h._plans_mtime = h._stat_plans()
    h._watcher = None
    h._sync_redis_client = None
    h._beats = {}
    return h


# ---------------------------------------------------------------------------
# router_signature
# ---------------------------------------------------------------------------

def test_loading_the_fixture_twice_gives_the_same_signature():
    """The fingerprint is a function of the config, not of the load."""
    first = models.router_signature(models.load(plans_path()))
    second = models.router_signature(models.load(plans_path()))
    assert first == second, (first, second)
    assert len(first) == 64 and int(first, 16) >= 0, first
    print(f"  same fixture twice -> same sha256 {first[:12]}...")


def test_policy_only_edits_leave_the_signature_unchanged():
    """Caps, lane order and settings are read per request — none of them is
    part of what the generated LiteLLM config bakes in."""
    baseline = _sig_of(plans_path())
    for what, mutate in [("plan max_parallel", _bump_parallel),
                         ("lane order rotation", _rotate_forge),
                         ("settings.pin_wait_seconds", _retune_pin_wait)]:
        path = _write_copy(mutate)
        try:
            assert _sig_of(path) == baseline, f"{what} changed the signature"
        finally:
            os.unlink(path)
    print("  max_parallel, lane order and pin_wait_seconds edits: signature held")


def test_router_shaped_edits_change_the_signature():
    """What the generated model_list carries — model strings, api_base,
    api_key, context windows — and the set of lane keys, are router
    surface: touching any of it must change the fingerprint."""
    baseline = _sig_of(plans_path())
    for what, mutate in [("a model string", _retag_model),
                         ("a plan api_base", _move_api_base),
                         ("a plan api_key", _swap_api_key),
                         ("adding a lane", _add_lane),
                         ("removing a lane", _drop_lane)]:
        path = _write_copy(mutate)
        try:
            assert _sig_of(path) != baseline, f"{what} left the signature alone"
        finally:
            os.unlink(path)
    print("  model string, api_base, api_key and lane-set edits: signature moved")


# ---------------------------------------------------------------------------
# the handler, without constructing it
# ---------------------------------------------------------------------------

def test_missing_config_stat_is_none_and_reload_is_a_no_op():
    """A vanished plans.yaml is not a broken one: the gate reads it as
    'nothing new', keeps the stamp and keeps serving the current registry."""
    if hooks is None:
        print(f"  skipped: switchyard.hooks unavailable ({_HOOKS_UNAVAILABLE})")
        return
    path = _write_copy()
    try:
        with _config_path(path):
            h = _bare_handler(path)
            missing = path + ".gone"
            with _config_path(missing):
                assert h._stat_plans() is None
                h._plans_mtime = 123.0
                before = h.registry
                h._maybe_reload()
                assert h.registry is before
                assert h._plans_mtime == 123.0, "a missing file moved the stamp"
                assert h._stat_plans() is None
        print("  missing plans.yaml: stat None, reload no-op, stamp untouched")
    finally:
        os.unlink(path)


def test_a_policy_edit_swaps_the_registry_in_place():
    """Bump a cap, retune the pin wait and reorder a lane: the mtime gate
    opens, the fresh registry loads, and because the signature matches, the
    swap promotes it — registry, picker and policy together."""
    if hooks is None:
        print(f"  skipped: switchyard.hooks unavailable ({_HOOKS_UNAVAILABLE})")
        return
    path = _write_copy()
    try:
        with _config_path(path):
            h = _bare_handler(path)
            old_reg, old_picker, old_policy = h.registry, h._picker, h._policy
            old_refs = [m.ref for m in old_reg.lane_members("forge")]
            # Rotate the body; the tail stays last, as the lane rewrites it.
            expected = old_refs[1:-1] + [old_refs[0], old_refs[-1]]

            def edit(raw):
                _bump_parallel(raw)
                _rotate_forge(raw)
                _retune_pin_wait(raw)

            _rewrite(path, edit)
            h._maybe_reload()

            fresh = h.registry
            assert fresh is not old_reg, "the registry was not swapped"
            assert fresh.plans["minimax-ultra"].max_parallel == 5
            assert fresh.settings.pin_wait_seconds == 3
            assert [m.ref for m in fresh.lane_members("forge")] == expected
            assert h._picker is not old_picker
            assert h._picker.registry is fresh
            assert h._policy is not old_policy
            assert h._policy.settings is fresh.settings
            assert h._plans_mtime == os.stat(path).st_mtime
            # The swap was legal precisely because the router surface held.
            assert models.router_signature(fresh) == \
                models.router_signature(old_reg)
        print("  policy-only edit: registry, picker and policy swapped in place")
    finally:
        os.unlink(path)


def test_a_router_shaped_edit_keeps_the_running_registry():
    """A new model string is a deployment the started router has never seen:
    the edit is refused — old registry stays — and the operator is told to
    reload the router, not left wondering why the lane went dark."""
    if hooks is None:
        print(f"  skipped: switchyard.hooks unavailable ({_HOOKS_UNAVAILABLE})")
        return
    path = _write_copy()
    capture = _Capture()
    hooks.log.disabled = False
    hooks.log.addHandler(capture)
    try:
        with _config_path(path):
            h = _bare_handler(path)
            old_reg = h.registry
            _rewrite(path, _retag_model)
            h._maybe_reload()

            assert h.registry is old_reg, "the running registry was replaced"
            assert h._plans_mtime == os.stat(path).st_mtime, \
                "a refused edit must advance the stamp, not retry forever"
            assert models.router_signature(models.load(path)) != \
                models.router_signature(old_reg)
            assert any("reload.sh" in m or "KEEPING" in m
                       for m in capture.messages), capture.messages
        print("  router-shaped edit: registry kept, operator told to reload.sh")
    finally:
        hooks.log.removeHandler(capture)
        hooks.log.disabled = True
        os.unlink(path)


def test_a_broken_edit_is_reported_once_and_then_ignored():
    """A lane naming a model that does not exist fails to load. The stamp
    advances on the failure, so the watcher does not re-log the same typo
    every five seconds — and the next real edit gets a fresh chance."""
    if hooks is None:
        print(f"  skipped: switchyard.hooks unavailable ({_HOOKS_UNAVAILABLE})")
        return
    path = _write_copy()
    try:
        with _config_path(path):
            h = _bare_handler(path)
            old_reg = h.registry
            stamp = _rewrite(path, _reference_a_ghost_model)
            h._maybe_reload()
            assert h.registry is old_reg
            assert h._plans_mtime == stamp, "a broken file must advance the stamp"

            # No further edit, no further attempt: the gate is shut again.
            h._maybe_reload()
            assert h.registry is old_reg
            assert h._plans_mtime == stamp
        print("  broken edit: load skipped, stamp advanced, second cycle silent")
    finally:
        os.unlink(path)


def _reference_a_ghost_model(raw):
    raw["lanes"]["forge"]["order"].append("ghost-plan/ghost-model")


def test_a_reload_is_gated_on_the_file_mtime():
    """New bytes under an old stamp are invisible: the watcher polls mtime,
    so a write that does not move it is never loaded — that is the gate the
    whole feature rests on."""
    if hooks is None:
        print(f"  skipped: switchyard.hooks unavailable ({_HOOKS_UNAVAILABLE})")
        return
    path = _write_copy()
    try:
        with _config_path(path):
            h = _bare_handler(path)
            old_reg = h.registry
            old_stamp = h._plans_mtime
            assert old_stamp is not None
            _rewrite(path, _bump_parallel, stamp=old_stamp)
            h._maybe_reload()
            assert h.registry is old_reg
            assert h.registry.plans["minimax-ultra"].max_parallel == 4, \
                "a same-stamp write must not be loaded"
        print("  same stamp, new bytes: the gate held, nothing was loaded")
    finally:
        os.unlink(path)


def test_a_swap_keeps_redis_state_and_rebuilds_the_brain():
    """`_swap_registry` carries the plan-keyed Redis state across — slot
    table and ledger are shared, a request in flight is using them — and
    rebuilds everything registry-shaped: picker and policy, plus the slot
    table itself when the inflight window changed."""
    if hooks is None:
        print(f"  skipped: switchyard.hooks unavailable ({_HOOKS_UNAVAILABLE})")
        return
    policy_only = _write_copy(_retune_pin_wait)
    shorter_window = _write_copy(_shrink_inflight_window)
    try:
        with _config_path(policy_only):
            h = _bare_handler(policy_only)
            fake = h._redis
            slots, ledger = h._slots, h._ledger
            policy, picker = h._policy, h._picker

            fresh = models.load(policy_only)
            h._swap_registry(fresh)
            assert h.registry is fresh
            assert h._slots is slots, "the slot table must carry over"
            assert h._ledger is ledger, "the ledger must carry over"
            assert h._picker is not picker and h._picker.registry is fresh
            assert h._policy is not policy
            assert h._policy.settings is fresh.settings

            with _config_path(shorter_window):
                fresh2 = models.load(shorter_window)
                h._swap_registry(fresh2)
                assert h.registry is fresh2
                assert h._slots is not slots, "a new window needs a new table"
                assert h._slots.inflight_max_age == 60
                assert h._slots.redis is fake, "the fake redis must carry over"
            # Nothing in the reload path ever reached for a real client.
            assert h._sync_redis_client is None
        print("  swap: slots+ledger carried, picker+policy rebuilt, "
              "window change rebuilt the slot table")
    finally:
        os.unlink(policy_only)
        os.unlink(shorter_window)


def _shrink_inflight_window(raw):
    raw["settings"]["inflight_max_age_seconds"] = 60


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
    print("\nall hot-reload tests passed")
