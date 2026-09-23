"""Example-config parity: every operator-facing dataclass field declared in
``switchyard.models`` is reachable as a key in ``config/plans.example.yaml``.

``config/plans.example.yaml`` is the tracked fixture every test loads. The
operator's live ``config/plans.yaml`` is gitignored and varies per deployment,
so the example is the only place to assert "the schema surface stays in sync
with the code that reads it" — a Settings field added in ``models.py`` without
a matching entry in the example stops being documented, and a fresh-clone user
has no way to discover it.

The check is name-only: every (dataclass, field) pair must have at least one
matching key somewhere in the parsed YAML. Tests never call a real provider
(``conftest.py`` blocks non-loopback sockets), so loading the example is just
``yaml.safe_load`` over a static file.

A small set of fields is intentionally not present in the YAML because the
loader assigns them from elsewhere rather than reading them — see
``_INTERNAL_OR_DERIVED`` below for the per-dataclass list and the reasons.
"""
from __future__ import annotations

import os
import sys
from dataclasses import fields as dc_fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml  # noqa: E402

from switchyard.models import (  # noqa: E402
    CallerEnvironmentSettings,
    ConcurrencyLearning,
    Group,
    Lane,
    Model,
    Pacing,
    Plan,
    Probe,
    Quota,
    Settings,
    TransientBreaker,
)
from tests.plans_path import EXAMPLE  # noqa: E402


# Per-dataclass field names that the loader does not read from a YAML key.
# Every entry here is internal — "missing" means "the YAML does not need to
# expose this", not "the field has been forgotten".
#
#   Model.key / Lane.key  -- the outer YAML mapping key, not a sub-key.
#   Model.plan_key        -- set by ``_parse_models`` from the parent plan.
#   Plan.key              -- the outer YAML mapping key.
#   Plan.configured_parallel
#                          -- the YAML key the operator types is
#                             ``max_parallel``; the loader rewrites
#                             ``auto`` -> ``None`` and ``<int>`` -> ``int``.
#   Group.gid             -- a hash computed by the parser from
#                             (lane, strategy, sorted refs); the YAML never
#                             sets it and the parser requires each group
#                             entry to have exactly one strategy key.
#   Group.members         -- reconstructed from the inner list of a strategy
#                             entry; same single-key shape restriction.
#   Group.strategy        -- the strategy name IS the single key of an inline
#                             mapping entry (``round_robin: [...]``), not a
#                             ``strategy:`` field on the group.
#   Group.weights         -- the ref -> weight dictionary is the value of the
#                             ``weighted:`` inline mapping entry, not a
#                             ``weights:`` field on the group.
_INTERNAL_OR_DERIVED: dict[type, set[str]] = {
    Model: {"key", "plan_key"},
    Plan: {"key", "configured_parallel"},
    Group: {"gid", "members", "strategy", "weights"},
    Lane: {"key"},
}


def _yaml_keys(node, out: set[str]) -> None:
    """Walk a parsed YAML structure and collect every dict key at any depth."""
    if isinstance(node, dict):
        for k, v in node.items():
            out.add(k)
            _yaml_keys(v, out)
    elif isinstance(node, list):
        for item in node:
            _yaml_keys(item, out)


def _missing_fields(cls, present: set[str]) -> list[str]:
    skip = _INTERNAL_OR_DERIVED.get(cls, set())
    return [f.name for f in dc_fields(cls)
            if f.name not in skip and f.name not in present]


def test_every_dataclass_field_appears_in_the_example_config():
    with open(EXAMPLE) as fh:
        raw = yaml.safe_load(fh)
    present: set[str] = set()
    _yaml_keys(raw, present)

    classes = (
        Quota, Probe, Model, ConcurrencyLearning, Pacing,
        TransientBreaker, CallerEnvironmentSettings, Settings,
        Plan, Group, Lane,
    )

    failures: list[str] = []
    for cls in classes:
        missing = _missing_fields(cls, present)
        if missing:
            failures.append(f"  {cls.__name__}: missing {sorted(missing)}")

    assert not failures, (
        "fields in switchyard.models have no matching key in "
        "config/plans.example.yaml. Either add them to the example config "
        "(genericised values mirroring what a fresh-clone user needs) or "
        "extend _INTERNAL_OR_DERIVED with a clear reason:\n"
        + "\n".join(failures))


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} config_parity tests passed")
