"""Which plans file the tests run against: always the example, never yours.

`config/plans.yaml` is the operator's live portfolio — their plans, their
limits, their expiry dates — and it changes whenever they change a
subscription. Tests that read it give different results on different machines
and on different days, and a green suite then says nothing about the code.

So the example is the only fixture. It is tracked, identical for everyone, and
if a test needs a plan that is expiring, cooled or exhausted it builds that
itself rather than hoping the operator happens to have one.
"""
from __future__ import annotations

import os

# LiteLLM's `__init__` fetches the live model-cost map from GitHub by default
# and falls back to its bundled backup only on network failure. With the
# socket guard active in plain-script mode (loaded a few lines below via
# `import conftest`), that fetch raises before the fallback can run, which
# blocks `import switchyard.hooks` and every test that reaches the hooks
# module. Pin LiteLLM to its bundled backup here, before the guard installs,
# so the hooks tests load cleanly under `python3 tests/test_x.py` the same
# way they do under pytest (which also blocks the fetch, but via the
# conftest auto-import that runs before switchyard is even loaded).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import conftest  # noqa: F401  (socket guard)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "config", "plans.example.yaml")


def plans_path() -> str:
    if not os.path.exists(EXAMPLE):
        raise RuntimeError(
            f"{EXAMPLE} is missing — it is the fixture every test loads. "
            "It is tracked in git; restore it rather than pointing tests at "
            "config/plans.yaml, which differs per operator.")
    return EXAMPLE
