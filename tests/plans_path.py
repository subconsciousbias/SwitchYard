"""Which plans file the tests run against.

`config/plans.yaml` is the operator's own portfolio and is gitignored, so a
fresh clone has only `config/plans.example.yaml`. Tests prefer the real file
when it exists — that is what catches a config the code cannot load — and fall
back to the example so `pytest` passes on a clean checkout.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plans_path() -> str:
    live = os.path.join(ROOT, "config", "plans.yaml")
    example = os.path.join(ROOT, "config", "plans.example.yaml")
    return live if os.path.exists(live) else example
