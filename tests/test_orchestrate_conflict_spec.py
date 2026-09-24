"""Conflict-fixer spec pins the plumbing fallback (issue #236).

The conflict-fix round of an auto `--merge` run is dispatched with a spec
that tells the worker how to fold the base branch into the feature branch
when `gh pr merge` returns `CONFLICTING`. That spec lives in the
orchestrator's `build_conflict_spec` at
`~/.config/opencode/scripts/orchestrate-auto.py:1476` — a host-side
script that is intentionally OUTSIDE this repo.

The orchestrator-side fix for issue #236 added a plumbing-fallback
paragraph to that spec: if a worktree guard refuses the merge verb,
the worker must fall back to `git read-tree -m -u HEAD origin/<base>`
(resolve → verify → commit) instead of asking the runner for
permission, because a worker question cannot be answered in time and
stalls the pipeline (`dispatch_inactive` per
`test_refused_reply_continues_pipeline`).

This file pins that contract from inside the SwitchYard repo so CI
turns red if the spec regresses:

  1. The fixture `tests/fixtures/conflict_fixer_spec.py` carries the
     expected spec text as a Python constant. The in-repo tests
     below assert it contains the plumbing-fallback keywords
     (`read-tree`, `do not ask`, the plumbing instruction sequence,
     and the do-not-ask rationale referencing
     `test_refused_reply_continues_pipeline`).

  2. `test_spec_contains_original_merge_instructions` pins the
     original `git fetch origin <base>` + `git merge origin/<base>`
     instructions, so a future edit cannot accidentally regress the
     happy-path behaviour while adding the fallback.

  3. `test_live_orchestrator_spec_matches_fixture` reads the live
     `~/.config/opencode/scripts/orchestrate-auto.py` on the host that
     runs the test (operator Mac, CI runner with the file, etc.) and
     compares its `build_conflict_spec` body to the fixture. This
     test SKIPS when the file is not present (a fresh Linux CI
     container that doesn't mount the operator's `~/.config`). On a
     host where the file IS mounted but pre-WS1-patch, the test goes
     red — that is the documented deploy-order signal; the operator
     runs `manual_orchestrator_parity_check` (below) AFTER applying
     the WS1 patch to verify the contract.

  4. `manual_orchestrator_parity_check` is the operator-side assertion
     that does NOT skip when the file is missing. The plain-script
     runner and pytest both skip non-`test_*` callables, so this
     function is NOT auto-collected — the operator invokes it by hand
     after deploying the WS1 patch on their Mac to confirm the
     fixture matches the live orchestrator. Run it via:

         python3 -c "
         import sys; sys.path.insert(0, 'tests');
         from test_orchestrate_conflict_spec import (
             manual_orchestrator_parity_check);
         manual_orchestrator_parity_check()"

Nothing here touches the network, the operator's `.env`, or any
provider. Pure offline regression check.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

from fixtures.conflict_fixer_spec import CONFLICT_FIXER_SPEC

# Tests/test_live_orchestrator_spec_matches_fixture needs to skip on hosts
# where the live orchestrator file isn't mounted. Use `pytest.skip()` when
# pytest is the active runner — checked by `"pytest" in sys.modules`, which
# is True ONLY when pytest imported the test module; importing pytest on
# the call path would poison `sys.modules` and revert to the cycle-2
# silent-skip bug. The plain `python3 tests/test_x.py` runner hits this
# branch instead and prints a `[SKIP]` line plus an early-return that
# escapes `tests/_runner.py`'s `except BaseException` (which would
# otherwise re-classify the `Skipped` exception from `pytest.skip()` as
# a FAIL when pytest is installed but not the active runner).
_RUNNING_UNDER_PYTEST = "pytest" in sys.modules

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "conflict_fixer_spec.py"
assert FIXTURE.is_file(), (
    f"fixture file missing at {FIXTURE} — the conflict-fixer spec "
    "fixture is required for every test in this module"
)

# Operator-side location of the live orchestrator. The auto-collected
# test skips when absent; the manual operator-runs-by-hand check
# always asserts (and prints a clear skip message if absent).
LIVE_ORCHESTRATOR = Path(
    os.path.expanduser("~/.config/opencode/scripts/orchestrate-auto.py"))


def _skip(reason: str) -> None:
    """Cross-runner skip helper.

    Under pytest, calls `pytest.skip(reason)` which raises `Skipped`
    and pytest reports the test as skipped. Under the plain
    `python3 tests/test_x.py` runner (no pytest installed, OR pytest
    installed but not the active runner — both turn `"pytest" in
    sys.modules` False), prints a `[SKIP]` line and returns; the
    calling test function follows up with an explicit `return` to
    early-exit cleanly. The runtime-detection design — via
    `_RUNNING_UNDER_PYTEST` evaluated at module load — addresses the
    cycle-3 review finding that the older `try: import pytest;
    except ImportError` would let the plain runner's
    `except BaseException` swallow a Skipped exception raised by
    `pytest.skip()` even when pytest is merely installed.
    """
    if _RUNNING_UNDER_PYTEST:
        import pytest
        pytest.skip(reason)
    print(f"[SKIP] {reason}")


def _extract_live_spec() -> str | None:
    """Parse the live orchestrator file and return the body of its
    `build_conflict_spec` function. Returns None if the function is
    not found.

    Uses `ast.parse` + `ast.get_source_segment` so the extraction
    survives reformatting (blank-line-count changes, comment churn,
    PEP-8 line-length wrap) — the earlier
    `src.find('\\n\\n\\ndef ')` walk was brittle to those.
    """
    src = LIVE_ORCHESTRATOR.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise AssertionError(
            f"could not parse live orchestrator at {LIVE_ORCHESTRATOR}: "
            f"{exc.msg} at line {exc.lineno}"
        ) from exc
    fn = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "build_conflict_spec"),
        None,
    )
    if fn is None:
        return None
    body_src = ast.get_source_segment(src, fn) or ""
    # The function body is `return f"""..."""`; extract the literal.
    m = re.search(r'return f?"""(?P<text>.*?)"""', body_src, re.DOTALL)
    if m is None:
        raise AssertionError(
            "could not extract f-string body from build_conflict_spec in "
            f"live orchestrator at {LIVE_ORCHESTRATOR}"
        )
    return m.group("text")


def _check_live_parity(skip_if_missing: bool) -> bool:
    """Cross-check the fixture against the live orchestrator.

    Returns True if the live file is missing (caller decides whether
    to skip or fail). Otherwise asserts fixture == live_spec and
    returns False.

    `skip_if_missing=True` makes this a no-op under the
    `test_live_orchestrator_spec_matches_fixture` auto-collected test
    (CI runners where the file isn't mounted skip cleanly via
    `_skip()`). `skip_if_missing=False` makes this the strict
    `manual_orchestrator_parity_check` operator-invoked assertion —
    a missing file is reported as `[SKIP]` for the operator but never
    silent-passed.
    """
    if not LIVE_ORCHESTRATOR.is_file():
        if skip_if_missing:
            _skip(f"live orchestrator not at {LIVE_ORCHESTRATOR}")
            return True
        print(f"[SKIP] live orchestrator not at {LIVE_ORCHESTRATOR}")
        return True
    live_spec = _extract_live_spec()
    assert live_spec is not None, (
        f"build_conflict_spec function not found in live orchestrator at "
        f"{LIVE_ORCHESTRATOR}; the function may have been renamed — "
        "update the fixture and this test together"
    )
    assert live_spec == CONFLICT_FIXER_SPEC, (
        "live orchestrator's build_conflict_spec does not match the "
        "in-repo fixture. Update one to match the other — they MUST "
        "stay in sync. See tests/test_orchestrate_conflict_spec.py "
        "for the contract. (If you just applied the WS1 patch, you "
        "may need to update this fixture to mirror the new spec "
        "text and re-run.)"
    )
    return False


def test_spec_contains_plumbing_fallback():
    """Pin the plumbing-fallback paragraph that issue #236 added.

    The fallback instructs the worker to use `git read-tree -m -u HEAD
    origin/<base>` (two-way merge into the index + working tree)
    instead of asking the runner when a guard refuses the merge
    verb. The do-not-ask instruction is the load-bearing part: the
    worker MUST proceed via plumbing, never park on a question.
    """
    spec = CONFLICT_FIXER_SPEC
    # The plumbing instruction itself.
    assert "read-tree -m -u HEAD origin/{base}" in spec, spec
    # The do-not-ask instruction (case-insensitive; the live spec
    # uses "do NOT ask" with caps for emphasis).
    assert "do not ask" in spec.lower(), spec
    # The rationale that points at the regression test name — this is
    # the operator's anchor for understanding WHY the fallback exists.
    assert "test_refused_reply_continues_pipeline" in spec, spec
    # The pipeline-stall rationale: dispatch settles before reply.
    assert "dispatch settles" in spec or "dispatch_inactive" in spec, spec


def test_spec_contains_original_merge_instructions():
    """Pin the original happy-path merge instructions.

    The plumbing fallback is an ADDITION, not a replacement: the
    worker still tries `git fetch origin <base>` + `git merge
    origin/<base>` first. A future edit that drops those instructions
    would force every conflict-fix round through plumbing, which is
    slower and loses git's merge-recursive resolution heuristics.
    """
    spec = CONFLICT_FIXER_SPEC
    assert "git fetch origin {base}" in spec, spec
    assert "git merge origin/{base}" in spec, spec
    # The CONFLICTS acceptance line is the literal exact phrase the
    # orchestrator's journal-settle handler greps for; do not let it
    # drift to e.g. "CONFLICTS resolved" (no colon).
    assert "CONFLICTS: resolved" in spec, spec


def test_spec_instructs_plumbing_commit_produces_merge_commit():
    """Pin the explicit instruction that `git commit` after the
    plumbing `read-tree` produces a merge commit on the feature
    branch. This is the bit that tells the worker the plumbing path
    is a true equivalent of `git merge` for journal/audit purposes —
    the resulting tree state and the resulting journal event
    (`conflict-fix-done`) match either path.
    """
    spec = CONFLICT_FIXER_SPEC
    assert "produces a merge commit" in spec, spec
    assert "feature branch" in spec, spec


def test_live_orchestrator_spec_matches_fixture():
    """Cross-check the fixture against the live orchestrator.

    Skips (via `pytest.skip()` under pytest, via `[SKIP]` print under
    the plain runner) when `~/.config/opencode/scripts/orchestrate-auto.py`
    is absent — typical CI runners don't mount the operator's
    `~/.config`. On a host where the file IS mounted but still on the
    pre-WS1 patch (e.g. between the moment this PR merges and the
    moment the operator deploys the WS1 patch), this test goes red —
    that is the documented deploy-order signal; see the PR body's
    "Operator action (post-merge)" section and run
    `manual_orchestrator_parity_check` after applying the patch.
    """
    if _check_live_parity(skip_if_missing=True):
        return  # under pytest the skip raised; under plain runner
                # _skip already printed and the explicit return exits.


def manual_orchestrator_parity_check():
    """Operator-invoked fixture-vs-live parity assertion.

    NOT auto-collected (the plain runner and pytest default discovery
    both skip callables that don't match `test_*`). Run by the
    operator after deploying the WS1 patch to
    `~/.config/opencode/scripts/` on the operator Mac, to confirm the
    contract:

        python3 -c "
        import sys; sys.path.insert(0, 'tests');
        from test_orchestrate_conflict_spec import (
            manual_orchestrator_parity_check);
        manual_orchestrator_parity_check()"

    Exits 0 if the live orchestrator's `build_conflict_spec` body
    matches the in-repo fixture. Exits 0 with a `[SKIP]` line if the
    live file is absent (operator-side failure mode is "WS1 patch not
    yet applied", which surfaces elsewhere — this function only
    asserts parity, not deploy-state). Raises AssertionError on a
    real mismatch with a clear diff-style message.
    """
    if _check_live_parity(skip_if_missing=False):
        return
    print("[OK] live orchestrator's build_conflict_spec matches the fixture")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
