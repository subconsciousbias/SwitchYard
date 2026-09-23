"""One runner for every tests/test_*.py plain-script entry point.

Each test_* file ends with an `if __name__ == "__main__":` block that walks
`globals()` for callables named `test_*` and runs them in definition order.
Appending a new test_* below that block is the silent-failure trap that
CLAUDE.md warns about: the runner is discovered too late to pick the new
function up, the suite still reports "N tests passed", and the new test
never runs.

`run(glbls)` is the mechanical guard against that trap:

  1. walk the call-site module's compiled top-level code object (the one
     `inspect.currentframe().f_back.f_code`) for nested `co_consts` that
     are themselves code objects named `test_*`. compile() materialises
     those code objects even for `def`s whose body never executes at
     import time, so the set is complete -- a function that hasn't been
     imported as a callable still shows up here.
  2. any whose `co_firstlineno` is greater than the call site's line
     number was defined after the runner block, which is precisely the
     bug CLAUDE.md warns about. Fail with the offending name(s) before
     running anything, so the suite cannot report "all green" while
     silently skipping a test.
  3. then iterate `glbls` in definition order (the order in which the
     names were added to the module namespace -- matching what the
     original per-file runners effectively did under
     `list(globals().items())`), run every `test_*` callable inside a
     per-test try/except, print `  ok  name` or ` FAIL name: ...`, and
     exit non-zero on any failure or zero collected tests.

Definition order (not alphabetical sort) is the order that matches the
suite's pre-existing behaviour: 17 of the 19 files already sorted by
name and were insensitive to ordering, but two files -- test_portal.py
and test_messages_hooks.py -- iterated `globals().items()` directly,
which is definition order. Those two files have tests whose state
assumes an earlier test has not yet run, and switching them to a
strict-alphabetical sort would expose that brittleness. Keeping the
definition-order convention here means the runner unifies failure
handling (the bug the task is fixing) without re-ordering the suite's
existing tests.

The check is mechanical because the underlying failure mode is
mechanical: a developer pasted a def below the runner block. A code-review
rule could catch it; a mechanical check catches it the moment the
developer hits save. Issue #134's "exit 0 runner in test_usage.py" was
the same class of bug -- a runner that does not notice it has gone
wrong -- and unifying on this one fixes that incidentally.
"""
from __future__ import annotations

import inspect
import sys
import types


def _missing_tests(frame: types.FrameType) -> list[str]:
    """Names of `test_*` code objects in the caller's module whose
    definition line comes AFTER the call site.

    The call site is the line where `run(glbls)` was invoked. Anything
    `def test_*()`-ed below it cannot have been reached at the time the
    call was made, and so cannot be discovered by the `globals()` walk.
    """
    caller = frame.f_back
    if caller is None:
        return []
    module_code = caller.f_code
    call_line = caller.f_lineno
    after: list[str] = []
    for const in module_code.co_consts:
        if not isinstance(const, types.CodeType):
            continue
        if not const.co_name.startswith("test_"):
            continue
        if const.co_firstlineno > call_line:
            after.append(const.co_name)
    return after


def run(glbls: dict) -> int:
    """Discover and run every `test_*` callable in `glbls`. Exit non-zero
    on any failure (including zero tests, or any test defined below the
    call site)."""
    frame = inspect.currentframe()
    missing = _missing_tests(frame)
    if missing:
        for name in missing:
            print(f"defined after the runner — it will never run: {name}")
        return 1

    # `glbls.items()` is insertion order (= definition order for a
    # module namespace), matching the per-file convention that 17 of the
    # 19 original runners used one of two orderings -- alphabetical or
    # definition order -- and the two test files whose tests depended
    # on a particular order (test_portal.py, test_messages_hooks.py)
    # used definition order. Sorting alphabetically here would expose
    # test-isolation bugs in those files; keeping the definition order
    # the rest of the suite already had is the smaller behavioural
    # change.
    tests = [
        (name, fn) for name, fn in glbls.items()
        if name.startswith("test_") and callable(fn)
    ]
    if not tests:
        print("no test_* callables collected")
        return 1

    failures: list[tuple[str, BaseException]] = []
    for name, fn in tests:
        try:
            fn()
        except BaseException as exc:           # noqa: BLE001 — runner prints the cause itself
            failures.append((name, exc))
            print(f" FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok  {name}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("failed: " + ", ".join(name for name, _ in failures))
        return 1
    return 0


if __name__ == "__main__":           # the runner itself is scriptable
    sys.exit(run(globals()))