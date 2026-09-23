# Working in `tests/`

The offline test suite. Plain scripts, no pytest plugins, no Redis, no real
provider.

## Plain-script runner

`tests/*.py` are designed to be runnable by anyone, on any machine, with no
credentials, no Docker and no subscriptions. The runner block lives at the
bottom of each file and delegates to a shared helper:

```python
if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
```

`tests/_runner.run(globals())` discovers test functions from `globals()` in
definition order, runs each inside a per-test try/except, prints `  ok  name`
or ` FAIL name: ...`, and exits non-zero on any failure or on zero tests
collected. `tests/render_preview.py`'s `__main__` is not a test runner; it is
the layout-preview script and stays as `raise SystemExit(main())`.

It also carries a mechanical guard against the silent-skip trap: `run()`
walks the call-site module's compiled `co_consts`, finds `test_*` code
objects whose `co_firstlineno` is greater than the call site's, and exits
non-zero naming them. Anything appended *after* that block is defined too
late to be collected, and the guard turns the silent-skip case into a hard
failure at import time so the suite cannot report "N tests passed" with a
new test absent. **Insert before the runner, and check the count went up.**

The single documented entry point is `scripts/test.sh`:

```bash
bash scripts/test.sh
```

which runs `python3 -m pytest -q` then `python3 tests/render_preview.py`
and exits non-zero on any failure. For a quick smoke run on a single file
or to iterate on a fix, the same tests work as plain scripts — `python3
tests/test_*.py` runs the `__main__` block, which delegates to `_runner`.
`tests/render_preview.py`'s `__main__` is not a test runner; it is the
layout-preview script. Keep it that way — anything needing a live provider
belongs in `scripts/smoke.py` (which checks a running deployment and says
so), or as a manual step in `TESTING.md`.

## Adding a new test file

  1. Create `tests/test_<area>.py`. Use `tests/_modules.py` if you need shared
     helpers that are not worth a fixture.
  2. Put all `test_*` functions in the file **before** the
     `if __name__ == "__main__":` runner block at the bottom. Append-only edits
     below the runner are caught by the mechanical guard in
     `tests/_runner.py` (see "Plain-script runner" above).
  3. Set `SWITCHYARD_PLANS` to `tests/plans_path.plans_path()` early (before
     importing `switchyard.*`), not via `os.environ.setdefault`. The module
     captures the env var at import time, and a stray export from a real shell
     would otherwise silently become the fixture.
  4. Use the shared `FakeRedis` from `tests/fake_redis.py` for any test that
     needs Redis behaviour — do not start a real Redis.
  5. Stub the network. `127.0.0.1` is allowed (stub servers and subprocess
     bridges run there on purpose); anything else is blocked under the pytest
     path (see below).
  6. Run `bash scripts/test.sh` and check your test count.

## The conftest socket guard — and its real scope

`tests/conftest.py` blocks every outbound socket to a non-loopback address,
so a test that reaches `api.x.ai`, `api.z.ai` or a local Ollama fails
instead of spending capacity. Loopback is allowed on purpose: stub HTTP
servers and subprocess bridges run on `127.0.0.1`, which is how the real
protocol gets exercised against a fake peer.

**The guard only fires when pytest auto-loads `conftest.py`.** Pytest does
that for `python3 -m pytest`; a plain `python3 tests/test_x.py` does not get
pytest's conftest auto-load, so under the plain-script runner the guard is
not present unless the test file itself imports the conftest (none currently
do). Honest scope:

  - The "tests never call a real provider" guarantee holds under `pytest`
    and where a script explicitly imports `conftest`.
  - Under the plain `python3 tests/test_x.py` runner, a test that reaches a
    real provider will *succeed* (and burn quota), because the guard isn't
    installed — the only thing stopping that is convention plus review.
  - Anything that genuinely needs a provider belongs in `scripts/smoke.py`,
    which checks a running deployment and says plainly that it spends quota.

This is why the loop above runs each file in its own process: the plain
runner does no setup, no teardown, no shared fixtures, no conftest.

## `tests/fake_redis.py` — limits

`tests/fake_redis.py` is an in-memory stand-in for the redis surface the
suite uses. The fake implements exactly what the suite calls — no more:

  - Strings, hashes, zsets, HLLs (as exact sets, not the ~1% approximate
    Redis register array), sets, scripts.
  - The Lua claim/bump-and-cool/bump-streak/touch-lease/drop-lease/set-lease
    scripts that `switchyard/slots.py` registers, dispatched by inspecting
    the Lua source. New Lua scripts added to `slots.py` need a matching
    branch in the fake's `register_script`, or the integration test
    (`tests/test_live_redis.py`) is the only place they get exercised.
  - Pipeline commands in the order they're queued, with one result per
    command. The pipeline used to return `[]`; the current fake returns
    ordered results because callers index `results[0]`.
  - `set_lease`'s SET TTL is a deliberate no-op in the fake (see the comment
    in `FakeRedis.register_script`); tests that want to assert atomicity
    across partial failures need a real Redis.

If a test reaches for a Redis command the fake does not implement, the
right move is to extend the fake's surface to match the real Redis behaviour
and add a comment about the gap. Real-Redis drift in tests is a known
failure mode, and the comment block in `tests/fake_redis.py` is where new
drift gets named so it shows up in review.
