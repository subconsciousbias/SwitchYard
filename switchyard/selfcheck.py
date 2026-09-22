"""Gateway startup self-check.

Two reasons the proxy might silently mis-route are caught here, on purpose,
at startup:

1. A LiteLLM upgrade could move `Router.async_pre_routing_hook` from the
   internal auto-router path (the one that dispatches to the strategy
   registry) into the CustomLogger-callback dispatch path. Switchyard's
   `async_pre_routing_hook` was DELETED because it could never have fired
   on the pinned build (issue #47); if a future upgrade resurrects
   callback dispatch here, the audit loudly notices so a regression gets
   a name and not a quiet silent-spill story in production.

2. A LiteLLM upgrade could change how `num_retries=0` shapes the retry
   loop, or how `Router.acompletion` reaches the upstream at all. The
   loopback probe exercises the live code path against two stub servers
   (one 200, one 429) on 127.0.0.1 and asserts:

   - the success case reaches `async_log_success_event` exactly once,
   - the 429 case reaches `async_log_failure_event` exactly once,
   - the 429 case lands exactly one HTTP call at the stub (no blind
     router retry), and
   - the 429 case raises `litellm.RateLimitError` (not some swallowed
     exception).

Failure of any check prints `CRITICAL` and exits non-zero. The container
restart policy turns that into a visible outage rather than a silently
broken proxy. No env-var escape hatch - this is a deliberate
"fail-loud" gate, run once per gateway start, whose output is the audit
trail in `docker logs`.

This file does not talk to a real provider. Loopback only, no credentials,
no Redis, no internet.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import logging
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

log = logging.getLogger("switchyard.selfcheck")


class _CheckFailed(SystemExit):
    """Raised on any failed check; carries the message for the CRITICAL line."""


def _critical(msg: str) -> None:
    """Print the CRITICAL line and exit 1 - a gateway that will not start
    beats one that silently mis-routes, every time."""
    log.critical("selfcheck FAILED: %s", msg)
    raise _CheckFailed(1)


def _audit_router_pre_routing_hook(router_src: str) -> None:
    """(a) Router.async_pre_routing_hook must NOT iterate callback registries.

    The hook IS supposed to fire - just only for the litellm auto-router's
    internal routing-strategy machinery (selected_strategy.strategy.async_
    pre_routing_hook). If a future version starts calling CustomLogger
    callbacks under this name, Switchyard could legally re-pick there and
    num_retries=0 is no longer the whole story. Catch it loudly.
    """
    tree = ast.parse(router_src)
    target: ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "Router":
            continue
        for item in node.body:
            if (isinstance(item, ast.AsyncFunctionDef)
                    and item.name == "async_pre_routing_hook"):
                target = item
                break
        if target is not None:
            break
    if target is None:
        _critical(
            "Router.async_pre_routing_hook not found in litellm/router.py - "
            "the proxy's internal pre-routing hook has been removed, which "
            "may indicate a larger API change worth a code review."
        )
        return

    body_src = ast.unparse(target)
    # Any reference to the CustomLogger dispatch registries inside this
    # function means the hook has started fanning out to user code under
    # this name. That is the regression we want to scream about.
    bad_refs = (
        "litellm.callbacks",
        "litellm.logging_callbacks",
        "litellm._async_success_callback",
        "litellm._async_failure_callback",
        "litellm._async_post_call_success_hook",
        "litellm._async_post_call_failure_hook",
    )
    leaked = [r for r in bad_refs if r in body_src]
    if leaked:
        _critical(
            "Router.async_pre_routing_hook now references callback registries "
            f"({', '.join(leaked)}). Switchyard's pre-routing hook could now "
            "fire - review whether num_retries=0 is still the right contract."
        )
        return
    log.info(
        "  ok  Router.async_pre_routing_hook: no CustomLogger dispatch, "
        "internal-strategy only"
    )


def _audit_acompletion_through_fallbacks(router_src: str) -> None:
    """(b) acompletion must reach async_function_with_fallbacks.

    The whole "the router retries" path lives in
    async_function_with_fallbacks -> async_function_with_retries. If
    Router.acompletion stops going through that path, num_retries has no
    effect and a caller retry is not the whole story any more.
    """
    tree = ast.parse(router_src)
    router_class: ast.ClassDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Router":
            router_class = node
            break
    if router_class is None:
        _critical("Router class not found in litellm/router.py")
        return

    # Locate the method that routes model_list->async_function_with_fallbacks.
    # Router has several async entry points (acompletion, abatch_completion,
    # aembedding, ...); any of them calling async_function_with_fallbacks
    # is enough to prove the contract is intact.
    found = False
    for item in router_class.body:
        if not (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name.startswith("a")):
            continue
        item_src = ast.unparse(item)
        if "self.async_function_with_fallbacks" in item_src:
            found = True
            break
    if not found:
        _critical(
            "no Router method dispatches through async_function_with_fallbacks "
            "- num_retries has no effect any more, a caller retry is not the "
            "whole spill story."
        )
        return
    log.info(
        "  ok  Router.acompletion routes through async_function_with_fallbacks"
    )


def _audit_num_retries_early_raise(router_src: str) -> None:
    """(c) async_function_with_retries must early-raise when num_retries<=0.

    The whole point of num_retries=0 is that the FIRST failed call raises
    the original exception immediately. If a future rewrite reorders that
    test (or removes it), the proxy starts doing blind retries again.
    """
    tree = ast.parse(router_src)
    target: ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.AsyncFunctionDef)
                and node.name == "async_function_with_retries"):
            target = node
            break
    if target is None:
        _critical(
            "async_function_with_retries not found in litellm/router.py - "
            "the retry loop has been restructured, verify num_retries still "
            "controls attempts."
        )
        return

    body_src = ast.unparse(target)
    # The early-raise shape on the pinned build is `if num_retries > 0: ...;
    # else: raise`. Future builds could rewrite this; what we must not see
    # is the retry loop `for current_attempt in range(num_retries)` running
    # unconditionally (i.e. for num_retries == 0, the range is empty and the
    # raise below the loop is what bails out). A regression that gates the
    # whole loop on num_retries > 0 with a fall-through is silent.
    if "for current_attempt in range(num_retries)" not in body_src:
        _critical(
            "async_function_with_retries no longer contains the for-loop over "
            "range(num_retries) - the retry path was rewritten. Verify "
            "num_retries=0 still produces exactly one attempt."
        )
        return
    if "raise" not in body_src.split(
        "for current_attempt in range(num_retries)", 1)[0]:
        _critical(
            "async_function_with_retries has no early raise BEFORE the retry "
            "loop - num_retries=0 may now retry anyway."
        )
        return
    log.info(
        "  ok  async_function_with_retries: num_retries<=0 short-circuits "
        "before the retry loop"
    )


def _audit_proxy_pre_call_hook_present() -> None:
    """(d) litellm/proxy must still dispatch pre_call_hook / async_pre_call_hook.

    This is the ONE hook Switchyard relies on. If a future liteLLM build
    renames it (or drops the proxy's CustomLogger path entirely), the
    picker stops firing and the proxy starts routing by tag, with no slot
    accounting and no quota gating. Loud failure beats quiet mis-routing.

    We inspect the source files rather than importing the proxy package:
    importing `litellm.proxy.proxy_server` pulls in the full FastAPI app
    and its transitive deps (websockets, prisma, ...), which the gateway
    image has but a local install may not. Source inspection is also
    more honest - it is the proxy's *source* we are pinning against.

    The presence check is an AST walk: there must be at least one
    `ast.Call` whose callee resolves to `async_pre_call_hook` (whether as
    a bare `Name`, an `Attribute` on a registered manager, or a
    subscript). A bare substring scan over the file would pass on a
    docstring, a comment, or a renamed-but-still-mentioned hook; we
    want the loud semantic the contract is pinning to.
    """
    import litellm
    pkg_dir = os.path.dirname(litellm.__file__)
    proxy_dir = os.path.join(pkg_dir, "proxy")
    if not os.path.isdir(proxy_dir):
        _critical(
            f"litellm/proxy package directory not found at {proxy_dir} - "
            "Switchyard cannot run on a litellm build without the proxy."
        )
        return

    def _callee_names(node):
        # Walk the callee node's identifier chain and return the names
        # we see. Supported shapes:
        #   `foo(...)` -> ['foo']            (bare Name)
        #   `a.b.c(...)` -> ['a', 'b', 'c']  (Attribute chain)
        # Not chased (return []): subscripts (`a[0](...)`), calls
        # (`f()(...)`), and any other ast.AST shape - the audit only
        # cares about a recognisable identifier that could be the
        # dispatch symbol, and a subscript hides that symbol behind a
        # container that we cannot statically dereference. In practice
        # the pinned litellm dispatches `async_pre_call_hook` via
        # attribute chains (`a.b.async_pre_call_hook(target)`), which
        # the loop below handles; if a future build dispatches it
        # through a subscript the audit needs a smarter walker, not
        # a comment-shrink.
        out = []
        cur = node.func
        while isinstance(cur, ast.Attribute):
            out.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            out.append(cur.id)
        return out

    found_any = False
    for root, _, files in os.walk(proxy_dir):
        # Skip pyc caches; their source is identical to the .py alongside
        # them and reading bytecode here would be brittle across Pythons.
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
            except OSError:
                continue
            try:
                tree = ast.parse(src, filename=path)
            except SyntaxError:
                # A future litellm may have a half-broken proxy file; we
                # refuse to silently green-light on parse failure, but
                # only this audit cares about the proxy package, so we
                # move on and the "found_any = False" path catches us.
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if "async_pre_call_hook" in _callee_names(node):
                    found_any = True
                    break
            if found_any:
                break
        if found_any:
            break
    if not found_any:
        _critical(
            "litellm/proxy has no AST-level call to async_pre_call_hook - "
            "Switchyard's pre-call hook will not fire. Verify the proxy still "
            "dispatches CustomLogger pre-call hooks."
        )
        return
    log.info(
        "  ok  litellm/proxy references async_pre_call_hook "
        "(Switchyard's pre-call hook still has a dispatch site)"
    )


# -- dynamic probe ------------------------------------------------------------

class _Stub(BaseHTTPRequestHandler):
    """OpenAI-compatible stub. Modes: 'ok' -> 200, 'ratelimit' -> 429.

    Counts every POST so the probe can prove there were no blind retries.
    """
    mode: str = "ok"
    request_count: int = 0

    def log_message(self, fmt, *args):
        return  # silence the parent class's stderr access logging

    def do_POST(self):
        type(self).request_count += 1
        if type(self).mode == "ratelimit":
            body = json.dumps({"detail": "sidecar at capacity (1)"}).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "20")
        else:
            body = json.dumps({
                "id": "selfcheck-1",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "selfcheck-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                            "total_tokens": 15},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_stub(mode: str) -> tuple[HTTPServer, str, int]:
    """Bind an ephemeral port, run the stub in a background thread. The URL
    and the request counter are returned together so the caller can prove
    nothing fired more than once per probe."""
    _Stub.mode = mode
    _Stub.request_count = 0
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = HTTPServer(("127.0.0.1", port), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}/v1", port


class _Probe:  # placeholder; the real subclass is built in _build_probe
    pass


def _build_probe():
    """Import CustomLogger lazily so a missing litellm is a CRITICAL with
    the real reason, not an obscure NameError at module import time."""
    from litellm.integrations.custom_logger import CustomLogger

    class _ProbeImpl(CustomLogger):
        success: int = 0
        failure: int = 0

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            type(self).success += 1

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            type(self).failure += 1

    return _ProbeImpl


def _make_router(api_base: str, model_name: str):
    import litellm
    from litellm import Router

    model_list = [{
        "model_name": model_name,
        "litellm_params": {
            "model": "openai/selfcheck-model",
            "api_key": "selfcheck-fake-key",
            "api_base": api_base,
        },
    }]
    return Router(
        model_list=model_list,
        num_retries=0,
        disable_cooldowns=True,
        set_verbose=False,
    )


async def _probe_once(router, model_name: str) -> tuple[bool, BaseException | None]:
    import litellm
    try:
        await router.acompletion(
            model=model_name,
            messages=[{"role": "user", "content": "selfcheck"}],
        )
        return True, None
    except Exception as exc:                 # noqa: BLE001  - we re-raise typed
        return False, exc


async def _dynamic_probe() -> None:
    """Two stub servers, two probes: one 200 path, one 429 path.

    The 200 path must reach `async_log_success_event` exactly once. The
    429 path must reach `async_log_failure_event` exactly once, the stub
    must have received exactly one HTTP call (the proof there was no
    blind retry), and the call must have raised `RateLimitError`.

    Probe CustomLoggers are registered via `litellm.logging_callback_manager`
    rather than the bare `litellm.callbacks` list - the manager is the one
    that actually wires async success/failure event dispatch. `litellm.
    callbacks` is consulted for the registered class but the per-event
    dispatch goes through the manager's async lists.
    """
    import litellm

    _ProbeImpl = _build_probe()

    # -- 200 case -----------------------------------------------------------
    ok_server, ok_base, _ = _start_stub(mode="ok")
    ok_probe = _ProbeImpl()
    litellm.logging_callback_manager.add_litellm_async_success_callback(ok_probe)
    litellm.logging_callback_manager.add_litellm_async_failure_callback(ok_probe)
    try:
        ok_router = _make_router(ok_base, "selfcheck-ok")
        ok_ok, ok_exc = await _probe_once(ok_router, "selfcheck-ok")
        # Allow any background-task fan-out to drain before we read counts.
        await asyncio.sleep(0.1)
    finally:
        ok_server.shutdown()
        # Three things are registered into litellm's callback lists here,
        # and each one needs its own teardown:
        #   1. ok_probe (instance, registered via add_*_callback) - the
        #      manager's default lookup is `c == obj` (require_self=False),
        #      which matches the stored CustomLogger instance.
        #   2. ok_router.deployment_callback_on_success / .async_deployment_
        #      callback_on_failure - registered by Router.__init__ as bound
        #      methods whose `__self__` is ok_router. The default `c == obj`
        #      lookup misses them (a bound method is not equal to its self),
        #      so we pass `require_self=True` to match the manager's
        #      `c.__self__ == obj` path. Without this the bound methods
        #      keep ok_router reachable from `litellm._async_*_callback`
        #      for the rest of the process, even after we drop the local
        #      reference (a leak the round-2 review caught).
        #   3. The local ok_router reference itself, dropped so the GC
        #      can collect it once the bound methods are gone.
        # Round-reviewer: this is the in-process caller path; the
        # production entrypoint runs selfcheck in a separate process
        # before `exec litellm`, so the leak is harmless there.
        litellm.logging_callback_manager.remove_callback_from_all_lists(
            ok_probe)
        litellm.logging_callback_manager.remove_callback_from_all_lists(
            ok_router, require_self=True)
        ok_router = None

    if not ok_ok:
        _critical(
            f"success-case probe raised {type(ok_exc).__name__ if ok_exc else 'no exception'}: "
            f"{ok_exc!r}"
        )
        return
    if _Stub.request_count != 1:
        _critical(
            f"success-case stub received {_Stub.request_count} requests, expected 1"
        )
        return
    if ok_probe.success != 1:
        _critical(
            f"success-case: async_log_success_event fired {ok_probe.success} times, "
            "expected 1"
        )
        return
    log.info(
        "  ok  success-case: 1 HTTP call, async_log_success_event fired once"
    )

    # -- 429 case -----------------------------------------------------------
    rl_server, rl_base, _ = _start_stub(mode="ratelimit")
    rl_probe = _ProbeImpl()
    litellm.logging_callback_manager.add_litellm_async_success_callback(rl_probe)
    litellm.logging_callback_manager.add_litellm_async_failure_callback(rl_probe)
    rl_router = None
    try:
        rl_router = _make_router(rl_base, "selfcheck-rl")
        rl_ok, rl_exc = await _probe_once(rl_router, "selfcheck-rl")
        await asyncio.sleep(0.1)
    finally:
        rl_server.shutdown()
        # Instance-based cleanup of the probe, then bound-method cleanup
        # of the Router with require_self=True. Same three-way teardown
        # as the 200 case above.
        litellm.logging_callback_manager.remove_callback_from_all_lists(
            rl_probe)
        if rl_router is not None:
            litellm.logging_callback_manager.remove_callback_from_all_lists(
                rl_router, require_self=True)
        rl_router = None

    if rl_ok:
        _critical("429-case probe returned success, expected RateLimitError")
        return
    if not isinstance(rl_exc, litellm.RateLimitError):
        _critical(
            f"429-case raised {type(rl_exc).__name__}: {rl_exc!r}, expected "
            "litellm.RateLimitError"
        )
        return
    if _Stub.request_count != 1:
        _critical(
            f"429-case stub received {_Stub.request_count} requests, expected 1 "
            "(a blind retry would have re-hit the stub)"
        )
        return
    if rl_probe.failure != 1:
        _critical(
            f"429-case: async_log_failure_event fired {rl_probe.failure} times, "
            "expected 1"
        )
        return
    log.info(
        "  ok  429-case: 1 HTTP call (no blind retry), "
        "RateLimitError raised, async_log_failure_event fired once"
    )


# -- entry point --------------------------------------------------------------

def _litellm_version() -> str:
    """Version string for the audit log; tolerates the per-version naming."""
    try:
        from importlib.metadata import version
        return version("litellm")
    except Exception:
        import litellm
        return getattr(litellm, "__version__", "unknown")


def _static_audit() -> None:
    """Run all four AST checks. Each prints its own ok/fail line."""
    import litellm.router
    import litellm.proxy
    router_src = inspect.getsource(litellm.router)
    _audit_router_pre_routing_hook(router_src)
    _audit_acompletion_through_fallbacks(router_src)
    _audit_num_retries_early_raise(router_src)
    _audit_proxy_pre_call_hook_present()


async def main_async() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="switchyard.selfcheck: %(message)s",
        stream=sys.stdout,
    )
    # LiteLLM installs its own handlers at import time and floods INFO
    # through the root logger; this gate's output is the audit trail in
    # `docker logs`, so silence anything that is not switchyard's own
    # selfcheck line. Anything that actually fails still gets surfaced
    # via the CRITICAL exit above.
    for noisy in ("LiteLLM", "LiteLLM Router", "litellm", "litellm.litellm_core_utils.litellm_logging"):
        logging.getLogger(noisy).setLevel(logging.WARNING + 1)
    log.info("litellm version: %s", _litellm_version())
    _static_audit()
    await _dynamic_probe()
    log.info("selfcheck PASSED")
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except _CheckFailed as exit_exc:
        return int(exit_exc.code) if isinstance(exit_exc.code, int) else 1
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1


if __name__ == "__main__":
    sys.exit(main())