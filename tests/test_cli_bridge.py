"""Parser tests for the CLI bridge, against output the CLIs really produced.

The OpenCode fixture is a verbatim capture from `opencode run --format json`.
It exists because the first parser was written from a guess at the event shape:
it found no text, fell back to returning the raw event stream as the assistant's
answer, and reported zero tokens — a failure that looks like a working call.
"""
from __future__ import annotations

import errno
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PROVIDER"] = "opencode"

from _modules import load  # noqa: E402

server = load("cli_bridge_server",
              os.path.join(os.path.dirname(HERE), "sidecars", "cli_bridge", "server.py"))

FIXTURE = os.path.join(HERE, "fixtures", "opencode-events.jsonl")


def test_opencode_stream_yields_text_and_usage():
    out = server.parse_output(open(FIXTURE).read())
    assert out["result"] == "OK", out["result"]
    u = out["usage"]
    assert u["input_tokens"] == 6194 and u["output_tokens"] == 18, u
    assert u["cache_read_tokens"] == 1280, u
    print(f"  text={out['result']!r} in={u['input_tokens']} out={u['output_tokens']} "
          f"cached={u['cache_read_tokens']}")


def test_openai_envelope_reports_real_token_counts():
    """Zero usage would leave the pacer unable to measure a per-slot rate."""
    out = server.to_openai(server.parse_output(open(FIXTURE).read()), "m")
    u = out["usage"]
    assert u["prompt_tokens"] == 6194 and u["completion_tokens"] == 18
    assert u["total_tokens"] == 6212
    assert out["choices"][0]["message"]["content"] == "OK"
    print(f"  openai usage: {u}")


def test_anthropic_cache_tokens_are_counted_in_prompt_tokens():
    """Anthropic counts cache reads and writes separately from input_tokens;
    without adding them here, a heavily-cached session reports as nearly idle.

    Repro from issue #25: a real bridged job that consumed ~64k tokens
    reported 95 to the ledger because the cache fields were dropped at the
    to_openai boundary. The fix has to add them to prompt_tokens, otherwise
    the gateway's per-seat headroom calculation sees a cached seat as
    almost empty and leans on it harder than the plan allows.
    """
    payload = {"result": "ok", "usage": {
        "input_tokens": 95,
        "cache_read_input_tokens": 63200,
        "cache_creation_input_tokens": 800,
        "output_tokens": 200,
    }}
    u = server.to_openai(payload, "claude-opus-5")["usage"]
    expected_prompt = 95 + 63200 + 800
    assert u["prompt_tokens"] == expected_prompt, u
    assert u["completion_tokens"] == 200, u
    assert u["total_tokens"] == expected_prompt + 200, u
    print(f"  anthropic cached prompt folded into total: {u}")


def test_openai_shape_cache_tokens_are_not_double_counted():
    """OpenCode/Codex feed `cache_read_tokens` separately, but their
    `input_tokens` already includes those reads per OpenAI convention.
    Only Anthropic's cache_*_input_tokens are excluded from input_tokens,
    so adding them here must be gated on the Anthropic field names —
    otherwise this branch would inflate an OpenCode-style prompt by the
    cached portion a second time.
    """
    payload = {"result": "ok", "usage": {
        "input_tokens": 6194,
        "output_tokens": 18,
        "cache_read_tokens": 1280,
    }}
    u = server.to_openai(payload, "m")["usage"]
    assert u["prompt_tokens"] == 6194, u
    assert u["total_tokens"] == 6212, u
    print(f"  openai-shape cache stays inside input_tokens: {u}")


def test_reasoning_tokens_are_billed_but_not_shown():
    """Reasoning is excluded from the answer text and included in output tokens."""
    stream = "\n".join([
        json.dumps({"type": "reasoning", "part": {"type": "reasoning",
                                                  "text": "thinking out loud"}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": "answer"}}),
        json.dumps({"type": "step_finish", "part": {
            "type": "step-finish",
            "tokens": {"input": 10, "output": 5, "reasoning": 40}}}),
    ])
    out = server.parse_output(stream)
    assert out["result"] == "answer", out["result"]
    assert out["usage"]["output_tokens"] == 45, out["usage"]
    print(f"  reasoning hidden from text, counted in output: {out['usage']}")


def test_concatenated_or_pretty_printed_output_still_parses():
    """A line-based parser silently yields nothing here — which is how the
    original bug surfaced as 'the model returned raw JSON'."""
    pretty = ('{\n  "type": "text",\n  "part": {"type": "text", "text": "hi"}\n}'
              '{"type":"step_finish","part":{"type":"step-finish",'
              '"tokens":{"input":1,"output":1}}}')
    out = server.parse_output(pretty)
    assert out["result"] == "hi", out["result"]
    print(f"  pretty-printed + concatenated parsed: {out['result']!r}")


def test_no_text_raises_rather_than_returning_the_raw_stream():
    try:
        server.parse_output('{"type":"step_start","part":{"type":"step-start"}}')
    except json.JSONDecodeError:
        print("  a stream with no answer raises instead of echoing itself")
        return
    raise AssertionError("expected a parse failure, not a silent fallback")


def test_structured_parser_failure_is_a_502_not_a_raw_passthrough():
    """An OpenCode/Codex stream with no answer must surface as a real error.

    The previous fallback returned the raw JSONL (step_start/step_finish lines
    and all) as the assistant content -- issue #3 had OpenCode clients see
    those raw events in place of a parsed answer. The fix: any structured
    parser that fails to find text now raises HTTP 502, so the gateway can
    retry on another lane rather than ship the raw stream to the caller.

    Drive the failure path with a fake CLI binary that prints the bad stream
    and exits 0: same contract as a real opencode run that produces only
    step_start / step_finish events with no text part.
    """
    import asyncio
    from fastapi import HTTPException

    bad_stream = ('{"type":"step_start","part":{"type":"step-start"}}\n'
                  '{"type":"step_finish","part":{"type":"step-finish",'
                  '"tokens":{"input":1,"output":1}}}')

    fake = tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix="clib-bad-", delete=False)
    fake.write("import sys\n"
               f"sys.stdout.write({bad_stream!r})\n")
    fake.close()
    real_cli = server.CLI
    real_bare = server.BARE
    real_args = server.PROFILE["args"]
    try:
        server.CLI = sys.executable
        server.BARE = False
        server.PROFILE["args"] = [fake.name]
        async def invoke():
            return await server._run_cli("hi", None, "xai/grok-4.6", [])
        try:
            asyncio.run(invoke())
        except HTTPException as exc:
            assert exc.status_code == 502, exc.status_code
            assert "no parsed answer" in str(exc.detail), exc.detail
            print(f"  structured-parser failure -> 502 ({exc.detail[:60]}...)")
            return
        raise AssertionError("expected a 502, not a raw passthrough")
    finally:
        server.CLI = real_cli
        server.BARE = real_bare
        server.PROFILE["args"] = real_args
        os.unlink(fake.name)


def test_max_tokens_becomes_an_instruction():
    """No CLI has a token cap, so it must not be dropped silently."""
    assert server.fold_max_tokens(None, None) is None
    hint = server.fold_max_tokens("Be terse.", 60)
    assert "42 words" in hint and "Be terse." in hint, hint
    print(f"  max_tokens=60 -> {hint.splitlines()[-1][:60]!r}...")


# --------------------------------------------------------------------- codex ---
# Verbatim messages from codex-cli 0.155.1 on a ChatGPT-account seat.
CODEX_QUOTA = ("You've hit your usage limit. Visit "
               "https://chatgpt.com/codex/settings/usage to purchase more "
               "credits or try again at Sep 22nd, 2026 4:37 AM.")
CODEX_BAD_MODEL = json.dumps({
    "type": "error",
    "message": json.dumps({"type": "error", "status": 400, "error": {
        "type": "invalid_request_error",
        "message": "The 'gpt-5' model is not supported when using Codex with a "
                   "ChatGPT account."}}),
})


def test_absolute_reset_time_beats_the_default_cooldown():
    """A 30-hour lockout must not get a one-hour cooldown.

    Without parsing the stated time, the lane would retry a dead plan every hour
    for thirty more hours.

    The timestamp is generated rather than hard-coded: an earlier version pinned
    the real message's "Sep 22nd, 2026 4:37 AM", and the assertion drifted out of
    range as actual time passed — eventually it would parse to the past and return
    None, failing for a reason unrelated to the parser.
    """
    from datetime import datetime, timedelta, timezone
    target = datetime.now(timezone.utc) + timedelta(hours=30)
    message = ("You've hit your usage limit. Visit x to purchase more credits or "
               f"try again at {target.strftime('%b %d, %Y %I:%M %p')}.")
    secs = server.seconds_until(message)
    assert secs is not None and 29 * 3600 < secs < 31 * 3600, secs
    print(f"  a stated reset 30h out -> {secs / 3600:.1f}h cooldown")


def test_the_real_codex_wording_parses():
    """Separately, the verbatim message format must still be recognised — only
    the format, not the magnitude, so real time passing cannot break it."""
    assert server._TRY_AGAIN_AT.search(server.normalise(CODEX_QUOTA)), CODEX_QUOTA
    assert server._LIMIT.search(server.normalise(CODEX_QUOTA))
    print(f"  verbatim wording recognised: ...{CODEX_QUOTA[-34:]}")


def test_quota_message_becomes_429_with_that_retry_after():
    """Generated timestamp, for the same reason as above: a hard-coded future
    date silently drifts out of range as real time passes."""
    from datetime import datetime, timedelta, timezone
    target = datetime.now(timezone.utc) + timedelta(hours=30)
    message = ("You've hit your usage limit. Visit x or try again at "
               f"{target.strftime('%b %d, %Y %I:%M %p')}.")
    exc = server._limit_error(message)
    assert exc.status_code == 429, exc.status_code
    retry = int(exc.headers["Retry-After"])
    assert 29 * 3600 < retry < 31 * 3600, retry
    print(f"  HTTP 429, Retry-After {retry}s ({retry / 3600:.1f}h)")


def test_implausible_reset_times_are_discarded():
    """A stated time carries no timezone, so a past or absurd value is dropped
    rather than trusted."""
    assert server.seconds_until("try again at Jan 1, 2020 1:00 AM") is None
    assert server.seconds_until("You've hit your usage limit.") is None
    print("  past dates and missing dates fall back to the plan default")


def test_the_real_error_is_read_from_stdout_not_stderr():
    """Codex reports failures in stdout JSON and leaves stderr with an unrelated
    line ('Reading additional input from stdin...'), so a stderr-only handler
    turned a clear 400 into an opaque 502."""
    status, message = server.error_from_events(CODEX_BAD_MODEL)
    assert status == 400, status
    assert "not supported when using Codex with a ChatGPT account" in message
    print(f"  extracted status={status}: {message[:60]}...")


def test_codex_is_invoked_without_an_explicit_model():
    """A ChatGPT-account seat rejects every explicit model id, so the CLI picks."""
    argv, stdin_data = server.build_argv("PROMPT", None)
    if server.PROVIDER == "codex":
        assert "--model" not in argv, argv
        assert "--skip-git-repo-check" in argv, argv
        print(f"  {' '.join(argv)}")
    else:
        print(f"  (skipped: PROVIDER={server.PROVIDER})")


def test_an_oversized_prompt_travels_on_stdin():
    """A prompt past MAX_ARG_STRLEN must not reach the exec() call at all.

    create_subprocess_exec dies with "[Errno 7] Argument list too long" on a
    single argv element over ~128 KiB. Over the limit the prompt slot must
    vanish from argv (codex keeps its "-" placeholder) and come back as stdin
    data; under it, argv behaviour is exactly as before.
    """
    huge = "x" * (server.STDIN_PROMPT_LIMIT + 10)
    argv, stdin_data = server.build_argv(huge, None)
    assert stdin_data == huge
    assert huge not in argv, argv
    assert argv[0] == server.CLI, argv
    if server.PROVIDER == "codex":
        assert "-" in argv, argv
    else:
        assert "-" not in argv, argv

    argv, stdin_data = server.build_argv("small", None)
    assert stdin_data is None
    assert "small" in argv, argv
    print(f"  {len(huge)}-char prompt kept off argv; small prompt unchanged")


CODEX_FIXTURE = os.path.join(HERE, "fixtures", "codex-events.jsonl")


def test_codex_stream_yields_text_and_usage():
    """Codex uses `item.agent_message.text` and a top-level usage object — a
    different shape from OpenCode's `part` events, and the original parser found
    neither, so it returned the raw stream as the answer."""
    out = server.parse_output(open(CODEX_FIXTURE).read())
    assert out["result"] == "SHAPE OK", out["result"]
    u = out["usage"]
    assert u["input_tokens"] == 14159 and u["output_tokens"] == 7, u
    assert u["cache_read_tokens"] == 12160, u
    print(f"  text={out['result']!r} in={u['input_tokens']} out={u['output_tokens']} "
          f"cached={u['cache_read_tokens']}")


def test_both_cli_shapes_parse_with_one_parser():
    for name, text in (("codex-events.jsonl", "SHAPE OK"),
                       ("opencode-events.jsonl", "OK")):
        out = server.parse_output(open(os.path.join(HERE, "fixtures", name)).read())
        assert out["result"] == text, (name, out["result"])
        assert out["usage"]["input_tokens"] > 0, (name, out["usage"])
    print("  one parser handles both the part-based and item-based shapes")


def test_codex_reasoning_tokens_count_as_output():
    stream = ('{"type":"turn.completed","usage":{"input_tokens":10,'
              '"output_tokens":5,"reasoning_output_tokens":40}}'
              '{"type":"item.completed","item":{"type":"agent_message","text":"x"}}')
    out = server.parse_output(stream)
    assert out["usage"]["output_tokens"] == 45, out["usage"]
    print(f"  reasoning billed into output: {out['usage']}")


def test_limit_detection_survives_rewording_and_curly_quotes():
    """The original pattern demanded the exact phrase "usage limit reached" and a
    straight apostrophe. Codex says "You\u2019ve hit your usage limit", so an
    exhausted subscription was reported as a 502 and treated as transient."""
    cases = {
        "You\u2019ve hit your usage limit.": True,
        "You've hit your usage limit.": True,
        "Claude usage limit reached. Your limit will reset at 3pm.": True,
        "You have run out of credits for this plan.": True,
        "purchase more credits": True,
        "ENOENT: no such file or directory": False,
    }
    for msg, expected in cases.items():
        hit = bool(server._LIMIT.search(server.normalise(msg))) or bool(server.seconds_until(msg))
        assert hit is expected, (msg, hit)
    print(f"  {sum(cases.values())} limit wordings matched, 1 unrelated error not")




# ----------------------------------------------------------- config reading ---
from plans_path import plans_path  # noqa: E402

PLANS = plans_path()


def _read_for(plan: str, provider: str):
    """read_config() for one plan, without disturbing the module's globals."""
    import _modules
    old = dict(os.environ)
    os.environ.update({"PROVIDER": provider, "SWITCHYARD_PLAN": plan,
                       "SWITCHYARD_PLANS": PLANS})
    for key in ("SIDECAR_CONCURRENCY", "CLAUDE_MODEL", "CODEX_MODEL", "OPENCODE_MODEL"):
        os.environ.pop(key, None)
    try:
        mod = _modules.reload(server)
        return mod.read_config()
    finally:
        os.environ.clear()
        os.environ.update(old)
        _modules.reload(server)


def test_sidecar_reads_models_from_the_plan():
    """Models are nested under their plan. An earlier version read a top-level
    `deployments:` map, found nothing, and silently served the profile default —
    so every sidecar reported a model nobody had configured."""
    cfg = _read_for("claude-max", "claude")
    assert cfg.source == "config", cfg
    assert "claude-opus-5" in cfg.models and "claude-sonnet-5" in cfg.models, cfg.models
    assert cfg.concurrency == 2, cfg.concurrency
    print(f"  claude-max: concurrency={cfg.concurrency} models={sorted(cfg.models)}")


def test_sidecar_model_aliases_keep_provider_prefixes_where_needed():
    """`openai/claude-opus-5` is `claude-opus-5` to the CLI, but
    `openai/opencode-go/glm-5.3-flash` must keep its provider/model shape.

    Only the leading `openai/` is a LiteLLM provider hint. Anything after it is
    the provider's own name for the model and has to survive, because OpenCode
    genuinely needs `opencode-go/glm-5.3-flash`. Conversely a plan calling an
    API directly must NOT carry such a prefix: `xai/grok-4.6` is what OpenCode
    calls it, and api.x.ai answers 404 for it.
    """
    assert _read_for("opencode-go", "opencode").model == "opencode-go/glm-5.3-flash"
    assert _read_for("openai", "codex").model.startswith("gpt-5.6-")
    print("  only the LiteLLM provider prefix is stripped")


def test_a_missing_plan_is_reported_not_papered_over():
    cfg = _read_for("no-such-plan", "claude")
    assert cfg.source == "fallback", cfg
    print(f"  unknown plan -> source={cfg.source!r}, /health reports ok=false")


def test_disabled_models_are_not_offered():
    cfg = _read_for("openai", "codex")
    import yaml
    plan = yaml.safe_load(open(PLANS))["plans"]["openai"]
    disabled = {k for k, v in plan["models"].items() if (v or {}).get("enabled") is False}
    for key in disabled:
        alias = plan["models"][key]["model"].split("/", 1)[1]
        assert alias not in cfg.models, alias
    print(f"  {len(disabled)} disabled model(s) withheld from the allowlist")


# ------------------------------------------ issue #29: argv E2BIG regression ---
def test_multibyte_prompts_are_measured_in_bytes_not_characters():
    """MAX_ARG_STRLEN counts bytes, but the old check used len() -- characters.

    A 40k-character CJK prompt is 120k UTF-8 bytes, comfortably over the
    128 KiB-per-element limit on paper as 40k characters, but the kernel
    would still refuse to exec it. Without the byte-based check, a CJK
    prompt slips onto argv where it fails execve with E2BIG (issue #29).
    """
    # Each "\u4e00" is 3 UTF-8 bytes; limit//3 chars is *exactly* under the
    # char-count threshold but well over the byte threshold. Add one more
    # so the byte count strictly exceeds STDIN_PROMPT_LIMIT.
    n = server.STDIN_PROMPT_LIMIT // 3 + 1
    huge = "\u4e00" * n
    assert len(huge) < server.STDIN_PROMPT_LIMIT, (len(huge), server.STDIN_PROMPT_LIMIT)
    assert len(huge.encode("utf-8")) > server.STDIN_PROMPT_LIMIT, \
        (len(huge.encode("utf-8")), server.STDIN_PROMPT_LIMIT)

    argv, stdin_data = server.build_argv(huge, None)
    assert stdin_data == huge, "oversized CJK prompt must travel on stdin"
    assert huge not in argv, argv
    print(f"  {n}-char CJK prompt ({len(huge.encode('utf-8'))} bytes) kept off argv")


def test_an_oversized_system_prompt_folds_into_the_stdin_prompt():
    """OpenCode has no system-prompt flag, so an oversized system block has
    nowhere to live but the prompt itself -- and the prompt rides stdin.

    fold_system prepends the caller's system to the prompt when the active
    profile has no system_args key; once both are over STDIN_PROMPT_LIMIT,
    build_argv must still recognise that and put the combined text on stdin
    rather than trying to fit it as one argv element (issue #29).
    """
    huge_system = "s" * (server.STDIN_PROMPT_LIMIT + 10)
    # Drive the same path run_cli does for opencode: fold_system prepends the
    # caller's system text into the prompt and clears it, then build_argv
    # sees the now-oversized prompt and routes it to stdin.
    folded_prompt, folded_system = server.fold_system("small", huge_system)
    argv, stdin_data = server.build_argv(folded_prompt, folded_system)
    assert stdin_data is not None, "oversized prompt+system must ride stdin"
    assert stdin_data.startswith(huge_system), \
        (stdin_data[:40], "...", stdin_data[-40:])
    # No element of argv may be the system text itself.
    assert all(huge_system != element for element in argv), argv
    # The plain prompt also isn't an element (it was folded into stdin).
    assert "small" not in argv, argv
    print(f"  {len(huge_system)}-char system prompt folded into {len(stdin_data)}-char stdin payload")


def test_a_real_spawn_delivers_an_oversized_prompt_on_stdin():
    """End-to-end: an oversized prompt must reach the fake CLI intact on stdin.

    The argv/stdin and spawn-guarding fixes are both necessary: this proves
    the spawn itself does not fail with E2BIG once the prompt is on stdin,
    which is what the gateway would otherwise 500 on (issue #29).
    """
    import asyncio

    fake = tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix="clib-over-", delete=False)
    fake.write(
        "import json, sys\n"
        "data = sys.stdin.read()\n"
        "print(json.dumps({'type':'text','part':{'type':'text','text':"
        "f'got {len(data)} chars'}}))\n")
    fake.close()
    real_cli = server.CLI
    real_bare = server.BARE
    real_args = server.PROFILE["args"]
    huge = "x" * (server.STDIN_PROMPT_LIMIT + 123)
    try:
        server.CLI = sys.executable
        server.BARE = False
        server.PROFILE["args"] = [fake.name]
        async def invoke():
            return await server.run_cli(huge, None, "m")
        payload = asyncio.run(invoke())
        assert payload["result"] == f"got {len(huge)} chars", payload
        print(f"  {len(huge)}-char prompt delivered whole to a real subprocess via stdin")
    finally:
        server.CLI = real_cli
        server.BARE = real_bare
        server.PROFILE["args"] = real_args
        os.unlink(fake.name)


def test_e2big_from_the_spawn_is_413_and_other_spawn_errors_are_502():
    """A failed execve must surface the right HTTP status.

    E2BIG ("Argument list too long") means the request is over what argv
    can carry even after the stdin and file escape hatches -- a 413, not a
    500, so a client can tell "too large" from "down" (issue #29). Any
    other spawn-time OSError is this plan's capacity being broken, and must
    read as 502 so the router cools the plan instead of retrying it.
    """
    import asyncio
    from fastapi import HTTPException

    real_exec = asyncio.create_subprocess_exec
    captured: list = []

    async def fake_exec(*args, **kwargs):
        # Capture the call so the test can confirm the spawn was attempted
        # before the error fired (it failed, but the spawn path was taken).
        captured.append(args[:1])
        raise captured_exc

    # Test both errno values; restore after each so one monkeypatch's
    # tear-down cannot mask the other's setup.
    for errno_val, expected_status, expected_msg in (
            (errno.E2BIG, 413, "too large"),
            (errno.EIO,   502, "could not be spawned")):
        captured_exc = OSError(errno_val, "Arg list too long"
                               if errno_val == errno.E2BIG else "I/O error")
        asyncio.create_subprocess_exec = fake_exec
        try:
            async def invoke():
                return await server._run_cli("hi", None, "m", [])
            try:
                asyncio.run(invoke())
            except HTTPException as exc:
                assert exc.status_code == expected_status, (exc.status_code, exc.detail)
                assert expected_msg in str(exc.detail), exc.detail
                assert captured, "the fake spawn was never called"
            else:
                raise AssertionError(
                    f"expected an HTTPException for errno={errno_val}, got none")
        finally:
            asyncio.create_subprocess_exec = real_exec
    print(f"  E2BIG -> 413, other spawn OSError -> 502 (no 500 leak)")


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} cli-bridge tests passed")
