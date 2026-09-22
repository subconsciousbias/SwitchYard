"""Parser tests for the CLI bridge, against output the CLIs really produced.

The OpenCode fixture is a verbatim capture from `opencode run --format json`.
It exists because the first parser was written from a guess at the event shape:
it found no text, fell back to returning the raw event stream as the assistant's
answer, and reported zero tokens — a failure that looks like a working call.
"""
from __future__ import annotations

import base64
import errno
import json
import os
import sys
import tempfile
from pathlib import Path

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

    Extended for issue #34: the same payload now also surfaces the cache
    breakdown, so LiteLLM and Anthropic-protocol clients (Claude Code via
    /v1/messages) can see reads under `prompt_tokens_details.cached_tokens`
    and creations under a top-level `cache_creation_input_tokens`. Totals
    stay exactly as before — the breakdown is additive, not a recount.
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
    # Issue #25 repro pinned numerically: 95 + 63200 + 800 = 64095.
    assert u["prompt_tokens"] == 64095, u
    assert u["prompt_tokens_details"] == {"cached_tokens": 63200}, u
    assert u["cache_creation_input_tokens"] == 800, u
    print(f"  anthropic cached prompt folded into total: {u}")


def test_openai_shape_cache_tokens_are_not_double_counted():
    """OpenCode/Codex feed `cache_read_tokens` separately, but their
    `input_tokens` already includes those reads per OpenAI convention.
    Only Anthropic's cache_*_input_tokens are excluded from input_tokens,
    so adding them here must be gated on the Anthropic field names —
    otherwise this branch would inflate an OpenCode-style prompt by the
    cached portion a second time.

    Issue #34 keeps the gate strict: an OpenAI-shaped payload carries no
    Anthropic cache field names, so no `prompt_tokens_details` and no
    `cache_creation_input_tokens` are emitted — nothing for LiteLLM to
    misattribute back to the caller's billing.
    """
    payload = {"result": "ok", "usage": {
        "input_tokens": 6194,
        "output_tokens": 18,
        "cache_read_tokens": 1280,
    }}
    u = server.to_openai(payload, "m")["usage"]
    assert u["prompt_tokens"] == 6194, u
    assert u["total_tokens"] == 6212, u
    assert "prompt_tokens_details" not in u, u
    assert "cache_creation_input_tokens" not in u, u
    print(f"  openai-shape cache stays inside input_tokens: {u}")


def test_anthropic_cache_breakdown_emitted_even_when_creation_is_absent():
    """An Anthropic payload may carry cache reads without a creation field
    (a pure-read turn) or carries zero of both — issue #34 asks for a
    stable shape so callers can rely on the keys being present, not on
    their value being non-zero. Reads default to 0; creation defaults
    to 0; the breakdown dict is still emitted.
    """
    payload = {"result": "ok", "usage": {
        "input_tokens": 200,
        "cache_read_input_tokens": 1500,
        "output_tokens": 50,
    }}
    u = server.to_openai(payload, "claude-opus-5")["usage"]
    assert u["prompt_tokens"] == 200 + 1500 + 0, u
    assert u["prompt_tokens_details"] == {"cached_tokens": 1500}, u
    assert u["cache_creation_input_tokens"] == 0, u
    assert u["total_tokens"] == u["prompt_tokens"] + 50, u
    print(f"  anthropic creation-absent still emits zero-valued shape: {u}")


def test_anthropic_cache_breakdown_keys_are_exactly_what_litellm_round_trips():
    """Lock the spelling: `prompt_tokens_details.cached_tokens` (the OpenAI
    slot the gateway maps back to `cache_read_input_tokens` for Anthropic-
    protocol clients) and a top-level `cache_creation_input_tokens` (the
    verbatim Anthropic key, since no OpenAI slot exists for cache creation).
    A different name here would break LiteLLM's translation silently.
    """
    payload = {"result": "ok", "usage": {
        "input_tokens": 1,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 3,
        "output_tokens": 4,
    }}
    u = server.to_openai(payload, "m")["usage"]
    assert set(u) >= {"prompt_tokens", "completion_tokens", "total_tokens",
                      "prompt_tokens_details", "cache_creation_input_tokens"}, u
    assert set(u["prompt_tokens_details"]) == {"cached_tokens"}, u["prompt_tokens_details"]
    assert u["prompt_tokens_details"]["cached_tokens"] == 2, u
    assert u["cache_creation_input_tokens"] == 3, u
    print(f"  anthropic keys locked: {sorted(u)}")


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

    # Zero-usage no-text: the legacy path that still surfaces "yielded no
    # parsed answer". PR #64 (issue #64) carves out nonzero-usage no-text
    # into CliNoTextError + the contract retry; that path is covered by
    # test_final_502_detail_carries_combined_usage_in_contract_shape below.
    bad_stream = ('{"type":"step_start","part":{"type":"step-start"}}\n'
                  '{"type":"step_finish","part":{"type":"step-finish"}}')

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


def test_the_claude_file_flags_carry_an_oversized_system_prompt():
    """The context manager's claude paths, which the opencode default never reaches.

    tests here run PROVIDER=opencode, whose profile has no file-args keys, so
    every other test exercises at most the no-op yield. Reload with
    PROVIDER=claude -- the same env-swap pattern _read_for uses -- to drive
    the real profile through all three branches: append-form file flag,
    replace-form file flag with --exclude-dynamic-system-prompt-sections,
    and the strict no-fallback when replace mode has no replace-form key
    (an override must never quietly become a stack-up; issue #29 review).
    """
    import _modules
    from pathlib import Path

    old = dict(os.environ)
    os.environ.update({"PROVIDER": "claude"})
    os.environ.pop("SYSTEM_MODE", None)
    os.environ.pop("SWITCHYARD_PLAN", None)
    try:
        mod = _modules.reload(server)
        huge = "s" * (mod.STDIN_PROMPT_LIMIT + 10)

        # Append mode: --append-system-prompt-file, temp file unlinked after.
        with mod.system_prompt_file(huge) as (args, system):
            assert system is None, "the file path must consume the system text"
            assert args[0] == "--append-system-prompt-file", args
            path = Path(args[1])
            assert path.read_text() == huge + "\n", "file must carry the caller's text"
        assert not path.exists(), "temp file must be unlinked when the with block exits"

        # Replace mode: --system-prompt-file plus --exclude-dynamic-system-
        # prompt-sections -- the load-bearing detail: replace_extra_args must
        # ride the file path too, or replace mode would stop stripping the
        # CLI's injected sections.
        os.environ["SYSTEM_MODE"] = "replace"
        mod = _modules.reload(server)
        with mod.system_prompt_file(huge) as (args, system):
            assert system is None
            assert args[0] == "--system-prompt-file", args
            assert args[2] == "--exclude-dynamic-system-prompt-sections", args
            assert Path(args[1]).read_text() == huge + "\n"

        # Replace mode with no replace-form key: unchanged yield, never the
        # append form.
        saved = mod.PROFILE.pop("system_file_args_replace")
        try:
            with mod.system_prompt_file(huge) as (args, system):
                assert args == [] and system == huge, (args, system[:40])
        finally:
            mod.PROFILE["system_file_args_replace"] = saved
    finally:
        os.environ.clear()
        os.environ.update(old)
        _modules.reload(server)
    print("  claude file flags: append + replace(+extras) covered; "
          "replace without the key stays no-op; temp files cleaned up")


# ----------------------------------------------------------- images (issue #30) ---
# A 1x1 magenta PNG. Decodable to bytes, recognised as image/png, small enough
# to inline anywhere. The bug being fixed: a request carrying this used to be
# flattened into an empty prompt and answered from prior, with the model
# confidently naming a near-white hex (e.g. #EDF6EC) as the swatch's colour.
PNG_MAGENTA = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_stage_images_writes_decodable_files_for_both_image_shapes():
    """Both the Anthropic and the OpenAI image shapes must be decoded and
    written to disk, and the original block must be replaced with a text
    marker that survives into flatten(). Otherwise the model never sees the
    bytes and the prompt loses any pointer to where they are.
    """
    import shutil
    img_dir = Path(tempfile.mkdtemp(prefix="cli-imgtest-"))
    try:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "what colour is this swatch?"},
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png",
                                              "data": base64.b64encode(PNG_MAGENTA).decode()}},
                {"type": "image_url", "image_url": {"url":
                    f"data:image/png;base64,{base64.b64encode(PNG_MAGENTA).decode()}"}},
            ]},
        ]
        paths = server.stage_images(messages, img_dir)
        assert len(paths) == 2, paths
        # The bytes round-trip back to what we put in.
        assert paths[0].read_bytes() == PNG_MAGENTA, paths[0]
        assert paths[1].read_bytes() == PNG_MAGENTA, paths[1]
        assert paths[0].suffix == ".png", paths[0].suffix
        # The original image blocks are gone; a text marker carries the path.
        content = messages[0]["content"]
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert len(text_blocks) == 3, content
        assert "[image 1:" in text_blocks[1]["text"], text_blocks[1]
        assert "[image 2:" in text_blocks[2]["text"], text_blocks[2]
        # flatten() then joins the markers into a single prompt the model sees.
        prompt, _ = server.flatten(messages)
        assert "[image 1:" in prompt and "[image 2:" in prompt, prompt
        print(f"  both shapes decoded to {paths[0].name}, {paths[1].name}; "
              f"markers survived into the flattened prompt")
    finally:
        shutil.rmtree(img_dir, ignore_errors=True)


def test_image_unsupported_for_remote_url_not_a_silent_drop():
    """A remote http(s) URL cannot be fetched from here, so the request must
    fail loudly with HTTP 400 images_unsupported -- never with a confident
    wrong answer (issue #30). The earlier behaviour was to flatten the
    request into nothing, then return the model's prior over a near-white
    hex.
    """
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
    ]}]
    try:
        server.stage_images(messages, Path(tempfile.mkdtemp(prefix="cli-imgru-")))
    except server.ImageUnsupportedError as exc:
        http = exc.http()
        assert http.status_code == 400, http.status_code
        assert http.detail["error"]["type"] == "images_unsupported", http.detail
        print(f"  remote URL -> HTTP 400 ({http.detail['error']['type']!r})")
        return
    raise AssertionError("a remote URL must raise ImageUnsupportedError, "
                         "not be silently dropped")


def test_stage_or_fail_cleans_up_on_failure():
    """stage_or_fail owns a temp dir it must remove when staging fails.
    Otherwise an unsupported block would leave an empty directory behind on
    every failed request -- a slow leak that no caller can see.
    """
    before = set(os.listdir(tempfile.gettempdir()))
    try:
        server.stage_or_fail([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
        ]}])
    except server.ImageUnsupportedError:
        after = set(os.listdir(tempfile.gettempdir()))
        leftover = (after - before) & {p for p in after if p.startswith("swimg-")}
        assert not leftover, leftover
        print("  stage_or_fail removed its temp dir on the failure path")
        return
    raise AssertionError("expected ImageUnsupportedError")


def test_stage_or_fail_cleans_up_on_unexpected_failure():
    """stage_or_fail broadened its cleanup so ANY exception removes the temp
    dir, not just ImageUnsupportedError. A disk full at write_bytes used to
    leak an empty swimg-* dir on every retry; the fix mirrors cli_bridge's
    text-path try/finally pattern. Verified by raising mid-stage with a
    monkey-patched _stage_image -- if the broadened cleanup is reverted to
    the old ImageUnsupportedError-only except, the leftover check fires.
    """
    before = set(os.listdir(tempfile.gettempdir()))
    real_stage_image = server._stage_image

    def boom(block, img_dir, n):
        # Something that isn't ImageUnsupportedError: the old code would
        # not clean the temp dir, leaking it.
        raise OSError("simulated full disk at write_bytes")

    server._stage_image = boom
    try:
        try:
            server.stage_or_fail([{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png",
                                              "data": base64.b64encode(PNG_MAGENTA).decode()}},
            ]}])
        except OSError:
            after = set(os.listdir(tempfile.gettempdir()))
            leftover = (after - before) & {p for p in after if p.startswith("swimg-")}
            assert not leftover, leftover
            print("  stage_or_fail removed its temp dir on a non-ImageUnsupportedError failure")
            return
    finally:
        server._stage_image = real_stage_image
    raise AssertionError("expected OSError from the simulated write_bytes failure")


def test_codex_argv_carries_one_minus_i_per_image():
    """codex takes images via repeatable `-i FILE`. Every staged file must
    appear on argv with its own flag, in order, so the CLI receives the
    same image set the caller sent.

    codex's profile has no `bare_args` list (the harness's tool suppression
    is built in via `--dangerously-bypass-approvals-and-sandbox` on the
    MCP path; the text path does not need it). So nothing about Read or
    disallowed-tools is asserted here -- only that the image flag survives.
    """
    import shutil
    saved_provider, saved_profile, saved_cli = server.PROVIDER, server.PROFILE, server.CLI
    try:
        server.PROVIDER = "codex"
        server.PROFILE = server.PROFILES["codex"]
        server.CLI = server.PROFILE["cli"]
        img = Path(tempfile.mkdtemp(prefix="cli-imgargv-")) / "00.png"
        img.write_bytes(PNG_MAGENTA)
        argv, stdin_data = server.build_argv("look at this", None, None,
                                              image_paths=[img])
        # codex takes `-i FILE` once per image (repeatable).
        i_args = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
        assert i_args == [str(img)], argv
        assert stdin_data is None
        print(f"  codex argv carries -i for the staged image")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI = saved_provider, saved_profile, saved_cli
        shutil.rmtree(img.parent, ignore_errors=True)


def test_opencode_argv_carries_one_minus_f_per_image():
    """opencode takes images via repeatable `-f FILE`. Same contract as codex
    with a different flag name; both profiles carry every staged file.
    """
    import shutil
    saved_provider, saved_profile, saved_cli = server.PROVIDER, server.PROFILE, server.CLI
    try:
        server.PROVIDER = "opencode"
        server.PROFILE = server.PROFILES["opencode"]
        server.CLI = server.PROFILE["cli"]
        img = Path(tempfile.mkdtemp(prefix="cli-imgargv2-")) / "00.png"
        img.write_bytes(PNG_MAGENTA)
        argv, stdin_data = server.build_argv("look at this", None, None,
                                              image_paths=[img])
        f_args = [argv[i + 1] for i, a in enumerate(argv) if a == "-f"]
        assert f_args == [str(img)], argv
        assert stdin_data is None
        print(f"  opencode argv carries -f for the staged image")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI = saved_provider, saved_profile, saved_cli
        shutil.rmtree(img.parent, ignore_errors=True)


def test_claude_argv_uses_add_dir_and_allows_read_for_images():
    """Claude Code has no image flag; the staged files must be reachable via
    Read. `--add-dir <imgdir>` adds the directory and `--allowed-tools
    Read(<imgdir>/**)` lifts Read out of the bare-mode disallowed-tools
    list. Read must NOT also be in --disallowed-tools (the bare list does
    include it, so the image variant's bare_args_images is the one used).
    """
    import shutil
    saved_provider, saved_profile, saved_cli = server.PROVIDER, server.PROFILE, server.CLI
    try:
        server.PROVIDER = "claude"
        server.PROFILE = server.PROFILES["claude"]
        server.CLI = server.PROFILE["cli"]
        img_dir = Path(tempfile.mkdtemp(prefix="cli-imgargv3-"))
        img = img_dir / "img" / "00.png"
        img.parent.mkdir(parents=True)
        img.write_bytes(PNG_MAGENTA)
        argv, _ = server.build_argv("look at this", None, None,
                                      image_paths=[img])
        # --add-dir <imgdir> + --allowed-tools Read(<imgdir>/**).
        assert "--add-dir" in argv, argv
        assert str(img.parent) in argv, argv
        assert "--allowed-tools" in argv, argv
        allowed = argv[argv.index("--allowed-tools") + 1]
        assert allowed.startswith("Read(") and allowed.endswith("/**)"), allowed
        # The image variant's bare_args_images is in argv: max-turns 4, no
        # Read in the disallowed list.
        assert "4" in argv, argv
        idx = argv.index("--disallowed-tools")
        disallowed = argv[idx + 1]
        for tool in disallowed.split(","):
            assert tool != "Read", (disallowed, tool)
        print(f"  claude argv: --add-dir={img.parent}, --allowed-tools={allowed}")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI = saved_provider, saved_profile, saved_cli
        shutil.rmtree(img_dir, ignore_errors=True)


def test_no_image_blocks_means_image_note_is_empty():
    """`has_image_blocks` is the gate for staging. A request with only text
    must not pay the staging cost, and `image_note` on an empty path list
    must still return something sensible (it is appended unconditionally
    by the handle_chat path; the contract is that an empty list is a
    no-op).
    """
    assert not server.has_image_blocks([{"role": "user", "content": "hi"}])
    assert not server.has_image_blocks([{"role": "user", "content": [
        {"type": "text", "text": "hi"}]}])
    assert server.has_image_blocks([{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                              + base64.b64encode(PNG_MAGENTA).decode()}}]}])
    print("  text-only requests skip staging; mixed requests are detected")


# ------------------------------------------------- issue #64: no-text + retry ---
def test_no_text_with_charged_usage_preserves_usage_on_parse():
    """Issue #64 acceptance (a): a bug-shaped stream (step_start + step_finish
    with nonzero tokens but no text event) must raise a parse failure that
    carries the usage, not discard it. The provider bills those tokens; the
    original JSONDecodeError swallowed them, so the ledger never saw the
    charge. CliNoTextError exists to carry them forward to the retry path."""
    stream = ('{"type":"step_start","part":{"type":"step-start"}}\n'
              '{"type":"step_finish","part":{"type":"step-finish",'
              '"tokens":{"input":42,"output":7,"reasoning":3,'
              '"cache":{"read":1280}}}}')
    try:
        server.parse_output(stream)
    except server.CliNoTextError as exc:
        u = exc.usage
        assert u["input_tokens"] == 42, u
        # 7 output + 3 reasoning billed into output_tokens, per the parser's
        # reasoning-into-output convention (see parse_output comments).
        assert u["output_tokens"] == 10, u
        assert u["cache_read_tokens"] == 1280, u
        print(f"  no-text-with-usage preserves usage on parse: {u}")
        return
    raise AssertionError("expected CliNoTextError on no-text-with-usage stream")


def test_final_502_detail_carries_combined_usage_in_contract_shape():
    """Issue #64 acceptance (b): on two consecutive no-text-with-usage
    failures, the final 502 detail must match the contract exactly:
        {PROVIDER} cli emitted tokens with no text
            (prompt_tokens=N, completion_tokens=M)
    where N/M are the combined charges of both attempts. The gateway's
    classifier matches the literal phrase and extracts the numbers from the
    parenthesised fields, so the format is load-bearing, not cosmetic."""
    import asyncio
    from fastapi import HTTPException

    bad_stream = ('{"type":"step_start","part":{"type":"step-start"}}\n'
                  '{"type":"step_finish","part":{"type":"step-finish",'
                  '"tokens":{"input":100,"output":50}}}')
    fake = tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix="clib-bad-", delete=False)
    fake.write("import sys\n"
               f"sys.stdout.write({bad_stream!r})\n")
    fake.close()
    real_cli, real_bare, real_args = server.CLI, server.BARE, server.PROFILE["args"]
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
            expected = ("opencode cli emitted tokens with no text "
                        "(prompt_tokens=200, completion_tokens=100)")
            assert exc.detail == expected, (exc.detail, expected)
            print(f"  final 502 detail matches contract: {exc.detail!r}")
            return
    finally:
        server.CLI, server.BARE, server.PROFILE["args"] = real_cli, real_bare, real_args
        os.unlink(fake.name)
    raise AssertionError("expected an HTTPException")


def test_retry_fires_only_on_no_text_with_usage_and_merges_first_usage():
    """Issue #64 acceptance (c, part 1): the in-process retry must fire
    only on the no-text-with-usage shape. Other failure modes — auth,
    quota, timeout, empty stdout, nonzero exit — keep their existing
    single-attempt behaviour, so the same upstream blip is not amplified
    into a double bill.

    Issue #64 acceptance (c, part 2): on a successful retry, the first
    attempt's tokens must be folded into the payload so the ledger books
    both attempts exactly once each. A dropped attempt-1 is the bug this
    retry is here to prevent.
    """
    import asyncio
    from fastapi import HTTPException

    counter_path = tempfile.NamedTemporaryFile(
        "w", suffix=".counter", prefix="clib-cnt-", delete=False).name
    open(counter_path, "w").write("0")

    def make_fake(*, on_call_one: str = "", on_call_other: str = "",
                  stderr: str = "", rc: int = 0) -> str:
        """Write a tiny Python CLI that increments counter_path each call.

        `on_call_one` is printed on the first call, `on_call_other` on every
        subsequent call. Used to script the "first attempt bad, second
        attempt good" retry shape without two separate fake files.
        """
        body = (f"import sys\n"
                f"p = {counter_path!r}\n"
                f"n = int(open(p).read() or '0') + 1\n"
                f"open(p, 'w').write(str(n))\n"
                f"if n == 1:\n"
                f"  sys.stdout.write({on_call_one!r})\n"
                f"else:\n"
                f"  sys.stdout.write({on_call_other!r})\n"
                f"sys.stderr.write({stderr!r})\n"
                f"sys.exit({rc})\n")
        f = tempfile.NamedTemporaryFile(
            "w", suffix=".py", prefix="clib-rf-", delete=False)
        f.write(body)
        f.close()
        return f.name

    real_cli, real_bare, real_args = server.CLI, server.BARE, server.PROFILE["args"]
    try:
        server.CLI = sys.executable
        server.BARE = False

        # --- auth failure: single attempt, 401 ---
        open(counter_path, "w").write("0")
        auth_fake = make_fake(stderr="please run `claude login` to authenticate",
                              rc=1)
        server.PROFILE["args"] = [auth_fake]
        try:
            asyncio.run(server._run_cli("hi", None, "xai/grok-4.6", []))
        except HTTPException as exc:
            assert exc.status_code == 401, (exc.status_code, exc.detail)
        else:
            raise AssertionError("expected a 401 HTTPException")
        calls = int(open(counter_path).read())
        assert calls == 1, f"auth failure must not retry (calls={calls})"
        os.unlink(auth_fake)

        # --- quota failure: single attempt, 429 ---
        open(counter_path, "w").write("0")
        quota_fake = make_fake(stderr="You've hit your usage limit.", rc=1)
        server.PROFILE["args"] = [quota_fake]
        try:
            asyncio.run(server._run_cli("hi", None, "xai/grok-4.6", []))
        except HTTPException as exc:
            assert exc.status_code == 429, (exc.status_code, exc.detail)
        else:
            raise AssertionError("expected a 429 HTTPException")
        calls = int(open(counter_path).read())
        assert calls == 1, f"quota failure must not retry (calls={calls})"
        os.unlink(quota_fake)

        # --- empty stdout: single attempt, 502 ---
        open(counter_path, "w").write("0")
        empty_fake = make_fake(rc=0, on_call_other="")   # both calls print nothing
        server.PROFILE["args"] = [empty_fake]
        try:
            asyncio.run(server._run_cli("hi", None, "xai/grok-4.6", []))
        except HTTPException as exc:
            assert exc.status_code == 502, (exc.status_code, exc.detail)
        else:
            raise AssertionError("expected a 502 HTTPException")
        calls = int(open(counter_path).read())
        assert calls == 1, f"empty stdout must not retry (calls={calls})"
        os.unlink(empty_fake)

        # --- successful retry merges first attempt's usage ---
        # First attempt emits no-text-with-usage; second emits a normal text
        # event with its own usage. The final payload must include both,
        # summed, so the ledger books the retry's first attempt too.
        open(counter_path, "w").write("0")
        no_text = ('{"type":"step_finish","part":{"type":"step-finish",'
                   '"tokens":{"input":777,"output":33}}}')
        good = ('{"type":"text","part":{"type":"text","text":"ANSWER"}}\n'
                '{"type":"step_finish","part":{"type":"step-finish",'
                '"tokens":{"input":111,"output":7,"reasoning":2}}}')
        retry_fake = make_fake(on_call_one=no_text, on_call_other=good)
        server.PROFILE["args"] = [retry_fake]
        payload = asyncio.run(server._run_cli("hi", None, "xai/grok-4.6", []))
        assert payload["result"] == "ANSWER", payload
        u = payload["usage"]
        assert u["input_tokens"] == 888, u      # 777 + 111
        # 33 from attempt 1, 7+2 reasoning from attempt 2; reasoning rolls
        # into output_tokens, so 33 + 9 = 42.
        assert u["output_tokens"] == 42, u
        calls = int(open(counter_path).read())
        assert calls == 2, f"successful retry must call CLI twice (calls={calls})"
        os.unlink(retry_fake)
        print("  auth/quota/empty single-attempt; no-text-with-usage retries "
              "and folds attempt 1's usage into attempt 2's payload")
    finally:
        server.CLI, server.BARE, server.PROFILE["args"] = real_cli, real_bare, real_args
        if os.path.exists(counter_path):
            os.unlink(counter_path)


def test_zero_usage_no_text_does_not_retry_and_keeps_existing_message():
    """Issue #64 acceptance (d): a no-text stream with ZERO usage must NOT
    trigger the retry, and its failure must keep the legacy 502 message
    shape ('yielded no parsed answer: ...'). Only nonzero charged usage
    unlocks the retry path — a plain empty-no-text stream is the legacy
    behaviour, and a silent change to it would alter the gateway's view
    of every other no-text failure mode."""
    import asyncio
    from fastapi import HTTPException

    # (i) parse_output directly: zero usage -> json.JSONDecodeError, NOT
    # CliNoTextError. CliNoTextError is reserved for nonzero usage, so an
    # empty stream must never reach _run_cli's retry branch.
    stream = ('{"type":"step_start","part":{"type":"step-start"}}\n'
              '{"type":"step_finish","part":{"type":"step-finish"}}')
    try:
        server.parse_output(stream)
    except json.JSONDecodeError as exc:
        assert "no assistant text" in str(exc), exc
        assert not isinstance(exc, server.CliNoTextError), \
            "zero-usage no-text must NOT raise CliNoTextError"
    else:
        raise AssertionError("expected json.JSONDecodeError on zero-usage stream")

    # (ii) end-to-end: a no-text no-usage fake CLI is called exactly once,
    # and the final 502 keeps the legacy "yielded no parsed answer" shape.
    counter_path = tempfile.NamedTemporaryFile(
        "w", suffix=".counter", prefix="clib-zu-", delete=False).name
    open(counter_path, "w").write("0")
    bad = ('{"type":"step_start","part":{"type":"step-start"}}\n'
           '{"type":"step_finish","part":{"type":"step-finish"}}')
    f = tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix="clib-zu-", delete=False)
    f.write(f"import sys\n"
            f"p = {counter_path!r}\n"
            f"n = int(open(p).read() or '0') + 1\n"
            f"open(p, 'w').write(str(n))\n"
            f"sys.stdout.write({bad!r})\n")
    f.close()
    real_cli, real_bare, real_args = server.CLI, server.BARE, server.PROFILE["args"]
    try:
        server.CLI = sys.executable
        server.BARE = False
        server.PROFILE["args"] = [f.name]
        try:
            asyncio.run(server._run_cli("hi", None, "xai/grok-4.6", []))
        except HTTPException as exc:
            assert exc.status_code == 502, exc.status_code
            assert "yielded no parsed answer" in str(exc.detail), exc.detail
            assert "emitted tokens with no text" not in str(exc.detail), exc.detail
        calls = int(open(counter_path).read())
        assert calls == 1, f"zero-usage no-text must NOT retry (calls={calls})"
    finally:
        server.CLI, server.BARE, server.PROFILE["args"] = real_cli, real_bare, real_args
        os.unlink(f.name)
        os.unlink(counter_path)
    print("  zero-usage no-text skips the retry and keeps the legacy 502 message")


# --------------------------------------- issue #44: caller-env block + first-turn reminder ---
def test_text_path_argv_carries_env_block_when_resolved():
    """When the request body names the caller's environment (an OpenCode
    `<environment>` block in the system message), the `[SwitchYard tool
    execution environment]` block is appended to the system string that
    build_argv passes to the CLI. The env values are what the request
    said, NOT what the relay container sees.

    Note: build_argv itself does not run the env resolution -- the
    resolution is _handle_chat's job, then it forwards the augmented
    (prompt, system) to build_argv. We mirror that here: resolve via
    caller_env, render the block, then build_argv carries the rendered
    text on the CLI's argv. This is the contract for what the CLI
    receives end to end.
    """
    saved_provider, saved_profile, saved_cli = server.PROVIDER, server.PROFILE, server.CLI
    try:
        server.PROVIDER = "claude"
        server.PROFILE = server.PROFILES["claude"]
        server.CLI = server.PROFILE["cli"]
        body = {
            "model": "m",
            "messages": [
                {"role": "system", "content": (
                    "<environment>\n"
                    "  <working_directory>C:\\Users\\demo\\proj</working_directory>\n"
                    "  <platform>windows</platform>\n"
                    "</environment>"
                )},
                {"role": "user", "content": "Hi"},
            ],
        }
        prompt, system = server.flatten(body["messages"])
        # Mirror the _handle_chat path: parse the request, render the
        # block + reminder, then build_argv.
        env = server._caller_env.parse_request(body)
        assert env is not None and env.platform == "windows", env
        system = (f"{system}\n\n{server._caller_env.render_system_block(env)}"
                  if system else server._caller_env.render_system_block(env))
        reminder = server._caller_env.render_first_turn_reminder(env)
        prompt = f"{reminder}\n\n{prompt}" if prompt else reminder
        argv, _ = server.build_argv(prompt, system, "m")
        # The system string is the second-to-last element of argv (the
        # --append-system-prompt flag and value are the only system-carrying
        # pair in the claude profile's append-mode path).
        i = argv.index("--append-system-prompt") + 1
        rendered_system = argv[i]
        assert "[SwitchYard tool execution environment]" in rendered_system, \
            argv
        assert "Caller platform: windows" in rendered_system, argv
        assert r"Caller working directory: C:\Users\demo\proj" in rendered_system, \
            argv
        # The reminder on the first user turn is the prompt (substituted
        # into {-p prompt}).
        i_p = argv.index("-p") + 1
        rendered_prompt = argv[i_p]
        assert "[SwitchYard: tools execute on windows in C:\\Users\\demo\\proj" \
            in rendered_prompt, argv
        print("  system argv carries the env block + first-turn reminder")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI = saved_provider, saved_profile, saved_cli


def test_text_path_unknown_env_renders_unknown_wording_never_linux():
    """When neither the request nor the metadata carries an env, the
    unknown-env wording is rendered -- explicitly, with the literal
    word "unknown" -- and the relay's own Linux / /app environment is
    NEVER substituted as a fallback. This is the whole bug we are
    avoiding; a regression here would have the inner CLI's `# Environment`
    block continue to mislead the model about the caller.
    """
    saved_provider, saved_profile, saved_cli = server.PROVIDER, server.PROFILE, server.CLI
    try:
        server.PROVIDER = "claude"
        server.PROFILE = server.PROFILES["claude"]
        server.CLI = server.PROFILE["cli"]
        body = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "no env here"},
                {"role": "user", "content": "Hi"},
            ],
        }
        prompt, system = server.flatten(body["messages"])
        # _handle_chat's path: parse_request -> unknown -> render unknown.
        env = server._caller_env.parse_request(body)
        if env is None:
            # Mirror the cli_bridge read path: use CallerEnvironmentSettings
            # from switchyard.models (or the local fallback).
            try:
                from switchyard.models import CallerEnvironmentSettings
                ce_cfg = CallerEnvironmentSettings()
            except ImportError:
                ce_cfg = None
            env = server._caller_env.resolve(body, ce_cfg) if ce_cfg else None
        if env is None:
            env = server._caller_env.CallerEnvironment.unknown()
        assert env.source == "unknown", env
        system = (f"{system}\n\n{server._caller_env.render_system_block(env)}"
                  if system else server._caller_env.render_system_block(env))
        reminder = server._caller_env.render_first_turn_reminder(env)
        prompt = f"{reminder}\n\n{prompt}" if prompt else reminder
        argv, _ = server.build_argv(prompt, system, "m")
        i = argv.index("--append-system-prompt") + 1
        rendered_system = argv[i]
        # The literal "unknown" string for every field.
        assert "Caller platform: unknown" in rendered_system, argv
        assert "Caller working directory: unknown" in rendered_system, argv
        assert "Caller shell: unknown" in rendered_system, argv
        # NEVER the relay's own Linux / /app wording.
        assert "Platform: linux" not in rendered_system, argv
        assert "/app/mcp_bridge" not in rendered_system, argv
        assert "/app/cli_bridge" not in rendered_system, argv
        # Reminder on the first turn uses the unknown-env wording.
        i_p = argv.index("-p") + 1
        rendered_prompt = argv[i_p]
        assert "[SwitchYard: caller environment unknown" in rendered_prompt, argv
        assert "do not assume the relay container" in rendered_prompt, argv
        print("  unknown env: literal 'unknown' wording, no Linux / /app leakage")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI = saved_provider, saved_profile, saved_cli


def test_text_path_spawn_cwd_is_a_fresh_temp_dir():
    """Each call to `_run_cli` spawns the inner CLI in a fresh
    `tempfile.mkdtemp(prefix='sy-cli-')`, NOT in /app/cli_bridge. A fake
    CLI prints its own cwd; the assert confirms the spawned process
    actually saw the per-call temp dir.

    Belt-and-braces: a session-shaped directory beneath the temp
    partition keeps paths underneath it from looking like a meaningful
    project root the model could safely use (issue #44).
    """
    import asyncio
    # Use the opencode parser shape so the events_json parser accepts
    # the fake CLI output (the opencode profile is the test default).
    fake_cli = tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix="clib-cwd-", delete=False)
    fake_cli.write(
        "import json, os\n"
        # opencode emits one event per line: a text event, then a step_finish.
        "cwd = os.getcwd()\n"
        "print(json.dumps({'type': 'text', 'part': "
        "    {'type': 'text', 'text': cwd}}))\n"
        "print(json.dumps({'type': 'step_finish', 'part': "
        "    {'type': 'step-finish', "
        "     'tokens': {'input': 1, 'output': 1, 'reasoning': 0}}}))\n")
    fake_cli.close()
    real_cli = server.CLI
    real_bare = server.BARE
    real_args = server.PROFILE["args"]
    try:
        server.CLI = sys.executable
        server.BARE = False
        server.PROFILE["args"] = [fake_cli.name]
        async def invoke():
            return await server.run_cli("hi", None, "m")
        payload = asyncio.run(invoke())
        cwd = payload["result"]
        assert cwd != "/app/cli_bridge", cwd
        assert os.path.basename(cwd).startswith("sy-cli-"), cwd
        print(f"  inner CLI spawned in cwd={cwd} (under tempfile, not /app)")
    finally:
        server.CLI = real_cli
        server.BARE = real_bare
        server.PROFILE["args"] = real_args
        os.unlink(fake_cli.name)


def test_text_path_no_tools_passthrough_with_metadata_still_parses():
    """A no-tools text request that DOES carry `metadata.switchyard.caller_env`
    is a valid passthrough: it parses the stamp (no tools to refuse), it
    does NOT probe (no tool round-trip), and the resolved env drives the
    system block. The 400-on-tools behaviour for CLI-backed plans is
    untouched -- this test uses no tools on purpose, to confirm the
    caller-env field flows through _handle_chat without short-circuiting.
    """
    from fastapi import HTTPException
    saved_provider, saved_profile, saved_cli = server.PROVIDER, server.PROFILE, server.CLI
    try:
        server.PROVIDER = "claude"
        server.PROFILE = server.PROFILES["claude"]
        server.CLI = server.PROFILE["cli"]
        # The text path is _handle_chat. We can't drive it via the http
        # route from a test cleanly, so call it directly. It raises
        # HTTPException for image-unsupported or capacity issues; we
        # only assert no tool-refusal 400 fires.
        body = {
            "model": "m",
            "metadata": {"switchyard": {"caller_env": {
                "platform": "darwin", "cwd": "/Users/x/p",
                "shell": "zsh", "source": "request"}}},
            "messages": [{"role": "user", "content": "hi"}],
        }
        # No tools -> not a 400. The gate / spawn guard may raise 429/502
        # under load, but the call MUST not 400 on the tools check.
        try:
            import asyncio as _asyncio
            _asyncio.run(server._handle_chat(body))
        except HTTPException as exc:
            assert exc.status_code != 400 or \
                (isinstance(exc.detail, dict) and
                 exc.detail.get("error", {}).get("type") != "tools_unsupported"), \
                f"text path must not 400 on tools when no tools present: {exc.detail}"
        except Exception:
            pass  # 429/502/etc. from gate / spawn is not under test here
        print("  no-tools + metadata-stamped caller_env -> no tools_unsupported 400")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI = saved_provider, saved_profile, saved_cli


def test_text_path_system_mode_replace_argv_unchanged_apart_from_system():
    """SYSTEM_MODE=replace is the path where `--system-prompt` replaces the
    CLI's own agent prompt instead of stacking on top of it. Adding the
    env block MUST NOT alter that contract: the replace-mode flag, the
    file-form flag, and `--exclude-dynamic-system-prompt-sections` all
    stay present; only the system content string changes.

    Verified by checking the argv keeps the replace-mode markers AND
    that the env block rides inside the system content (not on top of
    it).
    """
    import _modules
    old = dict(os.environ)
    os.environ.update({"PROVIDER": "claude"})
    os.environ.pop("SYSTEM_MODE", None)
    os.environ.pop("SWITCHYARD_PLAN", None)
    try:
        mod = _modules.reload(server)
        os.environ["SYSTEM_MODE"] = "replace"
        mod = _modules.reload(server)
        body = {
            "model": "m",
            "messages": [
                {"role": "system", "content": (
                    "<environment>\n"
                    "  <working_directory>C:\\Users\\demo\\proj</working_directory>\n"
                    "  <platform>windows</platform>\n"
                    "</environment>"
                )},
                {"role": "user", "content": "Hi"},
            ],
        }
        prompt, system = mod.flatten(body["messages"])
        # Mirror _handle_chat: resolve env, render block, append to system.
        env = mod._caller_env.parse_request(body)
        assert env is not None and env.platform == "windows", env
        env_block = mod._caller_env.render_system_block(env)
        system = (f"{system}\n\n{env_block}" if system else env_block)
        argv, _ = mod.build_argv(prompt, system, "m")
        # Replace-mode flags stay present.
        assert "--system-prompt" in argv, argv
        assert "--exclude-dynamic-system-prompt-sections" in argv, argv
        # The env block rides INSIDE the --system-prompt argument, not on
        # top of it.
        i = argv.index("--system-prompt") + 1
        sys_arg = argv[i]
        assert "[SwitchYard tool execution environment]" in sys_arg, argv
        assert "Caller platform: windows" in sys_arg, argv
        assert r"Caller working directory: C:\Users\demo\proj" in sys_arg, argv
        # The caller-supplied original system content survives in the same arg.
        assert "<environment>" in sys_arg, argv
        # Other replace-mode invariants: the file form's flag is NOT
        # present (the system block stays inline because the system
        # string is well under MAX_ARG_STRLEN, even with the env block
        # appended).
        assert "--system-prompt-file" not in argv, argv
        print("  SYSTEM_MODE=replace: --system-prompt carries env block; "
              "--exclude-dynamic-system-prompt-sections preserved")
    finally:
        # Restore env AND reload the module to flush SYSTEM_MODE=replace
        # state so subsequent tests see the original (append-mode) profile.
        os.environ.clear()
        os.environ.update(old)
        _modules.reload(server)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} cli-bridge tests passed")
