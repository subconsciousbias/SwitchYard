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
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# The provider every test here assumes `server` was loaded under.
MODULE_PROVIDER = "opencode"
os.environ["PROVIDER"] = MODULE_PROVIDER

from _modules import load  # noqa: E402

server = load("cli_bridge_server",
              os.path.join(os.path.dirname(HERE), "sidecars", "cli_bridge", "server.py"))


def _restore_env_and_reload(old: dict) -> None:
    """Undo a test's env swap and re-execute `server` under MODULE_PROVIDER.

    `old` cannot be trusted for PROVIDER: pytest imports every test module
    before running any, and test_mcp_bridge sets PROVIDER=claude at import,
    so by run time os.environ says claude. Reloading under that turned
    `server` into the claude profile for every later test in this file --
    which then fed opencode fixture output to the claude parser.
    """
    import _modules
    os.environ.clear()
    os.environ.update(old)
    os.environ["PROVIDER"] = MODULE_PROVIDER
    try:
        _modules.reload(server)
    finally:
        os.environ.clear()
        os.environ.update(old)

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
    """The post-hoc enforcer must round-trip an over-budget payload:
    estimate_tokens says it is too long, enforce_max_tokens cuts it to
    max_tokens*4 bytes, the unchanged-overage path stays untouched, and
    finish_reason flips between "length" and "stop" accordingly.

    The old fold-into-prompt shape is gone (see test_fold_max_tokens_is_gone);
    this test now exercises the replacement: a real cut on real bytes, with
    the helper's "stop" / "length" contract spelled out.
    """
    # estimate_tokens: 4 bytes/token, ceil. ASCII 4 bytes -> 1 token.
    assert server.estimate_tokens("") == 0
    assert server.estimate_tokens("abcd") == 1
    assert server.estimate_tokens("abcde") == 2    # 5 bytes ceil(/4) = 2

    # enforce_max_tokens round-trip on the over-budget path.
    payload = {"result": "x" * 256, "usage": {"output_tokens": 64}}
    cut, fr = server.enforce_max_tokens(payload, 50)   # 50 tokens -> 200 byte cap
    assert fr == "length"
    assert len(cut["result"]) == 200
    assert cut["usage"] == payload["usage"]    # usage left alone
    assert payload["result"] == "x" * 256        # original untouched

    # Under-cap is a no-op with finish_reason "stop".
    small = {"result": "x" * 40, "usage": {}}    # 40 bytes -> 10 tokens
    kept, fr2 = server.enforce_max_tokens(small, 50)
    assert fr2 == "stop"
    assert kept["result"] == "x" * 40

    # No max_tokens (None / 0 / negative) is the unchanged pass-through.
    for mt in (None, 0, -1):
        passthrough, fr3 = server.enforce_max_tokens(payload, mt)
        assert fr3 == "stop", (mt, fr3)
        assert passthrough["result"] == payload["result"]
    print(f"  estimate_tokens rounds 4 bytes/token; over -> 'length' at "
          f"{len(cut['result'])} bytes, usage preserved; under + no-cap -> 'stop'")


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
        _restore_env_and_reload(old)


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
    """A missing plan must reach the caller as `source=fallback`, not as a
    crash.

    `read_config()` propagates the missing-plan error (so a deploy bug or
    a typo is loud), but `config()` catches it and serves the fallback
    Config on cold start so /health still reports `source=fallback`
    rather than 500ing every request. Drive `config()` with `_config`
    reset to simulate that cold start -- the same reset pattern the new
    last-good regression tests use.
    """
    import _modules
    old = dict(os.environ)
    os.environ.update({"PROVIDER": "claude", "SWITCHYARD_PLAN": "no-such-plan",
                       "SWITCHYARD_PLANS": PLANS})
    for key in ("SIDECAR_CONCURRENCY", "CLAUDE_MODEL", "CODEX_MODEL", "OPENCODE_MODEL"):
        os.environ.pop(key, None)
    try:
        mod = _modules.reload(server)
        mod._config = None
        cfg = mod.config()
        assert cfg.source == "fallback", cfg
    finally:
        _restore_env_and_reload(old)
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


def test_an_oversized_system_prompt_never_reaches_opencode_argv():
    """OpenCode takes the caller's system prompt as the agent's `prompt:` in
    the spawn dir's opencode.json (issue #264), so an oversized system block
    is neither folded into the user turn nor put on argv (issue #29): the
    user prompt stays as small as it was and rides argv."""
    huge_system = "s" * (server.STDIN_PROMPT_LIMIT + 10)
    saved = (server.PROVIDER, server.PROFILE)
    try:
        server.PROVIDER, server.PROFILE = "opencode", server.PROFILES["opencode"]
        prompt, system = server.fold_system("small", huge_system)
        assert (prompt, system) == ("small", huge_system)
        argv, stdin_data = server.build_argv(prompt, system)
        assert stdin_data is None, "a small user prompt stays on argv"
        assert all(huge_system not in element for element in argv), "system on argv"
        assert argv[-1] == "small", argv
        cfg = server.opencode_config(prompt=system)
        assert cfg["agent"]["switchyard"]["prompt"] == huge_system
        print(f"  {len(huge_system)}-char system prompt -> agent prompt, argv untouched")
    finally:
        server.PROVIDER, server.PROFILE = saved


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
    print("  E2BIG -> 413, other spawn OSError -> 502 (no 500 leak)")


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
        _restore_env_and_reload(old)
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


def test_patch_codex_catalog_strips_code_mode_and_builtin_tools():
    """The catalog fields that put a codex model into code mode, multi-agent
    mode, or hand it the apply_patch / web-search tools are removed; every
    other field (context window, reasoning levels...) is left alone."""
    catalog = {"models": [
        {"slug": "a", "tool_mode": "code_mode_only", "multi_agent_version": "v2",
         "apply_patch_tool_type": "freeform", "web_search_tool_type": "text",
         "supports_search_tool": True, "experimental_supported_tools": ["clock"],
         "context_window": 272000},
        {"slug": "b", "context_window": 1}]}
    out = server.patch_codex_catalog(catalog)
    for model in out["models"]:
        for key in server.CODEX_CATALOG_DROP:
            assert key not in model, (key, model)
        assert model["supports_search_tool"] is False, model
        assert model["experimental_supported_tools"] == [], model
    assert out["models"][0]["context_window"] == 272000, out
    print("  patch_codex_catalog: code mode / agents / patch / search removed")


def test_codex_catalog_is_generated_from_the_cli_and_cached():
    """codex_catalog_path runs `<codex> debug models` (tests: the fake in
    tests/fixtures/fake_codex.py), writes the patched catalog, and reuses it
    for the rest of the process."""
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="clib-codexcat-"))
    saved = (server.CODEX_CATALOG_PATH, server._codex_catalog_ready)
    try:
        server.CODEX_CATALOG_PATH = tmp / "catalog.json"
        server._codex_catalog_ready = False
        path = server.codex_catalog_path()
        data = json.loads(path.read_text())
        slugs = [m["slug"] for m in data["models"]]
        assert slugs == ["gpt-5.6-terra", "gpt-5.5"], slugs
        assert "tool_mode" not in data["models"][0], data
        mtime = path.stat().st_mtime_ns
        assert server.codex_catalog_path() == path
        assert path.stat().st_mtime_ns == mtime, "regenerated instead of cached"
        print("  codex catalog generated once from `codex debug models`, then cached")
    finally:
        server.CODEX_CATALOG_PATH, server._codex_catalog_ready = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_concurrent_first_codex_requests_share_one_catalog():
    """Review of #270: two first requests racing through catalog generation
    must both get the catalog; the loser of a shared temp-file rename used
    to 502 a perfectly valid request."""
    import shutil
    import threading
    tmp = Path(tempfile.mkdtemp(prefix="clib-codexcat-race-"))
    saved = (server.CODEX_CATALOG_PATH, server._codex_catalog_ready)
    results, errors = [], []

    def first_request():
        try:
            results.append(server.codex_catalog_path())
        except Exception as exc:    # noqa: BLE001 -- any failure is the bug
            errors.append(exc)

    try:
        server.CODEX_CATALOG_PATH = tmp / "catalog.json"
        server._codex_catalog_ready = False
        threads = [threading.Thread(target=first_request) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert results == [tmp / "catalog.json"] * 8, results
        assert [p.name for p in tmp.iterdir()] == ["catalog.json"], list(tmp.iterdir())
        print("  8 concurrent first requests -> one catalog, no errors, no temp leftovers")
    finally:
        server.CODEX_CATALOG_PATH, server._codex_catalog_ready = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_codex_catalog_failure_refuses_with_502():
    """Without the catalog, codex runs in code mode with its own shell. The
    request must be refused (502, so the router cools the plan), never
    served on an unlocked CLI."""
    import shutil
    from fastapi import HTTPException
    tmp = Path(tempfile.mkdtemp(prefix="clib-codexcat-bad-"))
    saved = (server.CODEX_CATALOG_PATH, server._codex_catalog_ready,
             server.PROFILES["codex"]["cli"])
    try:
        server.CODEX_CATALOG_PATH = tmp / "catalog.json"
        server._codex_catalog_ready = False
        server.PROFILES["codex"]["cli"] = str(tmp / "no-such-codex")
        try:
            server.codex_lockdown_args()
            raise AssertionError("expected HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 502, exc.detail
            assert "lockdown unavailable" in str(exc.detail), exc.detail
        assert not server.CODEX_CATALOG_PATH.exists()
        print("  codex catalog failure -> 502, no argv built")
    finally:
        (server.CODEX_CATALOG_PATH, server._codex_catalog_ready,
         server.PROFILES["codex"]["cli"]) = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_codex_lockdown_disables_the_live_web_search_tool():
    """On a ChatGPT login codex's model has a working server-side web search
    (`web.run`) that neither the feature switches nor the patched catalog
    remove; live-verified, `web_search="disabled"` does. A fake model never
    sees it, so this pins the override itself."""
    assert 'web_search="disabled"' in server.CODEX_CONFIG_OVERRIDES
    saved = (server.PROVIDER, server.PROFILE, server.CLI)
    try:
        server.PROVIDER = "codex"
        server.PROFILE = server.PROFILES["codex"]
        server.CLI = server.PROFILE["cli"]
        argv, _ = server.build_argv("hi", None, None)
        assert argv[argv.index('web_search="disabled"') - 1] == "-c", argv
        print('  codex argv carries -c web_search="disabled"')
    finally:
        server.PROVIDER, server.PROFILE, server.CLI = saved


def test_wants_web_search_reads_every_request_shape():
    """LiteLLM turns Claude Code's server-side web_search tool into
    `web_search_options`; OpenAI-shaped callers may send it directly, or a
    web_search* tool type."""
    assert server.wants_web_search({"web_search_options": {}})
    for kind in ("web_search", "web_search_preview", "web_search_20250305",
                 "web_fetch_20250910"):
        assert server.wants_web_search({"tools": [{"type": kind}]}), kind
    assert not server.wants_web_search({"tools": [{"type": "function",
                                                   "function": {"name": "web_search"}}]})
    assert not server.wants_web_search({"messages": []})
    assert not server.wants_web_search({"tools": [{"type": "code_execution_20250825"}]})
    print("  web requested via web_search_options or a web_search*/web_fetch* tool type")


def test_web_search_request_enables_each_clis_own_search_only_then():
    """Claude: WebSearch joins the allowlist (and gets turns to search then
    answer). Codex: web_search flips from disabled to live. Without the
    request neither changes."""
    saved = (server.PROVIDER, server.PROFILE, server.CLI, server.BARE)
    try:
        server.BARE = True
        server.PROVIDER, server.PROFILE = "claude", server.PROFILES["claude"]
        server.CLI = server.PROFILE["cli"]
        plain, _ = server.build_argv("hi", None, None)
        web, _ = server.build_argv("hi", None, None, web=True)
        assert plain[plain.index("--tools") + 1] == "", plain
        assert web[web.index("--tools") + 1] == "WebSearch,WebFetch", web
        assert web[web.index("--max-turns") + 1] == "4", web
        assert web[web.index("--allowed-tools") + 1] == "WebSearch,WebFetch", web
        again = list(web)
        server.claude_web_args(again)
        assert again[again.index("--tools") + 1] == "WebSearch,WebFetch", "duplicated"
        assert "--allowed-tools" not in plain, plain
        server.PROVIDER, server.PROFILE = "codex", server.PROFILES["codex"]
        server.CLI = server.PROFILE["cli"]
        plain, _ = server.build_argv("hi", None, None)
        web, _ = server.build_argv("hi", None, None, web=True)
        assert 'web_search="disabled"' in plain and 'web_search="live"' not in plain
        assert 'web_search="live"' in web and 'web_search="disabled"' not in web
        print("  claude: --tools WebSearch,WebFetch; codex: web_search live -- only when asked")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI, server.BARE = saved


def test_opencode_text_path_web_search_config_and_env():
    """OpenCode's websearch needs the tool allowed (by name: `"*": "deny"`
    drops every built-in) and OPENCODE_ENABLE_EXA=true in its environment."""
    import asyncio
    fake_cli = tempfile.NamedTemporaryFile("w", suffix=".py", prefix="clib-ocweb-", delete=False)
    fake_cli.write(
        "import json, os\n"
        "cfg = json.load(open('opencode.json'))['agent']['switchyard']\n"
        "out = {'perm': cfg['permission'], 'tools': cfg['tools'],"
        " 'exa': os.environ.get('OPENCODE_ENABLE_EXA')}\n"
        "print(json.dumps({'type': 'text', 'part': {'type': 'text', 'text': json.dumps(out)}}))\n"
        "print(json.dumps({'type': 'step_finish', 'part': {'type': 'step-finish', "
        "'tokens': {'input': 1, 'output': 1, 'reasoning': 0}}}))\n")
    fake_cli.close()
    real = (server.CLI, server.BARE, server.PROFILE, server.PROVIDER)
    try:
        server.CLI, server.BARE, server.PROVIDER = sys.executable, False, "opencode"
        server.PROFILE = dict(server.PROFILES["opencode"], args=[fake_cli.name])
        web = json.loads(asyncio.run(server.run_cli("q", None, "m", web=True))["result"])
        plain = json.loads(asyncio.run(server.run_cli("q", None, "m"))["result"])
        assert web["exa"] == "true" and plain["exa"] is None, (web["exa"], plain["exa"])
        assert web["tools"]["websearch"] and web["tools"]["webfetch"], web["tools"]
        assert web["perm"]["websearch"] == "allow" and "*" not in web["perm"], web["perm"]
        assert web["perm"]["bash"] == "deny", web["perm"]
        assert plain["perm"] == {"*": "deny"} and not any(plain["tools"].values()), plain
        print("  opencode web request: websearch/webfetch allowed by name + Exa enabled")
    finally:
        server.CLI, server.BARE, server.PROFILE, server.PROVIDER = real
        os.unlink(fake_cli.name)


def test_web_only_tools_are_not_refused_on_the_text_path():
    """A request whose only `tools` entry is a server-side web-search tool is
    a web-search request, not a caller tool the CLI cannot run: no 400."""
    import asyncio
    seen = {}

    async def fake_invoke(prompt, system, model, image_paths=None, fmt=None, *,
                          web=False, effort=None, thinking=None):
        seen["web"] = web
        return {"result": "found it", "usage": {}}

    real = server.invoke
    server.invoke = fake_invoke
    try:
        messages = [{"role": "user", "content": "search x"}]
        for shape in ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
                      {"web_search_options": {}}):     # what LiteLLM makes of it
            seen.clear()
            asyncio.run(server._handle_chat({"model": "m", "messages": messages, **shape}))
            assert seen == {"web": True}, (shape, seen)
        print("  web tool / web_search_options -> text path with the CLI's search on, no 400")
    finally:
        server.invoke = real


def test_request_effort_reads_every_carrier():
    """OpenAI chat, Responses, Anthropic, and the gateway's carrier."""
    assert server.request_effort({"reasoning_effort": "High"}) == "high"
    assert server.request_effort({"reasoning": {"effort": "low"}}) == "low"
    assert server.request_effort({"output_config": {"effort": "max"}}) == "max"
    assert server.request_effort({"switchyard": {"reasoning_effort": "xhigh"}}) == "xhigh"
    assert server.request_effort({}) is None
    print("  effort read from reasoning_effort / reasoning / output_config / switchyard")


def test_effort_maps_to_each_clis_own_switch():
    """claude --effort, codex model_reasoning_effort, opencode --variant; an
    unknown level adds nothing rather than breaking the request."""
    saved = server.PROVIDER
    try:
        server.PROVIDER = "claude"
        assert server.effort_args("high") == ["--effort", "high"]
        assert server.effort_args("minimal") == ["--effort", "low"]
        assert server.effort_args("ultra") == ["--effort", "max"]
        assert server.effort_args("bogus") == [] and server.effort_args(None) == []
        server.PROVIDER = "codex"
        assert server.effort_args("xhigh") == ["-c", 'model_reasoning_effort="xhigh"']
        assert server.effort_args("none") == ["-c", 'model_reasoning_effort="low"']
        server.PROVIDER = "opencode"
        assert server.effort_args("high") == ["--variant", "high"]
        assert server.effort_args("xhigh") == ["--variant", "max"]
        print("  effort -> claude --effort / codex model_reasoning_effort / opencode --variant")
    finally:
        server.PROVIDER = saved


def test_text_path_argv_carries_the_effort():
    saved = (server.PROVIDER, server.PROFILE, server.CLI)
    try:
        server.PROVIDER, server.PROFILE = "claude", server.PROFILES["claude"]
        server.CLI = server.PROFILE["cli"]
        argv, _ = server.build_argv("hi", None, None, effort="high")
        assert argv[argv.index("--effort") + 1] == "high", argv
        argv, _ = server.build_argv("hi", None, None)
        assert "--effort" not in argv, argv
    finally:
        server.PROVIDER, server.PROFILE, server.CLI = saved
    print("  text-path argv carries --effort only when the caller asked")


def test_codex_text_path_argv_carries_the_lockdown():
    """The text path's codex has the same shell, patch tool and code-mode
    host as the tool path's -- and before this, nothing but the missing
    `tools` field stood between them and the model."""
    saved = (server.PROVIDER, server.PROFILE, server.CLI)
    try:
        server.PROVIDER = "codex"
        server.PROFILE = server.PROFILES["codex"]
        server.CLI = server.PROFILE["cli"]
        argv, _ = server.build_argv("hi", None, None)
        disabled = [argv[i + 1] for i, el in enumerate(argv) if el == "--disable"]
        assert list(server.CODEX_DISABLED_FEATURES) == disabled, disabled
        for override in server.CODEX_CONFIG_OVERRIDES:
            assert argv[argv.index(override) - 1] == "-c", (override, argv)
        assert any(el.startswith("model_catalog_json=") for el in argv), argv
        print("  codex text-path argv carries the full lockdown")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI = saved


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
        print("  codex argv carries -i for the staged image")
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
        print("  opencode argv carries -f for the staged image")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI = saved_provider, saved_profile, saved_cli
        shutil.rmtree(img.parent, ignore_errors=True)


def test_claude_media_ride_inline_on_stdin_not_via_read():
    """Claude Code has no image flag, and staging the files for a native Read
    put relay paths (/tmp/swimg-*/img/01.png) in front of the model, which
    handed them to the CALLER's Read tool (issue #264). Media now ride inline:
    `--input-format stream-json` + one stdin user message carrying the prompt
    and the base64 blocks. No Read, no --add-dir, no prompt in argv."""
    import shutil
    saved = (server.PROVIDER, server.PROFILE, server.CLI, server.BARE)
    img_dir = Path(tempfile.mkdtemp(prefix="cli-imgargv3-"))
    try:
        server.PROVIDER = "claude"
        server.PROFILE = server.PROFILES["claude"]
        server.CLI = server.PROFILE["cli"]
        server.BARE = True
        img = img_dir / "img" / "01.png"
        img.parent.mkdir(parents=True)
        img.write_bytes(PNG_MAGENTA)
        pdf = img_dir / "img" / "02.pdf"
        pdf.write_bytes(b"%PDF-1.4 marker")
        argv, stdin_data = server.build_argv("look at this", None, None,
                                             image_paths=[img, pdf])
        assert "--add-dir" not in argv and "--allowed-tools" not in argv, argv
        assert argv[argv.index("--tools") + 1] == "", argv
        assert argv[argv.index("--max-turns") + 1] == "1", argv
        assert argv[argv.index("--input-format") + 1] == "stream-json", argv
        assert argv[argv.index("--output-format") + 1] == "stream-json", argv
        assert argv.count("--output-format") == 1, argv
        assert "--verbose" in argv, argv
        assert "look at this" not in argv and not any(str(img_dir) in a for a in argv), argv
        msg = json.loads(stdin_data)
        assert msg["type"] == "user", msg
        content = msg["message"]["content"]
        assert content[0] == {"type": "text", "text": "look at this"}, content[0]
        assert content[1]["type"] == "image", content[1]
        assert content[1]["source"]["media_type"] == "image/png", content[1]
        assert base64.b64decode(content[1]["source"]["data"]) == PNG_MAGENTA
        assert content[2]["type"] == "document", content[2]
        assert content[2]["source"]["media_type"] == "application/pdf", content[2]
        assert base64.b64decode(content[2]["source"]["data"]) == b"%PDF-1.4 marker"
        assert str(img_dir) not in stdin_data, "relay path leaked to the model"
        print("  claude media: stream-json stdin with image + document, no Read")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI, server.BARE = saved
        shutil.rmtree(img_dir, ignore_errors=True)


def test_claude_media_with_web_and_effort_compose():
    """Web and media are independent on the claude text path: web adds
    WebSearch/WebFetch and raises --max-turns (claude_web_args), media switch
    the input to stream-json, and effort still lands its flag -- the media
    return must not skip any of them."""
    import shutil
    saved = (server.PROVIDER, server.PROFILE, server.CLI, server.BARE)
    img_dir = Path(tempfile.mkdtemp(prefix="cli-imgweb-"))
    try:
        server.PROVIDER = "claude"
        server.PROFILE = server.PROFILES["claude"]
        server.CLI = server.PROFILE["cli"]
        server.BARE = True
        img = img_dir / "01.png"
        img.write_bytes(PNG_MAGENTA)
        argv, stdin_data = server.build_argv("look and search", None, None,
                                             image_paths=[img], web=True, effort="high")
        assert argv[argv.index("--tools") + 1] == "WebSearch,WebFetch", argv
        assert argv[argv.index("--allowed-tools") + 1] == "WebSearch,WebFetch", argv
        assert int(argv[argv.index("--max-turns") + 1]) >= 4, argv
        assert argv[argv.index("--input-format") + 1] == "stream-json", argv
        assert argv[argv.index("--output-format") + 1] == "stream-json", argv
        assert argv[argv.index("--effort") + 1] == "high", argv
        assert "Read" not in argv[argv.index("--tools") + 1], argv
        assert json.loads(stdin_data)["message"]["content"][1]["type"] == "image"
        print("  claude web + media + effort compose on one argv")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI, server.BARE = saved
        shutil.rmtree(img_dir, ignore_errors=True)


def test_claude_media_markers_name_no_relay_path():
    """The in-prompt placeholder for a staged block must not carry the relay
    path on claude (the bytes are inline); codex/opencode keep the path since
    it is the file they attach."""
    import shutil
    saved = server.PROVIDER
    msgs = lambda: [{"role": "user", "content": [  # noqa: E731
        {"type": "text", "text": "what colour?"},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(PNG_MAGENTA).decode()}},
        {"type": "file", "file": {
            "file_data": "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.4").decode()}}]}]
    dirs = []
    try:
        server.PROVIDER = "claude"
        m = msgs()
        d = Path(tempfile.mkdtemp(prefix="cli-marker-"))
        dirs.append(d)
        paths = server.stage_images(m, d)
        texts = [b["text"] for b in m[0]["content"]]
        assert texts[1] == "[image 1: attached]", texts
        assert texts[2] == "[document 2: attached]", texts
        assert [p.suffix for p in paths] == [".png", ".pdf"], paths
        assert "attachment(s) follow" in server.image_note(paths)
        assert str(d) not in server.image_note(paths)
        server.PROVIDER = "codex"
        m = msgs()[:1]
        m[0]["content"] = m[0]["content"][:2]
        d = Path(tempfile.mkdtemp(prefix="cli-marker-"))
        dirs.append(d)
        paths = server.stage_images(m, d)
        assert m[0]["content"][1]["text"] == f"[image 1: {paths[0]}]", m
        print("  markers: claude 'attached', codex keeps its -i path")
    finally:
        server.PROVIDER = saved
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)


def _pdf_msg(data: bytes = b"%PDF-1.4") -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "text", "text": "summarise"},
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                        "data": base64.b64encode(data).decode()}},
        {"type": "text", "text": "and this:"},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(PNG_MAGENTA).decode()}}]}]


def _run_pdf_chat(provider: str, **extra) -> dict:
    """Drive _handle_chat with a PDF + an image on `provider`; the fake
    invoke records the prompt and the argv the CLI would get (built while
    the staged files still exist)."""
    import asyncio
    seen: dict = {}

    async def fake_invoke(prompt, system, model, image_paths=None, fmt=None, *,
                          web=False, effort=None, thinking=None):
        paths = list(image_paths or [])
        argv, stdin_data = server.build_argv(prompt, system, model, image_paths=paths)
        seen.update(prompt=prompt, paths=paths, argv=argv, stdin=stdin_data, fmt=fmt,
                    bytes=[p.read_bytes() for p in paths])
        return {"result": '{"a": "x"}', "usage": {}}     # valid for every format

    real = (server.invoke, server.PROVIDER, server.PROFILE, server.CLI)
    try:
        server.invoke = fake_invoke
        server.PROVIDER = provider
        server.PROFILE = server.PROFILES[provider]
        server.CLI = server.PROFILE["cli"]
        asyncio.run(server._handle_chat({"model": "m", "messages": _pdf_msg(), **extra}))
    finally:
        server.invoke, server.PROVIDER, server.PROFILE, server.CLI = real
    return seen


def test_pdf_prompt_keeps_the_response_format_on_codex_and_opencode():
    """#283 (response_format) and #286 (PDF -> text + pages) meet in
    _complete: the expanded PDF's text, the image note AND the schema
    instruction all reach the prompt; a codex strict schema still goes to
    the CLI natively (fmt passed to invoke, no prompt instruction), an
    opencode strict schema as the full instruction after the image note."""
    from _modules import FakePoppler
    for provider in ("codex", "opencode"):
        with FakePoppler():
            seen = _run_pdf_chat(provider, response_format={"type": "json_object"})
        prompt = seen["prompt"]
        assert "EXTRACTED TEXT" in prompt, prompt
        assert prompt.index("[3 image(s) attached to this prompt:") \
            < prompt.index("Respond with only a single JSON object"), prompt
        assert seen["fmt"] is None, seen["fmt"]
        assert len(seen["paths"]) == 3, seen["paths"]
    strict = {"type": "json_schema", "json_schema": {
        "name": "s", "strict": True, "schema": {
            "type": "object", "properties": {"a": {"type": "string"}},
            "required": ["a"], "additionalProperties": False}}}
    with FakePoppler():
        seen = _run_pdf_chat("codex", response_format=strict)
    assert seen["fmt"] and seen["fmt"]["strict"], seen["fmt"]
    assert "EXTRACTED TEXT" in seen["prompt"] and "JSON Schema" not in seen["prompt"], seen
    # opencode never enforces a schema itself: the strict schema rides the
    # prompt as an instruction (with the schema dump), after the image note.
    with FakePoppler():
        seen = _run_pdf_chat("opencode", response_format=strict)
    prompt = seen["prompt"]
    assert seen["fmt"] is None, seen["fmt"]
    assert "EXTRACTED TEXT" in prompt, prompt
    instruction = ("Respond with only a JSON value that validates against this JSON "
                   "Schema: no prose before or after it, no code fences.\n"
                   + json.dumps(strict["json_schema"]["schema"]))
    assert prompt.index("[3 image(s) attached to this prompt:") \
        < prompt.index(instruction), prompt
    print("  PDF prompt + response_format: text, page note and schema all survive")


def test_pdf_prompt_on_codex_and_opencode_is_text_plus_page_images():
    """codex -i is images-only and OpenCode's attachment docs reject PDF, so
    a PDF in the prompt goes as what both DO take: its extracted text at its
    place in the prompt, one PNG per page on -i / -f. No 400, nothing
    dropped, and the image note counts what was actually attached."""
    from _modules import FakePoppler
    for provider, flag in (("codex", "-i"), ("opencode", "-f")):
        with FakePoppler():
            seen = _run_pdf_chat(provider)
        paths = seen["paths"]
        assert [p.name for p in paths] == ["page-1.png", "page-2.png", "02.png"], paths
        assert seen["bytes"][:2] == [b"PNG1", b"PNG2"], seen["bytes"]
        assert seen["bytes"][2] == PNG_MAGENTA
        attached = [seen["argv"][i + 1] for i, a in enumerate(seen["argv"]) if a == flag]
        assert attached == [str(p) for p in paths], (provider, seen["argv"])
        assert not any(a.endswith(".pdf") for a in seen["argv"]), seen["argv"]
        prompt = seen["prompt"]
        assert "[document 1: PDF, 2 page(s) attached as images:" in prompt, prompt
        assert "extracted text follows]\nEXTRACTED TEXT" in prompt, prompt
        # the text sits where the PDF was: between the two caller texts
        assert prompt.index("summarise") < prompt.index("EXTRACTED TEXT") \
            < prompt.index("and this:") < prompt.index("[image 2:"), prompt
        assert "[3 image(s) attached to this prompt:" in prompt, prompt
    print("  PDF prompt: codex -i / opencode -f get page PNGs, text inline, no 400")


def test_pdf_prompt_cut_at_the_page_limit_says_so():
    import shutil
    from _modules import FakePoppler
    msgs = _pdf_msg()
    saved = (server.PROVIDER, server.PDF_PAGE_LIMIT)
    try:
        server.PROVIDER, server.PDF_PAGE_LIMIT = "codex", 2
        with FakePoppler():
            paths, img_dir = server.stage_or_fail(msgs)
        shutil.rmtree(img_dir, ignore_errors=True)
    finally:
        server.PROVIDER, server.PDF_PAGE_LIMIT = saved
    assert len(paths) == 3, paths
    assert "2 page(s) (first 2 only) attached as images" in msgs[0]["content"][1]["text"], msgs


def test_pdf_prompt_partial_render_still_delivers():
    """Text but no pages (or pages but no text) is still a delivery; the
    marker says which half is missing. Text alone attaches nothing."""
    from _modules import FakePoppler
    saved = server.PROVIDER
    try:
        server.PROVIDER = "opencode"
        msgs = _pdf_msg()[:1]
        msgs[0]["content"] = msgs[0]["content"][:2]
        with FakePoppler(pages_fail=True):
            paths, img_dir = server.stage_or_fail(msgs)
        assert (paths, img_dir) == ([], None), (paths, img_dir)
        marker = msgs[0]["content"][1]["text"]
        assert "no page images could be rendered" in marker and "EXTRACTED TEXT" in marker, marker
        msgs = _pdf_msg()
        with FakePoppler(text_fails=True):
            paths, img_dir = server.stage_or_fail(msgs)
        import shutil
        shutil.rmtree(img_dir, ignore_errors=True)
        assert len(paths) == 3, paths
        assert "no text could be extracted" in msgs[0]["content"][1]["text"], msgs
    finally:
        server.PROVIDER = saved


def test_unrenderable_pdf_prompt_is_a_400_on_codex_and_opencode():
    """Poppler producing neither text nor pages: refused loudly (400
    images_unsupported), never a PDF dropped from the prompt unseen."""
    import asyncio
    from fastapi import HTTPException
    from _modules import FakePoppler

    async def fake_invoke(*a, **k):
        raise AssertionError("the CLI ran without the PDF")

    real = (server.invoke, server.PROVIDER, server.PROFILE, server.CLI)
    try:
        server.invoke = fake_invoke
        for provider in ("codex", "opencode"):
            server.PROVIDER = provider
            server.PROFILE = server.PROFILES[provider]
            server.CLI = server.PROFILE["cli"]
            with FakePoppler(text_fails=True, pages_fail=True):
                try:
                    asyncio.run(server._handle_chat({"model": "m", "messages": _pdf_msg()}))
                except HTTPException as exc:
                    assert exc.status_code == 400, exc.status_code
                    assert exc.detail["error"]["type"] == "images_unsupported", exc.detail
                    assert "PDF" in exc.detail["error"]["message"], exc.detail
                else:
                    raise AssertionError(f"{provider} accepted an unrenderable PDF")
        print("  unrenderable PDF prompt -> 400 images_unsupported on codex and opencode")
    finally:
        server.invoke, server.PROVIDER, server.PROFILE, server.CLI = real


def test_pdf_prompt_on_claude_stays_an_inline_document():
    """Claude takes the PDF itself (stream-json document block): no render,
    even with poppler failing; the marker names no relay path."""
    from _modules import FakePoppler
    saved = server.PROVIDER
    msgs = _pdf_msg()
    try:
        server.PROVIDER = "claude"
        with FakePoppler(text_fails=True, pages_fail=True):
            paths, img_dir = server.stage_or_fail(msgs)
        try:
            assert [p.suffix for p in paths] == [".pdf", ".png"], paths
            assert paths[0].read_bytes() == b"%PDF-1.4"
            assert msgs[0]["content"][1] == {"type": "text", "text": "[document 1: attached]"}
        finally:
            import shutil
            shutil.rmtree(img_dir, ignore_errors=True)
        assert server.has_image_blocks(_pdf_msg())
        assert not server.has_image_blocks([{"role": "user", "content": [
            {"type": "document", "source": {"type": "text", "media_type": "text/plain",
                                            "data": "hi"}}]}])
        print("  PDF prompt on claude: staged as the PDF, sent inline")
    finally:
        server.PROVIDER = saved


def test_claude_json_parser_reads_stream_json_result():
    """Media requests switch claude to --output-format stream-json; the
    parser takes the final `result` event, which has the json output's shape."""
    stream = "\n".join(json.dumps(e) for e in [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "#ff00ff"}]}},
        {"type": "result", "subtype": "success", "result": "#ff00ff", "is_error": False,
         "usage": {"input_tokens": 3, "output_tokens": 2}}]) + "\n"
    out = server.parse_output(stream, "claude_json")
    assert out["type"] == "result" and out["result"] == "#ff00ff", out
    single = json.dumps({"type": "result", "result": "plain"})
    assert server.parse_output(single, "claude_json")["result"] == "plain"
    try:
        server.parse_output("not json at all\n", "claude_json")
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("garbage parsed")
    print("  claude_json parser: stream-json result event accepted")


def test_claude_json_buffered_collects_reasoning_from_content_blocks():
    """`claude --output-format json` (a single JSON object, not stream-json) has
    any reasoning blocks nested in `content[]` of the result, exactly the same
    shape the stream-json assistant events carry. The parser walks that list
    so a buffered single-object form gives the same `reasoning` field the
    stream-json form collects on its own loop. Before the fix both branches of
    the conditional returned the parsed object unchanged.
    """
    single = json.dumps({"type": "result", "result": "the answer",
                         "content": [
                             {"type": "thinking", "thinking": "first "},
                             {"type": "text", "text": "the answer"},
                             {"type": "redacted_thinking", "thinking": "redacted"},
                         ],
                         "is_error": False,
                         "usage": {"input_tokens": 4, "output_tokens": 5}})
    out = server.parse_output(single, "claude_json")
    assert out["result"] == "the answer", out
    assert out["reasoning"] == "first redacted", out
    assert out["type"] == "result", out

    # No thinking block -> no `reasoning` key (same absence-on-reasoning contract
    # the events_json path enforces).
    plain_single = json.dumps({"type": "result", "result": "hi",
                               "is_error": False,
                               "usage": {"input_tokens": 1, "output_tokens": 1}})
    out2 = server.parse_output(plain_single, "claude_json")
    assert "reasoning" not in out2, out2
    assert out2["result"] == "hi", out2
    print("  claude_json buffered: thinking blocks collected onto `reasoning`, "
          "absent when the model emitted none")


def test_claude_text_path_allows_no_builtin_tools():
    """The text path's --max-turns 1 turn ends in error_max_turns -> a paid
    502 whenever the model calls ANY tool (issue #195: Agent, ToolSearch,
    Skill; issue #256: AskUserQuestion). A denylist has to name every
    built-in and misses the next one a CLI release adds, so the bare args
    are an allowlist of zero (`--tools ""`) plus the shared lockdown:
    no account MCP connectors, no login-dir settings/CLAUDE.md/hooks, and
    no permission prompt anyone would have to answer."""
    saved = (server.PROVIDER, server.PROFILE, server.CLI, server.BARE)
    try:
        server.PROVIDER = "claude"
        server.PROFILE = server.PROFILES["claude"]
        server.CLI = server.PROFILE["cli"]
        server.BARE = True
        argv, _ = server.build_argv("hi", None, None)
        assert argv[argv.index("--tools") + 1] == "", argv
        assert "--disallowed-tools" not in argv, argv
        assert argv[argv.index("--max-turns") + 1] == "1", argv
        assert "--strict-mcp-config" in argv, argv
        assert argv[argv.index("--setting-sources") + 1] == "", argv
        assert argv[argv.index("--permission-prompts") + 1] == "none", argv
        assert tuple(server.CLAUDE_LOCKDOWN) == (
            "--strict-mcp-config", "--setting-sources", "",
            "--permission-prompts", "none"), server.CLAUDE_LOCKDOWN
        print("  claude text path: --tools '' + strict MCP + no settings + no prompts")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI, server.BARE = saved


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
            # 400 with tools_unsupported detail is the regression this test
            # is named for -- must not 400 on tools when no tools are
            # present. The gate/spawn path also legitimately raises
            # 429/502, and those stay tolerated. Any other HTTPException
            # from _handle_chat is a regression we want to surface (e.g. a
            # new error reason that doesn't fit the two contracts above).
            # Non-HTTPException errors are deliberately NOT caught here, so
            # a KeyError/TypeError regression fails the test loudly instead
            # of being silently swallowed.
            if (exc.status_code == 400 and isinstance(exc.detail, dict)
                    and exc.detail.get("error", {}).get("type")
                    == "tools_unsupported"):
                raise AssertionError(
                    f"text path must not 400 on tools when no tools present: "
                    f"{exc.detail}") from exc
            assert exc.status_code in (429, 502), (
                f"unexpected HTTPException from _handle_chat: "
                f"{exc.status_code} {exc.detail}")
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
        _restore_env_and_reload(old)


# ----------------------------------------- issue #70: enforce max_tokens post-hoc ---
def test_max_tokens_truncated_with_length_reason():
    """The post-hoc enforcement contract: over-budget content is cut to
    ~max_tokens*4 bytes with finish_reason "length", usage is left alone
    (the caller paid for the real tokens), an estimate <= cap leaves the
    content byte-identical with finish_reason "stop", no max_tokens is a
    pass-through with "stop", and sse_from_completion frames the truncated
    answer in the order delta -> terminal -> [DONE].
    """
    import asyncio
    # (a) over-budget: content cut to ~max_tokens*4 bytes; usage untouched.
    big = "x" * 1024
    payload = {"result": big,
               "usage": {"input_tokens": 5, "output_tokens": 200, "total_tokens": 205}}
    cut, fr = server.enforce_max_tokens(payload, 50)   # 50 tokens -> 200 byte cap
    assert fr == "length", fr
    assert len(cut["result"]) <= 200, len(cut["result"])
    assert cut["result"] == "x" * 200, len(cut["result"])
    # Usage stays exactly what the CLI reported -- the ledger books real tokens.
    assert cut["usage"] == payload["usage"], cut["usage"]
    # Original payload untouched.
    assert payload["result"] == big

    # (b) multibyte: cutting on a byte boundary must not raise UnicodeDecodeError.
    # Each '\u4e00' is 3 UTF-8 bytes. 200 copies -> 600 bytes -> estimate 150.
    cjk = "\u4e00" * 200
    payload2 = {"result": cjk, "usage": {"input_tokens": 0, "output_tokens": 150}}
    cut2, fr2 = server.enforce_max_tokens(payload2, 50)   # 200 byte cap
    assert fr2 == "length", fr2
    assert len(cut2["result"].encode("utf-8")) <= 200
    # Decode was errors="ignore"; the leading 200 bytes are a whole number of
    # 3-byte chars, so no splitting occurred. That is the test the plan calls
    # for ("multibyte cuts on a byte boundary").
    assert cut2["result"] == "\u4e00" * (200 // 3), cut2["result"]

    # (c) estimate <= cap: content untouched, finish_reason "stop".
    small = "x" * 100   # 100 bytes -> 25 tokens by ceil/4
    payload3 = {"result": small, "usage": {"output_tokens": 25}}
    unchanged, fr3 = server.enforce_max_tokens(payload3, 50)
    assert fr3 == "stop", fr3
    assert unchanged["result"] == small
    # Byte-for-byte identity: this is the path the unchanged-output behaviour
    # depends on, and any caching keyed on the result.
    assert unchanged["result"] is payload3["result"] or \
        unchanged["result"] == payload3["result"]

    # (d) no max_tokens: pass-through with "stop".
    payload4 = {"result": "anything", "usage": {}}
    no_cap, fr4 = server.enforce_max_tokens(payload4, None)
    assert fr4 == "stop", fr4
    no_cap2, fr4b = server.enforce_max_tokens(payload4, 0)
    assert fr4b == "stop", fr4b
    no_cap3, fr4c = server.enforce_max_tokens(payload4, -1)
    assert fr4c == "stop", fr4c
    assert no_cap["result"] == "anything"

    # (e) to_openai carries the truncated finish_reason, and sse_from_completion
    # frames it as delta -> terminal (length) -> [DONE].
    result = server.to_openai(cut, "m", finish_reason=fr)
    assert result["choices"][0]["finish_reason"] == "length", result
    assert result["choices"][0]["message"]["content"] == cut["result"]

    async def collect():
        out = []
        async for frame in server.sse_from_completion(result, "m"):
            out.append(frame)
        return out

    frames = asyncio.run(collect())
    assert len(frames) == 3, frames
    assert frames[-1] == "data: [DONE]\n\n", frames[-1]
    first = json.loads(frames[0][6:])
    assert first["choices"][0]["delta"]["content"] == cut["result"], first
    terminal = json.loads(frames[-2][6:])
    assert terminal["choices"][0]["finish_reason"] == "length", terminal
    print(f"  over-budget -> 'length' at {len(cut['result'])} bytes; "
          "estimate<=cap and no-cap stay 'stop'; sse ordering preserved")


def test_unenforceable_max_tokens_returns_400():
    """A lane that cannot honour an output cap refuses it with the #70
    contract detail -- under every spelling of the cap (max_completion_tokens
    is what LiteLLM sends for gpt-5 names, issue #133). No shipped profile
    is such a lane any more (codex enforces since the tool lockdown), so the
    refusal is driven through a synthetic one."""
    import asyncio
    from fastapi import HTTPException
    saved = (server.PROVIDER, server.PROFILE, server.CLI)
    try:
        server.PROVIDER = "opencode"
        server.PROFILE = dict(server.PROFILES["opencode"], enforce_max_tokens=False,
                              enforce_max_tokens_reason="test-lane")
        server.CLI = "/no/such/cli-binary-for-issue70"
        for spelling in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            async def go(key=spelling):
                try:
                    await server._handle_chat({
                        "model": "m", key: 100,
                        "messages": [{"role": "user", "content": "hi"}]})
                except HTTPException as exc:
                    return exc.status_code, exc.detail
            status, detail = asyncio.run(go())
            assert status == 400, (spelling, status, detail)
            assert detail["error"]["type"] == "max_tokens_unenforceable", detail
            assert detail["error"]["reason"] == "test-lane", detail
        print("  unenforceable lane: 400 under max_tokens / max_completion_tokens / max_output_tokens")
    finally:
        server.PROVIDER, server.PROFILE, server.CLI = saved


def test_request_max_tokens_reads_every_spelling():
    assert server.request_max_tokens({"max_tokens": 5}) == 5
    assert server.request_max_tokens({"max_completion_tokens": 7}) == 7
    assert server.request_max_tokens({"max_output_tokens": 9}) == 9
    assert server.request_max_tokens({"max_tokens": 0}) is None
    assert server.request_max_tokens({}) is None
    print("  output cap read from max_tokens / max_completion_tokens / max_output_tokens")


def test_health_reports_max_tokens_mode():
    """opencode and codex profiles -> enforces_max_tokens=True, no reason key.

    Same shape both bridges report; /health's callers can branch on the
    field and (when false) inspect the reason to tell a future streaming-
    only refusal apart from the current one.
    """
    import asyncio
    saved_provider, saved_profile = server.PROVIDER, server.PROFILE
    try:
        server.PROVIDER = "opencode"
        server.PROFILE = server.PROFILES["opencode"]
        h_open = asyncio.run(server.health())
        assert h_open["enforces_max_tokens"] is True, h_open
        assert "enforces_max_tokens_reason" not in h_open, h_open

        # Codex enforces post-hoc since the tool lockdown left its text path
        # a single answer (refusing rejected every Claude Code / OpenCode
        # request, which always carry a cap).
        server.PROVIDER = "codex"
        server.PROFILE = server.PROFILES["codex"]
        h_codex = asyncio.run(server.health())
        assert h_codex["enforces_max_tokens"] is True, h_codex
        assert "enforces_max_tokens_reason" not in h_codex, h_codex
    finally:
        server.PROVIDER, server.PROFILE = saved_provider, saved_profile

    print("  opencode and codex health -> enforces_max_tokens=True (no reason)")


def test_opencode_text_path_spawn_dir_carries_the_locked_down_agent():
    """OpenCode reads its project config from the working directory. The
    text path spawns in a fresh `sy-cli-*` temp dir (issue #44), so unless
    the bridge writes the config THERE, `--agent switchyard` is not found
    and OpenCode falls back to its `build` agent: every built-in tool,
    `permission: * allow`. A fake CLI reports the opencode.json it finds in
    its own cwd; the assert is that the spawned process saw the lockdown.
    """
    import asyncio
    fake_cli = tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix="clib-ocfg-", delete=False)
    fake_cli.write(
        "import json, os\n"
        "cfg = open('opencode.json').read() if os.path.exists('opencode.json') else 'MISSING'\n"
        "print(json.dumps({'type': 'text', 'part': {'type': 'text', 'text': cfg}}))\n"
        "print(json.dumps({'type': 'step_finish', 'part': "
        "    {'type': 'step-finish', "
        "     'tokens': {'input': 1, 'output': 1, 'reasoning': 0}}}))\n")
    fake_cli.close()
    # Pin the opencode profile outright: another test module may have loaded
    # this bridge under a different PROVIDER in the same process.
    real = (server.CLI, server.BARE, server.PROFILE, server.PROVIDER)
    try:
        server.CLI = sys.executable
        server.BARE = False
        server.PROVIDER = "opencode"
        server.PROFILE = dict(server.PROFILES["opencode"], args=[fake_cli.name])
        payload = asyncio.run(server.run_cli("hi", None, "m"))
        assert payload["result"] != "MISSING", "no opencode.json in the spawn dir"
        cfg = json.loads(payload["result"])
        assert cfg == server.opencode_config(), cfg
        agent = cfg["agent"]["switchyard"]
        assert agent["permission"] == {"*": "deny"}, agent
        assert cfg["permission"] == {"*": "deny"}, cfg
        assert not any(agent["tools"].values()), agent["tools"]
        assert cfg["agent"]["title"] == {"disable": True}, cfg
        print("  text-path spawn dir carries the switchyard agent, all built-ins denied")
    finally:
        server.CLI, server.BARE, server.PROFILE, server.PROVIDER = real
        os.unlink(fake_cli.name)


def test_opencode_spawn_dir_config_write_failure_is_cleaned_up_and_502():
    """Writing the spawn dir's opencode.json can fail (full disk, a temp
    dir made unwritable). That must go through the same path as a failed
    spawn: the `sy-cli-*` dir is removed, not orphaned, and the caller
    gets a 502 so the router cools the plan -- not an unclassified 500."""
    import asyncio
    from fastapi import HTTPException
    tmp_root = tempfile.mkdtemp(prefix="clib-ocfg-fail-")
    real = (server.PROVIDER, server.PROFILE, server.opencode_config,
            tempfile.tempdir)

    def broken_config(allow=(), prompt=None):
        raise OSError(errno.ENOSPC, "No space left on device")

    try:
        server.PROVIDER = "opencode"
        server.PROFILE = server.PROFILES["opencode"]
        server.opencode_config = broken_config
        tempfile.tempdir = tmp_root
        try:
            asyncio.run(server.run_cli("hi", None, "m"))
            raise AssertionError("expected HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 502, exc.detail
            assert "could not be spawned" in str(exc.detail), exc.detail
        leftovers = [n for n in os.listdir(tmp_root) if n.startswith("sy-cli-")]
        assert leftovers == [], leftovers
        print("  opencode.json write failure -> 502, sy-cli-* dir removed")
    finally:
        (server.PROVIDER, server.PROFILE, server.opencode_config,
         tempfile.tempdir) = real
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_opencode_text_path_carries_the_system_prompt_as_the_agent_prompt():
    """Bridge-siblings with mcp_bridge.write_opencode_dir (review of #274):
    on the text path too, the caller's system prompt is the agent's
    `prompt:` -- replacing OpenCode's base prompt -- and is not folded into
    the user turn."""
    import asyncio
    fake_cli = tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix="clib-ocprompt-", delete=False)
    fake_cli.write(
        "import json, sys\n"
        "cfg = json.load(open('opencode.json'))\n"
        "out = {'prompt': cfg['agent']['switchyard'].get('prompt'), 'argv_tail': sys.argv[-1]}\n"
        "print(json.dumps({'type': 'text', 'part': {'type': 'text', 'text': json.dumps(out)}}))\n"
        "print(json.dumps({'type': 'step_finish', 'part': "
        "    {'type': 'step-finish', "
        "     'tokens': {'input': 1, 'output': 1, 'reasoning': 0}}}))\n")
    fake_cli.close()
    real = (server.CLI, server.BARE, server.PROFILE, server.PROVIDER)
    try:
        server.CLI = sys.executable
        server.BARE = False
        server.PROVIDER = "opencode"
        server.PROFILE = dict(server.PROFILES["opencode"],
                              args=[fake_cli.name, "{prompt}"])
        payload = asyncio.run(server.run_cli("the user turn", "CALLER SYSTEM", "m"))
        seen = json.loads(payload["result"])
        assert seen["prompt"] == "CALLER SYSTEM", seen
        assert seen["argv_tail"] == "the user turn", seen
        print("  opencode text path: system -> agent prompt, user turn unfolded")
    finally:
        server.CLI, server.BARE, server.PROFILE, server.PROVIDER = real
        os.unlink(fake_cli.name)


def test_opencode_harness_file_matches_opencode_config():
    """harness/opencode.json ships in the image (Dockerfile.sidecar) and is
    what a manual `opencode run` from /app picks up. It must be exactly the
    config the bridge writes, or the two drift and one of them is the
    unlocked one."""
    harness = Path(server.__file__).resolve().parent / "harness" / "opencode.json"
    assert json.loads(harness.read_text()) == server.opencode_config(), \
        "regenerate sidecars/cli_bridge/harness/opencode.json from opencode_config()"
    print("  harness/opencode.json == opencode_config()")


def test_opencode_config_denies_every_builtin_unless_allowed():
    """Hiding a tool (`tools: {x: false}`) does not stop OpenCode executing
    it when the model names it; the permission layer does. `allow` is the
    only way through, and a builtin named in it is also un-hidden."""
    only_mcp = server.opencode_config(allow=("switchyard_*",))["agent"]["switchyard"]
    assert only_mcp["permission"] == {"*": "deny", "switchyard_*": "allow"}, only_mcp
    # With a BUILT-IN allowed, `"*": "deny"` would drop every built-in from
    # the tool list on the pinned OpenCode (the allowed one included), so the
    # other built-ins are denied by name instead.
    cfg = server.opencode_config(allow=("switchyard_*", "read"))
    agent = cfg["agent"]["switchyard"]
    assert "*" not in agent["permission"], agent["permission"]
    assert agent["permission"] == {
        **{t: "deny" for t in server.OPENCODE_BUILTIN_TOOLS if t != "read"},
        "switchyard_*": "allow", "read": "allow"}, agent["permission"]
    assert cfg["permission"] == agent["permission"], cfg
    assert agent["tools"]["read"] is True, agent["tools"]
    others = {k: v for k, v in agent["tools"].items() if k != "read"}
    assert others and not any(others.values()), others
    for tool in ("bash", "skill", "task", "webfetch"):
        assert tool in agent["tools"], tool
    print("  opencode_config: * deny, allow-list only, builtins hidden")


def test_spawn_sites_pass_only_the_allowlisted_env_to_the_cli():
    """Every `create_subprocess_exec` here hands the CLI an `env=` whose keys
    are a subset of SUBPROCESS_ENV_KEYS. A planted secret (LITELLM_MASTER_KEY,
    the kind compose's env_file would have injected into the container) must
    not appear. Otherwise a single prompt injection exfiltrates every
    provider key, the gateway master key and the OAuth grant (issue #116).

    Covers BOTH cli_bridge spawn sites: _run_cli_attempt (text path) and
    usage_report (the /usage slash-command path, which is otherwise an
    attractive injection target because its argv is fixed and predictable).
    mcp_bridge's spawn site is covered in test_mcp_bridge.py.
    """
    import asyncio
    from fastapi import HTTPException

    real_exec = asyncio.create_subprocess_exec
    captured_envs: list = []

    async def fake_exec(*args, **kwargs):
        # Capture the env keyword arg verbatim. A real spawn would also need
        # stdin/stdout/stderr PIPE objects, communicate(), etc.; we substitute
        # a sentinel Process-like object so the post-spawn code under test
        # doesn't choke. For usage_report the spawn returns early, so this
        # only needs to look real enough for the first await past exec.
        env = kwargs.get("env")
        captured_envs.append(env)
        class _Stub:
            returncode = 0

            async def communicate(self, input=None):
                # Text path: an empty but non-zero stdout trips the
                # "exit 0 but nothing" branch and lets _run_cli_attempt
                # raise its expected HTTPException (502). The test catches
                # that exception -- only the captured env matters.
                # usage_report path: never reads stdout.
                return (b"", b"")

            async def wait(self):
                return 0

            def kill(self):
                return None
        return _Stub()

    # Plant the secret AFTER importing server so the constant SUBPROCESS_ENV_KEYS
    # is already captured; this mirrors the real failure shape where the
    # operator's .env is mounted into a running sidecar mid-life.
    planted = "LITELLM_MASTER_KEY"
    previous = os.environ.get(planted)
    os.environ[planted] = "sk-test-planted-secret"
    # The default PROVIDER for these tests is opencode; usage_report only
    # spawns for PROVIDER=claude (the others raise 501). Snapshot the
    # current PROVIDER and toggle to claude for the second half so the
    # /usage spawn site actually fires.
    saved_provider = server.PROVIDER
    try:
        asyncio.create_subprocess_exec = fake_exec

        # 1) The text path. _run_cli_attempt is what /v1/chat/completions
        #    reaches; its spawn is the one that runs the user's prompt.
        #    The fake spawn returns exit 0 / empty stdout, which is the
        #    "exit 0 but nothing" branch -- _run_cli raises 502. We do
        #    not care about the raised exception; we only need the spawn
        #    to have happened so the env= capture is meaningful.
        try:
            asyncio.run(server._run_cli("hi", None, "m", []))
        except HTTPException:
            pass

        # 2) The /usage path. Different call site in the file; same env=
        #    contract must hold. PROVIDER=claude so the spawn actually
        #    fires (opencode/codex raise 501/503 without spawning). The
        #    fake spawn returns exit 0 -- _find_usage_report finds no
        #    transcripts in this fake workdir and raises 502. Irrelevant;
        #    only the spawn's env= matters here.
        server.PROVIDER = "claude"
        try:
            asyncio.run(server.usage_report())
        except HTTPException:
            pass
    finally:
        asyncio.create_subprocess_exec = real_exec
        server.PROVIDER = saved_provider
        if previous is None:
            os.environ.pop(planted, None)
        else:
            os.environ[planted] = previous

    # Exactly one capture per documented spawn site: the text-path
    # _run_cli_attempt and the /usage command. `_run_cli` would also spawn
    # a second time on its single no-text-with-usage retry -- that retry
    # does NOT fire here today because the empty-stdout stub trips
    # _run_cli_attempt's HTTPException(502) terminal branch, not the
    # retriable CliNoTextError branch -- but pin the count so a future
    # change that flips the empty-stdout shape into a retry (or adds
    # another spawn) is caught here instead of silently extending the
    # captured list past two.
    assert len(captured_envs) == 2, captured_envs
    for env in captured_envs:
        # Plain dict, not None (which would have meant "inherit parent") and
        # not os.environ (which would have meant copy-the-whole-bag).
        assert isinstance(env, dict), type(env)
        # Every key must be on the allowlist. No drift, no stragglers.
        bad = set(env) - set(server.SUBPROCESS_ENV_KEYS)
        assert not bad, f"unexpected env keys passed to CLI: {sorted(bad)}"
        # The planted secret must be absent: the allowlist is the only
        # source of truth, and LITELLM_MASTER_KEY is not on it.
        assert planted not in env, \
            f"{planted!r} was passed to the inner CLI: {sorted(env)[:5]}..."
        # And we did not accidentally widen to os.environ: at least one of
        # the well-known SECRET_KEYS is NOT here even if it is in os.environ.
        # (LITELLM_MASTER_KEY above already proves that; this is belt and braces.)
        assert env.keys() <= set(server.SUBPROCESS_ENV_KEYS)

    print(f"  spawn env allowlist honoured at both sites: "
          f"{len(captured_envs)} captures, all keys subset of "
          f"{len(server.SUBPROCESS_ENV_KEYS)}-key allowlist; "
          f"{planted} absent")


# --------------------------------------------------------------- sidecar image ---
# Dockerfile.sidecar must COPY switchyard/models.py into the image AND
# read_config() must build a real CallerEnvironmentSettings from it. Two
# tests pin both halves: the Dockerfile line itself (a static guard, the
# same style tests/test_litellm_patch.py uses), and an end-to-end mirror
# of the image layout so a dropped COPY or a silent fallback fails this
# suite, not just the running container.

DOCKERFILE_SIDECAR = os.path.join(os.path.dirname(HERE), "Dockerfile.sidecar")


def test_dockerfile_sidecar_copies_switchyard_models_py():
    """Dockerfile.sidecar must COPY switchyard/models.py into the image.

    The earlier build only carried caller_env.py, so `import switchyard.models`
    failed inside the container and read_config() silently turned plans.yaml's
    `settings.caller_environment` into None on every request. A regression that
    drops the COPY (or moves models.py under a different dst) brings the
    silent-None bug back without a commit message. The exact-occurrence assert
    is deliberate: this Dockerfile is part of the sidecar-image contract.
    """
    with open(DOCKERFILE_SIDECAR, encoding="utf-8") as fh:
        src = fh.read()

    expected = "COPY switchyard/models.py /app/switchyard/models.py"
    assert expected in src, (
        f"Dockerfile.sidecar no longer contains {expected!r}; "
        f"switchyard/models.py is no longer baked into the sidecar image, "
        f"so the bridges cannot build CallerEnvironmentSettings from "
        f"plans.yaml and caller_environment config will silently become None."
    )
    # Build-time assertion must follow: a copy that is present but whose
    # `python3 -c "import switchyard.models; ..."` step was lost would
    # still build a broken image silently, so guard the assertion line too.
    assert "import switchyard.models" in src, (
        "Dockerfile.sidecar lost its build-time assertion that switchyard.models "
        "imports and constructs CallerEnvironmentSettings; a future copy that "
        "misses the file would no longer fail the build."
    )
    # The constructor call is what proves the typed surface shipped, not
    # just the bare `import` -- a future edit that shortens the RUN to
    # `python3 -c "import switchyard.models"` would still pass the import
    # guard above but stop exercising CallerEnvironmentSettings(probe=...).
    # Pin the constructor substring so a "weakened but not deleted" edit
    # also fails this static guard, not only the image-layout test that
    # catches the regression end-to-end after a real `docker compose build`.
    assert "switchyard.models.CallerEnvironmentSettings" in src, (
        "Dockerfile.sidecar lost its build-time assertion that constructs "
        "CallerEnvironmentSettings(probe='required'); the import line may be "
        "present but the typed-surface check is gone, so a future edit that "
        "breaks the constructor (e.g. renames the field) would build a "
        "silent-503 image instead of failing fast."
    )
    print(f"  Dockerfile.sidecar: {expected!r} present (with build-time assertion)")


def _copy_layout_for_dockerfile(image_root: str) -> dict:
    """Parse Dockerfile.sidecar's COPY lines and copy each <src> into <dst>
    under `image_root`. Returns the (src, dst) pairs actually copied so the
    test can name what landed.

    Each `COPY` line may name a single src/dst pair or a directory src; both
    shapes are handled by shutil.copy2 (file) / shutil.copytree (dir). A
    src is treated as a directory when it ends in `/` OR when a path by that
    name exists on disk as a directory under the repo root -- Dockerfile
    `COPY sidecars/mcp_bridge /app/mcp_bridge` is a directory src even
    without a trailing slash, and a literal shutil.copy2 on it raises
    `IsADirectoryError`. The parser is intentionally narrow — it does NOT
    interpret Dockerfile variables, ARG defaults, or multi-stage builds —
    because the sidecar Dockerfile is short and the relevant lines are
    plain COPYs.
    """
    import shutil

    repo_root = os.path.dirname(HERE)
    pairs = []
    with open(DOCKERFILE_SIDECAR, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line.startswith("COPY "):
                continue
            parts = line.split()[1:]
            if len(parts) != 2:
                # Multi-source COPY (COPY a b c /dst/) is not used here.
                continue
            src, dst = parts
            full_dst = os.path.join(image_root, dst.lstrip("/"))
            os.makedirs(os.path.dirname(full_dst), exist_ok=True)
            src_on_disk = os.path.join(repo_root, src.rstrip("/"))
            if src.endswith("/") or os.path.isdir(src_on_disk):
                shutil.copytree(src_on_disk, full_dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src_on_disk, full_dst)
            pairs.append((src, dst))
    return pairs


def test_image_layout_builds_a_real_caller_environment_settings():
    """End-to-end mirror of Dockerfile.sidecar's image: a temp `/app` is
    populated from the Dockerfile's own COPY lines, a minimal plans.yaml
    sets `settings.caller_environment`, and a fresh subprocess runs
    `read_config()` against that layout. A dropped COPY line OR a silent
    `None` fallback in the code is caught here, not only by the running
    image.

    The reproduction runs `read_config()` in a clean subprocess so the
    sidecar module's import-time globals (PROVIDER, PLAN, PLANS_PATH,
    _models) are constructed fresh against the mirrored layout, the same
    way the running container constructs them. Asserting the typed
    surface — `caller_environment` is not None, `.probe == "required"`,
    `.platform == "win32"` — pins both halves of the contract.
    """
    import shutil
    import subprocess

    image_root = Path(tempfile.mkdtemp(prefix="clib-imagelayout-"))
    try:
        pairs = _copy_layout_for_dockerfile(str(image_root))
        # Sanity-check the layout that the parser produced: the bridge
        # module, the switchyard package init, and the two switchyard
        # modules the bridges need (caller_env + models). Anything else
        # the Dockerfile copies (harness configs, mcp_bridge/) is fine to
        # have but not relevant to this test's contract.
        copied = {dst for _src, dst in pairs}
        assert "/app/cli_bridge/server.py" in copied, sorted(copied)
        assert "/app/switchyard/__init__.py" in copied, sorted(copied)
        assert "/app/switchyard/caller_env.py" in copied, sorted(copied)
        assert "/app/switchyard/models.py" in copied, sorted(copied)

        # Minimal plans.yaml: one plan, settings.caller_environment set
        # to a recognisable shape so a successful build is observable
        # (probe + platform are the two fields CallerEnvironmentSettings
        # exposes at its constructor).
        plans_path = image_root / "plans.yaml"
        plans_path.write_text(
            "settings:\n"
            "  caller_environment:\n"
            "    probe: required\n"
            "    platform: win32\n"
            "plans:\n"
            "  regression:\n"
            "    max_parallel: 1\n"
            "    models:\n"
            "      m:\n"
            "        model: m\n"
            "        enabled: true\n"
        )

        # Import and call read_config() in a fresh subprocess against
        # the mirrored layout. cwd /app/cli_bridge mirrors the running
        # container's `cd /app/${BRIDGE:-cli}_bridge`; PYTHONPATH=<root>/app
        # mirrors the Dockerfile's ENV PYTHONPATH=/app so a plain
        # `import switchyard.models` resolves the package we copied.
        # The harness /mcp_bridge dirs are not needed for read_config()
        # but the Dockerfile COPY them; the parser above will have copied
        # them if present, which is harmless.
        env = os.environ.copy()
        env["PYTHONPATH"] = str(image_root / "app")
        env["PROVIDER"] = "claude"
        env["SWITCHYARD_PLAN"] = "regression"
        env["SWITCHYARD_PLANS"] = str(plans_path)
        # Keep litellm (pulled by switchyard.models' guarded import) on
        # the local backup map for the subprocess too, mirroring what
        # plans_path does for the test suite.
        env.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

        proc = subprocess.run(
            [sys.executable, "-c",
             "import sys, json;"
             "sys.path.insert(0, '.');"
             "import importlib.util;"
             "_spec = importlib.util.spec_from_file_location('regression_cli_bridge',"
             " 'server.py');"
             "_mod = importlib.util.module_from_spec(_spec);"
             # Register BEFORE exec: cli_bridge defines a @dataclass that"
             # looks itself up by cls.__module__ in sys.modules during class"
             # construction, so the spec-loaded module has to be in"
             # sys.modules before exec_module runs."
             "sys.modules['regression_cli_bridge'] = _mod;"
             "_spec.loader.exec_module(_mod);"
             "cfg = _mod.read_config();"
             "ce = cfg.caller_environment;"
             "print(json.dumps({'caller_environment_is_none': ce is None,"
             " 'probe': getattr(ce, 'probe', None),"
             " 'platform': getattr(ce, 'platform', None)}))"],
            cwd=str(image_root / "app" / "cli_bridge"),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, (
            f"subprocess failed (rc={proc.returncode}); "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        # Locate the JSON line: subprocess.run may have litellm's noisy
        # stdout ahead of it. Take the last line that parses as JSON.
        import json as _json
        payload = None
        for line in proc.stdout.splitlines()[::-1]:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            break
        assert payload is not None, (
            f"subprocess did not emit a parseable JSON result; "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        # The full contract: caller_environment is a typed settings
        # instance, the probe field is what plans.yaml said, and the
        # platform field round-trips. A dropped COPY line in the
        # Dockerfile would leave _models is None and `caller_environment`
        # would be dropped on the floor by the new log.error branch,
        # so the first assertion fails.
        assert payload["caller_environment_is_none"] is False, payload
        assert payload["probe"] == "required", payload
        assert payload["platform"] == "win32", payload
        print(f"  image-mirror: read_config().caller_environment "
              f"probe={payload['probe']!r} platform={payload['platform']!r} "
              f"(not None — Dockerfile COPY line + module-level import "
              f"both intact)")
    finally:
        shutil.rmtree(image_root, ignore_errors=True)


# ------------------------------------------ issue #127: envelope -> error ----
def test_exit_zero_envelope_error_is_classified_via_http_surface():
    """Issue #127 pin: cli_bridge's HTTP surface for the exit-0-but-erroring
    JSON envelope must not change when the inline check is replaced by
    ``check_result_envelope``. A usage-limit envelope is a 429 (with
    Retry-After); an error_max_turns envelope is a 502.

    The fixture parses via ``claude_json`` (``json.loads(stdout)``) so the
    envelope survives verbatim into ``check_result_envelope``; the opencode
    parser used by this file's default fixture strips unknown events.
    """
    import asyncio
    from fastapi import HTTPException

    def make_fake(*, stdout: str) -> str:
        f = tempfile.NamedTemporaryFile(
            "w", suffix=".py", prefix="clib-env-", delete=False)
        f.write("import sys\n"
                f"sys.stdout.write({stdout!r})\n")
        f.close()
        return f.name

    real_cli, real_bare, real_args, real_parser = (
        server.CLI, server.BARE, server.PROFILE["args"], server.PROFILE["parser"])
    try:
        server.CLI = sys.executable
        server.BARE = False
        # claude_json preserves the envelope verbatim; events_json/codex_jsonl
        # would strip is_error/subtype and rebuild result/usage from events.
        server.PROFILE["parser"] = "claude_json"

        # --- exit-0 with is_error=true + usage-limit text -> 429 ---
        limited = ('{"type":"result","is_error":true,'
                   '"result":"Claude AI usage limit reached|1760000000",'
                   '"usage":{"input_tokens":0,"output_tokens":0}}')
        limited_fake = make_fake(stdout=limited)
        server.PROFILE["args"] = [limited_fake]
        try:
            try:
                asyncio.run(server._run_cli("hi", None, "xai/grok-4.6", []))
            except HTTPException as exc:
                assert exc.status_code == 429, (exc.status_code, exc.detail)
                assert exc.headers and "Retry-After" in exc.headers, exc.headers
            else:
                raise AssertionError("expected a 429 HTTPException")
        finally:
            os.unlink(limited_fake)

        # --- exit-0 with error_max_turns subtype -> 502, no Retry-After ---
        max_turns = '{"subtype":"error_max_turns","result":"max turns exceeded"}'
        max_turns_fake = make_fake(stdout=max_turns)
        server.PROFILE["args"] = [max_turns_fake]
        try:
            try:
                asyncio.run(server._run_cli("hi", None, "xai/grok-4.6", []))
            except HTTPException as exc:
                assert exc.status_code == 502, (exc.status_code, exc.detail)
                assert "Retry-After" not in (exc.headers or {}), exc.headers
            else:
                raise AssertionError("expected a 502 HTTPException")
        finally:
            os.unlink(max_turns_fake)

        print("  cli_bridge HTTP surface: exit-0 is_error+limit -> 429, "
              "error_max_turns -> 502 (no Retry-After)")
    finally:
        server.CLI, server.BARE, server.PROFILE["args"], server.PROFILE["parser"] = \
            real_cli, real_bare, real_args, real_parser


# ------------------------------------------- issue #141: last-good on parse err ---
# `read_config()` previously caught only (OSError, ValueError, TypeError) when
# walking plans.yaml by hand, so a yaml.YAMLError (a stray tab is a
# `ScannerError`) fell through uncaught and `config()` raised on every request
# for as long as the file stayed broken. A value typo (`max_parallel: two`) WAS
# caught and silently served the profile-default model at concurrency 1.
#
# The fix routes read_config() through `models.load`, which raises on every
# parse / value / shape error, and config() catches to keep the last-good
# `_config` on the warm path. The four tests below pin each of the four
# outcomes the issue calls out (issue #141): last-good on bad yaml, last-good
# on a value error (model unchanged, source stays config), recovery when the
# file is fixed (new mtime), and the cold-start fallback so /health still
# reports source=fallback. Each scenario works on a tempfile copy of the
# tracked fixture; the runner resets `_config`, `_config_at`, and
# `_config_mtime` between scenarios so a previous test's state cannot leak.


def _load_server_with_plans(plans_path: str, *, preserve_escape_hatches: bool = False):
    """Reload server under `plans_path` and reset module globals.

    Returns (mod, old_env). The caller must `_restore_env_and_reload(old_env)`
    in a finally block. Resetting `_config`, `_config_at`, and
    `_config_mtime` puts the module at a cold start, the same way `_read_for`
    and the new `_fallback_config` path expect.

    `preserve_escape_hatches=True` keeps any SIDECAR_CONCURRENCY /
    <PROVIDER>_MODEL already in `os.environ` so the cold-start fallback
    can be driven by them (test (4)). Other tests pop them so an env
    inherited from a sibling test cannot leak.
    """
    import _modules
    old = dict(os.environ)
    os.environ.update({"PROVIDER": "claude", "SWITCHYARD_PLAN": "claude-max",
                       "SWITCHYARD_PLANS": plans_path})
    if not preserve_escape_hatches:
        for key in ("SIDECAR_CONCURRENCY", "CLAUDE_MODEL", "CODEX_MODEL", "OPENCODE_MODEL"):
            os.environ.pop(key, None)
    mod = _modules.reload(server)
    mod._config = None
    mod._config_at = 0.0
    mod._config_mtime = None
    return mod, old


def test_last_good_kept_on_invalid_yaml():
    """Good load then invalid YAML -> config() returns the previous config.

    A stray tab makes plans.yaml unparseable. The previous code's
    `except (OSError, ValueError, TypeError)` missed ``yaml.YAMLError``
    (issue #141), so every request 500'd for as long as the file stayed
    broken. With `models.load` driving `read_config()`, parse errors
    propagate and `config()` keeps the last-good Config so the sidecar
    serves the previous config until the operator fixes the file.
    """
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="clib-141-yaml-"))
    try:
        plans = tmp / "plans.yaml"
        plans.write_text(Path(PLANS).read_text())
        mod, old = _load_server_with_plans(str(plans))
        try:
            cfg_good = mod.config()
            assert cfg_good.source == "config", cfg_good
            assert "claude-opus-5" in cfg_good.models, cfg_good.models
            assert cfg_good.concurrency == 2, cfg_good.concurrency

            # Stray tab in `models:` -> ScannerError (subclass of
            # YAMLError, not a ValueError), which the old
            # `except (OSError, ValueError, TypeError)` clause missed.
            plans.write_text("plans:\n  claude-max:\n\tmodels: tab-indented\n")
            mod._config_at = 0.0      # force TTL refresh on the next config() call
            cfg_after = mod.config()
            # The warm-sidecar rule: a broken plans.yaml must not steal
            # the config the sidecar was already serving on.
            assert cfg_after.model == cfg_good.model, (cfg_good.model, cfg_after.model)
            assert cfg_after.concurrency == cfg_good.concurrency, cfg_after.concurrency
            assert cfg_after.models == cfg_good.models, cfg_after.models
            assert cfg_after.source == "config", cfg_after.source
        finally:
            _restore_env_and_reload(old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  invalid YAML (stray tab) -> last-good config preserved")


def test_last_good_kept_on_value_error():
    """Good load then `max_parallel: two` -> previous config, model
    unchanged (not the profile default), source stays "config".

    A value typo used to fall into the same broad `except ValueError` and
    silently serve the profile-default model at concurrency 1, looking
    healthy while doing it. With `models.load` raising on shape errors,
    `config()` keeps the last-good Config -- the warm-sidecar rule: a
    transient bad edit must never substitute the profile default for a
    model the operator vetted.
    """
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="clib-141-value-"))
    try:
        plans = tmp / "plans.yaml"
        plans.write_text(Path(PLANS).read_text())
        mod, old = _load_server_with_plans(str(plans))
        try:
            cfg_good = mod.config()
            assert cfg_good.source == "config", cfg_good
            opus_model = cfg_good.model
            assert "claude-opus-5" in cfg_good.models, cfg_good.models
            # Sanity: the profile default for claude is something else
            # (PROFILE["model"]), so `cfg_good.model == PROFILE["model"]`
            # would mean read_config() already silently fell back -- the
            # very regression this test pins.
            assert opus_model != mod.PROFILE["model"], \
                f"sanity: last-good model is NOT the profile default " \
                f"(got model={opus_model!r}, profile_default={mod.PROFILE['model']!r})"

            # Same plans.yaml but `max_parallel: two` for claude-max.
            # _parse_plan_max_parallel rejects strings other than "auto"
            # as a ValueError, which propagates out of read_config().
            content = Path(PLANS).read_text()
            claude_max_block = ("  claude-max:\n"
                                "    label: \"Claude Max $200\"\n"
                                "    auth: cli_sidecar             # OAuth; "
                                "the Claude CLI holds the credential\n"
                                "    monthly_cost: 100\n"
                                "    max_parallel: 2\n"
                                "    max_parallel_ceiling: 2       # never probe above it\n")
            assert claude_max_block in content, "anchor missing — plans.example.yaml moved"
            plans.write_text(content.replace(
                claude_max_block,
                claude_max_block.replace("max_parallel: 2\n", "max_parallel: two\n", 1),
                1))
            mod._config_at = 0.0      # force TTL refresh
            cfg_after = mod.config()
            # Warm-sidecar rule: model is the last-good (claude-opus-5),
            # NOT the profile default; source stays config.
            assert cfg_after.model == opus_model, \
                f"model unchanged on value error (was {opus_model!r}, " \
                f"got {cfg_after.model!r})"
            assert cfg_after.model != mod.PROFILE["model"], \
                "warm sidecar must never substitute the profile default on a bad edit"
            assert cfg_after.concurrency == cfg_good.concurrency, cfg_after.concurrency
            assert cfg_after.models == cfg_good.models, cfg_after.models
            assert cfg_after.source == "config", \
                f"source stays config on warm-sidecar last-good (got {cfg_after.source!r})"
        finally:
            _restore_env_and_reload(old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  invalid value (max_parallel: two) -> last-good, model not profile default")


def test_fixed_plans_yaml_is_picked_up_after_mtime_advance():
    """Once the file is fixed and the mtime advances past the logged
    failure, the next config() call picks up the new config.

    The mtime stamp on the failure is bumped to the broken-file mtime
    so a subsequent read of the same broken file does NOT log again
    (mirroring hooks.py:_maybe_reload). An edit that produces a new
    mtime clears that stamp and the next config() call reloads.
    """
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="clib-141-recover-"))
    try:
        plans = tmp / "plans.yaml"
        plans.write_text(Path(PLANS).read_text())
        mod, old = _load_server_with_plans(str(plans))
        try:
            cfg_good = mod.config()
            assert cfg_good.source == "config", cfg_good
            assert cfg_good.concurrency == 2, cfg_good.concurrency

            # Break plans.yaml with a value error in the claude-max
            # block; the anchor below uniquely identifies it.
            content = Path(PLANS).read_text()
            claude_max_block = ("  claude-max:\n"
                                "    label: \"Claude Max $200\"\n"
                                "    auth: cli_sidecar             # OAuth; "
                                "the Claude CLI holds the credential\n"
                                "    monthly_cost: 100\n"
                                "    max_parallel: 2\n"
                                "    max_parallel_ceiling: 2       # never probe above it\n")
            assert claude_max_block in content, "anchor missing — plans.example.yaml moved"
            plans.write_text(content.replace(
                claude_max_block,
                claude_max_block.replace("max_parallel: 2\n", "max_parallel: two\n", 1),
                1))
            mod._config_at = 0.0      # force TTL refresh
            cfg_after_break = mod.config()
            assert cfg_after_break.concurrency == cfg_good.concurrency

            # Fix the file with a different cap so the new config is
            # distinguishable from the last-good. Force a new mtime so
            # any mtime-stamp tracking on the failure path is cleared
            # before we ask config() to reload. The cap AND its ceiling
            # have to move together: max_parallel_ceiling must be >=
            # max_parallel or the loader rejects the file (models.py
            # _parse_plan_max_parallel_ceiling).
            plans.write_text(content.replace(
                claude_max_block,
                claude_max_block.replace("max_parallel: 2",
                                         "max_parallel: 3").replace(
                    "max_parallel_ceiling: 2", "max_parallel_ceiling: 3", 1),
                1))
            # Future mtime, well past any stamp on the broken-file
            # failure so the next config() call's read_config() runs.
            future = time.time() + 60
            os.utime(str(plans), (future, future))
            mod._config_at = 0.0      # force TTL refresh
            cfg_fixed = mod.config()
            # New cap is picked up; last-good is no longer served.
            assert cfg_fixed.concurrency == 3, \
                f"fixed plans.yaml -> new cap 3 (got {cfg_fixed.concurrency})"
            assert cfg_fixed.source == "config", cfg_fixed.source
            assert cfg_fixed.models == cfg_good.models, cfg_fixed.models
        finally:
            _restore_env_and_reload(old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  broken file -> fixed file (new mtime) -> new config picked up")


def test_cold_start_on_broken_file_serves_fallback_with_escape_hatches():
    """A sidecar that boots against a broken plans.yaml (cold start, no
    last-good) must keep /health reporting source=fallback instead of
    crashing every request with a load failure.

    The escape hatches (SIDECAR_CONCURRENCY / <PROVIDER>_MODEL) still
    apply so an operator can steer a freshly-deployed sidecar by hand
    until plans.yaml is repaired.
    """
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="clib-141-cold-"))
    try:
        plans = tmp / "plans.yaml"
        plans.write_text("plans:\n  claude-max:\n\tmodels: tab-indented\n")

        # Cold start: nothing yet in `_config`, broken plans.yaml.
        # `_load_server_with_plans` resets `_config = None` to simulate
        # the cold start, the same way the other tests do.
        mod, old = _load_server_with_plans(str(plans))
        try:
            cfg = mod.config()
            assert cfg.source == "fallback", \
                f"cold-start fallback reports source=fallback (got {cfg.source!r})"
            assert cfg.model == mod.PROFILE["model"], \
                f"cold-start model is the profile default (got {cfg.model!r})"
            assert cfg.concurrency == 1, \
                f"cold-start concurrency is 1 (got {cfg.concurrency})"
        finally:
            _restore_env_and_reload(old)

        # Escape hatches steer the cold-start fallback so an operator can
        # bring a freshly-deployed sidecar up by hand. Set them BEFORE
        # the helper runs, with `preserve_escape_hatches=True` so the
        # helper does not pop them.
        os.environ["SIDECAR_CONCURRENCY"] = "7"
        os.environ["CLAUDE_MODEL"] = "claude-escape-hatch"
        mod, old = _load_server_with_plans(str(plans), preserve_escape_hatches=True)
        try:
            cfg = mod.config()
            assert cfg.source == "fallback", cfg.source
            assert cfg.model == "claude-escape-hatch", \
                f"<PROVIDER>_MODEL escape hatch applied (got {cfg.model!r})"
            assert cfg.concurrency == 7, \
                f"SIDECAR_CONCURRENCY escape hatch applied (got {cfg.concurrency})"
        finally:
            _restore_env_and_reload(old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  cold start on broken file -> source=fallback (escape hatches honoured)")


# -------------------------------------------- issue #292: reasoning propagation ---
def test_request_thinking_reads_every_carrier_and_shape():
    """The reasoning-request/display policy reaches the bridges under several
    spellings:
        * `extra_body.switchyard.thinking` (the gateway's carrier, set by
          hooks.py:carry_to_cli_sidecar for CLI plans)
        * top-level `thinking` (a non-routed caller, or any Anthropic-shape
          request the gateway lifted onto the carrier but a test hits directly)
        * OpenAI's `reasoning` with only a `display` field (when the gateway
          already popped the effort-typed reason but kept the display marker
          on the carrier).
    A `thinking.type` value of `disabled` is treated as no policy at all (the
    user's explicit "off" -- not something to be lifted). Unknown
    `type`/`display` values are dropped silently so a future schema bump
    cannot silently disable reasoning on the sidecar.
    """
    assert server.request_thinking({"switchyard": {"thinking": {"type": "enabled"}}}) \
        == {"type": "enabled"}
    assert server.request_thinking({"switchyard": {"thinking": {"type": "adaptive",
                                                                 "display": "full"}}}) \
        == {"type": "adaptive", "display": "full"}
    assert server.request_thinking({"thinking": {"type": "enabled",
                                                  "display": "summarized"}}) \
        == {"type": "enabled", "display": "summarized"}
    assert server.request_thinking({"switchyard": {"thinking": {"type": "disabled"}}}) \
        is None, "disabled is off-by-explicit-choice; never lifted"
    assert server.request_thinking({"switchyard": {"thinking": {"type": "unknown"}}}) \
        is None, "unknown type does not silently keep reasoning on"
    assert server.request_thinking({"switchyard": {"thinking": {"display": "omitted"}}}) \
        == {"display": "omitted"}, "display alone is a valid policy (no type)"
    assert server.request_thinking({}) is None
    # Provider hint: display policy lives on a different key in some shapes.
    assert server.request_thinking({"reasoning": {"display": "omitted"}}) \
        == {"display": "omitted"}
    print("  reasoning policy read from extra_body.switchyard.thinking, "
          "thinking, and reasoning.display (disabled -> off, unknown -> off)")


def test_thinking_args_only_added_when_a_policy_was_requested():
    """The CLI's reasoning mode is only turned on when the caller asked; a
    request with no policy gets nothing on the argv. Each profile gates on
    `type in ("enabled", "adaptive")` -- not on a display value alone, which
    is a presentation choice, not an enablement signal.
    """
    # No policy -> no argv fragment, on every profile.
    for prov, prof in (("claude", server.PROFILES["claude"]),
                       ("codex", server.PROFILES["codex"]),
                       ("opencode", server.PROFILES["opencode"])):
        saved = (server.PROVIDER, server.PROFILE, server.CLI)
        try:
            server.PROVIDER, server.PROFILE, server.CLI = prov, prof, prof["cli"]
            assert server.thinking_args(None) == [], prov
            assert server.thinking_args({}) == [], prov
            # A display-only policy is NOT an enable signal.
            assert server.thinking_args({"display": "omitted"}) == [], prov
            # An explicit enable signal -> each profile's argv-builder
            # already covers reasoning via effort_args(); thinking_args adds
            # nothing extra because (a) claude/codex have no per-call
            # reasoning switch beyond the effort override, and (b) opencode
            # gates reasoning behind `--variant`, already handled. A future
            # profile that needs an explicit switch will land here.
            assert server.thinking_args({"type": "enabled"}) == [], prov
            assert server.thinking_args({"type": "adaptive"}) == [], prov
        finally:
            server.PROVIDER, server.PROFILE, server.CLI = saved
    print("  thinking_args: no policy -> nothing; display-only -> nothing; "
          "enabled/adaptive -> empty (effort already gates reasoning)")


def test_parse_output_separates_reasoning_from_answer_for_three_clis():
    """OpenCode's `part.type == "reasoning"` events collect into a separate
    `reasoning` field; so do Claude's `thinking` content blocks and Codex's
    `item.type == "reasoning"` items. The assistant text and the chain of
    thought are both kept (issue #292): a downstream adapter lifts the
    reasoning onto `reasoning_content` rather than burying it inside the
    answer the user reads.
    """
    opencode_stream = "\n".join([
        json.dumps({"type": "reasoning", "part": {"type": "reasoning",
                                                   "text": "let me think"}}),
        json.dumps({"type": "reasoning", "part": {"type": "reasoning",
                                                   "text": "step two"}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": "answer"}}),
        json.dumps({"type": "step_finish", "part": {
            "type": "step-finish",
            "tokens": {"input": 10, "output": 5, "reasoning": 15}}}),
    ])
    out = server.parse_output(opencode_stream)
    assert out["result"] == "answer", out
    assert out["reasoning"] == "let me thinkstep two", out
    assert out["usage"]["output_tokens"] == 20, out["usage"]

    codex_stream = "\n".join([
        json.dumps({"type": "item.completed",
                    "item": {"type": "reasoning", "text": "thinking..."}}),
        json.dumps({"type": "item.completed",
                    "item": {"type": "agent_message", "text": "ok"}}),
        json.dumps({"type": "turn.completed",
                    "usage": {"input_tokens": 4, "output_tokens": 3,
                              "reasoning_output_tokens": 12}}),
    ])
    out = server.parse_output(codex_stream, "codex_jsonl")
    assert out["result"] == "ok", out
    assert out["reasoning"] == "thinking...", out
    assert out["usage"]["output_tokens"] == 15, out["usage"]

    claude_stream_json = "\n".join(json.dumps(e) for e in [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "why"},
            {"type": "text", "text": "answer"},
        ]}},
        {"type": "result", "subtype": "success", "is_error": False,
         "result": "answer", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ])
    out = server.parse_output(claude_stream_json, "claude_json")
    assert out["result"] == "answer", out
    assert out["reasoning"] == "why", out
    print("  three CLIs: reasoning text on `payload.reasoning`, answer stays on `result`")


def test_parse_output_omits_reasoning_when_the_cli_emitted_none():
    """A regular answer carries no reasoning: the `reasoning` key is absent,
    not present-but-empty. LiteLLM and downstream consumers should treat its
    absence as the "no reasoning requested/returned" signal -- an empty string
    would coerce to a falsy value in many clients and confuse the
    omitted-display contract.
    """
    plain = "\n".join([
        json.dumps({"type": "text", "part": {"type": "text", "text": "hi"}}),
        json.dumps({"type": "step_finish", "part": {
            "type": "step-finish",
            "tokens": {"input": 1, "output": 1, "reasoning": 0}}}),
    ])
    out = server.parse_output(plain)
    assert "reasoning" not in out, out
    assert out["result"] == "hi"
    print("  parse_output: no reasoning emitted -> reasoning key absent (not empty)")


def test_to_openai_emits_reasoning_content_when_present():
    """The OpenAI chat-completion message gains a `reasoning_content` field
    on any turn the parser separated reasoning from answer. The field is
    omitted on turns without reasoning, exactly the absence-shape LiteLLM's
    Messages/Responses adapters pass through as "no reasoning requested"."""
    payload = {"result": "answer", "reasoning": "chain of thought",
               "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}}
    out = server.to_openai(payload, "m")
    msg = out["choices"][0]["message"]
    assert msg["content"] == "answer", msg
    assert msg["reasoning_content"] == "chain of thought", msg

    no_reasoning = {"result": "answer",
                    "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}}
    out2 = server.to_openai(no_reasoning, "m")
    assert "reasoning_content" not in out2["choices"][0]["message"], out2
    print("  to_openai: reasoning_content present iff reasoning present, "
          "absent otherwise")


def test_to_openai_omitted_display_suppresses_visible_content():
    """`thinking.display == "omitted"` means the caller wants the
    reasoning ONLY -- the assistant's visible text is suppressed. The
    reasoning is preserved on `reasoning_content` so an adapter can render
    it; a downstream `text`-only consumer sees an empty `content` rather
    than the duplicated answer. The shape stays stable: the message has
    both fields, one empty, one filled, so adapter paths don't have to
    branch on presence.
    """
    payload = {"result": "the answer", "reasoning": "the chain",
               "usage": {"input_tokens": 1, "output_tokens": 1}}
    out = server.to_openai(payload, "m", reasoning_display="omitted")
    msg = out["choices"][0]["message"]
    assert msg["content"] == "", msg
    assert msg["reasoning_content"] == "the chain", msg

    # Other display values don't suppress content.
    out2 = server.to_openai(payload, "m", reasoning_display="full")
    assert out2["choices"][0]["message"]["content"] == "the answer", out2
    print("  to_openai: display=omitted -> content=\"\"; display=full -> content=result")


def test_sse_from_completion_emits_reasoning_before_content():
    """An OpenAI chat-completion stream that carries reasoning has it on
    `choices[0].delta.reasoning_content` BEFORE `choices[0].delta.content`,
    matching the documented order on the wire for streams of native
    reasoning models. Reasoning-only turns get the field, content-only turns
    do not -- an empty first delta is never emitted on a regular turn.
    """
    import asyncio as _asyncio
    payload = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant",
                                         "content": "answer",
                                         "reasoning_content": "chain"}}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    frames = []
    async def collect():
        async for frame in server.sse_from_completion(payload, "m"):
            frames.append(frame)
    _asyncio.run(collect())
    assert len(frames) == 3, frames
    first_delta = json.loads(frames[0][len("data: "):])["choices"][0]["delta"]
    keys = list(first_delta.keys())
    r_idx = keys.index("reasoning_content")
    c_idx = keys.index("content")
    assert r_idx < c_idx, ("reasoning_content before content", keys)
    assert first_delta["reasoning_content"] == "chain"
    assert first_delta["content"] == "answer"

    # A no-reasoning payload: stream has no reasoning_content delta.
    plain = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
             "choices": [{"index": 0, "finish_reason": "stop",
                          "message": {"role": "assistant", "content": "answer"}}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    frames2 = []
    async def collect2():
        async for frame in server.sse_from_completion(plain, "m"):
            frames2.append(frame)
    _asyncio.run(collect2())
    delta2 = json.loads(frames2[0][len("data: "):])["choices"][0]["delta"]
    assert "reasoning_content" not in delta2, delta2
    print("  SSE: reasoning_content emitted before content when present; "
          "absent on regular turns")


def test_thinking_display_omitted_means_text_path_result_is_empty():
    """End-to-end on the text path: a request carrying the omitted-display
    policy produces a response whose `message.content` is empty and whose
    `message.reasoning_content` carries the chain. The round-trip is the
    whole contract -- an empty `content` with no `reasoning_content` would
    be a silent drop.
    """
    import asyncio as _asyncio
    captured = {}

    async def fake_invoke(prompt, system, model, image_paths=None, fmt=None, *,
                          web=False, effort=None, thinking=None):
        captured["thinking"] = thinking
        return {"result": "the answer", "reasoning": "the chain",
                "usage": {"input_tokens": 1, "output_tokens": 1}}

    real_invoke = server.invoke
    server.invoke = fake_invoke
    saved = (server.PROVIDER, server.PROFILE, server.CLI)
    try:
        server.PROVIDER, server.PROFILE, server.CLI = ("claude", server.PROFILES["claude"],
                                                      server.PROFILES["claude"]["cli"])
        body = {"model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "switchyard": {"thinking": {"type": "enabled",
                                             "display": "omitted"}}}
        result = _asyncio.run(server._handle_chat(body))
    finally:
        server.invoke = real_invoke
        server.PROVIDER, server.PROFILE, server.CLI = saved

    # The carrier reaches invoke, not just the response layer.
    assert captured.get("thinking") == {"type": "enabled", "display": "omitted"}, captured
    msg = result["choices"][0]["message"]
    assert msg["content"] == "", msg
    assert msg["reasoning_content"] == "the chain", msg
    print("  end-to-end: switchyard.thinking.type=enabled + display=omitted -> "
          "content=\"\", reasoning_content populated")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
