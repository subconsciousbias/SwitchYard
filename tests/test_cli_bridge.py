"""Parser tests for the CLI bridge, against output the CLIs really produced.

The OpenCode fixture is a verbatim capture from `opencode run --format json`.
It exists because the first parser was written from a guess at the event shape:
it found no text, fell back to returning the raw event stream as the assistant's
answer, and reported zero tokens — a failure that looks like a working call.
"""
from __future__ import annotations

import base64
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

if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} cli-bridge tests passed")
