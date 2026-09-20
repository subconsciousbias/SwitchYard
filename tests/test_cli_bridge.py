"""Parser tests for the CLI bridge, against output the CLIs really produced.

The OpenCode fixture is a verbatim capture from `opencode run --format json`.
It exists because the first parser was written from a guess at the event shape:
it found no text, fell back to returning the raw event stream as the assistant's
answer, and reported zero tokens — a failure that looks like a working call.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sidecars", "cli_bridge"))
os.environ.setdefault("PROVIDER", "opencode")

import server  # noqa: E402

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
    argv = server.build_argv("PROMPT", None)
    if server.PROVIDER == "codex":
        assert "--model" not in argv, argv
        assert "--skip-git-repo-check" in argv, argv
        print(f"  {' '.join(argv)}")
    else:
        print(f"  (skipped: PROVIDER={server.PROVIDER})")


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
PLANS = os.path.join(os.path.dirname(HERE), "config", "plans.yaml")


def _read_for(plan: str, provider: str):
    """read_config() for one plan, without disturbing the module's globals."""
    import importlib
    old = dict(os.environ)
    os.environ.update({"PROVIDER": provider, "SWITCHYARD_PLAN": plan,
                       "SWITCHYARD_PLANS": PLANS})
    for key in ("SIDECAR_CONCURRENCY", "CLAUDE_MODEL", "CODEX_MODEL", "OPENCODE_MODEL"):
        os.environ.pop(key, None)
    try:
        mod = importlib.reload(server)
        return mod.read_config()
    finally:
        os.environ.clear()
        os.environ.update(old)
        importlib.reload(server)


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
    `openai/opencode-go/glm-5.3-flash` must keep its provider/model shape."""
    assert _read_for("grok", "opencode").model == "xai/grok-4.6"
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

if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
            n += 1
    print(f"\n{n} cli-bridge tests passed")
