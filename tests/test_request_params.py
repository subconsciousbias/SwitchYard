"""OpenAI request parameters no CLI has a flag for: `stop`, `n`, and
`response_format` (issue #264's conformance matrix).

Each used to come back as a 200 that ignored it -- the one failure the router
cannot spill on. These pin the contract: executed (stop applied, n fanned
out, the schema enforced natively or instructed and validated) or refused
with a 4xx/502 before or instead of a wrong 200.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plans_path import plans_path  # noqa: E402

os.environ["SWITCHYARD_PLANS"] = plans_path()
from _modules import load  # noqa: E402

server = load("cli_bridge_request_params",
              os.path.join(os.path.dirname(HERE), "sidecars", "cli_bridge", "server.py"))
from fastapi import HTTPException  # noqa: E402

SUM = {"type": "object", "properties": {"answer": {"type": "integer"}},
       "required": ["answer"], "additionalProperties": False}


def _schema_body(strict: bool = True) -> dict:
    return {"response_format": {"type": "json_schema", "json_schema": {
        "name": "sum", "strict": strict, "schema": SUM}}}


def _status(fn, *args):
    try:
        fn(*args)
    except HTTPException as exc:
        return exc.status_code, exc.detail
    return None, None


class _Provider:
    """Swap the loaded server's provider for one block."""

    def __init__(self, provider: str):
        self.provider = provider

    def __enter__(self):
        self.saved = (server.PROVIDER, server.PROFILE, server.CLI)
        server.PROVIDER = self.provider
        server.PROFILE = server.PROFILES[self.provider]
        server.CLI = server.PROFILE["cli"]

    def __exit__(self, *exc):
        server.PROVIDER, server.PROFILE, server.CLI = self.saved


def test_stop_cuts_at_the_earliest_sequence():
    assert server.apply_stop("alpha STOP beta END", ["END", "STOP"]) == ("alpha ", True)
    assert server.apply_stop("alpha STOP", "STOP") == ("alpha ", True)
    assert server.apply_stop("alpha", ["zzz", ""]) == ("alpha", False)
    assert server.apply_stop("alpha", None) == ("alpha", False)


def test_n_is_validated():
    assert server.request_n({}) == 1
    assert server.request_n({"n": 3}) == 3
    for bad in (0, -1, server.MAX_N + 1, "2", True, 1.5):
        status, detail = _status(server.request_n, {"n": bad})
        assert status == 400 and detail["error"]["param"] == "n", (bad, status)


def test_response_format_parsing_refuses_what_it_cannot_honour():
    assert server.response_schema({}) is None
    assert server.response_schema({"response_format": {"type": "text"}}) is None
    obj = server.response_schema({"response_format": {"type": "json_object"}})
    assert obj["json_object"] and obj["schema"] == {"type": "object"}, obj
    fmt = server.response_schema(_schema_body())
    assert fmt["schema"] == SUM and fmt["strict"] is True, fmt
    for bad in ({"type": "json_schema", "json_schema": {"name": "x"}},
                {"type": "json_schema", "json_schema": {"schema": {"type": 12}}},
                {"type": "grammar"}, "garbage", {"schema": {"type": "object"}}, ["json"]):
        status, detail = _status(server.response_schema, {"response_format": bad})
        assert status == 400 and detail["error"]["param"] == "response_format", (bad, detail)


def test_structured_answers_are_validated_or_502():
    fmt = server.response_schema(_schema_body())
    assert server.conform_structured('{"answer": 5}', fmt) == '{"answer": 5}'
    for tag in ("json", "JSON", "json5", ""):
        fenced = f'```{tag}\n{{"answer": 5}}\n```'
        assert server.conform_structured(fenced, fmt) == '{"answer": 5}', tag
    for bad in ("The answer is 5.", '{"answer": "five"}', '{"answer": 5, "x": 1}'):
        status, detail = _status(server.conform_structured, bad, fmt)
        assert status == 502 and detail["error"]["type"] == "structured_output_invalid", bad
    obj = server.response_schema({"response_format": {"type": "json_object"}})
    assert _status(server.conform_structured, "[1, 2]", obj)[0] == 502
    assert server.conform_structured('{"a": [1]}', obj) == '{"a": [1]}'


def test_finish_completion_skips_tool_turns_and_length_cuts():
    def completion(content, finish="stop", tool_calls=None):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {"choices": [{"index": 0, "message": message, "finish_reason": finish}]}

    body = {**_schema_body(), "stop": ["}"]}
    tool = completion(None, "tool_calls", [{"id": "c", "function": {"name": "x"}}])
    assert server.finish_completion(tool, body) == tool
    cut = server.finish_completion(completion('{"answer": 5', "length"), _schema_body())
    assert cut["choices"][0]["message"]["content"] == '{"answer": 5'
    stopped = server.finish_completion(completion("alpha STOP beta", "length"),
                                       {"stop": "STOP"})
    assert stopped["choices"][0] == {"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "alpha "}}


def test_stop_inside_a_length_cut_is_not_validated():
    """max_tokens cut + a stop hit inside it + a schema: the answer was cut
    short twice over, so it is returned as cut (finish "stop": the model hit
    the stop before the cap), not 502'd for failing to validate."""
    out = server.finish_completion(
        {"choices": [{"index": 0, "finish_reason": "length",
                      "message": {"role": "assistant", "content": '{"answer": "fi|ve'}}]},
        {**_schema_body(), "stop": "|"})
    assert out["choices"][0]["message"]["content"] == '{"answer": "fi', out
    assert out["choices"][0]["finish_reason"] == "stop", out


def test_malformed_stop_is_refused_before_the_cli_runs():
    for good in (None, "x", ["a", "b"], []):
        server.validate_stop({"stop": good})
    for bad in (5, ["a", 5], {"a": 1}):
        status, detail = _status(server.validate_stop, {"stop": bad})
        assert status == 400 and detail["error"]["param"] == "stop", bad
    calls = []

    async def never(*a, **k):
        calls.append(a)
        return {"result": "x", "usage": {}}
    saved = server.invoke
    server.invoke = never
    try:
        asyncio.run(server._handle_chat({"model": "m", "stop": 5,
                                         "messages": [{"role": "user", "content": "hi"}]}))
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("stop: 5 accepted")
    finally:
        server.invoke = saved
    assert calls == [], "the CLI ran for a request that was going to be refused"


def test_claude_images_keep_native_schema():
    """Claude's media ride stream-json stdin on the ordinary argv, and
    --json-schema beside them is proven on the pinned CLI
    (test_lockdown_claude.test_claude_json_schema_with_inline_media), so an
    image request is enforced natively too, not just instructed."""
    fmt = server.response_schema(_schema_body())
    with _Provider("claude"):
        assert server.native_schema(fmt)
        assert server.native_schema(fmt, images=True)
        assert server.schema_instruction(fmt, server.native_schema(fmt, True)) == ""


def test_claude_schema_keeps_web_turns():
    """schema_args' --max-turns comes after claude_web_args' in the argv and
    wins, so with web it must not cut the search-and-answer turns."""
    fmt = server.response_schema(_schema_body())
    with _Provider("claude"):
        with server.schema_args(fmt, web=True) as args:
            assert int(args[args.index("--max-turns") + 1]) > 4, args
        with server.schema_args(fmt) as args:
            assert args[args.index("--max-turns") + 1] == "3", args


def test_merge_completions_reindexes_and_sums_usage():
    one = {"id": "a", "model": "m", "choices": [{"index": 0, "message": {"content": "x"},
                                                "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
                     "prompt_tokens_details": {"cached_tokens": 2}}}
    merged = server.merge_completions([one, json.loads(json.dumps(one))])
    assert [c["index"] for c in merged["choices"]] == [0, 1], merged
    assert merged["usage"] == {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8,
                               "prompt_tokens_details": {"cached_tokens": 4}}, merged


def test_native_schema_flags_per_cli():
    """claude --json-schema (and turns to use its StructuredOutput tool);
    codex --output-schema FILE for a strict schema only, the file removed
    after; opencode has no flag and gets the instruction instead."""
    strict = server.response_schema(_schema_body())
    loose = server.response_schema(_schema_body(strict=False))
    with _Provider("claude"):
        with server.schema_args(strict) as args:
            assert args[:2] == ["--json-schema", json.dumps(SUM)], args
            assert args[args.index("--max-turns") + 1] == "3", args
        assert server.schema_instruction(strict, server.native_schema(strict)) == ""
    with _Provider("codex"):
        with server.schema_args(strict) as args:
            path = Path(args[args.index("--output-schema") + 1])
            assert json.loads(path.read_text()) == SUM
        assert not path.exists(), "schema file leaked"
        with server.schema_args(loose) as args:
            assert args == [], args
        assert "JSON Schema" in server.schema_instruction(loose, server.native_schema(loose))
    with _Provider("opencode"):
        with server.schema_args(strict) as args:
            assert args == [], args
        note = server.schema_instruction(strict, server.native_schema(strict))
        assert json.dumps(SUM) in note, note


def _run_chat(body: dict, answers: list[str]):
    """Drive _handle_chat with invoke stubbed; return (result, calls)."""
    calls: list[tuple] = []
    saved = server.invoke

    async def fake_invoke(prompt, system, model, image_paths=None, fmt=None, *,
                          web=False, effort=None, thinking=None):
        calls.append((prompt, fmt))
        return {"result": answers[len(calls) - 1], "usage": {"input_tokens": 2,
                                                              "output_tokens": 1}}
    server.invoke = fake_invoke
    try:
        result = asyncio.run(server._handle_chat(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], **body}))
    finally:
        server.invoke = saved
    return result, calls


def test_text_path_fans_out_n_and_applies_stop():
    """n runs one after another, each on its own gate slot: a lane with
    concurrency 1 serves it too instead of 429ing against itself."""
    with _Provider("opencode"):
        result, calls = _run_chat({"n": 2, "stop": ["STOP"]},
                                  ["one STOP x", "two"])
    assert len(calls) == 2, calls
    assert [c["message"]["content"] for c in result["choices"]] == ["one ", "two"], result
    assert [c["index"] for c in result["choices"]] == [0, 1]
    assert result["usage"]["completion_tokens"] == 2, result["usage"]


def test_text_path_schema_instructed_and_validated():
    with _Provider("opencode"):
        result, calls = _run_chat(_schema_body(), ['{"answer": 5}'])
        assert "JSON Schema" in calls[0][0], calls[0][0]
        assert calls[0][1] is None, "opencode has no native schema flag"
        assert result["choices"][0]["message"]["content"] == '{"answer": 5}'
        try:
            _run_chat(_schema_body(), ["It is 5."])
        except HTTPException as exc:
            assert exc.status_code == 502, exc.detail
        else:
            raise AssertionError("prose accepted for a schema")
    with _Provider("claude"):
        _result, calls = _run_chat(_schema_body(), ['{"answer": 5}'])
        assert "JSON Schema" not in calls[0][0], "native schema still instructed"
        assert calls[0][1]["schema"] == SUM, "claude's native flag not requested"


def test_streamed_n_emits_every_choice():
    result = {"id": "x", "created": 0, "usage": {}, "choices": [
        {"index": 0, "message": {"content": "a"}, "finish_reason": "stop"},
        {"index": 1, "message": {"content": "b"}, "finish_reason": "stop"}]}

    async def collect():
        return [c async for c in server.sse_from_completion(result, "m")]
    chunks = [json.loads(c[6:]) for c in asyncio.run(collect()) if c != "data: [DONE]\n\n"]
    deltas = {ch["index"]: ch["delta"].get("content")
              for c in chunks for ch in c["choices"] if ch["delta"]}
    assert deltas == {0: "a", 1: "b"}, chunks
    assert [ch["index"] for ch in chunks[-1]["choices"]] == [0, 1], chunks[-1]


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
