"""Regression tests for the Dockerfile.gateway build-time patches on
litellm.litellm_core_utils.token_counter._format_type and on
litellm.llms.openai.responses.count_tokens.token_counter.OpenAITokenCounter.

Issue #59: litellm's `_format_type` was written against the OpenAI tool-schema
shape, which requires `items` on every `type: array` property. The Anthropic
tool spec does not require `items` — `{"type": "array"}` is a complete schema
on its own. So every legitimate Anthropic `tools=[...]` payload hit a bare
subscript and raised `KeyError: 'items'` on first use, taking token_counter
down with a stack trace and silently breaking quota accounting on every
request that carried tools.

Issue #198: litellm's `OpenAITokenCounter.count_tokens` always asks the
vendor's /v1/responses/input_tokens endpoint for an exact count. Switchyard's
`*-sidecar` plans (claude-max-sidecar, codex-sidecar, opencode-go-sidecar,
...) are internal CLI bridges that expose only /usage, /health, /v1/models
and /v1/chat/completions — none of them speak the Responses API input-tokens
route, so the call 404s and the upstream CountTokens handler logs an
`HTTP error in CountTokens handler ... 404` ERROR line per call (the
evidence issue #198 cites: 96 ERROR lines in ~6h on one plan). The fix is
to short-circuit the call for any api_base whose hostname ends in "-sidecar"
(the network-alias suffix docker-compose.yml uses for every `*-sidecar`
service the gateway dials), so real vendor lanes (api.openai.com, api.x.ai,
api.z.ai, ...) still get the real vendor count endpoint unchanged.

Dockerfile.gateway patches both files at build time: the _format_type
patch replaces `props['items']` with a tolerant
`props.get('items') or {'type': 'string'}`, and the OpenAITokenCounter
patch appends an api_base short-circuit after the
`api_base: Final = litellm_params.get("api_base")` assignment. switchyard/
selfcheck.py runs `_audit_token_counter_patch` AND
`_audit_count_tokens_sidecar_patch` on every gateway start (both wired
from `_static_audit()`) and exits CRITICAL if either patch is missing or
broken. These tests guard both ends of that contract: the Dockerfile
actually contains both patch blocks, each patched function does the right
thing, each unpatched function still shows the original bug (so a lost
patch would be caught), `selfcheck.py` wires both audits into the static
gate, and the test constants match the Dockerfile literals byte-for-byte.

Pure string/file manipulation; no litellm import. The litellm package only
ships inside the gateway image, and the test runner does not need it.
"""
from __future__ import annotations

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `import conftest` resolves under plain `python3`

import conftest  # noqa: F401  (socket guard for plain-script mode)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# The exact target line Dockerfile.gateway patches out, copied verbatim from
# the Dockerfile's `vulnerable` literal. Pinned together with the replacement
# so a drift between the Dockerfile and this file is a loud signal: if the
# PIN POLICY at the top of Dockerfile.gateway is followed and a future
# litellm bump re-derives the patch, both strings change together.
#
# These are raw-bytes substrings of the Dockerfile as written to disk, so the
# Python literal encodes the `\"` escapes the Dockerfile uses to embed a
# double quote inside a double-quoted Python string.
_OLD_LINE = '        return f\\"{_format_type(props[\'items\'], indent)}[]\\"'
# The tolerant replacement, as the Dockerfile writes it: two adjacent Python
# string literals whose concatenation IS the patched line. We pin both halves
# rather than the concatenated form because the continuation line's leading
# whitespace is cosmetic — it must realign with the opening parenthesis of
# `patched = (` if the file is reformatted, and the test should fail only on
# a real regression, not on a reindent.
_NEW_LEFT = '        return f\\"{_format_type(props.get(\'items\') or '
_NEW_RIGHT = '{\'type\': \'string\'}, indent)}[]\\"'

# The same OLD/NEW pair, in the form they take inside `_PINNED_FORMAT_TYPE_
# SOURCE` below: that fixture is plain Python source (no Dockerfile-style
# `\"` escapes, because `_format_type` writes its f-strings with plain `"`)
# so the replace we run on it uses the unescaped form. Two constants let the
# Dockerfile-file assertion and the source-replace stay readable.
_OLD_LINE_FOR_FIXTURE = '        return f"{_format_type(props[\'items\'], indent)}[]"'
_NEW_LINE_FOR_FIXTURE = (
    '        return f"{_format_type(props.get(\'items\') or '
    '{\'type\': \'string\'}, indent)}[]"'
)


# Pinned v1.101.0 `_format_type` source, taken verbatim from
# litellm/litellm_core_utils/token_counter.py at the digest Dockerfile.gateway
# pins. Only the lines whose shape determines the patch's correctness are
# preserved; the rest is collapsed to a stub because the test cases do not
# exercise them. The `array` branch's `return` line is preserved byte-for-byte
# so the Dockerfile's `replace(vulnerable, patched, 1)` call would land on the
# same string in this fixture.
_PINNED_FORMAT_TYPE_SOURCE = '''def _format_object_parameters(parameters, indent):
    return ""  # trivial stub: the test never exercises the object branch


def _format_type(props, indent):
    type = props.get("type")
    if type == "string":
        if "enum" in props:
            return " | ".join([f'"{item}"' for item in props["enum"]])
        return "string"
    elif type == "array":
        # items is required, OpenAI throws an error if it's missing
        return f"{_format_type(props['items'], indent)}[]"
    elif type == "object":
        return ""  # not exercised by these test cases
    elif type in ["integer", "number"]:
        if "enum" in props:
            return " | ".join([f'"{item}"' for item in props["enum"]])
        return "number"
    elif type == "boolean":
        return "boolean"
    elif type == "null":
        return "null"
    else:
        # This is a guess, as an empty string doesn't yield the expected token count
        return "any"
'''


def test_dockerfile_contains_the_patch_block():
    """Dockerfile.gateway still performs the OLD->NEW substitution on
    `_format_type`. A regression that deletes the patch block (or silently
    changes the target string) returns every Anthropic `tools=[...]` payload
    to the `KeyError` behaviour — the same bug, without a commit message.
    Fail loudly here, before the running image has a chance to surface it.
    """
    path = os.path.join(ROOT, "Dockerfile.gateway")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    assert _OLD_LINE in src, (
        f"Dockerfile.gateway no longer contains the vulnerable fragment "
        f"{_OLD_LINE!r}; the patch block was removed or the pinned litellm "
        f"has drifted past it."
    )
    assert _NEW_LEFT in src, (
        f"Dockerfile.gateway is missing the first half of the patched "
        f"replacement {(_NEW_LEFT + _NEW_RIGHT)!r}; the patch block was "
        f"truncated."
    )
    assert _NEW_RIGHT in src, (
        f"Dockerfile.gateway is missing the second half of the patched "
        f"replacement {(_NEW_LEFT + _NEW_RIGHT)!r}; the patch block was "
        f"truncated."
    )
    print("  Dockerfile.gateway: vulnerable fragment AND both halves of "
          "the tolerant replacement are present (patch block intact)")


def _exec_format_type(source: str):
    """exec() a source string defining `_format_type`, return the callable."""
    namespace: dict = {}
    exec(compile(source, "<pinned-format-type-fixture>", "exec"), namespace)
    return namespace["_format_type"]


def test_patched_format_type_handles_itemsless_array():
    """After the Dockerfile.gateway substitution, `_format_type`:

    - does NOT raise on `{"type": "array"}` (Anthropic-legal, no `items`);
    - renders it as `string[]` (the synthetic fallback the patch picks when
      `items` is absent);
    - STILL renders items-carrying arrays identically to the unpatched
      behaviour — the only thing the patch changes is the missing-`items`
      case.

    This is the count the bridge and the quota accounting both depend on:
    an items-less array must come back as a string the token-counter can
    price, not a stack trace that takes the whole counter down.
    """
    patched_src = _PINNED_FORMAT_TYPE_SOURCE.replace(_OLD_LINE_FOR_FIXTURE, _NEW_LINE_FOR_FIXTURE, 1)
    assert patched_src != _PINNED_FORMAT_TYPE_SOURCE, (
        "fixture replace() was a no-op — the source no longer contains the "
        "OLD line the Dockerfile patch is supposed to replace."
    )

    patched = _exec_format_type(patched_src)

    itemsless = patched({"type": "array"}, 0)
    assert itemsless == "string[]", itemsless

    # items=integer renders as `number[]` on the pinned litellm: the
    # `type in [integer, number]` branch returns `number` regardless of
    # which one was specified. The literal value the task spec mentions
    # (`integer[]`) is what an Anthropic-flavoured counter would emit; on
    # the actual pinned upstream, the integer and number branches share
    # one return — a test that asked for `integer[]` would fail against
    # the real behaviour. The point of this assertion is "well-formed
    # arrays render the same as they did unpatched", and the well-formed
    # rendering is `number[]`.
    with_items = patched({"type": "array", "items": {"type": "integer"}}, 0)
    assert with_items == "number[]", with_items

    print(f"  patched: items-less array -> {itemsless!r} (no raise); "
          f"items=integer -> {with_items!r} (well-formed path preserved)")


def test_unpatched_format_type_raises_on_itemsless_array():
    """The pinned (unpatched) `_format_type` raises KeyError on items-less
    arrays. This is what the patch exists to fix — a test that proves the
    patched version no longer raises would be a tautology without a control:
    without the patch, the source itself raises on this very input.
    """
    unpatched = _exec_format_type(_PINNED_FORMAT_TYPE_SOURCE)
    try:
        unpatched({"type": "array"}, 0)
    except KeyError as exc:
        assert exc.args == ("items",), exc.args
        print(f"  unpatched: items-less array raises KeyError{exc.args} "
              "(patch would have caught this; a lost patch fails loudly)")
        return
    raise AssertionError(
        "unpatched _format_type did NOT raise on items-less array - the "
        "fixture is stale, the litellm upstream has been fixed upstream "
        "(in which case the Dockerfile patch is obsolete and the test "
        "should be retired), or the test is missing its point."
    )


def test_selfcheck_wires_audit_token_counter_patch():
    """switchyard/selfcheck.py defines `_audit_token_counter_patch` AND calls
    it from `_static_audit()`. If a future edit removes the function
    definition or unhooks it from the static gate, the build-time patch
    would silently stop being audited at startup and a drift would not be
    caught. We assert both halves via a substring + AST check; no litellm
    import.
    """
    path = os.path.join(ROOT, "switchyard", "selfcheck.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    assert "def _audit_token_counter_patch" in src, (
        "switchyard/selfcheck.py no longer defines "
        "_audit_token_counter_patch - the token-counter audit was dropped."
    )

    tree = ast.parse(src)

    found_def = any(
        isinstance(node, ast.FunctionDef)
        and node.name == "_audit_token_counter_patch"
        for node in ast.walk(tree)
    )
    assert found_def, (
        "_audit_token_counter_patch is present as text but does not parse "
        "as a function definition."
    )

    static_audit = next(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.FunctionDef) and node.name == "_static_audit"),
        None,
    )
    assert static_audit is not None, (
        "_static_audit() is missing from switchyard/selfcheck.py - the "
        "startup gate itself has been removed."
    )

    called = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_audit_token_counter_patch"
        for node in ast.walk(static_audit)
    )
    assert called, (
        "_audit_token_counter_patch is defined but NOT called from "
        "_static_audit(); the build-time patch would silently stop being "
        "audited at startup, and a litellm-pin drift would not be caught."
    )

    print("  selfcheck.py: _audit_token_counter_patch defined and called "
          "from _static_audit()")


def test_selfcheck_wires_audit_count_tokens_patch():
    """switchyard/selfcheck.py defines `_audit_count_tokens_sidecar_patch`
    AND calls it from `_static_audit()`. Mirror of
    `test_selfcheck_wires_audit_token_counter_patch` for the sibling
    sidecar-patch (issue #198): the protection contract the Dockerfile
    block claims is "sibling of the _format_type patch", which means
    "Dockerfile build-time patch + offline tests + a `_static_audit`
    audit + an offline test that pins the wiring" - this test is the
    last of those four. A regression that drops the audit or unhooks
    it from the static gate silently reinstates the 404 ERROR log
    lines that issue #198 closes, with no startup-time CRITICAL to
    surface them.
    We assert both halves via a substring + AST check; no litellm
    import.
    """
    path = os.path.join(ROOT, "switchyard", "selfcheck.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    assert "def _audit_count_tokens_sidecar_patch" in src, (
        "switchyard/selfcheck.py no longer defines "
        "_audit_count_tokens_sidecar_patch - the count_tokens sidecar "
        "audit was dropped. The Dockerfile patch is still there, but "
        "without a runtime check a pin drift would silently bring the "
        "404 / 'token counting failed' behaviour back."
    )

    tree = ast.parse(src)

    found_def = any(
        isinstance(node, ast.FunctionDef)
        and node.name == "_audit_count_tokens_sidecar_patch"
        for node in ast.walk(tree)
    )
    assert found_def, (
        "_audit_count_tokens_sidecar_patch is present as text but does "
        "not parse as a function definition."
    )

    static_audit = next(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.FunctionDef) and node.name == "_static_audit"),
        None,
    )
    assert static_audit is not None, (
        "_static_audit() is missing from switchyard/selfcheck.py - the "
        "startup gate itself has been removed."
    )

    called = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_audit_count_tokens_sidecar_patch"
        for node in ast.walk(static_audit)
    )
    assert called, (
        "_audit_count_tokens_sidecar_patch is defined but NOT called from "
        "_static_audit(); the sidecar build-time patch would silently "
        "stop being audited at startup, and a litellm-pin drift would "
        "not be caught."
    )

    print("  selfcheck.py: _audit_count_tokens_sidecar_patch defined and "
          "called from _static_audit()")


# ---- Issue #198: sidecar api_base short-circuit ------------------------
#
# Same shape as the _format_type constants above: a Dockerfile-escaped form
# that pins the literal bytes Dockerfile.gateway's `anchor = "..."` and
# `patched = (...)` heredoc contains, and an unescaped form that pins the
# bytes that end up inside the patched token_counter.py file (and in this
# test's source fixture, which is plain Python source without Dockerfile-
# style `\"` escapes).

# The exact target line Dockerfile.gateway patches against, copied verbatim
# from the Dockerfile's `anchor` literal. Pinned together with the
# replacement so a drift between the Dockerfile and this file is a loud
# signal: if the PIN POLICY at the top of Dockerfile.gateway is followed
# and a future litellm bump re-derives the patch, both strings change
# together.
_ANCHOR_LINE = '        api_base: Final = litellm_params.get(\\"api_base\\")'

# The first two replacement lines as they appear inside the Dockerfile's
# `patched = (...)` heredoc - i.e., the contents of the double-quoted
# Python string literals that concatenate into the patched lines that
# land in token_counter.py.
_PATCHED_LINE_1 = (
    '        if (isinstance(api_base, str) and api_base.split(\\"//\\")'
)
_PATCHED_LINE_2 = (
    '            [-1].split(\\"/\\")[0].split(\\":\\")[0].endswith(\\"-sidecar\\")):'
)

# The third replacement line is split across two adjacent Python string
# literals in the Dockerfile (their concatenation IS the patched line).
# Pin both halves for the same reason as the _NEW_LEFT/_NEW_RIGHT pair
# in the _format_type regression above: a reindent that re-aligns the
# continuation line with the opening parenthesis of `patched = (` is
# cosmetic and should not fail the test.
_PATCHED_LINE_3_LEFT = (
    '            return None  # switchyard: *-sidecar plans have '
)
_PATCHED_LINE_3_RIGHT = (
    'no /v1/responses/input_tokens route; count locally (issue #198)'
)

# The same anchor and replacement lines, in the form they take inside
# this test's source fixture (`_PINNED_COUNT_TOKENS_SOURCE`): that
# fixture is plain Python source (no Dockerfile-style `\"` escapes,
# because token_counter.py writes its strings with plain `"`). Two
# constant sets keep the Dockerfile-file assertion and the source-
# replace readable.
_ANCHOR_LINE_FOR_FIXTURE = (
    '        api_base: Final = litellm_params.get("api_base")'
)
_PATCHED_LINE_1_FOR_FIXTURE = (
    '        if (isinstance(api_base, str) and api_base.split("//")'
)
_PATCHED_LINE_2_FOR_FIXTURE = (
    '            [-1].split("/")[0].split(":")[0].endswith("-sidecar")):'
)
_PATCHED_LINE_3_FOR_FIXTURE = (
    '            return None  # switchyard: *-sidecar plans have '
    'no /v1/responses/input_tokens route; count locally (issue #198)'
)


# Pinned v1.101.0 fragment of `OpenAITokenCounter.count_tokens`, taken
# verbatim from litellm/llms/openai/responses/count_tokens/token_counter.py
# at the digest Dockerfile.gateway pins. Only the lines whose shape
# determines the patch's correctness are preserved; the rest of the
# upstream method (handler call, TokenCountResponse shaping, OpenAIError
# mapping) is collapsed to a sentinel return so the test does not need
# to stand up a litellm handler or an HTTP client. The `api_base`
# assignment line is preserved byte-for-byte so the Dockerfile's
# `replace(anchor, patched, 1)` call lands on the same string in this
# fixture.
_PINNED_COUNT_TOKENS_SOURCE = '''from typing import Final


class OpenAITokenCounter:
    """Stub of litellm.llms.openai.responses.count_tokens.OpenAITokenCounter
    at the digest Dockerfile.gateway pins. The class structure and method
    signature mirror the upstream so the anchor line (the api_base
    assignment) sits at the same indent it does in token_counter.py; the
    method body is collapsed to a sentinel return so the test does not
    need to stand up a litellm handler or an HTTP client.

    The sentinel return lets the test tell which path the function took:
    returning None means the api_base short-circuit fired (patch working),
    returning ("vendor", api_base) means the function fell through to
    the (stubbed) vendor path (either the api_base is a real vendor, or
    the patch was lost).
    """

    async def count_tokens(self, model_to_use, messages, contents,
                           deployment=None, request_model="", tools=None,
                           system=None):
        if not messages:
            return None
        deployment = deployment or {}
        litellm_params: Final = deployment.get("litellm_params", {})
        api_base: Final = litellm_params.get("api_base")
        return ("vendor", api_base)
'''


def _exec_count_tokens(source: str):
    """exec() a source string defining `OpenAITokenCounter`, return the
    class. Tests then instantiate it and call `.count_tokens(...)`.
    """
    namespace: dict = {}
    exec(compile(source, "<pinned-count-tokens-fixture>", "exec"), namespace)
    return namespace["OpenAITokenCounter"]


def test_dockerfile_contains_the_sidecar_patch_block():
    """Dockerfile.gateway still performs the api_base short-circuit
    substitution on OpenAITokenCounter.count_tokens (issue #198). A
    regression that deletes the patch block (or silently changes the
    target string) returns every `*-sidecar` plan to the 404 behaviour -
    the same bug, without a commit message. Fail loudly here, before
    the running image has a chance to surface it.
    """
    path = os.path.join(ROOT, "Dockerfile.gateway")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    assert _ANCHOR_LINE in src, (
        f"Dockerfile.gateway no longer contains the count_tokens anchor "
        f"{_ANCHOR_LINE!r}; the sidecar patch block was removed or the "
        f"pinned litellm has drifted past it."
    )
    assert _PATCHED_LINE_1 in src, (
        f"Dockerfile.gateway is missing the first line of the sidecar "
        f"replacement {_PATCHED_LINE_1!r}; the patch block was truncated."
    )
    assert _PATCHED_LINE_2 in src, (
        f"Dockerfile.gateway is missing the second line of the sidecar "
        f"replacement {_PATCHED_LINE_2!r}; the patch block was truncated."
    )
    assert _PATCHED_LINE_3_LEFT in src, (
        f"Dockerfile.gateway is missing the first half of the sidecar "
        f"return-line {_PATCHED_LINE_3_LEFT!r}; the patch block was "
        f"truncated."
    )
    assert _PATCHED_LINE_3_RIGHT in src, (
        f"Dockerfile.gateway is missing the second half of the sidecar "
        f"return-line {_PATCHED_LINE_3_RIGHT!r}; the patch block was "
        f"truncated."
    )
    print("  Dockerfile.gateway: sidecar anchor AND all four lines of "
          "the replacement are present (sidecar short-circuit intact)")


def test_patched_count_tokens_short_circuits_on_sidecar_api_base():
    """After the Dockerfile.gateway substitution, `count_tokens`:

    - returns None for an api_base whose hostname ends with "-sidecar"
      (the patch short-circuits the vendor /v1/responses/input_tokens
      call so the gateway falls back to its local tokenizer);
    - PROCEEDS for an api_base like https://api.x.ai/v1 (real vendor
      lane, the endswith check is False, the function returns the
      sentinel ("vendor", api_base) tuple);
    - PROCEEDS for an absent api_base (the if-condition guards on
      isinstance(api_base, str), so a None api_base falls through to
      the vendor path; without that guard, None.split(...) would raise
      AttributeError at first call and take the counter down for every
      real vendor that did not configure an explicit api_base).

    The hostname parsing - split("//")[-1].split("/")[0].split(":")[0] -
    must tolerate the URL forms docker-compose.yml generates (e.g.
    http://claude-max-sidecar:8081/v1). The control tests below exercise
    each branch.
    """
    import asyncio
    patched = (
        _ANCHOR_LINE_FOR_FIXTURE + "\n"
        + _PATCHED_LINE_1_FOR_FIXTURE + "\n"
        + _PATCHED_LINE_2_FOR_FIXTURE + "\n"
        + _PATCHED_LINE_3_FOR_FIXTURE
    )
    patched_src = _PINNED_COUNT_TOKENS_SOURCE.replace(
        _ANCHOR_LINE_FOR_FIXTURE, patched, 1
    )
    assert patched_src != _PINNED_COUNT_TOKENS_SOURCE, (
        "fixture replace() was a no-op - the source no longer contains "
        "the anchor line the Dockerfile patch is supposed to replace."
    )

    OpenAITokenCounter = _exec_count_tokens(patched_src)
    obj = OpenAITokenCounter()
    msgs = [{"role": "user", "content": "hi"}]

    # Sidecar api_base -> short-circuits to None.
    sidecar = asyncio.run(obj.count_tokens(
        "m", msgs, None,
        deployment={"litellm_params": {"api_base": "http://claude-max-sidecar:8081/v1"}},
    ))
    assert sidecar is None, (
        f"sidecar api_base should have short-circuited to None; got "
        f"{sidecar!r} - the patch is not landing on the api_base "
        f"assignment line."
    )

    # codex-sidecar -> also short-circuits (same suffix).
    codex = asyncio.run(obj.count_tokens(
        "m", msgs, None,
        deployment={"litellm_params": {"api_base": "http://codex-sidecar:8082/v1"}},
    ))
    assert codex is None, codex

    # Real vendor lane -> proceeds to the (stubbed) vendor path.
    vendor = asyncio.run(obj.count_tokens(
        "m", msgs, None,
        deployment={"litellm_params": {"api_base": "https://api.x.ai/v1"}},
    ))
    assert vendor == ("vendor", "https://api.x.ai/v1"), (
        f"vendor api_base should have proceeded; got {vendor!r} - the "
        f"endswith('-sidecar') check is incorrectly matching a real "
        f"vendor hostname."
    )

    # api.openai.com (no /v1 suffix) -> proceeds.
    openai = asyncio.run(obj.count_tokens(
        "m", msgs, None,
        deployment={"litellm_params": {"api_base": "https://api.openai.com/v1"}},
    ))
    assert openai == ("vendor", "https://api.openai.com/v1"), openai

    # Absent api_base -> proceeds, returns ("vendor", None). This is
    # the guard that the isinstance(api_base, str) check provides: a
    # None api_base would otherwise raise AttributeError on .split.
    absent = asyncio.run(obj.count_tokens(
        "m", msgs, None,
        deployment={"litellm_params": {}},
    ))
    assert absent == ("vendor", None), (
        f"absent api_base should have proceeded; got {absent!r} - the "
        f"isinstance(api_base, str) guard is missing or incorrect."
    )

    print(f"  patched: *-sidecar api_base -> None (short-circuit); "
          f"vendor api_base -> {vendor!r}; absent api_base -> "
          f"{absent!r}")


def test_unpatched_count_tokens_still_forwards():
    """The pinned (unpatched) `count_tokens` forwards every api_base
    through to the vendor call - so an api_base like
    http://claude-max-sidecar:8081 returns ("vendor", "http://...")
    instead of None. This is the control: a test that proves the patched
    version short-circuits is a tautology without it. A lost patch
    would silently reinstating the 404 / "token counting failed"
    anomaly that issue #198 documents.
    """
    import asyncio
    OpenAITokenCounter = _exec_count_tokens(_PINNED_COUNT_TOKENS_SOURCE)
    obj = OpenAITokenCounter()
    msgs = [{"role": "user", "content": "hi"}]
    sidecar = asyncio.run(obj.count_tokens(
        "m", msgs, None,
        deployment={"litellm_params": {"api_base": "http://claude-max-sidecar:8081/v1"}},
    ))
    assert sidecar == ("vendor", "http://claude-max-sidecar:8081/v1"), (
        f"unpatched count_tokens should have forwarded the sidecar "
        f"api_base to the vendor path; got {sidecar!r}. Either the "
        f"fixture is stale or the pinned litellm upstream has been "
        f"fixed upstream (in which case the Dockerfile patch is "
        f"obsolete and this test should be retired)."
    )
    print(f"  unpatched: sidecar api_base -> {sidecar!r} "
          "(patch would have short-circuited this; a lost patch fails loudly)")


def test_anchor_strings_match_dockerfile_literals():
    """The anchor and replacement string constants in this file must
    match the literals in Dockerfile.gateway byte-for-byte. A drift
    between the two would mean the test is checking something the
    Dockerfile no longer contains - so even a green run would not
    catch a missing or replaced patch. Mirrors the same guard the
    existing _format_type tests carry for the OLD/NEW line pair.
    """
    with open(os.path.join(ROOT, "Dockerfile.gateway"), encoding="utf-8") as fh:
        dockerfile_src = fh.read()

    # The anchor literal: `anchor = "..."` in Dockerfile.gateway's
    # `python - <<'PY'` heredoc. The `\"` escapes are the Dockerfile's
    # source bytes; they are not the bytes that appear in the patched
    # token_counter.py file (those have plain `"`).
    assert _ANCHOR_LINE in dockerfile_src, (
        f"_ANCHOR_LINE {_ANCHOR_LINE!r} is not present in "
        f"Dockerfile.gateway; the test constant has drifted past the "
        f"literal it is supposed to pin."
    )

    # The first two replacement lines: each is the contents of one of
    # the double-quoted Python string literals in the Dockerfile's
    # `patched = (...)` block. They use the same Dockerfile-escaped
    # quote form as the anchor literal.
    for literal, label in (
        (_PATCHED_LINE_1, "first replacement line"),
        (_PATCHED_LINE_2, "second replacement line"),
    ):
        assert literal in dockerfile_src, (
            f"test constant for the {label} {literal!r} is not present "
            f"in Dockerfile.gateway; the test constant has drifted past "
            f"the literal it is supposed to pin."
        )

    # The third replacement line is split across two adjacent Python
    # string literals in the Dockerfile (their concatenation IS the
    # patched line). Each half must appear verbatim in the Dockerfile
    # source.
    for literal, label in (
        (_PATCHED_LINE_3_LEFT, "third replacement line (left half)"),
        (_PATCHED_LINE_3_RIGHT, "third replacement line (right half)"),
    ):
        assert literal in dockerfile_src, (
            f"test constant for the {label} {literal!r} is not present "
            f"in Dockerfile.gateway; the test constant has drifted past "
            f"the literal it is supposed to pin."
        )

    # The fixture-form anchor must appear in the unpatched fixture
    # source - that is the line test (b) and (c) replace against. The
    # fixture-form replacement lines do not appear in the unpatched
    # fixture (they are inserted by the replace() call); they are
    # verified by test (b) end-to-end via the actual patched source
    # that the test exec()s.
    assert _ANCHOR_LINE_FOR_FIXTURE in _PINNED_COUNT_TOKENS_SOURCE, (
        f"_ANCHOR_LINE_FOR_FIXTURE {_ANCHOR_LINE_FOR_FIXTURE!r} is not "
        f"present in the unpatched fixture; the fixture has drifted "
        f"out of sync with the FOR_FIXTURE constants."
    )

    # The fixture-form constants must NOT contain backslashes - they
    # are the plain Python source form, and a stray `\\"` here would
    # silently change the post-replace fixture shape. This catches a
    # copy-paste error between the Dockerfile-escaped and fixture
    # forms of the same constant.
    for name, val in (
        ("_ANCHOR_LINE_FOR_FIXTURE", _ANCHOR_LINE_FOR_FIXTURE),
        ("_PATCHED_LINE_1_FOR_FIXTURE", _PATCHED_LINE_1_FOR_FIXTURE),
        ("_PATCHED_LINE_2_FOR_FIXTURE", _PATCHED_LINE_2_FOR_FIXTURE),
        ("_PATCHED_LINE_3_FOR_FIXTURE", _PATCHED_LINE_3_FOR_FIXTURE),
    ):
        assert "\\" not in val, (
            f"fixture-form constant {name} {val!r} unexpectedly "
            f"contains a backslash; it should be the unescaped form."
        )

    print("  test constants match Dockerfile.gateway's anchor and "
          "patched literals byte-for-byte (no silent drift possible)")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
