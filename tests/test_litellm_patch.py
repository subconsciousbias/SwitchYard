"""Regression tests for the Dockerfile.gateway build-time patch on
litellm.litellm_core_utils.token_counter._format_type.

Issue #59: litellm's `_format_type` was written against the OpenAI tool-schema
shape, which requires `items` on every `type: array` property. The Anthropic
tool spec does not require `items` — `{"type": "array"}` is a complete schema
on its own. So every legitimate Anthropic `tools=[...]` payload hit a bare
subscript and raised `KeyError: 'items'` on first use, taking token_counter
down with a stack trace and silently breaking quota accounting on every
request that carried tools.

Dockerfile.gateway patches the installed litellm at build time, replacing
`props['items']` with a tolerant `props.get('items') or {'type': 'string'}`,
so an items-less array counts as `string[]` instead of raising. switchyard/
selfcheck.py runs `_audit_token_counter_patch` on every gateway start and
exits CRITICAL if the patch is missing or broken. These four offline tests
guard both ends of that contract: the Dockerfile actually contains the
patch block, the patched `_format_type` tolerates items-less arrays, the
unpatched `_format_type` still raises (so a lost patch would be caught),
and `selfcheck.py` wires the audit into the static gate.

Pure string/file manipulation; no litellm import. The litellm package only
ships inside the gateway image, and the test runner does not need it.
"""
from __future__ import annotations

import ast
import os

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


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))