"""Tests for `switchyard.session.derive()` — the lane-aware fallback order.

This file is the regression net for the `derive()` preference list:

  1. `x-switchyard-session` (and friends) on the inbound request header;
  2. `metadata.session_id` / `metadata.switchyard_session` /
     `metadata.conversation_id` (body-shaped session id);
  3. `metadata.user_id` (Claude Code's per-session UUID, prefixed `u:`);
  4. `data["user"]` / `data["prompt_cache_key"]` (Codex / OpenAI's
     equivalent client identity fields, prefixed `u:`);
  5. `metadata.litellm_trace_id` (LiteLLM's trace id, prefixed `t:`);
  6. the conversation-prefix fingerprint, suffixed with the requested
     lane so the same prefix on `forge` and on `judge` derives two
     different sessions.

The fingerprint is the load-bearing change in this round: it now walks
the first eight messages (was the first two), hashes `role` as well as
content, and recognises image data-URLs (hashed by their prefix +
length) and `tool_use` blocks (hashed by their `id`) so an image-only
first message yields a stable, non-`None` fingerprint. Without that,
Claude Code sessions that open with a screenshot would derive a
`None` session and lose lane stickiness on the first turn.

The accept tests are the three from the change request:

  (a) same body, different `metadata.user_id` → different sessions, and
      the same `user_id` across turns stays stable;
  (b) same body, different `data["model"]` → different sessions;
  (c) image-only first message → stable non-`None` fingerprint,
      distinct from a different image or a different length.

The regression tests pin the existing behaviour that *did* work:

  - `x-switchyard-session` header still wins;
  - `metadata.session_id` still wins;
  - `metadata.litellm_trace_id` still wins;
  - `data["prompt_cache_key"]` and `data["user"]` are recognised and
      preferred over the fingerprint in that order;
  - an image-only first message combined with a text turn does not
      collapse to `None` (the pre-change behaviour was: list-content
      blocks other than `text` were ignored, so a vision-first turn
      was indistinguishable from an empty body).

The `__main__` block at the bottom is the plain-script runner entry
point. The mechanical guard in `tests/_runner.py` flags any new
`test_*` defined below it as a silent skip, so any new test here goes
above the runner line — see `tests/CLAUDE.md`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans_path import plans_path  # noqa: E402

# Assigned, not setdefault: an exported SWITCHYARD_PLANS pointing at
# someone's real config would otherwise silently become the fixture.
os.environ["SWITCHYARD_PLANS"] = plans_path()


from switchyard.session import derive as derive_session  # noqa: E402


# --- helpers ---------------------------------------------------------------


def _img_block(data_b64: str) -> dict:
    """An Anthropic-shape image block — `type: image`, `source.data` is the
    base64 payload."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data_b64},
    }


def _tool_use_block(tool_id: str, name: str = "Bash", input_block: dict | None = None) -> dict:
    """An Anthropic-shape tool_use block. `id` is the load-bearing field
    for fingerprint stability across turns of the same conversation."""
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": name,
        "input": input_block if input_block is not None else {"command": "ls"},
    }


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


# --- accept: (a) user_id ---------------------------------------------------


def test_user_id_separates_sessions_with_identical_bodies():
    """Two requests, same system prompt, same first user turn, but different
    `metadata.user_id` — they must derive two different sessions.

    The acceptance test the change request calls out: Claude Code threads
    its per-session UUID through `metadata.user_id` (the value carries a
    `_session_` marker that the shape-gate keys on), so two terminals that
    happen to be reading the same system prompt are not the same session.
    """
    msgs = [
        {"role": "system", "content": "You are Claude Code."},
        {"role": "user", "content": "list the repo"},
    ]
    a = derive_session({"model": "forge", "messages": msgs, "metadata": {
        "user_id": "user_account_uuid-001_session_session-aaa__longlived",
    }})
    b = derive_session({"model": "forge", "messages": msgs, "metadata": {
        "user_id": "user_account_uuid-001_session_session-bbb__longlived",
    }})
    assert a is not None and b is not None
    assert a != b, (a, b)
    # The prefix is the spec'd marker for this channel — a regression that
    # dropped the `u:` would still be a different string from any other
    # channel's prefix, so this assertion catches the silent-loss case.
    assert a.startswith("u:") and b.startswith("u:")
    print(f"  different user_id -> distinct u: sessions: {a!r} vs {b!r}")


def test_user_id_stable_across_turns():
    """Same `metadata.user_id` across two turns of the same conversation
    (different user content, same Claude Code session UUID) — derived
    session is stable, so the second turn lands on the same lane lease.

    The reverse half of acceptance test (a): user_id is a session key, not
    a request key, so it must dominate the message content.
    """
    base = {"model": "forge", "metadata": {
        "user_id": "user_account_uuid-001_session_session-cccc__longlived",
    }}
    turn1 = derive_session(dict(base, messages=[
        {"role": "system", "content": "You are Claude Code."},
        {"role": "user", "content": "first turn"},
    ]))
    turn2 = derive_session(dict(base, messages=[
        {"role": "system", "content": "You are Claude Code."},
        {"role": "user", "content": "second turn"},
    ]))
    assert turn1 is not None and turn2 is not None
    assert turn1 == turn2, (turn1, turn2)
    print(f"  same user_id across turns -> stable: {turn1!r}")


# --- accept: (b) lane -----------------------------------------------------


def test_lane_separates_sessions_with_identical_bodies():
    """Same body, but `data["model"]` is `forge` on one and `judge` on
    the other — they must derive two different sessions.

    The acceptance test the change request calls out: two Claude Code
    terminals reading the same project and asking the same first question
    must not collapse onto one lane's lease. The fingerprint is the same
    prefix on both; only the lane differs.
    """
    msgs = [
        {"role": "system", "content": "common system prompt"},
        {"role": "user", "content": "common first turn"},
    ]
    a = derive_session({"model": "forge", "messages": msgs})
    b = derive_session({"model": "judge", "messages": msgs})
    assert a is not None and b is not None
    assert a != b, (a, b)
    # Both are fingerprint-derived; the lane suffix is the only difference.
    assert a.endswith("|forge"), a
    assert b.endswith("|judge"), b
    # The same-prefix / different-lane split shows up as the shared `fp:`
    # portion followed by a different lane suffix.
    assert a.split("|")[0] == b.split("|")[0]
    print(f"  forge vs judge -> distinct: {a!r} vs {b!r}")


def test_fingerprint_lane_defaults_to_empty_string():
    """No `data["model"]` at all: fingerprint still derives, and the lane
    suffix is the empty string (the spec'd fallback), not the literal
    "None" — otherwise two requests with no lane would collide on the
    default but a request with `model=None` would land elsewhere.
    """
    msgs = [{"role": "user", "content": "hi"}]
    a = derive_session({"messages": msgs})
    b = derive_session({"messages": msgs})
    assert a is not None and a == b, (a, b)
    assert a.endswith("|"), a
    print(f"  no data['model'] -> lane suffix is empty: {a!r}")


# --- accept: (c) image-only first message ---------------------------------


def test_image_only_first_message_yields_stable_fingerprint():
    """First message contains only an image block (data-URL form): the
    fingerprint is non-None, stable across repeats of the same image,
    and distinct from a different image or a different length.

    The acceptance test the change request calls out: vision-first
    sessions must derive a stable identity. The pre-change behaviour was
    to ignore any block whose `type` was not `text`, so an image-only
    first turn was indistinguishable from no first turn at all and
    `derive()` returned `None`.
    """
    img_a = "data:image/png;base64," + ("A" * 100)
    msgs_a = [{"role": "user", "content": [_img_block(img_a)]}]

    fp_a_1 = derive_session({"model": "forge", "messages": msgs_a})
    fp_a_2 = derive_session({"model": "forge", "messages": msgs_a})
    assert fp_a_1 is not None, (fp_a_1, msgs_a)
    assert fp_a_1 == fp_a_2, (fp_a_1, fp_a_2)

    # A different image (same media type, different bytes, different
    # length) must derive a different fingerprint.
    img_b = "data:image/png;base64," + ("B" * 200)
    msgs_b = [{"role": "user", "content": [_img_block(img_b)]}]
    fp_b = derive_session({"model": "forge", "messages": msgs_b})
    assert fp_b is not None and fp_b != fp_a_1, (fp_a_1, fp_b)

    # Same image, same length: the spec'd "data-URL prefix / source length"
    # hash treats them as the same. (Two different payloads of identical
    # length would still match the prefix/length hash by design — that is
    # the documented behaviour for session-stickiness, not for forensics.)
    img_c = "data:image/png;base64," + ("A" * 100)
    msgs_c = [{"role": "user", "content": [_img_block(img_c)]}]
    fp_c = derive_session({"model": "forge", "messages": msgs_c})
    assert fp_c == fp_a_1, (fp_a_1, fp_c)

    print(f"  image-only: stable across repeats ({fp_a_1!r}), "
          f"distinct from longer image ({fp_b!r})")


def test_image_only_plus_text_collage_is_not_none():
    """First turn is a vision question: a text block "what is this?" and
    an image block. Both blocks contribute to the fingerprint, so the
    result is non-None — this is the regression check the spec calls
    out ("image-only + text collage no longer returns None").

    Pre-change, the image block was ignored; only the text "what is this?"
    was hashed. The combined fingerprint is still non-None in the old
    code because the text block kept it alive, but adding the image now
    means the same turn on two different images derives different
    sessions — which it did not before.
    """
    msgs = [{"role": "user", "content": [
        _text_block("what is this?"),
        _img_block("data:image/png;base64," + ("X" * 50)),
    ]}]
    fp = derive_session({"model": "forge", "messages": msgs})
    assert fp is not None, fp

    msgs2 = [{"role": "user", "content": [
        _text_block("what is this?"),
        _img_block("data:image/png;base64," + ("Y" * 50)),
    ]}]
    fp2 = derive_session({"model": "forge", "messages": msgs2})
    assert fp2 is not None
    # Same text, different image: distinct sessions now (was identical
    # before — the pre-change code skipped the image block).
    assert fp != fp2, (fp, fp2)
    print(f"  text+image collage -> non-None, image distinguishes: "
          f"{fp!r} vs {fp2!r}")


# --- regression: existing identifier paths unchanged ----------------------


def test_x_switchyard_session_header_still_wins():
    """The operator-set session header is the strongest signal — it
    predates every body-shaped id and must still short-circuit above
    `user_id`, `prompt_cache_key`, and the fingerprint.

    The fixture sets *every* body-shaped identifier below the header
    (`metadata.session_id`, `metadata.user_id`, `metadata.litellm_trace_id`,
    top-level `user` and `prompt_cache_key`) so a future reader can see
    that the header wins regardless of which body channel is also set.
    Only the header value itself decides the session."""
    data = {
        "model": "forge",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {
            "user_id": "uuid-xyz",
            "session_id": "session-abc",
            "litellm_trace_id": "trace-123",
        },
        "user": "client-u",
        "prompt_cache_key": "pck",
        "proxy_server_request": {"headers": {"x-switchyard-session": "header-id"}},
    }
    out = derive_session(data)
    assert out == "h:header-id", out
    print(f"  x-switchyard-session header -> h: prefix still wins: {out!r}")


def test_metadata_session_id_still_wins():
    """Body-shaped `metadata.session_id` is still second in the order,
    above `user_id` — preserves the behaviour the previous version
    shipped."""
    data = {
        "model": "forge",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"session_id": "session-abc", "user_id": "uuid-xyz"},
    }
    out = derive_session(data)
    assert out == "m:session-abc", out
    print(f"  metadata.session_id -> m: prefix still wins: {out!r}")


def test_metadata_switchyard_session_and_conversation_id_aliases():
    """`switchyard_session` and `conversation_id` are accepted aliases for
    `session_id`. The order is fixed — `session_id` first, then the two
    aliases — and the spec keeps that."""
    data = {"model": "forge", "messages": [{"role": "user", "content": "hi"}]}
    a = derive_session(dict(data, metadata={"switchyard_session": "ss-1"}))
    b = derive_session(dict(data, metadata={"conversation_id": "cid-1"}))
    assert a == "m:ss-1", a
    assert b == "m:cid-1", b
    # session_id beats the aliases
    c = derive_session(dict(data, metadata={
        "session_id": "sid-1", "switchyard_session": "ss-1",
        "conversation_id": "cid-1"}))
    assert c == "m:sid-1", c
    print("  session_id/switchyard_session/conversation_id aliases: all m: prefix")


def test_litellm_trace_id_still_recognised():
    """`metadata.litellm_trace_id` keeps its `t:` prefix and stays ahead
    of the fingerprint fallback. Preserves the spec'd order."""
    data = {
        "model": "forge",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"litellm_trace_id": "trace-123"},
    }
    out = derive_session(data)
    assert out == "t:trace-123", out
    print(f"  litellm_trace_id -> t: prefix still recognised: {out!r}")


def test_prompt_cache_key_preferred_over_fingerprint():
    """Codex / OpenAI's `prompt_cache_key` carries a session id — it is
    body-shaped, prefixed `u:<key>:<val>`, and must beat the
    conversation-prefix fingerprint. The spec puts `user` and
    `prompt_cache_key` in the same tier; `user` is checked first.

    The `u:` channel is scoped by API key (key_hash or `anon`) so two
    virtual keys claiming the same `user` / `prompt_cache_key` value do
    not share a lease — see `test_user_id_scoped_by_api_key`."""
    msgs = [{"role": "user", "content": "hi"}]
    fp_only = derive_session({"model": "forge", "messages": msgs})

    # prompt_cache_key alone
    out = derive_session({"model": "forge", "messages": msgs,
                          "prompt_cache_key": "pck-1"})
    assert out == "u:anon:pck-1", out
    assert out != fp_only, (out, fp_only)

    # user alone
    out = derive_session({"model": "forge", "messages": msgs, "user": "u-1"})
    assert out == "u:anon:u-1", out

    # both present: `user` wins (it is checked first)
    out = derive_session({"model": "forge", "messages": msgs,
                          "user": "u-1", "prompt_cache_key": "pck-1"})
    assert out == "u:anon:u-1", out
    print("  prompt_cache_key + user -> u:anon: prefix; user wins when both set")


def test_user_id_scoped_by_api_key():
    """Same `user_id` (or `user` / `prompt_cache_key`) under two different
    API keys must derive two different sessions — without key scoping, the
    `u:` channel would let two virtual keys under one master (or two
    masters) claim the same identity and share one lease.

    This is the regression for the review's should-fix on
    `metadata.user_id` / `data["user"]` / `data["prompt_cache_key"]`:
    each of those channels is now keyed by `api_key_hash` (or `anon`),
    matching the fingerprint channel's pattern. The `metadata.user_id`
    case uses the Claude Code UUID form (the shape-gate accepts only
    values carrying `_session_`); see `test_user_id_shape_gated_on_session_marker`
    for the shape-gate behaviour and the residual for non-`_session_`
    values."""
    msgs = [{"role": "user", "content": "hi"}]

    # Same Claude-Code-shaped user_id, different api_key_hash:
    # distinct sessions.
    codex_uid = "user_account_uuid-001_session_session-cccc__longlived"
    a = derive_session(
        {"model": "forge", "messages": msgs,
         "metadata": {"user_id": codex_uid}},
        api_key_hash="key-aaa",
    )
    b = derive_session(
        {"model": "forge", "messages": msgs,
         "metadata": {"user_id": codex_uid}},
        api_key_hash="key-bbb",
    )
    assert a is not None and b is not None
    assert a != b, (a, b)
    assert a == f"u:key-aaa:{codex_uid}", a
    assert b == f"u:key-bbb:{codex_uid}", b

    # Same `data["user"]`, different api_key_hash: distinct sessions.
    a = derive_session(
        {"model": "forge", "messages": msgs, "user": "alice"},
        api_key_hash="key-aaa",
    )
    b = derive_session(
        {"model": "forge", "messages": msgs, "user": "alice"},
        api_key_hash="key-bbb",
    )
    assert a is not None and b is not None
    assert a != b, (a, b)

    # Same `data["prompt_cache_key"]`, different api_key_hash: distinct.
    a = derive_session(
        {"model": "forge", "messages": msgs, "prompt_cache_key": "pck"},
        api_key_hash="key-aaa",
    )
    b = derive_session(
        {"model": "forge", "messages": msgs, "prompt_cache_key": "pck"},
        api_key_hash="key-bbb",
    )
    assert a is not None and b is not None
    assert a != b, (a, b)

    # And the `api_key_hash=None` (anon) fallback matches the unscoped path.
    no_key = derive_session(
        {"model": "forge", "messages": msgs, "user": "alice"}
    )
    assert no_key == "u:anon:alice", no_key

    print(f"  u: channel scoped by key: {a!r} vs {b!r}")


def test_user_id_shape_gated_on_session_marker():
    """`metadata.user_id` is only honored as a session key when shaped
    like Claude Code's per-session UUID form (it carries a
    `_session_` marker). A plain end-user id — LiteLLM canonical
    semantics — falls through to the lane-aware fingerprint, which
    separates parallel conversations under one client key by their
    message content.

    This is the within-key prong of the review's should-fix: same key,
    same `user_id`, different conversations must derive different
    sessions. The shape-gate is what routes the non-`_session_`
    value down to the fingerprint below; without it, two parallel
    conversations under one client key would merge onto one
    `u:<key>:<id>` lease — the #162 failure class.
    """
    base = {"model": "forge", "metadata": {"user_id": "alice"}}

    # Two parallel conversations, same key, same plain end-user id:
    # the shape-gate does NOT match, so the lane-aware fingerprint
    # separates them by message content.
    a = derive_session(dict(base, messages=[
        {"role": "user", "content": "refactor the auth module"},
    ]))
    b = derive_session(dict(base, messages=[
        {"role": "user", "content": "write the release notes"},
    ]))
    assert a is not None and b is not None
    assert a != b, (a, b)
    # Both went through the fingerprint channel (fp: prefix), not the
    # u: channel.
    assert "fp:" in a and "fp:" in b, (a, b)

    # The Claude Code form (carries `_session_`) IS honored as a
    # session key — that is the documented session shortcut the
    # shape-gate exists to preserve.
    codex = {"model": "forge", "metadata": {
        "user_id": "user_account123_session_abc-def-ghi__longlived",
    }}
    a = derive_session(dict(codex, messages=[
        {"role": "user", "content": "refactor the auth module"},
    ]))
    b = derive_session(dict(codex, messages=[
        {"role": "user", "content": "write the release notes"},
    ]))
    assert a is not None and b is not None
    # Same Claude Code session UUID across turns -> same session.
    assert a == b, (a, b)
    assert a.startswith("u:"), a

    # Two terminals using different Claude Code session UUIDs ->
    # distinct sessions even with the same messages.
    c1 = {"model": "forge", "metadata": {
        "user_id": "user_account123_session_uuid-001__longlived",
    }}
    c2 = {"model": "forge", "metadata": {
        "user_id": "user_account123_session_uuid-002__longlived",
    }}
    same_msgs = [{"role": "user", "content": "same first turn"}]
    a = derive_session(dict(c1, messages=same_msgs))
    b = derive_session(dict(c2, messages=same_msgs))
    assert a is not None and b is not None
    assert a != b, (a, b)
    assert a.startswith("u:") and b.startswith("u:")

    print(f"  user_id shape-gate: non-_session_ -> fp; "
          f"_session_ -> u: ({a!r} vs {b!r})")


def test_user_id_substring_gate_excludes_false_positives():
    """The `_session_` substring gate is anchored on the canonical
    Claude Code form `user_<...>_session_<id>__<tag>` — the `__`
    trailing tag separator and the `_session_<id>__` segment are
    both required, not just the bare substring `_session_`.

    Without the anchor, a plain end-user id containing the substring
    `_session_` (e.g. `bob_session_counter`, an OpenAI `user_*` token
    that happens to mention "session" in its tag) was being promoted
    to a lease-merging session key, contradicting the docstring's
    fall-through claim and re-introducing the within-key collapse for
    those clients. The anchored regex keeps the Claude Code shortcut
    and lets realistic end-user ids fall through to the fingerprint.

    This is the regression for the review's should-fix on the
    substring-gate fragility (cycle 3)."""
    def fp_for(user_id: str, msgs=None):
        if msgs is None:
            msgs = [{"role": "user", "content": "hi"}]
        return derive_session(
            {"model": "forge", "messages": msgs,
             "metadata": {"user_id": user_id}},
            api_key_hash="abc12345",
        )

    # Canonical Claude Code form: still honored as a session key.
    codex = fp_for("user_account_uuid-001_session_session-aaa__longlived")
    assert codex.startswith("u:"), codex
    assert codex == "u:abc12345:user_account_uuid-001_session_session-aaa__longlived", codex

    # Short-form Claude Code: still honored (`user_X_session_Y__tag`).
    short = fp_for("user_a_session_abc__x")
    assert short.startswith("u:"), short

    # False positive: bare substring match without trailing `__<tag>`.
    # This is the reviewer's exact repro: `bob_session_counter` is
    # end-user identity, not a session UUID; it must fall through
    # to the fingerprint, not promote to a lease-merging session key.
    fp_a = fp_for("bob_session_counter", msgs=[
        {"role": "user", "content": "refactor the auth module"},
    ])
    fp_b = fp_for("bob_session_counter", msgs=[
        {"role": "user", "content": "write the release notes"},
    ])
    # Both fall through to the fingerprint and are separated by message
    # content (the within-key prong of the original #162 finding).
    assert fp_a is not None and fp_b is not None
    assert fp_a != fp_b, (fp_a, fp_b)
    assert "fp:" in fp_a and "fp:" in fp_b, (fp_a, fp_b)
    assert not fp_a.startswith("u:") and not fp_b.startswith("u:"), (fp_a, fp_b)

    # False positive: multi-`_session_` substring with no `__<tag>`.
    multi = fp_for("user_456_session_abc_session_xyz")
    # Falls through to the fingerprint (no `__` tag separator).
    assert "fp:" in multi, multi
    assert not multi.startswith("u:"), multi

    # Plain end-user id with no `_session_`: still falls through.
    plain = fp_for("alice")
    assert "fp:" in plain, plain
    assert not plain.startswith("u:"), plain

    print("  substring-gate anchored: canonical u:, false positives fp:")


def test_existing_text_only_fingerprint_stable_across_messages():
    """Regression: a plain text conversation still hashes to the same
    prefix (with the new lane suffix), and the role+content hash is
    stable across repeats."""
    msgs = [
        {"role": "system", "content": "You are Claude Code."},
        {"role": "user", "content": "list the repo"},
    ]
    a = derive_session({"model": "forge", "messages": msgs})
    b = derive_session({"model": "forge", "messages": msgs})
    assert a == b, (a, b)
    # fp: prefix survives; lane is suffixed.
    assert "fp:" in a and a.endswith("|forge"), a
    print(f"  text-only fingerprint: stable across repeats: {a!r}")


def test_empty_body_no_identifier_returns_none():
    """The pre-change behaviour for an empty body is unchanged: no
    messages, no identifier of any kind → `None`. The picker already
    handles `None` as 'no affinity, no slot pinning'."""
    out = derive_session({})
    assert out is None, out

    out = derive_session({"messages": []})
    assert out is None, out

    out = derive_session({"messages": None})
    assert out is None, out
    print("  empty body, no identifier -> None (unchanged)")


def test_tool_use_block_contributes_to_fingerprint():
    """A leading assistant tool_use block has a stable id across turns of
    one session. The fingerprint should distinguish two conversations
    that share the same text prefix but call different tools.

    (Tool_use is mostly an assistant-role event; the spec is explicit that
    the first eight messages are walked regardless of role, so the
    `tool_use:<id>` contribution is what makes this test pass.)"""
    msgs_a = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": [
            _tool_use_block("toolu_01AAA"),
        ]},
    ]
    msgs_b = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": [
            _tool_use_block("toolu_01BBB"),
        ]},
    ]
    a = derive_session({"model": "forge", "messages": msgs_a})
    b = derive_session({"model": "forge", "messages": msgs_b})
    assert a is not None and b is not None
    assert a != b, (a, b)
    print(f"  tool_use id distinguishes: {a!r} vs {b!r}")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
