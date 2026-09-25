"""Deriving a stable session identity from an arbitrary client's request.

Affinity needs to know what a session *is*, and no wire protocol tells us.
In order of preference:
  1. an explicit header (`x-switchyard-session` and friends), for clients
     you control;
  2. a metadata session id (`session_id` / `switchyard_session` /
     `conversation_id`), same idea via the request body;
  3. `metadata.user_id` — Claude Code threads a per-session UUID through
     there, and it is a better session key than the conversation prefix
     because it survives a refresh that drops the in-flight messages.
     Shape-gated on Claude Code's `user_<account>_session_<uuid>__<tag>`
     form (regex anchored on the `_session_<id>__` segment, see
     `_CLAUDE_CODE_USER_ID`): a plain end-user id (LiteLLM canonical
     semantic), even one that contains the substring `_session_` but
     lacks the trailing `__<tag>`, falls through to the lane-aware
     fingerprint instead, so parallel conversations under one client
     key get distinct fingerprints. The fingerprint is the designed
     degradation path if Claude Code's UUID form ever drifts off the
     regex (stickiness shifts from session to lane+conversation, but
     the picker still keeps the conversation on one plan).
  4. `data["user"]` or `data["prompt_cache_key"]` — Codex / OpenAI's
     equivalent client identity fields, also prefixed `u:` and scoped
     by API key. Within one key, two parallel conversations with the
     same `user` / `prompt_cache_key` value still merge onto one
     lease: a documented trade-off for a client that treats those
     fields as end-user identity rather than session identity;
  5. LiteLLM's trace id, when the client threads one through;
  6. a hash of the conversation prefix, suffixed with the requested lane
     so the same prefix on two lanes (e.g. the same conversation under
     `forge` and again under `judge`) derives different sessions. Stable
     across the turns of one agent session, distinct between sessions,
     and now also distinguishes a vision-first conversation that opens
     with an image (hash of the data-URL prefix / source length) or a
     tool_use block (hash of its `id`) — not just text turns. Good
     enough that a long Claude Code or Cursor session stays on one
     provider without any client changes at all.

The `derive()` function is called before the picker rewrites
`data["model"]` from the lane name to the chosen deployment, so the
"requested lane" in step 6 is read verbatim — it is the caller-supplied
lane, not the post-pick provider deployment.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

# Any of these names is accepted, so a client that already sets a session header
# needs no change. Add your own here if it uses a different one.
HEADERS = ("x-switchyard-session", "x-session-id", "x-conversation-id")

# Identifies the CLI harness in front of us, when the operator opts in by
# setting `x-switchyard-cli` on the inbound request. Same governance as
# `x-switchyard-session`: operator-set, no client in the wild sets it today,
# so the new behaviour (tool blocklist + first-turn injection) is inert until
# something does. The metadata field name is kept for symmetry with the other
# session identifiers; not actually read by anyone yet.
CLI_HEADER = "x-switchyard-cli"
CLI_META_FIELD = "switchyard_cli"

# How many leading messages the conversation-prefix fingerprint walks.
# Eight covers the system block plus the first few turns — enough that a
# conversation which opens with an image and follows with a text turn still
# yields a stable fingerprint, but short enough that a long-running
# conversation does not pay a hash cost on every request.
_FINGERPRINT_WINDOW = 8

# Characters from the head of an image data-URL that fingerprint the source
# (most image formats are recognisable by their first 30 bytes — PNG's
# `\x89PNG\r\n\x1a\n`, JPEG's `\xff\xd8\xff`, WebP's `RIFF…WEBP`, etc.) and
# the byte length, so a different image of the same kind still hashes
# distinctly. The body itself is not hashed: data-URLs are typically tens
# or hundreds of KB of base64, and the prefix + length are enough to
# distinguish images for a session key.
_IMAGE_PREFIX_LEN = 30

# Claude Code threads a per-session UUID through `metadata.user_id` shaped
# like `user_<account_uuid>_session_<session_uuid>__longlived`. The bare
# `_session_` substring match it used to use admitted false positives
# (e.g. `bob_session_counter` is end-user identity, not a session UUID,
# and was being promoted to a lease-merging session key). Anchored on
# the canonical shape: `_session_<id>__<tag>` — the `<id>` is the session
# UUID token (alphanumerics, dashes, internal underscores) and the `__`
# is the double-underscore tag separator Claude Code emits. A plain
# end-user id containing `_session_` but no trailing `__<tag>` falls
# through to the lane-aware fingerprint.
#
# Degradation path: if Claude Code's UUID form drifts and stops matching
# this regex, the channel silently routes to the fingerprint below,
# which separates parallel conversations by message content. That is
# the same fallback the docstring claims for "a refresh that drops the
# in-flight messages", so a format drift is a safe no-regression
# outcome (stickiness may shift from session to lane+conversation, but
# the picker still keeps the conversation on one plan). If you change
# this regex, diff a real Claude Code payload against it before
# trusting the shortcut.
_CLAUDE_CODE_USER_ID = re.compile(r"_session_[A-Za-z0-9_-]+__")


def _block_part(blk: Any) -> str | None:
    """Reduce one content block to a short stable string for hashing.

    Returns None for blocks the fingerprint does not care about (an unknown
    shape, an empty dict, anything we have not been taught to recognise),
    so the caller can drop the block from the hashed parts list without
    contaminating the hash with the literal `"None"`.
    """
    if not isinstance(blk, dict):
        return None
    btype = blk.get("type")
    if btype == "text":
        return "text:" + str(blk.get("text", ""))[:2000]
    if btype == "image" or "image_url" in blk:
        # OpenAI wire form: {"type": "image_url", "image_url": {"url": ...}}.
        # Anthropic wire form: {"type": "image", "source": {"type": "base64",
        # "media_type": ..., "data": ...}}. Both carry the image data in a
        # URL-shaped field (`image_url.url` or `source.data`); the prefix
        # and length give us a stable, distinct fingerprint without paying
        # the cost of hashing the full payload.
        url = (
            (blk.get("image_url") or {}).get("url")
            if isinstance(blk.get("image_url"), dict)
            else blk.get("image_url")
        )
        if url is None:
            source = blk.get("source") or {}
            if isinstance(source, dict):
                url = source.get("data") or source.get("url")
        if url is None:
            return None
        return f"image:{str(url)[:_IMAGE_PREFIX_LEN]}:{len(url)}"
    if btype == "tool_use":
        # Anthropic tool-use block: the `id` is a server-issued UUID scoped
        # to this conversation, so it is exactly the part that is stable
        # across turns of one session and distinct between sessions.
        tid = blk.get("id")
        if tid:
            return f"tool_use:{tid}"
        return None
    return None


def _prefix_fingerprint(messages: list[dict[str, Any]] | None) -> str | None:
    if not messages:
        return None
    parts: list[str] = []
    for msg in messages[:_FINGERPRINT_WINDOW]:
        role = msg.get("role")
        if role:
            parts.append(f"role:{role}")
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content[:2000])
        elif isinstance(content, list):
            for blk in content:
                part = _block_part(blk)
                if part is not None:
                    parts.append(part)
    if not parts:
        return None
    return "fp:" + hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:20]


def derive(data: dict[str, Any], api_key_hash: str | None = None) -> str | None:
    meta = data.get("metadata") or {}
    proxy_req = data.get("proxy_server_request") or {}
    headers = {k.lower(): v for k, v in (proxy_req.get("headers") or {}).items()}

    for h in HEADERS:
        if headers.get(h):
            return f"h:{headers[h]}"

    for field in ("session_id", "switchyard_session", "conversation_id"):
        if meta.get(field):
            return f"m:{meta[field]}"

    # The `u:` channels below are scoped by `api_key_hash` so two virtual
    # keys under one master (or two masters, or `anon`) cannot claim the
    # same `user_id` / `user` / `prompt_cache_key` and share one lease.
    # That handles the cross-key half of #162.
    #
    # The within-key half — one client key, same end-user id, parallel
    # conversations collapsing onto one lease — is handled per-channel:
    # `metadata.user_id` is shape-gated on Claude Code's `_session_` UUID
    # form (the only client documented as sending a per-session UUID
    # there); a plain end-user id falls through to the lane-aware
    # fingerprint, which separates parallel conversations by their
    # message content. `data["user"]` and `data["prompt_cache_key"]` are
    # left unscoped at the within-key level because clients that send
    # them are expected to send a session-shaped value (OpenAI's
    # `prompt_cache_key` is a cache/session id by design; Codex's `user`
    # field is treated as one in the issue spec); the within-key
    # collapse there is a documented trade-off for a client that
    # misuses those fields as end-user identity instead.
    key_prefix = api_key_hash or "anon"

    if meta.get("user_id"):
        # Claude Code threads a per-session UUID through `metadata.user_id`
        # in the canonical `user_<account>_session_<uuid>__<tag>` shape
        # (see `_CLAUDE_CODE_USER_ID` above). The regex match keeps the
        # Claude Code session shortcut and lets a plain end-user id
        # (including one that happens to contain the substring
        # `_session_` but lacks the trailing `__<tag>`) fall through
        # to the lane-aware fingerprint, so parallel conversations
        # under one client key get different fingerprints instead of
        # merging on one lease.
        if _CLAUDE_CODE_USER_ID.search(str(meta["user_id"])):
            return f"u:{key_prefix}:{meta['user_id']}"
        # else: fall through to the fingerprint below. The fingerprint
        # is the designed degradation path if Claude Code's UUID form
        # ever drifts off this regex — see the comment on
        # `_CLAUDE_CODE_USER_ID`.

    for field in ("user", "prompt_cache_key"):
        val = data.get(field)
        if val:
            return f"u:{key_prefix}:{val}"

    if meta.get("litellm_trace_id"):
        return f"t:{meta['litellm_trace_id']}"

    fp = _prefix_fingerprint(data.get("messages"))
    if fp:
        # `data["model"]` is the pre-rewrite lane name — derive() runs
        # before the picker has rewritten it to the chosen deployment.
        # Suffixing the fingerprint with the lane keeps a single
        # conversation on `forge` separate from the same conversation on
        # `judge`, so a session never leaks across lanes it was not
        # asked for. `""` is the documented fallback when the caller did
        # not name a lane.
        lane = data.get("model")
        if not isinstance(lane, str):
            lane = ""
        return f"{api_key_hash or 'anon'}:{fp}|{lane}"
    # `return None` only when there are no messages AND no identifier of
    # any kind. The early returns above catch every identifier channel
    # (header, meta session ids, user_id, data.user / prompt_cache_key,
    # litellm_trace_id), so reaching here means none of them were set.
    # A `None` return is the existing "empty body" behaviour the picker
    # already knew to handle (no affinity, no slot pinning).
    return None


def identify(data: dict[str, Any]) -> str | None:
    """Lowercased CLI harness name from `x-switchyard-cli`, or None.

    The header is read first (operator-set on the inbound request, same path
    `derive()` uses for `x-switchyard-session`); the metadata field is the
    body-shaped fallback, kept for symmetry with `session_id`. A leading/
    trailing whitespace or different casing is normalised before being
    returned, so blocklist lookups against the lowercased `Settings.cli_tool_block`
    keys match without surprise.

    Returns None when no CLI is named — the caller is responsible for treating
    that as "no blocklist, no injection" so an unset header is a no-op.
    """
    proxy_req = data.get("proxy_server_request") or {}
    headers = {k.lower(): v for k, v in (proxy_req.get("headers") or {}).items()}
    raw = headers.get(CLI_HEADER)
    if raw is None:
        meta = data.get("metadata") or {}
        raw = meta.get(CLI_META_FIELD)
    if raw is None:
        return None
    return str(raw).strip().lower() or None
