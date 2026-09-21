"""Deriving a stable session identity from an arbitrary client's request.

Affinity needs to know what a session *is*, and no wire protocol tells us.
In order of preference:
  1. an explicit header, for clients you control;
  2. a metadata field, same idea via the request body;
  3. LiteLLM's trace id, when the client threads one through;
  4. a hash of the conversation prefix, which is stable across the turns of
     one agent session and differs between sessions. Good enough that a long
     Claude Code or Cursor session stays on one provider without any client
     changes at all.
"""
from __future__ import annotations

import hashlib
from typing import Any

# Any of these names is accepted, so a client that already sets a session header
# needs no change. Add your own here if it uses a different one.
HEADERS = ("x-switchyard-session", "x-session-id", "x-conversation-id")


def _prefix_fingerprint(messages: list[dict[str, Any]] | None) -> str | None:
    if not messages:
        return None
    parts: list[str] = []
    for msg in messages[:2]:            # system + first user turn
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content[:2000])
        elif isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(str(blk.get("text", ""))[:2000])
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

    if meta.get("litellm_trace_id"):
        return f"t:{meta['litellm_trace_id']}"

    fp = _prefix_fingerprint(data.get("messages"))
    if fp:
        return f"{api_key_hash or 'anon'}:{fp}"
    return None
