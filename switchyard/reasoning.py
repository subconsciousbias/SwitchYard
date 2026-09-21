"""Split inline reasoning tags out of a streamed response.

Some providers wrap their chain of thought in `<think>...</think>` inside the
ordinary content. LiteLLM lifts that into `reasoning_content` for a buffered
response but not for a streamed one, so the same model answers cleanly when
buffered and leaks raw tags when streamed — and a client that renders content
verbatim shows the model thinking out loud to the user.

This is a small state machine rather than a regex because a stream splits
wherever it likes: `<thi` can arrive in one chunk and `nk>` in the next, and the
reasoning itself spans many. Text is held back only far enough to recognise a
tag that straddles a boundary.
"""
from __future__ import annotations

OPEN = "<think>"
CLOSE = "</think>"
# Never hold back more than this: enough to recognise a split tag, no more.
_HOLD = max(len(OPEN), len(CLOSE)) - 1


class ReasoningSplitter:
    """One per streamed response. Feed it content, get (content, reasoning)."""

    def __init__(self) -> None:
        self.pending = ""
        self.in_think = False

    def feed(self, text: str) -> tuple[str, str]:
        """Split `text` into (content, reasoning) safe to emit right now."""
        if not text:
            return "", ""
        self.pending += text
        content, reasoning = [], []

        while self.pending:
            if self.in_think:
                cut = self.pending.find(CLOSE)
                if cut == -1:
                    # Emit all but a possible partial closing tag.
                    keep = _tail_to_hold(self.pending, CLOSE)
                    if keep:
                        reasoning.append(self.pending[:-keep])
                        self.pending = self.pending[-keep:]
                    else:
                        reasoning.append(self.pending)
                        self.pending = ""
                    break
                reasoning.append(self.pending[:cut])
                self.pending = self.pending[cut + len(CLOSE):]
                self.in_think = False
                continue

            cut = self.pending.find(OPEN)
            if cut == -1:
                keep = _tail_to_hold(self.pending, OPEN)
                if keep:
                    content.append(self.pending[:-keep])
                    self.pending = self.pending[-keep:]
                else:
                    content.append(self.pending)
                    self.pending = ""
                break
            content.append(self.pending[:cut])
            self.pending = self.pending[cut + len(OPEN):]
            self.in_think = True

        return "".join(content), "".join(reasoning)

    def flush(self) -> tuple[str, str]:
        """Whatever is still held back when the stream ends.

        A tag that never closes is emitted as reasoning rather than dropped:
        losing the model's answer would be far worse than showing it raw.
        """
        rest, self.pending = self.pending, ""
        if not rest:
            return "", ""
        return ("", rest) if self.in_think else (rest, "")


def _tail_to_hold(text: str, tag: str) -> int:
    """How many trailing characters could be the start of `tag`."""
    for size in range(min(_HOLD, len(text)), 0, -1):
        if tag.startswith(text[-size:]):
            return size
    return 0
