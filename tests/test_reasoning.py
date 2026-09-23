"""The streamed-reasoning splitter.

MiniMax M3 and other reasoning models wrap their chain of thought in
`<think>...</think>` inside ordinary content. LiteLLM lifts that into
`reasoning_content` for a buffered response but NOT for a streamed one, so the
same model answers cleanly when buffered and leaks raw tags when streamed —
which is what a user reported seeing through the forge lane.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)  # so `import conftest` resolves under plain `python3`

import conftest  # noqa: F401  (socket guard for plain-script mode)

from switchyard.reasoning import ReasoningSplitter  # noqa: E402


def split(chunks: list[str]) -> tuple[str, str]:
    """Feed chunks through a splitter the way a stream would."""
    s = ReasoningSplitter()
    content = reasoning = ""
    for chunk in chunks:
        c, r = s.feed(chunk)
        content += c
        reasoning += r
    c, r = s.flush()
    return content + c, reasoning + r


def test_reasoning_is_moved_not_dropped():
    content, reasoning = split(
        ["<think>The user wants 2+2.</think>\n\n2 + 2 equals 4."])
    assert content == "\n\n2 + 2 equals 4.", content
    assert reasoning == "The user wants 2+2.", reasoning
    print("  buffered-shaped chunk split cleanly")


def test_a_tag_split_across_chunks_is_still_recognised():
    """A stream breaks wherever it likes, including inside `<think>`."""
    content, reasoning = split(["<thi", "nk>why</thi", "nk>the answer"])
    assert content == "the answer", content
    assert reasoning == "why", reasoning

    # And the closing tag split at every possible offset.
    for cut in range(1, len("</think>")):
        c, r = split(["<think>x" + "</think>"[:cut], "</think>"[cut:] + "done"])
        assert c == "done", (cut, c)
        assert r == "x", (cut, r)
    print("  tags reassembled across chunk boundaries at every offset")


def test_content_either_side_of_the_tags_survives():
    content, reasoning = split(["before ", "<think>mid</think>", " after"])
    assert content == "before  after", content
    assert reasoning == "mid", reasoning


def test_text_without_tags_is_passed_through_unchanged():
    text = "just an answer, no reasoning at all"
    content, reasoning = split([text[:5], text[5:]])
    assert content == text, content
    assert reasoning == ""


def test_an_unterminated_tag_is_flushed_as_reasoning():
    """Losing the model's output would be far worse than showing it raw."""
    content, reasoning = split(["<think>never closes"])
    assert content == "", content
    assert reasoning == "never closes", reasoning
    print("  an unclosed tag is flushed, never swallowed")


def test_nothing_is_held_back_indefinitely():
    """Only a partial tag may be buffered, so a long answer is not delayed."""
    s = ReasoningSplitter()
    content, _ = s.feed("a plain sentence with no tags in it at all")
    # At most len("</think>") - 1 characters may be withheld.
    assert len(content) >= len("a plain sentence with no tags in it at all") - 7
    print(f"  emitted {len(content)} chars immediately, holding at most 7 back")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))
