"""Sliding windows over text, carrying absolute offsets.

A statistical model has a bounded input, so long documents get processed in
pieces. The whole difficulty is that the pieces must not lie about where they
came from: an entity found at character 12 of window 7 is useless unless it can
be reported at its position in the original document.

So a window carries the offset it started at, and every position derived from
it is absolute from the moment it is computed. There is no "window-relative"
representation anywhere in this module to accidentally return.

Windows are measured in words, not characters, because splitting mid-token
would manufacture fragments that look like entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from ..common.config import WINDOW_OVERLAP_WORDS, WINDOW_WORDS

#: A word is any run of non-whitespace. Deliberately crude: this decides where
#: a window may be cut, not what counts as a token for scoring.
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Window:
    """A slice of a document that remembers where it came from.

    ``text`` is exactly ``document[start:end]``, which is what makes the offsets
    checkable rather than merely asserted.
    """

    text: str
    start: int
    end: int

    def absolute(self, relative_start: int, relative_end: int) -> tuple[int, int]:
        """Turn a position inside this window into a position in the document."""
        return self.start + relative_start, self.start + relative_end


def windows(
    text: str,
    *,
    size: int = WINDOW_WORDS,
    overlap: int = WINDOW_OVERLAP_WORDS,
) -> Iterator[Window]:
    """Slide a window of ``size`` words with ``overlap`` words of carry-over.

    Raises ``ValueError`` on a configuration that cannot make progress. An
    overlap equal to the window means every window starts where the last one
    did, which is an infinite loop rather than a slow run — worth refusing
    loudly at the call that configured it, not diagnosing later from a hang.

    Text shorter than one window yields one window covering it. Text with no
    words yields nothing: there is no such thing as an entity in whitespace.
    """
    if size < 1:
        raise ValueError(f"window size must be at least 1 word, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap cannot be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than the window ({size}); "
            "an overlap that large never advances"
        )

    spans = [match.span() for match in _WORD.finditer(text)]
    if not spans:
        return

    step = size - overlap
    for first in range(0, len(spans), step):
        chunk = spans[first : first + size]
        if not chunk:
            break

        start, end = chunk[0][0], chunk[-1][1]
        yield Window(text=text[start:end], start=start, end=end)

        # The last window is the one that reached the final word. Without this
        # the tail is emitted again as ever-shorter windows.
        if first + size >= len(spans):
            break
