"""Measure how much of each retrieved item the answer took up.

Two separate things live here, and keeping them separate is the point:

* ``overlap_score`` **measures**. It is a property of the text and does not
  change.
* ``is_used`` **judges**. It applies a threshold, and the threshold is a
  choice.

Only the measurement is written into a trace. The verdict is computed at read
time by whatever displays or consumes the trace, so changing the threshold
re-classifies every trace ever archived instead of freezing each one at
whatever cutoff was current the day it was written.

No model reads the text and judges it. A judged score cannot be reproduced or
checked; this one can be recomputed by hand from the trace.
"""

from __future__ import annotations

from ._text import MIN_TOKEN_LENGTH, tokens
from .schema import Trace

#: The one place the cutoff is defined. Consumers import it rather than
#: hard-coding a number.
DEFAULT_THRESHOLD = 0.2

__all__ = [
    "DEFAULT_THRESHOLD",
    "MIN_TOKEN_LENGTH",
    "overlap_score",
    "is_used",
    "score_overlaps",
    "used_count",
]


def overlap_score(content: str, answer: str) -> float:
    """What fraction of ``content``'s tokens appear in the answer.

    **The item is the denominator, and that fixes what the number means.** The
    score answers *how much of this item reached the answer*. A large file that
    contributed a few words scores low and reads as waste, which is the
    judgement wanted: retrieval that hauls in a thousand lines to use six of
    them is retrieval worth seeing. Dividing by the answer's tokens instead
    would answer *where the answer came from* — a real question, but a
    different one, and not the one this measurement is for.

    The ceiling, and why it decides the unit rather than the threshold
    -----------------------------------------------------------------

    The intersection cannot be larger than the answer, so for an answer of
    ``N`` tokens the highest score any item can reach is ``N / |I|``. Clearing
    a threshold ``t`` therefore requires ``|I| <= N / t`` — at ``t = 0.2``,
    ``|I| <= 5N``. Past that length an item is unreachable no matter how
    relevant it is, because every token it holds beyond ``5N`` is denominator
    it can never match.

    **The fix for that is chunk-sized items, not a lower threshold.** A
    threshold low enough to admit a thousand-token file admits everything else
    too, and the measurement stops discriminating at all. Length is the lever.

    Headroom on the corpus as it stands, measured over both stored traces:

        items                     14
        largest item              16 tokens
        median item                5 tokens
        answers                   17 and 21 tokens
        limit at threshold 0.2    |I| <= 85 and <= 105 tokens
        ceiling, median item      min(21, 5) / 5 = 1.00

    So nothing is near the ceiling here — the largest item is 16 tokens
    against a limit of 105, and every item shorter than the answer can still
    reach 1.0. Real source files average 8,378 bytes, hundreds of tokens, and
    would sit far past it. ``test_an_item_longer_than_the_ceiling_allows_
    cannot_reach_the_threshold`` pins the arithmetic.

    Both denominators are asserted against each other in
    ``test_trace_classify.py``, on a document long enough to tell them apart —
    the demo corpus tops out at 20 tokens and cannot.

    Returns 0.0 when either side has no tokens, so an empty answer never makes
    everything look used.
    """
    answer_tokens = tokens(answer)
    if not answer_tokens:
        return 0.0

    item_tokens = tokens(content)
    if not item_tokens:
        return 0.0

    return len(answer_tokens & item_tokens) / len(item_tokens)


def is_used(overlap: float | None, threshold: float = DEFAULT_THRESHOLD) -> bool | None:
    """Whether an overlap counts as used. Pure: no trace, no state.

    ``None`` in, ``None`` out — an unmeasured item is unclassified, which is
    not the same as ignored. An overlap exactly on the threshold counts as
    used.
    """
    if overlap is None:
        return None
    return overlap >= threshold


def score_overlaps(trace: Trace) -> Trace:
    """Measure every item against the answer, in place.

    Writes ``overlap`` and nothing else — no verdict is recorded, because the
    threshold that would produce one is not this function's business.

    A trace with no answer is left alone: without an answer there is nothing to
    have been used, and guessing would be worse than leaving items unmeasured.
    """
    if not trace.answer:
        return trace

    for item in trace.items:
        item.overlap = overlap_score(item.content, trace.answer)

    return trace


def used_count(trace: Trace, threshold: float = DEFAULT_THRESHOLD) -> int:
    """How many items clear the threshold. Convenience for consumers."""
    return sum(1 for item in trace.items if is_used(item.overlap, threshold))
