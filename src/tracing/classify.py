"""Measure how much of an answer each retrieved item accounts for.

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
    """What fraction of the answer's tokens appear in ``content``.

    **The answer is the denominator, deliberately. Do not invert this.** The
    other direction — what fraction of the *item* appears in the answer —
    punishes long documents: a thousand-line file that supplied the one
    function the answer quoted scores near zero and looks unused. Measured, a
    file supplying every token of the answer drops below the threshold once it
    carries about 36 tokens it did not contribute, and real source files carry
    hundreds. Dividing by the answer asks what the answer actually drew on,
    which is the question a trace exists to answer.

    Both directions are asserted against each other in
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

    return len(answer_tokens & item_tokens) / len(answer_tokens)


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
