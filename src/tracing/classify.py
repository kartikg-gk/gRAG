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

import re

from .schema import Trace

#: The one place the cutoff is defined. Consumers import it rather than
#: hard-coding a number.
DEFAULT_THRESHOLD = 0.2

#: Shortest token worth counting. Two-character fragments match everywhere and
#: say nothing.
MIN_TOKEN_LENGTH = 3

# Underscores, hashes and hyphens are kept inside tokens on purpose. A plain
# word-character split shatters `payment_service` into two common words, turns
# `#412` into a bare number, and splits `feature-flag` in half — which destroys
# most of the signal on a code corpus, where identifiers and issue references
# are the distinctive terms.
_WORD_PATTERN = re.compile(r"[a-z0-9_#\-]+")

# Words too common to say anything about whether a document was used.
_STOPWORDS = frozenset(
    """
    a an the and or but if in on at to of for with without from by as is are was
    were be been being it its this that these those there their them they he she
    his her you your we our us i not no do does did has have had will would can
    could should may might must so than then when where which who whom what why
    how all any both each few more most other some such only own same too very
    just about into over under again further once here
    """.split()
)


def _tokens(text: str) -> set[str]:
    """Distinct meaningful tokens.

    A set, not a list: repeating a word twenty times must not make a document
    look twenty times more relevant.
    """
    return {
        token
        for token in _WORD_PATTERN.findall(text.lower())
        if len(token) >= MIN_TOKEN_LENGTH and token not in _STOPWORDS
    }


def overlap_score(content: str, answer: str) -> float:
    """What fraction of the answer's tokens appear in ``content``.

    **Answer coverage, deliberately.** The other direction — what fraction
    of the *item* appears in the answer — punishes long documents, since a
    thousand-line file that supplied the one function the answer quoted
    scores near zero and looks unused. Measuring against the answer instead
    asks what the answer actually drew on, which is the question the trace
    exists to answer. Do not invert
    this.

    Returns 0.0 when either side has no tokens, so an empty answer never makes
    everything look used.
    """
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0

    item_tokens = _tokens(content)
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
