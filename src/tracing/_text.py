"""Shared lexical utilities for overlap scoring and query matching.

One tokenizer, used by the classifier and by any producer that needs to compare
text. Two copies would drift, and a drifted tokenizer changes every score
silently.
"""

from __future__ import annotations

import re
from typing import Final

#: Shortest token worth counting. Two-character fragments match everywhere and
#: say nothing.
MIN_TOKEN_LENGTH: Final[int] = 3

# A token starts with an alphanumeric or a hash, then continues with letters,
# digits, underscores, hashes and hyphens.
#
# The leading-character anchor keeps punctuation out of the front of a token, so
# a stray "-foo" does not become a distinct term from "foo". The hash is
# admitted as a start character on purpose, and the reason is measured rather
# than assumed: on the verification corpus, anchoring on [a-z0-9] alone makes
# five of the eleven issue references — #13, #42, #77, #88, #99 — disappear
# entirely, because two digits fall under MIN_TOKEN_LENGTH once the hash is
# stripped. A reference that produces no token at all cannot be searched for or
# scored against, which is worse than an ambiguous one.
#
# Note what the reason is *not*. It is tempting to say the hash stops a bare
# "412" colliding with some other "412" in the text. Measured, that fires zero
# times here: the only bare numeric token in the verification documents is
# "410", an HTTP status, and no ticket number strips down to it. Collision is a
# real hazard on a corpus that carries loose numbers; it is not what this
# pattern buys on this one, and pinning the wrong reason means the pattern gets
# "simplified" the first time someone checks the stated one and finds it idle.
# ``test_the_corpus_loses_five_references_without_the_leading_hash`` pins the
# real one.
#
# Underscores and hyphens are kept inside tokens for the same reason. A plain
# word-character split shatters "payment_service" into two common words and
# halves "feature-flag", which is most of the signal on a code corpus.
WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"[a-z0-9#][a-z0-9_#-]{%d,}" % (MIN_TOKEN_LENGTH - 1),
    re.IGNORECASE,
)

#: Words too common to say anything about whether a document was used.
#
# Nineteen words, and short on purpose. A ninety-six-word list was measured
# against this one over 219 scored pairs — trace items against their answer,
# every ordered pair of verification documents, and every graph node against
# the demo query. The long list moved two verdicts, both on
# document-to-document comparisons nothing in this project consumes. Retrieval
# ranking, the composed answer and every used/ignored verdict on the demo trace
# were identical under both. Only nineteen of the ninety-six ever appeared in
# the corpus at all.
#
# Note what the list does *not* need to carry. ``MIN_TOKEN_LENGTH`` already
# discards everything shorter than three characters, so "a", "an", "of", "to",
# "in", "is" and their kind are unreachable as tokens whether or not a stopword
# list names them — twelve of the long list's entries were dead weight for that
# reason alone. So the length is spent on "that", "this", "have", "were",
# "also", "after": long enough to survive tokenization, common enough to say
# nothing.
#
# Growing this list is allowed, but it should follow a measurement showing what
# the new words buy. ``test_text.py`` pins the count.
STOP: Final[frozenset[str]] = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "was", "were", "are",
    "has", "have", "not", "but", "its", "also", "into", "over", "after",
})


def tokens(
    text: str,
    *,
    stop: frozenset[str] = STOP,
    pattern: re.Pattern[str] = WORD_RE,
) -> set[str]:
    """Extract normalized, unique lexical tokens from text.

    A set, not a list: repeating a word twenty times must not make a document
    look twenty times more relevant.
    """
    return {
        token
        for token in (match.group(0).lower() for match in pattern.finditer(text))
        if token not in stop
    }
