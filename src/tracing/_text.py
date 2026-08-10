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
# admitted as a start character on purpose: "#412" is an issue reference and
# must survive whole, while a pattern anchored on [a-z0-9] alone would strip the
# hash and leave a bare number that collides with any other "412" in the text.
#
# Underscores and hyphens are kept inside tokens for the same reason. A plain
# word-character split shatters "payment_service" into two common words and
# halves "feature-flag", which is most of the signal on a code corpus.
WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"[a-z0-9#][a-z0-9_#-]{%d,}" % (MIN_TOKEN_LENGTH - 1),
    re.IGNORECASE,
)

#: Words too common to say anything about whether a document was used.
STOP: Final[frozenset[str]] = frozenset(
    """
    a an the and or but if in on at to of for with without from by as is are was
    were be been being it its this that these those there their them they he she
    his her you your we our us i not no do does did has have had will would can
    could should may might must so than then when where which who whom what why
    how all any both each few more most other some such only own same too very
    just about into over under again further once here
    """.split()
)


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
