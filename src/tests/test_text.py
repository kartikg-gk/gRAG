"""Tests for the shared tokenizer.

One tokenizer, used by the classifier and by any producer that compares text.
Two copies would drift, and a drifted tokenizer changes every score silently —
so this suite pins the pattern rather than leaving it to whichever caller
happens to exercise it.
"""

from __future__ import annotations

import re

import pytest

from examples.entity_demo import DOCUMENTS
from src.tracing._text import MIN_TOKEN_LENGTH, STOP, WORD_RE, tokens


# ==========================================================================
# Extraction
# ==========================================================================


def test_words_become_tokens():
    assert tokens("alpha bravo charlie") == {"alpha", "bravo", "charlie"}


def test_tokens_are_lowercased():
    assert tokens("ALPHA Bravo") == {"alpha", "bravo"}


def test_punctuation_is_not_part_of_a_token():
    assert tokens("alpha, bravo! charlie.") == {"alpha", "bravo", "charlie"}


def test_the_result_is_a_set_not_a_list():
    """Repetition must not inflate anything downstream."""
    assert isinstance(tokens("alpha alpha alpha"), set)
    assert tokens("alpha alpha alpha") == {"alpha"}


def test_empty_text_has_no_tokens():
    assert tokens("") == set()


def test_text_of_only_punctuation_has_no_tokens():
    assert tokens("!!! ... ???") == set()


# ==========================================================================
# What the pattern must preserve
# ==========================================================================


def test_an_identifier_survives_as_one_token():
    """`payment_service` split in two becomes two common, useless words."""
    assert tokens("payment_service failed") == {"payment_service", "failed"}


def test_an_issue_reference_keeps_its_hash():
    """A bare 412 collides with any other number; #412 is the reference."""
    assert tokens("fixes #412") == {"fixes", "#412"}


def test_a_hyphenated_name_survives_as_one_token():
    assert tokens("feature-flag rollout") == {"feature-flag", "rollout"}


def test_a_dotted_path_splits_on_the_dot():
    """Dots are not part of a token, so a path yields its segments."""
    assert tokens("src/billing/webhook.py") == {"src", "billing", "webhook"}


def test_a_token_cannot_begin_with_a_hyphen():
    """Otherwise '-foo' and 'foo' would be different terms."""
    assert tokens("-alpha") == {"alpha"}


def test_a_token_cannot_begin_with_an_underscore():
    assert tokens("_alpha") == {"alpha"}


# ==========================================================================
# Length
# ==========================================================================


def test_the_minimum_length_is_three():
    assert MIN_TOKEN_LENGTH == 3


@pytest.mark.parametrize("short", ["id", "os", "go", "a", "x1"])
def test_tokens_shorter_than_the_minimum_are_dropped(short):
    assert tokens(short) == set()


def test_a_token_at_the_minimum_length_is_kept():
    assert tokens("abc") == {"abc"}


def test_a_short_hash_reference_still_counts():
    """'#42' is three characters and is a real reference."""
    assert tokens("#42") == {"#42"}


# ==========================================================================
# Stopwords
# ==========================================================================


def test_stopwords_are_removed():
    assert tokens("the quick and the dead") == {"quick", "dead"}


@pytest.mark.parametrize(
    "word", ["the", "and", "for", "with", "from", "that", "were", "also"]
)
def test_each_common_word_is_filtered(word):
    assert tokens(f"alpha {word} bravo") == {"alpha", "bravo"}


@pytest.mark.parametrize(
    "word", ["which", "would", "when", "about", "same", "without"]
)
def test_a_word_outside_the_list_is_kept(word):
    """The other side of the short list, asserted rather than implied.

    All six were stopwords under a ninety-six-word list measured against this
    one, and are ordinary tokens now. That long list changed two verdicts
    across 219 scored pairs, both on document-to-document comparisons nothing
    consumes — see the note on ``STOP``.

    Pinned so the list cannot quietly grow back one word at a time without
    someone deciding to.
    """
    assert tokens(f"alpha {word} bravo") == {"alpha", word, "bravo"}


@pytest.mark.parametrize("word", ["a", "an", "of", "to", "in", "is", "by", "as"])
def test_a_short_function_word_never_needed_the_list(word):
    """Length does this work, not membership.

    ``STOP`` spends none of its nineteen slots on these, and it does not have
    to: ``MIN_TOKEN_LENGTH`` discards them first. A list that named them would
    look longer while filtering nothing, which is most of what made a
    ninety-six-word version look useful.
    """
    assert word not in STOP
    assert tokens(f"alpha {word} bravo") == {"alpha", "bravo"}


def test_text_of_only_stopwords_has_no_tokens():
    assert tokens("the and of to with") == set()


def test_a_stopword_inside_an_identifier_is_not_stripped():
    """Filtering is per token, not per substring."""
    assert tokens("the_and_service") == {"the_and_service"}


def test_the_stopword_list_holds_exactly_nineteen_words():
    """Pinned at nineteen. A change is a decision, not a tidy-up.

    Measured against a ninety-six-word list: the extra seventy-seven words
    moved two verdicts across 219 scored pairs, both on document-to-document
    comparisons nothing consumes, and twelve of them were words
    ``MIN_TOKEN_LENGTH`` discards anyway.

    Growing this list is allowed, but it should follow a measurement showing
    what the new words buy — not a reading of the code that finds the list
    looks short.
    """
    assert len(STOP) == 19


def test_the_caller_may_supply_its_own_stopwords():
    assert tokens("alpha bravo", stop=frozenset({"alpha"})) == {"bravo"}


def test_the_caller_may_supply_its_own_pattern():
    import re

    # A single-character pattern, which the default would reject as too short.
    # "a" is filtered anyway because it is a stopword — stop and pattern are
    # independent filters and both still apply.
    assert tokens("x-y z", pattern=re.compile(r"[a-z]")) == {"x", "y", "z"}


# ==========================================================================
# The pattern itself
# ==========================================================================


def test_the_pattern_is_case_insensitive():
    assert WORD_RE.findall("ALPHA") == ["ALPHA"]


def test_the_pattern_matches_a_hash_reference_whole():
    assert WORD_RE.findall("#412") == ["#412"]


# ==========================================================================
# Why the hash is a start character — measured, not assumed
#
# The tests below are the evidence for admitting "#" as a start character,
# which is not the obvious choice — anchoring on [a-z0-9] alone is simpler and
# looks equivalent. They are written against the verification corpus rather
# than against invented strings, because the claim being defended is a claim
# about this corpus.
# ==========================================================================

#: The simpler pattern: identical but for the leading character class. Kept
#: here so the choice can be asserted against the thing it was chosen over,
#: rather than against a description of it. Nothing in the project uses it.
WITHOUT_LEADING_HASH = re.compile(
    r"[a-z0-9][a-z0-9_#-]{%d,}" % (MIN_TOKEN_LENGTH - 1), re.IGNORECASE
)


def corpus_references() -> set[str]:
    """Every distinct hash-initial token the verification corpus produces."""
    return {
        token
        for document in DOCUMENTS
        for token in tokens(document)
        if token.startswith("#")
    }


def test_the_corpus_holds_eleven_distinct_hash_references():
    """Pinned count. A change here is not a number to update.

    If this fails, either ``examples/entity_demo.DOCUMENTS`` gained or lost a
    reference, or the tokenizer stopped producing them. Both are worth reading
    before anything is re-pinned: the count is the denominator of the test
    below, and silently bumping it would make that test weaker without anyone
    noticing.
    """
    assert len(corpus_references()) == 11


def test_the_corpus_loses_five_references_without_the_leading_hash():
    """The measured reason the hash is admitted as a start character.

    Anchoring on ``[a-z0-9]`` strips the hash and leaves a two-digit
    remainder, which ``MIN_TOKEN_LENGTH`` then discards. Those five references
    produce no token at all, so nothing can search for them or score against
    them.

    If this fails, either the corpus moved or ``MIN_TOKEN_LENGTH`` moved. At
    ``MIN_TOKEN_LENGTH == 2`` every one of these survives as a bare number and
    the leading hash stops paying for itself; at 4, references like ``#412``
    join them and it pays more. Read which happened rather than re-pinning the
    number — the count is the whole argument.
    """
    vanished = {
        reference
        for reference in corpus_references()
        if not tokens(reference, pattern=WITHOUT_LEADING_HASH)
    }

    assert vanished == {"#13", "#42", "#77", "#88", "#99"}
    assert len(vanished) == 5


def test_the_surviving_references_collapse_to_bare_numbers():
    """The other six do produce a token, just not a distinguishing one."""
    survivors = corpus_references() - {"#13", "#42", "#77", "#88", "#99"}

    assert len(survivors) == 6
    for reference in survivors:
        assert tokens(reference, pattern=WITHOUT_LEADING_HASH) == {reference[1:]}


def test_no_reference_actually_collides_with_a_bare_number():
    """The reason that used to be in the comment, kept as a failing witness.

    The old justification for the leading hash was collision: strip it and
    ``#412`` lands on any other ``412`` in the text. On this corpus that never
    happens — the only bare numeric token in ``DOCUMENTS`` is ``410``, the HTTP
    status in document 8, and no reference strips down to it. The claim is not
    wrong in general, it is simply not what this pattern buys here, and a test
    asserting the real reason is worth more than a comment asserting a
    hypothetical one.

    Scope note: ``101`` also appears as a bare token once the graph nodes and
    the demo query are included, since pull request 101 is in the fixtures.
    Still no collision — ``#101`` is not a reference this corpus contains — but
    the wider universe is not pinned here because only ``DOCUMENTS`` is stable
    enough to assert against.

    If this ever fails, the corpus gained a genuine collision and the comment
    in ``_text.py`` should gain it back as a second reason.
    """
    bare_numbers = {
        token
        for document in DOCUMENTS
        for token in tokens(document)
        if token.isdigit()
    }
    stripped = {reference[1:] for reference in corpus_references()}

    assert bare_numbers == {"410"}
    assert bare_numbers & stripped == set()
