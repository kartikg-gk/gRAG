"""Tests for the shared tokenizer.

One tokenizer, used by the classifier and by any producer that compares text.
Two copies would drift, and a drifted tokenizer changes every score silently —
so this suite pins the pattern rather than leaving it to whichever caller
happens to exercise it.
"""

from __future__ import annotations

import pytest

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


@pytest.mark.parametrize("word", ["the", "and", "for", "with", "which", "would"])
def test_each_common_word_is_filtered(word):
    assert tokens(f"alpha {word} bravo") == {"alpha", "bravo"}


def test_text_of_only_stopwords_has_no_tokens():
    assert tokens("the and of to with") == set()


def test_a_stopword_inside_an_identifier_is_not_stripped():
    """Filtering is per token, not per substring."""
    assert tokens("the_and_service") == {"the_and_service"}


def test_the_stopword_set_is_not_empty():
    assert len(STOP) > 20


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
