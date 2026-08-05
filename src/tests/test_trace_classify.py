"""Tests for M3, the used-vs-ignored classifier.

The threshold is pinned here rather than by a live run. Three fixtures do that
work: one item that overlaps the answer heavily, one that shares nothing, and
one built to land exactly on the threshold so the comparison direction is
nailed down and cannot drift silently.

The boundary fixtures use distinct nonsense tokens on purpose — with a
ten-word answer, an item sharing two of those words scores exactly 0.20, which
makes the assertions exact instead of approximate.
"""

from __future__ import annotations

import pytest

from src.tracing import (
    DEFAULT_THRESHOLD,
    MIN_TOKEN_LENGTH,
    is_used,
    used_count,
    Trace,
    TraceItem,
    capture,
    score_overlaps,
    overlap_score,
)

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

# Ten distinct content words, none of them stopwords.
BOUNDARY_ANSWER = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"

REALISTIC_ANSWER = (
    "Alice rewrote the auth middleware token expiry check, which fixed the "
    "login failures for expired tokens."
)

HEAVY = (
    "Rewrote the auth middleware token expiry check so expired tokens no "
    "longer cause login failures."
)

UNRELATED = "Document how to rotate service account credentials for staging."


def boundary_trace() -> Trace:
    """Items scoring exactly 0.30, 0.20 and 0.10 against the answer."""
    return Trace(
        query="q",
        answer=BOUNDARY_ANSWER,
        items=[
            TraceItem(id="above", content="alpha bravo charlie", source="stub"),
            TraceItem(id="at", content="alpha bravo", source="stub"),
            TraceItem(id="below", content="alpha", source="stub"),
        ],
    )


# --------------------------------------------------------------------------
# the score itself
# --------------------------------------------------------------------------


def test_an_item_sharing_nothing_scores_zero():
    assert overlap_score(UNRELATED, REALISTIC_ANSWER) == 0.0


def test_an_item_repeating_the_answer_scores_one():
    assert overlap_score(REALISTIC_ANSWER, REALISTIC_ANSWER) == 1.0


def test_a_heavily_overlapping_item_scores_high():
    assert overlap_score(HEAVY, REALISTIC_ANSWER) > 0.5


def test_the_boundary_fixtures_score_exactly_as_designed():
    assert overlap_score("alpha bravo charlie", BOUNDARY_ANSWER) == pytest.approx(0.30)
    assert overlap_score("alpha bravo", BOUNDARY_ANSWER) == pytest.approx(0.20)
    assert overlap_score("alpha", BOUNDARY_ANSWER) == pytest.approx(0.10)


def test_scoring_ignores_case():
    assert overlap_score("ALPHA BRAVO", BOUNDARY_ANSWER) == pytest.approx(0.20)


def test_scoring_ignores_punctuation():
    assert overlap_score("alpha, bravo!", BOUNDARY_ANSWER) == pytest.approx(0.20)


def test_repeating_a_word_does_not_inflate_the_score():
    assert overlap_score("alpha alpha alpha", BOUNDARY_ANSWER) == pytest.approx(0.10)


def test_stopwords_do_not_count_as_overlap():
    assert overlap_score("the and of to with", REALISTIC_ANSWER) == 0.0


def test_an_empty_answer_scores_zero():
    assert overlap_score(HEAVY, "") == 0.0


def test_an_empty_item_scores_zero():
    assert overlap_score("", REALISTIC_ANSWER) == 0.0


# --------------------------------------------------------------------------
# the threshold, applied at read time
# --------------------------------------------------------------------------


def test_the_default_threshold_is_pinned():
    assert DEFAULT_THRESHOLD == 0.2


def test_scoring_records_the_measurement_and_no_verdict():
    """The whole point of schema 2: a trace stores what was measured."""
    trace = score_overlaps(boundary_trace())

    assert [item.overlap for item in trace.items] == pytest.approx([0.30, 0.20, 0.10])
    assert not hasattr(trace.items[0], "used")


def test_an_overlap_exactly_on_the_threshold_counts_as_used():
    assert is_used(0.2, 0.2) is True


def test_an_overlap_just_below_the_threshold_is_ignored():
    assert is_used(0.1, 0.2) is False


def test_an_unmeasured_item_is_unclassified_not_ignored():
    assert is_used(None) is None


def test_the_same_trace_reclassifies_at_a_different_threshold():
    """An archived trace is not frozen at the cutoff current when written."""
    trace = score_overlaps(boundary_trace())

    strict = [is_used(item.overlap, 0.25) for item in trace.items]
    loose = [is_used(item.overlap, 0.05) for item in trace.items]

    assert strict == [True, False, False]
    assert loose == [True, True, True]


def test_used_count_applies_the_threshold_for_a_consumer():
    trace = score_overlaps(boundary_trace())

    assert used_count(trace, 0.2) == 2
    assert used_count(trace, 0.25) == 1
    assert used_count(trace, 0.05) == 3


def test_the_three_deliberate_fixtures_land_where_intended():
    trace = Trace(
        query="q",
        answer=REALISTIC_ANSWER,
        items=[
            TraceItem(id="heavy", content=HEAVY, source="stub"),
            TraceItem(id="none", content=UNRELATED, source="stub"),
        ],
    )

    score_overlaps(trace)

    by_id = {item.id: item for item in trace.items}
    assert is_used(by_id["heavy"].overlap) is True
    assert is_used(by_id["none"].overlap) is False


def test_a_trace_with_no_answer_is_left_unmeasured():
    trace = Trace(query="q", items=[TraceItem(id="a", content=HEAVY, source="stub")])

    score_overlaps(trace)

    assert trace.items[0].overlap is None
    assert is_used(trace.items[0].overlap) is None


def test_a_trace_with_an_empty_answer_is_left_unmeasured():
    trace = Trace(
        query="q", answer="", items=[TraceItem(id="a", content=HEAVY, source="stub")]
    )

    score_overlaps(trace)

    assert trace.items[0].overlap is None


def test_score_overlaps_returns_the_trace_it_was_given():
    trace = boundary_trace()

    assert score_overlaps(trace) is trace


def test_a_trace_with_no_items_scores_cleanly():
    trace = score_overlaps(Trace(query="q", answer=REALISTIC_ANSWER))

    assert trace.items == []


# --------------------------------------------------------------------------
# capture then score_overlaps, the order the pipeline uses
# --------------------------------------------------------------------------


def stub_retriever(query: str) -> list[dict]:
    return [
        {"id": "doc:1", "content": HEAVY, "source": "stub", "score": 0.9},
        {"id": "doc:2", "content": UNRELATED, "source": "stub", "score": 0.4},
    ]


def test_capture_leaves_classification_alone_and_classify_fills_it_in():
    trace = capture("q", stub_retriever("q"), REALISTIC_ANSWER)
    assert all(is_used(item.overlap) is None for item in trace.items)

    score_overlaps(trace)

    assert [is_used(item.overlap) for item in trace.items] == [True, False]


def test_the_viewer_reports_how_many_of_the_retrieved_items_were_used():
    from src.tracing import render

    trace = score_overlaps(capture("q", stub_retriever("q"), REALISTIC_ANSWER))

    assert "1 of 2" in render(trace)


# --------------------------------------------------------------------------
# tokenization
#
# A code corpus lives on identifiers and issue refs. A plain word-character
# split shatters exactly the tokens that carry the signal.
# --------------------------------------------------------------------------


def test_an_identifier_survives_as_one_token():
    """`payment_service` split in two becomes two common, useless words."""
    assert overlap_score("payment_service", "payment_service failed") == pytest.approx(0.5)
    assert overlap_score("payment", "payment_service failed") == 0.0


def test_an_issue_reference_survives_as_one_token():
    # The answer has three content tokens: see, #412, #999 ("and" is a stopword).
    assert overlap_score("fixes #412", "see #412 and #999") == pytest.approx(1 / 3)
    # The bare number is a different token, so a plain digit split loses the ref.
    assert overlap_score("412", "see #412 and #999") == 0.0


def test_a_hyphenated_name_survives_as_one_token():
    assert overlap_score("feature-flag", "feature-flag rollout") == pytest.approx(0.5)
    assert overlap_score("feature", "feature-flag rollout") == 0.0


def test_short_tokens_are_dropped():
    """Two-character fragments match everywhere and say nothing."""
    assert overlap_score("id os go", "id os go alpha") == 0.0


def test_a_token_at_the_minimum_length_is_kept():
    assert MIN_TOKEN_LENGTH == 3
    assert overlap_score("abc", "abc") == 1.0


def test_repetition_does_not_inflate_the_score():
    """Tokens are a set: saying a word twenty times is saying it once."""
    once = overlap_score("alpha", "alpha bravo")
    many = overlap_score("alpha alpha alpha alpha", "alpha bravo")

    assert once == many == pytest.approx(0.5)


def test_repetition_in_the_answer_does_not_shrink_the_score():
    assert overlap_score("alpha", "alpha alpha alpha") == 1.0


def test_the_ratio_is_answer_coverage_not_item_coverage():
    """A long document that supplied the answer is not punished for its length.

    The denominator is the answer, deliberately.
    """
    long_item = "alpha " + " ".join(f"unrelated{n}" for n in range(200))

    assert overlap_score(long_item, "alpha") == 1.0
