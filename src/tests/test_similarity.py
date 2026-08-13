"""Tests for the similarity sources.

The lexical source has to earn the default slot: the drift it is there to catch
is orthographic, so these tests are mostly about separator and case variants
scoring high while genuinely different identifiers score low.

The one property that matters most is the last section. ``#412`` and ``#413``
differ by one character out of four and must not merge — a resolver that folded
consecutive ticket references together would be worse than no resolution at all.
"""

from __future__ import annotations

import pytest

from src.analysis import LexicalSimilarity, load_similarity
from src.common.config import DEEP_THRESHOLD, FAST_THRESHOLD, QUERY_THRESHOLD


@pytest.fixture
def similarity() -> LexicalSimilarity:
    return LexicalSimilarity()


def score(similarity, left: str, right: str) -> float:
    return similarity.scores(left, [right])[0]


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------


def test_scores_come_back_one_per_candidate_in_order(similarity):
    scores = similarity.scores("payment_service", ["payment-service", "auth", "x"])

    assert len(scores) == 3
    assert scores[0] > scores[1]


def test_no_candidates_scores_nothing(similarity):
    assert similarity.scores("payment_service", []) == []


def test_every_score_is_a_cosine(similarity):
    for left, right in [("a_service", "b_service"), ("#412", "#413"), ("x", "y")]:
        assert 0.0 <= score(similarity, left, right) <= 1.0


def test_identical_forms_score_one(similarity):
    assert score(similarity, "payment_service", "payment_service") == pytest.approx(1.0)


def test_similarity_is_symmetric(similarity):
    assert score(similarity, "payment_service", "billing_svc") == pytest.approx(
        score(similarity, "billing_svc", "payment_service")
    )


def test_an_empty_form_scores_zero(similarity):
    assert score(similarity, "", "payment_service") == 0.0
    assert score(similarity, "***", "payment_service") == 0.0


# --------------------------------------------------------------------------
# the drift it exists to catch
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right",
    [
        ("notification-service", "notification_service"),
        ("order_service", "ORDER_SERVICE"),
        ("payment_service", "payment service"),
        ("PR #1290", "PR#1290"),
    ],
)
def test_separator_and_case_variants_merge_on_the_fast_path(left, right):
    """These are the real variants B9's own output produced."""
    assert score(LexicalSimilarity(), left, right) >= FAST_THRESHOLD


@pytest.mark.parametrize(
    "left,right",
    [
        ("#412", "#413"),
        ("payment_service", "auth_service"),
        ("notification_service", "order_service"),
        ("#42", "#77"),
    ],
)
def test_genuinely_different_identifiers_stay_apart(left, right):
    """Below the deep threshold: not merged, and not even worth asking about."""
    assert score(LexicalSimilarity(), left, right) < DEEP_THRESHOLD


def test_consecutive_ticket_numbers_are_not_confusable():
    """The worst possible merge in this corpus, pinned."""
    similarity = LexicalSimilarity()

    for other in ("#413", "#414", "#411"):
        assert score(similarity, "#412", other) < DEEP_THRESHOLD


# --------------------------------------------------------------------------
# mechanics
# --------------------------------------------------------------------------


def test_vectors_are_cached_across_calls(similarity):
    similarity.scores("payment_service", ["auth_service"])
    similarity.scores("payment_service", ["auth_service"])

    assert set(similarity._cache) == {"payment_service", "auth_service"}


def test_the_default_source_needs_no_optional_dependency():
    assert load_similarity().name == "lexical"


def test_an_unavailable_embedding_source_degrades_to_lexical():
    """Same contract as the extractor's backend loader."""
    assert load_similarity("embedding").name in {"lexical", "embedding"}


# --------------------------------------------------------------------------
# the separation the query floor rests on
#
# QUERY_THRESHOLD was measured against this distribution, so a change to the
# scoring function that closes the gap must fail here rather than silently
# making queries return the wrong entity. Found by mutation testing: NGRAM = 1
# passed every other test in this file while pushing two known-distinct pairs
# above the floor.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right",
    [
        ("pull request #1347", "pull request #204"),
        ("pull request #204", "pull requests #3"),
        ("payment_service", "auth-service"),
        ("auth-service", "order_service"),
        ("notification-service", "auth-service"),
        ("#42", "#412"),
    ],
)
def test_known_distinct_entities_stay_below_the_query_floor(left, right):
    """These are labels resolution deliberately kept apart. A query for one
    must never return the other."""
    assert score(LexicalSimilarity(), left, right) < QUERY_THRESHOLD


@pytest.mark.parametrize(
    "term,label",
    [
        ("notification_service", "notification-service"),
        ("PR#1290", "PR #1290"),
        ("ORDER_SERVICE", "order_service"),
    ],
)
def test_real_variants_stay_above_the_query_floor(term, label):
    assert score(LexicalSimilarity(), term, label) >= QUERY_THRESHOLD


def test_the_measured_gap_around_the_query_floor_still_holds():
    """The floor sits in a gap, not on top of the distribution."""
    similarity = LexicalSimilarity()
    distinct = max(
        score(similarity, a, b)
        for a, b in [
            ("pull request #1347", "pull request #204"),
            ("payment_service", "auth-service"),
            ("auth-service", "order_service"),
        ]
    )
    variant = min(
        score(similarity, a, b)
        for a, b in [
            ("notification_service", "notification-service"),
            ("PR#1290", "PR #1290"),
        ]
    )

    assert distinct < QUERY_THRESHOLD <= variant
