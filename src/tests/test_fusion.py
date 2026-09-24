"""Tests for intent classification, recency decay, and fusion.

No database except where the acceptance cases need one. Classification and
decay are pure functions over values, and fusion takes two finished result
sets — driving a real store for those would test the store.

**Every decay test passes a fixed ``now``.** A decay computed against the wall
clock is a different number tomorrow, so a test written without one passes
today and fails in six months for no reason anybody will be able to find.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.common.config import (
    DEFAULT_HALF_LIFE_DAYS,
    GRAPH_WEIGHT_CONCEPTUAL,
    GRAPH_WEIGHT_RELATIONAL,
    HALF_LIFE_DAYS,
    INTENT_CONCEPTUAL,
    INTENT_RELATIONAL,
    MIN_VECTOR_K,
    NODE_COMMIT,
    NODE_PR,
    NODE_TICKET,
    RECENCY_FLOOR,
    STAGE_FALLBACK,
    STAGE_MARKER,
    STAGE_MODEL,
    TOTAL_K,
    VECTOR_WEIGHT_CONCEPTUAL,
    VECTOR_WEIGHT_RELATIONAL,
)
from src.retrieval import (
    age_days,
    classify,
    decay_factor,
    fuse,
    half_life_for,
    vector_budget,
    weights_for,
)

#: Fixed reference point for every decay assertion in this file.
NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def days_before(days: float) -> datetime:
    return NOW - timedelta(days=days)


class StubHit:
    """Stands in for a VectorHit or a GraphHit, whichever is being fused."""

    def __init__(self, node_id, value):
        self.id = node_id
        self.similarity = value
        self.score = value


class RecordingJudge:
    def __init__(self, answer=True):
        self.answer = answer
        self.calls = 0

    def relational(self, query):
        self.calls += 1
        return self.answer


class RaisingJudge:
    def __init__(self):
        self.calls = 0

    def relational(self, query):
        self.calls += 1
        raise RuntimeError("the model is unreachable")


# ==========================================================================
# Intent classification
# ==========================================================================


def test_a_relational_marker_classifies_without_consulting_the_judge():
    """Stage one is the cheap one, so it must not pay for stage two."""
    judge = RecordingJudge()

    result = classify("who changed the token expiry?", judge)

    assert result.intent == INTENT_RELATIONAL
    assert result.stage == STAGE_MARKER
    assert judge.calls == 0


def test_a_semantic_marker_classifies_without_consulting_the_judge():
    judge = RecordingJudge()

    result = classify("explain the retrieval architecture", judge)

    assert result.intent == INTENT_CONCEPTUAL
    assert result.stage == STAGE_MARKER
    assert judge.calls == 0


def test_no_marker_match_consults_the_judge_exactly_once():
    judge = RecordingJudge(answer=True)

    result = classify("token expiry handling in checkout", judge)

    assert judge.calls == 1
    assert result.stage == STAGE_MODEL
    assert result.intent == INTENT_RELATIONAL


def test_the_judge_can_answer_conceptual():
    judge = RecordingJudge(answer=False)

    result = classify("token expiry handling in checkout", judge)

    assert result.intent == INTENT_CONCEPTUAL
    assert result.stage == STAGE_MODEL


def test_a_judge_failure_falls_back_to_conceptual_and_does_not_raise():
    """A wrong weighting returns worse results; an exception returns none."""
    judge = RaisingJudge()

    result = classify("token expiry handling in checkout", judge)

    assert judge.calls == 1
    assert result.intent == INTENT_CONCEPTUAL
    assert result.stage == STAGE_FALLBACK


def test_no_judge_means_an_unmatched_query_takes_the_fallback():
    result = classify("token expiry handling in checkout", None)

    assert result.intent == INTENT_CONCEPTUAL
    assert result.stage == STAGE_FALLBACK


def test_the_stage_distinguishes_a_fallback_from_a_judged_answer():
    """Both weight the query the same way and mean different things."""
    judged = classify("token expiry handling", RecordingJudge(answer=False))
    fell_back = classify("token expiry handling", None)

    assert judged.intent == fell_back.intent
    assert judged.stage != fell_back.stage


def test_a_relational_marker_wins_over_a_semantic_one():
    """The thing a query names is more specific than the thing it describes."""
    result = classify("who explains the architecture?", RecordingJudge())

    assert result.intent == INTENT_RELATIONAL


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_an_empty_query_is_conceptual_by_fallback(query):
    judge = RecordingJudge()

    result = classify(query, judge)

    assert result.intent == INTENT_CONCEPTUAL
    assert result.stage == STAGE_FALLBACK
    assert judge.calls == 0


def test_the_matched_marker_is_reported():
    result = classify("WHO wrote this?", RecordingJudge())

    assert result.marker == "who"


# ==========================================================================
# Weights
# ==========================================================================


def test_weights_are_applied_per_classification():
    assert weights_for(INTENT_RELATIONAL) == (
        VECTOR_WEIGHT_RELATIONAL,
        GRAPH_WEIGHT_RELATIONAL,
    )
    assert weights_for(INTENT_CONCEPTUAL) == (
        VECTOR_WEIGHT_CONCEPTUAL,
        GRAPH_WEIGHT_CONCEPTUAL,
    )


def test_relational_weighting_favours_the_graph_arm():
    alpha, beta = weights_for(INTENT_RELATIONAL)

    assert beta > alpha


def test_conceptual_weighting_favours_the_vector_arm():
    alpha, beta = weights_for(INTENT_CONCEPTUAL)

    assert alpha > beta


# ==========================================================================
# Recency decay
# ==========================================================================


def test_a_known_recent_timestamp_decays_barely():
    factor = decay_factor(days_before(1), NODE_PR, now=NOW)

    assert factor > 0.98
    assert factor < 1.0


def test_a_known_old_timestamp_sits_at_the_floor():
    """Ten years of a 21-day half-life is far past anything the floor allows."""
    factor = decay_factor(days_before(3650), NODE_TICKET, now=NOW)

    assert factor == RECENCY_FLOOR


def test_an_unknown_timestamp_is_exactly_one_not_the_floor():
    """Unknown means no penalty, not maximum penalty.

    An undated node has said nothing about its age. Ranking it below a node
    known to be ancient would be inventing evidence against it.
    """
    assert decay_factor(None, NODE_TICKET, now=NOW) == 1.0
    assert decay_factor(None, NODE_TICKET, now=NOW) != RECENCY_FLOOR


def test_a_future_timestamp_is_exactly_one_and_earns_no_bonus():
    """Clock skew between systems is ordinary; a future date is a quirk."""
    future = NOW + timedelta(days=365)

    assert decay_factor(future, NODE_PR, now=NOW) == 1.0


def test_age_is_floored_at_zero():
    assert age_days(NOW + timedelta(days=10), now=NOW) == 0.0


def test_an_unknown_age_stays_unknown_rather_than_becoming_zero():
    """Both collapsing to 0.0 would make an undated node look newest."""
    assert age_days(None, now=NOW) is None
    assert age_days(NOW, now=NOW) == 0.0


def test_one_half_life_gives_exactly_a_half():
    factor = decay_factor(days_before(HALF_LIFE_DAYS[NODE_PR]), NODE_PR, now=NOW)

    assert factor == pytest.approx(0.5)


def test_decay_uses_the_types_own_half_life():
    """A ticket at 45 days is staler than a commit at 45 days."""
    ticket = decay_factor(days_before(45), NODE_TICKET, now=NOW)
    commit = decay_factor(days_before(45), NODE_COMMIT, now=NOW)

    assert ticket < commit
    assert commit == pytest.approx(0.5)


def test_an_unknown_type_uses_the_default_half_life():
    unknown = decay_factor(days_before(DEFAULT_HALF_LIFE_DAYS), "Nonesuch", now=NOW)
    absent = decay_factor(days_before(DEFAULT_HALF_LIFE_DAYS), None, now=NOW)

    assert unknown == pytest.approx(0.5)
    assert absent == pytest.approx(0.5)


def test_half_life_lookup_falls_back_for_an_unknown_type():
    assert half_life_for("Nonesuch") == DEFAULT_HALF_LIFE_DAYS
    assert half_life_for(None) == DEFAULT_HALF_LIFE_DAYS
    assert half_life_for(NODE_TICKET) == HALF_LIFE_DAYS[NODE_TICKET]


def test_decay_never_falls_below_the_floor():
    for days in (100, 1000, 10_000, 100_000):
        assert decay_factor(days_before(days), NODE_TICKET, now=NOW) >= RECENCY_FLOOR


def test_recency_disabled_returns_one_for_everything(monkeypatch):
    monkeypatch.setattr("src.retrieval.recency.RECENCY_ENABLED", False)

    assert decay_factor(days_before(10_000), NODE_TICKET, now=NOW) == 1.0
    assert decay_factor(None, NODE_TICKET, now=NOW) == 1.0
    assert decay_factor(days_before(1), NODE_PR, now=NOW) == 1.0


def test_an_epoch_number_is_accepted_as_well_as_a_datetime():
    """The store hands back datetimes; a raw column value is a number."""
    as_datetime = decay_factor(days_before(30), NODE_PR, now=NOW)
    as_number = decay_factor(days_before(30).timestamp(), NODE_PR, now=NOW.timestamp())

    assert as_datetime == pytest.approx(as_number)


# ==========================================================================
# Result-set balancing
# ==========================================================================


def test_the_vector_budget_shrinks_as_graph_hits_grow():
    assert vector_budget(0, total_k=10) == 10
    assert vector_budget(3, total_k=10) == 7
    assert vector_budget(8, total_k=10) == 2


def test_the_vector_budget_never_falls_below_the_floor():
    """A query whose traversal reaches everything still carries text evidence."""
    assert vector_budget(10, total_k=10) == MIN_VECTOR_K
    assert vector_budget(50, total_k=10) == MIN_VECTOR_K
    assert vector_budget(1000, total_k=10) == MIN_VECTOR_K


def test_the_budget_holds_when_graph_hits_exceed_the_total():
    result = fuse(
        [StubHit(f"v{n}", 0.9) for n in range(10)],
        [StubHit(f"g{n}", 0.9) for n in range(20)],
        INTENT_CONCEPTUAL,
        total_k=10,
        now=NOW,
    )

    assert result.graph_hits == 20
    assert result.vector_k == MIN_VECTOR_K


def test_the_budget_limits_what_enters_the_pool_not_what_leaves_it():
    """Trimming after fusing would make the reported vector_k a lie."""
    result = fuse(
        [StubHit(f"v{n}", 0.9) for n in range(10)],
        [StubHit(f"g{n}", 0.1) for n in range(9)],
        INTENT_CONCEPTUAL,
        total_k=10,
        now=NOW,
    )

    vector_sourced = [hit for hit in result.hits if hit.vector_score > 0]

    assert result.vector_k == MIN_VECTOR_K
    assert len(vector_sourced) <= MIN_VECTOR_K


# ==========================================================================
# Fusion
# ==========================================================================


def test_a_node_found_by_one_arm_keeps_a_zero_from_the_other():
    """Absence from an arm is information, not a missing value."""
    result = fuse(
        [StubHit("vector-only", 0.8)],
        [StubHit("graph-only", 0.9)],
        INTENT_CONCEPTUAL,
        now=NOW,
    )
    by_id = {hit.id: hit for hit in result.hits}

    assert by_id["vector-only"].graph_score == 0.0
    assert by_id["graph-only"].vector_score == 0.0
    assert not by_id["vector-only"].found_by_both


def test_a_single_arm_score_is_not_renormalised():
    """0.8 found by one arm scores alpha * 0.8, not 0.8."""
    alpha, _ = weights_for(INTENT_CONCEPTUAL)

    result = fuse([StubHit("only", 0.8)], [], INTENT_CONCEPTUAL, now=NOW)

    assert result.hits[0].score == pytest.approx(alpha * 0.8)


def test_a_node_found_by_both_arms_sums_the_weighted_parts():
    alpha, beta = weights_for(INTENT_RELATIONAL)

    result = fuse(
        [StubHit("both", 0.5)], [StubHit("both", 0.9)], INTENT_RELATIONAL, now=NOW
    )

    assert len(result.hits) == 1
    assert result.hits[0].score == pytest.approx(alpha * 0.5 + beta * 0.9)
    assert result.hits[0].found_by_both


def test_decay_multiplies_the_fused_score():
    alpha, _ = weights_for(INTENT_CONCEPTUAL)
    attributes = {"aged": {"type": NODE_PR, "timestamp": days_before(60)}}

    result = fuse(
        [StubHit("aged", 1.0)],
        [],
        INTENT_CONCEPTUAL,
        attributes=attributes,
        now=NOW,
    )

    assert result.hits[0].decay == pytest.approx(0.5)
    assert result.hits[0].score == pytest.approx(alpha * 1.0 * 0.5)


def test_a_node_with_no_attributes_is_not_penalised():
    result = fuse([StubHit("unknown", 1.0)], [], INTENT_CONCEPTUAL, now=NOW)

    assert result.hits[0].decay == 1.0


def test_results_are_ranked_by_total_score_descending():
    result = fuse(
        [StubHit("low", 0.1), StubHit("high", 0.9), StubHit("mid", 0.5)],
        [],
        INTENT_CONCEPTUAL,
        now=NOW,
    )

    assert result.ids == ["high", "mid", "low"]


def test_the_set_is_capped_at_total_k():
    result = fuse(
        [StubHit(f"v{n}", 0.5) for n in range(20)],
        [],
        INTENT_CONCEPTUAL,
        total_k=4,
        now=NOW,
    )

    assert len(result) == 4


def test_the_components_survive_into_the_result():
    """A total of 0.42 cannot say which arm earned it."""
    result = fuse(
        [StubHit("x", 0.6)], [StubHit("x", 0.4)], INTENT_RELATIONAL, now=NOW
    )
    hit = result.hits[0]

    assert hit.vector_score == 0.6
    assert hit.graph_score == 0.4
    assert hit.decay == 1.0
    assert result.alpha, result.beta == weights_for(INTENT_RELATIONAL)


def test_fusing_nothing_produces_nothing():
    result = fuse([], [], INTENT_CONCEPTUAL, now=NOW)

    assert result.hits == []
    assert result.vector_k == TOTAL_K


# ==========================================================================
# The acceptance cases
#
# These are B12's two constructed cases, which separate the arms as sharply as
# a corpus can: one query where the graph arm scores 0.92 and the vector arm
# 0.0107, and one where the vector arm ranks the answer first at 0.7231 and
# the graph arm returns nothing at all.
#
# They are what the weights exist for. **If the weights cannot rank the right
# answer first on cases built to separate this cleanly, the weights are wrong
# rather than the corpus being hard** — and the correct response is to report
# that, not to adjust the weights until these two pass. Two data points cannot
# justify a weighting; they can only falsify one.
# ==========================================================================


def test_case_one_relational_the_graph_arms_answer_ranks_first():
    """A query naming an entity whose neighbour shares no text with it.

    Measured in B12: graph arm reaches ``pr:9`` at 0.92; the vector arm scores
    it 0.0107 and puts ``ticket:4821`` — the thing named, not the answer — at
    1.0. Relational weighting has to let the traversal result win.
    """
    query = "who fixed #4821?"
    intent = classify(query, None)

    assert intent.intent == INTENT_RELATIONAL
    assert intent.stage == STAGE_MARKER

    result = fuse(
        [StubHit("ticket:4821", 1.0), StubHit("pr:9", 0.0107)],
        [StubHit("pr:9", 0.92)],
        intent.intent,
        now=NOW,
    )

    assert result.ids[0] == "pr:9"


def test_case_two_conceptual_the_vector_arms_answer_ranks_first():
    """A query describing a concept and naming nothing.

    Measured in B12: the vector arm ranks ``pr:77`` first at 0.7231 with no
    shared vocabulary, and the graph arm has no seed so returns nothing.
    Conceptual weighting has to let the similarity result win.
    """
    query = "how was excessive RAM usage when uploading big files addressed?"
    intent = classify(query, None)

    assert intent.intent == INTENT_CONCEPTUAL

    result = fuse(
        [StubHit("pr:77", 0.7231), StubHit("pr:78", -0.0285)],
        [],
        intent.intent,
        now=NOW,
    )

    assert result.ids[0] == "pr:77"


def test_case_one_would_rank_wrongly_under_conceptual_weighting():
    """Why the classification matters, not just the weights.

    The same two arms fused as conceptual put the named ticket first and the
    answer second. The weighting is doing real work here, and misclassifying
    this query is what would break it.
    """
    result = fuse(
        [StubHit("ticket:4821", 1.0), StubHit("pr:9", 0.0107)],
        [StubHit("pr:9", 0.92)],
        INTENT_CONCEPTUAL,
        now=NOW,
    )

    assert result.ids[0] == "ticket:4821"
