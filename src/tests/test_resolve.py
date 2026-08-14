"""Tests for entity resolution.

Everything here runs on a **stub embedder**, not a stub scorer. The real
``Similarity.scores`` — normalise, then dot product — executes in every test
below; only the vectors are fabricated. That is the point of putting the seam
at vector production: which band a score falls in, which way each comparison
points, what a failure does and which label survives are all properties of the
resolution logic, and none of them should change with the source of a number.

It also removes a class of test that could previously be written but never
happen: stubbing the scoring function allowed identical text to score less than
1.0, which no embedder can produce. See
``test_a_repeat_never_reaches_the_model_whatever_the_band_is_set_to``.

The judge is a counting stub for a different reason — "the model was not
called" is an assertion that cannot be made against a real one.
"""

from __future__ import annotations

import math

import pytest

from datetime import datetime, timedelta, timezone

from src.analysis import Entity, ResolvedEntity, Resolver, Similarity
from src.common.config import (
    DEEP_THRESHOLD,
    ENTITY_PERSON,
    ENTITY_SERVICE,
    ENTITY_TICKET,
    FAST_THRESHOLD,
    QUERY_THRESHOLD,
)


class StubEmbedder:
    """Places dense vectors so that distinct texts score a chosen cosine.

    The seam is at vector production, so resolution's thresholds are driven by
    geometry rather than by a stubbed-out scoring function — the real
    ``Similarity.scores`` runs in every test below.

    Uniform mode gives every text weight ``sqrt(s)`` on a shared axis plus
    ``sqrt(1 - s)`` on an axis of its own, which makes every distinct pair
    score exactly ``s`` and each text score 1.0 with itself.

    ``by_candidate`` mode puts the named texts at a chosen cosine from the
    shared axis and everything else entirely on it, so an unnamed text scores
    each named one at its own value. Two unnamed texts then score 1.0 with each
    other, so do not put two of the same type in one resolver unless a merge is
    what you want.
    """

    dimension = 64
    identity = "stub"

    def __init__(
        self,
        score: float = 0.0,
        *,
        by_candidate: dict[str, float] | None = None,
    ):
        self.score = score
        self.by_candidate = by_candidate or {}
        self._axes: dict[str, int] = {}

    def _axis(self, text: str) -> int:
        """A private dimension per text. Axis 0 is the shared one."""
        if text not in self._axes:
            self._axes[text] = 1 + len(self._axes)
        return self._axes[text]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension

        if self.by_candidate:
            if text not in self.by_candidate:
                vector[0] = 1.0
                return vector
            target = self.by_candidate[text]
            vector[0] = target
            vector[self._axis(text)] = math.sqrt(max(0.0, 1.0 - target * target))
            return vector

        vector[0] = math.sqrt(self.score)
        vector[self._axis(text)] = math.sqrt(max(0.0, 1.0 - self.score))
        return vector


class StubSimilarity(Similarity):
    """Real scoring on stub vectors, counting the calls made into it."""

    def __init__(self, score: float = 0.0, *, by_candidate: dict[str, float] | None = None):
        super().__init__(StubEmbedder(score, by_candidate=by_candidate))
        self.calls: list[tuple[str, list[str]]] = []

    def scores(self, text, candidates):
        self.calls.append((text, list(candidates)))
        return super().scores(text, candidates)


class StubJudge:
    """Answers as instructed, and records that it was asked."""

    def __init__(self, answer=True):
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def same(self, left, right):
        self.calls.append((left, right))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def entity(text: str, kind: str = ENTITY_SERVICE) -> Entity:
    return Entity(text=text, type=kind, score=1.0, start=0, end=len(text), source="rules")


# --------------------------------------------------------------------------
# the three paths
# --------------------------------------------------------------------------


def test_a_score_above_the_fast_threshold_merges_without_a_model_call():
    similarity = StubSimilarity(0.97)
    judge = StubJudge()
    resolver = Resolver(similarity, judge)

    resolver.add(entity("payment_service"))
    resolver.add(entity("payment-service"))

    assert judge.calls == []
    assert len(resolver.entities) == 1
    assert resolver.stats.fast_merges == 1
    assert resolver.stats.model_calls == 0


def test_a_score_in_the_band_calls_the_model_exactly_once():
    judge = StubJudge(answer=True)
    resolver = Resolver(StubSimilarity(0.88), judge)

    resolver.add(entity("payment_service"))
    resolver.add(entity("billing_service"))

    assert len(judge.calls) == 1
    assert resolver.stats.model_calls == 1


def test_a_model_answering_yes_merges():
    resolver = Resolver(StubSimilarity(0.88), StubJudge(answer=True))

    resolver.add(entity("payment_service"))
    resolver.add(entity("billing_service"))

    assert len(resolver.entities) == 1
    assert resolver.stats.model_merges == 1
    assert resolver.stats.model_rejections == 0


def test_a_model_answering_no_creates_a_new_entity():
    resolver = Resolver(StubSimilarity(0.88), StubJudge(answer=False))

    resolver.add(entity("payment_service"))
    resolver.add(entity("billing_service"))

    assert len(resolver.entities) == 2
    assert resolver.stats.model_rejections == 1
    assert resolver.stats.model_merges == 0


def test_a_score_below_the_deep_threshold_creates_a_new_entity_with_no_call():
    judge = StubJudge()
    resolver = Resolver(StubSimilarity(0.4), judge)

    resolver.add(entity("payment_service"))
    resolver.add(entity("auth_service"))

    assert judge.calls == []
    assert len(resolver.entities) == 2
    assert resolver.stats.model_calls == 0
    assert resolver.stats.created == 2


def test_the_first_entity_is_created_without_scoring_anything():
    similarity = StubSimilarity(0.99)
    resolver = Resolver(similarity, StubJudge())

    resolver.add(entity("payment_service"))

    assert similarity.calls == []
    assert resolver.stats.comparisons == 0


# --------------------------------------------------------------------------
# failing toward the recoverable error
# --------------------------------------------------------------------------


def test_a_model_that_raises_creates_a_new_entity():
    resolver = Resolver(StubSimilarity(0.88), StubJudge(answer=RuntimeError("timeout")))

    resolver.add(entity("payment_service"))
    resolver.add(entity("billing_service"))

    assert len(resolver.entities) == 2
    assert resolver.stats.failures == 1
    assert resolver.stats.model_merges == 0


@pytest.mark.parametrize("answer", ["yes", 1, None, "true"])
def test_an_unparseable_answer_never_merges(answer):
    """A truthy string is not a yes. Only a real bool is an answer."""
    resolver = Resolver(StubSimilarity(0.88), StubJudge(answer=answer))

    resolver.add(entity("payment_service"))
    resolver.add(entity("billing_service"))

    assert len(resolver.entities) == 2
    assert resolver.stats.failures == 1


def test_no_judge_at_all_never_merges_in_the_band():
    resolver = Resolver(StubSimilarity(0.88), judge=None)

    resolver.add(entity("payment_service"))
    resolver.add(entity("billing_service"))

    assert len(resolver.entities) == 2
    assert resolver.stats.failures == 1


def test_no_judge_does_not_block_a_fast_merge():
    """The safe default must not disable the path that needs no model."""
    resolver = Resolver(StubSimilarity(0.99), judge=None)

    resolver.add(entity("payment_service"))
    resolver.add(entity("payment-service"))

    assert len(resolver.entities) == 1


# --------------------------------------------------------------------------
# the boundaries, pinned in both directions
# --------------------------------------------------------------------------


def test_a_score_exactly_on_the_fast_threshold_merges_without_a_model():
    judge = StubJudge()
    resolver = Resolver(StubSimilarity(FAST_THRESHOLD), judge)

    resolver.add(entity("a_service"))
    resolver.add(entity("b_service"))

    assert resolver.stats.fast_merges == 1
    assert judge.calls == []


def test_a_score_exactly_on_the_deep_threshold_asks_the_model():
    judge = StubJudge(answer=True)
    resolver = Resolver(StubSimilarity(DEEP_THRESHOLD), judge)

    resolver.add(entity("a_service"))
    resolver.add(entity("b_service"))

    assert len(judge.calls) == 1


def test_just_below_the_fast_threshold_asks_rather_than_merging():
    """Fails if >= flips to >. One ULP below the boundary."""
    judge = StubJudge(answer=True)
    resolver = Resolver(StubSimilarity(FAST_THRESHOLD - 1e-9), judge)

    resolver.add(entity("a_service"))
    resolver.add(entity("b_service"))

    assert resolver.stats.fast_merges == 0
    assert len(judge.calls) == 1


def test_just_below_the_deep_threshold_creates_rather_than_asking():
    """Fails if >= flips to > on the deep comparison."""
    judge = StubJudge()
    resolver = Resolver(StubSimilarity(DEEP_THRESHOLD - 1e-9), judge)

    resolver.add(entity("a_service"))
    resolver.add(entity("b_service"))

    assert judge.calls == []
    assert resolver.stats.created == 2


def test_a_deep_threshold_above_the_fast_one_is_refused():
    with pytest.raises(ValueError, match="cannot exceed"):
        Resolver(StubSimilarity(), fast=0.8, deep=0.9)


def test_the_shipped_thresholds_leave_a_band():
    assert DEEP_THRESHOLD < FAST_THRESHOLD


# --------------------------------------------------------------------------
# type isolation
# --------------------------------------------------------------------------


def test_candidates_of_a_different_type_are_excluded_before_scoring():
    """A person and a service with the same name are never compared at all."""
    similarity = StubSimilarity(0.99)
    resolver = Resolver(similarity, StubJudge())

    resolver.add(entity("Sonic", ENTITY_PERSON))
    resolver.add(entity("Sonic", ENTITY_SERVICE))

    assert len(resolver.entities) == 2
    # Not merely rejected afterwards — never offered as a candidate.
    assert similarity.calls == []


def test_scoring_only_ever_sees_candidates_of_the_matching_type():
    similarity = StubSimilarity(0.1)
    resolver = Resolver(similarity, StubJudge())

    resolver.add(entity("alpha_service", ENTITY_SERVICE))
    resolver.add(entity("Alice", ENTITY_PERSON))
    resolver.add(entity("beta_service", ENTITY_SERVICE))

    surface, candidates = similarity.calls[-1]
    assert surface == "beta_service"
    assert candidates == ["alpha_service"]


def test_each_type_keeps_its_own_canonical_set():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    resolver.add(entity("#412", ENTITY_TICKET))
    resolver.add(entity("#413", ENTITY_TICKET))
    resolver.add(entity("payment_service", ENTITY_SERVICE))

    assert {e.type for e in resolver.entities} == {ENTITY_TICKET, ENTITY_SERVICE}


# --------------------------------------------------------------------------
# the merge record
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# order independence
#
# The rule: most frequent surface form wins; ties break on sorted order. Both
# are properties of the set, so neither can depend on which document arrived
# first.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order",
    [
        ("a_service", "b_service", "c_service"),
        ("c_service", "b_service", "a_service"),
        ("b_service", "c_service", "a_service"),
    ],
)
def test_three_input_orders_give_the_same_entity_set(order):
    """Same count, same types, same merges — whatever order they arrive in.

    The label is deliberately *not* asserted. With write-once labels it is
    whichever form was seen first, so it varies with document order, and that
    is accepted behaviour rather than a regression. What must not vary is the
    shape of the result.
    """
    resolver = Resolver(StubSimilarity(0.99), StubJudge())
    for name in order:
        resolver.add(entity(name))

    assert len(resolver.entities) == 1
    assert [e.type for e in resolver.entities] == [ENTITY_SERVICE]
    assert resolver.stats.merges == 2
    assert resolver.stats.created == 1
    # And the label is one of the forms that arrived, not something invented.
    assert resolver.entities[0].canonical == order[0]


def test_orders_agree_on_the_set_across_a_mixed_corpus():
    """Several groups and types at once, not one group in isolation."""
    from itertools import permutations

    forms = [
        ("payment_service", ENTITY_SERVICE),
        ("payment-service", ENTITY_SERVICE),
        ("#412", ENTITY_TICKET),
        ("alice", ENTITY_PERSON),
    ]
    shapes = set()

    for order in permutations(forms):
        resolver = Resolver(
            StubSimilarity(
                0.0, by_candidate={"payment_service": 0.99, "payment-service": 0.99}
            ),
            StubJudge(),
        )
        for text, kind in order:
            resolver.add(entity(text, kind))
        shapes.add(
            (
                len(resolver.entities),
                tuple(sorted(e.type for e in resolver.entities)),
                resolver.stats.merges,
                resolver.stats.created,
            )
        )

    assert len(shapes) == 1


# --------------------------------------------------------------------------
# the case rule — frequency, then not-all-uppercase, then string order
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# the reverse index — what makes the canonical choice cosmetic
# --------------------------------------------------------------------------


def test_the_record_serialises_to_plain_data():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())
    resolver.add(entity("payment_service"))
    resolver.add(entity("payment-service"))

    payload = resolver.to_dict()

    assert payload["entities"] == [
        {
            "canonical": "payment_service",
            "type": ENTITY_SERVICE,
            "timestamp": None,
        }
    ]
    assert payload["stats"]["fast_merges"] == 1


def test_the_record_carries_no_variant_machinery():
    """No alias list, no per-merge score, no deciding path."""
    resolver = Resolver(StubSimilarity(0.99), StubJudge())
    resolver.add(entity("payment_service"))
    resolver.add(entity("payment-service"))

    entry = resolver.to_dict()["entities"][0]

    assert set(entry) == {"canonical", "type", "timestamp"}
    assert not hasattr(resolver, "lookup")
    assert not hasattr(resolver.entities[0], "variants")


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_a_run_that_merges_nothing_is_distinguishable_from_one_that_merges_all():
    nothing = Resolver(StubSimilarity(0.0), StubJudge())
    everything = Resolver(StubSimilarity(1.0), StubJudge())

    for resolver in (nothing, everything):
        for name in ("a_service", "b_service", "c_service"):
            resolver.add(entity(name))

    assert (nothing.stats.created, nothing.stats.merges) == (3, 0)
    assert (everything.stats.created, everything.stats.merges) == (1, 2)


# --------------------------------------------------------------------------
# repeats are not resolution
#
# The same string seen in two documents is a merge, but it discovers no
# variant and makes no new lookup possible. Counting it alongside real merges
# overstates what resolution achieved.
# --------------------------------------------------------------------------


def test_an_exact_repeat_counts_as_a_repeat_not_a_merge():
    resolver = Resolver(StubSimilarity(1.0), StubJudge())

    resolver.add(entity("payment_service"))
    resolver.add(entity("payment_service"))

    assert resolver.stats.repeats == 1
    assert resolver.stats.fast_merges == 0
    assert resolver.stats.variant_merges == 0


def test_a_distinct_form_counts_as_a_merge_not_a_repeat():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    resolver.add(entity("payment_service"))
    resolver.add(entity("payment-service"))

    assert resolver.stats.fast_merges == 1
    assert resolver.stats.variant_merges == 1
    assert resolver.stats.repeats == 0


def test_the_two_counters_move_independently():
    """Repeats up, merges flat — then merges up, repeats flat."""
    repeats_only = Resolver(StubSimilarity(1.0), StubJudge())
    for _ in range(3):
        repeats_only.add(entity("payment_service"))

    variants_only = Resolver(StubSimilarity(0.99), StubJudge())
    for name in ("a_service", "b_service", "c_service"):
        variants_only.add(entity(name))

    assert (repeats_only.stats.repeats, repeats_only.stats.variant_merges) == (2, 0)
    assert (variants_only.stats.repeats, variants_only.stats.variant_merges) == (0, 2)


def test_a_mixed_run_splits_the_counts():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    for name in ("payment_service", "payment_service", "payment-service"):
        resolver.add(entity(name))

    assert resolver.stats.repeats == 1
    assert resolver.stats.variant_merges == 1
    assert resolver.stats.merges == 2


def test_a_repeat_never_reaches_the_model_whatever_the_band_is_set_to():
    """Identical text cannot score below 1.0, so a repeat is always fast.

    This became true when the seam moved to the embedder. The same text yields
    the same vector, and a vector's cosine with itself is exactly 1.0 — so no
    embedder can put a repeat in the ambiguous band. An earlier version of this
    test asserted the opposite, which only a stubbed *scoring* function could
    produce; nothing reachable through a real embedder ever could.
    """
    judge = StubJudge(answer=True)
    resolver = Resolver(StubSimilarity(0.88), judge)

    resolver.add(entity("payment_service"))
    resolver.add(entity("payment_service"))

    assert resolver.stats.repeats == 1
    assert resolver.stats.fast_merges == 0
    assert resolver.stats.model_calls == 0
    assert judge.calls == []


def test_identical_text_scores_exactly_one_through_the_real_scorer():
    """The geometric fact the test above rests on."""
    similarity = StubSimilarity(0.4)

    assert similarity.scores("payment_service", ["payment_service"])[0] == 1.0


def test_the_totals_reconcile_with_what_was_seen():
    """seen == created + variant merges + repeats. No entity is unaccounted for."""
    resolver = Resolver(StubSimilarity(0.99), StubJudge())
    for name in ("a_service", "a_service", "b_service", "c_service"):
        resolver.add(entity(name))
    resolver.add(entity("#42", ENTITY_TICKET))

    stats = resolver.stats
    assert stats.seen == stats.created + stats.variant_merges + stats.repeats


def test_both_counts_are_serialised_separately():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())
    for name in ("payment_service", "payment_service", "payment-service"):
        resolver.add(entity(name))

    stats = resolver.to_dict()["stats"]

    assert stats["repeats"] == 1
    assert stats["variant_merges"] == 1
    assert stats["merges"] == 2


def test_every_entity_seen_is_counted():
    resolver = Resolver(StubSimilarity(0.5), StubJudge())

    resolver.resolve([entity("a_service"), entity("b_service")])

    assert resolver.stats.seen == 2


def test_the_band_fraction_reports_what_reached_the_model():
    resolver = Resolver(StubSimilarity(0.88), StubJudge(answer=False))

    for name in ("a_service", "b_service", "c_service"):
        resolver.add(entity(name))

    # Two entities had candidates; both landed in the band.
    assert resolver.stats.model_calls == 2
    assert resolver.stats.band_fraction > 0


def test_the_band_fraction_is_zero_when_nothing_was_compared():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    resolver.add(entity("only_service"))

    assert resolver.stats.band_fraction == 0.0


def test_resolve_returns_the_canonical_set():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    result = resolver.resolve([entity("a_service"), entity("b_service")])

    assert result == resolver.entities
    assert len(result) == 1


# --------------------------------------------------------------------------
# query-time resolution
#
# The same similarity function as merging, at a much looser floor. This is what
# replaces the alias index: variant spellings survive as a scoring property
# rather than as stored data.
# --------------------------------------------------------------------------


def test_a_query_term_matching_a_label_exactly_resolves_to_it():
    """Real similarity: a stub cannot express "identical strings score 1.0"."""
    resolver = Resolver()
    resolver.add(entity("notification-service"))

    assert resolver.find("notification-service") is resolver.entities[0]


def test_an_orthographic_variant_resolves_through_similarity_not_a_table():
    """Real lexical similarity, no stub: the variant is not stored anywhere."""
    resolver = Resolver()
    resolver.add(entity("notification-service"))

    found = resolver.find("notification_service")

    assert found is resolver.entities[0]
    assert found.canonical == "notification-service"


@pytest.mark.parametrize(
    "term,label",
    [
        ("notification_service", "notification-service"),
        ("PR#1290", "PR #1290"),
        ("ORDER_SERVICE", "order_service"),
    ],
)
def test_the_demo_corpus_variants_all_resolve(term, label):
    """The three pairs the corpus actually produces."""
    resolver = Resolver()
    resolver.add(entity(label))

    assert resolver.find(term) is resolver.entities[0]


def test_a_term_below_the_query_floor_resolves_to_nothing():
    resolver = Resolver(StubSimilarity(QUERY_THRESHOLD - 1e-9), StubJudge())
    resolver.add(entity("notification-service"))

    assert resolver.find("something else entirely") is None


def test_a_term_exactly_on_the_query_floor_resolves():
    """Pins the comparison direction, as the merge thresholds are pinned."""
    resolver = Resolver(StubSimilarity(QUERY_THRESHOLD), StubJudge())
    resolver.add(entity("notification-service"))

    assert resolver.find("anything") is resolver.entities[0]


def test_an_unrelated_term_does_not_reach_a_real_entity():
    """Measured: 'payment_service' vs 'auth-service' is 0.5186, under the floor."""
    resolver = Resolver()
    resolver.add(entity("auth-service"))

    assert resolver.find("payment_service") is None
    assert resolver.find("database") is None


def test_finding_in_an_empty_resolver_returns_nothing():
    assert Resolver().find("anything") is None


def test_the_best_scoring_label_is_selected_among_several_above_the_floor():
    similarity = StubSimilarity(
        0.0, by_candidate={"alpha_service": 0.85, "beta_service": 0.95}
    )
    resolver = Resolver(similarity, StubJudge())
    resolver._create(entity("alpha_service"))
    resolver._create(entity("beta_service"))

    assert resolver.find("query").canonical == "beta_service"


def test_a_tie_above_the_floor_is_broken_deterministically():
    """On the label, not on insertion order — so both orders agree."""
    forward = Resolver(StubSimilarity(0.99), StubJudge())
    forward._create(entity("zeta_service"))
    forward._create(entity("alpha_service"))

    backward = Resolver(StubSimilarity(0.99), StubJudge())
    backward._create(entity("alpha_service"))
    backward._create(entity("zeta_service"))

    assert forward.find("q").canonical == backward.find("q").canonical == "alpha_service"


def test_a_query_can_be_narrowed_by_type():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())
    resolver._create(entity("Sonic", ENTITY_PERSON))
    resolver._create(entity("Sonic", ENTITY_SERVICE))

    assert resolver.find("Sonic", ENTITY_PERSON).type == ENTITY_PERSON
    assert resolver.find("Sonic", ENTITY_SERVICE).type == ENTITY_SERVICE


def test_the_query_floor_is_independent_of_the_merge_thresholds():
    """Changing one must not move the other."""
    loose = Resolver(StubSimilarity(0.5), StubJudge(), query_floor=0.4)
    strict = Resolver(StubSimilarity(0.5), StubJudge(), query_floor=0.9)

    for resolver in (loose, strict):
        resolver.add(entity("payment_service"))
        resolver.add(entity("auth_service"))

    # Same merge behaviour: 0.5 is below both merge thresholds in each.
    assert loose.stats.created == strict.stats.created == 2
    # Different query behaviour, from the floor alone.
    assert loose.find("anything") is not None
    assert strict.find("anything") is None


def test_changing_the_merge_thresholds_does_not_move_the_query_floor():
    resolver = Resolver(StubSimilarity(0.82), StubJudge(), fast=0.6, deep=0.5)

    resolver.add(entity("payment_service"))
    resolver.add(entity("payment-service"))

    # Merged, because the merge thresholds were lowered.
    assert len(resolver.entities) == 1
    # Still below the untouched query floor of 0.80... and 0.82 is above it.
    assert resolver.query_floor == QUERY_THRESHOLD


def test_the_query_floor_is_looser_than_both_merge_thresholds():
    assert QUERY_THRESHOLD < DEEP_THRESHOLD < FAST_THRESHOLD


# --------------------------------------------------------------------------
# write-once label and type
#
# A later surface form is not better evidence than the first one, only later.
# Nothing may overwrite a canonical node with a noisier spelling.
# --------------------------------------------------------------------------


def test_creating_an_entity_sets_its_label_and_type():
    resolver = Resolver(StubSimilarity(0.0), StubJudge())

    resolved = resolver.add(entity("payment_service", ENTITY_SERVICE))

    assert resolved.canonical == "payment_service"
    assert resolved.type == ENTITY_SERVICE


def test_merging_leaves_the_label_unchanged_even_when_the_new_form_is_noisier():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    resolver.add(entity("payment_service"))
    resolver.add(entity("PAYMENT_SERVICE"))
    resolver.add(entity("  Payment-Service  "))

    assert resolver.entities[0].canonical == "payment_service"


def test_merging_leaves_the_type_unchanged():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    resolver.add(entity("payment_service", ENTITY_SERVICE))
    resolver.add(entity("payment-service", ENTITY_SERVICE))

    assert resolver.entities[0].type == ENTITY_SERVICE


def test_a_model_decided_merge_also_leaves_the_label_alone():
    """Both merge paths are write-once, not just the fast one."""
    resolver = Resolver(StubSimilarity(0.88), StubJudge(answer=True))

    resolver.add(entity("payment_service"))
    resolver.add(entity("PAYMENT_SVC"))

    assert resolver.entities[0].canonical == "payment_service"


def test_the_label_is_whichever_form_arrived_first():
    """Order-dependent, and accepted. find() is what makes it not matter."""
    forward = Resolver(StubSimilarity(0.99), StubJudge())
    forward.add(entity("ORDER_SERVICE"))
    forward.add(entity("order_service"))

    backward = Resolver(StubSimilarity(0.99), StubJudge())
    backward.add(entity("order_service"))
    backward.add(entity("ORDER_SERVICE"))

    assert forward.entities[0].canonical == "ORDER_SERVICE"
    assert backward.entities[0].canonical == "order_service"


def test_either_label_is_reachable_by_either_spelling():
    """Why the differing label above does not matter. Real similarity."""
    for first, second in [
        ("ORDER_SERVICE", "order_service"),
        ("order_service", "ORDER_SERVICE"),
    ]:
        resolver = Resolver()
        resolver.add(entity(first))
        resolver.add(entity(second))

        assert resolver.find("order_service") is resolver.entities[0]
        assert resolver.find("ORDER_SERVICE") is resolver.entities[0]


# --------------------------------------------------------------------------
# timestamps move forward only
# --------------------------------------------------------------------------

EARLY = datetime(2026, 1, 1, tzinfo=timezone.utc)
LATE = datetime(2026, 6, 1, tzinfo=timezone.utc)


def test_creating_an_entity_records_the_timestamp_it_was_given():
    resolver = Resolver(StubSimilarity(0.0), StubJudge())

    resolved = resolver.add(entity("payment_service"), EARLY)

    assert resolved.timestamp == EARLY


def test_merging_with_a_newer_timestamp_advances_it():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    resolver.add(entity("payment_service"), EARLY)
    resolver.add(entity("payment-service"), LATE)

    assert resolver.entities[0].timestamp == LATE


def test_merging_with_an_older_timestamp_leaves_it_unchanged():
    """A stale document processed last must not drag recency backward."""
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    resolver.add(entity("payment_service"), LATE)
    resolver.add(entity("payment-service"), EARLY)

    assert resolver.entities[0].timestamp == LATE


def test_an_equal_timestamp_leaves_it_unchanged():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    resolver.add(entity("payment_service"), EARLY)
    resolver.add(entity("payment-service"), EARLY)

    assert resolver.entities[0].timestamp == EARLY


def test_an_unknown_incoming_timestamp_never_erases_a_known_one():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    resolver.add(entity("payment_service"), LATE)
    resolver.add(entity("payment-service"), None)

    assert resolver.entities[0].timestamp == LATE


def test_a_known_timestamp_fills_in_an_unknown_one():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    resolver.add(entity("payment_service"), None)
    resolver.add(entity("payment-service"), LATE)

    assert resolver.entities[0].timestamp == LATE


def test_an_unknown_timestamp_stays_none_and_never_becomes_zero():
    """0 reads as 1970 to a recency scorer, which would bury the entity."""
    resolver = Resolver(StubSimilarity(0.99), StubJudge())

    resolver.add(entity("payment_service"))
    resolver.add(entity("payment-service"))

    assert resolver.entities[0].timestamp is None


def test_the_timestamp_is_a_maximum_across_many_merges():
    resolver = Resolver(StubSimilarity(0.99), StubJudge())
    middle = EARLY + timedelta(days=30)

    for stamp in (middle, LATE, EARLY, None, middle):
        resolver.add(entity("payment_service"), stamp)

    assert resolver.entities[0].timestamp == LATE
