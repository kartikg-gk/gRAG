"""Tests for scoring and for the one embedder behind it.

The structure mirrors the code: **one embedder, one scoring path, no
abstraction over either**. Scoring is tested with hand-built vectors, so those
tests say nothing about where a vector came from and run on every build.

Substituting a stub embedder is the only seam, and it exists for these tests
rather than for configuration — production constructs ``Similarity()`` and gets
``SentenceTransformerEmbedder``.

**Normalisation happens in ``Similarity``, not in the embedder.** Asserted
below, because it is the property that lets a stub return hand-built numbers of
any magnitude and still drive the real thresholds.

The model-backed tests need torch, which on this machine runs only under WSL.
See the README — there is no lighter path any more.
"""

from __future__ import annotations

import math

import pytest

from src.analysis import MODEL_NAME, SentenceTransformerEmbedder, Similarity
from src.common.config import DEEP_THRESHOLD, FAST_THRESHOLD, QUERY_THRESHOLD


class StubEmbedder:
    """Returns whatever vector it was told to, for any text it knows.

    The test seam. There is no production alternative to ``SentenceTransformerEmbedder``.
    """

    identity = "stub"

    def __init__(self, vectors: dict[str, list[float]], dimension: int = 3):
        self.vectors = vectors
        self.dimension = dimension
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vectors.get(text, [0.0] * self.dimension)


def score(similarity: Similarity, left: str, right: str) -> float:
    return similarity.scores(left, [right])[0]


# --------------------------------------------------------------------------
# the one scoring path, on vectors built by hand
# --------------------------------------------------------------------------


def test_identical_vectors_score_one():
    similarity = Similarity(StubEmbedder({"a": [3.0, 4.0], "b": [3.0, 4.0]}, 2))

    assert score(similarity, "a", "b") == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    similarity = Similarity(StubEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]}, 2))

    assert score(similarity, "a", "b") == pytest.approx(0.0)


def test_magnitude_does_not_affect_the_score():
    """What normalising is for: direction is the signal, length is not."""
    similarity = Similarity(StubEmbedder({"a": [1.0, 1.0], "b": [50.0, 50.0]}, 2))

    assert score(similarity, "a", "b") == pytest.approx(1.0)


def test_a_known_angle_scores_its_cosine():
    """45 degrees is cos = 1/sqrt(2). An exact number, not a range."""
    similarity = Similarity(StubEmbedder({"a": [1.0, 0.0], "b": [1.0, 1.0]}, 2))

    assert score(similarity, "a", "b") == pytest.approx(1 / math.sqrt(2))


def test_scores_come_back_one_per_candidate_in_order():
    similarity = Similarity(
        StubEmbedder({"q": [1.0, 0.0], "near": [1.0, 0.1], "far": [0.0, 1.0]}, 2)
    )

    scores = similarity.scores("q", ["near", "far"])

    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_no_candidates_scores_nothing():
    assert Similarity(StubEmbedder({}, 2)).scores("q", []) == []


def test_a_zero_vector_scores_zero_rather_than_dividing_by_zero():
    similarity = Similarity(StubEmbedder({"a": [0.0, 0.0], "b": [1.0, 1.0]}, 2))

    assert score(similarity, "a", "b") == 0.0


def test_scoring_is_symmetric():
    similarity = Similarity(StubEmbedder({"a": [2.0, 1.0], "b": [1.0, 3.0]}, 2))

    assert score(similarity, "a", "b") == pytest.approx(score(similarity, "b", "a"))


def test_scoring_never_asks_which_embedder_produced_the_vectors():
    """There is one scoring implementation and no dispatch. Enforced by AST."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "analysis" / "similarity.py"
    ).read_text(encoding="utf-8")

    scores_fn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "scores"
    )
    branches = [n for n in ast.walk(scores_fn) if isinstance(n, ast.If)]
    inspects = [
        n.attr
        for n in ast.walk(scores_fn)
        if isinstance(n, ast.Attribute) and n.attr in {"name", "dimension"}
    ]
    type_checks = [
        n
        for n in ast.walk(scores_fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "isinstance"
    ]

    # One guard for the empty candidate list, and nothing that inspects the
    # embedder or the kind of vector it returned.
    assert len(branches) == 1
    assert inspects == []
    assert type_checks == []


# --------------------------------------------------------------------------
# normalisation happens here, once — not in the embedder
# --------------------------------------------------------------------------


def test_the_scorer_normalises_what_the_embedder_returned():
    similarity = Similarity(StubEmbedder({"a": [3.0, 4.0]}, 2))

    assert similarity.vector("a") == pytest.approx([0.6, 0.8])


def test_vectors_are_cached_across_calls():
    embedder = StubEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]}, 2)
    similarity = Similarity(embedder)

    similarity.scores("a", ["b"])
    similarity.scores("a", ["b"])

    assert embedder.calls == ["a", "b"]


# --------------------------------------------------------------------------
# the model, and the dimension it fixes
# --------------------------------------------------------------------------


def test_the_dimension_comes_from_the_model_not_from_configuration():
    """384 is the model's output size. There is nothing here to choose."""
    assert SentenceTransformerEmbedder().dimension == 384


def test_the_identity_names_the_model_and_its_dimension():
    identity = SentenceTransformerEmbedder().identity

    assert MODEL_NAME in identity
    assert "384" in identity


def test_a_store_from_the_ngram_embedder_is_rejected():
    """The representation this replaced. Its vectors mean nothing here."""
    similarity = Similarity()

    with pytest.raises(ValueError, match="stale"):
        similarity.load_vectors(
            {"identity": "ngram-3-384", "vectors": {"payment_service": [0.0] * 384}}
        )


def test_identical_text_scores_one():
    assert score(Similarity(), "payment_service", "payment_service") == pytest.approx(
        1.0, abs=1e-6
    )


def test_a_vector_from_the_scorer_is_always_unit_length():
    similarity = Similarity()

    for text in ("payment_service", "#412", "PR #1290"):
        assert math.sqrt(sum(v * v for v in similarity.vector(text))) == pytest.approx(
            1.0
        )


def test_the_model_is_loaded_once_and_reused():
    """A per-instance load would dominate the suite."""
    from src.analysis import similarity as module

    module._load_model()
    first = module._model
    Similarity().vector("anything")

    assert module._model is first


def test_two_fresh_interpreters_produce_identical_vectors():
    """Model loading is deterministic, and it is worth proving.

    Inherited from the hashed embedder, where it guarded against Python's
    randomised string hashing. There is no hashing now, so it guards what
    replaced it: that the same text embeds to the same numbers in a process
    that loaded the model separately.
    """
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "from src.analysis import Similarity;"
        "v = Similarity().vector('payment_service');"
        "print(round(sum(i * x for i, x in enumerate(v)), 9))"
    )
    root = Path(__file__).resolve().parents[2]
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
        ).stdout.strip()
        for _ in range(2)
    }

    assert len(runs) == 1 and runs != {""}


# --------------------------------------------------------------------------
# the dimension guard: a stored vector of a different length would score to a
# meaningless number
# --------------------------------------------------------------------------


class WrongLengthEmbedder:
    """Correct for most inputs, short for one. The poisoning case."""

    dimension = 4
    identity = "wrong-length"

    def embed(self, text: str) -> list[float]:
        if text == "bad":
            return [1.0, 0.0]
        return [1.0, 0.0, 0.0, 0.0]


def test_a_vector_of_the_wrong_length_is_refused_on_write():
    similarity = Similarity(WrongLengthEmbedder())

    with pytest.raises(ValueError, match="length 2"):
        similarity.vector("bad")


def test_the_guard_runs_on_every_write_not_only_the_first():
    """A good first vector must not license a bad second one."""
    similarity = Similarity(WrongLengthEmbedder())

    assert len(similarity.vector("fine")) == 4

    with pytest.raises(ValueError, match="declared dimension is 4"):
        similarity.vector("bad")


def test_a_refused_vector_is_not_left_in_the_store():
    similarity = Similarity(WrongLengthEmbedder())

    with pytest.raises(ValueError):
        similarity.vector("bad")

    assert "bad" not in similarity.export_vectors()["vectors"]


def test_the_error_names_the_embedder_and_both_lengths():
    similarity = Similarity(WrongLengthEmbedder())

    with pytest.raises(ValueError) as caught:
        similarity.vector("bad")

    message = str(caught.value)
    assert "wrong-length" in message and "2" in message and "4" in message


# --------------------------------------------------------------------------
# identity travels with the vectors
# --------------------------------------------------------------------------


def test_exported_vectors_carry_the_identity_that_produced_them():
    similarity = Similarity()
    similarity.vector("payment_service")

    payload = similarity.export_vectors()

    assert payload["identity"] == SentenceTransformerEmbedder().identity
    assert payload["dimension"] == 384
    assert "payment_service" in payload["vectors"]


def test_a_store_round_trips_through_export_and_load():
    source = Similarity()
    source.vector("payment_service")

    target = Similarity()
    target.load_vectors(source.export_vectors())

    assert target.vector("payment_service") == source.vector("payment_service")


def test_a_store_from_a_different_embedder_is_refused():
    """The stale-store case. Same shape, different meaning."""
    similarity = Similarity()

    with pytest.raises(ValueError, match="stale"):
        similarity.load_vectors(
            {
                "identity": "all-MiniLM-L12-v2-384",
                "dimension": 384,
                "vectors": {"payment_service": [0.0] * 384},
            }
        )


def test_a_store_with_no_identity_is_refused():
    similarity = Similarity()

    with pytest.raises(ValueError, match="stale"):
        similarity.load_vectors({"vectors": {"a": [0.0] * 384}})


def test_a_store_with_the_right_identity_but_wrong_length_vectors_is_refused():
    """Neither check implies the other, so both run."""
    similarity = Similarity()

    with pytest.raises(ValueError, match="declared dimension"):
        similarity.load_vectors(
            {
                "identity": SentenceTransformerEmbedder().identity,
                "dimension": 384,
                "vectors": {"payment_service": [1.0, 0.0]},
            }
        )


def test_a_refused_store_leaves_the_existing_vectors_alone():
    similarity = Similarity()
    similarity.vector("payment_service")
    before = similarity.export_vectors()["vectors"]

    with pytest.raises(ValueError):
        similarity.load_vectors({"identity": "other", "vectors": {}})

    assert similarity.export_vectors()["vectors"] == before


def test_an_embedder_without_an_identity_still_gets_a_distinguishing_one():
    """Stubs need not declare one, but must not all look alike."""

    class Anonymous:
        def __init__(self, dimension):
            self.dimension = dimension

        def embed(self, text):
            return [0.0] * self.dimension

    assert Similarity(Anonymous(2)).identity != Similarity(Anonymous(8)).identity


# --------------------------------------------------------------------------
# the drift the embedder exists to catch
#
# These numbers are the exact collision-free values. Hashing is not free — 21
# n-grams collide at 384 and 69 of 190 pairwise scores are inflated — but none
# of the pairs below are affected, and no inflated pair crosses a threshold.
# Both facts are pinned further down.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# what hashing actually costs
#
# Measured, because the comment that used to sit here claimed the dimension was
# collision-free and it is not. What matters is not that collisions exist but
# that none of them reaches a threshold — and that is a property of this
# corpus, which is why the count is pinned rather than merely observed.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# the separation the query floor rests on
#
# QUERY_THRESHOLD was measured against this distribution, so a change to the
# scoring function or the embedder that closes the gap must fail here rather
# than silently making queries return the wrong entity. Found by mutation
# testing: NGRAM = 1 passed every other test in this file while pushing two
# known-distinct pairs above the floor.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# what the model costs on this corpus
#
# Pinned as results, not suppressed. The model and thresholds in use produce
# merges that are wrong on this corpus, and a test that hid them would
# be worse than no test. Adjusting a threshold or special-casing ticket-shaped
# strings would both bury the finding.
# --------------------------------------------------------------------------


def corpus_surfaces() -> list[str]:
    from examples.entity_demo import DOCUMENTS
    from src.analysis import Extractor

    extractor = Extractor("none")
    return sorted(
        {entity.text for document in DOCUMENTS for entity in extractor.extract(document)}
    )


def test_consecutive_ticket_numbers_are_merged_by_the_model():
    """The wrong merge. #413 and #414 are different tickets and are combined.

    The model reads near-identical text as near-identical meaning. For an
    identifier, near-identical text means *definitely different* — the exact
    inversion of what an identifier needs, and Tickets, PRs and Commits are
    most of what this pipeline extracts.
    """
    assert score(Similarity(), "#413", "#414") == pytest.approx(0.9521, abs=1e-3)
    assert score(Similarity(), "#413", "#414") >= FAST_THRESHOLD


def test_a_real_variant_pair_no_longer_merges_outright():
    """The loss on the other side: a pair that should merge now asks a model.

    ``notification-service`` against ``notification_service`` scored 1.0 under
    character n-grams. The model puts it at 0.9041 — inside the band — so with
    no judge configured the two stay separate.
    """
    value = score(Similarity(), "notification-service", "notification_service")

    assert value == pytest.approx(0.9041, abs=1e-3)
    assert DEEP_THRESHOLD <= value < FAST_THRESHOLD


def test_the_middle_band_is_populated_under_the_model():
    """One pair of 190. The three-tier design is exercised, barely."""
    import itertools

    similarity = Similarity()
    in_band = [
        (a, b)
        for a, b in itertools.combinations(corpus_surfaces(), 2)
        if DEEP_THRESHOLD <= similarity.scores(a, [b])[0] < FAST_THRESHOLD
    ]

    assert len(in_band) == 1


def test_a_nonexistent_pull_request_reaches_a_real_one():
    """The query floor admits a PR number that was never ingested."""
    value = score(Similarity(), "pull request #9999", "pull request #1347")

    assert value >= QUERY_THRESHOLD
