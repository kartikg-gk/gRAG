"""Tests for the retrieval sequence and the record it emits.

A real store, because this is the one place every stage runs together and the
thing worth checking is that the record matches what actually happened rather
than what was intended.

Two assertions here are about identity rather than value — that the hop
records are the objects traversal built, and that decay is not evaluated a
second time. Both would pass on equality while a reconstruction silently
drifted from the run it claims to describe, so both are written as identity
and call-count checks instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

ladybug = pytest.importorskip(
    "ladybug", reason="the graph store needs ladybug, which runs under WSL"
)

from src.common.config import (  # noqa: E402
    EMBEDDING_DIMENSION,
    INTENT_CONCEPTUAL,
    INTENT_RELATIONAL,
    NODE_PR,
    NODE_TICKET,
    RELATION_AUTHORED,
    RELATION_RESOLVES,
    STAGE_FALLBACK,
    STAGE_MARKER,
)
from src.graphdb import open_context_graph  # noqa: E402
from src.retrieval import TIER_EXACT, retrieve  # noqa: E402
from src.retrieval import recency as recency_module  # noqa: E402

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def basis(position: int) -> list[float]:
    values = [0.0] * EMBEDDING_DIMENSION
    values[position] = 1.0
    return values


class AxisEmbedder:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.calls = 0

    def vector(self, text):
        self.calls += 1
        return basis(self.mapping.get(text, 0))


class StubEntity:
    def __init__(self, text, type_=NODE_TICKET):
        self.text = text
        self.type = type_


class StubExtractor:
    def __init__(self, entities=()):
        self.entities = list(entities)

    def extract(self, text):
        return list(self.entities)


@pytest.fixture
def store(tmp_path):
    graph = open_context_graph(tmp_path / "graph")
    graph.upsert_entity(
        "ticket:412", "#412", NODE_TICKET, timestamp=NOW - timedelta(days=30),
        embedding=basis(0),
    )
    graph.upsert_entity(
        "pr:9", "Rewrite the scheduler", NODE_PR,
        timestamp=NOW - timedelta(days=120), embedding=basis(1),
    )
    graph.upsert_entity(
        "person:alice", "alice", "Person", embedding=basis(2),
    )
    graph.upsert_relationship("pr:9", "ticket:412", RELATION_RESOLVES)
    graph.upsert_relationship("person:alice", "pr:9", RELATION_AUTHORED)
    graph.build_vector_index(rebuild=True)
    yield graph
    graph.close()


def run(store, **kwargs):
    kwargs.setdefault("now", NOW)
    return retrieve(store, AxisEmbedder(), kwargs.pop("query", "#412"), **kwargs)


# ==========================================================================
# The sequence runs
# ==========================================================================


def test_a_query_returns_every_stage_not_just_the_ranking(store):
    result = run(store, extractor=StubExtractor([StubEntity("#412")]))

    assert result.intent is not None
    assert result.vector is not None
    assert result.seeds is not None
    assert result.graph is not None
    assert result.fused is not None


def test_the_fused_ranking_is_the_answer(store):
    result = run(store, extractor=StubExtractor([StubEntity("#412")]))

    assert result.ids == result.fused.ids
    assert len(result) == len(result.fused)


def test_it_runs_without_an_extractor_or_a_judge(store):
    """Both optional, both degrade rather than fail."""
    result = run(store, query="scheduler rewrite")

    assert result.fused is not None
    assert result.intent.stage == STAGE_FALLBACK


# ==========================================================================
# All four sections populated
# ==========================================================================


def test_a_round_trip_populates_all_four_sections(store):
    result = run(
        store,
        query="who fixed #412?",
        extractor=StubExtractor([StubEntity("#412")]),
    )
    log = result.trace_log

    assert log.intent is not None
    assert log.execution_path is not None
    assert log.recency
    assert log.metrics is not None

    assert log.intent.label == INTENT_RELATIONAL
    assert log.intent.stage == STAGE_MARKER
    assert log.intent.marker == "who"
    assert log.intent.alpha and log.intent.beta

    assert log.execution_path.seeds
    assert log.execution_path.tier == TIER_EXACT
    assert log.execution_path.hops
    assert log.query == "who fixed #412?"


def test_the_intent_section_carries_the_weights_that_were_used(store):
    result = run(store, query="who fixed #412?",
                 extractor=StubExtractor([StubEntity("#412")]))

    assert result.trace_log.intent.alpha == result.fused.alpha
    assert result.trace_log.intent.beta == result.fused.beta


def test_a_conceptual_query_records_its_own_weights(store):
    result = run(store, query="explain the scheduler")

    assert result.trace_log.intent.label == INTENT_CONCEPTUAL
    assert result.trace_log.intent.alpha == result.fused.alpha


def test_the_seed_tier_reaches_the_log(store):
    result = run(store, extractor=StubExtractor([StubEntity("#412")]))

    assert result.trace_log.execution_path.tier == result.seeds.tier
    assert [s.id for s in result.trace_log.execution_path.seeds] == result.seeds.seeds


# ==========================================================================
# The hops are the traversal's own, not a reconstruction
# ==========================================================================


def test_the_logged_hops_are_the_objects_traversal_built(store):
    result = run(store, extractor=StubExtractor([StubEntity("#412")]))

    assert result.trace_log.execution_path.hops is result.graph.hops
    for logged, traversed in zip(
        result.trace_log.execution_path.hops, result.graph.hops
    ):
        assert logged is traversed


def test_every_hop_carries_the_four_fields(store):
    result = run(store, extractor=StubExtractor([StubEntity("#412")]))

    assert result.trace_log.execution_path.hops
    for hop in result.trace_log.execution_path.hops:
        assert hop.source
        assert hop.target
        assert hop.relation
        assert hop.confidence > 0
        assert hop.depth >= 1


def test_the_hops_describe_the_edges_the_store_actually_holds(store):
    result = run(store, extractor=StubExtractor([StubEntity("#412")]))
    taken = {(h.source, h.target, h.relation) for h in result.trace_log.execution_path.hops}

    assert ("ticket:412", "pr:9", RELATION_RESOLVES) in taken


# ==========================================================================
# Decay is the one that ranked, not a second evaluation
# ==========================================================================


def test_logged_decay_matches_what_fusion_used(store):
    result = run(store, extractor=StubExtractor([StubEntity("#412")]))

    by_id = {hit.id: hit for hit in result.fused.hits}
    for record in result.trace_log.recency:
        assert record.decay == by_id[record.id].decay
        assert record.age_days == by_id[record.id].age_days
        assert record.node_type == by_id[record.id].node_type


def test_age_is_computed_once_per_node(monkeypatch, store):
    """Not twice. A second call could disagree with the first.

    ``decay_factor`` used to call ``age_days`` and discard the result, so
    surfacing the age naively would evaluate it again — and if ``now`` were
    left to the wall clock the two would differ by the time between them. The
    number reported beside a score would not be the number that produced it.
    """
    calls = []
    real = recency_module.age_days

    def counting(timestamp, now=None):
        calls.append(timestamp)
        return real(timestamp, now)

    monkeypatch.setattr(recency_module, "age_days", counting)

    result = run(store, extractor=StubExtractor([StubEntity("#412")]))

    assert len(calls) == len(result.fused.hits)


# ==========================================================================
# Metrics match what drove fusion
# ==========================================================================


def test_metrics_match_the_values_that_drove_fusion(store):
    result = run(store, extractor=StubExtractor([StubEntity("#412")]))
    metrics = result.trace_log.metrics

    assert metrics.graph_hits == result.fused.graph_hits
    assert metrics.vector_k == result.fused.vector_k


def test_visited_matches_what_traversal_reported(store):
    result = run(store, extractor=StubExtractor([StubEntity("#412")]))

    assert result.trace_log.metrics.visited == result.graph.visited


# ==========================================================================
# The arms stay independent through the sequence
# ==========================================================================


def test_running_the_sequence_does_not_couple_the_arms(store):
    """Same arms, run alone, produce the same results as inside the sequence."""
    from src.retrieval import search, select, traverse

    embedder = AxisEmbedder()
    alone_vector = search(store, embedder, "#412", k=10)
    alone_seeds = select(store, StubExtractor([StubEntity("#412")]),
                         alone_vector.hits, "#412")
    alone_graph = traverse(store, alone_seeds.seeds, k=10)

    result = run(store, extractor=StubExtractor([StubEntity("#412")]))

    assert result.vector.ids == alone_vector.ids
    assert result.seeds.seeds == alone_seeds.seeds
    assert result.graph.ids == alone_graph.ids


def test_the_query_is_encoded_once(store):
    """Seed tier 2 reads the vector arm's output rather than re-encoding."""
    embedder = AxisEmbedder()

    retrieve(store, embedder, "#412",
             extractor=StubExtractor([StubEntity("#412")]), now=NOW)

    assert embedder.calls == 1
