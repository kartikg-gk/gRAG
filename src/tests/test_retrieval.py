"""Tests for the two retrieval arms and for seed selection.

Hand-built graphs where an exact score matters, the demo corpus where a
realistic shape does. A traversal score is a product of known confidences, so
most of these assert exact arithmetic against a graph small enough to compute
by hand — a demo-corpus assertion would be checking a number nobody can verify
by reading.

The store is real. Traversal is expressed in Cypher through
``expand_frontier``, and hub suppression is the database's grouping rather than
something this layer does afterwards, so a fake store would assert the shape
that was imagined instead of the one the engine produces.
"""

from __future__ import annotations

import pytest

ladybug = pytest.importorskip(
    "ladybug", reason="the graph store needs ladybug, which runs under WSL"
)

from src.common.config import (  # noqa: E402
    EMBEDDING_DIMENSION,
    MAX_DEGREE,
    MAX_HOPS,
    SEED_MIN_SIM,
    SEED_TOP_N,
    RELATION_AUTHORED,
    RELATION_CO_OCCURS,
    RELATION_PART_OF,
    RELATION_REPORTED,
    RELATION_TOUCHES,
    CONFIDENCE,
)
from src.graphdb import open_context_graph  # noqa: E402
from src.retrieval import (  # noqa: E402
    TIER_EXACT,
    TIER_FALLBACK,
    TIER_FUZZY,
    search,
    select,
    traverse,
)


@pytest.fixture
def store(tmp_path):
    graph = open_context_graph(tmp_path / "graph")
    yield graph
    graph.close()


def basis(position: int) -> list[float]:
    """A unit vector along one axis, so a match is unambiguous."""
    values = [0.0] * EMBEDDING_DIMENSION
    values[position] = 1.0
    return values


class AxisEmbedder:
    """Maps named texts to basis vectors, so similarity is exactly known."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = 0

    def vector(self, text):
        self.calls += 1
        return basis(self.mapping.get(text, 0))


def node(store, node_id, label, kind="Person", embedding=None):
    store.upsert_entity(node_id, label, kind, embedding=embedding)


def chain(store):
    """a -AUTHORED-> b -TOUCHES-> c, confidences 0.95 and 0.80."""
    node(store, "a", "alice")
    node(store, "b", "bravo", "PR")
    node(store, "c", "charlie", "File")
    store.upsert_relationship("a", "b", RELATION_AUTHORED)
    store.upsert_relationship("b", "c", RELATION_TOUCHES)


# ==========================================================================
# The vector arm
# ==========================================================================


def test_the_vector_arm_returns_results_ordered_by_similarity_descending(store):
    node(store, "near", "near", embedding=basis(0))
    node(store, "middle", "middle", embedding=[0.6, 0.8] + [0.0] * (EMBEDDING_DIMENSION - 2))
    node(store, "far", "far", embedding=basis(1))
    store.build_vector_index(rebuild=True)

    result = search(store, AxisEmbedder({"q": 0}), "q", k=3)

    similarities = [hit.similarity for hit in result.hits]
    assert similarities == sorted(similarities, reverse=True)
    assert result.ids[0] == "near"


def test_the_vector_arm_applies_no_floor(store):
    """A raw list. A cutoff here would hide what the arm is bad at."""
    node(store, "near", "near", embedding=basis(0))
    node(store, "orthogonal", "orthogonal", embedding=basis(1))
    store.build_vector_index(rebuild=True)

    result = search(store, AxisEmbedder({"q": 0}), "q", k=10)

    assert len(result) == 2
    assert min(hit.similarity for hit in result.hits) < SEED_MIN_SIM


def test_an_empty_query_never_reaches_the_model(store):
    """Nothing to encode, and a vector of nothing is not a place in the space."""
    embedder = AxisEmbedder({})

    result = search(store, embedder, "   ", k=5)

    assert result.hits == []
    assert embedder.calls == 0


def test_encode_and_search_are_timed_separately(store):
    node(store, "a", "a", embedding=basis(0))
    store.build_vector_index(rebuild=True)

    result = search(store, AxisEmbedder({"q": 0}), "q", k=1)

    assert result.encode_seconds > 0
    assert result.search_seconds > 0


# ==========================================================================
# The graph arm
# ==========================================================================


def test_the_path_score_is_the_product_along_the_path(store):
    """Two hops of known confidence: 0.95 * 0.80 = 0.76, not 1.75."""
    chain(store)

    result = traverse(store, ["a"], max_hops=2)
    scores = {hit.id: hit.score for hit in result.hits}

    assert scores["b"] == pytest.approx(0.95)
    assert scores["c"] == pytest.approx(0.95 * 0.80)
    assert scores["c"] != pytest.approx(0.95 + 0.80)


def test_a_node_reachable_by_two_paths_keeps_the_higher_score(store):
    """One hop at 0.75 against two hops at 0.95 and 0.95 — 0.9025 wins."""
    node(store, "seed", "seed")
    node(store, "middle", "middle", "PR")
    node(store, "target", "target", "Ticket")
    store.upsert_relationship("seed", "target", RELATION_REPORTED)   # 0.75
    store.upsert_relationship("seed", "middle", RELATION_AUTHORED)   # 0.95
    store.upsert_relationship("middle", "target", RELATION_AUTHORED)  # 0.95

    result = traverse(store, ["seed"], max_hops=2)
    scores = {hit.id: hit.score for hit in result.hits}

    assert scores["target"] == pytest.approx(0.95 * 0.95)


def test_traversal_stops_at_the_hop_bound(store):
    """A third node exists one hop past the bound and is not reached."""
    chain(store)
    node(store, "d", "delta", "Repo")
    store.upsert_relationship("c", "d", RELATION_PART_OF)

    within = traverse(store, ["a"], max_hops=2)
    beyond = traverse(store, ["a"], max_hops=3)

    assert "d" not in within.ids
    assert "d" in beyond.ids


def test_a_cycle_terminates(store):
    """Returning to a node cannot improve its score, so it is not re-expanded.

    A confidence is at most 1.0, so a path that loops has multiplied at least
    once more than the path already recorded and cannot beat it. That is what
    ends the walk — no separate cycle check exists.
    """
    node(store, "a", "a")
    node(store, "b", "b", "PR")
    node(store, "c", "c", "File")
    store.upsert_relationship("a", "b", RELATION_AUTHORED)
    store.upsert_relationship("b", "c", RELATION_TOUCHES)
    store.upsert_relationship("c", "a", RELATION_PART_OF)

    result = traverse(store, ["a"], max_hops=10)

    assert {hit.id for hit in result.hits} == {"b", "c"}


def test_a_seed_is_not_returned_as_its_own_result(store):
    """The seed is the question. Returning it at 1.0 would top the ranking."""
    chain(store)

    result = traverse(store, ["a"], max_hops=2)

    assert "a" not in result.ids


def test_traversal_with_no_seeds_does_nothing(store):
    chain(store)

    result = traverse(store, [], max_hops=2)

    assert result.hits == []
    assert result.hops == []


# ==========================================================================
# Hop records
# ==========================================================================


def test_every_hop_carries_source_target_confidence_and_relation(store):
    chain(store)

    result = traverse(store, ["a"], max_hops=2)

    assert result.hops
    for hop in result.hops:
        assert hop.source
        assert hop.target
        assert hop.relation
        assert hop.confidence > 0
        assert hop.depth >= 1


def test_hop_records_describe_the_path_that_was_taken(store):
    """A final score cannot say how a node was reached; these can."""
    chain(store)

    result = traverse(store, ["a"], max_hops=2)
    taken = {(hop.source, hop.target, hop.relation) for hop in result.hops}

    assert ("a", "b", RELATION_AUTHORED) in taken
    assert ("b", "c", RELATION_TOUCHES) in taken


def test_a_hop_records_its_depth(store):
    chain(store)

    result = traverse(store, ["a"], max_hops=2)
    by_target = {hop.target: hop.depth for hop in result.hops}

    assert by_target["b"] == 1
    assert by_target["c"] == 2


# ==========================================================================
# Hub suppression
# ==========================================================================


def test_hub_bound_trims_the_broad_relation_and_keeps_the_narrow_one(store):
    """Per relation, so a broad relation cannot cost a node its narrow one.

    The hub below has 12 TOUCHES edges, over the cap, and 2 AUTHORED edges,
    under it. The authorship is kept whole; the file list is trimmed to the
    cap, not dropped.
    """
    node(store, "hub", "hub", "PR")
    for index in range(12):
        node(store, f"file{index}", f"file{index}", "File")
        store.upsert_relationship("hub", f"file{index}", RELATION_TOUCHES)
    for index in range(2):
        node(store, f"person{index}", f"person{index}")
        store.upsert_relationship(f"person{index}", "hub", RELATION_AUTHORED)

    result = traverse(store, ["hub"], max_hops=1, max_degree=MAX_DEGREE, k=50)
    reached = set(result.ids)

    assert {"person0", "person1"} <= reached
    assert sum(node_id.startswith("file") for node_id in reached) == MAX_DEGREE


def test_without_a_cap_the_broad_relation_is_kept(store):
    """The suppression is the cap's doing, not an accident of the fixture."""
    node(store, "hub", "hub", "PR")
    for index in range(12):
        node(store, f"file{index}", f"file{index}", "File")
        store.upsert_relationship("hub", f"file{index}", RELATION_TOUCHES)

    result = traverse(store, ["hub"], max_hops=1, max_degree=None, k=50)

    assert len([i for i in result.ids if i.startswith("file")]) == 12


# ==========================================================================
# Seed selection
# ==========================================================================


class StubEntity:
    def __init__(self, text, type_="Ticket"):
        self.text = text
        self.type = type_


class StubExtractor:
    """Returns fixed entities, and records that it was asked."""

    def __init__(self, entities):
        self.entities = entities
        self.calls = 0

    def extract(self, text):
        self.calls += 1
        return list(self.entities)


class StubHit:
    def __init__(self, node_id, similarity):
        self.id = node_id
        self.similarity = similarity


def test_tier_one_fires_when_the_query_names_a_stored_entity(store):
    node(store, "ticket:412", "#412", "Ticket")

    result = select(store, StubExtractor([StubEntity("#412")]), [], "what about #412")

    assert result.tier == TIER_EXACT
    assert result.seeds == ["ticket:412"]


def test_tier_two_is_not_consulted_when_tier_one_produces_a_seed(store):
    """A hard stop. Mixing in fuzzy neighbours reintroduces what tier 1 excludes."""
    node(store, "ticket:412", "#412", "Ticket")
    fuzzy = [StubHit("ticket:413", 0.99), StubHit("ticket:414", 0.98)]

    result = select(store, StubExtractor([StubEntity("#412")]), fuzzy, "#412")

    assert result.tier == TIER_EXACT
    assert result.seeds == ["ticket:412"]
    assert "ticket:413" not in result.seeds


def test_tier_two_fires_when_nothing_matched_exactly(store):
    hits = [StubHit("a", 0.90), StubHit("b", 0.50)]

    result = select(store, StubExtractor([]), hits, "some concept")

    assert result.tier == TIER_FUZZY
    assert result.seeds == ["a", "b"]


def test_tier_two_takes_at_most_the_cap_even_when_more_clear_the_floor(store):
    hits = [StubHit(f"n{index}", 0.9) for index in range(SEED_TOP_N + 4)]

    result = select(store, StubExtractor([]), hits, "q")

    assert len(result.seeds) == SEED_TOP_N


def test_tier_two_ignores_hits_below_the_floor(store):
    hits = [StubHit("above", SEED_MIN_SIM), StubHit("below", SEED_MIN_SIM - 0.01)]

    result = select(store, StubExtractor([]), hits, "q")

    assert result.seeds == ["above"]


def test_tier_three_fires_when_nothing_clears_the_floor(store):
    hits = [StubHit("weak", SEED_MIN_SIM - 0.2), StubHit("weaker", 0.01)]

    result = select(store, StubExtractor([]), hits, "q")

    assert result.tier == TIER_FALLBACK
    assert result.seeds == ["weak", "weaker"]


def test_no_seeds_at_all_when_there_is_nothing_to_seed_from(store):
    result = select(store, StubExtractor([]), [], "q")

    assert result.seeds == []
    assert result.tier is None


def test_seeds_are_deduplicated_with_order_preserved(store):
    hits = [StubHit("b", 0.9), StubHit("a", 0.8), StubHit("b", 0.7)]

    result = select(store, StubExtractor([]), hits, "q")

    assert result.seeds == ["b", "a"]


def test_entities_found_and_matched_are_reported_separately(store):
    """Zero seeds means different things depending on these two.

    Three entities and no matches is a linking problem. No entities is a query
    that named none. A seed count alone cannot separate them.
    """
    node(store, "ticket:412", "#412", "Ticket")
    extractor = StubExtractor([StubEntity("#412"), StubEntity("#999")])

    result = select(store, extractor, [], "#412 and #999")

    assert result.entities_found == 2
    assert result.matched == 1


# ==========================================================================
# The arms are independent
# ==========================================================================


def test_the_graph_arm_receives_no_vector_score(store):
    """It takes seed ids. There is no channel for a similarity to arrive on."""
    import inspect

    signature = inspect.signature(traverse)

    assert "seeds" in signature.parameters
    assert not any(
        "similarity" in name or "vector" in name for name in signature.parameters
    )


def test_the_vector_arm_receives_nothing_from_traversal(store):
    import inspect

    signature = inspect.signature(search)

    assert not any(
        name in signature.parameters for name in ("seeds", "hops", "graph", "traversal")
    )


def test_neither_arm_changes_what_the_other_returns(store):
    """Run both, in both orders, and compare. No shared state anywhere."""
    chain(store)
    node(store, "z", "zulu", embedding=basis(3))
    store.build_vector_index(rebuild=True)
    embedder = AxisEmbedder({"q": 3})

    vector_first = search(store, embedder, "q", k=5).ids
    graph_after = traverse(store, ["a"], max_hops=2).ids

    graph_first = traverse(store, ["a"], max_hops=2).ids
    vector_after = search(store, embedder, "q", k=5).ids

    assert vector_first == vector_after
    assert graph_after == graph_first


def test_the_defaults_come_from_config():
    """The constants are the contract; a literal here would drift from them."""
    import inspect

    assert inspect.signature(traverse).parameters["max_hops"].default == MAX_HOPS
    assert inspect.signature(traverse).parameters["max_degree"].default == MAX_DEGREE
    assert inspect.signature(select).parameters["min_similarity"].default == SEED_MIN_SIM
    assert inspect.signature(select).parameters["top_n"].default == SEED_TOP_N


def test_every_retrieval_constant_is_environment_overridable(monkeypatch):
    """A setting that needs a source edit to change is not a setting."""
    import importlib

    monkeypatch.setenv("GRAPHRAG_GRAPH_MAX_HOPS", "7")
    monkeypatch.setenv("GRAPHRAG_GRAPH_SEED_MIN_SIM", "0.9")
    monkeypatch.setenv("GRAPHRAG_GRAPH_SEED_TOP_N", "11")
    monkeypatch.setenv("GRAPHRAG_MAX_DEGREE", "99")
    monkeypatch.setenv("GRAPHRAG_TOP_K_VECTOR", "5")
    monkeypatch.setenv("GRAPHRAG_TOP_K_GRAPH", "6")

    from src.common import config

    reloaded = importlib.reload(config)
    try:
        assert reloaded.MAX_HOPS == 7
        assert reloaded.SEED_MIN_SIM == 0.9
        assert reloaded.SEED_TOP_N == 11
        assert reloaded.MAX_DEGREE == 99
        assert reloaded.TOP_K_VECTOR == 5
        assert reloaded.TOP_K_GRAPH == 6
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_a_malformed_environment_value_falls_back(monkeypatch):
    """A shell typo should not stop the process starting."""
    import importlib

    monkeypatch.setenv("GRAPHRAG_GRAPH_MAX_HOPS", "not-a-number")

    from src.common import config

    reloaded = importlib.reload(config)
    try:
        assert reloaded.MAX_HOPS == 2
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_graph_router_settings_use_the_pinned_environment_names(monkeypatch):
    import importlib
    from src.common import config

    monkeypatch.setenv("GRAPHRAG_GRAPH_MAX_HOPS", "7")
    monkeypatch.setenv("GRAPHRAG_GRAPH_SEED_MIN_SIM", "0.61")
    monkeypatch.setenv("GRAPHRAG_GRAPH_SEED_TOP_N", "4")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.MAX_HOPS == 7
        assert reloaded.SEED_MIN_SIM == 0.61
        assert reloaded.SEED_TOP_N == 4
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_database_pool_environment_value_is_clamped_to_one(monkeypatch):
    import importlib
    from src.common import config

    monkeypatch.setenv("GRAPHRAG_DB_POOL_SIZE", "-3")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.READ_POOL_SIZE == 1
    finally:
        monkeypatch.undo()
        importlib.reload(config)
