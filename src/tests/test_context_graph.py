"""Tests for the graph store.

Every test drives a real database in a temporary directory. There is no fake
driver: the behaviours worth testing here — MERGE semantics, write-once
columns, index rebuilds, connection lifetime — are the database's behaviours,
and a fake would assert what we imagined rather than what the engine does.

Embeddings come back as float32, so anything comparing a stored vector to the
list that went in uses a tolerance rather than equality.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

ladybug = pytest.importorskip(
    "ladybug", reason="the graph store needs ladybug, which runs under WSL"
)

from src.common.config import (  # noqa: E402
    CONFIDENCE,
    DOC_TABLE,
    EMBEDDING_DIMENSION,
    NODE_TABLE,
    REL_TABLE,
    RELATION_AUTHORED,
    RELATION_CO_OCCURS,
    RELATION_REPORTED,
    RELATION_TOUCHES,
    VECTOR_INDEX_NAME,
)
from src.graphdb import ContextGraph, open_context_graph  # noqa: E402

EARLY = datetime(2026, 1, 1, tzinfo=timezone.utc)
LATE = datetime(2026, 6, 1, tzinfo=timezone.utc)


def vector(value: float) -> list[float]:
    return [value] * EMBEDDING_DIMENSION


@pytest.fixture
def graph(tmp_path):
    store = open_context_graph(tmp_path / "graph")
    yield store
    store.close()


def person(store, entity_id, label, ts=None, embedding=None):
    store.upsert_entity(entity_id, label, "Person", ts, embedding)


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


def test_a_fresh_database_gets_every_table(graph):
    tables = {row[1] for row in graph.query("CALL SHOW_TABLES() RETURN *")}

    assert {NODE_TABLE, DOC_TABLE, REL_TABLE, "Mentions"} <= tables


def test_the_entity_table_has_the_declared_columns(graph):
    columns = graph.table_columns(NODE_TABLE)

    assert set(columns) == {"id", "label", "type", "ts", "embedding"}
    assert columns["ts"] == "INT64"


def test_the_embedding_column_is_the_configured_width(graph):
    assert f"[{EMBEDDING_DIMENSION}]" in graph.table_columns(NODE_TABLE)["embedding"]


def test_initialising_twice_is_harmless(graph):
    graph.initialize_schema()
    graph.initialize_schema()

    assert graph.count_nodes() == 0


def test_migrating_a_current_database_adds_nothing(graph):
    assert graph.migrate() == []


def test_migration_adds_a_missing_column_and_reports_it(tmp_path):
    """CREATE TABLE IF NOT EXISTS will not do this — it sees the table and stops."""
    store = ContextGraph(tmp_path / "old")
    store.execute(
        f"CREATE NODE TABLE {NODE_TABLE}("
        f"  id STRING PRIMARY KEY, label STRING, type STRING)"
    )
    store.execute(
        f"CREATE NODE TABLE {DOC_TABLE}(id STRING PRIMARY KEY, path STRING)"
    )
    try:
        added = store.migrate()

        assert f"{NODE_TABLE}.ts" in added
        assert f"{NODE_TABLE}.embedding" in added
        assert f"{DOC_TABLE}.content" in added
        assert "ts" in store.table_columns(NODE_TABLE)
    finally:
        store.close()


def test_migration_preserves_existing_rows(tmp_path):
    store = ContextGraph(tmp_path / "old")
    store.execute(
        f"CREATE NODE TABLE {NODE_TABLE}("
        f"  id STRING PRIMARY KEY, label STRING, type STRING)"
    )
    store.execute(
        f"CREATE (:{NODE_TABLE} {{id: 'kept', label: 'alice', type: 'Person'}})"
    )
    try:
        store.migrate()
        survivor = store.get_entity("kept")

        assert survivor["label"] == "alice"
        assert survivor["timestamp"] is None  # the new column, unknown not zero
    finally:
        store.close()


def test_migration_is_idempotent(tmp_path):
    store = ContextGraph(tmp_path / "old")
    store.execute(
        f"CREATE NODE TABLE {NODE_TABLE}(id STRING PRIMARY KEY, label STRING)"
    )
    try:
        first = store.migrate()
        second = store.migrate()

        assert first
        assert second == []
    finally:
        store.close()


# --------------------------------------------------------------------------
# entities
# --------------------------------------------------------------------------


def test_an_entity_round_trips(graph):
    person(graph, "p:alice", "alice", EARLY, vector(0.1))
    stored = graph.get_entity("p:alice")

    assert stored["id"] == "p:alice"
    assert stored["label"] == "alice"
    assert stored["type"] == "Person"
    assert stored["timestamp"] == EARLY


def test_the_label_is_write_once(graph):
    """A noisier surface form arriving later must not clobber the canonical."""
    person(graph, "p:alice", "alice", EARLY)
    person(graph, "p:alice", "ALICE_SHOUTING", LATE)

    assert graph.get_entity("p:alice")["label"] == "alice"


def test_the_type_is_write_once(graph):
    graph.upsert_entity("p:alice", "alice", "Person", EARLY)
    graph.upsert_entity("p:alice", "alice", "WrongType", LATE)

    assert graph.get_entity("p:alice")["type"] == "Person"


def test_a_newer_timestamp_advances(graph):
    person(graph, "p:alice", "alice", EARLY)
    person(graph, "p:alice", "alice", LATE)

    assert graph.get_entity("p:alice")["timestamp"] == LATE


def test_an_older_timestamp_does_not_move_it_backward(graph):
    person(graph, "p:alice", "alice", LATE)
    person(graph, "p:alice", "alice", EARLY)

    assert graph.get_entity("p:alice")["timestamp"] == LATE


def test_an_unknown_timestamp_never_erases_a_known_one(graph):
    person(graph, "p:alice", "alice", LATE)
    person(graph, "p:alice", "alice", None)

    assert graph.get_entity("p:alice")["timestamp"] == LATE


def test_an_unknown_timestamp_is_null_rather_than_zero(graph):
    """0 reads as 1970 to a recency scorer, which buries the row."""
    person(graph, "p:alice", "alice", None)

    assert graph.get_entity("p:alice")["timestamp"] is None


def test_upserting_the_same_id_does_not_duplicate_it(graph):
    for _ in range(4):
        person(graph, "p:alice", "alice", EARLY)

    assert graph.count_nodes() == 1


def test_an_embedding_round_trips_within_float32_tolerance(graph):
    """The column is FLOAT[dim], so this is float32 — close, not identical."""
    person(graph, "p:alice", "alice", EARLY, vector(0.1))
    stored = graph.get_embedding("p:alice")

    assert len(stored) == EMBEDDING_DIMENSION
    assert stored == pytest.approx(vector(0.1), abs=1e-6)
    assert stored[0] != 0.1  # float32, not the Python float that went in


@pytest.mark.parametrize("width", [1, EMBEDDING_DIMENSION - 1, EMBEDDING_DIMENSION + 1])
def test_an_embedding_of_the_wrong_width_is_refused(graph, width):
    with pytest.raises(ValueError, match="dimensions"):
        graph.upsert_entity("p:alice", "alice", "Person", EARLY, [0.1] * width)


def test_the_error_names_the_entity_and_both_widths(graph):
    with pytest.raises(ValueError) as caught:
        graph.upsert_entity("p:alice", "alice", "Person", EARLY, [0.1])

    message = str(caught.value)
    assert "p:alice" in message and str(EMBEDDING_DIMENSION) in message


def test_an_entity_may_have_no_embedding(graph):
    person(graph, "p:alice", "alice", EARLY)

    assert graph.get_embedding("p:alice") is None


def test_existence_check(graph):
    person(graph, "p:alice", "alice")

    assert graph.entity_exists("p:alice")
    assert not graph.entity_exists("p:nobody")


def test_lookup_by_label_ignores_case(graph):
    graph.upsert_entity("o:openai", "OpenAI", "Org")

    for spelling in ("OpenAI", "openai", "OPENAI"):
        assert [e["id"] for e in graph.find_by_label(spelling)] == ["o:openai"]


def test_lookup_by_label_is_exact_not_fuzzy(graph):
    """Anything looser belongs to vector search, not here."""
    graph.upsert_entity("o:openai", "OpenAI", "Org")

    assert graph.find_by_label("Open AI") == []
    assert graph.find_by_label("OpenA") == []


# --------------------------------------------------------------------------
# relationships
# --------------------------------------------------------------------------


def test_a_relationship_uses_the_configured_weight(graph):
    person(graph, "a", "alice")
    person(graph, "b", "bob")
    graph.upsert_relationship("a", "b", RELATION_AUTHORED)

    assert graph.get_relationship("a", "b")["confidence"] == CONFIDENCE[
        RELATION_AUTHORED
    ]


def test_stronger_evidence_upgrades_the_relation(graph):
    person(graph, "a", "alice")
    person(graph, "b", "bob")

    graph.upsert_relationship("a", "b", RELATION_CO_OCCURS)
    graph.upsert_relationship("a", "b", RELATION_AUTHORED)

    edge = graph.get_relationship("a", "b")
    assert edge["relation"] == RELATION_AUTHORED
    assert edge["confidence"] == CONFIDENCE[RELATION_AUTHORED]


def test_weaker_evidence_does_not_downgrade(graph):
    """The trap: comparing the new confidence against itself after writing it."""
    person(graph, "a", "alice")
    person(graph, "b", "bob")

    graph.upsert_relationship("a", "b", RELATION_AUTHORED)
    graph.upsert_relationship("a", "b", RELATION_CO_OCCURS)

    edge = graph.get_relationship("a", "b")
    assert edge["relation"] == RELATION_AUTHORED
    assert edge["confidence"] == CONFIDENCE[RELATION_AUTHORED]


def test_equal_confidence_keeps_the_existing_relation(graph):
    person(graph, "a", "alice")
    person(graph, "b", "bob")

    graph.upsert_relationship("a", "b", RELATION_TOUCHES)
    graph.upsert_relationship("a", "b", "PART_OF")  # same weight, 0.80

    assert graph.get_relationship("a", "b")["relation"] == RELATION_TOUCHES


def test_a_weak_edge_does_not_block_a_later_strong_one(graph):
    """The whole point of upgrade-on-stronger-evidence."""
    person(graph, "a", "alice")
    person(graph, "b", "bob")

    graph.upsert_relationship("a", "b", RELATION_CO_OCCURS)
    graph.upsert_relationship("a", "b", RELATION_REPORTED)
    graph.upsert_relationship("a", "b", RELATION_AUTHORED)

    assert graph.get_relationship("a", "b")["relation"] == RELATION_AUTHORED


def test_repeated_upserts_do_not_multiply_edges(graph):
    person(graph, "a", "alice")
    person(graph, "b", "bob")
    for _ in range(3):
        graph.upsert_relationship("a", "b", RELATION_AUTHORED)

    assert graph.node_degree("a") == 1


def test_co_occurs_is_the_weakest_relation():
    """Configured so a proximity guess never outranks a structural fact."""
    assert CONFIDENCE[RELATION_CO_OCCURS] == min(CONFIDENCE.values())
    assert CONFIDENCE[RELATION_CO_OCCURS] < CONFIDENCE[RELATION_REPORTED]


# --------------------------------------------------------------------------
# documents and mentions
# --------------------------------------------------------------------------


def test_a_document_round_trips(graph):
    graph.upsert_document("d1", "/src/a.py", "hello")
    stored = graph.get_document("d1")

    assert stored == {"id": "d1", "path": "/src/a.py", "content": "hello"}


def test_re_ingesting_a_document_refreshes_its_content(graph):
    """Unlike an entity label, the newer text is the source of truth."""
    graph.upsert_document("d1", "/src/a.py", "old text")
    graph.upsert_document("d1", "/src/moved.py", "new text")

    stored = graph.get_document("d1")
    assert stored["content"] == "new text"
    assert stored["path"] == "/src/moved.py"
    assert graph.count_documents() == 1


def test_a_mention_links_a_document_to_an_entity(graph):
    person(graph, "p:alice", "alice")
    graph.upsert_document("d1", "/a.md", "alice wrote this")
    graph.add_mention("d1", "p:alice")

    documents = graph.documents_for_entities(["p:alice"])
    assert [d["id"] for d in documents["p:alice"]] == ["d1"]


def test_adding_the_same_mention_twice_is_idempotent(graph):
    person(graph, "p:alice", "alice")
    graph.upsert_document("d1", "/a.md", "text")
    graph.add_mention("d1", "p:alice")
    graph.add_mention("d1", "p:alice")

    assert len(graph.documents_for_entities(["p:alice"])["p:alice"]) == 1


def test_documents_are_grouped_by_entity(graph):
    for entity_id in ("a", "b"):
        person(graph, entity_id, entity_id)
    graph.upsert_document("d1", "/1.md", "one")
    graph.upsert_document("d2", "/2.md", "two")
    graph.add_mention("d1", "a")
    graph.add_mention("d2", "a")
    graph.add_mention("d2", "b")

    grouped = graph.documents_for_entities(["a", "b"])

    assert [d["id"] for d in grouped["a"]] == ["d1", "d2"]
    assert [d["id"] for d in grouped["b"]] == ["d2"]


def test_an_entity_nothing_mentions_gets_an_empty_list_not_a_missing_key(graph):
    person(graph, "lonely", "lonely")

    assert graph.documents_for_entities(["lonely"]) == {"lonely": []}


# --------------------------------------------------------------------------
# vector search
# --------------------------------------------------------------------------


def test_the_vector_index_exists_after_initialisation(graph):
    names = {row[1] for row in graph.query("CALL SHOW_INDEXES() RETURN *")}

    assert VECTOR_INDEX_NAME in names


def test_nearest_neighbour_finds_the_closest_entity(graph):
    person(graph, "near", "near", EARLY, [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1))
    person(graph, "far", "far", EARLY, [0.0] * (EMBEDDING_DIMENSION - 1) + [1.0])
    graph.build_vector_index(rebuild=True)

    hits = graph.vector_search([1.0] + [0.0] * (EMBEDDING_DIMENSION - 1), k=2)

    assert hits[0]["id"] == "near"
    assert hits[0]["similarity"] > hits[1]["similarity"]


def test_a_search_hit_identifies_the_entity(graph):
    person(graph, "p:alice", "alice", LATE, vector(0.1))
    graph.build_vector_index(rebuild=True)

    hit = graph.vector_search(vector(0.1), k=1)[0]

    assert set(hit) == {"id", "label", "type", "timestamp", "distance", "similarity"}
    assert hit["label"] == "alice"
    assert hit["timestamp"] == LATE


def test_similarity_is_one_minus_distance(graph):
    person(graph, "p:alice", "alice", EARLY, vector(0.1))
    graph.build_vector_index(rebuild=True)

    hit = graph.vector_search(vector(0.1), k=1)[0]

    assert hit["similarity"] == pytest.approx(1.0 - hit["distance"])
    assert hit["similarity"] == pytest.approx(1.0, abs=1e-4)


def test_k_bounds_the_result(graph):
    for index in range(5):
        person(graph, f"e{index}", f"e{index}", EARLY, vector(0.1 * (index + 1)))
    graph.build_vector_index(rebuild=True)

    assert len(graph.vector_search(vector(0.1), k=3)) <= 3


def test_a_query_vector_of_the_wrong_width_is_refused(graph):
    with pytest.raises(ValueError, match="dimensions"):
        graph.vector_search([0.1, 0.2], k=1)


def test_rebuilding_the_index_is_safe_to_repeat(graph):
    person(graph, "p:alice", "alice", EARLY, vector(0.1))
    graph.build_vector_index(rebuild=True)
    graph.build_vector_index(rebuild=True)

    assert graph.vector_search(vector(0.1), k=1)[0]["id"] == "p:alice"


# --------------------------------------------------------------------------
# graph retrieval
# --------------------------------------------------------------------------


def build_star(graph):
    """One hub, three spokes, with different relation strengths."""
    for entity_id in ("hub", "s1", "s2", "s3"):
        person(graph, entity_id, entity_id)
    graph.upsert_relationship("hub", "s1", RELATION_AUTHORED)
    graph.upsert_relationship("hub", "s2", RELATION_REPORTED)
    graph.upsert_relationship("hub", "s3", RELATION_CO_OCCURS)


def test_neighbours_come_back_strongest_first(graph):
    build_star(graph)

    assert [n["id"] for n in graph.neighbors("hub")] == ["s1", "s2", "s3"]


def test_neighbours_break_ties_deterministically(graph):
    person(graph, "hub", "hub")
    for spoke in ("z", "m", "a"):
        person(graph, spoke, spoke)
        graph.upsert_relationship("hub", spoke, RELATION_AUTHORED)

    assert [n["id"] for n in graph.neighbors("hub")] == ["a", "m", "z"]


def test_neighbours_respect_k(graph):
    build_star(graph)

    assert len(graph.neighbors("hub", k=2)) == 2


def test_degree_counts_distinct_neighbours(graph):
    build_star(graph)

    assert graph.node_degree("hub") == 3
    assert graph.node_degree("s1") == 1


def test_degree_of_an_isolated_node_is_zero(graph):
    person(graph, "lonely", "lonely")

    assert graph.node_degree("lonely") == 0


def test_frontier_expansion_groups_by_origin(graph):
    build_star(graph)
    person(graph, "other", "other")
    person(graph, "x", "x")
    graph.upsert_relationship("other", "x", RELATION_AUTHORED)

    frontier = graph.expand_frontier(["hub", "other"])

    assert [n["id"] for n in frontier["hub"]] == ["s1", "s2", "s3"]
    assert [n["id"] for n in frontier["other"]] == ["x"]


def test_frontier_expansion_uses_one_query_for_many_nodes(graph):
    """The N+1 guard: expanding N nodes must not cost N queries."""
    build_star(graph)
    calls: list[str] = []
    original = graph.query

    def counting(cypher, parameters=None):
        calls.append(cypher)
        return original(cypher, parameters)

    graph.query = counting
    try:
        graph.expand_frontier(["hub", "s1", "s2", "s3"])
    finally:
        graph.query = original

    assert len(calls) == 1


def test_frontier_filters_a_hub_per_relation_not_per_node(graph):
    """A broad TOUCHES must not cost the node its three AUTHORED edges."""
    person(graph, "repo", "repo")
    for index in range(6):
        person(graph, f"file{index}", f"file{index}")
        graph.upsert_relationship("repo", f"file{index}", RELATION_TOUCHES)
    for index in range(2):
        person(graph, f"dev{index}", f"dev{index}")
        graph.upsert_relationship("repo", f"dev{index}", RELATION_AUTHORED)

    kept = graph.expand_frontier(["repo"], k=50, max_degree=3)["repo"]

    relations = {n["relation"] for n in kept}
    assert relations == {RELATION_AUTHORED}
    assert len(kept) == 2


def test_frontier_without_a_max_degree_keeps_everything(graph):
    build_star(graph)

    assert len(graph.expand_frontier(["hub"], k=50)["hub"]) == 3


def test_frontier_of_an_isolated_node_is_an_empty_list(graph):
    person(graph, "lonely", "lonely")

    assert graph.expand_frontier(["lonely"]) == {"lonely": []}


def test_subgraph_marks_requested_nodes(graph):
    build_star(graph)

    result = graph.subgraph(["hub"])
    requested = {n["id"]: n["requested"] for n in result["nodes"]}

    assert requested["hub"] is True
    assert requested["s1"] is False


def test_subgraph_includes_edges_among_visible_nodes(graph):
    build_star(graph)

    edges = graph.subgraph(["hub"])["edges"]

    assert {(e["source"], e["target"]) for e in edges} == {
        ("hub", "s1"),
        ("hub", "s2"),
        ("hub", "s3"),
    }


def test_an_isolated_requested_node_still_appears(graph):
    """Returning only nodes with edges would drop exactly what a caller needs."""
    person(graph, "lonely", "lonely")

    result = graph.subgraph(["lonely"])

    assert [n["id"] for n in result["nodes"]] == ["lonely"]
    assert result["nodes"][0]["requested"] is True
    assert result["edges"] == []


def test_subgraph_of_nothing_is_empty(graph):
    assert graph.subgraph([]) == {"nodes": [], "edges": []}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_counting_nodes(graph):
    for index in range(3):
        person(graph, f"e{index}", f"e{index}")

    assert graph.count_nodes() == 3


def test_most_connected_ranks_by_distinct_neighbours(graph):
    build_star(graph)

    ranked = graph.most_connected(limit=2)

    assert ranked[0]["id"] == "hub"
    assert ranked[0]["degree"] == 3
    assert set(ranked[0]) == {"id", "label", "type", "degree"}


def test_most_connected_breaks_ties_deterministically(graph):
    for pair in (("a", "b"), ("c", "d")):
        for entity_id in pair:
            person(graph, entity_id, entity_id)
        graph.upsert_relationship(pair[0], pair[1], RELATION_AUTHORED)

    ranked = graph.most_connected(limit=4)

    assert [r["id"] for r in ranked] == ["a", "b", "c", "d"]


# --------------------------------------------------------------------------
# results are plain data
# --------------------------------------------------------------------------


def test_nothing_driver_specific_escapes(graph):
    """Callers work in dicts and never import the driver."""
    build_star(graph)
    person(graph, "p", "p", EARLY, vector(0.1))
    graph.build_vector_index(rebuild=True)

    for value in (
        graph.get_entity("hub"),
        graph.neighbors("hub")[0],
        graph.vector_search(vector(0.1), k=1)[0],
        graph.subgraph(["hub"])["nodes"][0],
        graph.most_connected(1)[0],
    ):
        assert isinstance(value, dict)
        assert all(isinstance(key, str) for key in value)


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


def test_close_is_safe_to_call_twice(tmp_path):
    store = open_context_graph(tmp_path / "g")
    store.close()
    store.close()


def test_operations_after_close_are_refused(tmp_path):
    store = open_context_graph(tmp_path / "g")
    store.close()

    with pytest.raises(RuntimeError, match="closed"):
        store.count_nodes()
    with pytest.raises(RuntimeError, match="closed"):
        store.upsert_entity("a", "a", "Person")


def test_the_context_manager_closes_on_the_way_out(tmp_path):
    with open_context_graph(tmp_path / "g") as store:
        store.upsert_entity("a", "a", "Person")
        assert store.count_nodes() == 1

    with pytest.raises(RuntimeError, match="closed"):
        store.count_nodes()


def test_the_context_manager_closes_even_when_the_body_raises(tmp_path):
    store = open_context_graph(tmp_path / "g")

    with pytest.raises(ZeroDivisionError):
        with store:
            1 / 0

    with pytest.raises(RuntimeError, match="closed"):
        store.count_nodes()


def test_a_read_connection_returns_to_the_pool_after_a_failed_query(graph):
    """A leaked lease would exhaust the pool and look like a hang."""
    for _ in range(10):
        with pytest.raises(Exception):
            graph.query("MATCH (x:NoSuchTable) RETURN x")

    assert graph.count_nodes() == 0  # the pool still has connections


def test_the_factory_opens_a_usable_store(tmp_path):
    with open_context_graph(tmp_path / "g") as store:
        store.upsert_entity("a", "alice", "Person", EARLY, vector(0.1))

        assert store.get_entity("a")["label"] == "alice"


def test_the_factory_can_skip_schema_creation(tmp_path):
    store = ContextGraph(tmp_path / "g")
    store.close()

    with open_context_graph(tmp_path / "g", initialize=False, migrate=False) as opened:
        tables = {row[1] for row in opened.query("CALL SHOW_TABLES() RETURN *")}

    assert NODE_TABLE not in tables
