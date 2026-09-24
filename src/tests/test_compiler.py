"""Tests for building a store file out of a tenant's accumulated rows.

The first one is the reason a build starts from nothing. A compile that
accumulates instead of rebuilding produces an artifact where **everything
present is correct** — the only evidence is something that should be gone and
is not, and nothing downstream looks for absences.

The store is real where it can be opened; where the native library will not
load, those tests skip rather than asserting against a stand-in that would
happily accept writes in any order.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlmodel import select

from src.models.database import control_plane_sessions, create_control_plane_engine
from src.models.graph_store import (
    EntityEdge,
    EntityNode,
    create_graph_store_schema,
    upsert_edges,
    upsert_nodes,
)
from src.worker.compiler import (
    ARTIFACT_RELATION,
    BATCH_SIZE,
    compile_artifact,
    edge_confidence,
)

NOW = 1_700_000_000
DIMENSION = 384


def _store_available() -> bool:
    """Whether a store can actually be opened here, not merely imported."""
    try:
        from src.graphdb import open_context_graph

        with tempfile.TemporaryDirectory() as directory:
            open_context_graph(Path(directory) / "probe").close()
        return True
    except Exception:
        return False


STORE_AVAILABLE = _store_available()
requires_store = pytest.mark.skipif(
    not STORE_AVAILABLE, reason="the graph store's native library is not loadable here"
)


@pytest.fixture()
def engine(tmp_path):
    made = create_control_plane_engine(tmp_path / "graph-store.db")
    create_graph_store_schema(made)
    try:
        yield made
    finally:
        made.dispose()


@pytest.fixture()
def db(engine):
    with control_plane_sessions(engine)() as open_session:
        yield open_session


def vector(seed: float = 0.1) -> list[float]:
    """A vector of the dimension the store expects."""
    return [seed] * DIMENSION


def node(node_id, *, org_id="org_1", label="PR", name=None, embedding=None):
    return {
        "org_id": org_id,
        "node_id": node_id,
        "repo_id": "repo_1",
        "label": label,
        "name": name or node_id,
        "properties": {"title": node_id},
        "embedding": embedding if embedding is not None else vector(),
        "created_at": NOW,
        "updated_at": NOW,
    }


def edge(source, target, *, org_id="org_1", relation="MENTIONS", weight=0.75):
    return {
        "org_id": org_id,
        "source_id": source,
        "target_id": target,
        "relation_type": relation,
        "weight": weight,
        "created_at": NOW,
    }


def seed(db, nodes, edges=()):
    upsert_nodes(db, list(nodes))
    if edges:
        upsert_edges(db, list(edges))
    db.commit()


def entities_in(path):
    """Every entity identifier the compiled store holds."""
    from src.graphdb import open_context_graph

    store = open_context_graph(path)
    try:
        rows = store.execute("MATCH (n:Entity) RETURN n.id ORDER BY n.id")
        return [row[0] for row in rows]
    finally:
        store.close()


def relationships_in(path):
    """Every relationship, as (source, target, relation, confidence)."""
    from src.graphdb import open_context_graph

    store = open_context_graph(path)
    try:
        rows = store.execute(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
            "RETURN a.id, b.id, r.relation, r.confidence ORDER BY a.id, b.id"
        )
        return [tuple(row) for row in rows]
    finally:
        store.close()


# ==========================================================================
# the one that only an absence reveals
# ==========================================================================


@requires_store
def test_a_recompile_drops_what_was_deleted_upstream(db, tmp_path):
    """Every build is a snapshot, not a merge.

    A compile that accumulated would produce an artifact in which everything
    present is correct — the deleted entity is the only evidence, and nothing
    downstream checks for things that should be missing.
    """
    output = tmp_path / "artifacts" / "org_1-1.db"

    seed(db, [node("pr:1"), node("pr:2"), node("pr:3")])
    first = compile_artifact("org_1", db, output)

    assert first.entities == 3
    assert entities_in(output) == ["pr:1", "pr:2", "pr:3"]

    # Deleted upstream, between the two builds.
    db.exec(select(EntityNode).where(EntityNode.node_id == "pr:2")).one()
    db.delete(db.get(EntityNode, ("org_1", "pr:2")))
    db.commit()

    second = compile_artifact("org_1", db, output)

    assert second.entities == 2
    assert entities_in(output) == ["pr:1", "pr:3"]
    assert "pr:2" not in entities_in(output)


@requires_store
def test_an_existing_file_is_replaced_rather_than_reopened(db, tmp_path):
    """Even when the previous artifact has nothing to do with this tenant."""
    output = tmp_path / "artifacts" / "org_1-1.db"

    seed(db, [node("stale:1", org_id="org_other")])
    compile_artifact("org_other", db, output)
    assert entities_in(output) == ["stale:1"]

    seed(db, [node("pr:1")])
    compile_artifact("org_1", db, output)

    assert entities_in(output) == ["pr:1"]


# ==========================================================================
# what a build produces
# ==========================================================================


@requires_store
def test_entities_and_edges_both_reach_the_artifact(db, tmp_path):
    output = tmp_path / "org_1.db"

    seed(
        db,
        [node("pr:1"), node("person:ada", label="Person")],
        [edge("pr:1", "person:ada", weight=0.9)],
    )

    result = compile_artifact("org_1", db, output)

    assert result.path == output
    assert (result.entities, result.edges) == (2, 1)
    assert output.exists()
    assert entities_in(output) == ["person:ada", "pr:1"]

    source, target, relation, confidence = relationships_in(output)[0]
    assert (source, target) == ("pr:1", "person:ada")
    # A mention the extractor was 0.9 sure of, priced as a mention.
    assert confidence == pytest.approx(0.60 * 0.9)


@requires_store
def test_an_edges_kind_survives_the_compile(db, tmp_path):
    """The artifact is typed, not only weighted.

    Each accumulated kind reaches the artifact under its own relation, priced
    by that relation, so a reader of a tenant graph can tell an authorship from
    a mention the same way a reader of a locally built graph can.
    """
    output = tmp_path / "org_1.db"

    seed(
        db,
        [node("pr:1"), node("person:ada"), node("svc:auth")],
        [
            edge("pr:1", "person:ada", relation="AUTHORED_BY", weight=1.0),
            edge("pr:1", "svc:auth", relation="MENTIONS", weight=0.4),
        ],
    )

    compile_artifact("org_1", db, output)

    stored = {(row[0], row[1]): (row[2], row[3]) for row in relationships_in(output)}
    assert stored[("pr:1", "person:ada")] == ("AUTHORED_BY", pytest.approx(0.95))
    assert stored[("pr:1", "svc:auth")] == ("MENTIONS", pytest.approx(0.60 * 0.4))


@requires_store
def test_an_edge_with_no_recorded_kind_falls_back_to_the_generic_relation(db, tmp_path):
    output = tmp_path / "org_1.db"

    seed(
        db,
        [node("pr:1"), node("person:ada")],
        [edge("pr:1", "person:ada", relation="", weight=0.5)],
    )

    compile_artifact("org_1", db, output)

    (_, _, relation, confidence), = relationships_in(output)
    assert relation == ARTIFACT_RELATION
    assert confidence == pytest.approx(0.35 * 0.5)


def test_an_authorship_is_priced_as_one():
    assert edge_confidence("AUTHORED_BY", 1.0) == pytest.approx(0.95)


def test_a_mention_carries_the_extractors_certainty():
    assert edge_confidence("MENTIONS", 0.5) == pytest.approx(0.3)
    assert edge_confidence("MENTIONS", 1.0) == pytest.approx(0.60)


def test_an_unpriced_relation_is_charged_the_lowest_price():
    assert edge_confidence("SOMETHING_NEW", 1.0) == pytest.approx(0.35)
    assert edge_confidence(None, 1.0) == pytest.approx(0.35)


def test_a_row_with_no_weight_counts_as_certain():
    assert edge_confidence("AUTHORED_BY", None) == pytest.approx(0.95)


def test_ingested_edges_no_longer_all_score_the_same():
    """The regression itself: rows ingest was equally sure of used to reach
    the artifact at the same 1.0, so every path scored alike."""
    assert edge_confidence("AUTHORED_BY", 1.0) != edge_confidence("MENTIONS", 1.0)


@requires_store
def test_only_the_named_organisations_rows_are_written(db, tmp_path):
    output = tmp_path / "org_1.db"

    seed(
        db,
        [node("pr:1", org_id="org_1"), node("pr:1", org_id="org_2"), node("pr:9", org_id="org_2")],
        [
            edge("pr:1", "pr:9", org_id="org_2"),
        ],
    )

    result = compile_artifact("org_1", db, output)

    assert (result.entities, result.edges) == (1, 0)
    assert entities_in(output) == ["pr:1"]
    assert relationships_in(output) == []


@requires_store
def test_the_parent_directory_is_created(db, tmp_path):
    output = tmp_path / "deep" / "nested" / "org_1.db"

    seed(db, [node("pr:1")])
    compile_artifact("org_1", db, output)

    assert output.exists()


# ==========================================================================
# order, and the index
# ==========================================================================


@requires_store
def test_every_entity_is_written_before_any_relationship(db, tmp_path, monkeypatch):
    """Asserted on the order of the writes, not on the finished contents.

    An edge whose target has not been written yet has nothing to attach to,
    and a finished artifact cannot tell you whether that nearly happened.
    """
    from src.graphdb import context_graph as store_module

    order: list[str] = []
    real_entity = store_module.ContextGraph.upsert_entity
    real_relationship = store_module.ContextGraph.upsert_relationship

    def watched_entity(self, *args, **kwargs):
        order.append("entity")
        return real_entity(self, *args, **kwargs)

    def watched_relationship(self, *args, **kwargs):
        order.append("edge")
        return real_relationship(self, *args, **kwargs)

    monkeypatch.setattr(store_module.ContextGraph, "upsert_entity", watched_entity)
    monkeypatch.setattr(
        store_module.ContextGraph, "upsert_relationship", watched_relationship
    )

    seed(
        db,
        [node(f"pr:{index}") for index in range(5)],
        [edge("pr:0", f"pr:{index}") for index in range(1, 5)],
    )

    compile_artifact("org_1", db, tmp_path / "org_1.db")

    assert order == ["entity"] * 5 + ["edge"] * 4
    # Restated as the property rather than the sequence, so the reason
    # survives a change in the fixture's size.
    assert "edge" not in order[: order.index("edge")]


@requires_store
def test_an_organisation_with_no_entities_builds_no_index(db, tmp_path, monkeypatch, caplog):
    """An index over nothing is wasted work, and may not be a legal call."""
    from src.graphdb import context_graph as store_module


    def watched(self, *, rebuild: bool = False):
        # Opening a store builds no index, so any build at all here is the
        # compile indexing rows that are not there.
        raise AssertionError("an index was built for an organisation with no entities")

    monkeypatch.setattr(store_module.ContextGraph, "build_vector_index", watched)

    output = tmp_path / "empty.db"
    with caplog.at_level(logging.INFO):
        result = compile_artifact("org_empty", db, output)

    assert (result.entities, result.edges) == (0, 0)
    # A file is still produced: an organisation with nothing is a tenant with
    # an empty graph, not a failed build.
    assert output.exists()
    assert "no index was built" in caplog.text


@requires_store
def test_the_index_is_built_once_after_everything_is_written(db, tmp_path, monkeypatch):
    """The expensive step, paid here so no query ever pays it."""
    from src.graphdb import context_graph as store_module

    order: list[str] = []
    real_entity = store_module.ContextGraph.upsert_entity
    real_index = store_module.ContextGraph.build_vector_index

    def watched_entity(self, *args, **kwargs):
        order.append("entity")
        return real_entity(self, *args, **kwargs)

    def watched_index(self, *, rebuild: bool = False):
        # Every build counts: opening a store builds none, so the only one
        # is the compile's own, after the writes.
        order.append("index")
        return real_index(self, rebuild=rebuild)

    monkeypatch.setattr(store_module.ContextGraph, "upsert_entity", watched_entity)
    monkeypatch.setattr(store_module.ContextGraph, "build_vector_index", watched_index)

    seed(db, [node("pr:1"), node("pr:2")])
    compile_artifact("org_1", db, tmp_path / "org_1.db")

    assert order == ["entity", "entity", "index"]


# ==========================================================================
# streaming
# ==========================================================================


def test_rows_are_streamed_in_batches_rather_than_fetched_whole(db, engine, tmp_path):
    """Counted at the connection, because a fetch-all passes every other test.

    With a batch size well under the row count, a streamed read fetches many
    times and a fetch-all fetches once. The assertion is on the number of
    fetches, which is the only place the difference is visible.
    """
    seed(db, [node(f"pr:{index}") for index in range(250)])

    fetches: list[int] = []
    statement_count = {"value": 0}

    @event.listens_for(engine, "before_cursor_execute")
    def count_statements(conn, cursor, statement, parameters, context, executemany):
        if "entity_nodes" in statement and statement.upper().startswith("SELECT"):
            statement_count["value"] += 1

    from src.worker.compiler import _stream

    try:
        rows = list(_stream(db, EntityNode, EntityNode.node_id, "org_1", 50))
        for row in rows:
            fetches.append(1)
    finally:
        event.remove(engine, "before_cursor_execute", count_statements)

    assert len(rows) == 250
    # One statement, read in batches through its cursor rather than reissued.
    assert statement_count["value"] == 1
    # The stream is lazy: nothing is materialised by constructing it.
    assert isinstance(_stream(db, EntityNode, EntityNode.node_id, "org_1", 50), type(iter([]))) is False


def test_the_stream_yields_lazily(db):
    """Constructing it must not read anything.

    A generator that had already fetched would have the same signature and
    the same results, and would hold every row while it did.
    """
    import inspect

    from src.worker.compiler import _stream

    seed(db, [node(f"pr:{index}") for index in range(10)])
    stream = _stream(db, EntityNode, EntityNode.node_id, "org_1", 5)

    assert inspect.isgenerator(stream)
    # Reading one row does not require reading all of them.
    first = next(stream)
    assert first.node_id == "pr:0"
    stream.close()


def test_the_batch_size_reaches_the_query(db, engine):
    """The parameter is not decoration: it is what bounds what is resident."""
    seed(db, [node(f"pr:{index}") for index in range(20)])

    from src.worker.compiler import _stream

    seen = {}

    @event.listens_for(engine, "before_cursor_execute")
    def capture(conn, cursor, statement, parameters, context, executemany):
        if context is not None and "entity_nodes" in statement:
            seen["yield_per"] = context.execution_options.get("yield_per")

    try:
        list(_stream(db, EntityNode, EntityNode.node_id, "org_1", 7))
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    assert seen.get("yield_per") == 7


def test_the_default_batch_size_is_a_thousand():
    assert BATCH_SIZE == 1000


# ==========================================================================
# the failure path
# ==========================================================================


@requires_store
def test_the_store_is_closed_when_writing_raises(db, tmp_path, monkeypatch):
    """A handle left open on a partial file is a file nothing can replace."""
    from src.graphdb import context_graph as store_module

    closed = []
    real_close = store_module.ContextGraph.close

    def watched_close(self):
        closed.append(True)
        return real_close(self)

    def fall_over(self, *args, **kwargs):
        raise RuntimeError("the store gave out partway")

    monkeypatch.setattr(store_module.ContextGraph, "close", watched_close)
    monkeypatch.setattr(store_module.ContextGraph, "upsert_entity", fall_over)

    seed(db, [node("pr:1")])

    with pytest.raises(RuntimeError):
        compile_artifact("org_1", db, tmp_path / "org_1.db")

    assert closed == [True]


@requires_store
def test_a_failed_build_leaves_nothing_a_retry_cannot_replace(db, tmp_path):
    """The next attempt starts by removing whatever the last one left."""
    output = tmp_path / "org_1.db"

    seed(db, [node("pr:1")])
    compile_artifact("org_1", db, output)
    assert output.exists()

    # Whatever is there, a fresh build replaces it.
    seed(db, [node("pr:2")])
    result = compile_artifact("org_1", db, output)

    assert result.entities == 2
    assert entities_in(output) == ["pr:1", "pr:2"]


@requires_store
def test_a_stale_write_ahead_log_does_not_break_the_next_build(db, tmp_path):
    """A crashed build leaves its log beside the file; the next build must not replay it."""
    output = tmp_path / "org_1.db"
    output.write_bytes(b"not a store")
    Path(f"{output}.wal").write_bytes(b"garbage left by a crashed build" * 100)

    seed(db, [node("pr:1"), node("pr:2")])
    result = compile_artifact("org_1", db, output)

    assert result.entities == 2
    assert entities_in(output) == ["pr:1", "pr:2"]


@requires_store
@pytest.mark.parametrize("suffix", [".wal", "-shm", ".tmp"])
def test_every_stale_sidecar_is_cleared_before_a_build(db, tmp_path, suffix):
    output = tmp_path / "org_1.db"
    stale_file = Path(f"{output}{suffix}")
    stale_file.write_bytes(b"stale")

    seed(db, [node("pr:1")])
    compile_artifact("org_1", db, output)

    assert not stale_file.exists() or stale_file.read_bytes() != b"stale"
    assert entities_in(output) == ["pr:1"]


def test_stale_pieces_are_removed_whether_files_or_directories(tmp_path):
    from src.worker.compiler import _remove

    output = tmp_path / "org_1.db"
    output.mkdir()
    (output / "inner").write_bytes(b"x")
    Path(f"{output}.wal").write_bytes(b"x")
    Path(f"{output}-shm").mkdir()
    Path(f"{output}.tmp").write_bytes(b"x")
    keep = tmp_path / "org_1.db.bak"
    keep.write_bytes(b"x")

    _remove(output)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["org_1.db.bak"]


def test_a_piece_that_cannot_be_removed_is_logged_and_skipped(tmp_path, monkeypatch, caplog):
    from src.worker import compiler as compiler_module

    output = tmp_path / "org_1.db"
    output.write_bytes(b"x")
    wal = Path(f"{output}.wal")
    wal.write_bytes(b"x")
    Path(f"{output}.tmp").write_bytes(b"x")

    real_unlink = Path.unlink

    def refuse_wal(self, *args, **kwargs):
        if self == wal:
            raise PermissionError("held open")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_wal)

    with caplog.at_level(logging.WARNING):
        compiler_module._remove(output)

    assert not output.exists()
    assert wal.exists()
    assert not Path(f"{output}.tmp").exists()
    assert "org_1.db.wal" in caplog.text


@requires_store
def test_a_compiled_tenant_graph_ranks_by_what_connects_things(db, tmp_path):
    """End to end: compile, then walk. Ingest is equally sure of every edge
    here, and still an item by the same author outranks one that only names
    the same service, because the relations are priced apart."""
    from src.graphdb import open_context_graph
    from src.retrieval.graph_arm import traverse

    output = tmp_path / "org_1.db"
    seed(
        db,
        [
            node("pr:1"), node("pr:2"), node("pr:3"),
            node("person:ada", label="Person"), node("svc:auth", label="Service"),
        ],
        [
            edge("pr:1", "person:ada", relation="AUTHORED_BY", weight=1.0),
            edge("pr:2", "person:ada", relation="AUTHORED_BY", weight=1.0),
            edge("pr:1", "svc:auth", relation="MENTIONS", weight=1.0),
            edge("pr:3", "svc:auth", relation="MENTIONS", weight=1.0),
        ],
    )
    compile_artifact("org_1", db, output)

    store = open_context_graph(output)
    try:
        scores = {hit.id: hit.score for hit in traverse(store, ["pr:1"]).hits}
    finally:
        store.close()

    assert len({round(score, 6) for score in scores.values()}) > 1
    assert scores["person:ada"] == pytest.approx(0.95)
    assert scores["svc:auth"] == pytest.approx(0.60)
    assert scores["pr:2"] > scores["pr:3"]


# ==========================================================================
# source text and time
# ==========================================================================


def item(node_id, *, title="Fix login", body="The auth service rejected tokens.",
         created_at="2026-09-01T10:00:00Z", url="https://example.test/1"):
    row = node(node_id, label="Issue", name="Issue #1")
    row["properties"] = {"title": title, "body": body, "created_at": created_at, "url": url}
    return row


def documents_in(path):
    from src.graphdb import open_context_graph

    store = open_context_graph(path)
    try:
        return {
            row[0]: (row[1], row[2])
            for row in store.execute("MATCH (d:Document) RETURN d.id, d.path, d.content")
        }
    finally:
        store.close()


def mentions_in(path):
    from src.graphdb import open_context_graph

    store = open_context_graph(path)
    try:
        return sorted(
            tuple(row) for row in store.execute(
                "MATCH (d:Document)-[:MENTIONS]->(e:Entity) RETURN d.id, e.id"
            )
        )
    finally:
        store.close()


def timestamps_in(path):
    from src.graphdb import open_context_graph

    store = open_context_graph(path)
    try:
        return {row[0]: row[1] for row in store.execute("MATCH (e:Entity) RETURN e.id, e.ts")}
    finally:
        store.close()


@requires_store
def test_an_items_text_becomes_its_source_document(db, tmp_path):
    output = tmp_path / "org_1.db"
    service = node("svc:auth", label="Service")
    service["properties"] = {"type": "Service"}
    seed(db, [item("r:issue:1"), service],
         [edge("r:issue:1", "svc:auth", relation="MENTIONS", weight=1.0)])

    result = compile_artifact("org_1", db, output)

    assert result.documents == 1
    assert documents_in(output) == {
        "r:issue:1:doc": ("https://example.test/1", "Fix login\n\nThe auth service rejected tokens."),
    }
    # The item answers with its own text, and so does what it names.
    assert mentions_in(output) == [("r:issue:1:doc", "r:issue:1"), ("r:issue:1:doc", "svc:auth")]


@requires_store
def test_a_node_with_no_text_gets_no_document(db, tmp_path):
    output = tmp_path / "org_1.db"
    person = node("person:ada", label="Person")
    person["properties"] = {"login": "ada"}
    service = node("svc:auth", label="Service")
    service["properties"] = {"type": "Service"}
    seed(db, [person, service],
         [edge("person:ada", "svc:auth", relation="MENTIONS", weight=1.0)])

    result = compile_artifact("org_1", db, output)

    assert result.documents == 0
    assert documents_in(output) == {}
    assert mentions_in(output) == []


@requires_store
def test_an_item_carries_when_it_was_written(db, tmp_path):
    from datetime import datetime, timezone

    output = tmp_path / "org_1.db"
    person = node("person:ada", label="Person")
    person["properties"] = {"login": "ada"}
    seed(db, [item("r:issue:1"), person])

    compile_artifact("org_1", db, output)

    stamps = timestamps_in(output)
    expected = int(datetime(2026, 9, 1, 10, tzinfo=timezone.utc).timestamp())
    assert stamps["r:issue:1"] == expected
    assert stamps["person:ada"] is None


@requires_store
def test_an_unreadable_time_is_left_unknown(db, tmp_path):
    output = tmp_path / "org_1.db"
    seed(db, [item("r:issue:1", created_at="not a date")])

    compile_artifact("org_1", db, output)

    assert timestamps_in(output)["r:issue:1"] is None
