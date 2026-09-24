"""Tests for the tables a tenant's graph accumulates in.

The first one is the reason the key is composite. A node identifier means
something only inside the organisation that produced it, and two tenants
holding the same one is ordinary — but a single-column key would pass every
other test in this file and fail only when a second tenant arrived with a
familiar-looking identifier.

Statement counts are taken from the connection rather than inferred, because
"one statement for the batch" is the claim and a loop of inserts satisfies
every assertion about the resulting rows.
"""

from __future__ import annotations

import pytest
from sqlalchemy import event, inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, SQLModel, select

from src.models.control_plane import (
    CONTROL_PLANE_TABLES,
    Organization,
    create_control_plane_schema,
)
from src.models.database import control_plane_sessions, create_control_plane_engine
from src.models.graph_store import (
    GRAPH_STORE_TABLES,
    EntityEdge,
    EntityNode,
    create_graph_store_schema,
    ensure_graph_store_schema,
    reset_schema_guard,
    upsert_edges,
    upsert_nodes,
)

NOW = 1_700_000_000
LATER = NOW + 3600


@pytest.fixture()
def engine(tmp_path):
    made = create_control_plane_engine(tmp_path / "graph-store.db")
    create_graph_store_schema(made)
    try:
        yield made
    finally:
        made.dispose()


@pytest.fixture()
def sessions(engine):
    return control_plane_sessions(engine)


@pytest.fixture()
def db(sessions):
    with sessions() as open_session:
        yield open_session


@pytest.fixture(autouse=True)
def clean_guard():
    """The per-process guard is module state; no test may inherit it."""
    reset_schema_guard()
    yield
    reset_schema_guard()


def node(org_id="org_1", node_id="pr:41", **overrides):
    row = {
        "org_id": org_id,
        "node_id": node_id,
        "repo_id": "repo_1",
        "label": "PR",
        "name": "the auth change",
        "properties": {"state": "merged"},
        "embedding": [0.1, 0.2, 0.3],
        "created_at": NOW,
        "updated_at": NOW,
    }
    row.update(overrides)
    return row


def edge(org_id="org_1", source="pr:41", target="person:ada", relation="REVIEWED_BY", **overrides):
    row = {
        "org_id": org_id,
        "source_id": source,
        "target_id": target,
        "relation_type": relation,
        "weight": 1.0,
        "created_at": NOW,
    }
    row.update(overrides)
    return row


def statements_during(engine, work):
    """Every statement the connection issued while ``work`` ran."""
    seen: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def record(conn, cursor, statement, parameters, context, executemany):
        seen.append(" ".join(statement.split()))

    try:
        work()
    finally:
        event.remove(engine, "before_cursor_execute", record)

    return seen


# ==========================================================================
# the reason the key is composite
# ==========================================================================


def test_two_organisations_may_hold_the_same_node_identifier(db):
    """A node identifier is unique within a tenant, not across them.

    Both of these are pull request 41, and they are different entities. A
    single-column key would make the second overwrite the first, and every
    other test here would still pass.
    """
    upsert_nodes(
        db,
        [
            node("org_a", "pr:41", name="one tenant's pull request"),
            node("org_b", "pr:41", name="another tenant's, entirely"),
        ],
    )
    db.commit()

    stored = db.exec(select(EntityNode).order_by(EntityNode.org_id)).all()

    assert [(row.org_id, row.node_id) for row in stored] == [
        ("org_a", "pr:41"),
        ("org_b", "pr:41"),
    ]
    assert stored[0].name != stored[1].name


def test_two_organisations_may_hold_the_same_relationship(db):
    """The same, one level up: an edge's identity starts with the tenant."""
    upsert_edges(db, [edge("org_a"), edge("org_b")])
    db.commit()

    assert len(db.exec(select(EntityEdge)).all()) == 2


# ==========================================================================
# the schema calls
# ==========================================================================


def test_the_schema_call_creates_both_tables(engine):
    assert set(inspect(engine).get_table_names()) == {
        "entity_nodes",
        "entity_edges",
    }


def test_the_schema_call_does_not_create_the_control_plane_tables(tmp_path):
    """Sharing a database is not sharing a schema call.

    Each call brings up what it owns. A call that created everything in the
    registry would put the other's tables in whichever database was opened
    first.
    """
    made = create_control_plane_engine(tmp_path / "only-graph.db")
    try:
        create_graph_store_schema(made)
        created = set(inspect(made).get_table_names())
    finally:
        made.dispose()

    for table in CONTROL_PLANE_TABLES:
        assert table.name not in created


def test_a_model_from_another_concern_is_not_created(tmp_path):
    """The same assertion the control plane's schema call carries.

    Declared on the shared registry, which is the case that bites: a module
    imported for an unrelated reason must not put its tables here.
    """

    class SomethingElseEntirely(SQLModel, table=True):
        __tablename__ = "something_else_entirely"

        id: int = Field(primary_key=True)

    made = create_control_plane_engine(tmp_path / "not-everything.db")
    try:
        create_graph_store_schema(made)
        created = set(inspect(made).get_table_names())
    finally:
        made.dispose()
        SQLModel.metadata.remove(SomethingElseEntirely.__table__)

    assert created == {table.name for table in GRAPH_STORE_TABLES}
    assert "something_else_entirely" not in created


def test_both_sets_of_tables_can_share_one_database(tmp_path):
    """Which is the arrangement this is built for.

    A compile reads a tenant's entities and writes an artifact row about what
    it produced. Those are one transaction only if they are one database.
    """
    made = create_control_plane_engine(tmp_path / "shared.db")
    try:
        create_control_plane_schema(made)
        create_graph_store_schema(made)
        created = set(inspect(made).get_table_names())

        with control_plane_sessions(made)() as db:
            db.add(
                Organization(
                    org_id="org_1",
                    name="Acme",
                    plan="team",
                    status="active",
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
            upsert_nodes(db, [node("org_1")])
            # One commit over both, which is the point of one database.
            db.commit()

            assert db.get(Organization, "org_1") is not None
            assert db.get(EntityNode, ("org_1", "pr:41")) is not None
    finally:
        made.dispose()

    expected = {table.name for table in CONTROL_PLANE_TABLES} | {
        table.name for table in GRAPH_STORE_TABLES
    }
    assert created == expected


def test_the_indexes_all_lead_with_the_organisation(engine):
    """No query here is unscoped, so no index may be."""
    inspector = inspect(engine)

    for table in ("entity_nodes", "entity_edges"):
        compound = [
            index
            for index in inspector.get_indexes(table)
            if len(index["column_names"]) > 1
        ]
        assert compound, table
        for index in compound:
            assert index["column_names"][0] == "org_id", (table, index["name"])


def test_traversal_is_indexed_in_both_directions(engine):
    """An index on the source cannot answer what points at a node."""
    names = {index["name"] for index in inspect(engine).get_indexes("entity_edges")}

    assert "entity_edges_org_source" in names
    assert "entity_edges_org_target" in names


# ==========================================================================
# the per-process guard
# ==========================================================================


def test_the_guard_creates_once_and_then_does_nothing(tmp_path):
    """Asked before every batch, so it must not cost a round trip every time."""
    made = create_control_plane_engine(tmp_path / "guarded.db")
    try:
        assert ensure_graph_store_schema(made) is True
        assert set(inspect(made).get_table_names()) == {
            "entity_nodes",
            "entity_edges",
        }

        second = statements_during(made, lambda: ensure_graph_store_schema(made))

        assert second == []
    finally:
        made.dispose()


def test_the_guard_reports_that_it_did_nothing_the_second_time(tmp_path):
    made = create_control_plane_engine(tmp_path / "guarded-twice.db")
    try:
        assert ensure_graph_store_schema(made) is True
        assert ensure_graph_store_schema(made) is False
        assert ensure_graph_store_schema(made) is False
    finally:
        made.dispose()


def test_the_guard_can_be_forgotten(tmp_path):
    """A process that changed database has to be able to ask again."""
    first = create_control_plane_engine(tmp_path / "one.db")
    second = create_control_plane_engine(tmp_path / "two.db")
    try:
        assert ensure_graph_store_schema(first) is True
        # Without forgetting, the second database never gets its tables.
        assert ensure_graph_store_schema(second) is False
        assert inspect(second).get_table_names() == []

        reset_schema_guard()

        assert ensure_graph_store_schema(second) is True
        assert set(inspect(second).get_table_names()) == {
            "entity_nodes",
            "entity_edges",
        }
    finally:
        first.dispose()
        second.dispose()


# ==========================================================================
# writing nodes
# ==========================================================================


def test_a_node_round_trips_with_its_payload_and_vector(db):
    upsert_nodes(db, [node()])
    db.commit()

    stored = db.get(EntityNode, ("org_1", "pr:41"))

    assert stored.label == "PR"
    assert stored.name == "the auth change"
    assert stored.repo_id == "repo_1"
    assert stored.properties == {"state": "merged"}
    assert stored.embedding == [0.1, 0.2, 0.3]


def test_upserting_the_same_node_refreshes_rather_than_duplicating(db):
    """Re-learning an entity updates what is known about it."""
    assert upsert_nodes(db, [node()]) == 1
    db.commit()

    assert (
        upsert_nodes(
            db,
            [
                node(
                    name="the auth change, revised",
                    properties={"state": "closed"},
                    embedding=[0.9, 0.8, 0.7],
                    updated_at=LATER,
                )
            ],
        )
        == 1
    )
    db.commit()

    rows = db.exec(select(EntityNode)).all()
    assert len(rows) == 1

    stored = rows[0]
    assert stored.name == "the auth change, revised"
    assert stored.properties == {"state": "closed"}
    assert stored.embedding == [0.9, 0.8, 0.7]
    assert stored.updated_at == LATER
    # When it was first seen is not overwritten by learning about it again.
    assert stored.created_at == NOW


def test_an_empty_batch_writes_nothing_and_does_not_fail(db):
    assert upsert_nodes(db, []) == 0
    assert upsert_edges(db, []) == 0
    assert db.exec(select(EntityNode)).all() == []


def test_a_batch_of_nodes_is_one_statement(engine, sessions):
    """A statement per row turns one round trip into two hundred."""
    rows = [node(node_id=f"pr:{index}") for index in range(200)]

    with sessions() as db:
        statements = statements_during(engine, lambda: upsert_nodes(db, rows))
        db.commit()

    writes = [
        statement for statement in statements if statement.upper().startswith("INSERT")
    ]
    assert len(writes) == 1

    with sessions() as reading:
        assert len(reading.exec(select(EntityNode)).all()) == 200


# ==========================================================================
# writing edges
# ==========================================================================


def test_an_edge_round_trips(db):
    upsert_edges(db, [edge(weight=0.5)])
    db.commit()

    stored = db.exec(select(EntityEdge)).one()

    assert stored.org_id == "org_1"
    assert stored.source_id == "pr:41"
    assert stored.target_id == "person:ada"
    assert stored.relation_type == "REVIEWED_BY"
    assert stored.weight == 0.5
    assert stored.edge_id


def test_an_edge_weighs_one_unless_something_says_otherwise(db):
    row = edge()
    del row["weight"]
    upsert_edges(db, [row])
    db.commit()

    assert db.exec(select(EntityEdge)).one().weight == 1.0


def test_the_same_relationship_twice_is_one_edge(db):
    """Identity is what it connects and how, not the generated handle."""
    assert upsert_edges(db, [edge()]) == 1
    db.commit()

    assert upsert_edges(db, [edge(weight=2.0)]) == 1
    db.commit()

    rows = db.exec(select(EntityEdge)).all()
    assert len(rows) == 1
    assert rows[0].weight == 2.0


def test_the_same_pair_under_a_different_relation_is_a_second_edge(db):
    """Two things can be related in more than one way at once."""
    upsert_edges(
        db,
        [
            edge(relation="REVIEWED_BY"),
            edge(relation="AUTHORED_BY"),
        ],
    )
    db.commit()

    stored = db.exec(select(EntityEdge)).all()

    assert len(stored) == 2
    assert {row.relation_type for row in stored} == {"REVIEWED_BY", "AUTHORED_BY"}


def test_the_direction_of_an_edge_matters(db):
    """Reversing it is a different relationship, not the same one restated."""
    upsert_edges(
        db,
        [
            edge(source="pr:41", target="person:ada"),
            edge(source="person:ada", target="pr:41"),
        ],
    )
    db.commit()

    assert len(db.exec(select(EntityEdge)).all()) == 2


def test_the_constraint_refuses_a_duplicate_written_around_the_upsert(db):
    """The guarantee is the constraint, not the upsert's conflict clause.

    A writer that inserted directly must be refused too, or the uniqueness
    holds only for code that remembers to go through one function.
    """
    upsert_edges(db, [edge()])
    db.commit()

    db.add(EntityEdge(**edge()))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_a_batch_of_edges_is_one_statement(engine, sessions):
    rows = [
        edge(source=f"pr:{index}", target="person:ada") for index in range(200)
    ]

    with sessions() as db:
        statements = statements_during(engine, lambda: upsert_edges(db, rows))
        db.commit()

    writes = [
        statement for statement in statements if statement.upper().startswith("INSERT")
    ]
    assert len(writes) == 1

    with sessions() as reading:
        assert len(reading.exec(select(EntityEdge)).all()) == 200


def test_a_batch_containing_the_same_edge_twice_settles_on_one_row(db):
    """Two mentions of one relationship in a single batch."""
    upsert_edges(db, [edge(), edge(weight=3.0)])
    db.commit()

    assert len(db.exec(select(EntityEdge)).all()) == 1
