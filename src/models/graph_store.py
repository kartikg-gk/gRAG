"""What a tenant's graph actually is, between compiles.

Two tables: the entities an organisation has accumulated, and the
relationships between them. Everything ingestion learns lands here, and the
store file a process serves from is later *derived* from these rows.

That is a change in what counts as authoritative, and it is worth being
explicit about: on this path the database is the truth and the store file is a
build output, so a store that is lost or corrupted is rebuilt rather than
recovered. **Only on this path.** The single-process path still opens a store
directly and still treats it as the primary thing; nothing here touches it.

Why these live beside the control plane
---------------------------------------

Same database, same connection. The control plane holds which tenant is
served what; this holds what there is to serve. They are written by the same
processes in the same transactions — a compile reads an organisation's rows
and writes an artifact row about the result — and splitting them would make
that two connections and no transaction spanning both.

Splitting them later is one environment variable and a second engine. Merging
two databases back into one is a migration, so the reversible direction is the
one to start from.

The schema call still creates **only these two tables**, the way the control
plane's creates only its own. Both share one model registry, and a call that
created everything in it would mean importing a module for one constant was
enough to put somebody else's tables in your database.

Keys, and the tenant that scopes them
-------------------------------------

A node identifier is unique **within** an organisation and nowhere else. Two
tenants can each hold a pull request numbered 41, and they are different
nodes. So the key is the pair, and every index leads with the organisation
because there is no query here that is not scoped to one tenant.

An edge carries a generated key so a row has a stable handle, but its
*identity* is the four-tuple of organisation, source, target and relation
type. The constraint on those four is what stops the same relationship being
recorded twice; the surrogate is a handle, not an identity.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Column,
    Engine,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlmodel import Field, Session, SQLModel

#: The relationship kinds this project records between entities.
#:
#: Stored on every edge. What reads it is a separate question from what this
#: table holds -- a consumer that ignores it does not make the column
#: unnecessary, because the row is the record of what was found.
RELATION_TYPE_MAX = 64


class EntityNode(SQLModel, table=True):
    """One entity in one organisation's graph."""

    __tablename__ = "entity_nodes"
    __table_args__ = (
        # Both compound, both leading with the organisation: a scan that did
        # not start there would read every tenant's rows to answer a question
        # about one.
        Index("entity_nodes_org_repo", "org_id", "repo_id"),
        Index("entity_nodes_org_label", "org_id", "label"),
    )

    #: The pair is the key. A node identifier means something only inside the
    #: organisation that produced it, and two tenants holding the same one is
    #: ordinary rather than a collision.
    org_id: str = Field(primary_key=True)
    node_id: str = Field(primary_key=True)

    #: Which repository this came from, when it came from one. Nullable
    #: because not every entity is a repository's -- a person is a person
    #: across all of them.
    repo_id: Optional[str] = Field(default=None, index=True)

    label: str
    name: str

    #: Whatever the extractor learned about this entity. No fixed shape on
    #: purpose: the set of things worth recording about a pull request is not
    #: the set worth recording about a person, and a column per field would
    #: be a migration every time one is added.
    properties: Optional[dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )

    #: The vector, as a list. Stored beside the entity rather than in a
    #: separate index because this table is the durable record and the index
    #: is built from it.
    embedding: Optional[list[float]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )

    created_at: int
    updated_at: int


class EntityEdge(SQLModel, table=True):
    """One relationship, in one organisation's graph."""

    __tablename__ = "entity_edges"
    __table_args__ = (
        # The identity. Not the primary key -- that is the surrogate below --
        # because what makes two edges the same is what they connect and how,
        # and a generated key would happily store that twice.
        UniqueConstraint(
            "org_id",
            "source_id",
            "target_id",
            "relation_type",
            name="entity_edges_one_per_relation",
        ),
        # Traversal runs both ways, and one index cannot answer both
        # directions: an index on the source is no help to "what points at
        # this".
        Index("entity_edges_org_source", "org_id", "source_id"),
        Index("entity_edges_org_target", "org_id", "target_id"),
    )

    #: A handle for the row. Generated rather than composed, so a reference to
    #: one edge is one value.
    edge_id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)

    org_id: str = Field(index=True)
    source_id: str
    target_id: str
    relation_type: str = Field(max_length=RELATION_TYPE_MAX)

    #: How strongly the two are related. One unless something says otherwise.
    weight: float = Field(default=1.0)

    created_at: int


#: Exactly the tables this module owns, in creation order.
GRAPH_STORE_MODELS = (EntityNode, EntityEdge)
GRAPH_STORE_TABLES = tuple(model.__table__ for model in GRAPH_STORE_MODELS)

#: Whether this process has already brought the tables up.
#:
#: Module level, so it is per process and per interpreter -- see
#: ``ensure_graph_store_schema``.
_schema_ready = False


def create_graph_store_schema(engine: Engine) -> None:
    """Bring these two tables up. Safe to call on every start.

    Names its tables rather than creating everything the registry knows, for
    the same reason the control plane's call does: the two share a registry,
    and neither may drag the other's tables into a database that did not ask
    for them.
    """
    SQLModel.metadata.create_all(
        engine, tables=list(GRAPH_STORE_TABLES), checkfirst=True
    )


def ensure_graph_store_schema(engine: Engine) -> bool:
    """Make sure this process can write, cheaply. Returns whether it created.

    A different question from the call above, which is "set this database
    up". This one is asked by the ingestion path before every batch, and
    answering it honestly every time would mean a round trip to ask whether
    tables exist before each one.

    So the answer is remembered for the life of the process. The cost of
    being wrong is a startup that assumed tables it does not have, which
    fails immediately and loudly on the first write; the cost of asking every
    time is paid forever.
    """
    global _schema_ready
    if _schema_ready:
        return False

    create_graph_store_schema(engine)
    _schema_ready = True
    return True


def reset_schema_guard() -> None:
    """Forget that the tables were brought up. For tests, and for a process
    that has changed which database it is talking to."""
    global _schema_ready
    _schema_ready = False


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def upsert_nodes(db: Session, rows: list[dict[str, Any]]) -> int:
    """Write these entities, refreshing any that are already there.

    **One statement for the whole list.** These are called from the ingestion
    path with hundreds of rows at a time, and a statement per row turns one
    round trip into hundreds.

    Keyed on the organisation and the node identifier, so re-learning an
    entity updates what is known about it rather than storing it again.
    """
    if not rows:
        return 0

    statement = _insert_for(db).values(rows)
    db.execute(
        statement.on_conflict_do_update(
            index_elements=["org_id", "node_id"],
            set_={
                "repo_id": statement.excluded.repo_id,
                "label": statement.excluded.label,
                "name": statement.excluded.name,
                "properties": statement.excluded.properties,
                "embedding": statement.excluded.embedding,
                # Not created_at: the row is the same entity it always was,
                # and when it was first seen is worth keeping.
                "updated_at": statement.excluded.updated_at,
            },
        )
    )
    return len(rows)


def upsert_edges(db: Session, rows: list[dict[str, Any]]) -> int:
    """Write these relationships, refreshing any that are already there.

    Keyed on the four-tuple that is an edge's identity rather than on the
    generated handle, which would let the same relationship in twice under
    two different keys.
    """
    if not rows:
        return 0

    prepared = [
        {"edge_id": uuid.uuid4().hex, **row} if "edge_id" not in row else dict(row)
        for row in rows
    ]

    statement = _insert_for(db, EntityEdge).values(prepared)
    db.execute(
        statement.on_conflict_do_update(
            index_elements=["org_id", "source_id", "target_id", "relation_type"],
            set_={"weight": statement.excluded.weight},
        )
    )
    return len(prepared)


def _insert_for(db: Session, model=EntityNode):
    """The insert this database understands, with its conflict clause.

    Both dialects this project runs on spell an upsert the same way and
    neither spells it the way the generic construct does, so the statement is
    chosen from the connection rather than assumed.
    """
    dialect = db.get_bind().dialect.name

    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:  # pragma: no cover - the project runs on two dialects
        raise NotImplementedError(
            f"no upsert is written for {dialect!r}; this project runs on "
            "postgresql and sqlite"
        )

    return insert(model.__table__)
