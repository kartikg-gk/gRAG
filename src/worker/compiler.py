"""Turning a tenant's accumulated rows into a store file to serve from.

One direction, one output. This reads the graph-store tables and writes a
store, and it does nothing else — nothing here fetches, extracts, embeds,
uploads or registers anything.

Every build starts from nothing
-------------------------------

Any file already at the output path is removed first. The artifact is
*derived*, not accumulated: it is a snapshot of what the graph store holds
right now, and a version that is anything else is not a version of anything.

Building on top of a previous artifact would carry forward entities that have
since been deleted upstream, and nothing downstream could tell — the artifact
would look entirely correct, because everything in it *is* correct. Only the
absence of what should have gone would say otherwise, and nothing checks for
absences.

Streamed, one batch at a time
-----------------------------

Both tables are read through a server-side cursor and written into the store
as the rows arrive. A tenant with a million entities holds one batch in
memory, not a million rows — and a plain "read everything, then write it"
would pass every test written against a small fixture while being the thing
that fails on the tenant that matters.

Nodes first, then edges. An edge needs both endpoints present when it is
written, and interleaving them means an edge arriving before its target.

The index is built here, and this is where the cost belongs
-----------------------------------------------------------

Building the vector index is the expensive step. Doing it here means the
process that answers queries never pays for it — not at first query, not at
load. Deferring it would move CPU onto the latency-sensitive path, which is
the entire thing this build-and-ship arrangement exists to avoid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from sqlmodel import Session, select

from ..common.relations import CONFIDENCE, RELATION_CO_OCCURS, RELATION_MENTIONS
from ..models.graph_store import EntityEdge, EntityNode

logger = logging.getLogger("graphrag.worker.compiler")

#: How many rows are read at a time.
#:
#: A thousand is large enough that the round trips disappear against the work
#: of writing them and small enough that one batch is a bounded amount of
#: memory whatever the tenant's size.
BATCH_SIZE = 1000

#: The relation an edge is written under when its row records no kind.
#:
#: The generic relation the store already reserves for a link whose kind is
#: not being claimed, and the lowest-weighted in the vocabulary — the correct
#: price for an edge whose kind is unknown.
ARTIFACT_RELATION = RELATION_CO_OCCURS

#: The store itself and every sidecar a previous build may have left beside it.
STALE_SUFFIXES = ("", ".wal", "-shm", ".tmp")


@dataclass(frozen=True)
class CompiledArtifact:
    """What one build produced."""

    path: Path
    entities: int
    edges: int
    documents: int = 0


def document_id(node_id: str) -> str:
    """The id of the source document an item's text is stored under."""
    return f"{node_id}:doc"


def _document_text(properties: dict) -> str:
    """An item's title and body as one text, or nothing when it has neither."""
    title = (properties.get("title") or "").strip()
    body = (properties.get("body") or "").strip()
    return "\n\n".join(part for part in (title, body) if part)


def _created_at(properties: dict) -> datetime | None:
    """When the item was written, or ``None`` when that was not recorded.

    ``None`` rather than a guess: recency treats an unknown time as neutral,
    and rows written before this was recorded carry none.
    """
    value = properties.get("created_at")
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def edge_confidence(relation: str | None, weight: float | None) -> float:
    """What traversal multiplies by for one row: its relation's price times
    how sure ingest was of it.

    The same prices the locally built graph uses, so the two kinds of graph
    score alike. A relation the vocabulary does not price is charged the
    generic relation's, the lowest there is.
    """
    price = CONFIDENCE.get(relation or ARTIFACT_RELATION, CONFIDENCE[ARTIFACT_RELATION])
    evidence = 1.0 if weight is None else float(weight)
    return price * evidence


def compile_artifact(
    org_id: str,
    db: Session,
    output_path: str | Path,
    *,
    batch_size: int = BATCH_SIZE,
) -> CompiledArtifact:
    """Build a fresh store for ``org_id`` at ``output_path``.

    What the artifact contains
    --------------------------

    Every entity this organisation has accumulated, with its label, type,
    vector and — for an item — when it was written; every relationship between
    them, with its weight; and each item's title and body as a source document,
    linked to the item and to every entity it mentions, so an answer has the
    text behind a node and not only its name.

    Each relationship keeps its kind. The rows record two — authorship and
    mention — and each reaches the artifact under its own relation, so a
    tenant graph can be read the way a locally built one is: by what connects
    two things, not only by how strongly. An edge with no recorded kind is
    written under the generic relation.

    A row's weight is how sure ingest was that the edge exists; the artifact's
    confidence is what traversal multiplies by. The second is the first priced
    by its relation — see :func:`edge_confidence`. Without that, every edge
    ingest was sure of reaches the artifact at 1.0 and every path through the
    graph scores the same.

    Anything already at ``output_path`` is removed first, so the result is a
    snapshot of the graph store rather than a merge with whatever was there
    before.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Derived, not accumulated. See the module docstring: a build on top of a
    # previous artifact keeps entities that were deleted upstream, and the
    # result looks correct because everything in it is.
    _remove(destination)

    from ..graphdb import open_context_graph

    store = open_context_graph(destination)
    entities = 0
    edges = 0
    documents = 0

    try:
        # Every node before any edge: an edge whose target has not been
        # written yet has nothing to attach to.
        for node in _stream(db, EntityNode, EntityNode.node_id, org_id, batch_size):
            properties = node.properties or {}
            store.upsert_entity(
                node.node_id,
                node.name,
                node.label,
                timestamp=_created_at(properties),
                embedding=node.embedding,
            )
            entities += 1
            text = _document_text(properties)
            if text:
                document = document_id(node.node_id)
                store.upsert_document(
                    document, properties.get("url") or node.node_id, text
                )
                # The item's own text answers for the item itself, which is
                # what the vector arm finds first.
                store.add_mention(document, node.node_id)
                documents += 1

        for edge in _stream(db, EntityEdge, EntityEdge.edge_id, org_id, batch_size):
            store.upsert_relationship(
                edge.source_id,
                edge.target_id,
                edge.relation_type or ARTIFACT_RELATION,
                confidence=edge_confidence(edge.relation_type, edge.weight),
            )
            if edge.relation_type == RELATION_MENTIONS:
                # The text that named the entity, as its evidence. A no-op
                # when the source wrote no document.
                store.add_mention(document_id(edge.source_id), edge.target_id)
            edges += 1

        if entities:
            # Last, and only with something to index. The expensive step, run
            # here so no query ever pays for it.
            store.build_vector_index(rebuild=True)
        else:
            logger.info("%s: no entities, so no index was built", org_id)
    finally:
        # Even when the loop above raised. A handle left open on a partial
        # file is a file the next attempt cannot replace.
        store.close()

    logger.info(
        "%s: compiled %d entities and %d edges to %s",
        org_id,
        entities,
        edges,
        destination,
    )
    return CompiledArtifact(
        path=destination, entities=entities, edges=edges, documents=documents
    )


def _stream(
    db: Session, model, order_by, org_id: str, batch_size: int
) -> Iterator:
    """Rows for one organisation, ``batch_size`` at a time.

    A server-side cursor rather than a fetch: the point is that the number of
    rows resident never depends on how many rows there are.

    Ordered, because a cursor over an unordered result is a cursor over
    whatever order the database felt like, and a partial read of that cannot
    be reasoned about.
    """
    statement = select(model).where(model.org_id == org_id).order_by(order_by)
    results = db.exec(statement.execution_options(yield_per=batch_size))

    for row in results:
        yield row


def _remove(path: Path) -> None:
    """Delete whatever a previous build left at ``path``, file or directory.

    The store and each of its sidecars: a build that crashed leaves its log
    beside the file, and a fresh store opened next to that log replays it and
    refuses to open. Both kinds, because a store is a file in this project and
    a directory in some configurations of the engine underneath it, and a build
    that removed only one of those would silently accumulate into the other.

    A piece that cannot be removed is logged and skipped, so the rest still go.
    """
    import shutil

    for suffix in STALE_SUFFIXES:
        piece = Path(f"{path}{suffix}")
        try:
            if piece.is_dir():
                shutil.rmtree(piece)
            elif piece.exists():
                piece.unlink()
        except OSError as exc:
            logger.warning("could not remove %s before the build: %s", piece, exc)
