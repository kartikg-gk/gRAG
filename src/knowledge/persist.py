"""Write a built graph into the store.

Consumes ``GraphBuilder.nodes`` and ``GraphBuilder.edges``; produces rows in a
``ContextGraph``. This is the only place the two meet. The builder stays a pure
construction step that knows nothing about a database, and the store stays free
of any knowledge about GitHub — the mapping between them lives here, where it
can be read in one file.

Writing is additive and does not replace the JSON path. Both consume the same
builder output, so a graph can be written to a file, to a store, or to both in
one run, and the two are comparable row for row.

What a document is
------------------

The builder keeps a node's identifying text — a pull request's title, a
commit's message, a file's path — and does not keep the longer bodies those
payloads carried. So a document here is a node's own text, not the full
original prose. That is worth knowing when reading mention edges: a pull
request contributes its title, so a ticket reference living only in the body is
not extracted and produces no mention. Extending this means the builder
retaining bodies, which is a change to what the JSON path emits and therefore
its own decision.

Entities found in that text are written as nodes so a mention has something to
point at. Ticket and pull request references are mapped onto the same id scheme
the builder uses, so ``#12`` in a commit message produces a mention edge
pointing at the ticket node that already exists rather than at a second node
describing the same ticket. Entity types with no structural counterpart —
services, organisations, products — get an id built from their type and label.

Embeddings
----------

Each entity is written with a vector when an embedder is supplied. The vector
index is **not** built here. Building an index costs a pass over every row, so
doing it once after all writes have finished is one pass rather than one per
batch, and callers that write in several stages pay for it once. ``persist``
writes; ``ContextGraph.build_vector_index`` indexes; keeping them separate is
what lets a caller time them separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from ..common.config import (
    ENTITY_PR,
    ENTITY_TICKET,
    NODE_COMMIT,
    NODE_FILE,
    NODE_PERSON,
    NODE_PR,
    NODE_REPO,
    NODE_TICKET,
)

#: Fields carrying a node's identifying text, most specific first. A node with
#: none of them falls back to its id, so every node is still searchable rather
#: than silently textless.
TEXT_FIELDS = ("title", "message", "full_name", "path", "login")

#: Where a node's text came from, for the document's ``path`` column. The url
#: is preferred because it locates the thing outside this database.
PATH_FIELDS = ("url", "path", "full_name")

#: Node types the builder produces, keyed to the id prefix it uses for them.
#: Extracted references are mapped through this so they land on the structural
#: node rather than beside it.
STRUCTURAL_PREFIX = {
    NODE_TICKET: "ticket",
    NODE_PR: "pr",
    NODE_PERSON: "person",
    NODE_COMMIT: "commit",
    NODE_FILE: "file",
    NODE_REPO: "repo",
}

_NUMBER = re.compile(r"\d+")


class Embedder(Protocol):
    """Anything that turns text into a fixed-width vector."""

    def vector(self, text: str) -> list[float]:
        ...


class Extractor(Protocol):
    """Anything that finds entities in text."""

    def extract(self, text: str) -> Iterable[Any]:
        ...


@dataclass
class PersistStats:
    """What reached the store, counted as it was written.

    Counted rather than derived from a query afterwards: a count taken from the
    store would agree with the store by construction and could not show a write
    that was attempted and silently did nothing.
    """

    entities: int = 0
    relationships: int = 0
    documents: int = 0
    mentions: int = 0
    embedded: int = 0
    entities_from_text: int = 0
    documents_without_text: int = 0
    per_relation: dict[str, int] = field(default_factory=dict)


def node_text(node: dict[str, Any]) -> str:
    """The identifying text of a node, or its id when it carries none."""
    for field_name in TEXT_FIELDS:
        value = node.get(field_name)
        if value:
            return str(value)
    return str(node["id"])


def node_path(node: dict[str, Any]) -> str:
    """Where the node came from, for a document row's ``path``."""
    for field_name in PATH_FIELDS:
        value = node.get(field_name)
        if value:
            return str(value)
    return str(node["id"])


def document_id(node_id: str) -> str:
    """One document per node, named so the two never collide."""
    return f"doc:{node_id}"


def entity_id(entity) -> str:
    """The store id for an extracted entity.

    Ticket and pull request references carry a number, and that number is the
    identity — ``#12`` and ``issue 12`` are the same ticket written twice. Both
    map onto the prefix the builder already uses, so a reference to something
    that was ingested structurally points at that node instead of creating a
    parallel one.

    Everything else is keyed on its lowercased surface form. Case is not
    identity for a service name: ``ORDER_SERVICE`` in an environment variable
    and ``order_service`` in prose are one thing.
    """
    if entity.type in (ENTITY_TICKET, ENTITY_PR):
        number = _NUMBER.search(entity.text)
        if number is not None:
            return f"{STRUCTURAL_PREFIX[entity.type]}:{number.group(0)}"
    return f"{entity.type.lower()}:{entity.text.lower()}"


def persist(
    store,
    builder,
    *,
    embedder: Embedder | None = None,
    extractor: Extractor | None = None,
) -> PersistStats:
    """Write every node, edge, document and mention into ``store``.

    Order matters and is not incidental. Nodes are written before edges,
    because an edge whose endpoints do not exist yet matches nothing and is
    dropped without error. Documents come last, because a mention needs both
    the document and the entity it points at already present.

    Idempotent by construction: every write goes through ``MERGE``, so running
    this twice over the same build leaves the same row counts. Asserted by
    ``test_re_ingesting_the_same_source_does_not_multiply_edges``.
    """
    stats = PersistStats()

    for node in builder.nodes.values():
        text = node_text(node)
        vector = embedder.vector(text) if embedder is not None else None
        store.upsert_entity(
            node["id"],
            text,
            node.get("type", "Unknown"),
            timestamp=node.get("timestamp"),
            embedding=vector,
        )
        stats.entities += 1
        if vector is not None:
            stats.embedded += 1

    for edge in builder.edges:
        store.upsert_relationship(
            edge["source"],
            edge["target"],
            edge["type"],
            confidence=edge.get("confidence"),
            timestamp=edge.get("timestamp"),
        )
        stats.relationships += 1
        stats.per_relation[edge["type"]] = stats.per_relation.get(edge["type"], 0) + 1

    if extractor is not None:
        _write_documents(store, builder, extractor, embedder, stats)

    return stats


def _write_documents(store, builder, extractor, embedder, stats: PersistStats) -> None:
    """A document per node, and a mention per entity found in its text."""
    for node in builder.nodes.values():
        text = node_text(node)
        doc_id = document_id(node["id"])
        store.upsert_document(doc_id, node_path(node), text)
        stats.documents += 1

        found = list(extractor.extract(text))
        if not found:
            stats.documents_without_text += 1
            continue

        for entity in found:
            target = entity_id(entity)
            if target not in builder.nodes:
                # An entity the structural pass never produced — a service
                # name, or a reference to something outside what was ingested.
                # It still needs a node for the mention to point at.
                store.upsert_entity(
                    target,
                    entity.text,
                    entity.type,
                    timestamp=None,
                    embedding=(
                        embedder.vector(entity.text) if embedder is not None else None
                    ),
                )
                stats.entities_from_text += 1
                if embedder is not None:
                    stats.embedded += 1

            store.add_mention(doc_id, target)
            stats.mentions += 1
