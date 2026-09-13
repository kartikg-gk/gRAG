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

Documents are supplied by the caller, built from the payloads by
``knowledge.documents``. They are not derived from nodes: a node keeps a
thing's identity and not the prose it arrived with, so deriving documents from
nodes finds titles where the references live in bodies. Measured on the
verification fixtures, that produced 0 entities against the 2 present.

Entities found in document content are written as nodes so a mention has
something to point at. Ticket and pull request references are mapped onto the
same id scheme the builder uses, so ``#12`` in a commit message produces a
mention edge pointing at the ticket node that already exists rather than at a
second node describing the same ticket. Entity types with no structural
counterpart — services, organisations, products — get an id built from their
type and label.

Extraction runs over the text held in memory, before it is written. Nothing
here reads a stored document back, so writing documents does not make this a
reader of the ``content`` column — a distinction that matters, because whether
that column has a reader is still an open question and this is not the thing
that answers it.

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

    #: Every entity written, by either pass. ``entities_from_text`` is the
    #: share of this total that the mention pass created rather than a separate
    #: tally to add on.
    entities: int = 0
    relationships: int = 0
    documents: int = 0
    mentions: int = 0
    embedded: int = 0
    entities_from_text: int = 0
    documents_without_entities: int = 0
    per_relation: dict[str, int] = field(default_factory=dict)
    #: Mentions pointing at a node the structural pass already created,
    #: against mentions that had to create one. The split says whether
    #: extraction is finding things the graph already knows about or adding
    #: to it, and those are different kinds of value.
    mentions_to_existing: int = 0
    mentions_to_new: int = 0
    #: Entities found, by the payload field their document came from. A total
    #: alone cannot say whether review bodies were worth reading.
    entities_by_field: dict[str, int] = field(default_factory=dict)


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
    documents: Iterable[Any] = (),
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

    _write_documents(store, builder, documents, extractor, embedder, stats)

    return stats


def _write_documents(
    store, builder, documents, extractor, embedder, stats: PersistStats
) -> None:
    """Store each document, and a mention per entity found in its content.

    Documents are written whether or not an extractor is supplied. The text is
    the source record; mentions are an interpretation of it, and a caller that
    wants the prose stored without paying for extraction should get that.
    """
    known = set(builder.nodes)

    for document in documents:
        store.upsert_document(document.id, document.path, document.content)
        stats.documents += 1

        if extractor is None:
            continue

        found = list(extractor.extract(document.content))
        if not found:
            stats.documents_without_entities += 1
            continue

        for entity in found:
            target = entity_id(entity)

            if target in known:
                stats.mentions_to_existing += 1
            else:
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
                known.add(target)
                # Counted in the total as well as the breakdown. This entity
                # reached the store; a total that omitted it would report fewer
                # entities than were written, and the breakdown beside it would
                # name the ones missing from it.
                stats.entities += 1
                stats.entities_from_text += 1
                stats.mentions_to_new += 1
                if embedder is not None:
                    stats.embedded += 1

            store.add_mention(document.id, target)
            stats.mentions += 1
            stats.entities_by_field[document.field] = (
                stats.entities_by_field.get(document.field, 0) + 1
            )
