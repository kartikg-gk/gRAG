"""Turning a retrieval run into something a consumer outside this process can read.

Two things need this shape and neither should own it. The HTTP surface serves
it as a response body, and the retriever adapter that exposes this engine to
an external agent framework has to hand the same information back in that
framework's own container. Both need the per-arm scores, both need the source
text, and both need the trace. A formatting step living inside a route handler
would force the adapter either to import a web framework or to write the
conversion a second time, and two conversions of one shape drift.

So this module is plain data in and plain data out — dictionaries and lists,
no request, no response class, no framework type anywhere in it.

Why the scores stay apart
-------------------------

A result carries ``score``, ``vector_score``, ``graph_score`` and ``decay``
separately, and collapsing them would be the one change that makes this API
useless for the thing it exists to expose. A node at 0.42 says nothing on its
own: it could be a strong vector match that recency cut, a strong traversal
the vector arm never saw, or both arms agreeing weakly. Those are different
answers to "why am I looking at this", and the two-arm design is invisible
without them.

An arm that did not find a node reports **0.0 rather than being omitted**.
Absent and zero are the same number here but not the same fact, and a consumer
that has to distinguish "this key is missing" from "this arm scored nothing"
will get it wrong.

Why the text is stored once
---------------------------

Entities and chunks are different things. Entities are what retrieval ranked;
chunks are the prose those entities were found in, attached as payload. One
chunk commonly mentions several entities, so a response that inlined text per
result would repeat the same paragraph under every entity it names — inflating
the payload and, worse, inflating any context assembled from it, where the
same text arriving three times reads as three pieces of evidence.

So chunks are a table keyed by id, and a result references the ids it draws
on. The text appears exactly once however many results point at it.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

#: What a result reports when an arm did not contribute to it. Named rather
#: than written as a bare 0.0 at three call sites, because the decision it
#: encodes — report the zero, never drop the key — is the point.
NO_CONTRIBUTION = 0.0


def hit_payload(hit) -> dict[str, Any]:
    """One ranked entity, with every component that produced its position."""
    return {
        "id": hit.id,
        "score": float(hit.score),
        "vector_score": float(getattr(hit, "vector_score", NO_CONTRIBUTION)),
        "graph_score": float(getattr(hit, "graph_score", NO_CONTRIBUTION)),
        "decay": float(getattr(hit, "decay", 1.0)),
        "node_type": getattr(hit, "node_type", None),
        "age_days": getattr(hit, "age_days", None),
        "found_by_both": bool(getattr(hit, "found_by_both", False)),
    }


def chunk_table(
    documents: Mapping[str, Iterable[Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Split ``entity -> chunks`` into a deduplicated table and a reference map.

    Returns ``(chunks, by_entity)``. ``chunks`` holds each chunk once, keyed by
    its id; ``by_entity`` maps an entity id to the chunk ids it draws on, in
    the order they arrived.

    The store already returns every requested entity, including ones nothing
    mentions, so an entity with no prose gets an empty list rather than being
    missing from the map — the same reasoning the store applies, preserved
    rather than re-decided here.
    """
    chunks: dict[str, dict[str, Any]] = {}
    by_entity: dict[str, list[str]] = {}

    for entity_id, records in documents.items():
        references: list[str] = []
        for record in records or ():
            chunk_id = record.get("id")
            if chunk_id is None:
                continue
            if chunk_id not in chunks:
                chunks[chunk_id] = {
                    "id": chunk_id,
                    "path": record.get("path"),
                    "content": record.get("content"),
                }
            if chunk_id not in references:
                references.append(chunk_id)
        by_entity[entity_id] = references

    return chunks, by_entity


def trace_payload(trace) -> dict[str, Any] | None:
    """The per-hop record, as plain data.

    ``None`` when a run carries no trace. The trace is built from the finished
    results and never recomputes anything, so it cannot disagree with the
    ranking beside it — which is what makes it worth serving rather than
    describing.
    """
    if trace is None:
        return None
    if hasattr(trace, "to_dict"):
        return trace.to_dict()

    from dataclasses import asdict, is_dataclass

    if is_dataclass(trace):
        return asdict(trace)
    return None


def format_run(run, documents: Mapping[str, Iterable[Mapping[str, Any]]] | None = None):
    """A whole retrieval run as plain data.

    ``documents`` is what the store returned for the ranked ids. It is passed
    in rather than fetched here so this module needs no store handle and can
    be called on a run that was produced anywhere — which is what lets one
    formatter serve both consumers.
    """
    hits = list(getattr(run, "hits", ()) or ())
    chunks, by_entity = chunk_table(documents or {})

    results = []
    for hit in hits:
        payload = hit_payload(hit)
        payload["chunk_ids"] = by_entity.get(hit.id, [])
        results.append(payload)

    fused = getattr(run, "fused", None)
    return {
        "query": getattr(run, "query", ""),
        "intent": getattr(getattr(run, "intent", None), "intent", None),
        "alpha": getattr(fused, "alpha", None),
        "beta": getattr(fused, "beta", None),
        "results": results,
        "chunks": list(chunks.values()),
        "trace": trace_payload(getattr(run, "trace_log", None)),
        "seconds": float(getattr(run, "seconds", 0.0)),
    }


#: Headings for the two sections of a prompt string. Constants because the
#: assembler and its tests both need to agree on them, and a heading typed
#: twice is a heading that drifts.
ENTITY_HEADING = "Retrieved entities"
TEXT_HEADING = "Source text"


def as_prompt(hits, documents: Mapping[str, Iterable[Mapping[str, Any]]] | None = None) -> str:
    """A retrieval result as one string, ready to put in front of a model.

    Two labelled sections: what was retrieved, then the prose behind it.

    **The text is deduplicated by chunk and the entities are not.** Several
    entities routinely point at one chunk — a single pull request body naming
    three tickets is the ordinary case — so the first entity to reach a chunk
    contributes its text and every later one contributes a provenance line
    instead, naming itself and the chunk it shares. Without that, a chunk five
    entities point at is pasted five times and most of what reaches the model
    is repetition of the same paragraph.

    The provenance line sits with the entities rather than in the text
    section, because it is a fact about an entity — where its evidence is —
    and putting it among the prose would interrupt the thing being quoted.

    **An empty section is omitted, not emitted empty.** A heading with nothing
    under it tells a model there was a category and it came back blank, which
    is an invitation to comment on the absence.
    """
    chunks, by_entity = chunk_table(documents or {})

    entity_lines: list[str] = []
    text_blocks: list[str] = []
    shown: dict[str, str] = {}

    for hit in hits or ():
        node_type = getattr(hit, "node_type", None)
        label = f"- {hit.id}" + (f" ({node_type})" if node_type else "")

        fresh: list[str] = []
        repeated: list[str] = []
        for chunk_id in by_entity.get(hit.id, []):
            if chunk_id in shown:
                repeated.append(chunk_id)
            else:
                shown[chunk_id] = hit.id
                fresh.append(chunk_id)

        if repeated:
            label += "; text already shown under " + ", ".join(
                f"{chunk_id} (via {shown[chunk_id]})" for chunk_id in repeated
            )
        entity_lines.append(label)

        for chunk_id in fresh:
            chunk = chunks[chunk_id]
            content = (chunk.get("content") or "").strip()
            if not content:
                continue
            source = chunk.get("path") or chunk_id
            text_blocks.append(f"[{chunk_id}] {source}\n{content}")

    sections: list[str] = []
    if entity_lines:
        sections.append(ENTITY_HEADING + "\n" + "\n".join(entity_lines))
    if text_blocks:
        sections.append(TEXT_HEADING + "\n\n" + "\n\n".join(text_blocks))
    return "\n\n".join(sections)
