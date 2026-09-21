"""Three read-only tools an agent calls to work the graph step by step.

Retrieval hands a model one pre-walked result. These let the model do the
walking itself: look a name up, ask a question, or trace what hangs off an
entity — one bounded call at a time, so nothing ever needs to be stuffed into a
single context.

Each tool routes through the engine's own store and router rather than taking a
query language, so every call inherits the hop, degree and length limits and
cannot read outside the graph it was given. The limits live in
``common.config`` under ``TOOL_*``.

The functions take the engine explicitly. Which graph that is — the local one,
or a tenant's — is the caller's decision, made once where the engine is chosen.
"""

from __future__ import annotations

import logging
from typing import Any

from .common.config import (
    MAX_DEGREE,
    MAX_QUERY_CHARS,
    TOOL_CITATIONS_PER_NODE,
    TOOL_MAX_CANDIDATES,
    TOOL_MAX_HOPS,
    TOOL_MAX_IMPACT,
    TOOL_NEIGHBOR_K,
    TOOL_RESOLVE_MIN_SIM,
    TOOL_SEED_MAX_DEGREE,
    TOOL_SNIPPET_CHARS,
    TOP_K_VECTOR,
)

logger = logging.getLogger("graphrag.agent_tools")


def resolve_entities(engine, name: str, *, limit: int, min_score: float = 0.0) -> list[dict]:
    """A name as graph entities: exact label matches first, else nearest by meaning.

    Exact matches score 1.0 and always pass. The semantic fallback drops hits
    below ``min_score``, so a name that means nothing resolves to nothing
    rather than to the least-bad guess.
    """
    exact = engine.store.find_by_label(name)
    if exact:
        return [
            {"id": node["id"], "label": node["label"], "type": node["type"],
             "match": "exact", "score": 1.0}
            for node in exact[:limit]
        ]
    try:
        vector = engine.router.embed_query(name)
        hits = engine.store.vector_search(vector, k=limit)
    except Exception as exc:  # noqa: BLE001 - no index yet, and the like
        logger.debug("semantic lookup failed for %r (%s)", name, exc)
        return []
    return [
        {"id": hit["id"], "label": hit["label"], "type": hit["type"],
         "match": "semantic", "score": round(float(hit["similarity"]), 4)}
        for hit in hits
        if float(hit["similarity"]) >= min_score
    ]


def _snippets(documents: list[dict]) -> list[dict]:
    """At most TOOL_CITATIONS_PER_NODE documents, each cut to a snippet."""
    out: list[dict] = []
    for document in documents[:TOOL_CITATIONS_PER_NODE]:
        content = (document.get("content") or "").strip()
        if not content:
            continue
        out.append({
            "doc_id": document.get("doc_id", document.get("id")),
            "source": document.get("path"),
            "snippet": content[:TOOL_SNIPPET_CHARS],
        })
    return out


def citations_for(engine, entity_ids: list[str]) -> dict[str, list[dict]]:
    """The documents behind each entity, as short snippets an answer can cite."""
    if not entity_ids:
        return {}
    try:
        grouped = engine.store.documents_for_entities(entity_ids)
    except Exception as exc:  # noqa: BLE001
        logger.debug("citation lookup failed (%s)", exc)
        return {}
    out: dict[str, list[dict]] = {}
    for entity_id, documents in grouped.items():
        cites = _snippets(documents)
        if cites:
            out[entity_id] = cites
    return out


def trace_impact(engine, entity_name: str, max_hops: int = 3) -> dict:
    """Everything connected to one entity, strongest first, with how it was reached.

    Resolves the name to one entity, then walks outward: a path's strength is
    the product of its edges' confidences, so influence fades with distance.
    Each reached entity records its hop distance and the neighbour it came
    through, so the result explains itself rather than being a flat list.
    """
    name = (entity_name or "").strip()[:MAX_QUERY_CHARS]
    if not name:
        return {"query_entity": entity_name, "resolved": None, "impacted": [],
                "note": "Give an entity name to trace from."}
    hops_cap = max(1, min(int(max_hops), TOOL_MAX_HOPS))

    resolved = resolve_entities(engine, name, limit=1, min_score=TOOL_RESOLVE_MIN_SIM)
    if not resolved:
        return {"query_entity": name, "resolved": None, "impacted": [],
                "note": "Nothing in this graph matched that name closely enough. "
                        "Use find_entity to see the candidates."}
    seed = resolved[0]
    seed_id = seed["id"]

    labels: dict[str, str] = {seed_id: seed["label"]}
    best: dict[str, dict] = {}
    frontier: dict[str, float] = {seed_id: 1.0}
    visited: set[str] = {seed_id}
    for depth in range(1, hops_cap + 1):
        if not frontier:
            break
        # The first hop expands the entity asked about, which is often a hub
        # itself, so it gets the generous threshold. Later hops keep the normal
        # one, so a hub further out cannot blow the walk up.
        degree_cap = TOOL_SEED_MAX_DEGREE if depth == 1 else MAX_DEGREE
        expanded = engine.store.expand_frontier(list(frontier), TOOL_NEIGHBOR_K, degree_cap)
        next_frontier: dict[str, float] = {}
        for from_id, strength in frontier.items():
            for neighbour in expanded.get(from_id, []):
                to_id = neighbour["id"]
                labels.setdefault(to_id, neighbour["label"])
                score = strength * neighbour["confidence"]
                current = best.get(to_id)
                if current is None or score > current["confidence"]:
                    best[to_id] = {
                        "id": to_id, "label": neighbour["label"], "type": neighbour["type"],
                        "confidence": round(score, 4), "hops": depth,
                        "via": labels.get(from_id, from_id),
                    }
                if to_id not in visited:
                    visited.add(to_id)
                    if score > next_frontier.get(to_id, 0.0):
                        next_frontier[to_id] = score
        frontier = next_frontier

    ranked = sorted(best.values(), key=lambda row: (-row["confidence"], row["hops"], row["id"]))
    ranked = ranked[:TOOL_MAX_IMPACT]
    cites = citations_for(engine, [row["id"] for row in ranked])
    for row in ranked:
        row["citations"] = cites.get(row["id"], [])

    return {
        "query_entity": name,
        "resolved": seed,
        "hops_traversed": hops_cap,
        "blast_radius_count": len(ranked),
        "impacted": ranked,
    }


def search_context(engine, query: str, top_k: int = 8) -> dict:
    """A question answered with ranked passages, each carrying its sources."""
    text = (query or "").strip()
    if not text:
        return {"query": query, "passages": [], "note": "Give a question to search for."}
    k = max(1, min(int(top_k), TOP_K_VECTOR * 2))
    response = engine.router.route(text, top_k=k)

    passages: list[dict[str, Any]] = []
    for node in response.results:
        passages.append({
            "entity": node.label or node.id,
            "type": node.type,
            "relevance": round(node.score_total, 4),
            "citations": _snippets(node.documents),
        })

    intent = (response.trace_log.get("intent") or {}).get("type")
    return {"query": response.query, "intent": intent,
            "result_count": len(passages), "passages": passages}


def find_entity(engine, name: str, limit: int = 5) -> dict:
    """The entities a name could mean, exact matches first. Walks nothing."""
    text = (name or "").strip()
    if not text:
        return {"query": name, "candidates": [], "note": "Give a name to look up."}
    cap = max(1, min(int(limit), TOOL_MAX_CANDIDATES))
    candidates = resolve_entities(engine, text, limit=cap)
    return {"query": text, "match_count": len(candidates), "candidates": candidates}
