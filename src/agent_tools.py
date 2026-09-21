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


def _candidate(node: dict, how: str, score: float) -> dict:
    return {"id": node["id"], "label": node["label"], "type": node["type"],
            "match": how, "score": score}


def _nearest(engine, name: str, limit: int) -> list[tuple[float, dict]]:
    """Entities closest in meaning to ``name``, each with its raw similarity."""
    try:
        hits = engine.store.vector_search(engine.router.embed_query(name), k=limit)
    except Exception as exc:  # noqa: BLE001 - an empty store has no index to search
        logger.debug("no semantic candidates for %r: %s", name, exc)
        return []
    ranked = []
    for hit in hits:
        similarity = float(hit["similarity"])
        ranked.append((similarity, _candidate(hit, "semantic", round(similarity, 4))))
    return ranked


def resolve_entities(engine, name: str, *, limit: int, min_score: float = 0.0) -> list[dict]:
    """The entities ``name`` most plausibly refers to.

    A label that matches exactly settles it, at full score, however many
    entities carry it. Only when nothing matches exactly does meaning come in,
    and then a floor applies: a closeness below ``min_score`` is treated as no
    match at all, so a meaningless name yields an empty list.
    """
    exact = engine.store.find_by_label(name)
    if exact:
        return [_candidate(node, "exact", 1.0) for node in exact[:limit]]
    return [found for similarity, found in _nearest(engine, name, limit) if similarity >= min_score]


def _snippets(documents: list[dict]) -> list[dict]:
    """The first TOOL_CITATIONS_PER_NODE documents that have text, as snippets."""
    considered = documents[:TOOL_CITATIONS_PER_NODE]
    texts = ((document, (document.get("content") or "").strip()) for document in considered)
    return [
        {"doc_id": document.get("doc_id", document.get("id")),
         "source": document.get("path"),
         "snippet": text[:TOOL_SNIPPET_CHARS]}
        for document, text in texts
        if text
    ]


def citations_for(engine, entity_ids: list[str]) -> dict[str, list[dict]]:
    """For each entity, citable snippets of the documents that mention it.

    Entities with nothing citable are left out rather than mapped to an empty
    list, and a store that cannot answer yields no citations at all.
    """
    if not entity_ids:
        return {}
    try:
        by_entity = engine.store.documents_for_entities(entity_ids)
    except Exception as exc:  # noqa: BLE001 - citations are optional extras
        logger.debug("no citations available: %s", exc)
        return {}
    snippets = {entity_id: _snippets(documents) for entity_id, documents in by_entity.items()}
    return {entity_id: found for entity_id, found in snippets.items() if found}


def _sightings(engine, seed_id: str, hops: int):
    """Every neighbour the walk from ``seed_id`` meets, in the order it meets them.

    Yields ``(depth, origin, neighbour, strength)``. Strength multiplies along
    the path, so it only ever falls. An entity is carried forward only from the
    first time it is met, and only while its strength is above zero. The first
    ring gets the generous degree cap because the seed is usually the busiest
    entity in the question; later rings keep the usual one.
    """
    ring = {seed_id: 1.0}
    met = {seed_id}
    for depth in range(1, hops + 1):
        if not ring:
            return
        cap = TOOL_SEED_MAX_DEGREE if depth == 1 else MAX_DEGREE
        adjacency = engine.store.expand_frontier(list(ring), TOOL_NEIGHBOR_K, cap)
        next_ring: dict[str, float] = {}
        for origin, carried in ring.items():
            for neighbour in adjacency.get(origin, []):
                strength = carried * neighbour["confidence"]
                yield depth, origin, neighbour, strength
                if neighbour["id"] in met:
                    continue
                met.add(neighbour["id"])
                if strength > 0.0:
                    next_ring[neighbour["id"]] = strength
        ring = next_ring


def _strongest_reach(sightings, seed: dict) -> list[dict]:
    """Per entity, the sighting that reached it most strongly, as result rows.

    Recorded strengths are rounded to four places, and a later sighting wins
    only when it beats the rounded figure.
    """
    names = {seed["id"]: seed["label"]}
    # entity id -> (rounded strength, depth, name of the neighbour it came through, neighbour)
    strongest: dict[str, tuple[float, int, str, dict]] = {}
    for depth, origin, neighbour, strength in sightings:
        names.setdefault(neighbour["id"], neighbour["label"])
        previous = strongest.get(neighbour["id"])
        if previous is None or strength > previous[0]:
            strongest[neighbour["id"]] = (round(strength, 4), depth, names.get(origin, origin), neighbour)
    return [
        {"id": entity_id, "label": found["label"], "type": found["type"],
         "confidence": score, "hops": depth, "via": via}
        for entity_id, (score, depth, via, found) in strongest.items()
    ]


def trace_impact(engine, entity_name: str, max_hops: int = 3) -> dict:
    """What depends on, or hangs off, one named entity — strongest first.

    Each reached entity says how far away it is and which neighbour led to it,
    and carries the snippets that support it, so the answer can be checked.
    """
    name = (entity_name or "").strip()[:MAX_QUERY_CHARS]
    if not name:
        return {"query_entity": entity_name, "resolved": None, "impacted": [],
                "note": "Give an entity name to trace from."}
    hops = min(max(int(max_hops), 1), TOOL_MAX_HOPS)

    matches = resolve_entities(engine, name, limit=1, min_score=TOOL_RESOLVE_MIN_SIM)
    if not matches:
        return {"query_entity": name, "resolved": None, "impacted": [],
                "note": "Nothing in this graph matched that name closely enough. "
                        "Use find_entity to see the candidates."}
    seed = matches[0]

    reach = _strongest_reach(_sightings(engine, seed["id"], hops), seed)
    order = sorted(reach, key=lambda row: (-row["confidence"], row["hops"], row["id"]))
    impacted = order[:TOOL_MAX_IMPACT]
    support = citations_for(engine, [row["id"] for row in impacted])
    for row in impacted:
        row["citations"] = support.get(row["id"], [])
    return {
        "query_entity": name,
        "resolved": seed,
        "hops_traversed": hops,
        "blast_radius_count": len(impacted),
        "impacted": impacted,
    }


def _passage(node) -> dict[str, Any]:
    return {"entity": node.label or node.id, "type": node.type,
            "relevance": round(node.score_total, 4), "citations": _snippets(node.documents)}


def search_context(engine, query: str, top_k: int = 8) -> dict:
    """Ranked passages for a question, each with the snippets behind it."""
    text = (query or "").strip()
    if not text:
        return {"query": query, "passages": [], "note": "Give a question to search for."}
    response = engine.router.route(text, top_k=min(max(int(top_k), 1), TOP_K_VECTOR * 2))
    passages = [_passage(node) for node in response.results]
    return {"query": response.query,
            "intent": (response.trace_log.get("intent") or {}).get("type"),
            "result_count": len(passages), "passages": passages}


def find_entity(engine, name: str, limit: int = 5) -> dict:
    """The entities a name could mean, exact matches first. Walks nothing."""
    text = (name or "").strip()
    if not text:
        return {"query": name, "candidates": [], "note": "Give a name to look up."}
    candidates = resolve_entities(engine, text, limit=min(max(int(limit), 1), TOOL_MAX_CANDIDATES))
    return {"query": text, "match_count": len(candidates), "candidates": candidates}
