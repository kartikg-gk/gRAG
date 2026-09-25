"""Intent-routed hybrid retrieval with a stable response contract."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock, RLock

from ..common.config import (
    EMBED_WORKERS,
    GRAPH_NEIGHBOR_K,
    GRAPH_WEIGHT_CONCEPTUAL,
    GRAPH_WEIGHT_RELATIONAL,
    INTENT_RELATIONAL,
    MAX_DEGREE,
    MAX_HOPS,
    MAX_QUERY_CHARS,
    QUERY_EXTRACT_CACHE,
    QUERY_THRESHOLD,
    RECENCY_ENABLED,
    RECENCY_FLOOR,
    RELATION_CO_OCCURS,
    SEED_MIN_SIM,
    SEED_TOP_N,
    TOP_K_VECTOR,
    VECTOR_WEIGHT_CONCEPTUAL,
    VECTOR_WEIGHT_RELATIONAL,
)
from .intent import classify
from .recency import age_and_decay
from .response import RouterResponse, RoutedNode, build_context, documents_payload

logger = logging.getLogger(__name__)

#: The words of a query that may name a node: runs of letters, digits and the
#: punctuation usernames and package names carry.
_QUERY_WORD = re.compile(r"[\w][\w.-]*[\w]")

#: Shortest word looked up as a name. Shorter ones are articles and
#: prepositions, and an exact match on one is a coincidence.
NAMED_WORD_MIN_CHARS = 3

_EMBED_EXECUTOR: ThreadPoolExecutor | None = None
_EMBED_EXECUTOR_LOCK = Lock()
_QUERY_EXTRACT_STATE_LOCK = Lock()


def _extract_cache_state(extractor) -> tuple[OrderedDict[str, list], RLock]:
    """Return the LRU state owned by this shared extractor instance."""
    with _QUERY_EXTRACT_STATE_LOCK:
        cache = getattr(extractor, "_graphrag_query_extract_cache", None)
        lock = getattr(extractor, "_graphrag_query_extract_lock", None)
        if cache is None or lock is None:
            cache = OrderedDict()
            lock = RLock()
            setattr(extractor, "_graphrag_query_extract_cache", cache)
            setattr(extractor, "_graphrag_query_extract_lock", lock)
        return cache, lock


def _executor() -> ThreadPoolExecutor:
    global _EMBED_EXECUTOR
    with _EMBED_EXECUTOR_LOCK:
        if _EMBED_EXECUTOR is None:
            _EMBED_EXECUTOR = ThreadPoolExecutor(
                max_workers=EMBED_WORKERS,
                thread_name_prefix="embed",
            )
        return _EMBED_EXECUTOR


def shutdown_embed_executor(wait: bool = True) -> None:
    global _EMBED_EXECUTOR
    with _EMBED_EXECUTOR_LOCK:
        executor, _EMBED_EXECUTOR = _EMBED_EXECUTOR, None
    if executor is not None:
        executor.shutdown(wait=wait)


class RetrievalRouter:
    """Vector recall plus a bounded graph stream, fused by query intent."""

    def __init__(self, store, embedder, *, extractor=None, judge=None, now=None):
        self.store = store
        self.embedder = embedder
        self.extractor = extractor
        self.judge = judge
        self.now = now
        if extractor is None:
            self._extract_cache = None
            self._extract_lock = None
        else:
            self._extract_cache, self._extract_lock = _extract_cache_state(extractor)

    def _entities(self, query: str) -> list:
        if self.extractor is None:
            return []
        if not QUERY_EXTRACT_CACHE:
            try:
                return list(self.extractor.extract(query))
            except Exception as exc:
                logger.debug("query entity extraction failed: %s", exc)
                return []
        with self._extract_lock:
            if query in self._extract_cache:
                self._extract_cache.move_to_end(query)
                return self._extract_cache[query]
        try:
            entities = list(self.extractor.extract(query))
        except Exception as exc:
            logger.debug("query entity extraction failed: %s", exc)
            return []
        with self._extract_lock:
            self._extract_cache[query] = entities
            self._extract_cache.move_to_end(query)
            while len(self._extract_cache) > QUERY_EXTRACT_CACHE:
                self._extract_cache.popitem(last=False)
        return entities

    def _embed(self, text: str) -> list[float]:
        return _executor().submit(self.embedder.vector, text).result()

    def embed_query(self, text: str) -> list[float]:
        """A query's vector, through the same bounded pool retrieval uses."""
        return self._embed(text)

    def _vector_hits(self, query: str, meta: dict[str, dict]) -> list[tuple[str, float]]:
        vector = self._embed(query)
        try:
            rows = self.store.vector_search(vector, k=TOP_K_VECTOR)
        except Exception as exc:
            logger.debug("vector search returned no rows: %s", exc)
            return []
        hits = []
        for row in rows:
            node_id = row.get("id")
            if node_id is None:
                continue
            meta.setdefault(node_id, {
                "label": row.get("label"),
                "type": row.get("type"),
                "timestamp": row.get("timestamp", row.get("ts")),
            })
            hits.append((node_id, float(row["similarity"])))
        return hits

    def _named_nodes(self, query: str) -> list[dict]:
        """Nodes whose label is a word of the query, exactly (ignoring case).

        A username is a name the extractor does not recognise, and the text
        model cannot tell one from another: embedded, "kevinjosethomas" sits
        near "robertkeus" because their letters look alike. Looking each word
        up as a label finds the one the query actually names.
        """
        rows: list[dict] = []
        for word in dict.fromkeys(_QUERY_WORD.findall(query)):
            if len(word) >= NAMED_WORD_MIN_CHARS and not word.isdigit():
                rows.extend(self.store.find_by_label(word))
        return rows

    def _linked_seeds(
        self, query: str, meta: dict[str, dict], named: list[dict] | None = None
    ) -> list[str]:
        linked: list[str] = []
        for row in self._named_nodes(query) if named is None else named:
            node_id = row.get("id")
            if node_id is None:
                continue
            meta.setdefault(node_id, {
                "label": row.get("label"),
                "type": row.get("type"),
                "timestamp": row.get("timestamp", row.get("ts")),
            })
            linked.append(node_id)
        for entity in self._entities(query):
            rows = list(self.store.find_by_label(entity.text))
            if not rows:
                try:
                    rows = self.store.vector_search(self._embed(entity.text), k=1)
                except Exception:
                    rows = []
                rows = [row for row in rows if float(row.get("similarity", 0.0)) >= QUERY_THRESHOLD]
            for row in rows:
                node_id = row.get("id")
                if node_id is None:
                    continue
                meta.setdefault(node_id, {
                    "label": row.get("label"),
                    "type": row.get("type"),
                    "timestamp": row.get("timestamp", row.get("ts")),
                })
                linked.append(node_id)
        return list(dict.fromkeys(linked))

    def _graph_stream(self, seeds: list[str], meta: dict[str, dict]):
        scores = {seed: 1.0 for seed in seeds}
        frontier = {seed: 1.0 for seed in seeds}
        visited = set(seeds)
        hops: list[dict] = []
        for _ in range(MAX_HOPS):
            if not frontier:
                break
            expanded = self.store.expand_frontier(
                list(frontier), GRAPH_NEIGHBOR_K, MAX_DEGREE
            )
            next_frontier: dict[str, float] = {}
            for source, accumulator in frontier.items():
                for neighbor in expanded.get(source, []):
                    target = neighbor["id"]
                    confidence = float(neighbor["confidence"])
                    relation = neighbor.get("relation") or RELATION_CO_OCCURS
                    path_score = accumulator * confidence
                    hops.append({
                        "from_id": source,
                        "to_id": target,
                        "confidence": confidence,
                        "relation": relation,
                    })
                    meta.setdefault(target, {
                        "label": neighbor.get("label"),
                        "type": neighbor.get("type"),
                        "timestamp": neighbor.get("timestamp", neighbor.get("ts")),
                    })
                    scores[target] = max(scores.get(target, 0.0), path_score)
                    if target not in visited:
                        visited.add(target)
                        next_frontier[target] = max(next_frontier.get(target, 0.0), path_score)
            frontier = next_frontier
        return scores, hops

    def route(self, query: str, top_k: int | None = None) -> RouterResponse:
        query = (query or "")[:MAX_QUERY_CHARS]
        named = self._named_nodes(query)
        decision = classify(query, self.judge, named=bool(named))
        relational = decision.intent == INTENT_RELATIONAL
        alpha = VECTOR_WEIGHT_RELATIONAL if relational else VECTOR_WEIGHT_CONCEPTUAL
        beta = GRAPH_WEIGHT_RELATIONAL if relational else GRAPH_WEIGHT_CONCEPTUAL
        intent_type = "relational" if relational else "semantic"
        total_k = top_k if top_k is not None else TOP_K_VECTOR
        meta: dict[str, dict] = {}

        pool = self._vector_hits(query, meta)
        linked_seeds = self._linked_seeds(query, meta, named)
        # A query that names something starts from what it names. Similar
        # vectors are a way to find a start when nothing is named, not extra
        # starts beside a named one: added anyway, a lookalike username
        # brings its owner's work into an answer about someone else.
        fuzzy_seeds = [] if linked_seeds else [
            node_id for node_id, similarity in sorted(pool, key=lambda item: -item[1])
            if similarity >= SEED_MIN_SIM
        ][:SEED_TOP_N]
        seeds = list(dict.fromkeys(linked_seeds + fuzzy_seeds))
        if not seeds and pool:
            seeds = [node_id for node_id, _ in pool[:SEED_TOP_N]]

        graph_scores, hops = self._graph_stream(seeds, meta)
        graph_hits = sum(node_id not in set(seeds) for node_id in graph_scores)
        vector_k = max(2, total_k - graph_hits)
        vector_scores = dict(pool[:vector_k])

        results: list[RoutedNode] = []
        for node_id in set(vector_scores) | set(graph_scores):
            info = meta.get(node_id, {})
            age, decay = age_and_decay(
                info.get("timestamp"), info.get("type"), self.now
            )
            vector_score = vector_scores.get(node_id, 0.0)
            graph_score = graph_scores.get(node_id, 0.0)
            results.append(RoutedNode(
                id=node_id,
                label=info.get("label"),
                type=info.get("type"),
                score_total=(alpha * vector_score + beta * graph_score) * decay,
                score_vector=vector_score,
                score_graph=graph_score,
                recency=round(decay, 4),
                age_days=round(age, 1) if age is not None else None,
            ))
        results.sort(key=lambda row: row.score_total, reverse=True)
        results = results[:total_k]
        documents = self.store.documents_for_entities([row.id for row in results])
        for row in results:
            row.documents = documents_payload(documents.get(row.id, []))

        trace_log = {
            "intent": {"alpha": alpha, "beta": beta, "type": intent_type},
            "execution_path": {
                "linked_seeds": linked_seeds,
                "vector_seeds": seeds,
                "graph_hops": hops,
            },
            "recency": {
                "enabled": RECENCY_ENABLED,
                "floor": RECENCY_FLOOR,
                "applied": [
                    {"id": row.id, "age_days": row.age_days, "factor": row.recency}
                    for row in results if row.age_days is not None
                ],
            },
            "metrics": {
                "graph_hits": graph_hits,
                "vector_k": vector_k,
                "total_nodes_evaluated": len(vector_scores) + len(graph_scores),
            },
        }
        return RouterResponse(query=query, results=results, trace_log=trace_log)

    async def aroute(self, query: str, top_k: int | None = None) -> RouterResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.route, query, top_k))

    def build_context(self, results: list[RoutedNode]) -> str:
        return build_context(results)

    def warm(self) -> None:
        self._embed("warmup query")
        self._entities("warmup query")
        # The first search, walk and document read each pay a one-off cost of
        # their own: measured, the first trace after a start took 1.7s and the
        # next 70ms. Run each once here, without the judge, which is a network
        # call and warms nothing in this process.
        meta: dict[str, dict] = {}
        pool = self._vector_hits("warmup query", meta)
        seeds = [node_id for node_id, _ in pool[:SEED_TOP_N]]
        self._graph_stream(seeds, meta)
        self.store.documents_for_entities(seeds)
