"""One query in, ranked results and a record of how they got there out.

This owns the sequence — classify, seed, search, traverse, fuse — and nothing
else does. Before it existed every stage was independently runnable and
independently tested, and nothing ran them together outside a test, so there
was no point at which a complete record of a query could be assembled.

Order is not free choice
------------------------

Classification runs first and costs nothing that depends on the store, so a
malformed query is cheap to reject. The vector arm runs before seed selection
because tier 2 reads its output — selecting seeds first would mean encoding the
query twice, and encoding is the most expensive thing here by three orders of
magnitude. Traversal runs last of the three because it needs seeds. Fusion
needs everything.

The arms stay independent
-------------------------

Running them in sequence is not the same as coupling them. The graph arm still
receives only seed ids, and the vector arm still receives only a query and an
embedder. Neither is handed the other's scores, and neither knows a fusion step
exists — which is what keeps the case where one arm finds something the other
misses visible in the separate results this returns alongside the fused one.

Every intermediate result is returned, not just the fused ranking. A caller
comparing arms needs them; the trace log is built from them; and a caller that
only wants the ranking can ignore the rest at no cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..common.config import TOP_K_GRAPH, TOP_K_VECTOR, TOTAL_K
from . import trace_log as trace_log_module
from .fuse import FusedResult, attributes_for, fuse
from .graph_arm import GraphResult, traverse
from .intent import Intent, classify
from .seeds import SeedResult, select
from .vector_arm import VectorResult, search


@dataclass
class RetrievalRun:
    """Everything one query produced, at every stage.

    The fused ranking is the answer; the rest is what it was made from. Both
    are returned because a run that only handed back the ranking would make
    the arms' separate behaviour unobservable from outside, which is the thing
    keeping them separate was for.
    """

    query: str
    intent: Intent
    vector: VectorResult
    seeds: SeedResult
    graph: GraphResult
    fused: FusedResult
    trace_log: Any = None
    seconds: float = 0.0

    @property
    def hits(self):
        return self.fused.hits

    @property
    def ids(self) -> list[str]:
        return self.fused.ids

    def __len__(self) -> int:
        return len(self.fused)


def retrieve(
    store,
    embedder,
    query: str,
    *,
    extractor=None,
    judge=None,
    now=None,
    k_vector: int = TOP_K_VECTOR,
    k_graph: int = TOP_K_GRAPH,
    total_k: int = TOTAL_K,
) -> RetrievalRun:
    """Run the full retrieval sequence for ``query``.

    ``extractor`` and ``judge`` are both optional and both degrade rather than
    fail. Without an extractor, seed tier 1 cannot run and selection falls to
    the vector tiers; without a judge, a query matching no marker takes the
    conceptual fallback. Neither absence stops a query returning results, and
    the trace log records which path was taken so a thin result is explicable
    rather than mysterious.

    ``now`` is threaded through to decay so a caller can pin it. Left unset it
    is the wall clock, which is right in production and wrong in a test.
    """
    started = time.perf_counter()

    intent = classify(query, judge)
    vector = search(store, embedder, query, k=k_vector)
    seeds = select(store, extractor, vector.hits, query)
    graph = traverse(store, seeds.seeds, k=k_graph)

    # Type and timestamp for everything either arm returned, in one query.
    # Fusion needs them for decay and neither arm carries them.
    attributes = attributes_for(
        store, [hit.id for hit in vector.hits] + [hit.id for hit in graph.hits]
    )
    fused = fuse(
        vector.hits,
        graph.hits,
        intent.intent,
        attributes=attributes,
        total_k=total_k,
        now=now,
    )

    run = RetrievalRun(
        query=query,
        intent=intent,
        vector=vector,
        seeds=seeds,
        graph=graph,
        fused=fused,
        seconds=time.perf_counter() - started,
    )
    # Built last, from the finished results. It reads and never recomputes, so
    # it cannot disagree with what actually ranked.
    run.trace_log = trace_log_module.build(query, intent, seeds, vector, graph, fused)
    return run
