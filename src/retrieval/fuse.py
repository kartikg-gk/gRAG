"""One ranked set from two arms.

::

    score_total = (alpha * vector_score + beta * graph_score) * decay

**A node found by one arm scores zero on the other, and that zero is kept.**
No renormalisation, no substituted default, no averaging over the arms that
did find it. Absence from an arm is information: a node the vector arm never
returned is a node whose text does not match the query, and replacing that
zero with a mean would say the opposite. The weights already encode how much
each arm is trusted; rescaling by how many arms fired would apply that
judgement twice.

Result-set balancing
--------------------

::

    vector_k = max(MIN_VECTOR_K, total_k - graph_hits)

More graph results means fewer vector slots, so the set size stays constant.
This is a token-cost decision as much as a ranking one — a graph hit is an id
and a score, a vector hit carries the text that made it match, and a fused set
that grew with traversal breadth would blow up the context it feeds.

The floor is why ``MIN_VECTOR_K`` exists. A query whose traversal reaches
everything would otherwise contribute no vector evidence at all, and a result
set built entirely from link structure cannot show that the answer says
anything relevant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..common.config import (
    GRAPH_WEIGHT_CONCEPTUAL,
    GRAPH_WEIGHT_RELATIONAL,
    INTENT_RELATIONAL,
    MIN_VECTOR_K,
    NODE_TABLE,
    TOTAL_K,
    VECTOR_WEIGHT_CONCEPTUAL,
    VECTOR_WEIGHT_RELATIONAL,
)
from .recency import decay_factor


@dataclass(frozen=True)
class FusedHit:
    """One result, with every component that produced its score.

    The parts are kept rather than collapsed into the total. A fused ranking
    is otherwise unreadable: a node at 0.42 gives no indication of whether an
    arm found it strongly and decay cut it, or both arms found it weakly.
    """

    id: str
    score: float
    vector_score: float
    graph_score: float
    decay: float
    node_type: str | None = None

    @property
    def found_by_both(self) -> bool:
        return self.vector_score > 0.0 and self.graph_score > 0.0


@dataclass
class FusedResult:
    """The ranked set, and how it was produced."""

    intent: str
    alpha: float
    beta: float
    hits: list[FusedHit] = field(default_factory=list)
    vector_k: int = 0
    graph_hits: int = 0

    @property
    def ids(self) -> list[str]:
        return [hit.id for hit in self.hits]

    def __len__(self) -> int:
        return len(self.hits)


def weights_for(intent: str) -> tuple[float, float]:
    """``(alpha, beta)`` — how much the vector and graph arms count.

    A query naming a thing wants the arm that follows links from it; a query
    describing a concept wants the arm that matches meaning.
    """
    if intent == INTENT_RELATIONAL:
        return VECTOR_WEIGHT_RELATIONAL, GRAPH_WEIGHT_RELATIONAL
    return VECTOR_WEIGHT_CONCEPTUAL, GRAPH_WEIGHT_CONCEPTUAL


def vector_budget(graph_hits: int, total_k: int = TOTAL_K) -> int:
    """How many vector results a set with ``graph_hits`` graph results may hold."""
    return max(MIN_VECTOR_K, total_k - graph_hits)


def attributes_for(store, node_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Type and timestamp for each id, in one query rather than one per node.

    Decay needs both and neither arm returns them — the graph arm carries
    scores and hops, the vector arm carries similarity. Looking them up here
    keeps that out of both arms, which is what lets the arms stay independent.
    """
    wanted = list(dict.fromkeys(node_ids))
    if not wanted:
        return {}

    rows = store.query(
        f"UNWIND $ids AS wanted "
        f"MATCH (e:{NODE_TABLE} {{id: wanted}}) "
        f"RETURN e.id, e.type, e.ts",
        {"ids": wanted},
    )
    return {row[0]: {"type": row[1], "timestamp": row[2]} for row in rows}


def fuse(
    vector_hits,
    graph_hits,
    intent: str,
    *,
    attributes: Mapping[str, Mapping[str, Any]] | None = None,
    total_k: int = TOTAL_K,
    now=None,
) -> FusedResult:
    """Combine the two arms into one ranking.

    Balancing is applied to the vector side before fusing, so the budget
    decides what enters the pool rather than trimming it afterwards. Trimming
    after would let a vector hit occupy a slot in the arithmetic and then be
    dropped, which changes nothing about the result but makes the reported
    ``vector_k`` a lie.

    Ties break on id so a ranking is stable across runs.
    """
    alpha, beta = weights_for(intent)
    attributes = attributes or {}

    graph_scores = {hit.id: float(hit.score) for hit in graph_hits}
    budget = vector_budget(len(graph_scores), total_k)
    admitted = list(vector_hits)[:budget]
    vector_scores = {hit.id: float(hit.similarity) for hit in admitted}

    result = FusedResult(
        intent=intent,
        alpha=alpha,
        beta=beta,
        vector_k=budget,
        graph_hits=len(graph_scores),
    )

    for node_id in list(dict.fromkeys([*vector_scores, *graph_scores])):
        vector_score = vector_scores.get(node_id, 0.0)
        graph_score = graph_scores.get(node_id, 0.0)
        attribute = attributes.get(node_id, {})
        decay = decay_factor(attribute.get("timestamp"), attribute.get("type"), now)

        result.hits.append(
            FusedHit(
                id=node_id,
                score=(alpha * vector_score + beta * graph_score) * decay,
                vector_score=vector_score,
                graph_score=graph_score,
                decay=decay,
                node_type=attribute.get("type"),
            )
        )

    result.hits.sort(key=lambda hit: (-hit.score, hit.id))
    result.hits = result.hits[:total_k]
    return result
