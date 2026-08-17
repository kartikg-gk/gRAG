"""Retrieval by traversal, scored by the confidence of the path taken.

Breadth-first from a set of seeds. Each hop multiplies the running score by the
confidence of the relation crossed::

    path_score = accumulator * confidence

Multiplying rather than subtracting a fixed decay is what makes relation
confidence the thing that shapes traversal. A path of authorship edges at 0.95
is worth 0.90 after two hops and 0.86 after three; a path of proximity edges at
0.35 is worth 0.12 after two and 0.04 after three. Strong evidence survives
depth that weak evidence does not, without a separate depth penalty deciding
that on its own.

``MAX_HOPS`` bounds the walk. It is a rail rather than the mechanism: the decay
above already stops weak paths, and the bound stops a strong path from walking
the entire graph.

Why hop records exist
---------------------

Traversal cannot be reconstructed from a final score. A node scoring 0.76 gives
no indication of whether it was reached in one hop through a 0.76 relation or
two hops through 0.95 and 0.80, and those mean different things about why it is
in the result. So every hop is recorded as it is taken, carrying its source,
target, confidence and relation.

The records are built even when nothing currently reads them. Rebuilding them
later means re-running the traversal against a graph that may have changed,
which does not answer the question that was asked of the original run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from ..common.config import MAX_DEGREE, MAX_HOPS, TOP_K_GRAPH


@dataclass(frozen=True)
class Hop:
    """One edge crossed, and what it was worth.

    ``depth`` is the hop number, counting from 1 for an edge leaving a seed, so
    a record says how far from a seed it was taken as well as between what.
    """

    source: str
    target: str
    relation: str
    confidence: float
    depth: int
    score: float


@dataclass(frozen=True)
class GraphHit:
    """One node reached, at the best score any path to it achieved."""

    id: str
    score: float
    depth: int


@dataclass
class GraphResult:
    """Scored nodes, the hops taken to reach them, and what it cost."""

    seeds: list[str] = field(default_factory=list)
    hits: list[GraphHit] = field(default_factory=list)
    hops: list[Hop] = field(default_factory=list)
    visited: int = 0
    traverse_seconds: float = 0.0

    @property
    def ids(self) -> list[str]:
        return [hit.id for hit in self.hits]

    def __len__(self) -> int:
        return len(self.hits)


def traverse(
    store,
    seeds: Iterable[str],
    *,
    max_hops: int = MAX_HOPS,
    max_degree: int | None = MAX_DEGREE,
    k: int = TOP_K_GRAPH,
) -> GraphResult:
    """Walk out from ``seeds``, scoring each node by its best path.

    A seed starts at 1.0 — it was chosen, not reached, so nothing has been
    multiplied yet.

    **A node enters the next frontier only if its new score beats the best
    already recorded for it.** That is what terminates the walk on a cyclic
    graph: coming back around to a node costs at least one more multiplication
    by a confidence at most 1.0, so the returning score cannot exceed the one
    already held and the node is not expanded again. No separate cycle check is
    needed, and a visited set alone would be weaker — it would stop the walk
    but also stop a genuinely better path found later from improving a score.

    The whole frontier is expanded in one query per hop rather than one per
    node. A traversal that issues a query per frontier node spends its time in
    round trips.
    """
    result = GraphResult(seeds=list(dict.fromkeys(seeds)))
    if not result.seeds:
        return result

    start = time.perf_counter()

    best: dict[str, float] = {seed: 1.0 for seed in result.seeds}
    depth_of: dict[str, int] = {seed: 0 for seed in result.seeds}
    frontier = list(result.seeds)

    for depth in range(1, max_hops + 1):
        if not frontier:
            break

        grouped = store.expand_frontier(frontier, k=k, max_degree=max_degree)

        improved: list[str] = []
        for origin in frontier:
            accumulator = best[origin]
            for neighbor in grouped.get(origin, []):
                confidence = float(neighbor["confidence"])
                score = accumulator * confidence

                result.hops.append(
                    Hop(
                        source=origin,
                        target=neighbor["id"],
                        relation=neighbor["relation"],
                        confidence=confidence,
                        depth=depth,
                        score=score,
                    )
                )

                if score > best.get(neighbor["id"], 0.0):
                    best[neighbor["id"]] = score
                    depth_of[neighbor["id"]] = depth
                    improved.append(neighbor["id"])

        frontier = list(dict.fromkeys(improved))

    result.traverse_seconds = time.perf_counter() - start
    result.visited = len(best)

    # Seeds are excluded from the result: they are the question, not an answer,
    # and returning them at 1.0 would put them above everything the traversal
    # actually found.
    reached = [
        GraphHit(id=node, score=score, depth=depth_of[node])
        for node, score in best.items()
        if node not in set(result.seeds)
    ]
    reached.sort(key=lambda hit: (-hit.score, hit.id))
    result.hits = reached[:k]
    return result
