"""What retrieval did, recorded from inside it.

Four sections, one per stage that makes a decision: how the query was
classified, where traversal started and where it walked, how age moved each
score, and how much work the run did.

Why this is not the tracing schema
----------------------------------

``src.tracing`` records what an *external* workflow did, captured from outside
it by a caller that wraps it. This records what *this* engine's retrieval did,
emitted from inside. Different producer, different consumer, different rate of
change.

Coupling them would mean every engine-internal field forced a version bump on a
published format that other people load. The friction is already visible in the
shapes: a fused item's score components map onto ``TraceItem`` cleanly, and
query-level values — the weights used, which stage classified the query, how
many nodes traversal visited — have nowhere to go without extending ``Trace``
itself. A format read by outside consumers should not change every time an
internal metric is added.

So this stays its own structure. Anything that wants both can hold both.

Nothing here computes
---------------------

Every value is read from a result some stage already produced. The hop records
are the objects traversal built, not copies rebuilt from a summary, and the
decay factors are the ones fusion actually multiplied by rather than a second
evaluation of the same curve. A recomputed number can drift from the one that
ranked the results, and a report that disagrees with the ranking is worse than
no report — it looks like a ranking bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IntentSection:
    """How the query was classified, and what that bought it.

    ``stage`` matters as much as ``label``: a query classified conceptual
    because a marker said so and one classified conceptual because no judge
    was configured produce identical weights and mean different things.
    """

    label: str
    stage: str
    marker: str | None
    alpha: float
    beta: float


@dataclass(frozen=True)
class SeedRecord:
    """One seed, and which tier produced it."""

    id: str
    tier: str | None


@dataclass(frozen=True)
class ExecutionPath:
    """Where traversal started and every edge it crossed.

    ``hops`` holds the ``Hop`` objects traversal built. They are not rebuilt
    here: a final score cannot say whether a node was reached in one hop at
    0.76 or two at 0.95 and 0.80, so reconstructing the path from the result
    would be inventing one of several histories that fit.
    """

    seeds: list[SeedRecord] = field(default_factory=list)
    tier: str | None = None
    hops: list[Any] = field(default_factory=list)
    entities_found: int = 0
    entities_matched: int = 0


@dataclass(frozen=True)
class RecencyRecord:
    """How old one node was, and what that did to its score.

    ``age_days`` is ``None`` when the node carries no timestamp, and ``decay``
    is then 1.0. Both are reported because 1.0 arises three ways — no date,
    dated today, recency disabled — and only the pair separates them.
    """

    id: str
    node_type: str | None
    age_days: float | None
    decay: float


@dataclass(frozen=True)
class Metrics:
    """How much work the run did."""

    graph_hits: int
    vector_k: int
    visited: int
    vector_returned: int = 0
    fused_returned: int = 0


@dataclass(frozen=True)
class TraceLog:
    """One query's record, assembled from what the stages returned."""

    query: str
    intent: IntentSection
    execution_path: ExecutionPath
    recency: list[RecencyRecord] = field(default_factory=list)
    metrics: Metrics | None = None


def build(query, intent, seeds, vector, graph, fused) -> TraceLog:
    """Assemble a log from results that already exist.

    Takes finished results rather than a store or a query, so it can be tested
    without either and so it cannot accidentally re-run a stage. Every argument
    is something a caller already has by the time fusion is done.
    """
    return TraceLog(
        query=query,
        intent=IntentSection(
            label=intent.intent,
            stage=intent.stage,
            marker=intent.marker,
            alpha=fused.alpha,
            beta=fused.beta,
        ),
        execution_path=ExecutionPath(
            seeds=[SeedRecord(id=seed, tier=seeds.tier) for seed in seeds.seeds],
            tier=seeds.tier,
            # The list traversal produced, held rather than copied.
            hops=graph.hops,
            entities_found=seeds.entities_found,
            entities_matched=seeds.matched,
        ),
        recency=[
            RecencyRecord(
                id=hit.id,
                node_type=hit.node_type,
                age_days=hit.age_days,
                decay=hit.decay,
            )
            # Read off the fused hits, so these are the factors that ranked
            # the results rather than a second look at the same curve.
            for hit in fused.hits
        ],
        metrics=Metrics(
            graph_hits=fused.graph_hits,
            vector_k=fused.vector_k,
            visited=graph.visited,
            vector_returned=len(vector.hits),
            fused_returned=len(fused.hits),
        ),
    )
