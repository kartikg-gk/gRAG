"""Two retrieval arms, run independently.

The vector arm encodes a query and searches the index. The graph arm walks out
from seeds, scoring by the confidence of the path taken. Seed selection is a
third piece, sitting between them: it decides where the graph arm starts, and
one of its tiers reads the vector arm's output.

**The arms themselves never fuse.** Each returns its own ordered list and
neither can see the other's results — the graph arm receives seed ids and
never a vector score, and the vector arm receives nothing from traversal at
all. That separation is what makes the cases where one arm finds something the
other misses visible; a merged list has one ranking and no way to tell which
arm earned a position in it.

``fuse`` combines them afterwards, as a separate step over two finished
results. It reads scores and never calls either arm, so the independence above
survives it: an arm cannot be influenced by a blend it does not know exists.
Every fused hit keeps its component scores for the same reason — a total of
0.42 says nothing about whether one arm found the node strongly and decay cut
it, or both arms found it weakly.

The weights are starting values, not measured ones. What would justify or move
them is a per-query comparison of the fused ranking against each arm's raw
ranking, which needs the arms to stay separately runnable — hence both.
"""

from __future__ import annotations

from .fuse import FusedHit, FusedResult, attributes_for, fuse, vector_budget, weights_for
from .graph_arm import GraphHit, GraphResult, Hop, traverse
from .intent import Intent, classify, first_marker
from .pipeline import RetrievalRun, retrieve
from .recency import age_and_decay, age_days, decay_factor, half_life_for
from .router import RetrievalRouter, shutdown_embed_executor
from .seeds import (
    TIER_EXACT,
    TIER_FALLBACK,
    TIER_FUZZY,
    SeedResult,
    exact_seeds,
    select,
)
from .trace_log import (
    ExecutionPath,
    IntentSection,
    Metrics,
    RecencyRecord,
    SeedRecord,
    TraceLog,
)
from .vector_arm import VectorHit, VectorResult, search

__all__ = [
    # vector arm
    "search",
    "VectorHit",
    "VectorResult",
    # graph arm
    "traverse",
    "GraphHit",
    "GraphResult",
    "Hop",
    # seeds
    "select",
    "exact_seeds",
    "SeedResult",
    "TIER_EXACT",
    "TIER_FUZZY",
    "TIER_FALLBACK",
    # intent
    "classify",
    "first_marker",
    "Intent",
    # recency
    "decay_factor",
    "age_days",
    "half_life_for",
    # fusion
    "fuse",
    "weights_for",
    "vector_budget",
    "attributes_for",
    "FusedHit",
    "FusedResult",
    # the sequence, and what it records
    "retrieve",
    "RetrievalRun",
    "TraceLog",
    "IntentSection",
    "ExecutionPath",
    "SeedRecord",
    "RecencyRecord",
    "Metrics",
    "age_and_decay",
    "RetrievalRouter",
    "shutdown_embed_executor",
]
