"""Two retrieval arms, run independently.

The vector arm encodes a query and searches the index. The graph arm walks out
from seeds, scoring by the confidence of the path taken. Seed selection is a
third piece, sitting between them: it decides where the graph arm starts, and
one of its tiers reads the vector arm's output.

**Nothing here fuses the two.** There is no combined score, no weighting, no
merged ranking. Each arm returns its own ordered list and neither can see the
other's results — the graph arm receives seed ids and never a vector score,
and the vector arm receives nothing from traversal at all.

That separation is the point of this layer rather than an unfinished state. A
blend is a set of weights, and weights chosen before there is a measurement of
what each arm finds alone are chosen from intuition. Keeping the arms apart is
what makes the cases where one finds something the other misses visible; once
the lists are merged there is one ranking and no way to tell which arm earned
a position in it.
"""

from __future__ import annotations

from .graph_arm import GraphHit, GraphResult, Hop, traverse
from .seeds import (
    TIER_EXACT,
    TIER_FALLBACK,
    TIER_FUZZY,
    SeedResult,
    exact_seeds,
    select,
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
]
