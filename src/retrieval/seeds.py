"""Where a traversal starts, decided in three tiers.

1. **Exact entity linking.** Entities extracted from the query, matched by
   exact label against stored nodes. Deterministic; no similarity involved.
2. **Fuzzy vector seeds.** The top ``SEED_TOP_N`` vector hits at or above
   ``SEED_MIN_SIM``.
3. **Fallback.** If neither produced anything, the top vector hits whatever
   their similarity.

Why exact identity is tried first
---------------------------------

When a query names ``#412`` or ``payment_service`` there is a right answer, and
it is the node with that label. A nearest-neighbour search asked the same
question returns something plausible instead — measured on this project's
corpus, ``#413`` and ``#414`` embed at 0.9521 of each other because they are
near-identical as text, and for an identifier near-identical text means
definitely different. Starting a traversal from the wrong ticket produces a
confident, well-scored, wrong neighbourhood.

So the tiers are not ranked by quality of evidence in the abstract. They are
ordered because tier 1 either finds the thing named or finds nothing, and a
tier that cannot be plausibly wrong is worth consulting before one that can.

Tier 2 is capped rather than floored alone. ``SEED_MIN_SIM`` decides what is
admissible and ``SEED_TOP_N`` decides how much is taken: without the cap a
permissive floor seeds traversal from the whole result set, and a walk from
everywhere returns the graph ranked by nothing the walk contributed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..common.config import SEED_MIN_SIM, SEED_TOP_N

#: Which tier produced a seed. Recorded per run rather than derived, because
#: "tier 1 never fires" and "tier 1 fires and finds nothing" look identical in
#: a seed list and mean different things — one says queries do not name
#: entities, the other says linking is broken.
TIER_EXACT = "exact"
TIER_FUZZY = "fuzzy"
TIER_FALLBACK = "fallback"


class Extractor(Protocol):
    def extract(self, text: str) -> Any:
        ...


@dataclass
class SeedResult:
    """The chosen seeds and which tier chose them."""

    seeds: list[str] = field(default_factory=list)
    tier: str | None = None
    #: Entities the query produced, whether or not any matched a stored node.
    #: A run finding three entities and zero seeds is a linking problem; a run
    #: finding no entities is a query that named none.
    entities_found: int = 0
    matched: int = 0

    def __len__(self) -> int:
        return len(self.seeds)


def exact_seeds(store, extractor: Extractor, query: str) -> tuple[list[str], int, int]:
    """Nodes whose label exactly matches an entity named in the query.

    Returns the seeds, how many entities the query yielded, and how many of
    them matched. The three are reported together because a zero seed count
    means different things depending on the other two.

    Matching is the store's own label lookup, which is case-insensitive and
    exact. Anything looser is tier 2's job, and doing it here would remove the
    property that makes this tier worth consulting first.
    """
    entities = list(extractor.extract(query))
    seeds: list[str] = []
    matched = 0

    for entity in entities:
        found = store.find_by_label(entity.text)
        if found:
            matched += 1
        for row in found:
            seeds.append(row["id"])

    return list(dict.fromkeys(seeds)), len(entities), matched


def select(
    store,
    extractor: Extractor | None,
    vector_hits,
    query: str,
    *,
    min_similarity: float = SEED_MIN_SIM,
    top_n: int = SEED_TOP_N,
) -> SeedResult:
    """Choose seeds, trying each tier in order and stopping at the first hit.

    ``vector_hits`` is an already-computed ordered result, not a store handle:
    the caller has usually run the vector arm anyway, and encoding the query a
    second time here would double the model cost of every query.

    **Tier 2 is not consulted when tier 1 produces a seed.** That is the whole
    ordering, so it is a hard stop rather than a preference — mixing an exact
    match with fuzzy neighbours would put the plausible-but-wrong node back
    into a result the exact match was there to keep out.
    """
    result = SeedResult()

    if extractor is not None:
        seeds, found, matched = exact_seeds(store, extractor, query)
        result.entities_found = found
        result.matched = matched
        if seeds:
            result.seeds = seeds
            result.tier = TIER_EXACT
            return result

    above_floor = [hit for hit in vector_hits if hit.similarity >= min_similarity]
    if above_floor:
        result.seeds = list(dict.fromkeys(hit.id for hit in above_floor))[:top_n]
        result.tier = TIER_FUZZY
        return result

    if vector_hits:
        result.seeds = list(dict.fromkeys(hit.id for hit in vector_hits))[:top_n]
        result.tier = TIER_FALLBACK

    return result
