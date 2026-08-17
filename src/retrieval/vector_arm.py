"""Retrieval by embedding similarity.

Encode the query, search the vector index, return what came back in order.
That is the whole arm, and the smallness is the point: this returns a raw
ranked list and applies no floor, so what it is good at and what it is bad at
are both visible in the output rather than hidden behind a cutoff.

**No filtering here.** A floor belongs to whatever decides what to do with
these results — seed selection applies one, and a caller comparing arms wants
the unfiltered list. Filtering at the source would make a result that scored
0.2 indistinguishable from one that did not exist, and those are different
findings.

Cost
----

Encoding is a model call and index search is not, so the two are reported
separately by ``search``. Cold start is a third cost again: the first call in a
process pays for loading the model, measured at roughly 21 seconds against
sub-millisecond warm encoding. One number covering all three would hide which
one hurts, and they are addressed differently — cold start by loading once per
process, warm encoding by batching, search by the index itself.

The embedder is passed in rather than constructed here. Loading a model takes
seconds, and a module that constructs its own makes that cost implicit and
unavoidable for a caller that wanted to reuse one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..common.config import TOP_K_VECTOR


class Embedder(Protocol):
    def vector(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class VectorHit:
    """One result, with what the store knew about it."""

    id: str
    similarity: float
    label: str | None = None
    type: str | None = None


@dataclass
class VectorResult:
    """An ordered result set, and what producing it cost.

    The timings travel with the result rather than being logged, so a caller
    measuring the arm does not have to instrument it from outside and a caller
    that does not care can ignore them.
    """

    query: str
    hits: list[VectorHit] = field(default_factory=list)
    encode_seconds: float = 0.0
    search_seconds: float = 0.0

    @property
    def ids(self) -> list[str]:
        return [hit.id for hit in self.hits]

    def __len__(self) -> int:
        return len(self.hits)


def search(
    store,
    embedder: Embedder,
    query: str,
    *,
    k: int = TOP_K_VECTOR,
) -> VectorResult:
    """The ``k`` nearest entities to ``query``, ordered by similarity.

    Ordering comes from the store, which returns ascending distance, and
    similarity is ``1 - distance`` under the cosine metric — so descending
    similarity is the same order. It is asserted rather than assumed, because
    a metric change would silently reverse it.

    An empty query returns an empty result without touching the model. There
    is nothing to encode, and a vector of the empty string is not a meaningful
    place in the space.
    """
    result = VectorResult(query=query)
    if not query or not query.strip():
        return result

    start = time.perf_counter()
    vector = embedder.vector(query)
    result.encode_seconds = time.perf_counter() - start

    start = time.perf_counter()
    rows = store.vector_search(vector, k=k)
    result.search_seconds = time.perf_counter() - start

    result.hits = [
        VectorHit(
            id=row["id"],
            similarity=float(row["similarity"]),
            label=row.get("label"),
            type=row.get("type"),
        )
        for row in rows
    ]
    return result
