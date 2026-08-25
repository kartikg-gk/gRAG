"""One object over the four things a caller actually does.

Ingesting, indexing, querying and assembling context are each already a
callable, and each already works on its own. What was missing is the thing
that owns the store handle across all four, so a caller does not have to open
one, remember to close it, and thread it through every call by hand.

    with Engine("graph.db", embedder=embedder) as engine:
        engine.ingest(builder, documents=documents)
        engine.build_index()
        run = engine.retrieve("who reviewed the auth change?")
        context = engine.context(run)

That is the whole surface. This module sequences and owns a handle; it does
not retrieve, score, or decide anything. Every number in a result was computed
by the module that already computed it, and every one of those modules stays
independently callable — nothing here is a required path to reach them.

Why the handle is owned here
----------------------------

The store is the one piece with a lifetime. Everything else is a function.
Leaving the open and the close to the caller means every caller writes the
same ``try/finally``, and the one that forgets leaks connections until the
process ends. Entering opens; leaving closes, including when the body raises.

Nothing is opened in ``__init__``. Constructing an ``Engine`` is free and
touches no disk, so building one to inspect its configuration cannot leave a
database open. Calling an operation outside a ``with`` block raises rather
than opening one implicitly, because an implicit open has no matching implicit
close.

Why there is an async entry point
---------------------------------

Embedding and traversal are CPU-bound and slow enough to matter — encoding a
query dominates a retrieval by orders of magnitude. A caller that already runs
an event loop should not have it blocked for that. ``retrieve_async`` hands
the same synchronous call to an executor and awaits it.

**This is a threading concern and nothing else.** It listens on nothing,
serves nothing, and is not a step towards anything that does. It exists so
that a notebook or an agent loop can await a query instead of stalling on it.

The default executor is used rather than one owned here. An executor owned by
this object would be a second thing with a lifetime, and the store's read pool
already bounds how much concurrency reaches the database.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Iterable

from .common.config import TOP_K_GRAPH, TOP_K_VECTOR, TOTAL_K
from .graphdb import open_context_graph
from .knowledge.persist import persist
from .retrieval import retrieve as _retrieve


class EngineNotOpen(RuntimeError):
    """An operation was attempted outside a ``with`` block.

    Its own type rather than a bare ``RuntimeError`` so a caller can tell "you
    forgot the context manager" from a failure inside the store.
    """


class Engine:
    """The four operations, over one store, with one lifetime.

    ``embedder``, ``extractor``, ``judge`` and ``now`` are held here and passed
    to the calls that accept them. They are constructor arguments rather than
    per-call ones because they are properties of how this engine is
    configured, not of a single query — and they stay injectable because the
    paths below already degrade correctly without them:

    ==============  ==========================================================
    ``extractor``   Absent, seed tier 1 cannot run and selection falls to the
                    vector tiers.
    ``judge``       Absent, a query matching no marker takes the conceptual
                    fallback.
    ``now``         Absent, recency decays against the wall clock, which is
                    right in production and wrong in a test.
    ==============  ==========================================================

    None of that is re-implemented here. The absence is passed through and the
    existing code makes the same decision it already made.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedder=None,
        extractor=None,
        judge=None,
        now=None,
        initialize: bool = True,
        migrate: bool = True,
    ) -> None:
        self.path = path
        self.embedder = embedder
        self.extractor = extractor
        self.judge = judge
        self.now = now
        self._initialize = initialize
        self._migrate = migrate
        self._store = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Engine":
        self._store = open_context_graph(
            self.path, initialize=self._initialize, migrate=self._migrate
        )
        return self

    def __exit__(self, *exc_info) -> None:
        """Close on the way out, whatever the way out was.

        No exception is suppressed: the return is falsy, so a raising body
        still raises after the handle is closed.
        """
        self.close()

    def close(self) -> None:
        """Close the handle. Safe to call twice, and safe if never opened."""
        if self._store is not None:
            self._store.close()
            self._store = None

    @property
    def store(self):
        """The open handle.

        Exposed deliberately. The four operations below are the common path,
        not the only one — ``documents_for_entities``, ``most_connected`` and
        the rest stay reachable without this class growing a passthrough for
        each. Wrapping the whole store would make this a second interface to
        maintain in step with the first.
        """
        if self._store is None:
            raise EngineNotOpen(
                "the store is not open; use this object as a context manager"
            )
        return self._store

    # -- the four operations -----------------------------------------------

    def ingest(self, builder, *, documents: Iterable[Any] = ()):
        """Write a built graph and its documents into the store.

        Takes a builder rather than a repository name because fetching and
        building are a separate concern with their own failure modes — a
        rate limit, a transport error — and folding them in here would put a
        network call behind a method whose name says "write".

        Returns whatever the write path returns, unchanged.
        """
        return persist(
            self.store,
            builder,
            embedder=self._embedder(),
            extractor=self.extractor,
            documents=documents,
        )

    def build_index(self, *, rebuild: bool = True) -> None:
        """Build the vector index over what has been written.

        Separate from ``ingest`` because indexing costs a pass over every row.
        Doing it once after a batch of writes is one pass; doing it inside the
        write would be one per batch, and it would make the two costs
        impossible to time apart.

        ``rebuild`` defaults to true because the index raises rather than
        no-opping when one already exists, which makes false the setting that
        fails on the second run.
        """
        self.store.build_vector_index(rebuild=rebuild)

    def retrieve(
        self,
        query: str,
        *,
        k_vector: int = TOP_K_VECTOR,
        k_graph: int = TOP_K_GRAPH,
        total_k: int = TOTAL_K,
    ):
        """Run the full retrieval sequence for ``query``.

        Every stage's result comes back, not only the fused ranking, because
        that is what the sequence already returns and narrowing it here would
        hide the arms' separate behaviour from anyone holding this object.
        """
        return _retrieve(
            self.store,
            self._embedder(),
            query,
            extractor=self.extractor,
            judge=self.judge,
            now=self.now,
            k_vector=k_vector,
            k_graph=k_graph,
            total_k=total_k,
        )

    async def retrieve_async(self, query: str, **kwargs):
        """``retrieve``, off the calling thread.

        Same call, same arguments, same result — handed to the default
        executor so an event loop stays free while encoding and traversal run.
        A failure propagates exactly as it would synchronously.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self.retrieve, query, **kwargs)
        )

    def context(self, results) -> dict:
        """The retrieved entities, their neighbours, and the edges between.

        Accepts a retrieval run or any iterable of entity ids. A run is what a
        caller has just been handed, and ids are what a caller has when the
        selection came from somewhere else; requiring the first would force
        the second to construct one.
        """
        ids = getattr(results, "ids", None)
        if ids is None:
            ids = list(results)
        return self.store.subgraph(ids)

    # -- internals ---------------------------------------------------------

    def _embedder(self):
        """The configured embedder, or the default one, built on first use.

        Imported here rather than at module scope because the default drags in
        a model and the machinery under it. Constructing an ``Engine``, or
        using one for anything that needs no vectors, must not pay for that.
        """
        if self.embedder is None:
            from .analysis import Similarity

            self.embedder = Similarity()
        return self.embedder
