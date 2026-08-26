"""One object over the four things a caller actually does.

Ingesting, indexing, querying and assembling context are each already a
callable, and each already works on its own. What was missing is the thing
that owns the store handle across all four, so a caller does not have to open
one, remember to close it, and thread it through every call by hand.

    with Engine("graph.db", embedder=embedder) as engine:
        engine.ingest(builder, documents=documents)
        engine.build_index()
        run = engine.retrieve("who reviewed the auth change?")
        text = engine.context("who reviewed the auth change?")

That is the whole surface. This module sequences and owns a handle; it does
not retrieve, score, or decide anything. Every number in a result was computed
by the module that already computed it, and every one of those modules stays
independently callable — nothing here is a required path to reach them.

Why the handle is owned here
----------------------------

The store is the one piece with a lifetime. Everything else is a function.
Leaving the open and the close to the caller means every caller writes the
same ``try/finally``, and the one that forgets leaks connections until the
process ends.

**Constructing opens the store.** Entering the context manager returns the
object as it already is, and leaving closes it — including when the body
raises. So a caller in a notebook, or in a script that runs to completion, gets
something usable from the constructor and is not forced into a block to get
it; the block is how you ask for a close at a known point, not how you get an
engine.

The cost of that is real and worth stating: an engine built and then abandoned
holds an open handle until it is closed or the process ends. Construction is
no longer free, and anything that builds one to inspect its configuration
pays for a database.

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

from .graphdb import open_context_graph
from .knowledge.persist import persist
from .results import as_prompt
from .retrieval import retrieve as _retrieve


class EngineNotOpen(RuntimeError):
    """An operation was attempted on an engine whose store has been closed.

    Its own type rather than a bare ``RuntimeError`` so a caller can tell "this
    engine is spent" from a failure inside the store. Construction opens, so
    reaching this means ``close`` has already run.
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
        #: Set once the index has been built in this process. A query consults
        #: it so that forgetting the explicit build is not fatal.
        self._indexed = False
        # Opened here, not on entry. A caller in a notebook or a script that
        # runs to completion gets a working object from the constructor, and
        # the context manager becomes optional sugar for closing rather than
        # the only way to get a usable one.
        self._store = open_context_graph(
            self.path, initialize=self._initialize, migrate=self._migrate
        )

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Engine":
        """The object as it already is. Construction did the opening."""
        return self

    def __exit__(self, *exc_info) -> None:
        """Close on the way out, whatever the way out was.

        No exception is suppressed: the return is falsy, so a raising body
        still raises after the handle is closed.
        """
        self.close()

    def close(self) -> None:
        """Close the handle. Safe to call twice.

        Idempotent because two owners can reasonably both close — a context
        manager on the way out and whatever else holds the engine.
        """
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
            raise EngineNotOpen("this engine's store has been closed")
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
        self._indexed = True

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        *,
        k_vector: int | None = None,
        k_graph: int | None = None,
    ):
        """Run the full retrieval sequence for ``query``.

        Every stage's result comes back, not only the fused ranking, because
        that is what the sequence already returns and narrowing it here would
        hide the arms' separate behaviour from anyone holding this object.
        """
        if not self._indexed:
            # A caller who ingested and asked a question straight away gets
            # results rather than an error, or worse an empty set that says
            # nothing about what was forgotten. The explicit build still
            # exists; this only makes skipping it non-fatal, and the flag is
            # what stops it happening on every query.
            self.build_index()

        # Only the counts a caller actually gave are forwarded, so an unset
        # one keeps whatever default the sequence itself applies. Repeating
        # those defaults here would be a second copy to keep in step, and the
        # copy would win silently when the two disagreed.
        counts = {
            name: value
            for name, value in (
                ("total_k", k),
                ("k_vector", k_vector),
                ("k_graph", k_graph),
            )
            if value is not None
        }

        return _retrieve(
            self.store,
            self._embedder(),
            query,
            extractor=self.extractor,
            judge=self.judge,
            now=self.now,
            **counts,
        )

    async def retrieve_async(self, query: str, *args, **kwargs):
        """``retrieve``, off the calling thread.

        Same call, same arguments, same result — handed to the default
        executor so an event loop stays free while encoding and traversal run.
        A failure propagates exactly as it would synchronously.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self.retrieve, query, *args, **kwargs)
        )

    def context(self, question: str, k: int | None = None) -> str:
        """Everything retrieved for ``question``, as one prompt-ready string.

        Runs the query itself and returns text, not structure. A caller that
        wants the ranking, the per-arm scores or the trace has ``retrieve``
        for exactly that; this one exists to produce something that can be
        dropped in front of a model without further assembly, and returning a
        structure would leave every caller writing the same formatting.

        Two labelled sections — the entities, then the prose behind them —
        with the text deduplicated by chunk. See ``results.as_prompt`` for why
        that deduplication is the point rather than a nicety.
        """
        run = self.retrieve(question, k)
        documents = self.store.documents_for_entities(run.ids)
        return as_prompt(run.hits, documents)

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
