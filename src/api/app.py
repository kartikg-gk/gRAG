"""The HTTP surface.

Deliberately thin. Retrieval, the store and their lifecycle are all already
solved behind the facade, so this layer does three things and nothing else:
establish who is asking and which tenant they may read, hand the query to the
facade, and shape the result into a declared contract.

Everything it serves was computed by the module that already computed it. No
route re-scores, re-ranks, re-queries or paraphrases anything.

One store, opened once
----------------------

The store opens at startup and closes at shutdown, not per request. Opening a
graph database per request would pay the open on every call and hold no
connection pool across them, and the pool is the thing that makes concurrent
reads work at all.

The embedding model is warmed at startup for the same reason, and it is the
larger cost: a cold start is roughly fifteen seconds, paid by whoever calls
first. Left alone that is a user. Moved to boot it is nobody.

**A startup that fails part-way still closes what it opened.** The store is
entered with a ``with`` around the yield, so a warm-up that raises unwinds
through it and the handle does not leak into a process that then fails to
serve.

Failures are statuses, not empty sets
-------------------------------------

A query that cannot run returns 5xx. It does not return an empty result list,
because "the graph holds nothing matching this" and "the graph could not be
read" are different facts and a caller that cannot tell them apart will
report the first while the second is happening.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status

from ..common.config import (
    DEFAULT_TENANT_ORG_ID,
    STORE_PATH,
    WARM_EMBEDDER_ON_STARTUP,
)
from ..graphs import graph_label, graph_paths
from ..registry import GraphRegistry
from ..results import format_run
from ..suggestions import OVERFETCH, suggestions_from
from . import auth as auth_module
from .auth import get_current_tenant_org, get_current_user
from .models import (
    GraphSummary,
    GraphsResponse,
    Health,
    QueryRequest,
    QueryResponse,
    SubgraphRequest,
    SubgraphResponse,
    Suggestion,
    SuggestionsResponse,
    SwitchRequest,
    SwitchResponse,
)

#: The text embedded at startup to pay the model's cold start. Its content is
#: irrelevant — only that it is short and that something goes through the
#: model before a caller does.
WARMUP_TEXT = "warm"


def default_engine_factory():
    """Build the engine this process will serve from.

    Imported inside the function because constructing the embedder loads a
    model, and importing this module — to read its routes, to build an app for
    a test — must not.
    """
    from ..analysis import Similarity
    from ..engine import Engine

    embedder = Similarity()
    if WARM_EMBEDDER_ON_STARTUP:
        # The whole point of doing this at startup. It has to be a real embed:
        # constructing the object does not load the weights.
        embedder.vector(WARMUP_TEXT)
    return Engine(STORE_PATH, embedder=embedder)


def graph_loader(engine):
    """A loader that opens a graph reusing an existing engine's models.

    The registry stores opaque handles and must not know what an engine is, so
    the knowledge of how to open one lives here and is injected into it.

    **This is the reason a tenth graph does not cost a gigabyte.** The
    statistical extraction stage is roughly 65 MiB resident and the embedding
    model carries a multi-second cold start on top of its own footprint;
    constructing them per graph would multiply both by the number of graphs
    open. Every piece that has state is passed through from the engine that
    already built it, so what a new graph adds is a store handle and nothing
    else.
    """
    from ..engine import Engine

    def load(path: str):
        return Engine(
            path,
            embedder=engine.embedder,
            extractor=engine.extractor,
            judge=engine.judge,
            now=engine.now,
        ).__enter__()

    return load


def engine_for(request: Request, org_id: str):
    """The engine holding ``org_id``'s graph, or a 503.

    **The registry receives graphs; it does not find them.** Nothing here
    turns an identifier into a location and nothing opens a store because a
    request arrived — a graph is servable exactly when something has already
    attached it or pointed a key at a path.

    **The tenant is an argument, not ambient state.** A context variable
    carrying the current tenant exists and would work, and using it would make
    this function's result depend on something invisible at the call site.
    Passing it means a route that forgets to resolve one does not compile
    rather than quietly serving whatever was last set.

    A miss is **service-unavailable, not not-found**. The tenant may well
    exist and be perfectly valid; this process simply does not hold its graph.
    404 would say the tenant is unknown, which is a claim this layer is in no
    position to make.
    """
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the store is not open",
        )

    engine = registry.get(org_id)
    if engine is None and not auth_module.MULTI_TENANCY_ENABLED:
        # Single-tenant operation, unchanged: one store, opened at startup
        # from the configured path, serving whatever identifier arrives.
        engine = registry.get(DEFAULT_TENANT_ORG_ID)

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="this tenant's graph is not loaded in this process",
        )
    return engine


def _same_file(left, right) -> bool:
    """Whether two paths name the same file, comparing resolved forms.

    The same store reached by two spellings is one store, and marking the
    active graph has to say so or the list shows nothing active while
    something plainly is.
    """
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left).absolute() == Path(right).absolute()


def create_app(*, engine_factory=None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level instance so a test can build one per
    configuration, and ``engine_factory`` is injectable for the same reason —
    the default one loads a model and opens a database, neither of which a
    test of routing should do.
    """
    build_engine = engine_factory or default_engine_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = build_engine()
        registry = GraphRegistry()
        # Opening a second graph must not load a second copy of the models.
        # The loader closes over the instances this engine already warmed, so
        # every graph opened later shares them rather than paying the cold
        # start and the resident cost again.
        registry.set_loader(graph_loader(engine))

        # The `with` is the guarantee for a startup that fails before the
        # store reaches the registry. Past that point the registry owns the
        # close, and Engine.close is idempotent, so the two do not conflict.
        with engine:
            # Single-tenant operation is the ordinary path: the process store
            # is attached under the default identifier, so a run with no
            # tenancy configured serves from exactly the store it opens today
            # and never consults the tenant root at all.
            registry.attach(DEFAULT_TENANT_ORG_ID, engine, path=str(engine.path))
            app.state.engine = engine
            app.state.registry = registry
            app.state.active_path = Path(engine.path)
            try:
                yield
            finally:
                app.state.engine = None
                app.state.registry = None
                app.state.active_path = None
                # Every handle the registry holds, not just the one startup
                # opened -- anything attached during the run closes here too.
                registry.close_all()

    app = FastAPI(title="graphrag", lifespan=lifespan)

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        """Unauthenticated on purpose.

        A readiness probe that needs a credential cannot report that
        credentials are misconfigured, which is one of the things it most
        needs to be able to report.
        """
        return Health(
            status="ok",
            store_open=getattr(request.app.state, "engine", None) is not None,
        )

    @app.post("/query", response_model=QueryResponse)
    async def query(
        payload: QueryRequest,
        request: Request,
        user_id: str = Depends(get_current_user),
        org_id: str = Depends(get_current_tenant_org),
    ) -> QueryResponse:
        """Run one query and return the ranked set with its evidence.

        Both dependencies are required because this route reads graph data.
        A data-bearing route without a resolved tenant is a route that reads
        whichever store the process happens to hold, which is the failure the
        tenant check exists to prevent — so it is resolved here even though
        the store is currently one per process.

        The query goes through the facade's async entry point, so encoding and
        traversal run off the event loop rather than blocking every other
        request for the duration.
        """
        import asyncio

        engine = engine_for(request, org_id)
        try:
            run = await engine.retrieve_async(payload.query, payload.k)
            # A second blocking call, and the facade has no async form of it.
            # Handed to a thread here rather than reaching past the facade or
            # reshaping it for one consumer.
            documents = await asyncio.to_thread(
                engine.store.documents_for_entities, run.ids
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="the query could not be completed",
            ) from exc

        return QueryResponse(**format_run(run, documents))

    @app.post("/subgraph", response_model=SubgraphResponse)
    async def subgraph(
        payload: SubgraphRequest,
        request: Request,
        user_id: str = Depends(get_current_user),
        org_id: str = Depends(get_current_tenant_org),
    ) -> SubgraphResponse:
        """The given nodes, their one-hop neighbours, and the edges between.

        Reads graph data, so it resolves a tenant like the query route does.

        An unknown id contributes nothing rather than failing the request. A
        caller expanding a set it got from somewhere else should not have the
        whole call rejected because one node has since gone.

        The store read is blocking and goes to a thread for the same reason
        the query path does: on the event loop it would serialise every other
        request behind it.
        """
        import asyncio

        engine = engine_for(request, org_id)
        try:
            result = await asyncio.to_thread(engine.store.subgraph, payload.ids)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="the subgraph could not be read",
            ) from exc

        return SubgraphResponse(**result)

    @app.get("/suggestions", response_model=SuggestionsResponse)
    async def suggestions(
        request: Request,
        limit: int = 5,
        user_id: str = Depends(get_current_user),
        org_id: str = Depends(get_current_tenant_org),
    ) -> SuggestionsResponse:
        """Example questions built from what this graph actually holds.

        Over-fetches and diversifies -- see ``suggestions_from`` for why asking
        for exactly the limit returns the same kind of thing repeatedly.

        **A store that cannot answer yields an empty list, not an error.** A
        graph too small or too new to suggest anything is an ordinary state,
        and a caller rendering a prompt bar should show nothing rather than an
        error where its examples would be.
        """
        import asyncio

        engine = engine_for(request, org_id)
        try:
            entities = await asyncio.to_thread(
                engine.store.most_connected, max(limit, 1) * OVERFETCH
            )
        except Exception:
            return SuggestionsResponse(suggestions=[])

        return SuggestionsResponse(
            suggestions=[
                Suggestion(**item) for item in suggestions_from(entities, limit)
            ]
        )

    # ----------------------------------------------------------------------
    # The local single-store workflow.
    #
    # These two enumerate graph files in this checkout and change which one
    # this process serves. They are **not** multi-tenancy: they operate on the
    # default tenant only and never touch another tenant's registry entry.
    # Multi-tenancy is the registry path, where a graph arrives from outside
    # and is attached.
    #
    # **Both are deliberately unauthenticated.** They are a local developer
    # affordance -- somebody keeping a graph per repository and moving between
    # them -- and requiring a credential to switch a graph on your own machine
    # is friction for no protection there. The argument against is real and
    # was made: they change what the server serves for everybody, so on
    # anything reachable they are an unauthenticated state change. That
    # argument was heard and this is the decision anyway, recorded here so the
    # next reader sees a choice rather than an oversight. There is no flag to
    # toggle it: a third behaviour is a third thing to reason about.
    # ----------------------------------------------------------------------

    @app.get("/graphs", response_model=GraphsResponse)
    def graphs(request: Request) -> GraphsResponse:
        """Every graph this checkout can serve, and which one is active."""
        active_path = getattr(request.app.state, "active_path", None)

        summaries = []
        active_id = None
        for path in graph_paths():
            identifier = path.stem
            is_active = active_path is not None and _same_file(path, active_path)
            if is_active:
                active_id = identifier
            summaries.append(
                GraphSummary(id=identifier, label=graph_label(path), active=is_active)
            )

        return GraphsResponse(graphs=summaries, active=active_id)

    @app.post("/graphs/switch", response_model=SwitchResponse)
    def switch(payload: SwitchRequest, request: Request) -> SwitchResponse:
        """Serve a different discovered graph from now on.

        The id must be one discovery found. Anything else is a 404 rather than
        an attempt to open whatever path was sent -- this route selects among
        known files, it does not take instructions about the filesystem.

        Three things move together, and missing any one leaves the process
        half-switched:

        * the router is repointed at the new store **without being rebuilt**,
          so the loaded models are kept rather than paid for again
        * the recorded active store and path are updated
        * the **default tenant's registry entry is re-bound**, because tenant
          resolution reads the registry and would otherwise keep handing out
          the previous file while this route served the new one

        The displaced store is closed only if it really is a different object.
        Closing the one just installed would leave the process serving a shut
        handle.
        """
        registry = getattr(request.app.state, "registry", None)
        if registry is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="the store is not open",
            )

        match = next((path for path in graph_paths() if path.stem == payload.id), None)
        if match is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="no such graph"
            )

        try:
            # Opens through the registry loader, which carries the warm models,
            # and re-binds the default tenant in the same call.
            displaced = registry.replace(DEFAULT_TENANT_ORG_ID, path=str(match))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="the graph could not be opened",
            ) from exc

        engine = registry.get(DEFAULT_TENANT_ORG_ID)
        app.state.engine = engine
        app.state.active_path = match

        # Only if it is genuinely displaced. Closing the new one would leave
        # this process serving a shut handle.
        if displaced is not None and displaced.handle is not engine:
            registry.close_entries([displaced])

        return SwitchResponse(
            id=match.stem, label=graph_label(match), nodes=engine.store.count_nodes()
        )

    @app.get("/projects")
    def projects(
        request: Request,
        user_id: str = Depends(get_current_user),
        org_id: str = Depends(get_current_tenant_org),
    ) -> dict:
        """What the two dependencies were built against.

        Reads no graph data — it returns the two resolved ids and nothing
        else, which is what makes it a check on the credentials rather than a
        query path.
        """
        return {
            "user_id": user_id,
            "org_id": org_id,
            "state_user_id": request.state.user_id,
            "state_org_id": request.state.org_id,
        }

    return app
