"""The HTTP surface.

Deliberately thin. Retrieval, the store and their lifecycle are all already
solved behind the facade, so this layer does three things and nothing else:
establish who is asking and which tenant they may read, hand the query to the
facade, and shape the result into a declared contract.

Every route lives under ``/api``.

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

The pod agent, and why it is off by default
-------------------------------------------

A process can also run the agent that keeps this machine's loaded graphs in
agreement with what the control plane says it should hold. It is off unless
``GRAPHRAG_POD_AGENT`` is exactly ``"1"``.

An exact comparison rather than a truthiness test, because the two mistakes
are not the same size. A deployment that meant to switch it on and wrote
``"true"`` gets a process with no agent and a log that says so. A deployment
that never wanted it, with something unrelated left in that variable, would
otherwise get a background task polling a database it does not have.

Everything the agent needs is imported **inside** that gate, so a process
running without it neither pays for those imports nor fails to start because
something on that path is unavailable.

The model routes
----------------

Summaries and answers call the general judge client. They are limited per user
and cached, and a failure — including a client that is not configured — comes
back in the response body rather than as an error status, because the caller
has already been shown the trace and an absent summary is not a failed page.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from ..cache import LRUCache
from ..common.config import (
    ANSWER_CONTEXT_MAX_CHARS,
    ANSWER_MAX_TOKENS,
    API_DOCS_ENABLED,
    CORS_ORIGINS,
    DEFAULT_TENANT_ORG_ID,
    POD_ID,
    SENTRY_DSN,
    SENTRY_ENVIRONMENT,
    SENTRY_TRACES_SAMPLE_RATE,
    STORE_PATH,
    SUMMARY_MAX_TOKENS,
    SUMMARY_TEXT_MAX_CHARS,
    TOP_K_VECTOR,
    TRACE_MAX_TOP_K,
)
from ..graphs import graph_label, graph_paths
from ..registry import REGISTRY
from ..retrieval.response import format_page_content
from . import auth as auth_module
from .auth import get_current_tenant_org, get_current_user
from .history_routes import router as history_router
from .github_oauth import router as github_oauth_router
from .models import (
    AnswerRead,
    AnswerRequest,
    GraphSwitchRead,
    GraphsRead,
    HealthRead,
    SubgraphRead,
    SubgraphRequest,
    SuggestionsRead,
    SummaryRead,
    SummarizeRequest,
    SwitchRequest,
    TraceRequest,
    TraceResponseRead,
)
from .onboarding import router as onboarding_router
from .ratelimit import LLM_RATE_LIMIT, limiter, trace_rate_guard
from .routing import TenantRoutingMiddleware
from .webhooks import router as webhooks_router

logger = logging.getLogger("graphrag.api.app")

#: The variable that switches the pod agent on, and the one value that does
#: it. See the module docstring for why this is compared rather than tested
#: for truth.
POD_AGENT_VARIABLE = "GRAPHRAG_POD_AGENT"
POD_AGENT_ENABLED = "1"

#: The version the local default graph is bound under in the registry.
LOCAL_DEFAULT_VERSION = "0"

#: One-sentence summaries, by key. Emptied when the served graph changes,
#: because the snippets they summarise belong to the previous one.
SUMMARY_CACHE: LRUCache[str, str] = LRUCache(capacity=512)

#: Answers, by question and context. Shared by the blocking and the streaming
#: route, so either one can answer from what the other produced.
ANSWER_CACHE: LRUCache[str, str] = LRUCache(capacity=512)

SUMMARY_PROMPT = (
    "Condense this engineering note into a single sentence "
    "of at most 25 words covering what changed and why. Return only the "
    "sentence.\n\n"
)

NO_CONTEXT_ANSWER = "Nothing relevant was retrieved for this question."
STREAM_FAILED_ANSWER = "The answer could not be produced right now."

#: The question template each hub type is offered with; anything else gets
#: the default.
SUGGESTION_TEMPLATES: dict[str, str] = {
    "Person": "Show recent contributions by {label}.",
    "Team": "Which areas are maintained by {label}?",
    "Service": "Which components rely on {label}?",
    "Library": "Which changes involve {label}?",
    "Tool": "Where is {label} used?",
    "PR": "Which entities connect to {label}?",
    "Ticket": "Which work items connect to {label}?",
}
SUGGESTION_DEFAULT = "Explore connections around {label}."

_llm: dict = {}


def pod_agent_enabled() -> bool:
    """Whether this process should run the agent.

    Read when the application starts rather than when this module is
    imported, so what is set at the moment of starting is what decides.
    """
    return os.environ.get(POD_AGENT_VARIABLE) == POD_AGENT_ENABLED


def _bring_up_the_control_plane() -> None:
    """Create the control-plane tables, and shrug if there is no database.

    Wrapped, and deliberately not fatal. The serving path does not read these
    tables: a local run with nothing configured has to come up and answer
    queries exactly as it did before any of this existed. An unavailable
    control plane is a line in the log, not a process that will not start.
    """
    try:
        from ..models import create_control_plane_engine, create_control_plane_schema

        engine = create_control_plane_engine()
        try:
            create_control_plane_schema(engine)
        finally:
            engine.dispose()
    except Exception:  # noqa: BLE001 - serving does not depend on this
        logger.warning(
            "the control plane is unavailable; continuing without it", exc_info=True
        )


def _bring_up_history() -> None:
    """Create the history tables, and keep serving if there is no database."""
    try:
        from ..history import initialize

        initialize().dispose()
    except Exception as exc:  # noqa: BLE001 - the API boots without history
        logger.warning("history is disabled (%s)", exc)


def _start_error_reporting() -> None:
    """Start error reporting when a DSN is configured; otherwise do nothing."""
    if not SENTRY_DSN:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=SENTRY_ENVIRONMENT,
        traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
        send_default_pii=False,
    )
    logger.info(
        "error reporting on (environment=%s, traces_sample_rate=%.2f)",
        SENTRY_ENVIRONMENT,
        SENTRY_TRACES_SAMPLE_RATE,
    )


def default_engine_factory():
    """Build the engine this process will serve from.

    Imported inside the function because constructing the embedder loads a
    model, and importing this module — to read its routes, to build an app for
    a test — must not.
    """
    from ..analysis import Extractor, Similarity
    from ..common.judge import IntentJudge
    from ..engine import Engine

    embedder = Similarity()
    extractor = Extractor()
    try:
        judge = IntentJudge()
    except Exception:
        judge = None
    return Engine(STORE_PATH, embedder=embedder, extractor=extractor, judge=judge)


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

    A miss is **service-unavailable, not not-found**. The tenant may well
    exist and be perfectly valid; this process simply does not hold its graph.
    """
    engine = REGISTRY.get(org_id)
    if engine is None and not auth_module.MULTI_TENANCY_ENABLED:
        # Single-tenant operation: one store, opened at startup from the
        # configured path, serving whatever identifier arrives.
        engine = REGISTRY.get(DEFAULT_TENANT_ORG_ID)

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Tenant graph not loaded on this pod: {org_id}",
        )
    return engine


def _loaded_graph_id(org_id: str) -> str | None:
    """The file identity of the graph that currently answers for ``org_id``."""
    entry = REGISTRY.entry(org_id)
    if entry is None and not auth_module.MULTI_TENANCY_ENABLED:
        entry = REGISTRY.entry(DEFAULT_TENANT_ORG_ID)
    return Path(entry.path).name if entry is not None else None


def _same_file(left, right) -> bool:
    """Whether two paths name the same file, comparing resolved forms."""
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left).absolute() == Path(right).absolute()


def general_client():
    """The general judge client and its model, built once.

    Raises when the judge is not configured; each route turns that into its
    own failure answer, and a timeout is one more such failure. Bounded, so a
    provider that stops responding cannot hold a request open indefinitely.
    """
    from ..common.config import ANSWER_TIMEOUT_SECONDS, JUDGE_MODEL
    from ..common.judge import _required, chat_client

    if "client" not in _llm:
        model = _required(JUDGE_MODEL, "GRAPHRAG_JUDGE_MODEL")
        client = chat_client().with_options(timeout=ANSWER_TIMEOUT_SECONDS)
        _llm["client"] = (client, model)
    return _llm["client"]


def _summary_key(key: str | None, text: str) -> str:
    """The cache key for a summary: the caller's key and a digest of the text.

    The caller's key alone is not enough. It is usually a node id, one cache
    serves every tenant in the process, and tenant graphs share ids — two
    organisations that both mention the same service have the same entity id.
    Keyed by id alone, one tenant's summary of its own text would be served
    to another. With the text in the key, a hit can only return a summary of
    the text the caller sent.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{key or ''}#{digest}"


def _answer_key(query: str, context: str) -> str:
    return f"{query}\n#{hash(context)}"


def _answer_prompt(query: str, context: str) -> str:
    """The grounded-answer prompt the blocking and streaming routes share."""
    return (
        "Answer a colleague's question about a software "
        "knowledge graph using ONLY the context below. Reply in two or three "
        "plain sentences that a non-expert can follow, naming the exact "
        "PRs, people and components involved. If the context lacks the "
        "answer, say so plainly instead of guessing.\n\n"
        f"Question: {query}\n\nContext:\n{context}"
    )


def suggestions_from(hubs, limit: int) -> list[dict]:
    """Example questions built from the busiest entities.

    Variety comes first: the first entity of each type, in the order the hubs
    arrive, busiest first. Only when every type has had its turn do further
    entities fill the remaining places, and a question already on the list is
    not asked twice.
    """
    candidates = []
    for hub in hubs:
        label = (hub.get("label") or "").strip()
        if label:
            kind = hub.get("type") or ""
            question = SUGGESTION_TEMPLATES.get(kind, SUGGESTION_DEFAULT).format(label=label)
            candidates.append({"query": question, "entity": label, "type": kind})

    first_of_type: dict[str, dict] = {}
    for candidate in candidates:
        first_of_type.setdefault(candidate["type"], candidate)
    # At least one pick even for a non-positive limit, as the list has always
    # been read as "some examples" rather than validated here.
    picked = list(first_of_type.values())[:max(limit, 1)]

    asked = {candidate["query"] for candidate in picked}
    for candidate in candidates:
        if len(picked) >= limit:
            break
        if candidate["query"] not in asked:
            picked.append(candidate)
            asked.add(candidate["query"])
    return picked


def _complete(prompt: str, max_tokens: int) -> str:
    """One deterministic completion from the general model, trimmed."""
    client, model = general_client()
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return (reply.choices[0].message.content or "").strip()


def _stream_completion(prompt: str, max_tokens: int):
    """The general model's completion, yielded in the pieces it arrives in."""
    client, model = general_client()
    chunks = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in chunks:
        try:
            piece = chunk.choices[0].delta.content or ""
        except (AttributeError, IndexError):
            piece = ""
        if piece:
            yield piece


def _cached_completion(
    cache, key: str, prompt: str | None, field: str, *, max_tokens: int, without_prompt: str = ""
) -> dict:
    """``{field: text, "cached": ...}`` from ``cache`` or, failing that, the model.

    A ``prompt`` of ``None`` means there is nothing to ask, and the reply is
    ``without_prompt``. A failed call is reported in the body under ``error``
    rather than raised, and nothing is remembered for it.
    """
    remembered = cache.get(key)
    if remembered is not None:
        return {field: remembered, "cached": True}
    if prompt is None:
        return {field: without_prompt, "cached": False}
    try:
        text = _complete(prompt, max_tokens)
    except Exception as exc:  # noqa: BLE001 - the caller reads the error field
        logger.warning("the model gave no %s: %s", field, exc)
        return {field: "", "cached": False, "error": str(exc)}
    cache.set(key, text)
    return {field: text, "cached": False}


def _prepared_answer(req) -> tuple[str, str | None, str | None]:
    """Where an answer to ``req`` will come from.

    ``(cache key, remembered answer or None, prompt or None)``; the prompt is
    None when there is no context to ground an answer in.
    """
    key = _answer_key(req.query, req.context)
    remembered = ANSWER_CACHE.get(key)
    context = (req.context or "").strip()[:ANSWER_CONTEXT_MAX_CHARS]
    return key, remembered, (_answer_prompt(req.query, context) if context else None)


# --------------------------------------------------------------------------
# The model routes. Declared once, at import, because the rate limit is
# registered when the route is decorated and a route declared per application
# would register it again for every application built in the process.
# --------------------------------------------------------------------------

model_router = APIRouter(prefix="/api")


@model_router.post(
    "/summarize",
    response_model=SummaryRead,
    response_model_exclude_none=True,
)
@limiter.limit(LLM_RATE_LIMIT)
def summarize(
    request: Request,
    req: SummarizeRequest,
    _user: str = Depends(get_current_user),
) -> dict:
    """A one-sentence summary of a note, remembered per key and text."""
    text = (req.text or "").strip()[:SUMMARY_TEXT_MAX_CHARS]
    if not text:
        return {"summary": "", "cached": False}
    return _cached_completion(
        SUMMARY_CACHE, _summary_key(req.key, text), SUMMARY_PROMPT + text, "summary",
        max_tokens=SUMMARY_MAX_TOKENS,
    )


@model_router.post(
    "/answer",
    response_model=AnswerRead,
    response_model_exclude_none=True,
)
@limiter.limit(LLM_RATE_LIMIT)
def answer(
    request: Request,
    req: AnswerRequest,
    _user: str = Depends(get_current_user),
) -> dict:
    """A short plain-language answer drawn only from the supplied context."""
    context = (req.context or "").strip()[:ANSWER_CONTEXT_MAX_CHARS]
    return _cached_completion(
        ANSWER_CACHE,
        _answer_key(req.query, req.context),
        _answer_prompt(req.query, context) if context else None,
        "answer",
        max_tokens=ANSWER_MAX_TOKENS,
        without_prompt=NO_CONTEXT_ANSWER,
    )


@model_router.post("/answer/stream")
@limiter.limit(LLM_RATE_LIMIT)
def answer_stream(
    request: Request,
    req: AnswerRequest,
    _user: str = Depends(get_current_user),
) -> StreamingResponse:
    """The answer as plain text, sent piece by piece while the model writes it."""

    def pieces():
        # Looked up when the stream starts, not when the route returns.
        key, remembered, prompt = _prepared_answer(req)
        if remembered is not None or prompt is None:
            yield NO_CONTEXT_ANSWER if remembered is None else remembered
            return
        sent: list[str] = []
        try:
            for piece in _stream_completion(prompt, ANSWER_MAX_TOKENS):
                sent.append(piece)
                yield piece
        except Exception as exc:  # noqa: BLE001 - the response has already begun
            logger.warning("the answer stream broke off: %s", exc)
            if not sent:
                yield STREAM_FAILED_ANSWER
            return
        whole = "".join(sent).strip()
        if whole:
            ANSWER_CACHE.set(key, whole)

    return StreamingResponse(
        pieces(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def create_app(*, engine_factory=None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level instance so a test can build one per
    configuration, and ``engine_factory`` is injectable for the same reason —
    the default one loads a model and opens a database, neither of which a
    test of routing should do.
    """
    _start_error_reporting()
    build_engine = engine_factory or default_engine_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = build_engine()
        # The process's registry, not a new one. Routes, the agent and this
        # startup all reach the same object; two of them would be two sets
        # of graphs, one of which nothing serves from.
        registry = REGISTRY
        # Opening a second graph must not load a second copy of the models.
        registry.set_loader(graph_loader(engine))

        # The `with` is the guarantee for a startup that fails before the
        # store reaches the registry. Past that point the registry owns the
        # close, and Engine.close is idempotent, so the two do not conflict.
        with engine:
            registry.attach(
                DEFAULT_TENANT_ORG_ID,
                engine,
                path=str(engine.path),
                version=LOCAL_DEFAULT_VERSION,
            )
            app.state.engine = engine
            app.state.active_path = Path(engine.path)
            app.state.pod_agent = None
            app.state.pod_agent_stop = None

            try:
                await asyncio.to_thread(engine.warm)
            except Exception:  # noqa: BLE001 - cold is slower, not fatal
                logger.warning("retrieval warm-up failed; continuing cold", exc_info=True)

            _bring_up_history()
            _bring_up_the_control_plane()

            agent = None
            stop = None
            if pod_agent_enabled():
                # Imported here and nowhere else, so a process running
                # without the agent never loads any of it.
                from ..pod import boot, poll

                try:
                    hydrated = await asyncio.to_thread(boot, registry=registry)
                    logger.info("hydrated %d tenant(s) at startup", len(hydrated))
                except Exception:  # noqa: BLE001 - serving beats registering
                    logger.warning(
                        "the pod did not boot cleanly; continuing", exc_info=True
                    )

                stop = asyncio.Event()
                agent = asyncio.create_task(poll(registry=registry, stop=stop))
                app.state.pod_agent = agent
                app.state.pod_agent_stop = stop
                logger.info("pod agent running as %s", POD_ID)

            try:
                yield
            finally:
                # The agent stops **before** the stores close. A tick in
                # flight is holding handles and may be part way through
                # swapping one; closing underneath it is a use-after-close.
                if agent is not None:
                    try:
                        stop.set()
                        await agent
                    except Exception:  # noqa: BLE001 - shutdown continues
                        logger.warning(
                            "the pod agent did not stop cleanly", exc_info=True
                        )

                app.state.pod_agent = None
                app.state.pod_agent_stop = None
                app.state.engine = None
                app.state.active_path = None
                registry.close_all()
                from ..retrieval.router import shutdown_embed_executor

                shutdown_embed_executor()

    docs = {} if API_DOCS_ENABLED else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="graphRAG API", version="0.1.0", lifespan=lifespan, **docs)

    # Per-user limits on the model routes: the limiter lives on the
    # application state, and an exceeded limit becomes a 429.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(TenantRoutingMiddleware)
    # Added last, so it is the outermost layer: a refusal from the routing gate
    # still carries the headers a browser needs to read it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post(
        "/api/trace",
        response_model=TraceResponseRead,
        response_model_exclude_unset=True,
    )
    async def trace(
        req: TraceRequest,
        request: Request,
        user_id: str = Depends(get_current_user),
        org_id: str = Depends(get_current_tenant_org),
        _limited: None = Depends(trace_rate_guard),
    ) -> dict:
        """Run one query and return the ranked nodes, the trace and the context.

        A caller who names a session has the run recorded there, once the
        session is confirmed to be theirs; one that is missing or somebody
        else's is the same single refusal.
        """
        from .. import history as history_module

        engine = engine_for(request, org_id)
        top_k = min(max(req.top_k or TOP_K_VECTOR, 1), TRACE_MAX_TOP_K)
        response = await engine.route_async(req.query, top_k)

        payload = asdict(response)
        for node, result in zip(response.results, payload["results"]):
            result["page_content"] = format_page_content(node)
        payload["context"] = engine.router.build_context(response.results)

        if req.session_id:
            owner = await asyncio.to_thread(history_module.session_owner, req.session_id)
            if owner is None or owner != user_id:
                logger.warning("session %s refused for %s", req.session_id, user_id)
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="no such session",
                )
            payload["trace_id"] = await asyncio.to_thread(
                history_module.persist_trace,
                session_id=req.session_id,
                query=req.query,
                execution_plan=payload["trace_log"],
                graph_payload=payload["results"],
                graph_id=_loaded_graph_id(org_id),
            )
        return payload

    @app.post("/api/subgraph", response_model=SubgraphRead)
    async def subgraph(
        req: SubgraphRequest,
        request: Request,
        org_id: str = Depends(get_current_tenant_org),
    ) -> dict:
        """The requested nodes, their one-hop neighbours, and the edges between."""
        engine = engine_for(request, org_id)
        return await asyncio.to_thread(engine.store.subgraph, req.node_ids)

    @app.get("/api/suggestions", response_model=SuggestionsRead)
    async def suggestions(
        request: Request,
        limit: int = 5,
        org_id: str = Depends(get_current_tenant_org),
    ) -> dict:
        """Example questions built from this graph's most connected entities.

        A store that cannot list them suggests nothing rather than failing: a
        graph too small or too new to suggest anything is an ordinary state.
        """
        engine = engine_for(request, org_id)
        try:
            hubs = await asyncio.to_thread(engine.store.top_entities, limit * 4)
        except Exception as exc:  # noqa: BLE001 - an empty list, not an error
            logger.warning("top_entities failed: %s", exc)
            return {"suggestions": []}
        return {"suggestions": suggestions_from(hubs, limit)}

    # ----------------------------------------------------------------------
    # The local single-store workflow: list the graph files in this checkout
    # and change which one this process serves. They operate on the default
    # tenant only and are deliberately unauthenticated — a local developer
    # affordance, recorded here as a choice rather than an oversight.
    #
    # With tenancy on, the local files are not what anyone is served, so the
    # list is the caller's own graph and nothing else, and switching is off:
    # it would swap a file from this checkout in under the default tenant for
    # anyone who asked, with no credential.
    # ----------------------------------------------------------------------

    @app.get("/api/graphs", response_model=GraphsRead)
    def graphs(
        request: Request,
        org_id: str = Depends(get_current_tenant_org),
    ) -> dict:
        """Every graph this checkout can serve, and which one is active.

        With tenancy on: the caller's organisation's graph, if this pod holds
        it, as the one active entry; an empty list if it does not.
        """
        if auth_module.MULTI_TENANCY_ENABLED:
            entry = REGISTRY.entry(org_id)
            if entry is None:
                return {"graphs": [], "active": None}
            graph_id = Path(entry.path).name
            label = f"Version {entry.version}" if entry.version else graph_id
            return {
                "graphs": [{"id": graph_id, "label": label, "active": True}],
                "active": graph_id,
            }

        active_path = getattr(request.app.state, "active_path", None)
        listed = [
            {
                "id": path.name,
                "label": graph_label(path),
                "active": active_path is not None and _same_file(path, active_path),
            }
            for path in graph_paths()
        ]
        active = Path(active_path).name if active_path is not None else None
        return {"graphs": listed, "active": active}

    @app.post("/api/graphs/switch", response_model=GraphSwitchRead)
    def switch(req: SwitchRequest, request: Request) -> dict:
        """Serve a different discovered graph from now on.

        The id must be one discovery found; anything else is a 404 rather than
        an attempt to open whatever was sent. The new store opens through the
        registry loader, which keeps the loaded models, and re-binds the
        default tenant in the same call. Summaries are dropped, because they
        describe the previous graph.
        """
        if auth_module.MULTI_TENANCY_ENABLED:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="switching graphs is only available with tenancy off",
            )

        registry = REGISTRY

        match = next((path for path in graph_paths() if path.name == req.id), None)
        if match is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown graph: {req.id}",
            )

        displaced = registry.replace(
            DEFAULT_TENANT_ORG_ID, path=str(match), version=LOCAL_DEFAULT_VERSION
        )
        engine = registry.get(DEFAULT_TENANT_ORG_ID)
        app.state.engine = engine
        app.state.active_path = match
        SUMMARY_CACHE.clear()

        if displaced is not None and displaced.handle is not engine:
            try:
                registry.close_entries([displaced])
            except Exception as exc:  # noqa: BLE001 - the new graph is already live
                logger.warning("the graph switched away from did not close: %s", exc)

        logger.info("now serving the graph at %s", match)
        return {
            "active": match.name,
            "label": graph_label(match),
            "nodes": engine.store.count_nodes(),
        }

    @app.get("/api/health", response_model=HealthRead)
    def health(request: Request) -> dict:
        """Unauthenticated on purpose: a readiness probe that needs a
        credential cannot report that credentials are misconfigured.

        With tenancy on, the nodes are those of every tenant graph this pod
        holds; the local store it opened at startup serves nobody, and
        counting it would report an empty pod as the healthy one.
        """
        if auth_module.MULTI_TENANCY_ENABLED:
            nodes = 0
            for key in REGISTRY.keys():
                engine = REGISTRY.get(key) if key != DEFAULT_TENANT_ORG_ID else None
                if engine is not None:
                    nodes += engine.store.count_nodes()
            return {"status": "ok", "nodes": nodes}
        engine = REGISTRY.get(DEFAULT_TENANT_ORG_ID)
        return {"status": "ok", "nodes": engine.store.count_nodes()}

    app.include_router(model_router)
    app.include_router(history_router)
    app.include_router(github_oauth_router)
    app.include_router(onboarding_router)
    app.include_router(webhooks_router)

    return app
