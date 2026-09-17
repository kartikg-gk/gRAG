"""The request and response contract.

Declared types rather than free-form dictionaries, because a response shape
that is only whatever a handler happened to build is not a contract — it is a
description of today's code, and a consumer discovers a change by breaking.
Declaring it also means the schema is published, so a client can see that an
arm reports zero rather than omitting itself without reading this project's
source.

Pydantic is already a runtime dependency of this project, used to validate
payloads at the ingestion boundary. This is the same job at the other
boundary, so it adds nothing to the dependency list.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """What a caller asks for.

    ``k`` is optional and bounded. Unbounded, one request could ask for the
    whole graph and the cost of answering would be set by the caller rather
    than by the service.
    """

    query: str = Field(min_length=1, description="The question, as text.")
    k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description=(
            "How many ranked results to return. Unset uses the engine's own "
            "default, so the number lives in one place rather than two."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "A session to record this query in. Unset, nothing is recorded "
            "and the query is answered exactly as it was before history "
            "existed."
        ),
    )


class Chunk(BaseModel):
    """One piece of source prose, carried once however many results cite it."""

    id: str
    path: str | None = None
    content: str | None = None


class Result(BaseModel):
    """One ranked entity, with the components that produced its position.

    The three score fields are always present. An arm that did not find this
    entity reports ``0.0`` — omitting it would make "no contribution" and "not
    reported" the same thing to a client, and they are different facts.
    """

    id: str
    score: float
    vector_score: float = Field(description="0.0 when the vector arm did not find it.")
    graph_score: float = Field(description="0.0 when traversal did not reach it.")
    decay: float = Field(description="The recency multiplier applied to the blend.")
    node_type: str | None = None
    age_days: float | None = Field(
        default=None,
        description="None means no timestamp, which is not the same as new.",
    )
    found_by_both: bool = False
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunks this entity was found in. Text lives in `chunks`.",
    )


class QueryResponse(BaseModel):
    """A ranked set, the prose behind it, and how it was produced."""

    query: str
    intent: str | None = None
    alpha: float | None = Field(
        default=None, description="Weight the vector arm carried for this intent."
    )
    beta: float | None = Field(
        default=None, description="Weight the graph arm carried for this intent."
    )
    results: list[Result] = Field(default_factory=list)
    chunks: list[Chunk] = Field(
        default_factory=list,
        description="Deduplicated: one entry per chunk, whatever cites it.",
    )
    trace: dict[str, Any] | None = Field(
        default=None,
        description="Per-hop record, built from the finished results.",
    )
    seconds: float = 0.0


class Health(BaseModel):
    """Whether the service is up and whether it can answer.

    Two fields, not one. A process that is running with no store open is up
    and cannot serve a query, and a probe that collapsed those into ``ok``
    would report healthy while every request failed.
    """

    status: str
    store_open: bool


class SubgraphRequest(BaseModel):
    """Which nodes to expand around."""

    ids: list[str] = Field(
        default_factory=list,
        description="Node ids. Unknown ones yield nothing rather than an error.",
    )


class SubgraphResponse(BaseModel):
    """The requested nodes, their one-hop neighbours, and the edges between.

    Shaped as the store returns it. A node carries ``requested`` so a caller
    can tell what it asked for from what came back with it — without that, a
    neighbour and a hit look identical.
    """

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class Suggestion(BaseModel):
    """One example question, and what it was built from."""

    question: str
    entity_id: str
    type: str | None = None


class SuggestionsResponse(BaseModel):
    suggestions: list[Suggestion] = Field(default_factory=list)


class GraphSummary(BaseModel):
    """One graph this checkout can serve."""

    id: str
    label: str
    active: bool


class GraphsResponse(BaseModel):
    """Every discovered graph, and which one is being served."""

    graphs: list[GraphSummary] = Field(default_factory=list)
    active: str | None = None


class SwitchRequest(BaseModel):
    """Which discovered graph to serve.

    An id, never a path. A path would make this an instruction to open an
    arbitrary file; an id can only name something discovery already found.
    """

    id: str = Field(min_length=1)


class SwitchResponse(BaseModel):
    """What is being served now."""

    id: str
    label: str
    nodes: int


class SessionRequest(BaseModel):
    """Start a session, or retitle one the caller owns.

    No user field: who is asking comes from the session token alone, so a
    field of that name in the body is ignored rather than read.
    """

    title: str | None = Field(
        default=None,
        description="The session's title. Empty uses the default title.",
    )
    session_id: str | None = Field(
        default=None,
        description="Unset creates a session; set renames that session.",
    )
    email: str | None = Field(
        default=None,
        description="The caller's address, recorded the first time they are seen.",
    )


class SessionSummaryResponse(BaseModel):
    """One session, as its owner sees it."""

    session_id: str
    title: str
    created_at: int


class TraceRecordResponse(BaseModel):
    """One recorded question in a session and what answering it involved."""

    trace_id: str
    session_id: str
    query: str
    plan: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    created_at: int
