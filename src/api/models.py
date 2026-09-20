"""The request and response contract.

Declared types rather than free-form dictionaries, because a request shape
that is only whatever a handler happened to read is not a contract — it is a
description of today's code, and a consumer discovers a change by breaking.

Pydantic is already a runtime dependency of this project, used to validate
payloads at the ingestion boundary. This is the same job at the other
boundary, so it adds nothing to the dependency list.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """A wire contract that rejects accidental response fields."""

    model_config = ConfigDict(extra="forbid")


class TraceRequest(BaseModel):
    """A question to trace, and optionally the session to record it in."""

    query: str
    top_k: int | None = None
    session_id: str | None = None


class SubgraphRequest(BaseModel):
    """The nodes to expand by one hop."""

    node_ids: list[str]


class SummarizeRequest(BaseModel):
    """A note to summarise, and the key its summary is cached under."""

    key: str
    text: str


class AnswerRequest(BaseModel):
    """A question and the context an answer must be grounded in."""

    query: str
    context: str


class SwitchRequest(BaseModel):
    """Which discovered graph to serve, by its file name.

    A name, never a path. A path would make this an instruction to open an
    arbitrary file; a name can only select something discovery already found.
    """

    id: str


class SessionUpsert(BaseModel):
    """Start a session, or retitle one the caller owns.

    ``user_id`` is accepted and ignored: who is asking comes from the session
    token alone.
    """

    user_id: str | None = None
    email: str | None = None
    title: str = "New Chat"
    session_id: str | None = None


class SessionRead(ContractModel):
    """One session, as its owner sees it."""

    id: str
    user_id: str
    title: str
    created_at: datetime


class TraceRead(ContractModel):
    """One recorded question in a session and what answering it involved."""

    id: str
    session_id: str
    query: str
    execution_plan: dict
    graph_payload: dict | list
    created_at: datetime
    graph_id: str | None = None


class ResultDocumentRead(ContractModel):
    doc_id: str | None
    content: str | None
    path: str | None


class ResultRead(ContractModel):
    id: str
    label: str | None
    type: str | None
    score_total: float
    score_vector: float
    score_graph: float
    recency: float
    age_days: float | None
    documents: list[ResultDocumentRead]
    page_content: str


class IntentTraceRead(ContractModel):
    alpha: float
    beta: float
    type: str


class GraphHopRead(ContractModel):
    from_id: str
    to_id: str
    confidence: float
    relation: str


class ExecutionPathRead(ContractModel):
    linked_seeds: list[str]
    vector_seeds: list[str]
    graph_hops: list[GraphHopRead]


class AppliedRecencyRead(ContractModel):
    id: str
    age_days: float
    factor: float


class RecencyTraceRead(ContractModel):
    enabled: bool
    floor: float
    applied: list[AppliedRecencyRead]


class RetrievalMetricsRead(ContractModel):
    graph_hits: int
    vector_k: int
    total_nodes_evaluated: int


class RetrievalTraceRead(ContractModel):
    intent: IntentTraceRead
    execution_path: ExecutionPathRead
    recency: RecencyTraceRead
    metrics: RetrievalMetricsRead


class TraceResponseRead(ContractModel):
    query: str
    results: list[ResultRead]
    trace_log: RetrievalTraceRead
    context: str
    trace_id: str | None = None


class SubgraphNodeRead(ContractModel):
    id: str
    label: str | None
    type: str | None
    requested: bool


class SubgraphEdgeRead(ContractModel):
    source: str
    target: str
    confidence: float
    relation: str


class SubgraphRead(ContractModel):
    nodes: list[SubgraphNodeRead]
    edges: list[SubgraphEdgeRead]


class SuggestionRead(ContractModel):
    query: str
    entity: str
    type: str


class SuggestionsRead(ContractModel):
    suggestions: list[SuggestionRead]


class GraphRead(ContractModel):
    id: str
    label: str
    active: bool


class GraphsRead(ContractModel):
    graphs: list[GraphRead]
    active: str | None


class GraphSwitchRead(ContractModel):
    active: str
    label: str
    nodes: int


class HealthRead(ContractModel):
    status: str
    nodes: int


class SummaryRead(ContractModel):
    summary: str
    cached: bool
    error: str | None = None


class AnswerRead(ContractModel):
    answer: str
    cached: bool
    error: str | None = None


FRONTEND_RESPONSE_MODELS = {
    "trace": TraceResponseRead,
    "subgraph": SubgraphRead,
    "suggestions": SuggestionsRead,
    "graphs": GraphsRead,
    "graph_switch": GraphSwitchRead,
    "health": HealthRead,
    "summary": SummaryRead,
    "answer": AnswerRead,
    "session": SessionRead,
    "session_trace": TraceRead,
}
