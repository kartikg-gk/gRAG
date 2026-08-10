"""The trace schema: what one query did, as plain data.

Everything else in this package produces or consumes this shape, so it stays
small and stdlib-only — no pydantic, no httpx, nothing from the rest of the
pipeline.

Schema versions
---------------

**3** — retrieved items and edges nest under a ``Retrieval``, one per retriever
call, each carrying its own query, span, and arm. A run with two retrievers was
previously flattened into one undifferentiated list, which lost which retriever
answered what. ``TraceItem`` gained ``label``, ``kind``, ``source_uri`` and
``metadata``. ``Trace.items`` and ``Trace.edges`` survive as read-only views
across every retrieval, so consumers that do not care about the grouping did
not have to change.

**2** — ``used`` removed from items; only the ``overlap`` measurement is
stored. The verdict is threshold-dependent and the measurement is not, so
baking a boolean in freezes an archived trace at whatever cutoff happened to be
current when it was written. Added ``vector_score``/``graph_score``, ``arm``,
``producer`` and ``spans``. Edges renamed ``type``/``confidence`` to
``relation``/``weight``.

**1** — initial: query, answer, timing, items with ``used``, edges.

Versions 1 and 2 still load: their flat items and edges are wrapped into a
single ``Retrieval``. A version 1 ``used`` is dropped, because the threshold
that produced it was never recorded and cannot be recovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SCHEMA_VERSION = 3

# Which retrieval arm produced a result set.
ARM_VECTOR = "vector"
ARM_GRAPH = "graph"
ARM_HYBRID = "hybrid"
ARM_UNKNOWN = "unknown"
ARMS = (ARM_VECTOR, ARM_GRAPH, ARM_HYBRID, ARM_UNKNOWN)

# What produced the trace. A data field with a default rather than an
# assumption baked into the code, so a second producer does not have to lie.
PRODUCER_UNKNOWN = "unknown"

# Span lifecycle.
STATUS_RUNNING = "running"
STATUS_OK = "ok"
STATUS_ERROR = "error"

#: Default kind for a retrieved item whose payload does not say what it is.
KIND_DOCUMENT = "document"


@dataclass
class TraceItem:
    """One retrieved item and how much of the answer it accounts for.

    ``overlap`` is a measurement, not a verdict. Whether it counts as used is
    decided at read time by applying a threshold — see ``classify.is_used``.

    ``vector_score`` and ``graph_score`` are the per-arm contributions to a
    fused ``score``.
    """

    id: str
    content: str
    source: str
    label: str | None = None
    kind: str = KIND_DOCUMENT
    source_uri: str | None = None
    score: float | None = None
    vector_score: float | None = None
    graph_score: float | None = None
    overlap: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEdge:
    """A relation between two retrieved items, when retrieval knows of one."""

    source: str
    target: str
    relation: str
    weight: float | None = None


@dataclass
class Retrieval:
    """One retriever call: what was asked, what came back, and from which arm.

    Grouping matters once more than one retriever runs. A flat list cannot say
    that the graph arm returned these four and the vector arm those three, and
    that is exactly the comparison a fused pipeline needs to show.
    """

    query: str = ""
    span_id: str | None = None
    arm: str = ARM_UNKNOWN
    items: list[TraceItem] = field(default_factory=list)
    edges: list[TraceEdge] = field(default_factory=list)


@dataclass
class Span:
    """One unit of work.

    ``parent_id`` is what makes a trace a tree rather than a list — without it
    there is no way to say which retrieval belonged to which step.

    ``start_ms`` and ``end_ms`` are milliseconds from the start of the run, so
    a span's position is meaningful without knowing the wall clock.
    """

    id: str
    name: str
    kind: str
    parent_id: str | None = None
    start_ms: float | None = None
    end_ms: float | None = None
    status: str = STATUS_RUNNING


@dataclass
class Trace:
    """One query, start to finish."""

    query: str
    answer: str | None = None
    producer: str = PRODUCER_UNKNOWN
    started_at: datetime | None = None
    duration_ms: float | None = None
    retrievals: list[Retrieval] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)

    # -- views across every retrieval --------------------------------------
    #
    # Read-only on purpose. A consumer that wants every item regardless of
    # which retriever produced it reads these; a consumer that cares about the
    # split reads ``retrievals``. Appending to a view would silently vanish,
    # so these build a new list each time rather than exposing internals.

    @property
    def items(self) -> list[TraceItem]:
        """Every retrieved item, in retrieval order."""
        return [item for retrieval in self.retrievals for item in retrieval.items]

    @property
    def edges(self) -> list[TraceEdge]:
        """Every relation, in retrieval order."""
        return [edge for retrieval in self.retrievals for edge in retrieval.edges]

    @property
    def arm(self) -> str:
        """The arm this trace came from, across all retrievals.

        One arm throughout reports that arm; a mix reports ``hybrid``, which is
        what a mix actually is.
        """
        arms = {retrieval.arm for retrieval in self.retrievals}
        if not arms:
            return ARM_UNKNOWN
        if len(arms) == 1:
            return arms.pop()
        return ARM_HYBRID


def _isoformat(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _item_to_dict(item: TraceItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "label": item.label,
        "kind": item.kind,
        "content": item.content,
        "source": item.source,
        "source_uri": item.source_uri,
        "score": item.score,
        "vector_score": item.vector_score,
        "graph_score": item.graph_score,
        "overlap": item.overlap,
        "metadata": item.metadata,
    }


def _item_from_dict(payload: dict[str, Any]) -> TraceItem:
    return TraceItem(
        id=payload["id"],
        content=payload["content"],
        source=payload["source"],
        label=payload.get("label"),
        kind=payload.get("kind", KIND_DOCUMENT),
        source_uri=payload.get("source_uri"),
        score=payload.get("score"),
        vector_score=payload.get("vector_score"),
        graph_score=payload.get("graph_score"),
        # A version 1 "used" is dropped: the threshold that produced it was
        # never recorded, so the verdict cannot be trusted or reproduced. The
        # overlap it was derived from survives.
        overlap=payload.get("overlap"),
        metadata=payload.get("metadata") or {},
    )


def _edge_to_dict(edge: TraceEdge) -> dict[str, Any]:
    return {
        "source": edge.source,
        "target": edge.target,
        "relation": edge.relation,
        "weight": edge.weight,
    }


def _edge_from_dict(payload: dict[str, Any]) -> TraceEdge:
    return TraceEdge(
        source=payload["source"],
        target=payload["target"],
        # Version 1 called these "type" and "confidence".
        relation=payload.get("relation") or payload["type"],
        weight=payload.get("weight", payload.get("confidence")),
    )


def to_dict(trace: Trace) -> dict[str, Any]:
    """Turn a trace into plain JSON-serializable data."""
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": trace.producer,
        "query": trace.query,
        "answer": trace.answer,
        "started_at": _isoformat(trace.started_at),
        "duration_ms": trace.duration_ms,
        "retrievals": [
            {
                "query": retrieval.query,
                "span_id": retrieval.span_id,
                "arm": retrieval.arm,
                "items": [_item_to_dict(item) for item in retrieval.items],
                "edges": [_edge_to_dict(edge) for edge in retrieval.edges],
            }
            for retrieval in trace.retrievals
        ],
        "spans": [
            {
                "id": span.id,
                "name": span.name,
                "kind": span.kind,
                "parent_id": span.parent_id,
                "start_ms": span.start_ms,
                "end_ms": span.end_ms,
                "status": span.status,
            }
            for span in trace.spans
        ],
    }


def trace_from_dict(payload: dict[str, Any]) -> Trace:
    """Rebuild a trace from plain data, including versions 1 and 2."""
    version = payload.get("schema_version", SCHEMA_VERSION)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"trace has schema_version {version}, but this module understands "
            f"up to {SCHEMA_VERSION}"
        )

    if "retrievals" in payload:
        retrievals = [
            Retrieval(
                query=entry.get("query", ""),
                span_id=entry.get("span_id"),
                arm=entry.get("arm", ARM_UNKNOWN),
                items=[_item_from_dict(item) for item in entry.get("items", [])],
                edges=[_edge_from_dict(edge) for edge in entry.get("edges", [])],
            )
            for entry in payload["retrievals"]
        ]
    else:
        # Versions 1 and 2 stored one flat list. Wrapping it in a single
        # retrieval loses nothing: a flat trace never said which retriever
        # produced what, so there was only ever one group to recover.
        items = [_item_from_dict(item) for item in payload.get("items", [])]
        edges = [_edge_from_dict(edge) for edge in payload.get("edges", [])]
        retrievals = (
            [
                Retrieval(
                    query=payload.get("query", ""),
                    arm=payload.get("arm", ARM_UNKNOWN),
                    items=items,
                    edges=edges,
                )
            ]
            if items or edges
            else []
        )

    return Trace(
        query=payload["query"],
        answer=payload.get("answer"),
        producer=payload.get("producer", PRODUCER_UNKNOWN),
        started_at=_parse_time(payload.get("started_at")),
        duration_ms=payload.get("duration_ms"),
        retrievals=retrievals,
        spans=[
            Span(
                id=span["id"],
                name=span["name"],
                kind=span["kind"],
                parent_id=span.get("parent_id"),
                start_ms=span.get("start_ms"),
                end_ms=span.get("end_ms"),
                status=span.get("status", STATUS_RUNNING),
            )
            for span in payload.get("spans", [])
        ],
    )
