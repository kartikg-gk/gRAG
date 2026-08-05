"""The trace schema: what one query did, as plain data.

Everything else in this package produces or consumes this shape, so it stays
small and stdlib-only — no pydantic, no httpx, nothing from the rest of the
pipeline.

Schema versions
---------------

**2** — ``used`` removed from items; only the ``overlap`` measurement is
stored. The verdict is threshold-dependent and the measurement is not, so
baking a boolean in freezes an archived trace at whatever cutoff happened to be
current when it was written. With the float stored, changing the threshold
re-classifies every trace ever recorded. Added ``vector_score`` and
``graph_score`` per item, ``arm`` and ``producer`` on the trace, and ``spans``.
Trace edges renamed ``type``/``confidence`` to ``relation``/``weight``.

**1** — initial: query, answer, timing, items with ``used``, edges.

Version 1 files still load. ``used`` is dropped on read, because the threshold
that produced it is not recorded and cannot be recovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SCHEMA_VERSION = 2

# Which retrieval arm produced a result set. Fusion does not exist yet, so
# nothing sets anything but the default — the field exists so that adding
# fusion later is not a breaking format change.
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


@dataclass
class TraceItem:
    """One retrieved item and how much of the answer it accounts for.

    ``overlap`` is a measurement, not a verdict. Whether it counts as used is
    decided at read time by applying a threshold — see ``classify.is_used``.

    ``vector_score`` and ``graph_score`` are the per-arm contributions to a
    fused ``score``. Nothing populates them yet.
    """

    id: str
    content: str
    source: str
    score: float | None = None
    vector_score: float | None = None
    graph_score: float | None = None
    overlap: float | None = None


@dataclass
class TraceEdge:
    """A relation between two retrieved items, when retrieval knows of one."""

    source: str
    target: str
    relation: str
    weight: float | None = None


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
    arm: str = ARM_UNKNOWN
    started_at: datetime | None = None
    duration_ms: float | None = None
    items: list[TraceItem] = field(default_factory=list)
    edges: list[TraceEdge] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)


def _isoformat(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def to_dict(trace: Trace) -> dict[str, Any]:
    """Turn a trace into plain JSON-serializable data."""
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": trace.producer,
        "arm": trace.arm,
        "query": trace.query,
        "answer": trace.answer,
        "started_at": _isoformat(trace.started_at),
        "duration_ms": trace.duration_ms,
        "items": [
            {
                "id": item.id,
                "content": item.content,
                "source": item.source,
                "score": item.score,
                "vector_score": item.vector_score,
                "graph_score": item.graph_score,
                "overlap": item.overlap,
            }
            for item in trace.items
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "weight": edge.weight,
            }
            for edge in trace.edges
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
    """Rebuild a trace from plain data, including version 1 files."""
    version = payload.get("schema_version", SCHEMA_VERSION)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"trace has schema_version {version}, but this module understands "
            f"up to {SCHEMA_VERSION}"
        )

    return Trace(
        query=payload["query"],
        answer=payload.get("answer"),
        producer=payload.get("producer", PRODUCER_UNKNOWN),
        arm=payload.get("arm", ARM_UNKNOWN),
        started_at=_parse_time(payload.get("started_at")),
        duration_ms=payload.get("duration_ms"),
        items=[
            TraceItem(
                id=item["id"],
                content=item["content"],
                source=item["source"],
                score=item.get("score"),
                vector_score=item.get("vector_score"),
                graph_score=item.get("graph_score"),
                # A version 1 "used" is dropped: the threshold that produced it
                # was never recorded, so the verdict cannot be trusted or
                # reproduced. The overlap it was derived from survives.
                overlap=item.get("overlap"),
            )
            for item in payload.get("items", [])
        ],
        edges=[
            TraceEdge(
                source=edge["source"],
                target=edge["target"],
                # Version 1 called these "type" and "confidence".
                relation=edge.get("relation") or edge["type"],
                weight=edge.get("weight", edge.get("confidence")),
            )
            for edge in payload.get("edges", [])
        ],
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
