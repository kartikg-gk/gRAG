"""Accumulation: build a trace in memory.

This module never learns where a trace will be written. It has no path
argument, no filesystem call, and no import of ``store``. A complete trace can
be built and asserted on in a test without touching the disk — which is the
reason the split exists.

Two ways in:

* ``capture(...)`` — a one-shot for callers that already hold everything.
* ``Recorder`` — accumulates across a run, with spans and failures.

Neither writes a verdict. ``used`` is not a field; the threshold is applied at
read time.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterable, Iterator, Mapping

from .schema import (
    ARM_UNKNOWN,
    PRODUCER_UNKNOWN,
    STATUS_ERROR,
    STATUS_OK,
    Span,
    Trace,
    TraceEdge,
    TraceItem,
)

UNKNOWN_SOURCE = "unknown"

#: Span kinds the recorder expects. Not enforced — a caller may use its own.
KIND_CHAIN = "chain"
KIND_RETRIEVER = "retriever"
KIND_LLM = "llm"
KIND_TOOL = "tool"


def _as_item(item: TraceItem | Mapping[str, Any]) -> TraceItem:
    """Normalize one retrieved item, copying so the caller's data is untouched."""
    if isinstance(item, TraceItem):
        return TraceItem(
            id=item.id,
            content=item.content,
            source=item.source,
            score=item.score,
            vector_score=item.vector_score,
            graph_score=item.graph_score,
            overlap=item.overlap,
        )
    return TraceItem(
        id=item["id"],
        content=item["content"],
        source=item.get("source", UNKNOWN_SOURCE),
        score=item.get("score"),
        vector_score=item.get("vector_score"),
        graph_score=item.get("graph_score"),
    )


def _as_edge(edge: TraceEdge | Mapping[str, Any]) -> TraceEdge:
    """Normalize one relation.

    Accepts ``relation``/``weight`` and the graph builder's own
    ``type``/``confidence``, so a producer does not have to rename fields on the
    way in.
    """
    if isinstance(edge, TraceEdge):
        return TraceEdge(
            source=edge.source,
            target=edge.target,
            relation=edge.relation,
            weight=edge.weight,
        )
    return TraceEdge(
        source=edge["source"],
        target=edge["target"],
        relation=edge.get("relation") or edge["type"],
        weight=edge.get("weight", edge.get("confidence")),
    )


def capture(
    query: str,
    retrieved_items: Iterable[TraceItem | Mapping[str, Any]],
    answer: str | None = None,
    *,
    edges: Iterable[TraceEdge | Mapping[str, Any]] = (),
    spans: Iterable[Span] = (),
    producer: str = PRODUCER_UNKNOWN,
    arm: str = ARM_UNKNOWN,
    started_at: datetime | None = None,
    duration_ms: float | None = None,
) -> Trace:
    """Record one query as a trace.

    Knows nothing about any particular retriever: items arrive as plain
    mappings, so anything producing ``{"id", "content", "source", "score"}``
    plugs in unchanged.

    Timing is passed in rather than measured here, because only the caller
    knows where the work started and stopped.
    """
    return Trace(
        query=query,
        answer=answer,
        producer=producer,
        arm=arm,
        started_at=started_at,
        duration_ms=duration_ms,
        items=[_as_item(item) for item in retrieved_items],
        edges=[_as_edge(edge) for edge in edges],
        spans=list(spans),
    )


class Recorder:
    """Accumulates a trace across a run.

    Spans nest automatically: whatever is open when a new span starts becomes
    its parent, so a retrieval recorded inside a step belongs to that step
    rather than floating at the top level.

    Failed work stays in the trace. A span whose body raises closes with
    ``error`` status and the exception propagates — the run failing is exactly
    what the trace needs to show, and dropping it would make a broken run look
    like a short one.
    """

    def __init__(
        self, *, producer: str = PRODUCER_UNKNOWN, arm: str = ARM_UNKNOWN
    ) -> None:
        self.producer = producer
        self.arm = arm
        self.queries: list[str] = []
        self.answers: list[str] = []
        self.items: list[TraceItem] = []
        self.edges: list[TraceEdge] = []
        self.spans: list[Span] = []

        self.started_at = datetime.now(timezone.utc)
        self._began = perf_counter()
        self._open: list[str] = []
        self._next_id = 0

    # -- elapsed -----------------------------------------------------------

    def _elapsed_ms(self) -> float:
        return round((perf_counter() - self._began) * 1000, 3)

    # -- recording ---------------------------------------------------------

    def record_query(self, query: str) -> None:
        if query:
            self.queries.append(query)

    def record_answer(self, answer: str) -> None:
        if answer:
            self.answers.append(answer)

    def record_items(
        self, items: Iterable[TraceItem | Mapping[str, Any]]
    ) -> None:
        self.items.extend(_as_item(item) for item in items)

    def record_edges(
        self, edges: Iterable[TraceEdge | Mapping[str, Any]]
    ) -> None:
        self.edges.extend(_as_edge(edge) for edge in edges)

    # -- spans -------------------------------------------------------------

    def _new_span_id(self) -> str:
        self._next_id += 1
        return f"s{self._next_id}"

    @contextmanager
    def span(self, name: str, *, kind: str = KIND_CHAIN) -> Iterator[Span]:
        """Record a unit of work, closing it ``ok`` or ``error``."""
        record = Span(
            id=self._new_span_id(),
            name=name,
            kind=kind,
            parent_id=self._open[-1] if self._open else None,
            start_ms=self._elapsed_ms(),
        )
        self.spans.append(record)
        self._open.append(record.id)

        try:
            yield record
        except BaseException:
            record.end_ms = self._elapsed_ms()
            record.status = STATUS_ERROR
            raise
        else:
            record.end_ms = self._elapsed_ms()
            record.status = STATUS_OK
        finally:
            self._open.pop()

    # -- finishing ---------------------------------------------------------

    def finish(self, query: str | None = None, answer: str | None = None) -> Trace:
        """Build the trace.

        Both arguments are optional. ``query`` falls back to the first one
        recorded — the earliest is the question actually asked, before any
        rewriting. ``answer`` falls back to the last text generated, since a
        later generation supersedes an earlier one. A caller may override
        either but must never be required to supply them.
        """
        return capture(
            query if query is not None else (self.queries[0] if self.queries else ""),
            self.items,
            answer if answer is not None else (self.answers[-1] if self.answers else None),
            edges=self.edges,
            spans=self.spans,
            producer=self.producer,
            arm=self.arm,
            started_at=self.started_at,
            duration_ms=self._elapsed_ms(),
        )
