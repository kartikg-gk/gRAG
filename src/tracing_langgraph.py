"""LangGraph adapter: observe an agent run and emit a framework-neutral trace.

    from langchain_core.runnables import RunnableLambda
    from src.tracing import save
    from src.tracing_langgraph import LangGraphTracer, capture

    tracer = LangGraphTracer()
    result = graph.invoke(inputs, config={"callbacks": [tracer]})

    trace = capture(tracer, answer=result["answer"])
    save(trace, "trace.json")

**Pure observation.** Nothing here changes what the agent does. Callback
failures are swallowed by LangChain rather than propagated (``raise_error`` is
left False on purpose), so a bug in the tracer cannot break a run it is only
watching.

**This module lives outside ``src/tracing/`` deliberately.** That package is
stdlib-only, and a test copies it elsewhere and runs it to prove so. This
adapter needs ``langchain-core``, so keeping it a sibling means installing the
adapter is opt-in and the core schema stays dependency-free.

**Nothing LangChain-shaped survives into the output.** Documents, LLM results
and errors are converted to plain dicts and strings the moment they arrive, so
a trace written from a LangGraph run is indistinguishable from one written by
any other producer. That is what makes the schema framework-neutral rather than
merely framework-agnostic in name.

Only ``langchain_core`` is imported, and only its abstract callback interface,
so this will not fight whatever LangChain version the host application pins.
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Sequence
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

from src.tracing import (
    KIND_CHAIN,
    KIND_LLM,
    KIND_RETRIEVER,
    KIND_TOOL,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_RUNNING,
    Span,
    Trace,
)
from src.tracing import capture as capture_trace

#: Metadata keys a retriever might use for a relevance score, in priority order.
SCORE_KEYS = ("score", "relevance_score", "similarity", "_score", "vector_score")

#: Chain input keys that commonly hold the user's question.
QUERY_KEYS = ("query", "question", "input", "text")

DEFAULT_SOURCE = "retriever"


def _score_of(metadata: dict) -> float | None:
    """The first recognizable score in a document's metadata."""
    for key in SCORE_KEYS:
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _item_from_document(document: Any, index: int) -> dict:
    """Convert one LangChain ``Document`` into a plain trace item.

    Falls back to a positional id so a retriever that carries no metadata still
    produces a usable trace rather than colliding every document onto one id.
    """
    metadata = dict(getattr(document, "metadata", None) or {})
    identifier = metadata.get("id") or metadata.get("source") or f"doc:{index}"

    return {
        "id": str(identifier),
        "content": getattr(document, "page_content", "") or "",
        "source": str(metadata.get("source_type") or DEFAULT_SOURCE),
        "score": _score_of(metadata),
    }


def _edges_from_document(document: Any) -> list[dict]:
    """Relations a retriever attached to a document, if it knows any."""
    metadata = getattr(document, "metadata", None) or {}
    edges = metadata.get("edges")
    if not isinstance(edges, (list, tuple)):
        return []

    return [
        dict(edge)
        for edge in edges
        if isinstance(edge, dict) and {"source", "target", "type"} <= set(edge)
    ]


def _text_of_response(response: Any) -> str | None:
    """The generated text from an ``LLMResult``, chat or completion."""
    generations = getattr(response, "generations", None)
    if not generations:
        return None

    for batch in generations:
        for generation in batch:
            text = getattr(generation, "text", None)
            if text:
                return text
            message = getattr(generation, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str) and content:
                return content
    return None


def _name_of(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str | None:
    """The runnable's name.

    langchain-core passes ``name`` as a keyword and leaves ``serialized`` as
    ``None``; older versions put it inside ``serialized``. Both are read so the
    adapter does not silently record nothing against a version it did not
    expect.
    """
    name = kwargs.get("name")
    if isinstance(name, str) and name:
        return name

    serialized = serialized or {}
    name = serialized.get("name")
    if isinstance(name, str) and name:
        return name

    identifier = serialized.get("id")
    if isinstance(identifier, (list, tuple)) and identifier:
        return str(identifier[-1])
    if isinstance(identifier, str) and identifier:
        return identifier
    return None


def _query_from_inputs(inputs: Any) -> str | None:
    """The user's question, when a chain's inputs carry something recognizable."""
    if isinstance(inputs, str):
        return inputs
    if not isinstance(inputs, dict):
        return None
    for key in QUERY_KEYS:
        value = inputs.get(key)
        if isinstance(value, str) and value:
            return value
    return None


class LangGraphTracer(BaseCallbackHandler):
    """Records what an agent retrieved and what it answered.

    Attach it to a run and read the result afterwards::

        tracer = LangGraphTracer()
        graph.invoke(inputs, config={"callbacks": [tracer]})
        trace = capture(tracer)

    A retriever's query wins over a chain's inputs when both are seen, because
    the retriever was asked something specific while a chain's inputs may be
    the whole agent state.

    Documents are de-duplicated by id, keeping the first. A document retrieved
    twice is one document that was available to the answer, and showing it
    twice would make the used-versus-ignored count read wrong.
    """

    #: Never let a tracer bug break the run it is watching.
    raise_error = False

    def __init__(self) -> None:
        self.query: str | None = None
        self.answer: str | None = None
        self.items: list[dict] = []
        self.edges: list[dict] = []
        self.errors: list[dict] = []
        self.spans: list[Span] = []
        self.started_at: datetime | None = None
        self.duration_ms: float | None = None
        self.chains: list[str] = []
        self.tools: list[str] = []

        self._seen_item_ids: set[str] = set()
        self._seen_edges: set[tuple[str, str, str]] = set()
        self._spans_by_run: dict[str, Span] = {}
        self._began: float | None = None

    # -- lifecycle ---------------------------------------------------------

    def _mark_start(self) -> None:
        if self.started_at is None:
            self.started_at = datetime.now(timezone.utc)
            self._began = perf_counter()

    def _elapsed_ms(self) -> float:
        if self._began is None:
            return 0.0
        return round((perf_counter() - self._began) * 1000, 3)

    def _mark_end(self) -> None:
        if self._began is not None:
            self.duration_ms = round((perf_counter() - self._began) * 1000, 1)

    # -- spans -------------------------------------------------------------

    def _open_span(
        self,
        name: str,
        kind: str,
        run_id: UUID | None,
        parent_run_id: UUID | None,
    ) -> Span:
        """Start recording a unit of work.

        Nesting comes from LangChain's own ``run_id``/``parent_run_id`` rather
        than from a stack of whatever is currently open. Runs can interleave —
        two retrievals in flight at once, or an async branch — and a stack
        would attach a span to whichever sibling happened to open last.

        A ``parent_run_id`` for a run this tracer never saw start (attaching
        mid-tree) records no parent rather than a dangling reference.
        """
        self._mark_start()
        span_id = str(run_id) if run_id is not None else f"span-{len(self.spans) + 1}"
        parent = str(parent_run_id) if parent_run_id is not None else None

        span = Span(
            id=span_id,
            name=name,
            kind=kind,
            parent_id=parent if parent in self._spans_by_run else None,
            start_ms=self._elapsed_ms(),
            status=STATUS_RUNNING,
        )
        self.spans.append(span)
        self._spans_by_run[span_id] = span
        return span

    def _close_span(
        self,
        run_id: UUID | None,
        status: str,
        *,
        name: str = "unknown",
        kind: str = KIND_CHAIN,
    ) -> Span:
        """Finish a unit of work.

        A close with no matching start still produces a span. Some callbacks
        only fire one half — a chat model reports its end without a start this
        handler sees — and a failure that leaves no trace of the unit is worse
        than one recorded with a zero-length window.
        """
        span = self._spans_by_run.get(str(run_id)) if run_id is not None else None
        if span is None:
            span = self._open_span(name, kind, run_id, None)

        span.end_ms = self._elapsed_ms()
        span.status = status
        return span

    def _record_error(
        self,
        phase: str,
        error: BaseException,
        run_id: UUID | None = None,
    ) -> None:
        """Close the failed unit with error status and keep it in the trace.

        Failed work is exactly what a trace needs to show. Dropping it would
        make a broken run look like a short one, so the span survives with
        ``status="error"``. ``tracer.errors`` carries the message as well,
        since the schema has nowhere to put exception text.
        """
        self._mark_end()
        self._close_span(run_id, STATUS_ERROR, name=phase, kind=phase)
        self.errors.append({"phase": phase, "error": f"{type(error).__name__}: {error}"})

    # -- chain -------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = _name_of(serialized, kwargs) or "chain"
        self._open_span(name, KIND_CHAIN, run_id, parent_run_id)
        self.chains.append(name)
        if self.query is None:
            self.query = _query_from_inputs(inputs)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._mark_end()
        self._close_span(run_id, STATUS_OK, name="chain", kind=KIND_CHAIN)
        if self.answer is None and isinstance(outputs, dict):
            for key in ("answer", "output", "result", "text"):
                value = outputs.get(key)
                if isinstance(value, str) and value:
                    self.answer = value
                    break

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_error(KIND_CHAIN, error, run_id)

    # -- retriever ---------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._open_span(
            _name_of(serialized, kwargs) or "retriever",
            KIND_RETRIEVER,
            run_id,
            parent_run_id,
        )
        if query:
            self.query = query

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._mark_end()
        self._close_span(run_id, STATUS_OK, name="retriever", kind=KIND_RETRIEVER)
        for document in documents or ():
            item = _item_from_document(document, len(self.items))
            if item["id"] not in self._seen_item_ids:
                self._seen_item_ids.add(item["id"])
                self.items.append(item)

            for edge in _edges_from_document(document):
                key = (edge["source"], edge["type"], edge["target"])
                if key not in self._seen_edges:
                    self._seen_edges.add(key)
                    self.edges.append(edge)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_error(KIND_RETRIEVER, error, run_id)

    # -- llm ---------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._open_span(
            _name_of(serialized, kwargs) or "llm", KIND_LLM, run_id, parent_run_id
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._mark_end()
        self._close_span(run_id, STATUS_OK, name="llm", kind=KIND_LLM)
        text = _text_of_response(response)
        if text:
            self.answer = text

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_error(KIND_LLM, error, run_id)

    # -- tool --------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = _name_of(serialized, kwargs) or "tool"
        self._open_span(name, KIND_TOOL, run_id, parent_run_id)
        self.tools.append(name)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._mark_end()
        self._close_span(run_id, STATUS_OK, name="tool", kind=KIND_TOOL)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_error(KIND_TOOL, error, run_id)


def capture(
    tracer: LangGraphTracer,
    *,
    query: str | None = None,
    answer: str | None = None,
) -> Trace:
    """Build a trace from what the tracer observed.

    ``query`` and ``answer`` override what was seen, for the common case where
    the caller holds a cleaner version than the callbacks could infer — an
    agent's final answer often lives in the invoke result rather than in the
    last LLM call.

    Delegates to the same ``capture`` every other producer uses, so a
    LangGraph-sourced trace is built by exactly the same code path.
    """
    return capture_trace(
        query if query is not None else (tracer.query or ""),
        tracer.items,
        answer if answer is not None else tracer.answer,
        edges=tracer.edges,
        spans=tracer.spans,
        started_at=tracer.started_at,
        duration_ms=tracer.duration_ms,
    )
