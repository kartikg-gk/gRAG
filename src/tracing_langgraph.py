"""LangGraph adapter: observe an agent run and emit a framework-neutral trace.

    from langgraph.graph import StateGraph
    from src.tracing import save
    from src.tracing_langgraph import LangGraphTracer

    tracer = LangGraphTracer()
    result = graph.invoke(inputs, config={"callbacks": [tracer]})

    trace = tracer.finish(answer=result["answer"])
    save(trace)

**Pure observation.** Nothing here changes what the agent does. Callback
failures are swallowed by LangChain rather than propagated (``raise_error`` is
left False on purpose), so a bug in the tracer cannot break a run it is only
watching.

**This module lives outside ``src/tracing/`` deliberately.** That package is
stdlib-only, and a test copies it elsewhere and runs it to prove so. This
adapter needs ``langchain-core``, so keeping it a sibling means installing the
adapter is opt-in and the core schema stays dependency-free.

**Nothing LangChain-shaped survives into the output.** Documents, LLM results
and errors are converted to plain data the moment they arrive, so a trace
written from a LangGraph run is indistinguishable from one written by any other
producer.

**Framework internals are filtered out of the span list.** LangGraph wraps user
nodes in ``RunnableSequence``, ``ChannelWrite`` and friends; recording those
would bury the three steps someone actually wrote under a dozen they did not.
Only chains that LangGraph itself labels as a node are kept.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Final, Sequence
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

from .tracing import (
    ARM_GRAPH,
    ARM_VECTOR,
    KIND_DOCUMENT,
    KIND_LLM,
    KIND_RETRIEVER,
    KIND_TOOL,
    PRODUCER_UNKNOWN,
    STATUS_ERROR,
    STATUS_OK,
    Retrieval,
    Span,
    Trace,
    TraceEdge,
    TraceItem,
)

#: The span kind for a user-defined LangGraph node.
KIND_NODE = "node"

#: Metadata keys a retriever might use for a relevance score, in priority order.
SCORE_KEYS: Final[tuple[str, ...]] = (
    "score",
    "relevance_score",
    "similarity",
    "_score",
    "vector_score",
)

#: Chain input keys that commonly hold the user's question.
QUERY_KEYS: Final[tuple[str, ...]] = ("query", "question", "input", "text")

#: LangGraph's own plumbing. These are runnables the framework creates, not
#: steps anyone wrote, and recording them buries the real ones.
NOISE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "LangGraph",
        "RunnableSequence",
        "RunnableCallable",
        "RunnableLambda",
        "ChannelWrite",
        "ChannelRead",
        "_write",
        "_route",
        "__start__",
        "__end__",
    }
)

DEFAULT_SOURCE = "retriever"


def _score_of(metadata: dict) -> float | None:
    """The first recognizable score in a document's metadata."""
    for key in SCORE_KEYS:
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _document_id(document: Any, index: int) -> str:
    """A stable identifier for a retrieved document.

    Falls back to a hash of the content rather than a positional id: the same
    document retrieved twice in one run, or across two runs, must land on the
    same id, and position does not survive a reordering.
    """
    metadata = getattr(document, "metadata", None) or {}
    for key in ("id", "node_id", "doc_id"):
        value = metadata.get(key)
        if value:
            return str(value)

    identifier = getattr(document, "id", None)
    if identifier:
        return str(identifier)

    content = getattr(document, "page_content", "") or ""
    if content:
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]
        return f"doc_{digest}"

    return f"doc:{index}"


def _item_from_document(document: Any, index: int) -> TraceItem:
    """Convert one LangChain ``Document`` into a framework-neutral item."""
    metadata = dict(getattr(document, "metadata", None) or {})
    item_id = _document_id(document, index)
    score = _score_of(metadata)

    # Only scalars survive into the trace: a metadata value holding a framework
    # object would make the trace unserializable and leak LangChain into it.
    carried = {
        key: value
        for key, value in metadata.items()
        if key != "edges" and isinstance(value, (str, int, float, bool))
    }

    return TraceItem(
        id=item_id,
        content=getattr(document, "page_content", "") or "",
        source=str(metadata.get("source_type") or DEFAULT_SOURCE),
        label=str(metadata.get("label") or metadata.get("title") or item_id),
        kind=str(metadata.get("kind") or metadata.get("type") or KIND_DOCUMENT),
        source_uri=metadata.get("source") or metadata.get("source_uri"),
        score=score,
        vector_score=score,
        metadata=carried,
    )


def _edges_from_document(document: Any) -> list[TraceEdge]:
    """Relations a retriever attached to a document, if it knows any."""
    metadata = getattr(document, "metadata", None) or {}
    edges = metadata.get("edges")
    if not isinstance(edges, (list, tuple)):
        return []

    converted: list[TraceEdge] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        relation = edge.get("relation") or edge.get("type")
        if not source or not target or not relation:
            continue
        converted.append(
            TraceEdge(
                source=str(source),
                target=str(target),
                relation=str(relation),
                weight=edge.get("weight", edge.get("confidence")),
            )
        )
    return converted


def _text_of_response(response: Any) -> str | None:
    """The generated text from an ``LLMResult``, chat or completion."""
    generations = getattr(response, "generations", None)
    if not generations:
        return None

    for batch in reversed(list(generations)):
        for generation in reversed(list(batch)):
            text = getattr(generation, "text", None)
            if text:
                return text
            message = getattr(generation, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str) and content:
                return content
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


def _name_of(
    serialized: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    kwargs: dict[str, Any],
    fallback: str,
) -> str:
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
    for candidate in (serialized.get("name"), (metadata or {}).get("langgraph_node")):
        if isinstance(candidate, str) and candidate:
            return candidate

    identifier = serialized.get("id")
    if isinstance(identifier, (list, tuple)) and identifier:
        return str(identifier[-1])
    if isinstance(identifier, str) and identifier:
        return identifier
    return fallback


class LangGraphTracer(BaseCallbackHandler):
    """Collect spans and retrieval information from a LangGraph run.

    Attach it to a run and read the result afterwards::

        tracer = LangGraphTracer()
        graph.invoke(inputs, config={"callbacks": [tracer]})
        trace = tracer.finish()

    Nesting comes from LangChain's own ``run_id``/``parent_run_id`` rather than
    from a stack of whatever is currently open. Runs can interleave — two
    retrievals in flight at once, or an async branch — and a stack would attach
    a span to whichever sibling happened to open last.
    """

    #: Never let a tracer bug break the run it is watching.
    raise_error = False

    def __init__(self, *, producer: str = "langgraph") -> None:
        super().__init__()

        self.producer = producer
        self.started_at = datetime.now(timezone.utc)
        self._began = perf_counter()

        self.spans: list[Span] = []
        self.retrievals: list[Retrieval] = []
        self.errors: list[dict] = []

        self._spans_by_run: dict[str, Span] = {}
        self._retrieval_by_run: dict[str, Retrieval] = {}
        # Groups opened but not yet closed, so a hook called without a
        # run_id can still find the one it belongs to.
        self._open_retrievals: list[Retrieval] = []
        # Runnables filtered as framework noise, mapped to the nearest ancestor
        # that was kept, so their children do not lose their place in the tree.
        self._skipped_parents: dict[str, str | None] = {}
        self._first_query: str | None = None
        self._last_llm_text: str | None = None

    # -- timing ------------------------------------------------------------

    def _now_ms(self) -> float:
        return round((perf_counter() - self._began) * 1000, 3)

    # -- spans -------------------------------------------------------------

    def _open_span(
        self,
        run_id: UUID | None,
        parent_run_id: UUID | None,
        name: str,
        kind: str,
    ) -> Span:
        """Start recording a unit of work.

        A ``parent_run_id`` for a run this tracer never saw start — because the
        parent was filtered as framework noise, or because the tracer attached
        mid-tree — is resolved to the nearest ancestor that *was* recorded, so
        a user node under a ``RunnableSequence`` still reports the node above
        it rather than losing its place in the tree.
        """
        span_id = str(run_id) if run_id is not None else f"span-{len(self.spans) + 1}"
        span = Span(
            id=span_id,
            name=name,
            kind=kind,
            parent_id=self._nearest_recorded_ancestor(parent_run_id),
            start_ms=self._now_ms(),
        )
        self.spans.append(span)
        self._spans_by_run[span_id] = span
        return span

    def _nearest_recorded_ancestor(self, parent_run_id: UUID | None) -> str | None:
        """The closest ancestor that survived filtering, if any."""
        if parent_run_id is None:
            return None
        parent = str(parent_run_id)
        if parent in self._spans_by_run:
            return parent
        return self._skipped_parents.get(parent)

    def _close_span(
        self,
        run_id: UUID | None,
        status: str,
        *,
        name: str = "unknown",
        kind: str = KIND_NODE,
    ) -> Span:
        """Finish a unit of work.

        A close with no matching start still produces a span. Some callbacks
        only fire one half — a chat model reports its end without a start this
        handler sees — and a failure that leaves no trace of the unit is worse
        than one recorded with a zero-length window.
        """
        span = self._spans_by_run.get(str(run_id)) if run_id is not None else None
        if span is None:
            if run_id is not None and str(run_id) in self._skipped_parents:
                # A filtered runnable ending is not a unit of work.
                return Span(id="", name=name, kind=kind, status=status)
            span = self._open_span(run_id, None, name, kind)

        span.end_ms = self._now_ms()
        span.status = status
        return span

    def _record_error(
        self, phase: str, error: BaseException, run_id: UUID | None = None
    ) -> None:
        """Close the failed unit with error status and keep it in the trace.

        Failed work is exactly what a trace needs to show. Dropping it would
        make a broken run look like a short one, so the span survives with
        ``status="error"``. ``tracer.errors`` carries the message as well,
        since the schema has nowhere to put exception text.
        """
        self._close_span(run_id, STATUS_ERROR, name=phase, kind=phase)
        self.errors.append({"phase": phase, "error": f"{type(error).__name__}: {error}"})

    # -- chain / LangGraph nodes -------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._first_query is None:
            self._first_query = _query_from_inputs(inputs)

        name = _name_of(serialized, metadata, kwargs, "chain")
        node = (metadata or {}).get("langgraph_node")

        # Only user-defined LangGraph nodes become spans. Everything else is
        # framework plumbing; it is remembered only so its children can find
        # the real ancestor above it.
        if node is None or name in NOISE_NAMES or name != node:
            if run_id is not None:
                self._skipped_parents[str(run_id)] = self._nearest_recorded_ancestor(
                    parent_run_id
                )
            return

        self._open_span(run_id, parent_run_id, name, KIND_NODE)

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._close_span(run_id, STATUS_OK, name="chain", kind=KIND_NODE)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_error(KIND_NODE, error, run_id)

    # -- retriever ---------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if query:
            self._first_query = query

        span = self._open_span(
            run_id,
            parent_run_id,
            _name_of(serialized, metadata, kwargs, "retriever"),
            KIND_RETRIEVER,
        )

        # One retriever call is one retrieval group, so a run with two
        # retrievers keeps their results separable.
        retrieval = Retrieval(query=query or "", span_id=span.id)
        self.retrievals.append(retrieval)
        self._retrieval_by_run[span.id] = retrieval
        self._open_retrievals.append(retrieval)

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._close_span(run_id, STATUS_OK, name="retriever", kind=KIND_RETRIEVER)

        retrieval = self._retrieval_by_run.get(str(run_id))
        if retrieval is None and run_id is None and self._open_retrievals:
            # Hooks called by hand carry no run_id, so the start cannot be
            # matched by key. The most recently opened group is the only one it
            # could belong to. A real run always has a run_id and never lands
            # here.
            retrieval = self._open_retrievals[-1]
        if retrieval is None:
            retrieval = Retrieval(query=self._first_query or "", span_id=str(run_id))
            self.retrievals.append(retrieval)
        if retrieval in self._open_retrievals:
            self._open_retrievals.remove(retrieval)

        seen_items: set[str] = {item.id for item in retrieval.items}
        seen_edges: set[tuple[str, str, str]] = {
            (edge.source, edge.relation, edge.target) for edge in retrieval.edges
        }

        for index, document in enumerate(documents or ()):
            item = _item_from_document(document, len(retrieval.items) + index)
            if item.id not in seen_items:
                seen_items.add(item.id)
                retrieval.items.append(item)

            for edge in _edges_from_document(document):
                key = (edge.source, edge.relation, edge.target)
                if key not in seen_edges:
                    seen_edges.add(key)
                    retrieval.edges.append(edge)

        # A retriever that reported relations was walking a graph; one that
        # reported only similarity was not.
        retrieval.arm = ARM_GRAPH if retrieval.edges else ARM_VECTOR

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
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._open_span(
            run_id, parent_run_id, _name_of(serialized, metadata, kwargs, "llm"), KIND_LLM
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._close_span(run_id, STATUS_OK, name="llm", kind=KIND_LLM)
        text = _text_of_response(response)
        if text:
            self._last_llm_text = text

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
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._open_span(
            run_id,
            parent_run_id,
            _name_of(serialized, metadata, kwargs, "tool"),
            KIND_TOOL,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
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

    # -- the finished trace ------------------------------------------------

    def finish(
        self, query: str | None = None, answer: str | None = None
    ) -> Trace:
        """Build the project's ``Trace`` from what was observed.

        Both arguments are optional. ``query`` falls back to the first one
        seen — the earliest is the question actually asked, before any
        rewriting. ``answer`` falls back to the last text generated, since a
        later generation supersedes an earlier one.

        Overlap is deliberately *not* measured here. Scoring is
        ``classify.score_overlaps``, and duplicating it would make two places
        able to disagree about what a measurement means.
        """
        for span in self.spans:
            if span.end_ms is None:
                span.end_ms = self._now_ms()
                if span.status == "running":
                    span.status = STATUS_OK

        latency = max((span.end_ms or 0.0 for span in self.spans), default=None)

        return Trace(
            query=query if query is not None else (self._first_query or ""),
            answer=answer if answer is not None else self._last_llm_text,
            producer=self.producer or PRODUCER_UNKNOWN,
            started_at=self.started_at,
            duration_ms=latency,
            retrievals=list(self.retrievals),
            spans=list(self.spans),
        )
