"""A recorded agent run, reshaped into the state the Studio canvas draws.

The Studio renders a ``TraceState`` (``frontend/src/types/trace.ts``). A trace
recorded from an agent run carries the same facts in a different shape: the
items each retrieval returned, the edges between them, and the spans that
timed each step. ``to_tracestate`` converts one into the other, so a recorded
run can be opened on the canvas like a live query.

Positions are all zero on purpose: the Studio lays the graph out itself.

Scores are carried over as the retriever reported them and never recomputed.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any

from .classify import DEFAULT_THRESHOLD
from .schema import STATUS_OK, Trace

#: The words a retriever might use for each entity type the Studio draws.
#: Kinds are compared with case, spaces, dashes and underscores ignored, so
#: "Pull Request", "pull-request" and "pull_request" are one word. A kind that
#: names nothing here is drawn as a document.
ENTITY_VOCABULARY: dict[str, tuple[str, ...]] = {
    "PR": ("pr", "pullrequest", "mergerequest", "mr"),
    "Commit": ("commit", "changeset", "revision"),
    "Ticket": ("ticket", "issue", "bug"),
    "Person": ("person", "user", "author", "contributor", "maintainer", "reviewer"),
    "Team": ("team", "group"),
    "Repo": ("repo", "repository", "project"),
    "Service": ("service", "component"),
    "Library": ("library", "package", "dependency"),
    "Tool": ("tool",),
}
_TYPE_OF_WORD = {word: kind for kind, words in ENTITY_VOCABULARY.items() for word in words}
_SEPARATORS = re.compile(r"[\s_\-]+")

#: How much of the retrieved text is kept as the run's context.
MAX_CONTEXT_CHARS = 8000

#: How much of one item's text a node carries as its snippet.
SNIPPET_CHARS = 600

#: An edge with no recorded weight is drawn at this confidence.
DEFAULT_EDGE_CONFIDENCE = 0.7

#: Nothing chose these weights: with no router in the run they only record
#: which kinds of retrieval were seen — relations, or text alone.
_OBSERVED_ARMS = {
    True: {"vector": 0.5, "graph": 0.5, "intent": "relational"},
    False: {"vector": 1.0, "graph": 0.0, "intent": "conceptual"},
}


def entity_type(kind: str | None) -> str:
    """The Studio entity type for a retriever's free-form ``kind``."""
    return _TYPE_OF_WORD.get(_SEPARATORS.sub("", (kind or "").lower()), "Document")


def _all_items(trace: Trace):
    for retrieval in trace.retrievals:
        for item in retrieval.items:
            yield retrieval, item


def _judged(trace: Trace) -> bool:
    """Whether items can be told apart as used or unused.

    That takes an answer to compare against and at least one item that was
    scored against it; otherwise every item is simply drawn.
    """
    return trace.answer is not None and any(item.overlap is not None for _, item in _all_items(trace))


def _node(item, arm: str, producer: str, judged: bool) -> dict[str, Any]:
    if judged:
        used = (item.overlap or 0.0) >= DEFAULT_THRESHOLD
        # This reports a text comparison, nothing more: an item can shape an
        # answer without lending it any words.
        verdict = "answer shares source text" if used else "no source-text match detected"
        subtitle = f"{verdict} · via {producer}"
    else:
        used = True
        subtitle = f"via {producer} · {arm} arm"
    return {
        "id": item.id,
        "label": item.label or item.id,
        "type": entity_type(item.kind),
        "active": used,
        "position": {"x": 0, "y": 0},
        "similarity": item.vector_score,
        "score": item.score,
        "meta": {
            "subtitle": subtitle,
            "snippet": (item.content or "")[:SNIPPET_CHARS] or None,
            "scoreGraph": item.graph_score,
            "sourceUrl": item.source_uri,
        },
    }


def _nodes(trace: Trace) -> list[dict[str, Any]]:
    """One node per distinct item, drawn from where the item first appeared."""
    first: dict[str, tuple[Any, str]] = {}
    for retrieval, item in _all_items(trace):
        first.setdefault(item.id, (item, retrieval.arm))
    judged = _judged(trace)
    return [_node(item, arm, trace.producer, judged) for item, arm in first.values()]


def _edges(trace: Trace, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Edges whose two ends are both drawn; lit when both ends were used."""
    drawn = {node["id"] for node in nodes}
    lit = {node["id"] for node in nodes if node["active"]}
    out: list[dict[str, Any]] = []
    for retrieval in trace.retrievals:
        tag = (retrieval.span_id or "run")[:8]
        for position, edge in enumerate(retrieval.edges):
            if {edge.source, edge.target} <= drawn:
                out.append({
                    "id": f"e_{tag}_{position}",
                    "source": edge.source,
                    "target": edge.target,
                    "confidence": DEFAULT_EDGE_CONFIDENCE if edge.weight is None else edge.weight,
                    "active": edge.source in lit and edge.target in lit,
                    "relation": edge.relation,
                })
    return out


def _step(index: int, span, arms: dict[str, str]) -> dict[str, Any]:
    finished = span.start_ms is not None and span.end_ms is not None
    step: dict[str, Any] = {
        "id": span.id,
        "index": index,
        "title": span.name,
        "detail": f"{span.kind} · {span.status}",
        "status": "complete" if span.status == STATUS_OK else "pending",
        "badge": span.kind,
        "durationMs": round(span.end_ms - span.start_ms, 1) if finished else None,
    }
    if span.kind == "retriever" and arms.get(span.id) in ("vector", "graph"):
        step["arm"] = arms[span.id]
    return step


def _steps(trace: Trace) -> list[dict[str, Any]]:
    """The spans in start order; a retriever step names the arm it searched."""
    arms: dict[str, str] = {}
    for retrieval in trace.retrievals:
        arms.setdefault(retrieval.span_id, retrieval.arm)
    ordered = sorted(trace.spans, key=lambda span: span.start_ms or 0.0)
    return [_step(index, span, arms) for index, span in enumerate(ordered)]


def _mean_score(trace: Trace) -> float:
    reported = [item.score for _, item in _all_items(trace) if item.score is not None]
    return round(sum(reported) / len(reported), 3) if reported else 0.0


def to_tracestate(trace: Trace) -> dict[str, Any]:
    """The run as a ``TraceState``-shaped dictionary, ready for the Studio."""
    linked = any(retrieval.edges for retrieval in trace.retrievals)
    nodes = _nodes(trace)
    text = "\n\n".join(item.content for _, item in _all_items(trace) if item.content)
    return {
        "id": f"trace_run_{int(time.time() * 1000)}",
        "query": trace.query,
        "computedAt": datetime.now(timezone.utc).isoformat(),
        "weights": dict(_OBSERVED_ARMS[linked]),
        "confidence": {
            "score": _mean_score(trace),
            "uncertainty": 0.0,
            "rationale": (
                f"Recorded {trace.producer} run. Scores are as the retriever "
                "reported them; nothing here recomputes them."
            ),
        },
        "steps": _steps(trace),
        "metrics": {
            "queryTimeSec": round((trace.duration_ms or 0.0) / 1000.0, 3),
            "nodesEvaluated": len(nodes),
        },
        "graph": {"nodes": nodes, "edges": _edges(trace, nodes)},
        "context": text[:MAX_CONTEXT_CHARS] or None,
    }
