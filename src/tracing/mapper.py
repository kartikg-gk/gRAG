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

import time
from datetime import datetime, timezone
from typing import Any

from .classify import DEFAULT_THRESHOLD
from .schema import STATUS_OK, Trace

#: A retriever's free-form kind, as one of the Studio's entity types. Anything
#: not listed draws as a document.
KIND_TO_ENTITY = {
    "pr": "PR",
    "pull_request": "PR",
    "pullrequest": "PR",
    "service": "Service",
    "person": "Person",
    "user": "Person",
    "author": "Person",
    "document": "Document",
    "doc": "Document",
    "chunk": "Document",
    "commit": "Commit",
    "repo": "Repo",
    "repository": "Repo",
    "branch": "Repo",
    "library": "Library",
    "package": "Library",
    "ticket": "Ticket",
    "issue": "Ticket",
    "jira": "Ticket",
    "team": "Team",
    "tool": "Tool",
}

#: How much of the retrieved text is kept as the run's context.
MAX_CONTEXT_CHARS = 8000

#: How much of one item's text a node carries as its snippet.
SNIPPET_CHARS = 600

#: An edge with no recorded weight is drawn at this confidence.
DEFAULT_EDGE_CONFIDENCE = 0.7


def to_tracestate(trace: Trace) -> dict[str, Any]:
    """The run as a ``TraceState``-shaped dictionary, ready for the Studio."""
    has_edges = any(retrieval.edges for retrieval in trace.retrievals)

    # "Used" means something only when there is an answer and the items were
    # scored against it. Without both, every item is drawn plainly.
    has_usage = trace.answer is not None and any(
        item.overlap is not None for retrieval in trace.retrievals for item in retrieval.items
    )

    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    used_ids: set[str] = set()
    for retrieval in trace.retrievals:
        for item in retrieval.items:
            if item.id in seen:
                continue
            seen.add(item.id)
            if has_usage:
                used = (item.overlap or 0.0) >= DEFAULT_THRESHOLD
                # State what was measured, not what it implies: an item the
                # model used as a constraint can leave no words in the answer
                # and still have mattered.
                subtitle = (
                    f"overlaps the answer · via {trace.producer}"
                    if used
                    else f"no lexical trace in answer · via {trace.producer}"
                )
            else:
                used = True
                subtitle = f"via {trace.producer} · {retrieval.arm} arm"
            if used:
                used_ids.add(item.id)
            nodes.append({
                "id": item.id,
                "label": item.label or item.id,
                "type": KIND_TO_ENTITY.get((item.kind or "").lower().strip(), "Document"),
                # Unused items draw dimmed.
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
            })

    edges: list[dict[str, Any]] = []
    for retrieval in trace.retrievals:
        prefix = (retrieval.span_id or "run")[:8]
        for index, edge in enumerate(retrieval.edges):
            if edge.source not in seen or edge.target not in seen:
                continue
            edges.append({
                "id": f"e_{prefix}_{index}",
                "source": edge.source,
                "target": edge.target,
                "confidence": edge.weight if edge.weight is not None else DEFAULT_EDGE_CONFIDENCE,
                "active": edge.source in used_ids and edge.target in used_ids,
                "relation": edge.relation,
            })

    steps = []
    ordered = sorted(trace.spans, key=lambda span: span.start_ms or 0.0)
    for index, span in enumerate(ordered):
        duration = (
            span.end_ms - span.start_ms
            if span.end_ms is not None and span.start_ms is not None
            else None
        )
        step = {
            "id": span.id,
            "index": index,
            "title": span.name,
            "detail": f"{span.kind} · {span.status}",
            "status": "complete" if span.status == STATUS_OK else "pending",
            "badge": span.kind,
            "durationMs": round(duration, 1) if duration is not None else None,
        }
        if span.kind == "retriever":
            arm = next((r.arm for r in trace.retrievals if r.span_id == span.id), None)
            if arm in ("vector", "graph"):
                step["arm"] = arm
        steps.append(step)

    context = "\n\n".join(
        item.content for retrieval in trace.retrievals for item in retrieval.items if item.content
    )[:MAX_CONTEXT_CHARS] or None

    scores = [
        item.score
        for retrieval in trace.retrievals
        for item in retrieval.items
        if item.score is not None
    ]

    return {
        "id": f"trace_run_{int(time.time() * 1000)}",
        "query": trace.query,
        "computedAt": datetime.now(timezone.utc).isoformat(),
        # No router ran, so the weights describe which arms were observed,
        # not a fusion decision.
        "weights": {
            "vector": 0.5 if has_edges else 1.0,
            "graph": 0.5 if has_edges else 0.0,
            "intent": "relational" if has_edges else "conceptual",
        },
        "confidence": {
            "score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "uncertainty": 0.0,
            "rationale": (
                f"Recorded {trace.producer} run. Scores are as the retriever "
                "reported them; nothing here recomputes them."
            ),
        },
        "steps": steps,
        "metrics": {
            "queryTimeSec": round((trace.duration_ms or 0.0) / 1000.0, 3),
            "nodesEvaluated": len(nodes),
        },
        "graph": {"nodes": nodes, "edges": edges},
        "context": context,
    }
