"""What a traced query answers with, and the text built from it.

``RoutedNode`` is one ranked entity with its three scores, its recency and the
documents that mention it. ``RouterResponse`` is the ranking plus the trace
log. ``build_context`` and ``format_page_content`` turn nodes into the text a
model would be given; both write each document's text once, and refer back to
it by id when a later node draws on the same document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass
class RoutedNode:
    id: str
    label: str | None
    type: str | None
    score_total: float
    score_vector: float
    score_graph: float
    recency: float = 1.0
    age_days: float | None = None
    documents: list[dict] = field(default_factory=list)


@dataclass
class RouterResponse:
    query: str
    results: list[RoutedNode]
    trace_log: dict


def _names(node: RoutedNode) -> tuple[str, str]:
    return node.label or node.id, node.type or "Unknown"


def build_context(results: Iterable[RoutedNode]) -> str:
    """The ranked entities, then each document's text once."""
    lines: list[str] = []
    chunks: list[str] = []
    seen: set[str] = set()

    for node in results:
        label, kind = _names(node)
        lines.append(f"- {label} ({kind})")
        for document in node.documents:
            text = (document.get("content") or "").strip()
            if not text:
                continue
            doc_id = document.get("doc_id")
            if doc_id in seen:
                lines.append(f"  [Trace: {label} ({kind}) -> {doc_id}]")
                continue
            seen.add(doc_id)
            chunks.append(f"[{label}] {text}")

    parts = (("GRAPH FACTS", "\n".join(lines)), ("SOURCE TEXT", "\n\n".join(chunks)))
    return "\n\n".join(f"{heading}:\n{body}" for heading, body in parts if body)


def format_page_content(node: RoutedNode, seen: set[str] | None = None) -> str:
    """One node as text: its name, then the documents behind it.

    ``seen`` carries the document ids already written across several calls;
    without one, only repeats within this node collapse.
    """
    seen = seen if seen is not None else set()
    label, kind = _names(node)
    parts: list[str] = []
    for document in node.documents:
        text = (document.get("content") or "").strip()
        if not text:
            continue
        doc_id = document.get("doc_id")
        if doc_id in seen:
            parts.append(f"[Trace: {label} ({kind}) -> {doc_id}]")
            continue
        seen.add(doc_id)
        parts.append(text)

    content = f"Node: {label} ({kind})"
    if parts:
        content += "\n\nContext:\n" + "\n---\n".join(parts)
    return content


def documents_payload(records: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Store documents in the response's shape: ``doc_id``, ``content``, ``path``."""
    return [
        {"doc_id": record.get("id"), "content": record.get("content"), "path": record.get("path")}
        for record in records or ()
    ]


def routed_response(run, store) -> RouterResponse:
    """A retrieval run as a ``RouterResponse``, labels and documents attached."""
    from ..results import trace_payload

    hits = list(getattr(run, "hits", ()) or ())
    documents = store.documents_for_entities([hit.id for hit in hits]) if hits else {}

    results = []
    for hit in hits:
        entity = store.get_entity(hit.id) or {}
        age = getattr(hit, "age_days", None)
        results.append(
            RoutedNode(
                id=hit.id,
                label=entity.get("label"),
                type=entity.get("type") or getattr(hit, "node_type", None),
                score_total=float(hit.score),
                score_vector=float(getattr(hit, "vector_score", 0.0)),
                score_graph=float(getattr(hit, "graph_score", 0.0)),
                recency=round(float(getattr(hit, "decay", 1.0)), 4),
                age_days=round(age, 1) if age is not None else None,
                documents=documents_payload(documents.get(hit.id, [])),
            )
        )

    raw = getattr(run, "trace_log", None)
    trace_log = dict(raw) if isinstance(raw, dict) else (trace_payload(raw) or {})
    return RouterResponse(query=getattr(run, "query", ""), results=results, trace_log=trace_log)
