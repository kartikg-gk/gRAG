"""The demo workflow: retrieve, then answer.

This module is the one a developer replaces with their own code. It imports
``overlap_score`` and nothing else from ``src.tracing``, and it imports it as a
lexical-similarity utility rather than as instrumentation: ``retrieve`` uses it
to rank nodes against the query, ``answer`` uses it to decide which retrieved
items belong together. No trace is created here, nothing is recorded, and
neither function knows a trace exists. Instrumentation happens in the demo
entrypoints, which wrap these calls — that is the integration pattern the
examples exist to show.

That distinction is load-bearing, so it is worth stating rather than leaving to
be inferred. Overlap scoring lives in its own module because it has consumers
that are not the tracer, and this is one of them: two calls here that would
have to keep working if tracing were removed entirely. A reader who believed
the scoring function had a single consumer would reasonably conclude it could
be folded into the tracer and the module deleted.

Both functions are deliberately small and dependency-free. This is not the
project's retrieval layer; that lives in ``src/retrieval/`` and is a separate
piece of work. What is here is the least machinery that produces a realistic
trace to look at.
"""

from __future__ import annotations

from typing import Any, Iterable

from src.tracing import overlap_score

# How many retrieved items the answer is allowed to draw on. Everything else
# retrieved is, by construction, waste — which is what the trace makes visible.
ANSWER_USES = 3


def node_text(node: dict[str, Any]) -> str:
    """The searchable text of a graph node.

    Nodes are typed dicts, so each type carries its meaning in a different
    field. Falling back to the id keeps every node searchable rather than
    silently unreachable.
    """
    for field in ("title", "message", "path", "full_name", "login"):
        if node.get(field):
            return str(node[field])
    return node["id"]


def retrieve(
    nodes: dict[str, dict],
    edges: Iterable[dict],
    query: str,
    *,
    limit: int = 7,
) -> tuple[list[dict], list[dict]]:
    """Rank nodes against the query and return them with their relations.

    Scoring is lexical overlap between the query and the node's text. Ties break
    on id so the ranking is stable across runs.

    Note this scores against the **query**, while the classifier later scores
    against the **answer**. Two different questions: "does this match what was
    asked" versus "did this reach what was said". Keeping them separate is what
    makes retrieved-but-unused items possible to see at all.
    """
    scored = []
    for node in nodes.values():
        score = overlap_score(node_text(node), query)
        if score > 0:
            scored.append((score, node))

    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    top = scored[:limit]

    items = [
        {
            "id": node["id"],
            "content": node_text(node),
            "source": "graph",
            "score": round(score, 4),
        }
        for score, node in top
    ]

    retrieved_ids = {item["id"] for item in items}
    related = [
        edge
        for edge in edges
        if edge["source"] in retrieved_ids and edge["target"] in retrieved_ids
    ]

    return items, related


def answer(items: list[dict], *, uses: int = ANSWER_USES) -> str:
    """Compose an answer from the highest-ranked item and what relates to it.

    A template rather than a language model: an LLM would need a key, would put
    the demo online, and would make the output different on every run. The point
    being demonstrated is the trace, not the generation.

    Selection matters more than it looks. Taking the top ``uses`` items would
    make the classifier tautological — it would mark used exactly what the
    template had just quoted. Instead the lead item sets a theme and the answer
    follows it down the ranking, so the items that reach the answer are *not*
    the top of the list. That is the whole phenomenon worth seeing: a document
    can score well, get retrieved, and never be used.
    """
    if not items:
        return "Nothing in the corpus matches that question."

    lead, *rest = items
    chosen = [lead]
    for item in rest:
        if len(chosen) >= uses:
            break
        if overlap_score(item["content"], lead["content"]) > 0:
            chosen.append(item)

    quoted = "; ".join(f"{item['id']} ({item['content']})" for item in chosen)
    return f"The most relevant records are {quoted}."
