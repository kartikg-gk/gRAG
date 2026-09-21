"""The agent tools, as tools an agent framework can hold.

A thin boundary, like the retriever adapter beside it: each tool calls the
function in ``agent_tools`` and returns its dictionary unchanged. The
descriptions are what a model reads to decide which tool to call, so they say
what each is for, when to prefer another, and what comes back.

Lives outside every package for the same reason the retriever adapter does: it
needs ``langchain-core``, and nothing under ``src/`` should grow that
dependency because this exists.
"""

from __future__ import annotations

from . import agent_tools

TRACE_IMPACT = """Find everything connected to one entity in the repository graph \
— a PR, issue, person, file or module — ranked by how strongly it is connected.

Use it for questions about consequences and dependencies: what a change would \
affect, what depends on something, what sits downstream of a PR. It walks the \
graph outward from the entity, so it reaches things no text search would.

Prefer it over search_context whenever the question is about impact rather \
than explanation. If the name could mean more than one entity, call \
find_entity first and pass the exact label.

Arguments: entity_name, matched exactly first and by meaning otherwise; \
max_hops, how far to walk, 1 to 4, default 3 — further reaches weaker links.

Returns: resolved, the entity the walk started from, with how it matched, or \
null; impacted, the reached entities strongest first, each with its confidence \
(path strength, falling with every hop), hops, via (the neighbour it came \
through) and citations (source snippets). Cite these; do not add facts they \
do not contain."""

SEARCH_CONTEXT = """Answer a question about the repository with hybrid retrieval \
— meaning-based search combined with the graph — returning ranked passages \
with their sources.

Use it to understand or explain: why something changed, how a part works, what \
something is responsible for, what happened recently in an area.

Prefer trace_impact when the question is specifically what a change would \
affect or what depends on an entity.

Arguments: query, the question; top_k, most passages to return, default 8.

Returns: intent, how the question was routed (relational or semantic); \
passages, ranked, each with the entity, its type, relevance (0 to 1) and \
citations (source snippets). Base every claim on these citations and name the \
source; state nothing they do not contain."""

FIND_ENTITY = """Look up which entities in the repository graph a name refers \
to — exact matches first, then the closest by meaning.

Use it when a name might be ambiguous or might not exist, before calling \
trace_impact. It does not walk the graph or answer questions; it only \
identifies entities.

Arguments: name, the name or part of it; limit, most candidates, default 5.

Returns: candidates, each with id, label, type, match (exact or semantic) and \
score (1.0 for exact, similarity otherwise) — pass a candidate's label to \
trace_impact as entity_name; match_count, how many were found."""


def graph_tools(engine) -> list:
    """The three tools, bound to one engine's graph."""
    from langchain_core.tools import StructuredTool

    def trace_impact(entity_name: str, max_hops: int = 3) -> dict:
        return agent_tools.trace_impact(engine, entity_name, max_hops)

    def search_context(query: str, top_k: int = 8) -> dict:
        return agent_tools.search_context(engine, query, top_k)

    def find_entity(name: str, limit: int = 5) -> dict:
        return agent_tools.find_entity(engine, name, limit)

    return [
        StructuredTool.from_function(trace_impact, name="trace_impact", description=TRACE_IMPACT),
        StructuredTool.from_function(search_context, name="search_context", description=SEARCH_CONTEXT),
        StructuredTool.from_function(find_entity, name="find_entity", description=FIND_ENTITY),
    ]
