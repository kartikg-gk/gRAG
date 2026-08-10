"""LangGraph demo: trace a real StateGraph run.

    python -m examples.graph_demo

Builds a three-node graph — retrieve, then answer, with a tool call in between
— runs it with the tracer attached, and prints the trace.

This exists so the adapter is exercised against a real graph rather than
hand-called hooks. Node spans, ``langgraph_node`` metadata, callback
propagation into sub-runnables, and the filtering of framework internals only
happen under an actual LangGraph run; logic that ships without one is untested
code.

``langgraph`` is an example/test dependency, not a runtime one. The tracing
package needs nothing, and the adapter needs only ``langchain-core``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from src.tracing import render, save, score_overlaps

from . import fixtures
from .workflow import ANSWER_USES

DEFAULT_OUTPUT = Path("graph_trace.json")


# --------------------------------------------------------------------------
# the corpus, as LangChain documents
# --------------------------------------------------------------------------


def _documents() -> list[Document]:
    """The offline fixture corpus, shaped as documents a retriever returns."""
    documents: list[Document] = []

    for pull_request in fixtures.PULL_REQUESTS:
        documents.append(
            Document(
                page_content=pull_request["title"],
                metadata={
                    "id": f"pr:{pull_request['number']}",
                    "kind": "PR",
                    "label": pull_request["title"],
                    "score": 0.9 - 0.1 * len(documents),
                    "source": pull_request["html_url"],
                    "source_type": "graph",
                },
            )
        )

    for issue in fixtures.ISSUES:
        if "pull_request" in issue:
            continue
        documents.append(
            Document(
                page_content=issue["title"],
                metadata={
                    "id": f"ticket:{issue['number']}",
                    "kind": "Ticket",
                    "label": issue["title"],
                    "score": 0.5,
                    "source": issue["html_url"],
                    "source_type": "graph",
                    # A relation the retriever knows about, carried alongside
                    # the document so the tracer can record it.
                    "edges": [
                        {
                            "source": "pr:101",
                            "target": f"ticket:{issue['number']}",
                            "relation": "RESOLVES",
                            "weight": 0.92,
                        }
                    ],
                },
            )
        )

    return documents


class FixtureRetriever(BaseRetriever):
    """Ranks the fixture corpus by naive term overlap with the query."""

    documents: list[Document] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        wanted = set(query.lower().split())
        scored = [
            (len(wanted & set(document.page_content.lower().split())), document)
            for document in self.documents
        ]
        matches = [document for hits, document in scored if hits]
        matches.sort(key=lambda document: document.metadata["id"])
        return matches or list(self.documents)


@tool
def count_documents(count: str) -> str:
    """Report how many documents were retrieved."""
    return f"{count} documents available"


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    question: str
    documents: Annotated[list[Document], "retrieved"]
    note: str
    answer: str


def build_graph(retriever: BaseRetriever | None = None):
    """A retrieve -> inspect -> answer graph.

    Three named nodes, so the trace should show exactly three node spans and
    none of LangGraph's own plumbing.
    """
    retriever = retriever or FixtureRetriever(documents=_documents())

    def retrieve(state: AgentState) -> dict[str, Any]:
        # No config is threaded by hand: LangGraph propagates the callbacks it
        # was invoked with into every sub-runnable, which is what carries the
        # tracer into the retriever.
        return {"documents": retriever.invoke(state["question"])}

    def inspect(state: AgentState) -> dict[str, Any]:
        return {"note": count_documents.invoke({"count": str(len(state["documents"]))})}

    def answer(state: AgentState) -> dict[str, Any]:
        chosen = state["documents"][:ANSWER_USES]
        quoted = "; ".join(
            f"{document.metadata['id']} ({document.page_content})"
            for document in chosen
        )
        return {"answer": f"The most relevant records are {quoted}."}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("inspect", inspect)
    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "inspect")
    graph.add_edge("inspect", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def run(query: str, *, output: Path | None = None):
    """Run the graph with the tracer attached and return the trace."""
    from src.tracing_langgraph import LangGraphTracer

    tracer = LangGraphTracer()
    result = build_graph().invoke(
        {"question": query},
        # The whole integration: one config key.
        config={"callbacks": [tracer]},
    )

    trace = tracer.finish(query=query, answer=result["answer"])
    score_overlaps(trace)

    if output is not None:
        save(trace, output)
    return trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="graph_demo",
        description="Trace a real LangGraph run. No network, no model.",
    )
    parser.add_argument("--query", default=fixtures.DEFAULT_QUERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        trace = run(args.query, output=args.output)
    except OSError as exc:
        print(f"could not write {args.output}: {exc}", file=sys.stderr)
        return 1

    print(render(trace))
    print("\nspans")
    for span in trace.spans:
        indent = "  " if span.parent_id else ""
        print(f"  {indent}{span.kind:<10} {span.name:<12} {span.status}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
