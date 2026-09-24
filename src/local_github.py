"""Fetch GitHub history, query an in-memory graph, and save a local v4 trace.

The hosted graph store and its embedding model are deliberately not involved.
This path uses the same validated GitHub client, graph builder, document
chunker, and trace contract as the optional backend.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .ingestion import (
    GitHubError, collect_by_pull_request, fetch_changed_files, fetch_commits,
    fetch_issues, fetch_pull_requests, fetch_repository, fetch_reviews,
    in_ingest_order, make_session,
)
from .knowledge import GraphBuilder
from .knowledge.documents import source_documents
from .tracing import (
    Retrieval, Span, Trace, TraceEdge, TraceItem, overlap_score, save,
)
from .tracing._text import STOP, tokens

DEFAULT_OUTPUT = Path("graphrag_out/trace_state.json")
ITEM_EXCERPT_CHARS = 500
REPO_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


def _words(value: str) -> set[str]:
    return tokens(value, stop=STOP | {"who", "what", "when", "where", "which", "how", "why", "repo", "repository"})


def _answer_excerpt(item: TraceItem) -> str:
    """Keep evidence readable without allowing GitHub templates into stdout."""
    body = item.content.removeprefix(item.label or "")
    body = re.sub(r"<!--.*?(?:-->|$)", " ", body, flags=re.DOTALL)
    body = re.sub(r"\s+", " ", body.replace("\ufeff", " ")).strip(" .#")
    return f"{item.label}: {body[:120].rstrip()}" if body else str(item.label)


def fetch_graph(repo: str, *, prs: int, issues: int, commits: int,
                files: bool, reviews: bool, session):
    """Use the existing ingestion path without opening a database."""
    repository = fetch_repository(session, repo)
    pulls = in_ingest_order(fetch_pull_requests(session, repo, limit=prs))
    tickets = in_ingest_order(fetch_issues(session, repo, limit=issues))
    history = in_ingest_order(fetch_commits(session, repo, limit=commits))
    numbers = [pull.number for pull in pulls]
    review_rows, review_failures = (
        collect_by_pull_request(fetch_reviews, session, repo, numbers)
        if reviews else ({}, 0)
    )
    file_rows, file_failures = (
        collect_by_pull_request(fetch_changed_files, session, repo, numbers)
        if files else ({}, 0)
    )
    graph = GraphBuilder()
    stats = graph.build(
        repository=repository, pull_requests=pulls, issues=tickets,
        commits=history, reviews=review_rows, changed_files=file_rows,
        enrichment_failures_by_stage={
            "reviews": review_failures, "changed_files": file_failures,
        },
    )
    documents = source_documents(
        pull_requests=pulls, issues=tickets, commits=history, reviews=review_rows,
    )
    return graph, documents, stats


def build_trace(repo: str, question: str, graph: GraphBuilder, documents,
                stats, *, started_at: datetime | None = None,
                ingestion_ms: float = 0.0, top: int = 10,
                source_metrics: dict | None = None) -> Trace:
    """Rank graph nodes by question overlap, then expand through typed edges."""
    began = time.perf_counter()
    started_at = started_at or datetime.now(timezone.utc)
    by_origin: dict[str, list] = {}
    for document in documents:
        by_origin.setdefault(document.origin, []).append(document)
    question_words = _words(question)
    filename_query_words = question_words - _words(repo.replace("/", " "))
    content: dict[str, str] = {}
    lexical: dict[str, float] = {}
    filename_hits: dict[str, int] = {}
    selected_documents: dict[str, object] = {}
    for node_id, node in graph.nodes.items():
        filename = PurePosixPath(str(node.get("path") or "")).stem
        filename_hits[node_id] = len(filename_query_words & _words(filename.replace("_", " ").replace("-", " ")))
        fields = [str(node.get(key) or "") for key in ("title", "name", "login", "full_name", "message", "filename", "path")]
        attached = by_origin.get(node_id, ())
        searchable = " ".join(fields + [document.content for document in attached])
        best_chunk = max(attached, key=lambda document: len(question_words & _words(document.content)),
                         default=None)
        excerpt = " ".join(value for value in fields if value)
        if best_chunk:
            excerpt = f"{excerpt}. {best_chunk.content}"
            selected_documents[node_id] = best_chunk
        content[node_id] = excerpt.strip()[:ITEM_EXCERPT_CHARS] or node_id
        words = _words(searchable)
        lexical[node_id] = len(question_words & words) / len(question_words) if question_words else 0.0

    ranked = sorted(graph.nodes, key=lambda node_id: (lexical[node_id], filename_hits[node_id],
                    graph.nodes[node_id].get("timestamp") or datetime.min.replace(tzinfo=timezone.utc),
                    node_id), reverse=True)
    seeds = ranked[:max(1, top // 2)]
    scores = dict(lexical)
    graph_scores: dict[str, float] = {}
    for edge in graph.edges:
        for source, target in ((edge["source"], edge["target"]),
                               (edge["target"], edge["source"])):
            if source in seeds and target in graph.nodes:
                score = min(1.0, lexical[source] * float(edge["confidence"]) * 0.7)
                graph_scores[target] = max(graph_scores.get(target, 0.0), score)
                scores[target] = max(scores.get(target, 0.0), score)
    chosen = sorted(scores, key=lambda node_id: (scores[node_id], lexical[node_id],
                                                filename_hits[node_id], node_id), reverse=True)[:top]
    if not chosen:
        raise ValueError("GitHub returned no graph nodes")
    selected = set(chosen)

    def item(node_id: str) -> TraceItem:
        node = graph.nodes[node_id]
        node_type = str(node["type"])
        source_url = node.get("url") or f"https://github.com/{repo}"
        document = selected_documents.get(node_id)
        if node.get("indexed_source") and document is not None:
            source_url = document.path
        elif node_type == "Person":
            source_url = f"https://github.com/{node.get('login', '')}"
        timestamp = node.get("timestamp")
        return TraceItem(
            id=node_id, label=str(node.get("title") or node.get("name") or node.get("login") or node.get("filename") or node.get("path") or node_id),
            kind=node_type, content=content[node_id], source="github",
            source_uri=source_url, score=round(scores[node_id], 4),
            graph_score=round(graph_scores.get(node_id, 0.0), 4),
            metadata={"repository": repo, "node_type": node_type,
                      "lexical_score": round(lexical[node_id], 4),
                      "revision": node.get("revision"),
                      "indexed_source": bool(node.get("indexed_source")),
                      "timestamp": timestamp.isoformat() if timestamp else None,
                      "source_documents": [document.path for document in by_origin.get(node_id, ())]},
        )

    items = [item(node_id) for node_id in chosen]
    edges = [TraceEdge(source=edge["source"], target=edge["target"],
                       relation=edge["type"], weight=float(edge["confidence"]))
             for edge in graph.edges if edge["source"] in selected and edge["target"] in selected]
    retrieval_ms = (time.perf_counter() - began) * 1000
    answer_began = time.perf_counter()
    scope = "repository sample" if source_metrics else "recent history"
    evidence = [candidate for candidate in items if lexical[candidate.id] > 0][:3]
    if evidence:
        details = "; ".join(f"{_answer_excerpt(candidate)} ({candidate.source_uri})"
                            for candidate in evidence)
        answer = f"In the {repo} {scope} fetched for this run, the closest matches are: {details}. Open the trace for their source text and connections."
    else:
        answer = f"No text match for this question appeared in the fetched {repo} {scope}. Inspect the retrieved graph and sources; this is not a claim about the entire repository."
    for candidate in items:
        candidate.overlap = round(overlap_score(candidate.content, answer), 4)
    answer_ms = (time.perf_counter() - answer_began) * 1000
    total_ms = ingestion_ms + retrieval_ms + answer_ms
    trace = Trace(
        query=question, answer=answer, producer="graphrag-local-github",
        started_at=started_at, duration_ms=round(total_ms, 2),
        retrievals=[Retrieval(query=question, span_id="retrieve", arm="graph", items=items, edges=edges)],
        graph_nodes=items, graph_edges=edges,
        spans=[
            Span(id="ingest", name="Fetch and build repository graph", kind="tool", start_ms=0,
                 end_ms=round(ingestion_ms, 2), status="ok"),
            Span(id="retrieve", name="Rank and traverse graph", kind="retriever",
                 start_ms=round(ingestion_ms, 2), end_ms=round(ingestion_ms + retrieval_ms, 2), status="ok"),
            Span(id="answer", name="Summarize cited matches", kind="chain",
                 start_ms=round(ingestion_ms + retrieval_ms, 2), end_ms=round(total_ms, 2), status="ok"),
        ],
        metrics={"graph_nodes": len(graph.nodes), "graph_edges": len(graph.edges),
                 "source_chunks": len(documents), "retrieved_nodes": len(items),
                 "retrieved_edges": len(edges), "pull_requests": stats.pull_requests,
                 "issues": stats.issues, "commits": stats.commits,
                 "enrichment_failures": stats.enrichment_failures, **(source_metrics or {})},
    )
    return trace


def main(argv: list[str] | None = None) -> int:
    # GitHub titles may contain characters the Windows console code page
    # cannot print. The UTF-8 trace itself always retains the original text.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description="Build a local GitHub graph trace (no hosted services)")
    parser.add_argument("repository", help="GitHub owner/repo")
    parser.add_argument("question", help="question about recent repository activity")
    parser.add_argument("--prs", type=int, default=15)
    parser.add_argument("--issues", type=int, default=15)
    parser.add_argument("--commits", type=int, default=20)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--files", action="store_true", help="also fetch files changed by each PR")
    parser.add_argument("--reviews", action="store_true", help="also fetch PR reviews")
    parser.add_argument("--source", action="store_true", help="index a bounded sample of source files at HEAD")
    parser.add_argument("--max-source-files", type=int, default=20,
                        help="maximum source files to fetch with --source (default: 20)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not REPO_PATTERN.fullmatch(args.repository) or any(part in {".", ".."} for part in args.repository.split("/")):
        parser.error("repository must be owner/repo")
    if min(args.prs, args.issues, args.commits) < 0 or args.top < 1:
        parser.error("counts must be nonnegative and --top must be positive")
    if not args.question.strip():
        parser.error("question must contain text")
    if not 1 <= args.max_source_files <= 200:
        parser.error("--max-source-files must be between 1 and 200")
    started_at = datetime.now(timezone.utc)
    began = time.perf_counter()
    try:
        source_metrics = None
        with make_session() as session:
            graph, documents, stats = fetch_graph(
                args.repository, prs=args.prs, issues=args.issues,
                commits=args.commits, files=args.files, reviews=args.reviews,
                session=session,
            )
            if args.source:
                from .ingestion.source import index_source

                source_metrics = index_source(session, args.repository, args.question,
                                              graph, documents, max_files=args.max_source_files)
        ingestion_ms = (time.perf_counter() - began) * 1000
        trace = build_trace(args.repository, args.question, graph, documents, stats,
                            started_at=started_at, ingestion_ms=ingestion_ms, top=args.top,
                            source_metrics=source_metrics)
        destination = save(trace, args.out)
    except (GitHubError, OSError, ValueError) as exc:
        print(f"graphrag-github-trace: {exc}", file=sys.stderr)
        return 1
    print(trace.answer)
    print(f"Wrote {destination} (v4; {len(trace.items)} nodes, {len(trace.edges)} edges)")
    print(f"View with: graphrag {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
