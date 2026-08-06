"""Live demo: the same workflow against a real GitHub repository.

    export GITHUB_TOKEN=...
    python -m examples.live_demo acme/checkout --query "who changed auth recently?"

Structurally identical to ``offline_demo`` — same graph builder, same workflow,
same four SDK calls. Only where the payloads come from differs, which is the
point: swapping the data source touches nothing downstream.

Deliberately duplicates the instrumentation block rather than sharing it with
the offline demo. An example is read top to bottom, and a reader should see the
whole flow in one file instead of chasing a helper.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx

from src.ingestion import (
    GitHubError,
    STAGE_CHANGED_FILES,
    STAGE_REVIEWS,
    collect_by_pull_request,
    fetch_changed_files,
    fetch_commits,
    fetch_issues,
    fetch_pull_requests,
    fetch_repository,
    fetch_reviews,
    make_session,
)
from src.knowledge import GraphBuilder
from src.tracing import Trace, capture, score_overlaps, render, save

from .workflow import answer, retrieve

DEFAULT_OUTPUT = Path("live_trace.json")
DEFAULT_QUERY = "who changed authentication recently?"


def _repository(value: str) -> str:
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a repository; expected owner/name"
        )
    return value


def build_graph(
    session: httpx.Client,
    repo: str,
    *,
    prs: int,
    issues: int,
    commits: int,
    reviews: bool,
    files: bool,
) -> GraphBuilder:
    """Fetch a repository and build its graph.

    Pull requests are materialized because reviews and changed files are
    requested per pull request — one extra call each, which is why both are
    switchable off.
    """
    pull_requests = list(fetch_pull_requests(session, repo, limit=prs))
    numbers = [pr.number for pr in pull_requests]

    # One failed enrichment call drops that pull request's extras, not the run.
    review_map, review_failures = (
        collect_by_pull_request(fetch_reviews, session, repo, numbers)
        if reviews
        else ({}, 0)
    )
    file_map, file_failures = (
        collect_by_pull_request(fetch_changed_files, session, repo, numbers)
        if files
        else ({}, 0)
    )

    builder = GraphBuilder()
    builder.build(
        repository=fetch_repository(session, repo),
        pull_requests=pull_requests,
        issues=fetch_issues(session, repo, limit=issues),
        commits=fetch_commits(session, repo, limit=commits),
        reviews=review_map,
        changed_files=file_map,
        enrichment_failures_by_stage={
            STAGE_REVIEWS: review_failures,
            STAGE_CHANGED_FILES: file_failures,
        },
    )
    return builder


def run(
    session: httpx.Client,
    repo: str,
    query: str,
    *,
    prs: int,
    issues: int,
    commits: int,
    reviews: bool,
    files: bool,
    limit: int,
    output: Path | None,
) -> tuple[Trace, GraphBuilder]:
    started_at = datetime.now(timezone.utc)
    began = perf_counter()

    graph = build_graph(
        session,
        repo,
        prs=prs,
        issues=issues,
        commits=commits,
        reviews=reviews,
        files=files,
    )

    # --- your workflow, unaware of tracing ---------------------------------
    items, edges = retrieve(graph.nodes, graph.edges, query, limit=limit)
    response = answer(items)

    # --- the SDK, four calls ----------------------------------------------
    trace = capture(
        query,
        items,
        response,
        edges=edges,
        started_at=started_at,
        duration_ms=round((perf_counter() - began) * 1000, 1),
    )
    score_overlaps(trace)
    if output is not None:
        save(trace, output)
    return trace, graph


def main(argv: list[str] | None = None, *, session: httpx.Client | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="live_demo",
        description="Trace a retrieval workflow over a real GitHub repository.",
        epilog="Authentication is read from $GITHUB_TOKEN. Without one, GitHub "
        "allows 60 requests an hour, which is enough for a small --prs value.",
    )
    parser.add_argument("repository", type=_repository, help="owner/name")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--prs", type=int, default=20)
    parser.add_argument("--issues", type=int, default=20)
    parser.add_argument("--commits", type=int, default=50)
    parser.add_argument("--limit", type=int, default=7, help="items to retrieve")
    parser.add_argument("--no-reviews", dest="reviews", action="store_false")
    parser.add_argument("--no-files", dest="files", action="store_false")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.set_defaults(reviews=True, files=True)
    args = parser.parse_args(argv)

    if session is None and not os.environ.get("GITHUB_TOKEN"):
        print(
            "note: no GITHUB_TOKEN set, so GitHub allows only 60 requests an hour.",
            file=sys.stderr,
        )

    owned = session is None
    session = session or make_session()
    try:
        trace, graph = run(
            session,
            args.repository,
            args.query,
            prs=args.prs,
            issues=args.issues,
            commits=args.commits,
            reviews=args.reviews,
            files=args.files,
            limit=args.limit,
            output=args.output,
        )
    except GitHubError as exc:
        print(f"could not read {args.repository}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"could not write {args.output}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    finally:
        if owned:
            session.close()

    print(
        f"graph: {graph.stats.nodes_created} nodes, "
        f"{graph.stats.edges_created} edges from {args.repository}\n"
    )
    print(render(trace))
    print(f"\nwrote {args.output}")
    print(f"view it with: python -m src.cli view {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
