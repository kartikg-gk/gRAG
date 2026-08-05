"""Offline demo: the canonical hello-world for the tracing SDK.

No token, no network, no model. Run it and look at what comes out:

    python -m examples.offline_demo

It writes ``examples/demo_trace.json`` and prints the rendered trace. The same
file opens in the project's viewer:

    python -m src.cli view examples/demo_trace.json

Every run produces a byte-identical artifact, so the file can be committed and
diffed.

Read this file top to bottom: the four SDK calls at the end of ``run`` are the
entire integration. Note that ``examples/workflow.py`` — the part you would
replace with your own code — imports nothing from ``src.tracing``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.models import (
    ChangedFile,
    Commit,
    Issue,
    PullRequest,
    Repository,
    Review,
)
from src.knowledge import GraphBuilder
from src.tracing import Trace, capture, score_overlaps, render, save

from . import fixtures
from .workflow import answer, retrieve

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "demo_trace.json"

# A real integration passes the clock: capture(..., started_at=datetime.now(UTC),
# duration_ms=elapsed). This demo fixes the start and omits the duration so the
# artifact is identical on every run — a measured duration would change the file
# each time and make it useless to diff or commit.
FIXED_START = datetime(2024, 9, 1, 12, 0, 0, tzinfo=timezone.utc)


def build_graph() -> GraphBuilder:
    """Turn the canned payloads into a graph.

    Identical to what the live demo does — only the source of the payloads
    differs. The models validate the fixtures exactly as they validate real API
    responses, so a mistake in the fixtures fails here rather than silently
    producing a wrong graph.
    """
    builder = GraphBuilder()
    builder.build(
        repository=Repository.model_validate(fixtures.REPOSITORY),
        pull_requests=[PullRequest.model_validate(p) for p in fixtures.PULL_REQUESTS],
        # The issues fixture includes a pull request, as GitHub's endpoint does.
        # Filtering it is the ingestion layer's job, mirrored here.
        issues=[
            Issue.model_validate(i)
            for i in fixtures.ISSUES
            if "pull_request" not in i
        ],
        commits=[Commit.model_validate(c) for c in fixtures.COMMITS],
        reviews={
            number: [Review.model_validate(r) for r in reviews]
            for number, reviews in fixtures.REVIEWS.items()
        },
        changed_files={
            number: [ChangedFile.model_validate(f) for f in files]
            for number, files in fixtures.CHANGED_FILES.items()
        },
    )
    return builder


def run(query: str, *, limit: int = 7, output: Path | None = None) -> Trace:
    """Run the workflow and trace it."""
    graph = build_graph()

    # --- your workflow, unaware of tracing ---------------------------------
    items, edges = retrieve(graph.nodes, graph.edges, query, limit=limit)
    response = answer(items)

    # --- the SDK, four calls ----------------------------------------------
    trace = capture(query, items, response, edges=edges, started_at=FIXED_START)
    score_overlaps(trace)
    if output is not None:
        save(trace, output)
    return trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="offline_demo",
        description="Trace a retrieval workflow over a canned corpus. No network.",
    )
    parser.add_argument("--query", default=fixtures.DEFAULT_QUERY)
    parser.add_argument("--limit", type=int, default=7, help="items to retrieve")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="where to write the trace"
    )
    args = parser.parse_args(argv)

    try:
        trace = run(args.query, limit=args.limit, output=args.output)
    except OSError as exc:
        print(f"could not write {args.output}: {exc}", file=sys.stderr)
        return 1

    print(render(trace))
    print(f"\nwrote {args.output}")
    print(f"view it with: python -m src.cli view {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
