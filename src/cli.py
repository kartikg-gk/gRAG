"""Command-line entrypoint.

A launcher and nothing more: it parses arguments, resolves them into calls on
the modules that do the work, and turns failures into messages a person can act
on. No fetching, no graph building, no scoring happens here.

    python -m src.cli ingest octocat/Hello-World
    python -m src.cli view example_trace.json

Two commands, because two things exist to run. There is no server to start and
no client to configure, so there is nothing to orchestrate beyond these.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from .ingestion import (
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
from .knowledge import GraphBuilder
from .tracing import DEFAULT_THRESHOLD, load, render

DEFAULT_PRS = 50
DEFAULT_ISSUES = 50
DEFAULT_COMMITS = 100


# --------------------------------------------------------------------------
# argument types
# --------------------------------------------------------------------------


def _repository(value: str) -> str:
    """Accept ``owner/name`` and nothing else."""
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a repository; expected owner/name"
        )
    return value


def _threshold(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return number


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _ingest(args: argparse.Namespace, session: httpx.Client | None) -> int:
    """Fetch a repository and build its graph."""
    owned_session = session is None
    session = session or make_session()

    try:
        repository = fetch_repository(session, args.repository)
        pull_requests = list(
            fetch_pull_requests(session, args.repository, limit=args.prs)
        )
        numbers = [pr.number for pr in pull_requests]

        # Enrichment is one call per pull request, so a long run has many
        # chances to fail. A failure drops that pull request's extras and the
        # run continues; the count is recorded rather than swallowed.
        reviews, review_failures = (
            collect_by_pull_request(
                fetch_reviews, session, args.repository, numbers
            )
            if args.reviews
            else ({}, 0)
        )
        changed_files, file_failures = (
            collect_by_pull_request(
                fetch_changed_files, session, args.repository, numbers
            )
            if args.files
            else ({}, 0)
        )

        builder = GraphBuilder()
        stats = builder.build(
            repository=repository,
            pull_requests=pull_requests,
            issues=fetch_issues(session, args.repository, limit=args.issues),
            commits=fetch_commits(session, args.repository, limit=args.commits),
            reviews=reviews,
            changed_files=changed_files,
            enrichment_failures_by_stage={
                STAGE_REVIEWS: review_failures,
                STAGE_CHANGED_FILES: file_failures,
            },
        )
    except GitHubError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if owned_session:
            session.close()

    if args.output and not _write_graph(builder, stats, args.output):
        return 1

    _report(args.repository, stats)
    return 0


def _view(args: argparse.Namespace, session: httpx.Client | None) -> int:
    """Render a saved trace.

    A trace stores measurements, not verdicts, so viewing never rewrites it.
    ``--threshold`` changes only how the same file is judged on screen.
    """
    try:
        trace = load(args.trace)
    except FileNotFoundError:
        print(f"no such trace file: {args.trace}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"not a readable trace: {exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"not a readable trace: missing field {exc}", file=sys.stderr)
        return 1

    print(render(trace, threshold=args.threshold))
    return 0


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_graph(builder: GraphBuilder, stats, destination: str) -> bool:
    """Write the graph to disk. Returns False if it could not be written."""
    payload = {
        "nodes": list(builder.nodes.values()),
        "edges": builder.edges,
        "stats": vars(stats),
    }
    try:
        Path(destination).write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"could not write {destination}: {exc}", file=sys.stderr)
        return False
    return True


def _report(repository: str, stats) -> None:
    print(f"{repository}")
    print(f"  {stats.nodes_created} nodes, {stats.edges_created} edges")
    print(
        f"  {stats.pull_requests} pull requests, {stats.issues} issues, "
        f"{stats.commits} commits"
    )
    print(f"  {stats.reviews} reviews, {stats.changed_files} changed files")

    for label, count in (
        ("enrichment calls failed and skipped", stats.enrichment_failures),
        *(
            (f"of those, in {stage}", count)
            for stage, count in sorted(stats.enrichment_failures_by_stage.items())
        ),
        ("commits with no linked account", stats.commits_without_author),
        ("self-reviews skipped", stats.self_reviews_skipped),
        ("unresolved references", stats.unresolved_references),
        ("orphan associations", stats.orphan_associations),
        ("duplicate nodes skipped", stats.duplicate_nodes_skipped),
        ("duplicate edges skipped", stats.duplicate_edges_skipped),
    ):
        if count:
            print(f"  {count} {label}")


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphrag", description="Build and inspect a repository knowledge graph."
    )
    commands = parser.add_subparsers(dest="command")

    ingest = commands.add_parser(
        "ingest", help="fetch a GitHub repository and build its graph"
    )
    ingest.add_argument("repository", type=_repository, help="owner/name")
    # Limits default to something finite: a large repository has thousands of
    # pull requests, and an unbounded default would spend a rate limit by
    # accident on the first run.
    ingest.add_argument(
        "--prs", type=int, default=DEFAULT_PRS, help=f"default {DEFAULT_PRS}"
    )
    ingest.add_argument(
        "--issues", type=int, default=DEFAULT_ISSUES, help=f"default {DEFAULT_ISSUES}"
    )
    ingest.add_argument(
        "--commits",
        type=int,
        default=DEFAULT_COMMITS,
        help=f"default {DEFAULT_COMMITS}",
    )
    # Reviews and files are one request per pull request each, so they dominate
    # the request count. On by default because they carry REVIEWED and TOUCHES;
    # switchable off when a quick structural pass is all that is wanted.
    ingest.add_argument(
        "--no-reviews", dest="reviews", action="store_false", help="skip reviews"
    )
    ingest.add_argument(
        "--no-files", dest="files", action="store_false", help="skip changed files"
    )
    ingest.add_argument("--output", metavar="PATH", help="write the graph as JSON")
    ingest.set_defaults(handler=_ingest, reviews=True, files=True)

    view = commands.add_parser("view", help="render a saved trace")
    view.add_argument("trace", help="path to a trace JSON file")
    view.add_argument(
        "--threshold",
        type=_threshold,
        default=DEFAULT_THRESHOLD,
        metavar="N",
        help=(
            f"used/ignored cutoff, default {DEFAULT_THRESHOLD}. The trace stores "
            "the measurement; this only changes how it is judged on screen."
        ),
    )
    view.set_defaults(handler=_view)

    return parser


def main(argv: list[str] | None = None, *, session: httpx.Client | None = None) -> int:
    """Run one command.

    ``session`` exists so tests can drive ``ingest`` against a fake transport,
    matching how the ingestion module already takes a session as an argument.
    Left unset, the CLI builds one and closes it.

    There is no token flag on purpose: a token passed on the command line lands
    in shell history and in the process list, so it is read from ``$GITHUB_TOKEN``
    only.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_usage(sys.stderr)
        print("error: a command is required", file=sys.stderr)
        return 2

    try:
        return args.handler(args, session)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
