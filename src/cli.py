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
    in_ingest_order,
    make_session,
)
from .knowledge import GraphBuilder
from .knowledge.documents import source_documents
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

        # The one sort point in the pipeline. Everything below — enrichment
        # order, node insertion order, and therefore which surface form becomes
        # an entity's label — follows from here, so it must not depend on the
        # order GitHub happened to return things in. See ingestion/order.py.
        pull_requests = in_ingest_order(
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

        # Materialised rather than passed as generators straight into build().
        # The bodies these carry are needed twice — once by node construction
        # and once to build source documents — and a generator can only be
        # walked once. The sort order is unchanged; only the point at which
        # the sequence is realised moves.
        issues = in_ingest_order(
            fetch_issues(session, args.repository, limit=args.issues)
        )
        commits = in_ingest_order(
            fetch_commits(session, args.repository, limit=args.commits)
        )

        builder = GraphBuilder()
        stats = builder.build(
            repository=repository,
            pull_requests=pull_requests,
            issues=issues,
            commits=commits,
            reviews=reviews,
            changed_files=changed_files,
            enrichment_failures_by_stage={
                STAGE_REVIEWS: review_failures,
                STAGE_CHANGED_FILES: file_failures,
            },
        )

        # Built from the payloads, not from the graph: node construction does
        # not carry bodies, so this is the only place the prose still exists.
        source_text = source_documents(
            pull_requests=pull_requests,
            issues=issues,
            commits=commits,
            reviews=reviews,
        )
    except GitHubError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if owned_session:
            session.close()

    if args.output and not _write_graph(builder, stats, args.output):
        return 1

    if args.store and not _write_store(
        builder, args.store, embed=args.embed, documents=source_text
    ):
        return 1

    _report(args.repository, stats)
    return 0


def _trace(args: argparse.Namespace, session: httpx.Client | None) -> int:
    """Fetch, run a traced agent, and write both the trace and its canvas form."""
    from .github_trace import run

    return run(args)


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
    except OSError as exc:
        print(f"could not read {args.trace}: {exc}", file=sys.stderr)
        return 1
    except (ValueError, TypeError) as exc:
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


def _write_store(
    builder: GraphBuilder, destination: str, *, embed: bool, documents=()
) -> bool:
    """Write the graph to a store. Returns False if it could not be written.

    Imported here rather than at module scope on purpose. The store needs a
    database driver and embedding needs a model, and neither is a dependency of
    ``ingest --output`` or of ``view``. A top-level import would make every
    invocation of this CLI pay for both, including the ones that never touch a
    store.

    The vector index is built after every write, in one pass. Indexing costs a
    pass over the rows, so doing it once at the end is one pass rather than one
    per batch, and it lets the two costs be timed apart.
    """
    from .graphdb import open_context_graph
    from .knowledge.persist import persist

    embedder = None
    if embed:
        from .analysis import Similarity

        embedder = Similarity()

    from .analysis import Extractor

    # Variants of one entity are folded together as the text is read, with
    # the same similarity the store embeds by and a model asked about the
    # close calls. No model configured means close calls stay apart; no
    # similarity (--no-embed) means nothing is folded.
    resolver = None
    if embedder is not None:
        from .analysis import Resolver
        from .common.judge import JudgeError, MergeJudge

        try:
            judge = MergeJudge()
        except (JudgeError, ImportError) as exc:
            print(f"  entity merging: close calls kept apart ({exc})", file=sys.stderr)
            judge = None
        resolver = Resolver(embedder, judge)

    try:
        store = open_context_graph(destination)
    except Exception as exc:
        print(f"could not open {destination}: {exc}", file=sys.stderr)
        return False

    try:
        written = persist(
            store,
            builder,
            embedder=embedder,
            extractor=Extractor("none"),
            documents=documents,
            resolver=resolver,
        )
        store.build_vector_index(rebuild=True)
    except Exception as exc:
        print(f"could not write {destination}: {exc}", file=sys.stderr)
        return False
    finally:
        store.close()

    print(f"  store {destination}")
    print(
        f"    {written.entities} entities "
        f"({written.entities_from_text} from document text), "
        f"{written.relationships} relationships"
    )
    print(
        f"    {written.documents} documents, {written.mentions} mentions, "
        f"{written.embedded} embedded"
    )
    if resolver is not None:
        merged = resolver.stats
        print(
            f"    {merged.variant_merges} variants merged "
            f"({merged.fast_merges} outright, {merged.model_merges} by the model, "
            f"{merged.model_rejections} kept apart after asking)"
        )
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
    # Independent of --output. Both may be given, and the two destinations
    # receive the same build, so a JSON file and a store written in one run are
    # comparable row for row.
    ingest.add_argument(
        "--store", metavar="PATH", help="write the graph to a graph store"
    )
    ingest.add_argument(
        "--no-embed",
        dest="embed",
        action="store_false",
        help="write the store without entity vectors, skipping the model load",
    )
    ingest.set_defaults(handler=_ingest, reviews=True, files=True, embed=True)

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

    trace = commands.add_parser(
        "trace",
        help="fetch a repository, run a traced agent over it, and write the run out",
    )
    # The arguments live beside the command's implementation; importing that
    # module needs nothing beyond the standard library until it runs.
    from .github_trace import add_arguments as _trace_arguments

    _trace_arguments(trace)
    trace.set_defaults(handler=_trace)

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
