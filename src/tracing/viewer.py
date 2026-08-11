"""M1 — the minimal trace viewer.

Reads a trace and prints it. Deliberately plain: a header, a table of retrieved
items, and the relations between them.

This exists to prove the schema is sufficient. If something about a run cannot
be shown here, the schema is missing a field — which is why the viewer was
built before anything that writes traces.

    python -m src.tracing example_trace.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from .classify import DEFAULT_THRESHOLD, is_used
from .schema import STATUS_ERROR, Span, Trace
from .store import load

CONTENT_WIDTH = 60


def _label(overlap: float | None, threshold: float) -> str:
    """How an item's fate reads in the table.

    The verdict is computed here rather than read from the trace: the trace
    stores the measurement, and the cutoff is the viewer's to apply.
    """
    used = is_used(overlap, threshold)
    if used is None:
        return "unclassified"
    return "used" if used else "ignored"


def _shorten(text: str, width: int = CONTENT_WIDTH) -> str:
    """One line, no wider than the column.

    The marker is ASCII on purpose: a Windows console under cp1252 raises
    UnicodeEncodeError on a real ellipsis when the output is piped.
    """
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    return flat[: width - 3] + "..."


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _header(trace: Trace, threshold: float) -> list[str]:
    used = sum(1 for item in trace.items if is_used(item.overlap, threshold))
    lines = [
        f"query    {trace.query}",
        f"answer   {_shorten(trace.answer, 100) if trace.answer else '(none)'}",
        f"used     {used} of {len(trace.items)} retrieved",
    ]
    if trace.started_at is not None:
        lines.append(f"started  {trace.started_at.isoformat()}")
    if trace.duration_ms is not None:
        lines.append(f"took     {trace.duration_ms} ms")
    return lines


def _item_table(trace: Trace, threshold: float) -> list[str]:
    if not trace.items:
        return []

    rows = [
        (
            item.id,
            _label(item.overlap, threshold),
            _number(item.score),
            _number(item.overlap),
            item.source,
            _shorten(item.content),
        )
        for item in trace.items
    ]
    headings = ("id", "fate", "score", "overlap", "source", "content")
    widths = [
        max(len(headings[column]), max(len(row[column]) for row in rows))
        for column in range(len(headings))
    ]

    def line(values) -> str:
        return "  ".join(
            value.ljust(widths[i]) for i, value in enumerate(values)
        ).rstrip()

    return [
        "",
        "retrieved",
        line(headings),
        "  ".join("-" * width for width in widths),
        *(line(row) for row in rows),
    ]


def _edge_table(trace: Trace) -> list[str]:
    if not trace.edges:
        return []
    return [
        "",
        "relations",
        *(
            f"  {edge.source} -[{edge.relation} {_number(edge.weight)}]-> {edge.target}"
            for edge in trace.edges
        ),
    ]


def _duration(span: Span) -> str:
    if span.start_ms is None or span.end_ms is None:
        return "-"
    return f"{span.end_ms - span.start_ms:.1f} ms"


def _span_table(trace: Trace) -> list[str]:
    """Every span, with its status, in start order.

    Failures are shown, never filtered. A run that answered while its retriever
    raised looks identical to a clean run in the item table — the span status is
    the only place that difference survives, so hiding it would make the trace
    lie by omission.
    """
    if not trace.spans:
        return []

    # Spans arrive in whatever order they finished. Ordering by start reads as
    # the run happened; a span with no timing sorts last rather than crashing
    # the comparison.
    ordered = sorted(
        trace.spans,
        key=lambda span: (span.start_ms is None, span.start_ms or 0.0),
    )
    depth = _depths(trace.spans)

    rows = [
        (
            span.status,
            "  " * depth[span.id] + span.name,
            span.kind,
            _duration(span),
        )
        for span in ordered
    ]
    headings = ("status", "span", "kind", "took")
    widths = [
        max(len(headings[column]), max(len(row[column]) for row in rows))
        for column in range(len(headings))
    ]

    def line(values) -> str:
        return "  ".join(
            value.ljust(widths[i]) for i, value in enumerate(values)
        ).rstrip()

    failed = sum(1 for span in trace.spans if span.status == STATUS_ERROR)
    heading = "spans" if not failed else f"spans ({failed} failed)"

    return [
        "",
        heading,
        line(headings),
        "  ".join("-" * width for width in widths),
        *(line(row) for row in rows),
    ]


def _depths(spans: list[Span]) -> dict[str, int]:
    """How deep each span sits under its parent.

    A parent named in ``parent_id`` but absent from the trace is treated as a
    root: a partial trace should still render rather than recurse forever, and
    a cycle from a malformed file must not hang the viewer.
    """
    parents = {span.id: span.parent_id for span in spans}
    depths: dict[str, int] = {}
    for span_id in parents:
        depth, cursor, seen = 0, parents[span_id], {span_id}
        while cursor in parents and cursor not in seen:
            seen.add(cursor)
            depth += 1
            cursor = parents[cursor]
        depths[span_id] = depth
    return depths


def render(trace: Trace, *, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Render a trace as plain text, judging items at ``threshold``.

    The trace stores measurements; the cutoff is applied here. Rendering the
    same file at a different threshold reclassifies it, without rewriting it.
    """
    return "\n".join(
        [
            *_header(trace, threshold),
            *_item_table(trace, threshold),
            *_edge_table(trace),
            *_span_table(trace),
        ]
    )


def render_file(path: str | Path, *, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Render a trace stored on disk."""
    return render(load(path), threshold=threshold)


def _program_name() -> str:
    """What to call this command in a usage line.

    Installed as a console script, ``argv[0]`` is the command a person typed.
    Run as ``python -m src.tracing`` it is the package's ``__main__.py``, which
    nobody can type back, so that case names the module form instead.
    """
    name = Path(sys.argv[0]).name
    if not name or name.endswith(".py"):
        return "python -m src.tracing"
    return name


def main(argv: list[str] | None = None) -> int:
    """Render one trace file. The entry point behind the console command.

    Every failure leaves by a ``return``, never by an exception: a person who
    typed a wrong filename gets a sentence, not a stack trace of this package's
    internals.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(f"usage: {_program_name()} <trace.json>", file=sys.stderr)
        return 2
    try:
        print(render_file(args[0]))
    except FileNotFoundError:
        print(f"no such trace file: {args[0]}", file=sys.stderr)
        return 1
    except OSError as exc:
        # A directory, a permission denial, an unreadable device.
        print(f"could not read {args[0]}: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError) as exc:
        # ValueError covers malformed JSON and an unknown schema version;
        # KeyError a missing required field; TypeError a file whose JSON is
        # valid but is not an object at all.
        print(f"not a readable trace: {exc}", file=sys.stderr)
        return 1
    return 0
