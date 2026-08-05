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
from .schema import Trace
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
        ]
    )


def render_file(path: str | Path, *, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Render a trace stored on disk."""
    return render(load(path), threshold=threshold)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m src.tracing <trace.json>", file=sys.stderr)
        return 2
    try:
        print(render_file(args[0]))
    except FileNotFoundError:
        print(f"no such trace file: {args[0]}", file=sys.stderr)
        return 1
    except (ValueError, KeyError) as exc:
        print(f"not a readable trace: {exc}", file=sys.stderr)
        return 1
    return 0
