"""Tests for M1, the trace viewer.

The viewer exists to prove the M0 schema is sufficient — if something cannot be
shown, the schema is missing a field. So these tests assert that every part of a
trace reaches the output, not that the output is pretty.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.tracing import Trace, TraceEdge, TraceItem, render, render_file, save

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "example_trace.json"


def make_trace() -> Trace:
    return Trace(
        query="who changed authentication recently?",
        answer="Alice changed it in PR #1347.",
        started_at=datetime(2026, 8, 2, 10, 0, 0, tzinfo=timezone.utc),
        duration_ms=412.5,
        items=[
            TraceItem(
                id="pr:1347",
                content="Rewrite the auth middleware token check.",
                source="graph",
                score=0.91,
                overlap=0.41,
            ),
            TraceItem(
                id="pr:1290",
                content="Bump the auth library.",
                source="vector",
                score=0.58,
                overlap=0.05,
            ),
        ],
        edges=[
            TraceEdge(
                source="pr:1347", target="issue:42", relation="RESOLVES", weight=0.92
            )
        ],
    )


# --------------------------------------------------------------------------
# what has to appear
# --------------------------------------------------------------------------


def test_the_query_is_shown():
    output = render(make_trace())

    assert "who changed authentication recently?" in output


def test_the_answer_is_shown():
    output = render(make_trace())

    assert "Alice changed it in PR #1347." in output


def test_every_item_id_is_shown():
    output = render(make_trace())

    assert "pr:1347" in output
    assert "pr:1290" in output


def test_item_scores_are_shown():
    output = render(make_trace())

    assert "0.91" in output
    assert "0.58" in output


def test_item_sources_are_shown():
    output = render(make_trace())

    assert "graph" in output
    assert "vector" in output


def test_item_content_is_shown():
    output = render(make_trace())

    assert "Rewrite the auth middleware" in output


def test_used_and_ignored_are_distinguishable():
    output = render(make_trace())

    assert "used" in output.lower()
    assert "ignored" in output.lower()


def test_edges_are_shown():
    output = render(make_trace())

    assert "RESOLVES" in output
    assert "issue:42" in output


def test_timing_is_shown():
    output = render(make_trace())

    assert "412.5" in output


def test_a_summary_counts_used_against_retrieved():
    output = render(make_trace())

    assert "1 of 2" in output


# --------------------------------------------------------------------------
# edge cases the schema allows
# --------------------------------------------------------------------------


def test_an_unclassified_item_is_not_called_ignored():
    trace = Trace(
        query="q",
        items=[TraceItem(id="pr:1", content="x", source="graph", score=0.5)],
    )

    output = render(trace)

    assert "unclassified" in output.lower()


def test_a_trace_with_no_items_still_renders():
    output = render(Trace(query="nothing matched"))

    assert "nothing matched" in output
    assert "0 of 0" in output


def test_a_trace_with_no_answer_still_renders():
    trace = Trace(
        query="q", items=[TraceItem(id="pr:1", content="x", source="graph")]
    )

    output = render(trace)

    assert "pr:1" in output


def test_a_trace_with_no_edges_omits_the_relations_section():
    output = render(Trace(query="q"))

    assert "RESOLVES" not in output


def test_long_content_is_truncated_to_keep_rows_readable():
    trace = Trace(
        query="q",
        items=[TraceItem(id="pr:1", content="x" * 500, source="graph")],
    )

    output = render(trace)

    assert "x" * 500 not in output
    assert "..." in output


# --------------------------------------------------------------------------
# reading from disk
# --------------------------------------------------------------------------


def test_render_file_reads_a_trace_from_disk(tmp_path):
    path = tmp_path / "trace.json"
    save(make_trace(), path)

    output = render_file(path)

    assert "who changed authentication recently?" in output


def test_the_hand_written_example_renders():
    output = render_file(EXAMPLE_PATH)

    assert "who changed authentication recently?" in output
    assert "pr:1347" in output
    assert "3 of 7" in output
    assert "AUTHORED" in output
