"""Tests for M1, the trace viewer.

The viewer exists to prove the M0 schema is sufficient — if something cannot be
shown, the schema is missing a field. So these tests assert that every part of a
trace reaches the output, not that the output is pretty.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.tracing import (
    STATUS_ERROR,
    STATUS_OK,
    Span,
    Trace,
    TraceEdge,
    TraceItem,
    capture,
    render,
    render_file,
    save,
)
from src.tracing.viewer import main

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "example_trace.json"


def make_trace() -> Trace:
    return _trace(
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



def _trace(*, query: str = "", answer=None, items=(), edges=(), **kwargs):
    """Build a flat trace the way a one-shot producer does.

    Schema 3 nests items under a Retrieval, and ``capture`` is the supported
    way to make one from a flat list — so these tests exercise the real path
    instead of assembling the dataclass by hand.
    """
    return capture(query, items, answer, edges=edges, **kwargs)



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
    trace = _trace(
        query="q",
        items=[TraceItem(id="pr:1", content="x", source="graph", score=0.5)],
    )

    output = render(trace)

    assert "unclassified" in output.lower()


def test_a_trace_with_no_items_still_renders():
    output = render(_trace(query="nothing matched"))

    assert "nothing matched" in output
    assert "0 of 0" in output


def test_a_trace_with_no_answer_still_renders():
    trace = _trace(
        query="q", items=[TraceItem(id="pr:1", content="x", source="graph")]
    )

    output = render(trace)

    assert "pr:1" in output


def test_a_trace_with_no_edges_omits_the_relations_section():
    output = render(_trace(query="q"))

    assert "RESOLVES" not in output


def test_long_content_is_truncated_to_keep_rows_readable():
    trace = _trace(
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
    # The example answers despite a failed retrieval arm, so the run reads as
    # complete in every section but the spans. That is the case the span table
    # exists for.
    assert "vector_retrieve" in output
    assert "1 failed" in output


# --------------------------------------------------------------------------
# spans — including the ones that failed
# --------------------------------------------------------------------------


def _spanned(*spans: Span) -> Trace:
    trace = _trace(query="q", answer="a")
    trace.spans.extend(spans)
    return trace


def test_span_names_are_shown():
    output = render(
        _spanned(Span(id="s1", name="retrieve", kind="retriever", status=STATUS_OK))
    )

    assert "retrieve" in output


def test_a_failed_span_appears_with_its_error_state():
    """A failure must not be filtered out of the output. It is the whole point."""
    output = render(
        _spanned(
            Span(id="s1", name="retrieve", kind="retriever", status=STATUS_OK),
            Span(id="s2", name="generate", kind="llm", status=STATUS_ERROR),
        )
    )

    assert "generate" in output
    assert STATUS_ERROR in output
    assert "1 failed" in output


def test_span_timing_is_shown():
    output = render(
        _spanned(
            Span(
                id="s1",
                name="retrieve",
                kind="retriever",
                status=STATUS_OK,
                start_ms=10.0,
                end_ms=52.5,
            )
        )
    )

    assert "42.5 ms" in output


def test_a_trace_with_no_spans_omits_the_span_section():
    assert "spans" not in render(_trace(query="q"))


def test_a_span_whose_parent_is_missing_still_renders():
    """A partial trace is renderable; a broken parent link is not fatal."""
    output = render(
        _spanned(Span(id="s2", name="orphan", kind="llm", parent_id="gone"))
    )

    assert "orphan" in output


def test_a_cycle_in_span_parents_does_not_hang():
    output = render(
        _spanned(
            Span(id="s1", name="first", kind="chain", parent_id="s2"),
            Span(id="s2", name="second", kind="chain", parent_id="s1"),
        )
    )

    assert "first" in output
    assert "second" in output


# --------------------------------------------------------------------------
# the overlap measurement reaches the screen
# --------------------------------------------------------------------------


def test_the_overlap_measurement_is_shown():
    output = render(make_trace())

    assert "0.41" in output
    assert "0.05" in output


def test_the_threshold_is_applied_at_render_time_not_stored():
    """The same file, judged twice, classifies differently. Nothing is rewritten."""
    trace = make_trace()

    assert "1 of 2" in render(trace, threshold=0.2)
    assert "0 of 2" in render(trace, threshold=0.9)
    assert [item.overlap for item in trace.items] == [0.41, 0.05]


# --------------------------------------------------------------------------
# failures a person can act on — no traceback reaches the terminal
# --------------------------------------------------------------------------


def test_a_missing_file_reports_clearly(capsys):
    code = main(["no-such-trace.json"])

    assert code == 1
    assert "no such trace file" in capsys.readouterr().err


def test_a_malformed_file_reports_clearly(tmp_path, capsys):
    path = tmp_path / "broken.json"
    path.write_text("{not json at all", encoding="utf-8")

    code = main([str(path)])

    assert code == 1
    assert "not a readable trace" in capsys.readouterr().err


def test_a_file_of_valid_json_that_is_not_a_trace_reports_clearly(
    tmp_path, capsys
):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    code = main([str(path)])

    assert code == 1
    assert "not a readable trace" in capsys.readouterr().err


def test_a_trace_missing_a_required_field_reports_clearly(tmp_path, capsys):
    path = tmp_path / "no-query.json"
    path.write_text('{"schema_version": 3}', encoding="utf-8")

    code = main([str(path)])

    assert code == 1
    assert "not a readable trace" in capsys.readouterr().err


def test_a_directory_instead_of_a_file_reports_clearly(tmp_path, capsys):
    code = main([str(tmp_path)])

    assert code == 1
    assert "could not read" in capsys.readouterr().err


def test_wrong_argument_count_prints_usage(capsys):
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["{not json at all", "[1, 2, 3]", ""])
def test_no_traceback_ever_reaches_the_user(tmp_path, capsys, bad):
    path = tmp_path / "bad.json"
    path.write_text(bad, encoding="utf-8")

    main([str(path)])

    assert "Traceback" not in capsys.readouterr().err


# --------------------------------------------------------------------------
# the viewer reads; it never writes
# --------------------------------------------------------------------------


def test_rendering_a_file_does_not_modify_it(tmp_path):
    path = tmp_path / "trace.json"
    save(make_trace(), path)
    before = path.read_bytes()

    render_file(path)

    assert path.read_bytes() == before


def test_the_viewer_module_has_no_write_path():
    """No save, no open-for-write, no mkdir. Enforced, not just intended."""
    import ast
    from pathlib import Path as _Path

    source = (_Path(__file__).resolve().parents[1] / "tracing" / "viewer.py").read_text(
        encoding="utf-8"
    )
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }

    assert not called & {"save", "write_text", "write_bytes", "mkdir", "open", "unlink"}
