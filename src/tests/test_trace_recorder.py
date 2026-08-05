"""Tests for accumulation, spans, and persistence.

Two structural properties are checked hardest: a complete trace can be built
without touching the filesystem, and failed work survives into the output. A
trace that silently drops a failed step makes a broken run look like a short
one.
"""

from __future__ import annotations

import json

import pytest

from src.tracing import (
    ARM_UNKNOWN,
    ARM_VECTOR,
    EMPTY_SLUG,
    KIND_LLM,
    KIND_RETRIEVER,
    PRODUCER_UNKNOWN,
    SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_OK,
    Recorder,
    TraceItem,
    default_path,
    finish_and_save,
    load,
    save,
    slugify,
    to_dict,
    trace_dir,
    trace_filename,
)

ITEMS = [
    {"id": "pr:1", "content": "alpha bravo", "source": "graph", "score": 0.9},
    {"id": "pr:2", "content": "zulu", "source": "vector", "score": 0.2},
]


# ==========================================================================
# Accumulation happens without a filesystem
# ==========================================================================


def test_a_complete_trace_is_built_without_touching_the_disk():
    recorder = Recorder(producer="test-pipeline", arm=ARM_VECTOR)

    with recorder.span("retrieve", kind=KIND_RETRIEVER):
        recorder.record_query("why does login fail?")
        recorder.record_items(ITEMS)
    with recorder.span("answer", kind=KIND_LLM):
        recorder.record_answer("because the token expired")

    trace = recorder.finish()

    assert trace.query == "why does login fail?"
    assert trace.answer == "because the token expired"
    assert [item.id for item in trace.items] == ["pr:1", "pr:2"]
    assert trace.producer == "test-pipeline"
    assert trace.arm == ARM_VECTOR


def test_the_accumulator_takes_no_path_argument():
    """Persistence is a separate module; accumulation must not know about it."""
    import inspect

    for method in (Recorder.finish, Recorder.__init__, Recorder.span):
        assert "path" not in inspect.signature(method).parameters


def test_the_accumulator_does_not_import_persistence():
    """Checked structurally, so a convenience import cannot quietly re-couple."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "tracing" / "capture.py").read_text(
        encoding="utf-8"
    )
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }

    assert "store" not in imported
    assert "json" not in imported
    assert "pathlib" not in imported


def test_defaults_are_recorded_when_nothing_is_configured():
    trace = Recorder().finish()

    assert trace.producer == PRODUCER_UNKNOWN
    assert trace.arm == ARM_UNKNOWN


# ==========================================================================
# Spans
# ==========================================================================


def test_a_span_records_its_shape():
    recorder = Recorder()

    with recorder.span("retrieve", kind=KIND_RETRIEVER):
        pass

    span = recorder.finish().spans[0]
    assert span.name == "retrieve"
    assert span.kind == KIND_RETRIEVER
    assert span.status == STATUS_OK
    assert span.parent_id is None
    assert span.start_ms is not None
    assert span.end_ms is not None
    assert span.end_ms >= span.start_ms


def test_a_nested_span_names_its_parent():
    """Without parent_id the trace is flat and loses which step owned what."""
    recorder = Recorder()

    with recorder.span("agent") as outer:
        with recorder.span("retrieve", kind=KIND_RETRIEVER) as inner:
            pass

    assert inner.parent_id == outer.id
    assert outer.parent_id is None


def test_siblings_share_a_parent():
    recorder = Recorder()

    with recorder.span("agent") as outer:
        with recorder.span("first") as one:
            pass
        with recorder.span("second") as two:
            pass

    assert one.parent_id == outer.id
    assert two.parent_id == outer.id


def test_a_span_closed_is_not_still_the_parent():
    recorder = Recorder()

    with recorder.span("first"):
        pass
    with recorder.span("second") as second:
        pass

    assert second.parent_id is None


def test_span_ids_are_unique():
    recorder = Recorder()

    for name in ("a", "b", "c"):
        with recorder.span(name):
            pass

    ids = [span.id for span in recorder.finish().spans]
    assert len(set(ids)) == len(ids)


def test_spans_are_recorded_in_the_order_they_started():
    recorder = Recorder()

    with recorder.span("outer"):
        with recorder.span("inner"):
            pass

    assert [span.name for span in recorder.finish().spans] == ["outer", "inner"]


# ==========================================================================
# Failures survive
# ==========================================================================


def test_a_failed_step_appears_in_the_output_with_error_status():
    recorder = Recorder()

    with pytest.raises(RuntimeError):
        with recorder.span("retrieve", kind=KIND_RETRIEVER):
            raise RuntimeError("index offline")

    trace = recorder.finish()
    assert len(trace.spans) == 1
    assert trace.spans[0].name == "retrieve"
    assert trace.spans[0].status == STATUS_ERROR


def test_a_failed_span_is_still_closed_in_time():
    recorder = Recorder()

    with pytest.raises(RuntimeError):
        with recorder.span("retrieve"):
            raise RuntimeError("boom")

    span = recorder.finish().spans[0]
    assert span.end_ms is not None
    assert span.end_ms >= span.start_ms


def test_a_failure_propagates_rather_than_being_swallowed():
    """Recording a failure is not the same as handling it."""
    recorder = Recorder()

    with pytest.raises(ValueError, match="specific"):
        with recorder.span("step"):
            raise ValueError("specific message")


def test_a_failure_inside_a_step_does_not_lose_the_surrounding_work():
    recorder = Recorder()

    with recorder.span("agent"):
        recorder.record_query("q")
        try:
            with recorder.span("retrieve", kind=KIND_RETRIEVER):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        recorder.record_answer("answered anyway")

    trace = recorder.finish()
    statuses = {span.name: span.status for span in trace.spans}
    assert statuses == {"agent": STATUS_OK, "retrieve": STATUS_ERROR}
    assert trace.answer == "answered anyway"


def test_a_failed_span_survives_a_round_trip(tmp_path):
    recorder = Recorder()
    with pytest.raises(RuntimeError):
        with recorder.span("retrieve"):
            raise RuntimeError("boom")

    path = save(recorder.finish(), tmp_path / "t.json")

    assert load(path).spans[0].status == STATUS_ERROR


# ==========================================================================
# finish() fallbacks
# ==========================================================================


def test_the_query_falls_back_to_the_first_one_recorded():
    """The earliest is the question actually asked, before any rewriting."""
    recorder = Recorder()
    recorder.record_query("the original question")
    recorder.record_query("a rewritten question")

    assert recorder.finish().query == "the original question"


def test_the_answer_falls_back_to_the_last_text_generated():
    recorder = Recorder()
    recorder.record_answer("a first draft")
    recorder.record_answer("the final answer")

    assert recorder.finish().answer == "the final answer"


def test_both_may_be_supplied_by_the_caller():
    recorder = Recorder()
    recorder.record_query("observed")
    recorder.record_answer("observed")

    trace = recorder.finish(query="supplied", answer="supplied too")

    assert trace.query == "supplied"
    assert trace.answer == "supplied too"


def test_neither_is_required():
    trace = Recorder().finish()

    assert trace.query == ""
    assert trace.answer is None


def test_timing_is_recorded_without_being_asked_for():
    trace = Recorder().finish()

    assert trace.started_at is not None
    assert trace.duration_ms is not None


# ==========================================================================
# Naming and the output directory
# ==========================================================================


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Who changed auth?", "who-changed-auth"),
        ("payment_service AND #412", "payment-service-and-412"),
        ("   spaced   out   ", "spaced-out"),
        ("MiXeD CaSe", "mixed-case"),
        ("trailing---", "trailing"),
    ],
)
def test_a_query_becomes_a_readable_slug(query, expected):
    assert slugify(query) == expected


@pytest.mark.parametrize("query", ["", "   ", "!!!", "???---"])
def test_an_empty_slug_falls_back_to_a_constant(query):
    assert slugify(query) == EMPTY_SLUG


def test_a_long_query_is_truncated():
    assert len(slugify("word " * 100)) <= 60


def test_a_filename_is_a_timestamp_and_a_slug():
    from datetime import datetime, timezone

    name = trace_filename(
        "why does login fail?", datetime(2026, 8, 2, 10, 14, 7, tzinfo=timezone.utc)
    )

    assert name == "20260802T101407Z_why-does-login-fail.json"


def test_the_output_directory_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHRAG_TRACE_DIR", str(tmp_path))

    assert trace_dir() == tmp_path


def test_the_output_directory_defaults_to_here(monkeypatch):
    from pathlib import Path

    monkeypatch.delenv("GRAPHRAG_TRACE_DIR", raising=False)

    assert trace_dir() == Path(".")


def test_saving_without_a_path_uses_the_configured_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHRAG_TRACE_DIR", str(tmp_path / "traces"))
    recorder = Recorder()
    recorder.record_query("why does login fail?")

    written = save(recorder.finish())

    assert written.parent == tmp_path / "traces"
    assert written.name.endswith("_why-does-login-fail.json")


def test_the_default_path_is_derived_from_the_trace(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHRAG_TRACE_DIR", str(tmp_path))
    recorder = Recorder()
    recorder.record_query("a question")

    assert default_path(recorder.finish()).parent == tmp_path


# ==========================================================================
# Persistence
# ==========================================================================


def test_saving_returns_where_it_went(tmp_path):
    path = save(Recorder().finish(), tmp_path / "t.json")

    assert path == tmp_path / "t.json"
    assert path.exists()


def test_saving_creates_a_missing_directory(tmp_path):
    path = save(Recorder().finish(), tmp_path / "deep" / "nested" / "t.json")

    assert path.exists()


def test_a_saved_trace_reads_back_identical(tmp_path):
    recorder = Recorder(producer="p", arm=ARM_VECTOR)
    recorder.record_query("q")
    recorder.record_answer("a")
    recorder.record_items([TraceItem(id="x", content="c", source="s", score=0.5)])
    recorder.record_edges(
        [{"source": "x", "target": "y", "relation": "TOUCHES", "weight": 0.8}]
    )
    with recorder.span("step"):
        pass
    trace = recorder.finish()

    path = save(trace, tmp_path / "t.json")

    assert load(path) == trace


def test_the_written_file_carries_the_schema_version(tmp_path):
    path = save(Recorder().finish(), tmp_path / "t.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION


def test_finish_and_save_does_both(tmp_path):
    recorder = Recorder()
    recorder.record_query("q")

    trace, path = finish_and_save(recorder, tmp_path / "t.json")

    assert path.exists()
    assert load(path) == trace


def test_finish_and_save_accepts_the_same_overrides(tmp_path):
    recorder = Recorder()

    trace, _ = finish_and_save(
        recorder, tmp_path / "t.json", query="supplied", answer="also supplied"
    )

    assert trace.query == "supplied"
    assert trace.answer == "also supplied"


def test_edges_accept_the_graph_builders_own_field_names():
    """The builder emits type/confidence; a producer should not have to rename."""
    recorder = Recorder()
    recorder.record_edges(
        [{"source": "a", "target": "b", "type": "RESOLVES", "confidence": 0.92}]
    )

    edge = recorder.finish().edges[0]
    assert edge.relation == "RESOLVES"
    assert edge.weight == 0.92


def test_the_optional_fusion_fields_serialize_even_when_empty(tmp_path):
    recorder = Recorder()
    recorder.record_items(ITEMS)

    payload = to_dict(recorder.finish())

    assert payload["items"][0]["vector_score"] is None
    assert payload["items"][0]["graph_score"] is None
    assert payload["arm"] == ARM_UNKNOWN


def test_per_arm_scores_survive_when_a_producer_sets_them():
    recorder = Recorder(arm="hybrid")
    recorder.record_items(
        [
            {
                "id": "x",
                "content": "c",
                "source": "hybrid",
                "score": 0.8,
                "vector_score": 0.6,
                "graph_score": 0.9,
            }
        ]
    )

    item = recorder.finish().items[0]
    assert item.vector_score == 0.6
    assert item.graph_score == 0.9
