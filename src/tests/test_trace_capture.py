"""Tests for M2, the capture layer.

Capture knows nothing about any particular retriever. These tests prove that by
driving it with two stubs that share no code and return different shapes: a
flat vector-style retriever, and a graph-style one that also returns relations.

If capture ever needs changing to accommodate a real retriever, it was written
wrong — so the graph stub here deliberately looks like what M5 will produce.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.tracing import Trace, TraceEdge, TraceItem, capture, is_used

# --------------------------------------------------------------------------
# stub retrievers — no shared code, no dependency on the real pipeline
# --------------------------------------------------------------------------


def vector_retriever(query: str) -> list[dict]:
    """A flat similarity search. No relations."""
    return [
        {
            "id": "doc:1",
            "content": "The auth middleware rejects tokens one second early.",
            "source": "vector",
            "score": 0.88,
        },
        {
            "id": "doc:2",
            "content": "Release notes for version 2.3 of the auth library.",
            "source": "vector",
            "score": 0.41,
        },
    ]


def graph_retriever(query: str) -> tuple[list[dict], list[dict]]:
    """What M5 is expected to look like: items plus the relations between them."""
    items = [
        {
            "id": "pr:1347",
            "content": "Rewrite the auth middleware token check.",
            "source": "graph",
            "score": 0.91,
        },
        {
            "id": "issue:42",
            "content": "Login fails for expired tokens.",
            "source": "graph",
            "score": 0.87,
        },
    ]
    edges = [
        {
            "source": "pr:1347",
            "target": "issue:42",
            "type": "RESOLVES",
            "confidence": 0.92,
        }
    ]
    return items, edges


ANSWER = "Alice rewrote the auth middleware so expired tokens stop failing login."


# --------------------------------------------------------------------------
# what capture records
# --------------------------------------------------------------------------


def test_capture_returns_a_trace():
    trace = capture("why does login fail?", vector_retriever("q"), ANSWER)

    assert isinstance(trace, Trace)


def test_the_query_and_answer_are_recorded():
    trace = capture("why does login fail?", vector_retriever("q"), ANSWER)

    assert trace.query == "why does login fail?"
    assert trace.answer == ANSWER


def test_every_retrieved_item_is_recorded_in_order():
    trace = capture("q", vector_retriever("q"), ANSWER)

    assert [item.id for item in trace.items] == ["doc:1", "doc:2"]


def test_item_fields_survive_capture():
    trace = capture("q", vector_retriever("q"), ANSWER)

    first = trace.items[0]
    assert first.content == "The auth middleware rejects tokens one second early."
    assert first.source == "vector"
    assert first.score == 0.88


def test_capture_does_not_decide_what_was_used():
    trace = capture("q", vector_retriever("q"), ANSWER)

    assert all(is_used(item.overlap) is None for item in trace.items)
    assert all(item.overlap is None for item in trace.items)


def test_an_answer_is_optional():
    trace = capture("q", vector_retriever("q"))

    assert trace.answer is None
    assert len(trace.items) == 2


def test_retrieving_nothing_is_still_a_trace():
    trace = capture("q", [], ANSWER)

    assert trace.items == []
    assert trace.query == "q"


# --------------------------------------------------------------------------
# pure observation
# --------------------------------------------------------------------------


def test_capture_does_not_mutate_what_the_retriever_returned():
    items = vector_retriever("q")
    before = [dict(item) for item in items]

    capture("q", items, ANSWER)

    assert items == before


def test_capture_copies_rather_than_aliasing_the_retriever_output():
    items = vector_retriever("q")

    trace = capture("q", items, ANSWER)
    trace.items[0].content = "changed after capture"

    assert items[0]["content"] == "The auth middleware rejects tokens one second early."


def test_a_generator_of_items_is_accepted():
    trace = capture("q", (item for item in vector_retriever("q")), ANSWER)

    assert len(trace.items) == 2


# --------------------------------------------------------------------------
# the shapes capture must accept
# --------------------------------------------------------------------------


def test_trace_items_can_be_passed_directly():
    trace = capture(
        "q", [TraceItem(id="pr:1", content="x", source="graph", score=0.5)], ANSWER
    )

    assert trace.items[0].id == "pr:1"


def test_a_missing_score_is_allowed():
    trace = capture("q", [{"id": "pr:1", "content": "x", "source": "graph"}], ANSWER)

    assert trace.items[0].score is None


def test_a_missing_source_is_recorded_as_unknown():
    trace = capture("q", [{"id": "pr:1", "content": "x"}], ANSWER)

    assert trace.items[0].source == "unknown"


def test_an_item_without_an_id_is_refused():
    with pytest.raises(KeyError):
        capture("q", [{"content": "x", "source": "graph"}], ANSWER)


# --------------------------------------------------------------------------
# relations, so a graph retriever needs no change to capture
# --------------------------------------------------------------------------


def test_a_graph_retriever_captures_its_relations():
    items, edges = graph_retriever("q")

    trace = capture("q", items, ANSWER, edges=edges)

    assert [edge.relation for edge in trace.edges] == ["RESOLVES"]
    assert trace.edges[0].source == "pr:1347"
    assert trace.edges[0].target == "issue:42"
    assert trace.edges[0].weight == 0.92


def test_trace_edges_can_be_passed_directly():
    trace = capture(
        "q",
        [],
        ANSWER,
        edges=[TraceEdge(source="a", target="b", relation="TOUCHES", weight=0.8)],
    )

    assert trace.edges[0].relation == "TOUCHES"


def test_a_retriever_with_no_relations_produces_no_edges():
    trace = capture("q", vector_retriever("q"), ANSWER)

    assert trace.edges == []


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def test_timing_is_recorded_when_the_caller_measured_it():
    started = datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc)

    trace = capture(
        "q", vector_retriever("q"), ANSWER, started_at=started, duration_ms=412.5
    )

    assert trace.started_at == started
    assert trace.duration_ms == 412.5


def test_timing_is_absent_when_the_caller_did_not_measure_it():
    trace = capture("q", vector_retriever("q"), ANSWER)

    assert trace.started_at is None
    assert trace.duration_ms is None


# --------------------------------------------------------------------------
# round trip, since capture feeds the viewer
# --------------------------------------------------------------------------


def test_a_captured_trace_survives_a_write_and_read(tmp_path):
    from src.tracing import load, save

    items, edges = graph_retriever("q")
    trace = capture("q", items, ANSWER, edges=edges)
    path = tmp_path / "trace.json"

    save(trace, path)

    assert load(path) == trace


def test_a_captured_trace_renders_in_the_viewer():
    from src.tracing import render

    trace = capture("why does login fail?", vector_retriever("q"), ANSWER)

    output = render(trace)
    assert "why does login fail?" in output
    assert "doc:1" in output
    assert "unclassified" in output
