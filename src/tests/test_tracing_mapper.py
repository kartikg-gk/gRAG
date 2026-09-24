"""A recorded run reshaped into the Studio's TraceState."""

from __future__ import annotations

from src.tracing.mapper import (
    DEFAULT_EDGE_CONFIDENCE,
    MAX_CONTEXT_CHARS,
    SNIPPET_CHARS,
    to_tracestate,
)
from src.tracing.schema import Retrieval, Span, Trace, TraceEdge, TraceItem


def item(item_id, *, kind="pr", overlap=None, content="text", score=0.5, label=None):
    return TraceItem(
        id=item_id, content=content, source="github", label=label or item_id, kind=kind,
        score=score, vector_score=0.4, graph_score=0.6, overlap=overlap,
        source_uri=f"https://example.com/{item_id}",
    )


def run(items, edges=(), answer=None, spans=(), duration_ms=1500.0):
    return Trace(
        query="who fixed login?", answer=answer, producer="langgraph",
        duration_ms=duration_ms,
        retrievals=[Retrieval(query="q", span_id="span-1234567890", arm="graph",
                              items=list(items), edges=list(edges))],
        spans=list(spans),
    )


def test_every_field_the_studio_reads_is_present():
    state = to_tracestate(run([item("pr:1")]))
    assert set(state) == {"id", "query", "computedAt", "weights", "confidence",
                          "steps", "metrics", "graph", "context"}
    node = state["graph"]["nodes"][0]
    assert node["position"] == {"x": 0, "y": 0}
    assert (node["similarity"], node["score"], node["meta"]["scoreGraph"]) == (0.4, 0.5, 0.6)
    assert node["meta"]["sourceUrl"] == "https://example.com/pr:1"


def test_kinds_become_studio_entity_types_and_unknown_ones_documents():
    state = to_tracestate(run([item("a", kind="Issue"), item("b", kind=" author "), item("c", kind="file")]))
    types = {node["id"]: node["type"] for node in state["graph"]["nodes"]}
    assert types == {"a": "Ticket", "b": "Person", "c": "Document"}


def test_with_an_answer_items_below_the_threshold_draw_as_unused():
    state = to_tracestate(run([item("used", overlap=0.5), item("ignored", overlap=0.05)], answer="x"))
    active = {node["id"]: node["active"] for node in state["graph"]["nodes"]}
    assert active == {"used": True, "ignored": False}
    subtitles = {node["id"]: node["meta"]["subtitle"] for node in state["graph"]["nodes"]}
    assert subtitles["ignored"].startswith("no source-text match detected")


def test_without_an_answer_every_item_draws_plainly():
    state = to_tracestate(run([item("a", overlap=0.0)]))
    node = state["graph"]["nodes"][0]
    assert node["active"] is True
    assert node["meta"]["subtitle"] == "via langgraph · graph arm"


def test_a_repeated_item_is_drawn_once():
    state = to_tracestate(run([item("a"), item("a")]))
    assert len(state["graph"]["nodes"]) == 1


def test_edges_are_kept_only_between_retrieved_items():
    edges = [TraceEdge("a", "b", "AUTHORED", 0.9), TraceEdge("a", "missing", "TOUCHES"),
             TraceEdge("b", "a", "REVIEWED")]
    state = to_tracestate(run([item("a"), item("b")], edges))
    drawn = state["graph"]["edges"]
    assert [(edge["source"], edge["target"]) for edge in drawn] == [("a", "b"), ("b", "a")]
    assert drawn[0]["confidence"] == 0.9
    assert drawn[1]["confidence"] == DEFAULT_EDGE_CONFIDENCE
    assert drawn[0]["id"] == "e_span-123_0"


def test_an_edge_is_active_only_when_both_ends_were_used():
    edges = [TraceEdge("used", "ignored", "AUTHORED")]
    state = to_tracestate(run([item("used", overlap=0.9), item("ignored", overlap=0.0)], edges, answer="x"))
    assert state["graph"]["edges"][0]["active"] is False


def test_spans_become_ordered_steps_with_durations():
    spans = [
        Span(id="s2", name="answer", kind="llm", start_ms=50.0, end_ms=80.0, status="ok"),
        Span(id="span-1234567890", name="search", kind="retriever", start_ms=10.0, end_ms=40.25, status="ok"),
        Span(id="s3", name="late", kind="chain", start_ms=90.0, status="running"),
    ]
    steps = to_tracestate(run([item("a")], spans=spans))["steps"]
    assert [step["title"] for step in steps] == ["search", "answer", "late"]
    assert steps[0]["durationMs"] == 30.2 and steps[0]["arm"] == "graph"
    assert steps[2]["status"] == "pending" and steps[2]["durationMs"] is None


def test_weights_describe_the_arms_observed():
    plain = to_tracestate(run([item("a")]))["weights"]
    linked = to_tracestate(run([item("a"), item("b")], [TraceEdge("a", "b", "X")]))["weights"]
    assert plain == {"vector": 1.0, "graph": 0.0, "intent": "conceptual"}
    assert linked == {"vector": 0.5, "graph": 0.5, "intent": "relational"}


def test_confidence_is_the_mean_reported_score_and_never_recomputed():
    state = to_tracestate(run([item("a", score=0.2), item("b", score=0.6)]))
    assert state["confidence"]["score"] == 0.4
    assert state["confidence"]["uncertainty"] == 0.0


def test_context_and_snippets_are_bounded():
    long_text = "y" * (MAX_CONTEXT_CHARS + 100)
    state = to_tracestate(run([item("a", content=long_text)]))
    assert len(state["context"]) == MAX_CONTEXT_CHARS
    assert len(state["graph"]["nodes"][0]["meta"]["snippet"]) == SNIPPET_CHARS


def test_query_time_comes_from_the_run_duration():
    assert to_tracestate(run([item("a")], duration_ms=1234.0))["metrics"]["queryTimeSec"] == 1.234
