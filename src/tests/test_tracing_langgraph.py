"""Tests for the LangGraph adapter.

Driven through a real ``StateGraph`` with the tracer attached via
``config={"callbacks": [tracer]}``, so callback propagation into sub-runnables
is exercised rather than assumed. Where a hook is easier to provoke directly —
errors, LLM results — it is called directly and said so.

Three properties matter most: the agent behaves identically whether or not it
is traced, framework internals never reach the span list, and nothing
LangChain-shaped reaches the output.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.outputs import Generation, LLMResult
from langchain_core.retrievers import BaseRetriever

from examples.graph_demo import build_graph
from src.tracing import (
    ARM_GRAPH,
    ARM_VECTOR,
    KIND_DOCUMENT,
    KIND_LLM,
    KIND_RETRIEVER,
    KIND_TOOL,
    STATUS_ERROR,
    STATUS_OK,
    Trace,
    is_used,
    load,
    render,
    save,
    score_overlaps,
    to_dict,
)
from src.tracing_langgraph import (
    KIND_NODE,
    NOISE_NAMES,
    LangGraphTracer,
    _document_id,
    _score_of,
)

QUESTION = "who changed the authentication token expiry and the payment webhook recently?"


def traced_run(question: str = QUESTION) -> tuple[LangGraphTracer, dict]:
    """Run the real graph with the tracer attached."""
    tracer = LangGraphTracer()
    result = build_graph().invoke(
        {"question": question}, config={"callbacks": [tracer]}
    )
    return tracer, result


# ==========================================================================
# Attachment and pure observation
# ==========================================================================


def test_the_graph_returns_the_same_result_traced_or_not():
    graph = build_graph()
    inputs = {"question": QUESTION}

    untraced = graph.invoke(inputs)
    traced = graph.invoke(inputs, config={"callbacks": [LangGraphTracer()]})

    assert untraced["answer"] == traced["answer"]


def test_a_tracer_that_raises_does_not_break_the_run():
    """``raise_error`` is False on purpose: observation must not be fatal."""

    class BrokenTracer(LangGraphTracer):
        def on_retriever_end(self, documents, **kwargs):
            raise RuntimeError("tracer bug")

    result = build_graph().invoke(
        {"question": QUESTION}, config={"callbacks": [BrokenTracer()]}
    )

    assert result["answer"]


def test_callbacks_propagate_into_the_retriever_sub_runnable():
    """The retriever is never handed the config by hand; LangGraph threads it."""
    tracer, _ = traced_run()

    assert any(span.kind == KIND_RETRIEVER for span in tracer.spans)
    assert tracer.retrievals, "no retrieval was observed"


def test_an_untouched_tracer_records_nothing():
    tracer = LangGraphTracer()

    assert tracer.spans == []
    assert tracer.retrievals == []
    assert tracer.finish().query == ""


# ==========================================================================
# Span lifecycle and structure
# ==========================================================================


def test_every_graph_node_becomes_a_span():
    tracer, _ = traced_run()

    nodes = [span.name for span in tracer.spans if span.kind == KIND_NODE]
    assert nodes == ["retrieve", "inspect", "answer"]


def test_framework_internals_never_reach_the_span_list():
    """LangGraph wraps user nodes in plumbing; recording it buries the real steps."""
    tracer, _ = traced_run()

    recorded = {span.name for span in tracer.spans}
    assert not (recorded & NOISE_NAMES), f"framework internals leaked: {recorded}"


@pytest.mark.parametrize("noise", sorted(NOISE_NAMES))
def test_each_known_internal_name_is_absent(noise):
    tracer, _ = traced_run()

    assert noise not in {span.name for span in tracer.spans}


def test_a_retrieval_names_the_node_it_happened_inside():
    tracer, _ = traced_run()

    by_id = {span.id: span for span in tracer.spans}
    retrieval = next(s for s in tracer.spans if s.kind == KIND_RETRIEVER)

    assert retrieval.parent_id in by_id
    assert by_id[retrieval.parent_id].name == "retrieve"


def test_a_tool_call_names_the_node_it_happened_inside():
    tracer, _ = traced_run()

    by_id = {span.id: span for span in tracer.spans}
    tool = next(s for s in tracer.spans if s.kind == KIND_TOOL)

    assert by_id[tool.parent_id].name == "inspect"


def test_every_span_is_closed_with_a_window():
    tracer, _ = traced_run()

    for span in tracer.finish().spans:
        assert span.status == STATUS_OK
        assert span.start_ms is not None
        assert span.end_ms is not None
        assert span.end_ms >= span.start_ms


def test_span_ids_are_unique():
    tracer, _ = traced_run()

    ids = [span.id for span in tracer.spans]
    assert len(set(ids)) == len(ids)


def test_a_failing_retriever_leaves_an_error_span():
    class BrokenRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            raise RuntimeError("index offline")

    tracer = LangGraphTracer()
    with pytest.raises(RuntimeError):
        build_graph(BrokenRetriever()).invoke(
            {"question": QUESTION}, config={"callbacks": [tracer]}
        )

    failed = [s for s in tracer.finish().spans if s.status == STATUS_ERROR]
    assert any(s.kind == KIND_RETRIEVER for s in failed)


def test_a_failed_span_still_names_its_parent():
    class BrokenRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            raise RuntimeError("index offline")

    tracer = LangGraphTracer()
    with pytest.raises(RuntimeError):
        build_graph(BrokenRetriever()).invoke(
            {"question": QUESTION}, config={"callbacks": [tracer]}
        )

    trace = tracer.finish()
    by_id = {span.id: span for span in trace.spans}
    retrieval = next(s for s in trace.spans if s.kind == KIND_RETRIEVER)
    assert by_id[retrieval.parent_id].name == "retrieve"


# ==========================================================================
# Retrieval capture
# ==========================================================================


def test_a_retrieval_records_its_own_query_and_span():
    tracer, _ = traced_run()

    retrieval = tracer.retrievals[0]
    assert retrieval.query == QUESTION
    assert retrieval.span_id in {span.id for span in tracer.spans}


def test_items_capture_the_full_document_shape():
    tracer, _ = traced_run()

    item = next(i for i in tracer.finish().items if i.id == "pr:101")
    assert item.label
    assert item.kind == "PR"
    assert item.content
    assert item.score is not None
    assert item.source_uri and item.source_uri.startswith("https://")
    assert item.metadata


def test_metadata_only_carries_scalars():
    """A framework object in metadata would make the trace unserializable."""
    tracer, _ = traced_run()

    for item in tracer.finish().items:
        for value in item.metadata.values():
            assert isinstance(value, (str, int, float, bool))


def test_graph_edges_on_a_document_are_captured():
    tracer, _ = traced_run()

    edges = tracer.finish().edges
    assert edges
    assert all(edge.relation for edge in edges)
    assert any(edge.weight is not None for edge in edges)


def test_a_retrieval_reporting_relations_is_the_graph_arm():
    tracer, _ = traced_run()

    assert tracer.retrievals[0].arm == ARM_GRAPH


def test_a_retrieval_without_relations_is_the_vector_arm():
    tracer = LangGraphTracer()
    tracer.on_retriever_start({"name": "r"}, "q", run_id=None)
    tracer.on_retriever_end([Document(page_content="a", metadata={"id": "x"})])

    assert tracer.retrievals[0].arm == ARM_VECTOR


def test_a_document_retrieved_twice_is_recorded_once():
    document = Document(page_content="a", metadata={"id": "x"})
    tracer = LangGraphTracer()

    tracer.on_retriever_start({"name": "r"}, "q", run_id=None)
    tracer.on_retriever_end([document, document])

    assert [item.id for item in tracer.finish().items] == ["x"]


def test_a_malformed_edge_is_dropped_rather_than_written():
    tracer = LangGraphTracer()

    tracer.on_retriever_end(
        [
            Document(
                page_content="a",
                metadata={"id": "x", "edges": [{"source": "x"}, "nonsense"]},
            )
        ]
    )

    assert tracer.finish().edges == []


def test_an_edge_may_use_the_graph_builders_field_names():
    tracer = LangGraphTracer()

    tracer.on_retriever_end(
        [
            Document(
                page_content="a",
                metadata={
                    "id": "x",
                    "edges": [
                        {
                            "source": "x",
                            "target": "y",
                            "type": "RESOLVES",
                            "confidence": 0.92,
                        }
                    ],
                },
            )
        ]
    )

    edge = tracer.finish().edges[0]
    assert edge.relation == "RESOLVES"
    assert edge.weight == 0.92


# ==========================================================================
# Document identity and scores
# ==========================================================================


@pytest.mark.parametrize("key", ["id", "node_id", "doc_id"])
def test_a_document_id_is_read_from_metadata(key):
    assert _document_id(Document(page_content="a", metadata={key: "x:1"}), 0) == "x:1"


def test_a_document_id_falls_back_to_a_content_hash():
    """Positional ids do not survive a reordering; a content hash does."""
    first = _document_id(Document(page_content="same text"), 0)
    later = _document_id(Document(page_content="same text"), 7)

    assert first == later
    assert first.startswith("doc_")


def test_different_content_gets_different_ids():
    assert _document_id(Document(page_content="a"), 0) != _document_id(
        Document(page_content="b"), 0
    )


def test_an_empty_document_still_gets_an_id():
    assert _document_id(Document(page_content=""), 3) == "doc:3"


@pytest.mark.parametrize(
    "key", ["score", "relevance_score", "similarity", "_score", "vector_score"]
)
def test_any_recognized_score_key_is_read(key):
    assert _score_of({key: 0.7}) == 0.7


def test_a_non_numeric_score_is_ignored():
    assert _score_of({"score": "high"}) is None


def test_a_boolean_is_not_a_score():
    """``True`` is an int in Python; treating it as 1.0 would be nonsense."""
    assert _score_of({"score": True}) is None


def test_no_score_key_means_no_score():
    assert _score_of({"unrelated": 1}) is None


def test_an_item_without_a_kind_defaults_to_document():
    tracer = LangGraphTracer()

    tracer.on_retriever_end([Document(page_content="a", metadata={"id": "x"})])

    assert tracer.finish().items[0].kind == KIND_DOCUMENT


# ==========================================================================
# LLM and tool tracing
# ==========================================================================


def test_the_llm_answer_is_recorded():
    tracer = LangGraphTracer()

    tracer.on_llm_end(LLMResult(generations=[[Generation(text="the answer")]]))

    assert tracer.finish().answer == "the answer"


def test_a_chat_model_answer_is_recorded():
    tracer = LangGraphTracer()
    model = GenericFakeChatModel(messages=iter(["a chat answer"]))

    model.invoke("q", config={"callbacks": [tracer]})

    assert tracer.finish().answer == "a chat answer"


def test_an_llm_call_becomes_a_span():
    tracer = LangGraphTracer()
    GenericFakeChatModel(messages=iter(["x"])).invoke(
        "q", config={"callbacks": [tracer]}
    )

    assert any(span.kind == KIND_LLM for span in tracer.spans)


def test_a_tool_call_becomes_a_span():
    tracer, _ = traced_run()

    tools = [span for span in tracer.spans if span.kind == KIND_TOOL]
    assert [span.name for span in tools] == ["count_documents"]


@pytest.mark.parametrize(
    "hook, kind",
    [
        ("on_chain_error", KIND_NODE),
        ("on_retriever_error", KIND_RETRIEVER),
        ("on_llm_error", KIND_LLM),
        ("on_tool_error", KIND_TOOL),
    ],
)
def test_each_error_hook_records_its_phase(hook, kind):
    tracer = LangGraphTracer()

    getattr(tracer, hook)(ValueError("boom"))

    assert tracer.errors[0]["phase"] == kind
    assert "ValueError: boom" in tracer.errors[0]["error"]


def test_the_exception_text_stays_off_the_trace():
    """The span records that it failed; the schema has nowhere for the message."""
    tracer = LangGraphTracer()
    tracer.on_llm_error(ValueError("a very specific message"))

    assert "a very specific message" not in json.dumps(to_dict(tracer.finish()))
    assert "a very specific message" in tracer.errors[0]["error"]


# ==========================================================================
# The finished Trace
# ==========================================================================


def test_finish_produces_the_projects_trace_type():
    tracer, _ = traced_run()

    assert isinstance(tracer.finish(), Trace)


def test_finish_needs_no_arguments():
    tracer, _ = traced_run()

    trace = tracer.finish()

    assert trace.query == QUESTION
    assert trace.producer == "langgraph"
    assert trace.duration_ms is not None
    assert trace.started_at is not None


def test_an_explicit_answer_overrides_what_was_observed():
    """An agent's final answer lives in the invoke result, not the LLM call."""
    tracer, result = traced_run()

    trace = tracer.finish(answer=result["answer"])

    assert trace.answer == result["answer"]


def test_an_explicit_query_overrides_what_was_observed():
    tracer, _ = traced_run()

    assert tracer.finish(query="the real question").query == "the real question"


def test_the_trace_is_json_serializable():
    tracer, result = traced_run()

    json.dumps(to_dict(tracer.finish(answer=result["answer"])))


def test_no_langchain_object_reaches_the_serialized_trace():
    tracer, result = traced_run()

    payload = json.dumps(to_dict(tracer.finish(answer=result["answer"])))

    assert "langchain" not in payload.lower()
    assert "Document(" not in payload


def test_the_trace_round_trips_through_a_file(tmp_path):
    tracer, result = traced_run()
    trace = tracer.finish(answer=result["answer"])
    path = tmp_path / "t.json"

    save(trace, path)

    assert load(path) == trace


def test_the_trace_renders_in_the_viewer():
    tracer, result = traced_run()

    output = render(tracer.finish(answer=result["answer"]))

    assert QUESTION in output
    assert "pr:101" in output


def test_finish_leaves_overlap_to_the_classifier():
    """Two places able to measure would be two places able to disagree."""
    tracer, result = traced_run()

    trace = tracer.finish(answer=result["answer"])

    assert all(item.overlap is None for item in trace.items)


def test_the_full_pipeline_produces_a_used_and_ignored_split():
    tracer, result = traced_run()

    trace = score_overlaps(tracer.finish(answer=result["answer"]))

    assert any(is_used(item.overlap) for item in trace.items)
    assert any(is_used(item.overlap) is False for item in trace.items)


def test_the_core_tracing_package_still_needs_no_langchain():
    """The adapter is opt-in; importing the schema must not pull LangChain in."""
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [sys.executable, "-c", "import sys, src.tracing; print('langchain_core' in sys.modules)"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False", result.stderr
