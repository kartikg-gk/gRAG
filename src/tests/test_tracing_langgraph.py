"""Tests for the LangGraph adapter.

Driven through real LangChain runnables with the tracer attached via
``config={"callbacks": [tracer]}``, so the attach mechanism itself is exercised
rather than the hooks being called by hand. Where a hook is easier to provoke
directly — errors, LLM results — it is called directly and said so.

Two properties matter most and are checked hardest: the agent behaves
identically whether or not it is traced, and nothing LangChain-shaped reaches
the output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.outputs import Generation, LLMResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from src.tracing import (
    KIND_CHAIN,
    KIND_RETRIEVER,
    STATUS_ERROR,
    STATUS_OK,
    Trace,
    is_used,
    load,
    render,
    save,
    score_overlaps,
)
from src.tracing_langgraph import LangGraphTracer, capture

# --------------------------------------------------------------------------
# a small agent to watch
# --------------------------------------------------------------------------

CORPUS = [
    Document(
        page_content="The auth middleware rejects tokens one second early.",
        metadata={"id": "pr:101", "score": 0.91, "source_type": "graph"},
    ),
    Document(
        page_content="Release notes for version 2.3 of the auth library.",
        metadata={"id": "pr:290", "score": 0.44, "source_type": "vector"},
    ),
]

ANSWER = "The auth middleware rejects tokens one second early, fixed in pr:101."


class StubRetriever(BaseRetriever):
    """Returns the corpus, unchanged, for any query."""

    documents: list[Document] = CORPUS

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return list(self.documents)


def build_agent(retriever: BaseRetriever | None = None):
    """A retrieve-then-answer chain, the smallest thing worth tracing."""
    retriever = retriever or StubRetriever()

    def run(inputs: dict) -> dict:
        documents = retriever.invoke(inputs["question"])
        return {"answer": ANSWER, "documents": len(documents)}

    return RunnableLambda(run)


def traced_run(question: str = "why does login fail?") -> LangGraphTracer:
    tracer = LangGraphTracer()
    build_agent().invoke({"question": question}, config={"callbacks": [tracer]})
    return tracer


# ==========================================================================
# Attachment and pure observation
# ==========================================================================


def test_the_agent_returns_the_same_result_traced_or_not():
    agent = build_agent()
    inputs = {"question": "why does login fail?"}

    untraced = agent.invoke(inputs)
    traced = agent.invoke(inputs, config={"callbacks": [LangGraphTracer()]})

    assert untraced == traced


def test_a_tracer_that_raises_does_not_break_the_run():
    """``raise_error`` is False on purpose: observation must not be fatal."""

    class BrokenTracer(LangGraphTracer):
        def on_retriever_end(self, documents, **kwargs):
            raise RuntimeError("tracer bug")

    result = build_agent().invoke(
        {"question": "q"}, config={"callbacks": [BrokenTracer()]}
    )

    assert result["answer"] == ANSWER


def test_attaching_via_config_reaches_the_retriever():
    tracer = traced_run()

    assert tracer.items, "no retrieval was observed"


def test_an_untouched_tracer_records_nothing():
    tracer = LangGraphTracer()

    assert tracer.items == []
    assert tracer.query is None
    assert tracer.answer is None
    assert tracer.started_at is None


# ==========================================================================
# What the hooks record
# ==========================================================================


def test_the_retrievers_query_is_recorded():
    tracer = traced_run("why does login fail?")

    assert tracer.query == "why does login fail?"


def test_a_retrievers_query_beats_the_chain_inputs():
    """The chain sees whole state; the retriever was asked something specific."""
    tracer = LangGraphTracer()
    tracer.on_chain_start({"name": "agent"}, {"question": "broad state"})
    tracer.on_retriever_start({"name": "r"}, "the specific question")

    assert tracer.query == "the specific question"


def test_a_chain_input_is_used_when_no_retriever_ran():
    tracer = LangGraphTracer()
    tracer.on_chain_start({"name": "agent"}, {"question": "only the chain"})

    assert tracer.query == "only the chain"


@pytest.mark.parametrize("key", ["query", "question", "input", "text"])
def test_common_chain_input_keys_are_recognized(key):
    tracer = LangGraphTracer()
    tracer.on_chain_start({"name": "agent"}, {key: "the question"})

    assert tracer.query == "the question"


def test_retrieved_documents_become_items():
    tracer = traced_run()

    assert [item["id"] for item in tracer.items] == ["pr:101", "pr:290"]
    assert tracer.items[0]["content"].startswith("The auth middleware")
    assert tracer.items[0]["score"] == 0.91
    assert tracer.items[0]["source"] == "graph"


def test_a_document_retrieved_twice_is_recorded_once():
    tracer = LangGraphTracer()

    tracer.on_retriever_end(CORPUS)
    tracer.on_retriever_end(CORPUS)

    assert [item["id"] for item in tracer.items] == ["pr:101", "pr:290"]


def test_several_retrievals_accumulate():
    tracer = LangGraphTracer()

    tracer.on_retriever_end([CORPUS[0]])
    tracer.on_retriever_end([Document(page_content="other", metadata={"id": "x:1"})])

    assert [item["id"] for item in tracer.items] == ["pr:101", "x:1"]


def test_a_document_without_metadata_still_gets_a_usable_id():
    tracer = LangGraphTracer()

    tracer.on_retriever_end([Document(page_content="a"), Document(page_content="b")])

    assert [item["id"] for item in tracer.items] == ["doc:0", "doc:1"]
    assert all(item["score"] is None for item in tracer.items)
    assert all(item["source"] == "retriever" for item in tracer.items)


@pytest.mark.parametrize(
    "key", ["score", "relevance_score", "similarity", "_score", "vector_score"]
)
def test_any_recognized_score_key_is_read(key):
    tracer = LangGraphTracer()

    tracer.on_retriever_end([Document(page_content="a", metadata={"id": "x", key: 0.7})])

    assert tracer.items[0]["score"] == 0.7


def test_a_non_numeric_score_is_ignored():
    tracer = LangGraphTracer()

    tracer.on_retriever_end(
        [Document(page_content="a", metadata={"id": "x", "score": "high"})]
    )

    assert tracer.items[0]["score"] is None


def test_relations_on_a_document_are_captured():
    tracer = LangGraphTracer()

    tracer.on_retriever_end(
        [
            Document(
                page_content="a",
                metadata={
                    "id": "pr:1",
                    "edges": [
                        {"source": "pr:1", "target": "ticket:2", "type": "RESOLVES"}
                    ],
                },
            )
        ]
    )

    assert tracer.edges == [
        {"source": "pr:1", "target": "ticket:2", "type": "RESOLVES"}
    ]


def test_a_malformed_edge_is_dropped_rather_than_written():
    tracer = LangGraphTracer()

    tracer.on_retriever_end(
        [
            Document(
                page_content="a",
                metadata={"id": "pr:1", "edges": [{"source": "pr:1"}, "nonsense"]},
            )
        ]
    )

    assert tracer.edges == []


def test_the_same_relation_seen_twice_is_recorded_once():
    edge = {"source": "pr:1", "target": "ticket:2", "type": "RESOLVES"}
    document = Document(page_content="a", metadata={"id": "pr:1", "edges": [edge]})
    tracer = LangGraphTracer()

    tracer.on_retriever_end([document])
    tracer.on_retriever_end([document])

    assert len(tracer.edges) == 1


def test_the_llm_answer_is_recorded():
    tracer = LangGraphTracer()

    tracer.on_llm_end(LLMResult(generations=[[Generation(text="the answer")]]))

    assert tracer.answer == "the answer"


def test_a_chat_model_answer_is_recorded():
    tracer = LangGraphTracer()
    model = GenericFakeChatModel(messages=iter(["a chat answer"]))

    model.invoke("q", config={"callbacks": [tracer]})

    assert tracer.answer == "a chat answer"


def test_a_chain_output_supplies_the_answer_when_no_llm_ran():
    tracer = traced_run()

    assert tracer.answer == ANSWER


def test_tool_names_are_recorded():
    tracer = LangGraphTracer()

    tracer.on_tool_start({"name": "search"}, "query")
    tracer.on_tool_end("result")

    assert tracer.tools == ["search"]


def test_chain_names_are_recorded():
    tracer = traced_run()

    assert tracer.chains


# ==========================================================================
# Errors
# ==========================================================================


@pytest.mark.parametrize(
    "hook, phase",
    [
        ("on_chain_error", "chain"),
        ("on_retriever_error", "retriever"),
        ("on_llm_error", "llm"),
        ("on_tool_error", "tool"),
    ],
)
def test_each_error_hook_records_its_phase(hook, phase):
    tracer = LangGraphTracer()

    getattr(tracer, hook)(ValueError("boom"))

    assert tracer.errors == [{"phase": phase, "error": "ValueError: boom"}]


def test_a_failed_unit_survives_into_the_serialized_trace():
    """Superseded an earlier test that asserted the opposite.

    That test also passed vacuously: ``"error" not in payload`` inspects the
    top-level dict *keys*, never the span statuses, so it would have kept
    passing whatever the tracer did.
    """
    tracer = traced_run()
    tracer.on_llm_error(ValueError("boom"))

    payload = json.loads(json.dumps(_as_dict(capture(tracer))))

    failed = [span for span in payload["spans"] if span["status"] == STATUS_ERROR]
    assert failed, "the failed unit disappeared from the trace"
    assert tracer.errors, "the exception message is still available on the tracer"


def test_the_exception_text_stays_off_the_trace():
    """The span records that it failed; the schema has nowhere for the message."""
    tracer = traced_run()
    tracer.on_llm_error(ValueError("a very specific message"))

    payload = json.dumps(_as_dict(capture(tracer)))

    assert "a very specific message" not in payload
    assert "a very specific message" in tracer.errors[0]["error"]


# ==========================================================================
# Structure: nesting reconstructable from the trace alone
# ==========================================================================


def test_a_run_records_spans():
    trace = capture(traced_run())

    assert trace.spans, "a traced run produced no spans"


def test_a_retrieval_names_the_step_it_happened_inside():
    """Objective 2: 'this retrieval happened inside step X', from the trace only."""
    trace = capture(traced_run())

    by_id = {span.id: span for span in trace.spans}
    retrieval = next(span for span in trace.spans if span.kind == KIND_RETRIEVER)

    assert retrieval.parent_id in by_id
    assert by_id[retrieval.parent_id].kind == KIND_CHAIN


def test_a_top_level_span_has_no_parent():
    trace = capture(traced_run())

    roots = [span for span in trace.spans if span.parent_id is None]
    assert len(roots) == 1
    assert roots[0].kind == KIND_CHAIN


def test_every_span_records_a_kind_and_a_window():
    trace = capture(traced_run())

    for span in trace.spans:
        assert span.kind
        assert span.start_ms is not None
        assert span.end_ms is not None
        assert span.end_ms >= span.start_ms


def test_a_completed_span_is_marked_ok():
    trace = capture(traced_run())

    assert all(span.status == STATUS_OK for span in trace.spans)


def test_span_ids_are_unique():
    trace = capture(traced_run())

    ids = [span.id for span in trace.spans]
    assert len(set(ids)) == len(ids)


def test_a_parent_from_outside_the_traced_subtree_is_not_a_dangling_reference():
    """Attaching mid-tree must not name a parent this tracer never saw."""
    from uuid import uuid4

    tracer = LangGraphTracer()
    tracer.on_retriever_start({"name": "r"}, "q", run_id=uuid4(), parent_run_id=uuid4())

    assert tracer.spans[0].parent_id is None


def test_spans_survive_a_round_trip(tmp_path):
    trace = capture(traced_run())
    path = tmp_path / "t.json"

    save(trace, path)

    assert [s.parent_id for s in load(path).spans] == [
        s.parent_id for s in trace.spans
    ]


def test_a_failed_retrieval_keeps_its_place_in_the_tree():
    """A failure inside a step must still say which step it was inside."""

    class BrokenRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            raise RuntimeError("index offline")

    broken = BrokenRetriever()
    agent = RunnableLambda(lambda d: broken.invoke(d["question"]))
    tracer = LangGraphTracer()

    with pytest.raises(RuntimeError):
        agent.invoke({"question": "q"}, config={"callbacks": [tracer]})

    trace = capture(tracer)
    by_id = {span.id: span for span in trace.spans}
    retrieval = next(span for span in trace.spans if span.kind == KIND_RETRIEVER)

    assert retrieval.status == STATUS_ERROR
    assert retrieval.parent_id in by_id


def _as_dict(trace: Trace) -> dict:
    from src.tracing import to_dict

    return to_dict(trace)


def test_a_failing_retriever_is_recorded_and_the_error_still_propagates():
    class BrokenRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            raise RuntimeError("index offline")

    tracer = LangGraphTracer()

    with pytest.raises(RuntimeError):
        BrokenRetriever().invoke("q", config={"callbacks": [tracer]})

    assert tracer.errors[0]["phase"] == "retriever"
    assert "index offline" in tracer.errors[0]["error"]


# ==========================================================================
# Timing
# ==========================================================================


def test_a_traced_run_records_when_it_started_and_how_long_it_took():
    tracer = traced_run()

    assert isinstance(tracer.started_at, datetime)
    assert tracer.started_at.tzinfo is not None
    assert tracer.duration_ms is not None
    assert tracer.duration_ms >= 0


def test_the_start_time_is_the_first_event_not_the_last():
    tracer = LangGraphTracer()

    tracer.on_chain_start({"name": "a"}, {"question": "q"})
    first = tracer.started_at
    tracer.on_retriever_start({"name": "r"}, "q")

    assert tracer.started_at == first


# ==========================================================================
# The output is framework-neutral
# ==========================================================================


def test_no_langchain_type_reaches_the_trace():
    trace = capture(traced_run())

    for value in (*trace.items, *trace.edges):
        for field in vars(value).values() if hasattr(value, "__dict__") else []:
            assert "langchain" not in type(field).__module__


def test_the_trace_is_json_serializable():
    trace = capture(traced_run())

    json.dumps(_as_dict(trace))  # raises if a LangChain object leaked through


def test_a_captured_trace_round_trips_through_a_file(tmp_path):
    trace = capture(traced_run())
    path = tmp_path / "trace.json"

    save(trace, path)

    assert load(path) == trace


def test_a_captured_trace_renders_in_the_viewer():
    trace = capture(traced_run())

    output = render(trace)
    assert "why does login fail?" in output
    assert "pr:101" in output


def test_capture_leaves_classification_to_the_classifier():
    trace = capture(traced_run())

    assert all(is_used(item.overlap) is None for item in trace.items)


def test_the_full_pipeline_produces_a_used_and_ignored_split():
    trace = score_overlaps(capture(traced_run()))

    assert any(is_used(item.overlap) for item in trace.items)
    assert any(is_used(item.overlap) is False for item in trace.items)


# ==========================================================================
# capture()
# ==========================================================================


def test_capture_uses_what_the_tracer_observed():
    trace = capture(traced_run("the question"))

    assert trace.query == "the question"
    assert trace.answer == ANSWER


def test_an_explicit_answer_overrides_what_was_observed():
    """An agent's final answer often lives in the invoke result, not the LLM call."""
    trace = capture(traced_run(), answer="the real answer")

    assert trace.answer == "the real answer"


def test_an_explicit_query_overrides_what_was_observed():
    trace = capture(traced_run(), query="the real question")

    assert trace.query == "the real question"


def test_capturing_an_unused_tracer_produces_an_empty_trace():
    trace = capture(LangGraphTracer())

    assert trace.query == ""
    assert trace.items == []
    assert trace.answer is None


def test_capture_carries_the_observed_timing():
    tracer = traced_run()

    trace = capture(tracer)

    assert trace.started_at == tracer.started_at
    assert trace.duration_ms == tracer.duration_ms


def test_the_core_tracing_package_still_needs_no_langchain():
    """The adapter is opt-in; importing the schema must not pull LangChain in."""
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, src.tracing;"
            "print('langchain_core' in sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False", result.stderr
