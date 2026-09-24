"""Tests for the outward retriever adapter.

A conversion at a boundary, so the tests drive it with runs built by hand:
what matters is that the score breakdown survives, that the text is real, and
that a result with no text still produces a usable document. None of that is
made truer by a database.

The batch fetch is asserted by counting calls, because the failure it guards
against — one query per result on a path that already ran two arms — is
invisible in the output.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip(
    "langchain_core", reason="the outward adapter needs langchain-core"
)

from src.retriever_langchain import (  # noqa: E402
    GraphRetriever,
    documents_from,
)


# --------------------------------------------------------------------------
# fakes shaped like what retrieval actually returns
# --------------------------------------------------------------------------


class FusedHit:
    def __init__(
        self,
        node_id,
        score=0.5,
        vector_score=0.5,
        graph_score=0.0,
        decay=1.0,
        node_type="PR",
        age_days=None,
    ):
        self.id = node_id
        self.score = score
        self.vector_score = vector_score
        self.graph_score = graph_score
        self.decay = decay
        self.node_type = node_type
        self.age_days = age_days

    @property
    def found_by_both(self):
        return self.vector_score > 0.0 and self.graph_score > 0.0


class VectorHit:
    def __init__(self, node_id, label=None):
        self.id = node_id
        self.similarity = 0.5
        self.label = label
        self.type = "PR"


class Run:
    """What the facade's query path hands back, as much as the adapter reads."""

    def __init__(self, hits, vector_hits=()):
        self.hits = list(hits)
        self.ids = [hit.id for hit in hits]
        self.vector = type("V", (), {"hits": list(vector_hits)})()


class Store:
    def __init__(self, documents=None, error=None):
        self.documents = documents or {}
        self.error = error
        self.calls = []

    def documents_for_entities(self, ids):
        self.calls.append(list(ids))
        if self.error is not None:
            raise self.error
        return {node_id: self.documents.get(node_id, []) for node_id in ids}


class Engine:
    """Stands in for the facade: a store and the two query entry points."""

    def __init__(self, run, store=None):
        self.run = run
        self._store = store or Store()
        self.queries = []

    @property
    def store(self):
        return self._store

    def retrieve(self, query, k=None, **kwargs):
        self.queries.append((query, k))
        return self.run

    async def retrieve_async(self, query, k=None, **kwargs):
        return self.retrieve(query, k)


def chunk(chunk_id, content, path=None):
    return {"id": chunk_id, "content": content, "path": path}


# ==========================================================================
# both arms
# ==========================================================================


def test_results_from_both_arms_become_documents_with_content_and_metadata():
    run = Run([FusedHit("pr:1", score=0.42, vector_score=0.30, graph_score=0.55)])
    store = Store({"pr:1": [chunk("doc:1", "the prose", "https://example.invalid/1")]})

    documents = GraphRetriever(engine=Engine(run, store)).invoke("who reviewed it?")

    assert len(documents) == 1
    document = documents[0]
    assert document.page_content == "the prose"
    assert document.metadata["id"] == "pr:1"
    assert document.metadata["score"] == 0.42
    assert document.metadata["vector_score"] == 0.30
    assert document.metadata["graph_score"] == 0.55
    assert document.metadata["found_by_both"] is True
    assert document.metadata["sources"] == ["https://example.invalid/1"]


def test_the_arms_are_reported_apart_rather_than_collapsed():
    """A node at 0.42 says nothing without the parts that produced it."""
    run = Run([FusedHit("pr:1", score=0.42, vector_score=0.30, graph_score=0.55, decay=0.9)])

    document = documents_from(run)[0]

    assert document.metadata["score"] != document.metadata["vector_score"]
    assert document.metadata["vector_score"] != document.metadata["graph_score"]
    assert document.metadata["decay"] == 0.9


@pytest.mark.parametrize(
    "vector_score, graph_score, missing",
    [(0.0, 0.40, "vector_score"), (0.40, 0.0, "graph_score")],
)
def test_an_arm_that_found_nothing_reports_zero_rather_than_dropping_out(
    vector_score, graph_score, missing
):
    """Absent and zero are the same number and different facts."""
    run = Run([FusedHit("pr:1", vector_score=vector_score, graph_score=graph_score)])

    metadata = documents_from(run)[0].metadata

    assert missing in metadata
    assert metadata[missing] == 0.0
    assert metadata["found_by_both"] is False


def test_every_ranked_result_becomes_a_document_in_order():
    run = Run([FusedHit("pr:1"), FusedHit("ticket:2"), FusedHit("commit:3")])

    documents = documents_from(run)

    assert [d.metadata["id"] for d in documents] == ["pr:1", "ticket:2", "commit:3"]


def test_a_run_with_no_results_produces_no_documents():
    assert documents_from(Run([])) == []


# ==========================================================================
# text
# ==========================================================================


def test_several_chunks_for_one_result_are_joined():
    run = Run([FusedHit("pr:1")])
    store = Store({"pr:1": [chunk("doc:1", "first"), chunk("doc:2", "second")]})

    documents = GraphRetriever(engine=Engine(run, store)).invoke("q")

    assert documents[0].page_content == "first\n\nsecond"
    assert documents[0].metadata["chunk_ids"] == ["doc:1", "doc:2"]


def test_text_is_fetched_once_for_the_whole_result_set():
    """One query, not one per result, on a path that already ran two arms."""
    run = Run([FusedHit("pr:1"), FusedHit("ticket:2"), FusedHit("commit:3")])
    store = Store()

    GraphRetriever(engine=Engine(run, store)).invoke("q")

    assert len(store.calls) == 1
    assert store.calls[0] == ["pr:1", "ticket:2", "commit:3"]


def test_a_result_with_no_text_falls_back_to_its_label():
    """An empty document is indistinguishable from having found nothing."""
    run = Run([FusedHit("pr:1")], vector_hits=[VectorHit("pr:1", label="fix the auth check")])
    store = Store({"pr:1": []})

    documents = GraphRetriever(engine=Engine(run, store)).invoke("q")

    assert documents[0].page_content == "fix the auth check"


def test_a_result_with_no_text_and_no_label_falls_back_to_type_and_id():
    run = Run([FusedHit("pr:1", node_type="PR")])

    document = documents_from(run)[0]

    assert document.page_content == "PR pr:1"
    assert document.page_content.strip()


def test_a_result_with_nothing_at_all_still_names_itself():
    run = Run([FusedHit("pr:1", node_type=None)])

    assert documents_from(run)[0].page_content == "pr:1"


def test_blank_chunk_content_counts_as_no_text():
    run = Run([FusedHit("pr:1")], vector_hits=[VectorHit("pr:1", label="a label")])
    store = Store({"pr:1": [chunk("doc:1", "   \n  ")]})

    documents = GraphRetriever(engine=Engine(run, store)).invoke("q")

    assert documents[0].page_content == "a label"


def test_a_store_that_cannot_answer_still_produces_documents():
    """The ranking is a real answer even when the prose cannot be reached."""
    run = Run([FusedHit("pr:1", node_type="PR")])
    store = Store(error=RuntimeError("the store is unreachable"))

    documents = GraphRetriever(engine=Engine(run, store)).invoke("q")

    assert documents[0].page_content == "PR pr:1"
    assert documents[0].metadata["chunk_ids"] == []


# ==========================================================================
# the call into the facade
# ==========================================================================


def test_the_query_reaches_the_facade_unchanged():
    engine = Engine(Run([FusedHit("pr:1")]))

    GraphRetriever(engine=engine).invoke("who reviewed the auth change?")

    assert engine.queries == [("who reviewed the auth change?", None)]


def test_an_unset_count_is_forwarded_as_unset():
    """So the sequence applies its own default rather than a copy held here."""
    engine = Engine(Run([FusedHit("pr:1")]))

    GraphRetriever(engine=engine).invoke("q")

    assert engine.queries[0][1] is None


def test_a_count_is_forwarded_when_given():
    engine = Engine(Run([FusedHit("pr:1")]))

    GraphRetriever(engine=engine, k=3).invoke("q")

    assert engine.queries[0][1] == 3


def test_the_async_path_returns_what_the_sync_one_does():
    engine = Engine(Run([FusedHit("pr:1")]))
    retriever = GraphRetriever(engine=engine)

    synchronous = retriever.invoke("q")
    asynchronous = asyncio.run(retriever.ainvoke("q"))

    assert [d.metadata["id"] for d in synchronous] == [
        d.metadata["id"] for d in asynchronous
    ]
    assert synchronous[0].page_content == asynchronous[0].page_content


def test_it_is_a_retriever_the_framework_recognises():
    from langchain_core.retrievers import BaseRetriever

    assert isinstance(GraphRetriever(engine=Engine(Run([]))), BaseRetriever)
