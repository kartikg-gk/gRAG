"""Tests for the facade.

The facade owns a handle and sequences four existing callables. Those two
things are what is tested here, and they are tested apart:

* **On a fake store**, because lifecycle and passthrough are properties of
  this object alone. A real database cannot make "the handle was closed even
  though the body raised" any truer, and it cannot run on an interpreter
  without the driver's native library.
* **On the real store**, for the four operations in sequence — because
  "ingest then index then query then context" is a claim about the modules
  agreeing with each other, and a fake would assert the agreement rather than
  observe it. Those skip where the driver cannot load.

Nothing here re-tests retrieval, persistence or traversal. Each has its own
file, and this object adds no arithmetic to any of them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.engine import Engine, EngineNotOpen


class FakeStore:
    """Records what was asked of it, and whether it was closed."""

    def __init__(self, documents=None):
        self.closed = 0
        self.index_builds: list[bool] = []
        self.subgraph_calls: list[list[str]] = []
        self.document_calls: list[list[str]] = []
        self.documents = documents or {}

    def close(self):
        self.closed += 1

    def build_vector_index(self, *, rebuild=False):
        self.index_builds.append(rebuild)

    def documents_for_entities(self, ids):
        self.document_calls.append(list(ids))
        return {node_id: self.documents.get(node_id, []) for node_id in ids}

    def subgraph(self, ids):
        self.subgraph_calls.append(list(ids))
        return {"nodes": [{"id": node} for node in ids], "edges": []}


class FakeHit:
    def __init__(self, node_id, node_type="PR"):
        self.id = node_id
        self.node_type = node_type


class FakeRun:
    """What the query path returns, as much of it as the facade reads."""

    def __init__(self, ids, node_type="PR"):
        self.ids = list(ids)
        self.hits = [FakeHit(node_id, node_type) for node_id in ids]


class StubEmbedder:
    """Never called by these tests; present so the default is never built."""

    def vector(self, text):
        return [0.0]


@pytest.fixture
def opened(monkeypatch):
    """An engine whose store is a fake, and the fake itself."""
    store = FakeStore()
    monkeypatch.setattr(
        "src.engine.open_context_graph", lambda path, **kwargs: store
    )
    return Engine("unused", embedder=StubEmbedder()), store


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


def test_constructing_opens_the_store(monkeypatch, tmp_path):
    """A caller gets a usable object without a block. That costs an open."""
    opens = []
    monkeypatch.setattr(
        "src.engine.open_context_graph",
        lambda path, **kwargs: opens.append(path) or FakeStore(),
    )

    engine = Engine(tmp_path / "graph")

    assert opens == [tmp_path / "graph"]
    assert engine.store is not None


def test_an_engine_built_without_a_block_is_usable(monkeypatch, tmp_path):
    """The reason construction opens: a script should not need a `with`."""
    store = FakeStore()
    monkeypatch.setattr("src.engine.open_context_graph", lambda path, **kw: store)
    monkeypatch.setattr("src.engine._retrieve", lambda *a, **k: "result")

    engine = Engine(tmp_path / "graph", embedder=StubEmbedder())
    try:
        assert engine.store is store
        assert engine.retrieve("q") == "result"
    finally:
        engine.close()

    assert store.closed == 1


def test_entering_returns_the_object_and_leaving_closes(opened):
    engine, store = opened

    with engine as entered:
        assert entered is engine
        assert store.closed == 0

    assert store.closed == 1


def test_the_handle_is_closed_when_the_body_raises(opened):
    """The case the context manager exists for."""
    engine, store = opened

    with pytest.raises(ValueError, match="from the body"):
        with engine:
            raise ValueError("from the body")

    assert store.closed == 1


def test_an_operation_failing_still_closes_the_handle(monkeypatch, opened):
    engine, store = opened
    monkeypatch.setattr(
        "src.engine._retrieve", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        with engine:
            engine.retrieve("anything")

    assert store.closed == 1


def test_the_exception_is_not_swallowed(opened):
    """Closing on the way out must not turn a failure into a success."""
    engine, _store = opened

    with pytest.raises(KeyError):
        with engine:
            raise KeyError("still raised")


@pytest.mark.parametrize(
    "operation",
    [
        lambda engine: engine.build_index(),
        lambda engine: engine.retrieve("q"),
        lambda engine: engine.context([]),
        lambda engine: engine.store,
    ],
)
def test_using_a_closed_engine_raises_its_own_error(operation, monkeypatch):
    """Not a bare RuntimeError: 'this engine is spent' is worth telling apart."""
    monkeypatch.setattr("src.engine.open_context_graph", lambda path, **kw: FakeStore())
    engine = Engine("unused", embedder=StubEmbedder())
    engine.close()

    with pytest.raises(EngineNotOpen):
        operation(engine)


def test_closing_twice_is_safe(opened):
    engine, store = opened

    with engine:
        pass
    engine.close()

    assert store.closed == 1


def test_closing_twice_directly_is_safe(monkeypatch):
    """Two owners can both close: a context manager and whatever holds it."""
    store = FakeStore()
    monkeypatch.setattr("src.engine.open_context_graph", lambda path, **kw: store)
    engine = Engine("unused")

    engine.close()
    engine.close()

    assert store.closed == 1


# --------------------------------------------------------------------------
# the operations pass through
# --------------------------------------------------------------------------


def test_the_optional_pieces_reach_the_query_path(monkeypatch, opened):
    """Held on the engine, handed to the call that accepts them."""
    engine, _store = opened
    seen = {}

    def spy(store, embedder, query, **kwargs):
        seen.update(kwargs)
        seen["query"] = query
        seen["embedder"] = embedder
        return "result"

    monkeypatch.setattr("src.engine._retrieve", spy)
    engine.extractor = "an-extractor"
    engine.judge = "a-judge"
    engine.now = "a-clock"

    with engine:
        assert engine.retrieve("who reviewed it?") == "result"

    assert seen["extractor"] == "an-extractor"
    assert seen["judge"] == "a-judge"
    assert seen["now"] == "a-clock"
    assert seen["query"] == "who reviewed it?"


def test_absent_optional_pieces_are_passed_through_as_absent(monkeypatch, opened):
    """Not re-implemented, not defaulted here. The query path already decides."""
    engine, _store = opened
    seen = {}
    monkeypatch.setattr(
        "src.engine._retrieve",
        lambda store, embedder, query, **kwargs: seen.update(kwargs) or "result",
    )

    with engine:
        engine.retrieve("q")

    assert seen["extractor"] is None
    assert seen["judge"] is None
    assert seen["now"] is None


def test_the_default_embedder_is_never_built_when_one_is_supplied(opened):
    """Building it imports a model. A supplied one must prevent that entirely."""
    engine, _store = opened
    embedder = engine.embedder

    with engine:
        assert engine._embedder() is embedder


def test_indexing_rebuilds_by_default(opened):
    """False is the setting that fails on the second run, so it is not default."""
    engine, store = opened

    with engine:
        engine.build_index()
        engine.build_index(rebuild=False)

    assert store.index_builds == [True, False]


def test_ingest_hands_the_builder_and_documents_to_the_write_path(monkeypatch, opened):
    engine, store = opened
    seen = {}

    def spy(target, builder, **kwargs):
        seen["target"] = target
        seen["builder"] = builder
        seen.update(kwargs)
        return "stats"

    monkeypatch.setattr("src.engine.persist", spy)
    engine.extractor = "an-extractor"

    with engine:
        assert engine.ingest("a-builder", documents=["d1"]) == "stats"

    assert seen["target"] is store
    assert seen["builder"] == "a-builder"
    assert seen["documents"] == ["d1"]
    assert seen["extractor"] == "an-extractor"


def test_context_returns_a_string_of_entities_and_their_text(monkeypatch, opened):
    engine, store = opened
    store.documents = {
        "pr:1": [{"id": "doc:1", "path": "https://example.invalid/1", "content": "the prose"}]
    }
    monkeypatch.setattr("src.engine._retrieve", lambda *a, **k: FakeRun(["pr:1"]))

    with engine:
        text = engine.context("who reviewed it?")

    assert isinstance(text, str)
    assert "pr:1" in text
    assert "the prose" in text
    assert "doc:1" in text


def test_context_runs_the_query_itself(monkeypatch, opened):
    """One argument, the question. Not a run somebody else already ran."""
    engine, store = opened
    asked = []
    monkeypatch.setattr(
        "src.engine._retrieve",
        lambda store_, embedder, query, **kw: asked.append((query, kw)) or FakeRun(["pr:1"]),
    )

    with engine:
        engine.context("who reviewed it?", k=4)

    assert asked[0][0] == "who reviewed it?"
    assert asked[0][1]["total_k"] == 4
    assert store.document_calls == [["pr:1"]]


# --------------------------------------------------------------------------
# the async entry point
# --------------------------------------------------------------------------


def test_the_async_entry_point_returns_what_the_sync_one_does(monkeypatch, opened):
    engine, _store = opened
    monkeypatch.setattr(
        "src.engine._retrieve",
        lambda store, embedder, query, **kwargs: f"result for {query}",
    )

    with engine:
        synchronous = engine.retrieve("who reviewed it?")
        asynchronous = asyncio.run(engine.retrieve_async("who reviewed it?"))

    assert synchronous == asynchronous == "result for who reviewed it?"


def test_the_async_entry_point_runs_off_the_calling_thread(monkeypatch, opened):
    """The reason it exists: the loop's thread must not do the blocking work."""
    import threading

    engine, _store = opened
    threads = []
    monkeypatch.setattr(
        "src.engine._retrieve",
        lambda *a, **k: threads.append(threading.current_thread().name) or "ok",
    )

    async def drive():
        threads.append(f"loop:{threading.current_thread().name}")
        return await engine.retrieve_async("q")

    with engine:
        asyncio.run(drive())

    loop_thread = threads[0].removeprefix("loop:")
    assert threads[1] != loop_thread


def test_an_async_failure_propagates(monkeypatch, opened):
    engine, _store = opened
    monkeypatch.setattr(
        "src.engine._retrieve",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("async boom")),
    )

    with engine:
        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.run(engine.retrieve_async("q"))


def test_the_async_entry_point_forwards_the_result_count(monkeypatch, opened):
    engine, _store = opened
    seen = {}
    monkeypatch.setattr(
        "src.engine._retrieve",
        lambda store, embedder, query, **kwargs: seen.update(kwargs) or "ok",
    )

    with engine:
        asyncio.run(engine.retrieve_async("q", 3))

    assert seen["total_k"] == 3


# ==========================================================================
# the four operations against the real store
# ==========================================================================

from src.common.config import EMBEDDING_DIMENSION
from src.knowledge import GraphBuilder


def _store_available() -> bool:
    """Whether a store can actually be opened here.

    Not ``importorskip`` on the driver. The driver is a thin Python package
    over a native library that is shipped separately, so the module imports
    cleanly on a machine where the library is absent and every open then
    fails — turning a skip into an error. Opening one settles it.
    """
    import tempfile

    try:
        from src.graphdb import open_context_graph

        with tempfile.TemporaryDirectory() as directory:
            open_context_graph(Path(directory) / "probe").close()
        return True
    except Exception:
        return False


STORE_AVAILABLE = _store_available()

requires_store = pytest.mark.skipif(
    not STORE_AVAILABLE, reason="the graph store's native library is not loadable here"
)


class AxisEmbedder:
    """Named texts on their own axis, so a match is exact rather than close."""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def vector(self, text):
        values = [0.0] * EMBEDDING_DIMENSION
        values[self.mapping.get(text, 0)] = 1.0
        return values


@pytest.fixture
def builder():
    graph = GraphBuilder()
    graph.nodes.update(
        {
            "p:alice": {"id": "p:alice", "type": "Person", "label": "alice"},
            "pr:1": {"id": "pr:1", "type": "PR", "label": "fix the auth check"},
        }
    )
    graph.edges.append(
        {"source": "p:alice", "target": "pr:1", "type": "AUTHORED", "confidence": 0.95}
    )
    return graph


@requires_store
def test_the_four_operations_run_in_sequence(tmp_path, builder):
    """Ingest, index, query, context — on one store, through one object."""
    embedder = AxisEmbedder({"alice": 1, "fix the auth check": 2})

    with Engine(tmp_path / "graph", embedder=embedder) as engine:
        stats = engine.ingest(builder)
        assert stats.entities == 2

        engine.build_index()

        run = engine.retrieve("alice")
        assert run.ids

        text = engine.context("alice")
        assert isinstance(text, str)
        assert "p:alice" in text


@requires_store
def test_the_real_handle_is_closed_when_the_body_raises(tmp_path, builder):
    engine = Engine(tmp_path / "graph", embedder=AxisEmbedder())

    with pytest.raises(ValueError):
        with engine:
            engine.ingest(builder)
            raise ValueError("from the body")

    with pytest.raises(EngineNotOpen):
        engine.store


@requires_store
def test_a_query_with_no_extractor_or_judge_still_returns(tmp_path, builder):
    """Absent optional pieces degrade, they do not fail."""
    embedder = AxisEmbedder({"alice": 1})

    with Engine(tmp_path / "graph", embedder=embedder) as engine:
        engine.ingest(builder)
        engine.build_index()
        run = engine.retrieve("alice")

    assert run is not None
    assert run.intent is not None


@requires_store
def test_the_async_and_sync_paths_agree_on_the_real_store(tmp_path, builder):
    embedder = AxisEmbedder({"alice": 1})

    with Engine(tmp_path / "graph", embedder=embedder) as engine:
        engine.ingest(builder)
        engine.build_index()
        synchronous = engine.retrieve("alice")
        asynchronous = asyncio.run(engine.retrieve_async("alice"))

    assert synchronous.ids == asynchronous.ids


# --------------------------------------------------------------------------
# the prompt string
# --------------------------------------------------------------------------


CHUNK = {"id": "doc:1", "path": "https://example.invalid/1", "content": "shared prose"}


def test_a_chunk_two_entities_share_appears_once_with_a_provenance_line(
    monkeypatch, opened
):
    """Five entities on one chunk must not paste the same paragraph five times."""
    engine, store = opened
    store.documents = {"pr:1": [CHUNK], "ticket:2": [CHUNK]}
    monkeypatch.setattr(
        "src.engine._retrieve", lambda *a, **k: FakeRun(["pr:1", "ticket:2"])
    )

    with engine:
        text = engine.context("q")

    assert text.count("shared prose") == 1
    # The second entity is still listed, and says where its text went.
    assert "ticket:2" in text
    assert "already shown" in text
    # And that line sits with the entities, not among the prose.
    entities, prose = text.split("Source text")
    assert "already shown" in entities
    assert "already shown" not in prose


def test_a_section_with_nothing_in_it_is_omitted(monkeypatch, opened):
    """A heading over nothing invites a model to remark on the absence."""
    engine, store = opened
    store.documents = {}
    monkeypatch.setattr("src.engine._retrieve", lambda *a, **k: FakeRun(["pr:1"]))

    with engine:
        text = engine.context("q")

    assert "pr:1" in text
    assert "Source text" not in text


def test_no_results_at_all_produces_an_empty_string(monkeypatch, opened):
    engine, _store = opened
    monkeypatch.setattr("src.engine._retrieve", lambda *a, **k: FakeRun([]))

    with engine:
        assert engine.context("q") == ""


def test_an_entity_type_is_carried_into_the_string(monkeypatch, opened):
    engine, store = opened
    store.documents = {}
    monkeypatch.setattr(
        "src.engine._retrieve", lambda *a, **k: FakeRun(["ticket:9"], node_type="Ticket")
    )

    with engine:
        text = engine.context("q")

    assert "Ticket" in text


# --------------------------------------------------------------------------
# the index builds itself when it has to
# --------------------------------------------------------------------------


def test_a_query_builds_the_index_when_it_has_not_been_built(monkeypatch, opened):
    """Ingest then ask: results, not an error and not a silent empty set."""
    engine, store = opened
    monkeypatch.setattr("src.engine._retrieve", lambda *a, **k: "result")

    with engine:
        assert engine.retrieve("q") == "result"

    assert store.index_builds == [True]


def test_an_explicit_build_means_a_query_does_not_rebuild(monkeypatch, opened):
    """The flag exists so the automatic build happens once, not per query."""
    engine, store = opened
    monkeypatch.setattr("src.engine._retrieve", lambda *a, **k: "result")

    with engine:
        engine.build_index()
        engine.retrieve("q")
        engine.retrieve("q again")

    assert store.index_builds == [True]


def test_the_automatic_build_happens_once_across_several_queries(monkeypatch, opened):
    engine, store = opened
    monkeypatch.setattr("src.engine._retrieve", lambda *a, **k: "result")

    with engine:
        engine.retrieve("one")
        engine.retrieve("two")
        engine.retrieve("three")

    assert store.index_builds == [True]


def test_context_also_gets_an_index_without_an_explicit_build(monkeypatch, opened):
    engine, store = opened
    store.documents = {}
    monkeypatch.setattr("src.engine._retrieve", lambda *a, **k: FakeRun(["pr:1"]))

    with engine:
        engine.context("q")

    assert store.index_builds == [True]
