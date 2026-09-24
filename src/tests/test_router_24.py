from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep
from types import SimpleNamespace

from src.retrieval.router import RetrievalRouter


class Embedder:
    def __init__(self):
        self.calls = []

    def vector(self, text):
        self.calls.append(text)
        return [float(len(text))]


class Extractor:
    def extract(self, query):
        return [SimpleNamespace(text="#412")]


class Store:
    def vector_search(self, embedding, k):
        return [
            {"id": "v1", "label": "Vector one", "type": "PR", "timestamp": None, "similarity": .9},
            {"id": "v2", "label": "Vector two", "type": "File", "timestamp": None, "similarity": .4},
        ]

    def find_by_label(self, label):
        return [{"id": "linked", "label": "#412", "type": "Ticket", "timestamp": None}]

    def expand_frontier(self, ids, k, max_degree):
        if "linked" not in ids:
            return {node_id: [] for node_id in ids}
        return {
            node_id: ([{"id": "graph", "label": "Graph hit", "type": "PR", "relation": None,
                        "confidence": .8, "timestamp": None}] if node_id == "linked" else [])
            for node_id in ids
        }

    def documents_for_entities(self, ids):
        return {node_id: [{"doc_id": "d1", "content": "shared evidence", "path": "p"}] for node_id in ids}


def test_route_emits_the_exact_2_4_contract():
    response = RetrievalRouter(Store(), Embedder(), extractor=Extractor()).route("who fixed #412?", 3)

    assert response.query == "who fixed #412?"
    assert [field for field in vars(response)] == ["query", "results", "trace_log"]
    trace = response.trace_log
    assert set(trace) == {"intent", "execution_path", "recency", "metrics"}
    assert set(trace["intent"]) == {"alpha", "beta", "type"}
    assert trace["intent"] == {"alpha": .15, "beta": .85, "type": "relational"}
    assert trace["execution_path"]["linked_seeds"] == ["linked"]
    # Something is named, so traversal starts only there.
    assert trace["execution_path"]["vector_seeds"] == ["linked"]
    assert set(trace["execution_path"]["graph_hops"][0]) == {
        "from_id", "to_id", "confidence", "relation"
    }
    assert trace["execution_path"]["graph_hops"][0]["relation"] == "CO_OCCURS"
    assert isinstance(trace["recency"]["applied"], list)
    # linked, graph, v1, v2: the vector hits are still evaluated, only not walked from.
    assert trace["metrics"]["total_nodes_evaluated"] == 4


def test_semantic_markers_and_query_bound_are_pinned():
    response = RetrievalRouter(Store(), Embedder()).route("explain " + "x" * 3000)

    assert len(response.query) == 2000
    assert response.trace_log["intent"] == {"alpha": .8, "beta": .2, "type": "semantic"}


def test_context_deduplicates_document_text_but_keeps_the_trace():
    router = RetrievalRouter(Store(), Embedder())
    response = router.route("what is this?", 2)

    context = router.build_context(response.results)
    assert context.count("shared evidence") == 1
    assert "[Trace:" in context


def test_empty_and_whitespace_queries_still_run_vector_recall():
    for query in ("", "   "):
        embedder = Embedder()
        response = RetrievalRouter(Store(), embedder).route(query)
        assert embedder.calls == [query]
        assert response.results


def test_query_extraction_cache_is_shared_across_graph_swaps():
    extractor = Extractor()
    extractor.calls = 0
    original = extractor.extract

    def counted(query):
        extractor.calls += 1
        return original(query)

    extractor.extract = counted
    RetrievalRouter(Store(), Embedder(), extractor=extractor)._entities("swap query")
    RetrievalRouter(Store(), Embedder(), extractor=extractor)._entities("swap query")
    assert extractor.calls == 1


def test_query_extraction_cache_is_isolated_between_extractors():
    class NamedExtractor:
        def __init__(self, name):
            self.name = name
            self.calls = 0

        def extract(self, query):
            self.calls += 1
            return [SimpleNamespace(text=self.name)]

    first = NamedExtractor("first")
    second = NamedExtractor("second")
    first_entities = RetrievalRouter(Store(), Embedder(), extractor=first)._entities(
        "extractor-scoped query"
    )
    second_entities = RetrievalRouter(Store(), Embedder(), extractor=second)._entities(
        "extractor-scoped query"
    )

    assert [entity.text for entity in first_entities] == ["first"]
    assert [entity.text for entity in second_entities] == ["second"]
    assert first.calls == second.calls == 1


def test_query_extraction_cache_does_not_serialize_unrelated_misses():
    barrier = Barrier(3)

    class SlowExtractor(Extractor):
        def __init__(self):
            self.calls = 0
            self.active = 0
            self.max_active = 0
            self.lock = Lock()

        def extract(self, query):
            with self.lock:
                self.calls += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            sleep(.05)
            try:
                return super().extract(query)
            finally:
                with self.lock:
                    self.active -= 1

    extractor = SlowExtractor()
    router = RetrievalRouter(Store(), Embedder(), extractor=extractor)

    def read(query):
        barrier.wait(timeout=2)
        return router._entities(query)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(read, "threaded unique query one")
        second = pool.submit(read, "threaded unique query two")
        barrier.wait(timeout=2)
        first.result(timeout=2)
        second.result(timeout=2)
    assert extractor.calls == 2
    assert extractor.max_active == 2


def test_query_extraction_cache_capacity_and_zero_disable(monkeypatch):
    import src.retrieval.router as router_module

    class CountingExtractor(Extractor):
        def __init__(self):
            self.calls = []

        def extract(self, query):
            self.calls.append(query)
            return super().extract(query)

    extractor = CountingExtractor()
    router = RetrievalRouter(Store(), Embedder(), extractor=extractor)
    monkeypatch.setattr(router_module, "QUERY_EXTRACT_CACHE", 2)
    for query in ("bounded one", "bounded two", "bounded one", "bounded three", "bounded two"):
        router._entities(query)
    assert extractor.calls == ["bounded one", "bounded two", "bounded three", "bounded two"]

    monkeypatch.setattr(router_module, "QUERY_EXTRACT_CACHE", 0)
    router._entities("disabled cache")
    router._entities("disabled cache")
    assert extractor.calls[-2:] == ["disabled cache", "disabled cache"]


def test_warm_uses_the_shared_cached_extraction_path():
    extractor = Extractor()
    extractor.calls = 0
    original = extractor.extract

    def counted(query):
        extractor.calls += 1
        return original(query)

    extractor.extract = counted
    first = RetrievalRouter(Store(), Embedder(), extractor=extractor)
    first.warm()
    RetrievalRouter(Store(), Embedder(), extractor=extractor)._entities("warmup query")
    assert extractor.calls == 1


class LookalikeStore(Store):
    """One person the query names, and lookalike usernames the text model
    places near it."""

    def vector_search(self, embedding, k):
        return [
            {"id": "person:kevinjosethomas", "label": "kevinjosethomas", "type": "Person",
             "timestamp": None, "similarity": .78},
            {"id": "person:robertkeus", "label": "robertkeus", "type": "Person",
             "timestamp": None, "similarity": .44},
            {"id": "person:keejkrej", "label": "keejkrej", "type": "Person",
             "timestamp": None, "similarity": .41},
        ]

    def find_by_label(self, label):
        if label.lower() == "kevinjosethomas":
            return [{"id": "person:kevinjosethomas", "label": "kevinjosethomas",
                     "type": "Person", "timestamp": None}]
        return []

    def expand_frontier(self, ids, k, max_degree):
        authored = {
            "person:kevinjosethomas": "pr:kevins",
            "person:robertkeus": "pr:roberts",
            "person:keejkrej": "pr:keejs",
        }
        return {
            node_id: ([{"id": authored[node_id], "label": authored[node_id], "type": "PR",
                        "relation": "AUTHORED_BY", "confidence": .95, "timestamp": None}]
                      if node_id in authored else [])
            for node_id in ids
        }


class NoEntities:
    def extract(self, query):
        return []


def test_a_named_username_is_the_only_place_traversal_starts():
    response = RetrievalRouter(LookalikeStore(), Embedder(), extractor=NoEntities()).route(
        "What did kevinjosethomas work on?", 10
    )
    path = response.trace_log["execution_path"]
    reached = {result.id for result in response.results}

    assert path["linked_seeds"] == ["person:kevinjosethomas"]
    assert path["vector_seeds"] == ["person:kevinjosethomas"]
    assert "pr:kevins" in reached
    # A lookalike username no longer brings its owner's work in.
    assert "pr:roberts" not in reached and "pr:keejs" not in reached


def test_with_nothing_named_similar_vectors_still_choose_the_start():
    response = RetrievalRouter(LookalikeStore(), Embedder(), extractor=NoEntities()).route(
        "who changed the escaping code?", 10
    )
    path = response.trace_log["execution_path"]

    assert path["linked_seeds"] == []
    assert path["vector_seeds"] == [
        "person:kevinjosethomas", "person:robertkeus", "person:keejkrej",
    ]


def test_short_words_and_numbers_are_not_looked_up_as_names():
    looked_up = []

    class Recording(LookalikeStore):
        def find_by_label(self, label):
            looked_up.append(label)
            return []

    RetrievalRouter(Recording(), Embedder(), extractor=NoEntities()).route("is it 412 or kevin-j.t?", 5)

    assert looked_up == ["kevin-j.t"]


class CountingJudge:
    def __init__(self, answer=False):
        self.answer = answer
        self.asked = []

    def relational(self, query):
        self.asked.append(query)
        return self.answer


def test_a_query_naming_a_node_is_relational_without_asking_the_judge():
    judge = CountingJudge(answer=False)
    response = RetrievalRouter(LookalikeStore(), Embedder(), extractor=NoEntities(), judge=judge).route(
        "What did kevinjosethomas work on?", 10
    )

    assert response.trace_log["intent"]["type"] == "relational"
    assert response.trace_log["intent"]["beta"] == .85
    assert judge.asked == []


def test_a_query_naming_nothing_is_still_put_to_the_judge():
    judge = CountingJudge(answer=False)
    response = RetrievalRouter(LookalikeStore(), Embedder(), extractor=NoEntities(), judge=judge).route(
        "What changed in escaping lately?", 10
    )

    assert judge.asked == ["What changed in escaping lately?"]
    assert response.trace_log["intent"]["type"] == "semantic"


def test_a_semantic_marker_still_wins_over_a_name():
    from src.retrieval.intent import classify

    assert classify("explain kevinjosethomas", named=True).intent == "semantic"
    assert classify("what did kevinjosethomas do", named=True).stage == "named"
    assert classify("who is kevinjosethomas", named=True).stage == "marker"
