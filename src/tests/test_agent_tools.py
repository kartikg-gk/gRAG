"""The three agent tools: lookup, search and impact tracing, with their limits."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src import agent_tools
from src.common import config


class FakeStore:
    def __init__(self, labels=None, hits=None, edges=None, documents=None):
        self.labels = labels or {}
        self.hits = hits or []
        self.edges = edges or {}
        self.documents = documents or {}
        self.expansions = []

    def find_by_label(self, label):
        return self.labels.get(label.lower(), [])

    def vector_search(self, vector, k=10):
        return self.hits[:k]

    def expand_frontier(self, ids, k=10, max_degree=None):
        self.expansions.append((list(ids), k, max_degree))
        return {node_id: self.edges.get(node_id, [])[:k] for node_id in ids}

    def documents_for_entities(self, ids):
        return {node_id: self.documents.get(node_id, []) for node_id in ids}


@dataclass
class Node:
    id: str
    label: str
    type: str
    score_total: float
    documents: list = field(default_factory=list)


@dataclass
class Response:
    query: str
    results: list
    trace_log: dict


class FakeRouter:
    def __init__(self, results=None):
        self.results = results or []
        self.routed = []

    def embed_query(self, text):
        return [0.0]

    def route(self, query, top_k=None):
        self.routed.append((query, top_k))
        return Response(query, self.results[:top_k], {"intent": {"type": "semantic"}})


class FakeEngine:
    def __init__(self, store, router=None, warm_error=None):
        self.store = store
        self.router = router or FakeRouter()
        self.warmed = 0
        self.warm_error = warm_error

    def warm(self):
        self.warmed += 1
        if self.warm_error:
            raise self.warm_error


def entity(node_id, label, kind="PR"):
    return {"id": node_id, "label": label, "type": kind}


def neighbour(node_id, label, confidence, kind="Person", relation="AUTHORED"):
    return {"id": node_id, "label": label, "type": kind, "confidence": confidence, "relation": relation}


# -- resolve / find_entity ---------------------------------------------------

def test_an_exact_label_match_wins_and_scores_one():
    store = FakeStore(labels={"auth": [entity("svc:auth", "auth", "Service")]},
                      hits=[{**entity("x", "other"), "similarity": 0.99}])
    found = agent_tools.find_entity(FakeEngine(store), "auth")
    assert found["candidates"] == [{"id": "svc:auth", "label": "auth", "type": "Service",
                                    "match": "exact", "score": 1.0}]


def test_without_an_exact_match_the_nearest_by_meaning_are_returned():
    store = FakeStore(hits=[{**entity("pr:1", "Fix auth"), "similarity": 0.81234}])
    found = agent_tools.find_entity(FakeEngine(store), "authentication")
    assert found["candidates"] == [{"id": "pr:1", "label": "Fix auth", "type": "PR",
                                    "match": "semantic", "score": 0.8123}]


def test_find_entity_caps_the_number_of_candidates():
    hits = [{**entity(f"pr:{i}", f"p{i}"), "similarity": 0.9} for i in range(30)]
    found = agent_tools.find_entity(FakeEngine(FakeStore(hits=hits)), "p", limit=99)
    assert found["match_count"] == config.TOOL_MAX_CANDIDATES


def test_find_entity_with_an_empty_name_says_so():
    found = agent_tools.find_entity(FakeEngine(FakeStore()), "   ")
    assert found["candidates"] == [] and "note" in found


# -- trace_impact ------------------------------------------------------------

def test_a_weak_semantic_match_does_not_start_a_trace():
    store = FakeStore(hits=[{**entity("pr:1", "unrelated"), "similarity": config.TOOL_RESOLVE_MIN_SIM - 0.01}])
    traced = agent_tools.trace_impact(FakeEngine(store), "nonsense")
    assert traced["resolved"] is None and traced["impacted"] == []


def test_path_strength_multiplies_and_records_hops_and_via():
    store = FakeStore(
        labels={"fix": [entity("pr:1", "fix")]},
        edges={
            "pr:1": [neighbour("person:ada", "ada", 0.9)],
            "person:ada": [neighbour("pr:2", "other fix", 0.5, "PR")],
        },
    )
    traced = agent_tools.trace_impact(FakeEngine(store), "fix", max_hops=2)
    rows = {row["id"]: row for row in traced["impacted"]}
    assert rows["person:ada"]["confidence"] == pytest.approx(0.9)
    assert (rows["person:ada"]["hops"], rows["person:ada"]["via"]) == (1, "fix")
    assert rows["pr:2"]["confidence"] == pytest.approx(0.45)
    assert (rows["pr:2"]["hops"], rows["pr:2"]["via"]) == (2, "ada")


def test_results_come_strongest_first():
    store = FakeStore(labels={"fix": [entity("pr:1", "fix")]},
                      edges={"pr:1": [neighbour("a", "a", 0.4), neighbour("b", "b", 0.9)]})
    traced = agent_tools.trace_impact(FakeEngine(store), "fix", max_hops=1)
    assert [row["id"] for row in traced["impacted"]] == ["b", "a"]


def test_the_first_hop_uses_the_generous_hub_threshold_and_later_hops_the_normal_one():
    store = FakeStore(labels={"fix": [entity("pr:1", "fix")]},
                      edges={"pr:1": [neighbour("a", "a", 0.9)], "a": [neighbour("b", "b", 0.9)]})
    agent_tools.trace_impact(FakeEngine(store), "fix", max_hops=2)
    assert store.expansions[0][1:] == (config.TOOL_NEIGHBOR_K, config.TOOL_SEED_MAX_DEGREE)
    assert store.expansions[1][1:] == (config.TOOL_NEIGHBOR_K, config.MAX_DEGREE)


def test_hops_are_clamped_to_the_ceiling():
    store = FakeStore(labels={"fix": [entity("pr:1", "fix")]})
    traced = agent_tools.trace_impact(FakeEngine(store), "fix", max_hops=99)
    assert traced["hops_traversed"] == config.TOOL_MAX_HOPS


def test_the_blast_radius_is_capped():
    """One hop can only reach TOOL_NEIGHBOR_K, so the cap is exercised over two."""
    first = [neighbour(f"a{i}", f"a{i}", 0.9) for i in range(10)]
    second = {f"a{i}": [neighbour(f"b{i}_{j}", f"b{i}_{j}", 0.9) for j in range(10)] for i in range(10)}
    store = FakeStore(labels={"fix": [entity("pr:1", "fix")]}, edges={"pr:1": first, **second})
    traced = agent_tools.trace_impact(FakeEngine(store), "fix", max_hops=2)
    assert traced["blast_radius_count"] == config.TOOL_MAX_IMPACT


def test_citations_are_capped_per_entity_and_cut_to_snippets():
    long_text = "x" * (config.TOOL_SNIPPET_CHARS + 500)
    documents = [{"id": f"d{i}", "path": f"p{i}", "content": long_text} for i in range(5)]
    store = FakeStore(labels={"fix": [entity("pr:1", "fix")]},
                      edges={"pr:1": [neighbour("a", "a", 0.9)]},
                      documents={"a": documents})
    traced = agent_tools.trace_impact(FakeEngine(store), "fix", max_hops=1)
    cites = traced["impacted"][0]["citations"]
    assert len(cites) == config.TOOL_CITATIONS_PER_NODE
    assert all(len(cite["snippet"]) == config.TOOL_SNIPPET_CHARS for cite in cites)
    assert cites[0] == {"doc_id": "d0", "source": "p0", "snippet": long_text[:config.TOOL_SNIPPET_CHARS]}


# -- search_context ----------------------------------------------------------

def test_search_returns_ranked_passages_with_their_sources():
    node = Node("pr:1", "Fix login", "PR", 0.87654,
                [{"doc_id": "d1", "path": "p1", "content": "fixed the login"}])
    engine = FakeEngine(FakeStore(), FakeRouter([node]))
    found = agent_tools.search_context(engine, "why was login fixed?")
    assert found["intent"] == "semantic" and found["result_count"] == 1
    assert found["passages"][0] == {"entity": "Fix login", "type": "PR", "relevance": 0.8765,
                                    "citations": [{"doc_id": "d1", "source": "p1", "snippet": "fixed the login"}]}


def test_search_clamps_how_many_passages_it_asks_for():
    router = FakeRouter()
    agent_tools.search_context(FakeEngine(FakeStore(), router), "q", top_k=500)
    assert router.routed[0][1] == config.TOP_K_VECTOR * 2


def test_search_with_an_empty_question_says_so():
    found = agent_tools.search_context(FakeEngine(FakeStore()), "")
    assert found["passages"] == [] and "note" in found


# -- the LangChain boundary --------------------------------------------------

def test_the_three_tools_are_exposed_by_name():
    pytest.importorskip("langchain_core")
    from src.tools_langchain import graph_tools

    store = FakeStore(labels={"auth": [entity("svc:auth", "auth", "Service")]})
    tools = {tool.name: tool for tool in graph_tools(FakeEngine(store))}
    assert set(tools) == {"trace_impact", "search_context", "find_entity"}
    assert tools["find_entity"].invoke({"name": "auth"})["match_count"] == 1


def test_building_the_tools_warms_the_engine_once():
    pytest.importorskip("langchain_core")
    from src.tools_langchain import graph_tools

    engine = FakeEngine(FakeStore())
    graph_tools(engine)
    assert engine.warmed == 1


def test_a_failed_warm_up_still_returns_working_tools():
    pytest.importorskip("langchain_core")
    from src.tools_langchain import graph_tools

    store = FakeStore(labels={"auth": [entity("svc:auth", "auth", "Service")]})
    engine = FakeEngine(store, warm_error=RuntimeError("no model"))
    tools = {tool.name: tool for tool in graph_tools(engine)}
    assert tools["find_entity"].invoke({"name": "auth"})["match_count"] == 1


def test_warming_can_be_skipped():
    pytest.importorskip("langchain_core")
    from src.tools_langchain import graph_tools

    engine = FakeEngine(FakeStore())
    graph_tools(engine, warm=False)
    assert engine.warmed == 0
