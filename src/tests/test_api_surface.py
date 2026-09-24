"""The HTTP surface: every route under /api, and what each one answers.

The engine and its store are faked. What is under test here is the contract
each route keeps — paths, request and response shapes, the exact texts, the
caches, the rate limit — not retrieval, which is tested where it lives.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api import auth as auth_module
from src.api.app import create_app
from src.retrieval.response import RouterResponse, RoutedNode, build_context
from src.tests.contract_fixtures import trace_log

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PINNED_ROUTES = {
    ("POST", "/api/trace"),
    ("POST", "/api/subgraph"),
    ("POST", "/api/summarize"),
    ("GET", "/api/graphs"),
    ("POST", "/api/graphs/switch"),
    ("GET", "/api/suggestions"),
    ("POST", "/api/answer"),
    ("POST", "/api/answer/stream"),
    ("GET", "/api/health"),
    ("POST", "/api/sessions"),
    ("GET", "/api/sessions"),
    ("GET", "/api/sessions/{session_id}/traces"),
    ("POST", "/api/admin/onboarding/provision"),
    ("POST", "/api/webhooks/github"),
}

#: Pinned too, and served once the encrypted token store they write through
#: exists; until then the rest of the table must hold without them.
GITHUB_CONNECT_ROUTES = {("GET", "/api/github/login"), ("GET", "/api/github/callback")}


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeHit:
    def __init__(self, node_id, score, vector_score, graph_score, decay=1.0, age_days=3.04):
        self.id = node_id
        self.score = score
        self.vector_score = vector_score
        self.graph_score = graph_score
        self.decay = decay
        self.node_type = "PR"
        self.age_days = age_days


class FakeRun:
    def __init__(self, query, hits):
        self.query = query
        self.hits = hits
        self.ids = [hit.id for hit in hits]
        self.trace_log = {"intent": {"alpha": 0.15, "beta": 0.85}}


ENTITIES = {
    "pr:1": {"id": "pr:1", "label": "Fix login", "type": "PR", "timestamp": None},
    "person:ada": {"id": "person:ada", "label": "ada", "type": "Person", "timestamp": None},
}


class FakeStore:
    def __init__(self, documents=None, hubs=None, hub_error=None, nodes=7):
        self.documents = documents or {}
        self.hubs = hubs or []
        self.hub_error = hub_error
        self.nodes = nodes
        self.subgraph_calls = []
        self.hub_limits = []

    def get_entity(self, entity_id):
        return ENTITIES.get(entity_id)

    def documents_for_entities(self, ids):
        return {node_id: self.documents.get(node_id, []) for node_id in ids}

    def subgraph(self, ids):
        self.subgraph_calls.append(list(ids))
        return {"nodes": [{"id": i, "label": i, "type": "PR", "requested": True} for i in ids], "edges": []}

    def top_entities(self, limit=12):
        self.hub_limits.append(limit)
        if self.hub_error is not None:
            raise self.hub_error
        return self.hubs

    def count_nodes(self):
        return self.nodes

    def close(self):
        pass


class FakeEngine:
    def __init__(self, hits=None, store=None, path="fake.lbug"):
        self.hits = hits if hits is not None else [
            FakeHit("pr:1", 0.9, 0.8, 0.95),
            FakeHit("person:ada", 0.5, 0.0, 0.6, decay=0.51234, age_days=None),
        ]
        self._store = store or FakeStore()
        self.queries = []
        self.path = path
        self.embedder = object()
        self.extractor = None
        self.judge = None
        self.now = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        pass

    @property
    def store(self):
        return self._store

    @property
    def router(self):
        return self

    def warm(self):
        pass

    def build_context(self, results):
        return build_context(results)

    async def route_async(self, query, k=None, **kwargs):
        self.queries.append((query, k))
        results = []
        for hit in self.hits:
            entity = ENTITIES.get(hit.id, {})
            documents = [
                {"doc_id": item.get("id"), "path": item.get("path"), "content": item.get("content")}
                for item in self.store.documents_for_entities([hit.id]).get(hit.id, [])
            ]
            age = round(hit.age_days, 1) if hit.age_days is not None else None
            results.append(RoutedNode(hit.id, entity.get("label"), entity.get("type"), hit.score,
                                      hit.vector_score, hit.graph_score, round(hit.decay, 4), age, documents))
        return RouterResponse(query, results, trace_log())

    async def retrieve_async(self, query, k=None, **kwargs):
        self.queries.append((query, k))
        return FakeRun(query, self.hits)


@pytest.fixture(autouse=True)
def single_tenant(monkeypatch):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)


@pytest.fixture(autouse=True)
def fresh_limits():
    from src.api.ratelimit import limiter

    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture()
def engine():
    return FakeEngine()


@pytest.fixture()
def client(engine):
    with TestClient(create_app(engine_factory=lambda: engine)) as made:
        yield made


# ==========================================================================
# 1.1 the route table
# ==========================================================================


def _served(app) -> set[tuple[str, str]]:
    """Every (method, path) the application publishes, included routers too."""
    return {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    }


def test_every_route_is_the_pinned_set_under_api():
    app = create_app(engine_factory=lambda: FakeEngine())

    assert _served(app) - GITHUB_CONNECT_ROUTES == PINNED_ROUTES


def test_no_route_is_served_outside_api():
    app = create_app(engine_factory=lambda: FakeEngine())

    assert [path for _, path in _served(app) if not path.startswith("/api/")] == []


@pytest.mark.parametrize(
    "method,path",
    [("POST", "/query"), ("POST", "/subgraph"), ("GET", "/suggestions"), ("GET", "/graphs"),
     ("POST", "/graphs/switch"), ("GET", "/health"), ("GET", "/projects"), ("GET", "/sessions")],
)
def test_the_old_paths_are_gone(client, method, path):
    assert client.request(method, path).status_code in (404, 405)


def test_the_app_is_named_and_versioned():
    app = create_app(engine_factory=lambda: FakeEngine())
    assert (app.title, app.version) == ("graphRAG API", "0.1.0")


def test_the_routing_gate_covers_exactly_the_three_tenant_routes():
    from src.api.routing import TENANT_SCOPED_PREFIXES

    assert TENANT_SCOPED_PREFIXES == ("/api/trace", "/api/subgraph", "/api/suggestions")


# ==========================================================================
# 1.2 POST /api/trace
# ==========================================================================

NODE_FIELDS = {"id", "label", "type", "score_total", "score_vector", "score_graph",
               "recency", "age_days", "documents", "page_content"}


def test_a_trace_is_the_router_response_with_context(client, engine):
    engine.store.documents = {"pr:1": [{"id": "doc:1", "path": "https://x/1", "content": "Fixed the login"}]}

    body = client.post("/api/trace", json={"query": "who fixed login?"}).json()

    assert set(body) == {"query", "results", "trace_log", "context"}
    assert body["query"] == "who fixed login?"
    assert [set(row) for row in body["results"]] == [NODE_FIELDS, NODE_FIELDS]
    first, second = body["results"]
    assert (first["id"], first["label"], first["type"]) == ("pr:1", "Fix login", "PR")
    assert (first["score_total"], first["score_vector"], first["score_graph"]) == (0.9, 0.8, 0.95)
    assert first["age_days"] == 3.0
    assert first["documents"] == [{"doc_id": "doc:1", "content": "Fixed the login", "path": "https://x/1"}]
    assert first["page_content"] == "Node: Fix login (PR)\n\nContext:\nFixed the login"
    assert second["recency"] == 0.5123
    assert second["age_days"] is None
    assert second["page_content"] == "Node: ada (Person)"
    assert body["context"] == (
        "GRAPH FACTS:\n- Fix login (PR)\n- ada (Person)\n\nSOURCE TEXT:\n[Fix login] Fixed the login"
    )
    assert body["trace_log"] == trace_log()


def test_top_k_defaults_to_the_vector_k(client, engine):
    from src.common.config import TOP_K_VECTOR

    client.post("/api/trace", json={"query": "q"})
    client.post("/api/trace", json={"query": "q", "top_k": 3})

    assert engine.queries == [("q", TOP_K_VECTOR), ("q", 3)]


def test_a_named_session_that_is_the_callers_records_the_trace(monkeypatch, client):
    import src.history as history_module

    saved = {}
    monkeypatch.setattr(history_module, "session_owner", lambda session_id: auth_module.DEV_USER_ID)

    def persist(**kwargs):
        saved.update(kwargs)
        return "trace-1"

    monkeypatch.setattr(history_module, "persist_trace", persist)

    body = client.post("/api/trace", json={"query": "q", "session_id": "s-1"}).json()

    assert body["trace_id"] == "trace-1"
    assert saved["session_id"] == "s-1"
    assert saved["query"] == "q"
    assert saved["execution_plan"] == body["trace_log"]
    assert saved["graph_payload"] == body["results"]
    assert saved["graph_id"] == "fake.lbug"


@pytest.mark.parametrize("owner", [None, "someone-else"])
def test_a_missing_or_foreign_session_is_the_one_uniform_refusal(monkeypatch, client, engine, owner):
    import src.history as history_module

    monkeypatch.setattr(history_module, "session_owner", lambda session_id: owner)

    response = client.post("/api/trace", json={"query": "q", "session_id": "s-1"})

    assert response.status_code == 404
    assert response.json() == {"detail": "no such session"}


def test_a_tenant_this_pod_does_not_hold_is_unavailable(monkeypatch, engine):
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)

    async def some_org():
        yield "org_elsewhere"

    app = create_app(engine_factory=lambda: engine)
    app.dependency_overrides[auth_module.get_current_tenant_org] = some_org
    from src.api import routing

    monkeypatch.setattr(routing.TenantRoutingMiddleware, "dispatch", lambda self, request, call_next: call_next(request))
    with TestClient(app) as client:
        response = client.post("/api/trace", json={"query": "q"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Tenant graph not loaded on this pod: org_elsewhere"}


def test_the_context_and_page_content_texts_are_exact():
    from src.retrieval.response import RoutedNode, build_context, format_page_content

    doc = {"doc_id": "d1", "content": "  shared text  ", "path": "p"}
    first = RoutedNode("a", "Alpha", "PR", 1.0, 1.0, 1.0, documents=[doc, {"doc_id": "d2", "content": " ", "path": "p"}])
    second = RoutedNode("b", None, None, 0.5, 0.5, 0.5, documents=[doc])

    assert build_context([first, second]) == (
        "GRAPH FACTS:\n- Alpha (PR)\n- b (Unknown)\n  [Trace: b (Unknown) -> d1]"
        "\n\nSOURCE TEXT:\n[Alpha] shared text"
    )
    assert format_page_content(first) == "Node: Alpha (PR)\n\nContext:\nshared text"
    seen = {"d1"}
    assert format_page_content(second, seen) == "Node: b (Unknown)\n\nContext:\n[Trace: b (Unknown) -> d1]"
    assert format_page_content(RoutedNode("c", "C", "PR", 0, 0, 0)) == "Node: C (PR)"


# ==========================================================================
# 1.3 POST /api/subgraph
# ==========================================================================


def test_the_subgraph_takes_node_ids(client, engine):
    body = client.post("/api/subgraph", json={"node_ids": ["pr:1"]}).json()

    assert engine.store.subgraph_calls == [["pr:1"]]
    assert body == {"nodes": [{"id": "pr:1", "label": "pr:1", "type": "PR", "requested": True}], "edges": []}


# ==========================================================================
# 1.4 GET /api/suggestions
# ==========================================================================


def test_suggestions_take_one_per_type_then_backfill_by_degree(client, engine):
    engine.store.hubs = [
        {"id": "p1", "label": "ada", "type": "Person"},
        {"id": "p2", "label": "bob", "type": "Person"},
        {"id": "x1", "label": " ", "type": "PR"},
        {"id": "t1", "label": "Login bug", "type": "Ticket"},
        {"id": "s1", "label": "auth", "type": "Service"},
        {"id": "w1", "label": "widget", "type": "Weird"},
    ]

    body = client.get("/api/suggestions", params={"limit": 5}).json()

    assert engine.store.hub_limits == [20]
    assert body == {"suggestions": [
        {"query": "Show recent contributions by ada.", "entity": "ada", "type": "Person"},
        {"query": "Which work items connect to Login bug?", "entity": "Login bug", "type": "Ticket"},
        {"query": "Which components rely on auth?", "entity": "auth", "type": "Service"},
        {"query": "Explore connections around widget.", "entity": "widget", "type": "Weird"},
        {"query": "Show recent contributions by bob.", "entity": "bob", "type": "Person"},
    ]}


def test_a_store_that_cannot_list_hubs_suggests_nothing(client, engine):
    engine.store.hub_error = RuntimeError("no")

    assert client.get("/api/suggestions").json() == {"suggestions": []}


@pytest.mark.parametrize(
    "kind,template",
    [("Team", "Which areas are maintained by x?"), ("Library", "Which changes involve x?"),
     ("Tool", "Where is x used?"), ("PR", "Which entities connect to x?")],
)
def test_each_type_has_its_template(client, engine, kind, template):
    engine.store.hubs = [{"id": "1", "label": "x", "type": kind}]

    assert client.get("/api/suggestions").json()["suggestions"][0]["query"] == template


# ==========================================================================
# 1.5 graphs
# ==========================================================================


@pytest.fixture()
def graph_files(tmp_path, monkeypatch):
    import src.graphs as graphs_module

    directory = tmp_path / "graphs"
    directory.mkdir()
    (directory / "pallets__click.lbug").write_bytes(b"")
    (directory / "old.db").write_bytes(b"")
    default = tmp_path / "graph.lbug"
    default.write_bytes(b"")
    monkeypatch.setattr(graphs_module, "GRAPHS_DIRECTORY", directory)
    monkeypatch.setattr(graphs_module, "STORE_PATH", str(default))
    import src.api.app as app_module

    monkeypatch.setattr(app_module, "STORE_PATH", str(default))
    return directory, default


def test_graphs_are_lbug_files_named_by_file_name(graph_files):
    directory, default = graph_files
    engine = FakeEngine(path=str(default))

    with TestClient(create_app(engine_factory=lambda: engine)) as client:
        body = client.get("/api/graphs").json()

    assert body == {
        "graphs": [
            {"id": "graph.lbug", "label": "graph (default)", "active": True},
            {"id": "pallets__click.lbug", "label": "pallets/click", "active": False},
        ],
        "active": "graph.lbug",
    }


def test_switching_to_an_unknown_graph_is_not_found(graph_files):
    _, default = graph_files
    with TestClient(create_app(engine_factory=lambda: FakeEngine(path=str(default)))) as client:
        response = client.post("/api/graphs/switch", json={"id": "nope.lbug"})

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown graph: nope.lbug"}


def test_switching_answers_with_the_new_graph_and_clears_summaries(graph_files, monkeypatch):
    from src.api import app as app_module
    from src.registry import REGISTRY

    directory, default = graph_files
    engine = FakeEngine(path=str(default))
    swapped = FakeEngine(path=str(directory / "pallets__click.lbug"), store=FakeStore(nodes=42))

    with TestClient(create_app(engine_factory=lambda: engine)) as client:
        REGISTRY.set_loader(lambda path: swapped)
        app_module.SUMMARY_CACHE.set("k", "v")
        body = client.post("/api/graphs/switch", json={"id": "pallets__click.lbug"}).json()
        listed = client.get("/api/graphs").json()

    assert body == {"active": "pallets__click.lbug", "label": "pallets/click", "nodes": 42}
    assert len(app_module.SUMMARY_CACHE) == 0
    assert listed["active"] == "pallets__click.lbug"


def test_the_default_store_is_graph_lbug_in_the_project_root(tmp_path):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GRAPHRAG_")}
    environment.update(GRAPHRAG_ENV_FILE="", PYTHONPATH=str(PROJECT_ROOT))
    completed = subprocess.run(
        [sys.executable, "-c", "from src.common.config import STORE_PATH; print(STORE_PATH)"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == PROJECT_ROOT / "graph.lbug"


# ==========================================================================
# 1.6 the LRU cache
# ==========================================================================


def test_the_cache_refuses_a_non_positive_capacity():
    from src.cache import LRUCache

    assert LRUCache().capacity == 256
    for bad in (0, -1):
        with pytest.raises(ValueError):
            LRUCache(bad)


def test_the_cache_evicts_the_least_recently_used():
    from src.cache import LRUCache

    cache = LRUCache(2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1          # promotes a
    cache.set("c", 3)                   # evicts b
    assert "b" not in cache and cache.get("a") == 1 and cache.get("c") == 3
    assert cache.get("missing") is None


def test_membership_does_not_promote_but_overwrite_does():
    from src.cache import LRUCache

    cache = LRUCache(2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert "a" in cache                 # no promotion
    cache.set("c", 3)                   # evicts a
    assert "a" not in cache
    cache.set("b", 20)                  # promotes b
    cache.set("d", 4)                   # evicts c
    assert "c" not in cache and cache.get("b") == 20 and len(cache) == 2
    cache.clear()
    assert len(cache) == 0


# ==========================================================================
# 1.7 / 1.8 the model routes
# ==========================================================================


class FakeCompletions:
    def __init__(self, text="A short answer.", error=None, stream_parts=None):
        self.text = text
        self.error = error
        self.stream_parts = stream_parts
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return self._stream()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"  {self.text}  "))])

    def _stream(self):
        for part in self.stream_parts or []:
            if isinstance(part, Exception):
                raise part
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=part))])
        yield SimpleNamespace(choices=[])


@pytest.fixture()
def model(monkeypatch):
    import src.api.app as app_module

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(app_module, "general_client", lambda: (client, "general-model"))
    app_module.SUMMARY_CACHE.clear()
    app_module.ANSWER_CACHE.clear()
    return completions


SUMMARY_PROMPT = (
    "Condense this engineering note into a single sentence of at most 25 words "
    "covering what changed and why. Return only the sentence.\n\n"
)


def test_a_summary_uses_the_pinned_prompt_and_is_cached(client, model):
    first = client.post("/api/summarize", json={"key": "k1", "text": "note"}).json()
    second = client.post("/api/summarize", json={"key": "k1", "text": "note"}).json()

    assert first == {"summary": "A short answer.", "cached": False}
    assert second == {"summary": "A short answer.", "cached": True}
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["model"] == "general-model"
    assert call["temperature"] == 0
    assert call["messages"] == [{"role": "user", "content": SUMMARY_PROMPT + "note"}]


def test_a_summary_keys_on_the_text_when_no_key_is_given(client, model):
    client.post("/api/summarize", json={"key": "", "text": "x" * 100})
    again = client.post("/api/summarize", json={"key": "", "text": "x" * 100}).json()

    assert again["cached"] is True
    assert len(model.calls) == 1


def test_the_same_key_with_different_text_is_not_served_from_cache(client, model):
    """The cross-tenant leak: two organisations share an entity id, and the
    second must not receive a summary of the first one's text."""
    client.post("/api/summarize", json={"key": "entity:service:auth", "text": "tenant A notes"})
    second = client.post(
        "/api/summarize", json={"key": "entity:service:auth", "text": "tenant B notes"}
    ).json()

    assert second["cached"] is False
    assert len(model.calls) == 2
    assert model.calls[1]["messages"][0]["content"].endswith("tenant B notes")


def test_empty_text_summarises_to_nothing_without_a_call(client, model):
    assert client.post("/api/summarize", json={"key": "k", "text": "   "}).json() == {"summary": "", "cached": False}
    assert model.calls == []


def test_a_failed_summary_reports_the_error(client, model):
    model.error = RuntimeError("down")

    assert client.post("/api/summarize", json={"key": "k", "text": "t"}).json() == {
        "summary": "", "cached": False, "error": "down"}


def test_an_unconfigured_model_is_the_routes_own_failure(client, monkeypatch):
    import src.api.app as app_module
    from src.common.judge import JudgeError

    def unconfigured():
        raise JudgeError("not configured: set GRAPHRAG_JUDGE_MODEL")

    monkeypatch.setattr(app_module, "general_client", unconfigured)
    app_module.SUMMARY_CACHE.clear()

    body = client.post("/api/summarize", json={"key": "k", "text": "t"}).json()

    assert body["summary"] == "" and body["cached"] is False and "GRAPHRAG_JUDGE_MODEL" in body["error"]


ANSWER_PROMPT = (
    "Answer a colleague's question about a software knowledge graph using ONLY "
    "the context below. Reply in two or three plain sentences that a non-expert "
    "can follow, naming the exact PRs, people and components involved. If the "
    "context lacks the answer, say so plainly instead of guessing.\n\nQuestion: q\n\nContext:\nctx"
)


def test_an_answer_uses_the_pinned_prompt_and_is_cached(client, model):
    first = client.post("/api/answer", json={"query": "q", "context": "ctx"}).json()
    second = client.post("/api/answer", json={"query": "q", "context": "ctx"}).json()

    assert first == {"answer": "A short answer.", "cached": False}
    assert second == {"answer": "A short answer.", "cached": True}
    assert model.calls[0]["messages"] == [{"role": "user", "content": ANSWER_PROMPT}]
    assert model.calls[0]["temperature"] == 0


def test_an_answer_with_no_context_says_so(client, model):
    assert client.post("/api/answer", json={"query": "q", "context": "  "}).json() == {
        "answer": "Nothing relevant was retrieved for this question.", "cached": False}
    assert model.calls == []


def test_a_failed_answer_reports_the_error(client, model):
    model.error = RuntimeError("boom")

    assert client.post("/api/answer", json={"query": "q", "context": "ctx"}).json() == {
        "answer": "", "cached": False, "error": "boom"}


def test_a_streamed_answer_arrives_in_parts_and_is_cached(client, model):
    model.stream_parts = ["Ada ", "", "fixed it."]

    response = client.post("/api/answer/stream", json={"query": "q", "context": "ctx"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text == "Ada fixed it."
    assert model.calls[0]["stream"] is True
    again = client.post("/api/answer", json={"query": "q", "context": "ctx"}).json()
    assert again == {"answer": "Ada fixed it.", "cached": True}


def test_a_stream_with_no_context_says_so(client, model):
    assert client.post("/api/answer/stream", json={"query": "q", "context": ""}).text == (
        "Nothing relevant was retrieved for this question.")


def test_a_stream_that_fails_before_any_text_apologises(client, model):
    model.error = RuntimeError("down")

    assert client.post("/api/answer/stream", json={"query": "q", "context": "ctx"}).text == (
        "The answer could not be produced right now.")


def test_a_stream_that_fails_part_way_keeps_what_it_sent(client, model):
    model.stream_parts = ["Partial ", RuntimeError("cut")]

    assert client.post("/api/answer/stream", json={"query": "q", "context": "ctx"}).text == "Partial "


# ==========================================================================
# 1.9 rate limiting
# ==========================================================================


@pytest.mark.parametrize("path,body", [
    ("/api/summarize", {"key": "k", "text": "t"}),
    ("/api/answer", {"query": "q", "context": "c"}),
    ("/api/answer/stream", {"query": "q", "context": "c"}),
])
def test_the_model_routes_allow_ten_a_minute_per_user(client, model, path, body):
    codes = [client.post(path, json=body).status_code for _ in range(11)]

    assert codes == [200] * 10 + [429]


def test_the_limit_key_is_the_user_then_the_address():
    from src.api.ratelimit import LLM_RATE_LIMIT, _user_key

    with_user = SimpleNamespace(state=SimpleNamespace(user_id="u1"), client=SimpleNamespace(host="1.2.3.4"),
                                headers={}, scope={"client": ("1.2.3.4", 1)})
    without = SimpleNamespace(state=SimpleNamespace(), client=SimpleNamespace(host="1.2.3.4"),
                              headers={}, scope={"client": ("1.2.3.4", 1)})

    assert LLM_RATE_LIMIT == "10/minute"
    assert _user_key(with_user) == "user:u1"
    assert _user_key(without) == "ip:1.2.3.4"


def test_the_limiter_is_on_the_app():
    from src.api.ratelimit import limiter

    assert create_app(engine_factory=lambda: FakeEngine()).state.limiter is limiter


# ==========================================================================
# 1.11 health
# ==========================================================================


def test_health_reports_the_node_count(client):
    assert client.get("/api/health").json() == {"status": "ok", "nodes": 7}


# ==========================================================================
# 1.12 credentials
# ==========================================================================


def _fresh(code, **settings):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GRAPHRAG_")}
    environment.update(GRAPHRAG_ENV_FILE="", PYTHONPATH=str(PROJECT_ROOT), **settings)
    completed = subprocess.run([sys.executable, "-c", code], env=environment,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()[-1]


def test_verification_is_on_exactly_when_an_issuer_is_set():
    code = "from src.common import config as c; print(c.CLERK_ENABLED, c.CLERK_JWKS_URL, c.DEV_USER_ID)"

    assert _fresh(code) == "False  dev-user"
    assert _fresh(code, GRAPHRAG_CLERK_ISSUER="https://id.example/") == (
        "True https://id.example/.well-known/jwks.json dev-user")
    assert _fresh(code, GRAPHRAG_CLERK_ISSUER="https://id.example", GRAPHRAG_CLERK_JWKS_URL="https://k/j") == (
        "True https://k/j dev-user")


def test_the_old_verification_flag_is_gone():
    from src.common import config

    assert not hasattr(config, "CLERK_ENABLED_VARIABLE")
    assert "GRAPHRAG_CLERK_ENABLED" not in Path(config.__file__).read_text(encoding="utf-8")


def test_tenant_key_has_its_own_header_when_clerk_is_enabled(monkeypatch):
    from starlette.requests import Request

    def request(headers):
        return Request({"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]})

    monkeypatch.setattr(auth_module, "CLERK_ENABLED", True)
    both = request({
        "Authorization": "Bearer clerk-session",
        "X-Graphrag-API-Key": "sk-1",
    })
    assert auth_module.api_key_from(both) == "sk-1"
    assert auth_module.bearer_token(both) == "clerk-session"
    assert auth_module.api_key_from(request({"Authorization": "Bearer sk-1"})) is None


def test_bearer_api_key_remains_compatible_without_clerk(monkeypatch):
    from starlette.requests import Request

    request = Request({
        "type": "http",
        "headers": [(b"authorization", b"Bearer sk-1")],
    })
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)

    assert auth_module.api_key_from(request) == "sk-1"


def test_a_bad_key_is_refused_with_a_bearer_challenge(monkeypatch, engine):
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)

    def reject(key):
        raise auth_module._unauthorized()

    monkeypatch.setattr(auth_module, "resolve_org", reject)
    with TestClient(create_app(engine_factory=lambda: engine)) as client:
        response = client.post("/api/trace", json={"query": "q"}, headers={"Authorization": "Bearer nope"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"detail": auth_module.INVALID_CREDENTIALS}


# ==========================================================================
# 1.14 error reporting
# ==========================================================================


SENTRY_PROBE = """
import sys, types
calls = []
stub = types.ModuleType("sentry_sdk")
stub.init = lambda **kwargs: calls.append(kwargs)
sys.modules["sentry_sdk"] = stub
from src.api.app import create_app
create_app(engine_factory=lambda: None)
print(sorted(calls[0].items()) if calls else "none")
"""


def test_error_reporting_is_off_without_a_dsn():
    assert _fresh(SENTRY_PROBE) == "none"


def test_error_reporting_starts_with_the_pinned_settings():
    assert _fresh(SENTRY_PROBE, GRAPHRAG_SENTRY_DSN="https://k@example/1") == (
        "[('dsn', 'https://k@example/1'), ('environment', 'development'), "
        "('send_default_pii', False), ('traces_sample_rate', 1.0)]"
    )
    assert _fresh(SENTRY_PROBE, GRAPHRAG_SENTRY_DSN="d", GRAPHRAG_SENTRY_ENVIRONMENT="prod",
                  GRAPHRAG_SENTRY_TRACES_SAMPLE_RATE="0.25") == (
        "[('dsn', 'd'), ('environment', 'prod'), ('send_default_pii', False), ('traces_sample_rate', 0.25)]"
    )


def test_the_answer_client_gives_up_after_its_timeout(monkeypatch):
    import src.api.app as app_module
    import src.common.judge as judge_module
    from openai import OpenAI

    from src.common.config import ANSWER_TIMEOUT_SECONDS

    monkeypatch.setattr(
        judge_module, "chat_client",
        lambda: OpenAI(api_key="test", base_url="http://127.0.0.1:9"),
    )
    monkeypatch.setattr(app_module, "_llm", {})
    monkeypatch.setattr("src.common.config.JUDGE_MODEL", "m")

    client, model = app_module.general_client()

    assert model == "m"
    assert client.timeout == ANSWER_TIMEOUT_SECONDS
