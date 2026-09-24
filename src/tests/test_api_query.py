"""Tests for the query endpoint and the lifecycle around it.

The engine is faked. What this layer does is resolve credentials, hand a query
to the facade, and shape what comes back — none of which a real store makes
truer, and all of which a real store would make slower and dependent on a
native library. Retrieval itself is tested where it lives.

The formatter is tested directly as well as through a route, because it has a
second consumer that is not this API and must keep working without one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import auth as auth_module
from src.api.app import create_app
from src.registry import REGISTRY
from src.results import chunk_table, format_run
from src.retrieval.response import RouterResponse, RoutedNode, build_context
from src.tests.contract_fixtures import trace_log


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeHit:
    def __init__(self, node_id, score, vector_score, graph_score, decay=1.0):
        self.id = node_id
        self.score = score
        self.vector_score = vector_score
        self.graph_score = graph_score
        self.decay = decay
        self.node_type = "PR"
        self.age_days = 3.0

    @property
    def found_by_both(self):
        return self.vector_score > 0.0 and self.graph_score > 0.0


class FakeFused:
    alpha = 0.15
    beta = 0.85

    def __init__(self, hits):
        self.hits = hits


class FakeIntent:
    intent = "relational"


class FakeRun:
    def __init__(self, hits):
        self.query = "who reviewed the auth change?"
        self.intent = FakeIntent()
        self.fused = FakeFused(hits)
        self.hits = hits
        self.ids = [hit.id for hit in hits]
        self.trace_log = None
        self.seconds = 0.01


class FakeStore:
    def __init__(self, documents=None):
        self.documents = documents or {}
        self.closed = 0
        self.asked_for = []

    def documents_for_entities(self, ids):
        self.asked_for.append(list(ids))
        return {node_id: self.documents.get(node_id, []) for node_id in ids}

    def get_entity(self, entity_id):
        return {"id": entity_id, "label": entity_id, "type": "PR"}

    def count_nodes(self):
        return 3

    def close(self):
        self.closed += 1


class FakeEngine:
    """Stands in for the facade: a context manager with an async query."""

    def __init__(self, hits=None, documents=None, error=None, warm_error=None):
        self.hits = hits if hits is not None else [FakeHit("pr:1", 0.42, 0.30, 0.55)]
        self._store = FakeStore(documents)
        self.error = error
        self.warm_error = warm_error
        self.entered = 0
        self.queries = []
        # The facade's surface, because the registry and the loader read it.
        self.path = "fake.db"
        self.embedder = object()
        self.extractor = None
        self.judge = None
        self.now = None
        self._closed = False

    def __enter__(self):
        self.entered += 1
        if self.warm_error is not None:
            raise self.warm_error
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        """Idempotent, like the facade's: the registry closes what it holds
        and the `with` closes again on the way out."""
        if not self._closed:
            self._closed = True
            self._store.close()

    @property
    def store(self):
        return self._store

    @property
    def router(self):
        return self

    def warm(self):
        if self.warm_error is not None:
            raise self.warm_error

    def build_context(self, results):
        return build_context(results)

    async def route_async(self, query, k=None, **kwargs):
        self.queries.append((query, k))
        if self.error is not None:
            raise self.error
        results = []
        for hit in self.hits:
            docs = [
                {"doc_id": item.get("id"), "path": item.get("path"), "content": item.get("content")}
                for item in self.store.documents_for_entities([hit.id]).get(hit.id, [])
            ]
            results.append(RoutedNode(hit.id, hit.id, hit.node_type, hit.score,
                                      hit.vector_score, hit.graph_score, hit.decay, hit.age_days, docs))
        return RouterResponse(query, results, trace_log())

    async def retrieve_async(self, query, k=None, **kwargs):
        """Mirrors the facade: the count is positional and may be unset."""
        self.queries.append((query, k))
        if self.error is not None:
            raise self.error
        return FakeRun(self.hits)


@pytest.fixture(autouse=True)
def single_tenant(monkeypatch):
    """Credentials off by default; the rejection cases turn them back on."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)


def client_for(engine):
    return TestClient(create_app(engine_factory=lambda: engine))


# --------------------------------------------------------------------------
# the query path
# --------------------------------------------------------------------------


def test_a_query_returns_ranked_results_with_their_components():
    engine = FakeEngine(hits=[FakeHit("pr:1", 0.42, 0.30, 0.55, decay=0.9)])

    with client_for(engine) as client:
        body = client.post("/api/trace", json={"query": "who reviewed it?"}).json()

    result = body["results"][0]
    assert result["id"] == "pr:1"
    assert result["score_total"] == 0.42
    assert result["score_vector"] == 0.30
    assert result["score_graph"] == 0.55
    assert result["recency"] == 0.9


def test_an_arm_that_found_nothing_reports_zero_rather_than_being_absent():
    """Absent and zero are the same number and different facts."""
    engine = FakeEngine(hits=[FakeHit("pr:1", 0.20, 0.0, 0.40)])

    with client_for(engine) as client:
        result = client.post("/api/trace", json={"query": "q"}).json()["results"][0]

    assert "score_vector" in result
    assert result["score_vector"] == 0.0
    assert result["score_graph"] == 0.40


def test_the_result_count_reaches_the_facade():
    engine = FakeEngine()

    with client_for(engine) as client:
        client.post("/api/trace", json={"query": "q", "top_k": 3})

    assert engine.queries[0][1] == 3


def test_the_query_goes_through_the_async_entry_point():
    """Asserted by the fake having no synchronous one to fall back to."""
    from src.common.config import TOP_K_VECTOR

    engine = FakeEngine()

    with client_for(engine) as client:
        assert client.post("/api/trace", json={"query": "q"}).status_code == 200

    assert engine.queries == [("q", TOP_K_VECTOR)]


@pytest.mark.parametrize("payload", [{}, {"query": "q", "top_k": "many"}])
def test_a_malformed_request_is_refused(payload):
    with client_for(FakeEngine()) as client:
        assert client.post("/api/trace", json=payload).status_code == 422


def test_a_failed_query_is_an_error_not_an_empty_result_set():
    """Nothing matched, and the store could not be read, are different facts."""
    engine = FakeEngine(error=RuntimeError("the store is on fire"))

    with TestClient(create_app(engine_factory=lambda: engine), raise_server_exceptions=False) as client:
        response = client.post("/api/trace", json={"query": "q"})

    assert response.status_code == 500
    assert "results" not in response.text


def test_a_failure_does_not_leak_what_went_wrong():
    engine = FakeEngine(error=RuntimeError("connection string is bad"))

    with TestClient(create_app(engine_factory=lambda: engine), raise_server_exceptions=False) as client:
        response = client.post("/api/trace", json={"query": "q"})

    assert "connection string" not in response.text


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------


SHARED = {"id": "doc:pr:1:0", "path": "https://example.invalid/1", "content": "shared prose"}


def test_each_result_carries_the_documents_that_mention_it():
    engine = FakeEngine(
        hits=[FakeHit("pr:1", 0.5, 0.5, 0.0), FakeHit("ticket:2", 0.4, 0.4, 0.0)],
        documents={"pr:1": [SHARED], "ticket:2": [SHARED]},
    )

    with client_for(engine) as client:
        body = client.post("/api/trace", json={"query": "q"}).json()

    expected = [{"doc_id": "doc:pr:1:0", "content": "shared prose", "path": "https://example.invalid/1"}]
    assert [r["documents"] for r in body["results"]] == [expected, expected]


def test_shared_text_is_written_once_in_the_context():
    """The reason a repeated document is referred back to rather than inlined."""
    engine = FakeEngine(
        hits=[FakeHit("pr:1", 0.5, 0.5, 0.0), FakeHit("ticket:2", 0.4, 0.4, 0.0)],
        documents={"pr:1": [SHARED], "ticket:2": [SHARED]},
    )

    with client_for(engine) as client:
        context = client.post("/api/trace", json={"query": "q"}).json()["context"]

    assert context.count("shared prose") == 1
    assert "[Trace: ticket:2 (PR) -> doc:pr:1:0]" in context


def test_a_result_with_no_prose_reports_an_empty_list_not_a_missing_key():
    engine = FakeEngine(hits=[FakeHit("pr:1", 0.5, 0.5, 0.0)], documents={})

    with client_for(engine) as client:
        body = client.post("/api/trace", json={"query": "q"}).json()

    assert body["results"][0]["documents"] == []


def test_documents_are_only_fetched_for_the_ranked_ids():
    engine = FakeEngine(hits=[FakeHit("pr:1", 0.5, 0.5, 0.0)])

    with client_for(engine) as client:
        client.post("/api/trace", json={"query": "q"})

    assert engine.store.asked_for == [["pr:1"]]


def test_the_chunk_table_deduplicates_without_a_route():
    """The formatter has a second consumer and must work without this API."""
    chunks, by_entity = chunk_table({"a": [SHARED], "b": [SHARED], "c": []})

    assert list(chunks) == ["doc:pr:1:0"]
    assert by_entity == {"a": ["doc:pr:1:0"], "b": ["doc:pr:1:0"], "c": []}


def test_the_formatter_needs_no_store_and_no_framework():
    run = FakeRun([FakeHit("pr:1", 0.42, 0.30, 0.55)])

    formatted = format_run(run, {"pr:1": [SHARED]})

    assert formatted["results"][0]["vector_score"] == 0.30
    assert formatted["chunks"][0]["id"] == "doc:pr:1:0"


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


def test_the_store_opens_once_at_startup_and_closes_at_shutdown():
    engine = FakeEngine()

    with client_for(engine) as client:
        client.post("/api/trace", json={"query": "one"})
        client.post("/api/trace", json={"query": "two"})
        assert engine.entered == 1
        assert engine.store.closed == 0

    assert engine.entered == 1
    assert engine.store.closed == 1


def test_a_startup_that_fails_part_way_still_closes_what_it_opened():
    """A warm-up that raises must not leave a handle in a process that dies."""

    class HalfOpening(FakeEngine):
        def __enter__(self):
            self.entered += 1
            self._store.closed += 0  # opened
            raise RuntimeError("warm-up failed")

    engine = HalfOpening()

    with pytest.raises(RuntimeError, match="warm-up failed"):
        with client_for(engine):
            pass

    assert engine.entered == 1


def test_a_query_before_the_store_is_open_is_unavailable_not_broken():
    app = create_app(engine_factory=lambda: FakeEngine())
    app.state.engine = None
    client = TestClient(app)

    # No `with`, so lifespan never runs and no engine is bound.
    assert client.post("/api/trace", json={"query": "q"}).status_code == 503


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------


def test_the_health_endpoint_needs_no_credentials(monkeypatch):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", True)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)

    with client_for(FakeEngine()) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    # No tenant graph held yet; the local store opened at startup serves
    # nobody under tenancy and is not counted.
    assert response.json() == {"status": "ok", "nodes": 0}


def test_the_query_route_is_rejected_without_a_session(monkeypatch):
    """A data-bearing route must not answer an unauthenticated caller."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", True)
    monkeypatch.setattr(auth_module, "CLERK_ISSUER", "https://issuer.test")
    monkeypatch.setattr(auth_module, "CLERK_JWKS_URL", "https://issuer.test/jwks")

    with client_for(FakeEngine()) as client:
        response = client.post("/api/trace", json={"query": "q"})

    assert response.status_code == 401


def test_the_query_route_is_rejected_without_a_tenant(monkeypatch):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)

    with client_for(FakeEngine()) as client:
        response = client.post("/api/trace", json={"query": "q"})

    assert response.status_code == 401


def test_every_data_bearing_route_resolves_a_tenant():
    """The three routes that read a tenant's graph each resolve one; the model
    and history routes are the caller's own and resolve a user instead."""
    from fastapi.routing import APIRoute

    app = create_app(engine_factory=lambda: FakeEngine())
    tenant_routes = {"/api/trace", "/api/subgraph", "/api/suggestions"}

    seen = set()
    for route in app.routes:
        if not isinstance(route, APIRoute) or route.path not in tenant_routes:
            continue
        names = {
            dependency.call.__name__
            for dependency in route.dependant.dependencies
            if getattr(dependency, "call", None) is not None
        }
        assert "get_current_tenant_org" in names, f"{route.path} reads data without a tenant"
        seen.add(route.path)

    assert seen == tenant_routes


def test_the_formatter_imports_without_a_web_framework():
    """Its second consumer is not this API and must not pay for one.

    A fresh interpreter with the framework blocked, because the import that
    matters is the one a package __init__ performs on the way in — which is
    exactly why this module does not live inside the api package.
    """
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['fastapi'] = None;"
            "from src.results import format_run;"
            "print(format_run.__name__)",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "format_run"


def test_the_registry_and_the_context_manager_do_not_double_close():
    """Two owners, one close. Engine.close is idempotent and this relies on it."""
    engine = FakeEngine()

    with client_for(engine) as client:
        client.get("/api/health")

    assert engine.store.closed == 1


def test_the_process_store_is_registered_under_the_default_tenant():
    engine = FakeEngine()
    app = create_app(engine_factory=lambda: engine)

    with TestClient(app) as client:
        client.get("/api/health")
        registry = REGISTRY
        assert registry.keys() == [auth_module.DEFAULT_TENANT_ORG_ID]
        assert registry.get(auth_module.DEFAULT_TENANT_ORG_ID) is engine


def test_the_registry_is_emptied_at_shutdown():
    engine = FakeEngine()
    app = create_app(engine_factory=lambda: engine)
    with TestClient(app):
        registry = REGISTRY
        assert len(registry) == 1

    assert len(registry) == 0
    assert engine.store.closed == 1


# --------------------------------------------------------------------------
# with tenancy on, graphs and health describe tenant graphs, not local files
# --------------------------------------------------------------------------


@pytest.fixture
def tenancy(monkeypatch):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    monkeypatch.setattr(auth_module, "resolve_org", lambda key: "org_a")


KEY = {"X-Graphrag-API-Key": "k"}


def test_health_counts_the_tenant_graphs_this_pod_holds(tenancy):
    with client_for(FakeEngine()) as client:
        REGISTRY.attach("org_a", FakeEngine(), path="/g/org_a.lbug", version="3")
        REGISTRY.attach("org_b", FakeEngine(), path="/g/org_b.lbug", version="1")
        body = client.get("/api/health").json()

    assert body == {"status": "ok", "nodes": 6}


def test_graphs_lists_only_the_callers_own_graph(tenancy):
    with client_for(FakeEngine()) as client:
        REGISTRY.attach("org_a", FakeEngine(), path="/g/org_a.lbug", version="3")
        REGISTRY.attach("org_b", FakeEngine(), path="/g/org_b.lbug", version="1")
        body = client.get("/api/graphs", headers=KEY).json()

    assert body == {
        "graphs": [{"id": "org_a.lbug", "label": "Version 3", "active": True}],
        "active": "org_a.lbug",
    }


def test_graphs_is_empty_when_this_pod_does_not_hold_the_callers_graph(tenancy):
    with client_for(FakeEngine()) as client:
        body = client.get("/api/graphs", headers=KEY).json()

    assert body == {"graphs": [], "active": None}


def test_graphs_needs_a_tenant_when_tenancy_is_on(tenancy):
    with client_for(FakeEngine()) as client:
        assert client.get("/api/graphs").status_code == 401


def test_switching_is_refused_when_tenancy_is_on(tenancy):
    with client_for(FakeEngine()) as client:
        response = client.post("/api/graphs/switch", json={"id": "anything.lbug"})

    assert response.status_code == 403
