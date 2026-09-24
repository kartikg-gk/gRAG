"""Tests for the subgraph, suggestions, list and switch routes.

The engine is faked. These routes resolve a tenant, hand one call to the
store, and shape what comes back — a real database makes none of that truer
and makes all of it need a native library.

**The first test is the re-bind.** Switching repoints the router, and if it
does not also re-bind the default tenant's registry entry, tenant resolution
keeps handing out the previous store while the switch route reports success.
Silent, and wrong in the direction that looks fine.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from src.api import auth as auth_module
from src.api.app import create_app
from src.registry import REGISTRY
from src.graphs import graph_label, graph_paths
from src.api.app import suggestions_from
from src.tests.contract_fixtures import trace_log


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeHit:
    def __init__(self, node_id):
        self.id = node_id
        self.score = 0.5
        self.vector_score = 0.5
        self.graph_score = 0.0
        self.decay = 1.0
        self.node_type = "PR"
        self.age_days = None
        self.found_by_both = False


class FakeRun:
    def __init__(self, node_id):
        self.query = "q"
        self.intent = type("I", (), {"intent": "conceptual"})()
        self.fused = type("F", (), {"alpha": 0.8, "beta": 0.2})()
        self.hits = [FakeHit(node_id)]
        self.ids = [node_id]
        self.trace_log = None
        self.seconds = 0.0


class FakeStore:
    def __init__(self, node_id="default:node", *, hubs=None, nodes=1, delay=0.0):
        self.node_id = node_id
        self.hubs = hubs if hubs is not None else []
        self.nodes = nodes
        self.delay = delay
        self.closed = 0
        self.subgraph_calls = []
        self.top_entities_calls = []

    def documents_for_entities(self, ids):
        return {node_id: [] for node_id in ids}

    def get_entity(self, entity_id):
        return {"id": entity_id, "label": entity_id, "type": "PR"}

    def subgraph(self, ids):
        self.subgraph_calls.append(list(ids))
        if self.delay:
            time.sleep(self.delay)
        known = [node for node in ids if node.startswith("known")]
        if not known:
            return {"nodes": [], "edges": []}
        return {
            "nodes": [
                {"id": node, "label": node, "type": "PR", "requested": True}
                for node in known
            ]
            + [
                {
                    "id": "neighbour:1",
                    "label": "neighbour:1",
                    "type": "Person",
                    "requested": False,
                }
            ],
            "edges": [
                {
                    "source": known[0],
                    "target": "neighbour:1",
                    "confidence": 0.9,
                    "relation": "AUTHORED",
                }
            ],
        }

    def top_entities(self, limit=12):
        self.top_entities_calls.append(limit)
        if isinstance(self.hubs, Exception):
            raise self.hubs
        return self.hubs[:limit]

    def count_nodes(self):
        return self.nodes

    def close(self):
        self.closed += 1


class FakeEngine:
    def __init__(self, node_id="default:node", path="default.lbug", **store_kwargs):
        self.node_id = node_id
        self.path = path
        self._store = FakeStore(node_id, **store_kwargs)
        self.embedder = object()
        self.extractor = None
        self.judge = None
        self.now = None
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
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
        pass

    def build_context(self, results):
        return ""

    async def route_async(self, query, k=None, **kwargs):
        from src.retrieval.response import RouterResponse, RoutedNode
        node = RoutedNode(self.node_id, self.node_id, "PR", .5, .5, 0.0)
        return RouterResponse(query, [node], trace_log())

    async def retrieve_async(self, query, k=None, **kwargs):
        return FakeRun(self.node_id)


@pytest.fixture(autouse=True)
def single_tenant(monkeypatch):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)


def client_for(engine):
    return TestClient(create_app(engine_factory=lambda: engine))


def discovered(monkeypatch, tmp_path, *names):
    """Put ``names`` on disk and make discovery return exactly them."""
    paths = []
    for name in names:
        path = tmp_path / f"{name}.lbug"
        path.write_bytes(b"")
        paths.append(path)
    monkeypatch.setattr("src.api.app.graph_paths", lambda: paths)
    return paths


# ==========================================================================
# the re-bind — write this one first
# ==========================================================================


def test_after_a_switch_the_tenant_path_serves_the_new_graph(monkeypatch, tmp_path):
    """The registry must follow the switch, not keep serving the old file.

    Asserted through ``/api/trace``, which resolves its store through the tenant
    path, rather than through the switch response — the response would say
    success while resolution still handed out the previous store.
    """
    discovered(monkeypatch, tmp_path, "old", "new")
    startup = FakeEngine("old:node", path=str(tmp_path / "old.lbug"))
    replacement = FakeEngine("new:node", path=str(tmp_path / "new.lbug"))
    app = create_app(engine_factory=lambda: startup)

    with TestClient(app) as client:
        REGISTRY.set_loader(lambda path: replacement)

        before = client.post("/api/trace", json={"query": "q"}).json()
        assert before["results"][0]["id"] == "old:node"

        switched = client.post("/api/graphs/switch", json={"id": "new.lbug"})
        assert switched.status_code == 200, switched.text

        after = client.post("/api/trace", json={"query": "q"}).json()

    assert after["results"][0]["id"] == "new:node", (
        "the query still resolved the previous store: the default tenant's "
        "registry entry was not re-bound"
    )


def test_a_switch_closes_the_displaced_store(monkeypatch, tmp_path):
    discovered(monkeypatch, tmp_path, "old", "new")
    startup = FakeEngine("old:node", path=str(tmp_path / "old.lbug"))
    replacement = FakeEngine("new:node", path=str(tmp_path / "new.lbug"))
    app = create_app(engine_factory=lambda: startup)

    with TestClient(app) as client:
        REGISTRY.set_loader(lambda path: replacement)
        client.post("/api/graphs/switch", json={"id": "new.lbug"})

        assert startup.store.closed == 1, "the displaced store stayed open"
        assert replacement.store.closed == 0, "the new store was closed"


def test_switching_to_the_active_graph_does_not_close_what_it_installed(
    monkeypatch, tmp_path
):
    """The one case where displaced and installed can be the same object."""
    discovered(monkeypatch, tmp_path, "only")
    engine = FakeEngine("only:node", path=str(tmp_path / "only.lbug"))
    app = create_app(engine_factory=lambda: engine)

    with TestClient(app) as client:
        REGISTRY.set_loader(lambda path: engine)
        response = client.post("/api/graphs/switch", json={"id": "only.lbug"})

        assert response.status_code == 200
        assert engine.store.closed == 0
        assert client.post("/api/trace", json={"query": "q"}).status_code == 200


def test_a_switch_reports_the_new_graph(monkeypatch, tmp_path):
    discovered(monkeypatch, tmp_path, "old", "acme__checkout")
    startup = FakeEngine("old:node", path=str(tmp_path / "old.lbug"))
    replacement = FakeEngine("new:node", path=str(tmp_path / "acme__checkout.lbug"), nodes=42)
    app = create_app(engine_factory=lambda: startup)

    with TestClient(app) as client:
        REGISTRY.set_loader(lambda path: replacement)
        body = client.post("/api/graphs/switch", json={"id": "acme__checkout.lbug"}).json()

    assert body == {"active": "acme__checkout.lbug", "label": "acme/checkout", "nodes": 42}


def test_switching_to_an_id_that_is_not_a_discovered_file_is_refused(
    monkeypatch, tmp_path
):
    """An id names something discovery found. It is not a path to open."""
    discovered(monkeypatch, tmp_path, "known")
    app = create_app(engine_factory=lambda: FakeEngine())

    with TestClient(app) as client:
        for identifier in ("missing.lbug", "known", "../escape", "/etc/passwd"):
            response = client.post("/api/graphs/switch", json={"id": identifier})
            assert response.status_code == 404, identifier


def test_a_switch_does_not_touch_another_tenants_entry(monkeypatch, tmp_path):
    """These routes are the local workflow, not multi-tenancy."""
    discovered(monkeypatch, tmp_path, "old", "new")
    startup = FakeEngine("old:node", path=str(tmp_path / "old.lbug"))
    other = FakeEngine("other:node", path="other.lbug")
    app = create_app(engine_factory=lambda: startup)

    with TestClient(app) as client:
        REGISTRY.attach("org_other", other, path="other.lbug")
        REGISTRY.set_loader(
            lambda path: FakeEngine("new:node", path=str(tmp_path / "new.lbug"))
        )
        client.post("/api/graphs/switch", json={"id": "new.lbug"})

        assert REGISTRY.get("org_other") is other
        assert other.store.closed == 0


# ==========================================================================
# listing
# ==========================================================================


def test_listing_marks_exactly_one_graph_active(monkeypatch, tmp_path):
    paths = discovered(monkeypatch, tmp_path, "alpha", "beta", "gamma")
    engine = FakeEngine("n", path=str(paths[1]))
    app = create_app(engine_factory=lambda: engine)

    with TestClient(app) as client:
        body = client.get("/api/graphs").json()

    active = [graph for graph in body["graphs"] if graph["active"]]
    assert len(active) == 1
    assert active[0]["id"] == "beta.lbug"
    assert body["active"] == "beta.lbug"


def test_listing_follows_a_switch(monkeypatch, tmp_path):
    paths = discovered(monkeypatch, tmp_path, "old", "new")
    startup = FakeEngine("old:node", path=str(paths[0]))
    app = create_app(engine_factory=lambda: startup)

    with TestClient(app) as client:
        REGISTRY.set_loader(
            lambda path: FakeEngine("new:node", path=str(paths[1]))
        )
        client.post("/api/graphs/switch", json={"id": "new.lbug"})
        body = client.get("/api/graphs").json()

    assert body["active"] == "new.lbug"


def test_listing_needs_no_credentials(monkeypatch, tmp_path):
    """Deliberately open. Both of these are a local developer affordance.

    With tenancy on they describe tenant graphs instead; see test_api_query.
    """
    discovered(monkeypatch, tmp_path, "alpha")
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", True)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)

    with client_for(FakeEngine()) as client:
        assert client.get("/api/graphs").status_code == 200


def test_switching_needs_no_credentials(monkeypatch, tmp_path):
    discovered(monkeypatch, tmp_path, "only")
    engine = FakeEngine("only:node", path=str(tmp_path / "only.lbug"))
    app = create_app(engine_factory=lambda: engine)
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", True)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)

    with TestClient(app) as client:
        REGISTRY.set_loader(lambda path: engine)
        assert client.post("/api/graphs/switch", json={"id": "only.lbug"}).status_code == 200


# ==========================================================================
# subgraph
# ==========================================================================


def test_subgraph_returns_neighbours_and_edges_for_known_ids():
    with client_for(FakeEngine()) as client:
        body = client.post("/api/subgraph", json={"node_ids": ["known:1"]}).json()

    assert {node["id"] for node in body["nodes"]} == {"known:1", "neighbour:1"}
    assert body["edges"][0]["relation"] == "AUTHORED"


def test_subgraph_returns_an_empty_result_for_an_unknown_id():
    """Not an error: a caller expanding a stale set should not be rejected."""
    with client_for(FakeEngine()) as client:
        response = client.post("/api/subgraph", json={"node_ids": ["nothing:1"]})

    assert response.status_code == 200
    assert response.json() == {"nodes": [], "edges": []}


def test_subgraph_with_no_ids_is_an_empty_result():
    with client_for(FakeEngine()) as client:
        assert client.post("/api/subgraph", json={"node_ids": []}).json() == {
            "nodes": [],
            "edges": [],
        }


def test_subgraph_does_not_block_the_event_loop():
    """A blocking read on the loop serialises every other request behind it."""
    engine = FakeEngine(delay=0.25)
    app = create_app(engine_factory=lambda: engine)

    with TestClient(app) as client:
        done = []

        def call():
            client.post("/api/subgraph", json={"node_ids": ["known:1"]})
            done.append(time.perf_counter())

        start = time.perf_counter()
        threads = [threading.Thread(target=call) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

    elapsed = max(done) - start
    assert len(done) == 3
    # Serialised would be at least 0.75s; overlapped is close to 0.25s.
    assert elapsed < 0.6, f"requests serialised: {elapsed:.2f}s for three 0.25s reads"


def test_subgraph_requires_a_tenant(monkeypatch):
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)

    with client_for(FakeEngine()) as client:
        assert client.post("/api/subgraph", json={"node_ids": []}).status_code == 401


# ==========================================================================
# suggestions
# ==========================================================================


def hubs(*pairs):
    return [
        {"id": f"{kind.lower()}:{n}", "label": f"{kind} {n}", "type": kind, "degree": 50 - n}
        for n, kind in enumerate(pairs)
    ]


def test_suggestions_returns_at_most_the_limit():
    engine = FakeEngine(hubs=hubs("Person", "PR", "Ticket", "Commit", "File", "Repo"))

    with client_for(engine) as client:
        body = client.get("/api/suggestions?limit=3").json()

    assert len(body["suggestions"]) == 3


def test_suggestions_over_fetches_rather_than_asking_for_the_limit():
    """Asking for exactly the limit returns the top few, which cluster."""
    engine = FakeEngine(hubs=hubs("Person", "PR"))

    with client_for(engine) as client:
        client.get("/api/suggestions?limit=4")

    assert engine.store.top_entities_calls == [16]


def test_no_two_suggestions_share_a_type_while_other_types_remain():
    """Five questions about five people is one question asked five times."""
    engine = FakeEngine(
        hubs=hubs("Person", "Person", "Person", "PR", "Ticket", "Commit")
    )

    with client_for(engine) as client:
        body = client.get("/api/suggestions?limit=4").json()

    types = [item["type"] for item in body["suggestions"]]
    assert len(set(types)) == len(types), types


def test_a_small_graph_returns_fewer_rather_than_padding():
    engine = FakeEngine(hubs=hubs("Person"))

    with client_for(engine) as client:
        body = client.get("/api/suggestions?limit=5").json()

    assert len(body["suggestions"]) == 1


def test_a_graph_that_cannot_answer_returns_an_empty_list():
    """Too small or too new to suggest anything is ordinary, not a failure."""
    engine = FakeEngine(hubs=RuntimeError("no index yet"))

    with client_for(engine) as client:
        response = client.get("/api/suggestions")

    assert response.status_code == 200
    assert response.json() == {"suggestions": []}


def test_a_suggestion_carries_the_entity_it_was_built_from():
    engine = FakeEngine(hubs=hubs("Person"))

    with client_for(engine) as client:
        item = client.get("/api/suggestions?limit=1").json()["suggestions"][0]

    assert item == {"query": "Show recent contributions by Person 0.", "entity": "Person 0", "type": "Person"}


def test_suggestions_requires_a_tenant(monkeypatch):
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)

    with client_for(FakeEngine()) as client:
        assert client.get("/api/suggestions").status_code == 401


# --- the diversification rule, without a route -----------------------------


def test_backfill_takes_by_degree_once_the_types_run_out():
    entities = hubs("Person", "Person", "Person")

    chosen = suggestions_from(entities, 3)

    assert [item["entity"] for item in chosen] == ["Person 0", "Person 1", "Person 2"]


def test_a_type_with_no_template_still_gets_a_question():
    chosen = suggestions_from([{"id": "x:1", "label": "thing", "type": "Unheard"}], 1)

    assert chosen[0]["query"] == "Explore connections around thing."


def test_a_limit_of_zero_asks_the_store_for_nothing():
    engine = FakeEngine(hubs=hubs("Person"))

    with client_for(engine) as client:
        body = client.get("/api/suggestions?limit=0").json()

    assert engine.store.top_entities_calls == [0]
    assert body == {"suggestions": []}


def test_entities_without_a_label_are_skipped():
    assert suggestions_from([{"id": "p:1", "label": "  ", "type": "Person"}], 3) == []


# ==========================================================================
# health
# ==========================================================================


def test_health_works_with_no_credentials(monkeypatch):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", True)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)

    with client_for(FakeEngine()) as client:
        body = client.get("/api/health").json()

    assert body == {"status": "ok", "nodes": 1}


# ==========================================================================
# discovery and labels
# ==========================================================================


def test_the_default_store_is_listed_first(tmp_path):
    root = tmp_path / "graphs"
    root.mkdir()
    for name in ("aaa", "zzz"):
        (root / f"{name}.lbug").write_bytes(b"")
    default = tmp_path / "configured.lbug"
    default.write_bytes(b"")

    found = graph_paths(directory=root, default=default)

    assert found[0] == default
    assert [path.stem for path in found] == ["configured", "aaa", "zzz"]


def test_the_default_is_listed_once_when_it_lives_in_the_directory(tmp_path):
    """Deduplicated by resolved path, and it keeps its place at the front."""
    root = tmp_path / "graphs"
    root.mkdir()
    default = root / "shared.lbug"
    default.write_bytes(b"")
    (root / "other.lbug").write_bytes(b"")

    found = graph_paths(directory=root, default=root / ".." / "graphs" / "shared.lbug")

    assert [path.stem for path in found] == ["shared", "other"]


def test_a_missing_default_is_simply_not_listed(tmp_path):
    root = tmp_path / "graphs"
    root.mkdir()
    (root / "only.lbug").write_bytes(b"")

    found = graph_paths(directory=root, default=tmp_path / "absent.lbug")

    assert [path.stem for path in found] == ["only"]


def test_a_missing_directory_is_not_an_error(tmp_path):
    default = tmp_path / "configured.lbug"
    default.write_bytes(b"")

    assert graph_paths(directory=tmp_path / "nothing", default=default) == [default]


def test_paths_come_back_as_found_not_resolved(tmp_path):
    root = tmp_path / "graphs"
    root.mkdir()
    (root / "one.lbug").write_bytes(b"")
    default = tmp_path / "configured.lbug"
    default.write_bytes(b"")

    found = graph_paths(directory=root, default=default)

    assert found[0] == default


def test_the_default_is_labelled_as_the_default(tmp_path):
    default = tmp_path / "graph.lbug"
    default.write_bytes(b"")

    assert graph_label(default, default=default) == "graph (default)"


def test_a_doubled_underscore_reads_back_as_a_slash(tmp_path):
    path = tmp_path / "acme__checkout.lbug"

    assert graph_label(path, default=tmp_path / "other.lbug") == "acme/checkout"


def test_an_ordinary_name_is_left_alone(tmp_path):
    path = tmp_path / "notes.lbug"

    assert graph_label(path, default=tmp_path / "other.lbug") == "notes"
