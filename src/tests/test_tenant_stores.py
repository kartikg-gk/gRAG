"""Tests for a tenant identifier selecting an already-loaded graph.

The registry receives graphs; it does not find them. Nothing here derives a
path from an identifier and nothing opens a store because a request arrived —
so a tenant is servable exactly when something has attached its handle or
pointed its key at a path, and every other tenant is service-unavailable.

The isolation test is the one this exists for, and it runs against **real
stores**: a fake engine would prove the routing and prove nothing about
isolation, because a fake returns whatever it was told to whichever caller
asks. Both tenants are attached explicitly, which is the whole shape of the
current design.

It needs no model. The embedder is a stub with fixed vectors, shared through
the loader exactly as the real one would be.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import auth as auth_module
from src.api.app import create_app
from src.common.config import DEFAULT_TENANT_ORG_ID, EMBEDDING_DIMENSION, POD_ID
from src.registry import REGISTRY, GraphRegistry


def _store_available() -> bool:
    """Whether a store can actually be opened here, not merely imported."""
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
    """Fixed vectors, so a match is exact and no model is loaded."""

    def __init__(self):
        self.calls = 0

    def vector(self, text):
        self.calls += 1
        values = [0.0] * EMBEDDING_DIMENSION
        values[abs(hash(text)) % EMBEDDING_DIMENSION] = 1.0
        return values


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeStore:
    def __init__(self):
        self.closed = 0

    def documents_for_entities(self, ids):
        return {node_id: [] for node_id in ids}

    def get_entity(self, entity_id):
        return {"id": entity_id, "label": entity_id, "type": "PR"}

    def close(self):
        self.closed += 1


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


class FakeEngine:
    def __init__(self, node_id="default:node", path="fake.db"):
        self.node_id = node_id
        self.path = path
        self._store = FakeStore()
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
        from src.tests.contract_fixtures import trace_log

        return RouterResponse(query, [node], trace_log())

    async def retrieve_async(self, query, k=None, **kwargs):
        """Mirrors the facade: the count is positional and may be unset."""
        return FakeRun(self.node_id)


def provision(plane, *org_ids):
    """Create the tenants, so credentials can name them.

    A credential refers to its organisation, so the row comes first. These
    tests are about which store a resolved tenant reaches, not about
    provisioning, so this is setup rather than subject.
    """
    from src.models import Organization, control_plane_sessions

    with control_plane_sessions(plane.engine)() as session:
        for org_id in org_ids:
            session.add(
                Organization(
                    org_id=org_id,
                    name=org_id,
                    plan="team",
                    status="active",
                    created_at=0,
                    updated_at=0,
                )
            )
        session.commit()
    return plane


def issue_key(tmp_path, org_id):
    """A real credential for ``org_id``, and the control plane holding it."""
    from src.control_plane import open_control_plane

    plane = provision(open_control_plane(tmp_path / "control.db"), org_id)
    raw, _record = plane.issue(org_id)
    return raw, plane


def serve_here(plane, *org_ids):
    """Mark each tenant ready on this process's pod.

    Tenant routes are answered only by the pod a tenant is assigned to. These
    tests are about which store a tenant reaches once it gets there, so every
    tenant is assigned here and ready, and that is setup rather than subject.
    """
    import time

    from src.models import LOAD_READY, POD_READY, Pod, PodAssignment, control_plane_sessions

    now = int(time.time())
    with control_plane_sessions(plane.engine)() as session:
        session.add(
            Pod(
                pod_id=POD_ID,
                address="127.0.0.1",
                status=POD_READY,
                last_heartbeat_at=now,
                created_at=now,
            )
        )
        session.flush()
        for org_id in org_ids:
            session.add(
                PodAssignment(
                    pod_id=POD_ID, org_id=org_id, load_status=LOAD_READY, assigned_at=now
                )
            )
        session.commit()
    return plane


# ==========================================================================
# the test this exists for
# ==========================================================================


@requires_store
def test_a_request_for_one_tenant_never_sees_another_tenants_data(tmp_path):
    """Two tenants, two stores, different contents, no leakage either way.

    Both graphs are **attached explicitly**. Nothing discovers them, which is
    the point: the registry is handed graphs and serves what it was handed.
    """
    from src.control_plane import open_control_plane
    from src.engine import Engine
    from src.graphdb import open_context_graph

    embedder = AxisEmbedder()
    contents = {"org_a": ("a:only", "alpha private"), "org_b": ("b:only", "beta private")}
    for org_id, (node_id, label) in contents.items():
        store = open_context_graph(tmp_path / org_id)
        try:
            store.upsert_entity(node_id, label, "PR", embedding=embedder.vector(label))
            store.build_vector_index(rebuild=True)
        finally:
            store.close()

    plane = serve_here(provision(open_control_plane(tmp_path / "control.db"), *contents), *contents)
    keys = {org_id: plane.issue(org_id)[0] for org_id in contents}

    default_store = tmp_path / "default"
    open_context_graph(default_store).close()
    app = create_app(engine_factory=lambda: Engine(default_store, embedder=embedder))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(auth_module, "CLERK_ENABLED", False)
        patch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
        auth_module.set_control_plane(plane)
        try:
            with TestClient(app) as client:
                # The caller supplies each path. Nothing is derived from a key.
                for org_id in contents:
                    REGISTRY.replace(org_id, path=str(tmp_path / org_id))

                bodies = {}
                for org_id, key in keys.items():
                    response = client.post(
                        "/api/trace",
                        json={"query": "private"},
                        headers={"Authorization": f"Bearer {key}"},
                    )
                    assert response.status_code == 200, response.text
                    bodies[org_id] = response.json()
        finally:
            auth_module.set_control_plane(None)
            plane.close()

    a_ids = {result["id"] for result in bodies["org_a"]["results"]}
    b_ids = {result["id"] for result in bodies["org_b"]["results"]}

    assert "b:only" not in a_ids
    assert "a:only" not in b_ids
    assert a_ids <= {"a:only"}
    assert b_ids <= {"b:only"}


@requires_store
def test_two_tenants_are_two_open_handles_not_one(tmp_path):
    """Isolation is two stores, not one store filtered two ways."""
    from src.api.app import graph_loader
    from src.engine import Engine
    from src.graphdb import open_context_graph

    for org_id in ("org_a", "org_b"):
        open_context_graph(tmp_path / org_id).close()
    default_store = tmp_path / "default"
    open_context_graph(default_store).close()

    registry = GraphRegistry()
    embedder = AxisEmbedder()
    first = Engine(default_store, embedder=embedder).__enter__()
    registry.set_loader(graph_loader(first))

    registry.replace("org_a", path=str(tmp_path / "org_a"))
    registry.replace("org_b", path=str(tmp_path / "org_b"))
    a, b = registry.get("org_a"), registry.get("org_b")

    assert a is not b
    assert a.store is not b.store
    assert sorted(registry.keys()) == ["org_a", "org_b"]
    # Model sharing is unchanged, and is why the second one is cheap.
    assert a.embedder is b.embedder is embedder

    registry.close_all()
    first.close()


# ==========================================================================
# a graph this process does not hold
# ==========================================================================


def test_an_unattached_tenant_is_unavailable_not_missing(tmp_path, monkeypatch):
    """503, not 404. The tenant may exist; this process has not been given it."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    raw, plane = issue_key(tmp_path, "org_elsewhere")
    serve_here(plane, "org_elsewhere")
    auth_module.set_control_plane(plane)

    try:
        with TestClient(create_app(engine_factory=lambda: FakeEngine())) as client:
            response = client.post(
                "/api/trace", json={"query": "q"}, headers={"Authorization": f"Bearer {raw}"}
            )
    finally:
        auth_module.set_control_plane(None)
        plane.close()

    assert response.status_code == 503
    assert "results" not in response.json()


def test_an_unattached_tenant_opens_nothing_and_touches_no_disk(tmp_path, monkeypatch):
    """A miss must not be a lazy open in disguise."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    raw, plane = issue_key(tmp_path, "org_elsewhere")
    serve_here(plane, "org_elsewhere")
    auth_module.set_control_plane(plane)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    opened = []
    app = create_app(engine_factory=lambda: FakeEngine())

    try:
        with TestClient(app) as client:
            REGISTRY.set_loader(lambda path: opened.append(path))
            before = sorted(p.name for p in workspace.iterdir())
            response = client.post(
                "/api/trace", json={"query": "q"}, headers={"Authorization": f"Bearer {raw}"}
            )
            after = sorted(p.name for p in workspace.iterdir())
    finally:
        auth_module.set_control_plane(None)
        plane.close()

    assert response.status_code == 503
    assert opened == [], "a request caused a store to be opened"
    assert before == after == []


# ==========================================================================
# the two ways a graph gets in
# ==========================================================================


def test_an_attached_handle_serves_that_tenant(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    raw, plane = issue_key(tmp_path, "org_attached")
    serve_here(plane, "org_attached")
    auth_module.set_control_plane(plane)
    app = create_app(engine_factory=lambda: FakeEngine("default:node"))

    try:
        with TestClient(app) as client:
            REGISTRY.attach(
                "org_attached", FakeEngine("attached:node"), path="given.db"
            )
            body = client.post(
                "/api/trace", json={"query": "q"}, headers={"Authorization": f"Bearer {raw}"}
            ).json()
    finally:
        auth_module.set_control_plane(None)
        plane.close()

    assert body["results"][0]["id"] == "attached:node"


def test_pointing_a_key_at_a_supplied_path_opens_through_the_loader(tmp_path, monkeypatch):
    """The caller supplies the path. Nothing is inferred from the key."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    raw, plane = issue_key(tmp_path, "org_pointed")
    serve_here(plane, "org_pointed")
    auth_module.set_control_plane(plane)
    app = create_app(engine_factory=lambda: FakeEngine("default:node"))
    asked = []

    try:
        with TestClient(app) as client:
            def loader(path):
                asked.append(path)
                return FakeEngine("pointed:node", path=path)

            REGISTRY.set_loader(loader)
            REGISTRY.replace("org_pointed", path="/somewhere/given")
            body = client.post(
                "/api/trace", json={"query": "q"}, headers={"Authorization": f"Bearer {raw}"}
            ).json()
    finally:
        auth_module.set_control_plane(None)
        plane.close()

    assert asked == ["/somewhere/given"]
    assert body["results"][0]["id"] == "pointed:node"


# ==========================================================================
# single-tenant operation
# ==========================================================================


def test_single_tenant_mode_works_with_nothing_configured(monkeypatch):
    """No credentials, no tenant configuration, no per-tenant anything."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)
    engine = FakeEngine("default:node")

    with TestClient(create_app(engine_factory=lambda: engine)) as client:
        response = client.post("/api/trace", json={"query": "q"})

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "default:node"


def test_the_startup_store_is_attached_under_the_default_identifier(monkeypatch):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)
    engine = FakeEngine()
    app = create_app(engine_factory=lambda: engine)

    with TestClient(app) as client:
        client.get("/health")
        assert REGISTRY.keys() == [DEFAULT_TENANT_ORG_ID]
        assert REGISTRY.get(DEFAULT_TENANT_ORG_ID) is engine


def test_with_tenancy_disabled_any_identifier_reaches_the_startup_store(monkeypatch):
    """The single-store fallback, which is what keeps a local run first-class."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)
    engine = FakeEngine("default:node")
    app = create_app(engine_factory=lambda: engine)

    from src.api.app import engine_for

    with TestClient(app):
        request = type("R", (), {"app": app})()
        assert engine_for(request, "some-other-identifier") is engine


# ==========================================================================
# the route asks for what it authenticated as
# ==========================================================================


def test_the_route_serves_the_tenant_it_authenticated_as(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    raw, plane = issue_key(tmp_path, "org_alpha")
    serve_here(plane, "org_alpha")
    auth_module.set_control_plane(plane)
    app = create_app(engine_factory=lambda: FakeEngine("default:node"))

    try:
        with TestClient(app) as client:
            REGISTRY.attach(
                "org_alpha", FakeEngine("alpha:node"), path="alpha.db"
            )
            REGISTRY.attach(
                "org_beta", FakeEngine("beta:node"), path="beta.db"
            )
            body = client.post(
                "/api/trace", json={"query": "q"}, headers={"Authorization": f"Bearer {raw}"}
            ).json()
    finally:
        auth_module.set_control_plane(None)
        plane.close()

    assert body["results"][0]["id"] == "alpha:node"


# ==========================================================================
# shutdown
# ==========================================================================


def test_shutdown_closes_every_attached_handle(monkeypatch):
    """Not just the one startup opened."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)

    startup = FakeEngine("default:node")
    attached = FakeEngine("later:node", path="later.db")
    app = create_app(engine_factory=lambda: startup)

    with TestClient(app) as client:
        client.get("/health")
        REGISTRY.attach("org_later", attached, path="later.db")

    assert startup.store.closed == 1
    assert attached.store.closed == 1


def test_a_request_before_the_registry_exists_is_unavailable(monkeypatch):
    """Credentials are checked first, so this needs them out of the way."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)
    app = create_app(engine_factory=lambda: FakeEngine())
    REGISTRY = None

    assert TestClient(app).post("/api/trace", json={"query": "q"}).status_code == 503
