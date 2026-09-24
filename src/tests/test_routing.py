"""Answering a tenant's request only from the pod assigned to it.

The lookup runs over a real control plane in a temporary SQLite file. The
middleware is exercised two ways: on a small application holding nothing but
the middleware and plain handlers, so that what reaches a handler is exactly
what the middleware let through, and once through ``create_app`` to prove it
is installed there.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api import auth as auth_module
from src.common.config import POD_ID
from src.control_plane import open_control_plane
from src.models import (
    LOAD_PULLING,
    LOAD_READY,
    POD_READY,
    Organization,
    Pod,
    PodAssignment,
    control_plane_sessions,
)

ORG = "org_routed"
HERE = "pod_here"
OTHER = "pod_other"
THIRD = "pod_third"

PULLING_DETAIL = "this tenant's graph is still loading on this pod; retry shortly"
ELSEWHERE_DETAIL = "this tenant is served by a different pod"
UNASSIGNED_DETAIL = "no pod is serving this tenant yet"


@pytest.fixture
def routing():
    from src.api import routing as module

    return module


@pytest.fixture
def plane(tmp_path):
    store = open_control_plane(tmp_path / "control.db")
    auth_module.set_control_plane(store)
    yield store
    auth_module.set_control_plane(None)
    store.close()


@pytest.fixture
def tenancy_on(monkeypatch):
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)


def seed(plane, *assignments: tuple[str, str]) -> None:
    """The tenant, the pods named, and one assignment per ``(pod_id, status)``."""
    now = int(time.time())
    sessions = control_plane_sessions(plane.engine)
    with sessions() as session:
        session.add(
            Organization(
                org_id=ORG, name=ORG, plan="free", status="active",
                created_at=now, updated_at=now,
            )
        )
        for pod_id in {pod for pod, _ in assignments}:
            session.add(
                Pod(
                    pod_id=pod_id, address="127.0.0.1", status=POD_READY,
                    last_heartbeat_at=now, created_at=now,
                )
            )
        session.commit()
    with sessions() as session:
        for pod_id, load_status in assignments:
            session.add(
                PodAssignment(
                    pod_id=pod_id, org_id=ORG, load_status=load_status,
                    assigned_at=now,
                )
            )
        session.commit()


class Unreachable:
    """A control plane that fails on any use, to prove nothing consulted it."""

    @property
    def engine(self):
        raise AssertionError("the control plane was consulted")

    def record_for_hash(self, hashed_key):
        raise AssertionError("the control plane was consulted")


def gated_client(routing) -> TestClient:
    """The middleware in front of handlers that report what reached them."""
    app = FastAPI()
    app.add_middleware(routing.TenantRoutingMiddleware)

    @app.api_route("/api/trace", methods=["POST", "OPTIONS"])
    async def query(request: Request):
        return {"answered": True, "org_id": getattr(request.state, "org_id", None)}

    @app.get("/api/health")
    async def health():
        return {"answered": True}

    @app.get("/api/graphs")
    async def graphs():
        return {"answered": True}

    return TestClient(app, raise_server_exceptions=False)


# ==========================================================================
# assignment_status
# ==========================================================================


def test_a_ready_assignment_on_this_pod_is_ready_here(routing, plane):
    seed(plane, (HERE, LOAD_READY))

    verdict = routing.assignment_status(ORG, pod_id=HERE, engine=plane.engine)

    assert verdict == ("ready_here", {"pod_id": HERE})


def test_an_unready_assignment_on_this_pod_is_pulling_here(routing, plane):
    seed(plane, (HERE, LOAD_PULLING))

    verdict = routing.assignment_status(ORG, pod_id=HERE, engine=plane.engine)

    assert verdict == ("pulling_here", {"pod_id": HERE, "load_status": LOAD_PULLING})


def test_a_ready_pod_elsewhere_is_chosen_over_a_pulling_one(routing, plane):
    seed(plane, (OTHER, LOAD_PULLING), (THIRD, LOAD_READY))

    verdict = routing.assignment_status(ORG, pod_id=HERE, engine=plane.engine)

    assert verdict == ("ready_elsewhere", {"pod_id": THIRD, "load_status": LOAD_READY})


def test_a_pod_still_pulling_elsewhere_counts_as_elsewhere(routing, plane):
    seed(plane, (OTHER, LOAD_PULLING))

    verdict = routing.assignment_status(ORG, pod_id=HERE, engine=plane.engine)

    assert verdict == ("ready_elsewhere", {"pod_id": OTHER, "load_status": LOAD_PULLING})


def test_a_tenant_with_no_assignment_is_unassigned(routing, plane):
    seed(plane)

    verdict = routing.assignment_status(ORG, pod_id=HERE, engine=plane.engine)

    assert verdict == ("unassigned", {})


def test_the_pod_and_engine_default_to_this_process_and_its_control_plane(
    routing, plane
):
    seed(plane, (POD_ID, LOAD_READY))

    assert routing.assignment_status(ORG) == ("ready_here", {"pod_id": POD_ID})


# ==========================================================================
# the middleware: what it leaves alone
# ==========================================================================


def test_nothing_is_gated_when_tenancy_is_off(monkeypatch, routing, plane):
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)
    auth_module.set_control_plane(Unreachable())

    response = gated_client(routing).post("/api/trace")

    assert response.status_code == 200
    assert response.json()["answered"] is True


def test_a_preflight_request_is_not_gated(routing, plane, tenancy_on):
    auth_module.set_control_plane(Unreachable())

    response = gated_client(routing).options("/api/trace")

    assert response.status_code == 200
    assert response.json()["answered"] is True


@pytest.mark.parametrize("path", ["/api/health", "/api/graphs"])
def test_a_route_that_serves_no_tenant_is_not_gated(routing, plane, tenancy_on, path):
    auth_module.set_control_plane(Unreachable())

    response = gated_client(routing).get(path)

    assert response.status_code == 200
    assert response.json() == {"answered": True}


# ==========================================================================
# the middleware: the verdicts
# ==========================================================================


def test_a_missing_key_is_rejected(routing, plane, tenancy_on):
    response = gated_client(routing).post("/api/trace")

    assert response.status_code == 401
    assert response.json() == {"detail": auth_module.INVALID_CREDENTIALS}
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_a_key_that_does_not_resolve_is_rejected(routing, plane, tenancy_on):
    seed(plane, (POD_ID, LOAD_READY))

    response = gated_client(routing).post("/api/trace", headers={"Authorization": "Bearer not-a-key"})

    assert response.status_code == 401
    assert response.json() == {"detail": auth_module.INVALID_CREDENTIALS}
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_a_tenant_still_loading_here_is_told_to_retry(routing, plane, tenancy_on):
    seed(plane, (POD_ID, LOAD_PULLING))
    raw, _ = plane.issue(ORG)

    response = gated_client(routing).post("/api/trace", headers={"Authorization": f"Bearer {raw}"})

    assert response.status_code == 503
    assert response.json() == {"detail": PULLING_DETAIL, "org_id": ORG}
    assert response.headers["Retry-After"] == "5"


def test_a_tenant_served_elsewhere_is_pointed_at_its_pod(routing, plane, tenancy_on):
    seed(plane, (OTHER, LOAD_READY))
    raw, _ = plane.issue(ORG)

    response = gated_client(routing).post("/api/trace", headers={"Authorization": f"Bearer {raw}"})

    assert response.status_code == 421
    assert response.json() == {
        "detail": ELSEWHERE_DETAIL, "org_id": ORG, "assigned_pod": OTHER,
    }
    assert response.headers["X-Graphrag-Assigned-Pod"] == OTHER


def test_a_tenant_no_pod_serves_is_a_conflict(routing, plane, tenancy_on):
    seed(plane)
    raw, _ = plane.issue(ORG)

    response = gated_client(routing).post("/api/trace", headers={"Authorization": f"Bearer {raw}"})

    assert response.status_code == 409
    assert response.json() == {"detail": UNASSIGNED_DETAIL, "org_id": ORG}


def test_a_failed_lookup_is_unavailable(monkeypatch, routing, plane, tenancy_on):
    seed(plane)
    raw, _ = plane.issue(ORG)

    def broken(org_id, **kwargs):
        raise RuntimeError("control plane unreachable")

    monkeypatch.setattr(routing, "assignment_status", broken)

    response = gated_client(routing).post("/api/trace", headers={"Authorization": f"Bearer {raw}"})

    assert response.status_code == 503
    assert response.json() == {"detail": "routing unavailable"}


def test_a_tenant_ready_here_reaches_the_route(routing, plane, tenancy_on):
    seed(plane, (POD_ID, LOAD_READY))
    raw, _ = plane.issue(ORG)

    response = gated_client(routing).post("/api/trace", headers={"Authorization": f"Bearer {raw}"})

    assert response.status_code == 200
    assert response.json() == {"answered": True, "org_id": ORG}


# ==========================================================================
# installed in the application
# ==========================================================================


def test_the_application_gates_tenant_routes(plane, tenancy_on):
    from src.api.app import create_app

    seed(plane)
    raw, _ = plane.issue(ORG)
    client = TestClient(create_app(), raise_server_exceptions=False)

    response = client.post("/api/trace", json={"query": "q"}, headers={"Authorization": f"Bearer {raw}"})

    assert response.status_code == 409
    assert response.json() == {"detail": UNASSIGNED_DETAIL, "org_id": ORG}
