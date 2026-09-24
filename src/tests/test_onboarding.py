"""Setting up a whole tenant in one admin call.

Every test drives the real route over a real control plane in a temporary
SQLite file, with foreign keys enforced the way the engine always enforces
them. The only stand-in is the arming call, which would otherwise need a
queue server, and it is replaced where the route uses it.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from src.api import auth as auth_module
from src.control_plane import hash_api_key, open_control_plane
from src.models import (
    LOAD_PULLING,
    POD_READY,
    ApiKey,
    Organization,
    Pod,
    PodAssignment,
    Repository,
    control_plane_sessions,
)

SECRET = "admin-secret-for-tests"
ROUTE = "/api/admin/onboarding/provision"
BODY = {"tenant_name": "Acme", "repo_name": "acme/widgets"}
WARNING = "save this api_key now: it is shown once and cannot be recovered"
SEEDED_POD = "pod_seeded"


@pytest.fixture
def onboarding(monkeypatch):
    """The route's module, with arming recorded instead of queued."""
    from src.api import onboarding as module

    calls: list[str] = []
    monkeypatch.setattr(module, "arm_organization", lambda org_id: calls.append(org_id))
    monkeypatch.setattr(module, "ADMIN_SECRET_KEY", SECRET)
    module.arming_calls = calls
    return module


@pytest.fixture
def plane(tmp_path):
    store = open_control_plane(tmp_path / "control.db")
    auth_module.set_control_plane(store)
    yield store
    auth_module.set_control_plane(None)
    store.close()


@pytest.fixture
def client(onboarding, plane):
    from src.api.app import create_app

    return TestClient(create_app(), raise_server_exceptions=False)


def seed_pod(plane, pod_id: str = SEEDED_POD) -> None:
    now = int(time.time())
    with control_plane_sessions(plane.engine)() as session:
        session.add(
            Pod(
                pod_id=pod_id,
                address="127.0.0.1",
                status=POD_READY,
                last_heartbeat_at=now,
                created_at=now,
            )
        )
        session.commit()


def rows(plane, model) -> list:
    with control_plane_sessions(plane.engine)() as session:
        return session.exec(select(model)).all()


def admin(secret: str = SECRET) -> dict:
    return {"X-Admin-Secret": secret}


# ==========================================================================
# The guard
# ==========================================================================


@pytest.mark.parametrize("unset", [None, ""])
def test_an_unconfigured_secret_disables_the_route(monkeypatch, client, onboarding, unset):
    monkeypatch.setattr(onboarding, "ADMIN_SECRET_KEY", unset)

    response = client.post(ROUTE, json=BODY, headers=admin())

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "onboarding is disabled: GRAPHRAG_ADMIN_SECRET_KEY is not configured"
    )


def test_a_missing_admin_header_is_rejected(client):
    assert client.post(ROUTE, json=BODY).status_code == 401


def test_a_wrong_admin_secret_is_rejected(client):
    assert client.post(ROUTE, json=BODY, headers=admin("not-the-secret")).status_code == 401


def test_the_admin_secret_is_the_only_gate(monkeypatch, client, plane):
    """Session verification and tenancy both on, and no other credential sent."""
    seed_pod(plane)
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", True)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)

    assert client.post(ROUTE, json=BODY, headers=admin()).status_code == 200


# ==========================================================================
# Provisioning
# ==========================================================================


def test_provisioning_writes_one_row_in_each_table(client, plane):
    seed_pod(plane)

    response = client.post(ROUTE, json=BODY, headers=admin())

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "org_id", "api_key", "pod_id", "repository_id", "reconcile_armed", "warning",
    }
    assert body["org_id"].startswith("org_") and len(body["org_id"]) == len("org_") + 16
    assert body["pod_id"] == SEEDED_POD
    assert body["reconcile_armed"] is True
    assert body["warning"] == WARNING

    organizations = rows(plane, Organization)
    keys = rows(plane, ApiKey)
    repositories = rows(plane, Repository)
    assignments = rows(plane, PodAssignment)
    assert [len(organizations), len(keys), len(repositories), len(assignments)] == [1, 1, 1, 1]

    organization = organizations[0]
    assert (organization.org_id, organization.name, organization.plan, organization.status) == (
        body["org_id"], "Acme", "free", "active",
    )
    assert isinstance(organization.created_at, int)

    repository = repositories[0]
    assert repository.repo_id == body["repository_id"]
    assert (repository.org_id, repository.provider, repository.provider_repo_id) == (
        body["org_id"], "github", "acme/widgets",
    )
    assert (repository.name, repository.default_branch, repository.status) == (
        "acme/widgets", "main", "active",
    )
    assert repository.last_synced_cursor is None

    assignment = assignments[0]
    assert (assignment.pod_id, assignment.org_id, assignment.load_status) == (
        SEEDED_POD, body["org_id"], LOAD_PULLING,
    )
    assert isinstance(assignment.assigned_at, int)


def test_the_returned_key_resolves_to_the_new_tenant(client, plane):
    seed_pod(plane)

    body = client.post(ROUTE, json=BODY, headers=admin()).json()

    assert auth_module.resolve_org(body["api_key"]) == body["org_id"]


def test_the_stored_key_is_a_digest_not_the_key(client, plane):
    seed_pod(plane)

    body = client.post(ROUTE, json=BODY, headers=admin()).json()

    stored = rows(plane, ApiKey)[0]
    assert stored.hashed_key != body["api_key"]
    assert stored.hashed_key == hash_api_key(body["api_key"])
    assert stored.prefix == body["api_key"][: len(stored.prefix)]
    assert stored.org_id == body["org_id"]


# ==========================================================================
# Arming the first build
# ==========================================================================


def test_the_new_tenant_is_armed_exactly_once(client, plane, onboarding):
    seed_pod(plane)

    body = client.post(ROUTE, json=BODY, headers=admin()).json()

    assert onboarding.arming_calls == [body["org_id"]]


def test_an_arming_failure_still_returns_the_tenant(monkeypatch, client, plane, onboarding):
    """The tenant exists by then; refusing the request would hide the key."""
    seed_pod(plane)

    def broken(org_id):
        raise RuntimeError("queue server unreachable")

    monkeypatch.setattr(onboarding, "arm_organization", broken)

    response = client.post(ROUTE, json=BODY, headers=admin())

    assert response.status_code == 200
    assert response.json()["reconcile_armed"] is False
    assert len(rows(plane, Organization)) == 1


# ==========================================================================
# A pod that does not exist
# ==========================================================================


def test_a_pod_that_has_not_booted_is_registered_as_booting(client, plane, onboarding):
    """No pod has booted, so the chosen id names no pod row yet.

    Onboarding writes it, marked booting rather than ready, in the same
    transaction as the tenant; the pod agent marks it ready when it starts.
    """
    from src.models import Pod

    response = client.post(ROUTE, json=BODY, headers=admin())

    assert response.status_code == 200
    (pod,) = rows(plane, Pod)
    assert pod.status == "booting"
    assert pod.last_heartbeat_at is None
    (assignment,) = rows(plane, PodAssignment)
    assert assignment.pod_id == pod.pod_id


def test_a_failed_provision_rolls_back_every_row(client, plane, onboarding, monkeypatch, caplog):
    """Nothing the transaction wrote survives a failure, including the pod."""
    import src.api.onboarding as onboarding_module
    from src.models import Pod

    def refuse(**_):
        raise RuntimeError("assignment refused")

    monkeypatch.setattr(onboarding_module, "PodAssignment", refuse)

    with caplog.at_level("ERROR"):
        response = client.post(ROUTE, json=BODY, headers=admin())

    assert response.status_code == 500
    assert "assignment refused" not in response.text
    for model in (Organization, ApiKey, Repository, PodAssignment, Pod):
        assert rows(plane, model) == [], model.__name__
    assert onboarding.arming_calls == []
