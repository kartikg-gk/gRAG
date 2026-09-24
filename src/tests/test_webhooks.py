"""Accepting GitHub webhooks and starting a rebuild.

The route runs for real over a real control plane in a temporary SQLite file.
Bodies are signed the way GitHub signs them, over the exact bytes sent. The
only stand-in is the arming call, replaced where the route uses it, since the
real one needs a queue server. Background work runs before ``post`` returns,
so every arming call is visible to the assertions that follow it.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from src.api import auth as auth_module
from src.control_plane import open_control_plane
from src.models import Organization, Repository, control_plane_sessions

SECRET = "webhook-secret-for-tests"
ROUTE = "/api/webhooks/github"
REPO = "acme/widgets"


@pytest.fixture
def webhooks(monkeypatch):
    """The route's module, with arming recorded instead of queued."""
    from src.api import webhooks as module

    calls: list[str] = []
    monkeypatch.setattr(module, "arm_organization", lambda org_id: calls.append(org_id))
    monkeypatch.setattr(module, "GITHUB_WEBHOOK_SECRET", SECRET)
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
def client(webhooks, plane):
    from src.api.app import create_app

    return TestClient(create_app(), raise_server_exceptions=False)


def track(plane, *pairs: tuple[str, str]) -> None:
    """Record that each ``(org_id, repository name)`` pair is tracked."""
    with control_plane_sessions(plane.engine)() as session:
        for org_id in sorted({org for org, _ in pairs}):
            session.add(
                Organization(
                    org_id=org_id, name=org_id, plan="free", status="active",
                    created_at=0, updated_at=0,
                )
            )
        session.commit()
    with control_plane_sessions(plane.engine)() as session:
        for index, (org_id, name) in enumerate(pairs):
            session.add(
                Repository(
                    # The provider's id is unique per tenant; the name is what a
                    # delivery carries, and two rows can share it.
                    repo_id=f"repo_{index}", org_id=org_id, provider="github",
                    provider_repo_id=f"id-{index}", name=name, status="active",
                    created_at=0,
                )
            )
        session.commit()


def compact(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def signature(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def deliver(client, event: str, payload: dict, *, signed: str | None = "auto"):
    """Send ``payload`` as GitHub would, signed over the bytes actually sent."""
    body = compact(payload)
    headers = {"X-GitHub-Event": event, "Content-Type": "application/json"}
    if signed == "auto":
        headers["X-Hub-Signature-256"] = signature(body)
    elif signed is not None:
        headers["X-Hub-Signature-256"] = signed
    return client.post(ROUTE, content=body, headers=headers)


def merged_close(name: str = REPO) -> dict:
    return {
        "action": "closed",
        "pull_request": {"merged": True},
        "repository": {"full_name": name},
    }


# ==========================================================================
# The signature
# ==========================================================================


@pytest.mark.parametrize("unset", [None, ""])
def test_an_unconfigured_secret_refuses_every_delivery(
    monkeypatch, client, webhooks, unset
):
    monkeypatch.setattr(webhooks, "GITHUB_WEBHOOK_SECRET", unset)

    assert deliver(client, "ping", {"zen": "hi"}).status_code == 503


def test_a_missing_signature_is_rejected(client):
    assert deliver(client, "ping", {"zen": "hi"}, signed=None).status_code == 401


def test_a_signature_without_the_sha256_prefix_is_rejected(client):
    digest = signature(compact({"zen": "hi"})).removeprefix("sha256=")

    response = deliver(client, "ping", {"zen": "hi"}, signed="sha1=" + digest)

    assert response.status_code == 401


def test_a_wrong_digest_is_rejected(client):
    response = deliver(client, "ping", {"zen": "hi"}, signed="sha256=" + "0" * 64)

    assert response.status_code == 401


def test_a_body_changed_after_signing_is_rejected(client):
    """Same data, different bytes. Only a check over the raw body catches it."""
    payload = merged_close()
    signed_over = compact(payload)
    sent = json.dumps(payload, indent=2).encode("utf-8")
    assert signed_over != sent and json.loads(signed_over) == json.loads(sent)

    response = client.post(
        ROUTE,
        content=sent,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": signature(signed_over),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 401


def test_the_signature_is_checked_before_the_event_is_read(client, webhooks):
    response = deliver(client, "push", {"repository": {"full_name": REPO}},
                       signed="sha256=" + "0" * 64)

    assert response.status_code == 401
    assert webhooks.arming_calls == []


def test_the_signature_is_the_only_gate(monkeypatch, client):
    """Session verification and tenancy on, and no other credential sent."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", True)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)

    assert deliver(client, "ping", {"zen": "hi"}).status_code == 200


# ==========================================================================
# Events that start nothing
# ==========================================================================


def test_a_ping_is_answered(client):
    response = deliver(client, "ping", {"zen": "hi"})

    assert response.status_code == 200
    assert response.json() == {"status": "pong"}


def test_an_event_that_is_not_a_merge_or_push_is_ignored(client, webhooks):
    response = deliver(client, "issues", {"action": "opened"})

    assert response.json() == {"status": "ignored", "event": "issues"}
    assert webhooks.arming_calls == []


def test_a_pull_request_that_was_not_closed_is_ignored(client, plane, webhooks):
    track(plane, ("org_a", REPO))
    payload = {"action": "opened", "repository": {"full_name": REPO}}

    response = deliver(client, "pull_request", payload)

    assert response.json() == {"status": "ignored", "action": "opened"}
    assert webhooks.arming_calls == []


def test_a_pull_request_closed_without_merging_is_ignored(client, plane, webhooks):
    track(plane, ("org_a", REPO))
    payload = {
        "action": "closed",
        "pull_request": {"merged": False},
        "repository": {"full_name": REPO},
    }

    response = deliver(client, "pull_request", payload)

    assert response.json() == {"status": "ignored", "reason": "pr not merged"}
    assert webhooks.arming_calls == []


def test_a_payload_with_no_repository_is_ignored(client, webhooks):
    response = deliver(client, "push", {"ref": "refs/heads/main"})

    assert response.json() == {"status": "ignored", "reason": "no repository in payload"}
    assert webhooks.arming_calls == []


# ==========================================================================
# Events that arm a rebuild
# ==========================================================================


def test_a_merged_pull_request_arms_its_tenant_once(client, plane, webhooks):
    """Tracked twice by the same tenant, and still armed once."""
    track(plane, ("org_a", REPO), ("org_a", REPO))

    response = deliver(client, "pull_request", merged_close())

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "repository": REPO, "orgs": 1}
    assert webhooks.arming_calls == ["org_a"]


def test_a_push_arms_its_tenant(client, plane, webhooks):
    track(plane, ("org_a", REPO))

    response = deliver(client, "push", {"repository": {"full_name": REPO}})

    assert response.json() == {"status": "accepted", "repository": REPO, "orgs": 1}
    assert webhooks.arming_calls == ["org_a"]


def test_a_repository_tracked_by_two_tenants_arms_both(client, plane, webhooks):
    track(plane, ("org_a", REPO), ("org_b", REPO), ("org_c", "other/repo"))

    response = deliver(client, "pull_request", merged_close())

    assert response.json() == {"status": "accepted", "repository": REPO, "orgs": 2}
    assert sorted(webhooks.arming_calls) == ["org_a", "org_b"]


def test_an_arming_failure_is_still_accepted(monkeypatch, client, plane, webhooks):
    """GitHub has nothing to retry: the delivery itself was fine."""
    track(plane, ("org_a", REPO), ("org_b", REPO))
    armed: list[str] = []

    def flaky(org_id):
        if org_id == "org_a":
            raise RuntimeError("queue server unreachable")
        armed.append(org_id)

    monkeypatch.setattr(webhooks, "arm_organization", flaky)

    response = deliver(client, "push", {"repository": {"full_name": REPO}})

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "repository": REPO, "orgs": 2}
    assert armed == ["org_b"]


# ==========================================================================
# Bodies that are signed but malformed
# ==========================================================================


@pytest.mark.parametrize("body", [b"not json", b"[1, 2, 3]", b'"a string"'])
def test_a_signed_body_that_is_not_a_json_object_is_a_server_error(client, webhooks, body):
    """A signed delivery is always a JSON object. Anything else is a fault, not a no-op."""
    response = client.post(
        ROUTE,
        content=body,
        headers={
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": signature(body),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 500
    assert webhooks.arming_calls == []


def test_an_empty_signed_body_is_read_as_an_empty_object(client, webhooks):
    response = client.post(
        ROUTE,
        content=b"",
        headers={
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": signature(b""),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ignored", "reason": "no repository in payload"}
    assert webhooks.arming_calls == []
