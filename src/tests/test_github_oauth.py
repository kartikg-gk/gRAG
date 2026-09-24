from __future__ import annotations

import hashlib
import hmac
from urllib.parse import parse_qs, urlparse

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import select

from src.control_plane import open_control_plane
from src.models.control_plane import Organization, Repository
from src.models.database import control_plane_sessions


@pytest.fixture()
def oauth(monkeypatch, tmp_path):
    from src.api import auth as auth_module
    from src.api.github_oauth import router

    monkeypatch.setenv("GRAPHRAG_GITHUB_CLIENT_ID", "client-id")
    monkeypatch.setenv("GRAPHRAG_GITHUB_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("GRAPHRAG_GITHUB_OAUTH_STATE_SECRET", "state-secret")
    monkeypatch.setenv(
        "GRAPHRAG_ENCRYPTION_MASTER_KEY", Fernet.generate_key().decode()
    )
    plane = open_control_plane(tmp_path / "oauth.db")
    auth_module.set_control_plane(plane)
    app = FastAPI()
    app.include_router(router)
    try:
        yield TestClient(app), plane
    finally:
        auth_module.set_control_plane(None)
        plane.close()


def test_oauth_routes_are_registered(oauth):
    client, _ = oauth
    paths = set(client.app.openapi()["paths"])
    assert {"/api/github/login", "/api/github/callback"} <= paths


def test_connect_requires_github_configuration(oauth, monkeypatch):
    client, _ = oauth
    monkeypatch.delenv("GRAPHRAG_GITHUB_CLIENT_ID", raising=False)
    response = client.get("/api/github/login", params={"org_id": "org_1"})
    assert response.status_code == 503
    assert response.json()["detail"] == "GitHub OAuth is not configured."


def test_connect_requires_a_state_secret(oauth, monkeypatch):
    client, _ = oauth
    monkeypatch.delenv("GRAPHRAG_GITHUB_OAUTH_STATE_SECRET", raising=False)
    monkeypatch.delenv("GRAPHRAG_ADMIN_SECRET_KEY", raising=False)
    response = client.get("/api/github/login", params={"org_id": "org_1"})
    assert response.status_code == 503
    assert response.json()["detail"] == "OAuth state secret is not configured."


def test_connect_redirect_has_exact_parameters(oauth):
    client, _ = oauth
    response = client.get(
        "/api/github/login", params={"org_id": "org_1"}, follow_redirects=False
    )
    assert response.status_code == 302
    parsed = urlparse(response.headers["location"])
    assert parsed.scheme == "https"
    assert parsed.netloc == "github.com"
    assert parsed.path == "/login/oauth/authorize"
    expected_signature = hmac.new(
        b"state-secret", b"org_1", hashlib.sha256
    ).hexdigest()
    assert parse_qs(parsed.query) == {
        "client_id": ["client-id"],
        "redirect_uri": ["http://localhost:8000/api/github/callback"],
        "scope": ["repo read:org"],
        "state": [f"org_1:{expected_signature}"],
        "allow_signup": ["false"],
    }


@pytest.mark.parametrize(
    ("state", "detail"),
    [("broken", "Malformed state."), ("org_1:not-valid", "Invalid OAuth state.")],
)
def test_callback_rejects_bad_state(oauth, state, detail):
    client, _ = oauth
    response = client.get(
        "/api/github/callback", params={"code": "code", "state": state}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == detail


class ExchangeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _state(org_id="org_1"):
    signature = hmac.new(
        b"state-secret", org_id.encode(), hashlib.sha256
    ).hexdigest()
    return f"{org_id}:{signature}"


def test_callback_exchange_request_and_non_200(oauth, monkeypatch):
    from src.api import github_oauth

    client, _ = oauth
    seen = {}

    def post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return ExchangeResponse(500)

    monkeypatch.setattr(github_oauth.httpx, "post", post)
    response = client.get(
        "/api/github/callback", params={"code": "oauth-code", "state": _state()}
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "GitHub token exchange failed."
    assert seen == {
        "url": "https://github.com/login/oauth/access_token",
        "data": {
            "client_id": "client-id",
            "client_secret": "client-secret",
            "code": "oauth-code",
            "redirect_uri": "http://localhost:8000/api/github/callback",
        },
        "headers": {"Accept": "application/json"},
        "timeout": 15,
    }


def test_callback_reports_missing_access_token(oauth, monkeypatch):
    from src.api import github_oauth

    client, _ = oauth
    monkeypatch.setattr(
        github_oauth.httpx,
        "post",
        lambda *args, **kwargs: ExchangeResponse(200, {"error": "bad_verification_code"}),
    )
    response = client.get(
        "/api/github/callback", params={"code": "bad", "state": _state()}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "No access_token from GitHub (bad_verification_code)."
    )


def test_callback_encrypts_token_for_every_repository_without_disclosing_it(
    oauth, monkeypatch
):
    from src.api import github_oauth

    client, plane = oauth
    with control_plane_sessions(plane.engine)() as db:
        db.add(
            Organization(
                org_id="org_1",
                name="Acme",
                plan="team",
                status="active",
                created_at=1,
                updated_at=1,
            )
        )
        db.flush()
        for index in range(2):
            db.add(
                Repository(
                    repo_id=f"repo_{index}",
                    org_id="org_1",
                    provider="github",
                    provider_repo_id=f"acme/repo-{index}",
                    name=f"acme/repo-{index}",
                    status="active",
                    created_at=1,
                )
            )
        db.commit()

    token = "github-plain-token"
    monkeypatch.setattr(
        github_oauth.httpx,
        "post",
        lambda *args, **kwargs: ExchangeResponse(200, {"access_token": token}),
    )
    response = client.get(
        "/api/github/callback", params={"code": "good", "state": _state()}
    )
    assert response.status_code == 200
    assert response.json() == {
        "status": "connected",
        "org_id": "org_1",
        "repositories_updated": 2,
    }
    assert token not in response.text

    with control_plane_sessions(plane.engine)() as db:
        rows = db.exec(select(Repository)).all()
        assert len(rows) == 2
        assert all(row.github_token != token for row in rows)
        assert all(token not in row.github_token for row in rows)
        assert all(row.get_github_token() == token for row in rows)
