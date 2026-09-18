"""GitHub OAuth connection flow for repository credentials."""

from __future__ import annotations

import hashlib
import hmac
import os
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlmodel import select

from ..models.control_plane import Repository
from ..models.database import control_plane_sessions
from . import auth as auth_module

router = APIRouter(prefix="/api/github", tags=["github"])

AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
DEFAULT_REDIRECT_URI = "http://localhost:8000/api/github/callback"
DEFAULT_SCOPE = "repo read:org"


def _setting(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _oauth_configuration() -> tuple[str, str, str, str]:
    client_id = _setting("GRAPHRAG_GITHUB_CLIENT_ID")
    client_secret = _setting("GRAPHRAG_GITHUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth is not configured.",
        )
    redirect_uri = _setting(
        "GRAPHRAG_GITHUB_OAUTH_REDIRECT_URI", DEFAULT_REDIRECT_URI
    )
    scope = _setting("GRAPHRAG_GITHUB_OAUTH_SCOPE", DEFAULT_SCOPE)
    return client_id, client_secret, redirect_uri, scope


def _state_secret() -> str:
    secret = _setting("GRAPHRAG_GITHUB_OAUTH_STATE_SECRET") or _setting(
        "GRAPHRAG_ADMIN_SECRET_KEY"
    )
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OAuth state secret is not configured.",
        )
    return secret


def _signature(secret: str, org_id: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), org_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _state(org_id: str) -> str:
    return f"{org_id}:{_signature(_state_secret(), org_id)}"


def _validated_org(state_value: str) -> str:
    org_id, _, presented = (state_value or "").partition(":")
    if not org_id or not presented:
        raise HTTPException(status_code=400, detail="Malformed state.")
    expected = _signature(_state_secret(), org_id)
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")
    return org_id


@router.get("/login")
def connect(org_id: str) -> RedirectResponse:
    client_id, _, redirect_uri, scope = _oauth_configuration()
    query = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": _state(org_id),
            "allow_signup": "false",
        }
    )
    return RedirectResponse(f"{AUTHORIZE_URL}?{query}", status_code=302)


@router.get("/callback")
def callback(code: str, state: str) -> dict:
    client_id, client_secret, redirect_uri, _ = _oauth_configuration()
    org_id = _validated_org(state)
    try:
        response = httpx.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="GitHub token exchange failed.",
        ) from exc
    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="GitHub token exchange failed.",
        )

    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No access_token from GitHub ({payload.get('error') or 'unknown'}).",
        )

    sessions = control_plane_sessions(auth_module.control_plane().engine)
    with sessions() as db:
        repositories = db.exec(
            select(Repository).where(Repository.org_id == org_id)
        ).all()
        for repository in repositories:
            repository.set_github_token(access_token)
        db.commit()

    return {
        "status": "connected",
        "org_id": org_id,
        "repositories_updated": len(repositories),
    }
