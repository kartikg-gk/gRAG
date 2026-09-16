"""GitHub webhook deliveries, turned into rebuilds.

A merged pull request or a push changes what a tracked repository holds, so
every tenant tracking it is armed for a rebuild. Everything else GitHub sends
is acknowledged and ignored.

The signature is the authentication
-----------------------------------

GitHub signs each delivery with a shared secret, over the exact bytes it sent.
The check runs on the raw body before anything is parsed, because parsing and
re-serialising changes the bytes, and a check over the re-serialised form
would verify something GitHub never signed. It is also the only gate: a
delivery carries no user session and no tenant key.

With no secret configured the route refuses everything. Accepting unverified
deliveries would let anyone who can reach the route trigger builds.

Arming after the response
-------------------------

GitHub expects a quick answer and retries deliveries that time out. Arming
talks to the queue server, so it runs after the response has been sent, and a
failure there is logged rather than returned: the delivery itself was fine,
and a retry from GitHub would not reach the queue server either.

Several tenants can track the same public repository, so one delivery can
arm several. A tenant tracking the repository more than once is armed once.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from sqlmodel import select

from ..common.config import GITHUB_WEBHOOK_SECRET
from ..models import Repository, control_plane_sessions
from ..worker.tasks import arm_organization
from . import auth as auth_module

logger = logging.getLogger("graphrag.api.webhooks")

router = APIRouter()

SIGNATURE_HEADER = "X-Hub-Signature-256"
EVENT_HEADER = "X-GitHub-Event"
SIGNATURE_PREFIX = "sha256="

#: The events that change what a repository holds.
REBUILD_EVENTS = ("pull_request", "push")

DISABLED_DETAIL = (
    "webhooks are disabled: GRAPHRAG_GITHUB_WEBHOOK_SECRET is not configured"
)


def _verify(secret: str, body: bytes, presented: str | None) -> None:
    """Raise 401 unless ``presented`` is GitHub's signature over ``body``."""
    if not presented or not presented.startswith(SIGNATURE_PREFIX):
        logger.warning("webhook refused: missing or malformed signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=auth_module.INVALID_CREDENTIALS,
        )

    expected = SIGNATURE_PREFIX + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()

    # Constant-time, so how many leading characters of a forgery matched
    # cannot be learned from how long the refusal took.
    if not hmac.compare_digest(presented.encode(), expected.encode()):
        logger.warning("webhook refused: signature does not match the body")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=auth_module.INVALID_CREDENTIALS,
        )


def _organisations_tracking(full_name: str) -> list[str]:
    """Every distinct tenant with a repository of this name, in a stable order."""
    engine = auth_module.control_plane().engine
    with control_plane_sessions(engine)() as session:
        rows = session.exec(
            select(Repository.org_id).where(Repository.name == full_name).distinct()
        ).all()
    return sorted(rows)


def _arm(org_id: str) -> None:
    """Arm one tenant, logging a failure instead of raising it.

    Runs after the response, so there is nobody left to raise to, and one
    tenant failing must not stop the others from being armed.
    """
    try:
        arm_organization(org_id)
    except Exception:  # noqa: BLE001 - logged; the next delivery arms again
        logger.warning("%s: a webhook could not arm a rebuild", org_id, exc_info=True)


@router.post("/webhooks/github")
async def github_webhook(request: Request, background: BackgroundTasks) -> dict:
    """Verify a delivery, then arm every tenant tracking its repository."""
    secret = GITHUB_WEBHOOK_SECRET
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=DISABLED_DETAIL
        )

    body = await request.body()
    _verify(secret, body, request.headers.get(SIGNATURE_HEADER))

    event = request.headers.get(EVENT_HEADER)
    if event == "ping":
        return {"status": "pong"}
    if event not in REBUILD_EVENTS:
        return {"status": "ignored", "event": event}

    # Parsed only after the signature matched, so this is GitHub's own output,
    # and GitHub always sends a JSON object. A signed empty body is read as an
    # empty object; any other body that is not JSON is left to fail as a fault.
    payload = json.loads(body or b"{}")

    if event == "pull_request":
        action = payload.get("action")
        if action != "closed":
            return {"status": "ignored", "action": action}
        pull_request = payload.get("pull_request") or {}
        if pull_request.get("merged") is not True:
            return {"status": "ignored", "reason": "pr not merged"}

    repository = payload.get("repository") or {}
    full_name = repository.get("full_name")
    if not full_name:
        return {"status": "ignored", "reason": "no repository in payload"}

    # A database read, off the event loop.
    org_ids = await asyncio.to_thread(_organisations_tracking, full_name)
    for org_id in org_ids:
        background.add_task(_arm, org_id)

    return {"status": "accepted", "repository": full_name, "orgs": len(org_ids)}
