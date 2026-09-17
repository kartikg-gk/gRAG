"""Query history over HTTP: a caller's sessions and what was asked in them.

Scoped to the user, never to a tenant. Who is asking comes from the session
dependency and from nowhere else, so no body field or query parameter can name
another user.

**One refusal for two situations.** A session that does not exist and a
session that belongs to somebody else both answer 404 with the same detail the
query route uses, so these routes cannot be used to ask whether a given
identifier exists.

An unconfigured history database is not caught here: it is a deployment fault,
and it surfaces as a server error rather than as an empty history.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status

from .. import history
from .auth import get_current_user
from .models import SessionRequest, SessionSummaryResponse, TraceRecordResponse

router = APIRouter()

#: The detail for a session this caller cannot act on, whichever reason.
SESSION_NOT_AVAILABLE = "no such session"


def _not_available() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=SESSION_NOT_AVAILABLE,
    )


@router.post("/sessions", response_model=SessionSummaryResponse)
async def create_or_rename_session(
    payload: SessionRequest,
    user_id: str = Depends(get_current_user),
) -> SessionSummaryResponse:
    """Start a session, or retitle one this caller owns."""
    try:
        summary = await asyncio.to_thread(
            history.create_or_rename_session,
            user_id,
            payload.title,
            session_id=payload.session_id,
            email=payload.email,
        )
    except history.SessionNotAvailable:
        raise _not_available() from None
    return SessionSummaryResponse(**asdict(summary))


@router.get("/sessions", response_model=list[SessionSummaryResponse])
async def list_sessions(
    user_id: str = Depends(get_current_user),
) -> list[SessionSummaryResponse]:
    """This caller's sessions, newest first."""
    summaries = await asyncio.to_thread(history.list_sessions, user_id)
    return [SessionSummaryResponse(**asdict(summary)) for summary in summaries]


@router.get("/sessions/{session_id}/traces", response_model=list[TraceRecordResponse])
async def list_traces(
    session_id: str,
    user_id: str = Depends(get_current_user),
) -> list[TraceRecordResponse]:
    """Everything recorded in one of this caller's sessions, oldest first."""
    try:
        records = await asyncio.to_thread(history.list_traces, user_id, session_id)
    except history.SessionNotAvailable:
        raise _not_available() from None
    return [TraceRecordResponse(**asdict(record)) for record in records]
