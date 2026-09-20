"""Sessions and traces, and who is allowed to see them.

The operations around the history tables. Two rules run through all of them.

Ownership is checked on every path, separately
----------------------------------------------

Renaming somebody else's session is refused, and **so is reading one**. The
read check is written out where the read happens rather than being assumed to
follow from the write check: a system that guards the operation that changes
things and not the one that shows them is a system that looks protected from
the outside and is not, and the read is the one that leaks.

A session that belongs to somebody else and a session that does not exist
raise the **same** refusal, carrying nothing that distinguishes them. Telling
the two apart would let anyone enumerate which identifiers are real.

History never breaks a query
----------------------------

The two operations the query path calls — finding who owns a session, and
writing a trace into one — swallow every database failure and return nothing.
Answering a question without recording it is a materially better outcome than
refusing to answer because the recording failed, and the person asking cannot
do anything about a history database being down.

That tolerance stops at the query path. The operations a route calls
directly — creating, listing, reading — raise, because a caller who asked to
see their sessions and got an empty list would believe they had none.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from .models.history import (
    DEFAULT_SESSION_TITLE,
    ChatSession,
    TraceLog,
    User,
    create_history_engine,
    create_history_schema,
    history_sessions,
)

logger = logging.getLogger("graphrag.history")

#: The domain a made-up address is built under when a real one cannot be
#: used. Never deliverable, which is the point: it is a placeholder, not a way
#: to reach anybody.
PLACEHOLDER_DOMAIN = "users.graphrag.local"


class SessionNotAvailable(RuntimeError):
    """This session cannot be acted on by this caller.

    **One error for two situations**: no such session, and somebody else's
    session. Distinguishing them in the response would turn this into a way
    to ask whether a given identifier exists.
    """


@dataclass(frozen=True)
class SessionRecord:
    """One session, as its owner sees it."""

    id: str
    user_id: str
    title: str
    created_at: datetime


@dataclass(frozen=True)
class TraceRecord:
    """One recorded question and what answering it involved."""

    id: str
    session_id: str
    query: str
    execution_plan: dict[str, Any]
    graph_payload: Any
    created_at: datetime
    graph_id: str | None = None


# --------------------------------------------------------------------------
# the people
# --------------------------------------------------------------------------


def ensure_user(db: Session, user_id: str, email: str | None = None) -> User:
    """The row for ``user_id``, created on first sight.

    **An address that belongs to a different identifier does not fail the
    request.** Two identifiers claiming one address is a question about
    identity that this table cannot answer, and answering it wrongly in
    either direction is worse than not answering: refusing would block a
    session over an email, and merging would hand one person another's
    history. So the second identifier gets a placeholder address instead,
    which nothing sends mail to and nobody sees. No address at all gets the
    placeholder too.
    """
    user = db.get(User, user_id)
    if user is not None:
        return user

    placeholder = f"{user_id}@{PLACEHOLDER_DOMAIN}"
    candidate = email or placeholder
    clash = db.exec(select(User).where(User.email == candidate)).first()
    if clash is not None and clash.id != user_id:
        logger.warning(
            "%s claims an address already held by %s; using a placeholder",
            user_id,
            clash.id,
        )
        candidate = placeholder

    user = User(id=user_id, email=candidate)
    db.add(user)
    # Surfaces any remaining uniqueness conflict here rather than at commit.
    db.flush()
    return user


# --------------------------------------------------------------------------
# what a route calls
# --------------------------------------------------------------------------


def create_or_rename_session(
    user_id: str,
    title: str = DEFAULT_SESSION_TITLE,
    *,
    session_id: str | None = None,
    email: str | None = None,
    engine: Engine | None = None,
) -> SessionRecord:
    """Start a session, or retitle one this caller already owns."""
    made = engine if engine is not None else create_history_engine()
    sessions = history_sessions(made)

    with sessions() as db:
        if session_id:
            session = _owned(db, session_id, user_id)
            session.title = title
        else:
            # The user row is created here rather than anywhere earlier
            # because this is the first moment anything needs one to exist.
            ensure_user(db, user_id, email)
            session = ChatSession(user_id=user_id, title=title)
            db.add(session)
        db.commit()
        db.refresh(session)
        return _session_record(session)


def list_sessions(user_id: str, *, engine: Engine | None = None) -> list[SessionRecord]:
    """This caller's sessions, most recent first."""
    made = engine if engine is not None else create_history_engine()
    sessions = history_sessions(made)

    with sessions() as db:
        rows = db.exec(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc())
        ).all()

    return [_session_record(row) for row in rows]


def list_traces(
    user_id: str, session_id: str, *, engine: Engine | None = None
) -> list[TraceRecord]:
    """Everything recorded in one session, oldest first.

    **The ownership check here is its own**, not a consequence of the one on
    renaming. This is the path that would hand somebody another person's
    questions, and it is the one worth being explicit about.
    """
    made = engine if engine is not None else create_history_engine()
    sessions = history_sessions(made)

    with sessions() as db:
        _owned(db, session_id, user_id)

        rows = db.exec(
            select(TraceLog)
            .where(TraceLog.session_id == session_id)
            .order_by(TraceLog.created_at.asc())
        ).all()

    return [
        TraceRecord(
            id=row.id,
            session_id=row.session_id,
            query=row.query,
            execution_plan=row.execution_plan,
            graph_payload=_graph_results(row.graph_payload),
            created_at=row.created_at,
            graph_id=_graph_id(row.graph_payload),
        )
        for row in rows
    ]


def _owned(db: Session, session_id: str, user_id: str) -> ChatSession:
    """The session, if this caller owns it. Otherwise the same refusal either
    way — see ``SessionNotAvailable``."""
    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != user_id:
        raise SessionNotAvailable(session_id)
    return session


# --------------------------------------------------------------------------
# what the query path calls
# --------------------------------------------------------------------------


def session_owner(session_id: str, *, engine: Engine | None = None) -> str | None:
    """Who owns ``session_id``, or ``None``.

    ``None`` covers two things that the caller treats identically: no such
    session, and a history database that could not be reached. Nothing here
    raises, because this sits in front of answering a question and a history
    database being down is not the asker's problem.
    """
    try:
        made = engine if engine is not None else create_history_engine()
        with history_sessions(made)() as db:
            session = db.get(ChatSession, session_id)
            return session.user_id if session is not None else None
    except Exception:  # noqa: BLE001 - history must not break a query
        logger.warning("could not read the owner of session %s", session_id, exc_info=True)
        return None


def persist_trace(
    session_id: str,
    query: str,
    execution_plan: dict[str, Any],
    graph_payload: Any,
    *,
    graph_id: str | None = None,
    engine: Engine | None = None,
) -> str | None:
    """Write one trace. Returns its identifier, or ``None`` if it was lost.

    Swallows everything. A question that was answered and not recorded is a
    lost trace; a question refused because it could not be recorded is a lost
    answer, and the second is the worse of the two by some distance.
    """
    try:
        made = engine if engine is not None else create_history_engine()
        with history_sessions(made)() as db:
            stored_payload = graph_payload
            if graph_id:
                stored_payload = {"results": graph_payload, "graph_id": graph_id}
            trace = TraceLog(
                session_id=session_id,
                query=query,
                execution_plan=execution_plan,
                graph_payload=stored_payload,
            )
            db.add(trace)
            db.commit()
            return trace.id
    except Exception:  # noqa: BLE001 - history must not break a query
        logger.warning("could not record a trace for %s", session_id, exc_info=True)
        return None


def _graph_id(payload: Any) -> str | None:
    """The graph recorded in a wrapped payload, absent on legacy traces."""
    if not isinstance(payload, dict):
        return None
    value = payload.get("graph_id")
    return value if isinstance(value, str) and value else None


def _graph_results(payload: Any) -> Any:
    """Expose the original result payload while keeping provenance internal."""
    if isinstance(payload, dict) and "graph_id" in payload and "results" in payload:
        return payload["results"]
    return payload


# --------------------------------------------------------------------------
# bringing it up
# --------------------------------------------------------------------------


def initialize(engine: Engine | None = None) -> Engine:
    """Create the history tables."""
    made = engine if engine is not None else create_history_engine()
    try:
        create_history_schema(made)
    except SQLAlchemyError:
        logger.warning("could not bring up the history tables", exc_info=True)
    return made


def _session_record(session: ChatSession) -> SessionRecord:
    return SessionRecord(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
    )
