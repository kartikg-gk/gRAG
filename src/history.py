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
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from .models.history import (
    DEFAULT_SESSION_TITLE,
    QuerySession,
    QueryTrace,
    User,
    create_history_engine,
    create_history_schema,
    history_sessions,
)

logger = logging.getLogger("graphrag.history")

#: The domain a made-up address is built under when a real one cannot be
#: used. Reserved for exactly this purpose and never deliverable, which is
#: the point: it is a placeholder, not a way to reach anybody.
PLACEHOLDER_DOMAIN = "users.invalid"


class SessionNotAvailable(RuntimeError):
    """This session cannot be acted on by this caller.

    **One error for two situations**: no such session, and somebody else's
    session. Distinguishing them in the response would turn this into a way
    to ask whether a given identifier exists.
    """


@dataclass(frozen=True)
class SessionSummary:
    """One session, as a caller sees it."""

    session_id: str
    title: str
    created_at: int


@dataclass(frozen=True)
class TraceRecord:
    """One recorded question and what answering it involved."""

    trace_id: str
    session_id: str
    query: str
    plan: Optional[dict[str, Any]]
    result: Optional[dict[str, Any]]
    created_at: int


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
    which nothing sends mail to and nobody sees.
    """
    existing = db.get(User, user_id)
    if existing is not None:
        return existing

    address = email or _placeholder_email(user_id)
    taken = db.exec(select(User).where(User.email == address)).first()
    if taken is not None and taken.user_id != user_id:
        logger.warning(
            "%s claims an address already held by %s; using a placeholder",
            user_id,
            taken.user_id,
        )
        address = _placeholder_email(user_id)

    user = User(user_id=user_id, email=address, created_at=_now())
    db.add(user)
    db.commit()
    return user


def _placeholder_email(user_id: str) -> str:
    """An address derived from the identifier. Never user-facing."""
    return f"{user_id}@{PLACEHOLDER_DOMAIN}"


# --------------------------------------------------------------------------
# what a route calls
# --------------------------------------------------------------------------


def create_or_rename_session(
    user_id: str,
    title: str | None = None,
    *,
    session_id: str | None = None,
    email: str | None = None,
    engine: Engine | None = None,
) -> SessionSummary:
    """Start a session, or retitle one this caller already owns."""
    made = engine if engine is not None else create_history_engine()
    sessions = history_sessions(made)

    with sessions() as db:
        if session_id is None:
            # The user row is created here rather than anywhere earlier
            # because this is the first moment anything needs one to exist.
            ensure_user(db, user_id, email)
            session = QuerySession(
                session_id=uuid.uuid4().hex,
                user_id=user_id,
                title=title or DEFAULT_SESSION_TITLE,
                created_at=_now(),
            )
            db.add(session)
            db.commit()
            return _summary(session)

        session = _owned(db, session_id, user_id)
        if title:
            session.title = title
            db.commit()
        return _summary(session)


def list_sessions(
    user_id: str, *, engine: Engine | None = None
) -> list[SessionSummary]:
    """This caller's sessions, most recent first."""
    made = engine if engine is not None else create_history_engine()
    sessions = history_sessions(made)

    with sessions() as db:
        rows = db.exec(
            select(QuerySession)
            .where(QuerySession.user_id == user_id)
            .order_by(QuerySession.created_at.desc(), QuerySession.session_id.desc())
        ).all()

    return [_summary(row) for row in rows]


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
            select(QueryTrace)
            .where(QueryTrace.session_id == session_id)
            .order_by(QueryTrace.created_at, QueryTrace.trace_id)
        ).all()

    return [
        TraceRecord(
            trace_id=row.trace_id,
            session_id=row.session_id,
            query=row.query,
            plan=row.plan,
            result=row.result,
            created_at=row.created_at,
        )
        for row in rows
    ]


def _owned(db: Session, session_id: str, user_id: str) -> QuerySession:
    """The session, if this caller owns it. Otherwise the same refusal either
    way — see ``SessionNotAvailable``."""
    session = db.get(QuerySession, session_id)
    if session is None or session.user_id != user_id:
        raise SessionNotAvailable(session_id)
    return session


# --------------------------------------------------------------------------
# what the query path calls
# --------------------------------------------------------------------------


def session_owner(session_id: str, *, engine: Engine | None = None) -> str | None:
    """Who owns ``session_id``, or ``None``.

    ``None`` covers three things that the caller treats identically: no such
    session, and a history database that could not be reached. Nothing here
    raises, because this sits in front of answering a question and a history
    database being down is not the asker's problem.
    """
    try:
        made = engine if engine is not None else create_history_engine()
        with history_sessions(made)() as db:
            session = db.get(QuerySession, session_id)
            return session.user_id if session is not None else None
    except Exception:  # noqa: BLE001 - history must not break a query
        logger.warning("could not read the owner of session %s", session_id, exc_info=True)
        return None


def record_trace(
    session_id: str,
    query: str,
    *,
    plan: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
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
            trace = QueryTrace(
                trace_id=uuid.uuid4().hex,
                session_id=session_id,
                query=query,
                plan=plan,
                result=result,
                created_at=_now(),
            )
            db.add(trace)
            db.commit()
            return trace.trace_id
    except Exception:  # noqa: BLE001 - history must not break a query
        logger.warning("could not record a trace for %s", session_id, exc_info=True)
        return None


# --------------------------------------------------------------------------
# bringing it up
# --------------------------------------------------------------------------


def initialize(engine: Engine | None = None) -> Engine:
    """Create the history tables. Not called at startup — see the API layer,
    which reaches this store only when a caller names a session."""
    made = engine if engine is not None else create_history_engine()
    try:
        create_history_schema(made)
    except SQLAlchemyError:
        logger.warning("could not bring up the history tables", exc_info=True)
    return made


def _summary(session: QuerySession) -> SessionSummary:
    return SessionSummary(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at,
    )


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())
