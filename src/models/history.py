"""Who asked what, and what came back.

A third database concern, and the first one scoped to a **person** rather
than to a tenant. Everything else in this project divides the world by
organisation: which tenant owns a graph, which tenant a pod serves, which
tenant a compile is for. This divides it by who is sitting at the keyboard,
and the two axes do not line up — a user belongs to a tenant, but their
history is theirs.

Its own connection, deliberately
--------------------------------

Named by its own environment variable, falling back to the same general
application database the control plane falls back to. In development that
usually means one Postgres holding everything; in a deployment it can be
moved without touching code, which matters more here than elsewhere: this is
the data with a person's name on it.

The schema call creates **only these tables**. Every schema call in this
project carries that restriction, because they all share one model registry
and a call that created everything in it would put somebody else's tables in
whichever database happened to be opened first.

Open-ended payloads
-------------------

A trace's plan and result are structured data with no fixed shape. What a
trace records is decided by the retrieval path, which changes; a column per
field would make every new thing worth recording a schema change, and the
things worth recording are exactly the ones nobody predicted.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Column, Engine, create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlmodel import Field, Session, SQLModel

from .database import DATABASE_URL_VARIABLE, as_url

#: This database, and the general one it falls back to.
#:
#: The same shape as the control plane's pair and a different specific
#: variable, so a deployment can put a person's query history somewhere else
#: without moving anything about tenants.
HISTORY_URL_VARIABLE = "GRAPHRAG_HISTORY_DATABASE_URL"

#: What a session is called when nobody names it.
DEFAULT_SESSION_TITLE = "New Chat"


class HistoryNotConfigured(RuntimeError):
    """Neither environment variable names a database for history."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class User(SQLModel, table=True):
    """One person, known by the identifier their credential carries.

    There is no registration step. By the time anything here runs, the
    identifier came out of a verified token — asking the person to also exist
    in a table before they can have a session would be asking them to
    announce what has already been proved.
    """

    id: str = Field(primary_key=True)
    email: str = Field(unique=True, index=True)
    created_at: datetime = Field(default_factory=_utcnow)


class ChatSession(SQLModel, table=True):
    """A named run of questions, belonging to exactly one person."""

    id: str = Field(default_factory=_new_id, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    title: str = Field(default=DEFAULT_SESSION_TITLE)
    created_at: datetime = Field(default_factory=_utcnow)


class TraceLog(SQLModel, table=True):
    """One question asked in a session, and what answering it involved."""

    id: str = Field(default_factory=_new_id, primary_key=True)
    session_id: str = Field(foreign_key="chatsession.id", index=True)
    query: str
    execution_plan: dict = Field(default_factory=dict, sa_column=Column(JSON))
    graph_payload: Any = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)


HISTORY_MODELS = (User, ChatSession, TraceLog)
HISTORY_TABLES = tuple(model.__table__ for model in HISTORY_MODELS)


def history_url(environment: dict[str, str] | None = None) -> str:
    """Where history lives, from the environment.

    Raises rather than inventing one, naming both variables — the same rule
    the control plane follows, for the same reason: a default would work on
    the machine that wrote it and give every process its own private copy
    everywhere else.
    """
    source = environment if environment is not None else os.environ

    for variable in (HISTORY_URL_VARIABLE, DATABASE_URL_VARIABLE):
        value = (source.get(variable) or "").strip()
        if value:
            return value

    raise HistoryNotConfigured(
        f"history has no database: set {HISTORY_URL_VARIABLE}, or "
        f"{DATABASE_URL_VARIABLE} to share one database with the rest of the "
        "application. There is no default."
    )


def create_history_engine(target: str | None = None, **options) -> Engine:
    """An engine for the history database."""
    url = as_url(target) if target is not None else history_url()

    engine = create_engine(url, pool_pre_ping=True, **options)

    if engine.dialect.name == "sqlite":
        # Per connection, because that is the only scope SQLite offers. A
        # declared reference that is checked on some connections and not
        # others is worse than one that is never checked.
        @event.listens_for(engine, "connect")
        def _set_pragma(connection, _record):  # pragma: no branch
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def history_sessions(engine: Engine) -> sessionmaker[Session]:
    """Sessions bound to ``engine``, holding model instances after commit."""
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def create_history_schema(engine: Engine) -> None:
    """Bring up the history tables, and only those."""
    SQLModel.metadata.create_all(engine, tables=list(HISTORY_TABLES), checkfirst=True)
