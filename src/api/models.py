"""The request and response contract.

Declared types rather than free-form dictionaries, because a request shape
that is only whatever a handler happened to read is not a contract — it is a
description of today's code, and a consumer discovers a change by breaking.

Pydantic is already a runtime dependency of this project, used to validate
payloads at the ingestion boundary. This is the same job at the other
boundary, so it adds nothing to the dependency list.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class TraceRequest(BaseModel):
    """A question to trace, and optionally the session to record it in."""

    query: str
    top_k: int | None = None
    session_id: str | None = None


class SubgraphRequest(BaseModel):
    """The nodes to expand by one hop."""

    node_ids: list[str]


class SummarizeRequest(BaseModel):
    """A note to summarise, and the key its summary is cached under."""

    key: str
    text: str


class AnswerRequest(BaseModel):
    """A question and the context an answer must be grounded in."""

    query: str
    context: str


class SwitchRequest(BaseModel):
    """Which discovered graph to serve, by its file name.

    A name, never a path. A path would make this an instruction to open an
    arbitrary file; a name can only select something discovery already found.
    """

    id: str


class SessionUpsert(BaseModel):
    """Start a session, or retitle one the caller owns.

    ``user_id`` is accepted and ignored: who is asking comes from the session
    token alone.
    """

    user_id: str | None = None
    email: str | None = None
    title: str = "New Chat"
    session_id: str | None = None


class SessionRead(BaseModel):
    """One session, as its owner sees it."""

    id: str
    user_id: str
    title: str
    created_at: datetime


class TraceRead(BaseModel):
    """One recorded question in a session and what answering it involved."""

    id: str
    session_id: str
    query: str
    execution_plan: dict
    graph_payload: dict | list
    created_at: datetime
