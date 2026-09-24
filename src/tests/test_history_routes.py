"""The query-history routes: sessions and their traces, scoped to the caller.

The caller is whoever the session dependency says it is, swapped per test by
overriding that dependency. History is pointed at a throwaway database.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.api import auth as auth_module
from src.api.auth import get_current_user
from src.history import create_or_rename_session, persist_trace
from src.models.history import create_history_engine, create_history_schema

ALICE = "user_alice"
BOB = "user_bob"
NOT_AVAILABLE = {"detail": "no such session"}
SESSION_FIELDS = {"id", "user_id", "title", "created_at"}
TRACE_FIELDS = {"id", "session_id", "query", "execution_plan", "graph_payload", "graph_id", "created_at"}


@pytest.fixture()
def engine(tmp_path):
    made = create_history_engine(str(tmp_path / "history.db"))
    create_history_schema(made)
    try:
        yield made
    finally:
        made.dispose()


@pytest.fixture()
def caller():
    return {"user_id": ALICE}


@pytest.fixture()
def client(monkeypatch, engine, caller):
    import src.history as history_module
    from src.api.app import create_app

    monkeypatch.setattr(history_module, "create_history_engine", lambda *a, **k: engine)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)

    app = create_app(engine_factory=lambda: None)
    app.dependency_overrides[get_current_user] = lambda: caller["user_id"]
    return TestClient(app)


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# ==========================================================================
# sessions
# ==========================================================================


def test_a_created_session_is_listed(client):
    created = client.post("/api/sessions", json={"title": "first questions"})

    assert created.status_code == 200
    body = created.json()
    assert set(body) == SESSION_FIELDS
    assert body["title"] == "first questions"
    assert body["user_id"] == ALICE
    _iso(body["created_at"])

    listed = client.get("/api/sessions")

    assert listed.status_code == 200
    assert listed.json() == [body]


def test_a_session_with_no_title_is_a_new_chat(client):
    assert client.post("/api/sessions", json={}).json()["title"] == "New Chat"


def test_a_caller_can_rename_their_own_session(client):
    session_id = client.post("/api/sessions", json={"title": "draft"}).json()["id"]

    renamed = client.post("/api/sessions", json={"session_id": session_id, "title": "final"})

    assert renamed.status_code == 200
    assert renamed.json()["id"] == session_id
    assert renamed.json()["title"] == "final"
    assert [row["title"] for row in client.get("/api/sessions").json()] == ["final"]


def test_renaming_someone_elses_session_is_not_found(client, engine, caller):
    theirs = create_or_rename_session(BOB, "bob's", engine=engine)

    response = client.post("/api/sessions", json={"session_id": theirs.id, "title": "mine now"})

    assert response.status_code == 404
    assert response.json() == NOT_AVAILABLE
    caller["user_id"] = BOB
    assert [row["title"] for row in client.get("/api/sessions").json()] == ["bob's"]


def test_renaming_a_missing_session_is_the_same_not_found(client):
    response = client.post("/api/sessions", json={"session_id": "no-such-session", "title": "x"})

    assert response.status_code == 404
    assert response.json() == NOT_AVAILABLE


def test_the_list_holds_only_the_callers_sessions_newest_first(client, engine):
    create_or_rename_session(ALICE, "oldest", engine=engine)
    create_or_rename_session(BOB, "not alice's", engine=engine)
    create_or_rename_session(ALICE, "middle", engine=engine)
    create_or_rename_session(ALICE, "newest", engine=engine)

    listed = client.get("/api/sessions").json()

    assert [row["title"] for row in listed] == ["newest", "middle", "oldest"]


def test_a_user_id_in_the_body_is_ignored(client, caller):
    created = client.post("/api/sessions", json={"title": "whose?", "user_id": BOB})

    assert created.status_code == 200
    assert created.json()["user_id"] == ALICE
    caller["user_id"] = BOB
    assert client.get("/api/sessions").json() == []


def test_a_user_id_in_the_query_string_is_ignored(client, engine):
    create_or_rename_session(BOB, "bob's", engine=engine)

    assert client.get("/api/sessions", params={"user_id": BOB}).json() == []


# ==========================================================================
# traces
# ==========================================================================


def test_traces_come_back_oldest_first(client, engine):
    session = create_or_rename_session(ALICE, "asked", engine=engine)
    for question in ("first", "second", "third"):
        persist_trace(
            session_id=session.id,
            query=question,
            execution_plan={"intent": {"type": "relational"}},
            graph_payload=[{"id": "pr:1"}],
            engine=engine,
        )

    response = client.get(f"/api/sessions/{session.id}/traces")

    assert response.status_code == 200
    rows = response.json()
    assert [row["query"] for row in rows] == ["first", "second", "third"]
    assert set(rows[0]) == TRACE_FIELDS
    assert rows[0]["execution_plan"] == {"intent": {"type": "relational"}}
    assert rows[0]["graph_payload"] == [{"id": "pr:1"}]
    assert rows[0]["session_id"] == session.id
    _iso(rows[0]["created_at"])


def test_a_session_with_no_traces_lists_none(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]

    assert client.get(f"/api/sessions/{session_id}/traces").json() == []


def test_reading_someone_elses_traces_is_not_found(client, engine):
    theirs = create_or_rename_session(BOB, "bob's", engine=engine)
    persist_trace(session_id=theirs.id, query="private", execution_plan={}, graph_payload=[], engine=engine)

    response = client.get(f"/api/sessions/{theirs.id}/traces")

    assert response.status_code == 404
    assert response.json() == NOT_AVAILABLE


def test_reading_a_missing_sessions_traces_is_the_same_not_found(client):
    response = client.get("/api/sessions/no-such-session/traces")

    assert response.status_code == 404
    assert response.json() == NOT_AVAILABLE


# ==========================================================================
# what these routes do not depend on
# ==========================================================================


def test_with_tenancy_on_no_key_is_needed_and_no_control_plane_is_touched(monkeypatch, client):
    def untouchable(*args, **kwargs):
        raise AssertionError("the control plane was reached")

    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    monkeypatch.setattr(auth_module, "control_plane", untouchable)
    monkeypatch.setattr(auth_module, "resolve_org", untouchable)

    created = client.post("/api/sessions", json={"title": "no key"})
    session_id = created.json()["id"]

    assert created.status_code == 200
    assert client.get("/api/sessions").status_code == 200
    assert client.get(f"/api/sessions/{session_id}/traces").status_code == 200


def test_an_unconfigured_history_database_is_a_server_error(monkeypatch, caller):
    import src.history as history_module
    from src.api.app import create_app
    from src.models.history import history_url

    def unconfigured(*args, **kwargs):
        history_url({})  # raises the same error an unset environment does

    monkeypatch.setattr(history_module, "create_history_engine", unconfigured)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)
    app = create_app(engine_factory=lambda: None)
    app.dependency_overrides[get_current_user] = lambda: caller["user_id"]

    response = TestClient(app, raise_server_exceptions=False).get("/api/sessions")

    assert response.status_code == 500


def test_the_history_routes_are_not_tenant_gated():
    from src.api.routing import TENANT_SCOPED_PREFIXES

    assert not any("/api/sessions".startswith(prefix) for prefix in TENANT_SCOPED_PREFIXES)
