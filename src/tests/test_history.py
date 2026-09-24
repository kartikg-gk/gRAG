"""Tests for sessions, traces, and who may see them.

The first one is the hole worth checking for. A system that refuses to let
you *rename* somebody else's session and then hands you its contents when you
ask to read it looks protected from the outside and is not — and reading is
the direction that leaks. So the read check is tested on its own terms rather
than assumed to follow from the write check.

The history database here is real and separate, on its own file. Nothing in
these tests touches the control plane.
"""

from __future__ import annotations

import logging

import pytest
from sqlmodel import select

from src.history import (
    PLACEHOLDER_DOMAIN,
    SessionNotAvailable,
    create_or_rename_session,
    ensure_user,
    list_sessions,
    list_traces,
    persist_trace,
    session_owner,
)
from src.models.control_plane import CONTROL_PLANE_TABLES
from src.models.history import (
    DEFAULT_SESSION_TITLE,
    HISTORY_TABLES,
    HISTORY_URL_VARIABLE,
    HistoryNotConfigured,
    ChatSession,
    TraceLog,
    User,
    create_history_engine,
    create_history_schema,
    history_sessions,
    history_url,
)
from src.models.database import DATABASE_URL_VARIABLE

ALICE = "user_alice"
BOB = "user_bob"


@pytest.fixture()
def engine(tmp_path):
    made = create_history_engine(str(tmp_path / "history.db"))
    create_history_schema(made)
    try:
        yield made
    finally:
        made.dispose()


@pytest.fixture()
def db(engine):
    with history_sessions(engine)() as open_session:
        yield open_session


# ==========================================================================
# the hole that looks fixed from the outside
# ==========================================================================


def test_reading_another_persons_session_is_refused(engine):
    """The read path has its own check, and this is it.

    Guarding the rename and not the read protects the thing nobody wants and
    leaves open the thing they do: somebody else's questions.
    """
    mine = create_or_rename_session(ALICE, "mine", engine=engine)
    persist_trace(session_id=mine.id, query="who reviewed the auth change?", execution_plan={}, graph_payload=[], engine=engine)

    with pytest.raises(SessionNotAvailable):
        list_traces(BOB, mine.id, engine=engine)

    # And the owner still reads it perfectly well.
    assert len(list_traces(ALICE, mine.id, engine=engine)) == 1


def test_renaming_another_persons_session_is_refused(engine):
    mine = create_or_rename_session(ALICE, "mine", engine=engine)

    with pytest.raises(SessionNotAvailable):
        create_or_rename_session(BOB, "yours now", session_id=mine.id, engine=engine)

    assert list_sessions(ALICE, engine=engine)[0].title == "mine"


def test_a_missing_session_and_someone_elses_refuse_identically(engine):
    """Telling them apart would be a way to ask which identifiers are real."""
    mine = create_or_rename_session(ALICE, "mine", engine=engine)

    with pytest.raises(SessionNotAvailable) as theirs:
        list_traces(BOB, mine.id, engine=engine)
    with pytest.raises(SessionNotAvailable) as absent:
        list_traces(BOB, "no-such-session", engine=engine)

    assert type(theirs.value) is type(absent.value)
    # Neither carries anything the other does not.
    assert set(theirs.value.args) - {mine.id} == set()
    assert set(absent.value.args) - {"no-such-session"} == set()


def test_the_two_checks_are_written_separately():
    """Not one check the other happens to call.

    Read and write both resolve ownership through the same helper, and both
    call it themselves. A read that relied on the write path's check would be
    one refactor away from having none.
    """
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "history.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    checked = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Call)
            and getattr(inner.func, "id", None) == "_owned"
            for inner in ast.walk(node)
        )
    }

    assert "list_traces" in checked
    assert "create_or_rename_session" in checked


# ==========================================================================
# people
# ==========================================================================


def test_a_first_time_identifier_gets_a_user_row(engine):
    """No registration step: the identifier arrived from a verified token."""
    with history_sessions(engine)() as reading:
        assert reading.exec(select(User)).all() == []

    created = create_or_rename_session(
        ALICE, "first", email="alice@example.invalid", engine=engine
    )

    with history_sessions(engine)() as reading:
        user = reading.get(User, ALICE)

    assert user is not None
    assert user.email == "alice@example.invalid"
    assert created.id


def test_an_address_already_held_by_someone_else_falls_back(engine, caplog):
    """A collision must not stop a session being created.

    Two identifiers claiming one address is a question this table cannot
    answer. Refusing would block a session over an email; merging would hand
    one person another's history. So the newcomer gets a placeholder.
    """
    create_or_rename_session(ALICE, "first", email="shared@example.invalid", engine=engine)

    with caplog.at_level(logging.WARNING):
        created = create_or_rename_session(
            BOB, "also first", email="shared@example.invalid", engine=engine
        )

    with history_sessions(engine)() as reading:
        bob = reading.get(User, BOB)
        alice = reading.get(User, ALICE)

    assert created.id
    assert alice.email == "shared@example.invalid"
    assert bob.email == f"{BOB}@{PLACEHOLDER_DOMAIN}"
    assert "placeholder" in caplog.text


def test_an_identifier_with_no_address_gets_a_placeholder(engine):
    create_or_rename_session(ALICE, "first", engine=engine)

    with history_sessions(engine)() as reading:
        assert reading.get(User, ALICE).email.endswith(PLACEHOLDER_DOMAIN)


def test_the_tables_take_their_default_names():
    assert [table.name for table in HISTORY_TABLES] == ["user", "chatsession", "tracelog"]


def test_a_new_session_is_a_new_chat():
    assert DEFAULT_SESSION_TITLE == "New Chat"


def test_the_placeholder_address_is_under_the_projects_domain(engine):
    create_or_rename_session(ALICE, "first", engine=engine)

    with history_sessions(engine)() as reading:
        assert reading.get(User, ALICE).email == f"{ALICE}@users.graphrag.local"


def test_renaming_sets_the_title_it_is_given(engine):
    created = create_or_rename_session(ALICE, "before", engine=engine)

    renamed = create_or_rename_session(ALICE, session_id=created.id, engine=engine)

    assert renamed.title == DEFAULT_SESSION_TITLE


def test_the_user_row_is_created_once(db):
    first = ensure_user(db, ALICE, "alice@example.invalid")
    second = ensure_user(db, ALICE, "different@example.invalid")

    assert first.id == second.id
    # The address is not rewritten by a later sighting.
    assert second.email == "alice@example.invalid"
    assert len(db.exec(select(User)).all()) == 1


# ==========================================================================
# sessions
# ==========================================================================


def test_a_session_without_a_title_gets_a_generic_one(engine):
    created = create_or_rename_session(ALICE, engine=engine)

    assert created.title == DEFAULT_SESSION_TITLE


def test_renaming_keeps_the_identifier_and_changes_the_title(engine):
    created = create_or_rename_session(ALICE, "before", engine=engine)
    renamed = create_or_rename_session(
        ALICE, "after", session_id=created.id, engine=engine
    )

    assert renamed.id == created.id
    assert renamed.title == "after"
    assert len(list_sessions(ALICE, engine=engine)) == 1


def test_sessions_are_listed_most_recent_first(engine):
    first = create_or_rename_session(ALICE, "oldest", engine=engine)
    second = create_or_rename_session(ALICE, "middle", engine=engine)
    third = create_or_rename_session(ALICE, "newest", engine=engine)

    listed = list_sessions(ALICE, engine=engine)

    assert [row.title for row in listed] == ["newest", "middle", "oldest"]
    assert [row.id for row in listed] == [third.id, second.id, first.id]


def test_one_persons_sessions_are_not_anothers(engine):
    create_or_rename_session(ALICE, "mine", engine=engine)
    create_or_rename_session(BOB, "theirs", engine=engine)

    assert [row.title for row in list_sessions(ALICE, engine=engine)] == ["mine"]
    assert [row.title for row in list_sessions(BOB, engine=engine)] == ["theirs"]


# ==========================================================================
# traces
# ==========================================================================


def test_traces_are_listed_oldest_first(engine):
    created = create_or_rename_session(ALICE, "mine", engine=engine)

    for question in ("first?", "second?", "third?"):
        persist_trace(
            session_id=created.id, query=question, execution_plan={}, graph_payload=[], engine=engine
        )

    assert [row.query for row in list_traces(ALICE, created.id, engine=engine)] == [
        "first?",
        "second?",
        "third?",
    ]


def test_a_trace_keeps_its_open_ended_payloads(engine):
    created = create_or_rename_session(ALICE, "mine", engine=engine)

    persist_trace(
        session_id=created.id,
        query="who reviewed it?",
        execution_plan={"intent": {"type": "relational", "alpha": 0.15}},
        graph_payload=[{"id": "pr:41", "anything": ["at", "all"]}],
        engine=engine,
    )

    trace = list_traces(ALICE, created.id, engine=engine)[0]

    assert trace.execution_plan == {"intent": {"type": "relational", "alpha": 0.15}}
    assert trace.graph_payload[0]["anything"] == ["at", "all"]


def test_a_trace_records_the_graph_without_a_schema_migration(engine):
    created = create_or_rename_session(ALICE, "mine", engine=engine)

    persist_trace(
        session_id=created.id,
        query="where was this answered?",
        execution_plan={},
        graph_payload=[{"id": "pr:41"}],
        graph_id="owner__repo.lbug",
        engine=engine,
    )

    trace = list_traces(ALICE, created.id, engine=engine)[0]
    assert trace.graph_id == "owner__repo.lbug"
    assert trace.graph_payload == [{"id": "pr:41"}]


def test_who_owns_a_session(engine):
    created = create_or_rename_session(ALICE, "mine", engine=engine)

    assert session_owner(created.id, engine=engine) == ALICE
    assert session_owner("no-such-session", engine=engine) is None


# ==========================================================================
# a broken history store must not break anything
# ==========================================================================


class UnreachableEngine:
    """Stands in for a database that cannot be reached at all."""

    def __getattr__(self, name):
        raise RuntimeError("the history database is unreachable")


def test_recording_a_trace_swallows_a_database_failure(caplog):
    """A question answered and not recorded beats a question refused."""
    with caplog.at_level(logging.WARNING):
        assert (
            persist_trace(
                session_id="any-session",
                query="a question",
                execution_plan={},
                graph_payload=[],
                engine=UnreachableEngine(),
            )
            is None
        )

    assert "could not record a trace" in caplog.text


def test_looking_up_an_owner_swallows_a_database_failure(caplog):
    with caplog.at_level(logging.WARNING):
        assert session_owner("any-session", engine=UnreachableEngine()) is None

    assert "could not read the owner" in caplog.text


def test_a_route_operation_does_not_swallow_a_failure(engine, monkeypatch):
    """The tolerance stops where a person is waiting for an answer.

    Someone who asked for their sessions and got an empty list would believe
    they had none, which is worse than an error.
    """
    with pytest.raises(Exception):
        list_sessions(ALICE, engine=UnreachableEngine())


# ==========================================================================
# where it lives
# ==========================================================================


def test_the_schema_call_creates_only_the_history_tables(tmp_path):
    """Every schema call in this project carries this restriction.

    They share one model registry, and a call that created everything in it
    would put another concern's tables in whichever database opened first.
    """
    from sqlalchemy import inspect
    from sqlmodel import Field, SQLModel

    class SomeOtherConcern(SQLModel, table=True):
        __tablename__ = "some_other_concern"

        id: int = Field(primary_key=True)

    made = create_history_engine(str(tmp_path / "only-history.db"))
    try:
        create_history_schema(made)
        created = set(inspect(made).get_table_names())
    finally:
        made.dispose()
        SQLModel.metadata.remove(SomeOtherConcern.__table__)

    assert created == {table.name for table in HISTORY_TABLES}
    assert "some_other_concern" not in created
    for table in CONTROL_PLANE_TABLES:
        assert table.name not in created


def test_the_url_falls_back_to_the_general_database():
    assert (
        history_url({DATABASE_URL_VARIABLE: "postgresql+psycopg://h/app"})
        == "postgresql+psycopg://h/app"
    )


def test_the_specific_variable_wins():
    assert (
        history_url(
            {
                HISTORY_URL_VARIABLE: "postgresql+psycopg://h/history",
                DATABASE_URL_VARIABLE: "postgresql+psycopg://h/app",
            }
        )
        == "postgresql+psycopg://h/history"
    )


def test_neither_variable_set_is_an_error_naming_both():
    with pytest.raises(HistoryNotConfigured) as failure:
        history_url({})

    assert HISTORY_URL_VARIABLE in str(failure.value)
    assert DATABASE_URL_VARIABLE in str(failure.value)


def test_history_has_its_own_variable_and_its_own_tables():
    """Separate from the control plane in name as well as in schema."""
    from src.models.database import CONTROL_PLANE_URL_VARIABLE

    assert HISTORY_URL_VARIABLE != CONTROL_PLANE_URL_VARIABLE
    assert {table.name for table in HISTORY_TABLES}.isdisjoint(
        {table.name for table in CONTROL_PLANE_TABLES}
    )


def test_a_trace_cannot_name_a_session_that_does_not_exist(engine):
    """The reference is declared and enforced, so a trace belongs somewhere.

    Recorded through the raw model rather than ``persist_trace``, which
    swallows everything by design.
    """
    from sqlalchemy.exc import IntegrityError

    with history_sessions(engine)() as db:
        db.add(
            TraceLog(id="t1", session_id="no-such-session", query="a question")
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()


def test_a_session_cannot_name_a_user_that_does_not_exist(engine):
    from sqlalchemy.exc import IntegrityError

    with history_sessions(engine)() as db:
        db.add(
            ChatSession(id="s1", user_id="nobody", title="orphan")
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()


# ==========================================================================
# the query route, which works exactly as before unless a session is named
# ==========================================================================


def _query_client(monkeypatch, engine, *, retrieved=None):
    """The application with a stand-in engine, and history pointed here."""
    from fastapi.testclient import TestClient

    import src.history as history_module
    from src.api import auth as auth_module
    from src.api.app import create_app

    class FakeHit:
        id = "pr:41"
        score = 0.9
        vector_score = 0.5
        graph_score = 0.4
        decay = 1.0
        node_type = "PR"
        age_days = 1.0
        found_by_both = True

    class FakeIntent:
        intent = "relational"

    class FakeFused:
        alpha = 0.15
        beta = 0.85
        hits = [FakeHit()]

    class FakeRun:
        query = "who reviewed the auth change?"
        intent = FakeIntent()
        fused = FakeFused()
        hits = [FakeHit()]
        ids = ["pr:41"]
        from src.tests.contract_fixtures import trace_log

        trace_log = trace_log()

    class FakeStore:
        def documents_for_entities(self, ids):
            return {}

        def get_entity(self, entity_id):
            return {"id": entity_id, "label": entity_id, "type": "PR"}

    class FakeEngine:
        path = "fake-store.lbug"
        embedder = extractor = judge = now = None
        store = FakeStore()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def close(self):
            pass

        @property
        def router(self):
            return self

        def warm(self):
            pass

        def build_context(self, results):
            return ""

        async def route_async(self, query, k=None):
            from src.retrieval.response import RouterResponse, RoutedNode
            (retrieved if retrieved is not None else []).append(query)
            hit = FakeHit()
            node = RoutedNode(hit.id, hit.id, "PR", hit.score, hit.vector_score,
                              hit.graph_score, hit.decay, hit.age_days, [])
            return RouterResponse(query, [node], FakeRun.trace_log)

        async def retrieve_async(self, query, k=None):
            (retrieved if retrieved is not None else []).append(query)
            return FakeRun()

    # The history layer reaches the test database rather than the environment.
    original_engine = history_module.create_history_engine
    monkeypatch.setattr(
        history_module, "create_history_engine", lambda *a, **k: engine
    )
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)

    return TestClient(create_app(engine_factory=FakeEngine)), original_engine


def test_a_query_naming_no_session_records_nothing(engine, monkeypatch):
    """The path that existed before history did, unchanged.

    Nothing is looked up, nothing is written, and the history database is
    never reached — asserted by making any use of it fail the test.
    """
    import src.history as history_module

    client, _ = _query_client(monkeypatch, engine)

    def refuse(*args, **kwargs):
        raise AssertionError("the history store was consulted for a plain query")

    monkeypatch.setattr(history_module, "session_owner", refuse)
    monkeypatch.setattr(history_module, "persist_trace", refuse)

    with client:
        response = client.post("/api/trace", json={"query": "who reviewed it?"})

    assert response.status_code == 200
    with history_sessions(engine)() as reading:
        assert reading.exec(select(TraceLog)).all() == []


def test_a_query_naming_a_session_records_into_it(engine, monkeypatch):
    """Naming a session opts into recording, and the answer is unchanged."""
    from src.common.config import DEV_USER_ID

    created = create_or_rename_session(DEV_USER_ID, "mine", engine=engine)
    client, _ = _query_client(monkeypatch, engine)

    with client:
        response = client.post(
            "/api/trace",
            json={"query": "who reviewed it?", "session_id": created.id},
        )

    assert response.status_code == 200
    assert response.json()["query"] == "who reviewed it?"

    traces = list_traces(DEV_USER_ID, created.id, engine=engine)
    assert len(traces) == 1
    assert traces[0].query == "who reviewed it?"
    # The trace log and the ranked nodes are both kept, in whatever shape they had.
    from src.tests.contract_fixtures import trace_log

    assert traces[0].execution_plan == trace_log()
    assert traces[0].graph_payload[0]["id"] == "pr:41"
    assert response.json()["trace_id"] == traces[0].id


def test_a_session_that_is_not_the_callers_is_refused_and_nothing_is_recorded(
    engine, monkeypatch
):
    """The route runs the query, then checks the session before recording.

    A session that is somebody else's is the one uniform refusal, and nothing
    is written into it.
    """
    somebody_else = create_or_rename_session(BOB, "theirs", engine=engine)

    retrieved: list[str] = []
    client, _ = _query_client(monkeypatch, engine, retrieved=retrieved)

    with client:
        response = client.post(
            "/api/trace",
            json={"query": "who reviewed it?", "session_id": somebody_else.id},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "no such session"}
    assert list_traces(BOB, somebody_else.id, engine=engine) == []


def test_a_recording_failure_does_not_change_the_answer(engine, monkeypatch):
    """The whole reason recording swallows its failures."""
    from src.common.config import DEV_USER_ID

    created = create_or_rename_session(DEV_USER_ID, "mine", engine=engine)
    client, _ = _query_client(monkeypatch, engine)

    import src.history as history_module

    def fall_over(*args, **kwargs):
        raise RuntimeError("the history database went away mid-write")

    with client:
        monkeypatch.setattr(history_module, "TraceLog", fall_over)
        response = client.post(
            "/api/trace",
            json={"query": "who reviewed it?", "session_id": created.id},
        )

    # The patch is lifted before reading back, or the read would fail for the
    # same reason the write did and prove nothing about the answer.
    monkeypatch.undo()

    assert response.status_code == 200
    assert response.json()["query"] == "who reviewed it?"
    # Answered, not recorded.
    assert list_traces(DEV_USER_ID, created.id, engine=engine) == []
