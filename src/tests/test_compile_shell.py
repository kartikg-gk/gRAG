"""Tests for everything around a compile except the compile.

Two of these are the reason the rest exists.

**A failed job frees the slot.** The database refuses a second in-flight job
for one organisation, so a retry can only open a new row if the previous one
was finalised out of the in-flight set first. Get that wrong and every retry
is refused — not loudly, but as an organisation that quietly stops being
rebuilt while every dispatch reports success.

**The phases raising leaves a failed row and still propagates.** The row is
what anyone reading the database sees; the re-raise is what lets the task
decide about retrying. Dropping either half looks fine from the other side.

The control plane is a real database with its real constraints. The lock is a
real server. Only the phases and the retry are stood in for.
"""

from __future__ import annotations

import logging

import pytest
from sqlmodel import select

from src.ingestion.github import GitHubError, GitHubRateLimitError
from src.models.control_plane import (
    ARTIFACT_READY,
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_FETCHING,
    JOB_IN_FLIGHT,
    JOB_REGISTERED,
    GraphArtifact,
    IngestJob,
    Organization,
    create_control_plane_schema,
)
from src.models.database import control_plane_sessions, create_control_plane_engine
from src.worker import compile as compile_module
from src.worker.compile import (
    MAX_RETRIES,
    SKIPPED_LOCKED,
    UNKNOWN_ORGANIZATION,
    _reconcile,
    finalize_job,
    next_version,
    set_job_status,
)
from src.worker.config import REDIS_URL
from src.worker.locks import LOCK_EXPIRY_SECONDS, compile_lock, lock_key

NOW = 1_700_000_000


def _connection():
    """A client on the test database, or ``None`` if nothing is listening."""
    try:
        import redis
    except ImportError:  # pragma: no cover - the library is a dependency
        return None

    try:
        made = redis.Redis.from_url(
            REDIS_URL.rsplit("/", 1)[0] + "/15",
            decode_responses=True,
            socket_connect_timeout=2,
        )
        made.ping()
    except Exception:  # noqa: BLE001 - any failure means "no server here"
        return None
    return made


@pytest.fixture()
def locks(monkeypatch):
    """A real lock server, with this organisation's key cleared either side."""
    from src.worker import debounce

    made = _connection()
    if made is None:
        pytest.skip("no reachable server; a lock cannot be shown against a fake")

    for key in made.scan_iter("reconcile:lock:*"):
        made.delete(key)
    monkeypatch.setattr(debounce, "_client", made)
    try:
        yield made
    finally:
        for key in made.scan_iter("reconcile:lock:*"):
            made.delete(key)
        monkeypatch.setattr(debounce, "_client", None)
        made.close()


@pytest.fixture()
def engine(tmp_path):
    made = create_control_plane_engine(tmp_path / "control-plane.db")
    create_control_plane_schema(made)
    try:
        yield made
    finally:
        made.dispose()


@pytest.fixture()
def sessions(engine):
    return control_plane_sessions(engine)


@pytest.fixture()
def db(sessions):
    with sessions() as open_session:
        yield open_session


def organization(db, org_id="org_1"):
    db.add(
        Organization(
            org_id=org_id,
            name=org_id,
            plan="team",
            status="active",
            created_at=NOW,
            updated_at=NOW,
        )
    )
    db.commit()
    return org_id


def jobs_for(db, org_id):
    return db.exec(select(IngestJob).where(IngestJob.org_id == org_id)).all()


# ==========================================================================
# the retry interaction
# ==========================================================================


def test_a_failed_job_frees_the_slot_for_the_next_one(db, monkeypatch):
    """The whole reason the core finalises before it re-raises.

    The database allows one in-flight job per organisation. A first attempt
    that failed must leave a row the constraint no longer counts, or the
    retry's insert is refused — and that refusal is silent from the outside:
    the task reports having run, and the organisation never compiles again.
    """
    organization(db, "org_1")

    def refuse(org_id, job_id, session):
        raise RuntimeError("the phases fell over")

    monkeypatch.setattr(compile_module, "run_phases", refuse)

    with pytest.raises(RuntimeError):
        _reconcile("org_1", db)

    first = jobs_for(db, "org_1")
    assert len(first) == 1
    assert first[0].status == JOB_FAILED
    assert first[0].status not in JOB_IN_FLIGHT
    assert first[0].error == "the phases fell over"

    # The retry. It gets a second row precisely because the first no longer
    # counts as in flight.
    with pytest.raises(RuntimeError):
        _reconcile("org_1", db)

    both = jobs_for(db, "org_1")
    assert len(both) == 2
    assert {job.status for job in both} == {JOB_FAILED}


def test_the_constraint_still_refuses_a_second_job_that_is_in_flight(db):
    """The half of the interaction that must keep working.

    If finalising to failed also stopped the constraint applying to genuinely
    running jobs, the test above would pass while the guarantee was gone.
    """
    from sqlalchemy.exc import IntegrityError

    organization(db, "org_1")

    db.add(
        IngestJob(
            job_id="job_a",
            org_id="org_1",
            trigger="schedule",
            status=JOB_FETCHING,
            queued_at=NOW,
        )
    )
    db.commit()

    db.add(
        IngestJob(
            job_id="job_b",
            org_id="org_1",
            trigger="schedule",
            status=JOB_FETCHING,
            queued_at=NOW,
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


# ==========================================================================
# the failure path
# ==========================================================================


def test_the_phases_raising_leaves_a_failed_row_and_still_propagates(
    db, caplog, monkeypatch
):
    """Both halves. The row is the record; the exception is the signal.

    Dropping either looks fine from the other side: a row nobody wrote leaves
    the database silent about a failure, and a swallowed exception leaves the
    task believing the compile worked.
    """
    organization(db, "org_1")

    def fall_over(org_id, job_id, session):
        raise RuntimeError("the phases gave out")

    monkeypatch.setattr(compile_module, "run_phases", fall_over)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            _reconcile("org_1", db)

    job = jobs_for(db, "org_1")[0]

    assert job.status == JOB_FAILED
    assert job.error == "the phases gave out"
    assert job.finished_at is not None
    assert job.started_at is not None
    assert "compile failed" in caplog.text


def test_the_shell_calls_the_phases_with_the_job_it_opened(db, monkeypatch):
    """The identifier the phases record their progress against.

    Handed down rather than looked up, so the row the shell will finalise on
    failure and the row the phases move through their statuses are the same
    one by construction.
    """
    seen = {}

    def watch(org_id, job_id, session):
        seen["org_id"] = org_id
        seen["job_id"] = job_id
        seen["session"] = session
        return _summary(skipped=False)

    organization(db, "org_1")
    monkeypatch.setattr(compile_module, "run_phases", watch)

    _reconcile("org_1", db)

    opened = jobs_for(db, "org_1")[0]
    assert seen["org_id"] == "org_1"
    assert seen["job_id"] == opened.job_id
    assert seen["session"] is db


def test_an_unknown_organisation_opens_no_job_and_does_not_raise(db, caplog):
    """Armed, then deleted before the sweep came round. Not an error."""
    with caplog.at_level(logging.WARNING):
        assert _reconcile("org_gone", db) == {
            "org_id": "org_gone",
            "status": UNKNOWN_ORGANIZATION,
        }

    assert jobs_for(db, "org_gone") == []
    assert "no such organisation" in caplog.text


def test_a_job_is_opened_in_flight_before_the_phases_run(db, monkeypatch):
    """What a reader of the database sees while a compile is running."""
    seen = {}

    def look(org_id, job_id, session):
        job = session.get(IngestJob, job_id)
        seen["status"] = job.status
        seen["started_at"] = job.started_at
        seen["org_id"] = job.org_id
        seen["trigger"] = job.trigger
        return _summary(skipped=False)

    organization(db, "org_1")
    monkeypatch.setattr(compile_module, "run_phases", look)

    result = _reconcile("org_1", db)
    assert result["status"] == JOB_REGISTERED

    assert seen["status"] == JOB_FETCHING
    assert seen["status"] in JOB_IN_FLIGHT
    assert seen["started_at"] is not None
    assert seen["org_id"] == "org_1"
    assert seen["trigger"] == "schedule"


# ==========================================================================
# the lock
# ==========================================================================


def test_the_lock_is_acquired_when_it_is_free(locks):
    with compile_lock("org_1") as acquired:
        assert acquired is True
        assert locks.get(lock_key("org_1")) is not None

    # Released on the way out.
    assert locks.get(lock_key("org_1")) is None


def test_a_second_holder_is_told_no_rather_than_waiting(locks):
    """Not acquiring is an answer, not an exception and not a queue."""
    with compile_lock("org_1") as first:
        assert first is True

        with compile_lock("org_1") as second:
            assert second is False


def test_a_lock_someone_else_now_holds_is_not_deleted(locks):
    """The reason the stored value is unique to each acquisition.

    A holder that ran past its expiry does not own the key it can see. If it
    deleted that key on the way out, the worker that acquired it next would
    be running unprotected while a third worker took the lock.
    """
    with compile_lock("org_1") as acquired:
        assert acquired is True
        # Stand in for: this lock expired, and another worker took it.
        locks.set(lock_key("org_1"), "another-worker-token")

    assert locks.get(lock_key("org_1")) == "another-worker-token"


def test_each_acquisition_stores_a_different_value(locks):
    """A constant would make the release check meaningless."""
    with compile_lock("org_1"):
        first = locks.get(lock_key("org_1"))
    with compile_lock("org_1"):
        second = locks.get(lock_key("org_1"))

    assert first != second


def test_the_lock_expires_so_a_dead_worker_does_not_block_forever(locks):
    """A safety release, not a timeout on ordinary work."""
    with compile_lock("org_1"):
        assert 0 < locks.ttl(lock_key("org_1")) <= LOCK_EXPIRY_SECONDS

    assert LOCK_EXPIRY_SECONDS == 1800


def test_locks_are_held_per_organisation(locks):
    with compile_lock("org_1") as first:
        with compile_lock("org_2") as second:
            assert first is True
            assert second is True


def test_a_task_that_cannot_get_the_lock_returns_and_opens_no_job(
    locks, engine, monkeypatch, caplog
):
    """Another compile is legitimately running. That is not a failure."""
    monkeypatch.setattr(
        compile_module, "create_control_plane_engine", lambda *a, **k: engine
    )

    with compile_lock("org_1") as held:
        assert held is True

        with caplog.at_level(logging.INFO):
            assert compile_module.reconcile_org_to_head("org_1") == {
                "org_id": "org_1",
                "status": SKIPPED_LOCKED,
            }

    with control_plane_sessions(engine)() as reading:
        assert reading.exec(select(IngestJob)).all() == []
    assert "already compiling elsewhere" in caplog.text


# ==========================================================================
# what is retried, and what is not
# ==========================================================================


def test_only_the_quota_error_asks_for_a_retry(locks, engine, monkeypatch):
    """Everything else fails on the first attempt.

    A broken build is not made less broken by five more runs, and the job row
    already says why.
    """
    retries = []

    def record_retry(*, exc=None, countdown=None):
        retries.append(countdown)
        return RuntimeError("retry requested")

    monkeypatch.setattr(
        compile_module, "create_control_plane_engine", lambda *a, **k: engine
    )
    monkeypatch.setattr(
        compile_module.reconcile_org_to_head, "retry", record_retry, raising=False
    )

    with control_plane_sessions(engine)() as setup:
        organization(setup, "org_1")

    def fall_over(org_id, job_id, session):
        raise GitHubError("something else went wrong")

    monkeypatch.setattr(compile_module, "run_phases", fall_over)

    with pytest.raises(GitHubError):
        compile_module.reconcile_org_to_head("org_1")

    assert retries == []


@pytest.mark.parametrize(
    "retry_after, expected",
    [
        (None, 60),  # nothing said: the documented default
        (12.0, 12),  # what the source asked for
        (7200.0, 3600),  # capped, however long it asks for
    ],
)
def test_a_quota_failure_backs_off_by_what_the_source_says(
    locks, engine, monkeypatch, retry_after, expected
):
    """Asserted against a stand-in: the question is what delay was requested.

    Whether a worker actually waited is the library's business.
    """
    retries = []

    def record_retry(*, exc=None, countdown=None):
        retries.append(countdown)
        return RuntimeError("retry requested")

    monkeypatch.setattr(
        compile_module, "create_control_plane_engine", lambda *a, **k: engine
    )
    monkeypatch.setattr(
        compile_module.reconcile_org_to_head, "retry", record_retry, raising=False
    )

    with control_plane_sessions(engine)() as setup:
        organization(setup, "org_1")

    def out_of_quota(org_id, job_id, session):
        raise GitHubRateLimitError("spent", retry_after=retry_after)

    monkeypatch.setattr(compile_module, "run_phases", out_of_quota)

    with pytest.raises(RuntimeError):
        compile_module.reconcile_org_to_head("org_1")

    assert retries == [expected]

    # And the row was finalised on the way out, which is what lets the retry
    # open a fresh one.
    with control_plane_sessions(engine)() as reading:
        assert [job.status for job in reading.exec(select(IngestJob)).all()] == [
            JOB_FAILED
        ]


def test_the_task_gives_up_after_a_bounded_number_of_attempts():
    assert compile_module.reconcile_org_to_head.max_retries == MAX_RETRIES == 5


def test_the_task_is_registered_under_the_name_the_sweep_dispatches():
    from src.worker.app import COMPILE_TASK, app

    assert compile_module.reconcile_org_to_head.name == COMPILE_TASK
    assert COMPILE_TASK in app.tasks
    assert "src.worker" not in COMPILE_TASK


# ==========================================================================
# transitions and version numbers
# ==========================================================================


def test_a_status_update_for_a_job_that_is_gone_does_nothing(db, caplog):
    """Bookkeeping for a vanished row is not worth failing a compile over."""
    with caplog.at_level(logging.WARNING):
        set_job_status(db, "job_that_never_existed", JOB_COMPLETED)
        finalize_job(db, "job_that_never_existed", JOB_FAILED, error="nothing")

    assert "no job job_that_never_existed" in caplog.text


def test_a_status_update_moves_the_row(db):
    organization(db, "org_1")
    db.add(
        IngestJob(
            job_id="job_1",
            org_id="org_1",
            trigger="schedule",
            status=JOB_FETCHING,
            queued_at=NOW,
        )
    )
    db.commit()

    set_job_status(db, "job_1", JOB_COMPLETED)

    assert db.get(IngestJob, "job_1").status == JOB_COMPLETED
    # A plain status change is not a finish.
    assert db.get(IngestJob, "job_1").finished_at is None


def test_finalising_records_the_reason_and_the_time(db):
    organization(db, "org_1")
    db.add(
        IngestJob(
            job_id="job_1",
            org_id="org_1",
            trigger="schedule",
            status=JOB_FETCHING,
            queued_at=NOW,
        )
    )
    db.commit()

    finalize_job(db, "job_1", JOB_FAILED, error="the source refused")

    job = db.get(IngestJob, "job_1")
    assert job.status == JOB_FAILED
    assert job.error == "the source refused"
    assert job.finished_at is not None


def test_the_first_version_is_one(db):
    organization(db, "org_1")

    assert next_version(db, "org_1") == 1


def test_the_next_version_is_the_highest_plus_one(db):
    organization(db, "org_1")
    organization(db, "org_2")

    for version in (1, 2, 5):
        db.add(
            GraphArtifact(
                artifact_id=f"art_{version}",
                org_id="org_1",
                version=version,
                s3_uri=f"local://artifacts/org_1/v{version}.lbug",
                status=ARTIFACT_READY,
                created_at=NOW,
            )
        )
    db.commit()

    assert next_version(db, "org_1") == 6
    # Version numbers are per organisation, so another tenant starts at one.
    assert next_version(db, "org_2") == 1


def test_two_compiles_computing_the_same_version_are_settled_by_the_database(db):
    """Racy on purpose: the constraint is the guarantee, not the arithmetic.

    Locking around the computation would move the guarantee somewhere weaker
    than the database.
    """
    from sqlalchemy.exc import IntegrityError

    organization(db, "org_1")
    version = next_version(db, "org_1")
    assert next_version(db, "org_1") == version

    db.add(
        GraphArtifact(
            artifact_id="art_a",
            org_id="org_1",
            version=version,
            s3_uri="local://artifacts/org_1/v1.lbug",
            created_at=NOW,
        )
    )
    db.commit()

    db.add(
        GraphArtifact(
            artifact_id="art_b",
            org_id="org_1",
            version=version,
            s3_uri="local://artifacts/org_1/v1.lbug",
            created_at=NOW,
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


# ==========================================================================
# a run that found nothing to do
# ==========================================================================


def _summary(*, skipped):
    from src.worker.phases import CompileSummary

    return CompileSummary(
        org_id="org_1",
        artifact_id=None if skipped else "art_1",
        version=None if skipped else 1,
        entities=0 if skipped else 5,
        edges=0 if skipped else 4,
        items=0 if skipped else 3,
        s3_uri=None if skipped else "local://tenants/org_1/1.lbug",
        skipped=skipped,
    )


def test_a_run_that_changed_nothing_reports_skipped(db, monkeypatch):
    organization(db, "org_1")
    monkeypatch.setattr(
        compile_module, "run_phases", lambda org_id, job_id, session: _summary(skipped=True)
    )

    result = _reconcile("org_1", db)
    assert result["org_id"] == "org_1"
    assert result["status"] == JOB_COMPLETED


def test_a_run_that_built_something_still_reports_compiled(db, monkeypatch):
    organization(db, "org_1")
    monkeypatch.setattr(
        compile_module, "run_phases", lambda org_id, job_id, session: _summary(skipped=False)
    )

    result = _reconcile("org_1", db)
    assert set(result) == {
        "org_id", "job_id", "artifact_id", "version", "s3_uri", "status"
    }
    assert result["status"] == JOB_REGISTERED


def test_the_task_for_a_run_with_no_changes_says_skipped_and_closes_the_job(
    locks, engine, monkeypatch
):
    from src.worker import phases as phases_module

    class NothingNew:
        def __call__(self, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(cursor=None, items=0)

    def no_changes(org_id, job_id, session):
        return phases_module.run_phases(org_id, job_id, session, ingest=NothingNew())

    monkeypatch.setattr(
        compile_module, "create_control_plane_engine", lambda *a, **k: engine
    )
    monkeypatch.setattr(compile_module, "run_phases", no_changes)
    with control_plane_sessions(engine)() as seeding:
        organization(seeding, "org_1")

    result = compile_module.reconcile_org_to_head("org_1")
    assert result["status"] == JOB_COMPLETED

    with control_plane_sessions(engine)() as reading:
        job = reading.exec(select(IngestJob)).one()
    assert job.status == JOB_COMPLETED
    assert job.finished_at is not None


# ==========================================================================
# abandoned jobs
# ==========================================================================


def test_a_job_its_worker_abandoned_is_failed_and_frees_the_tenant(db):
    from src.models.control_plane import JOB_COMPILING, JOB_FAILED
    from src.worker.compile import LOCK_EXPIRY_SECONDS, fail_abandoned_jobs

    organization(db)
    started = NOW
    db.add(IngestJob(job_id="stuck", org_id="org_1", trigger="schedule",
                     status=JOB_COMPILING, queued_at=started, started_at=started))
    db.commit()

    failed = fail_abandoned_jobs(db, now=int(started + LOCK_EXPIRY_SECONDS + 1))

    assert failed == ["stuck"]
    job = db.get(IngestJob, "stuck")
    assert job.status == JOB_FAILED and "abandoned" in job.error

    # The one-in-flight index no longer refuses the tenant's next build.
    db.add(IngestJob(job_id="next", org_id="org_1", trigger="schedule",
                     status=JOB_FETCHING, queued_at=started))
    db.commit()


def test_a_job_inside_the_lock_lifetime_is_left_running(db):
    from src.models.control_plane import JOB_COMPILING
    from src.worker.compile import LOCK_EXPIRY_SECONDS, fail_abandoned_jobs

    organization(db)
    db.add(IngestJob(job_id="busy", org_id="org_1", trigger="schedule",
                     status=JOB_COMPILING, queued_at=NOW, started_at=NOW))
    db.commit()

    assert fail_abandoned_jobs(db, now=int(NOW + LOCK_EXPIRY_SECONDS - 1)) == []
    assert db.get(IngestJob, "busy").status == JOB_COMPILING


def test_a_job_waiting_on_a_pod_is_not_a_worker_s_to_abandon(db):
    from src.models.control_plane import JOB_REGISTERED
    from src.worker.compile import LOCK_EXPIRY_SECONDS, fail_abandoned_jobs

    organization(db)
    db.add(IngestJob(job_id="built", org_id="org_1", trigger="schedule",
                     status=JOB_REGISTERED, queued_at=NOW, started_at=NOW))
    db.commit()

    assert fail_abandoned_jobs(db, now=int(NOW + 10 * LOCK_EXPIRY_SECONDS)) == []


# ==========================================================================
# graph rows in a database of their own
# ==========================================================================


def test_graph_rows_share_the_control_plane_when_nothing_else_is_named(db, monkeypatch):
    import src.worker.phases as phases_module
    from src.worker.compile import run_phases

    seen = {}
    monkeypatch.delenv("GRAPHRAG_GRAPH_STORE_DATABASE_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_DATABASE_URL", raising=False)
    monkeypatch.setattr(phases_module, "run_phases",
                        lambda org, job, session, **kw: seen.update(kw) or "ran")

    assert run_phases("org_1", "job_1", db) == "ran"
    assert "store_db" not in seen


def test_a_named_graph_store_gets_its_own_session(db, monkeypatch, tmp_path):
    import src.worker.phases as phases_module
    from src.worker.compile import run_phases

    seen = {}
    store_path = tmp_path / "graph-store.db"
    monkeypatch.setenv("GRAPHRAG_GRAPH_STORE_DATABASE_URL", f"sqlite+pysqlite:///{store_path}")

    def record(org, job, session, **kw):
        seen["control"] = str(session.get_bind().url)
        seen["store"] = str(kw["store_db"].get_bind().url)
        return "ran"

    monkeypatch.setattr(phases_module, "run_phases", record)

    assert run_phases("org_1", "job_1", db) == "ran"
    assert seen["store"].endswith("graph-store.db")
    assert seen["control"] != seen["store"]


def test_graph_store_url_prefers_the_specific_variable():
    from src.models.database import graph_store_url

    assert graph_store_url({}) is None
    assert graph_store_url({"GRAPHRAG_DATABASE_URL": "postgresql://a/one"}) == "postgresql://a/one"
    assert graph_store_url({
        "GRAPHRAG_DATABASE_URL": "postgresql://a/one",
        "GRAPHRAG_GRAPH_STORE_DATABASE_URL": "postgresql://a/graphs",
    }) == "postgresql://a/graphs"
