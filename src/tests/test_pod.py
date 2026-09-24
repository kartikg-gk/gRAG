"""Tests for what a process does at boot, and what it does on an interval.

The two worth reading first are the failure-posture ones — boot swallowing a
per-tenant failure, and the loop swallowing a whole-pass failure. Those two
assertions are the entire reason these are separate pieces from the pass they
call.

The control plane is real, the artifacts are real files moved by the real
storage module, and the registry is the real one. Only the thing that opens a
graph is a stand-in, because opening a real store would be a test of the store
driver.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from src.artifacts import artifact_key, checksum, put_artifact
from src.artifacts import pod_cache_path
from src.models.control_plane import (
    ARTIFACT_ACTIVE,
    LOAD_PULLING,
    LOAD_READY,
    POD_BOOTING,
    POD_READY,
    GraphArtifact,
    Organization,
    Pod,
    PodAssignment,
    create_control_plane_schema,
)
from src.models.database import control_plane_sessions, create_control_plane_engine
from src.pod import Hydrated, boot, hydrate, poll, register_pod
from src.registry import GraphRegistry

POD = "pod_here"
NOW = 1_700_000_000


class FakeGraph:
    def __init__(self, path: str):
        self.path = path
        self.closed = False

    def close(self):
        self.closed = True


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
def registry():
    made = GraphRegistry()
    made.set_loader(FakeGraph)
    try:
        yield made
    finally:
        made.close_all()


@pytest.fixture()
def roots(tmp_path):
    return {"artifact_root": tmp_path / "artifacts", "cache_root": tmp_path / "cache"}


def seed_serving(
    sessions,
    tmp_path,
    roots,
    *,
    org_id="org_1",
    version=1,
    pod_id=POD,
    load_status=LOAD_READY,
    store_the_file=True,
):
    """A tenant this pod was serving when it stopped: ready, with an artifact."""
    source = tmp_path / f"{org_id}-{version}.built"
    source.write_bytes(f"graph for {org_id}".encode())
    uri = put_artifact(
        source,
        artifact_key(org_id, str(version)),
        backend="local",
        root=roots["artifact_root"],
    )
    if not store_the_file:
        (roots["artifact_root"] / artifact_key(org_id, str(version))).unlink()

    artifact_id = f"art_{org_id}_{version}"

    with sessions() as db:
        if db.get(Organization, org_id) is None:
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
        if db.get(Pod, pod_id) is None:
            db.add(Pod(pod_id=pod_id, address="10.0.0.1:8000", created_at=NOW))
        db.commit()

        db.add(
            GraphArtifact(
                artifact_id=artifact_id,
                org_id=org_id,
                version=version,
                s3_uri=uri,
                checksum_sha256=checksum(source),
                status=ARTIFACT_ACTIVE,
                created_at=NOW,
            )
        )
        db.commit()

        db.add(
            PodAssignment(
                pod_id=pod_id,
                org_id=org_id,
                artifact_id=artifact_id,
                load_status=load_status,
                assigned_at=NOW,
                confirmed_at=NOW,
            )
        )
        db.commit()

    return artifact_id


# ==========================================================================
# the failure posture, which is the point of this module
# ==========================================================================


def test_one_tenant_failing_does_not_stop_the_others_hydrating(
    engine, sessions, registry, tmp_path, roots, caplog
):
    """Storage raising on one tenant is logged and skipped. Nothing propagates.

    This is the reverse of the reconcile pass on purpose: a pass that runs once
    at startup cannot let one bad row stop a process from serving everything
    else it holds.
    """
    seed_serving(sessions, tmp_path, roots, org_id="org_a")
    seed_serving(sessions, tmp_path, roots, org_id="org_b", store_the_file=False)
    seed_serving(sessions, tmp_path, roots, org_id="org_c")

    with caplog.at_level(logging.ERROR):
        hydrated = boot(engine=engine, pod_id=POD, registry=registry, **roots)

    assert [entry.org_id for entry in hydrated] == ["org_a", "org_c"]
    assert "org_a" in registry
    assert "org_c" in registry
    assert "org_b" not in registry
    assert "could not hydrate org_b" in caplog.text


def test_the_loader_failing_on_one_tenant_is_also_survived(
    engine, sessions, registry, tmp_path, roots
):
    """A store that will not open is a skip here, not a failed boot."""
    seed_serving(sessions, tmp_path, roots, org_id="org_a")
    seed_serving(sessions, tmp_path, roots, org_id="org_b")

    def loader(path: str):
        if "org_a" in path:
            raise OSError("this store will not open")
        return FakeGraph(path)

    registry.set_loader(loader)

    hydrated = boot(engine=engine, pod_id=POD, registry=registry, **roots)

    assert [entry.org_id for entry in hydrated] == ["org_b"]
    assert "org_a" not in registry


def test_an_unreadable_assignments_query_returns_nothing_and_does_not_raise(
    engine, sessions, registry, roots, caplog
):
    """The query itself is covered, not only the loop over its results."""

    class BrokenSession:
        def exec(self, *args, **kwargs):
            raise RuntimeError("the control plane is unreachable")

    with caplog.at_level(logging.ERROR):
        hydrated = hydrate(
            BrokenSession(), pod_id=POD, registry=registry, **roots
        )

    assert hydrated == []
    assert "could not read this pod's assignments" in caplog.text


def test_registration_failing_still_hydrates(
    engine, sessions, registry, tmp_path, roots, monkeypatch, caplog
):
    """Unregistered but serving beats registered and serving nothing."""
    seed_serving(sessions, tmp_path, roots, org_id="org_a")

    import src.pod as pod_module

    def refuse(*args, **kwargs):
        raise RuntimeError("the pods table would not take the row")

    monkeypatch.setattr(pod_module, "register_pod", refuse)

    with caplog.at_level(logging.ERROR):
        hydrated = boot(engine=engine, pod_id=POD, registry=registry, **roots)

    assert [entry.org_id for entry in hydrated] == ["org_a"]
    assert "could not register this pod" in caplog.text


# ==========================================================================
# registration
# ==========================================================================


def test_registering_a_pod_that_does_not_exist_inserts_it_ready(engine, sessions):
    with sessions() as db:
        register_pod(db, pod_id="pod_new", address="10.0.0.7:8000")

    with sessions() as db:
        row = db.get(Pod, "pod_new")

    assert row is not None
    assert row.status == POD_READY
    assert row.address == "10.0.0.7:8000"
    assert row.last_heartbeat_at is not None
    assert row.created_at is not None


def test_registering_a_pod_that_exists_refreshes_rather_than_duplicating(
    engine, sessions
):
    """Boot runs this every time and must not care that the row survived."""
    with sessions() as db:
        db.add(
            Pod(
                pod_id="pod_old",
                address="10.0.0.7:8000",
                status=POD_BOOTING,
                last_heartbeat_at=NOW,
                created_at=NOW,
            )
        )
        db.commit()

    with sessions() as db:
        register_pod(db, pod_id="pod_old")

    with sessions() as db:
        rows = db.exec(__import__("sqlmodel").select(Pod)).all()
        row = db.get(Pod, "pod_old")

    assert len(rows) == 1
    assert row.status == POD_READY
    assert row.last_heartbeat_at > NOW
    # Told no new address, so the configured one is left alone.
    assert row.address == "10.0.0.7:8000"
    assert row.created_at == NOW


def test_a_new_address_replaces_the_recorded_one(engine, sessions):
    with sessions() as db:
        register_pod(db, pod_id="pod_moved", address="10.0.0.7:8000")
    with sessions() as db:
        register_pod(db, pod_id="pod_moved", address="10.0.0.8:9000")

    with sessions() as db:
        assert db.get(Pod, "pod_moved").address == "10.0.0.8:9000"


# ==========================================================================
# hydration
# ==========================================================================


def test_a_ready_assignment_is_reopened_and_pointed_at(
    engine, sessions, registry, tmp_path, roots
):
    seed_serving(sessions, tmp_path, roots, org_id="org_a", version=4)

    hydrated = boot(engine=engine, pod_id=POD, registry=registry, **roots)

    assert hydrated == [Hydrated(org_id="org_a", version=4)]

    cached = pod_cache_path(POD, "org_a", "4", root=roots["cache_root"])
    assert cached.is_file()
    assert registry.entry("org_a").path == str(cached)
    assert registry.entry("org_a").version == "4"


def test_an_assignment_that_is_not_ready_is_left_alone(
    engine, sessions, registry, tmp_path, roots
):
    """Only what this pod was actually serving comes back.

    A tenant still pulling was never answering queries, and re-opening it here
    would claim it was.
    """
    seed_serving(sessions, tmp_path, roots, org_id="org_a", load_status=LOAD_PULLING)

    assert boot(engine=engine, pod_id=POD, registry=registry, **roots) == []
    assert "org_a" not in registry


def test_a_ready_assignment_with_nothing_recorded_is_skipped_quietly(
    engine, sessions, registry, tmp_path, roots
):
    """Ready with no artifact is nothing to re-open, and nothing wrong."""
    seed_serving(sessions, tmp_path, roots, org_id="org_a")
    with sessions() as db:
        db.get(PodAssignment, (POD, "org_a")).artifact_id = None
        db.commit()

    assert boot(engine=engine, pod_id=POD, registry=registry, **roots) == []
    assert "org_a" not in registry


def test_a_row_whose_artifact_is_missing_is_skipped_and_the_others_hydrate(
    engine, sessions, registry, tmp_path, roots, caplog
):
    """An assignment naming an artifact the control plane does not have.

    **This state cannot be reached through the database.** The reference from
    an assignment to an artifact is declared and enforced, so the row cannot
    name something that was deleted — the delete is what fails. The branch is
    still worth having and worth covering: it is what stands between a hand-
    edited database, or a restore with constraints off, and a boot that dies
    on the first tenant. So the missing row is produced at the session, which
    is the only place it can be produced at all.
    """
    seed_serving(sessions, tmp_path, roots, org_id="org_a")
    seed_serving(sessions, tmp_path, roots, org_id="org_b")

    with sessions() as db:
        assignments = db.exec(
            __import__("sqlmodel").select(PodAssignment)
        ).all()
        real = {
            assignment.org_id: db.get(GraphArtifact, assignment.artifact_id)
            for assignment in assignments
        }

    class SessionWithOneArtifactGone:
        """The real rows, except that org_b's artifact cannot be found."""

        def exec(self, *args, **kwargs):
            class Rows:
                @staticmethod
                def all():
                    return list(assignments)

            return Rows()

        def get(self, model, key):
            if key == "art_org_b_1":
                return None
            return real["org_a"] if key == "art_org_a_1" else None

    with caplog.at_level(logging.WARNING):
        hydrated = hydrate(
            SessionWithOneArtifactGone(), pod_id=POD, registry=registry, **roots
        )

    assert [entry.org_id for entry in hydrated] == ["org_a"]
    assert "org_a" in registry
    assert "org_b" not in registry
    assert "which the control plane does not have" in caplog.text


def test_the_database_will_not_let_an_assignment_outlive_its_artifact(
    engine, sessions, registry, tmp_path, roots
):
    """Which is why the skip above had to be built at the session.

    Recorded as a test rather than a comment: if the reference is ever
    dropped, this fails and the branch above stops being theoretical.
    """
    from sqlalchemy.exc import IntegrityError

    seed_serving(sessions, tmp_path, roots, org_id="org_a")

    with sessions() as db:
        artifact = db.get(GraphArtifact, "art_org_a_1")
        db.delete(artifact)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()


def test_another_pods_assignments_are_not_hydrated_here(
    engine, sessions, registry, tmp_path, roots
):
    seed_serving(sessions, tmp_path, roots, org_id="org_mine")
    seed_serving(sessions, tmp_path, roots, org_id="org_theirs", pod_id="pod_elsewhere")

    hydrated = boot(engine=engine, pod_id=POD, registry=registry, **roots)

    assert [entry.org_id for entry in hydrated] == ["org_mine"]
    assert "org_theirs" not in registry


def test_boot_registers_before_it_hydrates(engine, sessions, registry, tmp_path, roots):
    """A process is visible to the fleet before it starts doing slow work."""
    seed_serving(sessions, tmp_path, roots, org_id="org_a")

    boot(engine=engine, pod_id=POD, registry=registry, **roots)

    with sessions() as db:
        row = db.get(Pod, POD)

    assert row.status == POD_READY
    assert row.last_heartbeat_at is not None


# ==========================================================================
# the loop
# ==========================================================================


def test_the_loop_reconciles_until_it_is_stopped(registry, monkeypatch):
    import src.pod as pod_module

    passes = []

    async def drive():
        stop = asyncio.Event()

        def fake_pass(**kwargs):
            passes.append(kwargs["pod_id"])
            if len(passes) >= 3:
                stop.set()
            return []

        monkeypatch.setattr(pod_module, "reconcile", fake_pass)
        await poll(registry=registry, stop=stop, pod_id=POD, interval=0.01)

    asyncio.run(drive())

    assert len(passes) >= 3
    assert set(passes) == {POD}


def test_a_pass_that_raises_is_caught_and_the_next_tick_still_runs(
    registry, monkeypatch, caplog
):
    """The pass is deliberately loud. This is the thing that listens.

    Without this, the reconcile pass's un-defensiveness would take the process
    down the first time storage hiccuped.
    """
    import src.pod as pod_module

    attempts = []

    async def drive():
        stop = asyncio.Event()

        def fake_pass(**kwargs):
            attempts.append(len(attempts))
            if len(attempts) == 1:
                raise RuntimeError("storage went away")
            if len(attempts) >= 3:
                stop.set()
            return []

        monkeypatch.setattr(pod_module, "reconcile", fake_pass)
        await poll(registry=registry, stop=stop, pod_id=POD, interval=0.01)

    with caplog.at_level(logging.ERROR):
        asyncio.run(drive())

    assert len(attempts) >= 3
    assert "reconcile pass failed" in caplog.text
    assert "storage went away" in caplog.text


def test_a_stop_mid_interval_takes_effect_without_waiting_it_out(
    registry, monkeypatch
):
    """Timed, because a test that only checks it stopped passes with a sleep.

    The interval is five seconds and the stop arrives after a tenth of one. If
    the wait were a plain sleep the loop would sit there for the remaining
    four and nine tenths, so the elapsed time is the assertion.
    """
    import src.pod as pod_module

    monkeypatch.setattr(pod_module, "reconcile", lambda **kwargs: [])

    async def drive():
        stop = asyncio.Event()

        async def stop_shortly():
            await asyncio.sleep(0.1)
            stop.set()

        await asyncio.gather(
            poll(registry=registry, stop=stop, pod_id=POD, interval=5.0),
            stop_shortly(),
        )

    started = time.monotonic()
    asyncio.run(drive())
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, elapsed


def test_the_loop_does_not_run_a_pass_when_it_is_already_stopped(
    registry, monkeypatch
):
    import src.pod as pod_module

    passes = []
    monkeypatch.setattr(pod_module, "reconcile", lambda **kwargs: passes.append(1))

    async def drive():
        stop = asyncio.Event()
        stop.set()
        await poll(registry=registry, stop=stop, pod_id=POD, interval=0.01)

    asyncio.run(drive())

    assert passes == []


def test_the_pass_runs_off_the_event_loop(registry, monkeypatch):
    """A blocking pass must not stall everything else in the process.

    Asserted by the thread it runs on: the pass opens a session and copies
    files, and on the event loop's thread that time is time no request is
    served.
    """
    import threading

    import src.pod as pod_module

    threads = []
    here = threading.current_thread().name

    async def drive():
        stop = asyncio.Event()

        def fake_pass(**kwargs):
            threads.append(threading.current_thread().name)
            stop.set()
            return []

        monkeypatch.setattr(pod_module, "reconcile", fake_pass)
        await poll(registry=registry, stop=stop, pod_id=POD, interval=0.01)

    asyncio.run(drive())

    assert threads and threads[0] != here


def test_the_loop_says_when_it_starts_and_when_it_stops(
    registry, monkeypatch, caplog
):
    """Agent activity is visible in a log without instrumenting anything."""
    import src.pod as pod_module

    monkeypatch.setattr(pod_module, "reconcile", lambda **kwargs: [])

    async def drive():
        stop = asyncio.Event()
        stop.set()
        await poll(registry=registry, stop=stop, pod_id=POD, interval=0.01)

    with caplog.at_level(logging.INFO):
        asyncio.run(drive())

    assert f"{POD}: reconcile loop starting" in caplog.text
    assert f"{POD}: reconcile loop stopped" in caplog.text
