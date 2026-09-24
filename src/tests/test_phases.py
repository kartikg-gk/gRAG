"""Tests for the compile itself: ingest, build, upload, register, flip.

The first one is the reason cursors are collected rather than written. If a
run advanced them during the ingest and then failed to register, the next run
would start from the *new* cursor — skipping everything it had already pulled
into the graph store and never shipped. Those rows would exist, be in no
artifact, and nothing would go back for them. No error anywhere.

The last one is the first time the whole chain moves: a compile here, and the
pod agent's pass finds work for the first time since the registry was built.

Storage is the real local backend under ``tmp_path``. The compiler and the
ingest are stood in for where the test is about what surrounds them, and real
where the test is about the chain.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
from sqlmodel import select

from src.artifacts import artifact_key, checksum
from src.models.control_plane import (
    ARTIFACT_ACTIVE,
    ARTIFACT_FAILED,
    ARTIFACT_READY,
    ARTIFACT_SUPERSEDED,
    JOB_COMPILING,
    JOB_COMPLETED,
    JOB_COMPUTING,
    JOB_FETCHING,
    JOB_REGISTERED,
    JOB_UPLOADING,
    GraphArtifact,
    IngestJob,
    Organization,
    Pod,
    PodAssignment,
    Repository,
    create_control_plane_schema,
)
from src.models.database import control_plane_sessions, create_control_plane_engine
from src.models.graph_store import create_graph_store_schema, upsert_nodes
from src.worker.phases import CompileSummary, run_phases

NOW = 1_700_000_000
DIMENSION = 384


def _store_available() -> bool:
    try:
        from src.graphdb import open_context_graph

        with tempfile.TemporaryDirectory() as directory:
            open_context_graph(Path(directory) / "probe").close()
        return True
    except Exception:
        return False


STORE_AVAILABLE = _store_available()
requires_store = pytest.mark.skipif(
    not STORE_AVAILABLE, reason="the graph store's native library is not loadable here"
)


@pytest.fixture()
def engine(tmp_path):
    made = create_control_plane_engine(tmp_path / "control-plane.db")
    create_control_plane_schema(made)
    create_graph_store_schema(made)
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


@pytest.fixture()
def storage(tmp_path, monkeypatch):
    """The real local backend, rooted under this test's directory."""
    root = tmp_path / "storage"

    def put(source, key, **kwargs):
        from src.artifacts import put_artifact

        return put_artifact(source, key, backend="local", root=root)

    return {"root": root, "put": put}


class FakeIngest:
    """Reports a fixed number of items per repository, and a new cursor."""

    def __init__(self, per_repo=None, items=3, cursor="2026-08-05T00:00:00+00:00"):
        self.per_repo = per_repo or {}
        self.items = items
        self.cursor = cursor
        self.calls: list[tuple[str, str | None]] = []
        self.tokens: list[str | None] = []

    def __call__(self, *, org_id, repo_id, repo_name, cursor, db, token=None, **kwargs):
        self.calls.append((repo_id, cursor))
        self.tokens.append(token)
        count = self.per_repo.get(repo_id, self.items)
        from src.worker.ingest import IngestResult

        return IngestResult(
            cursor=self.cursor if count else cursor,
            nodes=count,
            edges=count,
            items=count,
        )


class FakeCompiler:
    """Writes a small file where a real compile would put one."""

    def __init__(self, entities=5, edges=4, contents=b"a compiled graph"):
        self.entities = entities
        self.edges = edges
        self.contents = contents
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, org_id, db, output_path, **kwargs):
        from src.worker.compiler import CompiledArtifact

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.contents)
        self.calls.append((org_id, path))
        return CompiledArtifact(path=path, entities=self.entities, edges=self.edges)


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


def repository(db, repo_id="repo_1", *, org_id="org_1", cursor=None, name=None):
    db.add(
        Repository(
            repo_id=repo_id,
            org_id=org_id,
            provider="github",
            provider_repo_id=repo_id,
            name=name or f"acme/{repo_id}",
            status="active",
            last_synced_cursor=cursor,
            created_at=NOW,
        )
    )
    db.commit()
    return repo_id


def retire(db, job_id):
    """Move a finished job out of the in-flight set.

    Stands in for the reconcile pass, which retires a job when it swaps in
    the artifact that job produced. These tests are about the compile alone,
    so nothing here runs that pass — the chain of the two is tested in
    ``test_after_a_compile_the_pod_agent_finds_work``.
    """
    from src.models.control_plane import JOB_SWAPPED

    job = db.get(IngestJob, job_id)
    job.status = JOB_SWAPPED
    job.finished_at = NOW
    db.commit()


def job_for(db, org_id="org_1", status=JOB_FETCHING):
    row = IngestJob(
        job_id=f"job_{org_id}_{status}",
        org_id=org_id,
        trigger="schedule",
        status=status,
        queued_at=NOW,
        started_at=NOW,
    )
    db.add(row)
    db.commit()
    return row.job_id


def run(db, storage, *, org_id="org_1", job_id=None, ingest=None, compiler=None, **kwargs):
    return run_phases(
        org_id,
        job_id or job_for(db, org_id),
        db,
        build_root=storage["root"].parent / "builds",
        ingest=ingest or FakeIngest(),
        compile_to=compiler or FakeCompiler(),
        upload=storage["put"],
        now=NOW,
        **kwargs,
    )


# ==========================================================================
# the one that decides whether a failure loses work
# ==========================================================================


def test_cursors_do_not_advance_when_registration_fails(db, storage, monkeypatch):
    """A failed publish must leave the next run reading the same delta.

    Advanced during the ingest, a cursor would have moved past work that was
    pulled into the graph store and never shipped in any artifact. Nothing
    would go back for it and nothing would report it — the rows would simply
    never appear in anything served.
    """
    organization(db)
    repository(db, "repo_1", cursor="2026-08-01T00:00:00+00:00")
    repository(db, "repo_2", cursor="2026-08-02T00:00:00+00:00")

    real_commit = type(db).commit
    calls = {"count": 0}

    def commit_until_the_last(self):
        calls["count"] += 1
        # Every status update commits; the registration is the final one.
        if calls["count"] >= 5:
            raise RuntimeError("the registration would not commit")
        return real_commit(self)

    monkeypatch.setattr(type(db), "commit", commit_until_the_last)

    with pytest.raises(RuntimeError):
        run(db, storage)

    monkeypatch.undo()
    db.rollback()

    stored = {row.repo_id: row.last_synced_cursor for row in db.exec(select(Repository)).all()}

    assert stored == {
        "repo_1": "2026-08-01T00:00:00+00:00",
        "repo_2": "2026-08-02T00:00:00+00:00",
    }
    # And nothing was published either.
    assert db.exec(select(GraphArtifact)).all() == []
    assert db.get(Organization, "org_1").desired_artifact_id is None


def test_cursors_advance_for_every_repository_on_success(db, storage):
    """All of them, and only with the registration."""
    organization(db)
    repository(db, "repo_1", cursor="2026-08-01T00:00:00+00:00")
    repository(db, "repo_2", cursor=None)

    ingest = FakeIngest(cursor="2026-08-09T12:00:00+00:00")
    run(db, storage, ingest=ingest)

    stored = {row.repo_id: row for row in db.exec(select(Repository)).all()}

    assert stored["repo_1"].last_synced_cursor == "2026-08-09T12:00:00+00:00"
    assert stored["repo_2"].last_synced_cursor == "2026-08-09T12:00:00+00:00"
    assert stored["repo_1"].last_synced_at == NOW
    # Each repository was read from its own stored cursor, not a shared one.
    assert ingest.calls == [
        ("repo_1", "2026-08-01T00:00:00+00:00"),
        ("repo_2", None),
    ]


def test_the_registration_and_the_cursors_are_one_transaction(db, storage):
    """Both land together. Neither is visible without the other."""
    organization(db)
    repository(db, "repo_1", cursor="2026-08-01T00:00:00+00:00")

    summary = run(db, storage)

    with control_plane_sessions(db.get_bind())() as reading:
        artifact = reading.exec(select(GraphArtifact)).one()
        repo = reading.exec(select(Repository)).one()
        organisation = reading.get(Organization, "org_1")

    assert artifact.artifact_id == summary.artifact_id
    assert organisation.desired_artifact_id == artifact.artifact_id
    assert repo.last_synced_cursor == "2026-08-05T00:00:00+00:00"


# ==========================================================================
# what a run publishes
# ==========================================================================


def test_a_full_run_registers_an_artifact_and_points_the_organisation_at_it(
    db, storage
):
    organization(db)
    repository(db, "repo_1")

    summary = run(db, storage)

    artifact = db.exec(select(GraphArtifact)).one()

    assert artifact.org_id == "org_1"
    assert artifact.version == 1
    assert artifact.status == ARTIFACT_READY
    assert artifact.entity_count == 5
    assert artifact.size_bytes == len(b"a compiled graph")
    assert artifact.built_by_job_id is not None
    job = db.get(IngestJob, artifact.built_by_job_id)
    assert job.produced_artifact_id == artifact.artifact_id
    assert job.cursor_to == "2026-08-05T00:00:00+00:00"
    assert db.get(Organization, "org_1").desired_artifact_id == artifact.artifact_id
    assert db.get(Organization, "org_1").updated_at == NOW

    assert summary == CompileSummary(
        org_id="org_1",
        artifact_id=artifact.artifact_id,
        version=1,
        entities=5,
        edges=4,
        items=3,
        s3_uri=artifact.s3_uri,
        skipped=False,
    )


def test_build_path_and_repository_token_match_the_part4_contract(
    db, storage, monkeypatch
):
    from cryptography.fernet import Fernet

    monkeypatch.setenv(
        "GRAPHRAG_ENCRYPTION_MASTER_KEY", Fernet.generate_key().decode()
    )
    organization(db)
    repository(db, "repo_1")
    stored = db.get(Repository, "repo_1")
    stored.set_github_token("github-token")
    db.commit()

    ingest = FakeIngest()
    compiler = FakeCompiler()
    run(db, storage, ingest=ingest, compiler=compiler)

    assert ingest.tokens == ["github-token"]
    assert compiler.calls[0][1].name == "v1.lbug"


def test_the_uploaded_file_matches_the_recorded_checksum_and_size(db, storage):
    """What the pod agent will verify before it swaps anything in."""
    organization(db)
    repository(db, "repo_1")

    run(db, storage)

    artifact = db.exec(select(GraphArtifact)).one()
    uploaded = storage["root"] / artifact_key("org_1", "1")

    assert uploaded.is_file()
    assert artifact.checksum_sha256 == checksum(uploaded)
    assert artifact.size_bytes == uploaded.stat().st_size
    assert artifact.s3_uri.startswith("local://")


def test_the_file_is_uploaded_before_any_row_describes_it(db, storage, monkeypatch):
    """A crash in between leaves a file nothing points at, which corrects
    itself. The reverse leaves a row promising a file that is not there."""
    order: list[str] = []

    def watched_upload(source, key, **kwargs):
        order.append("upload")
        return storage["put"](source, key, **kwargs)

    organization(db)
    repository(db, "repo_1")

    real_add = type(db).add

    def watched_add(self, instance, **kwargs):
        if isinstance(instance, GraphArtifact):
            order.append("row")
        return real_add(self, instance, **kwargs)

    monkeypatch.setattr(type(db), "add", watched_add)

    run_phases(
        "org_1",
        job_for(db, "org_1"),
        db,
        build_root=storage["root"].parent / "builds",
        ingest=FakeIngest(),
        compile_to=FakeCompiler(),
        upload=watched_upload,
        now=NOW,
    )

    assert order == ["upload", "row"]


# ==========================================================================
# superseding, and versions
# ==========================================================================


def test_the_previous_artifact_is_superseded(db, storage):
    organization(db)
    repository(db, "repo_1")

    first_job = job_for(db, "org_1")
    first = run(db, storage, job_id=first_job)
    retire(db, first_job)

    second = run(db, storage, job_id=job_for(db, "org_1", status=JOB_COMPUTING))

    artifacts = {row.artifact_id: row for row in db.exec(select(GraphArtifact)).all()}

    assert artifacts[first.artifact_id].status == ARTIFACT_SUPERSEDED
    assert artifacts[second.artifact_id].status == ARTIFACT_READY
    assert db.get(Organization, "org_1").desired_artifact_id == second.artifact_id


def test_an_already_superseded_artifact_is_left_alone(db, storage):
    """It was retired for its own reason, and rewriting it would lose which
    artifact was actually in service when."""
    organization(db)
    repository(db, "repo_1")

    db.add(
        GraphArtifact(
            artifact_id="art_old",
            org_id="org_1",
            version=1,
            s3_uri="local://artifacts/org_1/v1.lbug",
            status=ARTIFACT_SUPERSEDED,
            created_at=NOW,
        )
    )
    db.commit()
    organisation = db.get(Organization, "org_1")
    organisation.desired_artifact_id = "art_old"
    db.commit()

    run(db, storage)

    assert db.get(GraphArtifact, "art_old").status == ARTIFACT_SUPERSEDED


def test_a_failed_artifact_is_not_marked_superseded(db, storage):
    organization(db)
    repository(db, "repo_1")

    db.add(
        GraphArtifact(
            artifact_id="art_broken",
            org_id="org_1",
            version=1,
            s3_uri="local://artifacts/org_1/v1.lbug",
            status=ARTIFACT_FAILED,
            created_at=NOW,
        )
    )
    db.commit()
    db.get(Organization, "org_1").desired_artifact_id = "art_broken"
    db.commit()

    run(db, storage)

    assert db.get(GraphArtifact, "art_broken").status == ARTIFACT_FAILED


def test_an_active_artifact_is_superseded_too(db, storage):
    """Ready and active are both "in service" as far as replacing goes."""
    organization(db)
    repository(db, "repo_1")

    db.add(
        GraphArtifact(
            artifact_id="art_serving",
            org_id="org_1",
            version=1,
            s3_uri="local://artifacts/org_1/v1.lbug",
            status=ARTIFACT_ACTIVE,
            created_at=NOW,
        )
    )
    db.commit()
    db.get(Organization, "org_1").desired_artifact_id = "art_serving"
    db.commit()

    run(db, storage)

    assert db.get(GraphArtifact, "art_serving").status == ARTIFACT_SUPERSEDED


def test_the_version_increments_across_runs(db, storage):
    organization(db)
    repository(db, "repo_1")

    versions = []
    for index in range(3):
        job_id = job_for(db, "org_1", status=f"fetching-{index}")
        versions.append(run(db, storage, job_id=job_id).version)
        retire(db, job_id)

    assert versions == [1, 2, 3]


def test_two_compiles_racing_on_a_version_are_settled_by_the_database(db, storage):
    """The arithmetic is the common path; the constraint is the guarantee.

    Both computed version 1 from the same empty table, and the second insert
    is refused rather than producing two artifacts claiming to be the same
    version of the same tenant.
    """
    from sqlalchemy.exc import IntegrityError

    organization(db)
    repository(db, "repo_1")

    run(db, storage)

    db.add(
        GraphArtifact(
            artifact_id="art_racing",
            org_id="org_1",
            version=1,
            s3_uri="local://artifacts/org_1/v1.lbug",
            status=ARTIFACT_READY,
            created_at=NOW,
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


# ==========================================================================
# statuses
# ==========================================================================


def test_each_status_is_set_before_the_phase_it_names(db, storage, monkeypatch):
    """Set afterwards, a long compile would read as still fetching.

    The status answers what the job is doing *now* for whoever is reading the
    database while it runs.
    """
    import src.worker.compile as compile_module

    order: list[str] = []
    # Captured before the patch below, or the wrapper would call itself.
    real_status = compile_module.set_job_status

    def watched_status(session, job_id, status):
        order.append(f"status:{status}")
        return real_status(session, job_id, status)

    class WatchingIngest(FakeIngest):
        def __call__(self, **kwargs):
            order.append("ingest")
            return super().__call__(**kwargs)

    class WatchingCompiler(FakeCompiler):
        def __call__(self, org_id, db, output_path, **kwargs):
            order.append("compile")
            return super().__call__(org_id, db, output_path, **kwargs)

    def watched_upload(source, key, **kwargs):
        order.append("upload")
        return storage["put"](source, key, **kwargs)

    organization(db)
    repository(db, "repo_1")

    monkeypatch.setattr(compile_module, "set_job_status", watched_status)

    run_phases(
        "org_1",
        job_for(db, "org_1"),
        db,
        build_root=storage["root"].parent / "builds",
        ingest=WatchingIngest(),
        compile_to=WatchingCompiler(),
        upload=watched_upload,
        now=NOW,
    )

    assert order == [
        f"status:{JOB_COMPUTING}",
        "ingest",
        f"status:{JOB_COMPILING}",
        "compile",
        f"status:{JOB_UPLOADING}",
        "upload",
    ]


def test_the_job_ends_registered(db, storage):
    """The last status this path sets. Anything later is somebody else's."""
    organization(db)
    repository(db, "repo_1")
    job_id = job_for(db, "org_1")

    run(db, storage, job_id=job_id)

    assert db.get(IngestJob, job_id).status == JOB_REGISTERED


# ==========================================================================
# nothing changed
# ==========================================================================


def test_no_changes_anywhere_compiles_nothing(db, storage, caplog):
    """The decision: short-circuit rather than ship an identical artifact.

    Compiling anyway would produce a byte-identical store under a new version
    and hand it to every pod holding this tenant, each of which would download
    it and swap it in for no difference at all. The saving is a branch; the
    cost avoided is fleet-wide.
    """
    organization(db)
    repository(db, "repo_1", cursor="2026-08-01T00:00:00+00:00")
    repository(db, "repo_2", cursor="2026-08-02T00:00:00+00:00")

    compiler = FakeCompiler()
    job_id = job_for(db, "org_1")

    with caplog.at_level(logging.INFO):
        summary = run(
            db, storage, job_id=job_id, ingest=FakeIngest(items=0), compiler=compiler
        )

    assert summary.skipped is True
    assert summary.artifact_id is None
    assert summary.version is None
    assert summary.items == 0
    # Nothing was built and nothing was published.
    assert compiler.calls == []
    assert db.exec(select(GraphArtifact)).all() == []
    assert db.get(Organization, "org_1").desired_artifact_id is None
    assert "nothing to compile" in caplog.text


def test_a_run_with_nothing_to_do_ends_complete_not_failed(db, storage):
    """A job that legitimately had nothing to do is finished, not broken.

    Terminal on purpose: it leaves the in-flight set, so this organisation is
    free for the next compile rather than blocked by a job that will never
    move again.
    """
    organization(db)
    repository(db, "repo_1")
    job_id = job_for(db, "org_1")

    run(db, storage, job_id=job_id, ingest=FakeIngest(items=0))

    from src.models.control_plane import JOB_IN_FLIGHT

    job = db.get(IngestJob, job_id)
    assert job.status == JOB_COMPLETED
    assert job.status not in JOB_IN_FLIGHT
    assert job.error is None


def test_one_busy_repository_among_quiet_ones_still_compiles(db, storage):
    """The short-circuit is on the total, not on any single repository."""
    organization(db)
    repository(db, "repo_quiet")
    repository(db, "repo_busy")

    summary = run(
        db, storage, ingest=FakeIngest(per_repo={"repo_quiet": 0, "repo_busy": 4})
    )

    assert summary.skipped is False
    assert summary.items == 4
    assert summary.version == 1


def test_an_organisation_with_no_repositories_compiles_nothing(db, storage):
    organization(db)

    summary = run(db, storage)

    assert summary.skipped is True
    assert db.exec(select(GraphArtifact)).all() == []


# ==========================================================================
# the whole chain, for the first time
# ==========================================================================


@requires_store
def test_after_a_compile_the_pod_agent_finds_work(db, storage, tmp_path):
    """The first time everything moves at once.

    A compile publishes an artifact and points the organisation at it; the
    pass that has been correctly doing nothing since the registry was built
    now finds a difference between intent and reality, downloads the file,
    verifies its checksum, and swaps it in.

    Every piece here is the real one: the real compiler over real graph-store
    rows, the real storage backend, the real reconcile pass. Only the source
    fetch is stood in for, because this is not a test of the network.
    """
    from src.models.control_plane import JOB_IN_FLIGHT
    from src.reconcile import reconcile
    from src.registry import GraphRegistry
    from src.worker.compiler import compile_artifact

    organization(db)
    repository(db, "repo_1")

    # Real rows in the graph store, as C2's ingest would have left them.
    upsert_nodes(
        db,
        [
            {
                "org_id": "org_1",
                "node_id": "repo_1:pullrequest:41",
                "repo_id": "repo_1",
                "label": "PullRequest",
                "name": "PullRequest #41",
                "properties": {"title": "the auth change"},
                "embedding": [0.1] * DIMENSION,
                "created_at": NOW,
                "updated_at": NOW,
            }
        ],
    )
    db.commit()

    # This pod is assigned the tenant and holds nothing yet, which is the
    # state every pod has been in until now.
    db.add(Pod(pod_id="pod_here", address="127.0.0.1", created_at=NOW))
    db.commit()
    db.add(
        PodAssignment(
            pod_id="pod_here", org_id="org_1", artifact_id=None, assigned_at=NOW
        )
    )
    db.commit()

    summary_job = job_for(db, "org_1")
    summary = run(db, storage, job_id=summary_job, compiler=compile_artifact)

    assert summary.artifact_id is not None
    assert summary.entities == 1

    opened: list[str] = []

    class OpenedStore:
        def __init__(self, path):
            opened.append(path)
            self.path = path

        def close(self):
            pass

    registry = GraphRegistry()
    registry.set_loader(OpenedStore)

    swaps = reconcile(
        engine=db.get_bind(),
        pod_id="pod_here",
        registry=registry,
        cache_root=tmp_path / "cache",
        artifact_root=storage["root"],
    )

    assert len(swaps) == 1
    assert swaps[0].org_id == "org_1"
    assert swaps[0].moved_from is None
    assert swaps[0].moved_to == summary.artifact_id
    assert swaps[0].version == 1

    # It downloaded, verified and opened the real file this compile produced.
    assert len(opened) == 1
    assert Path(opened[0]).is_file()
    assert checksum(opened[0]) == db.get(GraphArtifact, summary.artifact_id).checksum_sha256

    with control_plane_sessions(db.get_bind())() as reading:
        assignment = reading.get(PodAssignment, ("pod_here", "org_1"))
        assert assignment.artifact_id == summary.artifact_id
        assert assignment.load_status == "ready"
        # The artifact is in service now, not merely built.
        assert reading.get(GraphArtifact, summary.artifact_id).status == ARTIFACT_ACTIVE
        # And the build that produced it is over, which is what lets this
        # tenant compile again. Before the pass retired it, this job stayed
        # in flight and the next compile was refused forever.
        job = reading.get(IngestJob, summary_job)
        assert job.status == JOB_COMPLETED
        assert job.status not in JOB_IN_FLIGHT

    # And a second pass finds nothing, because intent and reality agree.
    assert reconcile(
        engine=db.get_bind(),
        pod_id="pod_here",
        registry=registry,
        cache_root=tmp_path / "cache",
        artifact_root=storage["root"],
    ) == []


def test_a_registered_job_stays_in_flight_until_something_swaps_it(db, storage):
    """This path stops at ``registered``, and that is inside the in-flight set.

    Deliberate: a build is not over when it is registered, it is over when a
    pod is serving it. Until then the tenant holds its job and the
    one-at-a-time rule refuses a second build — which is right, because the
    first one has not been picked up yet.

    What closes it is the reconcile pass, on the swap. Nothing here does, so
    this asserts the block and then the release once something retires it.
    """
    from sqlalchemy.exc import IntegrityError

    from src.models.control_plane import JOB_IN_FLIGHT

    organization(db)
    repository(db, "repo_1")
    first = job_for(db, "org_1")

    run(db, storage, job_id=first)

    assert db.get(IngestJob, first).status == JOB_REGISTERED
    assert JOB_REGISTERED in JOB_IN_FLIGHT

    db.add(
        IngestJob(
            job_id="job_next",
            org_id="org_1",
            trigger="schedule",
            status=JOB_FETCHING,
            queued_at=NOW,
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    # Once a swap has retired it, the next compile can open its job.
    retire(db, first)
    db.add(
        IngestJob(
            job_id="job_next",
            org_id="org_1",
            trigger="schedule",
            status=JOB_FETCHING,
            queued_at=NOW,
        )
    )
    db.commit()


def test_a_run_with_nothing_to_do_records_when_it_finished(db, storage):
    organization(db)
    repository(db, "repo_1", cursor="2026-08-01T00:00:00+00:00")
    job_id = job_for(db, "org_1")

    run(db, storage, job_id=job_id, ingest=FakeIngest(items=0))

    job = db.get(IngestJob, job_id)
    assert job.status == JOB_COMPLETED
    assert job.finished_at is not None
    assert job.error is None
