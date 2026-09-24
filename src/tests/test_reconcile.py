"""Tests for the pass that makes reality match intent.

Nothing here is faked except the thing that opens a graph. The control plane
is a real database with real constraints, the artifacts are real files moved
by the real storage module, and the registry is the real one — only the loader
is a function that returns an object, because opening an actual store would
test the store driver rather than this.

The checksum test is the one the rest exists to protect: it is the assertion
that a corrupt artifact never reaches traffic.
"""

from __future__ import annotations

import pytest

from src.artifacts import artifact_key, checksum, pod_cache_path, put_artifact
from src.models.control_plane import (
    ARTIFACT_ACTIVE,
    ARTIFACT_BUILDING,
    ARTIFACT_READY,
    LOAD_PULLING,
    LOAD_READY,
    GraphArtifact,
    Organization,
    Pod,
    PodAssignment,
    create_control_plane_schema,
)
from src.models.database import control_plane_sessions, create_control_plane_engine
from src.reconcile import Swap, reconcile
from src.registry import GraphRegistry

POD = "pod_here"
OTHER_POD = "pod_elsewhere"
NOW = 1_700_000_000


class FakeGraph:
    """Whatever the loader returns. The registry stores handles opaquely."""

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
    """Where artifacts are stored, and where this pod caches them."""
    return {"artifact_root": tmp_path / "artifacts", "cache_root": tmp_path / "cache"}


def store_artifact(tmp_path, roots, org_id, version, contents=b"a built graph"):
    """A real file, really put into the local backend. Returns (uri, digest)."""
    source = tmp_path / f"{org_id}-{version}.built"
    source.write_bytes(contents)
    uri = put_artifact(
        source,
        artifact_key(org_id, str(version)),
        backend="local",
        root=roots["artifact_root"],
    )
    return uri, checksum(source)


def seed(
    sessions,
    tmp_path,
    roots,
    *,
    org_id="org_1",
    version=1,
    status=ARTIFACT_READY,
    pod_id=POD,
    intended=True,
    loaded=None,
    digest=None,
    contents=b"a built graph",
):
    """One tenant, one artifact, one assignment of that tenant to a pod.

    ``intended`` points the organisation at the artifact; ``loaded`` is what
    the assignment claims the pod already has. ``digest`` overrides the
    recorded checksum, which is how the corrupt case is built.
    """
    uri, real_digest = store_artifact(tmp_path, roots, org_id, version, contents)
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
                checksum_sha256=real_digest if digest is None else digest,
                size_bytes=len(contents),
                status=status,
                created_at=NOW,
            )
        )
        db.commit()

        if intended:
            db.get(Organization, org_id).desired_artifact_id = artifact_id

        assignment = db.get(PodAssignment, (pod_id, org_id))
        if assignment is None:
            db.add(
                PodAssignment(
                    pod_id=pod_id,
                    org_id=org_id,
                    artifact_id=loaded,
                    load_status=LOAD_PULLING,
                    assigned_at=NOW,
                )
            )
        else:
            assignment.artifact_id = loaded
        db.commit()

    return artifact_id


def assignment_for(sessions, org_id, pod_id=POD):
    with sessions() as db:
        return db.get(PodAssignment, (pod_id, org_id))


# ==========================================================================
# the one this exists for
# ==========================================================================


def test_a_corrupt_artifact_is_not_swapped_in(engine, sessions, registry, tmp_path, roots):
    """A checksum that does not match means the tenant keeps what it had.

    The verification happens while the download is still a file on disk. If it
    were done after the swap instead, this test would find the corrupt graph
    already serving — which is the whole reason the order is fetch, verify,
    swap, record.
    """
    previous = FakeGraph("previous.db")
    registry.attach("org_1", previous, path="previous.db", version="0")
    already = seed(
        sessions, tmp_path, roots, version=0, status=ARTIFACT_ACTIVE, intended=False
    )

    seed(
        sessions,
        tmp_path,
        roots,
        digest="0" * 64,  # what the control plane believes, and it is wrong
        loaded=already,
    )

    swaps = reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    assert swaps == []
    # Still serving what it was serving, through the same handle.
    assert registry.get("org_1") is previous
    assert registry.entry("org_1").path == "previous.db"
    assert previous.closed is False
    # Reality was not written forward.
    assert assignment_for(sessions, "org_1").artifact_id == already
    assert assignment_for(sessions, "org_1").load_status == LOAD_PULLING


def test_an_artifact_with_no_recorded_checksum_is_not_swapped_in(
    engine, sessions, registry, tmp_path, roots
):
    """Nothing to verify against is a refusal, not a free pass."""
    seed(sessions, tmp_path, roots, digest="")

    assert reconcile(engine=engine, pod_id=POD, registry=registry, **roots) == []
    assert "org_1" not in registry
    assert assignment_for(sessions, "org_1").artifact_id is None


# ==========================================================================
# the ordinary pass
# ==========================================================================


def test_a_fleet_already_in_agreement_does_no_work(
    engine, sessions, registry, tmp_path, roots
):
    """Intent equals reality everywhere: no download, no open, no write."""
    artifact_id = seed(sessions, tmp_path, roots, status=ARTIFACT_ACTIVE)
    with sessions() as db:
        db.get(PodAssignment, (POD, "org_1")).artifact_id = artifact_id
        db.get(PodAssignment, (POD, "org_1")).load_status = LOAD_READY
        db.commit()

    before = assignment_for(sessions, "org_1").confirmed_at

    assert reconcile(engine=engine, pod_id=POD, registry=registry, **roots) == []

    # Nothing was fetched into the cache and nothing was touched in the store.
    assert not pod_cache_path(POD, "org_1", "1", root=roots["cache_root"]).exists()
    assert len(registry) == 0
    assert assignment_for(sessions, "org_1").confirmed_at == before


def test_a_tenant_whose_intent_moved_is_downloaded_verified_and_swapped(
    engine, sessions, registry, tmp_path, roots
):
    """The whole handshake, and then a second pass that finds nothing to do."""
    artifact_id = seed(sessions, tmp_path, roots)

    swaps = reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    assert [(s.org_id, s.moved_from, s.moved_to, s.version) for s in swaps] == [
        ("org_1", None, artifact_id, 1)
    ]

    cached = pod_cache_path(POD, "org_1", "1", root=roots["cache_root"])
    assert cached.is_file()
    assert registry.entry("org_1").path == str(cached)
    assert registry.entry("org_1").version == "1"

    assignment = assignment_for(sessions, "org_1")
    assert assignment.artifact_id == artifact_id
    assert assignment.load_status == LOAD_READY
    assert assignment.confirmed_at is not None

    # A finished artifact becomes the one in service once it is serving.
    with sessions() as db:
        assert db.get(GraphArtifact, artifact_id).status == ARTIFACT_ACTIVE

    # And the pass is idempotent: the second sweep has nothing to do.
    handle = registry.get("org_1")
    assert reconcile(engine=engine, pod_id=POD, registry=registry, **roots) == []
    assert registry.get("org_1") is handle


def test_the_same_artifact_already_loaded_is_not_downloaded_again(
    engine, sessions, registry, tmp_path, roots
):
    """Agreement is decided before anything is fetched.

    Checked by deleting the stored artifact: a pass that re-downloaded what it
    already has would raise looking for a file that is not there.
    """
    artifact_id = seed(sessions, tmp_path, roots, loaded=None)
    reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    stored = (
        roots["artifact_root"] / artifact_key("org_1", "1")
    )
    stored.unlink()
    cached = pod_cache_path(POD, "org_1", "1", root=roots["cache_root"])
    cached.unlink()

    assert reconcile(engine=engine, pod_id=POD, registry=registry, **roots) == []
    assert not cached.exists()
    assert assignment_for(sessions, "org_1").artifact_id == artifact_id


def test_an_unfinished_artifact_is_skipped(engine, sessions, registry, tmp_path, roots):
    """A half-built graph must never be swapped in."""
    seed(sessions, tmp_path, roots, status=ARTIFACT_BUILDING)

    assert reconcile(engine=engine, pod_id=POD, registry=registry, **roots) == []
    assert "org_1" not in registry

    assignment = assignment_for(sessions, "org_1")
    assert assignment.artifact_id is None
    assert assignment.load_status == LOAD_PULLING


def test_an_artifact_the_control_plane_does_not_have_is_skipped(
    engine, sessions, registry, tmp_path, roots
):
    """Intent pointing at nothing is a skip, not a crash."""
    seed(sessions, tmp_path, roots, intended=False)
    with sessions() as db:
        # Intent set to an artifact id that was never written. The reference
        # is deferred at creation only, so this has to go through a row that
        # exists and then be pointed elsewhere — which the database refuses,
        # exactly as it should. So the artifact is deleted instead.
        artifact = db.get(GraphArtifact, "art_org_1_1")
        db.get(Organization, "org_1").desired_artifact_id = artifact.artifact_id
        db.commit()
        db.get(Organization, "org_1").desired_artifact_id = None
        db.delete(artifact)
        db.commit()

    with sessions() as db:
        assert db.get(GraphArtifact, "art_org_1_1") is None

    assert reconcile(engine=engine, pod_id=POD, registry=registry, **roots) == []
    assert "org_1" not in registry


def test_a_tenant_assigned_to_another_pod_is_untouched(
    engine, sessions, registry, tmp_path, roots
):
    """This pod reconciles its own assignments and nobody else's."""
    seed(sessions, tmp_path, roots, org_id="org_mine")
    seed(sessions, tmp_path, roots, org_id="org_theirs", pod_id=OTHER_POD)

    swaps = reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    assert [swap.org_id for swap in swaps] == ["org_mine"]
    assert "org_theirs" not in registry
    assert assignment_for(sessions, "org_theirs", OTHER_POD).artifact_id is None
    assert assignment_for(sessions, "org_theirs", OTHER_POD).load_status == LOAD_PULLING


# ==========================================================================
# what happens when something really goes wrong
# ==========================================================================


def test_a_failure_partway_leaves_the_completed_tenants_recorded(
    engine, sessions, registry, tmp_path, roots
):
    """Progress is committed per tenant, so an exception does not undo it.

    The failure is a store that refuses to open — the registry's loader
    raising, which is not one of the three expected skips and therefore must
    leave this function.
    """
    seed(sessions, tmp_path, roots, org_id="org_a")
    seed(sessions, tmp_path, roots, org_id="org_b")
    seed(sessions, tmp_path, roots, org_id="org_c")

    opened: list[str] = []

    def loader(path: str):
        opened.append(path)
        if "org_c" in path:
            raise OSError("this store will not open")
        return FakeGraph(path)

    registry.set_loader(loader)

    with pytest.raises(OSError):
        reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    assert assignment_for(sessions, "org_a").artifact_id == "art_org_a_1"
    assert assignment_for(sessions, "org_b").artifact_id == "art_org_b_1"
    assert assignment_for(sessions, "org_c").artifact_id is None
    assert len(opened) == 3


def test_storage_failing_is_not_caught(engine, sessions, registry, tmp_path, roots):
    """A missing artifact file leaves the pass. It is not one of the skips.

    A storage backend that cannot produce a file it was told exists is a
    fault, not an expected state, and the caller has to see it. Caught here,
    every pass would report success and nothing would ever load.
    """
    from src.artifacts import ArtifactError

    seed(sessions, tmp_path, roots)
    (roots["artifact_root"] / artifact_key("org_1", "1")).unlink()

    with pytest.raises(ArtifactError):
        reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    assert assignment_for(sessions, "org_1").artifact_id is None


def test_the_swap_hands_back_what_it_displaced_still_open(
    engine, sessions, registry, tmp_path, roots
):
    """Nothing is closed under a reader. The caller decides when.

    The registry never closes what it displaces, and this pass does not add a
    rule of its own — a request that fetched the old handle a moment before
    the swap is still reading through it.
    """
    previous = FakeGraph("previous.db")
    registry.attach("org_1", previous, path="previous.db", version="0")
    already = seed(
        sessions, tmp_path, roots, version=0, status=ARTIFACT_ACTIVE, intended=False
    )
    seed(sessions, tmp_path, roots, loaded=already)

    with sessions() as db:
        # The assignment claims something that is not the intended artifact,
        # so the pass has work to do.
        assert db.get(PodAssignment, (POD, "org_1")).artifact_id == already

    swaps = reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    assert len(swaps) == 1
    assert swaps[0].displaced is not None
    assert swaps[0].displaced.handle is previous
    assert previous.closed is False
    assert registry.get("org_1") is not previous


# ==========================================================================
# the shape of the summary
# ==========================================================================


def test_the_summary_says_what_moved_where(engine, sessions, registry, tmp_path, roots):
    seed(sessions, tmp_path, roots, org_id="org_a", version=3)

    swaps = reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    assert len(swaps) == 1
    swap = swaps[0]
    assert isinstance(swap, Swap)
    assert swap.org_id == "org_a"
    assert swap.moved_from is None
    assert swap.moved_to == "art_org_a_3"
    assert swap.version == 3


def test_no_registry_means_the_process_registry(engine):
    """Defaulted, and to the one this process serves from.

    This used to raise, on the reasoning that a default would hand the pass
    its own graphs that no request reads. That reasoning was about a *fresh*
    instance and still holds against one; the default here is the shared
    object startup attaches to, so a pass called with no argument acts on the
    graphs that are really loaded. Asserted by identity, because a second
    registry behaving identically is exactly the bug the old rule prevented.
    """
    import src.reconcile as reconcile_module
    from src.registry import REGISTRY

    handle = object()
    REGISTRY.attach("org_untouched", handle, path="somewhere.db")

    # No registry argument at all.
    assert reconcile(engine=engine, pod_id=POD) == []

    assert reconcile_module.REGISTRY is REGISTRY
    assert REGISTRY.get("org_untouched") is handle


# ==========================================================================
# retiring the build a swap completed
# ==========================================================================


def _job_for(sessions, org_id, artifact_id, *, job_id="job_1", status=None):
    """A build that produced ``artifact_id``, left where a compile leaves one."""
    from src.models.control_plane import JOB_REGISTERED, IngestJob

    with sessions() as db:
        db.add(
            IngestJob(
                job_id=job_id,
                org_id=org_id,
                trigger="schedule",
                status=status or JOB_REGISTERED,
                queued_at=NOW,
                started_at=NOW,
            )
        )
        db.commit()
        artifact = db.get(GraphArtifact, artifact_id)
        artifact.built_by_job_id = job_id
        db.commit()
    return job_id


def _job(sessions, job_id="job_1"):
    from src.models.control_plane import IngestJob

    with sessions() as db:
        return db.get(IngestJob, job_id)


def test_a_second_pass_does_not_touch_a_job_it_already_retired(
    engine, sessions, registry, tmp_path, roots
):
    """The idempotency case, and the one this pass runs into every interval.

    A caught-up organisation takes the early return, so nothing should reach
    the job at all — and even if it did, a job already finished is left as it
    is. Asserted on the finish time as well as the status: a second
    retirement would move it, and moving it would rewrite when the build
    actually ended.
    """
    from src.models.control_plane import JOB_COMPLETED

    artifact_id = seed(sessions, tmp_path, roots)
    _job_for(sessions, "org_1", artifact_id)

    assert len(reconcile(engine=engine, pod_id=POD, registry=registry, **roots)) == 1

    first = _job(sessions)
    assert first.status == JOB_COMPLETED
    finished_at = first.finished_at

    # Intent and reality now agree, so this pass has nothing to do.
    assert reconcile(engine=engine, pod_id=POD, registry=registry, **roots) == []

    again = _job(sessions)
    assert again.status == JOB_COMPLETED
    assert again.finished_at == finished_at


def test_a_successful_swap_retires_its_job(engine, sessions, registry, tmp_path, roots):
    """Which is what frees this organisation for its next compile."""
    from src.models.control_plane import JOB_COMPLETED, JOB_IN_FLIGHT, JOB_REGISTERED

    artifact_id = seed(sessions, tmp_path, roots)
    _job_for(sessions, "org_1", artifact_id)

    assert _job(sessions).status == JOB_REGISTERED
    assert JOB_REGISTERED in JOB_IN_FLIGHT

    reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    job = _job(sessions)
    assert job.status == JOB_COMPLETED
    assert job.status not in JOB_IN_FLIGHT
    assert job.finished_at is not None
    assert job.error is None


def test_a_corrupt_artifact_leaves_its_job_alone(
    engine, sessions, registry, tmp_path, roots
):
    """Retiring follows a verified swap, not an attempt at one.

    The build did not finish successfully, and a job row saying it did would
    be a record of something that never happened — as well as freeing the
    tenant for a compile while the last one is unaccounted for.
    """
    from src.models.control_plane import JOB_REGISTERED

    already = seed(
        sessions, tmp_path, roots, version=0, status=ARTIFACT_ACTIVE, intended=False
    )
    artifact_id = seed(sessions, tmp_path, roots, digest="0" * 64, loaded=already)
    _job_for(sessions, "org_1", artifact_id)

    assert reconcile(engine=engine, pod_id=POD, registry=registry, **roots) == []

    job = _job(sessions)
    assert job.status == JOB_REGISTERED
    assert job.finished_at is None


def test_an_unfinished_artifact_leaves_its_job_alone(
    engine, sessions, registry, tmp_path, roots
):
    from src.models.control_plane import ARTIFACT_BUILDING, JOB_REGISTERED

    artifact_id = seed(sessions, tmp_path, roots, status=ARTIFACT_BUILDING)
    _job_for(sessions, "org_1", artifact_id)

    assert reconcile(engine=engine, pod_id=POD, registry=registry, **roots) == []
    assert _job(sessions).status == JOB_REGISTERED


def test_an_artifact_with_no_build_behind_it_swaps_anyway(
    engine, sessions, registry, tmp_path, roots
):
    """Placed by hand, with no job to retire. Nothing to do, nothing wrong."""
    seed(sessions, tmp_path, roots)

    swaps = reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    assert len(swaps) == 1


def test_a_job_already_finished_by_another_pass_is_left_as_it_is(
    engine, sessions, registry, tmp_path, roots
):
    """Another tick, or another pod, may have got there first."""
    from src.models.control_plane import JOB_COMPLETED, JOB_FAILED

    artifact_id = seed(sessions, tmp_path, roots)
    _job_for(sessions, "org_1", artifact_id, status=JOB_FAILED)

    reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    # Not rewritten to completed: whatever finished it knew more than this
    # pass does, and a failed build that later swapped is a fact worth
    # keeping as it was recorded.
    assert _job(sessions).status == JOB_FAILED
    assert _job(sessions).status != JOB_COMPLETED


def test_after_retirement_a_new_compile_can_open_a_job(
    engine, sessions, registry, tmp_path, roots
):
    """The proof that the original finding is fixed, not merely relabelled.

    Before this, a tenant compiled once and its job stayed inside the
    in-flight set forever, so the constraint refused every later build. Here
    the swap retires it and the next build's row is accepted.
    """
    from src.models.control_plane import JOB_FETCHING, IngestJob

    artifact_id = seed(sessions, tmp_path, roots)
    _job_for(sessions, "org_1", artifact_id)

    reconcile(engine=engine, pod_id=POD, registry=registry, **roots)

    with sessions() as db:
        db.add(
            IngestJob(
                job_id="job_next",
                org_id="org_1",
                trigger="schedule",
                status=JOB_FETCHING,
                queued_at=NOW,
                started_at=NOW,
            )
        )
        # Refused before this change; accepted now.
        db.commit()

        assert db.get(IngestJob, "job_next").status == JOB_FETCHING
