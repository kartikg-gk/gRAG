"""What a compile actually does, from fetching to the pointer flip.

Four phases, and one transaction at the end that either publishes everything
or publishes nothing.

The order, and the one thing it protects
----------------------------------------

Ingest every repository, compile an artifact, upload it, then register it and
point the organisation at it. The registration is where the cursors move too,
and that pairing is the whole safety of the pipeline.

**If cursors advanced during the ingest and the registration then failed**,
the next run would start from the new cursor: it would skip everything the
failed run had already pulled into the graph store, and nothing would ever go
back for it. The rows would be in the database, absent from every artifact
that ever ships, and no error anywhere.

So the cursors are collected during the ingest and written with the
registration. A failure at any point before that commit leaves them where
they were, the next run re-fetches the same delta, and the ingest is an
upsert, so re-fetching costs time and nothing else. That is what makes the
whole compile safe to retry.

Uploading before registering
----------------------------

The file goes to storage before any row describes it. A crash in between
leaves a file nothing references — and that corrects itself rather than
leaking: the version number was never recorded, so the next attempt computes
the same version, writes the same key, and overwrites what was left.

The reverse order would be worse in a way that does not correct itself: a row
promising an artifact that is not there, which every pod would then try to
download.

Statuses are set before each phase
----------------------------------

Not after. The status answers "what is this job doing right now" for somebody
reading the database while it runs, and a status written afterwards would
show a job as still fetching through the whole of a long compile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlmodel import Session, select

from ..artifacts import artifact_key, checksum, put_artifact
from ..models.control_plane import (
    ARTIFACT_ACTIVE,
    ARTIFACT_READY,
    ARTIFACT_SUPERSEDED,
    JOB_COMPILING,
    JOB_COMPLETED,
    JOB_COMPUTING,
    JOB_REGISTERED,
    JOB_UPLOADING,
    GraphArtifact,
    Organization,
    Repository,
)
from .compiler import compile_artifact
from .ingest import ingest_repository

logger = logging.getLogger("graphrag.worker.phases")

#: Where a compile builds before uploading. A working directory, not storage:
#: what is here is disposable the moment the upload succeeds.
BUILD_ROOT = "builds"

#: The artifact states a new version may replace.
#:
#: Anything already superseded or failed is left alone — it was retired for a
#: reason, and re-retiring it would overwrite the record of which one was in
#: service when.
SUPERSEDABLE = (ARTIFACT_READY, ARTIFACT_ACTIVE)


@dataclass(frozen=True)
class CompileSummary:
    """What one compile produced. ``artifact_id`` is unset when nothing was
    built because nothing had changed."""

    org_id: str
    artifact_id: Optional[str]
    version: Optional[int]
    entities: int
    edges: int
    items: int
    uri: Optional[str] = None
    skipped: bool = False


def run_phases(
    org_id: str,
    job_id: str,
    db: Session,
    *,
    build_root: str | Path = BUILD_ROOT,
    ingest: Callable[..., object] = ingest_repository,
    compile_to: Callable[..., object] = compile_artifact,
    upload: Callable[..., str] = put_artifact,
    now: int | None = None,
) -> CompileSummary:
    """Ingest, compile, upload, register, and point the organisation at it."""
    from .compile import finalize_job, next_version, set_job_status

    moment = now if now is not None else _now()
    repositories = db.exec(
        select(Repository).where(Repository.org_id == org_id)
    ).all()

    # -- ingest ------------------------------------------------------------
    set_job_status(db, job_id, JOB_COMPUTING)

    items = 0
    # Collected, not written. See the module docstring: these move with the
    # registration or they do not move at all.
    cursors: dict[str, str | None] = {}

    for repository in repositories:
        result = ingest(
            org_id=org_id,
            repo_id=repository.repo_id,
            repo_name=repository.name,
            cursor=repository.last_sync_cursor,
            db=db,
            token=None,
        )
        cursors[repository.repo_id] = result.cursor
        items += result.items

    if items == 0:
        # Nothing changed anywhere, so a compile would produce an artifact
        # identical to the one already in service, under a new version — and
        # every pod holding this tenant would download and swap it for no
        # difference at all. The fleet cost is real and the saving is a
        # branch.
        #
        # Finished, not failed: a job that legitimately had nothing to do is
        # complete. It is also terminal, which releases this organisation for
        # the next compile, and it records when it stopped like any other
        # finished job.
        finalize_job(db, job_id, JOB_COMPLETED)
        logger.info("%s: no changes in any repository; nothing to compile", org_id)
        return CompileSummary(
            org_id=org_id,
            artifact_id=None,
            version=None,
            entities=0,
            edges=0,
            items=0,
            skipped=True,
        )

    # -- compile -----------------------------------------------------------
    set_job_status(db, job_id, JOB_COMPILING)

    # Racy on purpose: two concurrent compiles compute the same number, and
    # the uniqueness constraint on organisation-and-version refuses the
    # second. The arithmetic is the common path, not the guarantee.
    version = next_version(db, org_id)
    build_path = Path(build_root) / org_id / f"{version}.db"
    built = compile_to(org_id, db, build_path)

    # -- upload ------------------------------------------------------------
    set_job_status(db, job_id, JOB_UPLOADING)

    key = artifact_key(org_id, str(version))
    digest = checksum(built.path)
    size = built.path.stat().st_size
    uri = upload(built.path, key)

    # -- register, and flip ------------------------------------------------
    set_job_status(db, job_id, JOB_REGISTERED)

    artifact = GraphArtifact(
        artifact_id=_new_artifact_id(),
        org_id=org_id,
        version=version,
        uri=uri,
        checksum=digest,
        size_bytes=size,
        entity_count=built.entities,
        status=ARTIFACT_READY,
        job_id=job_id,
        created_at=moment,
    )
    db.add(artifact)
    db.flush()

    organization = db.get(Organization, org_id)
    previous_id = organization.active_artifact_id if organization else None
    if previous_id is not None:
        previous = db.get(GraphArtifact, previous_id)
        # Only what is still in service. Something already superseded or
        # failed was retired for its own reason, and rewriting it would lose
        # which artifact was actually in service when.
        if previous is not None and previous.status in SUPERSEDABLE:
            previous.status = ARTIFACT_SUPERSEDED

    if organization is not None:
        organization.active_artifact_id = artifact.artifact_id
        organization.updated_at = moment

    for repository in repositories:
        moved = cursors.get(repository.repo_id)
        if moved is not None:
            repository.last_sync_cursor = moved
            repository.last_synced_at = moment

    # Everything above, or nothing. A partial publish is either an artifact
    # nobody points at or cursors that have skipped past work nobody shipped.
    db.commit()

    logger.info(
        "%s: version %d registered (%d entities, %d edges from %d item(s))",
        org_id,
        version,
        built.entities,
        built.edges,
        items,
    )
    return CompileSummary(
        org_id=org_id,
        artifact_id=artifact.artifact_id,
        version=version,
        entities=built.entities,
        edges=built.edges,
        items=items,
        uri=uri,
    )


def _new_artifact_id() -> str:
    import secrets

    return f"art_{secrets.token_hex(8)}"


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())
