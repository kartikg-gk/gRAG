"""Everything around a compile except the compile.

The task, the lock it takes, the job row it opens, the transitions it records
and what it does when the work raises. The work itself is one call, and what
it does lives beside this rather than in it.

The retry interaction, which is easy to break and silent when broken
--------------------------------------------------------------------

The core marks its job **failed** before re-raising. Failed is not one of the
in-flight statuses, so the database's one-in-flight-job-per-organisation
constraint stops counting it, and the retry that follows can open a fresh row.

Reverse those two — retry without finalising, or finalise to a status that is
still in flight — and every retry is refused by the constraint. The refusal
does not surface as a failed compile; it surfaces as an organisation that
quietly stops being rebuilt while every task reports having been dispatched.

What is retried, and what is not
--------------------------------

Only the transient one: the source refusing further requests until its quota
resets. That is a wait-and-try-again condition and nothing about the compile
is wrong. Everything else propagates on the first failure — a build that is
broken is not made less broken by being run five more times, and the job row
already records why.

Version numbers are computed racily on purpose
----------------------------------------------

The next version is the highest that exists plus one. Two compiles running
together compute the same number, and the uniqueness constraint on
organisation-and-version is what refuses the second. The computation is the
common path; the constraint is the guarantee. Putting a lock around the
computation would move the guarantee somewhere weaker.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import func
from sqlmodel import Session, select

from ..ingestion.github import GitHubRateLimitError
from ..models.control_plane import (
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_FETCHING,
    JOB_REGISTERED,
    GraphArtifact,
    IngestJob,
    Organization,
)
from ..models.database import control_plane_sessions, create_control_plane_engine
from .app import COMPILE_TASK, app
from .locks import compile_lock

logger = logging.getLogger("graphrag.worker.compile")

#: What a compile reports back. Strings rather than an enumeration, for the
#: same reason the statuses in the control plane are strings: every one of
#: these crosses a queue as text.
SKIPPED_LOCKED = "lock_held"
UNKNOWN_ORGANIZATION = "unknown_org"

#: How long to wait before trying again when the source is out of quota and
#: says nothing about when it will not be.
DEFAULT_RETRY_SECONDS = 60

#: The longest backoff, whatever the source asks for. A primary quota can be
#: an hour away, and a task sleeping longer than that is a task nobody can
#: tell apart from one that is stuck.
MAX_RETRY_SECONDS = 3600

#: How many times a quota failure is retried before the task gives up.
MAX_RETRIES = 5


def run_phases(org_id: str, job_id: str, db: Session):
    """Build the graph and return what it produced.

    Delegated rather than written here: this module is the shell around a
    compile — the lock, the job row, the retry — and what a compile *is* is
    long enough to be its own reading.
    """
    from .phases import run_phases as phases

    return phases(org_id, job_id, db)


@app.task(bind=True, name=COMPILE_TASK, max_retries=MAX_RETRIES)
def reconcile_org_to_head(self, org_id: str) -> dict:
    """Rebuild one organisation's graph, if nobody else is already doing it.

    Returns what happened. **Not acquiring the lock is a return, not a
    raise**: another compile for this organisation is legitimately running,
    and recording a failure for that would put a failure in the record of
    something that went right.
    """
    with compile_lock(org_id) as acquired:
        if not acquired:
            logger.info("%s: already compiling elsewhere; leaving it", org_id)
            return {"org_id": org_id, "status": SKIPPED_LOCKED}

        engine = create_control_plane_engine()
        sessions = control_plane_sessions(engine)

        try:
            with sessions() as db:
                return _reconcile(org_id, db)
        except GitHubRateLimitError as exc:
            # The one transient condition. The job row was already finalised
            # as failed on the way out of the core, so the constraint will
            # let the retry open a new one.
            delay = min(exc.retry_after or DEFAULT_RETRY_SECONDS, MAX_RETRY_SECONDS)
            logger.warning("%s: out of quota; trying again in %ds", org_id, delay)
            raise self.retry(exc=exc, countdown=delay)
        finally:
            engine.dispose()


def _reconcile(org_id: str, db: Session) -> dict:
    """Open a job, run the phases, and record what became of it."""
    organization = db.get(Organization, org_id)
    if organization is None:
        # Armed, then deleted before the sweeper came round. Not an error and
        # not worth a failed job row for an organisation that is not there.
        logger.warning("%s: no such organisation; nothing to compile", org_id)
        return {"org_id": org_id, "status": UNKNOWN_ORGANIZATION}

    job = IngestJob(
        job_id=_new_job_id(),
        org_id=org_id,
        trigger="schedule",
        status=JOB_FETCHING,
        queued_at=_now(),
        started_at=_now(),
    )
    db.add(job)
    db.commit()
    job_id = job.job_id

    try:
        summary = run_phases(org_id, job_id, db)
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        logger.exception("%s: compile failed", org_id)
        # Both halves matter. The row is what anyone reading the database
        # sees, and the re-raise is what lets the task decide about retrying.
        finalize_job(db, job_id, JOB_FAILED, error=str(exc))
        raise

    # A run that found nothing changed built nothing, and says so.
    if getattr(summary, "skipped", False):
        return {"org_id": org_id, "job_id": job_id, "status": JOB_COMPLETED}
    return {
        "org_id": org_id,
        "job_id": job_id,
        "artifact_id": summary.artifact_id,
        "version": summary.version,
        "s3_uri": summary.s3_uri,
        "status": JOB_REGISTERED,
    }


def set_job_status(db: Session, job_id: str, status: str) -> None:
    """Move a job to ``status``.

    A job that is no longer there is not worth failing over: whatever removed
    it knew more than this call does, and raising here would turn a bookkeeping
    update into a failed compile.
    """
    job = db.get(IngestJob, job_id)
    if job is None:
        logger.warning("no job %s to move to %s", job_id, status)
        return

    job.status = status
    db.commit()


def finalize_job(
    db: Session, job_id: str, status: str, *, error: str | None = None
) -> None:
    """Close a job out: its final status, why, and when it stopped."""
    job = db.get(IngestJob, job_id)
    if job is None:
        logger.warning("no job %s to finalise as %s", job_id, status)
        return

    job.status = status
    job.error = error
    job.finished_at = _now()
    db.commit()


def next_version(db: Session, org_id: str) -> int:
    """The version number the next artifact for ``org_id`` should carry.

    Racy, and left that way — see the module docstring. Two compiles agree on
    the same number and the database refuses the second.
    """
    highest = db.exec(
        select(func.max(GraphArtifact.version)).where(GraphArtifact.org_id == org_id)
    ).one()

    return int(highest) + 1 if highest is not None else 1


def _new_job_id() -> str:
    import secrets

    return f"job_{secrets.token_hex(8)}"


def _now() -> int:
    """Seconds since the epoch, the way every timestamp in these tables is."""
    return int(datetime.now(timezone.utc).timestamp())
