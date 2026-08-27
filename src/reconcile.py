"""Bringing what this process has loaded into agreement with what it should.

One sweep, synchronous
----------------------

``reconcile`` is a plain function. It is called, it makes one pass over the
tenants assigned to this process, it returns what it changed, and it is
finished. There is no loop here, nothing runs in the background, and nothing
schedules anything. A caller that wants this to happen repeatedly calls it
repeatedly — that caller is a separate piece and does not exist yet.

Written this way because the pass is the part worth testing and a scheduler is
not. A test seeds rows, constructs a registry, calls this, and reads the
summary; no clock, no environment, no waiting.

Intent and reality
------------------

An organisation's ``active_artifact_id`` is what it is **supposed** to be
serving. An assignment's ``artifact_id`` is what this process **has**. The
pass looks at every organisation assigned here and acts only where those two
differ. Where they agree it does nothing at all — no download, no open, no
write — so a sweep over a fleet that is already caught up costs one query.

The order of the handshake is the design
----------------------------------------

Fetch, verify, swap, *then* record. Reality moves forward only after a swap
that has already happened locally and has already been checked.

That ordering is what makes this self-healing without anything detecting a
failure. A crash anywhere before the final write leaves intent and reality
still disagreeing, so the next pass simply does the same work again. There is
no cleanup step, no half-finished state to repair, and nothing that has to
notice the crash happened.

Verifying *after* the swap would mean a corrupt download is answering queries
for however long it takes someone to look. The download is checked while it is
still just a file on disk, and a mismatch means the tenant keeps serving
whatever it was already serving.

What is caught, and what is not
-------------------------------

Three conditions are expected and are handled by skipping the tenant and
moving on: the intended artifact is missing, it is not in a state that means
finished, or what arrived does not match the checksum that was recorded for
it. Each is logged.

**Everything else propagates.** A database that cannot be reached, storage
that raises, a store that will not open — these leave this function. The
caller decides what a failed pass means, and its job will be to log it and try
again on the next one. Catching them here would produce a pass that always
reports success and quietly does nothing, which is the failure mode that takes
longest to notice.

Boot-time hydration will want the opposite rule — one unusable row must not
stop a process from starting. The difference is deliberate: something that
runs every few seconds can afford to fail loudly, and something that runs once
cannot.

Progress is committed per tenant, as each finishes, so a pass that gets three
done and then raises on the fourth leaves those three recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlmodel import select

from .artifacts import checksum, get_artifact, pod_cache_path
from .common.config import POD_ID
from .models.control_plane import (
    ARTIFACT_ACTIVE,
    ARTIFACT_READY,
    LOAD_READY,
    GraphArtifact,
    Organization,
    PodAssignment,
)
from .models.database import control_plane_sessions, create_control_plane_engine
from .registry import GraphEntry, GraphRegistry

logger = logging.getLogger("graphrag.reconcile")

#: The artifact states a tenant may be moved onto.
#:
#: ``ready`` means built and verified by whatever produced it; ``active``
#: means it is already the one in service. Every other state — building,
#: failed, superseded — is either unfinished or deliberately retired, and
#: swapping to one would put a half-written graph in front of traffic.
SERVABLE = (ARTIFACT_READY, ARTIFACT_ACTIVE)


@dataclass(frozen=True)
class Swap:
    """One tenant moved from one artifact to another, this pass.

    ``displaced`` is the registry entry this swap replaced, **still open**.
    The registry never closes what it displaces and neither does this pass: a
    reader that fetched the old handle a moment before the swap is still
    reading through it. The caller holds these and closes them when it knows
    nothing is using them.
    """

    org_id: str
    moved_from: str | None
    moved_to: str
    version: int
    displaced: GraphEntry | None = None


def reconcile(
    *,
    engine: Engine | None = None,
    pod_id: str = POD_ID,
    registry: GraphRegistry | None = None,
    cache_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> list[Swap]:
    """Make one pass, and return the swaps it performed.

    Everything is injectable and everything has a default: the engine comes
    from the environment, the identifier from configuration, and the registry
    has to be handed in by whatever owns the process's open graphs — there is
    no process-wide one to reach for, and inventing one here would give this
    pass its own set of graphs that nothing serves from.

    Returns an empty list when intent and reality already agree everywhere.
    """
    if registry is None:
        raise ValueError(
            "reconcile needs the registry this process serves from; pass the "
            "one the application built"
        )

    engine = engine if engine is not None else create_control_plane_engine()
    sessions = control_plane_sessions(engine)

    swaps: list[Swap] = []

    with sessions() as db:
        assignments = db.exec(
            select(PodAssignment, Organization)
            .join(Organization, Organization.org_id == PodAssignment.org_id)
            .where(PodAssignment.pod_id == pod_id)
        ).all()

        for assignment, organization in assignments:
            intended = organization.active_artifact_id
            loaded = assignment.artifact_id

            if intended is None or intended == loaded:
                # Nothing to do, and nothing done: no fetch, no open, no
                # write. This is the branch a caught-up fleet takes.
                continue

            artifact = db.get(GraphArtifact, intended)
            if artifact is None:
                logger.warning(
                    "%s: intended artifact %s is not in the control plane",
                    organization.org_id,
                    intended,
                )
                continue

            if artifact.status not in SERVABLE:
                logger.warning(
                    "%s: artifact %s is %s, not something to serve",
                    organization.org_id,
                    artifact.artifact_id,
                    artifact.status,
                )
                continue

            destination = pod_cache_path(
                pod_id,
                organization.org_id,
                str(artifact.version),
                root=cache_root,
            )
            get_artifact(artifact.uri, destination, root=artifact_root)

            if not _arrived_intact(organization.org_id, artifact, destination):
                continue

            # The swap. Local, verified, and only now visible to readers.
            displaced = registry.replace(
                organization.org_id,
                path=str(destination),
                version=str(artifact.version),
            )

            # Reality, written last. Anything that went wrong above left this
            # untouched, which is what the next pass reads as "still to do".
            assignment.artifact_id = artifact.artifact_id
            assignment.load_status = LOAD_READY
            assignment.confirmed_at = _now()
            if artifact.status == ARTIFACT_READY:
                artifact.status = ARTIFACT_ACTIVE
            # Per tenant, so a later failure does not undo this one.
            db.commit()

            logger.info(
                "%s: %s -> %s (version %s)",
                organization.org_id,
                loaded,
                artifact.artifact_id,
                artifact.version,
            )
            swaps.append(
                Swap(
                    org_id=organization.org_id,
                    moved_from=loaded,
                    moved_to=artifact.artifact_id,
                    version=artifact.version,
                    displaced=displaced,
                )
            )

    return swaps


def _arrived_intact(org_id: str, artifact: Any, destination: Path) -> bool:
    """Whether the downloaded file is what the control plane recorded.

    A recorded checksum that does not match is a corrupt or wrong artifact and
    the tenant keeps what it has. **No recorded checksum is also a refusal**:
    there is nothing to check against, and swapping anyway would mean the one
    artifact nobody can verify is the one that goes in unverified.
    """
    if not artifact.checksum:
        logger.error(
            "%s: artifact %s has no recorded checksum; not swapping",
            org_id,
            artifact.artifact_id,
        )
        return False

    arrived = checksum(destination)
    if arrived != artifact.checksum:
        logger.error(
            "%s: artifact %s failed its checksum; not swapping",
            org_id,
            artifact.artifact_id,
        )
        return False

    return True


def _now() -> int:
    """Seconds since the epoch, the way every other timestamp here is stored."""
    return int(datetime.now(timezone.utc).timestamp())
