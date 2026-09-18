"""What a serving process does when it starts, and what it does thereafter.

Two pieces. **Boot** makes a restarted process usable immediately: it says
this process exists and is ready, and it re-opens the graphs this process was
already serving before it stopped. **The loop** keeps it current afterwards by
running the reconcile pass on an interval.

Neither calls the other. They are here together because together they are the
whole of a process's fleet membership, and apart they are two functions each
small enough to read in one sitting.

Which registry these act on
---------------------------

Both take one, and both fall back to the process's own when they are given
none. That fallback is a *shared* object, not a fresh one: it is what
startup attached the store to and what every route resolves through, so a
boot or a tick called with no argument loads graphs that requests can
actually read. Constructing one here instead would give these passes their
own set that nothing serves from — every boot looking successful and every
query still finding nothing.

Boot catches everything. The loop catches everything. The pass in the middle catches almost nothing.
---------------------------------------------------------------------------------------------------

That is three different postures on purpose, and the reason is when each one
runs.

Boot runs **once**. One unreadable row, one artifact that storage cannot
produce, one store that will not open — none of those may stop a process from
starting and serving the other nine tenants. So every tenant is wrapped, the
query around them is wrapped, and ``boot`` does not raise at all. A process
that comes up serving nine of ten tenants and logging the tenth is strictly
better than one that comes up serving none.

The reconcile pass runs **every few seconds** and deliberately lets almost
everything out, because a pass that swallows its failures reports success
forever while accomplishing nothing.

The loop is what makes that affordable: it wraps each tick, logs the
traceback, and goes again on the next one. The pass gets to be loud precisely
because something above it is listening.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, select

from .artifacts import get_artifact, pod_cache_path
from .common.config import POD_ADDRESS, POD_ID, RECONCILE_INTERVAL_SECONDS
from .models.control_plane import (
    LOAD_READY,
    POD_READY,
    GraphArtifact,
    Pod,
    PodAssignment,
)
from .models.database import control_plane_sessions, create_control_plane_engine
from .reconcile import reconcile
from .registry import REGISTRY, GraphRegistry

logger = logging.getLogger("graphrag.pod")


@dataclass(frozen=True)
class Hydrated:
    """One graph re-opened at boot: which tenant, and which build."""

    org_id: str
    version: int


# --------------------------------------------------------------------------
# boot
# --------------------------------------------------------------------------


def register_pod(
    db: Session, *, pod_id: str = POD_ID, address: str | None = None
) -> None:
    """Say this process exists and is ready to serve.

    Idempotent across restarts, which is the only property that matters: this
    runs on every boot and must not care whether the row survived the last
    stop. A pod that is already known has its status and heartbeat refreshed;
    its address is left alone unless a new one was given, so a restart that
    was told nothing does not overwrite what somebody configured.
    """
    existing = db.get(Pod, pod_id)
    now = _now()

    if existing is None:
        db.add(
            Pod(
                pod_id=pod_id,
                address=address if address is not None else POD_ADDRESS,
                status=POD_READY,
                last_heartbeat_at=now,
                created_at=now,
            )
        )
    else:
        existing.status = POD_READY
        existing.last_heartbeat_at = now
        if address is not None:
            existing.address = address

    db.commit()


def hydrate(
    db: Session,
    *,
    pod_id: str = POD_ID,
    registry: GraphRegistry | None = None,
    cache_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> list[Hydrated]:
    """Re-open the graphs this process was serving before it stopped.

    The assignments whose load status is ready are exactly that set: a tenant
    reached ready only by completing a verified swap, so re-opening what it
    records is re-opening what this process was actually answering from.

    **Never raises.** Every tenant is wrapped, and so is the query that finds
    them — an unreadable control plane at boot returns an empty summary and a
    log line, because the alternative is a process that will not start.
    """
    registry = registry if registry is not None else REGISTRY

    try:
        assignments = db.exec(
            select(PodAssignment).where(
                PodAssignment.pod_id == pod_id,
                PodAssignment.load_status == LOAD_READY,
            )
        ).all()
    except Exception:  # noqa: BLE001 - a boot that cannot read must still boot
        logger.exception("%s: could not read this pod's assignments", pod_id)
        return []

    hydrated: list[Hydrated] = []

    for assignment in assignments:
        org_id = assignment.org_id

        if assignment.artifact_id is None:
            # Ready with nothing recorded. Nothing to re-open, and nothing
            # wrong either — the reconcile pass will give it something.
            continue

        try:
            artifact = db.get(GraphArtifact, assignment.artifact_id)
            if artifact is None:
                logger.warning(
                    "%s: %s records artifact %s, which the control plane does "
                    "not have",
                    pod_id,
                    org_id,
                    assignment.artifact_id,
                )
                continue
            destination = pod_cache_path(
                pod_id, org_id, str(artifact.version), root=cache_root
            )
            get_artifact(artifact.s3_uri, destination, root=artifact_root)
            registry.replace(
                org_id, path=str(destination), version=str(artifact.version)
            )
        except Exception:  # noqa: BLE001 - one bad tenant must not stop boot
            logger.exception("%s: could not hydrate %s", pod_id, org_id)
            continue

        hydrated.append(Hydrated(org_id=org_id, version=artifact.version))

    logger.info("%s: hydrated %d tenant(s)", pod_id, len(hydrated))
    return hydrated


def boot(
    *,
    registry: GraphRegistry | None = None,
    engine: Engine | None = None,
    pod_id: str = POD_ID,
    address: str | None = None,
    cache_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> list[Hydrated]:
    """Register this process, then re-open what it was serving.

    In that order, so a process is visible to the fleet before it starts doing
    anything that takes time.

    **Registration failing does not stop hydration.** Being unregistered is a
    visibility problem somebody can see and fix; refusing to serve tenants
    that are already assigned here would be the worse of the two failures.
    """
    registry = registry if registry is not None else REGISTRY

    engine = engine if engine is not None else create_control_plane_engine()
    sessions = control_plane_sessions(engine)

    with sessions() as db:
        try:
            register_pod(db, pod_id=pod_id, address=address)
        except Exception:  # noqa: BLE001 - unregistered still serves
            logger.exception("%s: could not register this pod", pod_id)
            db.rollback()

        return hydrate(
            db,
            pod_id=pod_id,
            registry=registry,
            cache_root=cache_root,
            artifact_root=artifact_root,
        )


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


async def poll(
    *,
    registry: GraphRegistry | None = None,
    stop: asyncio.Event,
    engine: Engine | None = None,
    pod_id: str = POD_ID,
    interval: float = RECONCILE_INTERVAL_SECONDS,
    cache_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> None:
    """Reconcile on an interval until ``stop`` is set.

    Three things about the shape, each of which is the whole reason for its
    line:

    The pass runs **off the event loop**. It opens a database session and
    copies files; run inline it would stall every request this process is
    serving for as long as a download takes.

    Every tick is **wrapped, with its traceback**. The pass is deliberately
    loud — this is what makes that affordable. A failed tick is one log line
    and the next tick tries again.

    The wait is on the **stop signal with a timeout**, not a sleep. A stop
    requested one second into a five-second interval takes effect now rather
    than four seconds from now, which is the difference between a process that
    shuts down and one that appears to hang. The timeout expiring is the
    ordinary case — it means the interval elapsed — and is not a failure.
    """
    registry = registry if registry is not None else REGISTRY

    logger.info("%s: reconcile loop starting, every %ss", pod_id, interval)
    try:
        while not stop.is_set():
            try:
                await asyncio.to_thread(
                    reconcile,
                    engine=engine,
                    pod_id=pod_id,
                    registry=registry,
                    cache_root=cache_root,
                    artifact_root=artifact_root,
                )
            except Exception:  # noqa: BLE001 - the pass is loud; this listens
                logger.exception("%s: reconcile pass failed", pod_id)

            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                # The interval elapsed with no stop. Go again.
                pass
    finally:
        logger.info("%s: reconcile loop stopped", pod_id)


def _now() -> int:
    """Seconds since the epoch, the way every other timestamp here is stored."""
    return int(datetime.now(timezone.utc).timestamp())
