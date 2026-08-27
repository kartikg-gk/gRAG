"""Which process a new tenant should be given to.

One function. It reads the fleet and returns an identifier, and it writes
nothing — creating the assignment is the job of whatever asked, because that
caller is the one inside a transaction with the rest of provisioning.

The policy
----------

Least loaded, among pods that are registered and healthy. Ties break on the
identifier, ascending, so the same fleet state always produces the same
answer: a caller that retries after a timeout places the tenant where the
first attempt would have, rather than scattering retries across the fleet.

Load is **every assignment a pod holds**, whatever state it is in. A tenant
that is still pulling is still one this pod is answerable for, and counting
only the ready ones would make a pod that is halfway through three loads look
like the emptiest machine available and earn it a fourth.

The empty pod problem
---------------------

A grouped count of assignments has one row per pod **that has some**. A pod
with none is simply absent from the result — and that pod is the one that
should win a least-loaded contest.

So the counts start as every healthy pod at zero and the query is laid over
the top. Without that, new capacity is invisible to placement: the fleet grows
and every tenant still lands on the machines that were already busy. Nothing
errors, nothing looks wrong, and the new pods stay empty forever.

Never raises
------------

The whole body is wrapped, and any failure returns this process's own
identifier.

Placement sits in front of a user-facing action. A control plane that is
briefly unreachable should not turn "add a tenant" into an error the user
sees; it should put the tenant somewhere that can serve it, and in a
single-process deployment that place is here. The failure is logged, so a
fleet that has quietly collapsed to one pod is visible to whoever reads logs
rather than only to whoever notices the load.

The same reasoning runs through the freshness rules below: each one errs
toward keeping a pod in the running, because the cost of including a doubtful
pod is one tenant on a slow machine, and the cost of excluding every pod is
the fallback for everybody.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import Engine, func
from sqlmodel import select

from .common.config import POD_HEARTBEAT_WINDOW_SECONDS, POD_ID
from .models.control_plane import POD_READY, Pod, PodAssignment
from .models.database import control_plane_sessions, create_control_plane_engine

logger = logging.getLogger("graphrag.placement")

#: Where a tenant goes when the fleet cannot be consulted.
#:
#: Resolved once, here, rather than per call: it is a property of this
#: process, it cannot change while the process runs, and reading it on the
#: failure path would mean the failure path had its own way to fail.
#:
#: A process with no identifier configured falls back to this literal, which
#: is deliberately a name and not an empty string — an assignment row naming
#: ``""`` would be a row nobody can trace back to a machine.
UNIDENTIFIED_POD = "pod-local"

FALLBACK_POD = POD_ID or UNIDENTIFIED_POD


def choose_pod(*, engine: Engine | None = None, now: datetime | None = None) -> str:
    """The pod a new tenant should be assigned to.

    Reads the fleet, returns an identifier, writes nothing. **Never raises**:
    every failure, including a control plane that cannot be reached at all,
    returns this process's own identifier.
    """
    moment = now if now is not None else datetime.now(timezone.utc)

    try:
        made = engine if engine is not None else create_control_plane_engine()
        sessions = control_plane_sessions(made)

        with sessions() as db:
            healthy = [
                pod
                for pod in db.exec(select(Pod)).all()
                if pod.status == POD_READY and _is_fresh(pod.last_heartbeat_at, moment)
            ]

            if not healthy:
                logger.warning(
                    "no healthy pod to place on; falling back to %s", FALLBACK_POD
                )
                return FALLBACK_POD

            # Every candidate starts at zero, and the query is laid over the
            # top. A pod with no assignments has no row to overlay, and it is
            # the one that should win.
            load = {pod.pod_id: 0 for pod in healthy}

            counted = db.exec(
                select(PodAssignment.pod_id, func.count())
                .group_by(PodAssignment.pod_id)
            ).all()
            for pod_id, count in counted:
                if pod_id in load:
                    load[pod_id] = count

            # Fewest tenants, then identifier, so one fleet state has one
            # answer however many times it is asked.
            return min(load.items(), key=lambda entry: (entry[1], entry[0]))[0]
    except Exception:  # noqa: BLE001 - placement degrades, it does not fail
        logger.exception(
            "could not choose a pod; falling back to %s", FALLBACK_POD
        )
        return FALLBACK_POD


def _is_fresh(heartbeat, moment: datetime) -> bool:
    """Whether a pod has beaten recently enough to be given work.

    Three cases lean the same way, toward keeping the pod:

    **Never beaten is fresh.** A pod that registered a moment ago has not had
    a chance yet, and calling that stale would make every newly booted process
    permanently unplaceable — the fleet could not grow.

    **A timestamp with no timezone is read as UTC.** Comparing a naive value
    against an aware one raises, and a store that hands one back would
    otherwise take the whole fleet out through the failure path below.

    **Anything unexpected is fresh.** Including a doubtful pod costs one
    tenant on a slow machine. Excluding every pod costs the fallback for
    everybody, which is the worse of the two.
    """
    if heartbeat is None:
        return True

    try:
        if isinstance(heartbeat, datetime):
            beat = (
                heartbeat.replace(tzinfo=timezone.utc)
                if heartbeat.tzinfo is None
                else heartbeat
            )
        else:
            # Stored as seconds since the epoch, like every other timestamp
            # in these tables.
            beat = datetime.fromtimestamp(float(heartbeat), timezone.utc)

        return (moment - beat).total_seconds() <= POD_HEARTBEAT_WINDOW_SECONDS
    except Exception:  # noqa: BLE001 - a doubtful pod stays a candidate
        logger.warning("could not read a heartbeat; treating the pod as fresh")
        return True
