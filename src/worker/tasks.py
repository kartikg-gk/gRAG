"""What the queue runs, and the one thing it deliberately does not.

Two operations. One is a queued task on a schedule; the other is a plain
function that a request handler calls directly, and the difference between
them is the point of this module.

Arming is not a task
--------------------

``arm_organization`` is what an incoming event calls. Its entire body is one
small write to the same store the broker is running on. Routed through the
queue it would be a round trip through that store to schedule a job whose
whole purpose is a single write to that store — the scheduling costing more
than the work, and the caller waiting on a broker for something it could have
done itself.

So the event path stays synchronous and cheap, and only the expensive half —
the compile — is queued.

It is idempotent by construction rather than by checking anything: arming an
organisation that is already armed slides its deadline, and there is nothing
to create twice.

Sweeping is a task, and safe to run everywhere at once
------------------------------------------------------

The sweeper claims every organisation whose window has closed and dispatches
a compile for each. The claim is one server-side operation, so several
workers running this on the same schedule cannot take the same organisation —
that property is what makes the schedule safe to leave on every worker rather
than needing exactly one scheduler process to be alive.

It dispatches and returns. It does not wait for a compile to finish; a
sweeper blocked on a build would miss the next window.
"""

from __future__ import annotations

import logging
import time

from .app import COMPILE_TASK, SWEEP_TASK, app
from .debounce import arm, claim

logger = logging.getLogger("graphrag.worker")


def arm_organization(org_id: str, *, now: float | None = None) -> float:
    """Record an event for ``org_id`` and return when its compile is due.

    **Not a task.** See the module docstring: this is the synchronous half,
    called by whatever receives the event.
    """
    deadline = arm(org_id, now=now)
    moment = now if now is not None else time.time()
    logger.info(
        "%s: compile due in %.0fs", org_id, max(0.0, deadline - moment)
    )
    return deadline


def _fail_abandoned() -> None:
    """Release jobs whose worker died, so their tenants can build again.

    Never raises: the sweep's job is dispatching, and a control plane that
    cannot be reached this time is reached on the next tick.
    """
    try:
        from ..models.database import control_plane_sessions, create_control_plane_engine
        from .compile import fail_abandoned_jobs

        engine = create_control_plane_engine()
        try:
            with control_plane_sessions(engine)() as db:
                fail_abandoned_jobs(db)
        finally:
            engine.dispose()
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning("could not check for abandoned jobs", exc_info=True)


@app.task(name=SWEEP_TASK)
def sweep(now: float | None = None) -> int:
    """Dispatch a compile for every organisation whose window has closed.

    Returns how many were dispatched. Zero is the ordinary answer on a quiet
    fleet and is not logged — a line every thirty seconds saying nothing
    happened is a line nobody reads by the time something does.
    """
    _fail_abandoned()
    due = claim(now=now)
    if not due:
        return 0

    for org_id in due:
        # By name, so this chunk stands on its own: the task on the other end
        # is registered by the module that implements it, and this does not
        # need to import it.
        app.send_task(COMPILE_TASK, args=[org_id])

    logger.info("dispatched %d compile(s): %s", len(due), ", ".join(due))
    return len(due)
