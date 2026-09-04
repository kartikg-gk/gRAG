"""One compile at a time, per organisation.

The second of three mechanisms that keep a burst of activity from becoming a
pile of duplicate builds, and each of the three catches something the others
structurally cannot:

The **debounce window** collapses a burst of *events* into one scheduled
compile. It cannot help once two compiles are legitimately due — both were
claimed fairly, and the window has already done its job.

**This lock** stops two workers holding the same organisation at once. It
cannot help when it expires under a worker that is still running, which is
exactly what its safety release is for.

The **one-in-flight-job constraint** in the control plane catches that last
case: a worker whose lock has expired can still be refused a second job row
by the database.

Remove any one and a specific hole opens. None of them is redundant.

Acquiring, and the value that is stored
---------------------------------------

Set-if-absent with an expiry, in one operation, so two workers arriving
together cannot both believe they set it.

The value stored is **unique to this acquisition** rather than a constant.
That is what makes releasing safe: a holder that finishes after its lock
expired — and after another worker acquired it — compares the stored value to
its own, sees somebody else's, and leaves it alone. Deleting it would hand a
third worker a lock the second still thinks it holds.

Not acquiring is not an error
-----------------------------

The context manager yields whether it got the lock. It does not raise and it
does not wait. A compile that cannot get the lock is not a failure: another
one for that organisation is running right now, and the sweeper will bring
this organisation round again.

Releasing is one operation
--------------------------

Read the value and then delete it, and a lock that expired in between can
still be deleted by whoever is finishing late: the check passes, the key turns
over, and the delete lands on somebody else's lock. The exposure is only the
gap between two round trips, and the consequence is bounded — the database
still refuses a second job row for an organisation already building.

It is closed anyway. The comparison and the delete are a script the server
runs as one unit, the same shape the sweeper's claim already uses, so removing
the last case where this lock does not hold needed nothing that was not here
already.

"""

from __future__ import annotations

import contextlib
import logging
import uuid

from .debounce import client

logger = logging.getLogger("graphrag.worker.locks")

#: How long a lock survives without anybody releasing it.
#:
#: A safety release for a worker that died mid-compile, not a timeout for
#: ordinary work: nothing here interrupts a compile that runs past it. Without
#: an expiry, one crashed worker blocks its organisation permanently and the
#: only cure is somebody noticing and deleting a key by hand.
LOCK_EXPIRY_SECONDS = 1800


#: Delete this lock, but only if the value is still the one that was written.
#:
#: One unit, so nothing can slip between the comparison and the delete. The
#: reply says whether anything was removed, which is what tells a late
#: finisher that its lock had already turned over.
RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


def lock_key(org_id: str) -> str:
    """Where the lock for one organisation lives."""
    return f"graphrag:compile:lock:{org_id}"


@contextlib.contextmanager
def compile_lock(org_id: str, *, connection=None, expiry: int = LOCK_EXPIRY_SECONDS):
    """Hold the compile lock for ``org_id`` if it is free.

    Yields ``True`` when this call acquired it and ``False`` when somebody
    else holds it. The caller decides what the second means — see the module
    docstring for why that is not an exception.
    """
    redis_client = connection if connection is not None else client()
    key = lock_key(org_id)
    # This acquisition, not this organisation. Two holders in sequence have
    # different values, which is what the release below compares against.
    token = uuid.uuid4().hex

    acquired = bool(redis_client.set(key, token, nx=True, ex=expiry))

    try:
        yield acquired
    finally:
        if acquired:
            _release(redis_client, key, token)


def _release(redis_client, key: str, token: str) -> None:
    """Delete the lock, but only if this holder still owns it.

    A holder that ran past the expiry no longer owns anything: the key it can
    see belongs to whoever acquired it next, and deleting that would release a
    lock somebody else is relying on.

    The comparison and the delete are one server-side operation, so there is
    no moment between them for the key to change hands.
    """
    removed = redis_client.eval(RELEASE_SCRIPT, 1, key, token)
    if not removed:
        logger.warning(
            "the compile lock at %s was no longer this worker's to release; "
            "leaving it alone",
            key,
        )
