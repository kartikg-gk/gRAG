"""Deciding when an organisation should be recompiled.

Nothing here compiles anything, enqueues anything, or opens a database. It
answers one question — *when* is this organisation due — and something else
acts on the answer.

Two clocks, and both are load-bearing
-------------------------------------

A repository can produce twenty events in two minutes. Compiling twenty times
is waste; compiling once, after things settle, is right. But an organisation
that never settles must not wait forever, and that is the second clock::

    deadline = min(now + window, first_seen + max_wait)

The first term slides forward on every event, which is what collapses a burst
into one compile. The second is anchored to the first event of the burst and
never moves, which is what stops a busy organisation being deferred out of
existence.

**Dropping the second term does not fail loudly.** An organisation receiving
an event every ninety seconds under a two-minute window simply never becomes
due — no error, no log line, just a graph that quietly stops being rebuilt
while every event is accepted normally.

Two keys
--------

A **sorted set** of organisations waiting, scored by the deadline above, and
a **hash** of when each was first seen in its current burst. The hash is what
anchors the second clock, and it is cleared on claim so the next burst starts
its own — otherwise max-wait would measure from the first event this
organisation ever sent.

Claiming is one operation, not two
----------------------------------

Reading the due set and then removing it would let two sweepers running at
the same moment both read the same organisations and both act on them, which
is precisely the duplicate compile this module exists to prevent. So the read
and both removals are a script the server runs as one unit.
"""

from __future__ import annotations

import threading
import time

from .config import (
    DEBOUNCE_MAX_WAIT_SECONDS,
    DEBOUNCE_WINDOW_SECONDS,
    REDIS_URL,
)

#: Organisations waiting to be compiled, scored by when they become due.
PENDING_KEY = "graphrag:compile:pending"

#: When each waiting organisation was first seen in its current burst.
FIRST_SEEN_KEY = "graphrag:compile:first-seen"

#: Read the due members, and if there are any remove them from both keys.
#:
#: One script so the whole thing is one unit: two sweepers cannot both see the
#: same organisation, because the second one runs after the first has already
#: removed what it took.
#:
#: The members are unpacked into the removal calls, which puts a ceiling on
#: how many can be claimed in one go — a few thousand, on the interpreter's
#: stack. That is far more than a sweep is ever expected to find, and the
#: alternative costs a loop for a case that would mean something else had
#: already gone badly wrong.
CLAIM_SCRIPT = """
local due = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if #due > 0 then
    redis.call('ZREM', KEYS[1], unpack(due))
    redis.call('HDEL', KEYS[2], unpack(due))
end
return due
"""

_client_lock = threading.Lock()
_client = None


def client():
    """The connection, made on first use and reused.

    **Responses are decoded to strings.** The first-seen value is read back
    and used in arithmetic, and bytes will not convert — a detail that would
    otherwise surface as a type error in the one branch that only runs when
    two events arrive for the same organisation.
    """
    global _client
    with _client_lock:
        if _client is None:
            import redis

            _client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        return _client


def set_client(connection) -> None:
    """Replace the connection. For tests, and for a caller that owns its own."""
    global _client
    with _client_lock:
        _client = connection


def arm(
    org_id: str,
    *,
    now: float | None = None,
    connection=None,
    window: float = DEBOUNCE_WINDOW_SECONDS,
    max_wait: float = DEBOUNCE_MAX_WAIT_SECONDS,
) -> float:
    """Record an event for ``org_id``, and return when it is now due.

    ``now`` is injectable because the alternative is a test that sleeps for
    two minutes to prove a two-minute window, which is a test nobody runs.
    """
    redis_client = connection if connection is not None else client()
    moment = now if now is not None else time.time()

    # Set-if-absent, in one operation that reports whether it set. Two events
    # arriving for one organisation at the same instant must not both believe
    # they were the first, or the anchor would move and the max-wait clock
    # would restart on every event.
    claimed_first = redis_client.hsetnx(FIRST_SEEN_KEY, org_id, moment)

    if claimed_first:
        first_seen = moment
    else:
        stored = redis_client.hget(FIRST_SEEN_KEY, org_id)
        # A hash entry that vanished between the two calls — a claim landing
        # in between — leaves this event as the start of the next burst.
        first_seen = float(stored) if stored is not None else moment

    deadline = min(moment + window, first_seen + max_wait)

    # Replaces any existing score, so an organisation has one entry however
    # many events it sends.
    redis_client.zadd(PENDING_KEY, {org_id: deadline})

    return deadline


def claim(*, now: float | None = None, connection=None) -> list[str]:
    """Take every organisation due at or before ``now``.

    Returns what it took, and what it took is no longer pending. The read and
    the removals are one server-side operation, so a second sweeper running
    at the same moment gets the ones this call did not.
    """
    redis_client = connection if connection is not None else client()
    moment = now if now is not None else time.time()

    return redis_client.eval(
        CLAIM_SCRIPT, 2, PENDING_KEY, FIRST_SEEN_KEY, moment
    )


def pending_count(*, connection=None) -> int:
    """How many organisations are waiting.

    The only way to see from outside whether anything is queued at all, and
    cheap enough to ask often.
    """
    redis_client = connection if connection is not None else client()
    return redis_client.zcard(PENDING_KEY)
