"""Settings for the compile path.

**Deliberately not the serving path's configuration module.** The two run as
separate processes and a deployment can have one without the other: a serving
process that had to resolve a queue address it will never dial, or a compile
process that had to resolve an embedding model it will never load, would each
be able to fail on a setting that has nothing to do with what it does.

Every value is read at import, the way the serving side reads its own, so a
misconfigured deployment fails when the process starts rather than on the
first event it receives.
"""

from __future__ import annotations

import os

from ..common.environment import load_environment

# Before any setting below is read: they are read at import.
load_environment()


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value if value is not None and value.strip() else default


def _env_float(name: str, default: float) -> float:
    """A number, or the default if what is set is not one.

    A malformed value falls back rather than raising: these are timing
    constants, and a process that will not start because a window was typed
    with a stray character is worse than one that runs on the documented
    number and says nothing. The wrong value is visible in the behaviour; a
    process that never starts is visible only in a log nobody is reading yet.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


#: Where the queue's data structures live.
#:
#: A default that points at a local instance, unlike the control plane, which
#: has none. The difference is what each holds: this is transient scheduling
#: state that is rebuilt by the next event, and a process that quietly used a
#: local one would at worst debounce nothing. The control plane holds which
#: tenant is served what, and a process quietly using a private copy of that
#: would serve the wrong graph.
REDIS_URL = _env_str("GRAPHRAG_REDIS_URL", "redis://localhost:6379/0")

#: How long after the most recent event an organisation waits before it is
#: due. Every further event pushes this out again, which is what turns a
#: burst of twenty into one compile.
DEBOUNCE_WINDOW_SECONDS = _env_float("GRAPHRAG_DEBOUNCE_WINDOW", 120.0)

#: The longest an organisation waits from the first event of a burst,
#: whatever arrives afterwards.
#:
#: This is the whole defence against starvation. Without it, an organisation
#: receiving an event every ninety seconds under a two-minute window is
#: pushed out forever and never compiles at all — and nothing errors, so the
#: only symptom is a graph that silently stops being rebuilt.
DEBOUNCE_MAX_WAIT_SECONDS = _env_float("GRAPHRAG_DEBOUNCE_MAX_WAIT", 600.0)

#: The queue this project's tasks travel on.
#:
#: Named rather than the library's generic default, because the broker is a
#: shared service: anything else pointed at the same instance with default
#: settings would be reading and writing the same queue, and the first symptom
#: would be a task disappearing into a worker that has no idea what it is.
QUEUE_NAME = _env_str("GRAPHRAG_QUEUE_NAME", "graphrag")

#: How often the sweeper looks for organisations whose window has closed.
#:
#: Half a minute, against a two-minute window: short enough that the delay it
#: adds is a rounding error on the wait it is checking, long enough that a
#: quiet fleet is doing one cheap read a minute rather than one a second.
SWEEP_INTERVAL_SECONDS = _env_float("GRAPHRAG_SWEEP_INTERVAL", 30.0)

#: How long a compile lock survives when nobody releases it.
#:
#: Half an hour. A safety release for a worker that died mid-compile, so it has
#: to outlast any compile that is still running: a lock that expires under a
#: live worker lets a second one start on the same organisation.
COMPILE_LOCK_TTL = _env_float("GRAPHRAG_COMPILE_LOCK_TTL", 1800.0)
