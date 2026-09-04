"""The queue this project's background work travels on.

One application, holding the broker connection and the schedule. It defines
no tasks itself — those live beside it and are imported by a worker on
startup through ``include``, so a worker process started against this module
knows about every task without anything importing them by hand.

The settings below are choices, not defaults
--------------------------------------------

**Acknowledged after completion, not on receipt.** The library's default
acknowledges as soon as a worker picks a task up, which means a worker that
dies mid-run takes the task with it — silently, and precisely in the case
where losing work is least acceptable. Acknowledging late means a compile
whose worker was killed is redelivered to another one. The cost is that a
task must tolerate being run twice, which a compile does: it rebuilds a graph
from the same source and produces the same artifact.

**One task at a time per worker.** Compiles are long and heavy. A worker that
prefetches four holds three of them idle for the duration of the first, and
no other worker can take them — with long tasks, prefetching is a way of
turning spare capacity into queueing delay.

**Start times are recorded**, so a task that is running can be told apart from
one that is merely queued. Without it both look identical, and the question
asked during an incident is always which of the two.

**UTC, stated twice**, as the timezone and as the flag that says to use it.
A schedule that runs on local time moves twice a year.

**A named queue**, because the broker is a shared service and the generic
default is what everything else pointed at it also uses.

Task names are explicit
-----------------------

Every task is registered under a name written down here rather than one
derived from its module path. A derived name changes when a module moves, and
the schedule below would then reference a task that no longer exists: the
scheduler keeps running, the entry keeps firing, and nothing arrives. Nothing
errors, which is what makes it worth spelling out.
"""

from __future__ import annotations

from celery import Celery
from celery.schedules import schedule

from .config import QUEUE_NAME, REDIS_URL, SWEEP_INTERVAL_SECONDS

#: The tasks, by the names they are registered under. Written here so the
#: schedule and the registration cannot drift apart -- both read these.
SWEEP_TASK = "graphrag.sweep"
COMPILE_TASK = "graphrag.compile"

#: The entry in the schedule that runs the sweeper.
SWEEP_SCHEDULE_ENTRY = "sweep-due-organisations"

app = Celery(
    "graphrag",
    broker=REDIS_URL,
    backend=REDIS_URL,
    # A worker started against this application imports this, which is what
    # registers the tasks in the process that has to run them.
    include=["src.worker.tasks", "src.worker.compile"],
)

app.conf.update(
    # See the module docstring: each of these is a decision.
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    timezone="UTC",
    enable_utc=True,
    task_default_queue=QUEUE_NAME,
    beat_schedule={
        SWEEP_SCHEDULE_ENTRY: {
            # By name, not by reference: the scheduler resolves this against
            # the registry, and it is the same string the task is registered
            # under.
            "task": SWEEP_TASK,
            "schedule": schedule(run_every=SWEEP_INTERVAL_SECONDS),
        }
    },
)
