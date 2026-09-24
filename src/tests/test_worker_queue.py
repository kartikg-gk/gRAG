"""Tests for the queue, the schedule, and the two operations on it.

Two of these guard failures that are invisible while they happen. Arming
quietly becoming a queued task would make every incoming event wait on a
broker round trip to schedule a single write to that broker. A schedule
naming a task that is not registered under that name fires forever and
delivers nothing, with no error anywhere.

Dispatch is asserted against a stand-in. The question is whether the right
calls were made with the right arguments, not whether a worker executed them
— that is the library's job and it is tested where it lives.
"""

from __future__ import annotations

import logging

import pytest

from src.worker.app import COMPILE_TASK, SWEEP_SCHEDULE_ENTRY, SWEEP_TASK, app
from src.worker.config import QUEUE_NAME, REDIS_URL, SWEEP_INTERVAL_SECONDS
from src.worker.debounce import FIRST_SEEN_KEY, PENDING_KEY, pending_count
from src.worker.tasks import _fail_abandoned as REAL_RELEASE
from src.worker.tasks import arm_organization, sweep


@pytest.fixture(autouse=True)
def no_control_plane(monkeypatch):
    """The sweep also fails abandoned jobs in the control plane; these tests
    are about dispatch and must not reach whatever database .env names."""
    import src.worker.tasks as tasks_module

    monkeypatch.setattr(tasks_module, "_fail_abandoned", lambda: None)

NOW = 1_700_000_000.0


def _connection():
    """A client on the test database, or ``None`` if nothing is listening."""
    try:
        import redis
    except ImportError:  # pragma: no cover - the library is a dependency
        return None

    try:
        made = redis.Redis.from_url(
            REDIS_URL.rsplit("/", 1)[0] + "/15",
            decode_responses=True,
            socket_connect_timeout=2,
        )
        made.ping()
    except Exception:  # noqa: BLE001 - any failure means "no server here"
        return None
    return made


@pytest.fixture()
def queue_store(monkeypatch):
    """The debounce store, on the test database, cleared either side.

    Installed as the module-wide client so the operations under test reach it
    without being handed one, which is how they run in a worker.
    """
    from src.worker import debounce

    made = _connection()
    if made is None:
        pytest.skip("no reachable broker; these operations have nothing to talk to")

    made.delete(PENDING_KEY, FIRST_SEEN_KEY)
    monkeypatch.setattr(debounce, "_client", made)
    try:
        yield made
    finally:
        made.delete(PENDING_KEY, FIRST_SEEN_KEY)
        monkeypatch.setattr(debounce, "_client", None)
        made.close()


@pytest.fixture()
def dispatched(monkeypatch):
    """Records what would have been enqueued, instead of enqueuing it."""
    sent: list[tuple[str, list]] = []

    def record(name, args=None, **kwargs):
        sent.append((name, list(args or [])))

    monkeypatch.setattr(app, "send_task", record)
    return sent


# ==========================================================================
# the two traps
# ==========================================================================


def test_arming_is_not_a_queued_task():
    """The event path stays synchronous.

    A plain function, and nothing in the registry is it. Making this a task
    later would cost every incoming event a broker round trip to schedule a
    single write to that same broker — and nothing would look broken, since
    the events would still be armed, only slower and through more machinery
    than the work itself.
    """
    assert not hasattr(arm_organization, "delay")
    assert not hasattr(arm_organization, "apply_async")
    assert callable(arm_organization)

    # Imported so the registry is complete however this file was reached:
    # a task only exists in it once its module has been imported, and an
    # assertion about "every task" has to know it is seeing every task.
    import src.worker.compile  # noqa: F401

    # Nothing this project registered wraps it, under any name. The
    # library's own built-in tasks are not this project's and are left out.
    ours = {
        name: task
        for name, task in app.tasks.items()
        if name in {SWEEP_TASK, COMPILE_TASK}
    }
    assert set(ours) == {"graphrag.sweep", "worker.tasks.reconcile_org_to_head"}
    for task in ours.values():
        assert getattr(task, "run", None) is not arm_organization


def test_the_schedule_names_a_task_that_is_registered_under_that_name():
    """The derived-name trap, caught here rather than in production silence.

    A task named after its module path changes name when the module moves.
    The schedule would then reference something that no longer exists: the
    scheduler keeps firing the entry, nothing is delivered, and nobody raises
    anything.
    """
    entry = app.conf.beat_schedule[SWEEP_SCHEDULE_ENTRY]

    assert entry["task"] == SWEEP_TASK
    assert SWEEP_TASK in app.tasks
    # The registered name is the constant, not something derived from where
    # this module happens to live today.
    assert sweep.name == SWEEP_TASK == "graphrag.sweep"
    assert "src.worker" not in sweep.name


def test_the_compile_is_dispatched_by_a_name_that_is_written_down():
    """This chunk stands alone: nothing imports the task on the other end.

    Until that task exists, the name is the whole of the contract.
    """
    assert COMPILE_TASK == "worker.tasks.reconcile_org_to_head"


# ==========================================================================
# arming
# ==========================================================================


def test_arming_returns_a_deadline_and_leaves_the_organisation_pending(queue_store):
    deadline = arm_organization("org_1", now=NOW)

    assert deadline > NOW
    assert pending_count() == 1
    assert queue_store.zscore(PENDING_KEY, "org_1") == deadline


def test_arming_twice_slides_rather_than_queueing_twice(queue_store):
    """Idempotent by construction: there is nothing to create a second time."""
    first = arm_organization("org_1", now=NOW)
    second = arm_organization("org_1", now=NOW + 30.0)

    assert second > first
    assert pending_count() == 1


def test_arming_says_how_long_until_it_fires(queue_store, caplog):
    with caplog.at_level(logging.INFO):
        arm_organization("org_1", now=NOW)

    assert "org_1" in caplog.text
    assert "due in" in caplog.text


# ==========================================================================
# sweeping
# ==========================================================================


def test_the_sweep_dispatches_one_compile_per_due_organisation(
    queue_store, dispatched
):
    arm_organization("org_a", now=NOW)
    arm_organization("org_b", now=NOW)
    arm_organization("org_later", now=NOW + 300.0)

    count = sweep(now=NOW + 200.0)

    assert count == 2
    assert sorted(dispatched) == [
        (COMPILE_TASK, ["org_a"]),
        (COMPILE_TASK, ["org_b"]),
    ]
    # The one whose window has not closed is still waiting.
    assert pending_count() == 1


def test_a_quiet_sweep_dispatches_nothing_and_returns_zero(queue_store, dispatched):
    """The ordinary case on a fleet with nothing happening."""
    assert sweep(now=NOW) == 0
    assert dispatched == []


def test_a_sweep_before_the_window_closes_dispatches_nothing(queue_store, dispatched):
    arm_organization("org_1", now=NOW)

    assert sweep(now=NOW + 10.0) == 0
    assert dispatched == []
    assert pending_count() == 1


def test_what_the_sweep_claimed_is_not_claimed_again(queue_store, dispatched):
    """A burst produces one compile, not one per sweep that sees it."""
    arm_organization("org_1", now=NOW)

    assert sweep(now=NOW + 200.0) == 1
    assert sweep(now=NOW + 200.0) == 0
    assert len(dispatched) == 1


def test_the_sweep_does_not_wait_for_the_compiles_it_dispatches(
    queue_store, monkeypatch
):
    """It dispatches and returns.

    A sweeper that waited on a build would miss every window that closed
    while it was waiting.
    """
    handles = []

    class Handle:
        def get(self, *args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("the sweep waited for a result")

    def record(name, args=None, **kwargs):
        handle = Handle()
        handles.append(handle)
        return handle

    monkeypatch.setattr(app, "send_task", record)
    arm_organization("org_1", now=NOW)

    assert sweep(now=NOW + 200.0) == 1
    assert len(handles) == 1


# ==========================================================================
# the settings, each of which is a decision
# ==========================================================================


def test_tasks_are_acknowledged_after_they_finish():
    """A worker that dies mid-compile must not take the work with it.

    The default acknowledges on receipt, which loses the task exactly when a
    worker crashes — the one case where redelivery is the whole point.
    """
    assert app.conf.task_acks_late is True


def test_a_worker_holds_one_task_at_a_time():
    """Compiles are long, and prefetching turns spare capacity into delay.

    A worker holding four runs one and keeps three where no idle worker can
    reach them.
    """
    assert app.conf.worker_prefetch_multiplier == 1


def test_a_running_task_is_distinguishable_from_a_queued_one():
    assert app.conf.task_track_started is True


def test_the_schedule_and_the_clock_are_explicit():
    """A schedule on local time moves twice a year."""
    assert app.conf.timezone == "UTC"
    assert app.conf.enable_utc is True


def test_the_queue_carries_this_project_name():
    """The broker is shared; the generic default is what everything else uses."""
    assert app.conf.task_default_queue == QUEUE_NAME == "graphrag"


def test_the_broker_and_the_result_backend_are_the_one_service():
    """One instance, doing both jobs and holding the debounce state as well."""
    assert app.conf.broker_url == REDIS_URL
    assert app.conf.result_backend == REDIS_URL


def test_the_sweep_runs_on_the_configured_interval():
    entry = app.conf.beat_schedule[SWEEP_SCHEDULE_ENTRY]

    assert entry["schedule"].run_every.total_seconds() == SWEEP_INTERVAL_SECONDS
    assert SWEEP_INTERVAL_SECONDS == 30.0


def test_a_worker_imports_the_module_that_registers_the_tasks():
    """Otherwise a worker starts, connects, and knows about nothing."""
    assert "src.worker.tasks" in app.conf.include


def test_the_sweep_releases_abandoned_jobs_before_dispatching(monkeypatch, queue_store, dispatched):
    import src.worker.tasks as tasks_module

    order = []
    monkeypatch.setattr(tasks_module, "_fail_abandoned", lambda: order.append("released"))

    sweep()

    assert order == ["released"]


def test_releasing_survives_an_unreachable_control_plane(monkeypatch, caplog):
    import src.models.database as database

    def down(*_args, **_kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(database, "create_control_plane_engine", down)

    with caplog.at_level("WARNING"):
        REAL_RELEASE()

    assert "abandoned jobs" in caplog.text
