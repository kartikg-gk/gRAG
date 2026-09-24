"""Tests for connecting the pod agent to the application's lifecycle.

No new behaviour is under test here — boot, the loop and the pass are tested
where they live. What is under test is the wiring: whether the gate holds,
whether startup survives the agent failing, and **the order things stop in**.

The ordering test is the one to read first. Both operations happening is not
the same as them happening in the right order, and a test that only checked
both ran would pass with the dangerous ordering: stores closed underneath a
tick that is still holding their handles.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import pytest
from fastapi.testclient import TestClient

from src.api.app import POD_AGENT_ENABLED, POD_AGENT_VARIABLE, create_app
from src.registry import REGISTRY, GraphRegistry


@pytest.fixture(autouse=True)
def no_control_plane(monkeypatch):
    """Nothing configured, which is the ordinary local case.

    The schema step then fails and is swallowed, which is exactly the
    behaviour every test here is running on top of.
    """
    monkeypatch.delenv("GRAPHRAG_CONTROL_PLANE_DATABASE_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_DATABASE_URL", raising=False)


@pytest.fixture()
def gate_off(monkeypatch):
    monkeypatch.delenv(POD_AGENT_VARIABLE, raising=False)


@pytest.fixture()
def gate_on(monkeypatch):
    monkeypatch.setenv(POD_AGENT_VARIABLE, POD_AGENT_ENABLED)


class FakeEngine:
    """A stand-in for the graph this process serves from.

    The lifecycle is what is under test, not retrieval, and a real engine
    would load a model and open a native store to prove nothing about it.
    """

    path = "fake-store.lbug"
    embedder = None
    extractor = None
    judge = None
    now = None

    class store:
        @staticmethod
        def count_nodes():
            return 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def close(self):
        pass


def app_with_no_store():
    """An application whose engine is a stand-in. Routing is not under test."""
    return create_app(engine_factory=FakeEngine)


class RefuseToImport:
    """A finder that fails the test if a named module gets imported."""

    def __init__(self, name: str):
        self.name = name
        self.attempted = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            self.attempted = True
            raise AssertionError(f"{fullname} was imported")
        return None


# ==========================================================================
# the ordering, which is the point
# ==========================================================================


def test_the_agent_is_stopped_before_the_stores_are_closed(gate_on, monkeypatch):
    """Order, not merely both.

    A tick in flight holds store handles and may be swapping one. Closing the
    registry underneath it is a use-after-close, so the agent has to be
    stopped and awaited first — and the reverse ordering would satisfy any
    test that only asked whether both happened.
    """
    order: list[str] = []
    import src.pod as pod_module

    async def fake_poll(*, registry, stop, **kwargs):
        await stop.wait()
        # Recorded when the loop actually finishes, not when the signal was
        # set, so awaiting rather than merely signalling is what is asserted.
        order.append("agent stopped")

    monkeypatch.setattr(pod_module, "poll", fake_poll)
    monkeypatch.setattr(pod_module, "boot", lambda **kwargs: [])

    original_close = GraphRegistry.close_all

    def recording_close(self):
        order.append("stores closed")
        return original_close(self)

    monkeypatch.setattr(GraphRegistry, "close_all", recording_close)

    with TestClient(app_with_no_store()):
        pass

    assert order == ["agent stopped", "stores closed"]


def test_shutdown_awaits_the_task_rather_than_cancelling_it(gate_on, monkeypatch):
    """A tick mid-swap finishes. It is not torn open.

    Cancelling could leave half a downloaded artifact in the cache with a
    registry entry pointing at it, which is a state nothing else here knows
    how to repair.
    """
    import src.pod as pod_module

    finished = []

    async def fake_poll(*, registry, stop, **kwargs):
        await stop.wait()
        # Stands in for a tick that is part way through something.
        await asyncio.sleep(0.05)
        finished.append(True)

    monkeypatch.setattr(pod_module, "poll", fake_poll)
    monkeypatch.setattr(pod_module, "boot", lambda **kwargs: [])

    with TestClient(app_with_no_store()):
        pass

    assert finished == [True]


def test_a_failure_stopping_the_agent_still_closes_the_stores(
    gate_on, monkeypatch, caplog
):
    """Shutdown continues. One broken step does not strand open handles."""
    import src.pod as pod_module

    closed = []

    async def fake_poll(*, registry, stop, **kwargs):
        await stop.wait()
        raise RuntimeError("the agent came apart on the way down")

    monkeypatch.setattr(pod_module, "poll", fake_poll)
    monkeypatch.setattr(pod_module, "boot", lambda **kwargs: [])

    original_close = GraphRegistry.close_all
    monkeypatch.setattr(
        GraphRegistry,
        "close_all",
        lambda self: (closed.append(True), original_close(self))[1],
    )

    with caplog.at_level(logging.WARNING):
        with TestClient(app_with_no_store()):
            pass

    assert closed == [True]
    assert "did not stop cleanly" in caplog.text


# ==========================================================================
# the gate
# ==========================================================================


def test_with_the_gate_unset_nothing_is_started_and_nothing_is_imported(gate_off):
    """A process without the agent must not even load it.

    The module is dropped from the import cache and a finder is installed that
    fails on any attempt to bring it back, so this asserts the import did not
    happen rather than that the task did not start.
    """
    dropped = {
        name: sys.modules.pop(name)
        for name in ("src.pod", "src.reconcile")
        if name in sys.modules
    }
    refuse = RefuseToImport("src.pod")
    sys.meta_path.insert(0, refuse)
    try:
        with TestClient(app_with_no_store()) as client:
            assert client.app.state.pod_agent is None
            assert client.app.state.pod_agent_stop is None
    finally:
        sys.meta_path.remove(refuse)
        sys.modules.update(dropped)

    assert refuse.attempted is False
    assert "src.pod" not in sys.modules or dropped


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "TRUE", "1 ", " 1", "on"])
def test_only_the_exact_value_switches_the_agent_on(monkeypatch, value):
    """Anything else is off, including the values a person would expect to work.

    That is the direction to fail in: a typo produces a process with no agent
    and a log that says so, rather than a background task polling a database
    this deployment does not have.
    """
    monkeypatch.setenv(POD_AGENT_VARIABLE, value)

    with TestClient(app_with_no_store()) as client:
        assert client.app.state.pod_agent is None


def test_with_the_gate_on_boot_runs_and_a_task_is_created(gate_on, monkeypatch):
    import src.pod as pod_module

    booted = []
    running = asyncio.Event()

    def fake_boot(*, registry, **kwargs):
        booted.append(registry)
        return []

    async def fake_poll(*, registry, stop, **kwargs):
        running.set()
        await stop.wait()

    monkeypatch.setattr(pod_module, "boot", fake_boot)
    monkeypatch.setattr(pod_module, "poll", fake_poll)

    with TestClient(app_with_no_store()) as client:
        agent = client.app.state.pod_agent
        assert agent is not None
        assert not agent.done()
        assert client.app.state.pod_agent_stop is not None
        # The registry it was handed is the one the application serves from.
        assert booted == [REGISTRY]

    assert agent.done()


def test_boot_runs_off_the_event_loop(gate_on, monkeypatch):
    """It opens a session and copies files. Inline, that is a stalled process."""
    import threading

    import src.pod as pod_module

    threads = []

    def fake_boot(*, registry, **kwargs):
        threads.append(threading.current_thread().name)
        return []

    async def fake_poll(*, registry, stop, **kwargs):
        await stop.wait()

    monkeypatch.setattr(pod_module, "boot", fake_boot)
    monkeypatch.setattr(pod_module, "poll", fake_poll)

    with TestClient(app_with_no_store()):
        pass

    assert threads and threads[0] != threading.main_thread().name


# ==========================================================================
# nothing here may stop the process starting
# ==========================================================================


def test_a_boot_that_raises_still_allows_startup(gate_on, monkeypatch, caplog):
    """Unregistered and serving beats registered and refusing to start."""
    import src.pod as pod_module

    def refuse(**kwargs):
        raise RuntimeError("the control plane would not answer")

    async def fake_poll(*, registry, stop, **kwargs):
        await stop.wait()

    monkeypatch.setattr(pod_module, "boot", refuse)
    monkeypatch.setattr(pod_module, "poll", fake_poll)

    with caplog.at_level(logging.WARNING):
        with TestClient(app_with_no_store()) as client:
            assert client.get("/api/health").status_code == 200
            # The loop still starts. A boot that failed is not a reason to
            # stop trying every interval.
            assert client.app.state.pod_agent is not None

    assert "did not boot cleanly" in caplog.text


def test_the_control_plane_schema_step_failing_does_not_prevent_startup(
    gate_off, caplog
):
    """With no database configured, this is the ordinary path, not an edge."""
    with caplog.at_level(logging.WARNING):
        with TestClient(app_with_no_store()) as client:
            assert client.get("/api/health").status_code == 200

    assert "control plane is unavailable" in caplog.text


def test_a_run_with_nothing_configured_serves_exactly_as_before(gate_off):
    """The case to protect most carefully: default behaviour is untouched."""
    with TestClient(app_with_no_store()) as client:
        health = client.get("/api/health")

        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert client.app.state.pod_agent is None
        assert REGISTRY is not None
