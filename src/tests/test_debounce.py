"""Tests for deciding when an organisation is due to be recompiled.

**Against a real server, on its own database.** The property that matters
most here — that claiming is one operation and not two — is invisible to a
fake: a stand-in will happily run the read and the removals separately and
pass every assertion. So these connect to a live instance and skip, loudly,
when there is not one.

Database 15 rather than the working one, so a run never disturbs whatever a
local process has queued.

Time is passed in everywhere. A test that slept for two minutes to prove a
two-minute window is a test that would stop being run.
"""

from __future__ import annotations

import pytest

from src.worker.config import REDIS_URL
from src.worker.debounce import (
    FIRST_SEEN_KEY,
    PENDING_KEY,
    arm,
    claim,
    pending_count,
)

#: A window and a max wait chosen to make the arithmetic obvious rather than
#: to match the configured defaults, which are tested separately.
WINDOW = 120.0
MAX_WAIT = 600.0

NOW = 1_700_000_000.0


def _connection():
    """A client on the test database, or ``None`` if nothing is listening.

    Probes with a real command rather than trusting that the import worked:
    the client library installs and connects lazily, so an unreachable server
    looks perfectly healthy until something is asked of it.
    """
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
def connection():
    made = _connection()
    if made is None:
        pytest.skip(
            "no reachable queue server; the atomicity of claiming cannot be "
            "shown against a stand-in"
        )
    made.delete(PENDING_KEY, FIRST_SEEN_KEY)
    try:
        yield made
    finally:
        made.delete(PENDING_KEY, FIRST_SEEN_KEY)
        made.close()


def deadline_of(connection, org_id) -> float:
    return connection.zscore(PENDING_KEY, org_id)


def armed(connection, org_id, moment):
    return arm(
        org_id,
        now=moment,
        connection=connection,
        window=WINDOW,
        max_wait=MAX_WAIT,
    )


# ==========================================================================
# the starvation case, which is the reason for the second clock
# ==========================================================================


def test_an_organisation_armed_forever_still_becomes_due(connection):
    """Events every ninety seconds, under a two-minute window, for an hour.

    The sliding half of the deadline is pushed out by every one of them and
    on its own would defer this organisation indefinitely. The anchored half
    — first seen plus ten minutes — does not move, so the organisation is due
    at 1_700_000_600 however many events arrive after it.

    Written first because the failure it guards against is silent: nothing
    errors, no event is rejected, the graph simply stops being rebuilt.
    """
    first_seen = NOW
    deadlines = []

    # Ninety seconds apart, for an hour: forty events, every one of them
    # inside the window that the previous one opened.
    for step in range(40):
        moment = NOW + step * 90.0
        deadlines.append(armed(connection, "org_busy", moment))

    # The first few are the sliding clock: each is exactly a window away.
    assert deadlines[0] == NOW + WINDOW
    assert deadlines[1] == NOW + 90.0 + WINDOW

    # From the moment the anchored clock becomes the smaller of the two, the
    # deadline stops moving. Ten minutes after the first event, and not one
    # second later, whatever else arrives.
    assert deadlines[-1] == first_seen + MAX_WAIT == 1_700_000_600.0

    # And it really is due: a sweep at that moment takes it, while the events
    # are still arriving.
    assert claim(now=first_seen + MAX_WAIT, connection=connection) == ["org_busy"]


def test_the_deadline_is_the_earlier_of_the_two_clocks(connection):
    """Neither clock wins by default. Whichever is sooner is the deadline."""
    # Early in a burst the sliding clock is sooner.
    assert armed(connection, "org_1", NOW) == NOW + WINDOW

    # Late in one, the anchor is.
    late = NOW + MAX_WAIT - 30.0
    assert armed(connection, "org_1", late) == NOW + MAX_WAIT
    assert NOW + MAX_WAIT < late + WINDOW


# ==========================================================================
# arming
# ==========================================================================


def test_arming_makes_an_organisation_pending(connection):
    assert pending_count(connection=connection) == 0

    deadline = armed(connection, "org_1", NOW)

    assert pending_count(connection=connection) == 1
    assert deadline == NOW + WINDOW
    assert deadline_of(connection, "org_1") == deadline


def test_arming_twice_slides_the_deadline_rather_than_adding_an_entry(connection):
    """A burst is one waiting organisation, not twenty."""
    armed(connection, "org_1", NOW)
    second = armed(connection, "org_1", NOW + 30.0)

    assert pending_count(connection=connection) == 1
    assert second == NOW + 30.0 + WINDOW
    assert deadline_of(connection, "org_1") == second


def test_the_first_seen_anchor_is_not_moved_by_later_events(connection):
    """The whole second clock depends on this value staying put."""
    armed(connection, "org_1", NOW)
    armed(connection, "org_1", NOW + 30.0)
    armed(connection, "org_1", NOW + 60.0)

    assert float(connection.hget(FIRST_SEEN_KEY, "org_1")) == NOW


def test_organisations_are_tracked_apart(connection):
    armed(connection, "org_1", NOW)
    armed(connection, "org_2", NOW + 45.0)

    assert pending_count(connection=connection) == 2
    assert deadline_of(connection, "org_1") == NOW + WINDOW
    assert deadline_of(connection, "org_2") == NOW + 45.0 + WINDOW


# ==========================================================================
# claiming
# ==========================================================================


def test_claiming_takes_what_is_due_and_leaves_what_is_not(connection):
    armed(connection, "org_due", NOW)
    armed(connection, "org_later", NOW + 300.0)

    taken = claim(now=NOW + WINDOW, connection=connection)

    assert taken == ["org_due"]
    assert pending_count(connection=connection) == 1
    assert deadline_of(connection, "org_later") is not None


def test_a_deadline_exactly_now_is_due(connection):
    """At or before, not strictly before. A boundary that is off by one here
    is a compile that waits a whole sweep interval for no reason."""
    armed(connection, "org_1", NOW)

    assert claim(now=NOW + WINDOW, connection=connection) == ["org_1"]


def test_claiming_removes_what_it_returned(connection):
    """The second sweep finds nothing, which is what makes a burst one compile.

    This is also what a second sweeper running at the same moment sees. The
    read and the removals are one server-side operation, so the two cannot
    both take the same organisation — a property that only means anything
    against a real server, which is what this suite talks to.
    """
    armed(connection, "org_1", NOW)
    armed(connection, "org_2", NOW)

    assert sorted(claim(now=NOW + WINDOW, connection=connection)) == ["org_1", "org_2"]
    assert claim(now=NOW + WINDOW, connection=connection) == []
    assert pending_count(connection=connection) == 0


def test_claiming_clears_the_anchor_so_the_next_burst_starts_fresh(connection):
    """Otherwise max-wait would measure from the first event ever sent.

    An organisation that has been active for a day would then be due the
    instant it sent anything, which is the opposite of what the window is
    for.
    """
    armed(connection, "org_1", NOW)
    claim(now=NOW + WINDOW, connection=connection)

    assert connection.hget(FIRST_SEEN_KEY, "org_1") is None

    # A day later, a single event gets a full window rather than being due at
    # once.
    much_later = NOW + 86_400.0
    assert armed(connection, "org_1", much_later) == much_later + WINDOW
    assert float(connection.hget(FIRST_SEEN_KEY, "org_1")) == much_later


def test_claiming_clears_both_keys(connection):
    """Named as its own test because leaving the hash behind is invisible.

    The sorted set is what a sweep reads, so a stale hash entry breaks
    nothing today — it only makes the *next* burst inherit an anchor from the
    last one and become due immediately.
    """
    armed(connection, "org_1", NOW)
    armed(connection, "org_2", NOW)

    assert connection.hlen(FIRST_SEEN_KEY) == 2
    assert connection.zcard(PENDING_KEY) == 2

    claim(now=NOW + WINDOW, connection=connection)

    assert connection.hlen(FIRST_SEEN_KEY) == 0
    assert connection.zcard(PENDING_KEY) == 0


def test_claiming_nothing_is_not_an_error(connection):
    assert claim(now=NOW, connection=connection) == []

    armed(connection, "org_1", NOW)

    assert claim(now=NOW, connection=connection) == []
    assert pending_count(connection=connection) == 1


# ==========================================================================
# the count
# ==========================================================================


def test_the_pending_count_follows_what_is_waiting(connection):
    assert pending_count(connection=connection) == 0

    armed(connection, "org_1", NOW)
    armed(connection, "org_2", NOW)
    armed(connection, "org_1", NOW + 10.0)

    assert pending_count(connection=connection) == 2

    # The later event pushed org_1 ten seconds past org_2, so a sweep at
    # org_2's deadline takes one and leaves the other.
    assert claim(now=NOW + WINDOW, connection=connection) == ["org_2"]
    assert pending_count(connection=connection) == 1

    claim(now=NOW + 10.0 + WINDOW, connection=connection)

    assert pending_count(connection=connection) == 0


# ==========================================================================
# the settings
# ==========================================================================


def test_the_configured_values_are_the_documented_ones():
    from src.worker import config

    assert config.DEBOUNCE_WINDOW_SECONDS == 120.0
    assert config.DEBOUNCE_MAX_WAIT_SECONDS == 600.0
    assert config.REDIS_URL == "redis://localhost:6379/0"


def test_a_malformed_timing_value_falls_back_rather_than_raising(monkeypatch):
    """A stray character in a window must not stop a process starting."""
    import importlib

    from src.worker import config as config_module

    monkeypatch.setenv("GRAPHRAG_DEBOUNCE_WINDOW", "two minutes")
    reloaded = importlib.reload(config_module)
    try:
        assert reloaded.DEBOUNCE_WINDOW_SECONDS == 120.0
    finally:
        monkeypatch.delenv("GRAPHRAG_DEBOUNCE_WINDOW", raising=False)
        importlib.reload(config_module)


def test_the_compile_path_does_not_import_the_serving_path():
    """The two configuration surfaces stay apart.

    Not a rule against importing anything — the compile path legitimately
    reads the control-plane tables and raises the source client's errors.
    What it must never do is reach for the serving side's settings: that is
    the module a compile process would then have to resolve in full, for
    values about embedding models and stores it will never open.
    """
    import ast
    from pathlib import Path

    worker = Path(__file__).resolve().parents[1] / "worker"
    forbidden = ("common.config", "src.common.config", "graphrag.common.config")
    reaching_out: list[str] = []

    for source in worker.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.endswith(forbidden) or module in forbidden:
                    reaching_out.append(f"{source.name}: {module}")
                # The API package is the serving path itself, whatever it is
                # reached for.
                if "api" in module.split("."):
                    reaching_out.append(f"{source.name}: {module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.endswith(forbidden):
                        reaching_out.append(f"{source.name}: {alias.name}")

    assert reaching_out == [], reaching_out
