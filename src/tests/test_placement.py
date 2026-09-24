"""Tests for choosing which process a new tenant goes to.

The first one is the one to read: a healthy pod holding nothing has to be able
to win. A grouped count has no row for it, so it is invisible to the selection
unless the counts were seeded — and the resulting bug is silent. Placement
keeps working, new capacity just never gets used.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from src.models.control_plane import (
    LOAD_PULLING,
    LOAD_READY,
    POD_BOOTING,
    POD_DEAD,
    POD_DRAINING,
    POD_READY,
    Organization,
    Pod,
    PodAssignment,
    create_control_plane_schema,
)
from src.models.database import control_plane_sessions, create_control_plane_engine
from src.placement import FALLBACK_POD, UNIDENTIFIED_POD, choose_pod

NOW = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
EPOCH_NOW = int(NOW.timestamp())


@pytest.fixture()
def engine(tmp_path):
    made = create_control_plane_engine(tmp_path / "control-plane.db")
    create_control_plane_schema(made)
    try:
        yield made
    finally:
        made.dispose()


@pytest.fixture()
def sessions(engine):
    return control_plane_sessions(engine)


def add_pod(sessions, pod_id, *, status=POD_READY, heartbeat=EPOCH_NOW):
    with sessions() as db:
        db.add(
            Pod(
                pod_id=pod_id,
                address="10.0.0.1",
                status=status,
                last_heartbeat_at=heartbeat,
                created_at=EPOCH_NOW,
            )
        )
        db.commit()
    return pod_id


def assign(sessions, pod_id, org_id, *, load_status=LOAD_READY):
    with sessions() as db:
        if db.get(Organization, org_id) is None:
            db.add(
                Organization(
                    org_id=org_id,
                    name=org_id,
                    plan="team",
                    status="active",
                    created_at=EPOCH_NOW,
                    updated_at=EPOCH_NOW,
                )
            )
            db.commit()
        db.add(
            PodAssignment(
                pod_id=pod_id,
                org_id=org_id,
                load_status=load_status,
                assigned_at=EPOCH_NOW,
            )
        )
        db.commit()


# ==========================================================================
# the one this exists for
# ==========================================================================


def test_an_empty_healthy_pod_wins_over_a_loaded_one(engine, sessions):
    """A pod with no assignments has no row in a grouped count.

    Seeded from the healthy pods first, it is a candidate at zero and wins.
    Unseeded it is invisible, and the fleet's newest and emptiest machine
    never receives a tenant — with nothing failing to say so.
    """
    add_pod(sessions, "pod_busy")
    add_pod(sessions, "pod_empty")
    assign(sessions, "pod_busy", "org_1")
    assign(sessions, "pod_busy", "org_2")

    assert choose_pod(engine=engine, now=NOW) == "pod_empty"


def test_two_empty_pods_still_resolve_to_one_answer(engine, sessions):
    """Neither has a row in the count, so the tie is broken on identity."""
    add_pod(sessions, "pod_b")
    add_pod(sessions, "pod_a")

    assert choose_pod(engine=engine, now=NOW) == "pod_a"


# ==========================================================================
# the policy
# ==========================================================================


def test_the_least_loaded_healthy_pod_wins(engine, sessions):
    add_pod(sessions, "pod_a")
    add_pod(sessions, "pod_b")
    add_pod(sessions, "pod_c")
    assign(sessions, "pod_a", "org_1")
    assign(sessions, "pod_b", "org_2")
    assign(sessions, "pod_b", "org_3")
    assign(sessions, "pod_c", "org_4")
    assign(sessions, "pod_c", "org_5")
    assign(sessions, "pod_c", "org_6")

    assert choose_pod(engine=engine, now=NOW) == "pod_a"


def test_ties_break_on_identifier_ascending(engine, sessions):
    add_pod(sessions, "pod_z")
    add_pod(sessions, "pod_m")
    add_pod(sessions, "pod_a")
    for pod_id in ("pod_z", "pod_m", "pod_a"):
        assign(sessions, pod_id, f"org_{pod_id}")

    assert choose_pod(engine=engine, now=NOW) == "pod_a"


def test_the_same_fleet_state_gives_the_same_answer(engine, sessions):
    """Which is what makes a retry land where the first attempt would have."""
    add_pod(sessions, "pod_a")
    add_pod(sessions, "pod_b")
    assign(sessions, "pod_a", "org_1")
    assign(sessions, "pod_b", "org_2")

    answers = {choose_pod(engine=engine, now=NOW) for _ in range(5)}

    assert answers == {"pod_a"}


def test_a_tenant_still_pulling_counts_toward_its_pod(engine, sessions):
    """Mid-load is still load.

    Counting only what is ready would make a pod that is halfway through three
    downloads look like the emptiest machine in the fleet and earn it a
    fourth.
    """
    add_pod(sessions, "pod_loading")
    add_pod(sessions, "pod_idle")
    assign(sessions, "pod_loading", "org_1", load_status=LOAD_PULLING)
    assign(sessions, "pod_loading", "org_2", load_status=LOAD_PULLING)

    assert choose_pod(engine=engine, now=NOW) == "pod_idle"

    # And the pulling pod is genuinely counted, not merely losing a tie.
    assign(sessions, "pod_idle", "org_3")
    assign(sessions, "pod_idle", "org_4")
    assign(sessions, "pod_idle", "org_5")

    assert choose_pod(engine=engine, now=NOW) == "pod_loading"


# ==========================================================================
# health
# ==========================================================================


@pytest.mark.parametrize("status", [POD_BOOTING, POD_DRAINING, POD_DEAD])
def test_a_pod_that_is_not_ready_is_excluded(engine, sessions, status):
    """Booting, draining and dead are all "do not send me a tenant"."""
    add_pod(sessions, "pod_unavailable", status=status)
    add_pod(sessions, "pod_ready")
    assign(sessions, "pod_ready", "org_1")

    assert choose_pod(engine=engine, now=NOW) == "pod_ready"


def test_a_stale_heartbeat_is_excluded(engine, sessions):
    stale = int((NOW - timedelta(seconds=600)).timestamp())
    add_pod(sessions, "pod_stale", heartbeat=stale)
    add_pod(sessions, "pod_beating")
    assign(sessions, "pod_beating", "org_1")

    assert choose_pod(engine=engine, now=NOW) == "pod_beating"


def test_a_heartbeat_inside_the_window_is_included(engine, sessions):
    """The boundary is a window, not the last tick."""
    recent = int((NOW - timedelta(seconds=100)).timestamp())
    add_pod(sessions, "pod_recent", heartbeat=recent)
    add_pod(sessions, "pod_busy")
    assign(sessions, "pod_busy", "org_1")

    assert choose_pod(engine=engine, now=NOW) == "pod_recent"


def test_a_pod_that_has_never_beaten_is_included(engine, sessions):
    """It has just registered. Excluding it means the fleet cannot grow."""
    add_pod(sessions, "pod_new", heartbeat=None)
    add_pod(sessions, "pod_old")
    assign(sessions, "pod_old", "org_1")

    assert choose_pod(engine=engine, now=NOW) == "pod_new"


def test_a_timestamp_with_no_timezone_does_not_raise(engine, sessions):
    """Some stores hand back naive datetimes. Read as UTC, not compared raw.

    Compared against an aware moment a naive value raises, and the failure
    path would take the whole fleet out — every tenant to the fallback because
    of one column's type.
    """
    from src.placement import _is_fresh

    assert _is_fresh(NOW.replace(tzinfo=None), NOW) is True
    assert _is_fresh((NOW - timedelta(seconds=600)).replace(tzinfo=None), NOW) is False


def test_an_unreadable_heartbeat_keeps_the_pod_in_the_running(caplog):
    """A doubtful pod costs one tenant. An empty fleet costs every tenant."""
    from src.placement import _is_fresh

    with caplog.at_level(logging.WARNING):
        assert _is_fresh("not a time at all", NOW) is True

    assert "treating the pod as fresh" in caplog.text


# ==========================================================================
# the fallback
# ==========================================================================


def test_no_healthy_pods_falls_back_rather_than_raising(engine, sessions, caplog):
    add_pod(sessions, "pod_dead", status=POD_DEAD)

    with caplog.at_level(logging.WARNING):
        assert choose_pod(engine=engine, now=NOW) == FALLBACK_POD

    assert "no healthy pod" in caplog.text


def test_an_empty_fleet_falls_back(engine):
    assert choose_pod(engine=engine, now=NOW) == FALLBACK_POD


def test_a_query_that_raises_falls_back_rather_than_raising(caplog):
    """A control plane that cannot be reached must not fail a user's action."""

    class UnreachableEngine:
        def __getattr__(self, name):
            raise RuntimeError("the control plane is unreachable")

    with caplog.at_level(logging.ERROR):
        assert choose_pod(engine=UnreachableEngine(), now=NOW) == FALLBACK_POD

    assert "could not choose a pod" in caplog.text


def test_the_fallback_is_this_process_or_a_name(engine):
    """Never an empty string.

    An assignment naming ``""`` is a row nobody can trace to a machine, which
    is worse than a placeholder that says plainly what it is.
    """
    assert FALLBACK_POD
    assert UNIDENTIFIED_POD == "pod-local"


def test_placement_writes_nothing(engine, sessions):
    """It reads and answers. Creating the assignment is the caller's job."""
    add_pod(sessions, "pod_a")
    add_pod(sessions, "pod_b")
    assign(sessions, "pod_a", "org_1")

    from sqlmodel import select

    with sessions() as db:
        before = len(db.exec(select(PodAssignment)).all())

    assert choose_pod(engine=engine, now=NOW) == "pod_b"

    with sessions() as db:
        assert len(db.exec(select(PodAssignment)).all()) == before
