"""Tests for M0, the trace schema.

The schema is stdlib-only on purpose, so these tests never touch pydantic or
httpx. A trace has to survive a round trip through JSON unchanged — that is the
whole contract every later milestone depends on.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.tracing import (
    capture,
    is_used,
    SCHEMA_VERSION,
    Trace,
    TraceEdge,
    TraceItem,
    load,
    to_dict,
    trace_from_dict,
    save,
)

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "example_trace.json"


def make_trace() -> Trace:
    return _trace(
        query="who changed authentication recently?",
        answer="Alice changed it in PR #1347.",
        started_at=datetime(2026, 8, 2, 10, 0, 0, tzinfo=timezone.utc),
        duration_ms=412.5,
        items=[
            TraceItem(
                id="pr:1347",
                content="Rewrite the auth middleware token check.",
                source="graph",
                score=0.91,
                overlap=0.34,
            ),
            TraceItem(
                id="issue:42",
                content="Login fails for expired tokens.",
                source="vector",
                score=0.55,
                overlap=0.02,
            ),
        ],
        edges=[
            TraceEdge(
                source="pr:1347", target="issue:42", relation="RESOLVES", weight=0.92
            )
        ],
    )



def _trace(*, query: str = "", answer=None, items=(), edges=(), **kwargs):
    """Build a flat trace the way a one-shot producer does.

    Schema 3 nests items under a Retrieval, and ``capture`` is the supported
    way to make one from a flat list — so these tests exercise the real path
    instead of assembling the dataclass by hand.
    """
    return capture(query, items, answer, edges=edges, **kwargs)



# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------


def test_trace_records_the_query_and_answer():
    trace = make_trace()

    assert trace.query == "who changed authentication recently?"
    assert trace.answer == "Alice changed it in PR #1347."


def test_item_records_content_source_and_score():
    item = make_trace().items[0]

    assert item.content == "Rewrite the auth middleware token check."
    assert item.source == "graph"
    assert item.score == 0.91


def test_items_carry_a_used_flag():
    used, ignored = make_trace().items

    assert is_used(used.overlap) is True
    assert is_used(ignored.overlap) is False


def test_used_is_unset_until_a_classifier_runs():
    item = TraceItem(id="pr:1", content="x", source="graph", score=0.5)

    assert is_used(item.overlap) is None
    assert item.overlap is None


def test_edges_are_optional():
    trace = _trace(query="q")

    assert trace.edges == []
    assert trace.items == []


def test_timing_is_recorded():
    trace = make_trace()

    assert trace.duration_ms == 412.5
    assert trace.started_at == datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------


def test_to_dict_is_json_serializable():
    payload = to_dict(make_trace())

    json.dumps(payload)  # raises if anything is not serializable


def test_timestamps_serialize_as_iso_strings():
    payload = to_dict(make_trace())

    assert payload["started_at"] == "2026-08-02T10:00:00+00:00"


def test_missing_timestamp_serializes_as_null():
    payload = to_dict(_trace(query="q"))

    assert payload["started_at"] is None


def test_dict_carries_the_schema_version():
    payload = to_dict(make_trace())

    assert payload["schema_version"] == SCHEMA_VERSION


def test_round_trip_through_a_dict_preserves_everything():
    original = make_trace()

    restored = trace_from_dict(to_dict(original))

    assert restored == original


def test_round_trip_through_a_file_preserves_everything(tmp_path):
    original = make_trace()
    path = tmp_path / "trace.json"

    save(original, path)

    assert load(path) == original


def test_written_file_is_readable_json(tmp_path):
    path = tmp_path / "trace.json"

    save(make_trace(), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["query"] == "who changed authentication recently?"
    items = [item for group in payload["retrievals"] for item in group["items"]]
    assert len(items) == 2


def test_reading_a_trace_from_a_future_version_is_refused(tmp_path):
    path = tmp_path / "trace.json"
    payload = to_dict(make_trace())
    payload["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load(path)

    assert "schema_version" in str(excinfo.value)


# --------------------------------------------------------------------------
# the hand-written example
# --------------------------------------------------------------------------


def test_the_hand_written_example_loads():
    trace = load(EXAMPLE_PATH)

    assert trace.query
    assert trace.answer
    assert trace.items


def test_the_example_shows_both_a_used_and_an_ignored_item():
    trace = load(EXAMPLE_PATH)

    assert any(is_used(item.overlap) for item in trace.items)
    assert any(is_used(item.overlap) is False for item in trace.items)


def test_the_example_exercises_every_field():
    trace = load(EXAMPLE_PATH)

    assert trace.started_at is not None
    assert trace.duration_ms is not None
    assert trace.edges, "the example should show at least one relation"
    assert all(item.score is not None for item in trace.items)
    assert all(item.overlap is not None for item in trace.items)
