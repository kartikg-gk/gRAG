"""Tests for the one sort point.

Arrival order reaches the output:
entity labels are first-seen-wins, and runs are diffed against committed
output. So the ordering has to be a property of the data, not of the order
GitHub happened to return things in.

The key must be *total*. A timestamp alone is not — equal timestamps are
common, and two items that compare equal can come back in either order — so
every test that matters here uses items sharing a timestamp.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from src.ingestion import in_ingest_order, ingest_key
from src.tests.test_ingestion_wiring import (
    commit_payload,
    issue_payload,
    person,
    pr_payload,
)
from src.ingestion.models import Commit, Issue, PullRequest

ALICE = person("alice", 1)
NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


# The shared payload builders derive created_at from the item number, which
# stops being a valid date past 9. These tests need dozens of items, so both
# timestamps are set explicitly here.
CREATED = datetime(2024, 1, 1, tzinfo=timezone.utc)


def pull_request(number: int, updated: datetime | None = NOW) -> PullRequest:
    payload = pr_payload(number, ALICE, "")
    payload["created_at"] = CREATED.isoformat()
    payload["updated_at"] = updated.isoformat() if updated else None
    return PullRequest.model_validate(payload)


def issue(number: int, updated: datetime | None = NOW) -> Issue:
    payload = issue_payload(number, ALICE)
    payload["created_at"] = CREATED.isoformat()
    payload["updated_at"] = updated.isoformat() if updated else None
    return Issue.model_validate(payload)


def commit(sha: str, date: datetime | None = NOW) -> Commit:
    payload = commit_payload(sha, linked=True)
    payload["commit"]["author"]["date"] = date.isoformat() if date else None
    return Commit.model_validate(payload)


# --------------------------------------------------------------------------
# the key is total
# --------------------------------------------------------------------------


def test_two_items_with_identical_timestamps_sort_by_the_tiebreak():
    """The case a timestamp-only key leaves undefined."""
    items = [pull_request(3), pull_request(1), pull_request(2)]

    assert [p.number for p in in_ingest_order(items)] == [3, 2, 1]


def test_commits_with_identical_timestamps_sort_by_sha():
    items = [commit("ccc"), commit("aaa"), commit("bbb")]

    assert [c.sha for c in in_ingest_order(items)] == ["ccc", "bbb", "aaa"]


def test_no_two_distinct_items_produce_the_same_key():
    """What "total" means, asserted directly."""
    items = [pull_request(n) for n in range(1, 20)]

    keys = [ingest_key(item) for item in items]

    assert len(set(keys)) == len(keys)


def test_the_timestamp_dominates_the_tiebreak():
    """The tiebreak only settles ties. It must not reorder different times."""
    older_but_higher_number = pull_request(99, NOW - timedelta(days=5))
    newer_but_lower_number = pull_request(1, NOW)

    ordered = in_ingest_order([older_but_higher_number, newer_but_lower_number])

    assert [p.number for p in ordered] == [1, 99]


# --------------------------------------------------------------------------
# order does not depend on arrival order
# --------------------------------------------------------------------------


def test_shuffled_input_produces_identical_order():
    """The property the whole module exists for."""
    items = [pull_request(n, NOW - timedelta(hours=n % 4)) for n in range(1, 30)]
    expected = [p.number for p in in_ingest_order(items)]

    generator = random.Random(20260301)
    for _ in range(20):
        shuffled = items[:]
        generator.shuffle(shuffled)
        assert [p.number for p in in_ingest_order(shuffled)] == expected


def test_every_permutation_of_a_tied_group_agrees():
    """Not a sampled shuffle — all of them, on items that all tie."""
    from itertools import permutations

    items = [pull_request(n) for n in (1, 2, 3, 4)]
    results = {
        tuple(p.number for p in in_ingest_order(order))
        for order in permutations(items)
    }

    assert len(results) == 1


def test_reversing_the_input_changes_nothing():
    items = [issue(n, NOW - timedelta(minutes=n)) for n in range(1, 12)]

    assert in_ingest_order(items) == in_ingest_order(list(reversed(items)))


# --------------------------------------------------------------------------
# items without a timestamp
# --------------------------------------------------------------------------


def test_an_item_with_no_timestamp_still_sorts_deterministically():
    """A commit payload can lack an author date. That must not raise."""
    items = [commit("bbb", None), commit("aaa", None), commit("ccc", NOW)]

    ordered = in_ingest_order(items)

    assert [c.sha for c in ordered] == ["ccc", "bbb", "aaa"]


def test_a_naive_timestamp_does_not_break_the_comparison():
    """Mixing naive and aware datetimes raises TypeError on comparison."""
    naive = pull_request(1, datetime(2026, 3, 1, 12, 0))
    aware = pull_request(2, NOW)

    assert len(in_ingest_order([naive, aware])) == 2


def test_an_unknown_type_is_refused_rather_than_sorted_arbitrarily():
    with pytest.raises(TypeError, match="no ingest ordering"):
        in_ingest_order(["not a model"])


# --------------------------------------------------------------------------
# there is one sort point, not one per call site
# --------------------------------------------------------------------------


def test_nothing_in_the_ingest_path_sorts_fetched_items_itself():
    """A sort repeated per call site is one that will eventually disagree.

    Scans real code rather than trusting the convention: any second sort of
    fetched items would be a place this module could be bypassed.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    offenders = []

    for path in [
        root / "src" / "cli.py",
        root / "examples" / "live_demo.py",
        *(root / "src" / "ingestion").glob("*.py"),
    ]:
        if path.name == "order.py":
            continue  # the one place that is allowed to sort
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                if name in {"sorted", "sort"}:
                    offenders.append(
                        (path.name, lines[node.lineno - 1].strip())
                    )

    # One known sort remains, and it is not fetched-item ordering: the CLI
    # sorts the enrichment-failure stages so the report reads consistently.
    # Matched on the source line rather than a line number, which would make
    # this test fail on any edit above it.
    assert offenders == [
        ("cli.py", "for stage, count in sorted(stats.enrichment_failures_by_stage.items())")
    ], offenders


def test_both_entry_points_use_the_shared_sort():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for path in (root / "src" / "cli.py", root / "examples" / "live_demo.py"):
        assert "in_ingest_order" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# the parameters actually put on the wire
# --------------------------------------------------------------------------


def captured_params(fetch, **kwargs) -> list[dict]:
    """Query parameters of every request one fetch makes."""
    import httpx

    seen: list[dict] = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=[])

    session = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )
    list(fetch(session, "acme/checkout", **kwargs))
    return seen


@pytest.mark.parametrize("fetch_name", ["fetch_pull_requests", "fetch_issues"])
def test_list_endpoints_send_explicit_ordering_parameters(fetch_name):
    """Server defaults are not part of the API contract and have changed."""
    from src import ingestion

    params = captured_params(getattr(ingestion, fetch_name))

    assert params, "no request was made"
    assert params[0]["sort"] == "updated"
    assert params[0]["direction"] == "desc"
    assert params[0]["state"] == "all"


@pytest.mark.parametrize(
    "fetch_name", ["fetch_pull_requests", "fetch_issues", "fetch_commits"]
)
def test_every_list_endpoint_asks_for_the_maximum_page_size(fetch_name):
    """100 is GitHub's cap. Less multiplies requests for no benefit."""
    from src import ingestion

    params = captured_params(getattr(ingestion, fetch_name))

    assert params[0]["per_page"] == "100"


def test_the_commits_endpoint_sends_no_sort_it_would_ignore():
    """It accepts neither sort nor state; sending them is noise on the wire."""
    from src import ingestion

    params = captured_params(ingestion.fetch_commits)

    assert "sort" not in params[0]
    assert "state" not in params[0]


# --------------------------------------------------------------------------
# end to end: the same source data ingests identically every time
# --------------------------------------------------------------------------


def test_three_ingest_runs_produce_byte_identical_node_order(tmp_path):
    """The test that existed before the sort. It must still pass."""
    import contextlib
    import io
    import json

    from src.cli import main
    from src.tests.test_ingestion_wiring import fake_github

    orders = []
    for run in range(3):
        destination = tmp_path / f"graph{run}.json"
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(
                ["ingest", "acme/checkout", "--output", str(destination)],
                session=fake_github(),
            ) == 0
        payload = json.loads(destination.read_text(encoding="utf-8"))
        orders.append([node["id"] for node in payload["nodes"]])

    assert orders[0] == orders[1] == orders[2]


def test_shuffled_fetch_results_produce_the_same_corpus_order():
    """The whole point, end to end rather than on the sort function alone."""
    from src.knowledge import GraphBuilder

    pulls = [pull_request(n, NOW - timedelta(hours=n % 3)) for n in range(1, 10)]

    def node_ids(items):
        builder = GraphBuilder()
        builder.build(pull_requests=in_ingest_order(items))
        return list(builder.nodes)

    expected = node_ids(pulls)
    generator = random.Random(7)
    for _ in range(10):
        shuffled = pulls[:]
        generator.shuffle(shuffled)
        assert node_ids(shuffled) == expected
