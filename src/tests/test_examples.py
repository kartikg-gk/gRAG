"""Tests for the demo applications.

The offline demo is a committed artifact, so the thing most worth protecting is
that it stays byte-identical between runs. The rest checks that both demos wire
the workflow to the SDK correctly — not that retrieval is good, which it is not
meant to be.
"""

from __future__ import annotations

import json

import httpx
import pytest

from examples import fixtures, live_demo, offline_demo
from examples.workflow import answer, node_text, retrieve
from src.ingestion import API_ROOT
from src.tracing import load, to_dict, is_used

# --------------------------------------------------------------------------
# the offline demo
# --------------------------------------------------------------------------


def test_the_offline_demo_runs_without_a_network(tmp_path):
    trace = offline_demo.run(fixtures.DEFAULT_QUERY, output=tmp_path / "t.json")

    assert trace.items
    assert trace.answer


def test_the_offline_demo_is_byte_identical_between_runs(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    offline_demo.run(fixtures.DEFAULT_QUERY, output=first)
    offline_demo.run(fixtures.DEFAULT_QUERY, output=second)

    assert first.read_bytes() == second.read_bytes()


def test_the_written_artifact_reads_back_as_the_same_trace(tmp_path):
    path = tmp_path / "t.json"

    trace = offline_demo.run(fixtures.DEFAULT_QUERY, output=path)

    assert load(path) == trace


def test_the_artifact_is_json_the_viewer_understands(tmp_path):
    path = tmp_path / "t.json"
    offline_demo.run(fixtures.DEFAULT_QUERY, output=path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 3
    assert payload["retrievals"], "items nest under a retrieval in schema 3"

    items = [item for group in payload["retrievals"] for item in group["items"]]
    edges = [edge for group in payload["retrievals"] for edge in group["edges"]]
    assert items
    assert edges
    assert "used" not in items[0], "a verdict must not be persisted"
    assert "overlap" in items[0]


def test_the_demo_retrieves_more_than_the_answer_uses(tmp_path):
    """A demo where everything retrieved was used would show nothing."""
    trace = offline_demo.run(fixtures.DEFAULT_QUERY, output=tmp_path / "t.json")

    used = [item for item in trace.items if is_used(item.overlap)]
    ignored = [item for item in trace.items if is_used(item.overlap) is False]

    assert used
    assert ignored


def test_a_high_scoring_item_is_among_the_ignored(tmp_path):
    """The point of the tool: rank does not decide what reaches the answer."""
    trace = offline_demo.run(fixtures.DEFAULT_QUERY, output=tmp_path / "t.json")

    best_ignored = max(
        (item for item in trace.items if is_used(item.overlap) is False), key=lambda i: i.score
    )
    worst_used = min(
        (item for item in trace.items if is_used(item.overlap)), key=lambda i: i.score
    )

    assert best_ignored.score > worst_used.score


def test_the_demo_relations_only_connect_retrieved_items(tmp_path):
    trace = offline_demo.run(fixtures.DEFAULT_QUERY, output=tmp_path / "t.json")

    ids = {item.id for item in trace.items}
    for edge in trace.edges:
        assert edge.source in ids
        assert edge.target in ids


def test_the_offline_graph_filters_the_pull_request_from_the_issues_fixture():
    graph = offline_demo.build_graph()

    assert "ticket:101" not in graph.nodes
    assert "pr:101" in graph.nodes


def test_the_offline_graph_exercises_the_documented_skips():
    graph = offline_demo.build_graph()

    assert graph.stats.self_reviews_skipped == 1
    assert graph.stats.commits_without_author == 1


def test_the_committed_artifact_matches_a_fresh_run(tmp_path):
    """The CI check for item 9.

    ``examples/demo_trace.json`` is committed. A diff against a fresh run means
    capture or classification changed behaviour — which is exactly the signal
    the committed artifact exists to give. If this fails, either the change was
    intended (regenerate with ``python -m examples.offline_demo``) or it was
    not (fix the regression).
    """
    fresh = tmp_path / "fresh.json"
    offline_demo.run(fixtures.DEFAULT_QUERY, output=fresh)

    def content(path):
        """Text with line endings normalised.

        The committed file was written on Windows and carries CRLF; the suite
        now runs under WSL, which writes LF. A byte comparison would fail on
        that alone and say nothing about behaviour, which is what this test is
        actually for. ``.gitattributes`` pins the file to LF so the difference
        stops arising, and this keeps the test honest either way.
        """
        return path.read_text(encoding="utf-8").replace(chr(13) + chr(10), chr(10))

    assert content(fresh) == content(offline_demo.DEFAULT_OUTPUT), (
        "examples/demo_trace.json is stale; regenerate with "
        "`python -m examples.offline_demo`"
    )


def test_the_offline_demo_cli_reports_success(capsys, tmp_path):
    exit_code = offline_demo.main(["--output", str(tmp_path / "t.json")])

    assert exit_code == 0
    assert "retrieved" in capsys.readouterr().out


def test_the_offline_demo_creates_a_missing_output_directory(tmp_path):
    """A caller who named a directory meant for it to be used."""
    destination = tmp_path / "not-yet" / "t.json"

    exit_code = offline_demo.main(["--output", str(destination)])

    assert exit_code == 0
    assert destination.exists()


# --------------------------------------------------------------------------
# the workflow itself
# --------------------------------------------------------------------------


def test_node_text_prefers_the_field_that_carries_meaning():
    assert node_text({"id": "pr:1", "type": "PullRequest", "title": "Fix it"}) == "Fix it"
    assert node_text({"id": "file:a.py", "type": "File", "path": "a.py"}) == "a.py"
    assert node_text({"id": "x:1", "type": "Odd"}) == "x:1"


def test_retrieval_is_ordered_by_score_then_id():
    nodes = {
        "b:1": {"id": "b:1", "title": "alpha"},
        "a:1": {"id": "a:1", "title": "alpha"},
        "c:1": {"id": "c:1", "title": "alpha bravo"},
    }

    items, _ = retrieve(nodes, [], "alpha bravo")

    assert [item["id"] for item in items] == ["c:1", "a:1", "b:1"]


def test_retrieval_drops_items_that_match_nothing():
    nodes = {
        "a:1": {"id": "a:1", "title": "alpha"},
        "b:1": {"id": "b:1", "title": "unrelated words"},
    }

    items, _ = retrieve(nodes, [], "alpha")

    assert [item["id"] for item in items] == ["a:1"]


def test_retrieval_honours_its_limit():
    nodes = {f"n:{i}": {"id": f"n:{i}", "title": "alpha"} for i in range(10)}

    items, _ = retrieve(nodes, [], "alpha", limit=3)

    assert len(items) == 3


def test_retrieval_returns_only_relations_between_retrieved_items():
    nodes = {
        "a:1": {"id": "a:1", "title": "alpha"},
        "b:1": {"id": "b:1", "title": "alpha"},
    }
    edges = [
        {"source": "a:1", "target": "b:1", "type": "X"},
        {"source": "a:1", "target": "gone:1", "type": "Y"},
    ]

    _, related = retrieve(nodes, edges, "alpha")

    assert [edge["type"] for edge in related] == ["X"]


def test_the_answer_follows_the_lead_item_rather_than_the_ranking():
    """Otherwise the classifier would only ever confirm the template's choice."""
    items = [
        {"id": "lead", "content": "authentication token expiry", "score": 0.9},
        {"id": "high", "content": "payment webhook retries", "score": 0.8},
        {"id": "low", "content": "authentication session", "score": 0.1},
    ]

    text = answer(items, uses=2)

    assert "lead" in text
    assert "low" in text
    assert "high" not in text


def test_an_empty_result_still_produces_an_answer():
    assert "Nothing" in answer([])


# --------------------------------------------------------------------------
# the live demo, against a fake GitHub
# --------------------------------------------------------------------------


def fake_github(*, fail_with: int | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if fail_with is not None:
            return httpx.Response(fail_with, json={"message": "Not Found"})
        path = request.url.path
        if path.endswith("/reviews"):
            return httpx.Response(200, json=fixtures.REVIEWS[101])
        if path.endswith("/files"):
            return httpx.Response(200, json=fixtures.CHANGED_FILES[101])
        if path.endswith("/pulls"):
            return httpx.Response(200, json=fixtures.PULL_REQUESTS)
        if path.endswith("/issues"):
            return httpx.Response(200, json=fixtures.ISSUES)
        if path.endswith("/commits"):
            return httpx.Response(200, json=fixtures.COMMITS)
        return httpx.Response(200, json=fixtures.REPOSITORY)

    return httpx.Client(base_url=API_ROOT, transport=httpx.MockTransport(handler))


def test_the_live_demo_runs_end_to_end(capsys, tmp_path):
    exit_code = live_demo.main(
        ["acme/checkout", "--output", str(tmp_path / "t.json")],
        session=fake_github(),
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "nodes" in output
    assert "retrieved" in output


def test_the_live_demo_writes_a_trace_the_viewer_can_read(tmp_path):
    path = tmp_path / "t.json"

    live_demo.main(["acme/checkout", "--output", str(path)], session=fake_github())

    trace = load(path)
    assert trace.items
    assert trace.duration_ms is not None


def test_the_live_demo_records_real_timing(tmp_path):
    path = tmp_path / "t.json"

    live_demo.main(["acme/checkout", "--output", str(path)], session=fake_github())

    payload = to_dict(load(path))
    assert payload["started_at"] is not None
    assert payload["duration_ms"] >= 0


def test_the_live_demo_rejects_a_bad_repository_argument():
    with pytest.raises(SystemExit) as excinfo:
        live_demo.main(["not-a-slug"], session=fake_github())

    assert excinfo.value.code == 2


def test_the_live_demo_reports_a_missing_repository(capsys, tmp_path):
    exit_code = live_demo.main(
        ["acme/nope", "--output", str(tmp_path / "t.json")],
        session=fake_github(fail_with=404),
    )

    assert exit_code == 1
    assert "could not read acme/nope" in capsys.readouterr().err


def test_the_live_demo_can_skip_enrichment(tmp_path):
    path = tmp_path / "t.json"

    exit_code = live_demo.main(
        ["acme/checkout", "--no-reviews", "--no-files", "--output", str(path)],
        session=fake_github(),
    )

    assert exit_code == 0
    assert load(path).items


def test_both_demos_share_one_workflow():
    """If these diverge, the examples stop demonstrating the same integration."""
    assert live_demo.retrieve is offline_demo.retrieve
    assert live_demo.answer is offline_demo.answer
