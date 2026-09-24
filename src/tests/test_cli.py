"""Tests for the CLI entrypoint.

The CLI is a launcher, so these tests check argument handling, delegation, and
error messages — not graph or ingestion behaviour, which is covered by their own
suites.

``ingest`` talks to GitHub, so every test here passes in an ``httpx.Client``
backed by ``MockTransport``. That is the same seam the ingestion module already
uses: the session is a plain argument everywhere in this project.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.cli import main
from src.ingestion import API_ROOT
from src.tracing import TraceItem, capture, save

# --------------------------------------------------------------------------
# a fake GitHub
# --------------------------------------------------------------------------


def user() -> dict:
    return {"login": "alice", "id": 1}


def repo_payload() -> dict:
    return {
        "id": 1,
        "name": "r",
        "full_name": "o/r",
        "private": False,
        "owner": user(),
        "html_url": "https://github.com/o/r",
        "description": "test repo",
        "language": "Python",
        "default_branch": "main",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "pushed_at": "2024-01-01T00:00:00Z",
    }


def pr_payload(number: int) -> dict:
    return {
        "id": number,
        "number": number,
        "state": "open",
        "title": f"PR {number}",
        "body": "Fixes #10",
        "user": user(),
        "html_url": f"https://github.com/o/r/pull/{number}",
        "draft": False,
        "merge_commit_sha": None,
        "labels": [],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "closed_at": None,
        "merged_at": None,
    }


def issue_payload(number: int) -> dict:
    return {
        "id": number,
        "number": number,
        "state": "open",
        "title": f"Issue {number}",
        "body": None,
        "user": user(),
        "html_url": f"https://github.com/o/r/issues/{number}",
        "labels": [],
        "created_at": "2024-02-01T00:00:00Z",
        "updated_at": "2024-02-01T00:00:00Z",
        "closed_at": None,
    }


def commit_payload(sha: str) -> dict:
    return {
        "sha": sha,
        "html_url": f"https://github.com/o/r/commit/{sha}",
        "commit": {
            "message": "work",
            "author": {
                "name": "Alice",
                "email": "alice@example.com",
                "date": "2024-03-01T00:00:00Z",
            },
        },
        "author": user(),
        "parents": [],
    }


def review_payload() -> dict:
    return {
        "id": 80,
        "state": "APPROVED",
        "body": "lgtm",
        "user": {"login": "bob", "id": 2},
        "html_url": "https://github.com/o/r/pull/1#pullrequestreview-80",
        "commit_id": "abc",
        "submitted_at": "2024-04-01T00:00:00Z",
    }


def file_payload() -> dict:
    return {
        "sha": "abc",
        "filename": "src/app/main.py",
        "status": "modified",
        "additions": 5,
        "deletions": 2,
        "changes": 7,
    }


def fake_github(recorder: list | None = None, *, fail_with: int | None = None):
    """A session serving one small repository."""

    def handler(request: httpx.Request) -> httpx.Response:
        if recorder is not None:
            recorder.append(request)
        if fail_with is not None:
            return httpx.Response(fail_with, json={"message": "Not Found"})

        path = request.url.path
        if path.endswith("/reviews"):
            return httpx.Response(200, json=[review_payload()])
        if path.endswith("/files"):
            return httpx.Response(200, json=[file_payload()])
        if path.endswith("/pulls"):
            return httpx.Response(200, json=[pr_payload(1), pr_payload(2)])
        if path.endswith("/issues"):
            return httpx.Response(200, json=[issue_payload(10)])
        if path.endswith("/commits"):
            return httpx.Response(200, json=[commit_payload("abc")])
        return httpx.Response(200, json=repo_payload())

    return httpx.Client(base_url=API_ROOT, transport=httpx.MockTransport(handler))


@pytest.fixture
def saved_trace(tmp_path):
    path = tmp_path / "trace.json"
    save(
        capture(
            "who changed auth?",
            [
                TraceItem(id="a", content="alpha bravo charlie", source="graph"),
                TraceItem(id="b", content="zulu yankee", source="vector"),
            ],
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
        ),
        path,
    )
    return path


# --------------------------------------------------------------------------
# usage
# --------------------------------------------------------------------------


def test_no_command_is_a_usage_error(capsys):
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_an_unknown_command_is_a_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["frobnicate"])

    assert excinfo.value.code == 2


def test_help_lists_both_commands(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "ingest" in output
    assert "view" in output


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


def test_ingest_reports_what_it_built(capsys):
    exit_code = main(["ingest", "o/r"], session=fake_github())

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "o/r" in output
    assert "nodes" in output
    assert "edges" in output


def test_ingest_reports_per_type_counts(capsys):
    main(["ingest", "o/r"], session=fake_github())

    output = capsys.readouterr().out
    assert "pull requests" in output
    assert "issues" in output
    assert "commits" in output


def test_a_repository_argument_without_a_slash_is_rejected(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["ingest", "not-a-slug"], session=fake_github())

    assert excinfo.value.code == 2
    assert "owner/name" in capsys.readouterr().err


def test_a_repository_argument_with_too_many_parts_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main(["ingest", "a/b/c"], session=fake_github())

    assert excinfo.value.code == 2


def test_a_missing_repository_fails_with_a_clear_message(capsys):
    exit_code = main(["ingest", "o/r"], session=fake_github(fail_with=404))

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "404" in error
    assert "o/r" in error


def test_bad_credentials_fail_with_a_clear_message(capsys):
    exit_code = main(["ingest", "o/r"], session=fake_github(fail_with=401))

    assert exit_code == 1
    assert "401" in capsys.readouterr().err


def test_limits_are_passed_through_to_the_fetchers():
    requests: list[httpx.Request] = []

    main(["ingest", "o/r", "--prs", "1"], session=fake_github(requests))

    pulls = [r for r in requests if r.url.path.endswith("/pulls")]
    assert pulls[0].url.params["per_page"] == "100"


def test_reviews_and_files_are_fetched_by_default():
    requests: list[httpx.Request] = []

    main(["ingest", "o/r"], session=fake_github(requests))

    paths = [r.url.path for r in requests]
    assert any(path.endswith("/reviews") for path in paths)
    assert any(path.endswith("/files") for path in paths)


def test_no_reviews_skips_the_review_requests():
    requests: list[httpx.Request] = []

    main(["ingest", "o/r", "--no-reviews"], session=fake_github(requests))

    assert not any(r.url.path.endswith("/reviews") for r in requests)


def test_no_files_skips_the_file_requests():
    requests: list[httpx.Request] = []

    main(["ingest", "o/r", "--no-files"], session=fake_github(requests))

    assert not any(r.url.path.endswith("/files") for r in requests)


def test_output_writes_the_graph_as_json(tmp_path):
    destination = tmp_path / "graph.json"

    exit_code = main(
        ["ingest", "o/r", "--output", str(destination)], session=fake_github()
    )

    assert exit_code == 0
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["nodes"]
    assert payload["edges"]
    assert payload["stats"]["pull_requests"] == 2


def test_written_graph_json_keeps_timestamps_readable(tmp_path):
    destination = tmp_path / "graph.json"

    main(["ingest", "o/r", "--output", str(destination)], session=fake_github())

    payload = json.loads(destination.read_text(encoding="utf-8"))
    pr_node = next(node for node in payload["nodes"] if node["id"] == "pr:1")
    assert pr_node["timestamp"] == "2024-01-01T00:00:00+00:00"


def test_output_to_an_unwritable_path_fails_clearly(tmp_path, capsys):
    destination = tmp_path / "missing-dir" / "graph.json"

    exit_code = main(
        ["ingest", "o/r", "--output", str(destination)], session=fake_github()
    )

    assert exit_code == 1
    assert "could not write" in capsys.readouterr().err


# --------------------------------------------------------------------------
# ingest --store
#
# Every test here passes --no-embed. The store path is what is under test, and
# embedding would put a model download and a multi-second load into a CLI test
# that is otherwise instant. Embedding itself is asserted in test_persist.py
# against a stub embedder, where the vector's provenance is not the point.
# --------------------------------------------------------------------------


def test_store_writes_the_graph_to_a_store(tmp_path):
    ladybug = pytest.importorskip("ladybug", reason="the store needs ladybug")
    from src.graphdb import open_context_graph

    destination = tmp_path / "graph-store"

    exit_code = main(
        ["ingest", "o/r", "--store", str(destination), "--no-embed"],
        session=fake_github(),
    )

    assert exit_code == 0
    store = open_context_graph(destination)
    try:
        assert store.count_nodes() > 0
        assert store.count_documents() > 0
    finally:
        store.close()


def test_store_and_output_can_be_given_together(tmp_path):
    """The two destinations are independent, so one run can feed both."""
    pytest.importorskip("ladybug", reason="the store needs ladybug")
    from src.graphdb import open_context_graph

    json_path = tmp_path / "graph.json"
    store_path = tmp_path / "graph-store"

    exit_code = main(
        [
            "ingest",
            "o/r",
            "--output",
            str(json_path),
            "--store",
            str(store_path),
            "--no-embed",
        ],
        session=fake_github(),
    )

    assert exit_code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    store = open_context_graph(store_path)
    try:
        assert store.count_nodes() == len(payload["nodes"])
    finally:
        store.close()


def test_ingest_without_store_writes_no_store(tmp_path):
    """The flag is additive: omitting it leaves the old behaviour exactly."""
    destination = tmp_path / "graph.json"

    exit_code = main(
        ["ingest", "o/r", "--output", str(destination)], session=fake_github()
    )

    assert exit_code == 0
    assert list(tmp_path.iterdir()) == [destination]


def test_a_store_that_cannot_be_opened_fails_clearly(tmp_path, capsys):
    pytest.importorskip("ladybug", reason="the store needs ladybug")

    # A regular file where a store directory has to go.
    destination = tmp_path / "occupied"
    destination.write_text("not a store", encoding="utf-8")

    exit_code = main(
        ["ingest", "o/r", "--store", str(destination), "--no-embed"],
        session=fake_github(),
    )

    assert exit_code == 1
    assert "could not" in capsys.readouterr().err


# --------------------------------------------------------------------------
# a rate limit during enrichment aborts rather than truncating
#
# Enrichment failures normally degrade per item, which is right for a failure
# belonging to one item. A rate limit belongs to the session: continuing spends
# the remaining calls proving it, then writes a store missing every edge for
# every pull request after the limit, with exit code 0.
# --------------------------------------------------------------------------


def rate_limited_github(limit_from_pr: int):
    """Serves the repository normally until an enrichment call for limit_from_pr."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/pulls/" in path and (path.endswith("/reviews") or path.endswith("/files")):
            number = int(path.split("/pulls/")[1].split("/")[0])
            if number >= limit_from_pr:
                return httpx.Response(
                    403,
                    headers={
                        "x-ratelimit-remaining": "0",
                        "x-ratelimit-reset": "1800000000",
                    },
                    json={"message": "API rate limit exceeded"},
                )
            return httpx.Response(200, json=[review_payload()])
        if path.endswith("/pulls"):
            return httpx.Response(200, json=[pr_payload(1), pr_payload(2)])
        if path.endswith("/issues"):
            return httpx.Response(200, json=[issue_payload(10)])
        if path.endswith("/commits"):
            return httpx.Response(200, json=[commit_payload("abc")])
        return httpx.Response(200, json=repo_payload())

    return httpx.Client(base_url=API_ROOT, transport=httpx.MockTransport(handler))


def test_a_rate_limit_during_enrichment_aborts_with_a_nonzero_exit(tmp_path, capsys):
    destination = tmp_path / "graph-store"

    exit_code = main(
        ["ingest", "o/r", "--store", str(destination), "--no-embed"],
        session=rate_limited_github(2),
    )

    assert exit_code == 1
    assert "ingest failed" in capsys.readouterr().err


def test_a_rate_limit_during_enrichment_writes_no_store(tmp_path):
    """The whole point: a truncated graph must not reach disk."""
    destination = tmp_path / "graph-store"

    main(
        ["ingest", "o/r", "--store", str(destination), "--no-embed"],
        session=rate_limited_github(2),
    )

    assert not destination.exists()


def test_a_rate_limit_during_enrichment_writes_no_json_either(tmp_path):
    destination = tmp_path / "graph.json"

    exit_code = main(
        ["ingest", "o/r", "--output", str(destination)],
        session=rate_limited_github(2),
    )

    assert exit_code == 1
    assert not destination.exists()


def test_a_generic_enrichment_failure_still_completes_and_writes(tmp_path):
    """Unchanged behaviour, asserted beside the new one.

    Same position, same command; only the status the transport returns differs.
    A 500 degrades per item and the run finishes with a store on disk.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/pulls/" in path and (path.endswith("/reviews") or path.endswith("/files")):
            number = int(path.split("/pulls/")[1].split("/")[0])
            if number >= 2:
                return httpx.Response(500, json={"message": "boom"})
            return httpx.Response(200, json=[review_payload()])
        if path.endswith("/pulls"):
            return httpx.Response(200, json=[pr_payload(1), pr_payload(2)])
        if path.endswith("/issues"):
            return httpx.Response(200, json=[issue_payload(10)])
        if path.endswith("/commits"):
            return httpx.Response(200, json=[commit_payload("abc")])
        return httpx.Response(200, json=repo_payload())

    destination = tmp_path / "graph.json"

    exit_code = main(
        ["ingest", "o/r", "--output", str(destination)],
        session=httpx.Client(
            base_url=API_ROOT, transport=httpx.MockTransport(handler)
        ),
    )

    assert exit_code == 0
    assert destination.exists()


# --------------------------------------------------------------------------
# view
# --------------------------------------------------------------------------


def test_view_renders_a_saved_trace(capsys, saved_trace):
    exit_code = main(["view", str(saved_trace)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "who changed auth?" in output
    assert "a" in output
    assert "b" in output


def test_view_shows_an_unmeasured_trace_as_unclassified(capsys, saved_trace):
    """No overlap recorded means no verdict — not a verdict of ignored."""
    main(["view", str(saved_trace)])

    output = capsys.readouterr().out
    assert "unclassified" in output
    assert "0 of 2" in output


def test_view_judges_the_recorded_measurements(capsys):
    """The example stores overlaps; three of seven clear the default cutoff."""
    main(["view", "example_trace.json"])

    assert "3 of 7" in capsys.readouterr().out


def test_view_never_rewrites_the_file_it_shows(tmp_path, saved_trace):
    before = saved_trace.read_bytes()

    main(["view", "--threshold", "0.9", str(saved_trace)])

    assert saved_trace.read_bytes() == before


def test_the_same_file_reads_differently_at_a_different_threshold(capsys):
    """An archived trace is reclassified by the reader, not frozen on disk."""
    main(["view", "--threshold", "0.2", "example_trace.json"])
    at_default = capsys.readouterr().out

    main(["view", "--threshold", "0.4", "example_trace.json"])
    stricter = capsys.readouterr().out

    assert "3 of 7" in at_default
    assert "1 of 7" in stricter


def test_a_threshold_outside_zero_to_one_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main(["view", "--threshold", "1.5", "example_trace.json"])

    assert excinfo.value.code == 2


def test_view_of_a_missing_file_fails_clearly(capsys, tmp_path):
    exit_code = main(["view", str(tmp_path / "nope.json")])

    assert exit_code == 1
    assert "no such trace file" in capsys.readouterr().err


def test_view_of_a_malformed_file_fails_clearly(capsys, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    exit_code = main(["view", str(path)])

    assert exit_code == 1
    assert "not a readable trace" in capsys.readouterr().err


def test_view_of_a_future_schema_version_fails_clearly(capsys, tmp_path, saved_trace):
    payload = json.loads(saved_trace.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    path = tmp_path / "future.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["view", str(path)])

    assert exit_code == 1
    assert "schema_version" in capsys.readouterr().err
