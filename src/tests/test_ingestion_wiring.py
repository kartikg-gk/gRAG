"""Integration tests: GitHub client wired to the graph builder.

These exercise the real pipeline — real pagination, real normalization, real
graph construction — with only the network replaced. What is under test is the
connection between components, not the components themselves: scoring rules,
model validation, and Link-header walking each have their own unit suite.

The fixture repository is small but deliberately shaped so that one run
produces every node type and every relation type the builder can emit, plus
each documented skip case.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion import (
    API_ROOT,
    GitHubError,
    collect_by_pull_request,
    fetch_changed_files,
    fetch_commits,
    fetch_issues,
    fetch_pull_requests,
    fetch_repository,
    fetch_reviews,
)
from src.knowledge import (
    RELATION_AUTHORED,
    NODE_COMMIT,
    NODE_FILE,
    NODE_TICKET,
    RELATION_PART_OF,
    NODE_PR,
    RELATION_REPORTED,
    NODE_REPO,
    RELATION_RESOLVES,
    RELATION_REVIEWED,
    RELATION_TOUCHES,
    NODE_PERSON,
    GraphBuilder,
)

# --------------------------------------------------------------------------
# the fixture repository
#
# alice  opened pr:1 ("Fixes #10") and wrote commit:abc
# bob    opened pr:2 and reviewed pr:1
# carol  reported issue:10
# pr:2   was reviewed by bob, who also opened it -> self-review, skipped
# both pull requests touch the same file
# commit:def has no linked account
# the issues endpoint also returns a pull request, which must be filtered
# pull requests arrive across two pages
# --------------------------------------------------------------------------


def person(login: str, id_: int) -> dict:
    return {"login": login, "id": id_}


ALICE = person("alice", 1)
BOB = person("bob", 2)
CAROL = person("carol", 3)


def repo_payload() -> dict:
    return {
        "id": 100,
        "name": "r",
        "full_name": "o/r",
        "private": False,
        "owner": ALICE,
        "html_url": "https://github.com/o/r",
        "description": "fixture repo",
        "language": "Python",
        "default_branch": "main",
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2024-06-01T00:00:00Z",
        "pushed_at": "2024-06-01T00:00:00Z",
    }


def pr_payload(number: int, author: dict, body: str | None) -> dict:
    return {
        "id": number,
        "number": number,
        "state": "open",
        "title": f"PR {number}",
        "body": body,
        "user": author,
        "html_url": f"https://github.com/o/r/pull/{number}",
        "draft": False,
        "merge_commit_sha": None,
        "labels": [],
        "created_at": f"2024-0{number}-01T00:00:00Z",
        "updated_at": f"2024-0{number}-02T00:00:00Z",
        "closed_at": None,
        "merged_at": None,
    }


def issue_payload(number: int, author: dict, *, is_pull_request: bool = False) -> dict:
    payload = {
        "id": number,
        "number": number,
        "state": "closed",
        "title": f"Issue {number}",
        "body": None,
        "user": author,
        "html_url": f"https://github.com/o/r/issues/{number}",
        "labels": [],
        "created_at": "2024-04-01T00:00:00Z",
        "updated_at": "2024-04-02T00:00:00Z",
        "closed_at": "2024-04-03T00:00:00Z",
    }
    if is_pull_request:
        payload["pull_request"] = {"url": "https://api.github.com/repos/o/r/pulls/1"}
    return payload


def commit_payload(sha: str, *, linked: bool) -> dict:
    return {
        "sha": sha,
        "html_url": f"https://github.com/o/r/commit/{sha}",
        "commit": {
            "message": f"work in {sha}",
            "author": {
                "name": "Alice Example",
                "email": "alice@example.com",
                "date": "2024-05-01T00:00:00Z",
            },
        },
        "author": ALICE if linked else None,
        "parents": [],
    }


def review_payload(review_id: int, reviewer: dict) -> dict:
    return {
        "id": review_id,
        "state": "APPROVED",
        "body": "lgtm",
        "user": reviewer,
        "html_url": f"https://github.com/o/r/pull/1#pullrequestreview-{review_id}",
        "commit_id": "abc",
        "submitted_at": "2024-06-01T00:00:00Z",
    }


def file_payload(filename: str) -> dict:
    return {
        "sha": "blob",
        "filename": filename,
        "status": "modified",
        "additions": 12,
        "deletions": 3,
        "changes": 15,
        "patch": "@@ -1 +1 @@",
    }


PR_PAGES = [
    [pr_payload(1, ALICE, "Fixes #10")],
    [pr_payload(2, BOB, None)],
]

REVIEWS = {
    1: [review_payload(80, BOB)],
    2: [review_payload(81, BOB)],  # bob opened pr:2 — this is a self-review
}


def fake_github(recorder: list | None = None, *, broken: set[str] | None = None):
    """A session serving the fixture repository.

    ``broken`` names path suffixes that should fail, for the degradation tests.
    """
    broken = broken or set()

    def handler(request: httpx.Request) -> httpx.Response:
        if recorder is not None:
            recorder.append(request)
        path = request.url.path

        for suffix in broken:
            if path.endswith(suffix):
                return httpx.Response(500, json={"message": "upstream is unhappy"})

        if path.endswith("/reviews"):
            number = int(path.split("/pulls/")[1].split("/")[0])
            return httpx.Response(200, json=REVIEWS.get(number, []))
        if path.endswith("/files"):
            return httpx.Response(200, json=[file_payload("src/auth.py")])
        if path.endswith("/pulls"):
            page = int(request.url.params.get("page", 1))
            headers = {}
            if page < len(PR_PAGES):
                next_url = str(request.url.copy_set_param("page", page + 1))
                headers["Link"] = f'<{next_url}>; rel="next"'
            return httpx.Response(200, json=PR_PAGES[page - 1], headers=headers)
        if path.endswith("/issues"):
            return httpx.Response(
                200,
                json=[
                    issue_payload(10, CAROL),
                    issue_payload(11, BOB, is_pull_request=True),
                ],
            )
        if path.endswith("/commits"):
            return httpx.Response(
                200,
                json=[commit_payload("abc", linked=True), commit_payload("def", linked=False)],
            )
        return httpx.Response(200, json=repo_payload())

    return httpx.Client(base_url=API_ROOT, transport=httpx.MockTransport(handler))


def ingest(
    session: httpx.Client,
    *,
    reviews: bool = True,
    files: bool = True,
    prs: int | None = None,
) -> GraphBuilder:
    """The production sequence: fetch everything, then build the graph."""
    repository = fetch_repository(session, "o/r")
    pull_requests = list(fetch_pull_requests(session, "o/r", limit=prs))
    numbers = [pr.number for pr in pull_requests]

    review_map, review_failures = (
        collect_by_pull_request(fetch_reviews, session, "o/r", numbers)
        if reviews
        else ({}, 0)
    )
    file_map, file_failures = (
        collect_by_pull_request(fetch_changed_files, session, "o/r", numbers)
        if files
        else ({}, 0)
    )

    builder = GraphBuilder()
    builder.build(
        repository=repository,
        pull_requests=pull_requests,
        issues=fetch_issues(session, "o/r"),
        commits=fetch_commits(session, "o/r"),
        reviews=review_map,
        changed_files=file_map,
        enrichment_failures=review_failures + file_failures,
    )
    return builder


@pytest.fixture
def graph() -> GraphBuilder:
    return ingest(fake_github())


def edges_of(builder: GraphBuilder, relation: str) -> list[dict]:
    return [edge for edge in builder.edges if edge["type"] == relation]


def types_of(builder: GraphBuilder) -> set[str]:
    return {node["type"] for node in builder.nodes.values()}


# ==========================================================================
# Everything the pipeline can produce, produced
# ==========================================================================


def test_every_node_type_is_produced(graph):
    assert types_of(graph) == {
        NODE_REPO,
        NODE_PR,
        NODE_TICKET,
        NODE_COMMIT,
        NODE_PERSON,
        NODE_FILE,
    }


def test_every_relation_type_is_produced(graph):
    produced = {edge["type"] for edge in graph.edges}

    assert produced == {RELATION_AUTHORED, RELATION_REPORTED, RELATION_REVIEWED, RELATION_TOUCHES, RELATION_PART_OF, RELATION_RESOLVES}


def test_the_expected_nodes_exist_by_id(graph):
    assert set(graph.nodes) == {
        "repo:o/r",
        "pr:1",
        "pr:2",
        "ticket:10",
        "commit:abc",
        "commit:def",
        "person:alice",
        "person:bob",
        "person:carol",
        "file:src/auth.py",
    }


# ==========================================================================
# Data flowing across the boundary
# ==========================================================================


def test_pull_request_authorship_reaches_the_graph(graph):
    authored = {(e["source"], e["target"]) for e in edges_of(graph, RELATION_AUTHORED)}

    assert ("person:alice", "pr:1") in authored
    assert ("person:bob", "pr:2") in authored


def test_issue_authorship_arrives_as_reporting(graph):
    reported = edges_of(graph, RELATION_REPORTED)

    assert [(e["source"], e["target"]) for e in reported] == [("person:carol", "ticket:10")]


def test_commit_authorship_reaches_the_graph(graph):
    authored = {(e["source"], e["target"]) for e in edges_of(graph, RELATION_AUTHORED)}

    assert ("person:alice", "commit:abc") in authored


def test_a_closing_keyword_in_a_pull_request_body_reaches_the_graph(graph):
    resolves = edges_of(graph, RELATION_RESOLVES)

    assert [(e["source"], e["target"]) for e in resolves] == [("pr:1", "ticket:10")]


def test_every_artifact_is_attached_to_the_repository(graph):
    sources = {e["source"] for e in edges_of(graph, RELATION_PART_OF)}

    assert sources == {"pr:1", "pr:2", "ticket:10", "commit:abc", "commit:def"}


# ==========================================================================
# Pagination across the boundary
# ==========================================================================


def test_pull_requests_from_every_page_reach_the_graph(graph):
    assert "pr:1" in graph.nodes
    assert "pr:2" in graph.nodes


def test_the_second_page_is_actually_requested():
    requests: list[httpx.Request] = []

    ingest(fake_github(requests))

    pull_pages = [r for r in requests if r.url.path.endswith("/pulls")]
    assert len(pull_pages) == 2


def test_a_limit_reaches_the_graph_as_fewer_nodes():
    builder = ingest(fake_github(), prs=1)

    assert "pr:1" in builder.nodes
    assert "pr:2" not in builder.nodes


def test_a_limit_also_narrows_the_enrichment_requests():
    requests: list[httpx.Request] = []

    ingest(fake_github(requests), prs=1)

    assert len([r for r in requests if r.url.path.endswith("/reviews")]) == 1
    assert len([r for r in requests if r.url.path.endswith("/files")]) == 1


# ==========================================================================
# Filtering across the boundary
# ==========================================================================


def test_a_pull_request_returned_by_the_issues_endpoint_never_becomes_a_node(graph):
    assert "ticket:11" not in graph.nodes


def test_only_the_real_issue_is_counted(graph):
    assert graph.stats.issues == 1


def test_a_self_review_produces_no_relationship(graph):
    reviewed = {(e["source"], e["target"]) for e in edges_of(graph, RELATION_REVIEWED)}

    assert reviewed == {("person:bob", "pr:1")}
    assert graph.stats.self_reviews_skipped == 1


def test_a_commit_with_no_linked_account_still_becomes_a_node(graph):
    assert "commit:def" in graph.nodes
    assert not any(e["target"] == "commit:def" for e in edges_of(graph, RELATION_AUTHORED))
    assert graph.stats.commits_without_author == 1


# ==========================================================================
# Batching: per-pull-request enrichment landing on the right edges
# ==========================================================================


def test_each_pull_requests_files_attach_to_that_pull_request(graph):
    touches = {(e["source"], e["target"]) for e in edges_of(graph, RELATION_TOUCHES)}

    assert touches == {
        ("pr:1", "file:src/auth.py"),
        ("pr:2", "file:src/auth.py"),
    }


def test_a_file_touched_twice_is_one_node(graph):
    files = [n for n in graph.nodes.values() if n["type"] == NODE_FILE]

    assert len(files) == 1


def test_enrichment_is_requested_once_per_pull_request():
    requests: list[httpx.Request] = []

    ingest(fake_github(requests))

    assert len([r for r in requests if r.url.path.endswith("/reviews")]) == 2
    assert len([r for r in requests if r.url.path.endswith("/files")]) == 2


# ==========================================================================
# Optional enrichment
# ==========================================================================


def test_skipping_reviews_removes_only_the_review_relations():
    builder = ingest(fake_github(), reviews=False)

    assert edges_of(builder, RELATION_REVIEWED) == []
    assert edges_of(builder, RELATION_TOUCHES)
    assert edges_of(builder, RELATION_AUTHORED)


def test_skipping_files_removes_only_the_file_relations_and_nodes():
    builder = ingest(fake_github(), files=False)

    assert edges_of(builder, RELATION_TOUCHES) == []
    assert NODE_FILE not in types_of(builder)
    assert edges_of(builder, RELATION_REVIEWED)


def test_skipping_both_still_produces_the_structural_graph():
    builder = ingest(fake_github(), reviews=False, files=False)

    assert {edge["type"] for edge in builder.edges} == {
        RELATION_AUTHORED,
        RELATION_REPORTED,
        RELATION_PART_OF,
        RELATION_RESOLVES,
    }


def test_skipping_enrichment_skips_its_requests():
    requests: list[httpx.Request] = []

    ingest(fake_github(requests), reviews=False, files=False)

    paths = [r.url.path for r in requests]
    assert not any(path.endswith("/reviews") for path in paths)
    assert not any(path.endswith("/files") for path in paths)


# ==========================================================================
# Metadata propagation
# ==========================================================================


def test_payload_timestamps_survive_onto_nodes(graph):
    assert graph.nodes["pr:1"]["timestamp"] == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert graph.nodes["ticket:10"]["timestamp"] == datetime(
        2024, 4, 1, tzinfo=timezone.utc
    )
    assert graph.nodes["commit:abc"]["timestamp"] == datetime(
        2024, 5, 1, tzinfo=timezone.utc
    )
    assert graph.nodes["repo:o/r"]["timestamp"] == datetime(
        2023, 1, 1, tzinfo=timezone.utc
    )


def test_payload_timestamps_survive_onto_edges(graph):
    authored = next(e for e in edges_of(graph, RELATION_AUTHORED) if e["target"] == "pr:1")
    reviewed = edges_of(graph, RELATION_REVIEWED)[0]

    assert authored["timestamp"] == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert reviewed["timestamp"] == datetime(2024, 6, 1, tzinfo=timezone.utc)


def test_entities_without_an_event_time_carry_none(graph):
    assert graph.nodes["person:alice"]["timestamp"] is None
    assert graph.nodes["file:src/auth.py"]["timestamp"] is None


def test_every_edge_carries_a_confidence(graph):
    assert all(edge["confidence"] is not None for edge in graph.edges)


def test_item_detail_survives_onto_nodes(graph):
    assert graph.nodes["pr:1"]["title"] == "PR 1"
    assert graph.nodes["pr:1"]["state"] == "open"
    assert graph.nodes["ticket:10"]["state"] == "closed"
    assert graph.nodes["commit:abc"]["message"] == "work in abc"
    assert graph.nodes["repo:o/r"]["language"] == "Python"
    assert graph.nodes["person:alice"]["github_id"] == 1


def test_relationship_detail_survives_onto_edges(graph):
    touch = edges_of(graph, RELATION_TOUCHES)[0]
    review = edges_of(graph, RELATION_REVIEWED)[0]

    assert touch["additions"] == 12
    assert touch["deletions"] == 3
    assert touch["status"] == "modified"
    assert review["state"] == "APPROVED"
    assert review["review_id"] == 80


def test_the_statistics_describe_what_actually_arrived(graph):
    assert graph.stats.pull_requests == 2
    assert graph.stats.issues == 1
    assert graph.stats.commits == 2
    assert graph.stats.reviews == 2  # both counted, one skipped as a self-review
    assert graph.stats.changed_files == 2
    assert graph.stats.nodes_created == len(graph.nodes)
    assert graph.stats.edges_created == len(graph.edges)


# ==========================================================================
# Determinism and isolation
# ==========================================================================


def test_two_runs_of_the_same_fixture_produce_the_same_graph():
    first = ingest(fake_github())
    second = ingest(fake_github())

    assert first.nodes == second.nodes
    assert first.edges == second.edges


def test_two_runs_produce_the_same_statistics():
    assert vars(ingest(fake_github()).stats) == vars(ingest(fake_github()).stats)


def test_one_run_does_not_leak_into_the_next():
    first = ingest(fake_github())
    second = ingest(fake_github())

    assert first.nodes is not second.nodes
    assert len(second.nodes) == len(first.nodes)


def test_edge_order_is_stable_across_runs():
    first = [(e["source"], e["target"], e["type"]) for e in ingest(fake_github()).edges]
    second = [(e["source"], e["target"], e["type"]) for e in ingest(fake_github()).edges]

    assert first == second


# ==========================================================================
# Failure behaviour
#
# Enrichment is per pull request and degrades: one failure costs that pull
# request's extras, nothing else. Everything else is fatal, because losing the
# repository or the pull request list means there is no graph to build.
# ==========================================================================


def test_a_failing_repository_fetch_stops_the_run():
    with pytest.raises(GitHubError):
        ingest(fake_github(broken={"/repos/o/r"}))


def test_a_failing_pull_request_listing_stops_the_run():
    with pytest.raises(GitHubError):
        ingest(fake_github(broken={"/pulls"}))


def test_a_failing_review_fetch_costs_only_the_reviews():
    builder = ingest(fake_github(broken={"/reviews"}))

    assert edges_of(builder, RELATION_REVIEWED) == []
    assert builder.stats.pull_requests == 2
    assert builder.stats.issues == 1
    assert builder.stats.commits == 2
    assert edges_of(builder, RELATION_TOUCHES)
    assert edges_of(builder, RELATION_AUTHORED)
    assert edges_of(builder, RELATION_RESOLVES)


def test_a_failing_file_fetch_costs_only_the_files():
    builder = ingest(fake_github(broken={"/files"}))

    assert edges_of(builder, RELATION_TOUCHES) == []
    assert NODE_FILE not in types_of(builder)
    assert edges_of(builder, RELATION_REVIEWED)
    assert builder.stats.pull_requests == 2


def test_a_mid_run_failure_still_yields_a_complete_graph_for_every_other_item():
    """The whole point: one bad pull request must not cost the other one."""
    builder = ingest(fake_github(broken={"/pulls/1/reviews"}))

    # pr:1 lost its reviews; pr:2's self-review was skipped for its own reason.
    assert builder.stats.enrichment_failures == 1
    # Everything else survived intact.
    assert set(builder.nodes) == {
        "repo:o/r",
        "pr:1",
        "pr:2",
        "ticket:10",
        "commit:abc",
        "commit:def",
        "person:alice",
        "person:bob",
        "person:carol",
        "file:src/auth.py",
    }
    assert edges_of(builder, RELATION_TOUCHES)
    assert edges_of(builder, RELATION_RESOLVES)
    assert edges_of(builder, RELATION_PART_OF)


def test_enrichment_failures_are_recorded_on_the_graph():
    builder = ingest(fake_github(broken={"/reviews"}))

    # Two pull requests, each failing on reviews.
    assert builder.stats.enrichment_failures == 2


def test_a_clean_run_records_no_enrichment_failures(graph):
    assert graph.stats.enrichment_failures == 0


# ==========================================================================
# Idempotency
# ==========================================================================


def test_ingesting_the_same_fixture_twice_does_not_double_the_edges():
    builder = GraphBuilder()
    session = fake_github()

    for _ in range(2):
        pull_requests = list(fetch_pull_requests(session, "o/r"))
        numbers = [pr.number for pr in pull_requests]
        review_map, _ = collect_by_pull_request(fetch_reviews, session, "o/r", numbers)
        file_map, _ = collect_by_pull_request(
            fetch_changed_files, session, "o/r", numbers
        )
        builder.build(
            repository=fetch_repository(session, "o/r"),
            pull_requests=pull_requests,
            issues=fetch_issues(session, "o/r"),
            commits=fetch_commits(session, "o/r"),
            reviews=review_map,
            changed_files=file_map,
        )
        if _ == 0:
            after_first = len(builder.edges)

    assert len(builder.edges) == after_first
    assert builder.stats.duplicate_edges_skipped > 0
