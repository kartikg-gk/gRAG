"""Tests for Module 3, the graph builder.

Inputs are the real Module 2 models, constructed from trimmed GitHub payloads,
so these tests fail if the ingestion contract changes underneath them.

Nodes and edges are plain dicts: a node is ``{"id", "type", "timestamp", ...}``
and an edge is ``{"source", "target", "type", "confidence", "timestamp", ...}``,
with any relationship detail flattened alongside.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.ingestion.models import (
    ChangedFile,
    Commit,
    Issue,
    PullRequest,
    Repository,
    Review,
)
from src.knowledge.graph_builder import (
    RELATION_AUTHORED,
    NODE_COMMIT,
    CONFIDENCE,
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
# fixtures
# --------------------------------------------------------------------------


def make_pull_request(
    number: int = 1, *, author: str = "octocat", body: str | None = None
) -> PullRequest:
    return PullRequest.model_validate(
        {
            "id": number,
            "number": number,
            "state": "open",
            "title": f"PR {number}",
            "body": body,
            "user": {"login": author, "id": 1},
            "html_url": f"https://github.com/o/r/pull/{number}",
            "draft": False,
            "merge_commit_sha": None,
            "labels": [],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "closed_at": None,
            "merged_at": None,
        }
    )


def make_issue(
    number: int = 10, *, author: str = "octocat", body: str | None = None
) -> Issue:
    return Issue.model_validate(
        {
            "id": number,
            "number": number,
            "state": "closed",
            "title": f"Issue {number}",
            "body": body,
            "user": {"login": author, "id": 1},
            "html_url": f"https://github.com/o/r/issues/{number}",
            "labels": [],
            "created_at": "2024-02-01T00:00:00Z",
            "updated_at": "2024-02-02T00:00:00Z",
            "closed_at": "2024-02-03T00:00:00Z",
        }
    )


def make_commit(
    sha: str = "abc123", *, author: str | None = "octocat", message: str = "Fix the thing"
) -> Commit:
    return Commit.model_validate(
        {
            "sha": sha,
            "html_url": f"https://github.com/o/r/commit/{sha}",
            "commit": {
                "message": message,
                "author": {
                    "name": "Octo Cat",
                    "email": "octo@example.com",
                    "date": "2024-03-01T00:00:00Z",
                },
            },
            "author": {"login": author, "id": 1} if author else None,
            "parents": [],
        }
    )


def make_repository(full_name: str = "octocat/Hello-World") -> Repository:
    return Repository.model_validate(
        {
            "id": 1,
            "name": full_name.split("/")[-1],
            "full_name": full_name,
            "private": False,
            "owner": {"login": full_name.split("/")[0], "id": 99},
            "html_url": f"https://github.com/{full_name}",
            "description": "test repo",
            "language": "Python",
            "default_branch": "main",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-06-01T00:00:00Z",
            "pushed_at": "2023-06-01T00:00:00Z",
        }
    )


def make_review(
    review_id: int = 80,
    *,
    reviewer: str = "dave",
    state: str = "APPROVED",
    submitted: str | None = "2024-04-01T00:00:00Z",
) -> Review:
    payload = {
        "id": review_id,
        "state": state,
        "body": "lgtm",
        "user": {"login": reviewer, "id": 7},
        "html_url": f"https://github.com/o/r/pull/1#pullrequestreview-{review_id}",
        "commit_id": "abc123",
    }
    if submitted is not None:
        payload["submitted_at"] = submitted
    return Review.model_validate(payload)


def make_changed_file(
    filename: str = "src/app/main.py",
    *,
    status: str = "modified",
    previous_filename: str | None = None,
) -> ChangedFile:
    payload = {
        "sha": "abc",
        "filename": filename,
        "status": status,
        "additions": 10,
        "deletions": 3,
        "changes": 13,
        "patch": "@@ -1 +1 @@",
    }
    if previous_filename is not None:
        payload["previous_filename"] = previous_filename
    return ChangedFile.model_validate(payload)


def edges_of_type(builder: GraphBuilder, relation: str) -> list[dict]:
    return [edge for edge in builder.edges if edge["type"] == relation]


def nodes_of_type(builder: GraphBuilder, node_type: str) -> list[dict]:
    return [node for node in builder.nodes.values() if node["type"] == node_type]


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------


def test_pull_request_becomes_a_node():
    builder = GraphBuilder()

    builder.build(pull_requests=[make_pull_request(1347)])

    node = builder.nodes["pr:1347"]
    assert node["type"] == NODE_PR
    assert node["number"] == 1347
    assert node["title"] == "PR 1347"
    assert node["state"] == "open"


def test_issue_becomes_a_node():
    builder = GraphBuilder()

    builder.build(issues=[make_issue(42)])

    node = builder.nodes["ticket:42"]
    assert node["type"] == NODE_TICKET
    assert node["number"] == 42
    assert node["state"] == "closed"


def test_commit_becomes_a_node_keyed_by_sha():
    builder = GraphBuilder()

    builder.build(commits=[make_commit("6dcb09b")])

    node = builder.nodes["commit:6dcb09b"]
    assert node["type"] == NODE_COMMIT
    assert node["sha"] == "6dcb09b"
    assert node["message"] == "Fix the thing"


def test_author_becomes_a_user_node():
    builder = GraphBuilder()

    builder.build(pull_requests=[make_pull_request(author="alice")])

    node = builder.nodes["person:alice"]
    assert node["type"] == NODE_PERSON
    assert node["login"] == "alice"


def test_every_node_carries_its_own_id():
    builder = GraphBuilder()

    builder.build(pull_requests=[make_pull_request(1)])

    assert builder.nodes["pr:1"]["id"] == "pr:1"


def test_nodes_carry_their_payload_timestamp():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1)],
        issues=[make_issue(10)],
        commits=[make_commit("abc123")],
    )

    assert builder.nodes["pr:1"]["timestamp"] == datetime(
        2024, 1, 1, tzinfo=timezone.utc
    )
    assert builder.nodes["ticket:10"]["timestamp"] == datetime(
        2024, 2, 1, tzinfo=timezone.utc
    )
    assert builder.nodes["commit:abc123"]["timestamp"] == datetime(
        2024, 3, 1, tzinfo=timezone.utc
    )


def test_undated_nodes_keep_none_rather_than_zero():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request()],
        changed_files={1: [make_changed_file("a.py")]},
    )

    assert builder.nodes["person:octocat"]["timestamp"] is None
    assert builder.nodes["file:a.py"]["timestamp"] is None


# --------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------


def test_every_relation_has_a_confidence():
    assert CONFIDENCE[RELATION_AUTHORED] == 0.95
    assert CONFIDENCE[RELATION_RESOLVES] == 0.92
    assert CONFIDENCE[RELATION_REVIEWED] == 0.85
    assert CONFIDENCE[RELATION_TOUCHES] == 0.80
    assert CONFIDENCE[RELATION_PART_OF] == 0.80
    assert CONFIDENCE[RELATION_REPORTED] == 0.75


@pytest.mark.parametrize(
    "relation",
    [RELATION_AUTHORED, RELATION_RESOLVES, RELATION_REVIEWED, RELATION_TOUCHES, RELATION_PART_OF, RELATION_REPORTED],
)
def test_edges_carry_the_confidence_of_their_relation(relation):
    builder = GraphBuilder()

    builder.build(
        repository=make_repository("o/r"),
        pull_requests=[make_pull_request(1, author="alice", body="Fixes #42")],
        issues=[make_issue(42, author="bob")],
        reviews={1: [make_review(reviewer="dave")]},
        changed_files={1: [make_changed_file("a.py")]},
    )

    edges = edges_of_type(builder, relation)
    assert edges, f"expected at least one {relation} edge"
    assert all(edge["confidence"] == CONFIDENCE[relation] for edge in edges)




# --------------------------------------------------------------------------
# authorship and reporting
# --------------------------------------------------------------------------


def test_pull_request_author_is_linked_with_authored():
    builder = GraphBuilder()

    builder.build(pull_requests=[make_pull_request(1347, author="alice")])

    edges = edges_of_type(builder, RELATION_AUTHORED)
    assert len(edges) == 1
    assert edges[0]["source"] == "person:alice"
    assert edges[0]["target"] == "pr:1347"


def test_commit_author_is_linked_with_authored():
    builder = GraphBuilder()

    builder.build(commits=[make_commit("abc123", author="carol")])

    edges = edges_of_type(builder, RELATION_AUTHORED)
    assert edges[0]["source"] == "person:carol"
    assert edges[0]["target"] == "commit:abc123"


def test_issue_author_is_linked_with_reported_not_authored():
    builder = GraphBuilder()

    builder.build(issues=[make_issue(42, author="bob")])

    reported = edges_of_type(builder, RELATION_REPORTED)
    assert len(reported) == 1
    assert reported[0]["source"] == "person:bob"
    assert reported[0]["target"] == "ticket:42"
    assert edges_of_type(builder, RELATION_AUTHORED) == []


def test_authorship_edge_carries_the_creation_timestamp():
    builder = GraphBuilder()

    builder.build(pull_requests=[make_pull_request(1)])

    assert edges_of_type(builder, RELATION_AUTHORED)[0]["timestamp"] == datetime(
        2024, 1, 1, tzinfo=timezone.utc
    )


def test_commit_without_a_linked_account_gets_no_authorship_edge():
    builder = GraphBuilder()

    builder.build(commits=[make_commit("abc123", author=None)])

    assert "commit:abc123" in builder.nodes
    assert edges_of_type(builder, RELATION_AUTHORED) == []


# --------------------------------------------------------------------------
# repository membership
# --------------------------------------------------------------------------


def test_repository_becomes_a_node():
    builder = GraphBuilder()

    builder.build(repository=make_repository("octocat/Hello-World"))

    node = builder.nodes["repo:octocat/Hello-World"]
    assert node["type"] == NODE_REPO
    assert node["full_name"] == "octocat/Hello-World"
    assert node["language"] == "Python"
    assert node["timestamp"] == datetime(2023, 1, 1, tzinfo=timezone.utc)


def test_every_top_level_artifact_is_part_of_the_repository():
    builder = GraphBuilder()

    builder.build(
        repository=make_repository("o/r"),
        pull_requests=[make_pull_request(1)],
        issues=[make_issue(10)],
        commits=[make_commit("abc")],
    )

    memberships = edges_of_type(builder, RELATION_PART_OF)
    assert {edge["source"] for edge in memberships} == {"pr:1", "ticket:10", "commit:abc"}
    assert all(edge["target"] == "repo:o/r" for edge in memberships)


def test_without_a_repository_there_are_no_membership_edges():
    builder = GraphBuilder()

    builder.build(pull_requests=[make_pull_request(1)])

    assert edges_of_type(builder, RELATION_PART_OF) == []
    assert nodes_of_type(builder, NODE_REPO) == []


# --------------------------------------------------------------------------
# reviews
# --------------------------------------------------------------------------


def test_review_links_reviewer_to_the_pull_request():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1347, author="alice")],
        reviews={1347: [make_review(reviewer="dave")]},
    )

    edges = edges_of_type(builder, RELATION_REVIEWED)
    assert len(edges) == 1
    assert edges[0]["source"] == "person:dave"
    assert edges[0]["target"] == "pr:1347"


def test_review_edge_carries_state_and_submission_time():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1)],
        reviews={1: [make_review(state="CHANGES_REQUESTED")]},
    )

    edge = edges_of_type(builder, RELATION_REVIEWED)[0]
    assert edge["state"] == "CHANGES_REQUESTED"
    assert edge["timestamp"] == datetime(2024, 4, 1, tzinfo=timezone.utc)


def test_pending_review_has_no_submission_time():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1)],
        reviews={1: [make_review(state="PENDING", submitted=None)]},
    )

    assert edges_of_type(builder, RELATION_REVIEWED)[0]["timestamp"] is None


def test_self_review_is_not_a_relationship():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1, author="alice")],
        reviews={1: [make_review(reviewer="alice")]},
    )

    assert edges_of_type(builder, RELATION_REVIEWED) == []


def test_self_reviews_are_counted():
    builder = GraphBuilder()

    stats = builder.build(
        pull_requests=[make_pull_request(1, author="alice")],
        reviews={1: [make_review(80, reviewer="alice"), make_review(81, reviewer="dave")]},
    )

    assert stats.self_reviews_skipped == 1
    assert len(edges_of_type(builder, RELATION_REVIEWED)) == 1


def test_several_reviews_by_one_person_collapse_to_one_relationship():
    """Two reviews by the same person on the same pull request are one fact.

    Edges are deduplicated on (source, relation, target), so the second review
    does not create a second edge. The first review's state and id are the ones
    kept.
    """
    builder = GraphBuilder()

    stats = builder.build(
        pull_requests=[make_pull_request(1)],
        reviews={
            1: [
                make_review(80, reviewer="dave", state="CHANGES_REQUESTED"),
                make_review(81, reviewer="dave", state="APPROVED"),
            ]
        },
    )

    edges = edges_of_type(builder, RELATION_REVIEWED)
    assert len(edges) == 1
    assert edges[0]["review_id"] == 80
    assert edges[0]["state"] == "CHANGES_REQUESTED"
    assert stats.reviews == 2
    assert stats.duplicate_edges_skipped == 1
    assert len(nodes_of_type(builder, NODE_PERSON)) == 2  # the author and dave


def test_rebuilding_the_same_payloads_does_not_double_the_edges():
    """Once builds are persisted and re-run, this is what stops edge growth."""
    builder = GraphBuilder()

    builder.build(
        repository=make_repository("o/r"),
        pull_requests=[make_pull_request(1, body="Fixes #42")],
        issues=[make_issue(42)],
        reviews={1: [make_review(reviewer="dave")]},
        changed_files={1: [make_changed_file("a.py")]},
    )
    after_first = len(builder.edges)

    builder.build(
        repository=make_repository("o/r"),
        pull_requests=[make_pull_request(1, body="Fixes #42")],
        issues=[make_issue(42)],
        reviews={1: [make_review(reviewer="dave")]},
        changed_files={1: [make_changed_file("a.py")]},
    )

    assert len(builder.edges) == after_first
    assert builder.stats.duplicate_edges_skipped > 0


# --------------------------------------------------------------------------
# touched files
# --------------------------------------------------------------------------


def test_changed_file_becomes_a_node_keyed_by_path():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1)],
        changed_files={1: [make_changed_file("src/app/main.py")]},
    )

    node = builder.nodes["file:src/app/main.py"]
    assert node["type"] == NODE_FILE
    assert node["path"] == "src/app/main.py"


def test_pull_request_touches_the_file():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1347)],
        changed_files={1347: [make_changed_file("src/app/main.py")]},
    )

    edge = edges_of_type(builder, RELATION_TOUCHES)[0]
    assert edge["source"] == "pr:1347"
    assert edge["target"] == "file:src/app/main.py"


def test_touch_edge_carries_the_per_pull_request_change_counts():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1)],
        changed_files={1: [make_changed_file(status="modified")]},
    )

    edge = edges_of_type(builder, RELATION_TOUCHES)[0]
    assert edge["status"] == "modified"
    assert edge["additions"] == 10
    assert edge["deletions"] == 3


def test_a_file_touched_by_two_pull_requests_is_one_node_with_two_edges():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1), make_pull_request(2)],
        changed_files={
            1: [make_changed_file("shared.py")],
            2: [make_changed_file("shared.py")],
        },
    )

    assert len(nodes_of_type(builder, NODE_FILE)) == 1
    assert len(edges_of_type(builder, RELATION_TOUCHES)) == 2


def test_rename_keeps_the_previous_path_on_the_edge():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1)],
        changed_files={
            1: [
                make_changed_file(
                    "src/new.py", status="renamed", previous_filename="src/old.py"
                )
            ]
        },
    )

    assert edges_of_type(builder, RELATION_TOUCHES)[0]["previous_path"] == "src/old.py"


# --------------------------------------------------------------------------
# closing references
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "Fixes #42",
        "fixes #42",
        "Fixed #42",
        "fix #42",
        "Closes #42",
        "closed #42",
        "close #42",
        "Resolves #42",
        "resolved #42",
        "resolve #42",
        "This one finally fixes #42, promise",
    ],
)
def test_closing_keyword_creates_a_resolves_relationship(body):
    builder = GraphBuilder()

    builder.build(pull_requests=[make_pull_request(1, body=body)], issues=[make_issue(42)])

    edges = edges_of_type(builder, RELATION_RESOLVES)
    assert len(edges) == 1
    assert edges[0]["source"] == "pr:1"
    assert edges[0]["target"] == "ticket:42"


@pytest.mark.parametrize(
    "body",
    [
        "see #42",
        "related to #42",
        "duplicate of #42",
        "#42",
        "follows up on #42",
        "prefix #42 is not a keyword",
    ],
)
def test_bare_mentions_create_no_relationship(body):
    builder = GraphBuilder()

    builder.build(pull_requests=[make_pull_request(1, body=body)], issues=[make_issue(42)])

    assert edges_of_type(builder, RELATION_RESOLVES) == []


def test_commit_message_closing_keyword_creates_a_resolves_relationship():
    builder = GraphBuilder()

    builder.build(commits=[make_commit("abc", message="fix #42")], issues=[make_issue(42)])

    edge = edges_of_type(builder, RELATION_RESOLVES)[0]
    assert edge["source"] == "commit:abc"
    assert edge["target"] == "ticket:42"


def test_pull_request_title_is_not_a_closing_source():
    builder = GraphBuilder()
    pull_request = make_pull_request(1).model_copy(update={"title": "Fixes #42"})

    builder.build(pull_requests=[pull_request], issues=[make_issue(42)])

    assert edges_of_type(builder, RELATION_RESOLVES) == []


def test_issue_body_is_not_a_closing_source():
    builder = GraphBuilder()

    builder.build(issues=[make_issue(10, body="Fixes #42"), make_issue(42)])

    assert edges_of_type(builder, RELATION_RESOLVES) == []


def test_resolves_target_ingested_after_the_referrer_still_links():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1, body="Fixes #42")],
        issues=[make_issue(42)],
    )

    assert edges_of_type(builder, RELATION_RESOLVES)[0]["target"] == "ticket:42"


def test_resolves_can_point_at_a_pull_request():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1, body="Fixes #2"), make_pull_request(2)]
    )

    edge = edges_of_type(builder, RELATION_RESOLVES)[0]
    assert edge["source"] == "pr:1"
    assert edge["target"] == "pr:2"


def test_resolves_to_an_uningested_number_creates_no_edge():
    builder = GraphBuilder()

    stats = builder.build(pull_requests=[make_pull_request(1, body="Fixes #999")])

    assert edges_of_type(builder, RELATION_RESOLVES) == []
    assert stats.unresolved_references == 1


def test_the_same_closing_reference_twice_produces_one_edge():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[make_pull_request(1, body="Fixes #42, really closes #42")],
        issues=[make_issue(42)],
    )

    assert len(edges_of_type(builder, RELATION_RESOLVES)) == 1


def test_an_item_does_not_resolve_itself():
    builder = GraphBuilder()

    builder.build(pull_requests=[make_pull_request(1, body="Fixes #1")])

    assert edges_of_type(builder, RELATION_RESOLVES) == []


def test_missing_body_is_not_a_closing_source():
    builder = GraphBuilder()

    builder.build(pull_requests=[make_pull_request(1)], issues=[make_issue(42)])

    assert edges_of_type(builder, RELATION_RESOLVES) == []


# --------------------------------------------------------------------------
# duplicate prevention
# --------------------------------------------------------------------------


def test_one_author_of_several_items_produces_one_user_node():
    builder = GraphBuilder()

    builder.build(
        pull_requests=[
            make_pull_request(1, author="alice"),
            make_pull_request(2, author="alice"),
        ]
    )

    assert len(nodes_of_type(builder, NODE_PERSON)) == 1


def test_repeated_payload_does_not_duplicate_a_node():
    builder = GraphBuilder()

    builder.build(pull_requests=[make_pull_request(1), make_pull_request(1)])

    assert len(nodes_of_type(builder, NODE_PR)) == 1


def test_first_version_of_a_node_wins():
    builder = GraphBuilder()
    first = make_pull_request(1)
    second = make_pull_request(1).model_copy(update={"title": "renamed later"})

    builder.build(pull_requests=[first, second])

    assert builder.nodes["pr:1"]["title"] == "PR 1"


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_build_reports_what_it_created():
    builder = GraphBuilder()

    stats = builder.build(
        pull_requests=[make_pull_request(1, author="alice")],
        issues=[make_issue(10, author="alice")],
        commits=[make_commit("abc123", author="bob")],
    )

    assert stats.nodes_created == 5  # 3 items + 2 distinct users
    assert stats.edges_created == 3
    assert stats.pull_requests == 1
    assert stats.issues == 1
    assert stats.commits == 1


def test_build_counts_skipped_duplicates():
    builder = GraphBuilder()

    stats = builder.build(
        pull_requests=[
            make_pull_request(1, author="alice"),
            make_pull_request(1, author="alice"),
        ]
    )

    # the repeated pull request and its already-known author
    assert stats.duplicate_nodes_skipped == 2


def test_build_counts_commits_with_no_linked_account():
    builder = GraphBuilder()

    stats = builder.build(
        commits=[make_commit("abc", author=None), make_commit("def", author="alice")]
    )

    assert stats.commits_without_author == 1


def test_stats_count_reviews_and_touched_files():
    builder = GraphBuilder()

    stats = builder.build(
        repository=make_repository("o/r"),
        pull_requests=[make_pull_request(1)],
        reviews={1: [make_review()]},
        changed_files={1: [make_changed_file("a.py"), make_changed_file("b.py")]},
    )

    assert stats.reviews == 1
    assert stats.changed_files == 2


def test_reviews_for_an_uningested_pull_request_are_counted_not_linked():
    builder = GraphBuilder()

    stats = builder.build(reviews={999: [make_review()]})

    assert edges_of_type(builder, RELATION_REVIEWED) == []
    assert stats.orphan_associations == 1


def test_changed_files_for_an_uningested_pull_request_are_counted_not_linked():
    builder = GraphBuilder()

    stats = builder.build(changed_files={999: [make_changed_file("a.py")]})

    assert edges_of_type(builder, RELATION_TOUCHES) == []
    assert stats.orphan_associations == 1


def test_building_nothing_is_an_empty_graph():
    builder = GraphBuilder()

    stats = builder.build()

    assert builder.nodes == {}
    assert builder.edges == []
    assert stats.nodes_created == 0
    assert stats.edges_created == 0


def test_generators_are_accepted_as_input():
    builder = GraphBuilder()

    stats = builder.build(pull_requests=(make_pull_request(n) for n in (1, 2)))

    assert stats.pull_requests == 2
