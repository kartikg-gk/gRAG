"""Tests for the normalized GitHub domain models.

Payload fixtures are trimmed copies of real GitHub REST API responses. Field
names in the models match GitHub's JSON keys, so these tests double as a record
of the payload shapes Module 2 promises to downstream modules.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.ingestion.models import (
    ChangedFile,
    Commit,
    GitHubUser,
    Issue,
    PullRequest,
    Repository,
    Review,
)


def test_github_user_parses_login_and_id():
    user = GitHubUser.model_validate({"login": "octocat", "id": 1})

    assert user.login == "octocat"
    assert user.id == 1


def test_unknown_fields_are_ignored():
    user = GitHubUser.model_validate(
        {"login": "octocat", "id": 1, "gravatar_id": "", "site_admin": False}
    )

    assert user.login == "octocat"


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        GitHubUser.model_validate({"id": 1})


def test_repository_parses_identity_and_timestamps():
    repo = Repository.model_validate(
        {
            "id": 1296269,
            "name": "Hello-World",
            "full_name": "octocat/Hello-World",
            "private": False,
            "owner": {"login": "octocat", "id": 1},
            "html_url": "https://github.com/octocat/Hello-World",
            "description": "This your first repo!",
            "language": "Python",
            "default_branch": "master",
            "created_at": "2011-01-26T19:01:12Z",
            "updated_at": "2011-01-26T19:14:43Z",
            "pushed_at": "2011-01-26T19:06:43Z",
        }
    )

    assert repo.full_name == "octocat/Hello-World"
    assert repo.owner.login == "octocat"
    assert repo.language == "Python"
    assert repo.created_at == datetime(2011, 1, 26, 19, 1, 12, tzinfo=timezone.utc)


def test_repository_allows_missing_description_and_language():
    repo = Repository.model_validate(
        {
            "id": 1,
            "name": "empty",
            "full_name": "octocat/empty",
            "private": False,
            "owner": {"login": "octocat", "id": 1},
            "html_url": "https://github.com/octocat/empty",
            "description": None,
            "language": None,
            "default_branch": "main",
            "created_at": "2011-01-26T19:01:12Z",
            "updated_at": "2011-01-26T19:14:43Z",
            "pushed_at": None,
        }
    )

    assert repo.description is None
    assert repo.language is None
    assert repo.pushed_at is None


def test_pull_request_parses_core_fields():
    pr = PullRequest.model_validate(
        {
            "id": 1,
            "number": 1347,
            "state": "open",
            "title": "Amazing new feature",
            "body": "Please pull these awesome changes in!",
            "user": {"login": "octocat", "id": 1},
            "html_url": "https://github.com/octocat/Hello-World/pull/1347",
            "draft": False,
            "merge_commit_sha": "e5bd3914e2e596debea16f433f57875b5b90bcd6",
            "labels": [{"name": "bug"}, {"name": "enhancement"}],
            "created_at": "2011-01-26T19:01:12Z",
            "updated_at": "2011-01-26T19:01:12Z",
            "closed_at": None,
            "merged_at": None,
        }
    )

    assert pr.number == 1347
    assert pr.state == "open"
    assert pr.user.login == "octocat"
    assert pr.draft is False
    assert pr.merge_commit_sha == "e5bd3914e2e596debea16f433f57875b5b90bcd6"
    assert [label.name for label in pr.labels] == ["bug", "enhancement"]
    assert pr.created_at == datetime(2011, 1, 26, 19, 1, 12, tzinfo=timezone.utc)


def test_pull_request_open_has_no_closed_or_merged_timestamp():
    pr = PullRequest.model_validate(
        {
            "id": 1,
            "number": 1,
            "state": "open",
            "title": "wip",
            "body": None,
            "user": {"login": "octocat", "id": 1},
            "html_url": "https://github.com/octocat/Hello-World/pull/1",
            "draft": True,
            "merge_commit_sha": None,
            "labels": [],
            "created_at": "2011-01-26T19:01:12Z",
            "updated_at": "2011-01-26T19:01:12Z",
            "closed_at": None,
            "merged_at": None,
        }
    )

    assert pr.closed_at is None
    assert pr.merged_at is None
    assert pr.labels == []


def test_pull_request_merged_keeps_merged_at():
    pr = PullRequest.model_validate(
        {
            "id": 1,
            "number": 2,
            "state": "closed",
            "title": "done",
            "body": None,
            "user": {"login": "octocat", "id": 1},
            "html_url": "https://github.com/octocat/Hello-World/pull/2",
            "draft": False,
            "merge_commit_sha": "abc123",
            "labels": [],
            "created_at": "2011-01-26T19:01:12Z",
            "updated_at": "2011-02-01T10:00:00Z",
            "closed_at": "2011-02-01T10:00:00Z",
            "merged_at": "2011-02-01T10:00:00Z",
        }
    )

    assert pr.merged_at == datetime(2011, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    assert pr.closed_at == pr.merged_at


def test_issue_parses_core_fields():
    issue = Issue.model_validate(
        {
            "id": 1,
            "number": 1347,
            "state": "open",
            "title": "Found a bug",
            "body": "I'm having a problem with this.",
            "user": {"login": "octocat", "id": 1},
            "html_url": "https://github.com/octocat/Hello-World/issues/1347",
            "labels": [{"name": "bug"}],
            "created_at": "2011-04-22T13:33:48Z",
            "updated_at": "2011-04-22T13:33:48Z",
            "closed_at": None,
        }
    )

    assert issue.number == 1347
    assert issue.title == "Found a bug"
    assert issue.user.login == "octocat"
    assert [label.name for label in issue.labels] == ["bug"]
    assert issue.closed_at is None


def test_commit_keeps_nested_commit_object():
    commit = Commit.model_validate(
        {
            "sha": "6dcb09b5b57875f334f61aebed695e2e4193db5e",
            "html_url": "https://github.com/octocat/Hello-World/commit/6dcb09b",
            "commit": {
                "message": "Fix all the bugs",
                "author": {
                    "name": "Monalisa Octocat",
                    "email": "support@github.com",
                    "date": "2011-04-14T16:00:49Z",
                },
            },
            "author": {"login": "octocat", "id": 1},
            "parents": [{"sha": "553c2077f0edc3d5dc5d17262f6aa498e69d6f8e"}],
        }
    )

    assert commit.sha == "6dcb09b5b57875f334f61aebed695e2e4193db5e"
    assert commit.commit.message == "Fix all the bugs"
    assert commit.commit.author.name == "Monalisa Octocat"
    assert commit.commit.author.date == datetime(
        2011, 4, 14, 16, 0, 49, tzinfo=timezone.utc
    )
    assert commit.author.login == "octocat"
    assert [parent.sha for parent in commit.parents] == [
        "553c2077f0edc3d5dc5d17262f6aa498e69d6f8e"
    ]


def test_commit_author_is_none_when_github_cannot_link_an_account():
    commit = Commit.model_validate(
        {
            "sha": "abc123",
            "html_url": "https://github.com/octocat/Hello-World/commit/abc123",
            "commit": {
                "message": "Drive-by fix",
                "author": {
                    "name": "Someone Unlinked",
                    "email": "nobody@example.com",
                    "date": "2011-04-14T16:00:49Z",
                },
            },
            "author": None,
            "parents": [],
        }
    )

    assert commit.author is None
    assert commit.commit.author.name == "Someone Unlinked"


def test_review_parses_state_and_submitted_at():
    review = Review.model_validate(
        {
            "id": 80,
            "state": "APPROVED",
            "body": "Here is the body for the review.",
            "user": {"login": "octocat", "id": 1},
            "html_url": "https://github.com/octocat/Hello-World/pull/12#pullrequestreview-80",
            "commit_id": "ecdd80bb57125d7ba9641ffaa4d7d2c19d3f3091",
            "submitted_at": "2019-11-17T17:43:43Z",
        }
    )

    assert review.state == "APPROVED"
    assert review.user.login == "octocat"
    assert review.submitted_at == datetime(2019, 11, 17, 17, 43, 43, tzinfo=timezone.utc)


def test_pending_review_has_no_submitted_at():
    review = Review.model_validate(
        {
            "id": 81,
            "state": "PENDING",
            "body": "",
            "user": {"login": "octocat", "id": 1},
            "html_url": "https://github.com/octocat/Hello-World/pull/12",
            "commit_id": None,
        }
    )

    assert review.state == "PENDING"
    assert review.submitted_at is None


def test_changed_file_parses_status_and_counts():
    changed = ChangedFile.model_validate(
        {
            "sha": "bbcd538c8e72b8c175046e27cc8f907076331401",
            "filename": "src/app/main.py",
            "status": "modified",
            "additions": 103,
            "deletions": 21,
            "changes": 124,
            "patch": "@@ -132,7 +132,7 @@",
        }
    )

    assert changed.filename == "src/app/main.py"
    assert changed.status == "modified"
    assert changed.additions == 103
    assert changed.deletions == 21
    assert changed.changes == 124


def test_renamed_file_keeps_previous_filename():
    changed = ChangedFile.model_validate(
        {
            "sha": "abc",
            "filename": "src/app/new_name.py",
            "status": "renamed",
            "additions": 0,
            "deletions": 0,
            "changes": 0,
            "previous_filename": "src/app/old_name.py",
        }
    )

    assert changed.previous_filename == "src/app/old_name.py"


def test_binary_file_has_no_patch():
    changed = ChangedFile.model_validate(
        {
            "sha": "abc",
            "filename": "docs/logo.png",
            "status": "added",
            "additions": 0,
            "deletions": 0,
            "changes": 0,
        }
    )

    assert changed.patch is None
    assert changed.previous_filename is None
