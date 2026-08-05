"""The offline corpus: a small checkout service, as GitHub would describe it.

Plain dicts in exactly the shape the GitHub REST API returns, so the offline
demo runs them through the same models and the same graph builder the live
demo uses. Nothing here is mocked or simplified — swapping this module for real
API responses changes nothing downstream.

The corpus is shaped so one query produces a genuinely mixed result: some
records about authentication, some about billing, and some about neither. That
is what gives the trace something to show.
"""

from __future__ import annotations

ALICE = {"login": "alice", "id": 1}
BOB = {"login": "bob", "id": 2}
CAROL = {"login": "carol", "id": 3}
DANA = {"login": "dana", "id": 4}


REPOSITORY = {
    "id": 900,
    "name": "checkout",
    "full_name": "acme/checkout",
    "private": False,
    "owner": ALICE,
    "html_url": "https://github.com/acme/checkout",
    "description": "Checkout and billing service",
    "language": "Python",
    "default_branch": "main",
    "created_at": "2023-03-01T00:00:00Z",
    "updated_at": "2024-09-01T00:00:00Z",
    "pushed_at": "2024-09-01T00:00:00Z",
}


def _pull_request(number, title, author, body, created):
    return {
        "id": number,
        "number": number,
        "state": "closed",
        "title": title,
        "body": body,
        "user": author,
        "html_url": f"https://github.com/acme/checkout/pull/{number}",
        "draft": False,
        "merge_commit_sha": None,
        "labels": [],
        "created_at": created,
        "updated_at": created,
        "closed_at": created,
        "merged_at": created,
    }


PULL_REQUESTS = [
    _pull_request(
        101,
        "Reject expired authentication tokens on refresh",
        ALICE,
        "The expiry comparison was strict, so a token was rejected a second "
        "early. Fixes #12.",
        "2024-08-01T00:00:00Z",
    ),
    _pull_request(
        102,
        "Retry the payment webhook on transient failures",
        BOB,
        "Adds three retries with backoff. Fixes #13.",
        "2024-08-05T00:00:00Z",
    ),
    _pull_request(
        103,
        "Update the README badges",
        CAROL,
        "Documentation only, no behaviour change.",
        "2024-08-09T00:00:00Z",
    ),
]


def _issue(number, title, author, created):
    return {
        "id": number,
        "number": number,
        "state": "closed",
        "title": title,
        "body": None,
        "user": author,
        "html_url": f"https://github.com/acme/checkout/issues/{number}",
        "labels": [],
        "created_at": created,
        "updated_at": created,
        "closed_at": created,
    }


ISSUES = [
    _issue(12, "Authentication session expires one second early", DANA, "2024-07-20T00:00:00Z"),
    _issue(13, "Webhook retries flood the payment queue", BOB, "2024-07-22T00:00:00Z"),
    # GitHub returns pull requests from the issues endpoint too. Left in on
    # purpose: the ingestion layer filters it, and the demo proves that.
    dict(
        _issue(101, "Reject expired authentication tokens on refresh", ALICE, "2024-08-01T00:00:00Z"),
        pull_request={"url": "https://api.github.com/repos/acme/checkout/pulls/101"},
    ),
]


def _commit(sha, message, linked, date):
    return {
        "sha": sha,
        "html_url": f"https://github.com/acme/checkout/commit/{sha}",
        "commit": {
            "message": message,
            "author": {"name": "Alice Example", "email": "alice@acme.test", "date": date},
        },
        "author": linked,
        "parents": [],
    }


COMMITS = [
    _commit("a1b2c3d", "Tighten the authentication token expiry comparison", ALICE, "2024-08-01T00:00:00Z"),
    _commit("e4f5a6b", "Add webhook retry backoff", BOB, "2024-08-05T00:00:00Z"),
    # No linked account: GitHub could not match the git email to a user.
    _commit("c7d8e9f", "Bump the lockfile", None, "2024-08-11T00:00:00Z"),
]


def _review(review_id, reviewer, state="APPROVED"):
    return {
        "id": review_id,
        "state": state,
        "body": "",
        "user": reviewer,
        "html_url": "https://github.com/acme/checkout/pull/101",
        "commit_id": "a1b2c3d",
        "submitted_at": "2024-08-02T00:00:00Z",
    }


REVIEWS = {
    101: [_review(1, BOB)],
    # bob opened 102, so this is a self-review and the builder drops it.
    102: [_review(2, BOB)],
    103: [],
}


def _file(filename, additions, deletions):
    return {
        "sha": "blob",
        "filename": filename,
        "status": "modified",
        "additions": additions,
        "deletions": deletions,
        "changes": additions + deletions,
        "patch": "@@ -1 +1 @@",
    }


CHANGED_FILES = {
    101: [_file("src/auth/tokens.py", 14, 6)],
    102: [_file("src/billing/webhook.py", 40, 3)],
    103: [_file("README.md", 2, 2)],
}


# Chosen to span both themes in the corpus. A narrow query would retrieve only
# the records the answer then uses, and a trace where everything retrieved was
# used demonstrates nothing.
DEFAULT_QUERY = (
    "who changed the authentication token expiry and the payment webhook recently?"
)
