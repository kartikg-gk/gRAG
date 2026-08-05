"""Unit tests for the GitHub client, in isolation.

Every test drives the real request, pagination, and validation code against
``httpx.MockTransport``. No network, no monkeypatching of internals: the session
is an ordinary ``httpx.Client`` passed in as the first argument, which is the
only seam these tests need.

Nothing downstream is exercised here — no graph building, no CLI. Wiring between
components is covered by ``test_ingestion_wiring.py``.
"""

from __future__ import annotations

import time

import httpx
import pytest

from src.ingestion.github import (
    API_ROOT,
    PER_PAGE,
    TIMEOUT,
    GitHubAuthError,
    GitHubError,
    GitHubRateLimitError,
    fetch_changed_files,
    fetch_commits,
    fetch_issues,
    fetch_pull_requests,
    fetch_repository,
    fetch_reviews,
    make_session,
)
from src.ingestion.models import (
    ChangedFile,
    Commit,
    Issue,
    PullRequest,
    Repository,
    Review,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def session_returning(handler) -> httpx.Client:
    """A session whose HTTP layer is the given handler function."""
    return httpx.Client(base_url=API_ROOT, transport=httpx.MockTransport(handler))


def session_with_pages(
    pages: list[list[dict]], recorder: list | None = None, *, headers: dict | None = None
) -> httpx.Client:
    """A session serving ``pages`` in order, linked by Link rel="next"."""

    def handler(request: httpx.Request) -> httpx.Response:
        if recorder is not None:
            recorder.append(request)
        page = int(request.url.params.get("page", 1))
        response_headers = dict(headers or {})
        if page < len(pages):
            next_url = str(request.url.copy_set_param("page", page + 1))
            response_headers["Link"] = f'<{next_url}>; rel="next"'
        return httpx.Response(200, json=pages[page - 1], headers=response_headers)

    return httpx.Client(base_url=API_ROOT, transport=httpx.MockTransport(handler))


def user() -> dict:
    return {"login": "octocat", "id": 1}


def pr_payload(number: int) -> dict:
    return {
        "id": number,
        "number": number,
        "state": "open",
        "title": f"PR {number}",
        "body": None,
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


def issue_payload(number: int, *, is_pull_request: bool = False) -> dict:
    payload = {
        "id": number,
        "number": number,
        "state": "open",
        "title": f"Issue {number}",
        "body": None,
        "user": user(),
        "html_url": f"https://github.com/o/r/issues/{number}",
        "labels": [],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "closed_at": None,
    }
    if is_pull_request:
        payload["pull_request"] = {"url": "https://api.github.com/repos/o/r/pulls/1"}
    return payload


def commit_payload(sha: str, *, linked: bool = True) -> dict:
    """``linked=False`` is a commit GitHub could not match to an account."""
    return {
        "sha": sha,
        "html_url": f"https://github.com/o/r/commit/{sha}",
        "commit": {
            "message": f"commit {sha}",
            "author": {
                "name": "Octo Cat",
                "email": "octo@example.com",
                "date": "2024-01-01T00:00:00Z",
            },
        },
        "author": user() if linked else None,
        "parents": [],
    }


def review_payload(review_id: int) -> dict:
    return {
        "id": review_id,
        "state": "APPROVED",
        "body": "lgtm",
        "user": user(),
        "html_url": f"https://github.com/o/r/pull/1#pullrequestreview-{review_id}",
        "commit_id": "abc123",
        "submitted_at": "2024-01-02T00:00:00Z",
    }


def changed_file_payload(filename: str) -> dict:
    return {
        "sha": "abc",
        "filename": filename,
        "status": "modified",
        "additions": 1,
        "deletions": 2,
        "changes": 3,
        "patch": "@@ -1 +1 @@",
    }


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


# ==========================================================================
# Authentication
# ==========================================================================


def test_a_token_is_sent_as_bearer_auth():
    session = make_session(token="secret-token")

    assert session.headers["authorization"] == "Bearer secret-token"


def test_no_token_means_no_auth_header(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    session = make_session()

    assert "authorization" not in session.headers


def test_the_token_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "env-token")

    session = make_session()

    assert session.headers["authorization"] == "Bearer env-token"


def test_an_explicit_token_beats_the_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "env-token")

    session = make_session(token="explicit-token")

    assert session.headers["authorization"] == "Bearer explicit-token"


def test_an_empty_environment_token_is_treated_as_absent(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "")

    session = make_session()

    assert "authorization" not in session.headers


def test_auth_is_sent_on_follow_up_pages_too():
    requests: list[httpx.Request] = []
    handler_session = session_with_pages(
        [[pr_payload(1)], [pr_payload(2)]], recorder=requests
    )
    handler_session.headers["Authorization"] = "Bearer secret-token"

    list(fetch_pull_requests(handler_session, "o/r"))

    assert len(requests) == 2
    assert all(r.headers["authorization"] == "Bearer secret-token" for r in requests)


def test_the_token_never_appears_in_an_error_message():
    session = session_returning(
        lambda request: httpx.Response(403, json={"message": "Forbidden"})
    )
    session.headers["Authorization"] = "Bearer super-secret-token"

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "super-secret-token" not in str(excinfo.value)


# ==========================================================================
# Request construction
# ==========================================================================


def test_the_session_targets_the_public_api_by_default():
    assert make_session(token="t").base_url == httpx.URL(API_ROOT)


def test_the_session_sets_a_timeout():
    session = make_session(token="t")

    assert session.timeout.read == TIMEOUT


def test_requests_ask_for_the_versioned_json_api():
    session = make_session(token="t")

    assert session.headers["accept"] == "application/vnd.github+json"
    assert session.headers["x-github-api-version"] == "2022-11-28"


@pytest.mark.parametrize(
    "call, expected_path",
    [
        (lambda s: fetch_repository(s, "o/r"), "/repos/o/r"),
        (lambda s: list(fetch_pull_requests(s, "o/r")), "/repos/o/r/pulls"),
        (lambda s: list(fetch_issues(s, "o/r")), "/repos/o/r/issues"),
        (lambda s: list(fetch_commits(s, "o/r")), "/repos/o/r/commits"),
        (lambda s: list(fetch_reviews(s, "o/r", 7)), "/repos/o/r/pulls/7/reviews"),
        (lambda s: list(fetch_changed_files(s, "o/r", 7)), "/repos/o/r/pulls/7/files"),
    ],
)
def test_each_endpoint_targets_its_own_path(call, expected_path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = repo_payload() if request.url.path == "/repos/o/r" else []
        return httpx.Response(200, json=body)

    call(session_returning(handler))

    assert requests[0].url.path == expected_path


@pytest.mark.parametrize(
    "call",
    [
        lambda s: list(fetch_pull_requests(s, "o/r")),
        lambda s: list(fetch_issues(s, "o/r")),
        lambda s: list(fetch_commits(s, "o/r")),
        lambda s: list(fetch_reviews(s, "o/r", 1)),
        lambda s: list(fetch_changed_files(s, "o/r", 1)),
    ],
)
def test_every_listing_asks_for_the_largest_page(call):
    requests: list[httpx.Request] = []
    call(session_with_pages([[]], recorder=requests))

    assert requests[0].url.params["per_page"] == str(PER_PAGE)


def test_the_page_size_is_githubs_maximum():
    assert PER_PAGE == 100


def test_pull_requests_default_to_every_state():
    requests: list[httpx.Request] = []

    list(fetch_pull_requests(session_with_pages([[]], recorder=requests), "o/r"))

    assert requests[0].url.params["state"] == "all"


def test_issues_default_to_every_state():
    requests: list[httpx.Request] = []

    list(fetch_issues(session_with_pages([[]], recorder=requests), "o/r"))

    assert requests[0].url.params["state"] == "all"


@pytest.mark.parametrize("state", ["open", "closed", "all"])
def test_an_explicit_state_is_passed_through(state):
    requests: list[httpx.Request] = []

    list(
        fetch_pull_requests(
            session_with_pages([[]], recorder=requests), "o/r", state=state
        )
    )

    assert requests[0].url.params["state"] == state


def test_commits_send_no_state_parameter():
    requests: list[httpx.Request] = []

    list(fetch_commits(session_with_pages([[]], recorder=requests), "o/r"))

    assert "state" not in requests[0].url.params


def test_relative_paths_resolve_against_the_configured_base_url():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=repo_payload())

    session = httpx.Client(
        base_url="https://github.example.com/api/v3",
        transport=httpx.MockTransport(handler),
    )

    fetch_repository(session, "o/r")

    assert str(requests[0].url).startswith(
        "https://github.example.com/api/v3/repos/o/r"
    )


@pytest.mark.parametrize(
    "repo", ["octo-org/repo.name", "a-b/c_d", "Octo/Repo", "o/r.py"]
)
def test_repository_names_survive_the_path_intact(repo):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=repo_payload())

    fetch_repository(session_returning(handler), repo)

    assert requests[0].url.path == f"/repos/{repo}"


def test_the_pull_request_number_reaches_the_path():
    requests: list[httpx.Request] = []

    list(fetch_reviews(session_with_pages([[]], recorder=requests), "o/r", 1347))

    assert requests[0].url.path == "/repos/o/r/pulls/1347/reviews"


def test_every_request_is_a_get():
    requests: list[httpx.Request] = []

    list(fetch_pull_requests(session_with_pages([[pr_payload(1)]], requests), "o/r"))

    assert all(request.method == "GET" for request in requests)


# ==========================================================================
# Pagination
# ==========================================================================


def test_next_links_are_followed_across_several_pages():
    session = session_with_pages(
        [[pr_payload(1), pr_payload(2)], [pr_payload(3)], [pr_payload(4)]]
    )

    assert [pr.number for pr in fetch_pull_requests(session, "o/r")] == [1, 2, 3, 4]


def test_a_page_without_a_next_link_ends_the_walk():
    requests: list[httpx.Request] = []
    session = session_with_pages([[pr_payload(1)]], recorder=requests)

    list(fetch_pull_requests(session, "o/r"))

    assert len(requests) == 1


def test_link_headers_without_a_next_relation_end_the_walk():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=[pr_payload(1)],
            headers={
                "Link": '<https://api.github.com/repos/o/r/pulls?page=1>; rel="prev", '
                '<https://api.github.com/repos/o/r/pulls?page=1>; rel="first"'
            },
        )

    list(fetch_pull_requests(session_returning(handler), "o/r"))

    assert len(requests) == 1


def test_query_parameters_are_not_resent_on_follow_up_pages():
    requests: list[httpx.Request] = []
    session = session_with_pages(
        [[pr_payload(1)], [pr_payload(2)]], recorder=requests
    )

    list(fetch_pull_requests(session, "o/r", state="closed"))

    # The first request carries the parameters; the second reuses GitHub's own
    # next URL, which already has them, so nothing is appended twice.
    assert requests[0].url.params["state"] == "closed"
    assert len(requests[1].url.params.get_list("state")) == 1


def test_the_next_url_is_followed_exactly_as_given():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "page" in request.url.params:
            return httpx.Response(200, json=[pr_payload(2)])
        return httpx.Response(
            200,
            json=[pr_payload(1)],
            headers={
                "Link": '<https://api.github.com/repos/o/r/pulls'
                '?state=all&per_page=100&page=2>; rel="next"'
            },
        )

    list(fetch_pull_requests(session_returning(handler), "o/r"))

    assert str(requests[1].url).endswith("state=all&per_page=100&page=2")


def test_a_limit_stops_before_the_next_page_is_requested():
    requests: list[httpx.Request] = []
    session = session_with_pages(
        [[pr_payload(1), pr_payload(2)], [pr_payload(3)]], recorder=requests
    )

    numbers = [pr.number for pr in fetch_pull_requests(session, "o/r", limit=2)]

    assert numbers == [1, 2]
    assert len(requests) == 1


def test_a_limit_larger_than_the_result_set_returns_everything():
    session = session_with_pages([[pr_payload(1)], [pr_payload(2)]])

    assert len(list(fetch_pull_requests(session, "o/r", limit=99))) == 2


def test_a_limit_of_zero_requests_nothing():
    requests: list[httpx.Request] = []
    session = session_with_pages([[pr_payload(1)]], recorder=requests)

    assert list(fetch_pull_requests(session, "o/r", limit=0)) == []
    assert requests == []


def test_pages_are_fetched_lazily():
    requests: list[httpx.Request] = []
    session = session_with_pages(
        [[pr_payload(1)], [pr_payload(2)], [pr_payload(3)]], recorder=requests
    )

    stream = fetch_pull_requests(session, "o/r")
    next(stream)

    assert len(requests) == 1


def test_pagination_works_the_same_for_every_listing():
    for call in (
        lambda s: list(fetch_issues(s, "o/r")),
        lambda s: list(fetch_commits(s, "o/r")),
        lambda s: list(fetch_reviews(s, "o/r", 1)),
        lambda s: list(fetch_changed_files(s, "o/r", 1)),
    ):
        requests: list[httpx.Request] = []
        session = session_with_pages([[], []], recorder=requests)
        call(session)
        assert len(requests) == 2


# ==========================================================================
# Response normalization
# ==========================================================================


def test_a_repository_becomes_a_repository_model():
    session = session_returning(lambda r: httpx.Response(200, json=repo_payload()))

    repo = fetch_repository(session, "o/r")

    assert isinstance(repo, Repository)
    assert repo.full_name == "o/r"
    assert repo.owner.login == "octocat"


def test_pull_requests_become_pull_request_models():
    session = session_with_pages([[pr_payload(1)]])

    results = list(fetch_pull_requests(session, "o/r"))

    assert isinstance(results[0], PullRequest)
    assert results[0].user.login == "octocat"


def test_issues_become_issue_models():
    session = session_with_pages([[issue_payload(7)]])

    results = list(fetch_issues(session, "o/r"))

    assert isinstance(results[0], Issue)
    assert results[0].number == 7


def test_commits_become_commit_models_with_nested_detail():
    session = session_with_pages([[commit_payload("abc")]])

    commit = list(fetch_commits(session, "o/r"))[0]

    assert isinstance(commit, Commit)
    assert commit.commit.message == "commit abc"
    assert commit.commit.author.name == "Octo Cat"


def test_a_commit_with_no_linked_account_normalizes_to_none():
    session = session_with_pages([[commit_payload("abc", linked=False)]])

    commit = list(fetch_commits(session, "o/r"))[0]

    assert commit.author is None
    assert commit.commit.author.name == "Octo Cat"


def test_reviews_become_review_models():
    session = session_with_pages([[review_payload(80)]])

    review = list(fetch_reviews(session, "o/r", 1))[0]

    assert isinstance(review, Review)
    assert review.state == "APPROVED"


def test_changed_files_become_changed_file_models():
    session = session_with_pages([[changed_file_payload("src/main.py")]])

    changed = list(fetch_changed_files(session, "o/r", 1))[0]

    assert isinstance(changed, ChangedFile)
    assert changed.filename == "src/main.py"


def test_timestamps_normalize_to_aware_datetimes():
    session = session_with_pages([[pr_payload(1)]])

    created = list(fetch_pull_requests(session, "o/r"))[0].created_at

    assert created.tzinfo is not None
    assert created.year == 2024


def test_fields_github_adds_later_are_ignored():
    payload = pr_payload(1) | {"a_brand_new_field": {"nested": True}}
    session = session_with_pages([[payload]])

    assert list(fetch_pull_requests(session, "o/r"))[0].number == 1


def test_pull_requests_returned_by_the_issues_endpoint_are_dropped():
    session = session_with_pages(
        [[issue_payload(1), issue_payload(2, is_pull_request=True), issue_payload(3)]]
    )

    assert [issue.number for issue in fetch_issues(session, "o/r")] == [1, 3]


def test_a_limit_counts_issues_rather_than_raw_items():
    session = session_with_pages(
        [[issue_payload(1, is_pull_request=True), issue_payload(2), issue_payload(3)]]
    )

    numbers = [issue.number for issue in fetch_issues(session, "o/r", limit=2)]

    assert numbers == [2, 3]


# ==========================================================================
# Boundary cases
# ==========================================================================


def test_an_empty_result_set_yields_nothing():
    assert list(fetch_pull_requests(session_with_pages([[]]), "o/r")) == []


def test_a_partial_final_page_is_returned_in_full():
    session = session_with_pages([[pr_payload(n) for n in range(1, 4)]])

    assert len(list(fetch_pull_requests(session, "o/r"))) == 3


def test_an_empty_page_with_a_next_link_keeps_walking():
    session = session_with_pages([[], [pr_payload(2)]])

    assert [pr.number for pr in fetch_pull_requests(session, "o/r")] == [2]


def test_a_page_of_only_pull_requests_yields_no_issues():
    session = session_with_pages([[issue_payload(1, is_pull_request=True)]])

    assert list(fetch_issues(session, "o/r")) == []


def test_an_object_where_a_list_was_expected_is_an_error():
    session = session_returning(lambda r: httpx.Response(200, json={"message": "nope"}))

    with pytest.raises(GitHubError) as excinfo:
        list(fetch_pull_requests(session, "o/r"))

    assert "expected a list" in str(excinfo.value)


def test_a_malformed_item_names_the_endpoint_it_came_from():
    session = session_with_pages([[{"number": "not-a-number"}]])

    with pytest.raises(GitHubError) as excinfo:
        list(fetch_pull_requests(session, "o/r"))

    message = str(excinfo.value)
    assert "PullRequest" in message
    assert "/repos/o/r/pulls" in message


def test_an_item_missing_a_required_field_is_an_error():
    payload = pr_payload(1)
    del payload["created_at"]
    session = session_with_pages([[payload]])

    with pytest.raises(GitHubError):
        list(fetch_pull_requests(session, "o/r"))


def test_a_malformed_repository_payload_is_an_error():
    session = session_returning(lambda r: httpx.Response(200, json={"id": "x"}))

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "Repository" in str(excinfo.value)


def test_one_bad_item_fails_the_stream_rather_than_being_skipped():
    """Silently dropping an item would make an incomplete graph look complete."""
    session = session_with_pages([[pr_payload(1), {"number": "bad"}]])

    stream = fetch_pull_requests(session, "o/r")
    assert next(stream).number == 1
    with pytest.raises(GitHubError):
        next(stream)


# ==========================================================================
# Error handling
# ==========================================================================


def test_a_missing_repository_reports_the_status_and_the_name():
    session = session_returning(
        lambda r: httpx.Response(404, json={"message": "Not Found"})
    )

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/does-not-exist")

    message = str(excinfo.value)
    assert "404" in message
    assert "o/does-not-exist" in message


def test_bad_credentials_report_401():
    session = session_returning(
        lambda r: httpx.Response(401, json={"message": "Bad credentials"})
    )

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "401" in str(excinfo.value)
    assert "Bad credentials" in str(excinfo.value)


@pytest.mark.parametrize("status", [400, 422, 500, 502, 503])
def test_other_failures_report_their_status(status):
    session = session_returning(lambda r: httpx.Response(status, json={"message": "x"}))

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert str(status) in str(excinfo.value)


def test_githubs_own_explanation_reaches_the_message():
    session = session_returning(
        lambda r: httpx.Response(422, json={"message": "Validation failed"})
    )

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "Validation failed" in str(excinfo.value)


def test_an_error_with_a_non_json_body_still_reports_cleanly():
    session = session_returning(lambda r: httpx.Response(500, text="<html>oops</html>"))

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "500" in str(excinfo.value)


def test_an_error_with_an_unexpected_json_shape_still_reports_cleanly():
    session = session_returning(lambda r: httpx.Response(500, json=["a", "list"]))

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "500" in str(excinfo.value)


def test_a_failure_partway_through_pagination_surfaces():
    def handler(request: httpx.Request) -> httpx.Response:
        if "page" in request.url.params:
            return httpx.Response(500, json={"message": "boom"})
        return httpx.Response(
            200,
            json=[pr_payload(1)],
            headers={
                "Link": '<https://api.github.com/repos/o/r/pulls?page=2>; rel="next"'
            },
        )

    stream = fetch_pull_requests(session_returning(handler), "o/r")
    assert next(stream).number == 1
    with pytest.raises(GitHubError):
        next(stream)


def test_a_connection_failure_becomes_a_github_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session_returning(handler), "o/r")

    assert "no route to host" in str(excinfo.value)


def test_a_timeout_becomes_a_github_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session_returning(handler), "o/r")

    assert "timed out" in str(excinfo.value)


def test_every_failure_is_the_same_exception_type():
    """One exception type, so a caller needs one except clause."""
    for handler in (
        lambda r: httpx.Response(404, json={"message": "x"}),
        lambda r: httpx.Response(500, text="x"),
        lambda r: httpx.Response(200, json={"not": "a list"}),
    ):
        with pytest.raises(GitHubError):
            list(fetch_pull_requests(session_returning(handler), "o/r"))


# ==========================================================================
# Rate limiting
#
# The client raises on a spent quota. It does not wait, back off, or retry —
# that was left out until real failures justify it, so these tests pin the
# reporting behaviour rather than any recovery behaviour.
# ==========================================================================


def rate_limited(status: int, *, remaining: str = "0", reset: str | None = None):
    headers = {"x-ratelimit-remaining": remaining}
    if reset is not None:
        headers["x-ratelimit-reset"] = reset
    return lambda request: httpx.Response(
        status, json={"message": "API rate limit exceeded"}, headers=headers
    )


@pytest.mark.parametrize("status", [403, 429])
def test_a_spent_quota_is_reported_as_a_rate_limit(status):
    reset_at = str(int(time.time()) + 1800)
    session = session_returning(rate_limited(status, reset=reset_at))

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "rate limit" in str(excinfo.value).lower()


def test_the_rate_limit_message_says_when_the_quota_resets():
    reset_at = str(int(time.time()) + 1800)
    session = session_returning(rate_limited(403, reset=reset_at))

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    message = str(excinfo.value)
    assert "resets at" in message
    assert reset_at in message


def test_a_rate_limit_without_a_reset_header_says_so():
    session = session_returning(rate_limited(403))

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "reset time not reported" in str(excinfo.value)


def test_an_unparseable_reset_header_does_not_crash():
    session = session_returning(rate_limited(403, reset="not-a-number"))

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "not-a-number" in str(excinfo.value)


def test_a_forbidden_response_with_quota_left_is_not_a_rate_limit():
    session = session_returning(
        lambda r: httpx.Response(
            403, json={"message": "Forbidden"}, headers={"x-ratelimit-remaining": "42"}
        )
    )

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "rate limit" not in str(excinfo.value).lower()


def test_a_forbidden_response_without_quota_headers_is_not_a_rate_limit():
    session = session_returning(
        lambda r: httpx.Response(403, json={"message": "Forbidden"})
    )

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "rate limit" not in str(excinfo.value).lower()


def test_a_forbidden_response_is_reported_even_with_no_body():
    session = session_returning(lambda r: httpx.Response(403))

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert "403" in str(excinfo.value)



