"""Tests for caller-side retrying and the typed GitHub exceptions.

Retry is not in the client, so it is tested away from it. What matters here is
which failures get another attempt and which do not — an auth failure retried
three times is three rejections, and a rate limit slept off can block a run for
an hour.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from src.common import retry
from src.common.retry import MAX_ATTEMPTS, with_retry
from src.ingestion.github import GitHubTransportError, is_transient
from src.ingestion import (
    API_ROOT,
    GitHubAuthError,
    GitHubError,
    GitHubRateLimitError,
    collect_by_pull_request,
    fetch_pull_requests,
    fetch_repository,
    fetch_reviews,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def session_returning(handler) -> httpx.Client:
    return httpx.Client(base_url=API_ROOT, transport=httpx.MockTransport(handler))


def repo_payload() -> dict:
    return {
        "id": 1,
        "name": "r",
        "full_name": "o/r",
        "private": False,
        "owner": {"login": "octocat", "id": 1},
        "html_url": "https://github.com/o/r",
        "description": None,
        "language": None,
        "default_branch": "main",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "pushed_at": None,
    }


def review_payload(review_id: int) -> dict:
    return {
        "id": review_id,
        "state": "APPROVED",
        "body": "",
        "user": {"login": "octocat", "id": 1},
        "html_url": "https://github.com/o/r/pull/1",
        "commit_id": "abc",
        "submitted_at": "2024-01-02T00:00:00Z",
    }


def pr_payload(number: int) -> dict:
    """The minimum a PullRequest validates from."""
    return {
        "id": 1000 + number,
        "number": number,
        "title": f"PR {number}",
        "state": "open",
        "user": {"login": "alice", "id": 1, "type": "User"},
        "body": "",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "closed_at": None,
        "merged_at": None,
        "html_url": "https://github.com/o/r/pull/1",
        "labels": [],
    }


def counting_session(statuses: list[int], recorder: list) -> httpx.Client:
    """Serves ``statuses`` in order, then 200 forever."""

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.append(request)
        index = len(recorder) - 1
        if index < len(statuses):
            return httpx.Response(statuses[index], json={"message": "boom"})
        return httpx.Response(200, json=repo_payload())

    return httpx.Client(base_url=API_ROOT, transport=httpx.MockTransport(handler))


# ==========================================================================
# The typed exceptions
# ==========================================================================


def test_a_401_is_an_auth_error():
    session = session_returning(
        lambda r: httpx.Response(401, json={"message": "Bad credentials"})
    )

    with pytest.raises(GitHubAuthError) as excinfo:
        fetch_repository(session, "o/r")

    assert excinfo.value.status_code == 401
    assert "Bad credentials" in str(excinfo.value)


def test_a_spent_quota_is_a_rate_limit_error_carrying_its_reset():
    reset_epoch = 1735689600  # 2025-01-01T00:00:00Z
    session = session_returning(
        lambda r: httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
            headers={
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": str(reset_epoch),
                "retry-after": "60",
            },
        )
    )

    with pytest.raises(GitHubRateLimitError) as excinfo:
        fetch_repository(session, "o/r")

    error = excinfo.value
    assert error.status_code == 403
    assert error.reset_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert error.retry_after == 60.0


def test_a_429_is_also_a_rate_limit_error():
    session = session_returning(
        lambda r: httpx.Response(
            429, json={"message": "too many"}, headers={"x-ratelimit-remaining": "0"}
        )
    )

    with pytest.raises(GitHubRateLimitError):
        fetch_repository(session, "o/r")


def test_a_rate_limit_without_headers_reports_no_reset():
    session = session_returning(
        lambda r: httpx.Response(
            429, json={"message": "too many"}, headers={"x-ratelimit-remaining": "0"}
        )
    )

    with pytest.raises(GitHubRateLimitError) as excinfo:
        fetch_repository(session, "o/r")

    assert excinfo.value.reset_at is None
    assert excinfo.value.retry_after is None


def test_a_forbidden_response_with_quota_left_is_a_plain_error():
    session = session_returning(
        lambda r: httpx.Response(
            403, json={"message": "Forbidden"}, headers={"x-ratelimit-remaining": "42"}
        )
    )

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert not isinstance(excinfo.value, GitHubRateLimitError)
    assert excinfo.value.status_code == 403


@pytest.mark.parametrize("status", [400, 404, 422, 500, 503])
def test_every_failure_carries_its_status_code(status):
    session = session_returning(lambda r: httpx.Response(status, json={"message": "x"}))

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session, "o/r")

    assert excinfo.value.status_code == status


def test_a_transport_failure_has_no_status_code():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with pytest.raises(GitHubError) as excinfo:
        fetch_repository(session_returning(handler), "o/r")

    assert excinfo.value.status_code is None


def test_the_typed_errors_are_all_catchable_as_one():
    """A caller that does not care about the distinction needs one clause."""
    assert issubclass(GitHubAuthError, GitHubError)
    assert issubclass(GitHubRateLimitError, GitHubError)


# ==========================================================================
# What is retryable
# ==========================================================================


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_a_server_error_is_transient(status):
    assert is_transient(GitHubError("boom", status_code=status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_a_client_error_is_not_retryable(status):
    assert not is_transient(GitHubError("nope", status_code=status))


def test_an_auth_error_is_never_retryable():
    assert not is_transient(GitHubAuthError("bad token", status_code=401))


def test_a_rate_limit_is_never_retryable_here():
    """Waiting out a primary limit can block for an hour — the caller decides."""
    assert not is_transient(GitHubRateLimitError("spent", status_code=403))


def test_a_statusless_error_that_is_not_a_transport_failure_is_not_retryable():
    """A malformed payload has no status code either, and retrying cannot fix it."""
    assert not is_transient(GitHubError("unexpected payload", status_code=None))


def test_a_transport_failure_is_transient():
    """No response arrived, so nothing was learned about the request."""
    assert is_transient(GitHubTransportError("read timeout"))


def test_an_unrelated_exception_is_not_retryable():
    assert not is_transient(ValueError("unrelated"))


# ==========================================================================
# The request layer retries. Every test below drives a real fetch and counts
# the requests that actually reached the transport, so the attempt bound is
# asserted rather than assumed.
# ==========================================================================


def test_a_successful_request_is_not_repeated():
    requests: list[httpx.Request] = []
    session = counting_session([], requests)

    fetch_repository(session, "o/r")

    assert len(requests) == 1


def test_a_500_followed_by_a_success_returns_the_success():
    """The retry is counted: two requests, one result."""
    requests: list[httpx.Request] = []
    session = counting_session([500], requests)

    repo = fetch_repository(session, "o/r")

    assert repo.full_name == "o/r"
    assert len(requests) == 2


def test_three_consecutive_500s_raise():
    requests: list[httpx.Request] = []
    session = counting_session([500] * 10, requests)

    with pytest.raises(GitHubError):
        fetch_repository(session, "o/r")

    assert len(requests) == MAX_ATTEMPTS


def test_the_attempt_count_is_bounded_at_three():
    """Pinned explicitly: a bound that drifts is a bound that is not one."""
    assert MAX_ATTEMPTS == 3


@pytest.mark.parametrize("status", [400, 404, 409, 422])
def test_a_client_error_raises_immediately_with_no_retry(status):
    requests: list[httpx.Request] = []
    session = counting_session([status] * 10, requests)

    with pytest.raises(GitHubError):
        fetch_repository(session, "o/r")

    assert len(requests) == 1


def test_an_authentication_failure_raises_immediately_with_no_retry():
    requests: list[httpx.Request] = []
    session = counting_session([401] * 10, requests)

    with pytest.raises(GitHubAuthError):
        fetch_repository(session, "o/r")

    assert len(requests) == 1


def test_a_rate_limit_raises_immediately_carrying_its_reset(monkeypatch):
    """No retry and no sleep — waiting one out can block for an hour."""
    slept: list[float] = []
    monkeypatch.setattr(retry.time, "sleep", lambda seconds: slept.append(seconds))

    reset = int(datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc).timestamp())
    requests: list[httpx.Request] = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            403,
            json={"message": "rate limit exceeded"},
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset)},
        )

    session = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )

    with pytest.raises(GitHubRateLimitError) as caught:
        fetch_repository(session, "o/r")

    assert len(requests) == 1
    assert slept == []
    assert caught.value.reset_at is not None


def test_a_timeout_is_retried():
    """A transport failure means no response arrived. Worth another attempt."""
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json=repo_payload())

    session = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )

    assert fetch_repository(session, "o/r").full_name == "o/r"
    assert len(attempts) == 3


def test_a_timeout_that_never_clears_raises_after_the_bound():
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        raise httpx.ConnectTimeout("timed out", request=request)

    session = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )

    with pytest.raises(GitHubTransportError):
        fetch_repository(session, "o/r")

    assert len(attempts) == MAX_ATTEMPTS


def test_the_backoff_grows_between_attempts(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(retry, "BACKOFF_SECONDS", 0.5)
    monkeypatch.setattr(retry.time, "sleep", lambda seconds: slept.append(seconds))

    session = counting_session([500] * 10, [])
    with pytest.raises(GitHubError):
        fetch_repository(session, "o/r")

    assert slept == [0.5, 1.0]


def test_a_paginated_walk_retries_only_the_failed_page():
    """Retrying inside one request resumes rather than restarting the walk."""
    seen: list[str] = []

    def handler(request):
        page = request.url.params.get("page", "1")
        seen.append(page)
        if page == "1":
            return httpx.Response(
                200,
                json=[pr_payload(1)],
                headers={
                    "link": '<https://api.github.com/repos/o/r/pulls?page=2>; rel="next"'
                },
            )
        if seen.count("2") == 1:
            return httpx.Response(500, json={"message": "boom"})
        return httpx.Response(200, json=[pr_payload(2)])

    session = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )

    assert len(list(fetch_pull_requests(session, "o/r"))) == 2
    # Page 1 fetched once, not re-fetched when page 2 failed.
    assert seen.count("1") == 1


# ==========================================================================
# Per-pull-request enrichment
#
# One request per pull request, so a long run has many chances to fail. A
# failure must cost that pull request's extras and nothing else — aborting
# the whole fetch would throw away forty-nine good pull requests for one bad.
# ==========================================================================


def enrichment_session(failing: set[int], recorder: list) -> httpx.Client:
    """Reviews succeed except for the pull request numbers in ``failing``."""

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.append(request)
        number = int(request.url.path.split("/pulls/")[1].split("/")[0])
        if number in failing:
            return httpx.Response(500, json={"message": "boom"})
        return httpx.Response(200, json=[review_payload(number)])

    return httpx.Client(base_url=API_ROOT, transport=httpx.MockTransport(handler))


def test_enrichment_collects_every_pull_request_when_nothing_fails():
    results, failures = collect_by_pull_request(
        fetch_reviews, enrichment_session(set(), []), "o/r", [1, 2, 3]
    )

    assert sorted(results) == [1, 2, 3]
    assert failures == 0


def test_one_failing_pull_request_does_not_lose_the_others():
    results, failures = collect_by_pull_request(
        fetch_reviews, enrichment_session({2}, []), "o/r", [1, 2, 3]
    )

    assert sorted(results) == [1, 3]
    assert failures == 1


def test_every_pull_request_failing_still_returns_rather_than_raising():
    results, failures = collect_by_pull_request(
        fetch_reviews, enrichment_session({1, 2, 3}, []), "o/r", [1, 2, 3]
    )

    assert results == {}
    assert failures == 3


def test_each_pull_request_is_retried_before_being_given_up_on():
    requests: list[httpx.Request] = []

    collect_by_pull_request(
        fetch_reviews, enrichment_session({2}, requests), "o/r", [1, 2, 3]
    )

    failed = [r for r in requests if "/pulls/2/" in r.url.path]
    assert len(failed) == MAX_ATTEMPTS


def test_retrying_fetches_the_number_it_started_with():
    """A late-bound loop variable would retry the wrong pull request."""
    requests: list[httpx.Request] = []

    collect_by_pull_request(
        fetch_reviews, enrichment_session({2}, requests), "o/r", [1, 2, 3]
    )

    retried = [r.url.path for r in requests if "/pulls/2/" in r.url.path]
    assert set(retried) == {"/repos/o/r/pulls/2/reviews"}


def test_an_auth_failure_during_enrichment_is_not_retried():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(401, json={"message": "Bad credentials"})

    _, failures = collect_by_pull_request(
        fetch_reviews, session_returning(handler), "o/r", [1]
    )

    assert failures == 1
    assert len(requests) == 1
