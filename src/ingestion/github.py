"""GitHub data acquisition for the ingestion pipeline.

The flow is deliberately flat: ``make_session`` builds an ``httpx.Client``,
``_request`` performs one GET and turns failures into ``GitHubError``,
``_paginate`` walks GitHub's ``Link`` headers, and each ``fetch_*`` function
turns raw pages into validated models.

Every public function takes the session as its first argument, so tests pass an
``httpx.Client`` backed by ``httpx.MockTransport`` and exercise this module for
real without touching the network.

**The client never retries and never sleeps.** It raises a typed exception and
lets the caller decide, because only the caller knows whether waiting is
acceptable. Auth failures must never be retried at all; a rate limit carries
its reset time so a caller can choose between waiting and giving up. Retrying
lives in ``src/common/retry.py``, above this module.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from itertools import islice
from typing import Any, Iterable, Iterator

import httpx
from pydantic import BaseModel, ValidationError

from .models import ChangedFile, Commit, Issue, PullRequest, Repository, Review

API_ROOT = "https://api.github.com"
PER_PAGE = 100
TIMEOUT = 30.0


class GitHubError(RuntimeError):
    """Any failure while acquiring or validating GitHub data.

    ``status_code`` is the HTTP status when there was a response, and ``None``
    for a transport failure or a malformed payload. Callers scope their retry
    policy off it — retrying a 404 changes nothing, retrying a 503 might.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GitHubAuthError(GitHubError):
    """Credentials are missing, wrong, or lack the required scope (401).

    Never retryable. The token will not become valid by asking again.
    """


class GitHubRateLimitError(GitHubError):
    """The quota is spent (403 or 429 with no requests remaining).

    Carries ``reset_at`` and ``retry_after`` so a caller can decide between
    waiting and stopping. The client does not decide, because waiting out a
    primary rate limit can block for an hour.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        reset_at: datetime | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.reset_at = reset_at
        self.retry_after = retry_after


# --------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------


def make_session(token: str | None = None) -> httpx.Client:
    """Build a session for the GitHub REST API.

    Falls back to ``$GITHUB_TOKEN``. Without a token GitHub allows 60 requests
    per hour, which is enough for a smoke test and not enough for an ingest.
    """
    token = token or os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=API_ROOT, headers=headers, timeout=TIMEOUT)


# --------------------------------------------------------------------------
# request / pagination helpers
# --------------------------------------------------------------------------


def _request(
    session: httpx.Client, url: str, params: dict[str, Any] | None = None
) -> httpx.Response:
    """GET ``url`` once, or raise the exception that says what went wrong.

    One attempt, no sleeping. Whether a failure is worth another try depends on
    how long the caller is willing to wait, which this module cannot know.
    """
    try:
        response = session.get(url, params=params)
    except httpx.HTTPError as exc:
        raise GitHubError(f"request to {url} failed: {exc}") from exc

    if response.is_success:
        return response

    raise _failure(url, response)


def _failure(url: str, response: httpx.Response) -> GitHubError:
    """Build the exception for a non-2xx response."""
    status = response.status_code
    detail = _github_message(response)

    if status == 401:
        return GitHubAuthError(
            f"GitHub rejected the credentials for {url} (HTTP {status}). "
            f"{detail}".strip(),
            status_code=status,
        )

    if _is_rate_limited(response):
        return GitHubRateLimitError(
            f"GitHub rate limit exceeded for {url} (HTTP {status}); "
            f"{_reset_description(response)}. {detail}".strip(),
            status_code=status,
            reset_at=_reset_at(response),
            retry_after=_retry_after(response),
        )

    return GitHubError(
        f"GitHub request to {url} failed with HTTP {status}. {detail}".strip(),
        status_code=status,
    )


def _reset_at(response: httpx.Response) -> datetime | None:
    """When the quota refills, from ``x-ratelimit-reset``."""
    reset = response.headers.get("x-ratelimit-reset")
    if not reset:
        return None
    try:
        return datetime.fromtimestamp(int(reset), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds GitHub asked us to wait, from ``Retry-After``."""
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _github_message(response: httpx.Response) -> str:
    """GitHub's own explanation, when the body carries one."""
    try:
        body = response.json()
    except ValueError:
        return ""
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return body["message"]
    return ""


def _is_rate_limited(response: httpx.Response) -> bool:
    """True when GitHub refused the request because the quota is spent."""
    if response.status_code not in (403, 429):
        return False
    return response.headers.get("x-ratelimit-remaining") == "0"


def _reset_description(response: httpx.Response) -> str:
    """Human-readable form of the ``x-ratelimit-reset`` epoch, when present."""
    reset = response.headers.get("x-ratelimit-reset")
    if not reset:
        return "reset time not reported"
    try:
        moment = datetime.fromtimestamp(int(reset), tz=timezone.utc)
    except (TypeError, ValueError):
        return f"resets at {reset}"
    return f"resets at {moment.isoformat()} (epoch {reset})"


def _paginate(
    session: httpx.Client, path: str, params: dict[str, Any] | None = None
) -> Iterator[dict]:
    """Yield raw items from ``path``, following ``Link`` ``rel="next"``.

    Pages are fetched lazily, so a caller that stops early stops the requests
    too. Query parameters apply to the first request only; GitHub's next URL
    already carries them.
    """
    url: str | None = path
    request_params: dict[str, Any] | None = {**(params or {}), "per_page": PER_PAGE}

    while url is not None:
        response = _request(session, url, params=request_params)
        request_params = None

        items = response.json()
        if not isinstance(items, list):
            raise GitHubError(f"expected a list of items from {url}, got {type(items).__name__}")

        yield from items
        url = response.links.get("next", {}).get("url")


def _validate(model: type[BaseModel], item: dict, source: str):
    """Turn one raw payload into a model, or fail loudly about where it came from."""
    try:
        return model.model_validate(item)
    except ValidationError as exc:
        raise GitHubError(
            f"unexpected {model.__name__} payload from {source}: {exc}"
        ) from exc


def _validated(
    model: type[BaseModel], items: Iterable[dict], source: str
) -> Iterator[Any]:
    """Validate a stream of raw payloads."""
    for item in items:
        yield _validate(model, item, source)


# --------------------------------------------------------------------------
# endpoint functions
# --------------------------------------------------------------------------


def fetch_repository(session: httpx.Client, repo: str) -> Repository:
    """Fetch repository metadata. ``repo`` is ``"owner/name"``."""
    path = f"/repos/{repo}"
    return _validate(Repository, _request(session, path).json(), path)


def fetch_pull_requests(
    session: httpx.Client, repo: str, *, state: str = "all", limit: int | None = None
) -> Iterator[PullRequest]:
    """Yield pull requests, newest first as GitHub orders them."""
    path = f"/repos/{repo}/pulls"
    items = _paginate(session, path, {"state": state})
    yield from islice(_validated(PullRequest, items, path), limit)


def fetch_issues(
    session: httpx.Client, repo: str, *, state: str = "all", limit: int | None = None
) -> Iterator[Issue]:
    """Yield issues.

    GitHub's issues endpoint also returns pull requests; those carry a
    ``pull_request`` key and are dropped here so ``limit`` counts real issues.
    """
    path = f"/repos/{repo}/issues"
    items = _paginate(session, path, {"state": state})
    issues = (item for item in items if "pull_request" not in item)
    yield from islice(_validated(Issue, issues, path), limit)


def fetch_commits(
    session: httpx.Client, repo: str, *, limit: int | None = None
) -> Iterator[Commit]:
    """Yield commits from the repository's default branch."""
    path = f"/repos/{repo}/commits"
    yield from islice(_validated(Commit, _paginate(session, path), path), limit)


def fetch_reviews(
    session: httpx.Client, repo: str, pr_number: int
) -> Iterator[Review]:
    """Yield reviews submitted on one pull request."""
    path = f"/repos/{repo}/pulls/{pr_number}/reviews"
    yield from _validated(Review, _paginate(session, path), path)


def fetch_changed_files(
    session: httpx.Client, repo: str, pr_number: int
) -> Iterator[ChangedFile]:
    """Yield files touched by one pull request.

    GitHub caps this endpoint at 3000 files, so a very large pull request is
    reported incompletely. See the module README.
    """
    path = f"/repos/{repo}/pulls/{pr_number}/files"
    yield from _validated(ChangedFile, _paginate(session, path), path)

