"""GitHub data acquisition for the ingestion pipeline.

The flow is deliberately flat: ``make_session`` builds an ``httpx.Client``,
``_request`` performs one GET and turns failures into ``GitHubError``,
``_paginate`` walks GitHub's ``Link`` headers, and each ``fetch_*`` function
turns raw pages into validated models.

Every public function takes the session as its first argument, so tests pass an
``httpx.Client`` backed by ``httpx.MockTransport`` and exercise this module for
real without touching the network.

Retrying transient failures
---------------------------

``_request`` retries in the request layer, three attempts with a short
exponential backoff, and only for failures that a second attempt could plausibly
fix: a 5xx, or a connection or read timeout where no response arrived at all.

Everything else raises on the first attempt. A 4xx is a statement about the
request and will say the same thing again. An auth failure never becomes valid
by asking twice. **A rate limit is never slept off here** — waiting out a
primary GitHub limit can block for an hour, so it propagates carrying its reset
time and the run's owner decides.

This retries at the level of one HTTP request, so a paginated walk resumes at
the page that failed instead of restarting from the first. An earlier version
retried caller-side around whole fetches, which could not do that.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from itertools import islice
from typing import Any, Iterable, Iterator

import httpx
from pydantic import BaseModel, ValidationError

from ..common.retry import with_retry

from .models import ChangedFile, Commit, Issue, PullRequest, Repository, Review

API_ROOT = "https://api.github.com"

#: GitHub's maximum. Anything smaller multiplies the request count for nothing.
PER_PAGE = 100
TIMEOUT = 30.0

#: How many pages ``fetch_issues`` may walk before giving up on filling its
#: limit.
#:
#: **Chosen, not measured**, because no measurement can settle it: the issues
#: endpoint returns pull requests too and they are dropped here, so a page can
#: contribute nothing and no bound on pages follows from a bound on results.
#: Without a cap the walk ends only when the repository does.
#:
#: Ten pages is a thousand raw items at ``PER_PAGE``. A tracker that has not
#: produced the asked-for issues in a thousand items is one where the caller
#: wanted a different repository or a different limit, not more requests.
#:
#: Hitting the cap under-delivers rather than raising, which is the behaviour a
#: limit already has: asking for ten issues from a repository holding two
#: returns two. The run summary reports what was found either way.
MAX_ISSUE_PAGES = 10

#: Sent explicitly on every list endpoint that accepts them, rather than
#: relying on GitHub's defaults. The defaults are not part of the API contract
#: and have changed before; a silent change to them would silently change which
#: items a limited fetch returns.
#:
#: ``sort=updated`` because the interesting items are the recently touched
#: ones, and a ``--prs 50`` run should get the 50 that matter rather than the
#: 50 oldest. ``direction=desc`` pairs with it to mean newest first.
#:
#: This does not make the pipeline's output order server-dependent — see
#: ``order.py``, which sorts locally before anything is built.
LIST_PARAMS = {"sort": "updated", "direction": "desc"}


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


class GitHubTransportError(GitHubError):
    """No response arrived: a connection failure or a timeout.

    Separated from ``GitHubError`` because it is the one failure with no status
    code that is still worth retrying. Judging it by ``status_code is None``
    would also sweep in malformed-payload errors, which retrying cannot fix —
    a response that arrived and could not be parsed will not parse on the
    second attempt, and re-requesting it spends a rate-limit token to learn
    nothing.

    This type exists because the absence of a status code means two different
    things and only one of them is transient. A dedicated type says which,
    where a ``None`` check can only say "no status".

    Worth recording how this was found: a timeout used to be permanent, since
    the transience predicate required a status code and a transport failure has
    none. A test asserted that permanence — so the suite was not silent about
    the behaviour, it was defending it. Inverting that test was part of the
    fix, which is why a green suite is not by itself evidence that a rule is
    the intended one.
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


def is_transient(error: BaseException) -> bool:
    """Whether a second attempt could plausibly succeed.

    Only two cases qualify. A 5xx is the server saying it failed, not that the
    request was wrong. A transport failure means no response arrived, so nothing
    has been learned about the request at all.

    Auth and rate-limit failures are checked first and excluded even though a
    rate limit can carry a 5xx-shaped status in odd deployments: neither is
    fixed by asking again, and sleeping off a rate limit here could block the
    run for an hour.
    """
    if isinstance(error, (GitHubAuthError, GitHubRateLimitError)):
        return False
    if isinstance(error, GitHubTransportError):
        return True
    if isinstance(error, GitHubError):
        return error.status_code is not None and error.status_code >= 500
    return False


def _request_once(
    session: httpx.Client, url: str, params: dict[str, Any] | None = None
) -> httpx.Response:
    """GET ``url`` once, or raise the exception that says what went wrong."""
    try:
        response = session.get(url, params=params)
    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
        # No response arrived. Worth another attempt.
        raise GitHubTransportError(f"request to {url} failed: {exc}") from exc
    except httpx.HTTPError as exc:
        # A protocol or decoding failure. A second attempt would fail the same.
        raise GitHubError(f"request to {url} failed: {exc}") from exc

    if response.is_success:
        return response

    raise _failure(url, response)


def _request(
    session: httpx.Client, url: str, params: dict[str, Any] | None = None
) -> httpx.Response:
    """GET ``url``, retrying only what a retry could fix.

    **The retry belongs here, around one request, and not around the caller.**
    A list endpoint is walked page by page, and a failure on page nine of
    twelve is a failure of that page only. Retrying at this level resumes on
    the page that failed; retrying around the caller can only restart the walk
    from page one, re-fetching eight pages that already succeeded and paying
    their rate-limit cost again. There is no way to express "resume here" from
    outside the loop, because the caller does not hold the cursor.

    It also reaches everything. Placed at one call site, a retry protects that
    call site; placed here, it covers the repository, pull request, issue and
    commit fetches alike, none of which would otherwise have any.

    Bounded at ``retry.MAX_ATTEMPTS`` total attempts. When they are exhausted
    the last exception propagates unchanged — there is no partial result and no
    bookkeeping, so a top-level fetch either completes or aborts the ingest.

    Exactly one layer retries. A second retry wrapped around a caller of this
    would multiply, not add: three attempts here inside three attempts there is
    nine requests for one logical fetch.
    """
    return with_retry(
        lambda: _request_once(session, url, params),
        retryable=is_transient,
    )


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
    session: httpx.Client,
    path: str,
    params: dict[str, Any] | None = None,
    *,
    max_pages: int | None = None,
) -> Iterator[dict]:
    """Yield raw items from ``path``, following ``Link`` ``rel="next"``.

    Pages are fetched lazily, so a caller that stops early stops the requests
    too. Query parameters apply to the first request only; GitHub's next URL
    already carries them.

    ``max_pages`` stops the walk after that many requests. It is for callers
    that drop items after they arrive: their own limit counts survivors, which
    bounds nothing when a page contributes none. A caller that keeps every item
    needs no bound, because its limit already stops the generator.
    """
    url: str | None = path
    request_params: dict[str, Any] | None = {**(params or {}), "per_page": PER_PAGE}
    pages = 0

    while url is not None:
        response = _request(session, url, params=request_params)
        request_params = None
        pages += 1

        items = response.json()
        if not isinstance(items, list):
            raise GitHubError(f"expected a list of items from {url}, got {type(items).__name__}")

        yield from items

        # Checked after yielding, so a consumer that stops inside this page
        # never pays for the check, and the page just served is counted.
        if max_pages is not None and pages >= max_pages:
            return

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
    items = _paginate(session, path, {"state": state, **LIST_PARAMS})
    yield from islice(_validated(PullRequest, items, path), limit)


def fetch_issues(
    session: httpx.Client, repo: str, *, state: str = "all", limit: int | None = None
) -> Iterator[Issue]:
    """Yield issues.

    GitHub's issues endpoint also returns pull requests; those carry a
    ``pull_request`` key and are dropped here so ``limit`` counts real issues.

    Because that filter runs after each page arrives, ``limit`` bounds results
    and not requests: a page of nothing but pull requests contributes no issue
    to stop on. ``MAX_ISSUE_PAGES`` bounds the walk so a tracker holding few
    issues cannot spend a whole rate limit looking for them.
    """
    path = f"/repos/{repo}/issues"
    items = _paginate(
        session, path, {"state": state, **LIST_PARAMS}, max_pages=MAX_ISSUE_PAGES
    )
    issues = (item for item in items if "pull_request" not in item)
    yield from islice(_validated(Issue, issues, path), limit)


def fetch_commits(
    session: httpx.Client, repo: str, *, limit: int | None = None
) -> Iterator[Commit]:
    """Yield commits from the repository's default branch.

    No ``sort`` or ``direction``: the commits endpoint accepts neither, and
    sending them is silently ignored rather than honoured. Commit order is
    settled locally instead — see ``order.py``.
    """
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

