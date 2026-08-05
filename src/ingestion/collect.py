"""Resilient collection across many pull requests.

This sits above the client: the client reports one request's outcome, and this
decides what a run does about it. Retrying and giving up both live here, not in
``github.py``.

**Deliberate, and not to be "corrected" back.** Aborting an entire fetch
because one pull request's enrichment call failed loses the other forty-nine
pull requests, the issues, and the commits. That is a real defect regardless of
what a simpler arrangement would do. Enrichment is per item and degrades per
item; the count of what was lost travels with the graph so a partial result is
never mistaken for a complete one.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import httpx

from ..common.retry import with_retry
from .github import GitHubAuthError, GitHubError, GitHubRateLimitError

PerPullRequestFetch = Callable[[httpx.Client, str, int], Iterable[Any]]


def is_retryable(error: BaseException) -> bool:
    """Only a server-side failure is worth a second attempt.

    An auth failure never becomes valid by asking again. A spent rate limit is
    not slept off here — waiting out a primary GitHub limit can block for an
    hour, so it propagates with its reset time and the run's owner decides.
    """
    if isinstance(error, (GitHubAuthError, GitHubRateLimitError)):
        return False
    if isinstance(error, GitHubError):
        return error.status_code is not None and error.status_code >= 500
    return False


def collect_by_pull_request(
    fetch: PerPullRequestFetch,
    session: httpx.Client,
    repo: str,
    numbers: Iterable[int],
) -> tuple[dict[int, list], int]:
    """Run a per-pull-request fetch across many numbers, surviving failures.

    Each call is retried on a server-side failure. A call that still fails
    drops that one pull request's data and the walk continues.

    Returns the results and how many pull requests were lost. The count is
    returned rather than logged because a graph that quietly lost a pull
    request's reviews looks identical to one that never had any.
    """
    results: dict[int, list] = {}
    failures = 0

    for number in numbers:
        try:
            # ``n=number`` binds the value now rather than closing over the
            # loop variable, so the retry cannot fetch a different number.
            results[number] = with_retry(
                lambda n=number: list(fetch(session, repo, n)),
                retryable=is_retryable,
            )
        except GitHubError:
            failures += 1

    return results, failures
