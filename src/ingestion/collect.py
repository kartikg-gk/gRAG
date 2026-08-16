"""Resilient collection across many pull requests.

This sits above the client. The client now retries a transient failure inside
one request — see ``github._request`` — and this decides what the run does when
that has already been tried and still failed. Retrying is not repeated here;
doing both would give one enrichment call nine attempts instead of three.

**Enrichment degrades per item; do not make it abort the run.** Aborting an
entire fetch because one pull request's enrichment call failed loses the other
forty-nine pull requests, the issues, and the commits — a whole ingest thrown
away for one missing review list. So a failed item is dropped and the walk
continues, and the count of what was lost travels with the graph so a partial
result is never mistaken for a complete one.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import httpx

from .github import GitHubError

PerPullRequestFetch = Callable[[httpx.Client, str, int], Iterable[Any]]

#: The enrichment stages, so a failure count can say which hole it left in the
#: graph. A total alone cannot distinguish a run that lost its reviews from one
#: that lost its file lists.
STAGE_REVIEWS = "reviews"
STAGE_CHANGED_FILES = "changed_files"


def collect_by_pull_request(
    fetch: PerPullRequestFetch,
    session: httpx.Client,
    repo: str,
    numbers: Iterable[int],
) -> tuple[dict[int, list], int]:
    """Run a per-pull-request fetch across many numbers, surviving failures.

    Transient failures are already retried inside the client, so a call that
    reaches here has failed every attempt. It drops that one pull request's
    data and the walk continues.

    Returns the results and how many pull requests were lost. The count is
    returned rather than logged because a graph that quietly lost a pull
    request's reviews looks identical to one that never had any.
    """
    results: dict[int, list] = {}
    failures = 0

    for number in numbers:
        try:
            results[number] = list(fetch(session, repo, number))
        except GitHubError:
            failures += 1

    return results, failures
