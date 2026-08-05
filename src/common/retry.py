"""Retrying a call, for callers that decide a failure is worth another attempt.

Generic on purpose. This module knows nothing about GitHub — what counts as
retryable is passed in, because the answer belongs to whoever understands the
errors. That also keeps ``common`` free of any dependency on ``ingestion``.

There is no budget tracking and no circuit breaking.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 0.5


def with_retry(
    call: Callable[[], T],
    *,
    retryable: Callable[[BaseException], bool],
    attempts: int | None = None,
    backoff: float | None = None,
) -> T:
    """Run ``call``, retrying with exponential backoff while ``retryable``.

    ``call`` takes no arguments so the whole operation is retried, not part of
    one. Note this means a paginated fetch restarts from the first page: there
    is no way to resume a half-walked sequence from out here, which is the
    honest cost of keeping retry out of the client.

    The defaults are read at call time rather than bound into the signature, so
    that setting ``BACKOFF_SECONDS`` on this module actually takes effect —
    a default argument would have been frozen at import.
    """
    attempts = MAX_ATTEMPTS if attempts is None else attempts
    backoff = BACKOFF_SECONDS if backoff is None else backoff

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as error:
            if attempt >= attempts or not retryable(error):
                raise
            time.sleep(backoff * 2 ** (attempt - 1))

    raise AssertionError("unreachable: the loop either returns or raises")
