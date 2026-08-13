"""The one place fetched items are put into a defined order.

**Sorting is required here, and the reason for it.** Three things make
arrival order reach the output, and none of them is optional. The sequence
a build sees decides what the graph says, so it cannot be left to whatever
the network returned first.

First, entity labels are first-seen-wins. Whichever surface form is read first
becomes the label permanently, so arrival order decides what a thing is called.

Second, we diff runs against committed output. An ordering that wobbles between
runs produces a diff with no change in it, which trains everyone to ignore
diffs.

Third, the order an API returns is not part of its contract. List endpoints
send ``sort`` and ``direction`` explicitly rather than trusting a server
default, and sorting locally on top of that makes the result independent
of that choice, and of any future change to GitHub's
default ordering.

Sorting happens **once**, between fetch and corpus construction, and nowhere
else. A sort repeated at each call site is a sort that will eventually disagree
with itself.

The key is total
----------------

A timestamp alone is not a total ordering: two pull requests updated in the
same second leave their relative order undefined, and undefined is exactly what
this module exists to remove. Every key therefore ends in a unique field — the
item number, or the commit SHA — so no two distinct items can compare equal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, TypeVar

from .models import Commit, Issue, PullRequest

#: Sorts before every real timestamp. Items with no time are a real case —
#: a commit payload can lack an author date — and they need a defined place
#: rather than an exception.
_NO_TIME = datetime.min.replace(tzinfo=timezone.utc)

T = TypeVar("T")


def _aware(moment: datetime | None) -> datetime:
    """A comparable timestamp. Naive values are read as UTC.

    Mixing naive and aware datetimes raises ``TypeError`` on comparison, which
    would turn a data quirk into a crash halfway through a sort.
    """
    if moment is None:
        return _NO_TIME
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def ingest_key(item: object) -> tuple[datetime, int, str]:
    """A total ordering key: when it changed, then who it is.

    The trailing fields are the tiebreak and are what make this total. A number
    is unique within a repository and a SHA is unique everywhere, so two
    distinct items can never produce equal keys — whatever their timestamps.
    """
    if isinstance(item, Commit):
        date = item.commit.author.date if item.commit.author else None
        return _aware(date), 0, item.sha

    if isinstance(item, (PullRequest, Issue)):
        # updated_at, falling back to created_at: an item that has never been
        # updated still has a creation time, and treating it as timeless would
        # bury it under every dated item.
        moment = item.updated_at or item.created_at
        return _aware(moment), item.number, ""

    raise TypeError(f"no ingest ordering defined for {type(item).__name__}")


def in_ingest_order(items: Iterable[T]) -> list[T]:
    """Newest first, ties broken by number or SHA. Materializes the input.

    Returns a list rather than an iterator because sorting has to see
    everything anyway, and a caller that received a lazy sort would be misled
    about when the requests happen.
    """
    return sorted(items, key=ingest_key, reverse=True)
