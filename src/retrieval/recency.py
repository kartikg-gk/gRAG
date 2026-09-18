"""How much a result is worth given how old it is.

Exponential decay, half-life keyed by node type::

    factor = 0.5 ** (age_days / half_life)

Three behaviours here are easy to get subtly wrong, and each is pinned by test
rather than left to reading:

**An unknown timestamp returns 1.0, not the floor.** Unknown means no penalty,
not maximum penalty. An undated node has said nothing about its age, and
ranking it below a node known to be ancient would be inventing evidence
against it. This is the same rule the store applies when it writes NULL rather
than 0 — zero is a real date in 1970 and absent is not a date.

**``now`` is injectable.** A decay computed against the wall clock is a
different number tomorrow, so a test written against it passes today and fails
in six months. Every caller that cares about determinism passes ``now``.

**Age is floored at zero.** A timestamp in the future gives a factor of 1.0
and no bonus. Clock skew between a source system and this one is ordinary, and
a future date is a data quirk rather than evidence of relevance.

The floor stops an old fact vanishing. A five-year-old commit that is the only
thing touching a file is still the answer, and a decay reaching zero would rank
it below anything recent and irrelevant.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..common.config import (
    DEFAULT_HALF_LIFE_DAYS,
    HALF_LIFE_DAYS,
    RECENCY_ENABLED,
    RECENCY_FLOOR,
    SECONDS_PER_DAY,
)


def _epoch(value) -> float | None:
    """Seconds since the epoch, from a datetime or a number.

    Accepts both because the store hands back ``datetime`` and callers holding
    raw column values hand back integers. Converting at the edge keeps the
    arithmetic below in one unit.
    """
    if value is None or value == 0:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def age_days(timestamp, now=None) -> float | None:
    """How old ``timestamp`` is in days, or ``None`` when it is unknown.

    ``None`` propagates rather than becoming zero. A caller has to be able to
    tell "brand new" from "no idea", and both collapsing to 0.0 would make an
    undated node look like the most recent thing in the graph.
    """
    moment = _epoch(timestamp)
    if moment is None:
        return None

    reference = _epoch(now)
    if reference is None:
        reference = datetime.now(timezone.utc).timestamp()

    return max(0.0, (reference - moment) / SECONDS_PER_DAY)


def half_life_for(node_type: str | None) -> float:
    """Days after which this type is worth half as much.

    The spread is the intent. A ticket goes stale in three weeks because it is
    about the present; a person stays relevant for six months because who works
    on what changes slowly; a repository effectively never decays because it is
    the container rather than an event.
    """
    return HALF_LIFE_DAYS.get(node_type or "", DEFAULT_HALF_LIFE_DAYS)


def age_and_decay(
    timestamp, node_type: str | None = None, now=None
) -> tuple[float | None, float]:
    """``(age_days, decay_factor)`` from one pass.

    Both come back together because both are wanted together and the second is
    derived from the first. Computing the factor and then asking for the age
    separately would call ``age_days`` twice per node, and — worse than the
    cost — the two calls could disagree: a caller that omits ``now`` gets the
    wall clock, so the age reported alongside a score would be fractionally
    later than the age that produced it. A reported number that differs from
    the one used to rank is a reporting bug that looks like a ranking bug.

    ``age`` is ``None`` when the timestamp is unknown, and the factor is then
    1.0 — unknown means no penalty. The pair therefore distinguishes "no date"
    from "dated today", which both produce a factor of 1.0 and mean different
    things.
    """
    if not RECENCY_ENABLED:
        return age_days(timestamp, now), 1.0

    age = age_days(timestamp, now)
    if age is None:
        return None, 1.0

    factor = 0.5 ** (age / half_life_for(node_type))
    return age, max(RECENCY_FLOOR, factor)


def decay_factor(timestamp, node_type: str | None = None, now=None) -> float:
    """A multiplier in ``[RECENCY_FLOOR, 1.0]`` for how recent this is.

    Returns exactly 1.0 for an unknown timestamp and for a future one, and
    exactly 1.0 for everything when recency is disabled — three separate
    reasons to apply no penalty, deliberately producing the same value.

    Delegates to ``age_and_decay`` so there is one implementation of the curve.
    A caller wanting the age as well should call that directly rather than
    calling this and then ``age_days``.
    """
    return age_and_decay(timestamp, node_type, now)[1]
