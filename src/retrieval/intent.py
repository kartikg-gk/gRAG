"""Whether a query traces links or describes a concept.

Two stages, cheap first. A marker match costs a substring scan over a short
tuple; only a query matching nothing pays for a model call. Most queries in
practice name something or ask something, and the phrasings that signal which
are few enough to enumerate.

Failure is never fatal
----------------------

Every path that can raise falls back to conceptual. A misclassified query
returns results weighted the wrong way, which is worse results; an exception
returns nothing at all. The first is recoverable by reading the output, the
second is not, so classification never propagates a failure to the caller.

Conceptual is the fallback rather than relational because it is the weaker
claim. Relational weighting leans hard on traversal — 0.85 against 0.15 — and
a traversal from a seed chosen for a query that never named anything is a
confident walk from an arbitrary place. Conceptual weighting still consults
both arms.

The judge
---------

Injected, like every other model client in this project. With none supplied,
a query matching no marker takes the fallback and the run records that it did.
That is why the stage is recorded per query rather than inferred: "no marker
matched and no judge was configured" and "the judge said conceptual" produce
the same weighting and mean different things.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..common.config import (
    INTENT_CONCEPTUAL,
    INTENT_RELATIONAL,
    RELATIONAL_MARKERS,
    SEMANTIC_MARKERS,
    STAGE_FALLBACK,
    STAGE_MARKER,
    STAGE_MODEL,
    STAGE_NAMED,
)


class Judge(Protocol):
    """Answers one closed question about a query."""

    def relational(self, query: str) -> bool:
        ...


@dataclass(frozen=True)
class Intent:
    """What a query was classified as, and how that was decided."""

    intent: str
    stage: str
    marker: str | None = None

    @property
    def is_relational(self) -> bool:
        return self.intent == INTENT_RELATIONAL


def first_marker(query: str, markers) -> str | None:
    """The first marker present in ``query``, or ``None``.

    Substring matching against the lowercased query, so ``who`` also catches
    ``whose`` and ``which`` catches ``which of``. That is deliberate breadth:
    a marker list that had to enumerate inflections would be long enough to
    start matching by accident.
    """
    lowered = query.lower()
    for marker in markers:
        if marker in lowered:
            return marker
    return None


def classify(query: str, judge: Judge | None = None, *, named: bool = False) -> Intent:
    """Classify ``query`` as relational or conceptual.

    Relational markers are checked before semantic ones. A query carrying both
    — "who explains the architecture" — is treated as relational, because the
    thing it names is more specific than the thing it describes and the
    specific evidence is the one worth following.

    With no marker, a query that names a node of the graph (``named``) is
    relational; only a query that names nothing is put to the judge.

    An empty query is conceptual by fallback. There is nothing to match and
    nothing to ask a judge about.
    """
    if not query or not query.strip():
        return Intent(INTENT_CONCEPTUAL, STAGE_FALLBACK)

    marker = first_marker(query, RELATIONAL_MARKERS)
    if marker is not None:
        return Intent(INTENT_RELATIONAL, STAGE_MARKER, marker)

    marker = first_marker(query, SEMANTIC_MARKERS)
    if marker is not None:
        return Intent(INTENT_CONCEPTUAL, STAGE_MARKER, marker)

    # A query naming a person, a pull request or a service asks about that
    # thing's connections. Left to the model, two questions of the same shape
    # — "what did X work on?" — came back one relational and one semantic,
    # and the semantic one weighted the walk from X at a fifth.
    if named:
        return Intent(INTENT_RELATIONAL, STAGE_NAMED)

    if judge is None:
        return Intent(INTENT_CONCEPTUAL, STAGE_FALLBACK)

    try:
        relational = judge.relational(query)
    except Exception:
        # Deliberately broad. A judge is a network call to something outside
        # this process, and every way it can fail — timeout, malformed reply,
        # auth, a library raising something undocumented — has the same right
        # answer here: weight the query the safer way and carry on.
        return Intent(INTENT_CONCEPTUAL, STAGE_FALLBACK)

    return Intent(
        INTENT_RELATIONAL if relational else INTENT_CONCEPTUAL, STAGE_MODEL
    )
