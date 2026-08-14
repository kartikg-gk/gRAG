"""Entity resolution: three names for one thing become one entity.

Three bands, two thresholds:

    similarity >= FAST_THRESHOLD   merge, no model call
    similarity >= DEEP_THRESHOLD   ask a model
    otherwise                      a new entity

The split exists to control cost. Almost every pair is obviously the same or
obviously different, and only the narrow middle is worth paying a model for. A
run reports how many pairs reached the model precisely so that claim stays
falsifiable — see ``ResolutionStats.band_fraction``.

Failing safely
--------------

Every failure path creates a new entity rather than merging. A wrong merge is
unrecoverable: two entities become one, their relationships pool, and no later
pass can tell which evidence belonged to which. A wrong split leaves a
duplicate, which is visible in the output and fixable by re-running with
different thresholds. So a timeout, an exception, an unparseable answer and an
absent judge all mean "not the same".

What a merge returns
--------------------

The canonical entity, with its label updated. No alias list, no per-merge
score, no record of which band decided it.

An earlier version stored all three. Keeping the variants meant candidate
search had to score against them, which made grouping depend on document
order in two ways: similarity above a threshold is not transitive, so which
form arrived first decided which group absorbed the next one; and two
candidates scoring identically were separated by insertion order. Both
problems were consequences of holding the list. Removing it removes them,
rather than patching each in turn.

One mechanism, two floors
-------------------------

Throwing the variants away would make a query for ``notification_service``
miss an entity labelled ``notification-service``, so ``Resolver.find`` runs the
same similarity function at a much looser floor: 0.92 to merge, 0.80 to match a
query. Variant spellings survive as a scoring property rather than as stored
data.

The asymmetry is the point. A merge combines two records permanently and
nothing downstream can separate them again; a query match only chooses what to
look at, and being wrong costs a wasted read. The looser number is attached to
the cheaper mistake.

Write-once labels
-----------------

A merge updates the timestamp and nothing else. The label and type belong to
whichever surface form created the entity and are never rewritten, because a
later form is not better evidence — only later. The label therefore depends on
document order, and that is accepted: what a thing is called matters much less
than whether a query can reach it, which ``find`` settles by similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Protocol, Sequence

from ..common.config import (
    DEEP_THRESHOLD,
    FAST_THRESHOLD,
    PATH_FAST,
    PATH_MODEL,
    QUERY_THRESHOLD,
)
from .extract import Entity
from .similarity import Similarity


class Judge(Protocol):
    """Decides the ambiguous band.

    Implementations must ask exactly one closed question — are these two the
    same entity, yes or no — and constrain the answer to a single token of
    meaning. Not a ranking, not an explanation, not a choice among candidates:
    those turn one cheap call into an expensive one and produce an answer that
    has to be parsed rather than read.

    Raising is a supported outcome. The caller treats any exception as "not the
    same" and counts it as a failure.
    """

    def same(self, left: str, right: str) -> bool:
        """Whether two surface forms name the same thing."""


@dataclass
class ResolvedEntity:
    """One real thing, under the label it was first created with.

    **Label and type are write-once.** They are set when the entity is created
    and never modified afterwards. A later surface form is not better evidence
    than the first one, it is only later — so a noisier spelling arriving in a
    subsequent document must never clobber the label a canonical node already
    has.

    This makes the label depend on which document was read first, and that is
    accepted rather than worked around. Earlier versions tried frequency, then
    a not-all-uppercase rule, then string ordering, to pick a "best" form; all
    of them were machinery for a choice that does not need making. What the
    label is matters far less than that a query can reach the entity, and
    ``Resolver.find`` handles that with similarity rather than with spelling.

    ``timestamp`` moves forward only — see ``absorb``.
    """

    canonical: str
    type: str
    timestamp: datetime | None = None

    def absorb(self, timestamp: datetime | None = None) -> None:
        """Fold a mention in. Touches the timestamp and nothing else.

        Recency is the newest fact about an entity: mentioned again today, it
        is more recent, whichever document did the mentioning. So this is a
        maximum rather than an assignment — a stale document processed last
        must not drag the timestamp backward.

        ``None`` means unknown and loses to any real time, in both directions:
        an unknown incoming time never erases a known one, and a known incoming
        time fills in an unknown one. Unknown stays ``None`` rather than
        becoming ``0``, which a recency scorer would read as 1970.
        """
        if timestamp is None:
            return
        if self.timestamp is None or timestamp > self.timestamp:
            self.timestamp = timestamp

    def to_dict(self) -> dict:
        """Plain data, for writing to a file."""
        return {
            "canonical": self.canonical,
            "type": self.type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class ResolutionStats:
    """What one resolution run did.

    A run that merged nothing and a run that merged everything have to be
    distinguishable from these numbers alone, without re-reading the output.
    """

    seen: int = 0
    #: A distinct surface form folded into an entity on the fast path. This is
    #: resolution doing work: a name the entity did not already answer to.
    fast_merges: int = 0
    #: The same, decided by the model.
    model_merges: int = 0
    #: The incoming form was already the label — the same string seen in
    #: another document. Counted apart from the two above because it is not
    #: resolution doing anything: no drift was reconciled. Folding it into the
    #: merge count overstates what resolution achieved.
    repeats: int = 0
    model_calls: int = 0
    model_rejections: int = 0
    created: int = 0
    failures: int = 0
    #: Candidate scores computed. The denominator for the band fraction, and
    #: the honest one: an entity with no candidate was never a pair.
    comparisons: int = 0

    @property
    def variant_merges(self) -> int:
        """Distinct forms resolved onto an existing entity. The real number."""
        return self.fast_merges + self.model_merges

    @property
    def merges(self) -> int:
        """Every absorb event, repeats included. Reconciles with ``seen``."""
        return self.variant_merges + self.repeats

    @property
    def band_fraction(self) -> float:
        """Fraction of scored pairs that needed a model.

        The number that says whether the thresholds are set right. High means
        the bands are wrong and the run is paying for calls it should not need.
        """
        if not self.comparisons:
            return 0.0
        return self.model_calls / self.comparisons

    def to_dict(self) -> dict:
        return {
            "seen": self.seen,
            "fast_merges": self.fast_merges,
            "model_merges": self.model_merges,
            "variant_merges": self.variant_merges,
            "repeats": self.repeats,
            "merges": self.merges,
            "model_calls": self.model_calls,
            "model_rejections": self.model_rejections,
            "created": self.created,
            "failures": self.failures,
            "comparisons": self.comparisons,
            "band_fraction": round(self.band_fraction, 4),
        }


class Resolver:
    """Folds entities into canonical ones, keeping the evidence.

    ``similarity`` and ``judge`` are both injected. That is what makes every
    rule here — which band a score falls in, which way each comparison points,
    what a failure does, what ends up on the record — testable with fixed
    numbers and no model of any kind installed.

    With no judge, the ambiguous band resolves to "not the same". That is the
    same answer a failed call gives, so the safe path is the default rather
    than an error branch that only runs when something breaks.
    """

    def __init__(
        self,
        similarity: Similarity | None = None,
        judge: Judge | None = None,
        *,
        fast: float = FAST_THRESHOLD,
        deep: float = DEEP_THRESHOLD,
        query_floor: float = QUERY_THRESHOLD,
    ) -> None:
        if deep > fast:
            raise ValueError(
                f"deep threshold ({deep}) cannot exceed the fast one ({fast}); "
                "there would be no band between them"
            )

        self.similarity = similarity if similarity is not None else Similarity()
        self.judge = judge
        self.fast = fast
        self.deep = deep
        # Independent of the merge thresholds on purpose: they answer
        # different questions and are validated against different data.
        self.query_floor = query_floor
        self.stats = ResolutionStats()
        # Keyed by type, because candidates are only ever drawn from one type.
        self._by_type: dict[str, list[ResolvedEntity]] = {}

    # -- reading the result ------------------------------------------------

    @property
    def entities(self) -> list[ResolvedEntity]:
        """Every canonical entity, grouped by type in first-seen order."""
        return [entity for group in self._by_type.values() for entity in group]

    def to_dict(self) -> dict:
        return {
            "stats": self.stats.to_dict(),
            "entities": [entity.to_dict() for entity in self.entities],
        }

    def find(self, term: str, kind: str | None = None) -> ResolvedEntity | None:
        """The entity a query term refers to, or ``None``.

        **The same similarity function as merging, at a much looser floor.**
        One mechanism, two jobs. A query for ``notification_service`` scores 1.0
        against a label of ``notification-service`` and reaches it, with no
        alias table anywhere — which is how variant spellings survive being
        thrown away at merge time.

        The floor is loose because the two decisions carry different costs. A
        merge combines two records permanently and nothing downstream can undo
        it; a query match only chooses what to look at, and being wrong costs a
        wasted read. So merging demands 0.92 and this settles for 0.80.

        Ties are broken on the label rather than on insertion order, so two
        equally similar entities resolve the same way on every run.
        """
        candidates = (
            self._by_type.get(kind, [])
            if kind is not None
            else [entity for group in self._by_type.values() for entity in group]
        )
        if not candidates:
            return None

        scores = self.similarity.scores(
            term, [candidate.canonical for candidate in candidates]
        )
        score, entity = min(
            zip(scores, candidates),
            key=lambda pair: (-pair[0], pair[1].canonical),
        )
        return entity if score >= self.query_floor else None

    # -- doing the work ----------------------------------------------------

    def resolve(
        self,
        entities: Iterable[Entity],
        timestamp: datetime | None = None,
    ) -> list[ResolvedEntity]:
        """Fold a stream of extracted entities. Returns the canonical set."""
        for entity in entities:
            self.add(entity, timestamp)
        return self.entities

    def add(
        self, entity: Entity, timestamp: datetime | None = None
    ) -> ResolvedEntity:
        """Fold one entity in, and return whichever entity now holds it.

        ``timestamp`` is when the mention was made — a property of the document
        the entity came from, not of the span, which is why it is an argument
        here rather than a field on ``Entity``. Omitted, the entity carries no
        time, and unknown stays unknown.
        """
        self.stats.seen += 1

        candidates = self._by_type.setdefault(entity.type, [])
        best, score = self._best_candidate(entity.text, candidates)

        if best is None:
            return self._create(entity, timestamp)

        if score >= self.fast:
            return self._absorb(best, entity, score, PATH_FAST, timestamp)

        if score >= self.deep:
            return self._ask(entity, best, score, timestamp)

        return self._create(entity, timestamp)

    def _best_candidate(
        self, surface: str, candidates: Sequence[ResolvedEntity]
    ) -> tuple[ResolvedEntity | None, float]:
        """The most similar candidate of the same type, and its score.

        Canonical forms only — they are the only forms kept. An earlier version
        scored every variant an entity answered to, which is what made grouping
        sensitive to arrival order; with no variants stored the question cannot
        arise.

        One call into the similarity source, so an implementation backed by a
        matrix does one product against the candidate matrix rather than a
        pairwise loop.
        """
        if not candidates:
            return None, 0.0

        scores = self.similarity.scores(
            surface, [candidate.canonical for candidate in candidates]
        )
        self.stats.comparisons += len(scores)

        best_index = max(range(len(scores)), key=scores.__getitem__)
        return candidates[best_index], scores[best_index]

    def _ask(
        self,
        entity: Entity,
        candidate: ResolvedEntity,
        score: float,
        timestamp: datetime | None = None,
    ) -> ResolvedEntity:
        """The ambiguous band. One closed question, and a safe answer on doubt."""
        self.stats.model_calls += 1

        if self.judge is None:
            # No judge configured. Indistinguishable from a failed call by
            # design: both mean "could not establish sameness".
            self.stats.failures += 1
            return self._create(entity, timestamp)

        try:
            same = self.judge.same(candidate.canonical, entity.text)
        except Exception:
            # Deliberately broad. A judge reaches a network, and every way that
            # can fail has to mean the same thing here: do not merge.
            self.stats.failures += 1
            return self._create(entity, timestamp)

        if not isinstance(same, bool):
            # An unparseable answer is a failure, not a falsy value to trust.
            self.stats.failures += 1
            return self._create(entity, timestamp)

        if same:
            return self._absorb(candidate, entity, score, PATH_MODEL, timestamp)

        self.stats.model_rejections += 1
        return self._create(entity, timestamp)

    def _absorb(
        self,
        into: ResolvedEntity,
        entity: Entity,
        score: float,
        path: str,
        timestamp: datetime | None = None,
    ) -> ResolvedEntity:
        """Fold a form into an entity. Advances the timestamp, nothing else.

        **The label and type are not touched.** This is the whole write path
        for a merge, so that guarantee is enforced by there being no assignment
        here rather than by a rule someone has to remember.

        The one place a merge is counted, so every path that absorbs a form
        agrees on what kind of event it was.

        A repeat is the incoming form already being the label. With no alias
        list this is all that can be known: a form that merged earlier is
        indistinguishable from a new variant and is counted as drift again.
        That undercount is the price of not keeping the list.

        Novelty is judged on the form rather than the path, because the two are
        independent — a repeat normally scores 1.0 and takes the fast path, but
        a similarity source that scored an identical string lower would send it
        to the model, and it would still be a repeat.
        """
        if entity.text == into.canonical:
            self.stats.repeats += 1
        elif path == PATH_FAST:
            self.stats.fast_merges += 1
        else:
            self.stats.model_merges += 1

        into.absorb(timestamp)
        return into

    def _create(
        self, entity: Entity, timestamp: datetime | None = None
    ) -> ResolvedEntity:
        """The only place a label and type are ever set."""
        created = ResolvedEntity(
            canonical=entity.text, type=entity.type, timestamp=timestamp
        )
        self._by_type.setdefault(entity.type, []).append(created)
        self.stats.created += 1
        return created
