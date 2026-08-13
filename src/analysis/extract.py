"""Entity extraction from free text.

Two stages, in this order and for this reason:

1. **Rules.** Ticket references, pull request references and service names have
   exact formats. A statistical model is worst at precisely these — it has no
   reason to believe ``#412`` is one token, and it will happily call
   ``payment_service`` an organisation — and on a code corpus they are the
   highest-value things in the text. So they are matched deterministically and
   score 1.0.

2. **Statistical NER**, for the open-ended remainder: people, organisations,
   products. Its generic labels reach domain labels through ``LABEL_MAP`` and
   nowhere else, so what the model calls something is translated in exactly one
   place.

A rule match wins over any statistical span it overlaps. That is the whole
point of running the precise stage first, and it is enforced here rather than
left to the ordering of a dictionary.

**This module builds no edges.** It returns entities. What relation an entity
earns depends on a scorer that does not exist yet, and picking a weight before
there is a baseline to measure against is tuning against nothing. See the note
in ``common/config.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol

from ..common.config import (
    ENTITY_ORG,
    ENTITY_PERSON,
    ENTITY_PR,
    ENTITY_PRODUCT,
    ENTITY_SERVICE,
    ENTITY_TICKET,
    MIN_ENTITY_LENGTH,
    RULE_SCORE,
    STATISTICAL_SCORE,
    WINDOW_OVERLAP_WORDS,
    WINDOW_WORDS,
)
from .chunking import windows

#: The backend used when the caller expresses no preference.
DEFAULT_BACKEND = "spacy"

#: What the ``source`` field says when no statistical backend was available.
#: Recorded rather than silently omitted: "rules only" and "the model found
#: nothing" are different results and must not look alike downstream.
SOURCE_RULES = "rules"
SOURCE_NONE = "none"

#: spaCy's labels, translated to this project's vocabulary. The single place
#: any label translation happens. A label absent from this table is dropped —
#: spaCy emits DATE, CARDINAL, MONEY and more that say nothing about a
#: repository, and admitting them would drown the useful entities.
LABEL_MAP = {
    "PERSON": ENTITY_PERSON,
    "ORG": ENTITY_ORG,
    "PRODUCT": ENTITY_PRODUCT,
}

# -- rule patterns ---------------------------------------------------------
#
# Ordered: the first pattern to claim a span keeps it. A pull request reference
# contains something that also looks like a ticket reference, so PR is matched
# first and the ticket pattern never sees those characters.

_PULL_REQUEST = re.compile(r"\b(?:PR|pull[ -]?requests?)\s*#?\d+\b", re.IGNORECASE)

# A bare "#412" is read as a ticket. On GitHub the number space is shared, so
# this is a genuine ambiguity that no rule can settle from the text alone; the
# choice is recorded here rather than hidden, and the pull request pattern above
# takes anything explicitly marked. Resolving a reference against nodes that
# actually exist is the graph builder's job, and it already does it.
#
# No ``\b`` before the hash: ``#`` is not a word character, so ``\b#`` requires
# a word character immediately before it and a bare " #412" would never match.
# The lookbehind does the intended job instead — it keeps "page#42" and
# "##42" out without demanding anything be there at all.
_TICKET = re.compile(r"(?:\bissues?\s*)?(?<![\w#])#\d+\b", re.IGNORECASE)

# service names: payment_service, auth-service, billing_svc.
_SERVICE = re.compile(r"\b[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*[_-](?:service|svc)\b", re.I)

#: Words that form an ordinary English compound with "service" rather than
#: naming one. Found by reading the output — "the self-service portal and the
#: customer-service dashboard" produced two confident Service entities, both
#: wrong. A rule stage claiming score 1.0 has to earn it, and the cost of a
#: false positive here is high: it outranks anything the statistical stage
#: would have said about the same span.
_NOT_A_SERVICE = frozenset({"self", "customer", "micro", "full", "in", "out"})

RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (_PULL_REQUEST, ENTITY_PR),
    (_TICKET, ENTITY_TICKET),
    (_SERVICE, ENTITY_SERVICE),
)

#: Stripped from both ends of a surface form. Bullets, quotes — including the
#: typographic ones a copied comment carries — brackets and sentence
#: punctuation. Nothing internal is touched: "payment_service" keeps its
#: underscore, which is most of what makes it identifiable.
_JUNK = " \t\r\n\"'`*-–—•·.,;:!?()[]{}<>|/\\“”‘’„«»"


@dataclass(frozen=True)
class Entity:
    """One extracted mention, locatable in the document it came from.

    ``start`` and ``end`` are absolute character offsets into the original
    text, never into a window, and ``end`` is exclusive — so
    ``document[entity.start:entity.end] == entity.text`` holds for every entity
    this module returns.
    """

    text: str
    type: str
    score: float
    start: int
    end: int
    source: str


class Backend(Protocol):
    """A statistical NER backend.

    Deliberately narrow: a name, and a function from text to labelled spans
    with offsets relative to the text it was given. Anything that satisfies
    this can replace spaCy without a caller changing.
    """

    name: str

    def entities(self, text: str) -> Iterable[tuple[str, str, int, int]]:
        """Yield ``(surface, label, start, end)`` for one piece of text."""


class NullBackend:
    """Finds nothing, always available.

    What the extractor degrades to when no model can be loaded. It exists so
    the rule stage keeps working on a machine with no spaCy installed, and so
    that "no backend" is a value rather than a ``None`` every caller checks.
    """

    name = SOURCE_NONE

    def entities(self, text: str) -> Iterable[tuple[str, str, int, int]]:
        return ()


class SpacyBackend:
    """spaCy's ``en_core_web_sm``.

    The model is loaded once and reused; loading it per call costs about a
    second each time.
    """

    name = "spacy"

    def __init__(self, model: str = "en_core_web_sm") -> None:
        import spacy

        # Only the entity recognizer is wanted. Disabling the rest is roughly a
        # threefold speedup on long documents.
        self._nlp = spacy.load(model, disable=["lemmatizer", "textcat"])

    def entities(self, text: str) -> Iterator[tuple[str, str, int, int]]:
        for ent in self._nlp(text).ents:
            yield ent.text, ent.label_, ent.start_char, ent.end_char


def load_backend(preference: str = DEFAULT_BACKEND) -> Backend:
    """Build the preferred backend, or the best available substitute.

    Unavailability is normal, not exceptional: spaCy is an optional dependency
    and its model is a separate download, so a machine that has neither must
    still extract ticket and service references. An unknown preference is
    treated the same way — the caller gets a working extractor and can see
    which backend it actually got from ``Entity.source``.
    """
    if preference == SOURCE_NONE:
        return NullBackend()

    if preference not in (DEFAULT_BACKEND,):
        preference = DEFAULT_BACKEND

    try:
        return SpacyBackend()
    except (ImportError, OSError):
        # ImportError: spaCy not installed. OSError: installed, model missing.
        return NullBackend()


def clean(surface: str, start: int, end: int) -> tuple[str, int, int] | None:
    """Trim junk from both ends, keeping the offsets true.

    Returns ``None`` for anything that does not survive, rather than an empty
    string: a caller that forgets to check a falsy string still inserts it, and
    a caller that forgets to check ``None`` crashes where the bug is.

    The offsets move with the trim. An entity whose reported span no longer
    contains its own text would be worse than no entity at all.
    """
    stripped = surface.lstrip(_JUNK)
    start += len(surface) - len(stripped)

    stripped = stripped.rstrip(_JUNK)
    end = start + len(stripped)

    if len(stripped) < MIN_ENTITY_LENGTH:
        return None
    return stripped, start, end


def _rule_entities(text: str) -> list[Entity]:
    """Every deterministic match, earlier patterns winning contested spans."""
    found: list[Entity] = []
    claimed: list[tuple[int, int]] = []

    for pattern, label in RULES:
        for match in pattern.finditer(text):
            start, end = match.span()
            if _overlaps(start, end, claimed):
                continue
            if label == ENTITY_SERVICE and _is_english_compound(match.group(0)):
                continue
            cleaned = clean(match.group(0), start, end)
            if cleaned is None:
                continue
            surface, start, end = cleaned
            claimed.append((start, end))
            found.append(
                Entity(
                    text=surface,
                    type=label,
                    score=RULE_SCORE,
                    start=start,
                    end=end,
                    source=SOURCE_RULES,
                )
            )
    return found


def _is_english_compound(surface: str) -> bool:
    """Whether "<word>-service" is ordinary English rather than a service name.

    Only the segment before the final separator is consulted, so a genuine
    ``customer_billing_service`` is unaffected — it is the bare
    ``customer-service`` that is prose.
    """
    head = re.split(r"[_-]", surface.lower())
    return len(head) == 2 and head[0] in _NOT_A_SERVICE


def _overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start < other_end and other_start < end for other_start, other_end in spans)


def dedupe(entities: Iterable[Entity]) -> list[Entity]:
    """One entity per surface form and type, keeping the best score.

    Overlapping windows mean the same mention is seen twice by construction,
    so this is not a defensive measure — it is the second half of how the
    overlap works. Ties keep the earlier occurrence, so the result is stable
    across runs rather than dependent on iteration order.
    """
    best: dict[tuple[str, str], Entity] = {}
    for entity in entities:
        key = (entity.text, entity.type)
        current = best.get(key)
        if current is None or entity.score > current.score:
            best[key] = entity
        elif entity.score == current.score and entity.start < current.start:
            best[key] = entity
    return sorted(best.values(), key=lambda entity: (entity.start, entity.end))


class Extractor:
    """Text in, entities out.

    ``backend`` names a preference, not a requirement — see ``load_backend``.
    Window geometry and the length floor come from ``common/config.py``; the
    constructor accepts overrides so a caller with unusual text can tune them
    without editing a shared constant.
    """

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        *,
        window: int = WINDOW_WORDS,
        overlap: int = WINDOW_OVERLAP_WORDS,
    ) -> None:
        # Validate the geometry now, so a bad configuration fails when it is
        # configured rather than on whichever document happens to be long
        # enough to need a second window.
        for _ in windows("probe probe", size=window, overlap=overlap):
            break

        self.backend = load_backend(backend)
        self.window = window
        self.overlap = overlap

    def extract(self, text: str) -> list[Entity]:
        """Every entity in ``text``, deduplicated, in document order."""
        if not text:
            return []

        rules = _rule_entities(text)
        claimed = [(entity.start, entity.end) for entity in rules]

        statistical = [
            entity
            for entity in self._statistical(text)
            if not _overlaps(entity.start, entity.end, claimed)
        ]

        return dedupe(rules + statistical)

    def _statistical(self, text: str) -> Iterator[Entity]:
        """The backend's findings, in absolute offsets.

        Rules run against the whole document, not per window: a regex has no
        input limit, and windowing them would let a boundary cut a match in
        half. Only the model needs the text in pieces.
        """
        for window in windows(text, size=self.window, overlap=self.overlap):
            for surface, label, start, end in self.backend.entities(window.text):
                mapped = LABEL_MAP.get(label)
                if mapped is None:
                    continue

                absolute_start, absolute_end = window.absolute(start, end)
                cleaned = clean(surface, absolute_start, absolute_end)
                if cleaned is None:
                    continue

                surface, absolute_start, absolute_end = cleaned
                yield Entity(
                    text=surface,
                    type=mapped,
                    score=STATISTICAL_SCORE,
                    start=absolute_start,
                    end=absolute_end,
                    source=self.backend.name,
                )
