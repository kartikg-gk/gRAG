"""How alike two surface forms are.

A similarity source turns one surface form and a list of candidates into one
score per candidate. That signature is the whole interface, and it is shaped
that way on purpose: scoring against the candidate list in a single call is
what lets an implementation do one dot product against a matrix instead of a
pairwise loop, and it means resolution never sees a full N-by-N matrix it
would have to throw most of away.

Two implementations ship:

* ``LexicalSimilarity`` — standard library only, and the default. Character
  n-grams, L2-normalised, compared by dot product. It is genuinely cosine
  similarity; the vectors are hashed character features rather than learned
  ones.
* ``EmbeddingSimilarity`` — sentence-transformers, optional, same interface.

The lexical one is the default rather than a fallback because the drift it has
to catch is overwhelmingly orthographic: ``notification-service`` against
``notification_service``, ``ORDER_SERVICE`` against ``order_service``. Learned
embeddings earn their cost on semantic drift — "the billing box" against
``payment_service`` — which no amount of character overlap will find.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Protocol, Sequence

#: Length of the character n-gram. Three is the usual choice for short strings:
#: two matches far too much, four stops matching across a single typo.
NGRAM = 3

#: Everything outside this is flattened to a single space before n-gramming, so
#: ``auth-service``, ``auth_service`` and ``auth service`` produce the same
#: features. This is exactly the drift the lexical source exists to catch.
_SEPARATORS = re.compile(r"[^a-z0-9]+")


class Similarity(Protocol):
    """One surface form against many candidates, in one call."""

    def scores(self, surface: str, candidates: Sequence[str]) -> list[float]:
        """Similarity of ``surface`` to each candidate, in order, 0.0 to 1.0."""


def _normalise(surface: str) -> str:
    return _SEPARATORS.sub(" ", surface.lower()).strip()


def _vector(surface: str) -> dict[str, float]:
    """An L2-normalised character n-gram vector.

    Normalised at construction, so every later comparison is a plain dot
    product with no division — which is the point of normalising rather than
    dividing by magnitudes inside the loop.

    Padded at both ends so that a short form still produces features and so
    that the first and last characters carry the weight they should.
    """
    text = _normalise(surface)
    if not text:
        return {}

    padded = f"{' ' * (NGRAM - 1)}{text}{' ' * (NGRAM - 1)}"
    counts = Counter(
        padded[index : index + NGRAM] for index in range(len(padded) - NGRAM + 1)
    )

    magnitude = math.sqrt(sum(count * count for count in counts.values()))
    if magnitude == 0:
        return {}
    return {gram: count / magnitude for gram, count in counts.items()}


def _dot(left: dict[str, float], right: dict[str, float]) -> float:
    """Cosine similarity of two already-normalised vectors.

    Iterates the smaller side. Both are sparse, and the cost is the length of
    the shorter surface form rather than the size of the shared vocabulary.
    """
    if not left or not right:
        return 0.0
    if len(right) < len(left):
        left, right = right, left
    return sum(weight * right.get(gram, 0.0) for gram, weight in left.items())


class LexicalSimilarity:
    """Character n-gram cosine. Standard library only.

    Vectors are cached by surface form: resolution scores the same candidate
    repeatedly as the canonical set grows, and rebuilding its vector each time
    is the one avoidable cost in the loop.
    """

    name = "lexical"

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, float]] = {}

    def _cached(self, surface: str) -> dict[str, float]:
        vector = self._cache.get(surface)
        if vector is None:
            vector = _vector(surface)
            self._cache[surface] = vector
        return vector

    def scores(self, surface: str, candidates: Sequence[str]) -> list[float]:
        query = self._cached(surface)
        return [_dot(query, self._cached(candidate)) for candidate in candidates]


class EmbeddingSimilarity:
    """sentence-transformers, behind the same interface.

    Encoded with ``normalize_embeddings=True`` so the comparison stays a dot
    product against the candidate matrix — one matrix-vector product, not a
    pairwise loop and not a full distance matrix.

    Optional: importing this class is what pulls the dependency in, so nothing
    that only wants lexical similarity pays for it.
    """

    name = "embedding"

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def scores(self, surface: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []

        vectors = self._model.encode(
            [surface, *candidates], normalize_embeddings=True
        )
        query, matrix = vectors[0], vectors[1:]
        return [float(value) for value in matrix @ query]


def load_similarity(preference: str = "lexical") -> Similarity:
    """The requested similarity source, or the one that works.

    Same contract as the extractor's backend loader: an unavailable dependency
    is a normal condition, not an error, and the caller can see which source it
    got from ``.name``.
    """
    if preference == "embedding":
        try:
            return EmbeddingSimilarity()
        except (ImportError, OSError):
            return LexicalSimilarity()
    return LexicalSimilarity()
