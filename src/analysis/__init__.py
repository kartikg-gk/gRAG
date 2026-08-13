"""Analysis: read text, return entities.

Extraction only. This package builds no edges and touches no graph — what
relation an extracted entity earns is a question for a scorer that does not
exist yet, and answering it early would mean choosing a weight with nothing to
measure it against.

    from src.analysis import Extractor

    extractor = Extractor()
    for entity in extractor.extract(pull_request.body):
        print(entity.type, entity.text, entity.start, entity.end)
"""

from .chunking import Window, windows
from .resolve import (
    Judge,
    ResolutionStats,
    ResolvedEntity,
    Resolver,
)
from .similarity import (
    EmbeddingSimilarity,
    LexicalSimilarity,
    Similarity,
    load_similarity,
)
from .extract import (
    DEFAULT_BACKEND,
    LABEL_MAP,
    RULES,
    SOURCE_NONE,
    SOURCE_RULES,
    Backend,
    Entity,
    Extractor,
    NullBackend,
    SpacyBackend,
    clean,
    dedupe,
    load_backend,
)

__all__ = [
    "Backend",
    "DEFAULT_BACKEND",
    "EmbeddingSimilarity",
    "Entity",
    "Extractor",
    "Judge",
    "LABEL_MAP",
    "LexicalSimilarity",
    "NullBackend",
    "RULES",
    "ResolutionStats",
    "ResolvedEntity",
    "Resolver",
    "SOURCE_NONE",
    "SOURCE_RULES",
    "Similarity",
    "SpacyBackend",
    "Window",
    "clean",
    "dedupe",
    "load_backend",
    "load_similarity",
    "windows",
]
