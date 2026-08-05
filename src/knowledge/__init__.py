"""Module 3: knowledge graph construction.

Turns the normalized payloads from Module 2 into nodes and typed, weighted
relationships. Persistence, querying, traversal, indexing, scoring, and text
extraction belong to later modules.

The vocabulary itself lives in ``src.common.config`` and is re-exported here so
a consumer of the graph has one import. A rename still costs one file.
"""

from ..common.config import (
    CONFIDENCE,
    NODE_COMMIT,
    NODE_FILE,
    NODE_PERSON,
    NODE_PR,
    NODE_REPO,
    NODE_TICKET,
    RELATION_AUTHORED,
    RELATION_PART_OF,
    RELATION_REPORTED,
    RELATION_RESOLVES,
    RELATION_REVIEWED,
    RELATION_TOUCHES,
)
from .graph_builder import GraphBuilder, GraphStats

__all__ = [
    "GraphBuilder",
    "GraphStats",
    "CONFIDENCE",
    "NODE_PERSON",
    "NODE_REPO",
    "NODE_PR",
    "NODE_TICKET",
    "NODE_COMMIT",
    "NODE_FILE",
    "RELATION_AUTHORED",
    "RELATION_RESOLVES",
    "RELATION_REVIEWED",
    "RELATION_TOUCHES",
    "RELATION_PART_OF",
    "RELATION_REPORTED",
]
