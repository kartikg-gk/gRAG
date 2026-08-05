"""The graph vocabulary: node labels, relation names, and what each is worth.

One module so a rename costs one file. Nothing else should spell a node label
or a relation name as a literal — call sites import from here.

Node labels follow the recency table rather than GitHub's API, because
half-lives are keyed by label and that table is the authority. The ingestion
models keep GitHub's own names (``Repository``, ``PullRequest``, ``Issue``);
they mirror the API and are a deliberately separate vocabulary.

Only names something actually emits live here. A declared constant with no
producer reads as a shipped feature, which is why there is no text-proximity
relation and no label for entity kinds the builder cannot produce. Both arrive
with the code that emits them.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Node labels
# --------------------------------------------------------------------------

NODE_PERSON = "Person"
NODE_REPO = "Repo"
NODE_PR = "PR"
NODE_TICKET = "Ticket"
NODE_COMMIT = "Commit"
NODE_FILE = "File"

# --------------------------------------------------------------------------
# Relations
# --------------------------------------------------------------------------

RELATION_AUTHORED = "AUTHORED"
RELATION_RESOLVES = "RESOLVES"
RELATION_REVIEWED = "REVIEWED"
RELATION_TOUCHES = "TOUCHES"
RELATION_PART_OF = "PART_OF"
RELATION_REPORTED = "REPORTED"

# --------------------------------------------------------------------------
# Confidence
#
# How much each relation is worth, so downstream scoring has something to
# multiply. Structure earns more than inference.
# --------------------------------------------------------------------------

CONFIDENCE = {
    RELATION_AUTHORED: 0.95,
    RELATION_RESOLVES: 0.92,
    RELATION_REVIEWED: 0.85,
    RELATION_TOUCHES: 0.80,
    RELATION_PART_OF: 0.80,
    RELATION_REPORTED: 0.75,
}
