"""The graph vocabulary: node labels, relation names, and what each is worth.

One module so a rename costs one file. Nothing else should spell a node label
or a relation name as a literal — call sites import from here.

Node labels follow the recency table rather than GitHub's API, because
half-lives are keyed by label and that table is the authority. The ingestion
models keep GitHub's own names (``Repository``, ``PullRequest``, ``Issue``);
they mirror the API and are a deliberately separate vocabulary.

Names here generally arrive with the code that emits them, because a declared
constant with no producer reads as a shipped feature. ``RELATION_CO_OCCURS`` is
the one exception, and it is deliberate — see the note beside it.
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

#: Text proximity: two entities mentioned near each other.
#:
#: **Declared but never emitted.** Nothing in this project produces a
#: CO_OCCURS edge yet. The constant and its weight are declared anyway so
#: the relation vocabulary is complete: a scorer added later needs a name
#: and a price already agreed, and choosing the price at the moment of the
#: first producer means choosing it under pressure to flatter that producer.
#:
#: If something starts emitting these, the weight below is what makes them lose
#: to every structural relation: a proximity guess must never outrank an
#: authorship fact.
RELATION_CO_OCCURS = "CO_OCCURS"

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
    # Weakest by a wide margin, and below every structural relation above.
    RELATION_CO_OCCURS: 0.35,
}

# --------------------------------------------------------------------------
# Entity extraction
#
# The extractor reads text and returns entities. It does not build edges, and
# there is deliberately no relation weight for a mention here: what an
# extracted entity is worth depends on a scorer that does not exist yet, and a
# weighted constant with no producer reads as a shipped feature. The edge these
# entities earn gets chosen when there is a baseline to measure it against.
# --------------------------------------------------------------------------

# Labels the extractor can produce. Three of them are the node labels above,
# reused rather than re-spelled: a "#412" found in prose is the same kind of
# thing as a Ticket node, and two spellings of one concept is how drift starts.
ENTITY_TICKET = NODE_TICKET
ENTITY_PR = NODE_PR
ENTITY_PERSON = NODE_PERSON

# Labels with no node type yet, because nothing in ingestion produces one.
ENTITY_SERVICE = "Service"
ENTITY_ORG = "Org"
ENTITY_PRODUCT = "Product"

#: Sliding window over words, and how much consecutive windows share. Overlap
#: exists so an entity straddling a window edge is still seen whole by one of
#: them; it must stay smaller than the window or the windows stop advancing.
WINDOW_WORDS = 200
WINDOW_OVERLAP_WORDS = 40

#: Shortest surface form worth keeping, after cleaning. Anything shorter is
#: punctuation or a fragment, not an entity.
MIN_ENTITY_LENGTH = 3

#: What a rule-stage match scores. Deterministic patterns are not guesses.
RULE_SCORE = 1.0

#: What a statistical match scores. A single constant on purpose: spaCy's NER
#: exposes no per-entity probability, so any number that varied per entity here
#: would be invented. One honest constant beats a fabricated distribution.
STATISTICAL_SCORE = 0.5

# --------------------------------------------------------------------------
# Entity resolution
#
# Two thresholds carve similarity into three bands, and the middle one is the
# only place a model is worth paying for. Most pairs are obviously the same or
# obviously different; if most pairs land in the middle, these numbers are
# wrong, which is why the run reports the fraction that did.
#
# The gap between them is deliberately narrow. Widening it does not buy
# accuracy, it buys model calls.
# --------------------------------------------------------------------------

#: At or above this, merge outright. No model call.
FAST_THRESHOLD = 0.92

#: At or above this but below the fast one, ask. Below it, a new entity.
DEEP_THRESHOLD = 0.85

#: Query-time floor. One similarity mechanism, two jobs: this is the loose end.
#:
#: A merge permanently combines two records and cannot be undone, so it demands
#: near-certainty. A query match only decides what to look at, so it can afford
#: to be generous — that is what lets a query for ``notification_service`` reach
#: an entity labelled ``notification-service`` without any alias table.
#:
#: **Set from measurement rather than intuition.** A floor at roughly half
#: the merge threshold — 0.40 — is the intuitive choice and it is wrong
#: here. That intuition assumes a scorer where unrelated strings land near
#: zero; character n-grams do not. Measured over the demo corpus's
#: canonical set — entities resolution kept apart, so a query must never
#: confuse them — the highest scoring known-distinct pair is
#: ``'pull request #1347'`` against ``'pull request #204'`` at 0.7030, because
#: they share most of their characters. Meanwhile every true orthographic
#: variant scores exactly 1.0, since normalisation makes them identical.
#:
#: That leaves a safe window of (0.7030, 1.0] and no reason to sit near either
#: edge. 0.40 would have admitted eight known-distinct pairs — a query for
#: ``payment_service`` would have reached ``auth-service`` at 0.5186.
#:
#: Re-measure this if the similarity source changes. It is a property of the
#: scoring function, not a universal constant.
QUERY_THRESHOLD = 0.80

# Which band decided a merge. Stored on the record, because "these two were
# obviously identical" and "a model was asked and said yes" are different
# claims and a later audit needs to tell them apart.
PATH_FAST = "fast"
PATH_MODEL = "model"
PATH_NEW = "none"


# --------------------------------------------------------------------------
# Graph store
#
# Table names, the embedding dimension, the index name and the distance metric
# live here rather than in the store, so a rename or a re-index is one edit and
# the store has nothing to hardcode.
# --------------------------------------------------------------------------

#: Node table holding canonical entities.
NODE_TABLE = "Entity"

#: Node table holding raw source documents. Documents are source material, not
#: canonical entities, so they are a separate table with their own columns.
DOC_TABLE = "Document"

#: One relationship table, not one per relation type. The relation name is a
#: STRING column, so AUTHORED and the rest are values rather than tables, and
#: their weights are the CONFIDENCE values above.
REL_TABLE = "Related"

#: Document to entity. Its own table because its endpoints differ from every
#: other relation — Document to Entity rather than Entity to Entity — which is
#: also why it carries no properties and is absent from the relation constants.
MENTIONS_TABLE = "Mentions"

#: Width of the embedding column. This must equal the embedding model's output
#: size: the column is fixed-width at CREATE TABLE time, so it cannot be
#: derived from the model the way ``Similarity.dimension`` is. Changing the
#: model to one of a different width means a new database, not a migration.
EMBEDDING_DIMENSION = 384

#: Vector index over the entity embedding column.
VECTOR_INDEX_NAME = "entity_embedding_index"

#: Distance metric for that index. Cosine, because embeddings are L2-normalised
#: before they are stored, which makes cosine distance and the dot product the
#: same ordering — and lets similarity be recovered as ``1 - distance``.
VECTOR_METRIC = "cosine"

#: Read connections held open for concurrent queries. The engine serialises
#: writes but reads scale across connections, so this bounds how many can run
#: at once rather than how many exist.
READ_POOL_SIZE = 4

#: Seconds to wait for a free read connection before giving up. A caller that
#: waits forever on an exhausted pool looks like a hang, not a queue.
POOL_TIMEOUT_SECONDS = 30.0
