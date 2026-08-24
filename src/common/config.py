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
#: **Declared but never emitted.** Nothing produces a CO_OCCURS edge yet. The
#: constant and its weight are declared anyway so the relation vocabulary is
#: complete: a scorer added later needs a name and a price already agreed, and
#: choosing the price at the moment of the first producer means choosing it
#: under pressure to make that producer's output look reasonable.
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
#: **Set from measurement.** A floor at roughly half the merge threshold —
#: 0.40 — is the intuitive choice and it is wrong here. That intuition assumes
#: a scorer where unrelated strings land near zero; character n-grams do not
#: behave that way, because two unrelated strings still share characters.
#: Measured over the demo corpus's
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


# --------------------------------------------------------------------------
# Retrieval
#
# Every value here is overridable from the environment. These bound how much
# work a query does and how permissive it is, and both are properties of the
# corpus rather than of the code — a setting that is right for a sixteen-node
# graph is not right for a repository with ten thousand pull requests, and
# nobody should have to edit a source file to find that out.
# --------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    """An integer from the environment, or the default.

    A malformed value falls back rather than raising. A typo in a shell
    variable should not stop the process from starting, and the default is
    always a working setting.
    """
    import os

    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """A float from the environment, or the default. Malformed falls back."""
    import os

    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


#: How many hops traversal may take from a seed.
#:
#: Measured on the demo graph, mean share of the graph reached from an average
#: seed: 1 hop 17%, 2 hops 53%, 3 hops 89%, 4 hops 100%. The bound sits where
#: reach stops being selective — at 3 the traversal returns almost everything
#: and the score is doing all the discriminating, which leaves the bound with
#: no work to do.
#:
#: **Those figures come from a 16-node graph and describe a toy.** The point at
#: which reach stops being selective moves with graph size and with how hub-like
#: the graph is; on a real repository 2 hops from a pull request reaches its
#: author and then every pull request that author touched, which may be
#: hundreds. Anyone raising this to 3 should meet the 89% first and re-measure
#: on the corpus they actually have.
#:
#: Confidence decay already bounds depth on its own — a path multiplies, so
#: 0.95 twice is 0.90 while 0.35 twice is 0.12. This bound is a rail against
#: runaway traversal rather than the thing shaping results.
MAX_HOPS = _env_int("GRAPHRAG_MAX_HOPS", 2)

#: Lowest vector similarity that may seed a traversal.
#:
#: **No measurement stands behind this number yet.** It is a starting point
#: chosen to make traversal runnable, and the seed tier firing rates are what
#: would justify moving it. Treat it as unmeasured until those exist.
#:
#: It sits far below the merge thresholds on purpose. A merge permanently
#: combines two records and cannot be undone, so it demands near-certainty; a
#: seed only decides where to start looking, and a wrong seed costs a traversal
#: that finds nothing. The query side can afford to be generous where the write
#: side cannot.
SEED_MIN_SIM = _env_float("GRAPHRAG_SEED_MIN_SIM", 0.35)

#: How many fuzzy seeds tier 2 may contribute.
#:
#: A cap, not a floor. Without it a permissive ``SEED_MIN_SIM`` seeds traversal
#: from the whole vector result set, and the graph arm stops being a traversal
#: from somewhere specific — it becomes a walk from everywhere, which returns
#: the graph and ranks it by nothing the traversal contributed.
SEED_TOP_N = _env_int("GRAPHRAG_SEED_TOP_N", 3)

#: Per-relation degree above which a node's neighbours are suppressed.
#:
#: Grouped per relation rather than per node. A repository node with 10,000
#: TOUCHES edges and 3 AUTHORED edges should lose the file list and keep the
#: authorship; capping per node would drop both, discarding the useful half
#: because the other half is broad.
MAX_DEGREE = _env_int("GRAPHRAG_MAX_DEGREE", 10)

#: How many results each arm returns. Separate constants because the two arms
#: are measured against each other and a shared value would make a difference
#: in their result counts impossible to attribute.
TOP_K_VECTOR = _env_int("GRAPHRAG_TOP_K_VECTOR", 10)
TOP_K_GRAPH = _env_int("GRAPHRAG_TOP_K_GRAPH", 10)


# --------------------------------------------------------------------------
# Intent
#
# A query either traces links between things or describes a concept, and the
# two want different evidence. Classification is two-stage and cheap first:
# a marker match costs a substring scan, and only a query matching nothing
# pays for a model call.
#
# The marker lists are deliberately small. A long list drifts into encoding
# one corpus's phrasing, and every word added removes a query from the stage
# that could have judged it properly. If most queries reach stage two the
# lists are too narrow; if none do, they are too broad and are classifying by
# accident.
# --------------------------------------------------------------------------

#: Phrasings that trace a sequence of events, people or links. Matched as
#: substrings against the lowercased query, so "who" also catches "whose".
RELATIONAL_MARKERS = (
    "who",
    "whom",
    "whose",
    "which",
    "what caused",
    "what closed",
    "what fixed",
    "what broke",
    "related to",
    "connected to",
    "depends on",
    "blocked by",
    "reviewed",
    "authored",
    "owner of",
    "linked to",
)

#: Phrasings that ask about meaning rather than connection.
SEMANTIC_MARKERS = (
    "explain",
    "architecture",
    "overview",
    "summary",
    "summarise",
    "summarize",
    "describe",
    "how does",
    "why does",
    "what is",
    "purpose of",
    "rationale",
)

INTENT_RELATIONAL = "relational"
INTENT_CONCEPTUAL = "conceptual"

#: How a query was classified. Recorded so a run can report the stage-two rate
#: rather than leaving it to be inferred from timing.
STAGE_MARKER = "marker"
STAGE_MODEL = "model"
STAGE_FALLBACK = "fallback"

# --------------------------------------------------------------------------
# Fusion weights
#
# **Starting values, not measured.** These are chosen so the two arms can be
# combined at all; what blend performs best has not been measured. The
# per-query comparison of fused ranking against each arm's raw ranking is
# what would justify or move them.
#
# The shape is the part with a reason behind it: a query naming a thing wants
# the arm that follows links from it, and a query describing a concept wants
# the arm that matches meaning. Measured on two cases built to separate the
# arms, a named entity scored 0.92 through traversal and 0.0107 by similarity,
# and a described concept scored 0.7231 by similarity with traversal returning
# nothing at all. The weights lean the way those numbers do.
# --------------------------------------------------------------------------

VECTOR_WEIGHT_RELATIONAL = _env_float("GRAPHRAG_VECTOR_WEIGHT_RELATIONAL", 0.15)
GRAPH_WEIGHT_RELATIONAL = _env_float("GRAPHRAG_GRAPH_WEIGHT_RELATIONAL", 0.85)
VECTOR_WEIGHT_CONCEPTUAL = _env_float("GRAPHRAG_VECTOR_WEIGHT_CONCEPTUAL", 0.80)
GRAPH_WEIGHT_CONCEPTUAL = _env_float("GRAPHRAG_GRAPH_WEIGHT_CONCEPTUAL", 0.20)

# --------------------------------------------------------------------------
# Recency
#
# Half-lives are per node type, and the spread is the intent: a ticket is
# stale in three weeks because an open ticket is about the present and a
# closed one stops being asked about; a person stays relevant for six months
# because who works on what changes slowly; a repository effectively never
# decays because it is the container rather than an event.
#
# The floor stops an old fact vanishing. A five-year-old commit that is the
# only thing touching a file is still the answer, and a decay that reached
# zero would rank it below anything recent and irrelevant.
# --------------------------------------------------------------------------

SECONDS_PER_DAY = 86400.0

RECENCY_ENABLED = _env_int("GRAPHRAG_RECENCY_ENABLED", 1) == 1

#: Lowest a decay factor may fall, however old the thing is.
RECENCY_FLOOR = _env_float("GRAPHRAG_RECENCY_FLOOR", 0.35)

#: Days after which a node of this type is worth half as much.
HALF_LIFE_DAYS = {
    NODE_TICKET: 21.0,
    NODE_COMMIT: 45.0,
    NODE_PR: 60.0,
    NODE_FILE: 120.0,
    NODE_PERSON: 180.0,
    ENTITY_SERVICE: 365.0,
    NODE_REPO: 365.0,
}

#: For a type with no entry. Between a pull request and a file, so an unknown
#: type is neither treated as breaking news nor as permanent.
DEFAULT_HALF_LIFE_DAYS = _env_float("GRAPHRAG_DEFAULT_HALF_LIFE_DAYS", 90.0)

#: Fewest vector results a fused set may carry, however many the graph arm
#: returned. A query whose traversal reaches everything must still carry some
#: evidence of what the text actually says.
MIN_VECTOR_K = _env_int("GRAPHRAG_MIN_VECTOR_K", 2)

#: How many results a fused set returns.
TOTAL_K = _env_int("GRAPHRAG_TOTAL_K", 10)


# --------------------------------------------------------------------------
# Model judges
#
# Two closed questions get asked of a model, and they are asked under
# different constraints, so they get two endpoints rather than one shared
# client.
#
# The ingest question — do these two surface forms name the same entity — runs
# off the clock, once per grey-band pair, and its answer is permanent: a wrong
# merge pools two entities' evidence and no later pass can separate them. No
# timeout, no token cap.
#
# The query question — does this query trace links or describe a concept —
# runs inside the user's latency budget, once per query, in front of a
# fallback that costs nothing. Hard timeout, tight token cap.
#
# **The asymmetry is those two bounds and nothing else.** Nothing here
# requires the two models to differ, and nothing should: one endpoint with one
# model configured twice is a legitimate setup. What keeps the query path
# inside its budget is the timeout, not which model is named.
#
# Neither base URL and neither model name has a default. A key with no default
# already fails hard; a model with a default would silently ship one
# particular choice to anyone who set only a key, which is the thing these
# names being role-based exists to prevent. The cost is that there is no
# zero-config path — an unset value fails when a judge is constructed, not at
# its first call — and that is accepted.
# --------------------------------------------------------------------------


def _env_str(name: str, default: str = "") -> str:
    """A string from the environment, or the default.

    Blank counts as unset, so an exported-but-empty variable behaves the same
    as one that was never exported. The alternative is a base URL of "" that
    fails somewhere far from the shell that caused it.
    """
    import os

    return os.environ.get(name, "").strip() or default


#: Where the ingest question is asked. **No default** — see the note above.
JUDGE_BASE_URL = _env_str("GRAPHRAG_JUDGE_BASE_URL")

#: Which model answers it. **No default**, same reason.
JUDGE_MODEL = _env_str("GRAPHRAG_JUDGE_MODEL")

#: Where the query question is asked, when its own key is set. **No default.**
JUDGE_FAST_BASE_URL = _env_str("GRAPHRAG_JUDGE_FAST_BASE_URL")

#: Which model answers it. **No default.**
JUDGE_FAST_MODEL = _env_str("GRAPHRAG_JUDGE_FAST_MODEL")

#: Which variables hold the two keys. **The names live here; the values never
#: do.** A key read at call time from a variable named here cannot end up in a
#: source file, a traceback, or a diff.
JUDGE_KEY_VAR = "GRAPHRAG_JUDGE_KEY"
JUDGE_FAST_KEY_VAR = "GRAPHRAG_JUDGE_FAST_KEY"

#: Seconds the query question may take before it is abandoned.
#:
#: **Chosen, not measured.** It is a latency budget rather than an observation:
#: classification sits in front of retrieval, and the fallback it takes on a
#: timeout is the answer an unmatched query gets anyway. Measuring an
#: endpoint's real latency would say what the call costs, not what it is worth.
#:
#: This is the bound that makes the query path safe, and it applies whichever
#: endpoint answers — including when the fast key is unset and the ingest
#: endpoint takes the question instead.
JUDGE_TIMEOUT_SECONDS = _env_float("GRAPHRAG_JUDGE_TIMEOUT_SECONDS", 2.0)

#: Tokens the query question may spend. **Chosen**: the answer is one word, and
#: a cap this tight means a model that starts explaining itself is cut off
#: rather than paid for. The reply is read by prefix, so a truncated word still
#: parses.
JUDGE_MAX_TOKENS = _env_int("GRAPHRAG_JUDGE_MAX_TOKENS", 4)

#: Which endpoint answered the query question. Recorded rather than inferred,
#: for the same reason the classification stage is: a run where the fast
#: endpoint was never configured and one where it answered every query produce
#: the same weights and mean different things, and timing them is not a way to
#: tell.
ENDPOINT_FAST = "fast"
ENDPOINT_GENERAL = "general"


# --------------------------------------------------------------------------
# HTTP API: identity and tenancy
#
# Two separate checks, deliberately not one. A session token says *who is
# asking*; an API key says *whose data is being asked about*. Conflating them
# is how a valid user reads another organisation's graph, so they are resolved
# by different dependencies, from different credentials, against different
# stores, and neither can stand in for the other.
#
# Both default to enabled. An installation that wants the checks off has to say
# so, because the failure mode of the opposite default is a deployment that
# looks authenticated and is not.
# --------------------------------------------------------------------------

#: Whether session tokens are verified. **Off is a development mode**, and the
#: dependency logs a warning on every request it lets through unverified, so a
#: process running this way cannot be mistaken for one that is not.
CLERK_ENABLED = _env_int("GRAPHRAG_CLERK_ENABLED", 1) == 1

#: Who must have issued the token. **No default**, for the same reason the
#: judge models have none: a default issuer is a trust decision nobody made.
#: With verification on and this unset, the dependency refuses to build.
CLERK_ISSUER = _env_str("GRAPHRAG_CLERK_ISSUER")

#: Where the issuer publishes its public signing keys. **No default.**
CLERK_JWKS_URL = _env_str("GRAPHRAG_CLERK_JWKS_URL")

#: Which authorized parties may present a token, as a comma-separated list.
#: Empty means the ``azp`` claim is not checked — an allow-list of nothing
#: would reject every token, which is not the same as not caring.
CLERK_AUTHORIZED_PARTIES = tuple(
    part.strip()
    for part in _env_str("GRAPHRAG_CLERK_AUTHORIZED_PARTIES").split(",")
    if part.strip()
)

#: Seconds of clock skew tolerated on ``exp`` and ``iat``.
#:
#: **Chosen, not measured.** Server clocks drift by seconds, and a token
#: rejected because two machines disagree by one second is an outage rather
#: than a security event. Small enough that an expired token stays expired.
CLERK_LEEWAY_SECONDS = _env_float("GRAPHRAG_CLERK_LEEWAY_SECONDS", 30.0)

#: Seconds a fetched signing key stays cached.
#:
#: **Chosen.** Long enough that key fetches are rare, short enough that a
#: rotation is picked up without a restart. Rotation does not wait for this:
#: an unknown ``kid`` triggers a fetch immediately, so this only bounds how
#: long a *withdrawn* key stays usable.
JWKS_CACHE_SECONDS = _env_float("GRAPHRAG_JWKS_CACHE_SECONDS", 300.0)

#: The only signature algorithm accepted. **Deliberately not configurable.**
#: Reading the algorithm from anywhere the request can influence is the
#: algorithm-confusion attack; reading it from the environment is the same
#: mistake one step removed, since it lets a misconfiguration accept ``none``
#: or an HMAC algorithm keyed on the public key.
CLERK_ALGORITHM = "RS256"

#: The user id requests run as when verification is off. The value says what
#: it is, so it is recognisable anywhere it surfaces — a log line, a stored
#: record, a bug report — as an unauthenticated request rather than a person.
DEV_USER_ID = _env_str("GRAPHRAG_DEV_USER_ID", "dev-user-AUTHENTICATION-DISABLED")

#: Whether the tenant is resolved from an API key. Off means one tenant and no
#: key required, which is the single-user development case.
MULTI_TENANCY_ENABLED = _env_int("GRAPHRAG_MULTI_TENANCY_ENABLED", 1) == 1

#: The organisation every request belongs to when tenancy is off. Named the
#: same way as the development user, and for the same reason.
DEFAULT_TENANT_ORG_ID = _env_str(
    "GRAPHRAG_DEFAULT_TENANT_ORG_ID", "dev-org-SINGLE-TENANT"
)

#: Where the control plane lives. Separate from the graph store on purpose:
#: the graph holds one organisation's data, and the control plane holds the
#: mapping from credential to organisation. One database holding both is a
#: single query away from a cross-tenant read.
CONTROL_PLANE_PATH = _env_str("GRAPHRAG_CONTROL_PLANE_PATH", "control-plane.db")


# --------------------------------------------------------------------------
# Document chunking
#
# A stored document is one chunk, not one body. The unit matters because
# coverage divides by the item's token count: an answer of N tokens caps every
# score at N/|I|, so clearing threshold t needs |I| <= N/t — at 0.2, |I| <= 5N.
# An item past that length is unreachable however relevant it is, and the
# response has to be a smaller unit rather than a looser threshold, because a
# threshold loose enough to admit a whole body admits everything.
#
# Measured on 600 documents drawn from three public repositories, in the four
# fields this project stores — pull request bodies, issue bodies, review
# bodies, commit messages:
#
#     pull request bodies   median 112 words, p90 300, max 4371
#     issue bodies          median 127 words, p90 317, max  614
#     commit messages       median   7 words, p90  34, max  271
#     all four              median  25 words, p90 210, p95 300
#
# Two things follow. Most documents are short — a 120-word window leaves 77%
# of them as a single chunk, and the median pull request body untouched — so
# chunking costs nothing on the common case and only splits the tail. And the
# tail is long enough to matter: the largest body sampled is 4,371 words.
#
# The ceiling is in scoring tokens, not words, and the two differ. Measured
# over 100 real bodies, tokenisation yields **0.523 tokens per word** after
# stopwords and the minimum length are applied. So:
#
#     window   resulting |I|          smallest answer that can clear 0.2
#     words    median / p90 tokens    median / p90
#      60        31 /  42                6.2 / 8.4
#     120        62 /  84               12.4 / 16.8
#     200       104 / 140               20.8 / 28.0
#
# **The size is set by the worst case, not the median.** 0.523 is prose. Text
# that is mostly identifiers, code, or long unrepeated words tokenises at up
# to 1.0 tokens per word, because nothing is a stopword and nothing falls
# under the minimum length — and a pull request body full of stack traces is
# exactly that. Sizing on the median would put those chunks over the ceiling
# while the average one looked fine, which is the failure that is invisible
# in an average.
#
# So the bound is taken at a ratio of 1.0: the shortest answer in the stored
# traces is 17 tokens, giving |I| <= 85, and a window of 80 words cannot
# exceed 80 tokens however dense its text. On real prose the same window lands
# at a median of 42 tokens and a p90 of 56, comfortably inside.
#
# What that costs: 67% of documents stay a single chunk instead of the 77% a
# 120-word window would leave, so more bodies split. That is the price of the
# bound holding for every input rather than for the typical one.
# --------------------------------------------------------------------------

#: Words per stored chunk. Sized so that even text tokenising at 1.0 tokens
#: per word stays under the coverage ceiling for a 17-token answer — the
#: shortest answer in the stored traces — which puts the limit at 85 tokens.
DOCUMENT_CHUNK_WORDS = 80

#: Words carried into the next chunk, so an entity spanning a boundary is
#: still whole in one of them.
#:
#: **Chosen, not measured.** The longest entity any rule can match is three
#: words — ``pull request #1347`` — so three is the measured floor. Fifteen is
#: five times that, leaving room for the statistical stage's multi-word spans
#: without measuring them, and costing 19% duplicated text. A measurement of
#: real span lengths would justify moving it.
DOCUMENT_CHUNK_OVERLAP_WORDS = 15
