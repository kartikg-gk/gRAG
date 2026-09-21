"""The graph vocabulary: node labels, relation names, and what each is worth.

One module so a rename costs one file. Nothing else should spell a node label
or a relation name as a literal — call sites import from here.

Node labels follow the recency table rather than GitHub's API, because
half-lives are keyed by label and that table is the authority. The ingestion
models keep GitHub's own names (``Repository``, ``PullRequest``, ``Issue``);
they mirror the API and are a deliberately separate vocabulary.

Names here generally arrive with the code that emits them, because a declared
constant with no producer reads as a shipped feature. ``RELATION_CO_OCCURS`` is
the one exception, and it is deliberate — see the note beside it in
``relations``.
"""

from __future__ import annotations

from .environment import load_environment

# Before any setting below is read: they are read at import.
load_environment()

import os
from pathlib import Path

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
# Relations and confidence
#
# Defined in ``relations`` so the compile path can read them without these
# settings; re-exported here for everything else.
# --------------------------------------------------------------------------

from .relations import (  # noqa: E402
    RELATION_AUTHORED,
    RELATION_RESOLVES,
    RELATION_REVIEWED,
    RELATION_TOUCHES,
    RELATION_PART_OF,
    RELATION_REPORTED,
    RELATION_CO_OCCURS,
    RELATION_AUTHORED_BY,
    RELATION_MENTIONS,
    CONFIDENCE,
)

# --------------------------------------------------------------------------
# Entity extraction
#
# The extractor reads text and returns entities. It does not build edges; the
# edge an extracted entity earns is ``MENTIONS``, priced above with the rest,
# and the extractor's own score rides on it as how sure the match was.
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
WINDOW_WORDS = 300
WINDOW_OVERLAP_WORDS = 50

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
REL_TABLE = "RELATES_TO"

#: Document to entity. Its own table because its endpoints differ from every
#: other relation — Document to Entity rather than Entity to Entity — which is
#: also why it carries no properties and is absent from the relation constants.
MENTIONS_TABLE = "MENTIONS"

#: Width of the embedding column. This must equal the embedding model's output
#: size: the column is fixed-width at CREATE TABLE time, so it cannot be
#: derived from the model the way ``Similarity.dimension`` is. Changing the
#: model to one of a different width means a new database, not a migration.
EMBEDDING_DIMENSION = 384

#: Vector index over the entity embedding column.
VECTOR_INDEX_NAME = "idx_entity_embedding"

#: Distance metric for that index. Cosine, because embeddings are L2-normalised
#: before they are stored, which makes cosine distance and the dot product the
#: same ordering — and lets similarity be recovered as ``1 - distance``.
VECTOR_METRIC = "cosine"

#: Read connections held open for concurrent queries. The engine serialises
#: writes but reads scale across connections, so this bounds how many can run
#: at once rather than how many exist.
try:
    READ_POOL_SIZE = max(1, int(os.environ.get("GRAPHRAG_DB_POOL_SIZE", "10")))
except ValueError:
    READ_POOL_SIZE = 10

#: Seconds to wait for a free read connection before giving up. A caller that
#: waits forever on an exhausted pool looks like a hang, not a queue.
POOL_TIMEOUT_SECONDS = 15.0


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
MAX_HOPS = _env_int("GRAPHRAG_GRAPH_MAX_HOPS", 2)

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
SEED_MIN_SIM = _env_float("GRAPHRAG_GRAPH_SEED_MIN_SIM", 0.35)

#: How many fuzzy seeds tier 2 may contribute.
#:
#: A cap, not a floor. Without it a permissive ``SEED_MIN_SIM`` seeds traversal
#: from the whole vector result set, and the graph arm stops being a traversal
#: from somewhere specific — it becomes a walk from everywhere, which returns
#: the graph and ranks it by nothing the traversal contributed.
SEED_TOP_N = _env_int("GRAPHRAG_GRAPH_SEED_TOP_N", 3)

#: Per-relation degree above which a node's neighbours are suppressed.
#:
#: Grouped per relation rather than per node. A repository node with 10,000
#: TOUCHES edges and 3 AUTHORED edges should lose the file list and keep the
#: authorship; capping per node would drop both, discarding the useful half
#: because the other half is broad.
MAX_DEGREE = _env_int("GRAPHRAG_MAX_DEGREE", 10)

# --------------------------------------------------------------------------
# Agent tools
#
# Bounds on what one tool call may return, so an agent working the graph step
# by step can neither walk all of it nor pull every document in one call.
# --------------------------------------------------------------------------

#: Ceiling on how far one impact trace walks, whatever the caller asks for.
TOOL_MAX_HOPS = _env_int("GRAPHRAG_TOOL_MAX_HOPS", 4)
#: Most entities one impact trace returns.
TOOL_MAX_IMPACT = _env_int("GRAPHRAG_TOOL_MAX_IMPACT", 50)
#: Most candidates one entity lookup returns.
TOOL_MAX_CANDIDATES = _env_int("GRAPHRAG_TOOL_MAX_CANDIDATES", 10)
#: Source documents attached to each returned entity.
TOOL_CITATIONS_PER_NODE = _env_int("GRAPHRAG_TOOL_CITATIONS_PER_NODE", 2)
#: Characters of each document returned: a snippet to cite, not the document.
TOOL_SNIPPET_CHARS = _env_int("GRAPHRAG_TOOL_SNIPPET_CHARS", 280)
#: Neighbours taken per node on each hop of an impact trace. Wider than the
#: retrieval arm's, because an impact question wants the whole neighbourhood.
TOOL_NEIGHBOR_K = _env_int("GRAPHRAG_TOOL_NEIGHBOR_K", 25)
#: Hub threshold on the first hop only, where the node asked about is often a
#: hub itself. Later hops keep MAX_DEGREE, so a hub further out cannot explode
#: the walk.
TOOL_SEED_MAX_DEGREE = _env_int("GRAPHRAG_TOOL_SEED_MAX_DEGREE", 200)
#: Below this similarity a name does not resolve at all, so a trace never
#: starts from a weak guess.
TOOL_RESOLVE_MIN_SIM = _env_float("GRAPHRAG_TOOL_RESOLVE_MIN_SIM", 0.40)

#: How many results each arm returns. Separate constants because the two arms
#: are measured against each other and a shared value would make a difference
#: in their result counts impossible to attribute.
TOP_K_VECTOR = _env_int("GRAPHRAG_TOP_K_VECTOR", 10)
TOP_K_GRAPH = _env_int("GRAPHRAG_TOP_K_GRAPH", 10)

#: Per-frontier neighbour cap used by the retrieval graph stream.
GRAPH_NEIGHBOR_K = _env_int("GRAPHRAG_GRAPH_NEIGHBOR_K", 5)

#: Query length and shared embedding concurrency bounds.
MAX_QUERY_CHARS = _env_int("GRAPHRAG_MAX_QUERY_CHARS", 2000)
EMBED_WORKERS = _env_int("GRAPHRAG_EMBED_WORKERS", 4)
QUERY_EXTRACT_CACHE = _env_int("GRAPHRAG_QUERY_EXTRACT_CACHE", 512)


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
    "caused by",
    "because of",
    "related to",
    "connected to",
    "depends on",
    "owns",
    "owned by",
    "between",
    "path from",
    "linked to",
    "responsible for",
    "which pr",
    "which ticket",
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
    "what is",
    "how does",
    "concept",
    "definition",
    "purpose of",
)

INTENT_RELATIONAL = "relational"
INTENT_SEMANTIC = "semantic"
# Compatibility for callers that import the old name; its value is pinned to
# the new public label.
INTENT_CONCEPTUAL = INTENT_SEMANTIC

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
    "Issue": 21.0,
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


def _env_raw(name: str) -> str:
    """A variable exactly as exported, or "" when unset. Nothing is trimmed."""
    import os

    return os.environ.get(name, "")


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

#: Seconds one attempt at an answer or a summary may take before it is
#: abandoned.
#:
#: **Chosen, not measured.** Without it the client waits as long as the library
#: lets it — ten minutes — and a provider that stops responding holds a request,
#: and a worker thread, for all of that. Long enough for a slow model to finish
#: a full answer; a stream only needs each piece to arrive inside it. Applies
#: per attempt, and the library's own retries still run, so the worst case is
#: a small multiple of this. The routes already turn the error into their
#: failure answer.
ANSWER_TIMEOUT_SECONDS = _env_float("GRAPHRAG_ANSWER_TIMEOUT_SECONDS", 60.0)

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

#: Who must have issued the token. **No default**: a default issuer is a trust
#: decision nobody made.
CLERK_ISSUER = _env_str("GRAPHRAG_CLERK_ISSUER")

#: Where the issuer publishes its public signing keys. Unset, the issuer's
#: standard well-known location.
CLERK_JWKS_URL = _env_str("GRAPHRAG_CLERK_JWKS_URL") or (
    f"{CLERK_ISSUER.rstrip('/')}/.well-known/jwks.json" if CLERK_ISSUER else ""
)

#: Whether session tokens are verified: exactly when an issuer is configured.
#: Without one every request runs as the development user, and the dependency
#: says so in the log.
CLERK_ENABLED = bool(CLERK_ISSUER)

#: Which authorized parties may present a token, as a comma-separated list.
#: Empty means the ``azp`` claim is not checked — an allow-list of nothing
#: would reject every token, which is not the same as not caring.
CLERK_AUTHORIZED_PARTIES = tuple(
    part.strip()
    for part in _env_str("GRAPHRAG_CLERK_AUTHORIZED_PARTIES").split(",")
    if part.strip()
)

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
DEV_USER_ID = _env_str("GRAPHRAG_DEV_USER_ID", "dev-user")

#: The exact values of GRAPHRAG_MULTI_TENANCY_ENABLED that turn tenancy on.
#: Compared as written: no trimming and no case folding.
MULTI_TENANCY_ON_VALUES = ("1", "true", "True")

#: Whether the tenant is resolved from an API key. Off means one tenant and no
#: key required, which is the single-user development case, and is what any
#: value outside MULTI_TENANCY_ON_VALUES gives, unset included.
MULTI_TENANCY_ENABLED = _env_raw("GRAPHRAG_MULTI_TENANCY_ENABLED") in MULTI_TENANCY_ON_VALUES

#: The organisation every request belongs to when tenancy is off. Named the
#: same way as the development user, and for the same reason.
DEFAULT_TENANT_ORG_ID = _env_str(
    "GRAPHRAG_DEFAULT_TENANT_ORG_ID", "dev-org-SINGLE-TENANT"
)

#: The shared secret the admin onboarding route checks. **No default**: unset,
#: that route refuses every request rather than accepting one it cannot check.
ADMIN_SECRET_KEY = _env_str("GRAPHRAG_ADMIN_SECRET_KEY")

#: The secret GitHub signs webhook deliveries with. **No default**: unset, the
#: webhook route refuses every delivery, because a payload that cannot be
#: verified is one anybody could have sent.
GITHUB_WEBHOOK_SECRET = _env_str("GRAPHRAG_GITHUB_WEBHOOK_SECRET")

#: The browser origins allowed to call the API, comma-separated. Any origin by
#: default, which suits local use; a deployment names its frontend's origin.
CORS_ORIGINS = [
    origin.strip()
    for origin in _env_str("GRAPHRAG_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

#: Where the control plane lives is **not** here. It is a database URL read at
#: connection time by ``src.models.database``, from
#: ``GRAPHRAG_CONTROL_PLANE_DATABASE_URL`` or ``GRAPHRAG_DATABASE_URL``, and it
#: has no default. Every setting in this module has one, and a default here
#: would be a local file — which would give each process its own private
#: control plane and no error, on the one system where several processes are
#: required to share it.


# --------------------------------------------------------------------------
# Document chunking
#
# Stored source documents use character windows. This is deliberately separate
# from entity extraction's word windows: storage boundaries preserve exact
# substrings and have their own geometry.
# --------------------------------------------------------------------------

#: Python string characters per stored source-document chunk.
DOCUMENT_CHUNK_CHARACTERS = 1200

#: Python string characters carried into the next stored document chunk.
DOCUMENT_CHUNK_OVERLAP_CHARACTERS = 150


# --------------------------------------------------------------------------
# The served store
#
# The HTTP surface opens one store for the life of the process. Which store
# is a deployment decision, so it is read from the environment like every
# other value here rather than passed on a command line the server does not
# have.
# --------------------------------------------------------------------------

#: The project root: the directory holding ``src/``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Where the served graph lives: ``graph.lbug`` in the project root unless
#: the environment names another file.
STORE_PATH = _env_str("GRAPHRAG_STORE_PATH", str(PROJECT_ROOT / "graph.lbug"))

#: Whether to embed a throwaway string at startup.
#:
#: **Measured.** The embedding model's cold start is roughly fifteen seconds,
#: and it is paid on the first call that needs a vector. Left to itself that
#: is the first user query, which turns a slow start into a slow product. One
#: embed at boot moves the cost to where nobody is waiting on it.
#:
#: Off is for tests and for any process that will never embed, where fifteen
#: seconds of model loading buys nothing.
WARM_EMBEDDER_ON_STARTUP = _env_int("GRAPHRAG_WARM_EMBEDDER", 1) == 1



# --------------------------------------------------------------------------
# Artifact storage
#
# A built graph is a file, and moving it between wherever it was built and
# wherever it is served is a separate concern from either. These say where it
# goes and which backend puts it there.
#
# The default needs nothing set up: files copied under a local directory, no
# service, no credentials, no optional package installed. That is what
# development and the test suite use, and it is a real backend rather than a
# stub — the cloud one is the alternative, not the real one.
# --------------------------------------------------------------------------

#: Which backend writes new artifacts. ``local`` or ``cloud``.
#:
#: Only writes consult this. A read dispatches on the scheme of the URI it was
#: given, so an artifact written to one backend stays readable after the
#: default changes to the other.
ARTIFACT_BACKEND = _env_str("GRAPHRAG_ARTIFACT_BACKEND", "local")

#: Where the local backend keeps things. Created on demand.
ARTIFACT_ROOT = _env_str("GRAPHRAG_ARTIFACT_ROOT", "artifacts")

#: Prefix used for artifact object keys in both storage backends.
ARTIFACT_PREFIX = _env_str("GRAPHRAG_ARTIFACT_S3_PREFIX", "artifacts")

#: Bucket for the cloud backend. **No default** — a default bucket name is a
#: default destination for somebody else's data.
ARTIFACT_BUCKET = _env_str("GRAPHRAG_ARTIFACT_BUCKET")

#: Region for the cloud backend, when its client needs one. **No default**,
#: for the same reason.
ARTIFACT_REGION = _env_str("GRAPHRAG_ARTIFACT_REGION")

#: Where a downloaded artifact lands before it is opened. Separate from the
#: artifact root because one is storage and the other is a cache: the cache is
#: keyed by which process holds it and is safe to delete.
POD_CACHE_ROOT = _env_str("GRAPHRAG_POD_CACHE_ROOT", "cache")

#: Which serving process this is.
#:
#: The local development identity. Deployments set this explicitly so two
#: processes never share the cache key represented by this value.
POD_ID = _env_str("GRAPHRAG_POD_ID", "pod-local")

#: Where other processes would reach this one.
#:
#: Recorded rather than used: nothing dials a pod yet. It defaults to the
#: loopback address because a single-machine deployment has no better answer,
#: and a wrong-but-honest local address is easier to notice than a blank one.
POD_ADDRESS = _env_str("GRAPHRAG_POD_ADDRESS", "127.0.0.1")

#: Seconds between one reconcile pass and the next.
#:
#: Five is short enough that a tenant whose intent moved is serving the new
#: graph within a few seconds, and long enough that a caught-up fleet's
#: repeated single query costs nothing worth measuring.
RECONCILE_INTERVAL_SECONDS = _env_float("GRAPHRAG_RECONCILE_INTERVAL", 5.0)

#: How long a pod's last heartbeat stays good enough to place work on it.
#:
#: Two minutes is many intervals' worth of beats, so a pod has to have missed
#: a run of them before it stops being a candidate — a single slow tick must
#: not take capacity out of the fleet. Tunable because the right number is a
#: property of the deployment's interval and its tolerance for placing a
#: tenant on something that has just died.
POD_HEARTBEAT_WINDOW_SECONDS = _env_float("GRAPHRAG_POD_HEARTBEAT_WINDOW", 120.0)


# --------------------------------------------------------------------------
# Error reporting, opt in
# --------------------------------------------------------------------------

#: Where errors are reported. Unset, nothing is reported and nothing is loaded.
SENTRY_DSN = _env_str("GRAPHRAG_SENTRY_DSN")

#: The environment name errors are filed under.
SENTRY_ENVIRONMENT = _env_str("GRAPHRAG_SENTRY_ENVIRONMENT", "development")

#: The share of requests traced for performance.
SENTRY_TRACES_SAMPLE_RATE = _env_float("GRAPHRAG_SENTRY_TRACES_SAMPLE_RATE", 1.0)
