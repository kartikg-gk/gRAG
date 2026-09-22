"""Reading one repository's changes into a tenant's accumulated graph.

**A second ingest path, beside the one that already exists and does not
change.** The older path opens a store file directly and serves one process
with no backend behind it; this one writes rows a tenant accumulates in a
database, which a later compile turns into a store. Neither is a draft of the
other, and the duplication is the design: collapsing them would mean the
local path could not run without a database.

What it emits, and what it does not
-----------------------------------

**Three relation types: authorship, mention and resolution.** Resolution is
a pull request whose body says it fixes, closes or resolves an issue, read by
the same rule the other path uses. That is fewer than the typed set the other
path records, which also tells reviewing and touching apart. An artifact
compiled from these rows therefore carries three kinds of edge, not six —
worth saying here rather than leaving to be discovered by whoever wonders
where the review edges went.

The identifiers are what make an entity one thing
-------------------------------------------------

A node identifier is derived from what the thing *is*, not from where it was
found, which is what lets a service named in twenty pull requests be one node
with twenty edges rather than twenty nodes. Slugs are the mechanism, and a
slug is never allowed to be empty: two differently-named things that both
slugify to nothing would otherwise become the same node.

Accumulation, and why the rules differ
--------------------------------------

**Nodes: the first write wins.** The same person appears in every pull
request they opened; embedding them once per appearance is a model call per
appearance for a single row.

**Edges: the highest weight wins.** The same entity can be mentioned twice
with different confidences, and keeping the last one seen would make the
result depend on the order the items came back in. The strongest claim is the
one worth keeping, and it does not move when the iteration order does.

A quiet repository costs nothing
--------------------------------

An empty delta returns before an extractor or an embedding model is built.
Both are expensive — the model in particular carries a multi-second load — and
a repository with no new activity is the common case on a fleet.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from sqlmodel import Session, delete, select

from ..common.relations import RELATION_AUTHORED_BY, RELATION_MENTIONS, RELATION_RESOLVES
from ..models.graph_store import (
    EntityEdge,
    EntityNode,
    ensure_graph_store_schema,
    upsert_edges,
    upsert_nodes,
)

logger = logging.getLogger("graphrag.worker.ingest")

#: The relations this path records, named and priced in the shared
#: vocabulary. These are the ones *this* path can tell apart from a title and
#: a body; see the module docstring: the richer set lives on the other path.
#:
#: What this path writes as an edge's weight is how sure it is the edge exists
#: — 1.0 for authorship, the extractor's score for a mention. What the relation
#: is worth is applied when the artifact is compiled.
RELATION_AUTHORED = RELATION_AUTHORED_BY

#: "Fixes #982", "closes #12", "resolved #7": a pull request naming the issue it
#: closes. The same rule the locally built graph applies, so a tenant graph
#: carries the link from a change to the problem it solved.
CLOSING_REFERENCE = re.compile(
    r"(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+#(\d+)", re.IGNORECASE
)

#: Extracted entity types that are a number, not a name: "#433". A number
#: means something only inside one repository, so it is resolved to that
#: repository's issue or pull request, never made a node of its own — keyed by
#: its text it would merge "#433" from every repository into one node.
NUMBERED_TYPES = ("Ticket", "PR")

#: Relations recomputed from an item's text on every read of it. Rewritten
#: whole, so a reference deleted from a body leaves the graph with it.
TEXT_RELATIONS = (RELATION_MENTIONS, RELATION_RESOLVES)

#: How an item's text names an issue or pull request by number. Each form
#: says which repository it means, and only this repository's are resolved:
#:
#: - ``#203`` — bare, so this repository;
#: - ``owner/repo#203`` — that repository;
#: - ``github.com/owner/repo/issues/203`` or ``/pull/203`` — that repository,
#:   and the same through dependabot's redirect host;
#: - ``<a href="…">#203</a>`` — the text of a link means what the link means.
#:
#: Read in that order, with each match blanked out before the next form is
#: looked for, so a bare ``#203`` inside a qualified or linked one is never
#: read a second time as this repository's.
_REPO_URL = re.compile(
    r"(?:github\.com|github-redirect\.dependabot\.com)/([\w.-]+)/([\w.-]+)/(?:issues|pull)/(\d+)",
    re.IGNORECASE,
)
_LINKED_NUMBER = re.compile(
    r"<a\b[^>]*\bhref=[\"']([^\"']*)[\"'][^>]*>\s*#(\d+)\s*</a>", re.IGNORECASE
)
_QUALIFIED_NUMBER = re.compile(r"(?<![\w./-])([\w.-]+)/([\w.-]+)#(\d+)\b")
_BARE_NUMBER = re.compile(r"(?<![\w/#&])#(\d+)\b")


def numbered_references(text: str, repo_name: str) -> list[str]:
    """The issue and pull request numbers ``text`` names in ``repo_name``.

    In the order first named, each once. References to another repository
    are left out, and so is a link whose target is not an issue or pull
    request at all: there is nothing here it could mean.
    """
    here = repo_name.lower()
    found: list[str] = []

    def blank(match: re.Match) -> str:
        return " " * (match.end() - match.start())

    def linked(match: re.Match) -> str:
        url = _REPO_URL.search(match.group(1))
        if url is not None and f"{url.group(1)}/{url.group(2)}".lower() == here:
            found.append(match.group(2))
        return blank(match)

    def qualified(match: re.Match) -> str:
        if f"{match.group(1)}/{match.group(2)}".lower() == here:
            found.append(match.group(3))
        return blank(match)

    text = _LINKED_NUMBER.sub(linked, text)
    text = _REPO_URL.sub(qualified, text)
    text = _QUALIFIED_NUMBER.sub(qualified, text)
    found.extend(match.group(1) for match in _BARE_NUMBER.finditer(text))
    return list(dict.fromkeys(found))

#: The label an item kind is stored under, where it differs from the kind.
#:
#: Recency picks a half-life by label, and the table spells a pull request
#: "PR". The kind stays in the node id, so relabelling never splits a node that
#: is already stored.
ITEM_LABELS = {"PullRequest": "PR"}

#: What a node identifier starts with, by what the node is.
PERSON_PREFIX = "person"
ENTITY_PREFIX = "entity"

#: What an empty slug becomes.
#:
#: A slug must never be empty. Two names that both reduce to nothing — one
#: punctuation, one another script — would otherwise be handed the same
#: identifier and become one node.
EMPTY_SLUG = "x"

#: Places past the decimal point kept on a mention's weight. A confidence is
#: a judgement to about this precision, and storing seventeen digits of one
#: invites comparisons that are really comparing float noise.
WEIGHT_PLACES = 4

_extractor = None
_embedder = None


@dataclass(frozen=True)
class IngestResult:
    """What one pass over a repository did."""

    cursor: Optional[str]
    nodes: int
    edges: int
    items: int


def slugify(text: str) -> str:
    """A lowercase, hyphenated form of ``text``, never empty.

    Every run of anything that is not a letter or a digit becomes one hyphen,
    and the ends are trimmed. What is left of a name with nothing to keep is
    the placeholder, so an identifier is always distinguishable rather than
    silently shared.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug or EMPTY_SLUG


def _iso(moment: Any) -> str | None:
    """A time as ISO text, for a JSON column. The fetcher hands back datetimes;
    a payload read elsewhere may already carry the string."""
    if moment is None:
        return None
    if isinstance(moment, datetime):
        return moment.isoformat()
    return str(moment)


def item_node_id(repo_id: str, kind: str, number: int | str) -> str:
    """The identifier for a pull request or an issue."""
    return f"{repo_id}:{kind.lower()}:{number}"


def person_node_id(name: str) -> str:
    """The identifier for a person, wherever they were named."""
    return f"{PERSON_PREFIX}:{slugify(name)}"


def _remove_orphaned_entities(db: Session, org_id: str) -> None:
    """Drop extracted entities nothing points at any more.

    An extracted entity exists only because some item mentioned it. Once the
    last mention is rewritten away it is a node with no reason to be in the
    graph. People and items are never touched here.
    """
    linked = select(EntityEdge.target_id).where(EntityEdge.org_id == org_id)
    db.exec(delete(EntityNode).where(
        EntityNode.org_id == org_id,
        EntityNode.node_id.like(f"{ENTITY_PREFIX}:%"),
        EntityNode.node_id.not_in(linked),
    ))


def entity_node_id(entity_type: str, text: str) -> str:
    """The identifier for something the extractor found."""
    return f"{ENTITY_PREFIX}:{slugify(entity_type)}:{slugify(text)}"


def default_extractor():
    """The extractor, built once for this process.

    Built lazily and kept, because construction is not free and every
    repository ingested in this process wants the same one.

    This project's own extractor, rather than the zero-shot model that would
    otherwise be the obvious default here: that model was measured on this
    corpus and rejected — more entities, none of them the ticket and pull
    request references that matter, thousands of times slower, and gigabytes
    of resident memory. The substitution is forced rather than chosen.
    """
    global _extractor
    if _extractor is None:
        from ..analysis.extract import Extractor

        _extractor = Extractor()
    return _extractor


def default_embed(text: str) -> list[float]:
    """Embed one string, through a model built once for this process."""
    global _embedder
    if _embedder is None:
        from ..analysis.similarity import SentenceTransformerEmbedder

        _embedder = SentenceTransformerEmbedder()
    return _embedder.embed(text)


def reset_defaults() -> None:
    """Drop the built extractor and model. For tests, and for a process that
    wants the memory back."""
    global _extractor, _embedder
    _extractor = None
    _embedder = None


def ingest_repository(
    *,
    org_id: str,
    repo_id: str,
    repo_name: str,
    cursor: str | None = None,
    db: Session,
    token: str | None = None,
    extractor=None,
    embed: Callable[[str], list[float]] | None = None,
    fetch: Callable[..., Iterable[Any]] | None = None,
    now: int | None = None,
) -> IngestResult:
    """Read what changed in ``repo_name`` since ``cursor`` into the graph store.

    Returns the new cursor and what was written. The cursor comes back
    unchanged when there was nothing new, so a caller can store the result
    without asking whether anything happened.
    """
    # The cheap guard, not the full call: this runs before every batch, and
    # asking the database whether its tables exist each time is a round trip
    # bought for nothing.
    ensure_graph_store_schema(db.get_bind())

    items = list((fetch or _fetch_delta)(repo_name, cursor, token))
    if not items:
        # Before anything expensive is built. A repository with no activity
        # is the ordinary case, and it must not load a model to discover it.
        logger.info("%s: nothing new since %s", repo_name, cursor)
        return IngestResult(cursor=cursor, nodes=0, edges=0, items=0)

    extract = extractor if extractor is not None else default_extractor()
    embedder = embed if embed is not None else default_embed
    moment = now if now is not None else _now()

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    #: (item, number, score) for a "#N" whose item was not known when found.
    #: Held here only: no node is written for it while it waits.
    deferred: list[tuple[str, str, float]] = []
    #: This repository's items already in the graph from earlier runs, read
    #: once, so resolving a number costs no query of its own.
    stored_items = set(db.exec(
        select(EntityNode.node_id).where(
            EntityNode.org_id == org_id,
            EntityNode.node_id.like(f"{repo_id}:%"),
        )
    ).all())

    def add_node(node_id: str, *, embed_text: str, **fields) -> None:
        """First write wins — see the module docstring.

        The text to embed is passed in rather than the vector, so the model
        is not called for a node that is already known. Embedding first and
        discarding the result afterwards produces identical rows and one
        model call per appearance, which is the whole cost this avoids.
        """
        if node_id in nodes:
            return
        nodes[node_id] = {
            "org_id": org_id,
            "node_id": node_id,
            "created_at": moment,
            "updated_at": moment,
            "embedding": embedder(embed_text),
            **fields,
        }

    def add_edge(source: str, target: str, relation: str, weight: float) -> None:
        """Highest weight wins, so iteration order cannot decide the answer."""
        key = (source, target, relation)
        existing = edges.get(key)
        if existing is not None and existing["weight"] >= weight:
            return
        edges[key] = {
            "org_id": org_id,
            "source_id": source,
            "target_id": target,
            "relation_type": relation,
            "weight": weight,
            "created_at": moment,
        }

    def link_number(source: str, number: str, score: float) -> bool:
        """Point ``source`` at this repository's item ``#number``, if it is known.

        Known means read earlier in this pass or stored by an earlier one.
        GitHub numbers issues and pull requests from one sequence, so at most
        one of the two exists. An item naming its own number links nothing.
        """
        for kind in ("Issue", "PullRequest"):
            target = item_node_id(repo_id, kind, number)
            if target in nodes or target in stored_items:
                if target != source:
                    add_edge(source, target, RELATION_MENTIONS, round(score, WEIGHT_PLACES))
                return True
        return False

    for item in items:
        kind = _kind_of(item)
        number = _field(item, "number")
        title = _field(item, "title") or ""
        body = _field(item, "body") or ""
        author = _author_of(item)

        node_id = item_node_id(repo_id, kind, number)
        add_node(
            node_id,
            repo_id=repo_id,
            label=ITEM_LABELS.get(kind, kind),
            name=f"{kind} #{number}",
            properties={
                "title": title,
                "url": _field(item, "html_url"),
                "author": author,
                "source_id": _field(item, "id"),
                "merged": _merged(item),
                # The prose and when it was written, for the compiled graph:
                # the body becomes the item's source document, and the time
                # its timestamp, which is what recency ranks by.
                "body": body,
                "created_at": _iso(_field(item, "created_at")),
            },
            # The prose, not the display name. Every item in a repository has
            # a name of the same shape, so embedding names would put them all
            # at nearly the same point and make the vector arm useless.
            embed_text=f"{kind} #{number}: {title}\n\n{body}",
        )

        if author:
            person = person_node_id(author)
            add_node(
                person,
                repo_id=None,
                label="Person",
                name=author,
                properties={"login": author},
                embed_text=author,
            )
            add_edge(node_id, person, RELATION_AUTHORED, 1.0)

        for entity in extract.extract(f"{title}\n\n{body}"):
            if entity.type in NUMBERED_TYPES:
                # Read from the text below instead, where the repository a
                # number belongs to can be seen; the extractor sees "#203".
                continue
            found = entity_node_id(entity.type, entity.text)
            add_node(
                found,
                repo_id=repo_id,
                label=entity.type,
                name=entity.text,
                properties={"type": entity.type, "source": entity.source},
                embed_text=entity.text,
            )
            add_edge(
                node_id, found, RELATION_MENTIONS, round(float(entity.score), WEIGHT_PLACES)
            )

        for number in numbered_references(f"{title}\n\n{body}", repo_name):
            reference = (node_id, number, 1.0)
            if not link_number(*reference):
                deferred.append(reference)

        if kind == "PullRequest":
            # To the issue's node whether or not it has been read yet: an edge
            # whose end is missing is not written when the graph is compiled,
            # and is there as soon as the issue is.
            for closed in sorted(set(CLOSING_REFERENCE.findall(body))):
                add_edge(node_id, item_node_id(repo_id, "Issue", closed), RELATION_RESOLVES, 1.0)

    # The second and last try for a number whose item had not been read when
    # it was found: an issue later in this same pass than the pull request
    # naming it. One that still names nothing is dropped — another
    # repository's, or an item not read yet, with nothing it could point at.
    linked_late = sum(1 for reference in deferred if link_number(*reference))
    if deferred:
        logger.info(
            "%s: %d reference(s) deferred, %d linked on retry, %d dropped",
            repo_name, len(deferred), linked_late, len(deferred) - linked_late,
        )

    # What these items' text said before is replaced by what it says now.
    read = [node_id for node_id in nodes if node_id.startswith(f"{repo_id}:")]
    if read:
        db.exec(delete(EntityEdge).where(
            EntityEdge.org_id == org_id,
            EntityEdge.source_id.in_(read),
            EntityEdge.relation_type.in_(TEXT_RELATIONS),
        ))

    written_nodes = upsert_nodes(db, list(nodes.values()))
    written_edges = upsert_edges(db, list(edges.values()))
    _remove_orphaned_entities(db, org_id)
    # One commit over both. A run that wrote its entities and lost its
    # relationships would leave a graph that looks populated and traverses
    # nowhere.
    db.commit()

    moved = _new_cursor(items, cursor)
    logger.info(
        "%s: %d item(s) -> %d node(s), %d edge(s)",
        repo_name,
        len(items),
        written_nodes,
        written_edges,
    )
    return IngestResult(
        cursor=moved, nodes=written_nodes, edges=written_edges, items=len(items)
    )


# --------------------------------------------------------------------------
# reading the source
# --------------------------------------------------------------------------


def _fetch_delta(repo_name: str, cursor: str | None, token: str | None):
    """Pull requests and issues touched since ``cursor``.

    The cursor is the last update time this repository was read to. The
    client lists newest first, so this stops at the first item that is not
    newer rather than reading a repository's whole history to find the few
    that changed.
    """
    from ..ingestion.github import fetch_issues, fetch_pull_requests, make_session

    since = _parse_cursor(cursor)

    with make_session(token) as session:
        for source in (fetch_pull_requests, fetch_issues):
            for item in source(session, repo_name):
                if since is not None and _updated_at(item) is not None:
                    if _updated_at(item) <= since:
                        # Newest first, so everything after this is older.
                        break
                yield item


def _parse_cursor(cursor: str | None) -> datetime | None:
    if not cursor:
        return None
    try:
        parsed = datetime.fromisoformat(cursor)
    except ValueError:
        logger.warning("unreadable cursor %r; reading from the beginning", cursor)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _new_cursor(items: list[Any], previous: str | None) -> str | None:
    """The latest update time seen, or what we started with."""
    times = [_updated_at(item) for item in items]
    latest = max((moment for moment in times if moment is not None), default=None)
    return latest.isoformat() if latest is not None else previous


def _updated_at(item: Any) -> datetime | None:
    moment = _field(item, "updated_at")
    if isinstance(moment, str):
        try:
            moment = datetime.fromisoformat(moment.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(moment, datetime):
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return None


def _kind_of(item: Any) -> str:
    """What this item is, as it will be labelled."""
    explicit = _field(item, "kind")
    if explicit:
        return str(explicit)
    return "PullRequest" if _field(item, "merge_commit_sha", missing=False) is not False else "Issue"


def _author_of(item: Any) -> str | None:
    user = _field(item, "user")
    if user is None:
        return None
    login = _field(user, "login")
    return str(login) if login else None


def _merged(item: Any) -> bool:
    merged_at = _field(item, "merged_at")
    if merged_at is not None:
        return True
    return bool(_field(item, "merge_commit_sha"))


def _field(item: Any, name: str, *, missing=None):
    """A field off a model or a mapping, whichever this is."""
    if isinstance(item, dict):
        return item.get(name, missing)
    return getattr(item, name, missing)


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())
