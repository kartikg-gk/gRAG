"""Tests for reading a repository's changes into a tenant's graph.

The first one is the reason first-wins accumulation exists. Embedding the
same person once per pull request they opened produces **exactly the same
rows** as embedding them once — the only difference is a model call per
appearance, and the only thing that catches it is a call count.

No model and no extractor are built here. Both are injectable precisely so a
test of what gets written does not load one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest
from sqlalchemy import event
from sqlmodel import select

from src.models.database import control_plane_sessions, create_control_plane_engine
from src.models.graph_store import (
    EntityEdge,
    EntityNode,
    create_graph_store_schema,
    reset_schema_guard,
)
from src.worker.ingest import (
    ENTITY_PREFIX,
    PERSON_PREFIX,
    RELATION_AUTHORED,
    RELATION_MENTIONS,
    entity_node_id,
    ingest_repository,
    item_node_id,
    person_node_id,
    reset_defaults,
    slugify,
)

NOW = 1_700_000_000


@dataclass
class FakeEntity:
    """What the extractor hands back, as far as this path cares."""

    text: str
    type: str
    score: float = 0.9
    source: str = "rules"


class CountingEmbedder:
    """Records every string it was asked to embed."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text)), 0.5]


class FixedExtractor:
    """Returns whatever it was told to, per text, and counts its calls."""

    def __init__(self, entities=None, by_text=None):
        self.entities = entities or []
        self.by_text = by_text or {}
        self.calls: list[str] = []

    def extract(self, text: str):
        self.calls.append(text)
        for marker, found in self.by_text.items():
            if marker in text:
                return found
        return self.entities


def item(number=1, *, kind="PullRequest", title="the auth change", body="body text",
         author="ada", updated="2026-08-01T12:00:00+00:00", merged=True):
    return {
        "kind": kind,
        "number": number,
        "title": title,
        "body": body,
        "user": {"login": author},
        "html_url": f"https://example.invalid/pull/{number}",
        "id": 1000 + number,
        "merged_at": "2026-08-01T12:00:00+00:00" if merged else None,
        "updated_at": updated,
    }


@pytest.fixture()
def engine(tmp_path):
    made = create_control_plane_engine(tmp_path / "graph-store.db")
    create_graph_store_schema(made)
    try:
        yield made
    finally:
        made.dispose()


@pytest.fixture()
def db(engine):
    with control_plane_sessions(engine)() as open_session:
        yield open_session


@pytest.fixture(autouse=True)
def clean_module_state():
    reset_schema_guard()
    reset_defaults()
    yield
    reset_schema_guard()
    reset_defaults()


def run(db, items, *, extractor=None, embed=None, cursor=None, **kwargs):
    return ingest_repository(
        org_id=kwargs.pop("org_id", "org_1"),
        repo_id=kwargs.pop("repo_id", "repo_1"),
        repo_name=kwargs.pop("repo_name", "acme/widgets"),
        cursor=cursor,
        db=db,
        extractor=extractor if extractor is not None else FixedExtractor(),
        embed=embed if embed is not None else CountingEmbedder(),
        fetch=lambda *args, **kw: list(items),
        now=NOW,
        **kwargs,
    )


# ==========================================================================
# the one a call count catches and the rows do not
# ==========================================================================


def test_a_person_in_many_items_is_embedded_once(db):
    """First-wins accumulation, which is invisible in the result.

    Twenty pull requests by one author produce one person row either way. The
    difference is twenty model calls against one, and nothing about the
    written rows can tell the two apart.
    """
    embedder = CountingEmbedder()
    items = [item(number=index, author="ada") for index in range(1, 21)]

    result = run(db, items, embed=embedder)

    person = person_node_id("ada")
    assert embedder.calls.count("ada") == 1
    # One call per item for the items themselves, plus the single person.
    assert len(embedder.calls) == 21
    assert result.nodes == 21

    stored = db.exec(select(EntityNode).where(EntityNode.node_id == person)).all()
    assert len(stored) == 1


def test_an_entity_in_many_items_is_embedded_once(db):
    """The same rule, for what the extractor finds."""
    embedder = CountingEmbedder()
    extractor = FixedExtractor([FakeEntity(text="auth-service", type="Service")])

    run(db, [item(number=1), item(number=2), item(number=3)], extractor=extractor, embed=embedder)

    assert embedder.calls.count("auth-service") == 1


# ==========================================================================
# a quiet repository
# ==========================================================================


def test_an_empty_delta_builds_neither_the_extractor_nor_the_embedder(db, caplog):
    """The common case on a fleet, and it must cost nothing.

    Asserted on the stand-ins never being called, because the point is not
    that nothing was written — it is that nothing was *built*.
    """
    embedder = CountingEmbedder()
    extractor = FixedExtractor()

    with caplog.at_level(logging.INFO):
        result = run(db, [], extractor=extractor, embed=embedder, cursor="2026-08-01T00:00:00+00:00")

    assert result.cursor == "2026-08-01T00:00:00+00:00"
    assert (result.nodes, result.edges, result.items) == (0, 0, 0)
    assert embedder.calls == []
    assert extractor.calls == []
    assert db.exec(select(EntityNode)).all() == []
    assert "nothing new since" in caplog.text


def test_an_empty_delta_does_not_reach_for_the_real_extractor(db, monkeypatch):
    """Not merely that the stand-in was unused: the default is not built.

    Reaching for the default would load a model, which is exactly the cost
    the early return exists to avoid.
    """
    import src.worker.ingest as ingest_module

    def refuse():
        raise AssertionError("the extractor was built for an empty delta")

    def refuse_embed(text):
        raise AssertionError("the model was built for an empty delta")

    monkeypatch.setattr(ingest_module, "default_extractor", refuse)
    monkeypatch.setattr(ingest_module, "default_embed", refuse_embed)

    result = ingest_repository(
        org_id="org_1",
        repo_id="repo_1",
        repo_name="acme/widgets",
        cursor=None,
        db=db,
        fetch=lambda *a, **k: [],
        now=NOW,
    )

    assert result.items == 0


# ==========================================================================
# what one item produces
# ==========================================================================


def test_an_item_produces_itself_its_author_and_an_authorship_edge(db):
    result = run(db, [item(number=41, author="ada")])

    node_id = item_node_id("repo_1", "PullRequest", 41)
    person = person_node_id("ada")

    stored = {row.node_id: row for row in db.exec(select(EntityNode)).all()}
    assert set(stored) == {node_id, person}

    pull_request = stored[node_id]
    assert pull_request.label == "PR"
    assert pull_request.name == "PullRequest #41"
    assert pull_request.repo_id == "repo_1"
    assert pull_request.properties["title"] == "the auth change"
    assert pull_request.properties["author"] == "ada"
    assert pull_request.properties["merged"] is True
    assert pull_request.properties["url"].endswith("/41")
    assert pull_request.properties["source_id"] == 1041

    author = stored[person]
    assert author.label == "Person"
    assert author.name == "ada"
    # A person belongs to no single repository.
    assert author.repo_id is None

    edges = db.exec(select(EntityEdge)).all()
    assert len(edges) == 1
    assert (edges[0].source_id, edges[0].target_id) == (node_id, person)
    assert edges[0].relation_type == RELATION_AUTHORED
    assert edges[0].weight == 1.0

    assert (result.nodes, result.edges, result.items) == (2, 1, 1)


def test_an_item_with_no_author_produces_no_person(db):
    """A record with nobody attached is still a record."""
    without = item(number=1)
    without["user"] = None

    run(db, [without])

    assert {row.label for row in db.exec(select(EntityNode)).all()} == {"PR"}
    assert db.exec(select(EntityEdge)).all() == []


def test_an_entity_found_in_two_items_is_one_node_with_two_edges(db):
    """Which is the whole reason identifiers are derived from what a thing is."""
    extractor = FixedExtractor([FakeEntity(text="auth-service", type="Service")])

    run(db, [item(number=1), item(number=2)], extractor=extractor)

    found = entity_node_id("Service", "auth-service")
    assert len(db.exec(select(EntityNode).where(EntityNode.node_id == found)).all()) == 1

    mentions = db.exec(
        select(EntityEdge).where(EntityEdge.relation_type == RELATION_MENTIONS)
    ).all()
    assert len(mentions) == 2
    assert {edge.source_id for edge in mentions} == {
        item_node_id("repo_1", "PullRequest", 1),
        item_node_id("repo_1", "PullRequest", 2),
    }
    assert {edge.target_id for edge in mentions} == {found}


def test_a_mention_is_weighted_by_the_extractor_and_rounded(db):
    extractor = FixedExtractor(
        [FakeEntity(text="auth-service", type="Service", score=0.8765432198)]
    )

    run(db, [item(number=1)], extractor=extractor)

    mention = db.exec(
        select(EntityEdge).where(EntityEdge.relation_type == RELATION_MENTIONS)
    ).one()

    assert mention.weight == 0.8765


def test_only_two_relation_types_are_emitted(db):
    """Smaller than the typed set on the other path, and deliberately so.

    Recorded as a test so an artifact built from these rows carrying two
    kinds of edge is a stated property rather than a surprise.
    """
    extractor = FixedExtractor([FakeEntity(text="auth-service", type="Service")])

    run(db, [item(number=1), item(number=2)], extractor=extractor)

    kinds = {edge.relation_type for edge in db.exec(select(EntityEdge)).all()}
    assert kinds == {RELATION_AUTHORED, RELATION_MENTIONS}
    # The literals, not only the constants. The authorship edge runs from the
    # item to the person, so it has to read in that direction — and the other
    # name for it in this project means the opposite direction, which is
    # exactly the confusion worth pinning shut.
    assert kinds == {"AUTHORED_BY", "MENTIONS"}


# ==========================================================================
# accumulation
# ==========================================================================


def test_an_edge_seen_twice_keeps_the_higher_weight(db):
    """Whichever order the two arrive in.

    Last-wins would make the stored weight depend on how the source happened
    to order its results, which is not a property of the relationship.
    """
    extractor = FixedExtractor(
        by_text={
            "first": [FakeEntity(text="auth-service", type="Service", score=0.4)],
            "second": [FakeEntity(text="auth-service", type="Service", score=0.9)],
        }
    )

    run(
        db,
        [item(number=1, title="first"), item(number=1, title="second")],
        extractor=extractor,
    )

    mention = db.exec(
        select(EntityEdge).where(EntityEdge.relation_type == RELATION_MENTIONS)
    ).one()
    assert mention.weight == 0.9


def test_the_higher_weight_wins_when_it_comes_first_too(db):
    """The reverse order, since order is the thing being ruled out."""
    extractor = FixedExtractor(
        by_text={
            "first": [FakeEntity(text="auth-service", type="Service", score=0.9)],
            "second": [FakeEntity(text="auth-service", type="Service", score=0.4)],
        }
    )

    run(
        db,
        [item(number=1, title="first"), item(number=1, title="second")],
        extractor=extractor,
    )

    mention = db.exec(
        select(EntityEdge).where(EntityEdge.relation_type == RELATION_MENTIONS)
    ).one()
    assert mention.weight == 0.9


# ==========================================================================
# the text that is embedded
# ==========================================================================


def test_an_item_is_embedded_from_its_prose_not_its_name(db):
    """Names in a repository are all the same shape.

    Embedding them would put every item at nearly the same point and leave
    the vector arm ranking noise.
    """
    embedder = CountingEmbedder()

    run(
        db,
        [item(number=41, title="the auth change", body="rotates the signing key")],
        embed=embedder,
    )

    embedded = embedder.calls[0]

    assert "the auth change" in embedded
    assert "rotates the signing key" in embedded
    assert embedded != "PullRequest #41"


def test_a_person_and_an_entity_are_embedded_from_their_names(db):
    """There is nothing else to embed for either."""
    embedder = CountingEmbedder()
    extractor = FixedExtractor([FakeEntity(text="auth-service", type="Service")])

    run(db, [item(number=1, author="ada")], extractor=extractor, embed=embedder)

    assert "ada" in embedder.calls
    assert "auth-service" in embedder.calls


# ==========================================================================
# identifiers
# ==========================================================================


def test_a_slug_is_lowercase_and_hyphenated():
    assert slugify("Auth Service") == "auth-service"
    assert slugify("  API   Gateway  ") == "api-gateway"
    assert slugify("v2.1/release") == "v2-1-release"


@pytest.mark.parametrize("name", ["", "   ", "!!!", "...", "///", "→→→"])
def test_a_slug_is_never_empty(name):
    """Two names with nothing to keep must not become one node."""
    assert slugify(name) == "x"
    assert slugify(name)


def test_identifiers_say_what_the_thing_is(db):
    assert item_node_id("repo_1", "PullRequest", 41) == "repo_1:pullrequest:41"
    assert person_node_id("Ada Lovelace") == f"{PERSON_PREFIX}:ada-lovelace"
    assert entity_node_id("Service", "Auth Service") == f"{ENTITY_PREFIX}:service:auth-service"


def test_two_repositories_keep_their_items_apart(db):
    """The identifier leads with the repository, so numbering cannot collide."""
    run(db, [item(number=1)], repo_id="repo_a")
    run(db, [item(number=1)], repo_id="repo_b")

    items = db.exec(
        select(EntityNode).where(EntityNode.label == "PR")
    ).all()
    assert {row.node_id for row in items} == {
        "repo_a:pullrequest:1",
        "repo_b:pullrequest:1",
    }


def test_two_tenants_ingesting_the_same_repository_do_not_collide(db):
    """The rows are scoped by organisation, and the identifiers repeat."""
    run(db, [item(number=1)], org_id="org_a")
    run(db, [item(number=1)], org_id="org_b")

    rows = db.exec(
        select(EntityNode).where(EntityNode.node_id == "repo_1:pullrequest:1")
    ).all()
    assert {row.org_id for row in rows} == {"org_a", "org_b"}


# ==========================================================================
# writing
# ==========================================================================


def test_both_upserts_land_in_one_commit(engine, db):
    """A run that wrote entities and lost relationships would look populated
    and traverse nowhere.

    Asserted as "no commit ever shows nodes without edges", which is the
    property that matters — a schema call commits too, and counting commits
    would be counting that.
    """
    from sqlalchemy import text

    seen: list[tuple[int, int]] = []

    @event.listens_for(engine, "commit")
    def record(connection):
        seen.append(
            (
                connection.execute(text("SELECT count(*) FROM entity_nodes")).scalar(),
                connection.execute(text("SELECT count(*) FROM entity_edges")).scalar(),
            )
        )

    extractor = FixedExtractor([FakeEntity(text="auth-service", type="Service")])
    try:
        run(db, [item(number=1)], extractor=extractor)
    finally:
        event.remove(engine, "commit", record)

    # Nothing was ever committed with entities present and relationships
    # missing, which is what a second commit between the two upserts would
    # have produced.
    assert all(edges > 0 for nodes, edges in seen if nodes > 0)
    # And by the end both are there.
    assert seen[-1] == (3, 2)


def test_the_cursor_moves_to_the_latest_item_seen(db):
    result = run(
        db,
        [
            item(number=1, updated="2026-08-01T10:00:00+00:00"),
            item(number=2, updated="2026-08-03T09:00:00+00:00"),
            item(number=3, updated="2026-08-02T11:00:00+00:00"),
        ],
        cursor="2026-07-01T00:00:00+00:00",
    )

    assert result.cursor == "2026-08-03T09:00:00+00:00"
    assert result.items == 3


def test_ingesting_the_same_item_twice_refreshes_rather_than_duplicating(db):
    """A second pass over an item that changed updates what is stored."""
    run(db, [item(number=1, title="before")])
    run(db, [item(number=1, title="after")])

    rows = db.exec(
        select(EntityNode).where(EntityNode.node_id == "repo_1:pullrequest:1")
    ).all()

    assert len(rows) == 1
    assert rows[0].properties["title"] == "after"


def test_the_schema_guard_is_used_rather_than_the_full_call(db, monkeypatch):
    """Called before every batch, so it must be the cheap question."""
    import src.worker.ingest as ingest_module

    def refuse(engine):
        raise AssertionError("the initialising call was used on the ingest path")

    monkeypatch.setattr(ingest_module, "ensure_graph_store_schema", lambda engine: False)
    monkeypatch.setattr(
        "src.models.graph_store.create_graph_store_schema", refuse, raising=False
    )

    run(db, [item(number=1)])


# ==========================================================================
# the path this one does not replace
# ==========================================================================


def test_the_existing_local_ingest_path_is_untouched():
    """Two paths coexisting is the design, not a migration in progress.

    The older path builds a graph for one process with no database behind it.
    Nothing here imports it, and nothing here is imported by it.
    """
    import ast
    from pathlib import Path

    older = Path(__file__).resolve().parents[1] / "knowledge" / "graph_builder.py"
    source = older.read_text(encoding="utf-8")

    assert "worker" not in source

    tree = ast.parse(Path(__file__).resolve().parents[1].joinpath("worker", "ingest.py").read_text(encoding="utf-8"))
    imported = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any("knowledge" in name for name in imported), imported


def test_every_relation_this_path_writes_has_a_price():
    """A new relation added here without a price would reach the artifact at
    the generic, lowest weight without anyone choosing that."""
    from src.common.config import CONFIDENCE

    assert RELATION_AUTHORED in CONFIDENCE
    assert RELATION_MENTIONS in CONFIDENCE


def test_a_mention_ranks_between_proximity_and_every_structural_fact():
    from src.common.config import CONFIDENCE, RELATION_CO_OCCURS

    structural = [
        price for relation, price in CONFIDENCE.items()
        if relation not in (RELATION_MENTIONS, RELATION_CO_OCCURS)
    ]
    assert CONFIDENCE[RELATION_CO_OCCURS] < CONFIDENCE[RELATION_MENTIONS] < min(structural)


def test_an_item_keeps_its_body_and_when_it_was_written(db):
    raw = item(number=41)
    raw["created_at"] = "2026-07-30T09:00:00+00:00"
    run(db, [raw])

    row = db.exec(
        select(EntityNode).where(EntityNode.node_id == item_node_id("repo_1", "PullRequest", 41))
    ).one()
    assert row.properties["body"] == "body text"
    assert row.properties["created_at"] == "2026-07-30T09:00:00+00:00"


def test_a_fetched_datetime_is_stored_as_text(db):
    from datetime import datetime, timezone

    raw = item(number=42)
    raw["created_at"] = datetime(2026, 7, 30, 9, tzinfo=timezone.utc)
    run(db, [raw])

    row = db.exec(
        select(EntityNode).where(EntityNode.node_id == item_node_id("repo_1", "PullRequest", 42))
    ).one()
    assert row.properties["created_at"] == "2026-07-30T09:00:00+00:00"


def test_a_pull_request_is_labelled_for_its_half_life_and_keeps_its_id(db):
    from src.common.config import HALF_LIFE_DAYS

    run(db, [item(number=7), item(number=8, kind="Issue")])

    labels = {row.node_id: row.label for row in db.exec(select(EntityNode)).all()
              if row.node_id.startswith("repo_1:")}
    assert labels == {"repo_1:pullrequest:7": "PR", "repo_1:issue:8": "Issue"}
    assert HALF_LIFE_DAYS["PR"] == HALF_LIFE_DAYS["PullRequest"] == 60.0


def test_a_pull_request_that_closes_an_issue_resolves_it(db):
    from src.common.relations import RELATION_RESOLVES

    run(db, [item(number=50, body="Speeds up escape. Fixes #433, closes #12.")])

    resolves = {
        (row.source_id, row.target_id, row.weight)
        for row in db.exec(select(EntityEdge).where(EntityEdge.relation_type == RELATION_RESOLVES)).all()
    }
    source = item_node_id("repo_1", "PullRequest", 50)
    assert resolves == {
        (source, item_node_id("repo_1", "Issue", "12"), 1.0),
        (source, item_node_id("repo_1", "Issue", "433"), 1.0),
    }


def test_a_bare_number_or_an_issue_body_resolves_nothing(db):
    from src.common.relations import RELATION_RESOLVES

    run(db, [
        item(number=51, body="Related to #433, see #12."),
        item(number=52, kind="Issue", body="Fixes #9 upstream."),
    ])

    assert db.exec(select(EntityEdge).where(EntityEdge.relation_type == RELATION_RESOLVES)).all() == []


def test_resolves_is_priced():
    from src.common.config import CONFIDENCE
    from src.common.relations import RELATION_RESOLVES

    assert CONFIDENCE[RELATION_RESOLVES] == 0.92


# ==========================================================================
# "#N" references
# ==========================================================================


def mentions(db):
    return {
        (row.source_id, row.target_id)
        for row in db.exec(select(EntityEdge).where(EntityEdge.relation_type == RELATION_MENTIONS)).all()
    }


def test_a_number_naming_an_item_in_this_repo_points_at_that_item(db):
    extractor = FixedExtractor(by_text={"see #7": [FakeEntity(text="#7", type="Ticket")]})

    run(db, [item(number=7, kind="Issue", body="the bug"), item(number=8, body="see #7")],
        extractor=extractor)

    pr, issue = item_node_id("repo_1", "PullRequest", 8), item_node_id("repo_1", "Issue", 7)
    assert mentions(db) == {(pr, issue)}
    assert not [row for row in db.exec(select(EntityNode)).all() if row.label == "Ticket"]


def test_a_number_resolves_to_an_item_read_in_an_earlier_run(db):
    run(db, [item(number=7, kind="Issue", body="the bug")])
    extractor = FixedExtractor([FakeEntity(text="#7", type="Ticket")])

    run(db, [item(number=9, body="follow-up to #7")], extractor=extractor)

    assert (item_node_id("repo_1", "PullRequest", 9), item_node_id("repo_1", "Issue", 7)) in mentions(db)


def test_a_number_naming_nothing_here_is_dropped(db):
    extractor = FixedExtractor([FakeEntity(text="#10007", type="Ticket")])

    run(db, [item(number=8, body="bumps pip, see #10007")], extractor=extractor)

    assert mentions(db) == set()
    assert not [row for row in db.exec(select(EntityNode)).all() if row.label == "Ticket"]


def test_rereading_an_item_replaces_what_its_text_said(db):
    first = FixedExtractor([FakeEntity(text="auth-service", type="Service")])
    run(db, [item(number=8, body="touches auth-service")], extractor=first)

    run(db, [item(number=8, body="nothing named now")], extractor=FixedExtractor([]))

    assert mentions(db) == set()
    # The service was named only there, so it leaves the graph with the mention.
    assert entity_node_id("Service", "auth-service") not in {
        row.node_id for row in db.exec(select(EntityNode)).all()
    }
    # The authorship is not text-derived and stays.
    assert db.exec(select(EntityEdge).where(EntityEdge.relation_type == RELATION_AUTHORED)).all()


def test_an_entity_still_named_elsewhere_is_kept(db):
    named = FixedExtractor([FakeEntity(text="auth-service", type="Service")])
    run(db, [item(number=8), item(number=9)], extractor=named)

    run(db, [item(number=8, body="nothing named now")], extractor=FixedExtractor([]))

    assert entity_node_id("Service", "auth-service") in {
        row.node_id for row in db.exec(select(EntityNode)).all()
    }


def test_a_number_read_before_its_item_is_linked_by_the_deferred_pass(db, caplog):
    """The pull request comes first in the pass and names #7; issue #7 comes
    after it. When #7 is found nothing is known by that number, so it waits,
    and the retry at the end of the pass links it."""
    extractor = FixedExtractor(by_text={"fixes the crash in #7": [FakeEntity(text="#7", type="Ticket")]})

    with caplog.at_level("INFO", logger="graphrag.worker.ingest"):
        run(db, [item(number=8, body="fixes the crash in #7"),
                 item(number=7, kind="Issue", body="it crashes")],
            extractor=extractor)

    # Found before its item, so not linked on sight; linked by the retry.
    assert "1 reference(s) deferred, 1 linked on retry, 0 dropped" in caplog.text

    pr, issue = item_node_id("repo_1", "PullRequest", 8), item_node_id("repo_1", "Issue", 7)
    assert (pr, issue) in mentions(db)
    # No stand-in node was written for the reference while it waited.
    assert not [row for row in db.exec(select(EntityNode)).all() if row.label == "Ticket"]
    assert not [row for row in db.exec(select(EntityNode)).all()
                if row.node_id.startswith("entity:ticket:")]


def test_a_number_still_unknown_after_the_retry_is_dropped(db, caplog):
    extractor = FixedExtractor([FakeEntity(text="#10007", type="Ticket")])

    with caplog.at_level("INFO", logger="graphrag.worker.ingest"):
        run(db, [item(number=8, body="bumps pip, see #10007")], extractor=extractor)

    assert "1 reference(s) deferred, 0 linked on retry, 1 dropped" in caplog.text
    assert mentions(db) == set()


def test_a_number_already_known_is_linked_on_sight(db, caplog):
    run(db, [item(number=7, kind="Issue", body="the bug")])
    extractor = FixedExtractor([FakeEntity(text="#7", type="Ticket")])

    with caplog.at_level("INFO", logger="graphrag.worker.ingest"):
        run(db, [item(number=9, body="follow-up to #7")], extractor=extractor)

    assert "deferred" not in caplog.text
    assert (item_node_id("repo_1", "PullRequest", 9), item_node_id("repo_1", "Issue", 7)) in mentions(db)


# ==========================================================================
# which repository a number belongs to
# ==========================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        ("see #203", ["203"]),
        ("see acme/widgets#203", ["203"]),
        ("see https://github.com/acme/widgets/issues/203", ["203"]),
        ("see https://github.com/acme/widgets/pull/203", ["203"]),
        ("see ACME/Widgets#203", ["203"]),
        ("see other/repo#203", []),
        ("see https://github.com/other/repo/issues/203", []),
        ("see https://github.com/other/repo/pull/203/files", []),
        ('bump <a href="https://github-redirect.dependabot.com/pypa/wheel/issues/480">#480</a>', []),
        ('see <a href="https://github.com/acme/widgets/issues/7">#7</a>', ["7"]),
        ('notes <a href="https://example.com/changelog">#12</a>', []),
        ("other/repo#5 and #6", ["6"]),
        ("#6 then #6 again", ["6"]),
        ("anchor foo#3 or &#39; entity", []),
    ],
)
def test_only_this_repositorys_numbers_are_read(text, expected):
    from src.worker.ingest import numbered_references

    assert numbered_references(text, "acme/widgets") == expected


def test_a_reference_to_another_repository_links_nothing_here(db):
    run(db, [
        item(number=203, kind="Issue", body="ours"),
        item(number=8, body="bumps pip, see other/repo#203"),
    ], repo_name="acme/widgets")

    assert mentions(db) == set()


def test_a_qualified_reference_to_this_repository_links(db):
    run(db, [
        item(number=203, kind="Issue", body="ours"),
        item(number=8, body="see https://github.com/acme/widgets/issues/203"),
    ], repo_name="acme/widgets")

    assert (item_node_id("repo_1", "PullRequest", 8), item_node_id("repo_1", "Issue", 203)) in mentions(db)


# ==========================================================================
# no GitHub token: the demonstration batch
# ==========================================================================


def test_with_no_token_the_first_read_is_the_demonstration_batch(monkeypatch):
    from src.worker.ingest import DEMO_ITEMS, _fetch_delta

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GRAPHRAG_GITHUB_DEMO", raising=False)

    items = list(_fetch_delta("acme/widgets", None, None))

    assert [item["number"] for item in items] == [item["number"] for item in DEMO_ITEMS]


def test_with_no_token_a_later_read_finds_nothing_new(monkeypatch):
    from src.worker.ingest import _fetch_delta

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GRAPHRAG_GITHUB_DEMO", raising=False)

    assert list(_fetch_delta("acme/widgets", "2026-07-05T09:15:00+00:00", None)) == []


def test_the_demonstration_batch_can_be_forced_with_a_token_present(monkeypatch):
    from src.worker.ingest import _fetch_delta

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GRAPHRAG_GITHUB_DEMO", "1")

    assert len(list(_fetch_delta("acme/widgets", None, None))) == 3


def test_the_demonstration_batch_builds_every_kind_of_edge(db, monkeypatch):
    from src.common.relations import RELATION_RESOLVES
    from src.worker.ingest import DEMO_ITEMS

    run(db, [dict(item) for item in DEMO_ITEMS], repo_name="example/demo")

    kinds = {row.relation_type for row in db.exec(select(EntityEdge)).all()}
    assert {RELATION_AUTHORED, RELATION_MENTIONS, RELATION_RESOLVES} <= kinds
