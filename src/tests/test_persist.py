"""Tests for writing a built graph into the store.

Drives a real database, like the rest of the store tests: the behaviours that
matter here are MERGE semantics and what actually lands in a table, and a fake
would assert what was imagined instead.

The demo corpus is used where a realistic shape matters and hand-built nodes
where an exact count matters. Both appear, and which one a test uses is
deliberate rather than incidental.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

ladybug = pytest.importorskip(
    "ladybug", reason="the graph store needs ladybug, which runs under WSL"
)

from src.common.config import (  # noqa: E402
    DOC_TABLE,
    EMBEDDING_DIMENSION,
    MENTIONS_TABLE,
    NODE_PR,
    NODE_TABLE,
    NODE_TICKET,
    REL_TABLE,
    RELATION_AUTHORED,
)
from src.analysis import Extractor  # noqa: E402
from src.graphdb import open_context_graph  # noqa: E402
from src.knowledge.persist import (  # noqa: E402
    document_id,
    entity_id,
    node_path,
    node_text,
    persist,
)

WHEN = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    graph = open_context_graph(tmp_path / "graph")
    yield graph
    graph.close()


class FakeBuilder:
    """Exactly the two attributes ``persist`` reads."""

    def __init__(self, nodes, edges=()):
        self.nodes = {node["id"]: node for node in nodes}
        self.edges = list(edges)


class CountingEmbedder:
    """A deterministic vector, and a record of what was asked for."""

    def __init__(self):
        self.seen = []

    def vector(self, text):
        self.seen.append(text)
        return [float(len(text) % 7)] * EMBEDDING_DIMENSION


def demo_builder():
    from examples.offline_demo import build_graph

    return build_graph()


def edge_rows(store):
    return store.query(
        f"MATCH (a:{NODE_TABLE})-[r:{REL_TABLE}]->(b:{NODE_TABLE}) "
        f"RETURN a.id, b.id, r.relation ORDER BY a.id, b.id"
    )


def mention_rows(store):
    return store.query(
        f"MATCH (d:{DOC_TABLE})-[:{MENTIONS_TABLE}]->(e:{NODE_TABLE}) "
        f"RETURN d.id, e.id ORDER BY d.id, e.id"
    )


# --------------------------------------------------------------------------
# the text a node contributes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node, expected",
    [
        ({"id": "pr:1", "title": "Fix the thing"}, "Fix the thing"),
        ({"id": "commit:a", "message": "Tighten expiry"}, "Tighten expiry"),
        ({"id": "repo:o/r", "full_name": "o/r"}, "o/r"),
        ({"id": "file:a.py", "path": "a.py"}, "a.py"),
        ({"id": "person:alice", "login": "alice"}, "alice"),
    ],
)
def test_a_node_contributes_its_identifying_text(node, expected):
    assert node_text(node) == expected


def test_a_node_with_no_text_field_falls_back_to_its_id():
    """Searchable rather than silently absent."""
    assert node_text({"id": "odd:1"}) == "odd:1"


def test_title_wins_over_a_later_field():
    """The order in TEXT_FIELDS is the specificity order, not an accident."""
    node = {"id": "pr:1", "title": "Fix the thing", "path": "somewhere.py"}

    assert node_text(node) == "Fix the thing"


def test_a_document_path_prefers_the_url():
    """The url locates the thing outside this database; the id only inside it."""
    assert node_path({"id": "pr:1", "url": "https://example.test/1"}) == (
        "https://example.test/1"
    )
    assert node_path({"id": "pr:1"}) == "pr:1"


def test_document_ids_cannot_collide_with_node_ids():
    assert document_id("pr:1") == "doc:pr:1"
    assert document_id("pr:1") != "pr:1"


# --------------------------------------------------------------------------
# extracted references land on the structural node
# --------------------------------------------------------------------------


class Entity:
    def __init__(self, text, type_):
        self.text = text
        self.type = type_


@pytest.mark.parametrize(
    "surface, kind, expected",
    [
        ("#12", NODE_TICKET, "ticket:12"),
        ("issue #12", NODE_TICKET, "ticket:12"),
        ("PR #101", NODE_PR, "pr:101"),
        ("pull request #101", NODE_PR, "pr:101"),
    ],
)
def test_a_reference_maps_onto_the_structural_id(surface, kind, expected):
    """The number is the identity, so both spellings reach one node."""
    assert entity_id(Entity(surface, kind)) == expected


def test_a_service_name_is_keyed_on_its_lowercased_form():
    """Case is not identity: an env var and prose name one service."""
    assert entity_id(Entity("ORDER_SERVICE", "Service")) == entity_id(
        Entity("order_service", "Service")
    )


# --------------------------------------------------------------------------
# what reaches the store
# --------------------------------------------------------------------------


def test_every_node_and_edge_reaches_the_store(store):
    builder = demo_builder()

    stats = persist(store, builder)

    assert stats.entities == len(builder.nodes)
    assert stats.relationships == len(builder.edges)
    assert store.count_nodes() == len(builder.nodes)
    assert len(edge_rows(store)) == len(builder.edges)


def test_per_relation_counts_match_the_build(store):
    """The JSON path and the store path must agree relation by relation.

    A total that matches while one relation is over-counted and another under
    would pass a count-only check and mean the graph was rewired.
    """
    from collections import Counter

    builder = demo_builder()
    persist(store, builder)

    expected = Counter(edge["type"] for edge in builder.edges)
    rows = store.query(
        f"MATCH (a:{NODE_TABLE})-[r:{REL_TABLE}]->(b:{NODE_TABLE}) RETURN r.relation"
    )

    assert Counter(row[0] for row in rows) == expected


def test_nodes_are_written_before_edges(store):
    """An edge whose endpoints are absent matches nothing and is lost silently.

    Asserted by writing a build whose only edge joins two nodes that appear
    later in iteration order than the edge does in the edge list.
    """
    builder = FakeBuilder(
        nodes=[
            {"id": "a", "type": "Person", "login": "alice", "timestamp": WHEN},
            {"id": "b", "type": "PR", "title": "something", "timestamp": WHEN},
        ],
        edges=[
            {
                "source": "a",
                "target": "b",
                "type": RELATION_AUTHORED,
                "confidence": 0.95,
                "timestamp": WHEN,
            }
        ],
    )

    persist(store, builder)

    assert edge_rows(store) == [["a", "b", RELATION_AUTHORED]]


def test_persisting_twice_leaves_the_same_counts(store):
    """MERGE throughout, so a re-ingest is not a duplication."""
    builder = demo_builder()

    persist(store, builder)
    first = (store.count_nodes(), len(edge_rows(store)))
    persist(store, builder)

    assert (store.count_nodes(), len(edge_rows(store))) == first


def test_timestamps_survive_the_write(store):
    builder = FakeBuilder(
        nodes=[{"id": "a", "type": "PR", "title": "t", "timestamp": WHEN}]
    )

    persist(store, builder)

    assert store.get_entity("a")["timestamp"] == WHEN


def test_a_node_without_a_timestamp_stays_absent(store):
    builder = FakeBuilder(nodes=[{"id": "a", "type": "PR", "title": "t"}])

    persist(store, builder)

    assert store.get_entity("a")["timestamp"] is None


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------


def test_no_embedder_means_no_vectors(store):
    builder = demo_builder()

    stats = persist(store, builder)

    assert stats.embedded == 0
    assert store.get_embedding(next(iter(builder.nodes))) is None


def test_every_entity_is_embedded_when_an_embedder_is_given(store):
    builder = demo_builder()
    embedder = CountingEmbedder()

    stats = persist(store, builder, embedder=embedder)

    assert stats.embedded == len(builder.nodes)
    for node_id in builder.nodes:
        assert store.get_embedding(node_id) is not None


def test_the_embedder_is_asked_for_the_node_text_not_the_id(store):
    """A vector of an id would make every search a search over id strings."""
    builder = FakeBuilder(
        nodes=[{"id": "pr:1", "type": "PR", "title": "Reject expired tokens"}]
    )
    embedder = CountingEmbedder()

    persist(store, builder, embedder=embedder)

    assert embedder.seen == ["Reject expired tokens"]


def test_persist_does_not_build_the_index():
    """Indexing is one pass over the rows, so it happens once, after writing.

    Asserted by reading the source rather than by behaviour, and the reason is
    worth recording: there is no observable difference to assert. Opening a
    store creates the index, and writes after a build are searchable without a
    rebuild, so a search succeeds either way. The separation exists so a caller
    writing in several batches pays for one index pass rather than one per
    batch, and nothing about a single-batch run can show that.

    A source scan is the honest instrument here. Behaviour cannot distinguish
    the two arrangements, so a behavioural test would be asserting something
    it is not actually measuring.
    """
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "knowledge" / "persist.py"
    ).read_text(encoding="utf-8")

    called = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "build_vector_index" not in called
    # The writes it does make, so the scan is known to be looking at the right
    # thing rather than passing because it found nothing at all.
    assert {"upsert_entity", "upsert_relationship", "upsert_document"} <= called


def test_search_works_once_the_index_is_built(store):
    builder = demo_builder()
    persist(store, builder, embedder=CountingEmbedder())

    store.build_vector_index(rebuild=True)

    assert store.vector_search([0.0] * EMBEDDING_DIMENSION, k=3)

# --------------------------------------------------------------------------
# documents and mentions
#
# Documents no longer come from nodes. A node carries a thing's identity and
# not the prose it arrived with, so documents are built from the payloads by
# knowledge.documents and handed to persist. These tests drive that shape.
# --------------------------------------------------------------------------


def document(doc_id, content, *, path="https://example.test/1", origin="pr:1",
             field="pull_request_body"):
    from src.knowledge.documents import SourceDocument

    return SourceDocument(
        id=doc_id, path=path, content=content, origin=origin, field=field
    )


def test_no_documents_means_no_document_rows(store):
    builder = demo_builder()

    stats = persist(store, builder, extractor=Extractor("none"))

    assert stats.documents == 0
    assert store.count_documents() == 0


def test_a_document_is_stored_with_its_content_and_path(store):
    builder = FakeBuilder(nodes=[{"id": "pr:1", "type": "PR", "title": "t"}])

    persist(
        store,
        builder,
        documents=[document("doc:pr:1", "Adds retries. Fixes #13.")],
    )

    stored = store.get_document("doc:pr:1")
    assert stored["content"] == "Adds retries. Fixes #13."
    assert stored["path"] == "https://example.test/1"


def test_documents_are_stored_even_without_an_extractor(store):
    """The prose is the source record; mentions are an interpretation of it.

    A caller that wants the text kept without paying for extraction gets that,
    so the two costs are separable.
    """
    builder = FakeBuilder(nodes=[{"id": "pr:1", "type": "PR", "title": "t"}])

    stats = persist(store, builder, documents=[document("doc:pr:1", "Fixes #13.")])

    assert stats.documents == 1
    assert stats.mentions == 0
    assert store.get_document("doc:pr:1")["content"] == "Fixes #13."


def test_a_reference_in_a_body_produces_a_mention(store):
    """The gap this closes: the reference lives in the body, not the title."""
    builder = FakeBuilder(
        nodes=[
            {"id": "pr:1", "type": "PR", "title": "Retry the payment webhook"},
            {"id": "ticket:13", "type": "Ticket", "title": "Webhook retries flood"},
        ]
    )

    stats = persist(
        store,
        builder,
        extractor=Extractor("none"),
        documents=[document("doc:pr:1", "Adds three retries with backoff. Fixes #13.")],
    )

    assert stats.mentions == 1
    assert ["doc:pr:1", "ticket:13"] in mention_rows(store)


def test_a_mention_reaches_the_node_the_structural_pass_made(store):
    """``#13`` points at the ticket already ingested, not a second one."""
    builder = FakeBuilder(
        nodes=[
            {"id": "pr:1", "type": "PR", "title": "Retry"},
            {"id": "ticket:13", "type": "Ticket", "title": "Webhook retries flood"},
        ]
    )

    stats = persist(
        store,
        builder,
        extractor=Extractor("none"),
        documents=[document("doc:pr:1", "Fixes #13.")],
    )

    assert stats.mentions_to_existing == 1
    assert stats.mentions_to_new == 0
    assert stats.entities_from_text == 0
    assert store.count_nodes() == 2
    assert store.get_entity("ticket:13")["label"] == "Webhook retries flood"


def test_an_entity_with_no_structural_node_gets_one(store):
    """A service name has nothing to point at until it is written."""
    builder = FakeBuilder(nodes=[{"id": "commit:a", "type": "Commit", "message": "m"}])

    stats = persist(
        store,
        builder,
        extractor=Extractor("none"),
        documents=[document("doc:commit:a", "Retry the payment_service webhook")],
    )

    assert stats.entities_from_text == 1
    assert stats.mentions_to_new == 1
    assert store.entity_exists("service:payment_service")


def test_entities_are_counted_by_the_field_they_came_from(store):
    """A total cannot say whether review bodies were worth reading."""
    builder = FakeBuilder(nodes=[{"id": "pr:1", "type": "PR", "title": "t"}])

    stats = persist(
        store,
        builder,
        extractor=Extractor("none"),
        documents=[
            document("doc:pr:1", "Fixes #13.", field="pull_request_body"),
            document("doc:review:1", "same bug as #88", field="review_body"),
        ],
    )

    assert stats.entities_by_field == {"pull_request_body": 1, "review_body": 1}


def test_a_document_with_no_entities_is_counted_separately(store):
    builder = FakeBuilder(nodes=[{"id": "pr:1", "type": "PR", "title": "t"}])

    stats = persist(
        store,
        builder,
        extractor=Extractor("none"),
        documents=[document("doc:pr:1", "Documentation only, no behaviour change.")],
    )

    assert stats.documents == 1
    assert stats.documents_without_entities == 1
    assert stats.mentions == 0


def test_mentions_do_not_multiply_across_runs(store):
    builder = FakeBuilder(nodes=[{"id": "pr:1", "type": "PR", "title": "t"}])
    docs = [document("doc:pr:1", "Fixes #13.")]

    persist(store, builder, extractor=Extractor("none"), documents=docs)
    first = len(mention_rows(store))
    persist(store, builder, extractor=Extractor("none"), documents=docs)

    assert len(mention_rows(store)) == first


def test_a_reference_can_be_both_a_resolves_edge_and_a_mention(store):
    """Expected, and not duplication. The two answer different questions.

    The closing-keyword scan produces a RESOLVES edge, which asserts that this
    pull request closes that ticket — a structural claim. A mention records
    that the ticket was named in this text — a provenance claim. One can be
    true without the other: prose naming a ticket it does not close produces a
    mention and no edge.
    """
    from src.common.config import RELATION_RESOLVES

    builder = demo_builder()

    persist(
        store,
        builder,
        extractor=Extractor("none"),
        documents=[
            document("doc:pr:101", "Fixes #12.", origin="pr:101")
        ],
    )

    resolves = [row for row in edge_rows(store) if row[2] == RELATION_RESOLVES]

    assert ["pr:101", "ticket:12", RELATION_RESOLVES] in resolves
    assert ["doc:pr:101", "ticket:12"] in mention_rows(store)


# --------------------------------------------------------------------------
# mentions after chunking
# --------------------------------------------------------------------------


def test_a_mention_from_a_chunked_body_reaches_a_stored_record():
    """A reference deep in a long body still has a document to originate from.

    Built through ``source_documents`` rather than by hand, because the thing
    at risk is the id: mentions point at whatever that function produced, and
    a chunk id that did not match a stored row would drop the mention with no
    error anywhere.
    """
    from src.graphdb import open_context_graph
    from src.ingestion.models import PullRequest
    from src.knowledge.documents import source_documents
    from src.common.config import DOCUMENT_CHUNK_CHARACTERS
    import tempfile
    from pathlib import Path as _Path

    # The reference sits past the first chunk, so it can only be found in a
    # later one — which is the case a single whole-body record never tested.
    filler = "x" * (DOCUMENT_CHUNK_CHARACTERS + 40)
    body = f"{filler} and this finally fixes #13."
    pull_request = PullRequest.model_validate(
        {
            "id": 1, "number": 1, "title": "Retry the payment webhook",
            "state": "open", "draft": False, "body": body,
            "html_url": "https://example.invalid/pull/1",
            "user": {"login": "alice", "id": 1},
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )
    documents = source_documents(pull_requests=[pull_request])
    assert len(documents) > 1, "the body must actually split for this to test anything"

    builder = FakeBuilder(
        nodes=[
            {"id": "pr:1", "type": "PR", "title": "Retry the payment webhook"},
            {"id": "ticket:13", "type": "Ticket", "title": "Webhook retries flood"},
        ]
    )

    with tempfile.TemporaryDirectory() as directory:
        store = open_context_graph(_Path(directory) / "graph")
        try:
            stats = persist(
                store, builder, extractor=Extractor("none"), documents=documents
            )
            rows = mention_rows(store)
            stored = {
                row[0]
                for row in store.query(f"MATCH (d:{DOC_TABLE}) RETURN d.id")
            }
        finally:
            store.close()

    assert stats.mentions >= 1
    mentioning = [row for row in rows if row[1] == "ticket:13"]
    assert mentioning, "the reference in a later chunk produced no mention"

    # Every mention originates from a document that was actually written.
    for document_id, _entity_id in rows:
        assert document_id in stored
        assert document_id.startswith("doc:pr:1:")


def test_the_entity_count_includes_entities_created_from_text(store):
    """The count reported is what reached the store, not one pass of it.

    The mention pass upserts entities the structural pass never produced. They
    are in the store, so a count that omits them tells the operator a number
    lower than what was written.
    """
    builder = FakeBuilder(nodes=[{"id": "commit:a", "type": "Commit", "message": "m"}])

    stats = persist(
        store,
        builder,
        extractor=Extractor("none"),
        documents=[document("doc:commit:a", "Retry the payment_service webhook")],
    )

    assert stats.entities_from_text == 1
    assert stats.entities == store.count_nodes()


def test_entities_from_text_remains_a_subset_of_the_total(store):
    """The breakdown still says how many of the total came from prose."""
    builder = FakeBuilder(nodes=[{"id": "commit:a", "type": "Commit", "message": "m"}])

    stats = persist(
        store,
        builder,
        extractor=Extractor("none"),
        documents=[document("doc:commit:a", "Retry the payment_service webhook")],
    )

    assert stats.entities == 2
    assert stats.entities_from_text == 1


def test_a_commit_is_labelled_by_its_subject_line():
    from src.knowledge.persist import node_label

    node = {"id": "commit:abc", "message": "Fix the auth race\n\nLong explanation " + "x" * 5000}
    assert node_label(node) == "Fix the auth race"


def test_a_single_line_wall_of_text_is_capped():
    from src.knowledge.persist import LABEL_MAX_CHARS, node_label

    label = node_label({"id": "commit:abc", "message": "y" * 5000})
    assert len(label) == LABEL_MAX_CHARS and label.endswith("…")


def test_a_node_with_only_blank_text_falls_back_to_its_id():
    from src.knowledge.persist import node_label

    assert node_label({"id": "commit:abc", "message": "  \n  "}) == "commit:abc"


# --------------------------------------------------------------------------
# Variants folded as the text is read
# --------------------------------------------------------------------------


class NamedEntities:
    """Returns the entities named for each document's text."""

    def __init__(self, by_text):
        self.by_text = by_text

    def extract(self, text):
        from src.analysis import Entity

        return [
            Entity(text=name, type=kind, score=1.0, start=0, end=len(name), source="rules")
            for name, kind in self.by_text.get(text, [])
        ]


class FixedSimilarity:
    """A fixed score per pair of surface forms, 0 for any other pair."""

    def __init__(self, pairs):
        self.pairs = {frozenset(pair): score for pair, score in pairs.items()}

    def scores(self, text, candidates):
        return [self.pairs.get(frozenset((text, other)), 0.0) for other in candidates]


class FixedJudge:
    def __init__(self, answer):
        self.answer = answer
        self.asked = []

    def same(self, left, right):
        self.asked.append((left, right))
        return self.answer


def _variants_build(store, similarity_score, judge=None):
    from src.analysis import Resolver

    builder = FakeBuilder(nodes=[{"id": "pr:1", "type": "PR", "title": "t"}])
    extractor = NamedEntities({
        "one": [("payment_service", "Service")],
        "two": [("PaymentService", "Service")],
    })
    resolver = Resolver(
        FixedSimilarity({("payment_service", "PaymentService"): similarity_score}), judge
    )
    persist(
        store, builder, extractor=extractor, resolver=resolver,
        documents=[document("doc:a", "one"), document("doc:b", "two")],
    )
    return resolver


def test_a_near_identical_variant_points_at_the_first_spelling(store):
    resolver = _variants_build(store, 0.95)

    assert resolver.stats.fast_merges == 1
    assert ("doc:b", "service:payment_service") in [tuple(row) for row in mention_rows(store)]
    assert not store.entity_exists("service:paymentservice")


def test_a_close_call_is_merged_when_the_judge_says_so(store):
    judge = FixedJudge(answer=True)
    resolver = _variants_build(store, 0.88, judge)

    assert judge.asked == [("payment_service", "PaymentService")]
    assert resolver.stats.model_merges == 1
    assert not store.entity_exists("service:paymentservice")


def test_a_close_call_stays_apart_without_a_judge(store):
    resolver = _variants_build(store, 0.88, judge=None)

    assert resolver.stats.variant_merges == 0
    assert store.entity_exists("service:paymentservice")


def test_without_a_resolver_every_spelling_is_its_own_entity(store):
    builder = FakeBuilder(nodes=[{"id": "pr:1", "type": "PR", "title": "t"}])
    extractor = NamedEntities({
        "one": [("payment_service", "Service")],
        "two": [("PaymentService", "Service")],
    })

    persist(store, builder, extractor=extractor,
            documents=[document("doc:a", "one"), document("doc:b", "two")])

    assert store.entity_exists("service:payment_service")
    assert store.entity_exists("service:paymentservice")
