"""Tests for turning payloads into source documents.

No database here. This step is a pure transformation from validated models to
plain records, so it is tested as one — the store's behaviour is asserted in
``test_persist.py``, where it belongs.

The fixtures are the same ones the demo corpus uses, validated through the
same models, so a change to either shows up here rather than only at runtime.
"""

from __future__ import annotations

import pytest

from src.ingestion.models import Commit, Issue, PullRequest, Review
from src.knowledge.documents import (
    FIELD_COMMIT_MESSAGE,
    FIELD_ISSUE_BODY,
    FIELD_PULL_REQUEST_BODY,
    FIELD_REVIEW_BODY,
    source_documents,
)
from examples import fixtures


def demo_payloads():
    return {
        "pull_requests": [
            PullRequest.model_validate(p) for p in fixtures.PULL_REQUESTS
        ],
        "issues": [
            Issue.model_validate(i)
            for i in fixtures.ISSUES
            if "pull_request" not in i
        ],
        "commits": [Commit.model_validate(c) for c in fixtures.COMMITS],
        "reviews": {
            number: [Review.model_validate(r) for r in reviews]
            for number, reviews in fixtures.REVIEWS.items()
        },
    }


# --------------------------------------------------------------------------
# what becomes a document
# --------------------------------------------------------------------------


def test_a_pull_request_body_becomes_a_document():
    documents = source_documents(**demo_payloads())
    by_id = {document.id: document for document in documents}

    assert by_id["doc:pr:101:0"].content == (
        "The expiry comparison was strict, so a token was rejected a second "
        "early. Fixes #12."
    )
    assert by_id["doc:pr:101:0"].field == FIELD_PULL_REQUEST_BODY
    assert by_id["doc:pr:101:0"].origin == "pr:101"


def test_a_commit_message_becomes_a_document():
    """Stored despite the node already carrying the message in full.

    The duplication is deliberate: a mention needs a document to originate
    from, so without this a reference written in a commit message could not
    produce one at all.
    """
    documents = source_documents(**demo_payloads())
    by_id = {document.id: document for document in documents}

    assert by_id["doc:commit:a1b2c3d:0"].content == (
        "Tighten the authentication token expiry comparison"
    )
    assert by_id["doc:commit:a1b2c3d:0"].field == FIELD_COMMIT_MESSAGE


def test_a_document_path_locates_the_thing_outside_this_database():
    documents = source_documents(**demo_payloads())
    by_id = {document.id: document for document in documents}

    assert by_id["doc:pr:101:0"].path == "https://github.com/acme/checkout/pull/101"


def test_an_absent_body_produces_no_document():
    """The demo issues carry no body, so no issue document exists."""
    documents = source_documents(**demo_payloads())

    assert not [d for d in documents if d.field == FIELD_ISSUE_BODY]


@pytest.mark.parametrize("body", [None, "", "   ", "\n\t "])
def test_blank_bodies_are_not_stored(body):
    """A row with no content carries a path, costs space and yields nothing."""
    payload = dict(fixtures.PULL_REQUESTS[0], body=body)

    documents = source_documents(
        pull_requests=[PullRequest.model_validate(payload)]
    )

    assert documents == []


def test_an_issue_body_becomes_a_document_when_there_is_one():
    payload = dict(fixtures.ISSUES[0], body="Same root cause as #99.")

    documents = source_documents(issues=[Issue.model_validate(payload)])

    assert len(documents) == 1
    assert documents[0].field == FIELD_ISSUE_BODY
    assert documents[0].id == "doc:ticket:12:0"


def test_a_review_body_becomes_a_document():
    """Review bodies are fetched and validated; until now nothing read them."""
    payload = dict(fixtures.REVIEWS[101][0], body="same bug as #88")

    documents = source_documents(
        reviews={101: [Review.model_validate(payload)]}
    )

    assert len(documents) == 1
    assert documents[0].field == FIELD_REVIEW_BODY
    assert documents[0].content == "same bug as #88"
    assert documents[0].origin == "pr:101"


def test_reviews_on_one_pull_request_do_not_overwrite_each_other():
    """Keyed on the review's own id: one pull request carries many reviews."""
    first = dict(fixtures.REVIEWS[101][0], id=1, body="looks wrong")
    second = dict(fixtures.REVIEWS[101][0], id=2, body="see #88")

    documents = source_documents(
        reviews={101: [Review.model_validate(first), Review.model_validate(second)]}
    )

    assert {document.id for document in documents} == {
        "doc:review:1:0",
        "doc:review:2:0",
    }


# --------------------------------------------------------------------------
# stability
# --------------------------------------------------------------------------


def test_the_same_payloads_produce_the_same_documents():
    """Two runs over one input are identical, so a diff means a change."""
    assert source_documents(**demo_payloads()) == source_documents(**demo_payloads())


def test_document_ids_are_unique():
    documents = source_documents(**demo_payloads())

    assert len({document.id for document in documents}) == len(documents)


def test_no_payloads_produce_no_documents():
    assert source_documents() == []


# --------------------------------------------------------------------------
# the gap this closes, measured on the fixtures
# --------------------------------------------------------------------------


def test_the_references_in_this_corpus_live_in_bodies_not_titles():
    """The measurement that motivated building documents at all.

    Extraction over node text found nothing, because every reference-shaped
    entity in these fixtures sits in a pull request body while nodes carry
    titles. Pinned so that stops being an assumption.

    A failure means the fixtures gained a reference in a title, which is fine
    but changes what the corpus demonstrates.
    """
    from src.analysis import Extractor

    extractor = Extractor("none")
    references = {"Ticket", "PR"}

    in_titles = [
        entity
        for payload in fixtures.PULL_REQUESTS
        for entity in extractor.extract(payload["title"])
        if entity.type in references
    ]
    in_bodies = [
        entity
        for payload in fixtures.PULL_REQUESTS
        for entity in extractor.extract(payload["body"] or "")
        if entity.type in references
    ]

    assert in_titles == []
    assert [entity.text for entity in in_bodies] == ["#12", "#13"]


# --------------------------------------------------------------------------
# one record per chunk, not per body
# --------------------------------------------------------------------------

from src.common.config import (  # noqa: E402
    DOCUMENT_CHUNK_CHARACTERS,
    DOCUMENT_CHUNK_OVERLAP_CHARACTERS,
)


def pull_request(number: int, body: str) -> PullRequest:
    """A payload carrying a chosen body, valid in every other respect."""
    return PullRequest.model_validate(
        {
            "id": 90_000 + number,
            "number": number,
            "title": "a title",
            "state": "open",
            "draft": False,
            "body": body,
            "html_url": f"https://example.invalid/pull/{number}",
            "user": {"login": "alice", "id": 1},
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )


def test_a_body_shorter_than_one_chunk_stays_one_record():
    """Most bodies are short. Chunking must cost them nothing."""
    body = "The expiry comparison was strict, so a token was rejected early."
    documents = source_documents(pull_requests=[pull_request(1, body)])

    assert len(documents) == 1
    assert documents[0].content == body


def test_a_body_exactly_one_chunk_long_stays_one_record():
    """The boundary case, so the split cannot fire one character early."""
    body = "x" * DOCUMENT_CHUNK_CHARACTERS
    documents = source_documents(pull_requests=[pull_request(2, body)])

    assert len(documents) == 1
    assert documents[0].content == body


def test_a_long_body_becomes_several_records():
    body = "x" * (DOCUMENT_CHUNK_CHARACTERS * 2 + 50)
    documents = source_documents(pull_requests=[pull_request(3, body)])

    assert len(documents) > 1
    assert all(len(d.content) <= DOCUMENT_CHUNK_CHARACTERS for d in documents)


def test_consecutive_chunks_overlap_by_the_configured_characters():
    body = "".join(chr(33 + index % 90) for index in range(2250))
    documents = source_documents(pull_requests=[pull_request(4, body)])

    first, second = documents[0].content, documents[1].content
    carried = first[-DOCUMENT_CHUNK_OVERLAP_CHARACTERS:]

    assert second[:DOCUMENT_CHUNK_OVERLAP_CHARACTERS] == carried


def test_non_overlapping_portions_reconstruct_the_body_exactly():
    body = "".join(chr(33 + index % 90) for index in range(5003))
    documents = source_documents(pull_requests=[pull_request(5, body)])

    reconstructed = documents[0].content + "".join(
        document.content[DOCUMENT_CHUNK_OVERLAP_CHARACTERS:]
        for document in documents[1:]
    )
    assert reconstructed == body


def test_chunk_ids_extend_the_body_id_rather_than_replacing_it():
    body = "x" * (DOCUMENT_CHUNK_CHARACTERS + 1)
    documents = source_documents(pull_requests=[pull_request(7, body)])

    assert len(documents) > 1
    assert [d.id for d in documents] == [
        f"doc:pr:7:{index}" for index in range(len(documents))
    ]
    assert all(d.origin == "pr:7" for d in documents)


def test_chunk_ids_are_unique_and_stable_across_two_runs():
    """Mentions point at these. A scheme that reshuffles breaks provenance."""
    body = "x" * (DOCUMENT_CHUNK_CHARACTERS * 3)
    first = source_documents(pull_requests=[pull_request(8, body)])
    second = source_documents(pull_requests=[pull_request(8, body)])

    assert [d.id for d in first] == [d.id for d in second]
    assert len({d.id for d in first}) == len(first)


def test_a_chunk_of_only_whitespace_is_not_stored():
    """``_usable`` applies per chunk, not only per field."""
    body = "x" + " " * (DOCUMENT_CHUNK_CHARACTERS * 2)
    documents = source_documents(pull_requests=[pull_request(9, body)])

    assert all(d.content.strip() for d in documents)
    assert len(documents) == 1
    assert documents[0].content == body[:DOCUMENT_CHUNK_CHARACTERS]


def test_every_field_is_chunked_not_only_pull_requests():
    """One rule for all four, so no field keeps whole-body records."""
    long_body = "x" * (DOCUMENT_CHUNK_CHARACTERS * 2)
    issue = Issue.model_validate(
        {
            "id": 5, "number": 5, "title": "t", "state": "open",
            "body": long_body,
            "html_url": "https://example.invalid/issues/5",
            "user": {"login": "alice", "id": 1},
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )
    documents = source_documents(issues=[issue])

    assert len(documents) > 1
    assert [d.id for d in documents] == [
        f"doc:ticket:5:{index}" for index in range(len(documents))
    ]


def test_unicode_is_counted_as_python_characters():
    documents = source_documents(
        pull_requests=[pull_request(10, "é" * (DOCUMENT_CHUNK_CHARACTERS + 1))]
    )

    assert [len(document.content) for document in documents] == [
        DOCUMENT_CHUNK_CHARACTERS,
        DOCUMENT_CHUNK_OVERLAP_CHARACTERS + 1,
    ]
