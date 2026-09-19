"""Source text, split into chunks, separately from the nodes built out of it.

A node carries a thing's identity — a pull request's title, a commit's SHA and
message. A document carries the prose that thing arrived with. They are
different and are stored differently: node fields are what the graph is keyed
and displayed on, and document content is the text extraction runs over.

Why this exists as its own step
------------------------------

Bodies are fetched from the API and validated into the models, and node
construction does not carry them: a pull request node keeps ``number``,
``title``, ``state``, ``draft`` and ``url``. Measured on the verification
fixtures, every reference-shaped entity in that corpus lives in a body and
none in a title, so extraction over node text alone found nothing at all —
0 entities against 2. That is the whole of the gap this closes.

Documents are built from the models rather than from the graph builder,
because the builder does not retain what it did not put on a node. That keeps
node construction unchanged: nothing here alters what a node holds, so the
JSON a build emits is unaffected.

What becomes a document
-----------------------

Pull request bodies, issue bodies, review bodies, and commit messages.

Review bodies are included because they are fetched, validated, and until now
read by nothing — a reviewer writing "same bug as #88" was information the
pipeline already had in memory and discarded.

**Commits become documents even though a commit node already holds its message
in full.** The message is therefore stored twice, once as the node's label and
once as document content, and that is a deliberate cost. The two serve
different roles: the label is the commit's display identity, and the document
is the text a mention edge points back to as its provenance. Without a commit
document, a reference written in a commit message could not produce a mention
at all, because a mention needs a document to originate from. Uniformity is
worth the duplicated string — every payload carrying prose produces a
document, so "where did this entity come from" has one answer shape rather
than one per payload type.

Empty and absent bodies produce no document. A row whose content is ``None``
or whitespace would carry a path, occupy space, and never yield a mention. The
same rule applies per chunk, not only per field.

One record per chunk, not per body
----------------------------------

A long body becomes several records; a short one stays a single record. The
unit is forced by how coverage is scored, not chosen for storage reasons:
coverage divides by the item's own token count, so an answer of ``N`` tokens
caps every score at ``N / |I|`` and an item longer than ``5N`` cannot clear a
0.2 threshold however relevant it is. Whole bodies run past that bound — the
largest sampled is 4,371 words — and the fix has to be a smaller unit, since a
threshold loose enough to admit a whole body admits everything.

The character size and overlap live in ``common.config``. They are independent
of the extractor's word windows: stored chunks preserve exact character slices
while extraction windows preserve whole words and absolute offsets.

Ids carry the chunk index
-------------------------

``doc:pr:101`` becomes ``doc:pr:101:0``, ``doc:pr:101:1``. The original id is
still the prefix, so a record's origin is readable from its id alone, and the
index comes from the chunk's position in the text rather than from iteration
order, so re-ingesting the same body produces the same ids.

Every record carries the suffix, including a body that produced only one
chunk. Two id shapes would mean every consumer has to handle both; one shape
means none do. The cost is that ids written before chunking do not match the
ones written after, so a store built earlier needs re-ingesting rather than
merging cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping

from ..common.config import (
    DOCUMENT_CHUNK_CHARACTERS,
    DOCUMENT_CHUNK_OVERLAP_CHARACTERS,
)


@dataclass(frozen=True)
class SourceDocument:
    """One piece of source prose, addressable and attributable.

    ``origin`` names the node the text arrived with, so a mention can be traced
    back past the document to the pull request or commit it came from. It is
    not written as an edge here — this is a value, and persistence is not this
    module's business.
    """

    id: str
    path: str
    content: str
    origin: str
    field: str


#: Which payload field a document's text came from. Reported after an ingest so
#: the contribution of each can be seen separately, rather than as one total
#: that cannot say whether review bodies were worth reading.
FIELD_PULL_REQUEST_BODY = "pull_request_body"
FIELD_ISSUE_BODY = "issue_body"
FIELD_REVIEW_BODY = "review_body"
FIELD_COMMIT_MESSAGE = "commit_message"


def _usable(text: str | None) -> bool:
    """Whether text is worth storing. Absent and blank are the same here."""
    return bool(text and text.strip())


def _character_windows(
    text: str,
    *,
    size: int = DOCUMENT_CHUNK_CHARACTERS,
    overlap: int = DOCUMENT_CHUNK_OVERLAP_CHARACTERS,
) -> Iterator[str]:
    """Yield exact character slices of ``text`` in deterministic order."""
    step = size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + size]
        if not chunk:
            break
        yield chunk
        if start + size >= len(text):
            break


def _chunks(
    base_id: str, path: str, content: str, origin: str, field: str
) -> Iterator[SourceDocument]:
    """One record per chunk of ``content``, in reading order.

    Stored chunks are exact character slices. Their geometry is independent of
    extraction's word windows and never normalizes the content between the
    selected boundaries.

    ``_usable`` is applied per chunk rather than only per field. A body whose
    tail is a signature line or a horizontal rule would otherwise store a row
    that carries a path, occupies space, and can never yield a mention.
    """
    for index, chunk in enumerate(_character_windows(content)):
        if not _usable(chunk):
            continue
        yield SourceDocument(
            id=f"{base_id}:{index}",
            path=path,
            content=chunk,
            origin=origin,
            field=field,
        )


def source_documents(
    *,
    pull_requests: Iterable = (),
    issues: Iterable = (),
    commits: Iterable = (),
    reviews: Mapping[int, Iterable] | None = None,
) -> list[SourceDocument]:
    """Every piece of source prose in one build, in a stable order.

    Ordered by payload type and then by the order each arrived in, so two runs
    over the same input produce the same list. Ingest order is already made
    total upstream, so this inherits that rather than sorting again.
    """
    documents: list[SourceDocument] = []

    for pull_request in pull_requests:
        if _usable(pull_request.body):
            documents.extend(
                _chunks(
                    f"doc:pr:{pull_request.number}",
                    pull_request.html_url,
                    pull_request.body,
                    f"pr:{pull_request.number}",
                    FIELD_PULL_REQUEST_BODY,
                )
            )

    for issue in issues:
        if _usable(issue.body):
            documents.extend(
                _chunks(
                    f"doc:ticket:{issue.number}",
                    issue.html_url,
                    issue.body,
                    f"ticket:{issue.number}",
                    FIELD_ISSUE_BODY,
                )
            )

    for commit in commits:
        if _usable(commit.commit.message):
            documents.extend(
                _chunks(
                    f"doc:commit:{commit.sha}",
                    commit.html_url,
                    commit.commit.message,
                    f"commit:{commit.sha}",
                    FIELD_COMMIT_MESSAGE,
                )
            )

    for number, submitted in (reviews or {}).items():
        for review in submitted:
            if _usable(review.body):
                documents.extend(
                    # Keyed on the review's own id, not the pull request's:
                    # one pull request carries many reviews and they would
                    # otherwise overwrite each other.
                    _chunks(
                        f"doc:review:{review.id}",
                        review.html_url,
                        review.body,
                        f"pr:{number}",
                        FIELD_REVIEW_BODY,
                    )
                )

    return documents
