"""Source text, kept whole, separately from the nodes built out of it.

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
or whitespace would carry a path, occupy space, and never yield a mention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


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
            documents.append(
                SourceDocument(
                    id=f"doc:pr:{pull_request.number}",
                    path=pull_request.html_url,
                    content=pull_request.body,
                    origin=f"pr:{pull_request.number}",
                    field=FIELD_PULL_REQUEST_BODY,
                )
            )

    for issue in issues:
        if _usable(issue.body):
            documents.append(
                SourceDocument(
                    id=f"doc:ticket:{issue.number}",
                    path=issue.html_url,
                    content=issue.body,
                    origin=f"ticket:{issue.number}",
                    field=FIELD_ISSUE_BODY,
                )
            )

    for commit in commits:
        if _usable(commit.commit.message):
            documents.append(
                SourceDocument(
                    id=f"doc:commit:{commit.sha}",
                    path=commit.html_url,
                    content=commit.commit.message,
                    origin=f"commit:{commit.sha}",
                    field=FIELD_COMMIT_MESSAGE,
                )
            )

    for number, submitted in (reviews or {}).items():
        for review in submitted:
            if _usable(review.body):
                documents.append(
                    SourceDocument(
                        # Keyed on the review's own id, not the pull request's:
                        # one pull request carries many reviews and they would
                        # otherwise overwrite each other.
                        id=f"doc:review:{review.id}",
                        path=review.html_url,
                        content=review.body,
                        origin=f"pr:{number}",
                        field=FIELD_REVIEW_BODY,
                    )
                )

    return documents
