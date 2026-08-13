"""Manual verification for the extractor.

Prints extracted entities with the text they came from, so a person can read
them and judge. There is no automated substitute: a wrong entity is invisible
once it is inside a score, and every quality number measured downstream is
capped by how good these are.

The text below is representative repository prose — pull request titles and
bodies, issue reports, review comments, commit messages — with the kinds of
noise that actually occur: bullets, quotes, trailing punctuation, code
identifiers, version numbers and dates.

    python -m examples.entity_demo

Each row prints the entity, its type, score and backend, then the surrounding
sentence with the matched span bracketed — so the offsets are checked by eye
at the same time as the label.
"""

from __future__ import annotations

import sys

from src.analysis import Extractor

#: Deliberately not cherry-picked to make the extractor look good. Several of
#: these are known-hard: a version number next to a hash, a person's name that
#: is also a common word, a service name inside a file path.
DOCUMENTS = [
    "Rewrite the auth middleware token check. The expiry comparison used < "
    "instead of <=, so tokens were rejected one second early. Fixes #42.",

    "Login fails for tokens that expire exactly on the boundary. Reported by "
    "Alice Mbeki after the payment_service deploy on Tuesday.",

    "* Bump the auth library from 2.1 to 2.3\n"
    "* No behaviour change expected\n"
    "* See PR #1290 for the earlier attempt",

    'Reviewed by Priya Raman: "the retry logic in billing_svc looks wrong, '
    'see #88 for the same bug in notification-service."',

    "Superseded by pull request #1347. The original fix in auth-service did "
    "not cover the refresh path.",

    "Webhook retries flood the payment queue when payment_service is slow. "
    "Related to issue #13 and the work Bob Chen did in Q3.",

    "Merge pull request #204 from acme/fix-token-expiry\n\n"
    "Tighten the authentication token expiry comparison.",

    "Document how to rotate service account credentials for staging. "
    "Owner: Dana Okafor. Blocked on #77.",

    "The file src/billing/webhook.py in notification_service still calls the "
    "deprecated endpoint. Microsoft Azure returns 410 now.",

    "closes #412, #413 and #414 — all three were the same root cause in "
    "auth-service's clock handling.",

    # From here the cases are chosen to break the rules, not to flatter them.
    # A verification corpus made only of clean positives measures nothing.
    "The self-service portal and the customer-service dashboard both call the "
    "same microservice, so #1 is not the right place to fix this.",

    "Reverted [#42](https://github.com/acme/repo/issues/42) — see the "
    "discussion in acme/repo#118 and the mirror at gitlab#77.",

    "Deployed v2.3.1-service to prod at 09:15 UTC. The order_service and "
    "ORDER_SERVICE env var disagree about the port.",

    "Closes #99. Also see PR#1290 (no space) and pull requests #3 and #4 "
    "which were both reverted.",
]


def main(argv: list[str] | None = None) -> int:
    backend = (argv or sys.argv[1:] or ["spacy"])[0]
    extractor = Extractor(backend)

    print(f"backend requested: {backend}")
    print(f"backend in use:    {extractor.backend.name}")
    if extractor.backend.name == "none":
        print("  (spaCy or its model is unavailable — rule stage only)")
    print()

    rows = []
    for index, document in enumerate(DOCUMENTS):
        for entity in extractor.extract(document):
            rows.append((index, document, entity))

    print(f"{len(rows)} entities from {len(DOCUMENTS)} documents; showing 20\n")

    for number, (index, document, entity) in enumerate(rows[:20], start=1):
        # The offsets are verified by eye here as well as by test: what is
        # bracketed is sliced from the document using the entity's own offsets,
        # so a wrong offset shows up as a bracket in the wrong place.
        before = document[max(0, entity.start - 45) : entity.start].replace("\n", " ")
        matched = document[entity.start : entity.end]
        after = document[entity.end : entity.end + 45].replace("\n", " ")

        print(
            f"{number:2}. {entity.text!r:28} {entity.type:8} "
            f"{entity.score:.2f}  {entity.source}"
        )
        print(f"    doc {index}: ...{before}[{matched}]{after}...")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
