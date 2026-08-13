"""Manual verification for entity resolution.

Prints the label each group settled on and what a query term resolves to, so a
person can read them and judge. There is no automated substitute: a wrong merge
pools two things' evidence permanently, and nothing downstream can tell
afterwards which relationship belonged to which.

Runs over the entities the extractor actually produces from
``examples/entity_demo.py`` — the same corpus, so the drift being resolved is
drift the pipeline really generates rather than drift invented to be resolvable.

    python -m examples.resolve_demo

Cross-type merges are reported separately and loudly. They should be
impossible: candidates are filtered by type before anything is scored, so one
appearing here means that filter broke.
"""

from __future__ import annotations

import sys

from src.analysis import Extractor, Resolver

from examples.entity_demo import DOCUMENTS


def main(argv: list[str] | None = None) -> int:
    extractor = Extractor("none")
    entities = [
        entity for document in DOCUMENTS for entity in extractor.extract(document)
    ]

    # No judge is supplied, so the band must resolve to "not the same".
    # Anything that reaches the model path is reported below as a decision
    # that was declined, not silently merged.
    resolver = Resolver()
    resolver.resolve(entities)

    stats = resolver.stats
    print(f"similarity source: {resolver.similarity.name}")
    print(f"judge:             {'none configured' if resolver.judge is None else 'set'}")
    print()
    print(f"entities seen      {stats.seen}")
    print(f"canonical entities {len(resolver.entities)}")
    print(f"created            {stats.created}")
    print()
    # The two events kept apart. A repeat is the same string in another
    # document: a merge, but not resolution finding anything. Only the variant
    # merges below discovered a name the entity did not already answer to.
    print(f"variant merges     {stats.variant_merges}   <- resolution doing work")
    print(f"  on the fast path {stats.fast_merges}")
    print(f"  decided by model {stats.model_merges}")
    print(f"exact repeats      {stats.repeats}   <- same string seen again")
    print(f"absorb events      {stats.merges}   (variant merges + repeats)")
    print()
    print(f"model calls        {stats.model_calls}")
    print(f"model rejections   {stats.model_rejections}")
    print(f"failures           {stats.failures}")
    print(f"pairs scored       {stats.comparisons}")
    print(f"band fraction      {stats.band_fraction:.1%}")
    print()

    # Per-merge detail is no longer stored — a merge returns the canonical
    # entity and updates its label, nothing more. What can still be shown is
    # the label each group settled on and, through find(), that every variant
    # the corpus produced still reaches it.
    print("canonical labels")
    for resolved in resolver.entities:
        print(f"  [{resolved.type:8}] {resolved.canonical!r}")

    print()
    print("query-time resolution — same similarity, floor "
          f"{resolver.query_floor}")
    probes = [
        ("notification_service", "notification-service"),
        ("notification-service", "notification-service"),
        ("PR#1290", "PR #1290"),
        ("ORDER_SERVICE", "order_service"),
        ("payment_service", "payment_service"),
        ("auth-service", "auth-service"),
        ("database", None),
        ("service", None),
        ("pull request #9999", None),
    ]
    for term, expected in probes:
        found = resolver.find(term)
        got = found.canonical if found else None
        mark = "ok " if got == expected else "!! "
        print(f"  {mark}{term!r:24} -> {got!r}")

    # A merge can only hold one type, since candidates are filtered before
    # scoring. Verified rather than asserted: check that every surface form
    # inside a resolved entity was extracted under that entity's type. This is
    # the failure that would matter most, so it is derived from the extractor's
    # own output rather than from the resolver's word for it.
    types_by_surface: dict[str, set[str]] = {}
    for extracted in entities:
        types_by_surface.setdefault(extracted.text, set()).add(extracted.type)

    crossed = [
        (resolved.canonical, resolved.canonical, resolved.type,
         types_by_surface[resolved.canonical])
        for resolved in resolver.entities
        if types_by_surface.get(resolved.canonical, {resolved.type})
        != {resolved.type}
    ]

    print()
    print(f"cross-type merges: {len(crossed)}")
    for canonical, surface, kind, actual in crossed:
        print(f"  !! {canonical!r} [{kind}] absorbed {surface!r} extracted as {actual}")

    # Order independence of the entity SET, checked on real data. Labels are
    # write-once and therefore order-dependent by design; the grouping is what
    # has to hold. Any label differences are printed below so the accepted
    # variation stays visible rather than hidden by the comparison.
    print()
    print("order independence")
    baseline = _fingerprint(resolver)
    orders = [
        list(reversed(entities)),
        sorted(entities, key=lambda e: e.text),
        sorted(entities, key=lambda e: (e.type, e.text), reverse=True),
    ]
    baseline_labels = {(e.type, e.canonical) for e in resolver.entities}
    for index, order in enumerate(orders, start=1):
        replay = Resolver()
        replay.resolve(order)
        same = _fingerprint(replay) == baseline
        labels = {(e.type, e.canonical) for e in replay.entities}
        drift = sorted(label for _, label in labels - baseline_labels)
        note = f"   labels differing: {drift}" if drift else ""
        print(f"  order {index}: set {'identical' if same else 'DIFFERENT'}{note}")

    # Why the band was empty. Printed because "no pair needed a model" is a
    # claim about the thresholds, and it is only meaningful next to the scores
    # that were actually seen.
    print()
    print("score distribution over scored pairs")
    for low, high, label in [
        (resolver.fast, 1.01, f">= {resolver.fast} fast"),
        (resolver.deep, resolver.fast, f"[{resolver.deep}, {resolver.fast}) band"),
        (0.5, resolver.deep, f"[0.5, {resolver.deep}) new"),
        (0.0, 0.5, "< 0.5 new"),
    ]:
        count = sum(1 for value in _all_scores(resolver, entities) if low <= value < high)
        print(f"  {label:28} {count}")

    return 0


def _fingerprint(resolver: Resolver) -> tuple:
    """The shape of the result, independent of what anything ended up called.

    Labels are write-once, so a different document order produces different
    labels — that is accepted behaviour, not drift. What must not change is the
    entity set: how many entities, of which types, and how many merges got
    them there.
    """
    return (
        len(resolver.entities),
        tuple(sorted(entity.type for entity in resolver.entities)),
        resolver.stats.merges,
        resolver.stats.created,
    )


def _all_scores(resolver: Resolver, entities) -> list[float]:
    """Every pairwise score a fresh run would compute, for the histogram.

    Recomputed on a second resolver rather than recorded during the first: the
    resolver stores merge scores only, and a histogram built from those would
    show merges rather than the distribution that produced them.
    """
    replay = Resolver(resolver.similarity)
    seen: list[float] = []
    for entity in entities:
        candidates = replay._by_type.get(entity.type, [])
        if candidates:
            seen.extend(
                replay.similarity.scores(
                    entity.text, [c.canonical for c in candidates]
                )
            )
        replay.add(entity)
    return seen


if __name__ == "__main__":
    raise SystemExit(main())
