"""Example questions built from what is actually in a graph.

A hardcoded list is wrong twice: it asks about things a particular graph may
not contain, and it never changes when the graph does. These are built from
the entities the graph actually holds, so an empty graph suggests nothing and
a graph about one repository suggests questions about that repository.

Over-fetch, then diversify
--------------------------

Asking the store for exactly the number wanted returns the top few by degree,
and the top few by degree are routinely all the same kind of thing — a busy
repository's hubs are its most active people, one after another. Five
questions about five people is one question asked five times.

So the store is asked for several times the limit, and the walk down that
longer list takes **at most one entity per type** before it will take a second
of any type. If that runs out of types before it runs out of room, the
remainder is filled by degree alone, skipping anything already suggested.

That ordering matters: diversity first, then rank. The reverse — rank with a
diversity tiebreak — produces the same clustering it was meant to avoid,
because rank alone already decided the top of the list.

This module takes rows and returns questions. It does not know what a store
is, which is what lets the over-fetch factor and the diversification be tested
against a handful of dictionaries.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

#: How many candidates to ask for per question wanted. Enough that a graph
#: dominated by one type still has other types within reach, small enough that
#: the query stays cheap. Chosen, not measured — the number that would justify
#: moving it is the observed type distribution of a real graph.
OVERFETCH = 5

#: What to ask about each kind of thing. The wording is per type because a
#: question that reads naturally about a person reads oddly about a file.
TEMPLATES = {
    "Person": "What has {label} been working on?",
    "PR": "What changed in {label}?",
    "Ticket": "What is {label} about?",
    "Commit": "What did {label} change?",
    "File": "What touches {label}?",
    "Repo": "What is happening in {label}?",
    "Service": "What depends on {label}?",
}

#: For a type with no template. Deliberately vague, because a type this does
#: not know about is a type whose entities it cannot phrase a sharp question
#: about — and a sharp question about the wrong thing is worse than a broad one.
DEFAULT_TEMPLATE = "What is {label} related to?"


def question_for(entity: Mapping[str, Any]) -> str:
    """The question text for one entity."""
    label = entity.get("label") or entity.get("id") or ""
    template = TEMPLATES.get(entity.get("type") or "", DEFAULT_TEMPLATE)
    return template.format(label=label)


def suggestions_from(
    entities: Iterable[Mapping[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Up to ``limit`` questions, at most one per type before any repeats.

    ``entities`` is expected in descending degree order — the order the store
    returns hubs in — and that order is preserved within each pass.

    Returns the question, the entity it came from and that entity's type. A
    caller rendering more than the string needs both, and recovering them from
    the text afterwards would mean parsing a sentence this module just built.
    """
    if limit <= 0:
        return []

    candidates = [entity for entity in entities if entity.get("id")]

    chosen: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    seen_types: set[str] = set()

    # First pass: one per type, in degree order.
    for entity in candidates:
        if len(chosen) >= limit:
            break
        node_type = entity.get("type") or ""
        if node_type in seen_types:
            continue
        seen_types.add(node_type)
        used_ids.add(entity["id"])
        chosen.append(
            {
                "question": question_for(entity),
                "entity_id": entity["id"],
                "type": entity.get("type"),
            }
        )

    # Second pass: fill by degree alone, skipping what is already there.
    for entity in candidates:
        if len(chosen) >= limit:
            break
        if entity["id"] in used_ids:
            continue
        used_ids.add(entity["id"])
        chosen.append(
            {
                "question": question_for(entity),
                "entity_id": entity["id"],
                "type": entity.get("type"),
            }
        )

    return chosen
