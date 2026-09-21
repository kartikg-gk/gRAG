"""The relation vocabulary and what each relation is worth.

Apart from the rest of the settings on purpose: this is plain data with no
environment behind it, so the compile path can price edges without resolving
the serving side's configuration. ``config`` re-exports every name here, so
serving code keeps importing them from there.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Relations
# --------------------------------------------------------------------------

RELATION_AUTHORED = "AUTHORED"
RELATION_RESOLVES = "RESOLVES"
RELATION_REVIEWED = "REVIEWED"
RELATION_TOUCHES = "TOUCHES"
RELATION_PART_OF = "PART_OF"
RELATION_REPORTED = "REPORTED"

#: Text proximity: two entities mentioned near each other.
#:
#: **Declared but never emitted.** Nothing produces a CO_OCCURS edge yet. The
#: constant and its weight are declared anyway so the relation vocabulary is
#: complete: a scorer added later needs a name and a price already agreed, and
#: choosing the price at the moment of the first producer means choosing it
#: under pressure to make that producer's output look reasonable.
#:
#: If something starts emitting these, the weight below is what makes them lose
#: to every structural relation: a proximity guess must never outrank an
#: authorship fact.
RELATION_CO_OCCURS = "CO_OCCURS"

#: The two relations a tenant's graph is built from.
#:
#: Written by the hosted ingest, which reads only an item's title, body and
#: author. ``AUTHORED_BY`` points from the item to the person, the other way
#: round from ``AUTHORED``, so it keeps its own name; it is the same fact and
#: carries the same weight. ``MENTIONS`` is an item naming an entity in prose.
RELATION_AUTHORED_BY = "AUTHORED_BY"
RELATION_MENTIONS = "MENTIONS"

# --------------------------------------------------------------------------
# Confidence
#
# How much each relation is worth, so downstream scoring has something to
# multiply. Structure earns more than inference.
# --------------------------------------------------------------------------

CONFIDENCE = {
    RELATION_AUTHORED: 0.95,
    RELATION_RESOLVES: 0.92,
    RELATION_REVIEWED: 0.85,
    RELATION_TOUCHES: 0.80,
    RELATION_PART_OF: 0.80,
    RELATION_REPORTED: 0.75,
    RELATION_AUTHORED_BY: 0.95,
    # Named outright in the text: stronger than a proximity guess, weaker than
    # every structural fact, because it is still read out of prose.
    RELATION_MENTIONS: 0.60,
    # Weakest by a wide margin, and below every structural relation above.
    RELATION_CO_OCCURS: 0.35,
}
