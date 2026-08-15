"""Graph persistence.

``ContextGraph`` is the whole surface. Everything database-specific stays
behind it — Cypher, connections, driver result objects — so callers work in
plain dicts and never import the driver.

    from src.graphdb import open_context_graph

    with open_context_graph("graph.db") as graph:
        graph.upsert_entity("person:alice", "alice", "Person")
"""

from .context_graph import ContextGraph, open_context_graph

__all__ = ["ContextGraph", "open_context_graph"]
