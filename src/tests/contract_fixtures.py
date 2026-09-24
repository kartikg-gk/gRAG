"""Shared complete response-contract values for API test doubles."""

from __future__ import annotations


def trace_log() -> dict:
    return {
        "intent": {"alpha": 0.15, "beta": 0.85, "type": "relational"},
        "execution_path": {
            "linked_seeds": [],
            "vector_seeds": [],
            "graph_hops": [],
        },
        "recency": {"enabled": True, "floor": 0.35, "applied": []},
        "metrics": {"graph_hits": 0, "vector_k": 9, "total_nodes_evaluated": 0},
    }
