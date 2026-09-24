"""Capture a LangGraph run without adding any service dependency."""

from __future__ import annotations

from pathlib import Path

from ..tracing import Trace, save, score_overlaps
from ..tracing_langgraph import LangGraphTracer

__all__ = ["LangGraphTracer", "save_run"]


def save_run(
    tracer: LangGraphTracer,
    *,
    query: str | None = None,
    answer: str | None = None,
    path: str | Path | None = None,
) -> tuple[Trace, Path]:
    """Finish a callback recording, score answer overlap, and write JSON."""
    trace = score_overlaps(tracer.finish(query=query, answer=answer))
    return trace, save(trace, path)
