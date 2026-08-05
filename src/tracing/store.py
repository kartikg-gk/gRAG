"""Persistence: put a trace on disk and read it back.

This module knows nothing about how a trace was recorded — it takes a finished
``Trace`` and writes it. The accumulator, correspondingly, knows nothing about
paths. Neither imports the other's concerns.

Naming is ``<utc-timestamp>_<query-slug>.json`` so a directory of traces sorts
chronologically and is still readable at a glance. The output directory comes
from ``$GRAPHRAG_TRACE_DIR``, defaulting to the working directory.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .schema import Trace, to_dict, trace_from_dict

#: Where traces land when no path is given.
TRACE_DIR_ENV = "GRAPHRAG_TRACE_DIR"
DEFAULT_TRACE_DIR = "."

#: Filename slug when a query has nothing usable in it.
EMPTY_SLUG = "query"
SLUG_MAX_LENGTH = 60

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def slugify(query: str) -> str:
    """Lowercase, hyphenate runs of anything else, and never return empty."""
    slug = _NON_ALPHANUMERIC.sub("-", query.lower()).strip("-")
    return slug[:SLUG_MAX_LENGTH].strip("-") or EMPTY_SLUG


def trace_dir() -> Path:
    """The configured output directory."""
    return Path(os.environ.get(TRACE_DIR_ENV) or DEFAULT_TRACE_DIR)


def trace_filename(query: str, moment: datetime | None = None) -> str:
    """``<utc-timestamp>_<query-slug>.json``."""
    moment = moment or datetime.now(timezone.utc)
    stamp = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{slugify(query)}.json"


def default_path(trace: Trace) -> Path:
    """Where this trace goes when the caller names no path."""
    return trace_dir() / trace_filename(trace.query, trace.started_at)


def save(trace: Trace, path: str | Path | None = None) -> Path:
    """Write a trace as readable JSON. Returns where it went.

    With no path, the name is derived from the trace and the directory from
    the environment. Parent directories are created, because a caller that
    configured an output directory meant for it to be used.
    """
    destination = Path(path) if path is not None else default_path(trace)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(to_dict(trace), indent=2) + "\n", encoding="utf-8"
    )
    return destination


def load(path: str | Path) -> Trace:
    """Read a trace back from disk."""
    return trace_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def finish_and_save(
    recorder,
    path: str | Path | None = None,
    *,
    query: str | None = None,
    answer: str | None = None,
) -> tuple[Trace, Path]:
    """Finalize a recording and write it, in one call.

    The convenience shortcut for the common ending. ``recorder`` is duck-typed
    rather than imported so persistence keeps no dependency on accumulation.
    """
    trace = recorder.finish(query=query, answer=answer)
    return trace, save(trace, path)
