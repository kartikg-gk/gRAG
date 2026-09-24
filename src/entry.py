"""Lightweight entry point for the local viewer and optional backend CLI."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "github-trace":
        from .local_github import main as github_main

        return github_main(args[1:])
    if args and args[0] in {"ingest", "view", "trace"}:
        from .cli import main as backend_main

        return backend_main(args)
    from .tracing.web import main as viewer_main

    return viewer_main(args)
