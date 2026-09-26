"""Fail a release if its wheel cannot serve a built viewer."""

import sys
from pathlib import Path
from zipfile import ZipFile

wheel = Path(sys.argv[1])
with ZipFile(wheel) as archive:
    names = archive.namelist()
    html = "graphrag/tracing/ui/viewer.html"
    if html not in names:
        raise SystemExit(f"{wheel.name} lacks {html}")
    if not any(name.startswith("graphrag/tracing/ui/assets/") and name.endswith(".js") for name in names):
        raise SystemExit(f"{wheel.name} lacks viewer JavaScript")
    if not any(name.startswith("graphrag/tracing/ui/assets/") and name.endswith(".css") for name in names):
        raise SystemExit(f"{wheel.name} lacks viewer CSS")
    if "graphrag/local_github.py" not in names:
        raise SystemExit(f"{wheel.name} lacks the local GitHub command")
    entry_points = next((name for name in names if name.endswith(".dist-info/entry_points.txt")), None)
    if entry_points is None or "graphweave-github-trace" not in archive.read(entry_points).decode("utf-8"):
        raise SystemExit(f"{wheel.name} lacks the GitHub CLI entry point")
    print(f"{wheel.name}: bundled HTML, JavaScript, and CSS present")
