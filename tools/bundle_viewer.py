"""Copy the production viewer into the Python package before building it."""

from pathlib import Path
from shutil import copytree, rmtree

ROOT = Path(__file__).resolve().parents[1]
BUILT = ROOT / "frontend" / "dist"
DESTINATION = ROOT / "src" / "tracing" / "ui"
BUILD_COPY = ROOT / "build" / "lib" / "graphrag" / "tracing" / "ui"
SOURCE_LIST = ROOT / "grag_trace_viewer.egg-info" / "SOURCES.txt"

if not (BUILT / "viewer.html").is_file():
    raise SystemExit("Build the viewer first: npm --prefix frontend run build")
if DESTINATION.exists():
    rmtree(DESTINATION)
copytree(BUILT, DESTINATION)
if BUILD_COPY.exists():
    rmtree(BUILD_COPY)
SOURCE_LIST.unlink(missing_ok=True)
print(f"Bundled viewer from {BUILT} into {DESTINATION}")
