"""M4 — proof the tracing package stands on its own.

The claim is that ``src.tracing`` is usable without the rest of the pipeline and
needs nothing beyond the standard library. That is easy to say and easy to break
with one convenience import, so it is checked here rather than asserted in a
README.

The import checks run in a subprocess: once this test session has imported
pydantic and httpx for the other suites, ``sys.modules`` in this process can no
longer prove anything.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "src" / "tracing"


def run_python(code: str, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_importing_tracing_pulls_in_no_third_party_packages():
    result = run_python(
        "import sys, src.tracing;"
        "third_party = [m for m in ('pydantic', 'httpx', 'certifi', 'anyio')"
        " if m in sys.modules];"
        "print(third_party)"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_importing_tracing_pulls_in_no_other_pipeline_module():
    result = run_python(
        "import sys, src.tracing;"
        "siblings = [m for m in sys.modules"
        " if m.startswith('src.') and not m.startswith('src.tracing')];"
        "print(siblings)"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_the_package_source_imports_nothing_from_the_rest_of_the_project():
    """Real imports only — docstrings showing usage examples do not count."""
    import ast

    offenders = []
    for path in PACKAGE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, which stays inside the package.
                names = [] if node.level else [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] == "src":
                    offenders.append(f"{path.name}:{node.lineno}: {name}")

    assert offenders == []


def test_the_package_runs_from_a_copy_outside_the_project(tmp_path):
    """Copied somewhere else under a different name, it still works."""
    import shutil

    destination = tmp_path / "tracekit"
    shutil.copytree(PACKAGE_DIR, destination, ignore=shutil.ignore_patterns("__pycache__"))

    result = run_python(
        "from tracekit import capture, score_overlaps, is_used;"
        "t = score_overlaps(capture('q',"
        " [{'id': 'a', 'content': 'alpha bravo', 'source': 's'}],"
        " 'alpha bravo charlie'));"
        "print(is_used(t.items[0].overlap))",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_the_viewer_runs_as_a_module():
    result = run_python(
        "import runpy, sys;"
        "sys.argv = ['src.tracing', 'example_trace.json'];"
        "runpy.run_module('src.tracing', run_name='__main__')"
    )

    assert "who changed authentication recently?" in result.stdout
    assert "3 of 7" in result.stdout


def test_the_viewer_reports_a_missing_file_clearly():
    result = subprocess.run(
        [sys.executable, "-m", "src.tracing", "no-such-file.json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "no such trace file" in result.stderr


def test_the_viewer_reports_bad_usage_clearly():
    result = subprocess.run(
        [sys.executable, "-m", "src.tracing"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr
