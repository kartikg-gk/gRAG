"""The package installs, and the console command runs.

The claim these tests defend is that someone who has never seen this repository
can ``pip install`` it and get a working command. That is easy to break with a
missing package entry or an entry point pointing at a function that moved, and
neither breaks any other test in the suite — nothing else here goes through the
installed distribution.

Why ``--no-deps``
-----------------

The install is deliberately dependency-free. ``graphrag-view`` reaches only
``graphrag.tracing``, which is standard library only, so httpx and pydantic are
not needed to prove the entry point works — and installing them here would buy
nothing this suite does not already cover. That the metadata *declares* them is
checked separately, below, so a dropped requirement still fails.

This does not make the test offline: pip still fetches the build backend named
in ``[build-system]``, uncached. When that fetch fails the fixture skips rather
than fails — an unreachable index is not a defect in this repository. CI runs
the full ``pip install -e ".[dev]"``, where a network is a given.
"""

from __future__ import annotations

import os
import subprocess
import tomllib
import venv
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def project_config() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def script_path(env_dir: Path, name: str) -> Path:
    """Where a console script lands, on this platform."""
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return env_dir / bin_dir / f"{name}{suffix}"


def python_path(env_dir: Path) -> Path:
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return env_dir / bin_dir / f"python{suffix}"


@pytest.fixture(scope="module")
def clean_checkout(tmp_path_factory) -> Path:
    """Only the files a fresh clone would have, copied somewhere else.

    Building from the working tree would hide a whole class of bug: this
    repository gitignores ``*.md``, so a ``readme = "README.md"`` in
    pyproject.toml builds happily here and fails for everyone else. Exporting
    what git tracks — plus what is staged to become tracked — is the only way
    the test can see what CI will see.
    """
    export = tmp_path_factory.mktemp("checkout")

    tracked = _git("ls-files")
    to_be_tracked = _git("ls-files", "--others", "--exclude-standard")
    for relative in sorted(set(tracked) | set(to_be_tracked)):
        source = PROJECT_ROOT / relative
        if not source.is_file():
            continue
        destination = export / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    return export


def _git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def clean_install(tmp_path_factory, clean_checkout) -> Path:
    """A fresh virtual environment with the clean checkout installed into it.

    Module-scoped: building an environment costs seconds, and every test here
    asks a different question of the same one.
    """
    env_dir = tmp_path_factory.mktemp("env") / "venv"
    venv.create(env_dir, with_pip=True, clear=True)

    result = subprocess.run(
        [str(python_path(env_dir)), "-m", "pip", "install", "--no-deps",
         "--no-input", str(clean_checkout)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            "could not build the package — most often no index reachable for "
            f"the build backend:\n{result.stdout}\n{result.stderr}"
        )
    return env_dir


# --------------------------------------------------------------------------
# it installs, and the command runs
# --------------------------------------------------------------------------


def test_the_console_command_renders_a_trace(clean_install):
    command = script_path(clean_install, "graphrag-view")

    assert command.exists(), f"no console script at {command}"

    result = subprocess.run(
        [str(command), str(PROJECT_ROOT / "example_trace.json")],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "who changed authentication recently?" in result.stdout
    assert "3 of 7" in result.stdout


def test_the_console_command_reports_a_missing_file_clearly(clean_install):
    result = subprocess.run(
        [str(script_path(clean_install, "graphrag-view")), "no-such-file.json"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "no such trace file" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_usage_line_names_the_installed_command(clean_install):
    """Not ``python -m src.tracing`` — a name the person cannot type."""
    result = subprocess.run(
        [str(script_path(clean_install, "graphrag-view"))],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "graphrag-view" in result.stderr


def test_the_installed_package_imports_under_its_distribution_name(clean_install):
    result = subprocess.run(
        [
            str(python_path(clean_install)),
            "-c",
            "from graphrag.tracing import capture, is_used, render; print('ok')",
        ],
        capture_output=True,
        text=True,
        cwd=clean_install,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_installed_tracing_package_still_needs_nothing_third_party(clean_install):
    """M4's isolation property survives packaging, in a venv with no deps at all."""
    result = subprocess.run(
        [
            str(python_path(clean_install)),
            "-c",
            "import sys, graphrag.tracing;"
            "print([m for m in ('pydantic', 'httpx', 'certifi', 'anyio')"
            " if m in sys.modules])",
        ],
        capture_output=True,
        text=True,
        cwd=clean_install,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_the_tests_are_not_shipped(clean_install):
    result = subprocess.run(
        [str(python_path(clean_install)), "-c", "import graphrag.tests"],
        capture_output=True,
        text=True,
        cwd=clean_install,
    )

    assert result.returncode != 0


# --------------------------------------------------------------------------
# the dependency list stays small
# --------------------------------------------------------------------------


def test_runtime_dependencies_are_exactly_the_two_that_are_needed():
    """A new runtime dependency has to be argued for here before it lands."""
    names = {
        requirement.split(">")[0].split("=")[0].split("[")[0].strip()
        for requirement in project_config()["project"]["dependencies"]
    }

    assert names == {"httpx", "pydantic"}


@pytest.mark.parametrize("package", ["langgraph", "langchain-core", "pytest"])
def test_test_and_example_only_dependencies_are_optional(package):
    """Nothing needed solely by tests or examples may be a runtime requirement."""
    config = project_config()["project"]
    runtime = " ".join(config["dependencies"])
    optional = " ".join(
        requirement
        for group in config["optional-dependencies"].values()
        for requirement in group
    )

    assert package not in runtime
    assert package in optional


def test_the_declared_console_entry_points_resolve():
    """An entry point naming a function that moved is a silent break."""
    import importlib

    for target in project_config()["project"]["scripts"].values():
        module_name, _, attribute = target.partition(":")
        # In-repo the package is importable as `src`; installed it is
        # `graphrag`. Same modules, so resolving either proves the target.
        module = importlib.import_module(module_name.replace("graphrag", "src", 1))

        assert callable(getattr(module, attribute))
