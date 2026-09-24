"""Loading the environment file.

Settings are read at import. Without a loader they come only from what the
shell exported, and a value written to ``.env`` and nowhere else is silently
absent: an unconfigured judge, a history database that raises, and nothing
saying the file was never read.

The loader tests drive the real function against real files in a temporary
directory. The import-order tests start a fresh interpreter, because what they
assert — that a value in the file is visible to a module-level constant — is
only true or false in a process that has not imported the module yet.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE = "GRAPHRAG_ENVIRONMENT_TEST_PROBE"
FILE_VARIABLE = "GRAPHRAG_ENV_FILE"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """An empty working directory, with the suite's opt-out lifted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(PROBE, raising=False)
    monkeypatch.delenv(FILE_VARIABLE, raising=False)
    return tmp_path


# ==========================================================================
# The loader
# ==========================================================================


def test_a_variable_in_the_file_reaches_the_environment(workdir):
    from src.common.environment import load_environment

    (workdir / ".env").write_text(f"{PROBE}=from-file\n")

    loaded = load_environment()

    assert os.environ[PROBE] == "from-file"
    assert loaded.resolve() == (workdir / ".env").resolve()


def test_a_variable_already_in_the_environment_wins(workdir, monkeypatch):
    """The file is a default. An exported value is a decision."""
    from src.common.environment import load_environment

    monkeypatch.setenv(PROBE, "from-shell")
    (workdir / ".env").write_text(f"{PROBE}=from-file\n")

    load_environment()

    assert os.environ[PROBE] == "from-shell"


def test_no_file_is_not_an_error(workdir):
    """A checkout with nothing configured is the ordinary starting state."""
    from src.common.environment import load_environment

    assert load_environment() is None


def test_an_empty_file_variable_loads_nothing(workdir, monkeypatch):
    from src.common.environment import load_environment

    monkeypatch.setenv(FILE_VARIABLE, "")
    (workdir / ".env").write_text(f"{PROBE}=from-file\n")

    assert load_environment() is None
    assert PROBE not in os.environ


def test_a_named_file_is_loaded_instead_of_the_default(workdir, monkeypatch):
    from src.common.environment import load_environment

    (workdir / ".env").write_text(f"{PROBE}=default-file\n")
    (workdir / "staging.env").write_text(f"{PROBE}=named-file\n")
    monkeypatch.setenv(FILE_VARIABLE, str(workdir / "staging.env"))

    load_environment()

    assert os.environ[PROBE] == "named-file"


def test_a_named_file_that_does_not_exist_is_an_error(workdir, monkeypatch):
    """Naming a file says it exists. Loading nothing instead is a silent default."""
    from src.common.environment import load_environment

    monkeypatch.setenv(FILE_VARIABLE, str(workdir / "absent.env"))

    with pytest.raises(FileNotFoundError):
        load_environment()


def test_the_suite_never_reads_the_developers_file():
    """The project root holds a real .env with real credentials in it."""
    assert os.environ.get(FILE_VARIABLE) == ""


# ==========================================================================
# Import order, in a fresh interpreter
# ==========================================================================


def _fresh_interpreter(workdir: Path, code: str) -> str:
    """The last line a new process prints, with no project settings inherited."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GRAPHRAG_") and name != "GITHUB_TOKEN"
    }
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=workdir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "module, attribute, variable, value",
    [
        ("src.common.config", "JUDGE_MODEL", "GRAPHRAG_JUDGE_MODEL", "model-from-file"),
        ("src.worker.config", "REDIS_URL", "GRAPHRAG_REDIS_URL", "redis://from-file:6379/0"),
    ],
)
def test_a_setting_read_at_import_sees_the_file(
    workdir, module, attribute, variable, value
):
    """Both configuration modules: the serving process and the compile process."""
    (workdir / ".env").write_text(f"{variable}={value}\n")

    printed = _fresh_interpreter(workdir, f"import {module} as m; print(m.{attribute})")

    assert printed == value


@pytest.mark.parametrize("entry_point", ["src.cli", "src.api.app", "src.worker.app"])
def test_a_setting_read_at_call_time_sees_the_file_from_every_entry_point(
    workdir, entry_point
):
    """The database URLs are read when a connection is made, not at import."""
    (workdir / ".env").write_text("GRAPHRAG_DATABASE_URL=sqlite:///from-file.db\n")

    printed = _fresh_interpreter(
        workdir,
        f"import {entry_point}; "
        "from src.models.history import history_url; print(history_url())",
    )

    assert printed == "sqlite:///from-file.db"
