"""Which spellings of the tenancy flag turn tenancy on.

The flag is read at import, so each case runs in a fresh interpreter with
exactly the environment it names. The comparison is exact: no trimming and no
case folding.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROBE = """
from src.common.config import MULTI_TENANCY_ENABLED
print(MULTI_TENANCY_ENABLED)
"""


def _tenancy_seen(tmp_path: Path, value: str | None) -> str:
    environment = {
        name: setting
        for name, setting in os.environ.items()
        if not name.startswith("GRAPHRAG_")
    }
    environment["GRAPHRAG_ENV_FILE"] = ""
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    if value is not None:
        environment["GRAPHRAG_MULTI_TENANCY_ENABLED"] = value
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("value", ["1", "true", "True"])
def test_these_values_turn_tenancy_on(tmp_path, value):
    assert _tenancy_seen(tmp_path, value) == "True"


@pytest.mark.parametrize("value", [None, "0", "false", "TRUE", "yes", " 1"])
def test_every_other_value_leaves_tenancy_off(tmp_path, value):
    assert _tenancy_seen(tmp_path, value) == "False"
