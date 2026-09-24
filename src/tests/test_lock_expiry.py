"""The compile lock's expiry is a setting.

Read at import, like every other setting on the compile path, so each case runs
in a fresh interpreter with exactly the environment it names. The lock is
driven through a stand-in connection that records what reaches ``set``, which
is the value Redis would actually receive.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROBE = """
from src.worker.locks import compile_lock

class Recording:
    def set(self, key, value, *, nx, ex):
        self.ex = ex
        return True

    def eval(self, script, numkeys, *args):
        return 1

connection = Recording()
with compile_lock("org_1", connection=connection):
    pass
print(connection.ex, type(connection.ex).__name__)
"""


def _expiry_seen(tmp_path: Path, **settings: str) -> str:
    """What reaches Redis as the expiry, in a process with only ``settings``."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GRAPHRAG_")
    }
    environment["GRAPHRAG_ENV_FILE"] = ""
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment.update(settings)
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


def test_an_exported_expiry_reaches_redis_as_an_integer(tmp_path):
    assert _expiry_seen(tmp_path, GRAPHRAG_COMPILE_LOCK_TTL="42") == "42 int"


def test_an_unset_expiry_stays_at_half_an_hour(tmp_path):
    assert _expiry_seen(tmp_path) == "1800 int"
