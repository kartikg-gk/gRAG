"""Shared test configuration.

The process registry is module-level state, so anything a test attaches to it
outlives that test unless something empties it. The fixture below does, for
every test in the suite — the same shape as the reset the graph store's
per-process schema guard carries, and for the same reason: state that lives
above a test is state the next test inherits without asking.

Retry backoff is real code with real sleeps, and several suites deliberately
serve 5xx responses. Left alone that costs the suite tens of seconds of pure
waiting, so the delay is zeroed everywhere by default. The backoff schedule
itself is asserted explicitly in ``test_retry.py``, where ``time.sleep`` is
captured rather than skipped.
"""

from __future__ import annotations

import os

# Before any project module is imported: configuration is read at import, and
# the project root holds a developer's real .env. Empty means load no file.
os.environ["GRAPHRAG_ENV_FILE"] = ""

import pytest  # noqa: E402

from src.common import retry


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch):
    monkeypatch.setattr(retry, "BACKOFF_SECONDS", 0)


@pytest.fixture(autouse=True)
def empty_process_registry():
    """Nothing a test attaches to the process registry reaches the next one.

    Emptied on the way in as well as out: a test that fails partway through
    leaves whatever it had attached, and the next one must not inherit it.
    """
    from src.registry import reset_registry

    reset_registry()
    yield
    reset_registry()
