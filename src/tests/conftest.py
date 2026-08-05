"""Shared test configuration.

Retry backoff is real code with real sleeps, and several suites deliberately
serve 5xx responses. Left alone that costs the suite tens of seconds of pure
waiting, so the delay is zeroed everywhere by default. The backoff schedule
itself is asserted explicitly in ``test_retry.py``, where ``time.sleep`` is
captured rather than skipped.
"""

from __future__ import annotations

import pytest

from src.common import retry


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch):
    monkeypatch.setattr(retry, "BACKOFF_SECONDS", 0)
