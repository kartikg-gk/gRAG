"""Cross-origin access: a browser app on another origin can call the API.

The allowed origins are read at import, so the configured-list case runs in a
fresh interpreter with exactly the environment it names.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import auth as auth_module

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_ORIGIN = "http://localhost:5173"


@pytest.fixture()
def client(monkeypatch):
    from src.api.app import create_app

    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)
    return TestClient(create_app(engine_factory=lambda: None))


def test_the_default_allows_any_origin():
    from src.common.config import CORS_ORIGINS

    assert CORS_ORIGINS == ["*"]


def test_a_preflight_from_the_dev_server_is_allowed(client):
    response = client.options(
        "/api/trace",
        headers={
            "Origin": DEV_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] in ("*", DEV_ORIGIN)


def test_a_preflight_is_answered_with_tenancy_on_and_no_key(monkeypatch, client):
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)

    response = client.options(
        "/api/trace",
        headers={"Origin": DEV_ORIGIN, "Access-Control-Request-Method": "POST"},
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


def test_a_refusal_from_the_routing_gate_still_carries_the_header(monkeypatch, client):
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)

    response = client.post(
        "/api/trace",
        json={"query": "who changed the parser?"},
        headers={"Origin": DEV_ORIGIN},
    )

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] in ("*", DEV_ORIGIN)


PROBE = """
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.common.config import CORS_ORIGINS

client = TestClient(create_app(engine_factory=lambda: None))
print(CORS_ORIGINS)
for origin in ("http://a.test", "http://c.test"):
    response = client.options(
        "/api/health",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )
    print(origin, response.headers.get("access-control-allow-origin"))
"""


def test_a_configured_list_allows_only_its_origins(tmp_path):
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GRAPHRAG_")
    }
    environment["GRAPHRAG_ENV_FILE"] = ""
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["GRAPHRAG_CORS_ORIGINS"] = "http://a.test, http://b.test,"
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()[-3:]
    assert lines == [
        "['http://a.test', 'http://b.test']",
        "http://a.test http://a.test",
        "http://c.test None",
    ]
