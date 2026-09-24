from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest
from fastapi.exceptions import ResponseValidationError
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api import auth as auth_module
from src.api.app import create_app, model_router
from src.api.models import FRONTEND_RESPONSE_MODELS, SessionRead, TraceRead
from src.retrieval.response import RoutedNode
from src.tests.test_api_surface import FakeEngine


FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "frontend_api_fixtures.json"
)


def test_frontend_fixtures_validate_against_strict_response_contracts():
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))

    assert set(fixtures) == set(FRONTEND_RESPONSE_MODELS)
    for name, model in FRONTEND_RESPONSE_MODELS.items():
        model.model_validate(fixtures[name])


def test_subgraph_fixture_contains_every_edge_endpoint():
    subgraph = json.loads(FIXTURES.read_text(encoding="utf-8"))["subgraph"]
    node_ids = {node["id"] for node in subgraph["nodes"]}

    assert {
        endpoint
        for edge in subgraph["edges"]
        for endpoint in (edge["source"], edge["target"])
    } <= node_ids


def test_openapi_publishes_the_frontend_response_models():
    schema = create_app(engine_factory=lambda: None).openapi()
    expected = {
        ("/api/trace", "post"): "TraceResponseRead",
        ("/api/subgraph", "post"): "SubgraphRead",
        ("/api/suggestions", "get"): "SuggestionsRead",
        ("/api/graphs", "get"): "GraphsRead",
        ("/api/graphs/switch", "post"): "GraphSwitchRead",
        ("/api/health", "get"): "HealthRead",
        ("/api/summarize", "post"): "SummaryRead",
        ("/api/answer", "post"): "AnswerRead",
        ("/api/sessions", "post"): "SessionRead",
        ("/api/sessions/{session_id}/traces", "get"): "TraceRead",
    }

    for (path, method), model_name in expected.items():
        response = schema["paths"][path][method]["responses"]["200"]["content"]
        serialized = json.dumps(response, sort_keys=True)
        assert model_name in serialized


def test_frozen_json_routes_have_runtime_response_models():
    app = create_app(engine_factory=lambda: None)
    expected = {
        ("/api/trace", "POST"),
        ("/api/subgraph", "POST"),
        ("/api/suggestions", "GET"),
        ("/api/graphs", "GET"),
        ("/api/graphs/switch", "POST"),
        ("/api/health", "GET"),
        ("/api/summarize", "POST"),
        ("/api/answer", "POST"),
    }
    routes = {
        (route.path, method): route
        for route in [*app.routes, *model_router.routes]
        if isinstance(route, APIRoute)
        for method in route.methods
    }

    assert all(routes[key].response_model is not None for key in expected)


TRACE_LOG = {
    "intent": {"alpha": 0.15, "beta": 0.85, "type": "relational"},
    "execution_path": {
        "linked_seeds": ["pr:1"],
        "vector_seeds": ["pr:1"],
        "graph_hops": [],
    },
    "recency": {"enabled": True, "floor": 0.35, "applied": []},
    "metrics": {"graph_hits": 1, "vector_k": 9, "total_nodes_evaluated": 1},
}


@dataclass
class _TracePayload:
    query: str
    results: list[RoutedNode]
    trace_log: dict


@dataclass
class _TracePayloadWithExtra(_TracePayload):
    unexpected: str = "must be rejected"


class _ContractEngine(FakeEngine):
    def __init__(self, *, top_level_extra=False, document_extra=False, trace_extra=False):
        super().__init__(hits=[])
        self.top_level_extra = top_level_extra
        self.document_extra = document_extra
        self.trace_extra = trace_extra

    async def route_async(self, query, k=None, **kwargs):
        document = {"doc_id": "doc:1", "content": "context", "path": None}
        if self.document_extra:
            document["unexpected"] = "must be rejected"
        result = RoutedNode(
            "pr:1", "Fix login", "PR", 0.9, 0.8, 0.7, 1.0, None, [document]
        )
        trace_log = json.loads(json.dumps(TRACE_LOG))
        if self.trace_extra:
            trace_log["metrics"]["unexpected"] = "must be rejected"
        payload = _TracePayload(query, [result], trace_log)
        if self.top_level_extra:
            payload = _TracePayloadWithExtra(query, [result], trace_log)
        return payload


def _trace_client(monkeypatch, **engine_options):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)
    return TestClient(create_app(engine_factory=lambda: _ContractEngine(**engine_options)))


@pytest.mark.parametrize(
    "engine_options",
    [
        {"top_level_extra": True},
        {"document_extra": True},
        {"trace_extra": True},
    ],
)
def test_trace_runtime_rejects_unexpected_response_fields(monkeypatch, engine_options):
    with _trace_client(monkeypatch, **engine_options) as client:
        with pytest.raises(ResponseValidationError):
            client.post("/api/trace", json={"query": "q"})


def test_trace_runtime_accepts_the_contract_and_omits_absent_trace_id(monkeypatch):
    with _trace_client(monkeypatch) as client:
        response = client.post("/api/trace", json={"query": "q"})

    assert response.status_code == 200
    assert "trace_id" not in response.json()
    assert response.json()["results"][0]["age_days"] is None


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            SessionRead,
            {
                "id": "s-1",
                "user_id": "u-1",
                "title": "title",
                "created_at": "2026-09-19T00:00:00Z",
                "unexpected": True,
            },
        ),
        (
            TraceRead,
            {
                "id": "t-1",
                "session_id": "s-1",
                "query": "q",
                "execution_plan": {},
                "graph_payload": [],
                "created_at": "2026-09-19T00:00:00Z",
                "unexpected": True,
            },
        ),
    ],
)
def test_history_response_contracts_reject_extra_fields(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)
