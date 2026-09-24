import base64

import httpx
import pytest

from src.ingestion.github import make_session
from src.ingestion.github import GitHubError
from src.ingestion.source import MAX_FILE_BYTES, eligible, index_source
from src.knowledge import GraphBuilder
from src.local_github import build_trace
from src.tracing import to_dict


def blob(path="src/cache.py", **changes):
    return {"path": path, "type": "blob", "mode": "100644", "sha": "b" * 40,
            "size": 100, **changes}


@pytest.mark.parametrize("entry", [
    blob(".env"), blob("config/secrets.py"), blob("node_modules/index.js"),
    blob("/absolute.py"), blob("../escape.py"), blob("key.pem"), blob("link.py", mode="120000"),
    blob(size=MAX_FILE_BYTES + 1), blob("image.png"), blob(type="tree"),
])
def test_source_excludes_sensitive_generated_binary_and_unsafe_entries(entry):
    assert not eligible(entry)


def test_private_source_index_uses_token_and_emits_immutable_line_citations(monkeypatch):
    token = "test-only-not-a-real-token"
    monkeypatch.setenv("GITHUB_TOKEN", token)
    revision = "a" * 40
    source = "# Cache policy\n\ndef refresh_cache():\n    return 'cache expiry'\n"
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {token}"
        assert request.url.host == "api.github.com"
        if request.url.path.endswith("/commits/HEAD"):
            return httpx.Response(200, json={"sha": revision, "commit": {"tree": {"sha": "c" * 40}}})
        if "/git/trees/" in request.url.path:
            return httpx.Response(200, json={"truncated": False, "tree": [blob(), blob("second.py")]})
        encoded = base64.b64encode(source.encode()).decode()
        return httpx.Response(200, json={"encoding": "base64", "content": encoded[:20] + "\n" + encoded[20:] + "\n"})

    with make_session() as configured:
        headers = configured.headers
    with httpx.Client(base_url="https://api.github.com", headers=headers,
                      transport=httpx.MockTransport(handle)) as session:
        graph = GraphBuilder()
        graph.nodes["repo:owner/private"] = {"id": "repo:owner/private", "type": "Repo", "name": "private"}
        documents = []
        metrics = index_source(session, "owner/private", "cache expiry", graph, documents, max_files=1)
        trace = build_trace("owner/private", "cache expiry", graph, documents, graph.stats, source_metrics=metrics)
    assert len(requests) == 3
    assert metrics["source_files_indexed"] == 1
    assert metrics["source_files_eligible"] == 2
    assert documents[0].path.endswith(f"/{revision}/src/cache.py#L1-L4")
    assert trace.items[0].kind == "File"
    assert trace.items[0].source_uri == documents[0].path
    assert "cache" in trace.answer
    assert token not in str(to_dict(trace))


def test_public_session_does_not_require_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with make_session() as session:
        assert "Authorization" not in session.headers


def test_invalid_base64_blob_is_reported_as_github_error():
    def handle(request):
        if request.url.path.endswith("/commits/HEAD"):
            return httpx.Response(200, json={"sha": "a" * 40, "commit": {"tree": {"sha": "c" * 40}}})
        if "/git/trees/" in request.url.path:
            return httpx.Response(200, json={"truncated": False, "tree": [blob()]})
        return httpx.Response(200, json={"encoding": "base64", "content": "not base64!"})

    with httpx.Client(base_url="https://api.github.com",
                      transport=httpx.MockTransport(handle)) as session:
        with pytest.raises(GitHubError, match="invalid base64"):
            index_source(session, "owner/repo", "cache", GraphBuilder(), [], max_files=1)
