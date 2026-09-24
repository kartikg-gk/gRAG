"""The no-service GitHub path from API payloads to a viewable v4 trace."""

import json

import httpx

from src import local_github
from src.tracing import load
from src.tracing import TraceItem
from src.tracing.web import discover, make_handler
from src.ingestion.models import Commit, Issue, PullRequest, Repository
from src.knowledge import GraphBuilder
from src.knowledge.documents import SourceDocument


def make_repository(name):
    return Repository.model_validate({
        "id": 1, "name": name.split("/")[-1], "full_name": name, "private": False,
        "owner": {"id": 1, "login": name.split("/")[0]},
        "html_url": f"https://github.com/{name}", "default_branch": "main",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z",
    })


def make_issue(number, body):
    return Issue.model_validate({
        "id": number, "number": number, "title": f"Issue {number}", "body": body,
        "state": "open", "user": {"id": 1, "login": "octocat"},
        "html_url": f"https://github.com/octocat/Hello-World/issues/{number}",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z",
    })


def make_pull_request(number, body):
    return PullRequest.model_validate(make_issue(number, body).model_dump())


def make_commit():
    return Commit.model_validate({
        "sha": "a" * 40, "html_url": "https://github.com/octocat/Hello-World/commit/" + "a" * 40,
        "commit": {"message": "Fix cache timeout", "author": {"name": "Octocat", "date": "2026-01-01T00:00:00Z"}},
        "author": {"id": 1, "login": "octocat"},
    })


def test_github_command_builds_viewable_v4_trace(tmp_path, monkeypatch):
    repository = make_repository("octocat/Hello-World")
    pull = make_pull_request(1, body="Fixes #10 by correcting the cache timeout")
    issue = make_issue(10, body="Cache timeout makes requests fail")
    commit = make_commit()
    calls = []

    def respond(request):
        calls.append(request.url.path)
        path = request.url.path
        rows = {
            "/repos/octocat/Hello-World": repository.model_dump(mode="json"),
            "/repos/octocat/Hello-World/pulls": [pull.model_dump(mode="json")],
            "/repos/octocat/Hello-World/issues": [issue.model_dump(mode="json")],
            "/repos/octocat/Hello-World/commits": [commit.model_dump(mode="json")],
        }
        return httpx.Response(200, json=rows[path])

    monkeypatch.setattr(local_github, "make_session", lambda: httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(respond)))
    destination = tmp_path / "graphrag_out" / "trace_state.json"
    assert local_github.main(["octocat/Hello-World", "who fixed the cache timeout?",
                              "--out", str(destination)]) == 0
    assert len(calls) == 4
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4
    assert payload["answer"]
    assert payload["retrievals"][0]["items"]
    assert payload["graph"]["edges"]
    assert payload["spans"] and payload["metrics"]["graph_nodes"] > 0
    assert all(item["source_uri"] for item in payload["retrievals"][0]["items"])
    assert load(destination).query == "who fixed the cache timeout?"
    assert len(discover(destination)) == 1
    assert make_handler(discover(destination))


def test_invalid_repository_rejected_without_fetch():
    try:
        local_github.main(["https://github.com/octocat/Hello-World", "question"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("invalid repository accepted")


def test_answer_excerpt_drops_github_template():
    item = TraceItem(id="pr:1", label="Fix parsing", content="Fix parsing. <!-- template --> Corrects nargs handling", source="github")
    assert "template" not in local_github._answer_excerpt(item)
    assert "Corrects nargs" in local_github._answer_excerpt(item)
    item.content = "Fix parsing. <!-- truncated template"
    assert local_github._answer_excerpt(item) == "Fix parsing"


def test_explicit_filename_question_ranks_that_file_first():
    graph = GraphBuilder()
    graph.nodes["file:README.md"] = {"id": "file:README.md", "type": "File", "path": "README.md", "indexed_source": True}
    graph.nodes["file:src/click/other.py"] = {"id": "file:src/click/other.py", "type": "File", "path": "src/click/other.py", "indexed_source": True}
    documents = [
        SourceDocument(id="readme", path="https://github.com/pallets/click/blob/abc/README.md#L1-L1",
                       content="Click is a library", origin="file:README.md", field="source_code"),
        SourceDocument(id="other", path="https://github.com/pallets/click/blob/abc/src/click/other.py#L1-L1",
                       content="Click readme", origin="file:src/click/other.py", field="source_code"),
    ]
    trace = local_github.build_trace("pallets/click", "What does the README say about Click?",
                                    graph, documents, graph.stats, top=2)
    assert trace.items[0].id == "file:README.md"
