"""The one-command traced run: corpus, retriever, traced agent, written output."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")

from src import github_trace  # noqa: E402

REPO = "acme/widgets"

RESPONSES = {
    f"/repos/{REPO}": {"description": "A widget library", "language": "Python",
                       "stargazers_count": 5, "open_issues_count": 1,
                       "html_url": "https://github.com/acme/widgets"},
    f"/repos/{REPO}/pulls?state=all&sort=updated&direction=desc&per_page=5": [
        {"number": 7, "title": "Fix login crash", "body": "Fixes #3 by checking the token",
         "user": {"login": "ada"}, "merged_at": "2026-09-01T00:00:00Z",
         "html_url": "https://github.com/acme/widgets/pull/7"},
    ],
    f"/repos/{REPO}/issues?state=all&sort=updated&direction=desc&per_page=5": [
        {"number": 3, "title": "Login crashes on expired token", "body": "Steps to reproduce",
         "user": {"login": "bob"}, "state": "closed", "html_url": "https://github.com/acme/widgets/issues/3"},
        {"number": 7, "title": "the PR again", "pull_request": {}, "user": {"login": "ada"}},
    ],
    f"/repos/{REPO}/commits?per_page=5": [
        {"sha": "abcdef1234", "author": {"login": "ada"},
         "commit": {"message": "Refresh tokens before expiry\n\nCloses #3"},
         "html_url": "https://github.com/acme/widgets/commit/abcdef1"},
    ],
}


def fetch(path, token):
    return RESPONSES[path]


def corpus():
    return github_trace.build_corpus(REPO, None, 5, 5, 5, fetch=fetch)


def by_id(documents):
    return {d.metadata["id"]: d for d in documents}


def test_the_corpus_holds_the_repository_its_work_and_its_people():
    docs = by_id(corpus())
    assert set(docs) == {"repo_widgets", "pr_7", "tkt_3", "commit_abcdef1", "person_ada", "person_bob"}
    assert docs["pr_7"].metadata["kind"] == "pull_request"


def test_pull_requests_returned_by_the_issues_endpoint_are_skipped():
    assert "tkt_7" not in by_id(corpus())


def test_relations_are_read_from_bodies_and_messages():
    docs = by_id(corpus())
    pr_edges = {(e["source"], e["relation"], e["target"]) for e in docs["pr_7"].metadata["edges"]}
    assert ("pr_7", "RESOLVES", "tkt_3") in pr_edges
    assert ("person_ada", "AUTHORED", "pr_7") in pr_edges
    commit_edges = {(e["relation"], e["target"]) for e in docs["commit_abcdef1"].metadata["edges"]}
    assert ("RESOLVES", "tkt_3") in commit_edges


def test_a_commit_is_labelled_by_its_first_line():
    assert by_id(corpus())["commit_abcdef1"].metadata["label"] == "Refresh tokens before expiry"


def test_a_person_records_everything_they_authored():
    edges = by_id(corpus())["person_ada"].metadata["edges"]
    assert {e["target"] for e in edges} == {"pr_7", "commit_abcdef1"}


def test_the_retriever_keeps_edges_only_between_documents_it_returned():
    retriever = github_trace.lexical_graph_retriever(corpus(), k=3)
    found = retriever.invoke("login crash")
    kept = {d.metadata["id"] for d in found}
    assert len(found) <= 3
    for document in found:
        for edge in document.metadata["edges"]:
            assert edge["source"] in kept and edge["target"] in kept


def test_a_document_with_no_word_in_common_carries_no_score():
    found = github_trace.lexical_graph_retriever(corpus(), k=10).invoke("login crash")
    scores = {d.metadata["id"]: d.metadata.get("score") for d in found}
    assert scores["pr_7"] and scores["pr_7"] > 0
    assert scores["repo_widgets"] is None


def test_the_traced_run_is_recorded_and_scored(tmp_path):
    pytest.importorskip("langgraph")
    trace = github_trace.run_traced(corpus(), "who fixed the login crash?", REPO, 5)
    assert trace.query == "who fixed the login crash?"
    assert trace.answer.startswith(f"In the sampled {REPO} activity,")
    items = [item for retrieval in trace.retrievals for item in retrieval.items]
    assert items and all(item.overlap is not None for item in items)

    recorded, drawable = github_trace.write_outputs(trace, tmp_path / "out")
    assert json.loads(recorded.read_text(encoding="utf-8"))["query"] == trace.query
    state = json.loads(drawable.read_text(encoding="utf-8"))
    assert state["graph"]["nodes"] and "steps" in state


def test_the_cli_offers_the_trace_command():
    from src.cli import _build_parser

    args = _build_parser().parse_args(["trace", "acme/widgets", "why?", "--top", "4"])
    assert (args.repository, args.question, args.top) == ("acme/widgets", "why?", 4)
