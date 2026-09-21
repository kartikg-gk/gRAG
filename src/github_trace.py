"""One command from a GitHub repository to a recorded agent run.

Fetch a repository's recent pull requests, issues and commits, build a small
corpus from them, run a two-step LangGraph agent over it with the tracer
attached, and write the run out twice: as the recorded trace, and reshaped
for the Studio canvas.

No database, no model and no key beyond an optional GitHub token. The agent's
answer is assembled from what it retrieved rather than generated, so the run
needs nothing to be configured and is the same every time for the same data.

Lives outside every package, like the other LangChain adapters: it needs
``langchain-core``, and the agent needs ``langgraph``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .tracing._text import STOP, tokens as _base_tokens
from .tracing.classify import score_overlaps
from .tracing.mapper import to_tracestate
from .tracing.schema import to_dict

API = "https://api.github.com"
USER_AGENT = "graphrag-trace/0.1"

#: "fixes #12", "closed #12", "resolves #12" in a body or commit message.
RESOLVE_RE = re.compile(r"(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+#(\d+)", re.IGNORECASE)

#: Looser than the scoring tokeniser on purpose: short words like "pr" and
#: "db" matter when matching a question against titles.
WORD_RE = re.compile(r"[a-z0-9_#-]+")
QUERY_STOP = STOP | {
    "a", "an", "or", "of", "in", "on", "to", "is", "it", "by", "at", "what",
    "who", "which", "how", "why", "when", "did", "do", "does", "recently",
    "recent", "latest", "github", "user", "author",
}

#: On an equal lexical score, substantive work outranks boilerplate.
KIND_PRIORITY = {"pull_request": 4, "commit": 3, "ticket": 2, "repo": 1, "person": 0}

EXCERPT_CHARS = 400
COMMIT_LABEL_CHARS = 48


def _get(path: str, token: str | None) -> Any:
    request = urllib.request.Request(
        f"{API}{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if err.code in (401, 403):
            raise SystemExit(
                f"GitHub answered {err.code} for {path}: a private repository or "
                "the rate limit. Pass --token or set GITHUB_TOKEN."
            ) from err
        if err.code == 404:
            raise SystemExit(f"GitHub answered 404 for {path}: check the repository name.") from err
        raise SystemExit(f"GitHub answered {err.code} for {path}.") from err


def _excerpt(text: str | None, limit: int = EXCERPT_CHARS) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def build_corpus(repo: str, token: str | None, n_prs: int, n_issues: int, n_commits: int, *, fetch=_get) -> list:
    """The repository's recent work as documents, each carrying its relations."""
    from langchain_core.documents import Document

    name = repo.split("/")[-1]
    meta = fetch(f"/repos/{repo}", token)
    repo_id = f"repo_{name}"
    documents = [
        Document(
            page_content=_excerpt(
                f"{repo}: {meta.get('description') or 'no description'}. "
                f"Primary language {meta.get('language') or 'n/a'}, "
                f"{meta.get('stargazers_count', 0)} stars, "
                f"{meta.get('open_issues_count', 0)} open issues."
            ),
            metadata={"id": repo_id, "label": name, "kind": "repo", "source": meta.get("html_url")},
        )
    ]

    authored: dict[str, list[str]] = {}

    pulls = fetch(f"/repos/{repo}/pulls?state=all&sort=updated&direction=desc&per_page={n_prs}", token)
    for pull in pulls:
        number = pull["number"]
        login = (pull.get("user") or {}).get("login") or "unknown"
        pr_id = f"pr_{number}"
        edges = [
            {"source": pr_id, "target": repo_id, "relation": "TOUCHES", "weight": 0.7},
            {"source": f"person_{login}", "target": pr_id, "relation": "AUTHORED", "weight": 0.9},
        ]
        for ref in RESOLVE_RE.findall(pull.get("body") or ""):
            edges.append({"source": pr_id, "target": f"tkt_{ref}", "relation": "RESOLVES", "weight": 0.85})
        state = "merged" if pull.get("merged_at") else pull.get("state", "open")
        documents.append(Document(
            page_content=_excerpt(f"PR #{number} '{pull['title']}' by {login} ({state}). {pull.get('body') or ''}"),
            metadata={"id": pr_id, "label": f"PR #{number}", "kind": "pull_request",
                      "source": pull.get("html_url"), "edges": edges},
        ))
        authored.setdefault(login, []).append(pr_id)

    issues = fetch(f"/repos/{repo}/issues?state=all&sort=updated&direction=desc&per_page={n_issues}", token)
    for issue in issues:
        # The issues endpoint returns pull requests too; those are already in.
        if "pull_request" in issue:
            continue
        number = issue["number"]
        login = (issue.get("user") or {}).get("login") or "unknown"
        ticket_id = f"tkt_{number}"
        documents.append(Document(
            page_content=_excerpt(
                f"Issue #{number} '{issue['title']}' by {login} ({issue.get('state', 'open')}). "
                f"{issue.get('body') or ''}"
            ),
            metadata={"id": ticket_id, "label": f"Issue #{number}", "kind": "ticket",
                      "source": issue.get("html_url"),
                      "edges": [{"source": ticket_id, "target": repo_id,
                                 "relation": "REPORTS_ON", "weight": 0.6}]},
        ))
        authored.setdefault(login, [])

    # A repository with one maintainer often has no pull requests or issues at
    # all; its commits carry the history.
    commits = fetch(f"/repos/{repo}/commits?per_page={n_commits}", token) if n_commits else []
    for commit in commits:
        sha = commit["sha"][:7]
        login = (
            (commit.get("author") or {}).get("login")
            or (commit.get("commit", {}).get("author") or {}).get("name")
            or "unknown"
        )
        commit_id = f"commit_{sha}"
        message = commit.get("commit", {}).get("message") or ""
        edges = [
            {"source": commit_id, "target": repo_id, "relation": "TOUCHES", "weight": 0.6},
            {"source": f"person_{login}", "target": commit_id, "relation": "AUTHORED", "weight": 0.9},
        ]
        for ref in RESOLVE_RE.findall(message):
            edges.append({"source": commit_id, "target": f"tkt_{ref}", "relation": "RESOLVES", "weight": 0.8})
        documents.append(Document(
            page_content=_excerpt(f"Commit {sha} by {login}: {message}"),
            metadata={"id": commit_id,
                      "label": _excerpt(message.splitlines()[0] if message else sha, COMMIT_LABEL_CHARS),
                      "kind": "commit", "source": commit.get("html_url"), "edges": edges},
        ))
        authored.setdefault(login, []).append(commit_id)

    for login, work in authored.items():
        documents.append(Document(
            page_content=f"GitHub user {login} authored {len(work)} of the recent changes "
                         f"(pull requests and commits) in {repo}.",
            metadata={"id": f"person_{login}", "label": login, "kind": "person",
                      "source": f"https://github.com/{login}",
                      "edges": [{"source": f"person_{login}", "target": item,
                                 "relation": "AUTHORED", "weight": 0.9} for item in work]},
        ))
    return documents


def _tokens(text: str) -> set[str]:
    return _base_tokens(text, stop=QUERY_STOP, pattern=WORD_RE)


def lexical_graph_retriever(corpus: list, k: int = 10):
    """Lexical seeds, then their one-hop neighbours, capped at ``k``."""
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class LexicalGraphRetriever(BaseRetriever):
        """Neighbours keep their own, often lower, lexical score. An edge is
        returned only when both of its ends were kept."""

        corpus: list
        k: int = 10

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list:
            question = _tokens(query) or {"?"}
            scored = sorted(
                ((round(len(question & _tokens(d.page_content)) / len(question), 3), d)
                 for d in self.corpus),
                key=lambda pair: (pair[0], KIND_PRIORITY.get(pair[1].metadata["kind"], 0)),
                reverse=True,
            )
            by_id = {d.metadata["id"]: (score, d) for score, d in scored}

            kept = {d.metadata["id"]: (score, d) for score, d in scored[: max(3, self.k // 2)]}
            for _, document in list(kept.values()):
                for edge in document.metadata.get("edges", []):
                    for neighbour in (edge["source"], edge["target"]):
                        if len(kept) >= self.k:
                            break
                        if neighbour not in kept and neighbour in by_id:
                            kept[neighbour] = by_id[neighbour]
            for score, document in scored:
                if len(kept) >= self.k:
                    break
                kept.setdefault(document.metadata["id"], (score, document))

            out = []
            for score, document in sorted(kept.values(), key=lambda pair: pair[0], reverse=True):
                edges = [e for e in document.metadata.get("edges", [])
                         if e["source"] in kept and e["target"] in kept]
                meta = {key: value for key, value in document.metadata.items() if key != "edges"}
                # Zero overlap means "here for its relations, not the question":
                # no score says that more honestly than a 0.00.
                if score > 0:
                    meta["score"] = score
                out.append(Document(page_content=document.page_content, metadata={**meta, "edges": edges}))
            return out

    return LexicalGraphRetriever(corpus=corpus, k=k)


def run_traced(corpus: list, question: str, repo: str, k: int):
    """Run the two-step agent with the tracer attached; return the scored trace."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as err:
        raise SystemExit('The agent needs LangGraph: pip install "graphrag[examples]"') from err
    from typing import TypedDict

    from .tracing_langgraph import LangGraphTracer

    retriever = lexical_graph_retriever(corpus, k)

    class State(TypedDict):
        question: str
        docs: list
        answer: str

    def retrieve(state: State, config) -> dict:
        return {"docs": retriever.invoke(state["question"], config=config)}

    def answer(state: State, config) -> dict:
        docs = state["docs"]
        top = docs[0]
        kinds: dict[str, int] = {}
        for document in docs:
            kinds[document.metadata["kind"]] = kinds.get(document.metadata["kind"], 0) + 1
        parts = ", ".join(f"{count} {kind.replace('_', ' ')}(s)" for kind, count in sorted(kinds.items()))
        return {"answer": (
            f"Top match in {repo}: {top.metadata['label']}. "
            f"{top.page_content[:160]}… Retrieved {len(docs)} items ({parts}); "
            "open the graph to see who and what they connect to."
        )}

    graph = (StateGraph(State)
             .add_node("retrieve", retrieve)
             .add_node("answer", answer)
             .add_edge(START, "retrieve")
             .add_edge("retrieve", "answer")
             .add_edge("answer", END)
             .compile())

    tracer = LangGraphTracer()
    result = graph.invoke({"question": question}, config={"callbacks": [tracer]})
    trace = tracer.finish(query=question, answer=result["answer"])
    # The tracer records; scoring is a separate step, so the run can say which
    # retrieved items the answer actually drew on.
    return score_overlaps(trace)


def write_outputs(trace, out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    recorded = out / "agent_trace.json"
    drawable = out / "trace_state.json"
    recorded.write_text(json.dumps(to_dict(trace), indent=2), encoding="utf-8")
    drawable.write_text(json.dumps(to_tracestate(trace), indent=2), encoding="utf-8")
    return recorded, drawable


def add_arguments(parser: argparse.ArgumentParser) -> None:
    import os

    parser.add_argument("repository", help="owner/name")
    parser.add_argument("question", nargs="?", default=None,
                        help='the question to trace (default: "What changed recently in <repo>, and who drove it?")')
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"),
                        help="GitHub token, or set GITHUB_TOKEN; needed for private repositories")
    parser.add_argument("--prs", type=int, default=25, help="recent pull requests to fetch, default 25")
    parser.add_argument("--issues", type=int, default=25, help="recent issues to fetch, default 25")
    parser.add_argument("--commits", type=int, default=25, help="recent commits to fetch, default 25")
    parser.add_argument("--top", type=int, default=10, help="items to retrieve, default 10")
    parser.add_argument("--out", type=Path, default=Path("trace_out"), help="output directory, default trace_out")


def run(args: argparse.Namespace) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
    if "/" not in args.repository:
        raise SystemExit("the repository must be owner/name")
    question = args.question or f"What changed recently in {args.repository}, and who drove it?"

    print(f"fetching {args.repository}: last {args.prs} pull requests, "
          f"{args.issues} issues, {args.commits} commits", file=sys.stderr)
    corpus = build_corpus(args.repository, args.token, args.prs, args.issues, args.commits)
    print(f"corpus   : {len(corpus)} documents", file=sys.stderr)

    trace = run_traced(corpus, question, args.repository, args.top)
    recorded, drawable = write_outputs(trace, args.out)

    items = sum(len(r.items) for r in trace.retrievals)
    edges = sum(len(r.edges) for r in trace.retrievals)
    arm = trace.retrievals[0].arm if trace.retrievals else "n/a"
    print(f"answer   : {trace.answer}")
    print(f"retrieved: {items} items · {edges} edges · arm={arm} · {trace.duration_ms or 0.0:.1f}ms")
    print(f"wrote    : {recorded}")
    print(f"wrote    : {drawable}")
    return 0
