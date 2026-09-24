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
from collections import Counter
from pathlib import Path
from typing import Any

from .tracing._text import STOP, tokens as _base_tokens
from .tracing.classify import score_overlaps
from .tracing.mapper import to_tracestate
from .tracing.schema import to_dict

API = "https://api.github.com"
USER_AGENT = "graphrag-trace/0.1"

#: The keywords GitHub itself treats as closing an issue when followed by
#: ``#<number>``; any of them in a pull request body or a commit message is
#: read as that change resolving the issue.
CLOSING_KEYWORDS = ("close", "closes", "closed", "fix", "fixes", "fixed",
                    "resolve", "resolves", "resolved")
CLOSING_RE = re.compile(r"(?:%s)\s+#(\d+)" % "|".join(CLOSING_KEYWORDS), re.IGNORECASE)

#: Question terms: any run of letters, digits, ``_``, ``#`` and ``-`` that does
#: not start with punctuation, in any case. Unlike the scoring tokeniser this
#: keeps one- and two-character terms, since "pr", "ci" and "db" are often the
#: very word a question about a repository turns on.
TERM_RE = re.compile(r"[a-z0-9#][a-z0-9_#-]*", re.IGNORECASE)

#: Words that shape a question without saying what it is about.
FUNCTION_WORDS = frozenset({"a", "an", "as", "at", "be", "by", "do", "if", "in", "is",
                            "it", "of", "on", "or", "so", "to", "up"})
QUESTION_WORDS = frozenset({"what", "which", "who", "whom", "whose", "when", "where",
                            "why", "how"})
AUXILIARIES = frozenset({"did", "does", "done", "can", "could", "should", "would", "will",
                         "been", "being"})
#: Words about the sample itself rather than anything in it: every document
#: comes from the same repository, recently, via GitHub.
SAMPLE_WORDS = frozenset({"github", "repository", "repo", "recent", "recently", "latest",
                          "activity"})
IGNORED_TERMS = STOP | FUNCTION_WORDS | QUESTION_WORDS | AUXILIARIES | SAMPLE_WORDS

#: A relation GitHub reports directly is certain; one read out of free text
#: ("fixes #12") is an inference and is held a little lower.
REPORTED = 1.0
INFERRED = 0.8

EXCERPT_CHARS = 400
COMMIT_LABEL_CHARS = 48

DEFAULT_FETCH = 25
DEFAULT_TOP = 10

#: What each refusal most likely means, and what to do about it.
_REFUSALS = {
    401: "GitHub refused {path} ({code}): the token is missing or wrong for this "
         "repository. Pass --token or set GITHUB_TOKEN.",
    403: "GitHub refused {path} ({code}): the repository is private or the rate "
         "limit is spent. Pass --token or set GITHUB_TOKEN.",
    404: "GitHub has nothing at {path} ({code}). Check the owner/name spelling.",
}


def _get(path: str, token: str | None) -> Any:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(API + path, headers=headers), timeout=30) as reply:
            body = reply.read()
    except urllib.error.HTTPError as err:
        message = _REFUSALS.get(err.code, "GitHub answered {path} with {code}.")
        raise SystemExit(message.format(path=path, code=err.code)) from err
    return json.loads(body.decode("utf-8"))


def _excerpt(text: str | None, limit: int = EXCERPT_CHARS) -> str:
    return " ".join((text or "").split())[:limit]


def _login(record: dict) -> str:
    return (record.get("user") or {}).get("login") or "unknown"


def _document(doc_id: str, label: str, kind: str, source: str | None, text: str, edges=None):
    from langchain_core.documents import Document

    metadata: dict[str, Any] = {"id": doc_id, "label": label, "kind": kind, "source": source}
    if edges is not None:
        metadata["edges"] = edges
    return Document(page_content=text, metadata=metadata)


def _edge(source: str, target: str, relation: str, weight: float) -> dict:
    return {"source": source, "target": target, "relation": relation, "weight": weight}


def _closes(text: str, closer: str) -> list[dict]:
    return [_edge(closer, f"tkt_{number}", "RESOLVES", INFERRED) for number in CLOSING_RE.findall(text)]


def build_corpus(repo: str, token: str | None, n_prs: int, n_issues: int, n_commits: int, *, fetch=_get) -> list:
    """The repository's recent work as documents, each carrying its relations.

    One document for the repository, one per pull request, issue and commit,
    and one per person who opened or wrote any of them. People who only filed
    issues get a document too, with nothing listed as authored.
    """
    name = repo.split("/")[-1]
    repo_id = f"repo_{name}"
    meta = fetch(f"/repos/{repo}", token)
    documents = [_document(
        repo_id, name, "repo", meta.get("html_url"),
        _excerpt(f"{repo} is a {meta.get('language') or 'n/a'} repository with "
                 f"{meta.get('stargazers_count', 0)} stars and {meta.get('open_issues_count', 0)} "
                 f"open issues. {meta.get('description') or 'No description given.'}"),
    )]
    authored: dict[str, list[str]] = {}

    for pull in fetch(f"/repos/{repo}/pulls?state=all&sort=updated&direction=desc&per_page={n_prs}", token):
        number, login = pull["number"], _login(pull)
        pr_id = f"pr_{number}"
        body = pull.get("body") or ""
        state = "merged" if pull.get("merged_at") else pull.get("state", "open")
        edges = [_edge(pr_id, repo_id, "TOUCHES", REPORTED),
                 _edge(f"person_{login}", pr_id, "AUTHORED", REPORTED), *_closes(body, pr_id)]
        documents.append(_document(
            pr_id, f"PR #{number}", "pull_request", pull.get("html_url"),
            _excerpt(f"Pull request #{number} by {login}, {state}: {pull['title']}. {body}"), edges,
        ))
        authored.setdefault(login, []).append(pr_id)

    for issue in fetch(f"/repos/{repo}/issues?state=all&sort=updated&direction=desc&per_page={n_issues}", token):
        if "pull_request" in issue:
            continue  # GitHub lists pull requests as issues as well; they are in already.
        number, login = issue["number"], _login(issue)
        ticket_id = f"tkt_{number}"
        documents.append(_document(
            ticket_id, f"Issue #{number}", "ticket", issue.get("html_url"),
            _excerpt(f"Issue #{number} by {login}, {issue.get('state', 'open')}: "
                     f"{issue['title']}. {issue.get('body') or ''}"),
            [_edge(ticket_id, repo_id, "REPORTS_ON", REPORTED)],
        ))
        authored.setdefault(login, [])

    # With a single maintainer there may be no pull requests or issues at all,
    # and the commits are the whole history.
    for commit in (fetch(f"/repos/{repo}/commits?per_page={n_commits}", token) if n_commits else []):
        sha = commit["sha"][:7]
        detail = commit.get("commit", {})
        login = (commit.get("author") or {}).get("login") or (detail.get("author") or {}).get("name") or "unknown"
        commit_id = f"commit_{sha}"
        message = detail.get("message") or ""
        headline = message.splitlines()[0] if message else sha
        edges = [_edge(commit_id, repo_id, "TOUCHES", REPORTED),
                 _edge(f"person_{login}", commit_id, "AUTHORED", REPORTED), *_closes(message, commit_id)]
        documents.append(_document(
            commit_id, _excerpt(headline, COMMIT_LABEL_CHARS), "commit", commit.get("html_url"),
            _excerpt(f"Commit {sha} by {login}: {message}"), edges,
        ))
        authored.setdefault(login, []).append(commit_id)

    for login, work in authored.items():
        person_id = f"person_{login}"
        documents.append(_document(
            person_id, login, "person", f"https://github.com/{login}",
            f"{login} authored {len(work)} of the sampled pull requests and commits in {repo}.",
            [_edge(person_id, item, "AUTHORED", REPORTED) for item in work],
        ))
    return documents


def _terms(text: str) -> set[str]:
    return _base_tokens(text, stop=IGNORED_TERMS, pattern=TERM_RE)


def _ranked(corpus: list, question: set[str]) -> list[tuple[float, Any]]:
    """Each document with the share of the question's terms it contains.

    Best share first. On an equal share the better-connected document goes
    first, since it gives the neighbour step more to follow; after that, corpus
    order stands.
    """
    shares = [(round(len(question & _terms(d.page_content)) / len(question), 3), d) for d in corpus]
    return sorted(shares, key=lambda pair: (-pair[0], -len(pair[1].metadata.get("edges", ()))))


def _related(seeds: list[tuple[float, Any]], by_id: dict[str, tuple[float, Any]]):
    """Documents one relation away from a seed, seed by seed."""
    for _, document in seeds:
        for edge in document.metadata.get("edges", ()):
            for end in (edge["source"], edge["target"]):
                if end in by_id:
                    yield by_id[end]


def lexical_graph_retriever(corpus: list, k: int = 10):
    """Question-term matches, then what they are related to, up to ``k``."""
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class LexicalGraphRetriever(BaseRetriever):
        """Documents that share a term with the question fill up to two
        thirds of ``k``, leaving room for relations. Their direct relations
        come next, strongest first, so a related document that also matches
        beats one that is merely attached; then the best of the rest. A
        related document keeps its own score, which may be none; an edge is
        returned only when both of its ends were."""

        corpus: list
        k: int = 10

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list:
            ranked = _ranked(self.corpus, _terms(query) or {"?"})
            by_id = {pair[1].metadata["id"]: pair for pair in ranked}
            seeds = [pair for pair in ranked if pair[0] > 0][:max(1, self.k - self.k // 3)]
            related = sorted(_related(seeds, by_id), key=lambda pair: -pair[0])

            chosen: dict[str, tuple[float, Any]] = {}
            for pair in (*seeds, *related, *ranked):
                if len(chosen) >= self.k:
                    break
                chosen.setdefault(pair[1].metadata["id"], pair)

            results = []
            for score, document in sorted(chosen.values(), key=lambda pair: -pair[0]):
                metadata = {key: value for key, value in document.metadata.items() if key != "edges"}
                metadata["edges"] = [edge for edge in document.metadata.get("edges", ())
                                     if edge["source"] in chosen and edge["target"] in chosen]
                if score > 0:
                    # Present only when the document shares a question term;
                    # the relations alone brought the others in.
                    metadata["score"] = score
                results.append(Document(page_content=document.page_content, metadata=metadata))
            return results

    return LexicalGraphRetriever(corpus=corpus, k=k)


def _summary(repo: str, documents: list) -> str:
    """The run's answer, put together from what was retrieved."""
    if not documents:
        return f"No recent evidence was retrieved from {repo}."
    lead = documents[0]
    counts = Counter(document.metadata["kind"] for document in documents)
    inventory = ", ".join(f"{kind}: {count}" for kind, count in sorted(counts.items()))
    return (
        f"In the sampled {repo} activity, {lead.metadata['label']} ranked first. "
        f"Evidence: {lead.page_content[:160].strip()}. The run retrieved {len(documents)} "
        f"connected items ({inventory}); inspect the trace for their scores and links."
    )


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

    def summarize(state: State, config) -> dict:
        return {"answer": _summary(repo, state["docs"])}

    workflow = StateGraph(State)
    for step, action in (("retrieve", retrieve), ("summarize", summarize)):
        workflow.add_node(step, action)
    for start, end in ((START, "retrieve"), ("retrieve", "summarize"), ("summarize", END)):
        workflow.add_edge(start, end)

    tracer = LangGraphTracer()
    result = workflow.compile().invoke({"question": question}, config={"callbacks": [tracer]})
    trace = tracer.finish(query=question, answer=result["answer"])
    # Recording and scoring are separate steps: scoring is what says which of
    # the retrieved items the answer drew on.
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

    parser.add_argument("repository", help="the repository to sample, as owner/name")
    parser.add_argument("question", nargs="?", default=None,
                        help="what to ask of the sample; by default, what changed recently and who drove it")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"),
                        help="a GitHub token (GITHUB_TOKEN is read if set); private repositories need one")
    for flag, what in (("--prs", "pull requests"), ("--issues", "issues"), ("--commits", "commits")):
        parser.add_argument(flag, type=int, default=DEFAULT_FETCH,
                            help=f"how many of the most recent {what} to sample (default {DEFAULT_FETCH})")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                        help=f"how many documents the agent retrieves (default {DEFAULT_TOP})")
    parser.add_argument("--out", type=Path, default=Path("trace_out"),
                        help="where to write the two JSON files (default trace_out)")


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
