"""Bounded source indexing through GitHub's authenticated Git objects API.

Files are read as data. No checkout, archive extraction, imports, or repository
commands run. Citations use an immutable commit and exact chunk line numbers.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import PurePosixPath
from urllib.parse import quote

from .github import GitHubError, _request
from ..analysis.chunking import windows
from ..knowledge.documents import SourceDocument
from ..tracing._text import tokens

EXTENSIONS = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
    ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".scala", ".sh", ".sql", ".md", ".rst", ".txt", ".toml",
})
EXCLUDED_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "vendor", "dist", "build",
    "__pycache__", ".next", "coverage", ".ssh", "secrets", "credentials",
})
MAX_FILE_BYTES = 128 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024


def eligible(entry: dict) -> bool:
    path = PurePosixPath(entry.get("path", ""))
    lowered = {part.lower() for part in path.parts}
    name = path.name.lower()
    return (
        entry.get("type") == "blob"
        and entry.get("mode") in {"100644", "100755"}
        and not path.is_absolute() and ".." not in path.parts
        and not (lowered & EXCLUDED_DIRS)
        and not name.startswith((".env", "id_rsa", "id_ed25519"))
        and "secret" not in name and "credential" not in name
        and path.suffix.lower() in EXTENSIONS
        and 0 < entry.get("size", 0) <= MAX_FILE_BYTES
    )


def index_source(session, repo: str, question: str, graph, documents: list,
                 *, max_files: int = 20) -> dict[str, int]:
    """Append selected files/chunks to the existing graph and source corpus."""
    commit = _request(session, f"/repos/{repo}/commits/HEAD").json()
    revision = commit["sha"]
    tree_sha = commit["commit"]["tree"]["sha"]
    tree = _request(session, f"/repos/{repo}/git/trees/{tree_sha}", {"recursive": "1"}).json()
    candidates = [entry for entry in tree["tree"] if eligible(entry)]
    query_words = tokens(question)
    candidates.sort(key=lambda entry: (
        -len(query_words & tokens(entry["path"].replace("/", " ").replace("_", " "))),
        entry["path"],
    ))
    metrics = {"source_files_indexed": 0, "source_files_eligible": len(candidates),
               "source_bytes": 0, "source_tree_truncated": int(bool(tree.get("truncated")))}
    for entry in candidates[:max_files]:
        if metrics["source_bytes"] + entry["size"] > MAX_TOTAL_BYTES:
            break
        blob = _request(session, f"/repos/{repo}/git/blobs/{entry['sha']}").json()
        if blob.get("encoding") != "base64":
            raise GitHubError("GitHub source blob did not use base64 encoding")
        encoded = blob.get("content", "")
        if not isinstance(encoded, str):
            raise GitHubError("GitHub source blob contained invalid base64")
        if len(encoded) > MAX_FILE_BYTES * 2:
            raise GitHubError("GitHub source blob exceeded the file size limit")
        try:
            raw = base64.b64decode("".join(encoded.split()), validate=True)
        except (binascii.Error, TypeError, ValueError) as exc:
            raise GitHubError("GitHub source blob contained invalid base64") from exc
        if len(raw) > MAX_FILE_BYTES or b"\0" in raw:
            continue
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            continue
        path = entry["path"]
        node_id = f"file:{path}"
        url = f"https://github.com/{repo}/blob/{revision}/{quote(path, safe='/')}"
        node = graph.nodes.setdefault(node_id, {"id": node_id, "type": "File", "timestamp": None})
        node.update(path=path, url=url, revision=revision, indexed_source=True)
        repo_id = f"repo:{repo}"
        if not any(edge["source"] == node_id and edge["target"] == repo_id
                   and edge["type"] == "PART_OF" for edge in graph.edges):
            graph.edges.append({"source": node_id, "target": repo_id,
                                "type": "PART_OF", "confidence": 0.8, "timestamp": None})
        for number, chunk in enumerate(windows(text, size=90, overlap=15)):
            first = text.count("\n", 0, chunk.start) + 1
            last = text.count("\n", 0, chunk.end) + 1
            documents.append(SourceDocument(
                id=f"doc:{node_id}:{number}", path=f"{url}#L{first}-L{last}",
                content=chunk.text, origin=node_id, field="source_code",
            ))
        metrics["source_files_indexed"] += 1
        metrics["source_bytes"] += len(raw)
    return metrics
