"""Reject known backend clients, backend URLs, and key formats in the static UI."""

import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("frontend/dist")
rules = {
    "backend client": r"VITE_GRAPHRAG_API_URL|/api/trace\b|/api/answer\b|/api/graphs\b|ClerkProvider",
    "backend URL": r"https?://(?:localhost|127\.0\.0\.1):(?:8000|8080)|api\.openai\.com|openrouter\.ai/api",
    "secret/key": r"sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|pk_(?:test|live)_[A-Za-z0-9]{20,}",
}
assets = [path for path in root.rglob("*") if path.suffix in {".js", ".html", ".css"}]
if not assets:
    raise SystemExit(f"No production assets found in {root}")
failures = []
for asset in assets:
    content = asset.read_text(encoding="utf-8")
    for name, pattern in rules.items():
        if re.search(pattern, content):
            # Print only the rule and filename; never echo a possible secret.
            failures.append(f"{asset}: {name}")
if failures:
    raise SystemExit("\n".join(failures))
print(f"Checked {len(assets)} static assets: no known backend clients, URLs, or key formats")
