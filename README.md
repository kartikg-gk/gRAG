# graphRAG

A repository knowledge graph, and a trace of what retrieval actually used.

## Running the tests

**The suite requires WSL.** Entity resolution embeds text with
`sentence-transformers`, which needs torch, and torch cannot be imported by
Windows-side Python on this machine — Smart App Control blocks it. WSL is
unaffected. Nothing about that policy needs changing.

There is no lighter path. The n-gram embedder that used to make a Windows-only
run possible has been removed, so this is the only supported way to run the
suite:

```bash
wsl -d Ubuntu bash -lc "cd ~/vllm-tutorial && source .venv/bin/activate && cd '/mnt/e/AI projects/graphRAG' && python -m pytest"
```

Running `python -m pytest` from Windows will fail at import, not at a test.

### What the test environment needs

```
pip install -e ".[dev]"
```

`dev` pulls in `sentence-transformers`, `spacy`, `langgraph` and
`langchain-core`. The first run downloads `all-MiniLM-L6-v2` (about 90 MB) and
caches it under `~/.cache/huggingface`; later runs load from that cache.

The model is loaded once per process and reused — see `_load_model` in
`src/analysis/similarity.py`. A per-instance load would dominate the run.

## Commands

```bash
graphrag ingest owner/name        # fetch a repository and build its graph
graphrag view trace.json          # render a saved trace
```

## Known results worth reading before trusting resolution

With the embedder, model and thresholds in use, on the demo corpus:

- **`#413` and `#414` merge.** They are different tickets. The model scores
  them 0.9521 — near-identical text reads as near-identical meaning, which is
  inverted for identifiers.
- **`notification-service` and `notification_service` no longer merge.** They
  score 0.9041, inside the ask-a-model band, and no judge is configured.
- **`pull request #9999` resolves to `pull request #1347`**, a pull request
  that was never ingested reaching one that was.

These are pinned by tests rather than tuned away. See
`test_similarity.py`.
