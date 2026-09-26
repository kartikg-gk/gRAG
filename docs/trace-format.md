# graphRAG trace JSON, version 4

The trace is one UTF-8 JSON object per agent run. `schema_version` is `4`.
The Python package can read versions 1–3; new producers should write version 4.
The machine-readable contract is [`trace.schema.json`](trace.schema.json).

| Field | Meaning |
| --- | --- |
| `query` | Original user question. |
| `answer` | Final answer, or `null` if unavailable. |
| `producer` | Adapter or application name. |
| `started_at`, `duration_ms` | Run start (ISO 8601) and elapsed milliseconds. |
| `graph.nodes` | Nodes with ID, label, kind, content, source, optional URL and metadata, scores, and answer overlap. |
| `graph.edges` | Directed source/target IDs, relation, and optional weight. |
| `retrievals` | Each retriever invocation: query, arm (`vector`, `graph`, `hybrid`, or `unknown`), items, and edges. The items carry the retrieval scores. |
| `spans` | Execution timeline. Each span has an ID, name, kind, optional parent ID, start/end offsets in milliseconds, and status. |
| `metrics` | Numeric run measurements. `duration_ms` is always emitted. Producers may add numeric measurements. |

Item scores are measurements, not verdicts. `score` is the combined retrieval
score; `vector_score` and `graph_score` are optional arm scores. `overlap` is
the fraction of unique non-stopword source tokens present in the answer. The
viewer calls an item **used** at overlap ≥ 0.2, **ignored** below 0.2, and
**unclassified** when overlap and answer are absent. It computes a missing
overlap in the browser when the answer and item content are present. The JSON
never stores a `used` boolean, so a new threshold can reclassify old runs.

`graph.nodes` and `graph.edges` may include context beyond retrieved items.
When absent from a version 4 producer, the Python serializer derives them from
the retrievals. A graph node uses the same shape as a retrieved item; an edge
uses the same shape inside and outside a retrieval.

Generate and save a trace from LangGraph:

```python
from graphrag.adapters.langgraph import LangGraphTracer, save_run

tracer = LangGraphTracer()
result = graph.invoke(inputs, config={"callbacks": [tracer]})
trace, path = save_run(tracer, query=inputs["query"], answer=result["answer"])
```

Install `graphweave[adapter]` for this adapter. `graphweave trace.json`
or `graphweave directory/` opens the local viewer. `graphweave-view` is an alias;
both accept `--port` and `--no-browser`. The viewer needs no backend service.
For a repository-to-trace run, `graphweave-github-trace owner/repo "question"`
fetches recent GitHub activity, builds an in-memory graph, and writes
`graphweave_out/trace_state.json` in this v4 format. `GITHUB_TOKEN` is optional
for public repositories and required for private repositories. This local
command uses lexical graph retrieval and a cited extractive answer; it does
not invoke the optional embedding model, graph database, or remote judge.
Add `--source` to fetch a bounded sample of source files at HEAD and cite
immutable commit URLs with line numbers. The default command examines recent
activity only, so neither mode claims to search every file or all history.
The separate `graphweave-backend` command retains the ingestion and graph memory
platform. Install `graphweave[backend,api,graph,embeddings,judge]` to
run that advanced mode; none of those extras are needed for local trace viewing.

For a static deployment, build `frontend/` and deploy `frontend/dist/`. Users
can drop or paste a trace, or link to `?src=https://host/path/trace.json`.
The remote JSON server must permit browser cross-origin requests. No trace is
uploaded by the drag/drop or paste flows.
