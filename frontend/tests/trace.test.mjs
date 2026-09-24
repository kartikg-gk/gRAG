import assert from "node:assert/strict";
import test from "node:test";
import { parseTrace, toGraphState, overlap } from "../.test-build/viewer/trace.js";

const item = { id: "one", kind: "File", content: "cache expiry", source: "github", score: 0.8 };
const trace = { schema_version: 4, query: "cache?", answer: "cache expiry", retrievals: [{ items: [item], edges: [] }] };

test("v4 preserves scores, file kinds, and measured overlap", () => {
  const parsed = parseTrace(trace);
  assert.equal(toGraphState(parsed).graph.nodes[0].score, 0.8);
  assert.equal(toGraphState(parsed).graph.nodes[0].type, "File");
  assert.equal(overlap(item, trace.answer), 1);
  assert.equal(overlap(item), null);
});

test("malformed nested input is rejected before rendering", () => {
  for (const invalid of [
    { ...trace, retrievals: [null] },
    { ...trace, retrievals: [{ items: [{}] }] },
    { ...trace, answer: {} },
    { ...trace, metrics: { value: {} } },
    { ...trace, schema_version: 3.5 },
    { ...trace, items: [{ ...item, score: "oops" }] },
  ]) assert.throws(() => parseTrace(invalid));
});

test("explicit empty graph is preserved", () => {
  assert.equal(toGraphState({ ...trace, graph: { nodes: [], edges: [] } }).graph.nodes.length, 0);
});

test("unsafe citations and dangling graph edges never reach the canvas", () => {
  const state = toGraphState({ ...trace, graph: {
    nodes: [{ ...item, source_uri: "javascript:alert(1)" }],
    edges: [{ source: "one", target: "missing", relation: "LINK" }],
  } });
  assert.equal(state.graph.nodes[0].meta.sourceUrl, undefined);
  assert.equal(state.graph.edges.length, 0);
});
