import { layoutGraph } from "@/lib/layout";
import type { TraceEdge, TraceNode, TraceState } from "@/types/trace";

const nodes: TraceNode[] = [
  { id: "person:river", label: "River Chen", type: "Person", active: true, score: 0.912, similarity: 0.884, position: { x: 0, y: 0 }, meta: { subtitle: "Contributor", scoreGraph: 0.937, connections: 2, linkedSeed: true, ageDays: 2, recency: 0.99, snippet: "Investigated the authentication regression and prepared the corrective pull request.", summaryText: "Investigated the authentication regression and prepared the corrective pull request." } },
  { id: "pr:142", label: "PR #142", type: "PR", active: true, score: 0.934, similarity: 0.902, position: { x: 0, y: 0 }, meta: { subtitle: "Pull Request", scoreGraph: 0.961, connections: 3, linkedSeed: true, ageDays: 2, recency: 0.99, snippet: "Corrects token refresh handling and adds regression coverage for expired sessions.", summaryText: "Corrects token refresh handling and adds regression coverage for expired sessions." } },
  { id: "repo:atlas", label: "atlas-web", type: "Repo", active: true, score: 0.801, position: { x: 0, y: 0 }, meta: { subtitle: "Repository", connections: 1, linkedSeed: false } },
  { id: "file:session", label: "session-manager.ts", type: "File", active: true, score: 0.726, position: { x: 0, y: 0 }, meta: { subtitle: "File", connections: 2, linkedSeed: false, snippet: "Session refresh utilities used by the application authentication boundary.", summaryText: "Session refresh utilities used by the application authentication boundary." } },
  { id: "doc:runbook", label: "Authentication runbook", type: "Document", active: false, score: 0.418, position: { x: 0, y: 0 }, meta: { subtitle: "Document", connections: 1, linkedSeed: false, snippet: "Operational notes for diagnosing login, token refresh, and session expiry incidents.", summaryText: "Operational notes for diagnosing login, token refresh, and session expiry incidents." } },
  { id: "ticket:204", label: "Ticket #204", type: "Ticket", active: false, position: { x: 0, y: 0 }, meta: { subtitle: "Ticket", connections: 1, linkedSeed: false } },
];

const edges: TraceEdge[] = [
  { id: "e-author", source: "person:river", target: "pr:142", confidence: 0.94, relation: "AUTHORED", active: true },
  { id: "e-merge", source: "pr:142", target: "repo:atlas", confidence: 0.88, relation: "MERGED_INTO", active: true },
  { id: "e-touch", source: "pr:142", target: "file:session", confidence: 0.91, relation: "TOUCHES", active: true },
  { id: "e-mention", source: "doc:runbook", target: "file:session", confidence: 0.71, relation: "MENTIONS", active: false },
  { id: "e-ticket", source: "ticket:204", target: "pr:142", confidence: 0.76, relation: "RESOLVED_BY", active: false },
];

export const sampleTrace: TraceState = {
  id: "sample-trace-frontend", query: "Who corrected the authentication regression?",
  computedAt: "2026-09-19T00:00:00Z", weights: { vector: 0.2, graph: 0.8, intent: "relational" },
  confidence: { score: 0.91, uncertainty: 0.04, rationale: "Sample trace for offline development." },
  steps: [], metrics: { queryTimeSec: 0.24, nodesEvaluated: 6 }, graph: layoutGraph(nodes, edges),
  context: "Generic local development fixture derived from the frontend API fixture shape.",
};
