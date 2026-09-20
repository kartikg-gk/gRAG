import { getAuthToken } from "@/lib/authToken";
import { layoutGraph } from "@/lib/layout";
import type {
  EntityType,
  ExecutionStep,
  RouterConfidence,
  TraceEdge,
  TraceNode,
  TraceState,
} from "@/types/trace";

export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const API_KEY: string | undefined = import.meta.env.VITE_API_KEY;

export interface ApiDocument {
  doc_id: string | null;
  content: string | null;
  path: string | null;
}

export interface ApiResult {
  id: string;
  label: string | null;
  type: string | null;
  score_total: number;
  score_vector: number;
  score_graph: number;
  recency: number;
  age_days: number | null;
  documents: ApiDocument[];
  page_content: string;
}

export interface ApiGraphHop {
  from_id: string;
  to_id: string;
  confidence: number;
  relation: string;
}

export interface ApiTraceLog {
  intent: { alpha: number; beta: number; type: string };
  execution_path: {
    linked_seeds: string[];
    vector_seeds: string[];
    graph_hops: ApiGraphHop[];
  };
  recency: {
    enabled: boolean;
    floor: number;
    applied: { id: string; age_days: number; factor: number }[];
  };
  metrics: {
    graph_hits: number;
    vector_k: number;
    total_nodes_evaluated: number;
  };
}

export interface ApiTraceResponse {
  query: string;
  results: ApiResult[];
  trace_log: ApiTraceLog;
  context: string;
  trace_id?: string | null;
}

export interface ApiSubgraphNode {
  id: string;
  label: string | null;
  type: string | null;
  requested: boolean;
}

export interface ApiSubgraphEdge {
  source: string;
  target: string;
  confidence: number;
  relation: string;
}

export interface ApiSubgraph {
  nodes: ApiSubgraphNode[];
  edges: ApiSubgraphEdge[];
}

export interface Suggestion {
  query: string;
  entity: string;
  type: string;
}

export interface ApiGraph {
  id: string;
  label: string;
  active: boolean;
}

export interface ApiGraphsResponse {
  graphs: ApiGraph[];
  active: string | null;
}

export interface ApiGraphSwitchResponse {
  active: string;
  label: string;
  nodes: number;
}

export interface ApiHealthResponse {
  status: string;
  nodes: number;
}

export interface ApiSummaryResponse {
  summary: string;
  cached: boolean;
  error?: string;
}

export interface ApiAnswerResponse {
  answer: string;
  cached: boolean;
  error?: string;
}

export interface SessionSummary {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
}

export interface TraceRecord {
  id: string;
  session_id: string;
  query: string;
  execution_plan: ApiTraceLog;
  graph_payload: ApiResult[] | Record<string, unknown>;
  created_at: string;
  graph_id?: string | null;
}

async function headersFor(init: RequestInit): Promise<Headers> {
  const headers = new Headers(init.headers);
  if (init.body !== undefined) headers.set("Content-Type", "application/json");
  const token = await getAuthToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (API_KEY) headers.set("X-Graphrag-API-Key", API_KEY);
  return headers;
}

function httpError(path: string, response: Response): Error {
  return new Error(`${path} → HTTP ${response.status} ${response.statusText}`);
}

async function rawRequest(path: string, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: await headersFor(init),
  });
  if (!response.ok) throw httpError(path, response);
  return response;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  return (await rawRequest(path, init)).json() as Promise<T>;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export function fetchTrace(query: string, sessionId?: string, topK?: number): Promise<ApiTraceResponse> {
  return post<ApiTraceResponse>("/api/trace", {
    query,
    ...(topK === undefined ? {} : { top_k: topK }),
    ...(sessionId === undefined ? {} : { session_id: sessionId }),
  });
}

export function fetchSubgraph(nodeIds: string[]): Promise<ApiSubgraph> {
  return post<ApiSubgraph>("/api/subgraph", { node_ids: nodeIds });
}

export async function fetchSuggestions(limit = 5): Promise<Suggestion[]> {
  try {
    const body = await request<{ suggestions: Suggestion[] }>(
      `/api/suggestions?limit=${encodeURIComponent(limit)}`,
    );
    return body.suggestions;
  } catch {
    return [];
  }
}

export function fetchGraphs(): Promise<ApiGraphsResponse> {
  return request<ApiGraphsResponse>("/api/graphs");
}

export function switchGraph(id: string): Promise<ApiGraphSwitchResponse> {
  return post<ApiGraphSwitchResponse>("/api/graphs/switch", { id });
}

export function fetchHealth(): Promise<ApiHealthResponse> {
  return request<ApiHealthResponse>("/api/health");
}

const summaryCache = new Map<string, string>();
const answerCache = new Map<string, string>();

export async function summarize(key: string, text: string): Promise<ApiSummaryResponse> {
  const cached = summaryCache.get(key);
  if (cached) return { summary: cached, cached: true };
  const response = await post<ApiSummaryResponse>("/api/summarize", { key, text });
  if (response.summary) summaryCache.set(key, response.summary);
  return response;
}

const answerKey = (query: string, context: string) => `${query}::${context.length}`;

export async function answer(query: string, context: string): Promise<ApiAnswerResponse> {
  const key = answerKey(query, context);
  const cached = answerCache.get(key);
  if (cached) return { answer: cached, cached: true };
  const response = await post<ApiAnswerResponse>("/api/answer", { query, context });
  if (response.answer) answerCache.set(key, response.answer);
  return response;
}

export async function streamAnswer(
  query: string,
  context: string,
  onToken: (text: string) => void,
): Promise<string> {
  const key = answerKey(query, context);
  const cached = answerCache.get(key);
  if (cached) {
    onToken(cached);
    return cached;
  }
  const response = await rawRequest("/api/answer/stream", {
    method: "POST",
    body: JSON.stringify({ query, context }),
  });
  if (!response.body) {
    const fallback = await answer(query, context);
    onToken(fallback.answer);
    return fallback.answer;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let cumulative = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    cumulative += decoder.decode(value, { stream: true });
    onToken(cumulative);
  }
  cumulative += decoder.decode();
  if (cumulative) {
    onToken(cumulative);
    answerCache.set(key, cumulative);
  }
  return cumulative;
}

export function createSession(title: string, email?: string, sessionId?: string): Promise<SessionSummary> {
  return post<SessionSummary>("/api/sessions", {
    title,
    ...(sessionId === undefined ? {} : { session_id: sessionId }),
    ...(email === undefined ? {} : { email }),
  });
}

export function listSessions(): Promise<SessionSummary[]> {
  return request<SessionSummary[]>("/api/sessions");
}

export function fetchSessionTraces(sessionId: string): Promise<TraceRecord[]> {
  return request<TraceRecord[]>(`/api/sessions/${encodeURIComponent(sessionId)}/traces`);
}

const KNOWN_TYPES: EntityType[] = [
  "PR", "Commit", "File", "Person", "Document", "Repo", "Ticket", "Service", "Library", "Team", "Tool",
];

export function normalizeType(type: string | null | undefined): EntityType {
  const wanted = (type ?? "").toLowerCase();
  return KNOWN_TYPES.find((known) => known.toLowerCase() === wanted) ?? "Document";
}

export function extractNodeIds(log: ApiTraceLog | null | undefined): string[] {
  if (!log) return [];
  const ids = new Set<string>();
  for (const id of log.execution_path.vector_seeds) ids.add(id);
  for (const id of log.execution_path.linked_seeds) ids.add(id);
  for (const hop of log.execution_path.graph_hops) {
    ids.add(hop.from_id);
    ids.add(hop.to_id);
  }
  return [...ids];
}

export function allTraceNodeIds(response: ApiTraceResponse): string[] {
  const ids = new Set(extractNodeIds(response.trace_log));
  for (const result of response.results) ids.add(result.id);
  return [...ids];
}

const SNIPPET_LIMIT = 240;

export function resultSnippet(result: ApiResult): string | undefined {
  const document = result.documents.find((item) => (item.content ?? "").trim() !== "");
  const text = (document?.content ?? result.page_content).trim();
  if (!text) return undefined;
  return text.length > SNIPPET_LIMIT ? `${text.slice(0, SNIPPET_LIMIT)}…` : text;
}

export function resultSummaryText(result: ApiResult): string | undefined {
  const document = result.documents.find((item) => (item.content ?? "").trim() !== "");
  return (document?.content ?? result.page_content).trim() || undefined;
}

export function resultSourceUrl(result: ApiResult): string | undefined {
  return result.documents.find((item) => /^https?:\/\//.test(item.path ?? ""))?.path ?? undefined;
}

export function deriveConfidence(log: ApiTraceLog | null | undefined): RouterConfidence {
  const hops = log?.execution_path.graph_hops ?? [];
  if (hops.length === 0) {
    return {
      score: 0.78,
      uncertainty: 0.08,
      rationale: "No graph hop corroborated the vector arm, so this confidence comes from a fixed prior.",
    };
  }
  const values = hops.map((hop) => hop.confidence);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const spread = (Math.max(...values) - Math.min(...values)) / 2;
  return {
    score: Math.min(1, Math.max(0, mean)),
    uncertainty: Math.min(0.2, Math.max(0.02, spread || 0.03)),
    rationale: `Averaged over ${hops.length} graph hop${hops.length === 1 ? "" : "s"} (mean confidence ${mean.toFixed(2)}). The band comes from the spread over the walked edges.`,
  };
}

const plural = (count: number) => (count === 1 ? "" : "s");

export function buildSteps(log: ApiTraceLog): ExecutionStep[] {
  const seeds = log.execution_path.vector_seeds.length;
  const hops = log.execution_path.graph_hops;
  const visited = log.metrics.total_nodes_evaluated;
  const relations = new Map<string, number>();
  for (const hop of hops) relations.set(hop.relation, (relations.get(hop.relation) ?? 0) + 1);
  const relationSummary = [...relations.entries()]
    .sort((left, right) => right[1] - left[1])
    .map(([relation, count]) => `${count}× ${relation}`)
    .join(", ");
  const aged = log.recency.applied.filter((row) => row.factor < 0.95);
  const oldest = aged.reduce<(typeof aged)[number] | null>(
    (current, row) => (current === null || row.age_days > current.age_days ? row : current),
    null,
  );
  return [
    {
      id: "vector-retrieval", index: 1, title: "Vector Retrieval",
      detail: `Pulled ${seeds} semantic chunk${plural(seeds)}.`,
      status: "complete", badge: `${seeds} seed${plural(seeds)}`, arm: "vector",
    },
    {
      id: "graph-walk", index: 2, title: "Graph Walk",
      detail: relationSummary
        ? `Followed ${hops.length} edge${plural(hops.length)}: ${relationSummary}.`
        : `Followed ${hops.length} edge${plural(hops.length)} through the relationship graph.`,
      status: "complete", badge: `${hops.length} hop${plural(hops.length)}`, arm: "graph",
    },
    oldest
      ? {
          id: "context-build", index: 3, title: "Context Build",
          detail: `Scored ${visited} node${plural(visited)}; age decay demoted ${aged.length} of them (oldest ${Math.round(oldest.age_days)} days, ×${oldest.factor.toFixed(2)}).`,
          status: "complete", badge: `${aged.length} aged`,
        }
      : {
          id: "context-build", index: 3, title: "Context Build",
          detail: `Scored ${visited} node${plural(visited)} to build the grounded context.`,
          status: "complete", badge: `${visited} nodes`,
        },
  ];
}

const pairKey = (left: string, right: string) => `${left}\u0000${right}`;

export function adaptToTraceState(
  query: string,
  response: ApiTraceResponse,
  subgraph: ApiSubgraph,
  elapsedMs: number,
): TraceState {
  const log = response.trace_log;
  const results = new Map(response.results.map((result) => [result.id, result]));
  const hops = log.execution_path.graph_hops;
  const linkedSeeds = new Set(log.execution_path.linked_seeds);
  const tracedIds = extractNodeIds(log);
  const emptyPath = tracedIds.length === 0;
  const active = new Set(emptyPath ? response.results.map((result) => result.id) : tracedIds);
  const resultData = (id: string) => {
    const result = results.get(id);
    if (!result) return { similarity: undefined, score: undefined, meta: {} };
    return {
      similarity: result.score_vector,
      score: result.score_total,
      meta: {
        snippet: resultSnippet(result), summaryText: resultSummaryText(result), scoreGraph: result.score_graph,
        sourceUrl: resultSourceUrl(result), recency: result.recency, ageDays: result.age_days,
        linkedSeed: linkedSeeds.has(id),
      },
    };
  };
  const nodes: TraceNode[] = subgraph.nodes.map((node) => {
    const result = results.get(node.id);
    const scored = resultData(node.id);
    const rawType = node.type ?? result?.type;
    return {
      id: node.id, label: node.label ?? result?.label ?? node.id, type: normalizeType(rawType),
      active: node.requested || active.has(node.id), position: { x: 0, y: 0 },
      similarity: scored.similarity, score: scored.score,
      meta: { subtitle: rawType ?? undefined, linkedSeed: linkedSeeds.has(node.id), ...scored.meta },
    };
  });
  const present = new Set(nodes.map((node) => node.id));
  for (const id of active) {
    if (present.has(id)) continue;
    const result = results.get(id);
    const scored = resultData(id);
    nodes.push({
      id, label: result?.label ?? id, type: normalizeType(result?.type), active: true,
      position: { x: 0, y: 0 }, similarity: scored.similarity, score: scored.score,
      meta: { subtitle: result?.type ?? "Inferred from trace", linkedSeed: linkedSeeds.has(id), ...scored.meta },
    });
    present.add(id);
  }
  const hopKeys = new Set(hops.map((hop) => pairKey(hop.from_id, hop.to_id)));
  const isTraced = (source: string, target: string) =>
    hopKeys.has(pairKey(source, target)) || hopKeys.has(pairKey(target, source));
  const edges: TraceEdge[] = subgraph.edges.map((edge, index) => ({
    id: `e-${edge.source}-${edge.target}-${index}`, source: edge.source, target: edge.target,
    confidence: edge.confidence, relation: edge.relation,
    active: isTraced(edge.source, edge.target) ||
      (emptyPath && active.has(edge.source) && active.has(edge.target)),
  }));
  const rendered = new Set(edges.map((edge) => pairKey(edge.source, edge.target)));
  hops.forEach((hop, index) => {
    const covered = rendered.has(pairKey(hop.from_id, hop.to_id)) ||
      rendered.has(pairKey(hop.to_id, hop.from_id));
    if (covered || !present.has(hop.from_id) || !present.has(hop.to_id)) return;
    edges.push({
      id: `hop-${hop.from_id}-${hop.to_id}-${index}`, source: hop.from_id, target: hop.to_id,
      confidence: hop.confidence, relation: hop.relation, active: true,
    });
    rendered.add(pairKey(hop.from_id, hop.to_id));
  });
  const connections = new Map<string, number>();
  for (const edge of edges) {
    connections.set(edge.source, (connections.get(edge.source) ?? 0) + 1);
    connections.set(edge.target, (connections.get(edge.target) ?? 0) + 1);
  }
  for (const node of nodes) node.meta = { ...node.meta, connections: connections.get(node.id) ?? 0 };

  return {
    id: response.trace_id ?? `trace_${Date.now().toString(36)}`,
    query,
    computedAt: new Date().toISOString(),
    weights: {
      vector: log.intent.alpha,
      graph: log.intent.beta,
      intent: log.intent.type === "semantic" ? "conceptual" : "relational",
    },
    confidence: deriveConfidence(log),
    steps: buildSteps(log),
    metrics: {
      tokens: { used: 0, budget: 0, reductionPct: 0 }, peakRamGb: 0,
      queryTimeSec: +(elapsedMs / 1000).toFixed(2),
      nodesEvaluated: log.metrics.total_nodes_evaluated,
    },
    graph: layoutGraph(nodes, edges),
    context: response.context,
  };
}

const EMPTY_SUBGRAPH: ApiSubgraph = { nodes: [], edges: [] };

export async function runTraceQuery(query: string, sessionId?: string): Promise<TraceState> {
  const started = performance.now();
  const response = await fetchTrace(query, sessionId);
  const ids = allTraceNodeIds(response);
  const subgraph = ids.length === 0 ? EMPTY_SUBGRAPH : await fetchSubgraph(ids);
  return adaptToTraceState(query, response, subgraph, performance.now() - started);
}

function historyResults(payload: TraceRecord["graph_payload"]): ApiResult[] {
  if (Array.isArray(payload)) return payload;
  const results = payload.results;
  return Array.isArray(results) ? (results as ApiResult[]) : [];
}

export async function hydrateTraceFromLog(record: TraceRecord): Promise<TraceState> {
  const response: ApiTraceResponse = {
    query: record.query,
    results: historyResults(record.graph_payload),
    trace_log: record.execution_plan,
    context: "",
    trace_id: record.id,
  };
  const ids = allTraceNodeIds(response);
  const subgraph = ids.length === 0 ? EMPTY_SUBGRAPH : await fetchSubgraph(ids);
  const state = adaptToTraceState(record.query, response, subgraph, 0);
  return { ...state, computedAt: record.created_at };
}
