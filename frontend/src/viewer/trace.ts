import type { EntityType, TraceState } from "@/types/trace";

export interface TraceItem {
  id: string; label?: string | null; kind?: string; content: string; source: string;
  source_uri?: string | null; score?: number | null; vector_score?: number | null;
  graph_score?: number | null; overlap?: number | null; metadata?: Record<string, unknown>;
}
export interface TraceEdge { source: string; target: string; relation: string; weight?: number | null }
export interface TraceSpan { id: string; name: string; kind: string; parent_id?: string | null;
  start_ms?: number | null; end_ms?: number | null; status?: string }
export interface TraceFile {
  schema_version: number; query: string; answer?: string | null; producer?: string;
  started_at?: string | null; duration_ms?: number | null;
  graph?: { nodes: TraceItem[]; edges: TraceEdge[] };
  retrievals?: Array<{ query?: string; arm?: string; items: TraceItem[]; edges?: TraceEdge[] }>;
  items?: TraceItem[]; edges?: TraceEdge[]; spans?: TraceSpan[];
  metrics?: Record<string, number | null>;
}

const kinds = new Set(["PR", "Commit", "File", "Person", "Document", "Repo", "Ticket", "Service", "Library", "Team", "Tool"]);
const stop = new Set(["the", "and", "for", "that", "this", "with", "from", "was", "were", "are", "has", "have", "not", "but", "its", "also", "into", "over", "after"]);
function safeHttpUrl(value?: string | null): string | undefined {
  if (!value) return undefined;
  try { const url = new URL(value); return url.protocol === "http:" || url.protocol === "https:" ? url.href : undefined; }
  catch { return undefined; }
}
function tokens(text: string): Set<string> {
  return new Set((text.match(/[a-z0-9#][a-z0-9_#-]{2,}/gi) ?? []).map((word) => word.toLowerCase()).filter((word) => !stop.has(word)));
}
export function overlap(item: TraceItem, answer?: string | null): number | null {
  if (typeof item.overlap === "number") return item.overlap;
  if (answer == null) return null;
  const used = tokens(answer); const source = tokens(item.content);
  if (!used.size || !source.size) return 0;
  return [...source].filter((word) => used.has(word)).length / source.size;
}
export function parseTrace(input: unknown): TraceFile {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("Trace must be a JSON object.");
  const value = input as Record<string, unknown>;
  if (typeof value.query !== "string") throw new Error("Trace needs a query string.");
  if (typeof value.schema_version !== "number" || !Number.isInteger(value.schema_version) || value.schema_version < 1 || value.schema_version > 4) throw new Error("Unsupported trace schema version.");
  const object = (entry: unknown, label: string): Record<string, unknown> => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) throw new Error(`${label} must be an object.`);
    return entry as Record<string, unknown>;
  };
  const list = (entry: unknown, label: string): unknown[] => {
    if (!Array.isArray(entry)) throw new Error(`${label} must be a list.`);
    return entry;
  };
  const text = (entry: unknown, label: string) => { if (typeof entry !== "string") throw new Error(`${label} must be text.`); };
  const number = (entry: unknown, label: string) => { if (entry != null && (typeof entry !== "number" || !Number.isFinite(entry))) throw new Error(`${label} must be a finite number.`); };
  const items = (entry: unknown) => list(entry, "Items").forEach((raw) => {
    const item = object(raw, "Item");
    for (const key of ["id", "content", "source"]) text(item[key], `Item ${key}`);
    for (const key of ["label", "kind", "source_uri"]) if (item[key] != null) text(item[key], `Item ${key}`);
    for (const key of ["score", "vector_score", "graph_score", "overlap"]) number(item[key], key);
    if (item.overlap != null && ((item.overlap as number) < 0 || (item.overlap as number) > 1)) throw new Error("Overlap must be between zero and one.");
  });
  const edges = (entry: unknown) => list(entry, "Edges").forEach((raw) => {
    const edge = object(raw, "Edge");
    for (const key of ["source", "target", "relation"]) text(edge[key], `Edge ${key}`);
    number(edge.weight, "Edge weight");
  });
  if (value.answer != null) text(value.answer, "Answer");
  if (value.producer != null) text(value.producer, "Producer");
  if (value.started_at != null) text(value.started_at, "Start time");
  number(value.duration_ms, "Duration");
  if (value.items != null) items(value.items);
  if (value.edges != null) edges(value.edges);
  if (value.retrievals != null) list(value.retrievals, "Retrievals").forEach((raw) => {
    const retrieval = object(raw, "Retrieval"); items(retrieval.items);
    if (retrieval.edges != null) edges(retrieval.edges);
  });
  if (value.graph != null) { const graph = object(value.graph, "Graph"); items(graph.nodes); edges(graph.edges); }
  if (value.spans != null) list(value.spans, "Spans").forEach((raw) => {
    const span = object(raw, "Span");
    for (const key of ["id", "name", "kind"]) text(span[key], `Span ${key}`);
    if (span.status != null) text(span.status, "Span status");
    number(span.start_ms, "Span start"); number(span.end_ms, "Span end");
  });
  if (value.metrics != null) Object.entries(object(value.metrics, "Metrics")).forEach(([key, entry]) => number(entry, key));
  return value as unknown as TraceFile;
}
export function retrievedItems(trace: TraceFile): TraceItem[] {
  return trace.retrievals?.flatMap((retrieval) => retrieval.items ?? []) ?? trace.items ?? [];
}
export function toGraphState(trace: TraceFile): TraceState {
  const items = retrievedItems(trace);
  const nodes = trace.graph?.nodes ?? items;
  const uniqueNodes = [...new Map(nodes.map((item) => [item.id, item])).values()];
  const edges = trace.graph?.edges ?? trace.retrievals?.flatMap((entry) => entry.edges ?? []) ?? trace.edges ?? [];
  const nodeIds = new Set(uniqueNodes.map((item) => item.id));
  const retrieved = new Set(items.map((item) => item.id));
  return {
    id: `${trace.started_at ?? "trace"}:${trace.query}`,
    query: trace.query,
    computedAt: trace.started_at ?? "",
    weights: { vector: 0, graph: 0, intent: "conceptual" },
    confidence: { score: 0, uncertainty: 0, rationale: "" },
    graph: {
      nodes: uniqueNodes.map((item) => ({
        id: item.id, label: item.label || item.id,
        type: ([...kinds].find((kind) => kind.toLowerCase() === item.kind?.toLowerCase()) ?? "Document") as EntityType,
        active: retrieved.has(item.id) || items.length === 0,
        position: { x: 0, y: 0 }, score: item.score ?? undefined,
        similarity: item.vector_score ?? undefined,
        meta: { subtitle: item.source, snippet: item.content,
          summaryText: item.content, scoreGraph: item.graph_score ?? undefined,
          sourceUrl: safeHttpUrl(item.source_uri) },
      })),
      edges: edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)).map((edge, index) => ({ id: `${edge.source}:${edge.target}:${index}`,
        source: edge.source, target: edge.target, relation: edge.relation,
        confidence: edge.weight ?? 1, active: true })),
    },
    steps: (trace.spans ?? []).map((span, index) => ({ id: span.id, index, title: span.name,
      detail: span.kind, status: span.status === "running" ? "active" : "complete",
      durationMs: span.end_ms != null && span.start_ms != null ? span.end_ms - span.start_ms : undefined })),
    metrics: { queryTimeSec: (trace.duration_ms ?? 0) / 1000 },
  };
}
