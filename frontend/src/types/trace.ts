export type EntityType = "PR" | "Commit" | "File" | "Person" | "Document" | "Repo" | "Ticket" | "Service" | "Library" | "Team" | "Tool";
export interface TraceNode { id: string; label: string; type: EntityType; active: boolean;
  position: { x: number; y: number }; similarity?: number; score?: number;
  meta?: { subtitle?: string; owner?: string; status?: string; timestamp?: string; snippet?: string;
    summaryText?: string; connections?: number; scoreGraph?: number; sourceUrl?: string; recency?: number;
    ageDays?: number | null; linkedSeed?: boolean };
  orphan?: boolean; }
export interface TraceEdge { id: string; source: string; target: string; confidence: number; active: boolean; relation?: string; }
export interface RouterWeights { vector: number; graph: number; intent: "relational" | "conceptual"; }
export interface RouterConfidence { score: number; uncertainty: number; rationale: string; }
export type StepStatus = "complete" | "active" | "pending";
export interface ExecutionStep { id: string; index: number; title: string; detail: string; status: StepStatus; badge?: string; arm?: "vector" | "graph"; durationMs?: number; }
export interface TraceMetrics { tokens?: { used: number; budget: number; reductionPct: number }; peakRamGb?: number; queryTimeSec: number; nodesEvaluated?: number; }
export interface TraceState { id: string; query: string; computedAt: string; weights: RouterWeights;
  confidence: RouterConfidence; steps: ExecutionStep[]; metrics: TraceMetrics;
  graph: { nodes: TraceNode[]; edges: TraceEdge[] }; context?: string; }
