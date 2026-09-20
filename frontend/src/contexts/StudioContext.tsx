import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
  type ReactNode,
} from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

import {
  API_BASE, createSession, fetchGraphs, fetchSessionTraces, fetchSuggestions,
  hydrateTraceFromLog, listSessions, runTraceQuery, streamAnswer, switchGraph as switchGraphApi,
  type ApiGraph, type SessionSummary, type Suggestion, type TraceRecord,
} from "@/lib/api";
import { sampleTrace } from "@/lib/sampleTrace";
import type { CitationFocusRequest } from "@/components/graph/GraphWorkspace";
import type { TraceState } from "@/types/trace";

export interface HostIdentity { userId: string | null; email: string | null; ready: boolean }

export const EMPTY_TRACE: TraceState = {
  id: "trace_empty",
  query: "",
  computedAt: new Date(0).toISOString(),
  weights: { vector: 0.5, graph: 0.5, intent: "conceptual" },
  confidence: { score: 0, uncertainty: 0, rationale: "No query yet." },
  steps: [],
  metrics: { queryTimeSec: 0 },
  graph: { nodes: [], edges: [] },
};

export const LOCAL_PRESETS = [
  "Who corrected the authentication regression?",
  "Which files did PR #142 touch?",
  "What does the authentication runbook mention?",
] as const;

const RECENTS_KEY = "graphrag.recent.queries";
const ACTIVE_GRAPH_KEY = "graphrag.activeGraph";

function safeRead(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}

function safeWrite(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch { /* storage is optional */ }
}

function safeRemove(key: string) {
  try { localStorage.removeItem(key); } catch { /* storage is optional */ }
}

function initialRecents(): string[] {
  try {
    const parsed: unknown = JSON.parse(safeRead(RECENTS_KEY) ?? "[]");
    return Array.isArray(parsed) && parsed.every((item) => typeof item === "string")
      ? [...new Set(parsed)].slice(0, 6)
      : [];
  } catch { return []; }
}

function initialTheme(): "dark" | "light" {
  return safeRead("graphrag.theme") === "light" ? "light" : "dark";
}

interface StudioContextValue {
  trace: TraceState;
  retrieving: boolean;
  answer: string | null;
  answerStreaming: boolean;
  suggestions: Suggestion[];
  suggestionQueries: string[];
  recents: string[];
  graphs: ApiGraph[];
  activeGraphId: string | null;
  graphsLoading: boolean;
  graphError: string | null;
  graphSwitching: boolean;
  identity: HostIdentity;
  historyEnabled: boolean;
  sessions: SessionSummary[];
  sessionsLoading: boolean;
  sessionsError: string | null;
  activeSessionId: string | null;
  activeTraces: TraceRecord[];
  tracesLoading: boolean;
  tracesError: string | null;
  citationFocus: CitationFocusRequest | null;
  theme: "dark" | "light";
  runQuery: (query: string) => Promise<void>;
  focusCitation: (nodeId: string) => void;
  changeGraph: (id: string) => Promise<void>;
  newChat: () => void;
  loadSessionsNow: () => Promise<void>;
  selectSession: (id: string) => Promise<void>;
  selectTrace: (record: TraceRecord) => Promise<void>;
  refreshActiveTraces: () => Promise<void>;
  toggleTheme: () => void;
  apiBase: string;
}

const StudioContext = createContext<StudioContextValue | null>(null);

interface StudioProviderProps { children: ReactNode; identity?: HostIdentity }

export function StudioProvider({ children, identity = { userId: null, email: null, ready: true } }: StudioProviderProps) {
  const navigate = useNavigate();
  const [trace, setTrace] = useState<TraceState>(EMPTY_TRACE);
  const [retrieving, setRetrieving] = useState(false);
  const retrievingRef = useRef(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [answerStreaming, setAnswerStreaming] = useState(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [recents, setRecents] = useState<string[]>(initialRecents);
  const [graphs, setGraphs] = useState<ApiGraph[]>([]);
  const [activeGraphId, setActiveGraphId] = useState<string | null>(null);
  const [graphsLoading, setGraphsLoading] = useState(true);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [graphSwitching, setGraphSwitching] = useState(false);
  const graphSwitchingRef = useRef(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState<string | null>(null);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const activeSessionRef = useRef<string | null>(null);
  const [activeTraces, setActiveTraces] = useState<TraceRecord[]>([]);
  const [tracesLoading, setTracesLoading] = useState(false);
  const [tracesError, setTracesError] = useState<string | null>(null);
  const [citationFocus, setCitationFocus] = useState<CitationFocusRequest | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">(initialTheme);
  const queryEpoch = useRef(0);
  const graphGeneration = useRef(0);
  const suggestionGeneration = useRef(0);
  const sessionGeneration = useRef(0);
  const traceGeneration = useRef(0);
  const historyEnabled = identity.ready && Boolean(identity.userId);

  const bumpEpoch = useCallback(() => ++queryEpoch.current, []);

  useEffect(() => {
    activeSessionRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    ++queryEpoch.current;
    ++sessionGeneration.current;
    ++traceGeneration.current;
    retrievingRef.current = false;
    activeSessionRef.current = null;
    setSessions([]);
    setActiveSessionId(null);
    setActiveTraces([]);
    setSessionsError(null);
    setTracesError(null);
    setSessionsLoading(false);
    setTracesLoading(false);
    setRetrieving(false);
    setAnswerStreaming(false);
    setAnswer(null);
    setCitationFocus(null);
  }, [identity.ready, identity.userId]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    safeWrite("graphrag.theme", theme);
  }, [theme]);

  const loadSuggestionsNow = useCallback(async () => {
    const generation = ++suggestionGeneration.current;
    const next = await fetchSuggestions(6);
    if (generation === suggestionGeneration.current) setSuggestions(next);
  }, []);

  const refreshGraphs = useCallback(async (confirmedId?: string) => {
    const generation = ++graphGeneration.current;
    try {
      const response = await fetchGraphs();
      if (generation !== graphGeneration.current) return;
      const active = confirmedId && response.graphs.some((graph) => graph.id === confirmedId)
        ? confirmedId : response.active;
      setGraphs(response.graphs.map((graph) => ({ ...graph, active: graph.id === active })));
      setActiveGraphId(active);
      setGraphError(null);
    } catch (error) {
      if (generation !== graphGeneration.current) return;
      if (!confirmedId) setGraphError(error instanceof Error ? error.message : "Unable to load graphs");
    }
  }, []);

  useEffect(() => {
    const generation = ++graphGeneration.current;
    setGraphsLoading(true);
    void (async () => {
      try {
        const response = await fetchGraphs();
        if (generation !== graphGeneration.current) return;
        const ids = new Set(response.graphs.map((graph) => graph.id));
        const saved = safeRead(ACTIVE_GRAPH_KEY);
        let active = response.active;
        if (saved && !ids.has(saved)) safeRemove(ACTIVE_GRAPH_KEY);
        if (saved && ids.has(saved) && saved !== response.active) {
          try {
            const confirmed = await switchGraphApi(saved);
            if (generation !== graphGeneration.current) return;
            active = confirmed.active;
          } catch (error) {
            if (generation !== graphGeneration.current) return;
            console.error("[graphRAG] Saved graph restore failed", error);
            setGraphError(error instanceof Error ? error.message : "Unable to restore saved graph");
            setGraphs(response.graphs.map((graph) => ({ ...graph, active: graph.id === response.active })));
            setActiveGraphId(response.active);
            return;
          }
        }
        setGraphs(response.graphs.map((graph) => ({ ...graph, active: graph.id === active })));
        setActiveGraphId(active);
        setGraphError(null);
      } catch (error) {
        if (generation === graphGeneration.current) {
          setGraphError(error instanceof Error ? error.message : "Unable to load graphs");
        }
      } finally {
        if (generation === graphGeneration.current) setGraphsLoading(false);
        void loadSuggestionsNow();
      }
    })();
  }, [loadSuggestionsNow]);

  const rememberRecent = useCallback((query: string) => {
    setRecents((current) => {
      const next = [query, ...current.filter((item) => item !== query)].slice(0, 6);
      safeWrite(RECENTS_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const runQuery = useCallback(async (rawQuery: string) => {
    const query = rawQuery.trim();
    if (!query || retrievingRef.current || graphSwitchingRef.current) return;
    const epoch = bumpEpoch();
    retrievingRef.current = true;
    setRetrieving(true);
    setAnswer(null);
    setAnswerStreaming(false);
    setGraphSwitching(false);
    setCitationFocus(null);
    rememberRecent(query);

    let sessionId = activeSessionRef.current ?? undefined;
    if (historyEnabled && !sessionId) {
      try {
        const created = await createSession(query.slice(0, 30) || "New chat", identity.email ?? undefined);
        if (epoch !== queryEpoch.current) return;
        sessionId = created.id;
        activeSessionRef.current = created.id;
        setActiveSessionId(created.id);
        setSessions((current) => [created, ...current.filter((item) => item.id !== created.id)]);
      } catch (error) {
        if (epoch === queryEpoch.current) console.error("[graphRAG] Session creation failed", error);
      }
      if (epoch !== queryEpoch.current) return;
    }

    try {
      const nextTrace = await runTraceQuery(query, sessionId);
      if (epoch !== queryEpoch.current) return;
      setTrace(nextTrace);
      setRetrieving(false);
      retrievingRef.current = false;
      if (nextTrace.context?.trim()) {
        setAnswer("");
        setAnswerStreaming(true);
        try {
          await streamAnswer(query, nextTrace.context, (text) => {
            if (epoch === queryEpoch.current) setAnswer(text);
          });
        } catch (error) {
          if (epoch !== queryEpoch.current) return;
          console.error("[graphRAG] Answer stream failed", error);
          setAnswer(null);
        } finally {
          if (epoch === queryEpoch.current) setAnswerStreaming(false);
        }
      }
    } catch (error) {
      if (epoch !== queryEpoch.current) return;
      console.error("[graphRAG] Trace retrieval failed", error);
      const message = "Backend unreachable — the sample trace is shown so the canvas stays usable.";
      setTrace({ ...sampleTrace, id: `sample_failed_${epoch}_${Date.now()}`, query });
      setAnswer(message);
      toast.error(message);
    } finally {
      if (epoch === queryEpoch.current) {
        retrievingRef.current = false;
        setRetrieving(false);
      }
    }
  }, [bumpEpoch, historyEnabled, identity.email, rememberRecent]);

  const focusCitation = useCallback((nodeId: string) => {
    if (graphSwitchingRef.current) return;
    setCitationFocus((current) => ({ nodeId, nonce: (current?.nonce ?? 0) + 1 }));
    navigate("/");
  }, [navigate]);

  const changeGraph = useCallback(async (id: string) => {
    if (graphSwitchingRef.current || id === activeGraphId) return;
    if (!graphs.some((graph) => graph.id === id)) {
      setGraphError("Unknown graph selection.");
      return;
    }
    bumpEpoch();
    ++traceGeneration.current;
    retrievingRef.current = false;
    setRetrieving(false);
    setAnswerStreaming(false);
    setTracesLoading(false);
    setTracesError(null);
    graphSwitchingRef.current = true;
    setGraphSwitching(true);
    try {
      const confirmed = await switchGraphApi(id);
      setActiveGraphId(confirmed.active);
      setGraphs((current) => current.map((graph) => ({ ...graph, active: graph.id === confirmed.active })));
      safeWrite(ACTIVE_GRAPH_KEY, confirmed.active);
      setTrace(EMPTY_TRACE);
      setAnswer(null);
      setActiveTraces([]);
      setCitationFocus(null);
      setGraphError(null);
      toast.success(`Switched to ${confirmed.label}`);
      await Promise.all([refreshGraphs(confirmed.active), loadSuggestionsNow()]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to switch graph";
      setGraphError(message);
      toast.error(message);
    } finally {
      graphSwitchingRef.current = false;
      setGraphSwitching(false);
    }
  }, [activeGraphId, bumpEpoch, graphs, loadSuggestionsNow, refreshGraphs]);

  const newChat = useCallback(() => {
    if (graphSwitchingRef.current) return;
    bumpEpoch();
    retrievingRef.current = false;
    activeSessionRef.current = null;
    setActiveSessionId(null);
    setActiveTraces([]);
    setTrace(EMPTY_TRACE);
    setAnswer(null);
    setRetrieving(false);
    setAnswerStreaming(false);
    setGraphSwitching(false);
    setCitationFocus(null);
    navigate("/");
  }, [bumpEpoch, navigate]);

  const loadSessionsNow = useCallback(async () => {
    if (!historyEnabled) return;
    const generation = ++sessionGeneration.current;
    setSessionsLoading(true);
    setSessionsError(null);
    try {
      const next = await listSessions();
      if (generation !== sessionGeneration.current) return;
      setSessions([...next].sort((a, b) => b.created_at.localeCompare(a.created_at)));
    } catch (error) {
      if (generation === sessionGeneration.current) setSessionsError(error instanceof Error ? error.message : "Unable to load sessions");
    } finally {
      if (generation === sessionGeneration.current) setSessionsLoading(false);
    }
  }, [historyEnabled, identity.ready, identity.userId]);

  useEffect(() => { void loadSessionsNow(); }, [loadSessionsNow]);

  const hydrateRecord = useCallback(async (record: TraceRecord, epoch: number) => {
    let restoringGraph = false;
    try {
      if (record.graph_id && record.graph_id !== activeGraphId) {
        if (!graphs.some((graph) => graph.id === record.graph_id)) {
          throw new Error(`The graph used by this trace is unavailable: ${record.graph_id}`);
        }
        restoringGraph = true;
        graphSwitchingRef.current = true;
        setGraphSwitching(true);
        const confirmed = await switchGraphApi(record.graph_id);
        if (epoch !== queryEpoch.current) return;
        setActiveGraphId(confirmed.active);
        setGraphs((current) => current.map((graph) => ({
          ...graph,
          active: graph.id === confirmed.active,
        })));
        safeWrite(ACTIVE_GRAPH_KEY, confirmed.active);
        setGraphError(null);
        void loadSuggestionsNow();
      }
      const hydrated = await hydrateTraceFromLog(record);
      if (epoch !== queryEpoch.current) return;
      setTrace(hydrated);
      navigate("/");
    } catch (error) {
      if (epoch !== queryEpoch.current) return;
      console.error("[graphRAG] History hydration failed", error);
      setTrace(EMPTY_TRACE);
      toast.error(error instanceof Error ? error.message : "This trace cannot be restored.");
    } finally {
      if (restoringGraph) {
        graphSwitchingRef.current = false;
        setGraphSwitching(false);
      }
      if (epoch === queryEpoch.current) {
        retrievingRef.current = false;
        setRetrieving(false);
      }
    }
  }, [activeGraphId, graphs, loadSuggestionsNow, navigate]);

  const selectSession = useCallback(async (id: string) => {
    if (graphSwitchingRef.current || !historyEnabled || !sessions.some((session) => session.id === id)) return;
    const epoch = bumpEpoch();
    ++traceGeneration.current;
    activeSessionRef.current = id;
    setActiveSessionId(id);
    setActiveTraces([]);
    setAnswer(null);
    setAnswerStreaming(false);
    setGraphSwitching(false);
    setCitationFocus(null);
    setTracesLoading(true);
    setTracesError(null);
    retrievingRef.current = true;
    setRetrieving(true);
    try {
      const records = await fetchSessionTraces(id);
      if (epoch !== queryEpoch.current) return;
      const original = [...records].sort((a, b) => a.created_at.localeCompare(b.created_at));
      setActiveTraces([...original].reverse());
      setTracesLoading(false);
      if (original.length === 0) {
        setTrace(EMPTY_TRACE);
        setRetrieving(false);
        retrievingRef.current = false;
        navigate("/");
        return;
      }
      await hydrateRecord(original[original.length - 1], epoch);
    } catch (error) {
      if (epoch !== queryEpoch.current) return;
      setTracesError(error instanceof Error ? error.message : "Unable to load traces");
      setRetrieving(false);
      retrievingRef.current = false;
    } finally {
      if (epoch === queryEpoch.current) setTracesLoading(false);
    }
  }, [bumpEpoch, historyEnabled, hydrateRecord, navigate, sessions]);

  const selectTrace = useCallback(async (record: TraceRecord) => {
    if (graphSwitchingRef.current) return;
    const epoch = bumpEpoch();
    setAnswer(null);
    setAnswerStreaming(false);
    setGraphSwitching(false);
    setCitationFocus(null);
    retrievingRef.current = true;
    setRetrieving(true);
    await hydrateRecord(record, epoch);
  }, [bumpEpoch, hydrateRecord]);

  const refreshActiveTraces = useCallback(async () => {
    if (graphSwitchingRef.current) return;
    const sessionId = activeSessionRef.current;
    if (!historyEnabled || !sessionId) return;
    const generation = ++traceGeneration.current;
    setTracesLoading(true);
    setTracesError(null);
    try {
      const records = await fetchSessionTraces(sessionId);
      if (generation !== traceGeneration.current || sessionId !== activeSessionRef.current) return;
      setActiveTraces([...records].sort((a, b) => b.created_at.localeCompare(a.created_at)));
    } catch (error) {
      if (generation === traceGeneration.current) setTracesError(error instanceof Error ? error.message : "Unable to load traces");
    } finally {
      if (generation === traceGeneration.current) setTracesLoading(false);
    }
  }, [historyEnabled]);

  const suggestionQueries = suggestions.length > 0
    ? suggestions.slice(0, 3).map((item) => item.query)
    : [...LOCAL_PRESETS];

  const value = useMemo<StudioContextValue>(() => ({
    trace, retrieving, answer, answerStreaming, suggestions, suggestionQueries, recents,
    graphs, activeGraphId, graphsLoading, graphError, graphSwitching,
    identity, historyEnabled, sessions, sessionsLoading, sessionsError,
    activeSessionId, activeTraces, tracesLoading, tracesError, citationFocus, theme,
    runQuery, focusCitation, changeGraph, newChat, loadSessionsNow, selectSession,
    selectTrace, refreshActiveTraces, toggleTheme: () => setTheme((current) => current === "dark" ? "light" : "dark"),
    apiBase: API_BASE,
  }), [
    activeGraphId, activeSessionId, activeTraces, answer, answerStreaming, changeGraph,
    citationFocus, focusCitation, graphError, graphSwitching, graphs, graphsLoading,
    historyEnabled, identity, loadSessionsNow, newChat, recents, refreshActiveTraces,
    retrieving, runQuery, selectSession, selectTrace, sessions, sessionsError,
    sessionsLoading, suggestionQueries, suggestions, theme, trace, tracesError, tracesLoading,
  ]);

  return <StudioContext.Provider value={value}>{children}</StudioContext.Provider>;
}

export function useStudio() {
  const value = useContext(StudioContext);
  if (!value) throw new Error("useStudio must be used inside StudioProvider");
  return value;
}
