import { useEffect } from "react";
import { Check, GitBranch, History, LoaderCircle, MessageSquare, Plus, RotateCw } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { useStudio } from "@/contexts/StudioContext";
import type { TraceRecord } from "@/lib/api";

function PageFrame({ title, action, children }: { title: string; action?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="h-full overflow-y-auto bg-paper p-4 scrollbar-thin md:p-8">
      <div className="mx-auto max-w-5xl">
        <header className="flex items-center justify-between gap-4">
          <h1 className="text-xl font-semibold tracking-[-0.02em] md:text-2xl">{title}</h1>
          {action}
        </header>
        <div className="mt-6 overflow-hidden rounded-sm border border-line bg-panel">{children}</div>
      </div>
    </section>
  );
}

const rowClass = "flex w-full items-center gap-3 border-b border-line px-4 py-3 text-left last:border-b-0 hover:bg-raised focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue";

function TraceRow({ record, onClick, disabled }: { record: TraceRecord; onClick: () => void; disabled?: boolean }) {
  return <button type="button" className={rowClass} disabled={disabled} onClick={onClick}><History size={17} className="shrink-0 text-blue" /><span className="min-w-0 flex-1"><span className="block truncate text-sm text-ink">{record.query}</span><span className="mt-1 block font-mono text-[11px] text-ink-muted">{new Date(record.created_at).toLocaleString()}</span></span></button>;
}

export function TracesPage() {
  const { historyEnabled, activeSessionId, activeTraces, tracesLoading, tracesError, graphSwitching, refreshActiveTraces, selectTrace } = useStudio();
  useEffect(() => { void refreshActiveTraces(); }, [refreshActiveTraces]);
  return <PageFrame title="Traces" action={historyEnabled && activeSessionId ? <button type="button" className="page-action" disabled={graphSwitching} onClick={() => void refreshActiveTraces()}><RotateCw size={15} /> Refresh</button> : undefined}>
    {!historyEnabled ? <p className="p-5 text-sm text-ink-dim">Sign in to keep sessions</p>
      : !activeSessionId ? <p className="p-5 text-sm text-ink-dim">Choose a session to inspect its traces.</p>
        : tracesLoading ? <p className="flex items-center gap-2 p-5 text-sm text-ink-dim" role="status" aria-live="polite"><LoaderCircle size={16} className="animate-spin" /> Loading traces…</p>
          : tracesError ? <p className="p-5 text-sm text-red" role="alert">{tracesError}</p>
            : activeTraces.length === 0 ? <p className="p-5 text-sm text-ink-dim">No traces in this session.</p>
              : activeTraces.map((record) => <TraceRow key={record.id} record={record} disabled={graphSwitching} onClick={() => void selectTrace(record)} />)}
  </PageFrame>;
}

export function QueriesPage() {
  const navigate = useNavigate();
  const { recents, runQuery, graphSwitching } = useStudio();
  const rerun = (query: string) => { navigate("/"); void runQuery(query); };
  return <PageFrame title="Queries">
    {recents.length === 0 ? <p className="p-5 text-sm text-ink-dim">Recent queries will appear here.</p>
      : recents.slice(0, 6).map((query) => <button key={query} type="button" disabled={graphSwitching} className={rowClass} onClick={() => rerun(query)}><MessageSquare size={17} className="shrink-0 text-blue" /><span className="truncate text-sm">{query}</span></button>)}
  </PageFrame>;
}

export function GraphsPage() {
  const { graphs, activeGraphId, graphsLoading, graphError, graphSwitching, changeGraph } = useStudio();
  return <PageFrame title="Graphs">
    {graphsLoading ? <p className="flex items-center gap-2 p-5 text-sm text-ink-dim" role="status" aria-live="polite"><LoaderCircle size={16} className="animate-spin" /> Loading graphs…</p>
      : graphError && graphs.length === 0 ? <p className="p-5 text-sm text-red" role="alert">{graphError}</p>
        : graphs.length === 0 ? <p className="p-5 text-sm text-ink-dim">No graphs available.</p>
          : graphs.map((graph) => <button key={graph.id} type="button" disabled={graphSwitching || graph.id === activeGraphId} className={rowClass} onClick={() => void changeGraph(graph.id)}><GitBranch size={17} className="shrink-0 text-blue" /><span className="min-w-0 flex-1"><span className="block truncate text-sm">{graph.label}</span><span className="block truncate font-mono text-[11px] text-ink-muted">{graph.id}</span></span>{graph.id === activeGraphId ? <span className="flex items-center gap-1 text-xs text-green"><Check size={14} /> Active</span> : null}</button>)}
    {graphError && graphs.length > 0 ? <p className="border-t border-line p-3 text-xs text-red" role="alert">{graphError}</p> : null}
  </PageFrame>;
}

export function SessionsPage() {
  const { historyEnabled, sessions, sessionsLoading, sessionsError, activeSessionId, graphSwitching, selectSession, newChat, loadSessionsNow } = useStudio();
  useEffect(() => { void loadSessionsNow(); }, [loadSessionsNow]);
  return <PageFrame title="Sessions" action={historyEnabled ? <button type="button" className="page-action" disabled={graphSwitching} onClick={newChat}><Plus size={15} /> New Chat</button> : undefined}>
    {!historyEnabled ? <p className="p-5 text-sm text-ink-dim">Sign in to keep sessions</p>
      : sessionsLoading ? <p className="flex items-center gap-2 p-5 text-sm text-ink-dim" role="status" aria-live="polite"><LoaderCircle size={16} className="animate-spin" /> Loading sessions…</p>
        : sessionsError ? <p className="p-5 text-sm text-red" role="alert">{sessionsError}</p>
          : sessions.length === 0 ? <p className="p-5 text-sm text-ink-dim">No saved sessions yet.</p>
            : sessions.map((session) => <button key={session.id} type="button" disabled={graphSwitching} className={rowClass} onClick={() => void selectSession(session.id)}><History size={17} className="shrink-0 text-blue" /><span className="min-w-0 flex-1"><span className="block truncate text-sm">{session.title}</span><span className="block font-mono text-[11px] text-ink-muted">{new Date(session.created_at).toLocaleString()}</span></span>{session.id === activeSessionId ? <span className="text-xs text-green">Active</span> : null}</button>)}
  </PageFrame>;
}

export function SettingsPage() {
  const { apiBase, theme, toggleTheme } = useStudio();
  return <PageFrame title="Settings">
    <dl className="divide-y divide-line text-sm">
      <div className="grid gap-2 p-5 sm:grid-cols-[160px_1fr]"><dt className="text-ink-muted">API base</dt><dd className="break-all font-mono text-ink">{apiBase}</dd></div>
      <div className="grid items-center gap-2 p-5 sm:grid-cols-[160px_1fr]"><dt className="text-ink-muted">Theme</dt><dd><button type="button" className="page-action" onClick={toggleTheme}>{theme === "dark" ? "Dark" : "Light"} · Toggle</button></dd></div>
    </dl>
  </PageFrame>;
}
