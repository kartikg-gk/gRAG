import { Fragment, useEffect, useMemo, useState, type ReactNode } from "react";
import { CornerDownLeft, LoaderCircle, Search, Sparkles } from "lucide-react";
import { useNavigate, useOutletContext } from "react-router-dom";

import { GraphWorkspace } from "@/components/graph/GraphWorkspace";
import { summarize } from "@/lib/api";
import type { ShellOutletContext } from "@/components/shell/AppShell";
import { useStudio } from "@/contexts/StudioContext";

function escapeRegex(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function HomePage() {
  const navigate = useNavigate();
  const { commandInputRef, focusRequest } = useOutletContext<ShellOutletContext>();
  const {
    trace, retrieving, answer, answerStreaming, suggestionQueries, runQuery,
    activeGraphId, citationFocus, focusCitation, graphSwitching,
  } = useStudio();
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (focusRequest > 0) commandInputRef.current?.focus();
  }, [commandInputRef, focusRequest]);

  const answerParts = useMemo<ReactNode[]>(() => {
    if (answer === null || answer === "") return [];
    const firstByLabel = new Map<string, string>();
    for (const node of trace.graph.nodes) {
      if (node.label.length >= 3 && !firstByLabel.has(node.label)) firstByLabel.set(node.label, node.id);
    }
    const labels = [...firstByLabel.keys()].sort((a, b) => b.length - a.length);
    if (labels.length === 0) return [answer];
    const matcher = new RegExp(`(${labels.map(escapeRegex).join("|")})`, "g");
    return answer.split(matcher).map((part, index) => {
      const nodeId = firstByLabel.get(part);
      return nodeId ? (
        <button key={`${part}-${index}`} type="button" className="font-medium text-blue underline decoration-blue/40 underline-offset-4" onClick={() => focusCitation(nodeId)}>{part}</button>
      ) : <Fragment key={index}>{part}</Fragment>;
    });
  }, [answer, focusCitation, trace.graph.nodes]);

  const submit = () => {
    const trimmed = query.trim();
    if (!trimmed || retrieving || graphSwitching) return;
    setQuery("");
    void runQuery(trimmed);
  };

  return (
    <section className="dotted-canvas flex h-full min-w-0 flex-col bg-paper" aria-label="Graph workspace">
      <div className="border-b border-line bg-panel/95 px-3 py-3 md:px-5">
        <div className="mx-auto max-w-3xl">
          <label className="flex h-11 items-center gap-3 rounded-md border border-line bg-paper px-3 text-ink-dim focus-within:border-blue focus-within:ring-1 focus-within:ring-blue">
            {retrieving ? <LoaderCircle size={17} className="animate-spin" aria-hidden="true" /> : <Search size={17} aria-hidden="true" />}
            <span className="sr-only">Query the graph</span>
            <input ref={commandInputRef} type="text" value={query} disabled={retrieving || graphSwitching}
              aria-busy={retrieving}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); submit(); } }}
              className="min-w-0 flex-1 bg-transparent text-sm text-ink outline-none placeholder:text-ink-muted disabled:cursor-wait"
              placeholder={retrieving ? "traversing graph…" : "Trace anything…"} />
            <span className="hidden items-center gap-1 font-mono text-[11px] text-ink-muted sm:flex"><CornerDownLeft size={13} aria-hidden="true" /> Enter</span>
          </label>
          {retrieving ? <p className="sr-only" role="status" aria-live="polite">Retrieving graph context and answer…</p> : null}
          <div className="mt-2 flex gap-2 overflow-x-auto pb-0.5 scrollbar-thin" aria-label="Suggested queries">
            {suggestionQueries.map((item) => <button key={item} type="button" disabled={retrieving || graphSwitching} onClick={() => { navigate("/"); void runQuery(item); }} className="shrink-0 rounded-sm border border-line bg-paper px-2.5 py-1 text-left text-[11px] text-ink-dim transition-colors hover:border-ink-muted hover:text-ink disabled:opacity-50">{item}</button>)}
          </div>
        </div>
      </div>
      <div className="relative min-h-0 flex-1" aria-label="Trace graph canvas">
        <GraphWorkspace trace={trace} activeGraphId={activeGraphId ?? undefined} citationFocus={citationFocus ?? undefined} summarize={summarize} />
        {trace.id === "idle-trace" && !retrieving ? <div className="pointer-events-none absolute inset-0 flex items-center justify-center"><p className="rounded-sm border border-line bg-panel/90 px-4 py-2 text-sm text-ink-muted">Run a trace to explore the graph.</p></div> : null}
        {answer !== null ? (
          <section className="absolute bottom-4 left-4 right-4 z-10 max-h-36 overflow-y-auto rounded-sm border border-line bg-panel/95 p-4 shadow-none backdrop-blur md:left-auto md:w-[520px]" aria-label="Answer" aria-live="polite" aria-busy={answerStreaming}>
            <h2 className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-ink"><Sparkles size={14} className="text-blue" /> Answer {answerStreaming ? <span className="font-normal normal-case text-ink-muted">streaming…</span> : null}</h2>
            {answer === "" && answerStreaming ? <div className="h-3 w-4/5 animate-pulse rounded-sm bg-raised" /> : <p className="text-sm leading-6 text-ink-dim">{answerParts}</p>}
          </section>
        ) : null}
      </div>
    </section>
  );
}
