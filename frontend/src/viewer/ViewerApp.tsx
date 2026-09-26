import { useCallback, useEffect, useRef, useState } from "react";
import { FileJson2, FolderOpen, Link2, Upload } from "lucide-react";

import { GraphWorkspace } from "@/components/graph/GraphWorkspace";
import { overlap, parseTrace, retrievedItems, toGraphState, type TraceFile } from "@/viewer/trace";

interface LocalRun { id: string; name: string; query: string; started_at: string | null }
const localMode = Boolean((window as Window & { __GRAPHRAG_LOCAL_VIEWER__?: boolean }).__GRAPHRAG_LOCAL_VIEWER__);

export function ViewerApp() {
  const [trace, setTrace] = useState<TraceFile | null>(null);
  const [runs, setRuns] = useState<LocalRun[]>([]);
  const [name, setName] = useState("");
  const [pasted, setPasted] = useState("");
  const [source, setSource] = useState("");
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = useCallback((raw: unknown, label: string) => {
    try { setTrace(parseTrace(raw)); setName(label); setError(""); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Invalid trace JSON."); }
  }, []);
  const readFile = useCallback(async (file: File) => {
    try { accept(JSON.parse(await file.text()), file.name); }
    catch { setError("Could not parse the selected JSON file."); }
  }, [accept]);
  const loadUrl = useCallback(async (url: string) => {
    try {
      const parsed = new URL(url);
      if (parsed.protocol !== "https:" && parsed.protocol !== "http:") throw new Error("Use an HTTP or HTTPS JSON URL.");
      const response = await fetch(parsed.href);
      if (!response.ok) throw new Error(`JSON request failed (${response.status}).`);
      accept(await response.json(), parsed.href);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not load JSON URL."); }
  }, [accept]);

  useEffect(() => {
    const initial = new URLSearchParams(window.location.search).get("src");
    if (initial) { setSource(initial); void loadUrl(initial); }
    if (!localMode) return;
    void fetch("/api/local/runs").then(async (response) => {
      if (!response.ok) throw new Error("Could not list local traces.");
      const listed = await response.json() as LocalRun[];
      setRuns(listed);
      if (listed.length && !initial) {
        const first = await fetch(`/api/local/traces/${listed[0].id}`);
        if (!first.ok) throw new Error("Could not load local trace.");
        accept(await first.json(), listed[0].name);
      }
    }).catch((cause: unknown) => setError(cause instanceof Error ? cause.message : "Could not load local traces."));
  }, [accept, loadUrl]);

  const selectRun = async (run: LocalRun) => {
    try {
      const response = await fetch(`/api/local/traces/${run.id}`);
      if (!response.ok) throw new Error(`Could not load ${run.name}.`);
      accept(await response.json(), run.name);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not load trace."); }
  };
  const items = trace ? retrievedItems(trace) : [];
  const graph = trace ? toGraphState(trace) : null;
  const used = trace ? items.filter((item) => { const score = overlap(item, trace.answer); return score !== null && score >= 0.2; }).length : 0;

  return <main className="min-h-dvh bg-paper text-ink" onDragOver={(event) => event.preventDefault()} onDrop={(event) => {
    event.preventDefault(); const file = event.dataTransfer.files[0]; if (file) void readFile(file);
  }}>
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-panel px-5 py-4">
      <div><h1 className="text-xl font-semibold tracking-tight">graphweave <span className="font-normal text-ink-dim">/ trace viewer</span></h1><p className="text-xs text-ink-muted">Explore retrieval and execution locally</p></div>
      <button className="page-action" type="button" onClick={() => inputRef.current?.click()}><Upload size={16} /> Open JSON</button>
      <input ref={inputRef} className="hidden" type="file" accept=".json,application/json" onChange={(event) => { const file = event.target.files?.[0]; if (file) void readFile(file); }} />
    </header>
    <div className="grid min-h-[calc(100dvh-81px)] lg:grid-cols-[290px_minmax(0,1fr)]">
      <aside className="border-b border-line bg-panel p-4 lg:border-b-0 lg:border-r">
        {runs.length > 0 && <section className="mb-6"><h2 className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-ink-dim"><FolderOpen size={15} /> Run history</h2><div className="space-y-1">{runs.map((run) => <button key={run.id} type="button" className="w-full rounded-sm px-2 py-2 text-left text-sm hover:bg-raised" onClick={() => void selectRun(run)}><span className="block truncate">{run.query || run.name}</span><span className="text-xs text-ink-muted">{run.name}</span></button>)}</div></section>}
        <section className="space-y-3"><h2 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-ink-dim"><FileJson2 size={15} /> Import trace</h2>
          <p className="text-xs leading-5 text-ink-dim">Drop a JSON file anywhere, paste JSON, or load a public URL. Imported files stay in this browser tab.</p>
          <textarea aria-label="Paste trace JSON" className="h-28 w-full rounded-sm border border-line bg-paper p-2 font-mono text-xs outline-none focus:border-blue" value={pasted} onChange={(event) => setPasted(event.target.value)} placeholder="Paste trace JSON" />
          <button type="button" className="page-action" onClick={() => { try { accept(JSON.parse(pasted), "Pasted JSON"); } catch { setError("Could not parse pasted JSON."); } }}>View pasted JSON</button>
          <div className="flex items-center gap-2 border-t border-line pt-3"><Link2 size={15} className="text-ink-muted" /><label className="text-xs text-ink-dim" htmlFor="source-url">Public JSON URL</label></div>
          <input id="source-url" className="w-full rounded-sm border border-line bg-paper p-2 text-xs outline-none focus:border-blue" value={source} onChange={(event) => setSource(event.target.value)} placeholder="https://example.com/trace.json" />
          <button type="button" className="page-action" onClick={() => void loadUrl(source)}>Load URL</button>
          <p className="text-xs text-ink-muted">The source must permit cross-origin browser requests.</p>
        </section>
      </aside>
      <section className="min-w-0 p-4 md:p-6">
        {error && <p role="alert" className="mb-4 rounded-sm border border-red/40 bg-red/10 p-3 text-sm text-red">{error}</p>}
        {!trace || !graph ? <div className="flex min-h-[50vh] items-center justify-center rounded-sm border border-dashed border-line text-center text-ink-dim"><p>Open a trace JSON file to inspect its graph, retrieval, and timeline.</p></div> : <>
          <div className="mb-4"><p className="truncate font-mono text-xs text-ink-muted">{name}</p><h2 className="mt-1 text-xl font-semibold">{trace.query || "Untitled query"}</h2><p className="mt-1 text-xs text-ink-dim">{trace.producer || "Unknown producer"} · {items.length} retrieved · {used} used · {trace.duration_ms ?? "—"} ms</p></div>
          <div className="h-[430px] overflow-hidden rounded-sm border border-line bg-panel"><GraphWorkspace trace={graph} /></div>
          <div className="mt-4 grid gap-4 xl:grid-cols-2">
            <section className="rounded-sm border border-line bg-panel p-4"><h3 className="mb-3 font-semibold">Retrieved vs used</h3>{items.length === 0 ? <p className="text-sm text-ink-dim">No retrieved items recorded.</p> : <div className="max-h-80 space-y-2 overflow-y-auto">{items.map((item, index) => { const score = overlap(item, trace.answer); return <article key={`${item.id}:${index}`} className="rounded-sm border border-line p-3"><div className="flex justify-between gap-2 text-sm"><span className="truncate font-medium">{item.label || item.id}</span><span className={score == null ? "text-ink-muted" : score >= 0.2 ? "text-green" : "text-ink-dim"}>{score == null ? "Unclassified" : score >= 0.2 ? "Used" : "Ignored"}</span></div><p className="mt-1 text-xs text-ink-dim">score {item.score ?? "—"} · overlap {score == null ? "—" : score.toFixed(3)} · {item.source}</p><p className="mt-2 line-clamp-3 text-xs text-ink-dim">{item.content}</p></article>; })}</div>}</section>
            <section className="rounded-sm border border-line bg-panel p-4"><h3 className="mb-3 font-semibold">Execution timeline</h3>{!trace.spans?.length ? <p className="text-sm text-ink-dim">No execution steps recorded.</p> : <ol className="max-h-80 space-y-2 overflow-y-auto">{[...trace.spans].sort((a, b) => (a.start_ms ?? Infinity) - (b.start_ms ?? Infinity)).map((span) => <li key={span.id} className="border-l-2 border-blue pl-3 text-sm"><span className="font-medium">{span.name}</span><span className="ml-2 text-xs text-ink-dim">{span.kind} · {span.status || "running"} · {span.start_ms ?? "—"}–{span.end_ms ?? "—"} ms</span></li>)}</ol>}</section>
          </div>
          {trace.answer && <section className="mt-4 rounded-sm border border-line bg-panel p-4"><h3 className="mb-2 font-semibold">Final answer</h3><p className="whitespace-pre-wrap text-sm leading-6 text-ink-dim">{trace.answer}</p></section>}
          {trace.metrics && <section className="mt-4 rounded-sm border border-line bg-panel p-4"><h3 className="mb-2 font-semibold">Metrics</h3><dl className="flex flex-wrap gap-4 text-xs">{Object.entries(trace.metrics).map(([key, value]) => <div key={key}><dt className="text-ink-muted">{key}</dt><dd className="font-mono">{value ?? "—"}</dd></div>)}</dl></section>}
        </>}
      </section>
    </div>
  </main>;
}
