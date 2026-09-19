import { useEffect, useRef, useState } from "react";
import { BarChart3, ExternalLink, FileText, Info, Link2, X } from "lucide-react";

import { EntityIcon } from "@/components/graph/EntityIcon";
import { summarize } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { TraceNode } from "@/types/trace";

const MIN_HEIGHT = 220;
const DEFAULT_HEIGHT = 320;

function repositoryName(graphId?: string) {
  if (!graphId) return undefined;
  const match = /^([^_]+)__([^/]+?)(?:\.lbug)?$/.exec(graphId);
  return match?.[2];
}

function ageLabel(age?: number | null) {
  if (age === undefined || age === null) return "—";
  if (age < 1) return "Today";
  return `${Math.round(age)} day${Math.round(age) === 1 ? "" : "s"} ago`;
}

interface NodeInspectorProps {
  node: TraceNode;
  activeGraphId?: string;
  maxHeight: () => number;
  onClose: () => void;
}

export function NodeInspector({ node, activeGraphId, maxHeight, onClose }: NodeInspectorProps) {
  const [height, setHeight] = useState(DEFAULT_HEIGHT);
  const [summary, setSummary] = useState<string>();
  const [summaryFailed, setSummaryFailed] = useState(false);
  const [loading, setLoading] = useState(false);
  const requestRef = useRef(0);
  const sourceText = node.meta?.summaryText;
  const sourceUrl = node.meta?.sourceUrl;
  const provenance = repositoryName(activeGraphId);
  const maximum = Math.max(MIN_HEIGHT, Math.floor(maxHeight() * 0.65));
  const clamp = (value: number) => Math.min(Math.max(MIN_HEIGHT, value), maximum);

  useEffect(() => {
    const current = ++requestRef.current;
    setSummary(undefined);
    setSummaryFailed(false);
    if (!sourceText) {
      setLoading(false);
      return;
    }
    setLoading(true);
    void summarize(node.id, sourceText)
      .then((response) => {
        if (requestRef.current !== current) return;
        if (response.error || response.summary.trim() === "") {
          setSummaryFailed(true);
          return;
        }
        setSummary(response.summary);
      })
      .catch(() => {
        if (requestRef.current === current) setSummaryFailed(true);
      })
      .finally(() => {
        if (requestRef.current === current) setLoading(false);
      });
    return () => { requestRef.current += 1; };
  }, [node.id, sourceText]);

  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, [onClose]);

  const startResize = (event: React.PointerEvent<HTMLDivElement>) => {
    const originY = event.clientY;
    const originHeight = height;
    event.currentTarget.setPointerCapture(event.pointerId);
    const move = (moveEvent: PointerEvent) => setHeight(clamp(originHeight + originY - moveEvent.clientY));
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  };

  const scores = [
    ["Combined score", node.score, true],
    ["Vector similarity", node.similarity, false],
    ["Graph relevance", node.meta?.scoreGraph, false],
  ] as const;

  return (
    <aside className="absolute inset-x-0 bottom-0 z-20 flex flex-col border-t border-line bg-panel/98 backdrop-blur" style={{ height: clamp(height) }} aria-label={`Inspector for ${node.label}`}>
      <div
        className="absolute inset-x-0 -top-2 flex h-4 cursor-ns-resize touch-none items-center justify-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue"
        role="separator" aria-orientation="horizontal" aria-label="Resize inspector"
        aria-valuenow={Math.round(clamp(height))} aria-valuemin={MIN_HEIGHT} aria-valuemax={maximum} tabIndex={0}
        onPointerDown={startResize}
        onKeyDown={(event) => {
          if (event.key === "ArrowUp") { event.preventDefault(); setHeight((value) => clamp(value + 16)); }
          if (event.key === "ArrowDown") { event.preventDefault(); setHeight((value) => clamp(value - 16)); }
        }}
      ><span className="h-0.5 w-10 rounded-full bg-ink-muted" /></div>
      <header className="flex min-h-[76px] items-center gap-3 border-b border-line px-4 md:px-6">
        <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-sm bg-blue/20 text-blue"><EntityIcon type={node.type} size={22} /></span>
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-base font-semibold text-ink md:text-lg">{node.label}</h2>
          <p className="truncate font-mono text-[11px] uppercase tracking-wide text-ink-dim">{node.type} · {node.id}</p>
        </div>
        {sourceUrl ? <a href={sourceUrl} target="_blank" rel="noreferrer" className="inspector-icon-button" aria-label="Open source"><ExternalLink size={17} /></a> : null}
        <button type="button" className="inspector-icon-button" onClick={onClose} aria-label="Close inspector"><X size={19} /></button>
      </header>
      <div className="inspector-grid min-h-0 flex-1 overflow-y-auto scrollbar-thin">
        <section className="inspector-section">
          <h3 className="inspector-title"><BarChart3 size={16} /> Scores</h3>
          <dl className="space-y-2 text-sm">
            {scores.map(([label, value, primary]) => value === undefined ? null : (
              <div key={label} className="flex justify-between gap-3"><dt className="text-ink-dim">{label}</dt><dd className={cn("font-mono text-ink", primary && "text-green")}>{value.toFixed(3)}</dd></div>
            ))}
          </dl>
        </section>
        <section className="inspector-section">
          <h3 className="inspector-title"><FileText size={16} /> Summary</h3>
          {loading ? <div className="space-y-2" role="status" aria-live="polite" aria-label="Loading summary"><div className="h-3 w-full animate-pulse rounded-sm bg-raised" /><div className="h-3 w-5/6 animate-pulse rounded-sm bg-raised" /><div className="h-3 w-2/3 animate-pulse rounded-sm bg-raised" /></div>
            : !sourceText ? <p className="text-sm text-ink-dim">No summary for this node</p>
              : <p className={cn("text-sm leading-6 text-ink-dim", summaryFailed && "line-clamp-6")}>{summaryFailed ? sourceText : summary}</p>}
        </section>
        <section className="inspector-section">
          <h3 className="inspector-title"><Info size={16} /> Metadata</h3>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
            <dt className="text-ink-muted">Type</dt><dd className="truncate font-mono text-ink-dim">{node.type}</dd>
            <dt className="text-ink-muted">ID</dt><dd className="truncate font-mono text-ink-dim" title={node.id}>{node.id}</dd>
            <dt className="text-ink-muted">Age</dt><dd className="text-ink-dim">{ageLabel(node.meta?.ageDays)}</dd>
            <dt className="text-ink-muted">Recency</dt><dd className="font-mono text-ink-dim">{node.meta?.recency?.toFixed(3) ?? "—"}</dd>
            <dt className="text-ink-muted">Connections</dt><dd className="font-mono text-ink-dim">{node.meta?.connections ?? "—"}</dd>
            <dt className="text-ink-muted">Linked seed</dt><dd className="text-ink-dim">{node.meta?.linkedSeed ? "Yes" : "No"}</dd>
          </dl>
        </section>
        <section className="inspector-section">
          <h3 className="inspector-title"><Link2 size={16} /> Source</h3>
          {provenance ? <p className="mb-2 text-xs text-ink-muted">Repository · <span className="font-mono text-ink-dim">{provenance}</span></p> : null}
          {sourceUrl && activeGraphId ? <a href={sourceUrl} target="_blank" rel="noreferrer" className="break-all text-sm text-blue underline decoration-blue/40 underline-offset-4">{sourceUrl}</a> : <p className="text-sm text-ink-dim">No source link</p>}
        </section>
      </div>
    </aside>
  );
}
