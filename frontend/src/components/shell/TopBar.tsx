import { useEffect, useRef } from "react";
import { Check, ChevronDown, GitBranch, LoaderCircle, Moon, Sun } from "lucide-react";

import { useAuthUi } from "@/contexts/AuthUiContext";
import { useStudio } from "@/contexts/StudioContext";
import { cn } from "@/lib/utils";

interface TopBarProps {
  graphMenuOpen: boolean;
  onGraphMenuChange: (open: boolean) => void;
  railOpen: boolean;
}

export function TopBar({ graphMenuOpen, onGraphMenuChange, railOpen }: TopBarProps) {
  const { graphs, activeGraphId, graphsLoading, graphError, graphSwitching, changeGraph, theme, toggleTheme } = useStudio();
  const authUi = useAuthUi();
  const graphTriggerRef = useRef<HTMLButtonElement>(null);
  const graphMenuRef = useRef<HTMLDivElement>(null);
  const active = graphs.find((graph) => graph.id === activeGraphId);
  const graphLabel = graphsLoading ? "Loading graph…" : active?.label ?? "No graphs";

  useEffect(() => {
    if (!graphMenuOpen) return;
    const close = () => {
      onGraphMenuChange(false);
      graphTriggerRef.current?.focus();
    };
    const pointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Node)) return;
      if (!graphMenuRef.current?.contains(event.target) && !graphTriggerRef.current?.contains(event.target)) close();
    };
    const keyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close();
      }
    };
    document.addEventListener("pointerdown", pointerDown);
    window.addEventListener("keydown", keyDown);
    return () => {
      document.removeEventListener("pointerdown", pointerDown);
      window.removeEventListener("keydown", keyDown);
    };
  }, [graphMenuOpen, onGraphMenuChange]);
  return (
    <header
      className={cn(
        "relative z-20 flex min-w-0 items-center justify-between border-b border-line bg-panel px-2 md:px-7",
        !railOpen && "pl-12 md:pl-14",
      )}
    >
      <div className="flex min-w-0 items-center gap-1.5 md:gap-2.5">
        <span className="hidden h-7 w-7 items-center justify-center rounded-md border border-line bg-raised text-blue sm:flex">
          <GitBranch size={15} aria-hidden="true" />
        </span>
        <span className="truncate text-sm font-semibold tracking-[-0.02em] md:text-lg">graphRAG</span>
      </div>
      <div className="flex min-w-0 items-center gap-1.5 md:gap-3">
        <div className="relative">
          <button
            ref={graphTriggerRef}
            type="button"
            className="flex h-9 w-[124px] items-center gap-1.5 rounded-md border border-line bg-paper px-2 text-sm text-ink-dim transition-colors hover:bg-raised hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue md:h-10 md:w-56 md:gap-2 md:px-3"
            aria-label="Select active graph"
            aria-haspopup="menu"
            aria-expanded={graphMenuOpen}
            aria-busy={graphsLoading || graphSwitching}
            onClick={() => onGraphMenuChange(!graphMenuOpen)}
          >
            <GitBranch size={16} aria-hidden="true" />
            <span className="min-w-0 flex-1 truncate text-left font-mono text-xs">{graphLabel}</span>
            <ChevronDown size={15} aria-hidden="true" />
          </button>
          {graphMenuOpen ? (
            <div ref={graphMenuRef} role="menu" className="absolute right-0 top-12 w-[min(16rem,calc(100vw-1rem))] rounded-md border border-line bg-panel p-2 text-xs text-ink-dim shadow-lg">
              {graphsLoading ? <p className="flex items-center gap-2 p-2" role="status" aria-live="polite"><LoaderCircle size={14} className="animate-spin" /> Loading graphs…</p> : null}
              {!graphsLoading && graphError ? <p className="p-2 text-red" role="alert">{graphError}</p> : null}
              {!graphsLoading && !graphError && graphs.length === 0 ? <p className="p-2">No graphs available</p> : null}
              {graphs.map((graph) => (
                <button key={graph.id} type="button" role="menuitem" aria-current={graph.id === activeGraphId ? "true" : undefined} disabled={graphSwitching || graph.id === activeGraphId} onClick={() => { onGraphMenuChange(false); void changeGraph(graph.id); }} className="flex w-full items-center gap-2 rounded-sm px-2 py-2 text-left hover:bg-raised disabled:opacity-50">
                  <Check size={14} className={graph.id === activeGraphId ? "opacity-100" : "opacity-0"} />
                  <span className="min-w-0 flex-1 truncate">{graph.label}</span>
                </button>
              ))}
            </div>
          ) : null}
        </div>
        <button
          type="button"
          className="flex h-9 w-9 items-center justify-center rounded-md border border-line bg-paper text-ink-dim transition-colors hover:bg-raised hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue md:h-10 md:w-10"
          aria-label={theme === "dark" ? "Use light theme" : "Use dark theme"}
          onClick={toggleTheme}
        >
          {theme === "dark" ? <Sun size={18} aria-hidden="true" /> : <Moon size={18} aria-hidden="true" />}
        </button>
        {authUi.enabled ? <div className="flex shrink-0 items-center">{authUi.control}</div> : null}
      </div>
    </header>
  );
}
