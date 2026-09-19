import { useEffect, useRef, type RefObject } from "react";
import { GitBranch, History, Home, Network, Search, Settings, Waypoints } from "lucide-react";
import { NavLink } from "react-router-dom";

import { cn } from "@/lib/utils";

const items = [
  { label: "Home", to: "/", icon: Home, end: true },
  { label: "Traces", to: "/traces", icon: Waypoints },
  { label: "Queries", to: "/queries", icon: Search },
  { label: "Graphs", to: "/graphs", icon: GitBranch },
  { label: "Sessions", to: "/sessions", icon: History },
  { label: "Settings", to: "/settings", icon: Settings },
] as const;

interface NavigationRailProps {
  mobile: boolean;
  open: boolean;
  closeButtonRef: RefObject<HTMLButtonElement>;
  onClose: () => void;
  onToggle: () => void;
}

export function NavigationRail({ mobile, open, closeButtonRef, onClose, onToggle }: NavigationRailProps) {
  const railRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (railRef.current) railRef.current.inert = !open;
  }, [open]);

  return (
    <aside
      ref={railRef}
      className={cn(
        "z-40 flex h-dvh w-20 flex-col border-r border-line bg-panel transition-transform duration-200 motion-reduce:transition-none",
        mobile && "fixed inset-y-0 left-0",
        !open && (mobile ? "-translate-x-full" : "pointer-events-none -translate-x-full"),
      )}
      aria-hidden={!open}
    >
      <button
        ref={closeButtonRef}
        type="button"
        className="flex h-20 items-center justify-center border-b border-line font-mono text-xl text-ink-dim transition-colors hover:bg-raised hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue max-md:h-14"
        aria-label="Collapse navigation"
        onClick={onToggle}
      >
        «
      </button>
      <nav className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto py-3" aria-label="Primary navigation">
        {items.map(({ label, to, icon: Icon, ...item }) => (
          <NavLink
            key={to}
            to={to}
            end={"end" in item ? item.end : undefined}
            onClick={mobile ? onClose : undefined}
            className={({ isActive }) =>
              cn(
                "mx-2 flex h-[66px] flex-col items-center justify-center gap-1.5 rounded-md text-ink-dim transition-colors hover:bg-raised hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue",
                isActive && "bg-raised text-ink",
              )
            }
          >
            <Icon size={20} strokeWidth={1.8} aria-hidden="true" />
            <span className="text-xs font-medium">{label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="flex h-12 items-center justify-center border-t border-line text-ink-muted" aria-hidden="true">
        <Network size={17} />
      </div>
    </aside>
  );
}
