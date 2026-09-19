import { useCallback, useEffect, useRef, useState } from "react";
import { Outlet, useNavigate } from "react-router-dom";

import { NavigationRail } from "@/components/shell/NavigationRail";
import { TopBar } from "@/components/shell/TopBar";

export interface ShellOutletContext {
  commandInputRef: React.RefObject<HTMLInputElement>;
  focusRequest: number;
}

const MOBILE_QUERY = "(max-width: 767px)";

export function AppShell() {
  const navigate = useNavigate();
  const commandInputRef = useRef<HTMLInputElement>(null);
  const railOpenButtonRef = useRef<HTMLButtonElement>(null);
  const railCloseButtonRef = useRef<HTMLButtonElement>(null);
  const [mobile, setMobile] = useState(() => window.matchMedia(MOBILE_QUERY).matches);
  const [railOpen, setRailOpen] = useState(() => !window.matchMedia(MOBILE_QUERY).matches);
  const [graphMenuOpen, setGraphMenuOpen] = useState(false);
  const [focusRequest, setFocusRequest] = useState(0);

  useEffect(() => {
    const media = window.matchMedia(MOBILE_QUERY);
    const update = (event: MediaQueryListEvent) => {
      setMobile(event.matches);
      setRailOpen(!event.matches);
    };
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  const focusCommand = useCallback(() => {
    navigate("/");
    setFocusRequest((request) => request + 1);
  }, [navigate]);

  const openRail = useCallback(() => {
    setRailOpen(true);
    window.requestAnimationFrame(() => railCloseButtonRef.current?.focus());
  }, []);

  const closeRail = useCallback((restoreFocus = true) => {
    setRailOpen(false);
    if (restoreFocus) window.requestAnimationFrame(() => railOpenButtonRef.current?.focus());
  }, []);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      const editable =
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement ||
        (event.target instanceof HTMLElement && event.target.isContentEditable);
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        focusCommand();
        return;
      }
      if (editable || event.defaultPrevented) return;
      if ((event.metaKey || event.ctrlKey) && event.key === "\\") {
        event.preventDefault();
        if (railOpen) closeRail();
        else openRail();
      } else if (event.key === "Escape") {
        if (mobile && railOpen) closeRail();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [closeRail, focusCommand, mobile, openRail, railOpen]);

  return (
    <div
      className="grid h-dvh overflow-hidden bg-paper text-ink transition-[grid-template-columns] duration-200 motion-reduce:transition-none"
      style={{ gridTemplateColumns: mobile ? "1fr" : railOpen ? "80px 1fr" : "0 1fr" }}
    >
      <NavigationRail
        mobile={mobile}
        open={railOpen}
        closeButtonRef={railCloseButtonRef}
        onClose={() => closeRail(false)}
        onToggle={() => closeRail()}
      />
      {mobile && railOpen ? (
        <button
          type="button"
          className="fixed inset-0 z-30 bg-black/50"
          aria-label="Close navigation"
          onClick={() => closeRail()}
        />
      ) : null}
      <div className="relative grid min-w-0 grid-rows-[56px_1fr] overflow-hidden md:grid-rows-[80px_1fr]">
        {!railOpen ? (
          <button
            ref={railOpenButtonRef}
            type="button"
            className="absolute left-0 top-0 z-50 flex h-10 w-10 items-center justify-center border-b border-r border-line bg-panel font-mono text-lg text-ink-dim transition-colors hover:bg-raised hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue"
            aria-label="Open navigation"
            onClick={openRail}
          >
            »
          </button>
        ) : null}
        <TopBar
          graphMenuOpen={graphMenuOpen}
          onGraphMenuChange={setGraphMenuOpen}
          railOpen={railOpen}
        />
        <main className="min-h-0 min-w-0 overflow-hidden">
          <Outlet context={{ commandInputRef, focusRequest } satisfies ShellOutletContext} />
        </main>
      </div>
    </div>
  );
}
