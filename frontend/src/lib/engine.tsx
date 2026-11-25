// Engine status shared across the app, kept in sync with backend events.
// Status and logs live in separate contexts: log lines arrive many times a
// second while a model loads or generates, and only the log view needs them,
// so pages that read the status do not re-render on every line.
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { listen } from "@tauri-apps/api/event";
import { api, type EngineEvent, type EngineStatus, type LoadProgress } from "./api";

const MAX_LOG_LINES = 2000;

const StatusCtx = createContext<EngineStatus | null>(null);
const LogsCtx = createContext<string[]>([]);
// Load progress arrives up to 5 times a second while a model loads; only the
// bar components read it, so it has its own context too.
const ProgressCtx = createContext<LoadProgress | null>(null);

export function EngineProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState<LoadProgress | null>(null);
  // Log lines arrive fast during load; batch them into one render per frame.
  const pending = useRef<string[]>([]);
  const frame = useRef<number | null>(null);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    api.engineStatus().then((s) => !cancelled && setStatus(s));
    api.engineLogs(MAX_LOG_LINES).then((l) => !cancelled && setLogs(l));
    listen<EngineEvent>("engine://event", (e) => {
      const ev = e.payload;
      if (ev.type === "status") {
        setStatus(ev.status);
        if (ev.status.state === "loading" && ev.status.pid == null) setLogs([]);
        // A new load starts from zero; an unload clears the last one. A
        // failed load keeps its last numbers next to the error.
        if ((ev.status.state === "loading" && ev.status.pid == null) || ev.status.state === "unloaded" || ev.status.state === "unloading") setProgress(null);
      } else if (ev.type === "load_progress") {
        setProgress(ev.progress);
      } else if (ev.type === "log") {
        pending.current.push(ev.line);
        if (frame.current == null) {
          frame.current = requestAnimationFrame(() => {
            frame.current = null;
            const add = pending.current;
            pending.current = [];
            setLogs((prev) => {
              const next = prev.concat(add);
              return next.length > MAX_LOG_LINES ? next.slice(next.length - MAX_LOG_LINES) : next;
            });
          });
        }
      }
    }).then((u) => {
      if (cancelled) u();
      else unlisten = u;
    });
    return () => {
      cancelled = true;
      unlisten?.();
      if (frame.current != null) cancelAnimationFrame(frame.current);
    };
  }, []);

  return (
    <StatusCtx.Provider value={status}>
      <ProgressCtx.Provider value={progress}>
        <LogsCtx.Provider value={logs}>{children}</LogsCtx.Provider>
      </ProgressCtx.Provider>
    </StatusCtx.Provider>
  );
}

/** The engine status (does not change on log lines). */
export function useEngine(): { status: EngineStatus | null } {
  return { status: useContext(StatusCtx) };
}

/** How far the current model load has got (null when none is running or finished). */
export function useLoadProgress(): LoadProgress | null {
  return useContext(ProgressCtx);
}

/** The backend log lines (only the log view should use this). */
export function useEngineLogs(): string[] {
  return useContext(LogsCtx);
}
