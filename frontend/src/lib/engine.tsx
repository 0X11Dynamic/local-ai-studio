// Engine status shared across the app, kept in sync with backend events.
// Status and logs live in separate contexts: log lines arrive many times a
// second while a model loads or generates, and only the log view needs them,
// so pages that read the status do not re-render on every line.
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { listen } from "@tauri-apps/api/event";
import { api, type EngineEvent, type EngineStatus } from "./api";

const MAX_LOG_LINES = 2000;

const StatusCtx = createContext<EngineStatus | null>(null);
const LogsCtx = createContext<string[]>([]);

export function EngineProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
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
      <LogsCtx.Provider value={logs}>{children}</LogsCtx.Provider>
    </StatusCtx.Provider>
  );
}

/** The engine status (does not change on log lines). */
export function useEngine(): { status: EngineStatus | null } {
  return { status: useContext(StatusCtx) };
}

/** The backend log lines (only the log view should use this). */
export function useEngineLogs(): string[] {
  return useContext(LogsCtx);
}
