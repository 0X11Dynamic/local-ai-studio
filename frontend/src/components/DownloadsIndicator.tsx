import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api2, type DownloadEvent } from "../lib/api2";
import { percent } from "../lib/format";
import type { PageId } from "../App";

/** Topbar summary of running downloads; opens Discover. */
export function DownloadsIndicator({ go }: { go: (p: PageId) => void }) {
  const [dl, setDl] = useState<Record<string, DownloadEvent>>({});
  useEffect(() => {
    api2.listDownloads().then((l) => setDl(Object.fromEntries(l.map((d) => [d.id, d]))));
    const un = listen<DownloadEvent>("download://event", (e) => setDl((d) => ({ ...d, [e.payload.id]: e.payload })));
    return () => {
      un.then((f) => f());
    };
  }, []);
  const running = Object.values(dl).filter((d) => d.state === "running" || d.state === "queued");
  if (!running.length) return null;
  const done = running.reduce((a, d) => a + d.downloaded, 0);
  const total = running.reduce((a, d) => a + d.total, 0);
  return (
    <button className="btn ghost engine-pill" onClick={() => go("discover")} title="Open Discover">
      Downloading {running.length} <span className="mono">{percent(done, total).toFixed(0)}%</span>
    </button>
  );
}
