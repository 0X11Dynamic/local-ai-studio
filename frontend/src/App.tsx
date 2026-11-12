import { useEffect, useState, type ComponentType } from "react";
import {
  ChatCircle, Compass, Cube, ListChecks, TerminalWindow, TreeStructure, BookOpen, Plugs,
  Code, Cpu, Gear, TextAlignLeft, Gauge, SidebarSimple, type IconProps,
} from "@phosphor-icons/react";
import { EngineProvider, useEngine } from "./lib/engine";
import { api } from "./lib/api";
import { Overview } from "./pages/Overview";
import { Runtimes } from "./pages/Runtimes";
import { Models } from "./pages/Models";
import { Settings } from "./pages/Settings";
import { Discover } from "./pages/Discover";
import { Chat } from "./pages/Chat";
import { Instructions } from "./pages/Instructions";
import { Developer } from "./pages/Developer";
import { Mcp } from "./pages/Mcp";
import { DownloadsIndicator } from "./components/DownloadsIndicator";
import { ApprovalsIndicator } from "./components/ApprovalsIndicator";
import { NotBuilt } from "./pages/NotBuilt";
import { Environments } from "./pages/Environments";
import { AgentTasks } from "./pages/AgentTasks";
import { Subagents } from "./pages/Subagents";
import { Skills } from "./pages/Skills";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { FirstRun } from "./components/FirstRun";

export type PageId =
  | "overview" | "chat" | "discover" | "models" | "runtimes" | "agents" | "subagents"
  | "environments" | "skills" | "mcp" | "instructions" | "developer" | "settings";

interface NavItem { id: PageId; label: string; icon: ComponentType<IconProps>; phase?: number }

// 13 destinations grouped into 4 sections so no group exceeds 4 items.
const NAV: { title: string; items: NavItem[] }[] = [
  {
    title: "Work",
    items: [
      { id: "chat", label: "Chat", icon: ChatCircle },
      { id: "agents", label: "Agent Tasks", icon: ListChecks },
      { id: "subagents", label: "Subagents", icon: TreeStructure },
    ],
  },
  {
    title: "Models",
    items: [
      { id: "discover", label: "Discover", icon: Compass },
      { id: "models", label: "My Models", icon: Cube },
      { id: "runtimes", label: "Runtimes", icon: Cpu },
    ],
  },
  {
    title: "Agent setup",
    items: [
      { id: "instructions", label: "System Instructions", icon: TextAlignLeft },
      { id: "skills", label: "Skills", icon: BookOpen },
      { id: "mcp", label: "MCP Connections", icon: Plugs },
      { id: "environments", label: "Execution Environments", icon: TerminalWindow },
    ],
  },
  {
    title: "System",
    items: [
      { id: "overview", label: "Overview", icon: Gauge },
      { id: "developer", label: "Developer / API", icon: Code },
      { id: "settings", label: "Settings", icon: Gear },
    ],
  },
];

const ALL = NAV.flatMap((g) => g.items);

function EngineIndicator({ go }: { go: (p: PageId) => void }) {
  const { status } = useEngine();
  if (!status) return null;
  const label = {
    unloaded: "No model loaded",
    loading: "Loading model",
    loaded: "Loaded",
    unloading: "Unloading",
    failed: "Load failed",
  }[status.state];
  const off = status.gpu_layers_offloaded;
  return (
    <button className="btn ghost engine-pill" onClick={() => go("models")} title="Open My Models">
      <span className={status.state === "failed" ? "danger-text" : status.state === "loaded" ? "" : "faint"}>{label}</span>
      {status.state === "loaded" && (
        <>
          <strong className="mono">{status.props?.model_path?.split(/[\\/]/).pop()}</strong>
          <span className="mono faint">{status.runtime_id}</span>
          {off && <span className="mono faint">GPU {off[0]}/{off[1]}</span>}
          {status.warnings.length > 0 && <span style={{ color: "var(--accent)" }}>Slow: VRAM full</span>}
        </>
      )}
    </button>
  );
}

function Shell() {
  const [page, setPage] = useState<PageId>("overview");
  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => {
    api.getSettings().then((s) => {
      if (s["ui.sidebar_collapsed"] === true) setCollapsed(true);
      if (s["ui.theme"] === "light") document.documentElement.dataset.theme = "light";
    });
  }, []);
  const toggle = () => {
    const next = !collapsed;
    setCollapsed(next);
    api.setSetting("ui.sidebar_collapsed", next);
  };
  const current = ALL.find((i) => i.id === page)!;

  return (
    <div className={`shell${collapsed ? " collapsed" : ""}`}>
      <nav className="sidebar" aria-label="Main">
        <div className="brand">
          <span className="brand-mark" aria-hidden />
          <span className="brand-name">Local AI Studio</span>
        </div>
        {NAV.map((g) => (
          <div className="nav-group" key={g.title}>
            <div className="nav-group-title">{g.title}</div>
            {g.items.map((it) => (
              <button
                key={it.id}
                className="nav-item"
                aria-current={page === it.id ? "page" : undefined}
                onClick={() => setPage(it.id)}
                title={collapsed ? it.label : undefined}
              >
                <it.icon size={16} weight={page === it.id ? "fill" : "regular"} aria-hidden />
                <span className="nav-label">{it.label}</span>
                {it.phase && <span className="nav-badge" title={`Arrives in build phase ${it.phase}`}>P{it.phase}</span>}
              </button>
            ))}
          </div>
        ))}
        <div className="sidebar-foot">
          <button className="nav-item" onClick={toggle} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
            <SidebarSimple size={16} aria-hidden />
            <span className="nav-label">Collapse</span>
          </button>
        </div>
      </nav>
      <div className="main">
        <header className="topbar">
          <h1>{current.label}</h1>
          <div className="topbar-right">
            <ErrorBoundary label="Status bar" scope="inline">
              <ApprovalsIndicator go={setPage} />
              <DownloadsIndicator go={setPage} />
              <EngineIndicator go={setPage} />
            </ErrorBoundary>
          </div>
        </header>
        <main className={`content${page === "chat" ? " flush" : ""}`}>
          {/* One boundary per page (keyed by page): a render error shows on that page only. */}
          <ErrorBoundary key={page} label={current.label}>
            {page === "chat" && <Chat go={setPage} />}
            {page === "discover" && <Discover go={setPage} />}
            {page === "instructions" && <Instructions />}
            {page === "developer" && <Developer />}
            {page === "mcp" && <Mcp />}
            {page === "environments" && <Environments />}
            {page === "agents" && <AgentTasks />}
            {page === "subagents" && <Subagents go={setPage} />}
            {page === "skills" && <Skills />}
            {page === "overview" && <Overview go={setPage} />}
            {page === "runtimes" && <Runtimes />}
            {page === "models" && <Models go={setPage} />}
            {page === "settings" && <Settings />}
            {current.phase && <NotBuilt label={current.label} phase={current.phase} />}
          </ErrorBoundary>
        </main>
      </div>
      <ErrorBoundary label="Setup" scope="inline"><FirstRun go={setPage} /></ErrorBoundary>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary label="app" scope="app">
      <EngineProvider>
        <Shell />
      </EngineProvider>
    </ErrorBoundary>
  );
}
