import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  /** Shown in the message, e.g. "Agent Tasks". */
  label: string;
  /**
   * "page": the error replaces only the page; "app": the whole window (last
   * resort); "inline": a one-line note (small areas such as the status bar).
   */
  scope?: "page" | "app" | "inline";
  children: ReactNode;
}

interface State {
  error: Error | null;
  stack: string | null;
  attempt: number;
}

/**
 * Catches a render error so one broken page shows the error with a reload
 * button instead of blanking the whole window (React unmounts everything on
 * an uncaught render error).
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, stack: null, attempt: 0 };

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error(`[${this.props.label}] render error:`, error, info.componentStack);
    this.setState({ stack: info.componentStack ?? null });
  }

  reload = () => {
    if (this.props.scope === "app") {
      window.location.reload();
      return;
    }
    // Remount the page's tree from scratch (fresh state, data fetched again).
    this.setState((s) => ({ error: null, stack: null, attempt: s.attempt + 1 }));
  };

  render() {
    const { error, stack, attempt } = this.state;
    if (!error) return <div key={attempt} style={{ display: "contents" }}>{this.props.children}</div>;
    if (this.props.scope === "inline") {
      return (
        <span className="row danger-text" role="alert" title={error.message} style={{ fontSize: 12 }}>
          {this.props.label} error <button className="btn sm ghost" onClick={this.reload}>Reload</button>
        </span>
      );
    }
    return (
      <div className="page page-error" role="alert" aria-label={`${this.props.label} failed`}>
        <div className="panel" style={{ display: "grid", gap: 10 }}>
          <strong style={{ fontWeight: 600 }}>{this.props.scope === "app" ? "The app hit an error" : `The ${this.props.label} page hit an error`}</strong>
          <span className="dim">
            {this.props.scope === "app" ? "Reload to continue." : "The rest of the app still works. Reload the page to try again, or pick another page."}
          </span>
          <pre className="mono danger-text" style={{ whiteSpace: "pre-wrap", margin: 0, fontSize: 12 }}>{error.message || String(error)}</pre>
          {stack && (
            <details>
              <summary className="faint">Details</summary>
              <pre className="mono faint" style={{ whiteSpace: "pre-wrap", fontSize: 11 }}>{stack.trim()}</pre>
            </details>
          )}
          <div className="row">
            <button className="btn sm primary" onClick={this.reload}>Reload page</button>
          </div>
        </div>
      </div>
    );
  }
}
