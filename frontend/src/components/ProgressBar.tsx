// A labelled progress bar (role=progressbar) that uses the app's .progress look.

/** "about 3 min left", "under a minute left"; null when there is no estimate. */
export function etaText(secs: number | null | undefined): string | null {
  if (secs == null || !Number.isFinite(secs)) return null;
  if (secs <= 0) return null;
  if (secs < 60) return secs < 10 ? "a few seconds left" : "under a minute left";
  const m = Math.round(secs / 60);
  if (m < 90) return `~${m} min left`;
  const h = Math.floor(m / 60);
  return `~${h} h ${m % 60} min left`;
}

export function ProgressBar({ percent, label, valueText }: { percent: number; label: string; valueText?: string }) {
  const p = Math.max(0, Math.min(100, Number.isFinite(percent) ? percent : 0));
  return (
    <div
      className="progress load-bar"
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(p)}
      aria-valuetext={valueText ?? `${Math.round(p)} percent`}
    >
      <span style={{ width: `${p}%` }} />
    </div>
  );
}
