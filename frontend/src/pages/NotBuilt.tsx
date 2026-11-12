// Honest placeholder for sections scheduled in later build phases. It has no
// controls, so nothing on it pretends to work.
const WHAT: Record<string, string> = {
  "Execution Environments": "Local PowerShell, SSH, Docker and a disposable WSL sandbox, each with a connection test.",
  "Agent Tasks": "Agent runs with budgets, approvals, live step logs and cancellation.",
  Subagents: "Bounded parallel child agents with their own budgets and a cancellable task tree.",
  Skills: "SKILL.md library with import, export, versions and approval for agent-proposed edits.",
};

export function NotBuilt({ label, phase }: { label: string; phase: number }) {
  return (
    <div className="page">
      <div className="empty">
        <strong>{label} is not built yet. It is scheduled for build phase {phase}.</strong>
        {WHAT[label] ?? ""}
      </div>
    </div>
  );
}
