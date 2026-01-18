export type HealthIssue = {
  code: string;
  severity: "error" | "warning";
  message: string;
  fix: string;
  action?: string;
};

export type HealthReport = {
  ok: boolean;
  issues: HealthIssue[];
  checks?: Record<string, any>;
};

export function StartupHealthModal(props: {
  open: boolean;
  report: HealthReport | null;
  onOpenSettings: () => void;
}) {
  const { open, report, onOpenSettings } = props;
  if (!open || !report) return null;

  const issues = report.issues || [];
  const hasIssues = issues.length > 0;
  const needsSettings = issues.some((i) => i.action === "open_settings");

  return (
    <div className="health-backdrop" role="dialog" aria-modal="true">
      <div className="health-modal">
        <div className="health-header">
          <div className="health-title">
            {hasIssues ? "Setup required" : "Startup checks"}
          </div>
        </div>
        <div className="health-body">
          {issues.length === 0 ? (
            <div className="health-empty">All checks passed.</div>
          ) : (
            issues.map((issue) => (
              <div
                key={issue.code}
                className={`health-issue health-issue-${issue.severity}`}
              >
                <div className="health-issue-title">{issue.message}</div>
                <div className="health-issue-fix">{issue.fix}</div>
              </div>
            ))
          )}
        </div>
        <div className="health-actions">
          {needsSettings ? (
            <button className="health-btn primary" onClick={onOpenSettings}>
              Open Settings
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
