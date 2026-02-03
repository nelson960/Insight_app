export function FirstRunSetupModal(props: {
  open: boolean;
  onOpenSettings: () => void;
}) {
  const { open, onOpenSettings } = props;
  if (!open) return null;

  return (
    <div className="setup-backdrop" role="dialog" aria-modal="true" aria-label="First run setup">
      <div className="setup-modal">
        <div className="setup-header">
          <div className="setup-title">Setup</div>
        </div>
        <div className="setup-body">
          <div className="setup-row">
            <div>
              <div className="setup-label">Model</div>
              <div className="setup-desc">
                Configure a GGUF model and download the embedding model.
              </div>
            </div>
            <button
              className="setup-btn primary"
              type="button"
              onClick={onOpenSettings}
            >
              Open Settings
            </button>
          </div>
        </div>
        <div className="setup-actions" />
      </div>
    </div>
  );
}
