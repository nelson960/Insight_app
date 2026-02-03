import { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

/**
 * Error Boundary component to catch and handle React component errors.
 * Prevents the entire app from crashing due to errors in subtree components.
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("ErrorBoundary caught an error:", error);
    console.error("Error Info:", errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: "2rem",
          textAlign: "center",
          fontFamily: "system-ui, -apple-system, sans-serif",
          color: "#e11d48",
          backgroundColor: "#fef2f2",
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}>
          <h1 style={{ fontSize: "1.5rem", marginBottom: "1rem" }}>
            Something went wrong
          </h1>
          <p style={{ color: "#64748b", marginBottom: "1.5rem" }}>
            The application encountered an unexpected error.
          </p>
          <details style={{
            marginBottom: "1.5rem",
            padding: "1rem",
            backgroundColor: "#fff",
            borderRadius: "0.5rem",
            border: "1px solid #fecaca",
            maxWidth: "600px",
            textAlign: "left",
            fontSize: "0.875rem"
          }}>
            <summary style={{ cursor: "pointer", fontWeight: "bold" }}>
              Error details
            </summary>
            <pre style={{
              marginTop: "1rem",
              overflow: "auto",
              fontSize: "0.75rem",
              color: "#64748b"
            }}>
              {this.state.error?.stack || String(this.state.error)}
            </pre>
          </details>
          <button
            onClick={() => window.location.reload()}
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: "#e11d48",
              color: "white",
              border: "none",
              borderRadius: "0.375rem",
              cursor: "pointer",
              fontSize: "1rem"
            }}
          >
            Reload Application
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
