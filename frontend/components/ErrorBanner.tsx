"use client";

import { ResearchApiError } from "../lib/api";

const KIND_HINT: Record<ResearchApiError["kind"], string> = {
  unavailable: "Check that the backend is running, then try again.",
  timeout: "The question may need a longer timeout, or the backend may be busy.",
  invalid_question: "Edit the question and submit again.",
  server_error: "The failure was logged server-side with the request ID above.",
  research_failed: "The failure was logged server-side with the request ID above.",
  unknown_job: "Submit the question again to start a new research request.",
};

/** Error states — user-safe message plus request ID, never a stack trace. */
export function ErrorBanner({ error, onDismiss }: { error: ResearchApiError; onDismiss?: () => void }) {
  return (
    <div role="alert" className="error-banner" data-testid="error-banner">
      <strong className="error-heading">Something went wrong</strong>
      <p className="error-message">{error.message}</p>
      {error.requestId && (
        <p className="error-request">
          Request ID: <code>{error.requestId}</code>
        </p>
      )}
      <p className="hint">{KIND_HINT[error.kind]}</p>
      {onDismiss && (
        <button type="button" onClick={onDismiss} className="secondary-button">
          Dismiss
        </button>
      )}
    </div>
  );
}
