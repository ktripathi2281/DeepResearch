"use client";

import type { AnswerStatus } from "../types/research";

const STATUS_COPY: Record<AnswerStatus, { heading: string; body: string }> = {
  answered: {
    heading: "Answered",
    body: "Based on the available evidence.",
  },
  insufficient_evidence: {
    heading: "Insufficient evidence",
    body: "DeepResearch could not find enough evidence to answer this reliably.",
  },
  conflicting_evidence: {
    heading: "Conflicting evidence",
    body: "The available sources disagree on this point. Both sides are preserved below.",
  },
  no_evidence: {
    heading: "No evidence",
    body: "No relevant evidence was found, so no grounded answer is offered.",
  },
};

/** Status banner — text plus shape, never color alone. */
export function StatusBanner({ status }: { status: AnswerStatus }) {
  const copy = STATUS_COPY[status];
  return (
    <div role="status" className={`status-banner status-${status}`} data-testid="status-banner">
      <strong className="status-heading">{copy.heading}</strong>
      <p className="status-body">{copy.body}</p>
    </div>
  );
}
