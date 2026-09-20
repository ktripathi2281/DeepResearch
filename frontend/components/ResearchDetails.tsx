"use client";

import type { ResearchResult } from "../types/research";

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

function verificationSummary(result: ResearchResult): string {
  const verification = result.verification;
  if (!verification) return "Not run";
  const counts = verification.counts;
  const interesting = ["supported", "unsupported", "insufficient_evidence", "unverifiable"]
    .filter((key) => (counts[key] ?? 0) > 0)
    .map((key) => `${counts[key]} ${key.replace(/_/g, " ")}`);
  return interesting.length > 0 ? interesting.join(", ") : "No cited claims";
}

/**
 * User-safe research details — identifiers, counts, durations, and
 * model names. Never chain-of-thought, prompts, or raw tool arguments.
 */
export function ResearchDetails({ result }: { result: ResearchResult }) {
  const details = result.research_details;
  return (
    <section aria-label="Research details" className="panel">
      <h2 className="panel-title">Research details</h2>
      <dl className="details-list">
        <div className="details-row">
          <dt>Request ID</dt>
          <dd>
            <code data-testid="details-request-id">{details.request_id}</code>
          </dd>
        </div>
        <div className="details-row">
          <dt>Time</dt>
          <dd>{formatMs(details.elapsed_ms)}</dd>
        </div>
        <div className="details-row">
          <dt>Evidence</dt>
          <dd data-testid="details-evidence-count">{details.evidence_count} items</dd>
        </div>
        <div className="details-row">
          <dt>Citations</dt>
          <dd data-testid="details-citation-count">
            {details.citation_count} valid
            {details.invalid_citation_count > 0 &&
              `, ${details.invalid_citation_count} invalid`}
          </dd>
        </div>
        {details.candidates !== null && details.candidates !== undefined && (
          <div className="details-row">
            <dt>Retrieval candidates</dt>
            <dd>{details.candidates}</dd>
          </div>
        )}
        <div className="details-row">
          <dt>Verification</dt>
          <dd data-testid="details-verification">{verificationSummary(result)}</dd>
        </div>
        {Object.keys(details.models).length > 0 && (
          <div className="details-row">
            <dt>Models</dt>
            <dd>
              {Object.entries(details.models)
                .map(([role, name]) => `${role}: ${name}`)
                .join(" · ")}
            </dd>
          </div>
        )}
      </dl>
      {result.conflicts.length > 0 && (
        <div className="conflicts">
          <h3 className="conflicts-title">Detected conflicts</h3>
          <ul>
            {result.conflicts.map((conflict) => (
              <li key={conflict.conflict_id}>
                {conflict.description} (evidence{" "}
                {conflict.citation_ids.map((id) => `[${id}]`).join(", ")})
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
