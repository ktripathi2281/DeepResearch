"use client";

import type { EvidenceInfo } from "../types/research";

interface EvidencePanelProps {
  evidence: EvidenceInfo[];
  highlightedCitation: number | null;
}

function metaLine(item: EvidenceInfo): string {
  const bits: string[] = [`Method: ${item.retrieval_method}`, `Rank: ${item.retrieval_rank}`];
  if (item.page !== null && item.page !== undefined) bits.push(`Page: ${item.page}`);
  if (item.section) bits.push(`Section: ${item.section}`);
  return bits.join(" · ");
}

/**
 * Source/evidence panel. Every chunk is quoted source material rendered
 * as inert text (React escapes by default; no `dangerouslySetInnerHTML`
 * anywhere). Internal-only fields (embeddings, DB internals) are never
 * part of the API shape, so they cannot leak here.
 */
export function EvidencePanel({ evidence, highlightedCitation }: EvidencePanelProps) {
  return (
    <section aria-label="Sources and evidence" className="panel">
      <h2 className="panel-title">Sources / Evidence</h2>
      {evidence.length === 0 && <p className="hint">No evidence items.</p>}
      <ol className="evidence-list">
        {evidence.map((item) => (
          <li
            key={item.citation_id}
            id={`evidence-${item.citation_id}`}
            data-testid={`evidence-${item.citation_id}`}
            className={
              highlightedCitation === item.citation_id
                ? "evidence-item evidence-highlight"
                : "evidence-item"
            }
            tabIndex={-1}
          >
            <h3 className="evidence-title">
              [{item.citation_id}] {item.document_title ?? item.document_source}
            </h3>
            <p className="evidence-meta">
              {item.document_source} · {item.document_type}
            </p>
            <p className="evidence-meta">{metaLine(item)}</p>
            <p className="evidence-meta">
              Relevance score: {item.retrieval_score.toFixed(3)}
            </p>
            <blockquote className="evidence-text">
              <span className="evidence-label">Source material:</span> {item.text}
            </blockquote>
          </li>
        ))}
      </ol>
    </section>
  );
}
