"use client";

import type { CitationInfo } from "../types/research";

interface AnswerViewProps {
  answer: string;
  citations: CitationInfo[];
  onCitationClick: (citationId: number) => void;
}

const MARKER_PATTERN = /\[(\d+)\]/g;

/**
 * Answer text with inline citations. Markers that the backend mapped
 * to evidence become buttons that jump to the evidence; unknown or
 * invalid markers stay plain text — never invented, never linked.
 */
export function AnswerView({ answer, citations, onCitationClick }: AnswerViewProps) {
  const known = new Set(citations.map((citation) => citation.citation_id));
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let key = 0;
  for (const match of answer.matchAll(MARKER_PATTERN)) {
    const start = match.index ?? 0;
    if (start > lastIndex) {
      parts.push(<span key={key++}>{answer.slice(lastIndex, start)}</span>);
    }
    const id = Number(match[1]);
    if (known.has(id)) {
      parts.push(
        <button
          key={key++}
          type="button"
          className="citation-button"
          aria-label={`Go to evidence ${id}`}
          onClick={() => onCitationClick(id)}
        >
          [{id}]
        </button>,
      );
    } else {
      parts.push(
        <span key={key++} title="No matching evidence for this marker">
          [{id}]
        </span>,
      );
    }
    lastIndex = start + match[0].length;
  }
  if (lastIndex < answer.length) {
    parts.push(<span key={key++}>{answer.slice(lastIndex)}</span>);
  }

  return (
    <section aria-label="Answer" className="panel">
      <h2 className="panel-title">Answer</h2>
      <p className="answer-text" data-testid="answer-text">
        {parts}
      </p>
      {citations.length === 0 && (
        <p className="hint">No citations — this answer is not backed by listed evidence.</p>
      )}
    </section>
  );
}
