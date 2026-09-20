/**
 * Deterministic API fixtures for frontend tests. No Ollama, no
 * Postgres, no network — plain objects shaped like the backend's
 * safe response models.
 */
import type {
  AnswerStatus,
  EvidenceInfo,
  ResearchDetailsInfo,
  ResearchJobSnapshot,
  ResearchResult,
} from "../types/research";

export function evidenceItem(overrides?: Partial<EvidenceInfo>): EvidenceInfo {
  return {
    citation_id: 1,
    chunk_id: "chunk-1",
    document_id: "doc-1",
    document_title: "Source Title",
    document_source: "doc.md",
    document_type: "markdown",
    page: 3,
    section: "Findings",
    text: "Alpha note.",
    retrieval_method: "reranked",
    retrieval_score: 0.9,
    retrieval_rank: 1,
    ...overrides,
  };
}

export function details(overrides?: Partial<ResearchDetailsInfo>): ResearchDetailsInfo {
  return {
    request_id: "req-123",
    elapsed_ms: 1500,
    stages: [
      { name: "embedding", duration_ms: 42, success: true },
      { name: "generation", duration_ms: 900, success: true },
    ],
    candidates: 20,
    evidence_count: 1,
    citation_count: 1,
    invalid_citation_count: 0,
    models: { llm: "qwen3:4b" },
    ...overrides,
  };
}

export function researchResult(overrides?: Partial<ResearchResult>): ResearchResult {
  const evidence = [evidenceItem()];
  return {
    request_id: "req-123",
    status: "answered",
    answer: "Hybrid retrieval combines vector and lexical search [1].",
    citations: [
      {
        citation_id: 1,
        document_title: "Source Title",
        document_source: "doc.md",
        document_type: "markdown",
        page: 3,
        section: "Findings",
        retrieval_method: "reranked",
        retrieval_score: 0.9,
        retrieval_rank: 1,
      },
    ],
    evidence,
    conflicts: [],
    verification: {
      counts: { supported: 1, unsupported: 0, insufficient_evidence: 0 },
      claims: [
        {
          claim_id: 1,
          citation_id: 1,
          claim_text: "Hybrid retrieval combines vector and lexical search [1].",
          status: "supported",
          explanation: "Stated in the evidence.",
        },
      ],
    },
    research_details: details(),
    ...overrides,
  };
}

export function snapshot(
  overrides?: Partial<ResearchJobSnapshot> & { status?: AnswerStatus },
): ResearchJobSnapshot {
  const { status, ...rest } = overrides ?? {};
  const requestId = rest.request_id ?? "req-123";
  // A real backend always agrees with itself: propagate the job ID
  // into the nested result and research details.
  const result = researchResult(status ? { status } : undefined);
  result.request_id = requestId;
  result.research_details.request_id = requestId;
  return {
    request_id: requestId,
    job_status: "completed",
    stages: result.research_details.stages,
    result,
    error: null,
    ...rest,
  };
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
