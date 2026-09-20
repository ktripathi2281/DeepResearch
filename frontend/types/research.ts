/**
 * Public research API contract (Milestone 18).
 *
 * These types mirror the backend's safe response models in
 * `src/deepresearch/research_api.py`. The frontend never invents
 * fields: unknown keys from the server are ignored, never rendered.
 */

/** Answer outcome taxonomy — owned by the backend, never duplicated. */
export type AnswerStatus =
  | "no_evidence"
  | "insufficient_evidence"
  | "answered"
  | "conflicting_evidence";

/** Research job lifecycle — distinct from the answer status above. */
export type JobStatus = "running" | "completed" | "failed";

export interface ErrorInfo {
  message: string;
  type?: string | null;
  request_id?: string | null;
}

export interface StageInfo {
  name: string;
  duration_ms: number;
  success: boolean;
}

export interface CitationInfo {
  citation_id: number;
  document_title: string | null;
  document_source: string;
  document_type: string;
  page: number | null;
  section: string | null;
  retrieval_method: string;
  retrieval_score: number;
  retrieval_rank: number;
}

export interface EvidenceInfo {
  citation_id: number;
  chunk_id: string;
  document_id: string;
  document_title: string | null;
  document_source: string;
  document_type: string;
  page: number | null;
  section: string | null;
  /** Source material: rendered as inert text, never executed. */
  text: string;
  retrieval_method: string;
  retrieval_score: number;
  retrieval_rank: number;
}

export interface ConflictInfo {
  conflict_id: string;
  conflict_type: string;
  description: string;
  citation_ids: number[];
}

export interface VerificationClaimInfo {
  claim_id: number;
  citation_id: number | null;
  claim_text: string;
  status: string;
  explanation: string;
}

export interface VerificationSummaryInfo {
  counts: Record<string, number>;
  claims: VerificationClaimInfo[];
}

export interface ResearchDetailsInfo {
  request_id: string;
  elapsed_ms: number;
  stages: StageInfo[];
  candidates: number | null;
  evidence_count: number;
  citation_count: number;
  invalid_citation_count: number;
  models: Record<string, string>;
}

export interface ResearchResult {
  request_id: string;
  status: AnswerStatus;
  answer: string;
  citations: CitationInfo[];
  evidence: EvidenceInfo[];
  conflicts: ConflictInfo[];
  verification: VerificationSummaryInfo | null;
  research_details: ResearchDetailsInfo;
}

export interface ResearchJobSnapshot {
  request_id: string;
  job_status: JobStatus;
  stages: StageInfo[];
  result: ResearchResult | null;
  error: ErrorInfo | null;
}

/** Backend enforces the same limit (`MAX_QUESTION_CHARS`); this is display-only. */
export const MAX_QUESTION_CHARS = 4000;
