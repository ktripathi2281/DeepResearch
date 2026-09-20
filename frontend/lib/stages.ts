/**
 * User-facing labels for real backend stage names.
 *
 * The names on the left are the exact `traced_stage(...)` identifiers
 * recorded by the backend pipeline; the labels on the right are what
 * the progress panel shows. Unknown stage names are rendered as-is so
 * the UI never hides real backend behavior — and never invents
 * fake progress (no percentages anywhere).
 */
export const STAGE_LABELS: Record<string, string> = {
  embedding: "Preparing search",
  vector_retrieval: "Searching evidence (semantic)",
  bm25_retrieval: "Searching evidence (keyword)",
  hybrid_fusion: "Combining search results",
  reranking: "Reranking evidence",
  conflict_detection: "Checking evidence for conflicts",
  citation_extraction: "Extracting citations",
  generation: "Generating answer",
  citation_verification: "Verifying citations",
};

/** Suggested pipeline order for display; unseen stages append at the end. */
const STAGE_ORDER = [
  "embedding",
  "vector_retrieval",
  "bm25_retrieval",
  "hybrid_fusion",
  "reranking",
  "conflict_detection",
  "citation_extraction",
  "generation",
  "citation_verification",
];

export function stageLabel(name: string): string {
  return STAGE_LABELS[name] ?? name;
}

export function orderStageNames(names: string[]): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const name of STAGE_ORDER) {
    if (names.includes(name)) {
      ordered.push(name);
      seen.add(name);
    }
  }
  for (const name of names) {
    if (!seen.has(name)) ordered.push(name);
  }
  return ordered;
}
