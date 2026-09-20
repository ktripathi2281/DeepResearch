# ADR-007: Hybrid retrieval with Reciprocal Rank Fusion

Date: 2026-09-20 | Status: Accepted | Milestone: M7

## Context

M5 (cosine vector search) and M6 (Okapi BM25) fail differently:
vectors blur exact terms (names, codes, rare keywords) while BM25
misses paraphrase entirely. M7 must combine them into one ranked
candidate set for the future reranker (M8), without touching either
implementation.

## Decision

- **Orchestration-only `hybrid.py`**: M5 `retrieve` and M6
  `retrieve_bm25` run unmodified (one query embedding total, inside
  the M5 call); hybrid only fuses. Same `RetrievalResult` model with
  `method="hybrid"`, fused score, reassigned rank.
- **Reciprocal Rank Fusion, `rrf_k=60` (literature default,
  `Settings.hybrid_rrf_k`)**: a chunk at 1-based rank `r`
  contributes `1 / (rrf_k + r)` per list it appears in; the fused
  score is the sum. Rank-based fusion is chosen precisely because
  raw cosine similarities and raw BM25 scores live on unrelated
  scales — adding them directly would let one method's range
  dominate for no principled reason, and any normalization would be
  another tunable to justify. RRF needs no score calibration.
- **Candidate pools default to `2 * top_k`** (capped at `max_top_k`)
  so fusion sees beyond the final cut; explicit per-side K values
  are validated like M5/M6 (raise, never clamp).
- **Dedup by chunk ID** (union, each chunk once); single-side chunks
  keep their single contribution. Ties break on chunk ID ascending.
- **Filters (`document_id`/`document_type`) pass through** to both
  retrievers; a supplied BM25 index keeps M6 freshness semantics
  (`StaleIndexError` propagates). Retriever failures propagate —
  no silent degradation.
- **Component scores stay internal** (`FusedCandidate`); the public
  result carries only the fused score.

## Alternatives considered

- Weighted raw-score sum: rejected — requires normalizing two
  unrelated scales, i.e. more parameters with no measurement yet.
- Weighted RRF / RRF with per-system weights: deferred to evaluated
  experiments, not the baseline.

## Consequences

- M8 reranks hybrid output; M16+ measures vector vs BM25 vs hybrid
  on the eval dataset. **No claim is made here that hybrid is
  better** — RRF is the baseline to beat, and the experiment may
  show otherwise on some query classes.
