# ADR-005: Vector retrieval (pgvector cosine, exact search)

Date: 2026-09-20 | Status: Accepted | Milestone: M5

## Context

M5 needs query → chunks via PostgreSQL + pgvector (FR-4, ARCHITECTURE
§5 RetrievalResult, EVALUATION §3 baseline), consistent with M4's
L2-normalized 384-d bge-small-en-v1.5 vectors. Decisions needed:
similarity definition, top-K policy, model consistency, filters,
indexing — without touching BM25/hybrid (M6/M7).

## Decision

- **Similarity = `1 − cosine_distance`**, computed in the database with
  pgvector's native `<=>` operator (`Chunk.embedding.cosine_distance`).
  Identical vectors score 1.0, orthogonal 0.0, opposite −1.0; higher
  always means more similar. The conversion is explicit because
  pgvector returns a distance and the architecture demands an
  intuitive score.
- **One query embedding per call** through the `EmbeddingProvider`
  Protocol (never sentence-transformers directly); its dimension is
  re-validated (384) before searching.
- **Results are `RetrievalResult` frozen dataclasses**: chunk/document
  ids, chunk_index, text, score, 1-based rank, title/type/source,
  page/section/metadata, `retrieval_method="vector"` for later fusion.
  Ordered score-descending; **ties break on `Chunk.id` ascending** so
  ranking is deterministic run to run.
- **Top-K: default 5** (matches the Recall@5 evaluation baseline),
  **cap 100** (`Settings.retrieval_top_k/max_top_k`). `top_k < 1`,
  non-integers, or over-cap requests raise `RetrievalError` — fail
  clearly, never silently clamp, so experiments cannot mistake a
  clamped K for a requested one.
- **Model/version filter always on**: only chunks whose
  `embedding_model/version` match the querying provider are searched.
  Mixed-model vectors are never compared; stale vectors are M4's
  `force` re-embed path, not retrieval's problem.
- **Unembedded chunks excluded** by `embedding IS NOT NULL`; empty
  corpus/filter match returns `[]`, never an error.
- **Filters limited to `document_id` / `document_type`** — columns that
  already exist. No filtering language, no query planning (M7+ concern
  if ever).
- **No approximate index at M5: exact search.** The corpus is small
  (tens of chunks); an HNSW/IVFFlat index would add recall/latency
  tradeoffs with zero measurable benefit, and exact search keeps the
  EVALUATION §3 baseline honest. Correctness does not depend on index
  choice — add HNSW with a latency experiment when the corpus grows.
- **Single joined query** (chunks + documents + score); read-only
  session, no commits, no N+1. Persistence lives in
  `repository.search_chunks_by_vector`; orchestration in
  `retrieval.retrieve`.

## Alternatives considered

- Inner-product operator (`<#>`): equivalent for normalized vectors,
  but cosine distance reads directly as "1 − similarity" and tolerates
  unnormalized legacy rows.
- L2 distance (`<->`): wrong geometry for a cosine contract.
- Silent clamping of over-limit K: rejected — hides experiment error.
- HNSW now: rejected per above; revisit with measurements.

## Consequences

- M6 builds BM25 separately; M7 fuses `RetrievalResult.score` values
  (both define higher = more similar).
- EVALUATION retrieval baselines (Recall@K, MRR) run against this
  exact-search implementation first.
