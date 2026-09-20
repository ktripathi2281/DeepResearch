# ADR-008: Cross-encoder reranking (bge-reranker-base)

Date: 2026-09-20 | Status: Accepted | Milestone: M8

## Context

M7 hybrid retrieval returns fused candidates, but RRF only reflects
rank positions — it never reads query and document together. M8 must
add a precision layer before generation (M10+), staying local-first
on a Lenovo LOQ (16 GB RAM, 6 GB VRAM).

## Decision

- **Model: `BAAI/bge-reranker-base` (~278M params) via
  sentence-transformers `CrossEncoder`.** No new ML framework; the M4
  dependency already covers it. No larger reranker, no cloud.
- **Bi-encoder vs cross-encoder:** bi-encoders (M4/M5) embed query
  and chunks independently for fast full-corpus search; the
  cross-encoder jointly encodes each `(query, chunk)` pair, which is
  more precise but too slow for whole-corpus scoring — hence it runs
  *after* retrieval on a small candidate set (default 20), not
  instead of it.
- **Scores are raw logits** — never normalized, never treated as
  probabilities. Sorted descending, ties by chunk ID ascending;
  results reuse M5 `RetrievalResult` with `method="reranked"`,
  rescored and re-ranked from 1.
- **Candidate/final split:** `candidate_top_k=20` scored at most,
  best `top_k=5` returned (`Settings.reranker_candidate_top_k`,
  `retrieval_top_k`). Fewer available → rerank all; empty → `[]`
  without touching the model.
- **Device: same `auto`/CPU/CUDA contract as M4** (shared
  `resolve_device`; explicit bad-CUDA raises `RerankerLoadError`).
  CPU-first initially: reranking 20 short pairs on CPU takes seconds
  and avoids GPU contention with the future Ollama generator.
  Batch 16 (`Settings.reranker_batch_size`), one model load per
  provider instance, lazy so imports stay light and offline-safe.
- **Version recorded from config (`"1"`)** — the runtime exposes no
  checkpoint revision, and none is invented.
- **Failures propagate** (`RerankerError` on load/inference/count
  mismatch); no silent fallback to hybrid scores.
- **Independence:** `rerank_results` accepts any `RetrievalResult`
  list (vector, BM25, or hybrid), so M16+ can compare all four
  configurations.

## Alternatives considered

- Larger cross-encoders (bge-reranker-large/v2): rejected — GPU/RAM
  budget reserved for generation; base is the documented stack pick.
- Cohere/API rerankers: rejected — cloud, paid, against local-first.
- Score normalization (sigmoid/min-max): rejected — adds
  uninterpretable scaling; downstream only needs ordering.

## Consequences

- One-time ~1.1 GB model download to the HF cache on first use.
- M10+ generation consumes reranked evidence; M16 measures
  hybrid vs hybrid+reranker (quality and latency) rather than
  assuming the win.
