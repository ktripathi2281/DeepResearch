# ADR-004: Local embeddings (bge-small-en-v1.5)

Date: 2026-09-20 | Status: Accepted | Milestone: M4

## Context

M4 must turn M3 chunks into 384-d vectors in the existing
`chunks.embedding VECTOR(384)` column (ARCHITECTURE §4
EmbeddingProvider, FR-3), staying local-first on a Lenovo LOQ
(16 GB RAM, 6 GB VRAM). No schema redesign, no retrieval, no cloud.

## Decision

- **Provider: `sentence-transformers` running `BAAI/bge-small-en-v1.5`
  (~33M params, 384 dims).** The standard maintained library for this
  model class; torch arrives as its dependency (CUDA build by default
  for the LOQ's NVIDIA GPU; a CPU-only torch install works identically
  through the same device-selection code and was used for verification
  here to avoid a 2 GB download).
- **Abstraction: `EmbeddingProvider` Protocol** (`model_name`,
  `model_version`, `dimension`, `device`, `embed_texts`) in
  `src/deepresearch/embeddings.py`. Domain/service code depends only on
  the Protocol; HF/torch imports live inside `LocalEmbeddingProvider`,
  lazily loaded, so the API and unit tests stay import-light and
  offline-safe.
- **Normalization: L2-normalized (`normalize_embeddings=True`).**
  Cosine similarity then equals dot product, which is exactly what M5
  pgvector search will rely on. Documented here so M5 has a contract.
- **Batching: `embed_texts` takes lists; default batch 32**
  (`Settings.embedding_batch_size`), conservative for 6 GB VRAM and
  trivial for bge-small. One model load per provider instance, never
  per chunk.
- **Device: `"auto"` → CUDA when `torch.cuda.is_available()` else CPU;**
  explicit `"cuda"` without a GPU raises `ModelLoadError` (loud, never
  silent); `"cpu"` always works — correctness never requires a GPU.
  The resolved device is exposed and logged.
- **Versioning: `embedding_model` = configured model name,
  `embedding_version` = configured string (default `"1"`).** Bumping
  either marks existing vectors stale. Reruns skip current vectors
  (zero model calls); stale vectors are reported and left untouched
  unless `force=True` re-embeds them explicitly — vectors from mixed
  models are never silently blended.
- **Validation: every batch is length-checked against 384 before any
  write;** mismatch raises `DimensionMismatchError` with nothing
  persisted (no truncation/padding, ever).
- **Transactions: one commit per batch.** A failed batch rolls back
  while earlier batches stay committed; the error surfaces, failed
  chunks are never marked embedded.
- **Logging: model/device/batch/processed/elapsed via the existing
  JSON logger.** Chunk text is never logged.

## Alternatives considered

- `transformers` + manual pooling: more control, but re-implements
  what sentence-transformers already standardizes (pooling,
  normalization, batching).
- ONNX/quantized runtimes: faster on CPU, but adds conversion
  infrastructure for a 33M-parameter model that already runs
  comfortably.
- Separate `embeddings` table: rejected in ADR-001; M4 writes the
  existing nullable columns, no DDL.

## Consequences

- New runtime dep: `sentence-transformers>=3.0` (local inference only;
  no API key, no cloud). One-time ~130 MB model download to the HF
  cache on first use.
- `Settings` gains `embedding_model/_version/_batch_size/_device/
  _normalize`. M5 reads `chunks.embedding` assuming L2-normalized
  384-d vectors with matching model/version metadata.
- Tests: deterministic fake provider for all logic/PG tests
  (13 unit + 2 PG); real-model verification isolated in
  `test_embedding_model_real.py` (skips offline, fails loudly on
  misbehavior).
