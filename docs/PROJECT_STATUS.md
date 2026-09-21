# DeepResearch — Project Status (M21, final engineering milestone)

Factual summary. No ratings, no superlatives — counts and names only.

## Milestones completed

M1 foundation · M2 schema + repository · M3 ingestion · M4 embeddings ·
M5 vector retrieval · M6 BM25 · M7 hybrid fusion · M8 reranking ·
M9 Ollama provider · M10 grounded generation · M11 citations ·
M12 verification · M13 answer statuses + conflicts · M14 bounded agent ·
M15 observability · M16 evaluation framework · M17 adversarial security ·
M18 research UI + API · M19 provider abstraction + cloud adapters ·
M20 production polish · M21 final validation.

## Test inventory

- Backend: 429 tests, all passing (`pytest`; includes opt-in live-model
  tests, which run when Ollama + `qwen3:4b` are present and skip
  otherwise), ~4 minutes with live models.
- Frontend: 46 tests across 7 files (`vitest run`); `tsc --noEmit`
  clean; `next build` succeeds.
- Lint: `ruff check` clean; `ruff format --check` clean.
- Live-validated: backend `/health` + `/ready`, end-to-end
  research scenarios, UI serving — see `docs/DEMO.md`.

## Data and models

- Evaluation dataset: `eval-dev-v1` (8 cases, 7 documents, 8 categories).
- Supported providers: `ollama`, `openai_compatible`, `gemini`
  (behind `LLMProvider`; single factory; cloud adapters mocked-test only).
- Default model: Ollama `qwen3:4b` (Q4_K_M), `http://localhost:11434`.
- Vector database: PostgreSQL + pgvector (`pgvector/pgvector:pg16`).
- Embeddings: `BAAI/bge-small-en-v1.5`, 384-d, L2-normalized.
- Reranker: `BAAI/bge-reranker-base` cross-encoder (top 20 → top 5).

## Major architectural components

FastAPI app (health/readiness/research API) → in-memory
ResearchService (bounded retention, graceful shutdown) → hybrid
retrieval (pgvector + BM25 + RRF) → cross-encoder rerank →
grounded generation → citation extraction → citation verification →
answer statuses and conflicts → Next.js UI (question, stages,
answer, citations, evidence, details). Cross-cutting: request
tracing, versioned evaluation, adversarial security suite, provider
factory, structured JSON logging. Full map: `docs/ARCHITECTURE.md`
§19 (Mermaid).

## Known limitations (pointers, not repetitions)

`README.md` → Limitations; `docs/SECURITY.md` → Known limitations;
per-ADR consequences sections.
