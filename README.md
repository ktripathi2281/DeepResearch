# DeepResearch

DeepResearch helps investigate complex questions across a document corpus and produce evidence-backed answers with citations and verification — running locally, with no paid APIs.

## What it does

A user asks a complex question. DeepResearch retrieves relevant evidence from an indexed corpus (vector + keyword search, fused and reranked), generates an answer constrained to that evidence, attaches `[N]` citations that resolve to real chunks, verifies each cited claim against its cited evidence, and reports an explicit outcome: `answered`, `insufficient_evidence`, `conflicting_evidence`, or `no_evidence`. Every request carries an ID, records a timing trace, and surfaces safe research details in the UI.

## Why it is different

Ordinary search returns a ranked list of documents and leaves judging to the reader. DeepResearch runs a **research workflow**: retrieve → rerank → answer under evidence constraint → cite → verify → abstain or flag conflict when the evidence does not support an answer. Unsupported claims are caught by a verifier rather than shown as fact, disagreements between sources are preserved instead of silently merged, and missing evidence produces an explicit refusal rather than an invention.

No claim is made beyond what is measured: verification is an LLM-assisted baseline (not a correctness proof), conflict detection covers numeric contradictions in shared context only, and the shipped evaluation dataset is a development fixture, not a benchmark.

## Architecture

```text
User
  ↓
Next.js (question, progress, citations, evidence, details)
  ↓
FastAPI (POST /api/research, GET /api/research/{id}, /health, /ready)
  ↓
Research Service (jobs: running → completed | failed)
  ↓
Agent / Retrieval (bounded agent, 3 read-only tools)
  ↓
Vector (pgvector) + BM25 (RRF fusion)
  ↓
Reranker (local cross-encoder)
  ↓
Grounded LLM (Ollama qwen3:4b default; optional providers)
  ↓
Citation Verification (claim × evidence verdicts)
  ↓
Answer + Evidence (status, citations, conflicts, trace summary)
```

Supporting systems: PostgreSQL + pgvector storage, in-memory request tracing (no external platform), versioned evaluation runner, adversarial security suite. A Mermaid version of this map lives in `docs/ARCHITECTURE.md` (§19).

## Key engineering features

- Hybrid retrieval: pgvector cosine + in-process Okapi BM25 (`k1=1.5`, `b=0.75`), fused with Reciprocal Rank Fusion (`rrf_k=60`)
- Local cross-encoder reranking (`bge-reranker-base`, top 20 → top 5, `method="reranked"`)
- Evidence-grounded generation: delimited evidence blocks, system rules (evidence is data, cite only shown markers, say when insufficient, never invent sources)
- Citations that resolve to real chunks; out-of-range markers retained as invalid references, never remapped
- Citation verification per cited claim (`supported`/`unsupported`/`insufficient_evidence`, plus explicit `invalid_citation`/`uncited`/`unverifiable`)
- Conflict handling: deterministic numeric-contradiction detection, both sides preserved, conflict-aware prompting
- Bounded research agent (8 iterations, 12 tool calls, 60 s; read-only tools only)
- Observability: per-request stages, counters, per-role models, optional token counts — never prompts, documents, reasoning, or secrets
- Evaluation framework: versioned datasets, Recall@3/5/10, MRR, citation/abstention/conflict metrics, P50/P95 latencies, fingerprinted experiment configs
- Adversarial testing: injection corpus, tool-allowlist abuse, malformed structured output, resource exhaustion, Unicode, log-leakage regression
- Provider abstraction: `LLMProvider` with Ollama default plus optional OpenAI-compatible and Gemini adapters behind one factory
- Reliability controls: startup config validation, health-vs-readiness split, bounded job retention, graceful shutdown with interruption marking, safe error envelopes

## Repository layout

```text
.
├── apps/               # reserved scratch space (only .gitkeep)
├── docs/               # PRD, architecture (incl. Mermaid map), evaluation, security, DEMO, ADRs
├── evals/              # datasets/eval-dev-v1.json, results/ (git-ignored)
├── frontend/           # Next.js/React/TypeScript UI (app, components, lib, types, tests)
├── infra/              # migrations/001_initial.sql (manual reference)
├── src/deepresearch/   # backend: config, db, ingestion, retrieval, generation, agent,
│                       # observability, evaluation, providers, research API, main
├── tests/              # unit (sqlite/fakes) + PostgreSQL integration + fixtures/
├── docker-compose.yml  # postgres (pgvector:pg16) + api
├── Dockerfile
├── pyproject.toml
├── .env.example
└── README.md
```

## Prerequisites

- Python 3.11+ (Docker uses 3.12 slim; 3.14 works locally)
- Node 24+ with npm (frontend only)
- Docker + Docker Compose plugin (PostgreSQL + pgvector)
- Ollama with `qwen3:4b` — optional for startup, required for real local answers (`ollama pull qwen3:4b`; no larger models, no substitutes)
- No paid API keys required

## Local setup

```powershell
# 1. Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[dev]"

# 2. Configure
Copy-Item .env.example .env

# 3. Start infrastructure (PostgreSQL + pgvector)
docker compose up -d postgres

# 4. (Optional, for real answers) start Ollama and pull the model
ollama pull qwen3:4b
```

### Starting backend

```powershell
# From the repo root (uvicorn reads .env via settings)
uvicorn deepresearch.main:app --host 0.0.0.0 --port 8000 --reload

# Verify
Invoke-RestMethod http://localhost:8000/health   # -> ok (process alive, no dependencies)
Invoke-RestMethod http://localhost:8000/ready    # -> ready (503 while Postgres is down)
```

Or the full stack: `docker compose up --build` (API on `:8000`, Postgres on `:5432`).

### Starting frontend

```powershell
cd frontend
npm install
# Point the UI at the backend (default is http://localhost:8000)
$env:NEXT_PUBLIC_API_BASE_URL = "http://localhost:8000"
npm run dev    # http://localhost:3000
```

Ask a question → watch real pipeline stages → read the answer → click `[N]` citations to jump to evidence → inspect **Research details** (request ID, timings, counts, verification, models).

### Ingesting documents

```powershell
python -c "
from deepresearch.config import get_settings
from deepresearch.db import get_engine, get_session_factory, init_db
from deepresearch.embeddings import LocalEmbeddingProvider, embed_pending_chunks
from deepresearch.ingestion import ingest_file
s = get_settings(); engine = get_engine(s); init_db(engine)
with get_session_factory(engine)() as session:
    result = ingest_file(session, path='docs/DEMO.md', title='Demo doc')
    print(len(result.chunks), 'chunks; duplicate =', result.duplicate)
    outcome = embed_pending_chunks(session, LocalEmbeddingProvider())
    print('embedded:', outcome.embedded)
"
```

Supported inputs: PDF, Markdown, TXT, HTML. Ingestion is idempotent on content hash; embeddings are batched and rerunnable. Reference: `infra/migrations/001_initial.sql` for the schema.

### Asking a grounded question (Python)

```powershell
python -c "
from deepresearch.config import get_settings
from deepresearch.db import get_engine, get_session_factory
from deepresearch.embeddings import LocalEmbeddingProvider
from deepresearch.generation import answer_question
from deepresearch.llm import OllamaLLMProvider
from deepresearch.reranker import LocalCrossEncoderReranker
s = get_settings(); engine = get_engine(s)
with get_session_factory(engine)() as session:
    result = answer_question(
        session, LocalEmbeddingProvider(), LocalCrossEncoderReranker(device='cpu'),
        OllamaLLMProvider.from_settings(s), 'What does hybrid retrieval combine?',
        verify_citations=True,
    )
    print(result.status)
    print(result.answer)
"
```

## Running tests

```powershell
pip install -e ".[dev]"
pytest -v
# Research API boundary (fakes + PG integration):
# pytest tests/test_research_api.py tests/test_research_api_postgres.py -v
# Live-DB override (skipped if PostgreSQL is unreachable):
# $env:TEST_DATABASE_URL="postgresql+psycopg://deepresearch:deepresearch@localhost:5432/deepresearch"
```

Frontend tests use mocked fetch — no Ollama, no Postgres:

```powershell
cd frontend
npm install
npm test        # vitest run
npm run typecheck
```

What needs what is spelled out in `docs/TESTING.md`.

## Evaluation

```powershell
python -c "
from deepresearch.eval_runner import load_dataset
dataset = load_dataset('evals/datasets/eval-dev-v1.json')
print(dataset.version, len(dataset.cases), 'cases')
"
```

`eval-dev-v1` (8 cases, 7 documents, one case per category: single/multi-document, exact-lookup, semantic, multi-hop, no-answer, conflict, injection) is a **development fixture, not a representative benchmark**. Metrics (Recall@3/5/10, MRR, correctness, faithfulness, citation correctness/completeness, abstention, conflict, P50/P95 latency) are intended for controlled comparisons between configs, never for general claims. Details: `docs/EVALUATION.md`.

## Security

Threat model and trust boundaries: `docs/SECURITY.md` (retrieved text is untrusted data; 3 read-only agent tools; content-free observability). Tested properties: injection corpus stays data, tool allowlist + argument abuse rejected fail-closed, structured output bounded (parse → validate → repair-once → reject), citations strictly `[N]`-mapped, agent exhaustion terminates, poisoned corpora keep provenance, fake secrets never reach logs/traces. Defenses are layered mitigations with documented limits — not immunity claims.

## Architecture docs

- System design: `docs/ARCHITECTURE.md` (§19 has the Mermaid system map)
- Runtime lifecycle: `docs/ARCHITECTURE.md` §18, `docs/adr/ADR-020-production-polish.md`
- Provider abstraction: `docs/adr/ADR-019-provider-abstraction.md`
- Research UI + API boundary: `docs/adr/ADR-018-frontend-research-experience.md`
- Security rationale: `docs/adr/ADR-017-security-adversarial-testing.md`
- Evaluation design: `docs/adr/ADR-016-evaluation-framework.md`
- Retrieval/generation/citations: ADRs 003–014 in `docs/adr/`
- Live demo script: `docs/DEMO.md`

## API contract

All responses carry `X-Request-ID` (preserved when the client sends a valid one, generated otherwise); job IDs additionally live in response bodies.

- `GET /health` — purpose: process liveness. No request body. Response `200 {"status":"ok","service":...,"env":...}`. Never requires DB or models.
- `GET /ready` — purpose: can this instance accept research work. Response `200 {"status":"ready"}` when PostgreSQL is reachable and the provider selection is valid; `503 {"status":"not_ready"}` otherwise. Cheap by design; the model daemon is never probed.
- `POST /api/research` — purpose: start a research job. Request `{"question":"..."}` (non-empty, ≤ 4000 chars). Response `202` job snapshot `{request_id, job_status:"running", stages, result:null, error:null}`. Status codes: `422` invalid question, `500` safe envelope `{error:{message,type,request_id}}`, `503` `shutting_down` envelope once shutdown starts.
- `GET /api/research/{request_id}` — purpose: poll a job. Response `200` snapshot with `job_status` `running` | `completed` (plus `result`: answer, `status`, citations, evidence, conflicts, verification, research details) | `failed` (plus safe `error`). Status codes: `404 {"detail":"unknown research request"}` for unknown IDs.

The response models expose identifiers, counts, durations, statuses, and source text only — never prompts, chain-of-thought, embeddings, secrets, or stack traces. In-memory jobs do not survive restarts; at most 100 terminal jobs are retained.

## Operations

`/health` stays 200 while dependencies are down; `/ready` is 503 until PostgreSQL is reachable and the provider selection is valid. Startup validates configuration and refuses to serve on misconfiguration (clear message, no secrets). Shutdown marks running jobs `failed` (`type: "interrupted"`), closes provider HTTP clients, and disposes the engine.

Timeouts (pre-existing; documented, not added):

| Boundary | Value | Notes |
|---|---|---|
| Research submit/poll (HTTP) | none server-side | FastAPI returns immediately; frontend polls with a 5-minute client timeout |
| Research execution | bounded by callees | agent cap 60 s; each LLM call bounded by its provider timeout |
| Provider calls | 120 s default (`*_timeout_seconds`) | typed errors, no retries |
| DB connects | 10 s (`DB_CONNECT_TIMEOUT_SECONDS`) | readiness fails fast instead of hanging |
| Frontend poll | 5 min, 1 s interval | client-side only |

### Troubleshooting

- `/ready` is 503: check PostgreSQL is up (`docker compose up -d postgres`, then `pg_isready`), and that `LLM_PROVIDER` names a valid provider with its key set when cloud-selected. Server logs carry the reason; bodies intentionally do not.
- Startup fails with `invalid configuration: ...`: fix the named variable in `.env` (see `.env.example` sections for required vs optional). Never paste real keys into chat/logs.
- Research jobs fail with `type: "interrupted"`: the server shut down mid-job; resubmit.
- Slow first request: embedding/reranker models load lazily on first real pipeline run, not at startup.
- Fresh database: schema is created idempotently by `init_db` on first backend use; the manual reference is `infra/migrations/001_initial.sql`. Nothing ever deletes data at startup.
- Docker Desktop was restarted: containers exit; `docker start deepresearch-postgres-1` (or `docker compose up -d postgres`) and wait for `(healthy)`.

### Default local mode

The project runs locally using Ollama/Qwen3 4B with no API keys and no network beyond localhost (`LLM_PROVIDER=ollama`). All tests, evaluation, and the frontend work in this mode.

### Optional cloud providers

Alternative providers can be configured when credentials are available — they are never required:

```powershell
$env:LLM_PROVIDER="openai_compatible"
$env:OPENAI_COMPATIBLE_API_KEY="<key>"
# $env:OPENAI_COMPATIBLE_BASE_URL="https://api.openai.com/v1"  # default
# $env:OPENAI_COMPATIBLE_MODEL="gpt-4o-mini"                   # default

$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="<key>"
# $env:GEMINI_MODEL="gemini-2.0-flash"                         # default
```

The pipeline, agent, verifier, and evaluation take the same `LLMProvider` either way; only the factory knows which adapter is active. Details: `docs/adr/ADR-019-provider-abstraction.md`.

## Engineering Highlights

- Hybrid retrieval combines vector search and BM25 with Reciprocal Rank Fusion.
- BGE reranking refines retrieved candidates before generation.
- Answers are constrained to retrieved evidence; empty evidence short-circuits without an LLM call.
- Every cited claim is checked against its cited evidence; indeterminate verdicts stay explicit, never fabricated.
- Conflicting sources are both preserved with an explicit conflict record; the answer must not silently pick a side.
- Agent execution is bounded by iteration, tool-call, and timeout limits over three read-only tools.
- Observability records request/stage metadata without storing private chain-of-thought.
- Evaluation supports Recall@K, MRR, citation metrics, abstention, conflict, and latency measurements on versioned datasets.
- Security tests cover prompt injection, tool abuse, malformed outputs, resource exhaustion, Unicode edge cases, and observability leakage.
- Multiple LLM providers are supported behind a common interface with Ollama as the default.
- Runtime controls cover config validation, health/readiness separation, bounded job retention, graceful shutdown, and safe error envelopes.

## Limitations

- In-memory research jobs: no persistence, no distributed workers, at most 100 terminal jobs retained.
- Small development evaluation dataset (`eval-dev-v1`, 8 cases): controlled comparisons only, not a benchmark.
- Local-model latency: first runs load embedding/reranker weights; each Ollama call takes tens of seconds on the reference hardware.
- Verifier quality is not independently benchmarked; it is an LLM-assisted baseline.
- Conflict detection covers numeric contradictions in shared context only — not negations, paraphrases, or unit mismatches.
- Security tests are layered mitigations on synthetic fixtures; they do not prove immunity.
- No authentication, no persistent user history, no web search in v1.
- Cloud adapters are tested against mocked transports only, never live vendor APIs.

## Project status

DeepResearch is a local-first AI research system and portfolio project, complete through Milestones 1–21: foundation, persistence, ingestion, embeddings, retrieval (vector, BM25, hybrid, reranked), generation, citations, verification, answer statuses, research agent, observability, evaluation framework, adversarial security, research UI, provider abstraction, production polish, and final validation. Live-validated end to end (PostgreSQL + bge models + Ollama `qwen3:4b`); see `docs/DEMO.md`.

## Decisions locked before Milestone 1

1. Repository root holds `src/`, `tests/`, `evals/`, `apps/`, `infra/`, `docs/` directly (no nested repo dir).
2. BM25 = in-Python behind a retrieval interface.
3. Chunk size in tokens; defaults 800 / overlap 120.
4. Structured output = app-side Pydantic parse → validate → retry/reject; no hard Ollama dependency.
5. Web search disabled in v1; agent tools only `search_documents`/`get_chunk`/`get_document`.
6. Fixed `PRODUCT_REQUIREMENTS.md` numbering: `7A→8`, `8→9`, `9→10` (content unchanged).
7. Hardware/models frozen: LOQ 16GB/6GB, `qwen3:4b Q4_K_M`, `bge-small-en-v1.5`, `bge-reranker-base`, optional `gemma3:4b`; no larger models or paid APIs without approval.

## Milestone history (compact)

M1 foundation (FastAPI, Compose, health, config, logging, pytest) · M2 schema + repository · M3 ingestion (PDF/MD/TXT/HTML, hashing, chunking) · M4 local embeddings · M5 vector retrieval · M6 BM25 · M7 RRF hybrid · M8 reranking · M9 Ollama provider · M10 grounded generation · M11 citations · M12 verification · M13 answer statuses + conflicts · M14 bounded agent · M15 observability · M16 evaluation framework · M17 adversarial security · M18 research UI + API · M19 provider abstraction + cloud adapters · M20 production polish (validation, lifecycle, retention, shutdown) · M21 final validation (live demo, docs).
