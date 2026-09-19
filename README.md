# DeepResearch — Milestones 1–2: Foundation + Database Schema

Local-first, evidence-based research assistant. M1 built the development
foundation (Python project, FastAPI skeleton, PostgreSQL + pgvector via
Docker Compose, health/readiness, env config, logging, pytest). M2 adds
the persistence foundation: `documents` + `chunks` tables, pgvector
`VECTOR(384)` as schema capability (NULL until M4), repository layer.

No ingestion, retrieval, BM25, reranking, generation, agent, eval, or
frontend yet.

## Repository layout (root = `C:\Users\tripa\Projects\DeepResearch`)

```text
.
├── apps/               # reserved (future web/API wrappers, M19)
├── docs/               # PRD, architecture, evaluation, prompts
├── evals/              # reserved (datasets/runners, M16+)
├── infra/              # migrations/001_initial.sql + deploy extras
├── src/deepresearch/   # config, logging, db (engine+init_db), models, repository, main
├── tests/              # health/config/db + test_models (sqlite) + test_postgres_schema (PG)
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
└── README.md
```

No nested `deepresearch/` repo directory: `src/`, `tests/`, `evals/`,
`apps/`, `infra/`, `docs/` live directly under the root.

## Prerequisites

- Python 3.11+ (dev machine uses 3.12 slim in Docker; 3.14 works locally)
- Docker + Docker Compose plugin (for `postgres` + `api` services)
- No paid API keys required

## Quickstart (local, without Docker)

```powershell
# 1. Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install (editable + dev)
pip install -U pip
pip install -e ".[dev]"

# 3. Configure
Copy-Item .env.example .env

# 4. Run API
uvicorn deepresearch.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Verify
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready   # 503 if Postgres is down (expected)
```

## Quickstart (Docker — full local stack)

```powershell
Copy-Item .env.example .env
docker compose up --build
# API:     http://localhost:8000/health
# Postgres: localhost:5432 (user/pass/db: deepresearch)
docker compose down
```

Postgres image is `pgvector/pgvector:pg16` so the vector extension is
available for Milestone 2 without changing images.

## Database schema (M2)

Tables: `documents` (id UUID, title, source, content_hash UNIQUE,
document_type, metadata JSON, created_at) and `chunks` (id UUID,
document_id FK CASCADE, text, chunk_index, section/page nullable,
metadata JSON, embedding VECTOR(384) NULL, embedding_model/version NULL,
created_at; UNIQUE(document_id, chunk_index)).

`content_hash` is the ingestion idempotency key (M3). `embedding` stays
NULL until M4 — see `docs/adr/001-postgres-pgvector-schema.md`.
Migration strategy (`create_all` + versioned SQL, Alembic deferred):
`docs/adr/002-migration-strategy.md`.

```powershell
# SQLite-safe default not needed — schema targets Postgres:
docker compose up -d postgres
python -c "from deepresearch.config import get_settings; from deepresearch.db import get_engine, init_db; init_db(get_engine(get_settings()))"
# Canonical DDL for review: infra/migrations/001_initial.sql
```

## Tests

```powershell
pip install -e ".[dev]"
pytest -v
# Live-DB check only (skipped if Postgres is unreachable):
# $env:TEST_DATABASE_URL="postgresql+psycopg://deepresearch:deepresearch@localhost:5432/deepresearch"
# pytest -v
```

## Lint / format (ruff)

```powershell
ruff check src tests
ruff format --check src tests
ruff format src tests   # apply fixes
```

## Endpoints

- `GET /health` → `{"status":"ok",...}` (liveness, no DB)
- `GET /ready` → `200 {"status":"ready"}` or `503 {"status":"not_ready"}` (DB check)

## Decisions locked before Milestone 1

1. Root is `C:\Users\tripa\Projects\DeepResearch`; no nested repo dir.
2. BM25 = in-Python behind a retrieval interface (impl in M6; ADR then).
3. Chunk size in tokens; defaults 800 / overlap 120 (M3, experiments later).
4. Structured output = app-side Pydantic parse→validate→retry/reject; no hard Ollama dependency.
5. Web search disabled in v1; agent tools only `search_documents/get_chunk/get_document`.
6. Fixed `PRODUCT_REQUIREMENTS.md` numbering: `7A→8`, `8→9`, `9→10` (content unchanged).
7. Hardware/models frozen: LOQ 16GB/6GB, `qwen3:4b Q4_K_M`, `bge-small-en-v1.5`, `bge-reranker-base`, optional `gemma3:4b`; no larger models or paid APIs without approval.

## What's next (not in M2)

Milestone 3: document ingestion (parsing, normalization, configurable
token chunking with defaults 800/120, metadata preservation, hashing,
idempotency) — no embeddings/retrieval yet.
