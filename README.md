# DeepResearch — Milestone 1: Repository and Local Infrastructure

Local-first, evidence-based research assistant. Milestone 1 is the
development foundation only: Python project, FastAPI skeleton,
PostgreSQL + pgvector via Docker Compose, health/readiness endpoints,
env-based config, structured logging, pytest.

No RAG, ingestion, retrieval, generation, or frontend yet.

## Repository layout (root = `C:\Users\tripa\Projects\DeepResearch`)

```text
.
├── apps/               # reserved (future web/API wrappers, M19)
├── docs/               # PRD, architecture, evaluation, prompts
├── evals/              # reserved (datasets/runners, M16+)
├── infra/              # reserved (deploy extras)
├── src/deepresearch/   # config.py, logging.py, db.py, main.py
├── tests/              # test_health.py, test_config.py, test_db.py
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

## What's next (not in M1)

Milestone 2: DB schema + migrations + pgvector column + repository layer.
