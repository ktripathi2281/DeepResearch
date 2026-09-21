# DeepResearch — Testing Guide

## Backend tests

```powershell
pip install -e ".[dev]"
pytest -v
```

Needs: nothing (fakes + SQLite by default). ~430 tests, ~1 minute.

## Frontend tests

```powershell
cd frontend
npm install
npm test        # vitest run (mocked fetch)
npm run typecheck
npm run build   # production build check
```

Needs: Node 24 + npm only. No Ollama, no Postgres. 46 tests.

## PostgreSQL integration tests

Run automatically inside `pytest -v` when PostgreSQL is reachable;
each file skips cleanly otherwise:

```powershell
docker compose up -d postgres
# $env:TEST_DATABASE_URL="postgresql+psycopg://deepresearch:deepresearch@localhost:5432/deepresearch"
pytest tests/test_retrieval_postgres.py tests/test_research_api_postgres.py -v
```

Needs: Docker + healthy `deepresearch-postgres-1`. All files
matching `*_postgres.py` live here.

## Security tests

```powershell
pytest tests/test_security.py tests/test_security_agent.py tests/test_security_postgres.py tests/test_providers_security.py -v
```

Needs: nothing for `test_security.py` / `test_security_agent.py` /
`test_providers_security.py` (fakes + mocks); Postgres for
`test_security_postgres.py` (skips without it). No Ollama, no cloud
credentials — sentinel keys stand in for real secrets and are
asserted absent from logs and errors.

## Evaluation tests

```powershell
pytest tests/test_evaluation.py tests/test_evaluation_postgres.py -v
```

Needs: nothing for the unit file; Postgres for the integration
file. Both use fakes. The shipped `eval-dev-v1` is a development
fixture, not a benchmark.

## Reliability tests

```powershell
pytest tests/test_reliability.py -v
```

Needs: nothing (fakes + monkeypatched probes). Covers config
validation, health/readiness, job lifecycle, retention,
concurrency, error contract, shutdown, request IDs, and client
reuse.

## Live-model tests (opt-in, skipped by default)

Files ending `_ollama_live.py` or `_model_real.py` run only with the
real thing present and skip otherwise:

- Ollama daemon + `qwen3:4b`: `test_llm_ollama_live.py`,
  `test_generation_ollama_live.py`, `test_verifier_ollama_live.py`,
  `test_agent_ollama_live.py`
- HuggingFace cache with the pinned models:
  `test_embedding_model_real.py`, `test_reranker_model_real.py`

Live-model note: `qwen3:4b` is a thinking model. Small
`max_tokens` caps can be consumed entirely by thinking, yielding
empty text; the live smoke tests use generous or no caps for this
reason, while cap plumbing stays covered by unit tests.

## Full verification

```powershell
pip install -e ".[dev]"
pytest -v                                  # backend + PG integration
ruff check src tests
ruff format --check src tests
cd frontend
npm test; npm run typecheck; npm run build
cd ..
docker compose config                      # compose file validation
```

Requirement matrix:

| Suite | Docker/PG | Ollama | Cloud keys | Network |
|---|---|---|---|---|
| Backend unit | no (skips) | no (skips) | no | no |
| PG integration | yes | no | no | no |
| Frontend | no | no | no | no (mocked) |
| Security | partial (1 file) | no | no (sentinels) | no |
| Evaluation | partial (1 file) | no | no | no |
| Reliability | no | no | no | no |
| Live-model | no | yes | no | localhost only |
