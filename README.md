# DeepResearch — Milestones 1–17: + Security & Adversarial Testing

Local-first, evidence-based research assistant. M1 built the development
foundation (Python project, FastAPI skeleton, PostgreSQL + pgvector via
Docker Compose, health/readiness, env config, logging, pytest). M2 adds
the persistence foundation: `documents` + `chunks` tables, pgvector
`VECTOR(384)`, repository layer. M3 adds deterministic ingestion:
PDF/Markdown/TXT/HTML parsing, sha256 content hashing, idempotent
persist, token chunking (800/120 defaults). M4 adds local embeddings:
`BAAI/bge-small-en-v1.5` (384-d, L2-normalized) via `sentence-transformers`,
batched `embed_pending_chunks()` with idempotent reruns. M5 adds cosine
vector retrieval: query → provider → pgvector `<=>` → ranked
`RetrievalResult`s (score = `1 − distance`, top-K default 5 / max 100).
M6 adds in-process Okapi BM25 over `Chunk.text` (`k1=1.5`, `b=0.75`):
same `RetrievalResult` shape with `method="bm25"`, snapshot index with
explicit refresh — fully independent from vector retrieval. M7 fuses
both paths with Reciprocal Rank Fusion (`rrf_k=60`, candidate pools
`2 × top_k`) into `method="hybrid"` results. M8 reranks those
candidates with local `BAAI/bge-reranker-base` (raw cross-encoder
scores, top 20 → top 5, `method="reranked"`). M9 adds the generation
capability: `LLMProvider` abstraction + `OllamaLLMProvider`
(`qwen3:4b`, timeouts, typed errors). M10 wires it up: hybrid →
rerank → delimited grounded prompt → plain-text `GroundedAnswer`
with evidence and model identity. M11 adds citations: evidence
blocks carry `Citation [N]` markers, the model is instructed to cite
only shown markers, and markers are extracted back to evidence
(first-use order, invalid references retained). M12 verifies each
cited claim against its cited evidence with the local LLM
(JSON verdicts: supported/unsupported/insufficient_evidence, plus
explicit invalid/uncited/unverifiable states) — an LLM-assisted
baseline that itself needs evaluation, not a correctness proof.
M13 adds explicit outcomes (`no_evidence` / `insufficient_evidence` /
`answered` / `conflicting_evidence`). M14 adds a bounded research
agent (3 read-only tools, hard caps). M15 adds request tracing
(stages, counters, per-role models). M16 makes it measurable:
versioned eval datasets, Recall@K/MRR, deterministic
answer/citation/abstention/conflict metrics, P50/P95 latencies, and
fingerprinted experiments across vector/BM25/hybrid/reranked
configs — measurements only, never winners.

No frontend yet.

## Repository layout (root = `C:\Users\tripa\Projects\DeepResearch`)

```text
.
├── apps/               # reserved (future web/API wrappers, M19)
├── docs/               # PRD, architecture, evaluation, prompts
├── evals/              # reserved (datasets/runners, M16+)
├── infra/              # migrations/001_initial.sql + deploy extras
├── src/deepresearch/   # config, logging, db, models, repository, parsing/chunking/ingestion/embeddings/retrieval/bm25/hybrid/reranker/llm/generation/citations/verification/answer_status/agent/observability/evaluation/eval_runner, main
├── tests/              # unit (sqlite) + PG integration + fixtures/
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

## Ingestion (M3)

```powershell
docker compose up -d postgres
python -c "from deepresearch.config import get_settings; from deepresearch.db import get_engine, init_db; init_db(get_engine(get_settings()))"
python -c "
from pathlib import Path
from deepresearch.config import get_settings
from deepresearch.db import get_engine, get_session_factory, init_db
from deepresearch.ingestion import ingest_file
engine = get_engine(get_settings()); init_db(engine)
with get_session_factory(engine)() as s:
    r = ingest_file(s, path='tests/fixtures/sample.md')
    print(r.document.document_type, len(r.chunks), r.duplicate)
"
```

Pipeline: bytes → `parsing.parse_bytes` → sha256 hash → hash lookup →
`chunking.chunk_units` (target/overlap from `Settings`: 800/120) →
`repository` persist. Re-ingesting identical content returns existing
rows (`duplicate=True`). Details: `docs/adr/003-ingestion-chunking.md`.

## Embeddings (M4)

```powershell
docker compose up -d postgres
python -c "
from deepresearch.config import get_settings
from deepresearch.db import get_engine, get_session_factory, init_db
from deepresearch.embeddings import LocalEmbeddingProvider, embed_pending_chunks
s = get_settings(); engine = get_engine(s); init_db(engine)
provider = LocalEmbeddingProvider(model_name=s.embedding_model, device=s.embedding_device)
with get_session_factory(engine)() as session:
    print(embed_pending_chunks(session, provider, batch_size=s.embedding_batch_size))
"
```

First run downloads `BAAI/bge-small-en-v1.5` (~130 MB) once to the HF
cache; reruns skip embedded chunks (`embedded=0`). Vectors are
L2-normalized 384-d; model/version recorded per chunk. Details:
`docs/adr/004-embeddings.md`.

## Retrieval (M5)

```powershell
python -c "
from deepresearch.config import get_settings
from deepresearch.db import get_engine, get_session_factory
from deepresearch.embeddings import LocalEmbeddingProvider
from deepresearch.retrieval import retrieve
s = get_settings(); engine = get_engine(s)
provider = LocalEmbeddingProvider(model_name=s.embedding_model, device=s.embedding_device)
with get_session_factory(engine)() as session:
    for r in retrieve(session, provider, 'hybrid retrieval', top_k=s.retrieval_top_k):
        print(round(r.score, 4), r.chunk_index, r.text[:80])
"
```

Cosine similarity (`1 − pgvector distance`, higher = more similar),
exact search (no approximate index at this corpus size), deterministic
tie-breaks, unembedded/foreign-model chunks excluded. Details:
`docs/adr/005-vector-retrieval.md`.

## Lexical retrieval (M6)

```powershell
python -c "
from deepresearch.config import get_settings
from deepresearch.db import get_engine, get_session_factory
from deepresearch.bm25 import BM25Retriever
s = get_settings(); engine = get_engine(s)
with get_session_factory(engine)() as session:
    retriever = BM25Retriever(k1=s.bm25_k1, b=s.bm25_b)
    for r in retriever.retrieve(session, 'hybrid retrieval', top_k=s.retrieval_top_k):
        print(round(r.score, 4), r.chunk_index, r.text[:80])
"
```

Okapi BM25 (`k1=1.5`, `b=0.75`) over `Chunk.text`, lowercased M3
tokens, no stemming. Snapshot index with explicit refresh (stale use
raises); zero-score docs excluded. Details: `docs/adr/006-bm25-lexical.md`.

## Hybrid retrieval (M7)

```powershell
python -c "
from deepresearch.config import get_settings
from deepresearch.db import get_engine, get_session_factory
from deepresearch.embeddings import LocalEmbeddingProvider
from deepresearch.hybrid import retrieve_hybrid
s = get_settings(); engine = get_engine(s)
provider = LocalEmbeddingProvider(model_name=s.embedding_model, device=s.embedding_device)
with get_session_factory(engine)() as session:
    for r in retrieve_hybrid(session, provider, 'hybrid retrieval', top_k=s.retrieval_top_k):
        print(round(r.score, 4), r.chunk_index, r.text[:80])
"
```

Vector + BM25 candidates fused with RRF (`1 / (rrf_k + rank)`,
`rrf_k=60`): candidate pools `2 × top_k`, union dedup by chunk ID,
single query embedding, failures propagate. Baseline fusion — M16
experiments will measure it against each path alone. Details:
`docs/adr/ADR-007-hybrid-retrieval.md`.

## Reranking (M8)

```powershell
python -c "
from deepresearch.config import get_settings
from deepresearch.db import get_engine, get_session_factory
from deepresearch.embeddings import LocalEmbeddingProvider
from deepresearch.hybrid import retrieve_hybrid
from deepresearch.reranker import LocalCrossEncoderReranker, rerank_results
s = get_settings(); engine = get_engine(s)
provider = LocalEmbeddingProvider(model_name=s.embedding_model, device=s.embedding_device)
reranker = LocalCrossEncoderReranker(model_name=s.reranker_model, device=s.reranker_device)
with get_session_factory(engine)() as session:
    hybrid = retrieve_hybrid(session, provider, 'hybrid retrieval', top_k=s.reranker_candidate_top_k)
    for r in rerank_results('hybrid retrieval', hybrid, reranker, top_k=s.retrieval_top_k):
        print(round(r.score, 4), r.chunk_index, r.text[:80])
"
```

Top 20 hybrid candidates jointly scored with the query by local
`BAAI/bge-reranker-base` (~278M params, CPU-first; one-time ~1.1 GB
download to the HF cache). Raw scores, ties by chunk ID, failures
propagate. Details: `docs/adr/ADR-008-reranking.md`.

## Generation provider (M9)

Ollama must be installed and running; install the model once:

```powershell
ollama pull qwen3:4b
ollama list   # verify qwen3:4b is present
```

Do not pull larger variants (no 14B/30B/32B, no 12B/27B). GPU use
depends on Ollama and the local environment — it is not guaranteed.

```powershell
python -c "
from deepresearch.config import get_settings
from deepresearch.llm import OllamaLLMProvider
s = get_settings()
provider = OllamaLLMProvider.from_settings(s)
try:
    print(provider.generate('Reply with exactly: OK'))
finally:
    provider.close()
"
```

The provider (`qwen3:4b`, 120 s timeout, temperature 0.0) feeds the
M10 grounded pipeline below. The live smoke test
(`tests/test_llm_ollama_live.py`) runs against the same daemon and
skips with setup instructions when Ollama is absent. Details:
`docs/adr/ADR-009-ollama-generation-provider.md`.

## Grounded answers (M10)

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
        session,
        LocalEmbeddingProvider(),
        LocalCrossEncoderReranker(device='cpu'),
        OllamaLLMProvider.from_settings(s),
        'What does hybrid retrieval combine?',
    )
    print(result.answer)
"
```

`answer_question` runs hybrid → rerank → delimited evidence prompt →
LLM, returning a `GroundedAnswer` (answer text, evidence, citations,
model identity). Evidence blocks carry `Citation [N]` markers; the
answer's markers are extracted back to evidence in first-use order,
with out-of-range markers retained as invalid references. Pass
`verify_citations=True` to judge each cited claim against its cited
evidence with the local LLM (JSON verdicts; `GroundedAnswer.
verification_report`): supported / unsupported / insufficient_evidence,
plus explicit invalid / uncited / unverifiable states. Empty evidence
short-circuits without calling any LLM; retrieved text stays untrusted
data inside evidence blocks. Requires ingested + embedded chunks and
a running Ollama (also for live verification). Details:
`docs/adr/ADR-010-grounded-generation.md`,
`docs/adr/ADR-011-citations.md`,
`docs/adr/ADR-012-citation-verification.md`. Verification is an
LLM-assisted baseline that itself needs evaluation — not a
correctness proof. M13 adds explicit outcomes: `status` is
`no_evidence` (fixed message, zero LLM calls), `insufficient_evidence`
(unsupported/unverifiable cited rows), `answered`, or
`conflicting_evidence` (deterministic numeric contradictions; both
sources preserved, conflict-aware prompting). Details:
`docs/adr/ADR-013-no-answer-and-conflict-handling.md`. Conflict
detection is conservative — no perfect-detection claim is made. No
factual-accuracy or production-readiness claim is made.

## Research agent (M14)

```powershell
python -c "
from deepresearch.config import get_settings
from deepresearch.db import get_engine, get_session_factory
from deepresearch.agent import run_research_agent
from deepresearch.embeddings import LocalEmbeddingProvider
from deepresearch.llm import OllamaLLMProvider
s = get_settings(); engine = get_engine(s)
with get_session_factory(engine)() as session:
    result = run_research_agent(
        session, LocalEmbeddingProvider(), OllamaLLMProvider.from_settings(s),
        'What does hybrid retrieval combine?',
    )
    print(result.termination_reason, len(result.evidence))
"
```

`run_research_agent` loops LLM decisions over exactly three
read-only tools (`search_documents`, `get_chunk`, `get_document`)
with hard caps (8 iterations, 12 tool calls, 60 s) and returns
deduplicated evidence for the existing answer pipeline. Local corpus
only — no web search, no code execution, no writes. Details:
`docs/adr/ADR-014-bounded-research-agent.md`. Not a general
autonomous agent; live reliability is smoke-tested, not asserted.

## Observability (M15)

```powershell
python -c "
from deepresearch.observability import traced_request, get_current_trace
with traced_request() as trace:
    pass  # run any pipeline call here; stages/counters attach to trace
print(trace.to_dict())
"
```

Every request gets an `X-Request-ID` (preserved or minted) with an
isolated in-memory `RequestTrace`: monotonic stage timings, candidate
counts per pipeline stage, model identity per role (`llm`,
`verifier`, `agent`), token/cost fields that stay `None` unless a
provider reports them, and explicit termination. Logs and traces
carry identifiers and counts only — never prompts, documents,
reasoning, or secrets. No external platform, no dashboard, no
persistence. Details: `docs/adr/ADR-015-observability.md`.

## Evaluation (M16)

```powershell
python -c "
from deepresearch.eval_runner import load_dataset, run_evaluation, save_result
from deepresearch.evaluation import ExperimentConfig
dataset = load_dataset('evals/datasets/eval-dev-v1.json')
print(dataset.version, len(dataset.cases), 'cases')
"
```

Versioned datasets (`evals/datasets/`, 8 explicit categories),
source-level ground truth, Recall@3/5/10 + MRR (`None` means
unavailable, never zero), deterministic answer/citation metrics
from verification rows and statuses, abstention/conflict rates,
P50/P95 from real trace timings, and sha256-fingerprinted
experiments across vector/BM25/hybrid/reranked configs. Results
serialize to `evals/results/` (git-ignored). The shipped
`eval-dev-v1` is a development fixture, not a representative
benchmark. Details: `docs/EVALUATION.md`,
`docs/adr/ADR-016-evaluation-framework.md`.

## Security (M17)

Threat model and trust boundaries: `docs/SECURITY.md` (retrieved
text is untrusted data; 3 read-only agent tools; content-free
observability). Tested properties: 8-attack injection corpus stays
data (`tests/fixtures/adversarial.txt`), tool allowlist + argument
abuse rejected fail-closed, structured output bounded
(parse→validate→repair→reject), citations strictly `[N]`-mapped,
agent exhaustion terminates, poisoned corpora keep provenance with
conflict status, fake secrets never reach logs/traces. Explicit
guards only where inputs were unbounded: 4000-char questions, 10 MB
documents, NUL-byte sanitization. No absolute-security claim —
see limitations in `docs/SECURITY.md`. Details:
`docs/adr/ADR-017-security-adversarial-testing.md`.

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

## What's next (not in M17)

Milestone 18: security evaluation suite (per-attack pass/fail
records with mitigations, rerun after fixes) — no production
serving yet.
