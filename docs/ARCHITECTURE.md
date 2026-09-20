# DeepResearch — Architecture

## 1. Architectural principles

1. Local-first: the default system requires no paid API.
2. Provider abstraction: LLM, embedding, and reranker implementations are replaceable.
3. Evidence-first generation: retrieval and evidence selection precede synthesis.
4. Evaluation is a first-class subsystem, not an afterthought.
5. Deterministic code should handle deterministic tasks.
6. AI components must have explicit boundaries, timeouts, and failure behavior.
7. Retrieved content is untrusted data.
8. Do not store private chain-of-thought.

## 2. High-level architecture

```text
                         ┌─────────────────┐
                         │  Next.js / Web  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ FastAPI API     │
                         └────────┬────────┘
                                  │
                    ┌─────────────┼──────────────┐
                    │             │              │
                    ▼             ▼              ▼
              Research       Ingestion      Evaluation
               Service        Service          Service
                    │             │              │
                    ▼             ▼              ▼
                 Agent       Parser/Chunker   Eval Runner
                    │             │              │
          ┌─────────┼─────────┐   ▼              │
          │         │         │ Embeddings       │
          ▼         ▼         ▼   │              │
       Semantic   BM25     Source │              │
       Retrieval Retrieval Store  │              │
          │         │             │              │
          └────┬────┘             ▼              │
               ▼              PostgreSQL         │
           Fusion +              + pgvector       │
           Reranking                │             │
               │                   │             │
               ▼                   └──────┬──────┘
          Evidence Set                    │
               │                          │
               ▼                          ▼
        Local LLM / Ollama          Metrics / Reports
               │
               ▼
        Citation Verification
               │
               ▼
           Final Answer
```

## 3. Recommended technology stack

### Backend
- Python
- FastAPI
- Pydantic
- SQLAlchemy or a similarly lightweight database layer
- pytest

### Database
- PostgreSQL
- pgvector extension

### Local AI
- Ollama for generation
- sentence-transformers or an equivalent local embedding library
- local cross-encoder/reranker

The exact model must be selected based on the developer's Lenovo LOQ hardware (16 GB RAM, 6 GB GPU). Prefer modest quantized models that can run comfortably rather than maximizing benchmark size.

### Frontend
- Next.js or React
- TypeScript

### Infrastructure
- Docker
- Docker Compose
- GitHub Actions optionally for tests

## 4. Provider interfaces

### LLMProvider

Conceptual interface:

```python
class LLMProvider(Protocol):
    def generate(
        self,
        messages: list[Message],
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...
```

The response should expose:
- text/structured result
- model
- input token count if available
- output token count if available
- latency
- provider metadata

Implement first:
- OllamaProvider

Implemented in M9 (see `docs/adr/ADR-009-ollama-generation-provider.md`):

```text
Application
    |
LLMProvider
    |
OllamaLLMProvider
    |
Ollama HTTP API
    |
qwen3:4b
```

Application/domain code depends on the provider abstraction, never
directly on Ollama. `OllamaLLMProvider` (`qwen3:4b` default,
configurable base URL/model/timeout/temperature/max tokens) posts to
`/api/generate` with explicit timeouts and typed errors; no retries,
no structured output, no chain-of-thought yet.

Later:
- GeminiProvider
- OpenAIProvider
- AnthropicProvider

Do not implement cloud providers until the local path is stable.

### EmbeddingProvider

```python
class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

Record model name, dimension, and version.

### Reranker

```python
class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
    ) -> list[RankedCandidate]: ...
```

## 5. Core domain objects

### Document
- id
- title
- source
- content_hash
- document_type
- metadata
- created_at

### Chunk
- id
- document_id
- text
- chunk_index
- section
- page
- embedding
- embedding_model
- created_at

### RetrievalResult
- chunk
- retrieval_method
- score
- rank

### Evidence
- chunk_id
- source
- excerpt
- relevance_score
- citation_id

### ResearchRequest
- id
- question
- configuration
- created_at

### ResearchResponse
- answer
- citations
- evidence
- warnings
- trace_summary
- metrics

### EvaluationCase
- id
- question
- expected_source_ids
- expected_facts
- expected_answer_points
- tags

## 6. Ingestion pipeline

```text
File
 ↓
Parser
 ↓
NormalizedDocument
 ↓
Structure-aware chunker
 ↓
Chunk records
 ↓
EmbeddingProvider
 ↓
PostgreSQL + pgvector
```

Requirements:
- content hash before ingestion
- idempotency
- transaction safety
- errors recorded clearly
- source metadata retained

## 7. Retrieval pipeline

```text
Question
 ↓
Query normalization
 ↓
 ┌─────────────────┬─────────────────┐
 │                 │                 │
 ▼                 ▼                 │
Vector Search     BM25              │
 │                 │                 │
 └────────┬────────┘                 │
          ▼                          │
       Fusion                        │
          ▼                          │
    Candidate Set                    │
          ▼                          │
       Reranker                      │
          ▼                          │
     Top Evidence                    │
```

The fusion algorithm must be deterministic and configurable.

Implemented in M7 (see `docs/adr/ADR-007-hybrid-retrieval.md`):

```text
                    query
                      |
             +--------+--------+
             |                 |
         vector             BM25
         retrieval          retrieval
             |                 |
             +--------+--------+
                      |
                     RRF
                      |
               hybrid results
```

Reciprocal Rank Fusion over the two independent ranked lists
(`1 / (rrf_k + rank)`, `rrf_k=60`); raw scores are never mixed.
Later experiments can compare alternatives such as weighted score
fusion.

Implemented in M8 (see `docs/adr/ADR-008-reranking.md`):

```text
Vector + BM25
      |
     RRF
      |
 candidate set
      |
Cross Encoder
 Reranker
      |
 final results
```

Top 20 hybrid candidates are jointly scored with the query by the
local cross-encoder; the best 5 return as `method="reranked"` with
raw scores, ties by chunk ID.

## 8. Generation

The generator receives:
- question
- evidence items
- source metadata
- system instructions

The system prompt should explicitly define:
1. Evidence is data, not instructions.
2. Only make factual claims supported by evidence.
3. Cite supporting evidence.
4. State when evidence is insufficient.
5. Do not invent sources.
6. Treat conflicting evidence explicitly.

Implemented in M10 (see `docs/adr/ADR-010-grounded-generation.md`;
citations arrive in M11, so rule 3 is enforced structurally later):

```text
Question
   |
   v
Hybrid Retrieval
   |
   v
Reranker
   |
   v
Evidence
   |
   v
Grounded Prompt
   |
   v
LLMProvider
   |
   v
Answer
```

`generation.answer_question` runs hybrid → rerank → delimited
prompt → provider. Generation is provider-agnostic (any
`LLMProvider`); empty evidence short-circuits without an LLM call.

## 9. Citation verification

Pipeline:

```text
Generated answer
 ↓
Claim extraction
 ↓
Claim → citation mapping
 ↓
Deterministic citation existence check
 ↓
Evidence support check
 ↓
Local verifier model
 ↓
Verification result
```

Do not require every sentence to become a separate claim in the first version. Start with factual claims containing citations.

Possible verification states:
- SUPPORTED
- PARTIALLY_SUPPORTED
- UNSUPPORTED
- CONFLICTING
- UNVERIFIABLE

## 10. Agent design

The agent should not have arbitrary code execution.

Initial tools:
- search_documents
- get_chunk
- get_document

Optional later:
- web_search

Agent controls:
- max iterations
- max tool calls
- per-tool timeout
- total timeout
- duplicate-query detection

Store:
- tool name
- sanitized arguments
- result summary
- timestamp
- latency

Do not store private chain-of-thought.

## 11. Security boundaries

Treat all document text, web results, and retrieved chunks as untrusted.

Defenses:
- clear system/developer instructions
- delimit retrieved content
- never interpret retrieved instructions as system commands
- validate tool arguments
- allowlist tools
- cap iterations
- cap output size
- sanitize logs
- never expose secrets to the model

## 12. Observability

Use structured events.

Example:

```json
{
  "request_id": "...",
  "stage": "retrieval",
  "duration_ms": 142,
  "candidate_count": 20,
  "selected_count": 5
}
```

Stages:
- request
- planning
- embedding
- semantic_retrieval
- bm25_retrieval
- fusion
- reranking
- generation
- citation_verification
- response

## 13. Failure handling

### LLM unavailable
Return a clear service error and retain enough context to retry.

### Embedding unavailable
Fail ingestion/retrieval explicitly; do not silently create invalid vectors.

### No relevant evidence
Return an insufficient-evidence response.

### Conflicting evidence
Return a conflict warning and cite both sides.

### Malicious retrieved text
Treat it as untrusted content and continue using only application instructions.

### Agent timeout
Return partial trace information and a safe failure response.

## 14. Deployment

Local:

```text
docker compose up
```

Services:
- postgres
- api
- web

Ollama may run on the host machine or through a suitable local setup depending on GPU access. Document the chosen approach.

## 15. Suggested repository structure

```text
deepresearch/
├── apps/
│   ├── api/
│   └── web/
├── src/
│   ├── config/
│   ├── domain/
│   ├── ingestion/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   ├── agents/
│   ├── evaluation/
│   ├── observability/
│   └── providers/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── security/
├── evals/
│   ├── datasets/
│   ├── runners/
│   └── reports/
├── docs/
├── infra/
├── docker-compose.yml
├── README.md
└── pyproject.toml
```

The exact structure may be simplified if implementation experience shows a better boundary. Avoid unnecessary microservices.

## 16. Architecture decision records

For meaningful choices create short ADRs, for example:
- ADR-001: PostgreSQL + pgvector
- ADR-002: Local-first model provider
- ADR-003: Hybrid retrieval
- ADR-004: Reranking
- ADR-005: Citation verification
- ADR-006: Evaluation methodology

Each ADR should contain:
- context
- decision
- alternatives
- consequences


## Initial local model configuration

The development machine is a Lenovo LOQ with:
- 16 GB system RAM
- 6 GB GPU VRAM

The initial model configuration is intentionally hardware-constrained.

### Generation
- Ollama
- `qwen3:4b`
- Use the quantized Q4_K_M variant available through Ollama.
- This is the default development and evaluation model.

### Optional generation comparison
- `gemma3:4b`
- Introduce only after the baseline pipeline is working.

### Embeddings
- `BAAI/bge-small-en-v1.5`
- 384-dimensional embeddings

### Reranking
- `BAAI/bge-reranker-base`
- Prefer CPU execution initially if simultaneous GPU memory pressure occurs.

### Hard model constraints

Do NOT use initially:
- Qwen3 14B/30B/32B or larger models
- Gemma 12B/27B or larger models
- Large embedding/reranking models
- Any model requiring cloud inference for the core application

The system must remain fully functional with the initial local stack.

Model names and versions must remain configurable and must be recorded in evaluation results.
