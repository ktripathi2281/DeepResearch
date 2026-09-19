# DeepResearch — Product Requirements

## 1. Product overview

DeepResearch is a local-first, evidence-based research assistant. A user submits a research question and the system retrieves relevant evidence from an indexed document corpus, optionally searches approved web sources, synthesizes an answer, attaches source citations, verifies that important claims are supported by evidence, and records evaluation/observability data.

The project is intentionally designed to run at ₹0 using local models and open-source components. Paid model providers may be added later through provider adapters, but no core feature may require a paid API.

## 2. Goals

### Primary goals
1. Build a production-style RAG system rather than a basic "chat with PDFs" demo.
2. Demonstrate understanding of ingestion, chunking, embeddings, vector search, BM25, hybrid retrieval, reranking, generation, citations, and evaluation.
3. Make the system locally runnable with Ollama, local embeddings, PostgreSQL + pgvector, and Docker.
4. Make important AI behavior measurable.
5. Demonstrate defensive behavior against hallucination, insufficient evidence, conflicting evidence, and prompt injection.
6. Keep the architecture provider-agnostic.

### Secondary goals
1. Provide a clean web UI.
2. Expose research traces without exposing private chain-of-thought.
3. Track latency, token usage where available, retrieval statistics, and estimated cost.
4. Make experiments reproducible through versioned configurations and evaluation datasets.

## 3. Target users

Primary user: a developer, researcher, or technically curious user who wants evidence-backed answers from a controlled corpus.

This is a portfolio project, so developer usability, architecture clarity, testing, and measurable behavior are more important than consumer-scale polish.

## 4. Core user journey

1. User opens the web application.
2. User submits a research question.
3. System creates a research plan or search query set.
4. System retrieves candidate evidence.
5. System reranks candidates.
6. System generates an answer constrained by retrieved evidence.
7. System attaches citations to claims.
8. Citation verification checks important claims against evidence.
9. If evidence is insufficient, the system says so rather than inventing an answer.
10. User sees the answer, citations, sources, and high-level research trace.
11. The request is recorded for evaluation/observability.

## 5. Functional requirements

### FR-1: Document ingestion
Support at minimum:
- PDF
- Markdown
- TXT
- HTML

For every document store:
- document ID
- title
- source
- content hash
- document type
- ingestion timestamp
- metadata

For every chunk store:
- chunk ID
- document ID
- chunk text
- chunk index
- section heading when available
- page number when available
- source metadata
- embedding
- embedding model/version

Ingestion must be idempotent based on a content hash.

### FR-2: Chunking
Implement a configurable chunking strategy.
At minimum support:
- target chunk size
- overlap
- structure-aware metadata

The implementation must make chunking configuration explicit so experiments can compare strategies.

### FR-3: Embeddings
Use a local embedding model by default.
The embedding provider must be abstracted behind an interface.

The system must record:
- embedding model
- embedding dimension
- embedding version

### FR-4: Semantic retrieval
Retrieve top-K chunks using pgvector similarity.

### FR-5: Lexical retrieval
Implement BM25 retrieval.

### FR-6: Hybrid retrieval
Combine semantic and lexical candidates using a documented score-fusion strategy.
The strategy must be configurable.

### FR-7: Reranking
Rerank the hybrid candidate set with a local reranker.
Make the reranker configurable behind an interface.

### FR-8: Answer generation
Use a local Ollama model by default.
The model must receive:
- user question
- selected evidence
- citation metadata
- explicit instruction to avoid unsupported claims

The model must produce structured output internally where practical.

### FR-9: Citations
Every source citation must point to a real indexed source/chunk.
The UI must allow a user to inspect the supporting evidence.

### FR-10: Citation verification
For generated claims that are presented as factual:
- identify the associated evidence
- determine whether the evidence supports the claim
- flag unsupported claims
- do not silently fabricate citations

The first implementation may use a local LLM judge plus deterministic checks. The verification method must be documented.

### FR-11: Insufficient evidence
When retrieval confidence is low or evidence does not support an answer, the system must explicitly communicate insufficient evidence.

It must never fill missing evidence with invented facts.

### FR-12: Conflicting evidence
When credible indexed sources disagree:
- retain both pieces of evidence
- identify the conflict
- avoid silently selecting one as true
- cite the conflicting sources

### FR-13: Prompt-injection defense
Retrieved text must be treated as untrusted data.
Instructions embedded inside documents must not override system/application instructions.

Include test fixtures containing obvious injection attempts.

### FR-14: Research agent
The research agent may use tools such as:
- search indexed documents
- retrieve a document/chunk
- search approved web sources if web search is enabled

The agent must have:
- a maximum iteration count
- a maximum tool-call count
- timeout handling
- failure handling

The first release must work without web search.

### FR-15: Observability
Record per research request:
- request ID
- timestamps
- model/provider
- retrieval latency
- reranking latency
- generation latency
- total latency
- candidate counts
- selected chunk count
- token counts when available
- estimated cost when calculable
- evaluation/version identifiers

Do not store private chain-of-thought. Store tool events, inputs/outputs needed for debugging, and concise decision metadata instead.

### FR-16: Evaluation
Provide an offline evaluation runner using a versioned dataset.
At minimum evaluate:
- Recall@K
- MRR
- answer correctness
- faithfulness
- citation correctness
- citation completeness
- latency

### FR-17: Web UI
Provide:
- question input
- streaming or progressive status where practical
- final answer
- inline citations
- source/evidence viewer
- research trace summary
- error/insufficient-evidence states

## 6. Non-goals

Do not build:
- a general-purpose autonomous web crawler
- multi-user enterprise auth
- billing
- mobile apps
- fine-tuning infrastructure
- distributed GPU serving
- Kubernetes deployment
- production-scale distributed infrastructure
- arbitrary external tool execution

Do not add frameworks merely because they are popular. Prefer simple Python implementations where they improve understanding.

## 7. Local-first constraints

The default stack must work without paid APIs:
- Ollama for generation
- local embedding model
- local reranker
- PostgreSQL + pgvector
- BM25 implementation
- FastAPI
- React/Next.js
- Docker

Optional commercial providers may be added later behind interfaces.


## 8. Initial hardware and model configuration

Development hardware:
- Lenovo LOQ
- 16 GB RAM
- 6 GB GPU VRAM

Initial local stack:
- Generation: Ollama `qwen3:4b` Q4_K_M
- Embeddings: `BAAI/bge-small-en-v1.5` (384 dimensions)
- Reranking: `BAAI/bge-reranker-base`, preferably CPU initially if needed

Optional later generation comparison:
- `gemma3:4b`

The project must not require larger models or paid APIs for its core functionality.

## 9. Quality requirements

- Unit tests for core algorithms.
- Integration tests for ingestion/retrieval.
- Evaluation tests for RAG quality.
- Clear error handling.
- Type hints.
- Configuration through environment/config files.
- No secrets committed to Git.
- Docker-based local startup.
- README with architecture and setup instructions.

## 10. Acceptance criteria for MVP

The MVP is complete when:
1. A user can ingest a corpus.
2. Documents are parsed, chunked, embedded, and stored.
3. A question retrieves relevant chunks.
4. Hybrid retrieval works.
5. Reranking works.
6. A local LLM generates an evidence-grounded answer.
7. Citations resolve to actual evidence.
8. Unsupported/no-answer cases are handled.
9. Prompt injection fixtures are blocked.
10. An evaluation dataset can be executed.
11. Retrieval and generation metrics are recorded.
12. The full system starts locally with documented commands.
