# ADR-001: PostgreSQL + pgvector schema (documents, chunks)

Date: 2026-09-19 | Status: Accepted | Milestone: M2

## Context

M2 needs the persistence foundation for `document → chunks →
embeddings/retrieval → evidence → generation` (ARCHITECTURE §5–§6,
FR-1/FR-2/FR-3). M1 already fixed PostgreSQL + pgvector
(`pgvector/pgvector:pg16`) as the vector store. Open questions were
PK choice, idempotency mechanism, embedding column shape, and which
tables to create now.

## Decision

- **PostgreSQL + pgvector only.** No second store for BM25 (in-Python,
  M6) or metadata.
- **Two tables in M2: `documents`, `chunks`.** Deferred until their
  milestone justifies them: `research_requests` (M14 agent),
  `evaluation_cases` (M16), observability/event tables (M15).
- **UUIDv4 primary keys** (`sqlalchemy.Uuid`). Evidence/citation
  references must not leak ingestion order, and UUIDs keep future
  distributed ingestion simple.
- **`documents.content_hash` UNIQUE (sha256 hex)** is the ingestion
  idempotency key. PK stays synthetic because the same hash must be
  rejected regardless of title/source.
- **`chunks(document_id, chunk_index)` UNIQUE + `chunk_index >= 0`
  CHECK.** Re-ingestion reproduces chunk positions deterministically;
  negative indexes are rejected at the DB, not just in Python.
- **`section` / `page` nullable.** Structure-aware chunking (M3)
  preserves them when the parser provides them; TXT sources have none.
- **`metadata` JSON column on both tables** (ORM attributes
  `doc_metadata` / `chunk_metadata` — `metadata` alone would clash
  with `DeclarativeBase.metadata`). Carries source extras without
  schema churn.
- **Embedding as schema capability, not data:** `chunks.embedding`
  `VECTOR(384)` NULLABLE + `embedding_model` / `embedding_version`
  NULLABLE. 384 matches the frozen `BAAI/bge-small-en-v1.5` (M1
  decision 7). All NULL in M2; M4 is the first writer, M5 the first
  reader. Nullable (not absent) so M4 adds code, not DDL.
- **Immutable ingestion records:** `created_at` only, no `updated_at`.
  Documents/chunks are never updated in place; corrections ingest a
  new document (new hash).
- **`document_type` unconstrained String(32).** The four M3 input
  types (pdf/md/txt/html) are validated in Python, not via CHECK, so
  adding a type never requires a migration.
- **Cascade delete** `chunks → documents`. Deleting a document always
  removes its chunks; chunks never outlive their source.

## Alternatives considered

- Integer PKs: simpler, but leak order and complicate citation IDs.
- Separate `embeddings` table: rejected — 1:1 with chunks, a nullable
  column is simpler and keeps vector-index locality.
- `updated_at` on both tables: rejected — contradicts immutable
  ingestion; would suggest in-place mutation we forbid.

## Consequences

- M3 implements hash-check → parse → chunk → insert against these
  constraints; duplicate content fails fast with `IntegrityError`.
- M4 backfills `embedding`/`embedding_model`/`embedding_version`
  without DDL changes.
- SQLite unit tests work (JSON + `VECTOR(384)` degrade gracefully;
  embedding stays NULL), PG integration tests prove the real DDL.
