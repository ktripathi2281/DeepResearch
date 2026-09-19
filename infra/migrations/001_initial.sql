-- DeepResearch migration 001: initial schema (Milestone 2).
--
-- Canonical DDL matching src/deepresearch/models.py. The ORM models are
-- the source of truth; this file is the reviewable artifact and the
-- reference for `docker compose` first-start. Idempotent via IF NOT EXISTS.
-- Alembic is deferred (see docs/adr/002-migration-strategy.md).

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(512),
    source TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL UNIQUE,
    document_type VARCHAR(32) NOT NULL,
    metadata JSON,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_documents_content_hash ON documents (content_hash);
CREATE INDEX IF NOT EXISTS ix_documents_document_type ON documents (document_type);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    section VARCHAR(512),
    page INTEGER CHECK (page IS NULL OR page >= 1),
    metadata JSON,
    embedding VECTOR(384),
    embedding_model VARCHAR(128),
    embedding_version VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_chunks_document_chunk_index UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS ix_chunks_document_id ON chunks (document_id);
