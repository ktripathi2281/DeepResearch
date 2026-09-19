"""M2 PostgreSQL integration tests (skipped without a live DB).

Proves the real DDL: pgvector extension, UUID PKs, UNIQUE/CHECK/FK
constraints, and the repository round-trip. Run with:

    docker compose up -d postgres
    python -m pytest tests/test_postgres_schema.py -v

Unit coverage without Docker lives in tests/test_models.py.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.config import get_settings
from deepresearch.db import check_connection, init_db


def _pg_engine() -> Engine:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    return engine


def test_init_db_creates_extension_and_tables() -> None:
    engine = _pg_engine()
    try:
        init_db(engine)
        with engine.connect() as conn:
            ext = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
            assert ext == 1
        tables = set(inspect(engine).get_table_names())
        assert {"documents", "chunks"} <= tables
    finally:
        engine.dispose()


def test_document_chunk_roundtrip_on_postgres() -> None:
    engine = _pg_engine()
    content_hash = f"itest-{uuid.uuid4().hex}"[:64]
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            doc = repository.create_document(
                session,
                title="PG smoke doc",
                source="corpus/smoke.md",
                content_hash=content_hash,
                document_type="markdown",
                metadata={"suite": "postgres-schema"},
            )
            repository.create_chunk(
                session,
                document_id=doc.id,
                text="smoke chunk",
                chunk_index=0,
                section="Intro",
                page=1,
            )
            session.commit()
            assert repository.get_document_by_hash(session, content_hash) is not None
            chunks = repository.list_chunks_by_document(session, doc.id)
            assert len(chunks) == 1
            assert chunks[0].embedding is None
            with pytest.raises(IntegrityError):
                repository.create_document(
                    session,
                    title="dup",
                    source="other.md",
                    content_hash=content_hash,
                    document_type="markdown",
                )
            session.rollback()
    finally:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "DELETE FROM chunks WHERE document_id IN "
                        "(SELECT id FROM documents WHERE content_hash = :h)"
                    ),
                    {"h": content_hash},
                )
                conn.execute(
                    text("DELETE FROM documents WHERE content_hash = :h"), {"h": content_hash}
                )
        finally:
            engine.dispose()
