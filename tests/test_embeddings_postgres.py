"""M4 PostgreSQL integration — fake-provider persistence, idempotency, mismatch.

Skipped without a live DB. Real-model persistence is verified in
tests/test_embedding_model_real.py.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.config import get_settings
from deepresearch.db import check_connection, init_db
from deepresearch.embeddings import EMBEDDING_DIMENSION, embed_pending_chunks
from deepresearch.ingestion import ingest_bytes
from tests.fakes import FakeEmbeddingProvider

FIXTURES = Path(__file__).parent / "fixtures"


def _pg_engine() -> Engine:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    return engine


def _cleanup(engine: Engine, source: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM chunks WHERE document_id IN "
                "(SELECT id FROM documents WHERE source = :s)"
            ),
            {"s": source},
        )
        conn.execute(text("DELETE FROM documents WHERE source = :s"), {"s": source})


def test_embed_persist_readback_and_idempotent_rerun() -> None:
    engine = _pg_engine()
    raw = (FIXTURES / "sample.md").read_bytes()
    source = f"pg-m4-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            ingested = ingest_bytes(session, raw=raw, document_type="markdown", source=source)
            assert len(ingested.chunks) >= 1
            doc_id = ingested.document.id
            count = len(ingested.chunks)
            session.expunge_all()

            first = embed_pending_chunks(session, FakeEmbeddingProvider(), batch_size=2)
            assert first.embedded == count
            assert first.skipped_uptodate == 0
            session.expunge_all()

            chunks = repository.list_chunks_by_document(session, doc_id)
            assert len(chunks) == count
            for chunk in chunks:
                assert chunk.embedding is not None
                assert len(chunk.embedding) == EMBEDDING_DIMENSION == 384
                assert chunk.embedding_model == "fake-test-model"
                assert chunk.embedding_version == "test-v1"
            session.expunge_all()

            second = embed_pending_chunks(session, FakeEmbeddingProvider())
            assert (second.embedded, second.skipped_uptodate) == (0, count)
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_model_mismatch_reported_and_force_reembeds() -> None:
    engine = _pg_engine()
    source = f"pg-m4x-{uuid.uuid4().hex[:8]}.txt"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            ingested = ingest_bytes(
                session,
                raw=b"mismatch behavior probe text for embeddings",
                document_type="txt",
                source=source,
            )
            count = len(ingested.chunks)
            doc_id = ingested.document.id
            session.expunge_all()

            embed_pending_chunks(session, FakeEmbeddingProvider(model_name="model-a"))
            session.expunge_all()
            reported = embed_pending_chunks(session, FakeEmbeddingProvider(model_name="model-b"))
            assert (reported.embedded, reported.skipped_mismatch) == (0, count)
            session.expunge_all()
            forced = embed_pending_chunks(
                session, FakeEmbeddingProvider(model_name="model-b"), force=True
            )
            assert forced.embedded == count
            session.expunge_all()
            assert {
                c.embedding_model for c in repository.list_chunks_by_document(session, doc_id)
            } == {"model-b"}
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()
