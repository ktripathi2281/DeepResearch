"""M3 PostgreSQL integration — ingestion persistence/idempotency on real DDL.

Skipped without a live DB (`docker compose up -d postgres`). Unit
coverage lives in tests/test_ingestion.py.
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
from deepresearch.ingestion import ingest_bytes

FIXTURES = Path(__file__).parent / "fixtures"


def _pg_engine() -> Engine:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    return engine


def test_ingest_markdown_persists_and_deduplicates() -> None:
    engine = _pg_engine()
    raw = (FIXTURES / "sample.md").read_bytes()
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            first = ingest_bytes(session, raw=raw, document_type="markdown", source="pg-sample.md")
            assert first.duplicate is False
            assert len(first.chunks) >= 1
            assert [c.chunk_index for c in first.chunks] == list(range(len(first.chunks)))
            first_id = first.document.id
            first_count = len(first.chunks)
            session.expunge_all()
            second = ingest_bytes(
                session, raw=raw, document_type="markdown", source="pg-sample-copy.md"
            )
            assert second.duplicate is True
            assert second.document.id == first_id
            assert len(second.chunks) == first_count
            assert len(repository.list_chunks_by_document(session, first_id)) == first_count
    finally:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "DELETE FROM chunks WHERE document_id IN "
                        "(SELECT id FROM documents WHERE source LIKE 'pg-sample%')"
                    )
                )
                conn.execute(text("DELETE FROM documents WHERE source LIKE 'pg-sample%'"))
        finally:
            engine.dispose()


def test_ingest_pdf_preserves_pages() -> None:
    engine = _pg_engine()
    raw = (FIXTURES / "sample.pdf").read_bytes()
    source = f"pg-pdf-{uuid.uuid4().hex[:8]}.pdf"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            result = ingest_bytes(
                session,
                raw=raw,
                document_type="pdf",
                source=source,
                target_tokens=8,
                overlap_tokens=0,
            )
            assert result.duplicate is False
            assert {c.page for c in result.chunks} == {1, 2}
    finally:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "DELETE FROM chunks WHERE document_id IN "
                        "(SELECT id FROM documents WHERE source = :s)"
                    ),
                    {"s": source},
                )
                conn.execute(text("DELETE FROM documents WHERE source = :s"), {"s": source})
        finally:
            engine.dispose()
