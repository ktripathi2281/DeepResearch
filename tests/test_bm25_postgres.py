"""M6 PostgreSQL integration — BM25 over persisted chunks.

Proves the corpus loads from PG, ranking/metadata/filters match the
SQLite behavior, rebuilds are equivalent, and the refresh strategy
detects new chunks. No embeddings, no network, no model.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.bm25 import (
    BM25Retriever,
    StaleIndexError,
    build_index_from_session,
    retrieve_bm25,
)
from deepresearch.config import get_settings
from deepresearch.db import check_connection, init_db


def _pg_engine() -> Engine:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    return engine


def _seed(session: Session, source: str, texts: list[str]) -> None:
    doc = repository.create_document(
        session,
        title="PG BM25 Doc",
        source=source,
        content_hash=f"m6pg-{uuid.uuid4().hex}",
        document_type="markdown",
        metadata=None,
    )
    for index, chunk_text in enumerate(texts):
        repository.create_chunk(
            session,
            document_id=doc.id,
            text=chunk_text,
            chunk_index=index,
            section="PGSec" if index == 0 else None,
            page=1 if index == 0 else None,
            metadata=None,
        )
    session.commit()


def _cleanup(engine: Engine, like: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM chunks WHERE document_id IN "
                "(SELECT id FROM documents WHERE source LIKE :p)"
            ),
            {"p": like},
        )
        conn.execute(text("DELETE FROM documents WHERE source LIKE :p"), {"p": like})


def test_corpus_ids_metadata_and_filters() -> None:
    engine = _pg_engine()
    prefix = f"pg-m6-{uuid.uuid4().hex[:8]}"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, f"{prefix}-a.md", ["pgvector cosine similarity search"])
            _seed(session, f"{prefix}-b.md", ["unrelated cooking recipes"])
            session.expunge_all()

            results = retrieve_bm25(session, "pgvector cosine similarity", top_k=5)
            assert results
            assert results[0].text == "pgvector cosine similarity search"
            assert results[0].document_title == "PG BM25 Doc"
            assert results[0].page == 1
            assert results[0].section == "PGSec"
            assert results[0].retrieval_method == "bm25"
            session.expunge_all()

            one = retrieve_bm25(session, "pgvector", document_id=results[0].document_id)
            assert [r.text for r in one] == ["pgvector cosine similarity search"]
    finally:
        try:
            _cleanup(engine, f"{prefix}%")
        finally:
            engine.dispose()


def test_rebuild_equivalence_and_refresh_strategy() -> None:
    engine = _pg_engine()
    source = f"pg-m6r-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source, ["first persisted chunk words"])
            session.expunge_all()

            retriever = BM25Retriever()
            assert retriever.refresh(session) == 1
            before = [(r.chunk_id, r.score) for r in retriever.retrieve(session, "persisted")]
            session.expunge_all()

            # A rebuild from the same corpus gives identical results.
            rebuilt, _ = build_index_from_session(session)
            via_rebuilt = [
                (r.chunk_id, r.score) for r in retrieve_bm25(session, "persisted", index=rebuilt)
            ]
            assert via_rebuilt == before
            session.expunge_all()

            # New chunk: held index goes stale explicitly, refresh recovers.
            _seed(session, source, ["second persisted chunk words"])
            session.expunge_all()
            assert retriever.index is not None and not retriever.index.is_fresh(session)
            with pytest.raises(StaleIndexError):
                retrieve_bm25(session, "second", index=retriever.index)
            after = [r.text for r in retriever.retrieve(session, "second")]
            assert after == ["second persisted chunk words"]
            assert retriever.index is not None and retriever.index.is_fresh(session)
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()
