"""M4 real-model verification — BAAI/bge-small-en-v1.5 only.

Loads the actual local model (uses the HF cache when offline,
downloads once when online), then proves: 384 dimensions,
deterministic output, L2 normalization, and PostgreSQL persistence.

Skipped — never failed — when the model is neither cached nor
reachable, so the suite stays offline-safe. A loaded model that
misbehaves fails loudly instead.
"""

from __future__ import annotations

import math
import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.config import get_settings
from deepresearch.db import check_connection, init_db
from deepresearch.embeddings import (
    DEFAULT_MODEL,
    EMBEDDING_DIMENSION,
    LocalEmbeddingProvider,
    embed_pending_chunks,
)
from deepresearch.ingestion import ingest_bytes


def _model_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False
    try:
        snapshot_download(repo_id=DEFAULT_MODEL, local_files_only=True)
    except Exception:
        return False
    return True


def _network_up() -> bool:
    try:
        import urllib.request

        return urllib.request.urlopen("https://huggingface.co", timeout=10).status == 200
    except Exception:
        return False


def _pg_engine() -> Engine:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    return engine


@pytest.fixture(scope="module")
def provider() -> LocalEmbeddingProvider:
    if not (_model_cached() or _network_up()):
        pytest.skip(f"{DEFAULT_MODEL} not cached and no network; skipping real-model check")
    return LocalEmbeddingProvider()


def test_real_model_dimension_determinism_normalization(provider: LocalEmbeddingProvider) -> None:
    texts = ["DeepResearch verification probe sentence.", "A second probe sentence."]
    first = provider.embed_texts(texts)
    assert len(first) == 2
    assert all(len(v) == EMBEDDING_DIMENSION == 384 for v in first)
    assert provider.dimension == 384
    assert provider.device in {"cpu", "cuda"}
    assert provider.embed_texts(texts) == first  # deterministic same model/device
    for vector in first:
        assert math.isclose(sum(v * v for v in vector), 1.0, rel_tol=1e-4)


def test_real_model_persist_to_postgres(provider: LocalEmbeddingProvider) -> None:
    engine = _pg_engine()
    source = f"pg-m4real-{uuid.uuid4().hex[:8]}.txt"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            ingested = ingest_bytes(
                session,
                raw=b"real model persistence probe for bge-small-en-v1.5",
                document_type="txt",
                source=source,
            )
            count = len(ingested.chunks)
            doc_id = ingested.document.id
            session.expunge_all()
            result = embed_pending_chunks(session, provider)
            assert result.embedded == count
            assert result.model_name == DEFAULT_MODEL
            session.expunge_all()
            for chunk in repository.list_chunks_by_document(session, doc_id):
                assert chunk.embedding is not None
                assert len(chunk.embedding) == 384
                assert chunk.embedding_model == DEFAULT_MODEL
                assert chunk.embedding_version == "1"
            session.expunge_all()
            rerun = embed_pending_chunks(session, provider)
            assert (rerun.embedded, rerun.skipped_uptodate) == (0, count)
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
