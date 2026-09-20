"""M4 unit tests — embedding provider, service, idempotency (SQLite + fake provider).

No model download, no network, no GPU: the deterministic fake stands in
for BAAI/bge-small-en-v1.5. Real-model behavior is verified in isolation
by tests/test_embedding_model_real.py; PostgreSQL persistence by
tests/test_embeddings_postgres.py.
"""

from __future__ import annotations

import math

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.config import Settings
from deepresearch.db import init_db
from deepresearch.embeddings import (
    DEFAULT_MODEL,
    EMBEDDING_DIMENSION,
    DimensionMismatchError,
    EmbeddingError,
    EmbeddingProvider,
    LocalEmbeddingProvider,
    ModelLoadError,
    embed_pending_chunks,
    resolve_device,
)
from deepresearch.ingestion import ingest_bytes
from tests.fakes import FakeEmbeddingProvider


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enforce_fk(dbapi_conn, _record):  # type: ignore[no-untyped-def]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    init_db(engine)
    sess = Session(bind=engine)
    try:
        yield sess
    finally:
        sess.close()


def _ingest_two_docs(session: Session, *, chunks_each: int = 3) -> int:
    total = 0
    for doc_no in range(2):
        words = " ".join(f"doc{doc_no}w{i}" for i in range(chunks_each * 10))
        result = ingest_bytes(
            session,
            raw=words.encode(),
            document_type="txt",
            source=f"m4-{doc_no}.txt",
            target_tokens=10,
            overlap_tokens=0,
        )
        total += len(result.chunks)
    assert total == 2 * chunks_each
    return total


def test_provider_interface_conformance() -> None:
    assert isinstance(FakeEmbeddingProvider(), EmbeddingProvider)


def test_embedding_config_defaults() -> None:
    settings = Settings()
    assert settings.embedding_model == DEFAULT_MODEL == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_model_version == "1"
    assert settings.embedding_batch_size == 32
    assert settings.embedding_device == "auto"
    assert settings.embedding_normalize is True


def test_embed_texts_dimensionality_and_order() -> None:
    provider = FakeEmbeddingProvider()
    vectors = provider.embed_texts(["alpha", "beta", "gamma"])
    assert len(vectors) == 3
    assert all(len(v) == EMBEDDING_DIMENSION == 384 for v in vectors)
    assert vectors[0] == provider.embed_texts(["alpha"])[0]
    assert provider.embed_texts([]) == []


def test_batching_splits_calls(session: Session) -> None:
    total = _ingest_two_docs(session)
    provider = FakeEmbeddingProvider()
    result = embed_pending_chunks(session, provider, batch_size=2)
    assert result.embedded == total == 6
    assert provider.calls == [2, 2, 2]


def test_device_resolution() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}
    try:
        cuda = resolve_device("cuda")
    except ModelLoadError:
        cuda = None
    assert cuda in {"cuda", None}
    with pytest.raises(ModelLoadError):
        resolve_device("tpu")


def test_vectors_are_normalized() -> None:
    vectors = FakeEmbeddingProvider().embed_texts(["some chunk text"])
    assert math.isclose(sum(v * v for v in vectors[0]), 1.0, rel_tol=1e-6)


def test_wrong_dimension_rejected_before_write(session: Session) -> None:
    _ingest_two_docs(session)
    with pytest.raises(DimensionMismatchError):
        embed_pending_chunks(session, FakeEmbeddingProvider(dimension=128))
    assert repository.list_chunks_missing_embeddings(session) != []
    assert all(c.embedding is None for c in repository.list_chunks_missing_embeddings(session))


def test_provider_failure_preserves_earlier_batches(session: Session) -> None:
    _ingest_two_docs(session, chunks_each=2)  # 4 chunks, one contains the poison text
    provider = FakeEmbeddingProvider(fail_on="POISON")
    poison = repository.list_chunks_missing_embeddings(session)[2]
    poison.text = "this chunk contains POISON text"
    session.commit()
    with pytest.raises(EmbeddingError):
        embed_pending_chunks(session, provider, batch_size=2)
    remaining = repository.list_chunks_missing_embeddings(session)
    assert len(remaining) == 2  # first batch committed, failed batch rolled back


def test_rerun_is_idempotent(session: Session) -> None:
    total = _ingest_two_docs(session)
    provider = FakeEmbeddingProvider()
    first = embed_pending_chunks(session, provider)
    assert (first.embedded, first.skipped_uptodate) == (total, 0)
    second = embed_pending_chunks(session, provider)
    assert (second.embedded, second.skipped_uptodate) == (0, total)
    assert provider.calls == [total]  # no model calls on the rerun


def test_model_mismatch_skips_unless_forced(session: Session) -> None:
    total = _ingest_two_docs(session)
    embed_pending_chunks(session, FakeEmbeddingProvider(model_name="model-a"))
    other = FakeEmbeddingProvider(model_name="model-b")
    skipped = embed_pending_chunks(session, other)
    assert (skipped.embedded, skipped.skipped_mismatch) == (0, total)
    forced = embed_pending_chunks(session, other, force=True)
    assert forced.embedded == total
    assert forced.skipped_mismatch == 0


def test_limit_caps_work(session: Session) -> None:
    _ingest_two_docs(session)
    result = embed_pending_chunks(session, FakeEmbeddingProvider(), limit=2)
    assert result.embedded == 2


def test_invalid_batch_size_rejected(session: Session) -> None:
    with pytest.raises(EmbeddingError):
        embed_pending_chunks(session, FakeEmbeddingProvider(), batch_size=0)
    with pytest.raises(EmbeddingError):
        LocalEmbeddingProvider(batch_size=0)


def test_local_provider_configuration_without_loading() -> None:
    provider = LocalEmbeddingProvider()
    assert provider.model_name == DEFAULT_MODEL
    assert provider.model_version == "1"
    assert provider.dimension == EMBEDDING_DIMENSION == 384
    assert provider.batch_size == 32
    assert provider.normalized is True
    assert isinstance(provider, EmbeddingProvider)
