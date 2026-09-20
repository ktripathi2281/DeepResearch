"""M5 unit tests — result model, validation, mapping (no database ranking).

pgvector operators need PostgreSQL, so the SQL layer is stubbed here
and proven for real in tests/test_retrieval_postgres.py. Provider is
the deterministic fake: no model, no network, no GPU.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.config import Settings
from deepresearch.db import init_db
from deepresearch.embeddings import EMBEDDING_DIMENSION
from deepresearch.models import Chunk, Document
from deepresearch.retrieval import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    RetrievalError,
    RetrievalResult,
    retrieve,
    validate_top_k,
)
from tests.fakes import FakeEmbeddingProvider


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    sess = Session(bind=engine)
    try:
        yield sess
    finally:
        sess.close()


def _row(score: float, index: int = 0) -> tuple[Chunk, Document, float]:
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Doc",
        source="s.md",
        content_hash="h" * 64,
        document_type="markdown",
        doc_metadata=None,
    )
    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=doc_id,
        text=f"chunk {index}",
        chunk_index=index,
        section="S",
        page=2,
        chunk_metadata={"k": "v"},
    )
    return chunk, doc, score


def test_result_model_defaults_and_fields() -> None:
    chunk, doc, _ = _row(0.9)
    result = RetrievalResult(
        chunk_id=chunk.id,
        document_id=doc.id,
        chunk_index=0,
        text="t",
        score=0.9,
        rank=1,
        document_title="Doc",
        document_type="markdown",
        document_source="s.md",
        page=2,
        section="S",
        chunk_metadata=None,
    )
    assert result.retrieval_method == "vector"
    assert result.rank == 1


def test_top_k_validation() -> None:
    assert validate_top_k(1) == 1
    assert validate_top_k(DEFAULT_TOP_K) == DEFAULT_TOP_K
    for bad in (0, -3, MAX_TOP_K + 1, True, "5", 2.5, None):
        with pytest.raises(RetrievalError):
            validate_top_k(bad)  # type: ignore[arg-type]
    with pytest.raises(RetrievalError):
        validate_top_k(11, maximum=10)


def test_retrieval_config_defaults() -> None:
    settings = Settings()
    assert (settings.retrieval_top_k, settings.retrieval_max_top_k) == (
        DEFAULT_TOP_K,
        MAX_TOP_K,
    )


def test_blank_query_rejected_without_provider_call(session: Session) -> None:
    provider = FakeEmbeddingProvider()
    for bad in ("", "   "):
        with pytest.raises(RetrievalError):
            retrieve(session, provider, bad)
    assert provider.calls == []


def test_query_embedded_once_and_results_mapped(session: Session, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    rows = [_row(0.95, 0), _row(0.5, 1), _row(0.1, 2)]
    captured: dict = {}

    def fake_search(s, **kw):  # type: ignore[no-untyped-def]
        captured.update(kw)
        assert s is session
        return rows

    monkeypatch.setattr(repository, "search_chunks_by_vector", fake_search)
    provider = FakeEmbeddingProvider()
    results = retrieve(session, provider, "some question")
    assert provider.calls == [1]  # exactly one query embedding
    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.score for r in results] == [0.95, 0.5, 0.1]
    assert results[0].text == "chunk 0"
    assert results[0].document_title == "Doc"
    assert results[0].document_type == "markdown"
    assert results[0].page == 2
    assert results[0].section == "S"
    assert results[0].chunk_metadata == {"k": "v"}
    assert captured["limit"] == DEFAULT_TOP_K
    assert captured["model_name"] == "fake-test-model"


def test_empty_result_set(session: Session, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        repository,
        "search_chunks_by_vector",
        lambda s, **kw: [],  # type: ignore[no-untyped-def]
    )
    assert retrieve(session, FakeEmbeddingProvider(), "anything") == []


def test_filters_forwarded(session: Session, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict = {}

    def fake_search(s, **kw):  # type: ignore[no-untyped-def]
        captured.update(kw)
        return []

    monkeypatch.setattr(repository, "search_chunks_by_vector", fake_search)
    doc_id = uuid.uuid4()
    retrieve(session, FakeEmbeddingProvider(), "q", document_id=doc_id, document_type="pdf")
    assert captured["document_id"] == doc_id
    assert captured["document_type"] == "pdf"


def test_query_dimension_mismatch_rejected(session: Session) -> None:
    provider = FakeEmbeddingProvider(dimension=EMBEDDING_DIMENSION - 1)
    with pytest.raises(RetrievalError):
        retrieve(session, provider, "q")
