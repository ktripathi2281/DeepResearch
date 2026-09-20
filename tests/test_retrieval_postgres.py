"""M5 PostgreSQL integration — real pgvector ranking with controlled vectors.

Deterministic fake vectors make expected ranking obvious; the real
bge-small-en-v1.5 model is exercised once in the smoke test (skipped
when uncached and offline). Skipped entirely without a live DB.
"""

from __future__ import annotations

import math
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
from deepresearch.embeddings import (
    DEFAULT_MODEL,
    EMBEDDING_DIMENSION,
    LocalEmbeddingProvider,
    embed_pending_chunks,
)
from deepresearch.ingestion import ingest_bytes
from deepresearch.retrieval import RetrievalError, retrieve

FIXTURES = Path(__file__).parent / "fixtures"
DIM = EMBEDDING_DIMENSION
E1 = [1.0] + [0.0] * (DIM - 1)
E2 = [0.0, 1.0] + [0.0] * (DIM - 2)
EMIX = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)] + [0.0] * (DIM - 2)


class FixedQueryProvider:
    """Returns one fixed vector per input; model identity is configurable."""

    def __init__(self, vector: list[float], model_name: str = "fixed-model") -> None:
        self._vector = vector
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return "fixed-v1"

    @property
    def dimension(self) -> int:
        return DIM

    @property
    def device(self) -> str:
        return "cpu"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


def _pg_engine() -> Engine:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    return engine


def _seed(
    session: Session,
    source: str,
    specs: list[tuple[str, list[float] | None]],
    *,
    document_type: str = "markdown",
    title: str = "Seed doc",
) -> list:
    doc = repository.create_document(
        session,
        title=title,
        source=source,
        content_hash=f"m5-{uuid.uuid4().hex}",
        document_type=document_type,
        metadata=None,
    )
    chunks = []
    for index, (chunk_text, vector) in enumerate(specs):
        chunk = repository.create_chunk(
            session,
            document_id=doc.id,
            text=chunk_text,
            chunk_index=index,
            section="SeedSection" if index == 0 else None,
            page=1 if index == 0 else None,
            metadata={"seed": True},
        )
        if vector is not None:
            chunk.embedding = vector
            chunk.embedding_model = "fixed-model"
            chunk.embedding_version = "fixed-v1"
        chunks.append(chunk)
    session.commit()
    return chunks


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


def test_ranking_order_and_similarity_scores() -> None:
    engine = _pg_engine()
    source = f"pg-m5-rank-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source, [("alpha chunk", E1), ("beta chunk", E2), ("mix chunk", EMIX)])
            session.expunge_all()
            results = retrieve(session, FixedQueryProvider(E1), "query", top_k=3)
            assert [r.text for r in results] == ["alpha chunk", "mix chunk", "beta chunk"]
            assert results[0].score == pytest.approx(1.0)
            assert results[1].score == pytest.approx(1.0 / math.sqrt(2))
            assert results[2].score == pytest.approx(0.0)
            assert all(a.score >= b.score for a, b in zip(results, results[1:], strict=False))
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_top_k_limits_results() -> None:
    engine = _pg_engine()
    source = f"pg-m5-topk-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source, [("a", E1), ("b", E2), ("c", EMIX)])
            session.expunge_all()
            assert len(retrieve(session, FixedQueryProvider(E1), "q", top_k=2)) == 2
            with pytest.raises(RetrievalError):
                retrieve(session, FixedQueryProvider(E1), "q", top_k=101)
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_unembedded_chunks_excluded_and_metadata_returned() -> None:
    engine = _pg_engine()
    source = f"pg-m5-meta-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(
                session,
                source,
                [("embedded one", E1), ("no vector here", None)],
                title="Meta Doc",
            )
            session.expunge_all()
            results = retrieve(session, FixedQueryProvider(E1), "q", top_k=10)
            assert [r.text for r in results] == ["embedded one"]
            hit = results[0]
            assert hit.document_title == "Meta Doc"
            assert hit.document_type == "markdown"
            assert hit.document_source == source
            assert hit.page == 1
            assert hit.section == "SeedSection"
            assert hit.chunk_metadata == {"seed": True}
            assert hit.retrieval_method == "vector"
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_empty_corpus_returns_empty_list() -> None:
    engine = _pg_engine()
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            # A filter matching nothing must yield [], never an error.
            assert retrieve(session, FixedQueryProvider(E1), "q", document_id=uuid.uuid4()) == []
    finally:
        engine.dispose()


def test_model_version_mismatch_excluded() -> None:
    engine = _pg_engine()
    source = f"pg-m5-model-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source, [("foreign vector", E1)])
            session.expunge_all()
            assert retrieve(session, FixedQueryProvider(E1, model_name="other"), "q") == []
            assert len(retrieve(session, FixedQueryProvider(E1), "q")) == 1
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_tied_scores_rank_deterministically() -> None:
    engine = _pg_engine()
    source = f"pg-m5-tie-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source, [("tie one", E1), ("tie two", E1)])
            session.expunge_all()
            first = retrieve(session, FixedQueryProvider(E1), "q", top_k=2)
            session.expunge_all()
            second = retrieve(session, FixedQueryProvider(E1), "q", top_k=2)
            assert len(first) == 2
            assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
            assert [r.rank for r in first] == [1, 2]
            assert first[0].score == pytest.approx(first[1].score)
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_document_type_filter() -> None:
    engine = _pg_engine()
    prefix = f"pg-m5f-{uuid.uuid4().hex[:8]}"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, f"{prefix}-a.md", [("md chunk", E1)], document_type="markdown")
            _seed(session, f"{prefix}-b.txt", [("txt chunk", E1)], document_type="txt")
            session.expunge_all()
            results = retrieve(session, FixedQueryProvider(E1), "q", top_k=10, document_type="txt")
            assert [r.text for r in results] == ["txt chunk"]
    finally:
        try:
            _cleanup(engine, f"{prefix}%")
        finally:
            engine.dispose()


def _real_model_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

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


def test_real_model_smoke_semantic_match() -> None:
    if not (_real_model_cached() or _network_up()):
        pytest.skip(f"{DEFAULT_MODEL} not cached and no network")
    engine = _pg_engine()
    source = f"pg-m5real-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        provider = LocalEmbeddingProvider()
        with Session(bind=engine) as session:
            ingested = ingest_bytes(
                session,
                raw=(FIXTURES / "sample.md").read_bytes(),
                document_type="markdown",
                source=source,
            )
            doc_id = ingested.document.id
            session.expunge_all()
            embedded = embed_pending_chunks(session, provider)
            assert embedded.embedded >= 1
            session.expunge_all()
            results = retrieve(
                session, provider, "How does hybrid retrieval combine search methods?"
            )
            assert len(results) >= 1
            assert all(r.document_id == doc_id for r in results)
            assert all(a.score >= b.score for a, b in zip(results, results[1:], strict=False))
            assert results[0].score > 0.0
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()
