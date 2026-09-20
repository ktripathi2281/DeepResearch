"""M11 PostgreSQL integration — retrieval → generation → citation mapping.

Real hybrid retrieval with controlled vectors; reranker and LLM are
deterministic fakes. Verifies citations resolve to the correct chunk,
document, page, and section rows. No Ollama, no network.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.citations import InvalidCitationReference
from deepresearch.config import get_settings
from deepresearch.db import check_connection, init_db
from deepresearch.generation import answer_question
from tests.fakes import FakeLLMProvider, FakeReranker
from tests.test_retrieval_postgres import E1, E2, FixedQueryProvider


def _pg_engine() -> Engine:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    return engine


def _seed(session: Session, source: str) -> None:
    doc = repository.create_document(
        session,
        title="Citation Doc",
        source=source,
        content_hash=f"m11-{uuid.uuid4().hex}",
        document_type="markdown",
        metadata=None,
    )
    for index, (chunk_text, vector) in enumerate(
        [("alpha retrieval fact", E1), ("beta retrieval fact", E2)]
    ):
        chunk = repository.create_chunk(
            session,
            document_id=doc.id,
            text=chunk_text,
            chunk_index=index,
            section="CiteSec" if index == 0 else None,
            page=4 if index == 0 else None,
            metadata=None,
        )
        chunk.embedding = vector
        chunk.embedding_model = "fixed-model"
        chunk.embedding_version = "fixed-v1"
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


def test_citations_map_to_persisted_rows() -> None:
    engine = _pg_engine()
    source = f"pg-m11-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source)
            session.expunge_all()
            llm = FakeLLMProvider(answer="Alpha holds [1]. Beta holds [2]. Alpha again [1].")
            reranker = FakeReranker(
                scores={"alpha retrieval fact": 2.0, "beta retrieval fact": 1.0}
            )
            result = answer_question(
                session,
                FixedQueryProvider(E1),
                reranker,
                llm,
                "What holds?",
                retrieval_top_k=2,
                evidence_top_k=2,
            )
            assert result.has_evidence is True
            assert [c.citation_id for c in result.citations] == [1, 2]
            assert result.invalid_citations == []
            by_number = {c.citation_id: c for c in result.citations}
            assert by_number[1].document_title == "Citation Doc"
            assert by_number[1].document_source == source
            assert by_number[1].document_type == "markdown"
            assert by_number[1].page == 4
            assert by_number[1].section == "CiteSec"
            assert by_number[2].page is None and by_number[2].section is None
            # Chunk IDs resolve to the persisted rows with matching text.
            for citation in result.citations:
                row = repository.get_chunk_by_id(session, citation.chunk_id)
                assert row is not None
                assert row.document_id == citation.document_id
            texts = {
                c.citation_id: repository.get_chunk_by_id(session, c.chunk_id).text
                for c in result.citations
            }
            assert texts[1] == "alpha retrieval fact"
            assert texts[2] == "beta retrieval fact"
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_invalid_markers_recorded_not_remapped() -> None:
    engine = _pg_engine()
    source = f"pg-m11i-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source)
            session.expunge_all()
            llm = FakeLLMProvider(answer="Real [1], invented [5].")
            result = answer_question(
                session,
                FixedQueryProvider(E1),
                FakeReranker(),
                llm,
                "Q?",
                retrieval_top_k=2,
                evidence_top_k=2,
            )
            assert [c.citation_id for c in result.citations] == [1]
            assert result.invalid_citations == [InvalidCitationReference(citation_id=5)]
            assert "[5]" in result.answer
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()
