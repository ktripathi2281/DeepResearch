"""M7 PostgreSQL integration — hybrid fusion over real M5 + M6 paths.

Controlled vectors (query E1) plus lexical overlap make fusion
deterministic and assertable. No model download, no network.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.bm25 import StaleIndexError, build_index_from_session, retrieve_bm25
from deepresearch.config import get_settings
from deepresearch.db import check_connection, init_db
from deepresearch.hybrid import retrieve_hybrid
from tests.test_retrieval_postgres import E1, E2, EMIX, FixedQueryProvider


def _pg_engine() -> Engine:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    return engine


def _seed(session: Session, source: str, specs: list[tuple[str, list[float]]]) -> None:
    doc = repository.create_document(
        session,
        title="Hybrid Doc",
        source=source,
        content_hash=f"m7-{uuid.uuid4().hex}",
        document_type="markdown",
        metadata=None,
    )
    for index, (chunk_text, vector) in enumerate(specs):
        chunk = repository.create_chunk(
            session, document_id=doc.id, text=chunk_text, chunk_index=index
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


def _corpus() -> list[tuple[str, list[float]]]:
    return [
        ("alpha retrieval", E1),  # both retrievers
        ("beta retrieval", E2),  # BM25 hit, vector rank 3
        ("gamma unrelated", EMIX),  # vector-only
    ]


def test_fusion_order_dedup_scores() -> None:
    engine = _pg_engine()
    source = f"pg-m7-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source, _corpus())
            session.expunge_all()
            provider = FixedQueryProvider(E1)
            bm25_ranks = {r.text: r.rank for r in retrieve_bm25(session, "retrieval", top_k=3)}
            session.expunge_all()
            results = retrieve_hybrid(session, provider, "retrieval", top_k=3)
            assert [r.text for r in results] == [
                "alpha retrieval",
                "beta retrieval",
                "gamma unrelated",
            ]
            assert len({r.chunk_id for r in results}) == 3  # deduplicated
            expected_alpha = 1 / 61 + 1 / (60 + bm25_ranks["alpha retrieval"])
            expected_beta = 1 / 63 + 1 / (60 + bm25_ranks["beta retrieval"])
            assert results[0].score == pytest.approx(expected_alpha)
            assert results[1].score == pytest.approx(expected_beta)
            assert results[2].score == pytest.approx(1 / 62)  # vector-only contribution
            assert [r.rank for r in results] == [1, 2, 3]
            assert all(r.retrieval_method == "hybrid" for r in results)
            assert results[0].document_title == "Hybrid Doc"
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_candidate_pools_and_filters() -> None:
    engine = _pg_engine()
    prefix = f"pg-m7f-{uuid.uuid4().hex[:8]}"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, f"{prefix}-a.md", [("alpha retrieval", E1)])
            session.expunge_all()
            provider = FixedQueryProvider(E1)
            narrow = retrieve_hybrid(
                session, provider, "retrieval", top_k=5, vector_top_k=1, bm25_top_k=1
            )
            assert len(narrow) == 1
            session.expunge_all()
            doc_id = repository.list_chunks_for_bm25(session)[0][0].document_id
            session.expunge_all()
            scoped = retrieve_hybrid(session, provider, "retrieval", document_id=doc_id)
            assert scoped
            assert all(r.document_id == doc_id for r in scoped)
            session.expunge_all()
            assert retrieve_hybrid(session, provider, "retrieval", document_id=uuid.uuid4()) == []
    finally:
        try:
            _cleanup(engine, f"{prefix}%")
        finally:
            engine.dispose()


def test_deterministic_across_runs_and_stale_index() -> None:
    engine = _pg_engine()
    source = f"pg-m7d-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source, _corpus())
            session.expunge_all()
            provider = FixedQueryProvider(E1)
            first = [(r.chunk_id, r.score) for r in retrieve_hybrid(session, provider, "retrieval")]
            session.expunge_all()
            second = [
                (r.chunk_id, r.score) for r in retrieve_hybrid(session, provider, "retrieval")
            ]
            assert first == second
            session.expunge_all()

            stale, _ = build_index_from_session(session)
            _seed(session, source, [("delta retrieval newcomer", E1)])
            session.expunge_all()
            with pytest.raises(StaleIndexError):
                retrieve_hybrid(session, provider, "retrieval", index=stale)
            # Without a supplied index the hybrid path rebuilds and finds it.
            assert "delta retrieval newcomer" in [
                r.text for r in retrieve_hybrid(session, provider, "newcomer")
            ]
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()
