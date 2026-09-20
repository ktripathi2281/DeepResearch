"""M8 PostgreSQL integration — hybrid retrieval → fake rerank → finals.

Uses the real M7 hybrid path with controlled vectors; only the
cross-encoder is faked (deterministic scores by text). No GPU.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.config import get_settings
from deepresearch.db import check_connection, init_db
from deepresearch.hybrid import retrieve_hybrid
from deepresearch.reranker import rerank_results
from tests.fakes import FakeReranker
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
        title="Rerank Doc",
        source=source,
        content_hash=f"m8-{uuid.uuid4().hex}",
        document_type="markdown",
        metadata=None,
    )
    for index, (chunk_text, vector) in enumerate(
        [("alpha retrieval", E1), ("beta retrieval", E2), ("gamma retrieval", E1)]
    ):
        chunk = repository.create_chunk(
            session,
            document_id=doc.id,
            text=chunk_text,
            chunk_index=index,
            metadata={"seed": index},
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


def test_hybrid_then_rerank_pipeline() -> None:
    engine = _pg_engine()
    source = f"pg-m8-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source)
            session.expunge_all()

            hybrid = retrieve_hybrid(session, FixedQueryProvider(E1), "retrieval", top_k=3)
            assert len(hybrid) == 3
            assert all(r.retrieval_method == "hybrid" for r in hybrid)
            session.expunge_all()

            # Fake reranker reverses the hybrid order on purpose (higher rank → higher score).
            scores = {r.text: float(r.rank) for r in hybrid}
            finals = rerank_results("retrieval", hybrid, FakeReranker(scores=scores), top_k=2)
            assert [r.text for r in finals] == [r.text for r in reversed(hybrid)][:2]
            assert [r.rank for r in finals] == [1, 2]
            assert all(r.retrieval_method == "reranked" for r in finals)
            assert finals[0].document_title == "Rerank Doc"
            assert finals[0].chunk_metadata == hybrid[-1].chunk_metadata
            session.expunge_all()

            # candidate_top_k restricts scoring to the first N hybrids.
            partial = rerank_results(
                "retrieval", hybrid, FakeReranker(), candidate_top_k=1, top_k=5
            )
            assert len(partial) == 1
            assert partial[0].chunk_id == hybrid[0].chunk_id
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()
