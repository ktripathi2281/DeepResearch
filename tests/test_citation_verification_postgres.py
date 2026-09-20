"""M12 PostgreSQL integration — provenance chain through verification.

Real hybrid retrieval with controlled vectors; reranker, answer LLM,
and verifier LLM are deterministic fakes. Asserts evidence identity
survives retrieval → rerank → generation → citations → verification,
and that verification defaults off. No Ollama, no network.
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
from deepresearch.generation import answer_question
from tests.fakes import FakeLLMProvider, FakeReranker, FakeVerifierLLM
from tests.test_citation_verification import SUPPORTED_JSON, UNSUPPORTED_JSON
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
        title="Verify Doc",
        source=source,
        content_hash=f"m12-{uuid.uuid4().hex}",
        document_type="markdown",
        metadata=None,
    )
    for index, (chunk_text, vector) in enumerate(
        [("alpha retrieval fact", E1), ("beta retrieval fact", E2)]
    ):
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


def test_verified_pipeline_preserves_provenance() -> None:
    engine = _pg_engine()
    source = f"pg-m12-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source)
            session.expunge_all()
            llm = FakeLLMProvider(answer="Alpha holds [1]. Beta fails [2].")
            verifier = FakeVerifierLLM(responses=[SUPPORTED_JSON, UNSUPPORTED_JSON])
            result = answer_question(
                session,
                FixedQueryProvider(E1),
                FakeReranker(scores={"alpha retrieval fact": 2.0, "beta retrieval fact": 1.0}),
                llm,
                "What holds?",
                retrieval_top_k=2,
                evidence_top_k=2,
                verify_citations=True,
                verifier=verifier,
            )
            assert result.has_evidence is True
            assert [c.citation_id for c in result.citations] == [1, 2]
            report = result.verification_report
            assert report is not None
            assert [(r.claim_id, r.citation_id, r.status) for r in report.results] == [
                (1, 1, "supported"),
                (2, 2, "unsupported"),
            ]
            # Verified evidence is the persisted chunk row.
            first_evidence = report.results[0].evidence
            assert first_evidence is not None
            row = repository.get_chunk_by_id(session, first_evidence.chunk_id)
            assert row is not None and row.text == "alpha retrieval fact"
            assert report.counts()["supported"] == 1
            assert len(verifier.calls) == 2  # one call per cited claim
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_verification_defaults_off_and_no_evidence_empty() -> None:
    engine = _pg_engine()
    source = f"pg-m12o-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source)
            session.expunge_all()
            verifier = FakeVerifierLLM(responses=[SUPPORTED_JSON])
            plain = answer_question(
                session,
                FixedQueryProvider(E1),
                FakeReranker(),
                FakeLLMProvider(answer="Alpha [1]."),
                "Q?",
                retrieval_top_k=2,
                evidence_top_k=2,
            )
            assert plain.verification_report is None
            assert verifier.calls == []

            session.expunge_all()
            empty = answer_question(
                session,
                FixedQueryProvider(E1),
                FakeReranker(),
                FakeLLMProvider(),
                "Q?",
                document_id=uuid.uuid4(),
                verify_citations=True,
                verifier=verifier,
            )
            assert empty.has_evidence is False
            assert empty.verification_report is not None
            assert empty.verification_report.results == []
            assert verifier.calls == []
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()
