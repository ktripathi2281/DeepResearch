"""M13 PostgreSQL integration — none/consistent/conflicting evidence statuses.

Real hybrid retrieval with controlled vectors; reranker, answer LLM,
and verifier LLM are deterministic fakes. No Ollama, no network.
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
from deepresearch.generation import NO_EVIDENCE_MESSAGE, answer_question
from tests.fakes import FakeLLMProvider, FakeReranker, FakeVerifierLLM
from tests.test_answer_status import SUPPORTED_JSON, UNSUPPORTED_JSON
from tests.test_retrieval_postgres import E1, E2, FixedQueryProvider


def _pg_engine() -> Engine:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    return engine


def _seed(session: Session, source: str, specs: list[tuple[str, list[float]]]) -> None:
    doc = repository.create_document(
        session,
        title="Status Doc",
        source=source,
        content_hash=f"m13-{uuid.uuid4().hex}",
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


def _answer(session: Session, llm: FakeLLMProvider, reranker_scores: dict | None = None, **kwargs):  # type: ignore[no-untyped-def]
    return answer_question(
        session,
        FixedQueryProvider(E1),
        FakeReranker(scores=reranker_scores or {}),
        llm,
        "Q?",
        retrieval_top_k=3,
        evidence_top_k=3,
        **kwargs,
    )


def test_no_matching_documents_status() -> None:
    engine = _pg_engine()
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            llm, verifier = FakeLLMProvider(), FakeVerifierLLM(responses=[])
            result = _answer(
                session, llm, document_id=uuid.uuid4(), verify_citations=True, verifier=verifier
            )
            assert result.status == "no_evidence"
            assert result.answer == NO_EVIDENCE_MESSAGE
            assert result.evidence == [] and result.citations == []
            assert result.verification_report is not None
            assert result.verification_report.results == []
            assert llm.calls == [] and verifier.calls == []
    finally:
        engine.dispose()


def test_consistent_evidence_answered_with_provenance() -> None:
    engine = _pg_engine()
    source = f"pg-m13c-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(
                session,
                source,
                [("alpha retrieval note", E1), ("beta retrieval note", E2)],
            )
            session.expunge_all()
            llm = FakeLLMProvider(answer="Alpha note [1].")
            verifier = FakeVerifierLLM(responses=[SUPPORTED_JSON])
            result = _answer(
                session,
                llm,
                {"alpha retrieval note": 2.0, "beta retrieval note": 1.0},
                verify_citations=True,
                verifier=verifier,
            )
            assert result.status == "answered"
            assert result.has_evidence is True
            assert result.conflicts == []
            assert [c.citation_id for c in result.citations] == [1]
            assert result.verification_report is not None
            assert result.verification_report.counts()["supported"] == 1
            row = repository.get_chunk_by_id(session, result.citations[0].chunk_id)
            assert row is not None and row.text == "alpha retrieval note"
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_conflicting_evidence_status_and_preservation() -> None:
    engine = _pg_engine()
    source = f"pg-m13x-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(
                session,
                source,
                [
                    ("The policy was introduced in 2022.", E1),
                    ("The policy was introduced in 2024.", E2),
                ],
            )
            session.expunge_all()
            llm = FakeLLMProvider(answer="Introduced in 2022 [1].")
            verifier = FakeVerifierLLM(responses=[SUPPORTED_JSON])
            result = _answer(
                session,
                llm,
                {
                    "The policy was introduced in 2022.": 2.0,
                    "The policy was introduced in 2024.": 1.0,
                },
                verify_citations=True,
                verifier=verifier,
            )
            assert result.status == "conflicting_evidence"
            assert len(result.conflicts) == 1
            conflict = result.conflicts[0]
            assert conflict.values == ["2022", "2024"]
            assert len(result.evidence) == 2  # both sources preserved
            assert "disagreement" in llm.calls[0]["system_prompt"]
            # Verification still ran and is preserved alongside the status.
            assert result.verification_report is not None
            assert result.verification_report.counts()["supported"] == 1
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_unsupported_claim_becomes_insufficient() -> None:
    engine = _pg_engine()
    source = f"pg-m13u-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source, [("alpha retrieval note", E1)])
            session.expunge_all()
            llm = FakeLLMProvider(answer="Alpha note [1].")
            result = _answer(
                session,
                llm,
                verify_citations=True,
                verifier=FakeVerifierLLM(responses=[UNSUPPORTED_JSON]),
            )
            assert result.status == "insufficient_evidence"
            assert result.citations  # citations kept; status carries the judgment
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()
