"""M18 PostgreSQL integration — full job lifecycle against real pgvector.

The research service runs against a real Postgres database with fake
embedding/reranker/LLM providers (as in other *postgres tests): the
pipeline path and the public serialization are exercised end to end.
No Ollama, no network.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.config import get_settings
from deepresearch.db import check_connection, get_session_factory, init_db
from deepresearch.generation import answer_question
from deepresearch.research_api import ResearchService
from tests.fakes import FakeLLMProvider, FakeReranker, FakeVerifierLLM
from tests.test_answer_status import SUPPORTED_JSON
from tests.test_retrieval_postgres import E1, E2, FixedQueryProvider


def _pg_factory() -> tuple[Engine, Callable[[], Session]]:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    init_db(engine)
    return engine, get_session_factory(engine)


def _seed(session: Session, source: str, specs: list[tuple[str, list[float]]]) -> None:
    doc = repository.create_document(
        session,
        title="API Seed",
        source=source,
        content_hash=f"m18-{uuid.uuid4().hex}",
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


def _service(factory: Callable[[], Session], researcher) -> ResearchService:  # type: ignore[no-untyped-def]
    return ResearchService(session_factory=factory, researcher=researcher, run_async=False)


def test_job_completes_with_answered_status_and_safe_shape() -> None:
    engine, factory = _pg_factory()
    source = f"pg-m18a-{uuid.uuid4().hex[:8]}.md"
    try:
        with Session(bind=engine) as session:
            _seed(session, source, [("alpha retrieval note", E1), ("beta retrieval note", E2)])
            session.expunge_all()

        def researcher(session: Session, question: str, request_id: str) -> object:
            return answer_question(
                session,
                FixedQueryProvider(E1),
                FakeReranker(scores={"alpha retrieval note": 2.0}),
                FakeLLMProvider(answer="The alpha note is the answer [1]."),
                question,
                retrieval_top_k=3,
                evidence_top_k=3,
                verify_citations=True,
                verifier=FakeVerifierLLM(responses=[SUPPORTED_JSON] * 3),
                request_id=request_id,
            )

        svc = _service(factory, researcher)
        snapshot = svc.submit("What is in the alpha note?", "pg-job-1")
        assert snapshot.job_status == "completed"
        result = snapshot.result
        assert result is not None
        assert result.status == "answered"
        assert result.request_id == "pg-job-1"
        assert result.answer.startswith("The alpha note")
        assert result.evidence and result.evidence[0].citation_id == 1
        assert result.citations[0].citation_id == 1
        assert result.verification is not None
        assert result.verification.counts["supported"] >= 1
        details = result.research_details
        assert details.request_id == "pg-job-1"
        assert details.evidence_count >= 1
        assert details.elapsed_ms >= 0
        stage_names = [stage.name for stage in details.stages]
        assert "embedding" in stage_names
        assert "generation" in stage_names
        # Safe shape: no internal keys exposed.
        raw = snapshot.model_dump_json()
        assert "system_prompt" not in raw
        assert "chain" not in raw
        assert "Traceback" not in raw
    finally:
        _cleanup(engine, "pg-m18a-%")
        engine.dispose()


def test_no_evidence_job_short_circuits() -> None:
    engine, factory = _pg_factory()

    def researcher(session: Session, question: str, request_id: str) -> object:
        return answer_question(
            session,
            FixedQueryProvider(E1),
            FakeReranker(),
            FakeLLMProvider(),
            question,
            retrieval_top_k=3,
            evidence_top_k=3,
            document_id=uuid.uuid4(),
            verify_citations=True,
            verifier=FakeVerifierLLM(responses=[]),
            request_id=request_id,
        )

    svc = _service(factory, researcher)
    try:
        snapshot = svc.submit("Nothing here at all?", "pg-job-2")
        result = snapshot.result
        assert result is not None
        assert result.status == "no_evidence"
        assert result.answer  # fixed abstention message
        assert result.evidence == []
        assert result.citations == []
        assert result.conflicts == []
    finally:
        engine.dispose()


def test_conflicting_evidence_surfaces_conflicts() -> None:
    engine, factory = _pg_factory()
    source = f"pg-m18b-{uuid.uuid4().hex[:8]}.md"
    try:
        with Session(bind=engine) as session:
            _seed(
                session,
                source,
                [
                    ("The plant was introduced in 2022 AD.", E1),
                    ("The plant was introduced in 2024 AD.", E2),
                ],
            )
            session.expunge_all()

        def researcher(session: Session, question: str, request_id: str) -> object:
            return answer_question(
                session,
                FixedQueryProvider(E1),
                FakeReranker(
                    scores={
                        "The plant was introduced in 2022 AD.": 2.0,
                        "The plant was introduced in 2024 AD.": 1.0,
                    }
                ),
                FakeLLMProvider(answer="Some sources say 2022 [1], others say 2024 [2]."),
                question,
                retrieval_top_k=3,
                evidence_top_k=3,
                verify_citations=False,
                request_id=request_id,
            )

        svc = _service(factory, researcher)
        snapshot = svc.submit("When was the plant introduced?", "pg-job-3")
        result = snapshot.result
        assert result is not None
        assert result.status == "conflicting_evidence"
        assert result.conflicts
        assert result.conflicts[0].citation_ids
        assert len(result.evidence) >= 2  # both sides preserved
    finally:
        _cleanup(engine, "pg-m18b-%")
        engine.dispose()
