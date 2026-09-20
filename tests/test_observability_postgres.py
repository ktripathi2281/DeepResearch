"""M15 PostgreSQL integration — traced pipeline counters/stages/models.

Real database with controlled vectors; reranker/LLM/verifier are
deterministic fakes. No Ollama, no network.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import deepresearch.repository as repository_mod
from deepresearch import repository
from deepresearch.agent import run_research_agent
from deepresearch.config import get_settings
from deepresearch.db import check_connection, init_db
from deepresearch.generation import answer_question
from deepresearch.observability import get_current_trace, traced_request
from deepresearch.retrieval import RetrievalError
from tests.fakes import FakeAgentLLM, FakeLLMProvider, FakeReranker, FakeVerifierLLM
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
        title="Observed Doc",
        source=source,
        content_hash=f"m15-{uuid.uuid4().hex}",
        document_type="markdown",
        metadata=None,
    )
    for index, (chunk_text, vector) in enumerate(
        [("alpha observed fact", E1), ("beta observed fact", E2)]
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


def _decision(action: str, arguments: dict | None = None) -> str:
    import json as json_mod

    return json_mod.dumps({"action": action, "arguments": arguments or {}, "reason": "pg step"})


def test_traced_answer_pipeline() -> None:
    engine = _pg_engine()
    source = f"pg-m15-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source)
            session.expunge_all()
            with traced_request("pg-req-1") as trace:
                result = answer_question(
                    session,
                    FixedQueryProvider(E1),
                    FakeReranker(),
                    FakeLLMProvider(answer="Alpha fact [1]."),
                    "What fact?",
                    retrieval_top_k=2,
                    evidence_top_k=2,
                    verify_citations=True,
                    verifier=FakeVerifierLLM(
                        responses=['{"verdict": "supported", "explanation": "Stated."}']
                    ),
                )
            assert result.status == "answered"
            stages = {s.stage for s in trace.stages}
            assert {
                "embedding",
                "vector_retrieval",
                "bm25_retrieval",
                "hybrid_fusion",
                "reranking",
                "conflict_detection",
                "citation_extraction",
                "generation",
                "citation_verification",
            } <= stages
            assert all(s.success for s in trace.stages)
            assert trace.counters["retrieval_hybrid_candidates"] >= 1
            assert trace.counters["reranker_results"] == 2
            assert trace.counters["evidence_chunks"] == 2
            assert trace.counters["llm_calls"] == 2  # answer + one verification call
            assert trace.counters["citation_count"] == 1
            assert trace.counters["verification_calls"] == 1
            assert trace.counters["verification_supported"] == 1
            assert trace.models["llm"] == "fake-llm"
            assert trace.models["verifier"] == "fake-verifier"
            assert trace.models["embedding"] == "fixed-model"
            assert trace.models["reranker"] == "fake-reranker"
            assert trace.request_id == "pg-req-1"
            payload = json.dumps(trace.to_dict())  # serializable
            assert "alpha observed fact" not in payload  # no document content
            assert get_current_trace() is None  # context reset
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_failed_stage_recorded_and_propagated(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    engine = _pg_engine()
    try:
        init_db(engine)
        with Session(bind=engine) as session:

            def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("vector store down")

            monkeypatch.setattr(repository_mod, "search_chunks_by_vector", boom)
            with traced_request("pg-req-fail") as trace:
                with pytest.raises(RetrievalError):
                    answer_question(
                        session,
                        FixedQueryProvider(E1),
                        FakeReranker(),
                        FakeLLMProvider(),
                        "Q?",
                    )
            failed = [s for s in trace.stages if s.stage == "vector_retrieval"]
            assert len(failed) == 1
            assert failed[0].success is False
            assert failed[0].error_type == "RuntimeError"
    finally:
        engine.dispose()


def test_traced_agent_execution() -> None:
    engine = _pg_engine()
    source = f"pg-m15a-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source)
            session.expunge_all()
            llm = FakeAgentLLM(
                decisions=[
                    _decision("search_documents", {"query": "observed fact"}),
                    _decision("finish"),
                ]
            )
            with traced_request("pg-req-agent") as trace:
                result = run_research_agent(session, FixedQueryProvider(E1), llm, "Q?")
            assert result.termination_reason == "finished"
            assert trace.counters["agent_iterations"] == 2
            assert trace.counters["agent_tool_calls"] == 1
            assert trace.counters["llm_calls"] == 2  # two decision generations
            tool_stages = [s for s in trace.stages if s.stage == "agent.tool.search_documents"]
            assert len(tool_stages) == 1 and tool_stages[0].success is True
            assert any(s.stage == "agent_execution" and s.success for s in trace.stages)
            assert trace.attributes["agent_termination"] == "finished"
            assert trace.models["agent"] == "fake-agent"
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()
