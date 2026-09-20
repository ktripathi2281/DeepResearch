"""M14 PostgreSQL integration — real tools, bounded loop, pipeline handoff.

Hybrid search runs for real with controlled vectors; decisions come
from the scripted fake LLM. No Ollama, no network.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.agent import (
    get_chunk,
    get_document,
    run_research_agent,
    search_documents,
)
from deepresearch.config import get_settings
from deepresearch.db import check_connection, init_db
from deepresearch.generation import build_grounded_prompt
from deepresearch.reranker import rerank_results
from tests.fakes import FakeAgentLLM, FakeReranker
from tests.test_retrieval_postgres import E1, E2, FixedQueryProvider


def _pg_engine() -> Engine:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    return engine


def _seed(session: Session, source: str) -> tuple:
    doc = repository.create_document(
        session,
        title="Agent PG Doc",
        source=source,
        content_hash=f"m14-{uuid.uuid4().hex}",
        document_type="markdown",
        metadata=None,
    )
    chunks = []
    for index, (chunk_text, vector) in enumerate(
        [("alpha agent evidence", E1), ("beta agent evidence", E2)]
    ):
        chunk = repository.create_chunk(
            session, document_id=doc.id, text=chunk_text, chunk_index=index
        )
        chunk.embedding = vector
        chunk.embedding_model = "fixed-model"
        chunk.embedding_version = "fixed-v1"
        chunks.append(chunk)
    session.commit()
    return doc, chunks


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
    return json.dumps({"action": action, "arguments": arguments or {}, "reason": "pg step"})


def test_real_tools_against_postgres() -> None:
    engine = _pg_engine()
    source = f"pg-m14-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            doc, chunks = _seed(session, source)
            first_id, doc_id = chunks[0].id, doc.id
            session.expunge_all()

            hits = search_documents(session, FixedQueryProvider(E1), "agent evidence", top_k=5)
            assert {h.text for h in hits} == {"alpha agent evidence", "beta agent evidence"}

            detail = get_chunk(session, first_id)
            assert detail.text == "alpha agent evidence"
            assert detail.document_title == "Agent PG Doc"

            doc_detail = get_document(session, doc_id)
            assert doc_detail.title == "Agent PG Doc"
            assert len(doc_detail.chunk_refs) == 2
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_bounded_loop_with_provenance() -> None:
    engine = _pg_engine()
    source = f"pg-m14b-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            doc, chunks = _seed(session, source)
            first_id, doc_id = chunks[0].id, doc.id
            session.expunge_all()

            llm = FakeAgentLLM(
                decisions=[
                    _decision("search_documents", {"query": "agent evidence"}),
                    _decision("get_chunk", {"chunk_id": str(first_id)}),
                    _decision("get_document", {"document_id": str(doc_id)}),
                    _decision("finish"),
                ]
            )
            result = run_research_agent(session, FixedQueryProvider(E1), llm, "Agent evidence?")
            assert result.termination_reason == "finished"
            assert result.iteration_count == 4
            assert result.tool_call_count == 3
            assert {e.chunk_id for e in result.evidence} >= {first_id}
            assert result.trace is not None and len(result.trace.steps) == 3
            assert all(s.success for s in result.trace.steps)
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()


def test_evidence_handoff_to_generation_pipeline() -> None:
    engine = _pg_engine()
    source = f"pg-m14h-{uuid.uuid4().hex[:8]}.md"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, source)
            session.expunge_all()

            llm = FakeAgentLLM(
                decisions=[
                    _decision("search_documents", {"query": "agent evidence"}),
                    _decision("finish"),
                ]
            )
            result = run_research_agent(session, FixedQueryProvider(E1), llm, "Q?")
            assert result.evidence
            reranked = rerank_results(
                "Q?", result.evidence, FakeReranker(), candidate_top_k=10, top_k=2
            )
            prompt = build_grounded_prompt("Q?", reranked)
            assert "agent evidence" in prompt.user
    finally:
        try:
            _cleanup(engine, source)
        finally:
            engine.dispose()
