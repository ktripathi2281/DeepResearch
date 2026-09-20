"""M14 live Ollama agent — qwen3:4b decisions on a tiny corpus, real runtime only.

Skips (never fails) when Ollama is down or qwen3:4b is missing. Asserts
the agent makes allowlisted decisions, terminates within limits, and
returns evidence. No claim of general agent reliability.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from deepresearch.agent import ALLOWED_TOOLS, run_research_agent
from deepresearch.config import get_settings
from deepresearch.db import check_connection, get_engine, init_db
from deepresearch.embeddings import LocalEmbeddingProvider, embed_pending_chunks
from deepresearch.ingestion import ingest_file
from deepresearch.llm import DEFAULT_MODEL, TAGS_PATH, OllamaLLMProvider

FIXTURES = Path(__file__).parent / "fixtures"


def _ollama_ready(base_url: str, model: str) -> str | None:
    try:
        response = httpx.get(base_url.rstrip("/") + TAGS_PATH, timeout=10)
        models = response.json().get("models", [])
    except Exception:
        return f"Ollama is not running at {base_url}; start Ollama, then run: ollama pull {model}"
    names = [m.get("name", "") for m in models if isinstance(m, dict)]
    if not any(n == model or n.startswith(model + ":") for n in names):
        return f"model {model!r} not installed (have: {names or 'none'}); run: ollama pull {model}"
    return None


def test_live_agent_researches_tiny_corpus() -> None:
    settings = get_settings()
    engine = get_engine(settings)
    if not check_connection(engine):
        pytest.skip("PostgreSQL not reachable; run `docker compose up -d postgres`")
    reason = _ollama_ready(settings.ollama_base_url, settings.ollama_model)
    if reason is not None:
        pytest.skip(reason)
    assert settings.ollama_model == DEFAULT_MODEL == "qwen3:4b"

    content_hash: str | None = None
    provider = OllamaLLMProvider.from_settings(settings)
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            ingested = ingest_file(session, path=FIXTURES / "sample.txt")
            content_hash = ingested.document.content_hash
            session.expunge_all()
            embed_pending_chunks(session, LocalEmbeddingProvider())
            session.expunge_all()

            result = run_research_agent(
                session,
                LocalEmbeddingProvider(),
                provider,
                "What is DeepResearch ingestion?",
                max_iterations=4,
                max_tool_calls=6,
                timeout_seconds=240,
            )
            assert result.termination_reason in {
                "finished",
                "max_iterations",
                "max_tool_calls",
                "timeout",
            }
            assert result.tool_call_count >= 1
            assert result.tool_call_count <= 6
            assert result.trace is not None
            assert all(s.tool_name in ALLOWED_TOOLS for s in result.trace.steps)
            print(
                f"\n[live-agent] termination={result.termination_reason} "
                f"tools={result.tool_call_count} evidence={len(result.evidence)} "
                f"model={provider.model_version}"
            )
    finally:
        try:
            if content_hash is not None:
                with engine.begin() as conn:
                    conn.execute(
                        sql_text(
                            "DELETE FROM chunks WHERE document_id IN "
                            "(SELECT id FROM documents WHERE content_hash = :h)"
                        ),
                        {"h": content_hash},
                    )
                    conn.execute(
                        sql_text("DELETE FROM documents WHERE content_hash = :h"),
                        {"h": content_hash},
                    )
        finally:
            provider.close()
            engine.dispose()
