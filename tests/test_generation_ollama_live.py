"""M10 live Ollama generation — full grounded pipeline, real LLM only.

Skips (never fails) when Ollama is down or qwen3:4b is missing; the
skip message states the exact setup command. Embeddings/reranker use
the cached local models; only generation requires the daemon. No
model is ever substituted.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest
from sqlalchemy.orm import Session

from deepresearch.config import get_settings
from deepresearch.db import check_connection, get_engine, init_db
from deepresearch.embeddings import LocalEmbeddingProvider, embed_pending_chunks
from deepresearch.generation import answer_question
from deepresearch.ingestion import ingest_file
from deepresearch.llm import DEFAULT_MODEL, TAGS_PATH, OllamaLLMProvider
from deepresearch.reranker import LocalCrossEncoderReranker

FIXTURES = Path(__file__).parent / "fixtures"


def _ollama_ready(base_url: str, model: str) -> str | None:
    """Return a skip reason, or None when the daemon + model are usable."""
    try:
        response = httpx.get(base_url.rstrip("/") + TAGS_PATH, timeout=10)
        models = response.json().get("models", [])
    except Exception:
        return f"Ollama is not running at {base_url}; start Ollama, then run: ollama pull {model}"
    names = [m.get("name", "") for m in models if isinstance(m, dict)]
    if not any(n == model or n.startswith(model + ":") for n in names):
        return f"model {model!r} not installed (have: {names or 'none'}); run: ollama pull {model}"
    return None


def test_live_grounded_generation() -> None:
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
            ingested = ingest_file(session, path=FIXTURES / "sample.md")
            assert ingested.chunks
            content_hash = ingested.document.content_hash
            session.expunge_all()
            embedder = LocalEmbeddingProvider()
            embedded = embed_pending_chunks(session, embedder)
            assert embedded.embedded >= 0
            session.expunge_all()

            started = time.perf_counter()
            result = answer_question(
                session,
                embedder,
                LocalCrossEncoderReranker(device="cpu"),
                provider,
                "What does hybrid retrieval combine?",
                retrieval_top_k=3,
                evidence_top_k=2,
                # No max_tokens cap here: qwen3:4b thinking consumes capped
                # budgets and returns empty text (M21 live finding:
                # eval_count == cap, done_reason == length). Cap plumbing
                # is covered by unit tests; this test proves the grounded
                # pipeline end to end.
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            assert result.has_evidence is True
            assert result.answer.strip()
            assert result.model_name == "qwen3:4b"
            # M13: fixture evidence carries no contradictions → answered.
            assert result.status == "answered"
            assert result.conflicts == []
            print(
                f"\n[live-generation] model={result.model_version} "
                f"latency_ms={elapsed_ms} evidence={len(result.evidence)} "
                f"answer={result.answer.strip()[:120]!r}"
            )

            # M11 extension: extraction must be consistent with the answer.
            # No correctness claim — only that markers map within range and
            # in first-use order.
            from deepresearch.citations import extract_citations

            repeat = extract_citations(result.answer, result.evidence)
            assert [c.citation_id for c in repeat.citations] == [
                c.citation_id for c in result.citations
            ]
            assert all(1 <= c.citation_id <= len(result.evidence) for c in result.citations)
            print(
                f"[live-citations] citations={[c.citation_id for c in result.citations]} "
                f"invalid={[i.citation_id for i in result.invalid_citations]}"
            )
    finally:
        try:
            if content_hash is not None:
                from sqlalchemy import text as sql_text

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
