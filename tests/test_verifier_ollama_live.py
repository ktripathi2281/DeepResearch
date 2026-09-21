"""M12 live Ollama verifier — qwen3:4b JSON decisions, real runtime only.

Skips (never fails) when Ollama is down or qwen3:4b is missing. Asserts
only that a valid JSON decision is produced and parsed — never that
the verdict is correct. No model substitution, ever.
"""

from __future__ import annotations

import pytest

from deepresearch.citation_verification import (
    CitationVerificationReport,
    verify_answer_citations,
)
from deepresearch.config import get_settings
from deepresearch.llm import DEFAULT_MODEL, TAGS_PATH, OllamaLLMProvider
from tests.test_citation_verification import _evidence


def _ollama_ready(base_url: str, model: str) -> str | None:
    try:
        import httpx

        response = httpx.get(base_url.rstrip("/") + TAGS_PATH, timeout=10)
        models = response.json().get("models", [])
    except Exception:
        return f"Ollama is not running at {base_url}; start Ollama, then run: ollama pull {model}"
    names = [m.get("name", "") for m in models if isinstance(m, dict)]
    if not any(n == model or n.startswith(model + ":") for n in names):
        return f"model {model!r} not installed (have: {names or 'none'}); run: ollama pull {model}"
    return None


def test_live_verifier_returns_parsable_decision() -> None:
    settings = get_settings()
    reason = _ollama_ready(settings.ollama_base_url, settings.ollama_model)
    if reason is not None:
        pytest.skip(reason)
    assert settings.ollama_model == DEFAULT_MODEL == "qwen3:4b"

    provider = OllamaLLMProvider.from_settings(settings)
    try:
        report = verify_answer_citations(
            "Hybrid retrieval combines vector and lexical search [1].",
            [_evidence("Hybrid retrieval combines dense vector search with lexical matching.")],
            provider,
            # Generous bound (thinking alone runs ~2700 tokens on this
            # prompt; M21 live finding): qwen3:4b consumes capped
            # budgets thinking and returns empty text, which the repair
            # path turns into unverifiable instead of a verdict.
            max_tokens=4096,
        )
    finally:
        provider.close()
    assert isinstance(report, CitationVerificationReport)
    assert len(report.results) == 1
    assert report.results[0].status in {"supported", "unsupported", "insufficient_evidence"}
    assert report.results[0].explanation.strip()
    print(f"\n[live-verifier] status={report.results[0].status} model={provider.model_version}")
