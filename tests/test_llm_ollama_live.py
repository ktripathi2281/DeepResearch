"""M9 live Ollama smoke test — real runtime only, never fails the suite.

Skips (never fails) when Ollama is not running or qwen3:4b is not
installed; the skip message states the exact setup command. No model
is ever substituted: any model other than the configured one aborts
the check explicitly.
"""

from __future__ import annotations

import time

import httpx
import pytest

from deepresearch.config import get_settings
from deepresearch.llm import DEFAULT_MODEL, TAGS_PATH, OllamaLLMProvider


def _ollama_tags(base_url: str) -> list[dict] | None:
    try:
        response = httpx.get(base_url.rstrip("/") + TAGS_PATH, timeout=10)
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return None
    if response.status_code != 200:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    return body.get("models") if isinstance(body, dict) else None


def test_live_ollama_qwen_smoke() -> None:
    settings = get_settings()
    assert settings.ollama_model == DEFAULT_MODEL == "qwen3:4b"
    models = _ollama_tags(settings.ollama_base_url)
    if models is None:
        pytest.skip(
            f"Ollama is not running at {settings.ollama_base_url}; "
            "start Ollama, then run: ollama pull qwen3:4b"
        )
    names = [m.get("name", "") for m in models if isinstance(m, dict)]
    wanted = settings.ollama_model
    if not any(n == wanted or n.startswith(wanted + ":") for n in names):
        pytest.skip(
            f"model {settings.ollama_model!r} not installed (have: {names or 'none'}); "
            f"run: ollama pull {settings.ollama_model}"
        )

    provider = OllamaLLMProvider.from_settings(settings)
    try:
        started = time.perf_counter()
        # No max_tokens cap here: qwen3:4b is a thinking model and spends
        # small token budgets thinking, returning empty text (M21 live
        # finding). Cap plumbing is covered by unit tests; this smoke
        # test only proves daemon + model + version capture.
        answer = provider.generate("Reply with exactly: OK", temperature=0.0)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
    finally:
        provider.close()
    assert answer.strip(), "Ollama returned an empty response"
    assert provider.model_version is not None
    print(
        f"\n[live-ollama] model={provider.model_version} "
        f"latency_ms={elapsed_ms} answer={answer.strip()[:80]!r}"
    )
