"""M9 unit tests — Ollama provider over a mock HTTP transport (no Ollama needed)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from deepresearch.config import Settings
from deepresearch.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMResponseError,
    LLMTimeoutError,
    OllamaLLMProvider,
)

SRC = Path(__file__).parent.parent / "src" / "deepresearch"


def _ok_response(text: str = "Hello!", model: str = "qwen3:4b") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "response": text,
            "done": True,
            "done_reason": "stop",
        },
    )


def _provider(handler, **kwargs) -> OllamaLLMProvider:  # type: ignore[no-untyped-def]
    return OllamaLLMProvider(transport=httpx.MockTransport(handler), **kwargs)


def test_config_defaults() -> None:
    settings = Settings()
    assert settings.ollama_base_url == DEFAULT_BASE_URL == "http://localhost:11434"
    assert settings.ollama_model == DEFAULT_MODEL == "qwen3:4b"
    assert settings.ollama_timeout_seconds == 120
    assert settings.ollama_temperature == 0.0
    assert settings.ollama_max_tokens is None
    provider = OllamaLLMProvider.from_settings(settings)
    assert isinstance(provider, LLMProvider)
    assert provider.model_name == "qwen3:4b"
    assert provider.model_version is None  # unknown until a call happens
    provider.close()


def test_request_construction() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content.decode())
        return _ok_response()

    provider = _provider(handler)
    try:
        assert provider.generate("Hi there", system_prompt="Be brief.", temperature=0.0) == "Hello!"
    finally:
        provider.close()
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/generate"
    assert seen["body"]["model"] == "qwen3:4b"
    assert seen["body"]["prompt"] == "Hi there"
    assert seen["body"]["system"] == "Be brief."
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"]["temperature"] == 0.0
    assert "num_predict" not in seen["body"]["options"]  # None → provider default


def test_max_tokens_sent_when_set() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return _ok_response()

    provider = _provider(handler, max_tokens=256)
    try:
        provider.generate("Hi", temperature=0.5, max_tokens=64)
    finally:
        provider.close()
    assert seen["body"]["options"] == {"temperature": 0.5, "num_predict": 64}


def test_blank_prompt_rejected_without_http_call() -> None:
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _ok_response()

    provider = _provider(handler)
    try:
        for bad in ("", "   "):
            with pytest.raises(LLMError):
                provider.generate(bad)
    finally:
        provider.close()
    assert calls == []


def test_empty_response_rejected() -> None:
    for empty in ("", "   "):
        provider = _provider(lambda request, text=empty: _ok_response(text=text))
        try:
            with pytest.raises(LLMResponseError):
                provider.generate("Hi")
        finally:
            provider.close()


def test_malformed_responses_rejected() -> None:
    bodies = ["not json{{{", "[1, 2]", '{"done": true}', '{"response": 42}']
    for body in bodies:
        provider = _provider(_response_with_body(body))
        try:
            with pytest.raises(LLMResponseError):
                provider.generate("Hi")
        finally:
            provider.close()


def _response_with_body(body: str):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    return handler


def test_connection_failure_names_ollama() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    provider = _provider(handler, base_url="http://localhost:11434")
    try:
        with pytest.raises(LLMConnectionError, match="Ollama"):
            provider.generate("Hi")
    finally:
        provider.close()


def test_timeout_surfaces() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    provider = _provider(handler, timeout_seconds=5)
    try:
        with pytest.raises(LLMTimeoutError, match="5"):
            provider.generate("Hi")
    finally:
        provider.close()


def test_missing_model_identifies_pull_command() -> None:
    provider = _provider(
        lambda request: httpx.Response(404, text="model 'qwen3:4b' not found"),
        model="qwen3:4b",
    )
    try:
        with pytest.raises(LLMModelNotFoundError, match="ollama pull qwen3:4b"):
            provider.generate("Hi")
    finally:
        provider.close()


def test_server_error_surfaces_status() -> None:
    provider = _provider(lambda request: httpx.Response(500, text="boom"))
    try:
        with pytest.raises(LLMResponseError, match="500"):
            provider.generate("Hi")
    finally:
        provider.close()


def test_runtime_model_metadata_captured() -> None:
    provider = _provider(lambda request: _ok_response(model="qwen3:4b-runtime-id"))
    try:
        assert provider.model_version is None
        provider.generate("Hi")
        assert provider.model_version == "qwen3:4b-runtime-id"
    finally:
        provider.close()


def test_temperature_validation() -> None:
    provider = _provider(lambda request: _ok_response())
    try:
        with pytest.raises(LLMError):
            OllamaLLMProvider(temperature=-1.0)
        with pytest.raises(LLMError):
            OllamaLLMProvider(max_tokens=0)
        with pytest.raises(LLMError):
            provider.generate("Hi", temperature=-0.5)
        # Default temperature 0.0 is deterministic-friendly.
        assert provider.generate("Hi") == "Hello!"
    finally:
        provider.close()


# Since M19 the provider *name token* ("ollama" as an LLM_PROVIDER
# value, identity default, or doc word) is a legitimate cross-cutting
# concern. What must not leak outside llm.py/config.py is provider
# HTTP/API knowledge: the concrete class, endpoint port/path, CLI
# hints, and direct settings-attribute access (config.py stays the
# indirection). See ADR-019.
OLLAMA_API_DETAILS = (
    "ollamallmprovider",
    "11434",
    "/api/generate",
    "ollama pull",
    "ollama_base_url",
    "ollama_model",
    "ollama_timeout",
    "ollama_temperature",
    "ollama_max_tokens",
)


def test_no_direct_ollama_dependency_elsewhere() -> None:
    # Settings in config.py are the sanctioned indirection; anything
    # else touching local-provider HTTP/API details is a layering leak.
    offenders = []
    for path in SRC.glob("*.py"):
        if path.name in {"llm.py", "config.py", "__init__.py"}:
            continue
        content = path.read_text(encoding="utf-8").lower()
        hits = [marker for marker in OLLAMA_API_DETAILS if marker in content]
        if hits:
            offenders.append(f"{path.name}: {hits}")
    assert offenders == []
