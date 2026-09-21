"""M19 security regression — credentials never leak, cloud stays dormant.

Sentinel keys stand in for real credentials. Every failure mode is
exercised over ``httpx.MockTransport`` (no network): if a key,
authorization header, or credential-bearing URL ever reaches a log
record or an exception message, these tests fail.
"""

from __future__ import annotations

import importlib
import logging

import httpx
import pytest

from deepresearch.config import Settings
from deepresearch.llm import LLMConfigurationError, LLMProvider, OllamaLLMProvider
from deepresearch.providers import (
    GeminiLLMProvider,
    OpenAICompatibleLLMProvider,
    create_llm_provider,
)

SENTINEL = "sk-test-sentinel-9f8e7d6c5b"


def _logged_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


def _openai(handler, **kwargs) -> OpenAICompatibleLLMProvider:  # type: ignore[no-untyped-def]
    return OpenAICompatibleLLMProvider(
        api_key=SENTINEL, transport=httpx.MockTransport(handler), **kwargs
    )


def _gemini(handler, **kwargs) -> GeminiLLMProvider:  # type: ignore[no-untyped-def]
    return GeminiLLMProvider(api_key=SENTINEL, transport=httpx.MockTransport(handler), **kwargs)


def _fail_modes_openai():  # type: ignore[no-untyped-def]
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    return [
        _openai(boom),
        _openai(slow),
        _openai(lambda request: httpx.Response(401, text="bad key")),
        _openai(lambda request: httpx.Response(500, text="boom")),
        _openai(lambda request: httpx.Response(200, text="not json")),
    ]


def _fail_modes_gemini():  # type: ignore[no-untyped-def]
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    return [
        _gemini(boom),
        _gemini(slow),
        _gemini(lambda request: httpx.Response(400, text="bad key")),
        _gemini(lambda request: httpx.Response(200, text="not json")),
    ]


def test_api_key_never_in_logs_openai(caplog) -> None:  # type: ignore[no-untyped-def]
    with caplog.at_level(logging.INFO):
        for provider in [
            *_fail_modes_openai(),
            _openai(
                lambda request: httpx.Response(
                    200, json={"choices": [{"message": {"content": "ok"}}]}
                )
            ),
        ]:
            try:
                provider.generate("Hi")
            except Exception:  # noqa: BLE001, S110 — exercising failure modes
                pass
            finally:
                provider.close()
    assert SENTINEL not in _logged_text(caplog)


def test_api_key_never_in_logs_gemini(caplog) -> None:  # type: ignore[no-untyped-def]
    with caplog.at_level(logging.INFO):
        for provider in [
            *_fail_modes_gemini(),
            _gemini(
                lambda request: httpx.Response(
                    200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
                )
            ),
        ]:
            try:
                provider.generate("Hi")
            except Exception:  # noqa: BLE001, S110 — exercising failure modes
                pass
            finally:
                provider.close()
    logged = _logged_text(caplog)
    assert SENTINEL not in logged
    # Belt and braces: no credential-bearing query string may ever be logged.
    assert "key=" not in logged


def test_api_key_and_auth_never_in_errors() -> None:
    checked = 0
    for provider in [*_fail_modes_openai(), *_fail_modes_gemini()]:
        try:
            provider.generate("Hi")
            raise AssertionError("expected a provider failure") from None  # pragma: no cover
        except LLMConfigurationError:
            raise AssertionError("unexpected configuration error") from None  # pragma: no cover
        except Exception as exc:
            assert SENTINEL not in str(exc), f"key leaked in {type(exc).__name__}"
            assert "Bearer" not in str(exc), f"auth scheme leaked in {type(exc).__name__}"
            checked += 1
        finally:
            provider.close()
    assert checked == len(_fail_modes_openai()) + len(_fail_modes_gemini())


def test_import_and_construct_perform_no_network(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    created: list = []

    class NoNetworkClient(httpx.Client):  # type: ignore[no-untyped-def]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            created.append(True)
            raise AssertionError("network client must not be created")

    monkeypatch.setattr(httpx, "Client", NoNetworkClient)
    import deepresearch.providers as providers_mod

    importlib.reload(providers_mod)
    try:
        # Construct everything selectable — still no client.
        providers_mod.OpenAICompatibleLLMProvider(api_key=SENTINEL)
        providers_mod.OpenAICompatibleLLMProvider()
        providers_mod.GeminiLLMProvider(api_key=SENTINEL)
        providers_mod.GeminiLLMProvider()
        provider = providers_mod.create_llm_provider(Settings())
        assert isinstance(provider, OllamaLLMProvider)
        # Missing-key failures happen before any client exists.
        with pytest.raises(LLMConfigurationError):
            providers_mod.OpenAICompatibleLLMProvider().generate("Hi")
        with pytest.raises(LLMConfigurationError):
            providers_mod.GeminiLLMProvider().generate("Hi")
    finally:
        importlib.reload(providers_mod)
    assert created == []


def test_selecting_local_needs_no_cloud_credentials(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for variable in ("OPENAI_COMPATIBLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    provider = create_llm_provider(Settings())
    assert isinstance(provider, LLMProvider)
    assert isinstance(provider, OllamaLLMProvider)
    assert provider.model_name == "qwen3:4b"


def test_cloud_not_contacted_unless_selected(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    created: list = []
    real_client = httpx.Client

    class RecordingClient(real_client):  # type: ignore[no-untyped-def]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            created.append(True)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", RecordingClient)
    try:
        create_llm_provider(Settings())
        OpenAICompatibleLLMProvider(api_key=SENTINEL).close()
        GeminiLLMProvider(api_key=SENTINEL).close()
    finally:
        pass
    assert created == []
