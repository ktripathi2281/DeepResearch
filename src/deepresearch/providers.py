"""Optional LLM provider adapters + centralized factory — Milestone 19.

The research pipeline depends on ``LLMProvider`` only; this module
holds every concrete choice behind that boundary:

- ``OpenAICompatibleLLMProvider`` — generic Chat Completions HTTP
  adapter (any ``/chat/completions`` endpoint, not only OpenAI).
- ``GeminiLLMProvider`` — small ``generateContent`` REST adapter, no
  vendor SDK (dependency discipline: ``httpx`` is already required).
- ``create_llm_provider(settings)`` — the single selection point
  (``LLM_PROVIDER``: default local provider, ``openai_compatible``,
  ``gemini``).

Rules shared by all adapters:

- Lazy: importing or constructing an adapter performs no network I/O;
  the HTTP client is created on first use. A missing credential
  fails with ``LLMConfigurationError`` only when the provider is
  actually selected or called — never at import or startup.
- Secret-free diagnostics: error messages and logs carry status
  codes, model names, and timeouts — never API keys, authorization
  headers, credential-bearing URLs, or full provider bodies.
- Token counts come only from numbers the provider returned
  (``usage`` / ``usageMetadata``); ``None`` means unavailable, never
  estimated.

NOTE on the M9 layering guard
(``test_no_direct_ollama_dependency_elsewhere``): this module maps the
*provider-name token* for the default branch to
``default_llm_provider``. Name tokens are legitimate since M19; the
guard forbids local-provider HTTP/API details, of which this module
contains none.
"""

from __future__ import annotations

import time

import httpx

from deepresearch.config import Settings
from deepresearch.llm import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    default_llm_provider,
    optional_count,
    optional_total,
)
from deepresearch.logging import get_logger

logger = get_logger(__name__)

PROVIDER_LOCAL = "ollama"
PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
PROVIDER_GEMINI = "gemini"
KNOWN_PROVIDERS = (
    PROVIDER_LOCAL,
    PROVIDER_OPENAI_COMPATIBLE,
    PROVIDER_GEMINI,
)

CHAT_COMPLETIONS_PATH = "/chat/completions"
GEMINI_GENERATE_METHOD = ":generateContent"

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"


def _validate_common(
    *, base_url: str, model: str, timeout_seconds: float, temperature: object, max_tokens: object
) -> tuple[str, str, float, float, int | None]:
    """Shared constructor validation; returns normalized (base_url, model, ...)."""
    if not base_url or not base_url.strip():
        raise LLMError("base_url must be non-empty")
    if not model or not model.strip():
        raise LLMError("model must be non-empty")
    if timeout_seconds <= 0:
        raise LLMError(f"timeout_seconds must be > 0, got {timeout_seconds}")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise LLMError(f"temperature must be a number, got {temperature!r}")
    if float(temperature) < 0:
        raise LLMError(f"temperature must be >= 0, got {temperature}")
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 1):
        raise LLMError(f"max_tokens must be a positive integer or None, got {max_tokens!r}")
    return base_url.rstrip("/"), model.strip(), timeout_seconds, float(temperature), max_tokens


def _coerce_temperature(temperature: float | None, default: float) -> float:
    temp = default if temperature is None else temperature
    if not isinstance(temp, (int, float)) or isinstance(temp, bool) or float(temp) < 0:
        raise LLMError(f"temperature must be a number >= 0, got {temperature!r}")
    return float(temp)


def _coerce_max_tokens(max_tokens: int | None, default: int | None) -> int | None:
    limit = default if max_tokens is None else max_tokens
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise LLMError(f"max_tokens must be a positive integer or None, got {max_tokens!r}")
    return limit


class OpenAICompatibleLLMProvider:
    """Generic OpenAI-compatible Chat Completions adapter (M19, optional).

    Works with any server implementing ``POST {base}/chat/completions``
    (OpenAI, local OpenAI-style servers, third-party gateways). The API
    key travels only as an ``Authorization`` header on actual calls and
    is never logged, never echoed in errors, and never required until
    this provider is selected or called.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 120,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url, self._model, self._timeout_seconds, self._temperature, self._max_tokens = (
            _validate_common(
                base_url=base_url,
                model=model,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        self._api_key = api_key or ""
        self._transport = transport
        self._client: httpx.Client | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def model_version(self) -> str | None:
        return None  # Chat Completions reports no stable version id; never invented.

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _client_or_create(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        return self.generate_response(
            prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
        ).text

    def generate_response(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise LLMError("prompt must be non-empty text")
        if not self._api_key.strip():
            raise LLMConfigurationError(
                "openai_compatible provider needs an API key "
                "(OPENAI_COMPATIBLE_API_KEY); refusing to call without credentials"
            )
        temp = _coerce_temperature(temperature, self._temperature)
        limit = _coerce_max_tokens(max_tokens, self._max_tokens)
        messages: list[dict] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict = {"model": self._model, "messages": messages, "temperature": temp}
        if limit is not None:
            payload["max_tokens"] = limit

        started = time.perf_counter()
        try:
            response = self._client_or_create().post(
                CHAT_COMPLETIONS_PATH,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"openai_compatible request timed out after {self._timeout_seconds}s "
                f"(model {self._model!r})"
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"cannot reach openai_compatible endpoint at {self._base_url} "
                f"(model {self._model!r})"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(
                f"openai_compatible transport failed at {self._base_url} (model {self._model!r})"
            ) from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code in (401, 403):
            raise LLMResponseError(
                f"openai_compatible rejected credentials (status {response.status_code}, "
                f"model {self._model!r}); check OPENAI_COMPATIBLE_API_KEY"
            )
        if response.status_code == 404:
            raise LLMModelNotFoundError(
                f"model {self._model!r} not found at {self._base_url}; check "
                "OPENAI_COMPATIBLE_MODEL"
            )
        if response.status_code != 200:
            raise LLMResponseError(
                f"openai_compatible returned status {response.status_code} "
                f"for model {self._model!r}: {response.text[:200]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMResponseError("openai_compatible returned non-JSON output") from exc
        text, usage = _parse_chat_completions(body, self._model)
        logger.info(
            "llm generation finished",
            extra={
                "stage": "generation",
                "method": self._model,
                "candidate_count": len(prompt),
                "selected_count": len(text),
                "duration_ms": elapsed_ms,
            },
        )
        return LLMResponse(
            text=text,
            model=self._model,
            provider=type(self).__name__,
            input_tokens=optional_count(usage.get("prompt_tokens")),
            output_tokens=optional_count(usage.get("completion_tokens")),
            total_tokens=optional_total(usage.get("prompt_tokens"), usage.get("completion_tokens"))
            if usage.get("total_tokens") is None
            else optional_count(usage.get("total_tokens")),
        )


def _parse_chat_completions(body: object, model: str) -> tuple[str, dict]:
    """Extract (text, usage) from a Chat Completions body; strict, no guessing."""
    if not isinstance(body, dict):
        raise LLMResponseError("openai_compatible response is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("openai_compatible response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMResponseError("openai_compatible choice is malformed")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LLMResponseError("openai_compatible choice has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError(f"openai_compatible returned no usable content for model {model!r}")
    usage = body.get("usage")
    return content, usage if isinstance(usage, dict) else {}


class GeminiLLMProvider:
    """Google Gemini ``generateContent`` REST adapter (M19, optional, no SDK).

    The API key travels only as an ``x-goog-api-key`` header on actual
    calls — deliberately not as a ``?key=`` query parameter, so the key
    can never appear in a URL (HTTP client request logs record URLs,
    never headers). The key is never logged, never echoed in errors,
    and never required until this provider is selected or called.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_GEMINI_BASE_URL,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        timeout_seconds: float = 120,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url, self._model, self._timeout_seconds, self._temperature, self._max_tokens = (
            _validate_common(
                base_url=base_url,
                model=model,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        self._api_key = api_key or ""
        self._transport = transport
        self._client: httpx.Client | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def model_version(self) -> str | None:
        return None  # generateContent reports no stable version id; never invented.

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _client_or_create(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        return self.generate_response(
            prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
        ).text

    def generate_response(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise LLMError("prompt must be non-empty text")
        if not self._api_key.strip():
            raise LLMConfigurationError(
                "gemini provider needs an API key (GEMINI_API_KEY); "
                "refusing to call without credentials"
            )
        temp = _coerce_temperature(temperature, self._temperature)
        limit = _coerce_max_tokens(max_tokens, self._max_tokens)
        generation_config: dict = {"temperature": temp}
        if limit is not None:
            generation_config["maxOutputTokens"] = limit
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        if system_prompt is not None:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        started = time.perf_counter()
        try:
            # NOTE: the key travels as a header, never as ``?key=``: HTTP
            # clients log request URLs, so a key in the URL would leak
            # into logs. It is never interpolated into messages either.
            response = self._client_or_create().post(
                f"/v1beta/models/{self._model}{GEMINI_GENERATE_METHOD}",
                headers={"x-goog-api-key": self._api_key},
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"gemini request timed out after {self._timeout_seconds}s (model {self._model!r})"
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"cannot reach gemini endpoint (model {self._model!r})"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"gemini transport failed (model {self._model!r})") from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code in (400, 401, 403):
            raise LLMResponseError(
                f"gemini rejected the request (status {response.status_code}, "
                f"model {self._model!r}); check GEMINI_API_KEY and GEMINI_MODEL"
            )
        if response.status_code == 404:
            raise LLMModelNotFoundError(
                f"model {self._model!r} not found via gemini; check GEMINI_MODEL"
            )
        if response.status_code != 200:
            raise LLMResponseError(
                f"gemini returned status {response.status_code} for model {self._model!r}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMResponseError("gemini returned non-JSON output") from exc
        text, usage = _parse_generate_content(body, self._model)
        logger.info(
            "llm generation finished",
            extra={
                "stage": "generation",
                "method": self._model,
                "candidate_count": len(prompt),
                "selected_count": len(text),
                "duration_ms": elapsed_ms,
            },
        )
        return LLMResponse(
            text=text,
            model=self._model,
            provider=type(self).__name__,
            input_tokens=optional_count(usage.get("promptTokenCount")),
            output_tokens=optional_count(usage.get("candidatesTokenCount")),
            total_tokens=optional_count(usage.get("totalTokenCount")),
        )


def _parse_generate_content(body: object, model: str) -> tuple[str, dict]:
    """Extract (text, usageMetadata) from a generateContent body; strict."""
    if not isinstance(body, dict):
        raise LLMResponseError("gemini response is not a JSON object")
    feedback = body.get("promptFeedback")
    if isinstance(feedback, dict) and isinstance(feedback.get("blockReason"), str):
        raise LLMResponseError(
            f"gemini blocked the request (reason {feedback['blockReason']!r}, model {model!r})"
        )
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LLMResponseError(f"gemini returned no candidates for model {model!r}")
    first = candidates[0]
    if not isinstance(first, dict):
        raise LLMResponseError("gemini candidate is malformed")
    content = first.get("content")
    if not isinstance(content, dict):
        raise LLMResponseError("gemini candidate has no content")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise LLMResponseError(f"gemini returned no content parts for model {model!r}")
    texts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    text = "".join(texts)
    if not text.strip():
        raise LLMResponseError(f"gemini returned no usable text for model {model!r}")
    usage = body.get("usageMetadata")
    return text, usage if isinstance(usage, dict) else {}


# --- factory -----------------------------------------------------------------


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Select the generation provider from ``settings.llm_provider``.

    The single selection point for the whole application: ``"ollama"``
    (default, local), ``"openai_compatible"``, or ``"gemini"``.
    Unknown names and missing cloud credentials fail here with
    ``LLMConfigurationError`` — never at import, never at startup
    unless this factory is actually invoked.
    """
    name = (settings.llm_provider or "").strip().lower()
    if name == PROVIDER_LOCAL:
        return default_llm_provider(settings)
    if name == PROVIDER_OPENAI_COMPATIBLE:
        if not (settings.openai_compatible_api_key or "").strip():
            raise LLMConfigurationError(
                "LLM_PROVIDER=openai_compatible needs OPENAI_COMPATIBLE_API_KEY "
                "set in the environment"
            )
        return OpenAICompatibleLLMProvider(
            base_url=settings.openai_compatible_base_url,
            api_key=settings.openai_compatible_api_key,
            model=settings.openai_compatible_model,
            timeout_seconds=settings.openai_compatible_timeout_seconds,
            temperature=settings.openai_compatible_temperature,
            max_tokens=settings.openai_compatible_max_tokens,
        )
    if name == PROVIDER_GEMINI:
        if not (settings.gemini_api_key or "").strip():
            raise LLMConfigurationError(
                "LLM_PROVIDER=gemini needs GEMINI_API_KEY set in the environment"
            )
        return GeminiLLMProvider(
            base_url=settings.gemini_base_url,
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout_seconds=settings.gemini_timeout_seconds,
            temperature=settings.gemini_temperature,
            max_tokens=settings.gemini_max_tokens,
        )
    raise LLMConfigurationError(
        f"unknown LLM_PROVIDER {settings.llm_provider!r}; "
        f"expected one of: {', '.join(KNOWN_PROVIDERS)}"
    )
