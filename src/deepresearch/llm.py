"""Local LLM generation provider — Milestone 9.

Establishes the generation capability only: an ``LLMProvider`` Protocol
plus ``OllamaLLMProvider`` over the local Ollama HTTP API
(``qwen3:4b`` default). No retrieval wiring, no structured output, no
citations, no agents — those belong to M10+. Answer text only; no
chain-of-thought is requested, stored, or returned.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import httpx

from deepresearch.config import Settings
from deepresearch.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_TEMPERATURE = 0.0
GENERATE_PATH = "/api/generate"
TAGS_PATH = "/api/tags"


class LLMError(RuntimeError):
    """Base error for LLM generation failures."""


class LLMConnectionError(LLMError):
    """Ollama could not be reached at the configured base URL."""


class LLMTimeoutError(LLMError):
    """An Ollama request exceeded the configured timeout."""


class LLMResponseError(LLMError):
    """Ollama returned an error status or an unusable response."""


class LLMModelNotFoundError(LLMResponseError):
    """The configured model is not installed in Ollama."""


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal generation boundary; future cloud providers implement this."""

    @property
    def model_name(self) -> str: ...
    @property
    def model_version(self) -> str | None: ...

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Generate answer text for one prompt. Raises ``LLMError`` on failure."""
        ...


class OllamaLLMProvider:
    """Local Ollama provider (``/api/generate``, non-streaming JSON).

    Ollama's request/response schema lives only in this class — the
    rest of the application depends on ``LLMProvider``. One request
    per ``generate`` call with an explicit timeout; no retries (a
    failed call surfaces immediately instead of hanging the pipeline).
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
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
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._temperature = float(temperature)
        self._max_tokens = max_tokens
        self._transport = transport
        self._client: httpx.Client | None = None
        self._runtime_model: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaLLMProvider:
        return cls(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
            temperature=settings.ollama_temperature,
            max_tokens=settings.ollama_max_tokens,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def model_version(self) -> str | None:
        # Runtime identifier reported by Ollama, if a call has happened;
        # never invented when unknown.
        return self._runtime_model

    @property
    def base_url(self) -> str:
        return self._base_url

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

    def _options(self, temperature: float, max_tokens: int | None) -> dict:
        options: dict = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens  # Ollama's max-output-tokens option
        return options

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise LLMError("prompt must be non-empty text")
        temp = self._temperature if temperature is None else temperature
        if not isinstance(temp, (int, float)) or isinstance(temp, bool) or float(temp) < 0:
            raise LLMError(f"temperature must be a number >= 0, got {temperature!r}")
        limit = self._max_tokens if max_tokens is None else max_tokens
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise LLMError(f"max_tokens must be a positive integer or None, got {max_tokens!r}")

        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(float(temp), limit),
        }
        if system_prompt is not None:
            payload["system"] = system_prompt

        started = time.perf_counter()
        try:
            response = self._client_or_create().post(GENERATE_PATH, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama request timed out after {self._timeout_seconds}s "
                f"at {self._base_url} (model {self._model!r})"
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"cannot reach Ollama at {self._base_url} "
                f"(model {self._model!r}); is Ollama running?"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"Ollama transport failed at {self._base_url}: {exc}") from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code == 404:
            raise LLMModelNotFoundError(
                f"model {self._model!r} not found in Ollama at {self._base_url}; "
                f"run: ollama pull {self._model}"
            )
        if response.status_code != 200:
            raise LLMResponseError(
                f"Ollama returned status {response.status_code} "
                f"for model {self._model!r}: {response.text[:200]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMResponseError(f"Ollama returned non-JSON output: {exc}") from exc
        if not isinstance(body, dict) or not isinstance(body.get("response"), str):
            raise LLMResponseError("Ollama response has no 'response' text field")
        answer = body["response"]
        if not answer.strip():
            raise LLMResponseError(f"Ollama returned an empty response for model {self._model!r}")
        runtime_model = body.get("model")
        if isinstance(runtime_model, str) and runtime_model:
            self._runtime_model = runtime_model
        logger.info(
            "llm generation finished",
            extra={
                "stage": "generation",
                "method": self._model,
                "candidate_count": len(prompt),
                "selected_count": len(answer),
                "duration_ms": elapsed_ms,
            },
        )
        return answer
