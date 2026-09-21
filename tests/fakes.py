"""Shared deterministic test doubles (no models, no network, no GPU)."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence

from deepresearch.embeddings import EMBEDDING_DIMENSION, EmbeddingError
from deepresearch.llm import LLMError, LLMResponse
from deepresearch.reranker import RerankerError


class FakeEmbeddingProvider:
    """Deterministic L2-normalized vectors standing in for bge-small-en-v1.5."""

    def __init__(
        self,
        *,
        model_name: str = "fake-test-model",
        model_version: str = "test-v1",
        dimension: int = EMBEDDING_DIMENSION,
        device: str = "cpu",
        fail_on: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_version = model_version
        self._dimension = dimension
        self._device = device
        self._fail_on = fail_on
        self.calls: list[int] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def device(self) -> str:
        return self._device

    def _vector(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self._dimension)]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        if self._fail_on is not None and any(self._fail_on in t for t in texts):
            raise EmbeddingError("fake provider forced failure")
        return [self._vector(t) for t in texts]


class FakeReranker:
    """Deterministic test double: scores come from a text map (default fallback)."""

    def __init__(
        self,
        *,
        scores: dict[str, float] | None = None,
        default: float = 0.0,
        model_name: str = "fake-reranker",
        device: str = "cpu",
        batch_size: int = 16,
        fail: bool = False,
    ) -> None:
        self._scores = scores or {}
        self._default = default
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._fail = fail
        self.calls: list[int] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return "fake-v1"

    @property
    def device(self) -> str:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        docs = list(documents)
        self.calls.append(len(docs))
        if self._fail:
            raise RerankerError("fake reranker forced failure")
        return [self._scores.get(doc, self._default) for doc in docs]


class FakeLLMProvider:
    """Deterministic test double: fixed answer, recorded calls, optional failure."""

    def __init__(
        self,
        *,
        answer: str = "Fake grounded answer.",
        model_name: str = "fake-llm",
        model_version: str | None = "fake-v1",
        fail: bool = False,
    ) -> None:
        self._answer = answer
        self._model_name = model_name
        self._model_version = model_version
        self._fail = fail
        self.calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str | None:
        return self._model_version

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self._fail:
            raise LLMError("fake LLM forced failure")
        return self._answer

    def generate_response(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Text-only fake: delegates to ``generate``; usage honestly unknown (None)."""
        return LLMResponse(
            text=self.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model=self.model_name,
            provider=type(self).__name__,
        )


class FakeVerifierLLM(FakeLLMProvider):
    """Scripted verifier: pops one canned JSON/text response per call."""

    def __init__(self, *, responses: list[str], model_name: str = "fake-verifier") -> None:
        super().__init__(answer="", model_name=model_name, model_version="fake-vv")
        self._responses = list(responses)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise LLMError("fake verifier has no scripted response left")
        return self._responses.pop(0)


class FakeAgentLLM(FakeLLMProvider):
    """Scripted agent decisions: pops one canned JSON decision per call."""

    def __init__(self, *, decisions: list[str], model_name: str = "fake-agent") -> None:
        super().__init__(answer="", model_name=model_name, model_version="fake-av")
        self._decisions = list(decisions)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._decisions:
            raise LLMError("fake agent has no scripted decision left")
        return self._decisions.pop(0)
