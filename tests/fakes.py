"""Shared deterministic test double for M4 (no model, no network, no GPU)."""

from __future__ import annotations

import hashlib
import math
import random

from deepresearch.embeddings import EMBEDDING_DIMENSION, EmbeddingError


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
