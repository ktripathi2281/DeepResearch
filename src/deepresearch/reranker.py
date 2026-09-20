"""Local cross-encoder reranking — Milestone 8.

Pipeline: hybrid (or vector/BM25) candidates → ``(query, text)`` pairs →
``Reranker`` (batched) → raw scores → ``RetrievalResult`` list with
``method="reranked"``, sorted score-descending, ties by chunk ID.

Independent from retrieval and generation: any ``RetrievalResult``
list can be reranked. Heavy ML imports are lazy so unit tests and the
API stay import-light and offline-safe.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from deepresearch.embeddings import EmbeddingError, resolve_device
from deepresearch.logging import get_logger
from deepresearch.observability import (
    COUNTER_RERANKER_CANDIDATES,
    COUNTER_RERANKER_RESULTS,
    count,
    traced_stage,
)
from deepresearch.retrieval import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    RetrievalError,
    RetrievalResult,
    validate_top_k,
)

logger = get_logger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_MODEL_VERSION = "1"
RERANKED_METHOD = "reranked"
DEFAULT_RERANKER_BATCH_SIZE = 16
DEFAULT_CANDIDATE_TOP_K = 20


class RerankerError(RuntimeError):
    """Base error for reranker load/inference failures."""


class RerankerLoadError(RerankerError):
    """The local reranker model could not be loaded (reported, not hidden)."""


@runtime_checkable
class Reranker(Protocol):
    """Minimal reranker boundary later milestones can depend on."""

    @property
    def model_name(self) -> str: ...
    @property
    def model_version(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def batch_size(self) -> int: ...

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """Score candidates in order; one raw score per document."""
        ...


class LocalCrossEncoderReranker:
    """sentence-transformers CrossEncoder provider for bge-reranker-base.

    The model loads once on first use (never per request) and is
    reusable across requests. Scores are raw cross-encoder logits —
    not normalized, not probabilities. ``bge-reranker-base`` (~278M
    params) runs on CPU; CUDA is used only when available and selected.
    """

    def __init__(
        self,
        *,
        model_name: str = RERANKER_MODEL,
        model_version: str = RERANKER_MODEL_VERSION,
        device: str = "auto",
        batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise RerankerError(f"batch_size must be > 0, got {batch_size}")
        self._model_name = model_name
        self._model_version = model_version
        self._device_request = device
        self._batch_size = batch_size
        self._model = None
        self._resolved_device: str | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        # The runtime exposes no checkpoint revision; the configured
        # version is recorded instead of an invented one.
        return self._model_version

    @property
    def device(self) -> str:
        return self._resolved_device if self._resolved_device is not None else self._device_request

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _ensure_model(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model
        try:
            device = resolve_device(self._device_request)
        except EmbeddingError as exc:
            raise RerankerLoadError(str(exc)) from exc
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self._model_name, device=device)
        except Exception as exc:
            raise RerankerLoadError(
                f"cannot load reranker model {self._model_name!r} on {device}: {exc}"
            ) from exc
        self._model = model
        self._resolved_device = device
        logger.info(
            "reranker model loaded",
            extra={"stage": "reranking", "method": self._model_name, "path": device},
        )
        return model

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        texts = list(documents)
        if not texts:
            return []
        model = self._ensure_model()
        try:
            scores = model.predict(
                [(query, doc) for doc in texts],
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RerankerError(f"reranker inference failed for {len(texts)} pairs: {exc}") from exc
        result = [float(s) for s in scores]
        if len(result) != len(texts):
            raise RerankerError(
                f"reranker returned {len(result)} scores for {len(texts)} candidates"
            )
        return result


def rerank_results(
    query: str,
    results: list[RetrievalResult],
    reranker: Reranker,
    *,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    top_k: int = DEFAULT_TOP_K,
    max_top_k: int = MAX_TOP_K,
) -> list[RetrievalResult]:
    """Rerank candidates with a cross-encoder and return the top-K finals.

    Only the first ``candidate_top_k`` inputs are scored (never more
    than necessary); the best ``top_k`` of those are returned with
    ``method="reranked"`, raw reranker scores, and reassigned ranks.
    Empty input returns ``[]`` without touching the provider.
    """
    if not query or not query.strip():
        raise RetrievalError("query must be non-empty text")
    candidate_limit = validate_top_k(candidate_top_k, maximum=max_top_k)
    limit = validate_top_k(top_k, maximum=max_top_k)
    candidates = list(results[:candidate_limit])
    if not candidates:
        return []

    started = time.perf_counter()
    count(COUNTER_RERANKER_CANDIDATES, len(candidates))
    try:
        with traced_stage("reranking"):
            scores = reranker.rerank(query.strip(), [c.text for c in candidates])
    except RerankerError:
        raise
    except Exception as exc:
        raise RerankerError(f"reranker failed: {exc}") from exc
    if len(scores) != len(candidates):
        raise RerankerError(
            f"reranker returned {len(scores)} scores for {len(candidates)} candidates; "
            "nothing was fabricated"
        )

    ordered = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: (-item[1], item[0].chunk_id),
    )
    final = [
        RetrievalResult(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            chunk_index=candidate.chunk_index,
            text=candidate.text,
            score=score,
            rank=rank,
            document_title=candidate.document_title,
            document_type=candidate.document_type,
            document_source=candidate.document_source,
            page=candidate.page,
            section=candidate.section,
            chunk_metadata=candidate.chunk_metadata,
            retrieval_method=RERANKED_METHOD,
        )
        for rank, (candidate, score) in enumerate(ordered[:limit], start=1)
    ]
    count(COUNTER_RERANKER_RESULTS, len(final))
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "reranking finished",
        extra={
            "stage": "reranking",
            "method": reranker.model_name,
            "candidate_count": len(candidates),
            "selected_count": len(final),
            "duration_ms": elapsed_ms,
        },
    )
    return final
