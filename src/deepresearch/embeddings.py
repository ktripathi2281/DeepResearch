"""Local embedding generation and persistence — Milestone 4.

Pipeline: pending ``Chunk`` rows → ``EmbeddingProvider`` (batched) →
dimension validation → ``Chunk.embedding`` + model/version metadata.

Only the M2 schema fields are used (``embedding VECTOR(384)``,
``embedding_model``, ``embedding_version``). No retrieval, no reranking,
no generation. Heavy ML libraries (torch, sentence-transformers) are
imported lazily so unit tests and the API stay import-light and
offline-safe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session

from deepresearch.logging import get_logger
from deepresearch.observability import traced_stage

logger = get_logger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_VERSION = "1"
EMBEDDING_DIMENSION = 384
DEFAULT_BATCH_SIZE = 32


class EmbeddingError(RuntimeError):
    """Base error for embedding generation/persistence failures."""


class DimensionMismatchError(EmbeddingError):
    """A vector did not have exactly 384 dimensions. Never persisted."""


class ModelLoadError(EmbeddingError):
    """The local embedding model could not be loaded (reported, not hidden)."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal provider boundary later retrieval code (M5) can depend on."""

    @property
    def model_name(self) -> str: ...
    @property
    def model_version(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    @property
    def device(self) -> str: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; returns one vector per input, in order."""
        ...


def resolve_device(requested: str) -> str:
    """Map a device request to a concrete torch device name.

    - ``"auto"`` → ``"cuda"`` when a GPU is available, else ``"cpu"``.
    - ``"cpu"`` → ``"cpu"`` (always works, correctness never needs a GPU).
    - ``"cuda"`` → ``"cuda"`` only when available, else ``ModelLoadError``.
    """
    normalized = requested.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized in ("auto", "cuda"):
        try:
            import torch
        except ImportError as exc:
            if normalized == "cuda":
                raise ModelLoadError("device 'cuda' requested but torch is not installed") from exc
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if normalized == "cuda":
            raise ModelLoadError("device 'cuda' requested but torch reports no GPU")
        return "cpu"
    raise ModelLoadError(f"unknown embedding device: {requested!r} (expected auto/cpu/cuda)")


class LocalEmbeddingProvider:
    """sentence-transformers provider for BAAI/bge-small-en-v1.5.

    The model loads once on first use (never per chunk). Embeddings are
    L2-normalized so M5 cosine search can rely on dot-product geometry.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        model_version: str = DEFAULT_MODEL_VERSION,
        device: str = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise EmbeddingError(f"batch_size must be > 0, got {batch_size}")
        self._model_name = model_name
        self._model_version = model_version
        self._device_request = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._model = None
        self._resolved_device: str | None = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return self._dimension if self._dimension is not None else EMBEDDING_DIMENSION

    @property
    def device(self) -> str:
        return self._resolved_device if self._resolved_device is not None else self._device_request

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def normalized(self) -> bool:
        return self._normalize

    def _ensure_model(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model
        device = resolve_device(self._device_request)
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._model_name, device=device)
        except Exception as exc:
            raise ModelLoadError(
                f"cannot load embedding model {self._model_name!r} on {device}: {exc}"
            ) from exc
        self._model = model
        self._resolved_device = device
        if hasattr(model, "get_embedding_dimension"):
            self._dimension = int(model.get_embedding_dimension())
        else:  # older sentence-transformers API name
            self._dimension = int(model.get_sentence_embedding_dimension())
        logger.info(
            "embedding model loaded",
            extra={"stage": "embedding", "method": self._model_name, "path": device},
        )
        return model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        try:
            vectors = model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"inference failed for {len(texts)} texts: {exc}") from exc
        return [[float(v) for v in row] for row in vectors]


@dataclass(frozen=True)
class EmbeddingResult:
    embedded: int
    skipped_uptodate: int
    skipped_mismatch: int
    model_name: str
    model_version: str
    device: str
    elapsed_ms: int


def _validate_batch(vectors: list[list[float]], *, expected: int) -> None:
    for i, vector in enumerate(vectors):
        if len(vector) != expected:
            raise DimensionMismatchError(
                f"vector {i} has {len(vector)} dimensions, expected {expected}; "
                "nothing from this batch was persisted"
            )


def embed_pending_chunks(
    session: Session,
    provider: EmbeddingProvider,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    force: bool = False,
) -> EmbeddingResult:
    """Embed chunks missing vectors and persist them with model metadata.

    - Chunks with no embedding are always embedded.
    - Chunks already embedded with the same model+version are skipped.
    - Chunks embedded with a different model/version are skipped unless
      ``force=True`` (explicit re-embedding; never silently mixed).
    - One transaction per batch: a failed batch rolls back, earlier
      batches stay committed, and the error surfaces to the caller.
    """
    from deepresearch import repository

    if batch_size <= 0:
        raise EmbeddingError(f"batch_size must be > 0, got {batch_size}")
    if provider.dimension != EMBEDDING_DIMENSION:
        raise DimensionMismatchError(
            f"provider {provider.model_name!r} reports dimension {provider.dimension}, "
            f"expected {EMBEDDING_DIMENSION}; refusing to write"
        )

    started = time.perf_counter()
    pending = repository.list_chunks_missing_embeddings(session)
    stale = repository.list_chunks_with_stale_embeddings(
        session, model_name=provider.model_name, model_version=provider.model_version
    )
    uptodate = repository.count_chunks_with_embeddings(
        session, model_name=provider.model_name, model_version=provider.model_version
    )
    targets = list(pending) + ([c for c in stale] if force else [])
    if limit is not None:
        targets = targets[:limit]

    logger.info(
        "embedding run started",
        extra={
            "stage": "embedding",
            "method": provider.model_name,
            "path": provider.device,
            "candidate_count": len(targets),
            "selected_count": len(pending),
        },
    )

    embedded = 0
    with traced_stage("embedding"):
        for offset in range(0, len(targets), batch_size):
            batch = targets[offset : offset + batch_size]
            vectors = provider.embed_texts([c.text for c in batch])
            if len(vectors) != len(batch):
                session.rollback()
                raise EmbeddingError(
                    f"provider returned {len(vectors)} vectors for {len(batch)} chunks"
                )
            _validate_batch(vectors, expected=EMBEDDING_DIMENSION)
            try:
                for chunk, vector in zip(batch, vectors, strict=True):
                    chunk.embedding = vector
                    chunk.embedding_model = provider.model_name
                    chunk.embedding_version = provider.model_version
                session.commit()
            except Exception as exc:
                session.rollback()
                raise EmbeddingError(f"failed persisting batch at offset {offset}: {exc}") from exc
            embedded += len(batch)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    result = EmbeddingResult(
        embedded=embedded,
        skipped_uptodate=uptodate,
        skipped_mismatch=0 if force else len(stale),
        model_name=provider.model_name,
        model_version=provider.model_version,
        device=provider.device,
        elapsed_ms=elapsed_ms,
    )
    logger.info(
        "embedding run finished",
        extra={
            "stage": "embedding",
            "method": provider.model_name,
            "candidate_count": len(targets),
            "selected_count": embedded,
            "duration_ms": elapsed_ms,
        },
    )
    return result
