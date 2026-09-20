"""Semantic vector retrieval — Milestone 5.

Path: text query → ``EmbeddingProvider`` (one call) → 384-d vector →
PostgreSQL + pgvector cosine search → ranked ``RetrievalResult`` list.

Depends on the M4 provider abstraction, never on
sentence-transformers internals. Read-only: no commits, no writes.
No BM25, no hybrid fusion, no reranking (later milestones).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.embeddings import EMBEDDING_DIMENSION, EmbeddingProvider
from deepresearch.logging import get_logger
from deepresearch.observability import (
    COUNTER_RETRIEVAL_VECTOR,
    count,
    traced_stage,
)

logger = get_logger(__name__)

DEFAULT_TOP_K = 5
MAX_TOP_K = 100
RETRIEVAL_METHOD = "vector"


class RetrievalError(RuntimeError):
    """Base error for retrieval validation/query failures."""


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked hit with everything later evidence/citation code needs."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    text: str
    score: float
    rank: int
    document_title: str | None
    document_type: str
    document_source: str
    page: int | None
    section: str | None
    chunk_metadata: dict | None
    retrieval_method: str = RETRIEVAL_METHOD


def validate_top_k(top_k: int, *, maximum: int = MAX_TOP_K) -> int:
    """Validate ``top_k``; out-of-range values fail loudly (never clamped)."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise RetrievalError(f"top_k must be an integer, got {top_k!r}")
    if top_k < 1:
        raise RetrievalError(f"top_k must be >= 1, got {top_k}")
    if top_k > maximum:
        raise RetrievalError(f"top_k ({top_k}) exceeds maximum ({maximum})")
    return top_k


def retrieve(
    session: Session,
    provider: EmbeddingProvider,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_top_k: int = MAX_TOP_K,
    document_id: uuid.UUID | None = None,
    document_type: str | None = None,
) -> list[RetrievalResult]:
    """Embed one query and return the top-K most similar embedded chunks.

    Only chunks embedded with the provider's own model/version are
    searched; unembedded or foreign-model chunks never appear. An empty
    corpus yields ``[]``, never an error.
    """
    if not query or not query.strip():
        raise RetrievalError("query must be non-empty text")
    limit = validate_top_k(top_k, maximum=max_top_k)

    started = time.perf_counter()
    with traced_stage("embedding"):
        vectors = provider.embed_texts([query.strip()])
    if len(vectors) != 1:
        raise RetrievalError(f"provider returned {len(vectors)} vectors for 1 query")
    query_vector = vectors[0]
    if len(query_vector) != EMBEDDING_DIMENSION:
        raise RetrievalError(
            f"query vector has {len(query_vector)} dimensions, "
            f"expected {EMBEDDING_DIMENSION}; refusing to search"
        )

    try:
        with traced_stage("vector_retrieval"):
            rows = repository.search_chunks_by_vector(
                session,
                query_vector=query_vector,
                model_name=provider.model_name,
                model_version=provider.model_version,
                limit=limit,
                document_id=document_id,
                document_type=document_type,
            )
    except Exception as exc:
        raise RetrievalError(f"vector search failed: {exc}") from exc
    count(COUNTER_RETRIEVAL_VECTOR, len(rows))

    results = [
        RetrievalResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            score=score,
            rank=rank,
            document_title=doc.title,
            document_type=doc.document_type,
            document_source=doc.source,
            page=chunk.page,
            section=chunk.section,
            chunk_metadata=chunk.chunk_metadata,
        )
        for rank, (chunk, doc, score) in enumerate(rows, start=1)
    ]
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "vector retrieval finished",
        extra={
            "stage": "semantic_retrieval",
            "method": provider.model_name,
            "candidate_count": len(results),
            "selected_count": len(results),
            "duration_ms": elapsed_ms,
        },
    )
    return results
