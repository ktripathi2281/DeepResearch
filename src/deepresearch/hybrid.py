"""Hybrid retrieval — Milestone 7.

Orchestration + Reciprocal Rank Fusion over the independent M5 vector
and M6 BM25 paths:

    query
      |
      +----> vector retrieval (M5)
      |
      +----> BM25 retrieval (M6)
      |
      v
    RRF fusion
      |
      v
    ranked RetrievalResult list (method="hybrid")

No pgvector SQL, BM25 scoring, embedding, or tokenization is duplicated
here — both retrievers run unmodified. No reranking (M8).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from deepresearch.bm25 import DEFAULT_B, DEFAULT_K1, BM25Index, retrieve_bm25
from deepresearch.embeddings import EmbeddingProvider
from deepresearch.logging import get_logger
from deepresearch.observability import (
    COUNTER_RETRIEVAL_HYBRID,
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
from deepresearch.retrieval import retrieve as retrieve_vector

logger = get_logger(__name__)

HYBRID_METHOD = "hybrid"
DEFAULT_FUSION_METHOD = "rrf"
SUPPORTED_FUSION_METHODS = ("rrf",)
DEFAULT_RRF_K = 60
CANDIDATE_MULTIPLIER = 2


@dataclass(frozen=True)
class FusedCandidate:
    """Internal fusion trace: winning result plus its fused score.

    Component (vector/BM25) scores stay internal — the public
    ``RetrievalResult`` carries only the fused score. Kept separate so
    debugging never leaks into the public model.
    """

    result: RetrievalResult
    fused_score: float


def validate_rrf_k(rrf_k: int) -> int:
    """RRF smoothing constant must be a positive integer."""
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool):
        raise RetrievalError(f"rrf_k must be an integer, got {rrf_k!r}")
    if rrf_k < 1:
        raise RetrievalError(f"rrf_k must be >= 1, got {rrf_k}")
    return rrf_k


def validate_fusion_method(fusion_method: str) -> str:
    """Only implemented fusion strategies are accepted."""
    if fusion_method not in SUPPORTED_FUSION_METHODS:
        raise RetrievalError(
            f"unsupported fusion_method: {fusion_method!r} "
            f"(supported: {', '.join(SUPPORTED_FUSION_METHODS)})"
        )
    return fusion_method


def fuse_rrf(
    vector_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[FusedCandidate]:
    """Fuse two ranked lists with Reciprocal Rank Fusion.

    A chunk at 1-based rank ``r`` contributes ``1 / (rrf_k + r)`` per
    list it appears in; the fused score is the sum of contributions.
    Raw vector/BM25 scores are never mixed (different scales).
    Sorted by fused score descending, ties by chunk ID ascending.
    """
    validate_rrf_k(rrf_k)
    contributions: dict[uuid.UUID, float] = {}
    winners: dict[uuid.UUID, RetrievalResult] = {}
    for results in (vector_results, bm25_results):
        for result in results:
            contributions[result.chunk_id] = contributions.get(result.chunk_id, 0.0) + 1.0 / (
                rrf_k + result.rank
            )
            # Metadata is identical for the same chunk on either side;
            # prefer the vector result deterministically when present.
            if result.chunk_id not in winners or result.retrieval_method == "vector":
                winners[result.chunk_id] = result
    ordered = sorted(contributions.items(), key=lambda item: (-item[1], item[0]))
    return [
        FusedCandidate(result=winners[chunk_id], fused_score=score) for chunk_id, score in ordered
    ]


def _candidate_pool_size(top_k: int, explicit: int | None, *, max_top_k: int) -> int:
    if explicit is None:
        return min(CANDIDATE_MULTIPLIER * top_k, max_top_k)
    return validate_top_k(explicit, maximum=max_top_k)


def retrieve_hybrid(
    session: Session,
    embedding_provider: EmbeddingProvider,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_top_k: int = MAX_TOP_K,
    vector_top_k: int | None = None,
    bm25_top_k: int | None = None,
    document_id: uuid.UUID | None = None,
    document_type: str | None = None,
    index: BM25Index | None = None,
    fusion_method: str = DEFAULT_FUSION_METHOD,
    rrf_k: int = DEFAULT_RRF_K,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> list[RetrievalResult]:
    """Fuse M5 vector and M6 BM25 candidates with RRF and return top-K hybrids.

    Candidate pools default to ``2 * top_k`` (capped at ``max_top_k``) so
    fusion sees beyond the final cut. Exactly one query embedding is
    generated (inside the M5 call). Retriever failures — including a
    stale BM25 index — propagate instead of degrading silently.
    """
    if not query or not query.strip():
        raise RetrievalError("query must be non-empty text")
    if not isinstance(max_top_k, int) or isinstance(max_top_k, bool) or max_top_k < 1:
        raise RetrievalError(f"max_top_k must be an integer >= 1, got {max_top_k!r}")
    limit = validate_top_k(top_k, maximum=max_top_k)
    validate_fusion_method(fusion_method)
    validate_rrf_k(rrf_k)
    vector_limit = _candidate_pool_size(top_k, vector_top_k, max_top_k=max_top_k)
    bm25_limit = _candidate_pool_size(top_k, bm25_top_k, max_top_k=max_top_k)

    started = time.perf_counter()
    vector_results = retrieve_vector(
        session,
        embedding_provider,
        query,
        top_k=vector_limit,
        max_top_k=max_top_k,
        document_id=document_id,
        document_type=document_type,
    )
    bm25_results = retrieve_bm25(
        session,
        query,
        top_k=bm25_limit,
        max_top_k=max_top_k,
        document_id=document_id,
        document_type=document_type,
        index=index,
        k1=k1,
        b=b,
    )

    if fusion_method == "rrf":
        with traced_stage("hybrid_fusion"):
            fused = fuse_rrf(vector_results, bm25_results, rrf_k=rrf_k)
    else:  # validated above; guard against future bypass
        raise RetrievalError(f"unsupported fusion_method: {fusion_method!r}")

    results = [
        RetrievalResult(
            chunk_id=candidate.result.chunk_id,
            document_id=candidate.result.document_id,
            chunk_index=candidate.result.chunk_index,
            text=candidate.result.text,
            score=candidate.fused_score,
            rank=rank,
            document_title=candidate.result.document_title,
            document_type=candidate.result.document_type,
            document_source=candidate.result.document_source,
            page=candidate.result.page,
            section=candidate.result.section,
            chunk_metadata=candidate.result.chunk_metadata,
            retrieval_method=HYBRID_METHOD,
        )
        for rank, candidate in enumerate(fused[:limit], start=1)
    ]
    count(COUNTER_RETRIEVAL_HYBRID, len(fused))
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "hybrid retrieval finished",
        extra={
            "stage": "hybrid_retrieval",
            "method": fusion_method,
            "candidate_count": len(fused),
            "selected_count": len(results),
            "duration_ms": elapsed_ms,
        },
    )
    return results
