"""Lexical BM25 retrieval — Milestone 6.

Path: text query → normalize/tokenize → in-process Okapi BM25 index
(built deterministically from persisted chunks) → ranked
``RetrievalResult`` list with ``method="bm25"``.

Fully independent from M5 vector retrieval: separate module, separate
index, separate scores. No fusion here (M7), no reranking, no external
search service. Zero new dependencies — Okapi BM25 is ~60 lines and
stays readable.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.chunking import tokenize
from deepresearch.logging import get_logger
from deepresearch.retrieval import (
    MAX_TOP_K,
    RetrievalError,
    RetrievalResult,
    validate_top_k,
)

logger = get_logger(__name__)

BM25_METHOD = "bm25"
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class StaleIndexError(RetrievalError):
    """A provided BM25 index no longer matches the persisted corpus."""


def normalize_tokens(text: str) -> list[str]:
    """Tokenize query/corpus text identically: M3 regex tokens, lowercased.

    Pure-punctuation tokens (``,``, ``!`` …) are dropped — they carry no
    lexical signal and would only bloat document frequencies. No
    stemming or lemmatization: exact lexical behavior stays obvious.
    """
    return [t.lower() for t in tokenize(text) if any(ch.isalnum() for ch in t)]


@dataclass
class BM25Index:
    """In-process Okapi BM25 index over a snapshot of persisted chunks.

    ``chunk_ids``/``doc_freqs``/``doc_lens`` are parallel lists in
    deterministic build order (``Chunk.id`` ascending). ``filters``
    records the corpus scope the index was built for. Chunks are
    immutable (ADR-001), so freshness reduces to chunk id-set equality.
    """

    chunk_ids: list[uuid.UUID] = field(default_factory=list)
    doc_freqs: list[Counter] = field(default_factory=list)
    doc_lens: list[int] = field(default_factory=list)
    term_doc_count: Counter = field(default_factory=Counter)
    avg_length: float = 0.0
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    document_id: uuid.UUID | None = None
    document_type: str | None = None

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def _idf(self, term: str) -> float:
        # Lucene-style positive IDF: never negative, new terms score 0.
        df = self.term_doc_count.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.size - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        """Score every indexed doc; returns ``(position, score)`` desc, ties by position."""
        if not self.chunk_ids or not query_tokens:
            return []
        query_terms = Counter(query_tokens)
        ranked: list[tuple[int, float]] = []
        for pos, (freqs, length) in enumerate(zip(self.doc_freqs, self.doc_lens, strict=True)):
            total = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                denom = tf + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                total += idf * tf * (self.k1 + 1) / denom
            if total > 0:
                ranked.append((pos, total))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def is_fresh(
        self,
        session: Session,
        *,
        document_id: uuid.UUID | None = None,
        document_type: str | None = None,
    ) -> bool:
        """True when the persisted corpus (same scope) still matches this snapshot."""
        if document_id != self.document_id or document_type != self.document_type:
            return False
        current = repository.list_chunk_ids_for_freshness(
            session, document_id=document_id, document_type=document_type
        )
        return current == self.chunk_ids


def build_index(
    entries: list[tuple[uuid.UUID, list[str]]],
    *,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    document_id: uuid.UUID | None = None,
    document_type: str | None = None,
) -> BM25Index:
    """Build an index from ``(chunk_id, normalized tokens)`` pairs (pure, deterministic)."""
    if k1 <= 0:
        raise RetrievalError(f"k1 must be > 0, got {k1}")
    if not 0 <= b <= 1:
        raise RetrievalError(f"b must be in [0, 1], got {b}")
    index = BM25Index(k1=k1, b=b, document_id=document_id, document_type=document_type)
    total_length = 0
    for chunk_id, tokens in entries:
        freqs = Counter(tokens)
        index.chunk_ids.append(chunk_id)
        index.doc_freqs.append(freqs)
        index.doc_lens.append(len(tokens))
        total_length += len(tokens)
        for term in freqs:
            index.term_doc_count[term] += 1
    index.avg_length = total_length / len(entries) if entries else 0.0
    return index


def build_index_from_session(
    session: Session,
    *,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    document_id: uuid.UUID | None = None,
    document_type: str | None = None,
) -> tuple[BM25Index, dict[uuid.UUID, tuple]]:
    """Load the corpus from PostgreSQL and build a fresh index plus row map."""
    rows = repository.list_chunks_for_bm25(
        session, document_id=document_id, document_type=document_type
    )
    entries = [(chunk.id, normalize_tokens(chunk.text)) for chunk, _ in rows]
    index = build_index(entries, k1=k1, b=b, document_id=document_id, document_type=document_type)
    return index, {chunk.id: (chunk, doc) for chunk, doc in rows}


def retrieve_bm25(
    session: Session,
    query: str,
    *,
    top_k: int = 5,
    max_top_k: int = MAX_TOP_K,
    document_id: uuid.UUID | None = None,
    document_type: str | None = None,
    index: BM25Index | None = None,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> list[RetrievalResult]:
    """Lexical top-K retrieval over persisted chunks.

    With ``index=None`` a fresh index is built from the current corpus
    (simple, always correct). A provided index is used as-is but must
    be fresh for the same filter scope, else ``StaleIndexError`` —
    staleness is explicit, never silent.
    """
    if not query or not query.strip():
        raise RetrievalError("query must be non-empty text")
    limit = validate_top_k(top_k, maximum=max_top_k)
    query_tokens = normalize_tokens(query)
    if not query_tokens:
        return []  # punctuation-only query: content but no indexable terms

    started = time.perf_counter()
    if index is None:
        index, row_map = build_index_from_session(
            session, k1=k1, b=b, document_id=document_id, document_type=document_type
        )
    else:
        if not index.is_fresh(session, document_id=document_id, document_type=document_type):
            raise StaleIndexError(
                "provided BM25 index is stale for this corpus/scope; "
                "call refresh() or omit index to rebuild"
            )
        row_map = repository.get_chunks_with_documents_by_ids(
            session, [index.chunk_ids[pos] for pos, _ in index.score(query_tokens)[:limit]]
        )
        # Re-derive rows for winners only; metadata stays fresh from the DB.
    scored = index.score(query_tokens)[:limit]
    results = [
        RetrievalResult(
            chunk_id=row_map[index.chunk_ids[pos]][0].id,
            document_id=row_map[index.chunk_ids[pos]][0].document_id,
            chunk_index=row_map[index.chunk_ids[pos]][0].chunk_index,
            text=row_map[index.chunk_ids[pos]][0].text,
            score=score,
            rank=rank,
            document_title=row_map[index.chunk_ids[pos]][1].title,
            document_type=row_map[index.chunk_ids[pos]][1].document_type,
            document_source=row_map[index.chunk_ids[pos]][1].source,
            page=row_map[index.chunk_ids[pos]][0].page,
            section=row_map[index.chunk_ids[pos]][0].section,
            chunk_metadata=row_map[index.chunk_ids[pos]][0].chunk_metadata,
            retrieval_method=BM25_METHOD,
        )
        for rank, (pos, score) in enumerate(scored, start=1)
    ]
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "bm25 retrieval finished",
        extra={
            "stage": "bm25_retrieval",
            "method": BM25_METHOD,
            "candidate_count": len(results),
            "selected_count": len(results),
            "duration_ms": elapsed_ms,
        },
    )
    return results


class BM25Retriever:
    """Persistent in-process BM25 retriever with explicit refresh.

    Holds one index snapshot; ``retrieve`` auto-refreshes when no index
    exists or the corpus changed, so repeated queries pay the rebuild
    only when data actually changed. Refreshes are logged, never silent.
    """

    def __init__(self, *, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self._k1 = k1
        self._b = b
        self._index: BM25Index | None = None

    @property
    def index(self) -> BM25Index | None:
        return self._index

    def refresh(self, session: Session) -> int:
        """Rebuild the index from the current corpus; returns chunk count."""
        rows = repository.list_chunks_for_bm25(session)
        entries = [(chunk.id, normalize_tokens(chunk.text)) for chunk, _ in rows]
        self._index = build_index(entries, k1=self._k1, b=self._b)
        logger.info(
            "bm25 index refreshed",
            extra={
                "stage": "bm25_retrieval",
                "method": BM25_METHOD,
                "selected_count": len(entries),
            },
        )
        return len(entries)

    def retrieve(
        self,
        session: Session,
        query: str,
        *,
        top_k: int = 5,
        max_top_k: int = MAX_TOP_K,
        document_id: uuid.UUID | None = None,
        document_type: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve with the held index, refreshing first when stale/missing."""
        if (
            self._index is None
            or document_id is not None
            or document_type is not None
            or not self._index.is_fresh(session)
        ):
            # Filtered scopes always build ad hoc; the held index covers
            # the unfiltered corpus only. Both paths are explicit.
            if document_id is None and document_type is None:
                self.refresh(session)
                index = self._index
            else:
                return retrieve_bm25(
                    session,
                    query,
                    top_k=top_k,
                    max_top_k=max_top_k,
                    document_id=document_id,
                    document_type=document_type,
                    k1=self._k1,
                    b=self._b,
                )
        else:
            index = self._index
        return retrieve_bm25(
            session,
            query,
            top_k=top_k,
            max_top_k=max_top_k,
            index=index,
            k1=self._k1,
            b=self._b,
        )
