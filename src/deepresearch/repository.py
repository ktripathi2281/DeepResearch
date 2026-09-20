"""Repository / data-access layer — Milestones 2–5.

Thin, explicit helpers over SQLAlchemy sessions. No query builder
abstraction, no generic base class: each function maps to one
persistence operation the pipeline needs.

Idempotency contract: ``create_document`` raises
``sqlalchemy.exc.IntegrityError`` on a duplicate ``content_hash``;
callers (M3 ingestion) translate that into "already ingested".
"""

from __future__ import annotations

import uuid

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from deepresearch.models import Chunk, Document


def create_document(
    session: Session,
    *,
    title: str | None,
    source: str,
    content_hash: str,
    document_type: str,
    metadata: dict | None = None,
) -> Document:
    doc = Document(
        title=title,
        source=source,
        content_hash=content_hash,
        document_type=document_type,
        doc_metadata=metadata,
    )
    session.add(doc)
    session.flush()  # surface IntegrityError here, not at commit
    return doc


def get_document_by_id(session: Session, document_id: uuid.UUID) -> Document | None:
    return session.get(Document, document_id)


def get_document_by_hash(session: Session, content_hash: str) -> Document | None:
    return session.scalar(select(Document).where(Document.content_hash == content_hash))


def list_documents(session: Session, *, limit: int = 100, offset: int = 0) -> list[Document]:
    return list(
        session.scalars(select(Document).order_by(Document.created_at).limit(limit).offset(offset))
    )


def create_chunk(
    session: Session,
    *,
    document_id: uuid.UUID,
    text: str,
    chunk_index: int,
    section: str | None = None,
    page: int | None = None,
    metadata: dict | None = None,
) -> Chunk:
    chunk = Chunk(
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
        section=section,
        page=page,
        chunk_metadata=metadata,
    )
    session.add(chunk)
    session.flush()
    return chunk


def get_chunk_by_id(session: Session, chunk_id: uuid.UUID) -> Chunk | None:
    return session.get(Chunk, chunk_id)


def list_chunks_by_document(session: Session, document_id: uuid.UUID) -> list[Chunk]:
    return list(
        session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        )
    )


def list_chunks_missing_embeddings(session: Session, *, limit: int | None = None) -> list[Chunk]:
    """Chunks with no vector yet, oldest first (M4 embedding candidates)."""
    query = select(Chunk).where(Chunk.embedding.is_(None)).order_by(Chunk.created_at)
    if limit is not None:
        query = query.limit(limit)
    return list(session.scalars(query))


def list_chunks_with_stale_embeddings(
    session: Session, *, model_name: str, model_version: str, limit: int | None = None
) -> list[Chunk]:
    """Embedded chunks whose model/version differs from the configured provider.

    Returned for reporting and explicit ``force`` re-embedding; never
    silently mixed with current-model vectors (see ADR-004).
    """
    query = (
        select(Chunk)
        .where(Chunk.embedding.is_not(None))
        .where((Chunk.embedding_model != model_name) | (Chunk.embedding_version != model_version))
        .order_by(Chunk.created_at)
    )
    if limit is not None:
        query = query.limit(limit)
    return list(session.scalars(query))


def count_chunks_with_embeddings(session: Session, *, model_name: str, model_version: str) -> int:
    """Chunks already embedded with the given model/version (skipped by M4 reruns)."""
    query = (
        select(Chunk)
        .where(Chunk.embedding.is_not(None))
        .where(Chunk.embedding_model == model_name)
        .where(Chunk.embedding_version == model_version)
    )
    return len(list(session.scalars(query)))


def search_chunks_by_vector(
    session: Session,
    *,
    query_vector: list[float],
    model_name: str,
    model_version: str,
    limit: int,
    document_id: uuid.UUID | None = None,
    document_type: str | None = None,
) -> list[tuple[Chunk, Document, float]]:
    """Cosine-similarity search over embedded chunks (M5, PostgreSQL + pgvector).

    Ranking executes in the database via the native ``<=>`` operator;
    embeddings are never loaded into Python for scoring. Only chunks
    embedded with the given model/version are comparable, so the filter
    is always applied — incompatible vectors are never mixed. Ties
    break deterministically on ``Chunk.id``. Returns
    ``(chunk, document, similarity)`` with similarity ``1 - distance``
    (higher = more similar), ordered best-first.
    """
    distance = Chunk.embedding.cosine_distance(query_vector)
    similarity = (1 - distance).label("similarity")
    query = (
        select(Chunk, Document, similarity)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.embedding.is_not(None))
        .where(Chunk.embedding_model == model_name)
        .where(Chunk.embedding_version == model_version)
    )
    if document_id is not None:
        query = query.where(Chunk.document_id == document_id)
    if document_type is not None:
        query = query.where(Document.document_type == document_type)
    query = query.order_by(desc(similarity), Chunk.id.asc()).limit(limit)
    return [(chunk, doc, float(score)) for chunk, doc, score in session.execute(query).all()]
