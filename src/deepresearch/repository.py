"""Repository / data-access layer — Milestone 2.

Thin, explicit helpers over SQLAlchemy sessions. No query builder
abstraction, no generic base class: each function maps to one
persistence operation the ingestion pipeline (M3+) will need.

Idempotency contract: ``create_document`` raises
``sqlalchemy.exc.IntegrityError`` on a duplicate ``content_hash``;
callers (M3 ingestion) translate that into "already ingested".
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
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
