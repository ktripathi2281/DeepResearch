"""SQLAlchemy ORM models — Milestone 2.

Persistence foundation for the future pipeline:

    document → chunks → embeddings/retrieval → evidence → generation

Scope (M2 only):
- ``documents`` and ``chunks`` tables with the fields required by
  FR-1/FR-2/FR-3 and ARCHITECTURE §5 (Document, Chunk).
- pgvector ``VECTOR(384)`` column present as *schema capability only*;
  no embedding code populates it until Milestone 4.
- No ingestion, retrieval, BM25, reranking, LLM, agent, eval, or UI code.

Deferred (see docs/adr/001-*):
- research_requests / evaluation_cases / observability tables.
  They are justified only when the agent (M14), observability (M15),
  and evaluation (M16) milestones arrive.
"""

from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    """An ingested source file (PDF/MD/TXT/HTML).

    ``content_hash`` (sha256 hex) is the idempotency key: ingestion must
    refuse a second row with the same hash (M3 enforces this via the
    UNIQUE constraint defined here). The PK is a UUID so chunk/evidence
    references never leak ingestion order.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    document_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # Attribute name avoids clashing with DeclarativeBase.metadata;
    # column name stays ``metadata`` to match the architecture docs.
    doc_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Chunk.chunk_index",
        passive_deletes=True,
    )


class Chunk(Base):
    """An ordered text unit belonging to one document.

    ``(document_id, chunk_index)`` is UNIQUE so re-ingestion of the same
    document deterministically reproduces chunk positions. ``section`` /
    ``page`` preserve structure where the parser provides it (nullable
    otherwise). ``embedding`` stays NULL until Milestone 4.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
        Index("ix_chunks_document_id", "document_id"),
        CheckConstraint("chunk_index >= 0", name="ck_chunks_chunk_index_nonnegative"),
        CheckConstraint("page IS NULL OR page >= 1", name="ck_chunks_page_positive"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    # Schema capability for M4/M5 (bge-small-en-v1.5 → 384 dims). NULL in M2.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")
