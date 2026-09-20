"""Ingestion service — Milestone 3.

Internal pipeline (no public upload API in M3):

    raw bytes → parse → normalize → hash → idempotency check
              → chunk → persist Document + Chunks

Idempotency: ``content_hash = sha256("<type>\\n<normalized text>")``.
The type prefix keeps a PDF and a TXT with identical extracted text
from colliding. Duplicates return the existing rows with
``duplicate=True`` and write nothing. No embeddings (M4), no retrieval.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.chunking import chunk_units
from deepresearch.models import Chunk, Document
from deepresearch.parsing import (
    ParsingError,  # re-exported for callers/tests
    normalize_document_type,
    parse_bytes,
)

TOKENIZER_NAME = "deepresearch-regex-v1"

_FILENAME_SUFFIXES = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "txt",
    ".html": "html",
    ".htm": "html",
}


@dataclass(frozen=True)
class IngestionResult:
    document: Document
    chunks: list[Chunk]
    duplicate: bool


def compute_content_hash(*, document_type: str, normalized_text: str) -> str:
    canonical = f"{normalize_document_type(document_type)}\n{normalized_text}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def type_from_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    try:
        return _FILENAME_SUFFIXES[suffix]
    except KeyError:
        raise ParsingError(f"cannot infer document_type from filename: {filename!r}") from None


def ingest_bytes(
    session: Session,
    *,
    raw: bytes,
    document_type: str,
    source: str,
    title: str | None = None,
    metadata: dict | None = None,
    target_tokens: int = 800,
    overlap_tokens: int = 120,
) -> IngestionResult:
    parsed = parse_bytes(raw, document_type=document_type, source=source)
    doc_type = parsed.document_type
    content_hash = compute_content_hash(
        document_type=doc_type, normalized_text=parsed.normalized_text
    )

    existing = repository.get_document_by_hash(session, content_hash)
    if existing is not None:
        return IngestionResult(
            document=existing,
            chunks=repository.list_chunks_by_document(session, existing.id),
            duplicate=True,
        )

    drafts = chunk_units(parsed.units, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    resolved_title = title if title is not None else parsed.title
    try:
        doc = repository.create_document(
            session,
            title=resolved_title,
            source=source,
            content_hash=content_hash,
            document_type=doc_type,
            metadata=metadata,
        )
        chunks: list[Chunk] = []
        for index, draft in enumerate(drafts):
            chunks.append(
                repository.create_chunk(
                    session,
                    document_id=doc.id,
                    text=draft.text,
                    chunk_index=index,
                    section=draft.section,
                    page=draft.page,
                    metadata={"tokenizer": TOKENIZER_NAME, "token_count": draft.token_count},
                )
            )
        session.commit()
    except IntegrityError:
        # Lost a race with a concurrent ingest of the same content:
        # roll back and return the winner's rows instead of failing.
        session.rollback()
        winner = repository.get_document_by_hash(session, content_hash)
        if winner is None:
            raise
        return IngestionResult(
            document=winner,
            chunks=repository.list_chunks_by_document(session, winner.id),
            duplicate=True,
        )
    return IngestionResult(document=doc, chunks=chunks, duplicate=False)


def ingest_file(
    session: Session,
    *,
    path: str | Path,
    document_type: str | None = None,
    title: str | None = None,
    metadata: dict | None = None,
    target_tokens: int = 800,
    overlap_tokens: int = 120,
) -> IngestionResult:
    file_path = Path(path)
    raw = file_path.read_bytes()
    doc_type = document_type or type_from_filename(file_path.name)
    return ingest_bytes(
        session,
        raw=raw,
        document_type=doc_type,
        source=str(file_path),
        title=title,
        metadata=metadata,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )
