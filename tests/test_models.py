"""M2 unit tests — documents/chunks on SQLite (deterministic, no Docker).

SQLite stands in for PostgreSQL here: JSON columns and the pgvector
``VECTOR(384)`` type degrade gracefully for DDL, and ``embedding``
stays NULL throughout M2. Real PG DDL (extension, UUID, vector) is
covered by tests/test_postgres_schema.py, which skips without a DB.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.db import init_db


def _sqlite_session() -> Session:
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enforce_fk(dbapi_conn, _record):  # type: ignore[no-untyped-def]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    init_db(engine)
    return Session(bind=engine)


@pytest.fixture
def session() -> Session:
    s = _sqlite_session()
    try:
        yield s
    finally:
        s.close()


def _doc(session: Session, content_hash: str = "a" * 64, **kw) -> object:
    defaults = {
        "title": "Hallucinations survey",
        "source": "corpus/llm-survey.md",
        "document_type": "markdown",
        "metadata": {"corpus": "eval-v1"},
    }
    defaults.update(kw)
    doc = repository.create_document(session, content_hash=content_hash, **defaults)  # type: ignore[arg-type]
    session.commit()
    return doc


def test_document_creation(session: Session) -> None:
    doc = _doc(session)
    fetched = repository.get_document_by_id(session, doc.id)  # type: ignore[attr-defined]
    assert fetched is not None
    assert fetched.title == "Hallucinations survey"
    assert fetched.source == "corpus/llm-survey.md"
    assert fetched.content_hash == "a" * 64
    assert fetched.created_at is not None
    assert repository.get_document_by_hash(session, "a" * 64).id == doc.id  # type: ignore[attr-defined]


def test_chunk_creation_and_ordered_relationship(session: Session) -> None:
    doc = _doc(session)
    repository.create_chunk(
        session,
        document_id=doc.id,
        text="second",
        chunk_index=1,
        section="S2",  # type: ignore[attr-defined]
    )
    repository.create_chunk(
        session,
        document_id=doc.id,  # type: ignore[attr-defined]
        text="first",
        chunk_index=0,
        section="S1",
        page=3,
        metadata={"heading_level": 2},
    )
    session.commit()

    chunks = repository.list_chunks_by_document(session, doc.id)  # type: ignore[attr-defined]
    assert [c.text for c in chunks] == ["first", "second"]
    assert chunks[0].section == "S1"
    assert chunks[0].page == 3
    assert chunks[0].chunk_metadata == {"heading_level": 2}
    assert repository.get_chunk_by_id(session, chunks[1].id).text == "second"


def test_content_hash_uniqueness_enforces_idempotency(session: Session) -> None:
    _doc(session, content_hash="b" * 64)
    with pytest.raises(IntegrityError):
        repository.create_document(
            session,
            title="dup",
            source="elsewhere.md",
            content_hash="b" * 64,
            document_type="markdown",
        )
    session.rollback()
    assert repository.get_document_by_hash(session, "b" * 64) is not None


def test_metadata_and_nullable_structure_fields(session: Session) -> None:
    doc = _doc(session, content_hash="c" * 64, title=None, metadata=None)
    chunk = repository.create_chunk(
        session,
        document_id=doc.id,
        text="plain",
        chunk_index=0,  # type: ignore[attr-defined]
    )
    session.commit()
    assert doc.doc_metadata is None  # type: ignore[attr-defined]
    assert chunk.section is None
    assert chunk.page is None
    assert chunk.chunk_metadata is None
    assert chunk.embedding is None
    assert chunk.embedding_model is None


def test_invalid_document_reference_rejected(session: Session) -> None:
    with pytest.raises(IntegrityError):
        repository.create_chunk(session, document_id=uuid.uuid4(), text="orphan", chunk_index=0)
    session.rollback()


def test_duplicate_chunk_index_rejected(session: Session) -> None:
    doc = _doc(session, content_hash="d" * 64)
    repository.create_chunk(session, document_id=doc.id, text="one", chunk_index=0)  # type: ignore[attr-defined]
    session.commit()
    with pytest.raises(IntegrityError):
        repository.create_chunk(
            session,
            document_id=doc.id,
            text="dup",
            chunk_index=0,  # type: ignore[attr-defined]
        )
    session.rollback()


def test_negative_chunk_index_rejected(session: Session) -> None:
    doc = _doc(session, content_hash="e" * 64)
    with pytest.raises(IntegrityError):
        repository.create_chunk(
            session,
            document_id=doc.id,
            text="neg",
            chunk_index=-1,  # type: ignore[attr-defined]
        )
    session.rollback()


def test_delete_document_cascades_to_chunks(session: Session) -> None:
    doc = _doc(session, content_hash="f" * 64)
    repository.create_chunk(session, document_id=doc.id, text="x", chunk_index=0)  # type: ignore[attr-defined]
    session.commit()
    session.delete(doc)
    session.commit()
    assert repository.list_chunks_by_document(session, doc.id) == []  # type: ignore[attr-defined]
