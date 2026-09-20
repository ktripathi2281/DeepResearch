"""M3 unit tests — parsing, hashing, chunking, ingestion (SQLite, no network).

PostgreSQL persistence/idempotency against the real DDL lives in
tests/test_ingestion_postgres.py (skipped without a live DB).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.chunking import ChunkingError, chunk_units, detokenize, tokenize
from deepresearch.config import Settings
from deepresearch.db import init_db
from deepresearch.ingestion import (
    ParsingError,
    compute_content_hash,
    ingest_bytes,
    ingest_file,
    type_from_filename,
)
from deepresearch.parsing import TextUnit, parse_bytes

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enforce_fk(dbapi_conn, _record):  # type: ignore[no-untyped-def]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    init_db(engine)
    sess = Session(bind=engine)
    try:
        yield sess
    finally:
        sess.close()


# --- extraction -----------------------------------------------------------


def test_txt_extraction() -> None:
    parsed = parse_bytes(_read("sample.txt"), document_type="txt")
    assert parsed.document_type == "txt"
    assert len(parsed.units) == 3
    assert all(u.page is None and u.section is None for u in parsed.units)
    assert "plain text documents" in parsed.normalized_text


def test_markdown_extraction_preserves_sections() -> None:
    parsed = parse_bytes(_read("sample.md"), document_type="md")
    assert parsed.document_type == "markdown"
    assert parsed.title == "DeepResearch Sample Document"
    sections = {u.section for u in parsed.units}
    assert sections == {"Retrieval", "Reranking"}
    assert all(u.page is None for u in parsed.units)


def test_html_extraction_strips_scripts() -> None:
    parsed = parse_bytes(_read("sample.html"), document_type="html")
    assert parsed.title == "Sample Research Note"
    assert not any("IGNORE ALL PREVIOUS" in u.text for u in parsed.units)
    assert not any("<script" in u.text for u in parsed.units)
    sections = {u.section for u in parsed.units}
    assert {"Background", "Method"} <= sections
    assert "Evidence-first generation" in parsed.normalized_text


def test_pdf_extraction_preserves_pages() -> None:
    parsed = parse_bytes(_read("sample.pdf"), document_type="pdf", source="sample.pdf")
    assert parsed.document_type == "pdf"
    assert len(parsed.units) == 2
    assert [u.page for u in parsed.units] == [1, 2]
    assert "page one about retrieval" in parsed.units[0].text
    assert "page two about reranking" in parsed.units[1].text


# --- hashing --------------------------------------------------------------


def test_content_hash_deterministic_and_content_based() -> None:
    raw = _read("sample.txt")
    h1 = compute_content_hash(document_type="txt", normalized_text="hello")
    h2 = compute_content_hash(document_type="txt", normalized_text="hello")
    assert h1 == h2 and len(h1) == 64
    # Same text, different container type → different hash (no cross-type collision).
    assert compute_content_hash(document_type="txt", normalized_text="hello") != (
        compute_content_hash(document_type="markdown", normalized_text="hello")
    )
    # Filename plays no role: identical bytes from two sources share a hash.
    t1 = parse_bytes(raw, document_type="txt").normalized_text
    t2 = parse_bytes(raw, document_type="txt").normalized_text
    assert compute_content_hash(document_type="txt", normalized_text=t1) == (
        compute_content_hash(document_type="txt", normalized_text=t2)
    )


def test_type_from_filename() -> None:
    assert type_from_filename("a.pdf") == "pdf"
    assert type_from_filename("a.MD") == "markdown"
    assert type_from_filename("a.htm") == "html"
    with pytest.raises(ParsingError):
        type_from_filename("a.exe")


# --- chunking -------------------------------------------------------------


def _word_units(n: int) -> list[TextUnit]:
    return [TextUnit(text=f"word{i}") for i in range(n)]


def test_chunking_deterministic() -> None:
    units = _word_units(50)
    first = chunk_units(units, target_tokens=10, overlap_tokens=2)
    second = chunk_units(units, target_tokens=10, overlap_tokens=2)
    assert [c.text for c in first] == [c.text for c in second]
    assert [c.token_count for c in first] == [c.token_count for c in second]


def test_configurable_target_and_overlap() -> None:
    units = _word_units(50)
    small = chunk_units(units, target_tokens=10, overlap_tokens=0)
    large = chunk_units(units, target_tokens=25, overlap_tokens=0)
    assert len(small) == 5
    assert len(large) == 2
    assert all(c.token_count <= 10 for c in small)
    assert all(c.token_count <= 25 for c in large)
    # Overlap shares exact tokens across the boundary for full windows.
    overlapped = chunk_units(units, target_tokens=10, overlap_tokens=2)
    assert len(overlapped) == 6
    for prev, nxt in zip(overlapped, overlapped[1:], strict=False):
        assert tokenize(prev.text)[-2:] == tokenize(nxt.text)[:2]


def test_no_empty_chunks_and_sequential_windows() -> None:
    units = [TextUnit(text="   "), TextUnit(text="real content here"), TextUnit(text="")]
    drafts = chunk_units(units, target_tokens=800, overlap_tokens=120)
    assert len(drafts) == 1
    assert drafts[0].text == "real content here"
    assert chunk_units([], target_tokens=800, overlap_tokens=120) == []


def test_tokenizer_round_trip() -> None:
    tokens = tokenize("Hello, world! (test)")
    assert tokens == ["Hello", ",", "world", "!", "(", "test", ")"]
    assert tokenize(detokenize(tokens)) == tokens


def test_invalid_chunk_config_rejected() -> None:
    with pytest.raises(ChunkingError):
        chunk_units(_word_units(5), target_tokens=0, overlap_tokens=0)
    with pytest.raises(ChunkingError):
        chunk_units(_word_units(5), target_tokens=10, overlap_tokens=10)
    with pytest.raises(ChunkingError):
        chunk_units(_word_units(5), target_tokens=10, overlap_tokens=11)


# --- ingestion persistence / idempotency ----------------------------------


def test_ingest_persists_document_and_chunks(session: Session) -> None:
    result = ingest_bytes(
        session, raw=_read("sample.md"), document_type="markdown", source="sample.md"
    )
    assert result.duplicate is False
    assert result.document.title == "DeepResearch Sample Document"
    assert result.document.document_type == "markdown"
    assert len(result.chunks) >= 1
    assert [c.chunk_index for c in result.chunks] == list(range(len(result.chunks)))
    assert all(c.text.strip() for c in result.chunks)
    assert all(c.section in {"Retrieval", "Reranking"} for c in result.chunks)
    assert result.chunks[0].chunk_metadata["tokenizer"] == "deepresearch-regex-v1"
    # Repository round-trip sees the same rows.
    assert repository.get_document_by_hash(session, result.document.content_hash) is not None


def test_ingest_is_idempotent(session: Session) -> None:
    raw = _read("sample.txt")
    first = ingest_bytes(session, raw=raw, document_type="txt", source="sample.txt")
    count_before = len(repository.list_chunks_by_document(session, first.document.id))
    second = ingest_bytes(session, raw=raw, document_type="txt", source="renamed-copy.txt")
    assert second.duplicate is True
    assert second.document.id == first.document.id
    assert len(second.chunks) == count_before
    assert len(repository.list_chunks_by_document(session, first.document.id)) == count_before


def test_ingest_preserves_pdf_pages(session: Session) -> None:
    result = ingest_bytes(
        session,
        raw=_read("sample.pdf"),
        document_type="pdf",
        source="sample.pdf",
        target_tokens=800,
        overlap_tokens=0,
    )
    assert len(result.chunks) == 1
    assert result.chunks[0].page == 1


def test_empty_document_persists_without_chunks(session: Session) -> None:
    result = ingest_bytes(session, raw=b"  \n  ", document_type="txt", source="empty.txt")
    assert result.duplicate is False
    assert result.chunks == []
    again = ingest_bytes(session, raw=b"", document_type="txt", source="empty.txt")
    assert again.duplicate is True
    assert again.document.id == result.document.id


def test_settings_chunk_defaults() -> None:
    settings = Settings()
    assert settings.chunk_target_tokens == 800
    assert settings.chunk_overlap_tokens == 120


def test_ingest_file_infers_type(session: Session) -> None:
    result = ingest_file(session, path=FIXTURES / "sample.txt")
    assert result.document.document_type == "txt"
    assert len(result.chunks) >= 1


# --- malformed input ------------------------------------------------------


def test_malformed_and_unsupported_inputs_raise() -> None:
    with pytest.raises(ParsingError):
        parse_bytes(b"%PDF-1.4 not a real pdf", document_type="pdf", source="bad.pdf")
    with pytest.raises(ParsingError):
        parse_bytes(b"hello", document_type="docx")
    with pytest.raises(ParsingError):
        parse_bytes(b"\xff\xfe\x00bad", document_type="txt")
    with pytest.raises(ParsingError):
        parse_bytes(b"", document_type="pdf", source="empty.pdf")
