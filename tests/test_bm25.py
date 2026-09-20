"""M6 unit tests — BM25 tokenization, scoring, lifecycle (SQLite end-to-end).

BM25 is pure Python, so the full path (persist → index → retrieve)
runs on SQLite. PostgreSQL parity lives in test_bm25_postgres.py.
No network, no model, no GPU.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.bm25 import (
    BM25_METHOD,
    BM25Retriever,
    StaleIndexError,
    build_index,
    build_index_from_session,
    normalize_tokens,
    retrieve_bm25,
)
from deepresearch.config import Settings
from deepresearch.db import init_db
from deepresearch.retrieval import RetrievalError


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


def _seed(
    session: Session,
    texts: list[str],
    *,
    source: str = "bm25-seed.md",
    document_type: str = "markdown",
    title: str = "BM25 Doc",
) -> uuid.UUID:
    doc = repository.create_document(
        session,
        title=title,
        source=source,
        content_hash=f"m6-{uuid.uuid4().hex}",
        document_type=document_type,
        metadata=None,
    )
    for index, chunk_text in enumerate(texts):
        repository.create_chunk(
            session,
            document_id=doc.id,
            text=chunk_text,
            chunk_index=index,
            section="Sec" if index == 0 else None,
            page=1 if index == 0 else None,
            metadata={"seed": True},
        )
    session.commit()
    return doc.id


def test_normalizer_lowercases_and_drops_punctuation() -> None:
    assert normalize_tokens("Hello, WORLD!") == ["hello", "world"]
    assert normalize_tokens("!!! ...") == []
    assert normalize_tokens("  spaced   out  ") == ["spaced", "out"]
    assert normalize_tokens("model-v2 test") == ["model", "v2", "test"]


def test_query_corpus_tokenization_consistent(session: Session) -> None:
    _seed(session, ["Hybrid Retrieval Systems"])
    results = retrieve_bm25(session, "HYBRID, retrieval!")
    assert [r.text for r in results] == ["Hybrid Retrieval Systems"]


def test_index_construction_stats() -> None:
    index = build_index(
        [(uuid.uuid4(), ["a", "b", "a"]), (uuid.uuid4(), ["b", "c"])],
    )
    assert index.size == 2
    assert index.avg_length == pytest.approx(2.5)
    assert index.term_doc_count["a"] == 1
    assert index.term_doc_count["b"] == 2
    with pytest.raises(RetrievalError):
        build_index([], k1=0)
    with pytest.raises(RetrievalError):
        build_index([], b=1.5)


def test_exact_match_ranked_first(session: Session) -> None:
    _seed(
        session,
        [
            "totally unrelated weather report",
            "pgvector cosine similarity search",
            "another document about cooking recipes",
        ],
    )
    results = retrieve_bm25(session, "pgvector cosine similarity")
    assert results[0].text == "pgvector cosine similarity search"
    assert all(a.score >= b.score for a, b in zip(results, results[1:], strict=False))


def test_partial_match_more_terms_win(session: Session) -> None:
    _seed(session, ["retrieval systems overview", "retrieval reranking fusion overview"])
    results = retrieve_bm25(session, "retrieval reranking fusion")
    assert results[0].text == "retrieval reranking fusion overview"


def test_irrelevant_chunks_excluded(session: Session) -> None:
    _seed(session, ["quantum chromodynamics", "hybrid retrieval fusion"])
    results = retrieve_bm25(session, "retrieval")
    assert [r.text for r in results] == ["hybrid retrieval fusion"]


def test_no_match_returns_empty(session: Session) -> None:
    _seed(session, ["alpha beta gamma"])
    assert retrieve_bm25(session, "xylophone zebra") == []


def test_blank_query_rejected(session: Session) -> None:
    for bad in ("", "   "):
        with pytest.raises(RetrievalError):
            retrieve_bm25(session, bad)


def test_punctuation_only_query_returns_empty(session: Session) -> None:
    _seed(session, ["alpha beta"])
    assert retrieve_bm25(session, "!!! ... ,,,") == []


def test_repeated_query_terms_are_deterministic(session: Session) -> None:
    # Plain Okapi BM25 sums over unique query terms (no query-tf weight),
    # so repetition must not change the score — only remain deterministic.
    _seed(session, ["retrieval systems", "other words here"])
    single = retrieve_bm25(session, "retrieval")
    repeated = retrieve_bm25(session, "retrieval retrieval retrieval")
    assert [r.chunk_id for r in repeated] == [r.chunk_id for r in single]
    assert repeated[0].score == pytest.approx(single[0].score)


def test_repeated_chunk_terms_score_higher(session: Session) -> None:
    _seed(session, ["retrieval once here", "retrieval retrieval retrieval retrieval"])
    results = retrieve_bm25(session, "retrieval")
    assert results[0].text == "retrieval retrieval retrieval retrieval"


def test_top_k_validation_and_limit(session: Session) -> None:
    _seed(session, ["cat", "cat dog", "cat dog bird"])
    assert len(retrieve_bm25(session, "cat", top_k=2)) == 2
    for bad in (0, -1, 101):
        with pytest.raises(RetrievalError):
            retrieve_bm25(session, "cat", top_k=bad)


def test_ties_break_deterministically(session: Session) -> None:
    _seed(session, ["identical duplicate text", "identical duplicate text"])
    first = retrieve_bm25(session, "identical duplicate")
    second = retrieve_bm25(session, "identical duplicate")
    assert len(first) == 2
    assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
    assert [r.rank for r in first] == [1, 2]
    assert first[0].score == pytest.approx(first[1].score)


def test_metadata_mapping(session: Session) -> None:
    _seed(session, ["retrieval chunk"], title="Meta Title")
    (hit,) = retrieve_bm25(session, "retrieval")
    assert hit.document_title == "Meta Title"
    assert hit.document_type == "markdown"
    assert hit.document_source == "bm25-seed.md"
    assert hit.page == 1
    assert hit.section == "Sec"
    assert hit.chunk_metadata == {"seed": True}
    assert hit.retrieval_method == BM25_METHOD == "bm25"
    assert hit.chunk_index == 0


def test_document_id_filter(session: Session) -> None:
    wanted = _seed(session, ["retrieval alpha"], source="wanted.md")
    _seed(session, ["retrieval beta"], source="other.md")
    results = retrieve_bm25(session, "retrieval", document_id=wanted)
    assert [r.text for r in results] == ["retrieval alpha"]
    assert retrieve_bm25(session, "retrieval", document_id=uuid.uuid4()) == []


def test_document_type_filter(session: Session) -> None:
    _seed(session, ["retrieval md"], source="a.md", document_type="markdown")
    _seed(session, ["retrieval txt"], source="b.txt", document_type="txt")
    results = retrieve_bm25(session, "retrieval", document_type="txt")
    assert [r.text for r in results] == ["retrieval txt"]


def test_empty_corpus_returns_empty(session: Session) -> None:
    assert retrieve_bm25(session, "anything") == []


def test_short_chunks_retrievable(session: Session) -> None:
    _seed(session, ["up", "down"])
    assert [r.text for r in retrieve_bm25(session, "up")] == ["up"]


def test_k1_b_configuration_changes_scores() -> None:
    entries = [(uuid.uuid4(), ["a"] * 10), (uuid.uuid4(), ["a"])]
    default = build_index(entries)
    flat = build_index(entries, b=0.0)
    assert default.score(["a"])[0] != flat.score(["a"])[0]


def test_bm25_config_defaults() -> None:
    settings = Settings()
    assert (settings.bm25_k1, settings.bm25_b) == (1.5, 0.75)


def test_index_refresh_lifecycle(session: Session) -> None:
    retriever = BM25Retriever()
    _seed(session, ["first chunk words"], source=" evolving.md")
    assert retriever.refresh(session) == 1
    assert [r.text for r in retriever.retrieve(session, "first")] == ["first chunk words"]

    # New chunk without refresh: held index is stale and must say so.
    _seed(session, ["second chunk words"], source="evolving.md")
    assert retriever.index is not None and not retriever.index.is_fresh(session)
    assert [r.text for r in retriever.retrieve(session, "second")] == ["second chunk words"]
    assert retriever.index is not None and retriever.index.is_fresh(session)

    # A hand-held stale index passed explicitly raises instead of lying.
    stale, _ = build_index_from_session(session)
    _seed(session, ["third chunk words"], source="evolving.md")
    with pytest.raises(StaleIndexError):
        retrieve_bm25(session, "third", index=stale)


def test_rebuild_is_deterministic(session: Session) -> None:
    _seed(session, ["retrieval one", "retrieval two three", "unrelated"])
    first = retrieve_bm25(session, "retrieval")
    second = retrieve_bm25(session, "retrieval")
    assert [(r.chunk_id, r.score) for r in first] == [(r.chunk_id, r.score) for r in second]
