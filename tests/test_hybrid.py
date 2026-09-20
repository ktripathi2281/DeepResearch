"""M7 unit tests — RRF math, ranking, validation, orchestration (stubbed retrievers).

The M5/M6 implementations run unmodified in production; here they are
stubbed so fusion logic is deterministic and fast. Session is unused by
the stubs. No network, no model, no database.
"""

from __future__ import annotations

import uuid

import pytest

import deepresearch.hybrid as hybrid_mod
from deepresearch.config import Settings
from deepresearch.hybrid import (
    DEFAULT_RRF_K,
    fuse_rrf,
    retrieve_hybrid,
    validate_fusion_method,
    validate_rrf_k,
)
from deepresearch.retrieval import RetrievalError, RetrievalResult


def _result(
    chunk_id: uuid.UUID,
    rank: int,
    method: str,
    *,
    score: float = 0.0,
    text: str = "chunk text",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        chunk_index=0,
        text=text,
        score=score,
        rank=rank,
        document_title="Title",
        document_type="markdown",
        document_source="s.md",
        page=1,
        section="Sec",
        chunk_metadata={"k": "v"},
        retrieval_method=method,
    )


def _ids(n: int) -> list[uuid.UUID]:
    return [uuid.uuid4() for _ in range(n)]


# --- RRF math -------------------------------------------------------------


def test_rrf_vector_only_chunk() -> None:
    (cid,) = _ids(1)
    (fused,) = fuse_rrf([_result(cid, 1, "vector")], [], rrf_k=60)
    assert fused.fused_score == pytest.approx(1 / 61)
    assert fused.result.chunk_id == cid


def test_rrf_bm25_only_chunk() -> None:
    (cid,) = _ids(1)
    (fused,) = fuse_rrf([], [_result(cid, 3, "bm25")], rrf_k=60)
    assert fused.fused_score == pytest.approx(1 / 63)


def test_rrf_both_lists_sum_contributions() -> None:
    (cid,) = _ids(1)
    (fused,) = fuse_rrf([_result(cid, 1, "vector")], [_result(cid, 2, "bm25")], rrf_k=60)
    assert fused.fused_score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_raw_scores_never_mixed() -> None:
    (cid,) = _ids(1)
    (fused,) = fuse_rrf(
        [_result(cid, 1, "vector", score=999.0)],
        [_result(cid, 1, "bm25", score=-500.0)],
        rrf_k=60,
    )
    assert fused.fused_score == pytest.approx(2 / 61)


def test_rrf_custom_k() -> None:
    (cid,) = _ids(1)
    (fused,) = fuse_rrf([_result(cid, 4, "vector")], [], rrf_k=10)
    assert fused.fused_score == pytest.approx(1 / 14)


# --- ranking / dedup ------------------------------------------------------


def test_fused_ordering_and_tie_break() -> None:
    low, high, tie_a, tie_b = _ids(4)
    tie_ids = sorted([tie_a, tie_b])
    fused = fuse_rrf(
        [_result(high, 1, "vector"), _result(tie_a, 2, "vector")],
        [_result(low, 1, "bm25"), _result(tie_b, 2, "bm25")],
        rrf_k=60,
    )
    # high: 1/61; low: 1/61; ties: 1/62 each → high/low tie broken by chunk id.
    assert [f.result.chunk_id for f in fused] == sorted([high, low]) + tie_ids
    assert fused[0].fused_score == pytest.approx(1 / 61)


def test_duplicate_chunk_appears_once() -> None:
    (cid,) = _ids(1)
    fused = fuse_rrf([_result(cid, 1, "vector")], [_result(cid, 1, "bm25")])
    assert len(fused) == 1


def test_empty_inputs() -> None:
    assert fuse_rrf([], []) == []


# --- validation -----------------------------------------------------------


def test_validate_rrf_k() -> None:
    assert validate_rrf_k(1) == 1
    assert validate_rrf_k(DEFAULT_RRF_K) == DEFAULT_RRF_K
    for bad in (0, -5, True, "60", 6.0, None):
        with pytest.raises(RetrievalError):
            validate_rrf_k(bad)  # type: ignore[arg-type]


def test_validate_fusion_method() -> None:
    assert validate_fusion_method("rrf") == "rrf"
    for bad in ("weighted", "RRF", "", None):
        with pytest.raises(RetrievalError):
            validate_fusion_method(bad)  # type: ignore[arg-type]


def test_hybrid_config_default() -> None:
    assert Settings().hybrid_rrf_k == DEFAULT_RRF_K == 60


# --- orchestration (stubbed retrievers) -----------------------------------


class _StubProvider:
    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def model_version(self) -> str:
        return "v1"

    @property
    def dimension(self) -> int:
        return 384

    @property
    def device(self) -> str:
        return "cpu"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("hybrid must not embed directly; M5 owns the one call")


def _run(
    monkeypatch,  # type: ignore[no-untyped-def]
    vector: list[RetrievalResult],
    bm25: list[RetrievalResult],
    **kwargs,  # type: ignore[no-untyped-def]
):
    calls: dict = {}

    def fake_vector(session, provider, query, **kw):  # type: ignore[no-untyped-def]
        calls["vector"] = kw
        calls["vector_calls"] = calls.get("vector_calls", 0) + 1
        calls["query"] = query
        return vector

    def fake_bm25(session, query, **kw):  # type: ignore[no-untyped-def]
        calls["bm25"] = kw
        calls["query_bm25"] = query
        return bm25

    monkeypatch.setattr(hybrid_mod, "retrieve_vector", fake_vector)
    monkeypatch.setattr(hybrid_mod, "retrieve_bm25", fake_bm25)
    results = retrieve_hybrid(None, _StubProvider(), "test query", **kwargs)  # type: ignore[arg-type]
    return results, calls


def test_top_k_limits_final_results(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    ids = _ids(4)
    vector = [_result(cid, rank + 1, "vector") for rank, cid in enumerate(ids)]
    results, _ = _run(monkeypatch, vector, [], top_k=2)
    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]


def test_candidate_pools_default_and_explicit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _, calls = _run(monkeypatch, [], [], top_k=5)
    assert calls["vector"]["top_k"] == 10
    assert calls["bm25"]["top_k"] == 10
    _, calls = _run(monkeypatch, [], [], top_k=5, vector_top_k=3, bm25_top_k=7)
    assert calls["vector"]["top_k"] == 3
    assert calls["bm25"]["top_k"] == 7
    # Defaults respect the cap instead of exceeding it.
    _, calls = _run(monkeypatch, [], [], top_k=90, max_top_k=100)
    assert calls["vector"]["top_k"] == 100


def test_candidate_validation(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for kwargs in ({"vector_top_k": 0}, {"bm25_top_k": -1}, {"vector_top_k": 101}):
        with pytest.raises(RetrievalError):
            _run(monkeypatch, [], [], **kwargs)


def test_hybrid_validation(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(RetrievalError):
        _run(monkeypatch, [], [], top_k=0)
    with pytest.raises(RetrievalError):
        _run(monkeypatch, [], [], max_top_k=0)
    with pytest.raises(RetrievalError):
        _run(monkeypatch, [], [], rrf_k=0)
    with pytest.raises(RetrievalError):
        _run(monkeypatch, [], [], fusion_method="weighted")
    for bad_query in ("", "   "):
        with pytest.raises(RetrievalError):
            retrieve_hybrid(None, _StubProvider(), bad_query)  # type: ignore[arg-type]


def test_empty_branches(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    (cid,) = _ids(1)
    only_vector, _ = _run(monkeypatch, [_result(cid, 1, "vector")], [])
    assert len(only_vector) == 1
    assert only_vector[0].score == pytest.approx(1 / 61)
    only_bm25, _ = _run(monkeypatch, [], [_result(cid, 2, "bm25")])
    assert only_bm25[0].score == pytest.approx(1 / 62)
    both_empty, _ = _run(monkeypatch, [], [])
    assert both_empty == []


def test_metadata_method_and_rank(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cid_v, cid_b = _ids(2)
    results, _ = _run(
        monkeypatch,
        [_result(cid_v, 1, "vector", text="vector text")],
        [_result(cid_b, 1, "bm25", text="bm25 text")],
    )
    assert all(r.retrieval_method == "hybrid" for r in results)
    assert [r.rank for r in results] == [1, 2]
    assert results[0].text == "vector text"
    assert results[0].document_title == "Title"
    assert results[0].chunk_metadata == {"k": "v"}
    assert results[1].text == "bm25 text"


def test_filters_forwarded_to_both(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import uuid as uuid_mod

    doc_id = uuid_mod.uuid4()
    _, calls = _run(monkeypatch, [], [], document_id=doc_id, document_type="pdf")
    assert calls["vector"]["document_id"] == doc_id
    assert calls["bm25"]["document_id"] == doc_id
    assert calls["vector"]["document_type"] == "pdf"
    assert calls["bm25"]["document_type"] == "pdf"


def test_single_embedding_call(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # The stub provider raises if hybrid embeds directly; the M5 stub
    # records exactly one invocation (its one-call guarantee).
    _, calls = _run(monkeypatch, [], [])
    assert calls["vector_calls"] == 1
    assert calls["query"] == "test query"
    assert calls["query_bm25"] == "test query"


def test_retriever_failure_propagates(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RetrievalError("vector backend down")

    monkeypatch.setattr(hybrid_mod, "retrieve_vector", boom)
    with pytest.raises(RetrievalError):
        retrieve_hybrid(None, _StubProvider(), "q")  # type: ignore[arg-type]
