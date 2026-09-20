"""M8 unit tests — fake reranker, rerank_results, validation (no model/GPU).

Real bge-reranker-base behavior is verified in isolation by
tests/test_reranker_model_real.py; the hybrid→rerank pipeline on
PostgreSQL by tests/test_reranker_postgres.py.
"""

from __future__ import annotations

import uuid

import pytest

from deepresearch.config import Settings
from deepresearch.reranker import (
    DEFAULT_CANDIDATE_TOP_K,
    RERANKED_METHOD,
    RERANKER_MODEL,
    LocalCrossEncoderReranker,
    Reranker,
    RerankerError,
    RerankerLoadError,
    rerank_results,
)
from deepresearch.retrieval import RetrievalError, RetrievalResult
from tests.fakes import FakeReranker


def _result(chunk_id: uuid.UUID, text: str, *, score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        chunk_index=0,
        text=text,
        score=score,
        rank=1,
        document_title="Title",
        document_type="markdown",
        document_source="s.md",
        page=1,
        section="Sec",
        chunk_metadata={"k": "v"},
        retrieval_method="hybrid",
    )


def _ids(n: int) -> list[uuid.UUID]:
    return [uuid.uuid4() for _ in range(n)]


# --- A. fake behavior -----------------------------------------------------


def test_fake_reranker_count_order_and_empty() -> None:
    fake = FakeReranker(scores={"b": 2.0})
    assert fake.rerank("q", ["a", "b", "c"]) == [0.0, 2.0, 0.0]
    assert fake.calls == [3]
    assert isinstance(fake, Reranker)
    assert fake.rerank("q", []) == []
    assert fake.calls == [3, 0]


# --- B. rerank_results ----------------------------------------------------


def test_reranked_order_method_rank_and_metadata() -> None:
    low, high = _ids(2)
    results = rerank_results(
        "query",
        [_result(low, "low text"), _result(high, "high text")],
        FakeReranker(scores={"low text": -1.0, "high text": 5.0}),
    )
    assert [r.chunk_id for r in results] == [high, low]
    assert [r.score for r in results] == [5.0, -1.0]
    assert [r.rank for r in results] == [1, 2]
    assert all(r.retrieval_method == RERANKED_METHOD == "reranked" for r in results)
    assert results[0].text == "high text"
    assert results[0].document_title == "Title"
    assert results[0].chunk_metadata == {"k": "v"}
    assert results[0].page == 1 and results[0].section == "Sec"


def test_tie_break_by_chunk_id() -> None:
    tie_ids = sorted(_ids(2))
    other = _ids(1)[0]
    candidates = [_result(tie_ids[1], "b"), _result(other, "c"), _result(tie_ids[0], "a")]
    results = rerank_results("q", candidates, FakeReranker(scores={"c": -1.0}))
    assert [r.chunk_id for r in results] == tie_ids + [other]


def test_final_and_candidate_top_k() -> None:
    ids = _ids(5)
    candidates = [_result(cid, f"doc {i}") for i, cid in enumerate(ids)]
    scores = {f"doc {i}": float(5 - i) for i in range(5)}
    fake = FakeReranker(scores=scores)
    # Only 3 candidates scored, best 2 returned.
    results = rerank_results("q", candidates, fake, candidate_top_k=3, top_k=2)
    assert fake.calls == [3]
    assert [r.text for r in results] == ["doc 0", "doc 1"]
    # Fewer available than candidate_top_k: rerank all.
    short = rerank_results("q", candidates[:2], FakeReranker(), candidate_top_k=20, top_k=5)
    assert len(short) == 2


# --- C. validation --------------------------------------------------------


def test_blank_query_rejected() -> None:
    (cid,) = _ids(1)
    for bad in ("", "   "):
        with pytest.raises(RetrievalError):
            rerank_results(bad, [_result(cid, "t")], FakeReranker())


def test_invalid_k_values_rejected() -> None:
    (cid,) = _ids(1)
    candidates = [_result(cid, "t")]
    for kwargs in ({"top_k": 0}, {"top_k": -1}, {"top_k": 101}, {"top_k": True}):
        with pytest.raises(RetrievalError):
            rerank_results("q", candidates, FakeReranker(), **kwargs)  # type: ignore[arg-type]
    for kwargs in ({"candidate_top_k": 0}, {"candidate_top_k": -2}, {"candidate_top_k": 500}):
        with pytest.raises(RetrievalError):
            rerank_results("q", candidates, FakeReranker(), **kwargs)  # type: ignore[arg-type]


def test_invalid_provider_batch_size_rejected() -> None:
    with pytest.raises(RerankerError):
        LocalCrossEncoderReranker(batch_size=0)
    with pytest.raises(RerankerError):
        LocalCrossEncoderReranker(batch_size=-4)


def test_reranker_config_defaults() -> None:
    settings = Settings()
    assert settings.reranker_model == RERANKER_MODEL == "BAAI/bge-reranker-base"
    assert settings.reranker_model_version == "1"
    assert settings.reranker_device == "auto"
    assert settings.reranker_batch_size == 16
    assert settings.reranker_candidate_top_k == DEFAULT_CANDIDATE_TOP_K == 20


def test_local_provider_configuration_without_loading() -> None:
    provider = LocalCrossEncoderReranker()
    assert provider.model_name == RERANKER_MODEL
    assert provider.model_version == "1"
    assert provider.batch_size == 16
    assert isinstance(provider, Reranker)


# --- D. failures ----------------------------------------------------------


def test_provider_failure_propagates_without_fallback() -> None:
    (cid,) = _ids(1)
    with pytest.raises(RerankerError):
        rerank_results("q", [_result(cid, "t")], FakeReranker(fail=True))


def test_score_count_mismatch_raises() -> None:
    class ShortFake(FakeReranker):
        def rerank(self, query: str, documents) -> list[float]:  # type: ignore[no-untyped-def]
            super().rerank(query, documents)
            return [1.0]  # wrong count on purpose

    (cid,) = _ids(1)
    with pytest.raises(RerankerError):
        rerank_results("q", [_result(cid, "a"), _result(cid, "b")], ShortFake())


def test_empty_input_never_calls_provider() -> None:
    fake = FakeReranker()
    assert rerank_results("q", [], fake) == []
    assert fake.calls == []


# --- E. batching ----------------------------------------------------------


def test_single_provider_call_with_all_candidates() -> None:
    ids = _ids(4)
    fake = FakeReranker()
    rerank_results("q", [_result(cid, f"t{i}") for i, cid in enumerate(ids)], fake)
    assert fake.calls == [4]  # provider batches internally via its batch_size


def test_local_provider_forwards_batch_size(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import sentence_transformers

    seen: dict = {}

    class StubModel:
        def predict(self, pairs, batch_size=None, show_progress_bar=False):  # type: ignore[no-untyped-def]
            seen["batch_size"] = batch_size
            seen["pairs"] = len(pairs)
            return [0.1 * i for i in range(len(pairs))]

    monkeypatch.setattr(
        sentence_transformers,
        "CrossEncoder",
        lambda *a, **k: StubModel(),  # type: ignore[no-untyped-def]
    )
    provider = LocalCrossEncoderReranker(batch_size=4, device="cpu")
    assert provider.rerank("q", ["a", "b", "c"]) == [0.0, 0.1, 0.2]
    assert seen == {"batch_size": 4, "pairs": 3}
    assert provider.device == "cpu"


# --- F. device ------------------------------------------------------------


def test_explicit_cuda_without_gpu_fails_clearly() -> None:
    import torch

    provider = LocalCrossEncoderReranker(device="cuda")
    if torch.cuda.is_available():
        pytest.skip("CUDA available; unavailable-CUDA path not testable here")
    with pytest.raises(RerankerLoadError):
        provider.rerank("q", ["doc"])


def test_invalid_device_rejected() -> None:
    with pytest.raises(RerankerLoadError):
        LocalCrossEncoderReranker(device="tpu").rerank("q", ["doc"])
