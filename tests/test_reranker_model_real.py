"""M8 real-model smoke — BAAI/bge-reranker-base only.

Loads the actual cross-encoder (uses the HF cache when offline,
downloads once when online) and proves: load works, pairs score,
count matches, output is deterministic. Skipped — never failed —
when the model is neither cached nor reachable. A loaded model that
misbehaves fails loudly; no substitution, ever.
"""

from __future__ import annotations

import pytest

from deepresearch.reranker import RERANKER_MODEL, LocalCrossEncoderReranker, Reranker


def _model_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=RERANKER_MODEL, local_files_only=True)
    except Exception:
        return False
    return True


def _network_up() -> bool:
    try:
        import urllib.request

        return urllib.request.urlopen("https://huggingface.co", timeout=10).status == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def provider() -> LocalCrossEncoderReranker:
    if not (_model_cached() or _network_up()):
        pytest.skip(f"{RERANKER_MODEL} not cached and no network")
    return LocalCrossEncoderReranker(device="cpu")


def test_real_reranker_loads_scores_and_agrees_with_itself(
    provider: LocalCrossEncoderReranker,
) -> None:
    assert isinstance(provider, Reranker)
    pairs = [
        "Hybrid retrieval combines vector and lexical search.",
        "The weather in Lisbon is mild in spring.",
        "Unrelated cooking recipe with no retrieval content.",
    ]
    query = "How does hybrid retrieval work?"
    scores = provider.rerank(query, pairs)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    assert provider.device == "cpu"
    assert provider.rerank(query, pairs) == scores  # deterministic on fixed device
    assert provider.rerank(query, []) == []
