"""M16 unit tests — dataset, metrics, experiments, runner (no DB/LLM/files beyond fixtures).

PostgreSQL execution lives in test_evaluation_postgres.py. No Ollama anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from deepresearch.eval_runner import load_dataset, save_result
from deepresearch.evaluation import (
    REQUIRED_CATEGORIES,
    EvaluationCase,
    EvaluationResult,
    ExperimentConfig,
    abstention_ok,
    calculate_percentile,
    check_category_coverage,
    citation_completeness,
    citation_correctness,
    conflict_ok,
    facts_present,
    faithfulness,
    injection_structural_ok,
    mrr,
    recall_at_k,
    summarize_verification,
)

FIXTURE = Path(__file__).parent.parent / "evals" / "datasets" / "eval-dev-v1.json"


class _Row:
    def __init__(self, claim_id: int, citation_id: int | None, status: str) -> None:
        self.claim_id = claim_id
        self.citation_id = citation_id
        self.status = status


# --- dataset ----------------------------------------------------------------------


def test_fixture_loads_covers_categories_and_version() -> None:
    dataset = load_dataset(FIXTURE)
    assert dataset.version == "eval-dev-v1"
    assert len(dataset.cases) == 8
    assert check_category_coverage(dataset) == set()
    assert {c.category for c in dataset.cases} == set(REQUIRED_CATEGORIES)
    assert len(dataset.documents) == 7


def test_invalid_cases_rejected() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(case_id="x", question="q", category="no-such-category")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EvaluationCase(case_id="x", question="")  # type: ignore[arg-type]


def test_load_rejects_bad_files(tmp_path: Path) -> None:
    missing_version = tmp_path / "a.json"
    missing_version.write_text('{"cases": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(missing_version)
    dupes = tmp_path / "b.json"
    dupes.write_text(
        '{"version": "v", "cases": ['
        '{"case_id": "c", "question": "q", "category": "semantic"},'
        '{"case_id": "c", "question": "q", "category": "semantic"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(dupes)


# --- retrieval metrics ---------------------------------------------------------------


def test_recall_at_3_5_10() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = ["b", "d", "z"]
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, 10) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        recall_at_k(retrieved, relevant, 0)


def test_recall_no_ground_truth_is_none_not_zero() -> None:
    assert recall_at_k(["a"], [], 5) is None
    assert recall_at_k([], ["a"], 5) == 0.0


def test_recall_deduplicates() -> None:
    assert recall_at_k(["a", "a", "b"], ["a", "b"], 3) == 1.0


def test_mrr_first_hit_multiple_and_missing() -> None:
    assert mrr(["x", "a"], ["a"]) == pytest.approx(0.5)
    assert mrr(["a", "b"], ["a", "b"]) == 1.0
    assert mrr(["x", "y"], ["a"]) == 0.0
    assert mrr(["a"], []) is None


# --- percentiles ------------------------------------------------------------------------


def test_percentile_edge_cases() -> None:
    assert calculate_percentile([], 50) is None
    assert calculate_percentile([7.0], 50) == 7.0
    assert calculate_percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert calculate_percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5  # unsorted input
    assert calculate_percentile([1.0, 2.0, 3.0, 4.0], 0) == 1.0
    assert calculate_percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0
    with pytest.raises(ValueError):
        calculate_percentile([1.0], 101)


# --- answer metrics -----------------------------------------------------------------------


def test_facts_present_and_missing() -> None:
    result = facts_present("Dense vector search is used.", ["dense vector search", "BM25"])
    assert result["present"] == ["dense vector search"]
    assert result["missing"] == ["BM25"]
    assert result["score"] == pytest.approx(0.5)
    assert facts_present("anything", [])["score"] is None


def test_verification_aggregates() -> None:
    rows = [
        _Row(1, 1, "supported"),
        _Row(1, 2, "supported"),
        _Row(2, 3, "unsupported"),
        _Row(3, None, "uncited"),
        _Row(4, 9, "invalid_citation"),
    ]
    tallies = summarize_verification(rows)
    assert (tallies.supported, tallies.unsupported, tallies.cited_rows) == (2, 1, 3)
    assert (tallies.cited_claims, tallies.fully_supported_claims) == (2, 1)
    assert citation_correctness(tallies) == pytest.approx(2 / 3)
    assert faithfulness(tallies) == pytest.approx(0.5)
    empty = summarize_verification([_Row(1, None, "uncited")])
    assert citation_correctness(empty) is None
    assert faithfulness(empty) is None


def test_completeness() -> None:
    cited = ["dense vector search is used here", "other text"]
    result = citation_completeness("ans", ["dense vector search", "missing"], cited)
    assert result == pytest.approx(0.5)
    assert citation_completeness("ans", [], cited) is None
    assert citation_completeness("ans", ["x"], []) is None


def test_abstention_and_conflict() -> None:
    assert abstention_ok("no_evidence", 0, True) is True
    assert abstention_ok("insufficient_evidence", 0, True) is True
    assert abstention_ok("answered", 0, True) is False
    assert abstention_ok("no_evidence", 2, True) is False  # fake citations fail
    assert abstention_ok("answered", 0, False) is None  # not applicable
    assert conflict_ok("conflicting_evidence", 1, 2) is True
    assert conflict_ok("answered", 0, 2) is False
    assert conflict_ok("answered", 0, 1) is None


def test_injection_structural() -> None:
    out = injection_structural_ok("An answer.", ["doc with IGNORE marker"], ["IGNORE"])
    assert out == {
        "markers_retrieved": {"IGNORE": True},
        "all_markers_retrieved": True,
        "completed": True,
    }
    out = injection_structural_ok("  ", ["plain"], ["MISSING"])
    assert out["all_markers_retrieved"] is False and out["completed"] is False
    out = injection_structural_ok("Answer.", ["text"], [])
    assert out["all_markers_retrieved"] is None


# --- experiments ----------------------------------------------------------------------------


def test_experiment_identity_and_metadata() -> None:
    base = ExperimentConfig(dataset_version="eval-dev-v1", retrieval_method="hybrid")
    same = ExperimentConfig(dataset_version="eval-dev-v1", retrieval_method="hybrid")
    other = ExperimentConfig(dataset_version="eval-dev-v1", retrieval_method="vector")
    assert base.identity == same.identity
    assert base.identity != other.identity
    assert len(base.identity) == 16
    assert base.embedding_model == "BAAI/bge-small-en-v1.5"
    assert base.llm_model == "qwen3:4b"


def test_result_serialization_round_trip(tmp_path: Path) -> None:
    config = ExperimentConfig(dataset_version="eval-dev-v1")
    result = EvaluationResult(
        experiment=config,
        dataset_version="eval-dev-v1",
        total_cases=1,
        by_category={"semantic": 1},
        retrieval={"recall@5": 1.0, "mrr": None},
        answers={},
        latency_ms={"p50": 12.5, "p95": None},
    )
    target = save_result(result, tmp_path / "nested" / "result.json")
    payload = target.read_text(encoding="utf-8")
    assert '"mrr": null' in payload  # unavailable stays null, never zero
    assert f'"experiment_identity": "{config.identity}"' in payload
    assert target.read_text(encoding="utf-8").endswith("\n")
