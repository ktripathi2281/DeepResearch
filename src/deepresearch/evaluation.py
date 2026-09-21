"""Evaluation domain and deterministic metrics — Milestone 16.

Covers the evaluation data model (cases, datasets, experiments,
results) and every metric that can be computed without a judge
model: Recall@K, MRR, percentiles, fact presence, and the
verification/status-based answer metrics. The runner that executes
experiments against a live database lives in ``eval_runner.py``.

Conventions:

- Unavailable metrics are ``None`` — never silently zero.
- Relevance is matched on ``document.source`` so fixtures need no
  knowledge of database UUIDs.
- Keyword fact checks are documented approximations, not semantic
  judgments; anything requiring a judge is out of scope here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from deepresearch.answer_status import AnswerStatus

Category = Literal[
    "single-document",
    "multi-document",
    "exact-lookup",
    "semantic",
    "multi-hop",
    "no-answer",
    "conflict",
    "injection",
]

REQUIRED_CATEGORIES: tuple[str, ...] = (
    "single-document",
    "multi-document",
    "exact-lookup",
    "semantic",
    "multi-hop",
    "no-answer",
    "conflict",
    "injection",
)

RetrievalMethod = Literal["vector", "bm25", "hybrid", "hybrid_reranked"]


class EvalDocument(BaseModel):
    """One fixture corpus document to ingest before running a dataset."""

    source: str
    title: str = ""
    document_type: str = "markdown"
    text: str

    model_config = {"extra": "forbid"}


class EvaluationCase(BaseModel):
    """One evaluation case. Only fields relevant to a metric are required."""

    case_id: str
    question: str
    category: Category
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    relevant_sources: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    expected_status: AnswerStatus | None = None
    abstain_expected: bool = False
    injection_markers: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class EvalDataset(BaseModel):
    """Versioned dataset: fixture corpus plus cases."""

    version: str
    documents: list[EvalDocument] = Field(default_factory=list)
    cases: list[EvaluationCase] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class ExperimentConfig(BaseModel):
    """Everything that defines an experiment run (timestamp/notes excluded from identity)."""

    dataset_version: str
    retrieval_method: RetrievalMethod = "hybrid"
    top_k: int = 5
    reranker_enabled: bool = False
    chunk_target_tokens: int = 800
    chunk_overlap_tokens: int = 120
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_version: str = "1"
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:4b"
    fusion_method: str = "rrf"
    rrf_k: int = 60
    notes: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    model_config = {"extra": "forbid"}

    @property
    def identity(self) -> str:
        """Stable fingerprint: same config → same identity, always."""
        canonical = self.model_dump(exclude={"timestamp", "notes"}, mode="json")
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SkippedCase:
    case_id: str
    reason: str


@dataclass(frozen=True)
class CaseError:
    case_id: str
    error: str


@dataclass
class EvaluationResult:
    """Serializable experiment outcome; nulls mean unavailable, never zero."""

    experiment: ExperimentConfig
    dataset_version: str
    total_cases: int
    by_category: dict[str, int] = field(default_factory=dict)
    retrieval: dict[str, float | None] = field(default_factory=dict)
    answers: dict[str, float | None] = field(default_factory=dict)
    latency_ms: dict[str, float | None] = field(default_factory=dict)
    errors: list[CaseError] = field(default_factory=list)
    skipped: list[SkippedCase] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment.model_dump(mode="json"),
            "experiment_identity": self.experiment.identity,
            "dataset_version": self.dataset_version,
            "total_cases": self.total_cases,
            "by_category": dict(self.by_category),
            "retrieval": dict(self.retrieval),
            "answers": dict(self.answers),
            "latency_ms": dict(self.latency_ms),
            "errors": [asdict(error) for error in self.errors],
            "skipped": [asdict(skip) for skip in self.skipped],
        }


def check_category_coverage(dataset: EvalDataset) -> set[str]:
    """Categories from REQUIRED_CATEGORIES missing in a dataset."""
    return set(REQUIRED_CATEGORIES) - {case.category for case in dataset.cases}


def _deduplicated(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float | None:
    """Share of relevant items in the top K. None without ground truth."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not relevant_ids:
        return None
    top = _deduplicated(retrieved_ids)[:k]
    return len([rid for rid in top if rid in set(relevant_ids)]) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float | None:
    """Reciprocal rank of the first relevant hit. None without ground truth."""
    if not relevant_ids:
        return None
    relevant = set(relevant_ids)
    for rank, rid in enumerate(_deduplicated(retrieved_ids), start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def calculate_percentile(values: list[float], percentile: float) -> float | None:
    """Linear-interpolation percentile (0–100). None on empty input."""
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def p50(values: list[float]) -> float | None:
    return calculate_percentile(values, 50)


def p95(values: list[float]) -> float | None:
    return calculate_percentile(values, 95)


def facts_present(answer_text: str, expected_facts: list[str]) -> dict:
    """Case-insensitive substring check per expected fact (approximation).

    Keyword presence is NOT semantic correctness — it only detects
    whether the answer mentions the fact's wording. Documented as a
    deterministic baseline; a judge model would be a separate metric.
    """
    lowered = (answer_text or "").lower()
    present = [fact for fact in expected_facts if fact.lower() in lowered]
    missing = [fact for fact in expected_facts if fact.lower() not in lowered]
    total = len(expected_facts)
    return {
        "present": present,
        "missing": missing,
        "score": (len(present) / total) if total else None,
    }


@dataclass(frozen=True)
class VerificationTallies:
    supported: int = 0
    unsupported: int = 0
    insufficient_evidence: int = 0
    cited_rows: int = 0
    cited_claims: int = 0
    fully_supported_claims: int = 0


def summarize_verification(results: list) -> VerificationTallies:  # type: ignore[no-untyped-def]
    """Aggregate M12 rows: per-citation counts plus per-claim support.

    Only evaluated rows (supported/unsupported/insufficient_evidence)
    form claims; invalid markers and uncited claims are tracked by M12
    separately and never inflate the denominators.
    """
    supported = unsupported = insufficient = 0
    by_claim: dict[int, list[str]] = {}
    for row in results:
        if row.citation_id is None or row.status not in (
            "supported",
            "unsupported",
            "insufficient_evidence",
        ):
            continue
        by_claim.setdefault(row.claim_id, []).append(row.status)
        if row.status == "supported":
            supported += 1
        elif row.status == "unsupported":
            unsupported += 1
        else:
            insufficient += 1
    cited_rows = supported + unsupported + insufficient
    fully = sum(
        1 for statuses in by_claim.values() if statuses and all(s == "supported" for s in statuses)
    )
    return VerificationTallies(
        supported=supported,
        unsupported=unsupported,
        insufficient_evidence=insufficient,
        cited_rows=cited_rows,
        cited_claims=len(by_claim),
        fully_supported_claims=fully,
    )


def citation_correctness(tallies: VerificationTallies) -> float | None:
    """Supported citations over evaluated citations. None when none evaluated."""
    if tallies.cited_rows == 0:
        return None
    return tallies.supported / tallies.cited_rows


def faithfulness(tallies: VerificationTallies) -> float | None:
    """Fully supported cited claims over cited claims. None when none cited."""
    if tallies.cited_claims == 0:
        return None
    return tallies.fully_supported_claims / tallies.cited_claims


def citation_completeness(
    answer_text: str, expected_facts: list[str], cited_texts: list[str]
) -> float | None:
    """Share of expected facts appearing in cited evidence text.

    None without expected facts or without citations. Like
    ``facts_present``, a wording-level approximation.
    """
    if not expected_facts or not cited_texts:
        return None
    joined = "\n".join(cited_texts).lower()
    found = sum(1 for fact in expected_facts if fact.lower() in joined)
    return found / len(expected_facts)


def abstention_ok(status: str, invalid_count: int, abstain_expected: bool) -> bool | None:
    """True when a no-answer case abstained cleanly; None when not applicable."""
    if not abstain_expected:
        return None
    return status in ("no_evidence", "insufficient_evidence") and invalid_count == 0


def conflict_ok(status: str, conflict_count: int, evidence_count: int) -> bool | None:
    """True when a conflict case surfaced disagreement with sources kept."""
    if evidence_count < 2:
        return None
    return status == "conflicting_evidence" and conflict_count >= 1


def injection_structural_ok(
    answer_text: str, evidence_texts: list[str], markers: list[str]
) -> dict:
    """Structural injection checks (NOT an adversarial proof).

    Verifies each marker was retrieved as data (present in evidence)
    and the pipeline completed with an answer. Whether the model obeyed
    the injected instruction cannot be judged deterministically here —
    that requires a judge or manual review.
    """
    retrieved = (
        {marker: any(marker in text for text in evidence_texts) for marker in markers}
        if markers
        else {}
    )
    return {
        "markers_retrieved": retrieved,
        "all_markers_retrieved": all(retrieved.values()) if retrieved else None,
        "completed": bool((answer_text or "").strip()),
    }
