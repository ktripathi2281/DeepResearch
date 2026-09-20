"""Evaluation runner — Milestone 16.

Executes versioned datasets against the real pipeline with injectable
providers (fakes in tests, local models in practice): retrieval
dispatch per configured method, answer flow, metric aggregation, and
deterministic JSON export. Reports measurements only — never declares
a winner. No live model runtime, no network required.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from deepresearch.bm25 import BM25Index, retrieve_bm25
from deepresearch.embeddings import EmbeddingProvider
from deepresearch.evaluation import (
    CaseError,
    EvalDataset,
    EvaluationCase,
    EvaluationResult,
    ExperimentConfig,
    SkippedCase,
    abstention_ok,
    calculate_percentile,
    citation_completeness,
    citation_correctness,
    conflict_ok,
    facts_present,
    faithfulness,
    mrr,
    recall_at_k,
    summarize_verification,
)
from deepresearch.generation import GroundedAnswer, answer_question
from deepresearch.hybrid import retrieve_hybrid
from deepresearch.llm import LLMProvider
from deepresearch.observability import get_current_trace, traced_request, traced_stage
from deepresearch.reranker import Reranker, rerank_results
from deepresearch.retrieval import RetrievalResult
from deepresearch.retrieval import retrieve as retrieve_vector


@dataclass
class RetrievalDeps:
    """Injectable pipeline dependencies (fakes in tests)."""

    embedding_provider: EmbeddingProvider
    reranker: Reranker | None = None
    bm25_index: BM25Index | None = None
    llm_provider: LLMProvider | None = None
    verifier: LLMProvider | None = None


@dataclass
class CaseRetrieval:
    case_id: str
    category: str
    retrieved_sources: list[str] = field(default_factory=list)
    recall_at_3: float | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    mrr: float | None = None
    latency_ms: int = 0


@dataclass
class CaseAnswer:
    case_id: str
    category: str
    status: str = ""
    correctness: float | None = None
    faithfulness: float | None = None
    citation_correctness: float | None = None
    citation_completeness: float | None = None
    abstention: bool | None = None
    conflict: bool | None = None
    latency_ms: int = 0


def load_dataset(path: str | Path) -> EvalDataset:
    """Load and validate a versioned JSON dataset (duplicate IDs rejected)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw.get("version"):
        raise ValueError("dataset must be a JSON object with a non-empty 'version'")
    dataset = EvalDataset.model_validate(raw)
    seen: set[str] = set()
    for case in dataset.cases:
        if case.case_id in seen:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        seen.add(case.case_id)
    return dataset


def dispatch_retrieval(
    session: Session,
    query: str,
    config: ExperimentConfig,
    deps: RetrievalDeps,
) -> list[RetrievalResult]:
    """Run the configured retrieval method (comparison stays out of retrieval code)."""
    if config.retrieval_method == "vector":
        return retrieve_vector(session, deps.embedding_provider, query, top_k=config.top_k)
    if config.retrieval_method == "bm25":
        return retrieve_bm25(session, query, top_k=config.top_k, index=deps.bm25_index)
    if config.retrieval_method == "hybrid":
        return retrieve_hybrid(
            session, deps.embedding_provider, query, top_k=config.top_k, index=deps.bm25_index
        )
    if config.retrieval_method == "hybrid_reranked":
        if deps.reranker is None:
            raise ValueError("hybrid_reranked requires a reranker")
        candidates = retrieve_hybrid(
            session, deps.embedding_provider, query, top_k=config.top_k, index=deps.bm25_index
        )
        return rerank_results(query, candidates, deps.reranker, top_k=config.top_k)
    raise ValueError(f"unknown retrieval_method: {config.retrieval_method}")


def _ranked_sources(results: list[RetrievalResult]) -> list[str]:
    seen: list[str] = []
    for result in results:
        if result.document_source not in seen:
            seen.append(result.document_source)
    return seen


def evaluate_retrieval_case(
    session: Session,
    case: EvaluationCase,
    config: ExperimentConfig,
    deps: RetrievalDeps,
) -> CaseRetrieval:
    """Execute one case: dispatch, wall latency, source-level metrics."""
    started = time.perf_counter()
    with traced_stage("eval_retrieval_case"):
        results = dispatch_retrieval(session, case.question, config, deps)
    latency_ms = int((time.perf_counter() - started) * 1000)
    sources = _ranked_sources(results)
    return CaseRetrieval(
        case_id=case.case_id,
        category=case.category,
        retrieved_sources=sources,
        recall_at_3=recall_at_k(sources, case.relevant_sources, 3),
        recall_at_5=recall_at_k(sources, case.relevant_sources, 5),
        recall_at_10=recall_at_k(sources, case.relevant_sources, 10),
        mrr=mrr(sources, case.relevant_sources),
        latency_ms=latency_ms,
    )


def evaluate_answer_scores(answer: GroundedAnswer, case: EvaluationCase) -> CaseAnswer:
    """Score one generated answer deterministically (no judge model)."""
    facts = facts_present(answer.answer, case.expected_facts)
    report = answer.verification_report
    tallies = summarize_verification(getattr(report, "results", []) if report else [])
    cited_ids = {citation.chunk_id for citation in answer.citations}
    cited_texts = [item.text for item in answer.evidence if item.chunk_id in cited_ids]
    return CaseAnswer(
        case_id=case.case_id,
        category=case.category,
        status=answer.status,
        correctness=facts["score"],
        faithfulness=faithfulness(tallies),
        citation_correctness=citation_correctness(tallies),
        citation_completeness=citation_completeness(
            answer.answer, case.expected_facts, cited_texts
        ),
        abstention=abstention_ok(
            answer.status, len(answer.invalid_citations), case.abstain_expected
        ),
        conflict=conflict_ok(answer.status, len(answer.conflicts), len(answer.evidence)),
    )


def _mean(values: list[float | None]) -> float | None:
    measured = [v for v in values if v is not None]
    return sum(measured) / len(measured) if measured else None


def _rate(values: list[bool | None]) -> float | None:
    applicable = [v for v in values if v is not None]
    return sum(1 for v in applicable if v) / len(applicable) if applicable else None


def run_evaluation(
    session: Session,
    dataset: EvalDataset,
    config: ExperimentConfig,
    deps: RetrievalDeps,
    *,
    run_answers: bool = True,
) -> EvaluationResult:
    """Run every case: retrieval metrics, optional answer flow, latencies.

    Failures are recorded per case (never raised); cases needing a
    missing provider are skipped with a reason. No winner is declared.
    """
    if config.dataset_version != dataset.version:
        raise ValueError(
            f"config targets dataset {config.dataset_version!r} but loaded {dataset.version!r}"
        )
    retrieval_rows: list[CaseRetrieval] = []
    answer_rows: list[CaseAnswer] = []
    latencies: list[float] = []
    errors: list[CaseError] = []
    skipped: list[SkippedCase] = []
    by_category: dict[str, int] = {}

    for case in dataset.cases:
        by_category[case.category] = by_category.get(case.category, 0) + 1
        try:
            with traced_request(f"eval-{case.case_id}"):
                retrieval = evaluate_retrieval_case(session, case, config, deps)
                retrieval_rows.append(retrieval)
                latencies.append(float(retrieval.latency_ms))
                if run_answers:
                    if deps.llm_provider is None:
                        skipped.append(SkippedCase(case.case_id, "no llm_provider for answer flow"))
                        continue
                    answer_started = time.perf_counter()
                    answer = answer_question(
                        session,
                        deps.embedding_provider,
                        deps.reranker or _noop_reranker(),
                        deps.llm_provider,
                        case.question,
                        retrieval_top_k=config.top_k,
                        evidence_top_k=min(config.top_k, 5),
                        verify_citations=True,
                        verifier=deps.verifier or deps.llm_provider,
                    )
                    answer_latency = int((time.perf_counter() - answer_started) * 1000)
                    scores = evaluate_answer_scores(answer, case)
                    scores.latency_ms = answer_latency
                    answer_rows.append(scores)
                    latencies.append(float(answer_latency))
        except Exception as exc:
            errors.append(CaseError(case.case_id, f"{type(exc).__name__}: {exc}"))

    result = EvaluationResult(
        experiment=config,
        dataset_version=dataset.version,
        total_cases=len(dataset.cases),
        by_category=by_category,
        retrieval={
            "recall@3": _mean([r.recall_at_3 for r in retrieval_rows]),
            "recall@5": _mean([r.recall_at_5 for r in retrieval_rows]),
            "recall@10": _mean([r.recall_at_10 for r in retrieval_rows]),
            "mrr": _mean([r.mrr for r in retrieval_rows]),
        },
        answers={
            "correctness": _mean([a.correctness for a in answer_rows]),
            "faithfulness": _mean([a.faithfulness for a in answer_rows]),
            "citation_correctness": _mean([a.citation_correctness for a in answer_rows]),
            "citation_completeness": _mean([a.citation_completeness for a in answer_rows]),
            "abstention_rate": _rate([a.abstention for a in answer_rows]),
            "conflict_rate": _rate([a.conflict for a in answer_rows]),
        },
        latency_ms={
            "p50": calculate_percentile(latencies, 50),
            "p95": calculate_percentile(latencies, 95),
        },
        errors=errors,
        skipped=skipped,
    )
    trace = get_current_trace()
    if trace is not None:
        trace.note("eval_experiment", config.identity)
    return result


class _NoopReranker:
    """Pass-through reranker used only when evaluation runs without one."""

    model_name = "noop"
    model_version = "0"
    device = "cpu"
    batch_size = 1

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [0.0 for _ in documents]


def _noop_reranker() -> _NoopReranker:
    return _NoopReranker()


def save_result(result: EvaluationResult, path: str | Path) -> Path:
    """Write a deterministic JSON result (sorted keys, stable formatting)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target
