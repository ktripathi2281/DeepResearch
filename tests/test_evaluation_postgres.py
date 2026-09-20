"""M16 PostgreSQL integration — comparison runs and answer flow on real DB.

Controlled vectors make recall deterministic; answer/verifier LLMs are
scripted fakes. No Ollama, no network.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deepresearch import repository
from deepresearch.config import get_settings
from deepresearch.db import check_connection, init_db
from deepresearch.eval_runner import load_dataset, run_evaluation, save_result
from deepresearch.evaluation import (
    EvalDataset,
    EvaluationCase,
    ExperimentConfig,
    check_category_coverage,
)
from tests.fakes import FakeLLMProvider, FakeReranker, FakeVerifierLLM
from tests.test_evaluation import FIXTURE
from tests.test_retrieval_postgres import E1, E2, EMIX, FixedQueryProvider


def _pg_engine() -> Engine:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    if not check_connection(engine):
        pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    return engine


def _seed(session: Session, source: str, chunks: list[tuple[str, list[float]]]) -> None:
    doc = repository.create_document(
        session,
        title=f"Eval {source}",
        source=source,
        content_hash=f"m16-{uuid.uuid4().hex}",
        document_type="markdown",
        metadata=None,
    )
    for index, (chunk_text, vector) in enumerate(chunks):
        chunk = repository.create_chunk(
            session, document_id=doc.id, text=chunk_text, chunk_index=index
        )
        chunk.embedding = vector
        chunk.embedding_model = "fixed-model"
        chunk.embedding_version = "fixed-v1"
    session.commit()


def _cleanup(engine: Engine, like: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM chunks WHERE document_id IN "
                "(SELECT id FROM documents WHERE source LIKE :p)"
            ),
            {"p": like},
        )
        conn.execute(text("DELETE FROM documents WHERE source LIKE :p"), {"p": like})


def _mini_dataset(prefix: str) -> EvalDataset:
    return EvalDataset(
        version="pg-mini-v1",
        documents=[],
        cases=[
            EvaluationCase(
                case_id="r1",
                question="alpha retrieval",
                category="single-document",
                relevant_sources=[f"{prefix}-a.md"],
                expected_facts=["alpha"],
            ),
            EvaluationCase(
                case_id="r2",
                question="something absent entirely",
                category="no-answer",
                relevant_sources=[],
                expected_facts=[],
                abstain_expected=True,
            ),
        ],
    )


def test_four_way_comparison_reports_without_ranking() -> None:
    engine = _pg_engine()
    prefix = f"pg-m16-{uuid.uuid4().hex[:8]}"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, f"{prefix}-a.md", [("alpha retrieval fact", E1)])
            _seed(session, f"{prefix}-b.md", [("beta retrieval fact", E2)])
            _seed(session, f"{prefix}-c.md", [("gamma cooking note", EMIX)])
            session.expunge_all()

            dataset = _mini_dataset(prefix)
            identities: set[str] = set()
            for method in ("vector", "bm25", "hybrid", "hybrid_reranked"):
                config = ExperimentConfig(
                    dataset_version="pg-mini-v1",
                    retrieval_method=method,
                    top_k=5,  # type: ignore[arg-type]
                )
                deps_kwargs = {
                    "embedding_provider": FixedQueryProvider(E1),
                    "reranker": FakeReranker(),
                    "llm_provider": None,
                }
                from deepresearch.eval_runner import RetrievalDeps

                result = run_evaluation(
                    session, dataset, config, RetrievalDeps(**deps_kwargs), run_answers=False
                )
                identities.add(result.experiment.identity)
                assert result.errors == []
                assert result.retrieval["mrr"] is not None  # case r1 has ground truth
                assert result.retrieval["recall@5"] is not None
                assert result.latency_ms["p50"] is not None
            assert len(identities) == 4  # distinct configs, distinct identities
    finally:
        try:
            _cleanup(engine, f"{prefix}%")
        finally:
            engine.dispose()


def test_answer_flow_metrics_and_skips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    engine = _pg_engine()
    prefix = f"pg-m16a-{uuid.uuid4().hex[:8]}"
    try:
        init_db(engine)
        with Session(bind=engine) as session:
            _seed(session, f"{prefix}-a.md", [("alpha retrieval fact", E1)])
            session.expunge_all()

            dataset = _mini_dataset(prefix)
            config = ExperimentConfig(
                dataset_version="pg-mini-v1", retrieval_method="bm25", top_k=3
            )
            from deepresearch.eval_runner import RetrievalDeps

            llm = FakeLLMProvider(answer="Alpha holds [1].")
            verifier = FakeVerifierLLM(
                responses=['{"verdict": "supported", "explanation": "Stated."}'] * 8
            )
            result = run_evaluation(
                session,
                dataset,
                config,
                RetrievalDeps(
                    embedding_provider=FixedQueryProvider(E1),
                    reranker=FakeReranker(),
                    llm_provider=llm,
                    verifier=verifier,
                ),
            )
            assert result.answers["correctness"] == 1.0  # "alpha" present
            assert result.answers["faithfulness"] == 1.0
            assert result.answers["citation_correctness"] == 1.0
            assert result.answers["abstention_rate"] is not None

            target = save_result(result, tmp_path / "eval-result.json")
            payload = json.loads(target.read_text(encoding="utf-8"))
            assert payload["experiment_identity"] == config.identity
            assert payload["dataset_version"] == "pg-mini-v1"

            # Missing LLM → answer flow skipped with reasons, retrieval intact.
            skipped_result = run_evaluation(
                session,
                dataset,
                config,
                RetrievalDeps(embedding_provider=FixedQueryProvider(E1)),
            )
            assert skipped_result.retrieval["mrr"] is not None
            assert len(skipped_result.skipped) == 2
            assert skipped_result.answers["correctness"] is None
    finally:
        try:
            _cleanup(engine, f"{prefix}%")
        finally:
            engine.dispose()


def test_fixture_dataset_loads_with_full_coverage() -> None:
    dataset = load_dataset(FIXTURE)
    assert check_category_coverage(dataset) == set()
