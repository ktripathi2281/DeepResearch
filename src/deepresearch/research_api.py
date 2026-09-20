"""Public research HTTP boundary — Milestone 18.

A minimal, safe facade over the existing research pipeline for the
Next.js frontend. It defines the ONLY HTTP surface a research request
needs:

    POST /api/research          (question)          → 202 job started
    GET  /api/research/{id}                          → job snapshot

The pipeline itself is untouched: ``generation.answer_question``
(hybrid → rerank → grounded generation → citation extraction →
optional verification) stays the source of truth and runs unchanged in
a background thread per request. Nothing here re-implements retrieval,
generation, verification, or agent logic.

Security contract (M17 continues to apply):
- The response model is an explicit public representation: identifiers,
  counts, durations, statuses, and source text only.
- No prompts, no system instructions, no chain-of-thought, no raw
  model reasoning, no embeddings, no internals of the trace object.
- Job failures return a user-safe error plus a request ID; server-side
  detail goes to logs via ``logger.exception``, never to the response.
- ``MAX_QUESTION_CHARS`` (4000) is enforced server-side on submit —
  the frontend layer can never bypass it.

Progress: a running job exposes ``RequestTrace.stages`` — real backend
stage transitions with durations recorded by ``traced_stage``. The
frontend maps internal stage names to user-facing labels; no fake
percentages are produced anywhere.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session, sessionmaker

from deepresearch.answer_status import AnswerStatus
from deepresearch.config import get_settings
from deepresearch.db import get_engine, get_session_factory, init_db
from deepresearch.embeddings import LocalEmbeddingProvider
from deepresearch.generation import MAX_QUESTION_CHARS, GroundedAnswer, answer_question
from deepresearch.llm import default_llm_provider
from deepresearch.logging import get_logger
from deepresearch.observability import (
    COUNTER_RETRIEVAL_HYBRID,
    REQUEST_ID_HEADER,
    RequestTrace,
    normalize_request_id,
    traced_request,
)
from deepresearch.reranker import LocalCrossEncoderReranker

logger = get_logger(__name__)

JobStatus = Literal["running", "completed", "failed"]


# --- public (safe) response models -------------------------------------------


class ErrorInfo(BaseModel):
    """User-safe error: short message + identifiers, never a stack trace."""

    message: str
    type: str | None = None
    request_id: str | None = None


class StageInfo(BaseModel):
    """One real backend stage transition (name, duration, success)."""

    name: str
    duration_ms: int
    success: bool = True


class CitationInfo(BaseModel):
    """Provenance for one ``[N]`` marker found in the answer."""

    citation_id: int
    document_title: str | None
    document_source: str
    document_type: str
    page: int | None
    section: str | None
    retrieval_method: str
    retrieval_score: float
    retrieval_rank: int


class EvidenceInfo(BaseModel):
    """One source chunk presented as material (data, never instructions)."""

    citation_id: int
    chunk_id: str
    document_id: str
    document_title: str | None
    document_source: str
    document_type: str
    page: int | None
    section: str | None
    text: str
    retrieval_method: str
    retrieval_score: float
    retrieval_rank: int


class ConflictInfo(BaseModel):
    """One detected conflict between two preserved evidence items."""

    conflict_id: str
    conflict_type: str
    description: str
    citation_ids: list[int]


class VerificationClaimInfo(BaseModel):
    """One claim×citation verifier judgment (design verdict output, not reasoning)."""

    claim_id: int
    citation_id: int | None
    claim_text: str
    status: str
    explanation: str = ""


class VerificationSummaryInfo(BaseModel):
    """Aggregate + per-claim verification state for the research details."""

    counts: dict[str, int]
    claims: list[VerificationClaimInfo]


class ResearchDetailsInfo(BaseModel):
    """User-safe research details (identifiers, stages, counts, models)."""

    request_id: str
    elapsed_ms: int
    stages: list[StageInfo]
    candidates: int | None
    evidence_count: int
    citation_count: int
    invalid_citation_count: int
    models: dict[str, str]


class ResearchResult(BaseModel):
    """The completed research response in its public representation."""

    request_id: str
    status: AnswerStatus
    answer: str
    citations: list[CitationInfo]
    evidence: list[EvidenceInfo]
    conflicts: list[ConflictInfo]
    verification: VerificationSummaryInfo | None
    research_details: ResearchDetailsInfo


class ResearchJobSnapshot(BaseModel):
    """A pollable snapshot of one research job (running/completed/failed)."""

    request_id: str
    job_status: JobStatus
    stages: list[StageInfo]
    result: ResearchResult | None = None
    error: ErrorInfo | None = None


class ResearchQuestionInput(BaseModel):
    """Submit payload; server-side validation is authoritative."""

    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("question must be non-empty")
        if len(value) > MAX_QUESTION_CHARS:
            raise ValueError(
                f"question must be at most {MAX_QUESTION_CHARS} characters ({len(value)} given)"
            )
        return value.strip()


# --- serialization -----------------------------------------------------------


def to_research_result(
    grounded: GroundedAnswer,
    request_id: str,
    trace: RequestTrace | None,
    elapsed_ms: int,
) -> ResearchResult:
    """Map the existing ``GroundedAnswer`` domain object to the safe public view."""
    citations = [
        CitationInfo(
            citation_id=citation.citation_id,
            document_title=citation.document_title,
            document_source=citation.document_source,
            document_type=citation.document_type,
            page=citation.page,
            section=citation.section,
            retrieval_method=citation.retrieval_method,
            retrieval_score=citation.retrieval_score,
            retrieval_rank=citation.retrieval_rank,
        )
        for citation in grounded.citations
    ]
    evidence = [
        EvidenceInfo(
            citation_id=position,
            chunk_id=str(item.chunk_id),
            document_id=str(item.document_id),
            document_title=item.document_title,
            document_source=item.document_source,
            document_type=item.document_type,
            page=item.page,
            section=item.section,
            text=item.text,
            retrieval_method=item.retrieval_method,
            retrieval_score=item.score,
            retrieval_rank=item.rank,
        )
        for position, item in enumerate(grounded.evidence, start=1)
    ]
    conflicts = [
        ConflictInfo(
            conflict_id=conflict.conflict_id,
            conflict_type=conflict.conflict_type,
            description=conflict.description,
            citation_ids=list(conflict.citation_ids),
        )
        for conflict in grounded.conflicts
    ]
    verification = None
    if grounded.verification_report is not None:
        verification = VerificationSummaryInfo(
            counts=grounded.verification_report.counts(),
            claims=[
                VerificationClaimInfo(
                    claim_id=result.claim_id,
                    citation_id=result.citation_id,
                    claim_text=result.claim_text,
                    status=result.status,
                    explanation=result.explanation,
                )
                for result in grounded.verification_report.results
            ],
        )
    stages: list[StageInfo] = []
    counters: dict[str, int] = {}
    models: dict[str, str] = {}
    if trace is not None:
        stages = [
            StageInfo(name=stage.stage, duration_ms=stage.duration_ms, success=stage.success)
            for stage in trace.stages
        ]
        counters = dict(trace.counters)
        models = dict(trace.models)
    return ResearchResult(
        request_id=request_id,
        status=grounded.status,
        answer=grounded.answer,
        citations=citations,
        evidence=evidence,
        conflicts=conflicts,
        verification=verification,
        research_details=ResearchDetailsInfo(
            request_id=request_id,
            elapsed_ms=elapsed_ms,
            stages=stages,
            candidates=counters.get(COUNTER_RETRIEVAL_HYBRID),
            evidence_count=len(grounded.evidence),
            citation_count=len(grounded.citations),
            invalid_citation_count=len(grounded.invalid_citations),
            models=models,
        ),
    )


# --- research service (in-memory job store + background execution) ----------


class ResearchJob:
    """Mutable per-request job state; mutated in the worker thread."""

    def __init__(self, *, request_id: str, question: str) -> None:
        self.request_id = request_id
        self.question = question
        self.job_status: JobStatus = "running"
        self._started = time.monotonic()
        self.trace: RequestTrace | None = None
        self.result: ResearchResult | None = None
        self.error: ErrorInfo | None = None
        self._lock = threading.Lock()

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._started) * 1000)

    def snapshot(self) -> ResearchJobSnapshot:
        stages: list[StageInfo] = []
        if self.trace is not None:
            stages = [
                StageInfo(name=stage.stage, duration_ms=stage.duration_ms, success=stage.success)
                for stage in list(self.trace.stages)
            ]
        with self._lock:
            result = self.result.model_copy(deep=True) if self.result is not None else None
            error = self.error.model_copy(deep=True) if self.error is not None else None
            return ResearchJobSnapshot(
                request_id=self.request_id,
                job_status=self.job_status,
                stages=stages,
                result=result,
                error=error,
            )


class ResearchService:
    """Owns the jobs dict and runs ``researcher`` in a background thread.

    ``researcher(session, question, request_id) -> GroundedAnswer`` is
    the single injection point: the default implementation runs the
    unchanged M10 pipeline (``answer_question`` with verification);
    tests inject deterministic fakes so no database or model is needed.
    """

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        researcher: Callable[[Session, str, str], GroundedAnswer],
        run_async: bool = True,
    ) -> None:
        self._session_factory = session_factory
        self._researcher = researcher
        self._run_async = run_async
        self._jobs: dict[str, ResearchJob] = {}
        self._lock = threading.Lock()

    def submit(self, question: str, request_id: str) -> ResearchJobSnapshot:
        job = ResearchJob(request_id=request_id, question=question.strip())
        with self._lock:
            self._jobs[request_id] = job
        if self._run_async:
            thread = threading.Thread(
                target=self._execute,
                args=(job,),
                daemon=True,
                name=f"deepresearch-{request_id[:8]}",
            )
            thread.start()
        else:
            self._execute(job)
        return self.get(request_id)

    def get(self, request_id: str) -> ResearchJobSnapshot | None:
        with self._lock:
            job = self._jobs.get(request_id)
            return job.snapshot() if job is not None else None

    def _execute(self, job: ResearchJob) -> None:
        try:
            with traced_request(job.request_id) as trace:
                job.trace = trace
                session = self._session_factory()
                try:
                    grounded = self._researcher(session, job.question, job.request_id)
                finally:
                    session.close()
                result = to_research_result(grounded, job.request_id, trace, job.elapsed_ms())
            with job._lock:  # noqa: SLF001 — internal job state
                job.result = result
                job.job_status = "completed"
        except Exception as exc:  # noqa: BLE001 — job boundary: report, never crash the thread
            logger.exception("research job %s failed", job.request_id)
            with job._lock:  # noqa: SLF001 — internal job state
                job.job_status = "failed"
                job.error = ErrorInfo(
                    message="Research request failed.",
                    type=type(exc).__name__,
                    request_id=job.request_id,
                )


# --- default wiring (real pipeline) ------------------------------------------


_PROVIDER_LOCK = threading.Lock()
_default_providers: dict[str, object] | None = None


def _ensure_default_providers() -> dict[str, object]:
    """Build the local embedding/reranker/LLM providers once per process."""
    global _default_providers
    if _default_providers is None:
        with _PROVIDER_LOCK:
            if _default_providers is None:
                settings = get_settings()
                _default_providers = {
                    "embedding": LocalEmbeddingProvider(
                        model_name=settings.embedding_model,
                        model_version=settings.embedding_model_version,
                        device=settings.embedding_device,
                        batch_size=settings.embedding_batch_size,
                    ),
                    "reranker": LocalCrossEncoderReranker(
                        model_name=settings.reranker_model,
                        model_version=settings.reranker_model_version,
                        device=settings.reranker_device,
                        batch_size=settings.reranker_batch_size,
                    ),
                    "llm": default_llm_provider(settings),
                }
    return _default_providers


def default_researcher(session: Session, question: str, request_id: str) -> GroundedAnswer:
    """Run the unchanged M10 grounded pipeline plus citation verification."""
    providers = _ensure_default_providers()
    return answer_question(
        session,
        providers["embedding"],  # type: ignore[arg-type]
        providers["reranker"],  # type: ignore[arg-type]
        providers["llm"],  # type: ignore[arg-type]
        question,
        verify_citations=True,
        request_id=request_id,
    )


def default_session_factory() -> Callable[[], Session]:
    """Prepared DB session factory; schema sync is best-effort at boot."""
    settings = get_settings()
    engine = get_engine(settings)
    try:
        init_db(engine)
    except Exception:  # noqa: BLE001 — availability is reported by job failure
        logger.exception("database schema init failed; research jobs will fail cleanly")
    factory: sessionmaker[Session] = get_session_factory(engine)
    return factory


_default_service: ResearchService | None = None


def get_research_service() -> ResearchService:
    """FastAPI dependency: one shared in-memory service per process."""
    global _default_service
    if _default_service is None:
        _default_service = ResearchService(
            session_factory=default_session_factory(),
            researcher=default_researcher,
        )
    return _default_service


# --- routes ------------------------------------------------------------------

router = APIRouter(prefix="/api", tags=["research"])


@router.post(
    "/research",
    status_code=202,
    response_model=ResearchJobSnapshot,
)
def start_research(  # type: ignore[no-untyped-def]
    payload: ResearchQuestionInput,
    request: Request,
    service: ResearchService = Depends(get_research_service),  # noqa: B008 — FastAPI DI idiom
) -> ResearchJobSnapshot:
    """Start research. M15 request ID is preserved or minted; no client ID needed."""
    request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
    return service.submit(payload.question, request_id)


@router.get("/research/{request_id}", response_model=ResearchJobSnapshot)
def poll_research(  # type: ignore[no-untyped-def]
    request_id: str,
    service: ResearchService = Depends(get_research_service),  # noqa: B008 — FastAPI DI idiom
) -> ResearchJobSnapshot:
    """Return the current job snapshot for polling (running/completed/failed)."""
    job = service.get(request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown research request")
    return job
