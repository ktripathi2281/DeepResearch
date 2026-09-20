"""Local-first observability and request tracing — Milestone 15.

In-memory, dependency-free instrumentation answering: what ran, how
long each stage took, what counts resulted, which models were used,
and how it terminated. No OpenTelemetry/Prometheus/dashboards, no
database persistence, no external services.

Core rules:

- Passive: every helper no-ops when no trace is active, so existing
  code paths behave identically outside a traced request.
- Private: traces and logs carry identifiers and counts — never
  prompts, document contents, chain-of-thought, secrets, or headers.
- Isolated: request state lives in a ``ContextVar``, never a global,
  so concurrent requests cannot overwrite each other.
- Honest: failures record stage failure + duration and propagate;
  token/cost fields stay ``None`` unless a provider reports them.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

REQUEST_ID_HEADER = "X-Request-ID"
MAX_REQUEST_ID_LENGTH = 128

# Canonical counter keys used across the pipeline (extensible dict).
COUNTER_RETRIEVAL_VECTOR = "retrieval_vector_candidates"
COUNTER_RETRIEVAL_BM25 = "retrieval_bm25_candidates"
COUNTER_RETRIEVAL_HYBRID = "retrieval_hybrid_candidates"
COUNTER_RERANKER_CANDIDATES = "reranker_candidates"
COUNTER_RERANKER_RESULTS = "reranker_results"
COUNTER_EVIDENCE = "evidence_chunks"
COUNTER_LLM_CALLS = "llm_calls"
COUNTER_CITATIONS = "citation_count"
COUNTER_INVALID_CITATIONS = "invalid_citation_count"
COUNTER_VERIFICATION_CALLS = "verification_calls"
COUNTER_VERIFICATION_CLAIMS = "verification_claims"
COUNTER_AGENT_ITERATIONS = "agent_iterations"
COUNTER_AGENT_TOOL_CALLS = "agent_tool_calls"
COUNTER_DUPLICATE_EVIDENCE = "duplicate_evidence_count"
COUNTER_VERIFICATION_SUPPORTED = "verification_supported"
COUNTER_VERIFICATION_UNSUPPORTED = "verification_unsupported"
COUNTER_VERIFICATION_INSUFFICIENT = "verification_insufficient_evidence"
COUNTER_VERIFICATION_UNVERIFIABLE = "verification_unverifiable"


@dataclass(frozen=True)
class StageTiming:
    """One timed stage: monotonic duration, explicit success/failure."""

    stage: str
    duration_ms: int
    success: bool = True
    error_type: str | None = None


@dataclass
class RequestTrace:
    """Mutable per-request trace; serializable, never shared between requests."""

    request_id: str
    status: str = "ok"
    error_type: str | None = None
    stages: list[StageTiming] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, str] = field(default_factory=dict)
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: float | None = None

    def increment(self, key: str, amount: int = 1) -> int:
        self.counters[key] = self.counters.get(key, 0) + amount
        return self.counters[key]

    def set_model(self, role: str, name: str) -> None:
        self.models[role] = name

    def note(self, key: str, value: str) -> None:
        """Record a small string fact (e.g. termination reason)."""
        self.attributes[key] = value

    def record_llm_call(
        self,
        *,
        model: str,
        provider: str,
        duration_ms: int,
        role: str = "llm",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        estimated_cost: float | None = None,
    ) -> None:
        self.increment(COUNTER_LLM_CALLS)
        self.set_model(role, model)
        self.set_model(f"{role}_provider", provider)
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens
        if self.input_tokens is not None or self.output_tokens is not None:
            self.total_tokens = (self.input_tokens or 0) + (self.output_tokens or 0)
        if estimated_cost is not None:
            self.estimated_cost = (self.estimated_cost or 0.0) + estimated_cost

    def finish(self, status: str = "ok", error_type: str | None = None) -> RequestTrace:
        self.status = status
        self.error_type = error_type
        return self

    def to_dict(self) -> dict:
        return asdict(self)


_current_trace: ContextVar[RequestTrace | None] = ContextVar("deepresearch_trace", default=None)


def get_current_trace() -> RequestTrace | None:
    """Return the active request trace, or None outside a traced request."""
    return _current_trace.get()


def normalize_request_id(value: str | None) -> str:
    """Preserve a supplied ID when usable; otherwise generate a UUID."""
    if value and value.strip() and len(value.strip()) <= MAX_REQUEST_ID_LENGTH:
        return value.strip()
    return uuid.uuid4().hex


@contextmanager
def traced_request(request_id: str | None = None) -> Iterator[RequestTrace]:
    """Establish an isolated request trace for the enclosed block."""
    trace = RequestTrace(request_id=normalize_request_id(request_id))
    token = _current_trace.set(trace)
    try:
        yield trace
    finally:
        _current_trace.reset(token)


@contextmanager
def traced_stage(stage: str) -> Iterator[None]:
    """Time one stage on the active trace (no-op without one).

    Records duration and success/failure, then re-raises any
    exception — instrumentation never swallows pipeline errors.
    """
    trace = get_current_trace()
    if trace is None:
        yield
        return
    started = time.monotonic_ns()
    try:
        yield
    except Exception as exc:
        elapsed_ms = int((time.monotonic_ns() - started) / 1_000_000)
        trace.stages.append(
            StageTiming(
                stage=stage, duration_ms=elapsed_ms, success=False, error_type=type(exc).__name__
            )
        )
        raise
    elapsed_ms = int((time.monotonic_ns() - started) / 1_000_000)
    trace.stages.append(StageTiming(stage=stage, duration_ms=elapsed_ms, success=True))


def count(counter: str, amount: int = 1) -> None:
    """Increment a counter on the active trace (no-op without one)."""
    trace = get_current_trace()
    if trace is not None:
        trace.increment(counter, amount)
