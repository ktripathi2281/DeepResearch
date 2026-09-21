"""M20 reliability tests — config, health/readiness, job lifecycle,
retention, concurrency, error contract, shutdown, request IDs.

Deterministic fakes/mocks throughout: no Ollama, no cloud network,
no live database required (readiness probes are monkeypatched).
"""

from __future__ import annotations

import threading
import time
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import deepresearch.main as main_mod
import deepresearch.research_api as research_api_mod
from deepresearch.citation_verification import CitationVerificationReport
from deepresearch.config import ConfigurationError, Settings
from deepresearch.generation import GroundedAnswer
from deepresearch.llm import LLMTimeoutError
from deepresearch.observability import traced_request
from deepresearch.research_api import (
    ResearchService,
    ResearchServiceClosedError,
    close_default_resources,
)

client = TestClient(main_mod.app, raise_server_exceptions=False)


# --- local builders (no DB, no models) ----------------------------------------


class FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _grounded(answer: str = "Answer [1].") -> GroundedAnswer:
    from deepresearch.citations import Citation
    from deepresearch.retrieval import RetrievalResult

    chunk = uuid.uuid4()
    doc = uuid.uuid4()
    item = RetrievalResult(
        chunk_id=chunk,
        document_id=doc,
        chunk_index=0,
        text="Evidence.",
        score=0.9,
        rank=1,
        document_title="T",
        document_type="markdown",
        document_source="s.md",
        page=None,
        section=None,
        chunk_metadata=None,
    )
    citation = Citation(
        citation_id=1,
        chunk_id=chunk,
        document_id=doc,
        document_title="T",
        document_source="s.md",
        document_type="markdown",
        page=None,
        section=None,
        retrieval_method="reranked",
        retrieval_score=0.9,
        retrieval_rank=1,
    )
    return GroundedAnswer(
        answer=answer,
        evidence=[item],
        citations=[citation],
        model_name="fake-llm",
        verification_report=CitationVerificationReport(),
    )


def _service(researcher, **kwargs) -> ResearchService:  # type: ignore[no-untyped-def]
    return ResearchService(session_factory=FakeSession, researcher=researcher, **kwargs)


@pytest.fixture(autouse=True)
def _clear_overrides():  # type: ignore[no-untyped-def]
    yield
    main_mod.app.dependency_overrides.clear()


# --- A. configuration ----------------------------------------------------------


def test_valid_local_config_passes() -> None:
    settings = Settings().validate_for_runtime()
    assert settings.llm_provider == "ollama"


def test_cloud_keys_not_required_for_local() -> None:
    settings = Settings(llm_provider="ollama", openai_compatible_api_key=None, gemini_api_key=None)
    assert settings.validate_for_runtime().llm_provider == "ollama"


@pytest.mark.parametrize("bad", ["anthropic", "", "   ", "ollama2"])
def test_invalid_provider_rejected(bad: str) -> None:
    with pytest.raises(ConfigurationError, match="expected one of"):
        Settings(llm_provider=bad).validate_for_runtime()


@pytest.mark.parametrize(
    "field,value",
    [
        ("api_port", 0),
        ("api_port", 99999),
        ("ollama_timeout_seconds", 0),
        ("ollama_max_tokens", 0),
        ("ollama_temperature", -0.5),
        ("retrieval_top_k", 0),
        ("agent_max_iterations", 0),
        ("research_max_retained_jobs", 0),
    ],
)
def test_invalid_numeric_config_rejected_at_load(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


@pytest.mark.parametrize("bad", ["not-a-url", "*", "ftp://x.test", "http://x.test/path?q=1"])
def test_invalid_cors_rejected(bad: str) -> None:
    with pytest.raises(ConfigurationError, match="CORS_ORIGINS"):
        Settings(cors_origins=bad).validate_for_runtime()


def test_empty_cors_means_api_only_mode() -> None:
    assert Settings(cors_origins="").validate_for_runtime().cors_origins == ""


@pytest.mark.parametrize("bad", ["sqlite:///x.db", "not a url", "redis://localhost:6379/0"])
def test_invalid_database_scheme_rejected(bad: str) -> None:
    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        Settings(database_url=bad).validate_for_runtime()


def test_config_errors_never_contain_secrets(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-live-SENTINEL-123")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSENTINEL-456")
    settings = Settings(llm_provider="bogus-provider")
    with pytest.raises(ConfigurationError) as excinfo:
        settings.validate_for_runtime()
    assert "sk-live-SENTINEL-123" not in str(excinfo.value)
    assert "AIzaSENTINEL-456" not in str(excinfo.value)


# --- B. health / readiness ------------------------------------------------------


def test_health_needs_no_dependencies() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "x-request-id" in response.headers


def test_readiness_ok_when_dependencies_ready(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(main_mod, "check_connection", lambda engine: True)
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_fails_when_db_down(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(main_mod, "check_connection", lambda engine: False)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_readiness_fails_on_bad_provider_config(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(main_mod, "check_connection", lambda engine: True)
    monkeypatch.setattr(main_mod, "settings", Settings(llm_provider="bogus"))
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_readiness_bodies_carry_no_secrets(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-live-SENTINEL-789")
    monkeypatch.setattr(main_mod, "check_connection", lambda engine: False)
    response = client.get("/ready")
    assert response.status_code == 503
    assert "sk-live-SENTINEL-789" not in response.text
    assert "deepresearch:" not in response.text  # no DB credentials either


# --- C. job lifecycle -------------------------------------------------------------


def test_unknown_job_returns_controlled_404() -> None:
    response = client.get("/api/research/no-such-job")
    assert response.status_code == 404
    assert response.json() == {"detail": "unknown research request"}


def test_submit_running_completed_failed_cycle() -> None:
    started = threading.Event()
    release = threading.Event()

    def researcher(session, question, request_id):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(timeout=5)
        return _grounded()

    svc = _service(researcher, run_async=True)
    assert svc.submit("Q?", "cycle-1").job_status == "running"
    assert started.wait(timeout=5)
    assert svc.get("cycle-1").job_status == "running"  # type: ignore[union-attr]
    release.set()
    deadline = time.monotonic() + 5
    while svc.get("cycle-1").job_status == "running":  # type: ignore[union-attr]
        assert time.monotonic() < deadline
        time.sleep(0.01)
    final = svc.get("cycle-1")
    assert final is not None and final.job_status == "completed"
    assert final.result is not None
    # Terminal state is stable: repeated polls never regress.
    assert svc.get("cycle-1").job_status == "completed"  # type: ignore[union-attr]


def test_pipeline_timeout_becomes_safe_failed_job() -> None:
    def researcher(session, question, request_id):  # type: ignore[no-untyped-def]
        raise LLMTimeoutError("fake provider timed out")

    svc = _service(researcher, run_async=False)
    snapshot = svc.submit("Q?", "timeout-1")
    assert snapshot.job_status == "failed"
    assert snapshot.error is not None
    assert snapshot.error.message == "Research request failed."
    assert snapshot.error.type == "LLMTimeoutError"
    assert snapshot.error.request_id == "timeout-1"
    assert snapshot.result is None


def test_session_closed_even_on_failure() -> None:
    sessions: list[FakeSession] = []

    def factory() -> FakeSession:
        session = FakeSession()
        sessions.append(session)
        return session  # type: ignore[return-value]

    def researcher(session, question, request_id):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    svc = ResearchService(session_factory=factory, researcher=researcher, run_async=False)
    assert svc.submit("Q?", "close-1").job_status == "failed"
    assert sessions and all(session.closed for session in sessions)


def test_resubmit_running_id_is_idempotent() -> None:
    calls: list[str] = []
    release = threading.Event()

    def researcher(session, question, request_id):  # type: ignore[no-untyped-def]
        calls.append(question)
        assert release.wait(timeout=5)
        return _grounded()

    svc = _service(researcher, run_async=True)
    first = svc.submit("Same?", "dup-1")
    second = svc.submit("Same?", "dup-1")
    assert first.job_status == second.job_status == "running"
    release.set()
    deadline = time.monotonic() + 5
    while svc.get("dup-1").job_status == "running":  # type: ignore[union-attr]
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert calls == ["Same?"]  # exactly one execution, no duplicate thread


def test_resubmit_terminal_id_starts_fresh() -> None:
    answers = iter(["First [1].", "Second [1]."])

    def researcher(session, question, request_id):  # type: ignore[no-untyped-def]
        return _grounded(next(answers))

    svc = _service(researcher, run_async=False)
    assert svc.submit("Q?", "fresh-1").result.answer == "First [1]."  # type: ignore[union-attr]
    second = svc.submit("Q?", "fresh-1")
    assert second.job_status == "completed"
    assert second.result is not None and second.result.answer == "Second [1]."


def test_max_retained_jobs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_retained_jobs"):
        _service(lambda s, q, r: _grounded(), max_retained_jobs=0)


def test_terminal_eviction_is_oldest_first_and_bounded() -> None:
    svc = _service(
        lambda s, q, r: _grounded(f"Answer {q} [1]."), run_async=False, max_retained_jobs=2
    )
    for index in range(4):
        svc.submit(f"Q{index}?", f"evict-{index}")
    assert svc.get("evict-0") is None
    assert svc.get("evict-1") is None
    assert svc.get("evict-2") is not None
    assert svc.get("evict-3") is not None
    assert svc.get("evict-3").result.answer == "Answer Q3? [1]."  # type: ignore[union-attr]


def test_running_jobs_never_evicted() -> None:
    release = threading.Event()

    def researcher(session, question, request_id):  # type: ignore[no-untyped-def]
        if request_id == "keep-running":
            assert release.wait(timeout=10)
        return _grounded()

    svc = _service(researcher, run_async=True, max_retained_jobs=1)
    svc.submit("Q?", "keep-running")
    deadline = time.monotonic() + 5
    while svc.get("keep-running") is None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    for index in range(3):
        done = svc.submit(f"Q{index}?", f"done-{index}")
        assert done.job_status == "completed"
    assert svc.get("keep-running") is not None
    assert svc.get("keep-running").job_status == "running"  # type: ignore[union-attr]
    assert svc.get("done-2") is not None  # newest terminal retained
    release.set()


def test_shutdown_marks_running_interrupted_and_blocks_submits() -> None:
    release = threading.Event()
    started = threading.Event()

    def researcher(session, question, request_id):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(timeout=10)
        return _grounded()  # pragma: no cover — shutdown wins the race by design

    svc = _service(researcher, run_async=True)
    svc.submit("Q?", "sd-1")
    assert started.wait(timeout=5)
    marked = svc.shutdown()
    assert marked == 1
    snapshot = svc.get("sd-1")
    assert snapshot is not None and snapshot.job_status == "failed"
    assert snapshot.error is not None
    assert snapshot.error.type == "interrupted"
    assert "shutdown" in snapshot.error.message
    assert snapshot.error.request_id == "sd-1"
    assert snapshot.result is None
    with pytest.raises(ResearchServiceClosedError):
        svc.submit("Q?", "sd-2")
    release.set()


def test_shutdown_leaves_terminal_jobs_alone() -> None:
    svc = _service(lambda s, q, r: _grounded(), run_async=False)
    svc.submit("Q?", "term-1")
    assert svc.shutdown() == 0
    assert svc.get("term-1") is not None
    assert svc.get("term-1").job_status == "completed"  # type: ignore[union-attr]


def test_close_default_resources_safe_when_uninitialized() -> None:
    assert close_default_resources() == 0


def test_close_default_resources_marks_running(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    release = threading.Event()

    def researcher(session, question, request_id):  # type: ignore[no-untyped-def]
        assert release.wait(timeout=10)
        return _grounded()  # pragma: no cover

    svc = _service(researcher, run_async=True)
    saved_service = research_api_mod._default_service
    saved_providers = research_api_mod._default_providers
    saved_engine = research_api_mod._default_engine
    research_api_mod._default_service = svc
    research_api_mod._default_providers = None
    research_api_mod._default_engine = None
    try:
        svc.submit("Q?", "close-default-1")
        assert close_default_resources() == 1
        snapshot = svc.get("close-default-1")
        assert snapshot is not None and snapshot.job_status == "failed"
        assert snapshot.error is not None and snapshot.error.type == "interrupted"
    finally:
        release.set()
        research_api_mod._default_service = saved_service
        research_api_mod._default_providers = saved_providers
        research_api_mod._default_engine = saved_engine


# --- D. concurrency ------------------------------------------------------------------


def test_simultaneous_submissions_stay_isolated() -> None:
    count = 8
    release = threading.Event()
    svc = _service(
        lambda s, q, r: _grounded() if release.wait(timeout=10) else _grounded(),
        run_async=True,
    )
    ids = [f"conc-{index}" for index in range(count)]
    errors: list[BaseException] = []

    def submit_one(request_id: str) -> None:
        try:
            svc.submit(f"Question {request_id}?", request_id)
        except BaseException as exc:  # pragma: no cover — any failure is a bug
            errors.append(exc)

    threads = [threading.Thread(target=submit_one, args=(rid,)) for rid in ids]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5
    while any(svc.get(rid) is None for rid in ids):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert not errors
    assert len({rid for rid in ids}) == count  # unique IDs, all stored
    snapshots = [svc.get(rid) for rid in ids]
    assert all(s is not None and s.job_status == "running" for s in snapshots)
    release.set()
    for thread in threads:
        thread.join(timeout=10)
    deadline = time.monotonic() + 10
    while any(
        (s := svc.get(rid)) is None or s.job_status == "running"  # noqa: E731
        for rid in ids
    ):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    for rid in ids:
        final = svc.get(rid)
        assert final is not None and final.job_status == "completed"
        assert final.request_id == rid
        assert final.result is not None and final.result.request_id == rid
        assert final.result.research_details.request_id == rid


def test_concurrent_polling_is_consistent() -> None:
    release = threading.Event()

    def researcher(session, question, request_id):  # type: ignore[no-untyped-def]
        assert release.wait(timeout=10)
        return _grounded()

    svc = _service(researcher, run_async=True)
    svc.submit("Q?", "poll-1")
    results: list = []
    errors: list[BaseException] = []

    def poll_many() -> None:
        try:
            for _ in range(25):
                snapshot = svc.get("poll-1")
                assert snapshot is not None
                results.append((snapshot.job_status, snapshot.request_id))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=poll_many) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    release.set()
    assert not errors
    assert len(results) == 150
    assert {request_id for _, request_id in results} == {"poll-1"}


# --- E. error contract -----------------------------------------------------------------


def test_422_malformed_requests_have_no_traceback() -> None:
    for payload in [{}, {"question": ""}, {"question": "   "}, {"nonsense": 1}]:
        response = client.post("/api/research", json=payload)
        assert response.status_code == 422
        assert "Traceback" not in response.text
        assert "x-request-id" in response.headers


def test_500_envelope_matches_request_header() -> None:
    def boom() -> None:
        raise RuntimeError("C:\\private\\detail")

    main_mod.app.dependency_overrides[research_api_mod.get_research_service] = boom  # type: ignore[assignment]
    try:
        response = client.post(
            "/api/research", json={"question": "Q?"}, headers={"X-Request-ID": "hdr-500"}
        )
        assert response.status_code == 500
        body = response.json()
        assert set(body["error"]) == {"message", "type", "request_id"}
        assert body["error"]["message"] == "Internal server error."
        assert body["error"]["request_id"] == "hdr-500"
        assert response.headers["x-request-id"] == "hdr-500"
        assert "C:\\private" not in response.text
        assert "Traceback" not in response.text
    finally:
        main_mod.app.dependency_overrides.clear()


def test_503_when_service_closed() -> None:
    svc = _service(lambda s, q, r: _grounded(), run_async=False)
    svc.shutdown()
    main_mod.app.dependency_overrides[research_api_mod.get_research_service] = lambda: svc
    try:
        response = client.post("/api/research", json={"question": "Q?"})
        assert response.status_code == 503
        body = response.json()
        assert body["error"]["type"] == "shutting_down"
        assert body["error"]["request_id"]
        assert "Traceback" not in response.text
    finally:
        main_mod.app.dependency_overrides.clear()


def test_error_bodies_carry_no_secrets(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-live-SENTINEL-321")
    response = client.post("/api/research", json={"question": ""})
    assert response.status_code == 422
    assert "sk-live-SENTINEL-321" not in response.text
    unknown = client.get("/api/research/missing")
    assert "sk-live-SENTINEL-321" not in unknown.text


# --- F. request ID consistency ------------------------------------------------------------


def test_request_id_flows_end_to_end() -> None:
    shared = _service(lambda s, q, r: _grounded(), run_async=False)
    main_mod.app.dependency_overrides[research_api_mod.get_research_service] = lambda: shared
    try:
        submitted = client.post(
            "/api/research",
            json={"question": "Trace me?"},
            headers={"X-Request-ID": "e2e-trace-1"},
        )
        assert submitted.status_code == 202
        assert submitted.json()["request_id"] == "e2e-trace-1"
        assert submitted.headers["x-request-id"] == "e2e-trace-1"
        polled = client.get("/api/research/e2e-trace-1", headers={"X-Request-ID": "e2e-poll-1"})
        assert polled.status_code == 200
        body = polled.json()
        assert body["request_id"] == "e2e-trace-1"
        # Two ID levels, each consistent: the header echoes THIS http
        # request's ID, while the body carries the background job's ID.
        assert polled.headers["x-request-id"] == "e2e-poll-1"
        assert body["result"]["request_id"] == "e2e-trace-1"
        assert body["result"]["research_details"]["request_id"] == "e2e-trace-1"
    finally:
        main_mod.app.dependency_overrides.clear()


# --- G. lifespan / shutdown ---------------------------------------------------------------


def test_lifespan_starts_and_stops_cleanly(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    saved = (
        research_api_mod._default_service,
        research_api_mod._default_providers,
        research_api_mod._default_engine,
    )
    try:
        with TestClient(main_mod.app) as live:
            assert live.get("/health").status_code == 200
    finally:
        (
            research_api_mod._default_service,
            research_api_mod._default_providers,
            research_api_mod._default_engine,
        ) = saved


def test_lifespan_rejects_bad_config_with_clear_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(main_mod, "settings", Settings(llm_provider="bogus"))
    with pytest.raises(RuntimeError, match="expected one of"):
        with TestClient(main_mod.app):
            pass  # pragma: no cover — startup must fail before serving


# --- H. provider client lifecycle ------------------------------------------------------------


def test_provider_http_client_reused_and_closed() -> None:
    from deepresearch.providers import OpenAICompatibleLLMProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleLLMProvider(
        api_key="sk-test", transport=httpx.MockTransport(handler)
    )
    try:
        provider.generate("Hi")
        first = provider._client  # noqa: SLF001 — lifecycle assertion needs the handle
        provider.generate("Hi again")
        assert provider._client is first  # noqa: SLF001 — reused, not rebuilt
        assert first is not None
    finally:
        provider.close()
    assert provider._client is None  # noqa: SLF001 — close releases the client


# --- I. performance sanity (loose bounds; documents adequacy) ----------------------------------


def test_submission_latency_sane() -> None:
    svc = _service(lambda s, q, r: _grounded(), run_async=False)
    started = time.perf_counter()
    snapshot = svc.submit("Q?", "perf-1")
    elapsed = time.perf_counter() - started
    assert snapshot.job_status == "completed"
    assert elapsed < 2.0  # fake pipeline: milliseconds; bound guards regressions


def test_health_latency_sane() -> None:
    started = time.perf_counter()
    assert client.get("/health").status_code == 200
    assert time.perf_counter() - started < 1.0


def test_trace_creation_scales_linearly() -> None:
    started = time.perf_counter()
    for index in range(100):
        with traced_request(f"perf-{index}"):
            pass
    assert time.perf_counter() - started < 2.0
