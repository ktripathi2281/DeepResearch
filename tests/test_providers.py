"""M19 provider tests — factory, adapters (mocked HTTP), contract, independence.

No cloud credentials, no network: cloud adapters run over
``httpx.MockTransport``. The local default needs no keys at all.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import deepresearch.generation as generation_mod
from deepresearch.agent import run_research_agent
from deepresearch.citation_verification import verify_answer_citations
from deepresearch.config import Settings
from deepresearch.db import init_db
from deepresearch.evaluation import ExperimentConfig
from deepresearch.generation import answer_question
from deepresearch.llm import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    OllamaLLMProvider,
)
from deepresearch.observability import traced_request
from deepresearch.providers import (
    GeminiLLMProvider,
    OpenAICompatibleLLMProvider,
    create_llm_provider,
)
from deepresearch.retrieval import RetrievalResult
from tests.fakes import (
    FakeAgentLLM,
    FakeEmbeddingProvider,
    FakeLLMProvider,
    FakeReranker,
    FakeVerifierLLM,
)
from tests.test_citation_verification import SUPPORTED_JSON

SENTINEL_KEY = "sk-test-sentinel-key-000"


# --- helpers -----------------------------------------------------------------


class CloudLikeFake(FakeLLMProvider):
    """A fake whose identity looks nothing like the local provider."""

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(model_name="cloud-model-x", model_version="cx-1", **kwargs)


class CloudLikeVerifier(FakeVerifierLLM):
    """A scripted verifier whose identity looks nothing like the local provider."""

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(model_name="cloud-verifier-x", **kwargs)


class TokenFake(FakeLLMProvider):
    """A fake that reports real (scripted) token counts."""

    def generate_response(  # type: ignore[no-untyped-def]
        self, prompt, *, system_prompt=None, temperature=0.0, max_tokens=None
    ) -> LLMResponse:
        text = self.generate(
            prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
        )
        return LLMResponse(
            text=text,
            model=self.model_name,
            provider=type(self).__name__,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )


def _evidence(text: str = "evidence text") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        text=text,
        score=0.9,
        rank=1,
        document_title="Title",
        document_type="markdown",
        document_source="doc.md",
        page=None,
        section=None,
        chunk_metadata=None,
    )


def _stub_pipeline(monkeypatch, evidence: list) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(generation_mod, "retrieve_hybrid", lambda *a, **k: [])
    monkeypatch.setattr(generation_mod, "rerank_results", lambda *a, **k: evidence)


def _chat_ok(text: str = "Hello from cloud.", usage: dict | None = None) -> httpx.Response:
    body: dict = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def _gemini_ok(text: str = "Hello from Gemini.", usage: dict | None = None) -> httpx.Response:
    body: dict = {
        "candidates": [{"content": {"parts": [{"text": text}]}, "index": 0}],
    }
    if usage is not None:
        body["usageMetadata"] = usage
    return httpx.Response(200, json=body)


# --- A. factory ---------------------------------------------------------------


def test_factory_default_is_local() -> None:
    provider = create_llm_provider(Settings())
    assert isinstance(provider, OllamaLLMProvider)
    assert isinstance(provider, LLMProvider)
    assert provider.model_name == "qwen3:4b"


def test_factory_explicit_local() -> None:
    provider = create_llm_provider(Settings(llm_provider="ollama"))
    assert isinstance(provider, OllamaLLMProvider)


def test_factory_names_are_case_insensitive() -> None:
    settings = Settings(
        llm_provider="  OpenAI_Compatible  ", openai_compatible_api_key=SENTINEL_KEY
    )
    provider = create_llm_provider(settings)
    assert isinstance(provider, OpenAICompatibleLLMProvider)


def test_factory_openai_compatible() -> None:
    provider = create_llm_provider(
        Settings(llm_provider="openai_compatible", openai_compatible_api_key=SENTINEL_KEY)
    )
    assert isinstance(provider, OpenAICompatibleLLMProvider)
    assert provider.model_name == "gpt-4o-mini"


def test_factory_gemini() -> None:
    provider = create_llm_provider(Settings(llm_provider="gemini", gemini_api_key=SENTINEL_KEY))
    assert isinstance(provider, GeminiLLMProvider)
    assert provider.model_name == "gemini-2.0-flash"


def test_factory_unknown_provider_rejected() -> None:
    with pytest.raises(LLMConfigurationError, match="anthropic"):
        create_llm_provider(Settings(llm_provider="anthropic"))


def test_factory_blank_provider_rejected() -> None:
    with pytest.raises(LLMConfigurationError, match="expected one of"):
        create_llm_provider(Settings(llm_provider="   "))


def test_factory_openai_missing_key() -> None:
    with pytest.raises(LLMConfigurationError, match="OPENAI_COMPATIBLE_API_KEY"):
        create_llm_provider(Settings(llm_provider="openai_compatible"))


def test_factory_gemini_missing_key() -> None:
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        create_llm_provider(Settings(llm_provider="gemini"))


# --- B. strengthened contract --------------------------------------------------


def test_generate_response_default_delegates() -> None:
    fake = FakeLLMProvider(answer="Hi.")
    response = fake.generate_response("Hello?")
    assert response.text == "Hi."
    assert response.model == "fake-llm"
    assert response.provider == "FakeLLMProvider"
    assert response.input_tokens is None
    assert response.output_tokens is None
    assert response.total_tokens is None


def test_ollama_reports_real_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "qwen3:4b", "response": "Hi.", "prompt_eval_count": 12, "eval_count": 3},
        )

    provider = OllamaLLMProvider(transport=httpx.MockTransport(handler))
    try:
        response = provider.generate_response("Hello?")
        assert (response.input_tokens, response.output_tokens, response.total_tokens) == (12, 3, 15)
        assert response.text == "Hi."
    finally:
        provider.close()


def test_ollama_missing_counts_stay_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "qwen3:4b", "response": "Hi."})

    provider = OllamaLLMProvider(transport=httpx.MockTransport(handler))
    try:
        response = provider.generate_response("Hello?")
        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.total_tokens is None
    finally:
        provider.close()


# --- C. OpenAI-compatible adapter ----------------------------------------------


def _openai_provider(handler, **kwargs) -> OpenAICompatibleLLMProvider:  # type: ignore[no-untyped-def]
    return OpenAICompatibleLLMProvider(
        api_key=SENTINEL_KEY, transport=httpx.MockTransport(handler), **kwargs
    )


def test_openai_request_shape_and_metadata() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content.decode())
        return _chat_ok()

    provider = _openai_provider(handler, model="test-model")
    try:
        assert provider.generate("Hi", system_prompt="Be brief.") == "Hello from cloud."
        assert provider.model_name == "test-model"
        assert provider.model_version is None
        assert isinstance(provider, LLMProvider)
    finally:
        provider.close()
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/chat/completions"  # base_url prefix + chat path
    assert seen["auth"] == f"Bearer {SENTINEL_KEY}"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]


def test_openai_usage_reported() -> None:
    provider = _openai_provider(
        lambda request: _chat_ok(
            usage={"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
        )
    )
    try:
        response = provider.generate_response("Hi")
        assert (response.input_tokens, response.output_tokens, response.total_tokens) == (7, 5, 12)
    finally:
        provider.close()


def test_openai_usage_absent_stays_none() -> None:
    provider = _openai_provider(lambda request: _chat_ok())
    try:
        response = provider.generate_response("Hi")
        assert response.input_tokens is None and response.total_tokens is None
    finally:
        provider.close()


@pytest.mark.parametrize(
    "body",
    [
        "not json{{{",
        "[1, 2]",
        "{}",
        '{"choices": []}',
        '{"choices": [{}]}',
        '{"choices": [{"message": {}}]}',
        '{"choices": [{"message": {"content": "   "}}]}',
    ],
)
def test_openai_malformed_responses_rejected(body: str) -> None:
    provider = _openai_provider(lambda request: httpx.Response(200, text=body))
    try:
        with pytest.raises(LLMResponseError):
            provider.generate("Hi")
    finally:
        provider.close()


def test_openai_timeout_and_connection_mapped() -> None:
    slow = _openai_provider(
        lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=request)),
        timeout_seconds=5,
    )
    try:
        with pytest.raises(LLMTimeoutError, match="5"):
            slow.generate("Hi")
    finally:
        slow.close()
    dead = _openai_provider(
        lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused", request=request))
    )
    try:
        with pytest.raises(LLMConnectionError):
            dead.generate("Hi")
    finally:
        dead.close()


def test_openai_credential_and_model_errors() -> None:
    denied = _openai_provider(lambda request: httpx.Response(401, text="bad key"))
    try:
        with pytest.raises(LLMResponseError, match="credentials"):
            denied.generate("Hi")
    finally:
        denied.close()
    missing = _openai_provider(lambda request: httpx.Response(404, text="no model"))
    try:
        with pytest.raises(LLMModelNotFoundError):
            missing.generate("Hi")
    finally:
        missing.close()
    broken = _openai_provider(lambda request: httpx.Response(500, text="boom"))
    try:
        with pytest.raises(LLMResponseError, match="500"):
            broken.generate("Hi")
    finally:
        broken.close()


def test_openai_missing_key_fails_before_http() -> None:
    calls: list = []
    provider = OpenAICompatibleLLMProvider(
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or _chat_ok()  # pragma: no cover
        )
    )
    try:
        with pytest.raises(LLMConfigurationError, match="OPENAI_COMPATIBLE_API_KEY"):
            provider.generate("Hi")
    finally:
        provider.close()
    assert calls == []


def test_openai_constructor_validation() -> None:
    with pytest.raises(LLMError):
        OpenAICompatibleLLMProvider(api_key=SENTINEL_KEY, timeout_seconds=0)
    with pytest.raises(LLMError):
        OpenAICompatibleLLMProvider(api_key=SENTINEL_KEY, model="  ")


# --- D. Gemini adapter ----------------------------------------------------------


def _gemini_provider(handler, **kwargs) -> GeminiLLMProvider:  # type: ignore[no-untyped-def]
    return GeminiLLMProvider(api_key=SENTINEL_KEY, transport=httpx.MockTransport(handler), **kwargs)


def test_gemini_request_shape_and_metadata() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["api_key_header"] = request.headers.get("x-goog-api-key")
        seen["query"] = str(request.url.query)
        seen["body"] = json.loads(request.content.decode())
        return _gemini_ok()

    provider = _gemini_provider(handler, model="gemini-2.0-flash")
    try:
        assert provider.generate("Hi", system_prompt="Be brief.") == "Hello from Gemini."
        assert provider.model_name == "gemini-2.0-flash"
        assert provider.model_version is None
        assert isinstance(provider, LLMProvider)
    finally:
        provider.close()
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1beta/models/gemini-2.0-flash:generateContent"
    assert seen["api_key_header"] == SENTINEL_KEY
    assert "key=" not in seen["query"]  # never in the URL: clients log URLs
    assert seen["body"]["contents"][0]["parts"] == [{"text": "Hi"}]
    assert seen["body"]["systemInstruction"]["parts"] == [{"text": "Be brief."}]


def test_gemini_usage_reported() -> None:
    provider = _gemini_provider(
        lambda request: _gemini_ok(
            usage={"promptTokenCount": 9, "candidatesTokenCount": 4, "totalTokenCount": 13}
        )
    )
    try:
        response = provider.generate_response("Hi")
        assert (response.input_tokens, response.output_tokens, response.total_tokens) == (9, 4, 13)
    finally:
        provider.close()


def test_gemini_blocked_and_empty_rejected() -> None:
    blocked = _gemini_provider(
        lambda request: httpx.Response(
            200, json={"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []}
        )
    )
    try:
        with pytest.raises(LLMResponseError, match="blocked"):
            blocked.generate("Hi")
    finally:
        blocked.close()
    empty = _gemini_provider(lambda request: httpx.Response(200, json={"candidates": []}))
    try:
        with pytest.raises(LLMResponseError, match="no candidates"):
            empty.generate("Hi")
    finally:
        empty.close()


@pytest.mark.parametrize("body", ["not json{{{", "[1]", "{}", '{"candidates": [{}]}'])
def test_gemini_malformed_responses_rejected(body: str) -> None:
    provider = _gemini_provider(lambda request: httpx.Response(200, text=body))
    try:
        with pytest.raises(LLMResponseError):
            provider.generate("Hi")
    finally:
        provider.close()


def test_gemini_timeout_connection_and_status_mapped() -> None:
    slow = _gemini_provider(
        lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=request)),
        timeout_seconds=7,
    )
    try:
        with pytest.raises(LLMTimeoutError, match="7"):
            slow.generate("Hi")
    finally:
        slow.close()
    dead = _gemini_provider(
        lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused", request=request))
    )
    try:
        with pytest.raises(LLMConnectionError):
            dead.generate("Hi")
    finally:
        dead.close()
    denied = _gemini_provider(lambda request: httpx.Response(400, text="bad key"))
    try:
        with pytest.raises(LLMResponseError, match="GEMINI_API_KEY"):
            denied.generate("Hi")
    finally:
        denied.close()
    missing = _gemini_provider(lambda request: httpx.Response(404, text="no model"))
    try:
        with pytest.raises(LLMModelNotFoundError):
            missing.generate("Hi")
    finally:
        missing.close()


def test_gemini_missing_key_fails_before_http() -> None:
    calls: list = []
    provider = GeminiLLMProvider(
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or _gemini_ok()  # pragma: no cover
        )
    )
    try:
        with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
            provider.generate("Hi")
    finally:
        provider.close()
    assert calls == []


# --- E. provider independence ----------------------------------------------------


def test_generation_works_with_nonlocal_named_provider(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _stub_pipeline(monkeypatch, [_evidence("Key fact.")])
    llm = CloudLikeFake(answer="The key fact is X [1].")
    result = answer_question(None, None, None, llm, "What is the fact?")  # type: ignore[arg-type]
    assert result.status == "answered"
    assert result.model_name == "cloud-model-x"


def test_verifier_works_with_nonlocal_named_provider() -> None:
    report = verify_answer_citations(
        "Claim [1].", [_evidence()], CloudLikeVerifier(responses=[SUPPORTED_JSON])
    )
    assert [r.status for r in report.results] == ["supported"]


def test_agent_works_with_nonlocal_named_provider() -> None:
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    try:
        with Session(bind=engine) as session:
            decision = json.dumps({"action": "finish", "arguments": {}, "reason": "done"})
            llm = FakeAgentLLM(decisions=[decision], model_name="cloud-agent-model")
            result = run_research_agent(session, FakeEmbeddingProvider(), llm, "Q?")
            assert result.termination_reason == "finished"
    finally:
        engine.dispose()


# --- F. observability roles and tokens --------------------------------------------


def test_trace_records_tokens_and_keeps_roles_separate(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _stub_pipeline(monkeypatch, [_evidence("Key fact.")])
    llm = TokenFake(answer="The key fact is X [1].")
    verifier = FakeVerifierLLM(responses=[SUPPORTED_JSON])
    with traced_request("m19-tokens") as trace:
        result = answer_question(
            None,  # type: ignore[arg-type]
            FakeEmbeddingProvider(),
            FakeReranker(),
            llm,
            "What is the fact?",
            verify_citations=True,
            verifier=verifier,
        )
    assert result.status == "answered"
    assert trace.models["llm"] == "fake-llm"
    assert trace.models["llm_provider"] == "TokenFake"
    assert trace.models["verifier"] == "fake-verifier"
    assert trace.models["verifier_provider"] == "FakeVerifierLLM"
    assert trace.counters["llm_calls"] == 2  # one llm call, one verifier call
    assert trace.input_tokens == 10  # only the reporting provider contributed
    assert trace.output_tokens == 20
    assert trace.total_tokens == 30


# --- G. evaluation identity --------------------------------------------------------


def test_experiment_identity_distinguishes_providers() -> None:
    local = ExperimentConfig(dataset_version="eval-dev-v1")
    assert local.llm_provider == "ollama"
    cloud = ExperimentConfig(
        dataset_version="eval-dev-v1", llm_provider="openai_compatible", llm_model="gpt-4o-mini"
    )
    assert local.identity != cloud.identity
    same = ExperimentConfig(
        dataset_version="eval-dev-v1", llm_provider="openai_compatible", llm_model="gpt-4o-mini"
    )
    assert cloud.identity == same.identity
