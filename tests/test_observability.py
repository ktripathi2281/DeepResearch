"""M15 unit tests — traces, stages, IDs, isolation, log privacy (no DB/model)."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from fastapi.testclient import TestClient

from deepresearch.main import app
from deepresearch.observability import (
    RequestTrace,
    count,
    get_current_trace,
    normalize_request_id,
    traced_request,
    traced_stage,
)

client = TestClient(app)


# --- request IDs --------------------------------------------------------------


def test_normalize_request_id() -> None:
    assert normalize_request_id("abc-123") == "abc-123"
    assert normalize_request_id("  abc  ") == "abc"
    generated = normalize_request_id(None)
    assert len(generated) == 32  # uuid4 hex
    assert normalize_request_id("") != normalize_request_id("")
    assert len(normalize_request_id("x" * 200)) == 32  # too long → regenerate


def test_middleware_preserves_and_generates_ids() -> None:
    first = client.get("/health", headers={"X-Request-ID": "req-1"})
    assert first.headers["x-request-id"] == "req-1"
    second = client.get("/health")
    assert len(second.headers["x-request-id"]) == 32
    assert second.headers["x-request-id"] != first.headers["x-request-id"]


def test_health_stays_quiet_but_traced(caplog) -> None:  # type: ignore[no-untyped-def]
    with caplog.at_level(logging.INFO):
        client.get("/health", headers={"X-Request-ID": "quiet-1"})
    assert not [r for r in caplog.records if getattr(r, "event", None) == "request_finished"]
    assert get_current_trace() is None  # context reset after the request


# --- timing ---------------------------------------------------------------------


def test_stage_records_duration_and_success() -> None:
    with traced_request("t1") as trace:
        with traced_stage("work"):
            pass
        assert len(trace.stages) == 1
        stage = trace.stages[0]
        assert stage.stage == "work"
        assert isinstance(stage.duration_ms, int) and stage.duration_ms >= 0
        assert stage.success is True and stage.error_type is None


def test_failed_stage_records_and_propagates() -> None:
    with traced_request("t2") as trace:
        with pytest.raises(ValueError, match="boom"):
            with traced_stage("risky"):
                raise ValueError("boom")
        (stage,) = trace.stages
        assert stage.success is False
        assert stage.error_type == "ValueError"


def test_nested_stages() -> None:
    with traced_request("t3") as trace:
        with traced_stage("outer"):
            with traced_stage("inner"):
                pass
        assert [s.stage for s in trace.stages] == ["inner", "outer"]
        assert all(s.success for s in trace.stages)


def test_no_trace_no_crash_no_record() -> None:
    assert get_current_trace() is None
    with traced_stage("ghost"):
        pass
    count("anything", 5)  # must not raise outside a trace


# --- trace ------------------------------------------------------------------------


def test_trace_finish_counters_serialization() -> None:
    with traced_request("t4") as trace:
        count("hits", 2)
        count("hits")
        trace.set_model("llm", "qwen3:4b")
        trace.note("agent_termination", "finished")
        trace.record_llm_call(model="qwen3:4b", provider="OllamaLLMProvider", duration_ms=10)
        trace.finish("error", error_type="RuntimeError")
        assert trace.counters["hits"] == 3
        assert trace.models == {"llm": "qwen3:4b", "llm_provider": "OllamaLLMProvider"}
        assert trace.attributes["agent_termination"] == "finished"
        assert trace.status == "error" and trace.error_type == "RuntimeError"
        assert trace.input_tokens is None and trace.estimated_cost is None
        payload = json.dumps(trace.to_dict())  # serializable
        assert '"request_id": "t4"' in payload


def test_llm_usage_accumulates_only_when_reported() -> None:
    with traced_request("t5") as trace:
        trace.record_llm_call(model="m", provider="p", duration_ms=1)
        assert trace.input_tokens is None and trace.total_tokens is None
        trace.record_llm_call(
            model="m", provider="p", duration_ms=1, input_tokens=10, output_tokens=5
        )
        assert (trace.input_tokens, trace.output_tokens, trace.total_tokens) == (10, 5, 15)


# --- isolation ----------------------------------------------------------------------


def test_sequential_contexts_do_not_leak() -> None:
    with traced_request("a") as first:
        first.increment("x")
    with traced_request("b") as second:
        assert second.request_id == "b"
        assert second.counters == {}
    assert get_current_trace() is None


def test_concurrent_tasks_stay_isolated() -> None:
    async def worker(name: str, counter: str) -> tuple[str, dict]:
        with traced_request(name) as trace:
            await asyncio.sleep(0)
            trace.increment(counter)
            await asyncio.sleep(0)
            return trace.request_id, dict(trace.counters)

    async def main() -> list:
        return await asyncio.gather(worker("req-a", "alpha"), worker("req-b", "beta"))

    (id_a, counts_a), (id_b, counts_b) = asyncio.run(main())
    assert (id_a, counts_a) == ("req-a", {"alpha": 1})
    assert (id_b, counts_b) == ("req-b", {"beta": 1})
    assert get_current_trace() is None


# --- security / privacy ---------------------------------------------------------------


def test_formatter_excludes_sensitive_keys() -> None:
    from deepresearch.logging import JsonFormatter

    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", None, None)
    record.api_key = "sk-secret"  # type: ignore[attr-defined]
    record.authorization = "Bearer x"  # type: ignore[attr-defined]
    record.prompt = "full prompt text"  # type: ignore[attr-defined]
    record.document = "full document"  # type: ignore[attr-defined]
    record.request_id = "r1"  # type: ignore[attr-defined]
    output = json.loads(JsonFormatter().format(record))
    assert output["request_id"] == "r1"
    for leaked in ("sk-secret", "Bearer x", "full prompt text", "full document"):
        assert leaked not in json.dumps(output)
    assert "api_key" not in output and "prompt" not in output


def test_trace_carries_no_content_or_reasoning() -> None:
    trace = RequestTrace(request_id="r2")
    payload = json.dumps(trace.to_dict())
    for absent in ("prompt", "document", "reasoning", "chain_of_thought", "secret"):
        assert absent not in payload


def test_agent_step_shape_unchanged() -> None:
    from deepresearch.agent import AgentStep

    step = AgentStep(1, "search_documents", {}, "summary", 5, True)
    assert set(step.__dict__) <= {
        "step_number",
        "tool_name",
        "tool_input",
        "output_summary",
        "latency_ms",
        "success",
        "duplicate",
        "error",
    }
