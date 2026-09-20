# ADR-015: Local-first observability and request tracing

Date: 2026-09-20 | Status: Accepted | Milestone: M15

## Context

M1–M14 proved behavior but never measured it in one place: stage
latencies, candidate counts, model identity, and termination reasons
were scattered across per-module logs or absent. M15 must make every
research request traceable without external platforms, dashboards, or
persistence — and without touching retrieval/generation semantics.

## Decision

- **One module, stdlib only** (`observability.py`): `RequestTrace`
  (id, status, stages, counters, models, attributes, optional
  token/cost fields), `StageTiming`, and helpers. No OpenTelemetry,
  Prometheus, or SaaS — a local project needs inspectable data
  structures, not infrastructure.
- **ContextVar request scope**: `traced_request()` establishes an
  isolated trace; `get_current_trace()` exposes it; the FastAPI
  middleware preserves a usable `X-Request-ID` or mints a UUID and
  finishes the trace from the response status. No globals, so
  concurrent requests cannot overwrite each other (tested with
  interleaved asyncio tasks).
- **Passive instrumentation**: `traced_stage()` / `count()` no-op
  without an active trace, so all 200+ pre-existing tests run
  unchanged. Stages use `monotonic_ns` (never wall-clock);
  failures record `success=False` + error type and re-raise —
  instrumentation never swallows pipeline errors and trivially
  cannot fail the request (pure in-memory appends).
- **Coverage**: embedding, vector/BM25 retrieval, hybrid fusion,
  reranking, conflict detection, citation extraction, generation,
  citation verification, agent execution plus per-tool stages.
  Counts use canonical keys (candidates per stage, evidence,
  llm_calls by role — `llm`/`verifier`/`agent` kept distinct,
  citations, verification outcomes, agent iterations/tools/dups).
- **Models recorded, tokens honest**: model/provider names per
  role; token and cost fields exist but stay `None` because local
  Ollama/sentence-transformers report no usage — never fabricated,
  never a reason to add a paid API.
- **Privacy by construction**: the JSON formatter emits only an
  explicit key allowlist (request_id, stage, duration, path,
  method, status, event, status, error_type); prompts, documents,
  reasoning, secrets, and headers have no key and cannot leak.
  Traces hold identifiers and counts only. Health/ready stay
  access-log-quiet (header still set).
- **Agent compatibility**: M14 `AgentTrace` is untouched; the
  request trace adds iterations, tool calls, per-tool timings,
  duplicate-evidence counts, and a termination note. The existing
  `reason` field remains the sole operational text.

## Alternatives considered

- OpenTelemetry/Prometheus/Grafana: real power, real ops burden —
  unjustified before any deployment target exists.
- Database-backed trace store: rejected for M15 (explicitly out
  of scope); `to_dict()` serialization keeps the door open.
- Per-module trace objects threaded as parameters: rejected —
  invasive across 15 milestones of signatures; the ContextVar
  keeps instrumentation additive.

## Consequences

- M16 reads counters/reports instead of adding measurement code;
  any endpoint can return `trace.to_dict()` later without new
  plumbing. Limits: in-memory only (a restart loses traces),
  single-process scope, no sampling or aggregation — appropriate
  until traffic says otherwise.
