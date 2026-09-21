# ADR-019: Provider abstraction and optional cloud providers

Date: 2026-09-21 | Status: Accepted | Milestone: M19

## Context

M9 established the `LLMProvider` Protocol with a single local
implementation (`OllamaLLMProvider`, `qwen3:4b`) and a layering guard
forbidding local-provider details outside `llm.py`/`config.py`.
M10–M18 built the whole pipeline (generation, agent, verifier,
evaluation, research API) against that abstraction. M19 must open the
abstraction to alternative providers without changing the pipeline,
without weakening the local-first default, and without leaking
credentials.

## Decision

- **Strengthened contract, same shape.** `LLMProvider` keeps
  `generate() -> str` untouched and gains `generate_response() ->
  LLMResponse` (text, model, provider, optional token counts) with a
  protocol-level default that delegates to `generate()`. Token counts
  come only from numbers providers actually return
  (`prompt_eval_count`/`eval_count`, `usage`, `usageMetadata`);
  `None` means unavailable — never zero, never estimated.
- **Local default preserved.** `OllamaLLMProvider`, `qwen3:4b`,
  `http://localhost:11434`, bounded timeouts, and the M9 error
  taxonomy are unchanged. `LLM_PROVIDER` unset or `ollama` selects
  it; no cloud credential is ever needed for the default path.
- **Two optional adapters, no new dependencies.** A generic
  OpenAI-compatible Chat Completions adapter and a small Gemini
  `generateContent` REST adapter live in
  `src/deepresearch/providers.py`, using the already-required
  `httpx` — no vendor SDKs (dependency discipline). Both use lazy
  clients: import and construction perform no I/O; a missing key
  raises `LLMConfigurationError` only at selection or call time.
- **One factory.** `create_llm_provider(settings)` is the sole
  selection point (`ollama` / `openai_compatible` / `gemini`).
  Unknown names fail listing valid values. Generation, agent,
  verifier, eval runner, and the research API take an `LLMProvider`
  and contain no provider branching (verified by name-agnostic fake
  tests).
- **Error translation at the boundary.** Adapter failures map to the
  M9 taxonomy (`LLMConnectionError`, `LLMTimeoutError`,
  `LLMResponseError`, `LLMModelNotFoundError`); selection and
  credential problems use the new `LLMConfigurationError`.
- **Secret-free diagnostics.** Messages and logs carry statuses,
  models, and timeouts — never keys, auth headers, or
  credential-bearing URLs. Gemini uses the `x-goog-api-key` header
  instead of `?key=` precisely because HTTP clients log request
  URLs; a security test caught the query-param variant leaking
  through httpx's own request logs, and the header design fixed it.
- **Observability and evaluation.** `record_llm_call` additionally
  stores reported input/output tokens; role keys (`llm`/`verifier`/
  `agent`) are untouched. `ExperimentConfig` records `llm_provider`
  alongside `llm_model`, so cross-provider runs always differ in
  identity. No metric changed.
- **Guard refined, not weakened.** The M9 test now forbids
  local-provider *API details* (concrete class, port, API path, CLI
  hints, direct settings-attribute access) outside
  `llm.py`/`config.py`; the bare provider-name token is a legitimate
  cross-cutting concern (selection values, identity defaults).

## Alternatives considered

- **Gemini vendor SDK**: rejected — a ~30-line REST mapping over the
  existing HTTP client does the job; an SDK adds weight and a second
  configuration surface for no pipeline benefit.
- **Server-side key in URL query (`?key=`)**: rejected after a test
  proved httpx request logs echo the full URL. Header transport
  keeps keys out of every log path by construction.
- **Automatic fallback / routing / retries**: explicitly deferred —
  silent substitution between providers would violate the
  no-silent-substitution constraint; failures surface immediately as
  today.
- **Auto-detecting provider identity in eval configs**: rejected —
  explicit `llm_provider`/`llm_model` fields keep experiment
  identity reproducible from the config file alone.

## Consequences

- The pipeline is provider-agnostic in fact, not just in name:
  generation, agent, and verifier each pass with a
  foreign-named fake, and traces/eval identities distinguish
  providers.
- Cloud adapters are usable today via environment config but inert
  by default: no keys, no clients, no network unless explicitly
  selected — proven by regression tests.
- Explicitly out of scope (unchanged): model routing, fallback,
  cost optimization, rate limiting, caching, billing, auth, web
  search, MCP, streaming, frontend changes.
