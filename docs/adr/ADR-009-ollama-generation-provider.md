# ADR-009: Ollama generation provider (qwen3:4b)

Date: 2026-09-20 | Status: Accepted | Milestone: M9

## Context

M9 needs a generation capability for future grounded answers (M10+):
a reusable local provider behind an abstraction, honoring the
hardware/model constraints (LOQ 16 GB/6 GB, `qwen3:4b Q4_K_M`).
No retrieval wiring, no structured output, no agents in this step.

## Decision

- **Runtime: local Ollama HTTP API** (`/api/generate`, non-streaming
  JSON). Ollama owns model serving/quantization/GPU handling; the
  project owns prompting and orchestration. Explicitly not
  LangChain/LlamaIndex — the project intentionally owns its loop.
- **Model: `qwen3:4b` (Q4_K_M via Ollama)**, configurable
  (`Settings.ollama_model`). The 4B quantized footprint fits alongside
  embeddings/reranking on 6 GB VRAM; 14B+ variants are rejected by the
  frozen constraints. Never silently substituted: a missing model
  raises `LLMModelNotFoundError` naming `ollama pull <model>`.
- **Abstraction: `LLMProvider` Protocol** (`model_name`,
  `model_version`, `generate(prompt, *, system_prompt, temperature,
  max_tokens) -> str`) in `src/deepresearch/llm.py`. Application code
  depends on the Protocol; Ollama's schema lives only in
  `OllamaLLMProvider`. A future cloud provider implements the same
  Protocol with no domain changes.
- **HTTP: `httpx`** (moved to runtime deps; already used by tests) —
  smallest sufficient client, no Ollama SDK. Transport injectable
  (`httpx.MockTransport` in tests), one reusable client per provider.
- **Timeouts mandatory** (`ollama_timeout_seconds=120`); no retries
  yet — failures surface immediately rather than hanging. Errors are
  typed: `LLMConnectionError` (names the base URL + asks if Ollama
  runs), `LLMTimeoutError` (names the seconds), `LLMResponseError`
  (status/body snippet, empty output, malformed JSON), all under
  `LLMError`.
- **Semantics: temperature default 0.0, `max_tokens=None` means
  Ollama's default** (`num_predict` sent only when set), blank
  prompts and empty responses rejected, no JSON-schema mode yet.
- **Version: configured name always; runtime `"model"` id captured
  when a call succeeds, else `None`** — never invented.
- **No chain-of-thought**: answer text only, no reasoning fields;
  logs record lengths/latency, never prompt content.
- **No new endpoint** for the Ollama check in M9 scope: verification
  is a README snippet plus the live smoke test, keeping transport
  code untouched.

## Alternatives considered

- Ollama Python SDK: thin wrapper over the same HTTP calls — direct
  `httpx` keeps the schema visible and dependency-free.
- llama.cpp server directly: viable, but Ollama is the documented
  project runtime with simpler model management (`pull`/`list`).
- Retries with backoff (OPENCODE M9 sketch): deferred — premature
  without production traffic patterns; explicit errors first.

## Consequences

- Local setup is `ollama pull qwen3:4b` + running daemon (~2.5 GB
  download); GPU use depends on Ollama/environment, not promised.
- M10 builds grounded generation on `LLMProvider`; structured
  output/validation arrives with its own milestone, not here.
- No claim is made that `qwen3:4b` is the best model — it is the
  constrained baseline; `gemma3:4b` comparison comes later.
