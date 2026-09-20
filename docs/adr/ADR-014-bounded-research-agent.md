# ADR-014: Bounded research agent

Date: 2026-09-20 | Status: Accepted | Milestone: M14

## Context

Single-shot hybrid retrieval always spends the same effort
regardless of the question. M14 needs an agent that can search,
then inspect promising chunks/documents before answering — while
staying a constrained research navigator, not an autonomous
general agent. No frameworks (the project owns its loop), no web
search in v1.

## Decision

- **Exactly three read-only tools**: `search_documents` (existing
  hybrid path, chunk ID/title/section/score + concise text),
  `get_chunk` (full chunk view, structured not-found),
  `get_document` (metadata + bounded chunk pointers guiding deeper
  inspection). Unknown tool names fail Pydantic Literal validation;
  execution has a defense-in-depth allowlist. Never: code, SQL,
  shell, filesystem, HTTP, writes.
- **Decisions are `AgentDecision{action, arguments, reason}`**
  (Pydantic, extras forbidden, reason ≤500 chars of operational
  justification — not chain-of-thought). Parse → per-action
  argument validation → one bounded repair → `invalid_decision`
  termination. Malformed output never becomes an arbitrary action.
- **Hard caps guarantee termination**: 8 iterations / 12 tool
  calls / 60 s wall-clock (`Settings.agent_*`, validated, never
  raisable by the model). Six explicit terminations: finished,
  max_iterations, max_tool_calls, timeout, invalid_decision,
  tool_failure. Not-found is recoverable (failed step, loop
  continues); unexpected errors terminate with the trace intact.
- **Memory is bounded**: prompts carry the question plus truncated
  step summaries and an evidence count — never the full transcript.
  Duplicate calls are flagged in the trace and counted (caps, not
  caches, prevent infinite loops).
- **Evidence is deduplicated by chunk ID** with provenance from
  real rows; the agent returns references, and the existing M10
  pipeline answers. No combined agent→answer helper — the
  boundary keeps generation contracts intact.
- **Timeout is checked before every iteration and every tool**;
  nothing runs after it trips. Model is lazy-loaded nowhere here:
  the agent takes an `LLMProvider`, so tests script decisions and
  the live path uses qwen3:4b unchanged.
- **Injection boundary is structural**: tool/document text enters
  prompts as observation lines only; tests prove attack text never
  becomes a tool call. Broader adversarial testing belongs to the
  security milestone — no solved-claim is made.

## Alternatives considered

- LangChain/LangGraph/AutoGen/CrewAI: rejected — the portfolio
  point is implementing the loop directly; frameworks hide the
  termination and validation logic under test here.
- Skipping re-execution of duplicate calls (cache): rejected —
  executing + flagging is simpler and the caps already bound cost.
- Web search tool now: rejected — v1 answers from the controlled
  corpus only, per the frozen scope.

## Consequences

- M10 `answer_question` is unchanged; callers compose
  `run_research_agent` → rerank/generate manually (proven in PG
  tests). M15 observes the trace (counts, latencies, termination);
  M16 can compare single-shot vs agentic evidence. Live-Ollama
  behavior is smoke-tested, never asserted as reliable.
