# ADR-018: Frontend research experience

Date: 2026-09-20 | Status: Accepted | Milestone: M18

## Context

M1–M17 built a complete local-first RAG pipeline (ingestion,
embeddings, hybrid retrieval, reranking, grounded generation,
citations, verification, statuses, agent, observability, evaluation,
security) with no user-facing interface: the only HTTP surface was
`GET /health` and `GET /ready`. M18 must deliver the first proper
interface — ask → investigate → evidence-backed answer — without
duplicating pipeline logic, exposing private cognition, or weakening
M17 security boundaries.

## Decision

- **Next.js App Router + React + TypeScript in `frontend/`**
  (`app/`, `components/`, `lib/`, `types/`, `tests/`), plain CSS, no
  UI framework. Smallest dependency set that satisfies the brief:
  `next`/`react`/`react-dom` + `typescript`, with `vitest` +
  Testing Library + `jsdom` for tests only. No state library, no
  markdown renderer (a renderer would need a sanitizer to keep
  hostile evidence inert — plain-text rendering needs none).
- **Minimal backend boundary, not a second backend**
  (`src/deepresearch/research_api.py`): `POST /api/research` → 202
  job snapshot, `GET /api/research/{id}` for polling. A
  `ResearchService` holds in-memory jobs and runs the *unchanged*
  `answer_question(..., verify_citations=True)` in a background
  thread per request; `researcher(session, question, request_id)` is
  the single injection point so tests use deterministic fakes. The
  M14 agent is deliberately *not* wired in: the default research
  path is the evaluated M10 pipeline; agent-backed research stays a
  future decision.
- **Explicit safe response models** (`ErrorInfo`, `StageInfo`,
  `CitationInfo`, `EvidenceInfo`, `ConflictInfo`,
  `VerificationClaimInfo`, `VerificationSummaryInfo`,
  `ResearchDetailsInfo`, `ResearchResult`, `ResearchJobSnapshot`):
  identifiers, counts, durations, statuses, and source text only.
  Prompts, chain-of-thought, embeddings, DB internals, secrets, and
  stack traces have no field to travel in — enforced by tests
  asserting exact key sets and scanning serialized bodies.
- **Honest progress from real stages.** Running jobs expose live
  `RequestTrace.stages` (the exact `traced_stage` names); the UI maps
  them to friendly labels (`lib/stages.ts`), shows durations, and
  passes unknown future stages through as-is. No percentages, no
  invented steps, no streaming/WebSockets (polling at 1 s is
  sufficient for minute-scale local research).
- **Server-owned request IDs.** `X-Request-ID` preserved-or-minted by
  the backend (M15); the UI stores the returned ID for polling and
  display and never generates its own. CORS allowlist
  (`CORS_ORIGINS`, credentials off) is the only browser-specific
  backend config.
- **Citation UX from backend authority.** The answer is split on
  `[N]` markers client-side for rendering only: markers present in
  `citations[]` become buttons that scroll-to/highlight the matching
  evidence; unknown/invalid markers stay plain text. Duplicates map
  to the same evidence; absent citations get an explicit notice.
- **Evidence rendered as data.** Chunk text is a string value
  displayed in a labeled "Source material" blockquote via default
  React escaping. No `dangerouslySetInnerHTML` anywhere; tests assert
  `<script>alert("x")</script>` plus instruction-like content produce
  no elements and no behavior.
- **Status wording from the M13 taxonomy**, no second taxonomy:
  answered / insufficient_evidence / conflicting_evidence /
  no_evidence, each with a factual banner; conflicts listed with both
  sides preserved. "Research details" (request ID, elapsed,
  candidates, evidence/citation/invalid counts, verification counts,
  per-role models) replaces any "reasoning" view.
- **Errors are user-safe on both sides.** 422 (validation), 500
  envelope, fetch failure, timeout, failed job, and unknown job each
  map to a typed `ResearchApiError` with message + request ID and an
  accessible `role="alert"` banner. Server detail stays in logs via
  `logger.exception`.

## Alternatives considered

- **Server-Sent Events / WebSockets for progress**: rejected — the
  backend has no streaming primitive, and polling snapshots already
  carry live stage transitions. Adds a protocol for no new
  information.
- **Markdown rendering for answers**: rejected — requires a
  sanitizer dependency to keep hostile evidence inert; the backend
  returns plain text, so plain-text rendering is faithful and safe.
- **Wiring the M14 agent as the default researcher**: rejected for
  now — the agent path is bounded and tested but unevaluated as an
  answering strategy; the M10 pipeline is the measured default.
- **Persisting jobs in Postgres**: rejected — in-memory jobs match
  the demo scope; a restart drops history explicitly rather than
  pretending durability.

## Consequences

- The UI demonstrates exactly what the backend can do — no more, no
  less. New pipeline capabilities appear automatically when the safe
  models extend (unknown stage names already pass through).
- `frontend/` builds and tests independently (`npm test`,
  `npm run typecheck`); frontend tests use mocked fetch — no Ollama,
  no Postgres. Backend tests cover the boundary with fakes plus
  real-pgvector integration.
- Explicitly out of scope (unchanged): auth, billing, web search,
  conversation history, upload UI, dashboards, deployment config,
  analytics, new agent tools.
