# ADR-010: Evidence-grounded generation

Date: 2026-09-20 | Status: Accepted | Milestone: M10

## Context

M9 proved the LLM provider; M5–M8 proved retrieval. M10 must connect
them into answers that stay inside the evidence, with a first
injection boundary and an honest no-evidence path. Citations (M11),
verification (M12), and abstention/conflict frameworks (M13) come
later and must not be half-built here.

## Decision

- **Orchestration in `generation.py`: hybrid → rerank → prompt →
  `LLMProvider.generate`.** Retrieval/reranking run unmodified;
  generation depends on the provider Protocol (any implementation
  works); Postgres is touched only through existing retrieval APIs.
  K defaults are imported from the sibling modules, never redefined.
- **Types: `ResearchQuestion` (text + optional `request_id` for M15),
  `Evidence` as an alias of `RetrievalResult`** (provenance is
  already complete — duplicating it would rot), **`GroundedAnswer`**
  (answer, evidence, model name/version, `has_evidence`). No citation
  fields by design; adding them now would pre-empt M11's mapping
  contract.
- **Prompt split into trusted `system` + data `user` parts** (Ollama
  `system` support). The system part orders: evidence-only, no
  outside knowledge, no fabrication, untrusted-evidence rule,
  explicit-insufficiency rule, conflict acknowledgement, plain text
  without hidden reasoning. Evidence blocks are numbered with
  source/origin/section/page headers in reranked order — delimiters
  make data-vs-instruction structure explicit to the model.
- **No-evidence short-circuit:** empty evidence returns
  `GroundedAnswer(has_evidence=False)` with a fixed message and zero
  LLM calls — the model is never asked to answer from nothing.
  Insufficiency *within* evidence is a prompt instruction (the model
  says so); scoring confidence is explicitly out of scope.
- **Conflicts are prompt-level only:** the model is told to surface
  disagreement, not resolve it. No resolution algorithm (M13).
- **Injection boundary = structure, not a claim of safety:**
  malicious text stays inside evidence blocks and out of the system
  part (tested); dedicated adversarial evaluation belongs to the
  security milestone.
- **Plain-text output only.** Empty provider output already raises
  in M9, so M10 adds no JSON/retry/repair — structured generation
  arrives with citations (M11+).
- **Evidence budget:** `max_evidence_chars=None` keeps everything;
  when set, whole trailing blocks drop (never mid-block, never
  silent). Deterministic given the same ranked evidence.
- **Observability prep:** counts, model identity, and total latency
  are logged (prompt text never); `request_id` threads through for
  M15 correlation.

## Alternatives considered

- Merging citations now (OPENCODE M10 sketch lists answer +
  citations + warnings): rejected — citation IDs need M11's
  validation contract; emitting unvalidated `[1]` markers would
  teach the model to fabricate references.
- Confidence scores: rejected — uncalibrated numbers dressed as
  measurement; the honest signal is `has_evidence` plus the text.
- Automatic no-evidence retries with broader K: rejected — silent
  behavior change; callers choose their K explicitly.

## Consequences

- M11 maps answer spans to `Evidence` chunk IDs; M12 verifies them;
  M13 hardens abstention/conflict; M15 records the logged fields.
- No factual-accuracy or production-readiness claim is made here —
  the pipeline is grounded by construction, not verified by
  measurement (that is M16+).
