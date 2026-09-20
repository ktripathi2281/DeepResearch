# ADR-012: Citation verification (LLM-assisted, claim vs evidence)

Date: 2026-09-20 | Status: Accepted | Milestone: M12

## Context

M11 markers prove the model *pointed* at evidence, not that the
evidence *supports* the claim. M12 must judge the claim↔evidence
relationship with the local stack (qwen3:4b, no structured-output
guarantee), keeping association (M11) and judgment (here) separate.

## Decision

- **Claims are deterministic sentence units** (terminator/newline
  split, markers attached). Documented limitation: segmentation is
  heuristic, not factual-claim detection — the verifier judges each
  unit as written.
- **One verifier call per cited claim with all its cited evidence.**
  Per-claim×citation result rows share the claim's verdict so the
  relationship stays explicit without multiplying model calls.
- **Structured output at the application layer**: the model returns
  JSON `{verdict, explanation}` against a Pydantic schema
  (`supported`/`unsupported`/`insufficient_evidence` + non-empty
  explanation, extras ignored). Parse → validate → one bounded
  repair (previous output echoed back with the schema) → else
  `unverifiable`. Never indefinite, never another model, never a
  silent "supported".
- **Statuses**: `supported`, `unsupported`, `insufficient_evidence`
  (model verdicts); `invalid_citation` (M11 out-of-range markers,
  reported never sent); `uncited` (tracked, never verified —
  uncited ≠ unsupported); `unverifiable` (output unusable after
  retry). The last two extend the M12 brief because silent gaps
  would be worse; unexpected LLM errors still propagate as
  `LLMError` instead of becoming verdicts.
- **Verifier prompt isolates trusted instructions** from untrusted
  claim+evidence (injection text stays data; contradictory evidence
  must yield `unsupported`, not a pick). Explanation is a concise
  grounding justification, never chain-of-thought.
- **Integration is additive and off by default**:
  `GroundedAnswer.verification_report` (None unless requested) and
  `answer_question(verify_citations=False, verifier=None)` — the
  verifier defaults to the generator LLM. No-evidence answers carry
  an empty report with zero calls.
- **Report exposes counts** (per status, cited-claim total) for M16;
  no accuracy score is computed here.

## Alternatives considered

- Entailment classifiers (e.g. NLI models): deterministic and
  cheap, but another model to qualify and weaker on nuanced
  support; the local LLM reuses the qualified stack.
- Verifying every citation independently: multiplies calls for
  multi-citation claims with no judgment gain — the claim is the
  natural unit.
- Raising on malformed output: rejected — one bad JSON block
  should not void a whole report; `unverifiable` records it.

## Consequences

- Verification quality is itself unevaluated: M16 must measure the
  verifier against labeled claim/evidence pairs rather than
  assuming it. M13 builds abstention/conflict handling on these
  statuses; the UI/API can render per-claim results from the report.
