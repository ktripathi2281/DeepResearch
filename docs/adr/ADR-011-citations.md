# ADR-011: Source citations (association, not verification)

Date: 2026-09-20 | Status: Accepted | Milestone: M11

## Context

M10 answers were grounded but unreferenceable: nothing tied answer
spans to the evidence that produced them. M11 must establish the
citation contract — numbering, mapping, extraction — while leaving
correctness judgments to M12.

## Decision

- **Numbering: `[N]` from evidence order (1-based).** Stable,
  deterministic, human-readable; database UUIDs never face the user.
  The number is positional, so reordered evidence renumbers —
  citation identity is per-answer, not global.
- **`Citation` is a thin provenance view** (number, chunk/document
  IDs, title, source, type, page, section, retrieval score/method/
  rank) derived from `RetrievalResult`. No second identity system.
- **Prompt carries the markers**: each evidence block shows
  `Citation [N]`, and the system part orders the model to cite only
  shown markers, only for supported claims, never inventing them.
- **Extraction is strict stdlib regex on `[N]` digits only.**
  `[Source A]` and `[-1]` are ignored (not citations, not errors);
  `[0]` and out-of-range numbers become `InvalidCitationReference`
  records — retained for M12, never remapped, never deleted, while
  the answer text itself is preserved verbatim.
- **First-use order, deduplicated.** The citation list mirrors how
  the answer references evidence (`[3],[1],[3]` → `[3],[1]`),
  preserving the text↔evidence relationship M12 will judge.
- **Uncited evidence stays in `GroundedAnswer.evidence`** so M12 can
  measure completeness; `citations` holds only referenced items.
  No-evidence answers keep `citations == []` with no LLM call.
- **Association ≠ correctness, stated plainly.** A present marker
  proves the model pointed at evidence, not that the evidence
  supports the claim. M12 judges support; nothing here scores it.

## Alternatives considered

- UUID/database IDs as markers: precise but unreadable and prompt-
  heavy; rejected for the user-facing format.
- Dropping invalid markers at extraction: rejected — hides model
  misbehavior M12 must see.
- Rewriting the answer to fix citations: rejected — fabricates
  model output; the record must show what was generated.

## Consequences

- M12 consumes `citations` + `invalid` + full `evidence` to judge
  support per claim. M16 can report citation accuracy from the same
  fields. Any future citation rendering (UI/API) reads `Citation`
  without touching retrieval.
