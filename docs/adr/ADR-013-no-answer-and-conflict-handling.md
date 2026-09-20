# ADR-013: No-answer and conflict handling

Date: 2026-09-20 | Status: Accepted | Milestone: M13

## Context

M12 answers carried citations and verdicts but no machine-readable
outcome: callers could not tell abstention from grounding, or a
settled answer from a disputed one. M13 must make
no-evidence / insufficient / conflicting / answered explicit and
deterministic at the application level — without building M14's
agent or M16's evaluation.

## Decision

- **Statuses**: `no_evidence`, `insufficient_evidence`, `answered`,
  `conflicting_evidence` (`AnswerStatus`, distinct from M12
  verdicts: verdicts judge claim↔evidence pairs, status judges the
  answer). Precedence is fixed:
  no_evidence → conflicting_evidence → verification-based
  insufficient_evidence → answered. An invalid secondary marker
  alone never forces insufficient — the §11 example (answered with
  one invalid citation) is a tested rule.
- **No-evidence is structural**: empty evidence short-circuits
  before any LLM/verifier call (tested zero-call guarantee), with
  the canonical fixed message, empty citations/report, and an empty
  verification report object.
- **Insufficient has two legs**: the M10 prompt already orders the
  model to declare insufficiency, and verification (when enabled)
  converts `unsupported`/`unverifiable` cited rows into the status
  deterministically. Without verification there is no deterministic
  signal, so the status stays `answered` — the pipeline grounded
  the answer as instructed rather than guessing from LLM wording.
- **Conflicts are deterministic numeric contradictions**: two items
  conflict when number mentions (digits plus zero–twenty words)
  share ≥2 non-stopword context words within ±4 tokens but differ
  in value ("introduced in 2022" vs "2024"; "5 stages" vs "eight
  stages"). Pairwise over evidence order, one record per pair
  (`conflict-N`, type `numeric_mismatch`, both chunk/document IDs,
  values, shared context in the description). Detector exceptions
  propagate — a crash never becomes a silent "no conflict".
- **No LLM-assisted detector.** The deterministic scope covers the
  required fixtures; a second model call would add failure modes
  for cases evaluation has not shown to need it. The prompt still
  tells the model to surface any disagreement it notices, as a
  backstop outside the status path.
- **Conflict-aware prompt section** appended to the trusted system
  part only when conflicts exist, naming each disagreement and
  forbidding silent resolution; both sources stay in evidence and
  in the answer. Status is `conflicting_evidence` even when
  individual citations verify as supported — supported parts do
  not settle a disputed whole.
- **Known limits, stated plainly**: digits/zero–twenty only;
  negations ("safe"/"unsafe"), paraphrased quantities, unit
  mismatches, and coincidental shared contexts are out of scope
  (the last can over-flag). Broad factual reasoning waits for
  evaluation data (M16), not for a bigger heuristic here.

## Alternatives considered

- LLM conflict judge now: rejected — nondeterministic status
  path plus new failure modes, for scope the fixtures do not
  require.
- Threshold/config surface (evidence-count cutoffs): rejected —
  invented cutoffs would pretend to measure grounding; the
  verification report already carries the real signal.
- Silent conflict resolution (newest/highest-score wins):
  rejected — the portfolio point is preserving disagreement.

## Consequences

- Callers branch on `status`; UI/API can render conflicts from
  `GroundedAnswer.conflicts` with full provenance. M14's agent
  can treat `conflicting_evidence`/`insufficient_evidence` as
  search-more-or-abstain signals. M16 measures detector
  precision/recall on labeled conflict fixtures.
