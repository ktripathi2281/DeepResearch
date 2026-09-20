# ADR-017: Security and adversarial testing

Date: 2026-09-20 | Status: Accepted | Milestone: M17

## Context

M1–M16 built the pipeline with per-milestone safety properties
(untrusted-evidence prompts, tool allowlist, bounded retries,
content-free logs) but never attacked them systematically. M17 must
verify the trust boundaries actually hold, harden genuinely
unbounded inputs, and document the threat model — without adding
execution capabilities, cloud services, or new dependencies.

## Decision

- **Threat model written down first** (`docs/SECURITY.md`):
  trusted code/config/schema/providers vs untrusted documents,
  questions, model outputs, and metadata, with three explicit
  boundaries (evidence→LLM, LLM→tools, application→logs).
- **Retrieved documents stay untrusted by structure, not by model
  obedience**: delimited evidence blocks, instruction text never
  promoted, proven by an 8-attack adversarial corpus
  (`tests/fixtures/adversarial.txt`) asserting attacks remain data
  through ingestion, prompts, traces, and answers.
- **Agent allowlist enforced twice** (Pydantic Literal at parse,
  explicit guard at execution) plus a source tripwire test
  scanning `agent.py` for execution/IO/HTTP primitives with
  dot-guarded patterns (so `re.compile` prose doesn't false-positive).
- **Argument validation hardened, not weakened**: unknown argument
  keys now rejected (fail closed), UUID/type/range checks kept;
  oversized `top_k`, wrong types, nulls, and 5000-char IDs all
  rejected without crashing.
- **Only two genuinely unbounded inputs got new guards**:
  question length (4000 chars, both entries) and document bytes
  (10 MB, parse entry) — each the smallest explicit documented
  limit for its real risk (prompt/context blowup, memory
  exhaustion). Everything else already had bounds (top-K caps,
  agent caps, timeouts, single-repair, 500-char reason cap), now
  covered by boundary tests. LLM response length stays
  caller-bounded via existing `max_tokens`.
- **NUL bytes stripped at parse** (TXT/MD/HTML decode, PDF
  text/title): PostgreSQL rejects `\x00` outright, so a hostile
  or corrupt file now parses cleanly instead of crashing a query.
- **Metadata tested, not sanitized**: caller-owned JSON round-trips
  hostile-but-valid values (huge/unicode/nested/secret-looking)
  as inert data; invalid values fail at the database loudly.
- **Observability proven leak-free under attack**: poisoned-pipeline
  runs with planted fake secrets assert absence from logs and
  trace payloads; the formatter allowlist is the mechanism.
- **Errors fail closed**: 503 shape without tracebacks, agent
  failures explicit with intact traces, no silent privilege
  expansion or infinite retries anywhere.

## Alternatives considered

- Broad input sanitization/normalization engine (Unicode NFKC,
  confusable resolution, HTML-escaping everything): rejected —
  crash-safety plus explicit boundaries match the actual threat;
  a normalization engine would risk corrupting legitimate
  multilingual content for no measured benefit.
- Rate limiting / auth: rejected — no network service boundary
  exists in this local-first milestone; documented as production
  work.
- Third-party scanners / fuzz harnesses: rejected — deterministic
  regression tests fit the suite; tooling without a target
  deployment is ceremony.

## Consequences

- Security properties are tested, not asserted — but the suite is
  small and synthetic (stated in SECURITY.md); absence of
  vulnerabilities is explicitly not claimed.
- M18+ inherits: fail-closed validation, content-free logging,
  and the adversarial corpus as regression fixtures. Any new tool
  or endpoint must extend the allowlist tests and the threat
  model, not bypass them.
