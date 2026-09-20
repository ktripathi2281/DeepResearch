# DeepResearch — Security

Defensive posture for a local-first research system. Nothing here
claims absolute security: these are layered, tested boundaries plus
explicitly documented limitations. See `docs/adr/ADR-017-security-adversarial-testing.md`
for decision rationale.

## Threat model

### Trusted

- Application code in this repository.
- Deployment-controlled configuration (environment, `.env`).
- Database schema and migrations.
- Explicitly configured local model providers (Ollama daemon,
  HuggingFace model cache as configured by the operator).

### Untrusted

- Uploaded documents and all document text/chunks.
- Document metadata originating from ingestion.
- User questions.
- LLM-generated tool arguments, citations, and decisions.
- All model outputs (answers, verdicts, rerank scores).

### Trust boundaries

```text
User
  ↓
API
  ↓
Research pipeline
  ↓
Retrieval
  ↓
UNTRUSTED evidence
  ↓
LLM
```

```text
LLM
  ↓
validated agent decision
  ↓
explicit tool allowlist (3 read-only tools)
  ↓
tool execution
```

```text
Application
  ↓
observability
  ↓
structured logs (identifiers and counts only)
```

Retrieved text must never become executable instructions. Model
output must never become trusted configuration.

## Policies

- **Retrieved content is data.** Prompts delimit trusted
  instructions from evidence blocks; injection strings stay inside
  evidence (tested with an 8-attack adversarial corpus).
- **Agent tools are allowlisted**: `search_documents`, `get_chunk`,
  `get_document` — read-only, validated arguments, unknown names
  rejected at parse and at execution. No code/SQL/shell/HTTP.
- **Structured output is parsed → validated → repaired once →
  rejected.** No unbounded retries; malformed output never becomes a
  valid action, a "supported" verdict, or a silent success.
- **Citations are validated, not trusted**: strict `[N]` parsing,
  out-of-range markers retained as invalid, verification never
  receives invalid references.
- **Fail closed**: malformed input → validation error; exhausted
  limits → explicit termination; stale index → explicit error;
  oversized input → explicit rejection. No silent fallbacks.
- **Observability is content-free**: the log formatter emits an
  explicit key allowlist; traces hold IDs and counts. Prompts,
  documents, reasoning, secrets, and headers have no logging path
  (tested with planted fake secrets).

## Explicit resource limits

| Input | Limit | Behavior |
|---|---|---|
| Research question | 4000 chars (`MAX_QUESTION_CHARS`) | `GenerationError`/`AgentError` |
| Raw document | 10 MB (`MAX_DOCUMENT_BYTES`) | `ParsingError` |
| Retrieval `top_k` | 1–100 (`validate_top_k`) | `RetrievalError` |
| Agent search `top_k` | 1–10 | `AgentError` |
| Agent iterations / tool calls | 8 / 12 defaults, ≥1 | explicit termination |
| Agent timeout | 60 s default, >0 | explicit timeout |
| LLM calls | explicit timeout, no retries | typed errors |

NUL bytes are stripped at parse time (PostgreSQL rejects them).
LLM response length is caller-bounded via `max_tokens` (live paths
set it); no global cap is imposed on local generation.

## Known limitations (explicit non-claims)

- Prompt-injection defenses are layered mitigations, not proof of
  immunity; local model behavior varies by version and device.
- Conflict detection covers numeric contradictions in shared
  context only — not negations, paraphrases, or unit mismatches.
- Unicode handling is crash-safe and deterministic, not a
  confusable-resolution engine.
- The security suite is small and synthetic; it does not establish
  absence of vulnerabilities.
- Production deployment additionally needs authentication,
  transport security, secret management, resource isolation, and
  dependency review — all out of scope here.

## Testing approach

`tests/test_security.py` (injection, validation, resources,
unicode, metadata, errors), `tests/test_security_agent.py`
(exhaustion boundaries, allowlist tripwires), and
`tests/test_security_postgres.py` (poisoning, abstention attacks,
log leakage, hostile metadata) run on fakes + local Postgres —
no Ollama, no network. `tests/fixtures/adversarial.txt` is the
shared attack corpus. Report injection findings as attack /
expected / observed / mitigation, and fix-then-rerun.
