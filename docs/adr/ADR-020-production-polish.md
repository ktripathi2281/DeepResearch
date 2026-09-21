# ADR-020: Production polish and reliability

Date: 2026-09-21 | Status: Accepted | Milestone: M20

## Context

M1–M19 built capabilities (pipeline, agent, observability, eval,
security, frontend, providers) but left runtime behavior
development-grade: configuration failed late as mysterious job
errors, readiness checked only the database, the in-memory job store
grew without bound, shutdown had no defined job semantics, and the
500 path dropped the request-ID response header. M20 hardens the
service without adding AI capabilities, queues, auth, or new
dependencies.

## Decision

- **Fail fast on configuration.** Field-level pydantic bounds
  (including a new `api_port` 1–65535 range) catch bad numbers at
  load; `Settings.validate_for_runtime()` checks the cross-cutting
  trio — provider name, CORS origins (explicit http(s) origins, no
  `*`, no paths), database scheme (postgresql family) — and runs in
  the lifespan before serving. Messages name valid values and never
  carry secrets. `Settings(...)` itself stays permissive so tests
  and tools can construct arbitrary instances.
- **Bounded database waits.** `get_engine` sets a 10 s
  `connect_timeout` after an observed ~260 s psycopg retry storm
  hung first-request initialization. This is failure bounding, not
  pool tuning: pool behavior is untouched.
- **Explicit health/readiness split.** `/health` = process alive, no
  dependencies, unchanged shape. `/ready` = can accept research
  work: provider selection valid (lazy construction, no network)
  AND PostgreSQL reachable. The model daemon is deliberately never
  probed — it may be down at boot; jobs then fail cleanly. Bodies
  stay exactly `{"status": ...}` (an M17 test pins them); detail
  goes to logs.
- **Explicit job lifecycle.** `running → completed | failed` per job
  object. Resubmitting a running ID returns the running snapshot
  (no second thread); resubmitting a terminal ID starts a fresh job.
  `shutdown()` rejects new work (503 `shutting_down` envelope) and
  marks running jobs `failed` with `type: "interrupted"` — completed
  results are never fabricated across restarts, and the documented
  in-memory limitation stands.
- **Bounded retention.** `research_max_retained_jobs` (default 100,
  configurable) evicts oldest terminal jobs first; running jobs are
  never candidates. ~KBs per job keeps worst-case memory in
  single-digit MB.
- **Consistent safe errors.** 422 (FastAPI), 404
  `{"detail": "unknown research request"}`, 500
  `{"error": {message, type, request_id}}`, 503 shutdown envelope —
  no stack traces, keys, headers, or SQL anywhere. The 500 handler
  now stamps `X-Request-ID` itself: unhandled exceptions propagate
  through `call_next` after the safe response is generated, so the
  middleware stamp never runs on that path (found by an M20
  regression test).
- **Two-level request IDs (documented, tested).** The
  `X-Request-ID` header identifies one HTTP request; the job ID
  lives in bodies. Submit-with-ID round-trips identically in body
  and header; polls echo their own HTTP ID in the header and the
  job's ID in the body.
- **Graceful shutdown.** Lifespan closes provider HTTP clients
  (reused single clients per adapter, verified), disposes the
  engine, and reports interrupted-job counts via the JSON log
  allowlist (extended with documented operational keys only).
- **Timeouts documented, not added.** Submission is immediate;
  execution is bounded by existing provider timeouts (120 s),
  the agent cap (60 s), and verification's single repair; the
  frontend polls with a 5-minute client timeout. No new timeout
  was necessary.
- **Docker stays minimal.** One addition: an `api` healthcheck
  hitting dependency-free `/health` via stdlib urllib (slim image
  has no curl), with a generous start period. No new containers,
  no Kubernetes, no Alembic — fresh databases initialize through
  the existing idempotent `init_db`, documented in README
  troubleshooting.

## Alternatives considered

- **Persisting jobs or adding a queue/Redis**: rejected — out of
  scope; the bounded in-memory store matches the documented
  single-process model.
- **Probing the model daemon in /ready**: rejected — flaps
  readiness on an optional-at-boot dependency and adds network
  cost to every probe.
- **Enriching /ready bodies with per-check detail**: rejected —
  an M17 test pins the exact shapes, and detail belongs in logs.
- **A new "interrupted" job status**: rejected — the three-state
  contract (`running`/`completed`/`failed`) is what the frontend
  understands; interruption is a `failed` job with
  `type: "interrupted"`.

## Consequences

- The service starts, checks, serves, sheds load at shutdown, and
  diagnoses predictably with zero new dependencies.
- `tests/test_reliability.py` (54 tests) locks the contract:
  config, health/readiness, lifecycle, retention, concurrency
  isolation, error shapes, shutdown, request IDs, client reuse,
  and loose performance sanity bounds.
- Explicitly out of scope (unchanged): new AI capabilities, RAG
  changes, routing/fallback, MCP, web search, auth, billing,
  Kubernetes, cloud deployment, workers, queues, monitoring
  platforms.
