# ADR-002: Migration strategy (defer Alembic)

Date: 2026-09-19 | Status: Accepted | Milestone: M2

## Context

M2 requires "migrations" (OPENCODE_PROMPTS M2). Options: (a) Alembic
now, (b) SQLAlchemy `create_all` + versioned SQL files, defer Alembic.
There is no production data yet; schema changes are cheap and the team
is one developer on a local-first stack.

## Decision

- **Source of truth: `src/deepresearch/models.py`.**
- **Apply path: `init_db(engine)`** in `src/deepresearch/db.py` —
  `CREATE EXTENSION IF NOT EXISTS vector` (PostgreSQL only) then
  `Base.metadata.create_all`. Idempotent; safe on startup and in tests.
- **Review artifact: `infra/migrations/001_initial.sql`.** Canonical
  DDL kept in sync with the models by hand for M2. Used for review
  and as the `docker compose` first-start reference.
- **Alembic deferred** until schema must evolve with real data
  (expected M3–M5 or M21 polish). Adding it now is ceremony: no
  downgrades, no data migrations, no multi-developer branches exist.

## Alternatives considered

- Alembic from day one: industry-standard, but adds a dependency,
  env/config surface, and autogenerate noise for zero benefit while
  tables are greenfield and empty (`drop/recreate` is acceptable).
- Hand-written SQL only (no ORM DDL): rejected — doubles the drift
  risk between code and schema with no gain.

## Consequences

- M2–M5 recreate freely; `001_initial.sql` is re-issued per change.
- When the first breaking change with data arrives, introduce Alembic
  with this SQL as the baseline, and record it in a new ADR.
- CI/dev must run `init_db` before PG-backed tests (integration tests
  skip clearly when PostgreSQL is unreachable).
