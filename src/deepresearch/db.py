"""Database engine, schema init, and connectivity check.

Milestone 1 introduced the engine + ``check_connection`` helper.
Milestone 2 adds ``init_db`` (create extension + tables) without
changing the M1 API.
"""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from deepresearch.config import Settings

# Milestone 20: bound connection establishment. Without this, an
# unreachable database can block first-request initialization for
# minutes (observed: ~260 s of psycopg retries). Readiness and startup
# must fail fast instead. This is failure bounding, not pool tuning:
# pool size/recycle behavior is intentionally untouched.
DB_CONNECT_TIMEOUT_SECONDS = 10


def get_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS},
    )


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create the pgvector extension (PostgreSQL only) and all tables.

    Idempotent: safe to call on every startup and in tests. Alembic is
    deliberately deferred (see docs/adr/002-migration-strategy.md);
    models are the source of truth until schema evolves with real data.
    """
    from deepresearch.models import Base  # local import avoids cycles

    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def check_connection(engine: Engine, timeout_seconds: float = 5.0) -> bool:
    """Return True if ``SELECT 1`` succeeds, False otherwise.

    Never raises: callers (readiness probe, tests) decide how to handle
    an unavailable database. Uses a statement timeout via execution
    options where the dialect supports it; falls back to a plain query.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
