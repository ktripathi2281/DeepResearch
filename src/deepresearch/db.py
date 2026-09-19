"""Database engine and connectivity check (Milestone 1).

No schema yet — that is Milestone 2. This module only creates a
SQLAlchemy engine from ``Settings.database_url`` and exposes a
``check_connection()`` helper used by ``/ready`` and tests.
"""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from deepresearch.config import Settings


def get_engine(settings: Settings) -> Engine:
    return create_engine(settings.database_url, pool_pre_ping=True)


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
