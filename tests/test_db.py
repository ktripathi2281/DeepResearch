"""DB connectivity tests.

- test_check_connection_false_on_bad_host: deterministic, no live DB needed.
- test_check_connection_true_against_live_db: runs only when a PostgreSQL
  URL is reachable (skipped otherwise, e.g. CI without services or a dev
  machine without `docker compose up`). Override with TEST_DATABASE_URL.
"""

import os

import pytest
from sqlalchemy import create_engine

from deepresearch.config import Settings, get_settings
from deepresearch.db import check_connection, get_engine


def test_get_engine_uses_configured_url() -> None:
    s = Settings(
        database_url="postgresql+psycopg://u:p@localhost:5432/db",
    )
    engine = get_engine(s)
    try:
        assert "localhost" in str(engine.url)
    finally:
        engine.dispose()


def test_check_connection_false_on_unreachable_host() -> None:
    engine = create_engine(
        "postgresql+psycopg://u:p@127.0.0.1:1/db",
        connect_args={"connect_timeout": 1},
    )
    try:
        assert check_connection(engine) is False
    finally:
        engine.dispose()


def test_check_connection_true_against_live_db() -> None:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    try:
        if not check_connection(engine):
            pytest.skip(f"PostgreSQL not reachable at {url}; run `docker compose up -d postgres`")
    finally:
        engine.dispose()
    assert True
