"""Config tests: defaults and env-var overrides."""

import os

from deepresearch.config import Settings


def test_settings_defaults() -> None:
    s = Settings()
    assert s.app_name == "deepresearch"
    assert s.api_port == 8000
    assert "postgresql" in s.database_url


def test_settings_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("APP_NAME", "custom-app")
    monkeypatch.setenv("API_PORT", "9001")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    # os.environ check guards against pydantic-settings caching surprises.
    assert os.environ["APP_NAME"] == "custom-app"
    s = Settings()
    assert s.app_name == "custom-app"
    assert s.api_port == 9001
    assert s.database_url.endswith("/db")
