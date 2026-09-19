"""Application configuration via environment variables.

Milestone 1 only: no RAG/model settings beyond what is needed for
service startup and DB connectivity. Model names remain configurable
placeholders for later milestones; hardware constraints (Lenovo LOQ,
16 GB RAM, 6 GB VRAM, qwen3:4b / bge-small-en-v1.5 / bge-reranker-base)
are documented in docs/ and must not change without explicit approval.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="deepresearch")
    app_env: str = Field(default="local")
    log_level: str = Field(default="INFO")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    database_url: str = Field(
        default="postgresql+psycopg://deepresearch:deepresearch@localhost:5432/deepresearch"
    )


def get_settings() -> Settings:
    return Settings()
