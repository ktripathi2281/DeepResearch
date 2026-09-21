"""Application configuration via environment variables.

Milestones 3–4 add chunking (token-based: target 800, overlap 120) and
embedding configuration (bge-small-en-v1.5, batch 32, device auto).
Hardware constraints (Lenovo LOQ, 16 GB RAM, 6 GB VRAM,
qwen3:4b / bge-small-en-v1.5 / bge-reranker-base) are documented in
docs/ and must not change without explicit approval.

Milestone 20 adds explicit runtime validation
(``Settings.validate_for_runtime``): field-level pydantic bounds catch
bad numbers/ports at load, while cross-cutting checks (provider name,
CORS origins, database scheme) run eagerly at startup via the
lifespan so misconfiguration fails fast with a clear, secret-free
message instead of surfacing as a mysterious job failure later.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(ValueError):
    """A setting is invalid for runtime use; message names the problem, never secrets."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="deepresearch")
    app_env: str = Field(default="local")
    log_level: str = Field(default="INFO")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    # Milestone 18: browser origins allowed to call the research API.
    # Comma-separated list; credentials are never enabled, so this is
    # a simple allowlist for the local Next.js frontend.
    cors_origins: str = Field(default="http://localhost:3000")

    database_url: str = Field(
        default="postgresql+psycopg://deepresearch:deepresearch@localhost:5432/deepresearch"
    )

    # Milestone 3: token-based chunking (see ADR-003). Overlap must be
    # smaller than the target; validated in chunking, not here, so that
    # misconfiguration raises a clear ChunkingError at ingest time.
    chunk_target_tokens: int = Field(default=800, gt=0)
    chunk_overlap_tokens: int = Field(default=120, ge=0)

    # Milestone 4: local embeddings (see ADR-004). BAAI/bge-small-en-v1.5
    # outputs 384 dimensions; batch 32 is conservative for 16 GB RAM /
    # 6 GB VRAM. Device "auto" uses CUDA when available, else CPU.
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_model_version: str = Field(default="1")
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_device: str = Field(default="auto")
    embedding_normalize: bool = Field(default=True)

    # Milestone 5: vector retrieval (see ADR-005). Default top-K 5 matches
    # the evaluation baseline (Recall@5); the cap prevents accidentally
    # returning the whole corpus. Over-limit requests raise, never clamp.
    retrieval_top_k: int = Field(default=5, ge=1)
    retrieval_max_top_k: int = Field(default=100, ge=1)

    # Milestone 6: BM25 defaults (see ADR-006). Standard Okapi values;
    # configurable so M7 experiments can vary them explicitly.
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)

    # Milestone 7: hybrid RRF fusion (see ADR-007). rrf_k=60 is the
    # literature default; candidate pools default to 2 * top_k.
    hybrid_rrf_k: int = Field(default=60, ge=1)

    # Milestone 8: local cross-encoder reranking (see ADR-008).
    # bge-reranker-base (~278M params) runs on CPU; batch 16 is
    # conservative for 16 GB RAM / 6 GB VRAM. Candidate pool 20 with
    # final top-K 5 mirrors the retrieval conventions.
    reranker_model: str = Field(default="BAAI/bge-reranker-base")
    reranker_model_version: str = Field(default="1")
    reranker_device: str = Field(default="auto")
    reranker_batch_size: int = Field(default=16, gt=0)
    reranker_candidate_top_k: int = Field(default=20, ge=1)

    # Milestone 9: local Ollama generation (see ADR-009). qwen3:4b is the
    # constrained baseline (never silently substituted); max_tokens=None
    # means Ollama's own default (num_predict sent only when set).
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen3:4b")
    ollama_timeout_seconds: float = Field(default=120, gt=0)
    ollama_temperature: float = Field(default=0.0, ge=0)
    ollama_max_tokens: int | None = Field(default=None, gt=0)

    # Milestone 19: provider selection (see ADR-019). The default stays
    # the local path; cloud adapters are optional, disabled without
    # credentials, and selected only through the factory. Secrets are
    # environment-only (None by default) and never logged.
    llm_provider: str = Field(default="ollama")
    openai_compatible_base_url: str = Field(default="https://api.openai.com/v1")
    openai_compatible_api_key: str | None = Field(default=None)
    openai_compatible_model: str = Field(default="gpt-4o-mini")
    openai_compatible_timeout_seconds: float = Field(default=120, gt=0)
    openai_compatible_temperature: float = Field(default=0.0, ge=0)
    openai_compatible_max_tokens: int | None = Field(default=None, gt=0)
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com")
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.0-flash")
    gemini_timeout_seconds: float = Field(default=120, gt=0)
    gemini_temperature: float = Field(default=0.0, ge=0)
    gemini_max_tokens: int | None = Field(default=None, gt=0)

    # Milestone 14: bounded research agent (see ADR-014). Caps guarantee
    # termination; the agent itself can never raise them.
    agent_max_iterations: int = Field(default=8, ge=1)
    agent_max_tool_calls: int = Field(default=12, ge=1)
    agent_timeout_seconds: float = Field(default=60, gt=0)

    # Milestone 20: bounded in-memory job retention (see ADR-020). Each
    # retained job holds its answer plus evidence text; 100 bounds
    # worst-case memory to single-digit megabytes. Only terminal jobs
    # (completed/failed) are ever evicted, oldest first; running jobs
    # are never evicted.
    research_max_retained_jobs: int = Field(default=100, ge=1)

    def validate_for_runtime(self) -> Settings:
        """Fail fast on configuration that cannot serve research.

        Called once at application startup (lifespan), not at import:
        ``Settings(...)`` itself stays permissive so tests and tools
        can construct arbitrary instances. Every message names valid
        values; no message ever includes credentials, keys, or URLs
        carrying them.
        """
        from deepresearch.providers import KNOWN_PROVIDERS  # deferred: providers imports config

        name = (self.llm_provider or "").strip().lower()
        if name not in KNOWN_PROVIDERS:
            raise ConfigurationError(
                f"invalid LLM_PROVIDER {self.llm_provider!r}; "
                f"expected one of: {', '.join(KNOWN_PROVIDERS)}"
            )
        self._validate_cors_origins()
        self._validate_database_url()
        return self

    def _validate_cors_origins(self) -> None:
        from urllib.parse import urlparse

        entries = [entry.strip() for entry in (self.cors_origins or "").split(",")]
        entries = [entry for entry in entries if entry]
        for entry in entries:
            if entry == "*":
                raise ConfigurationError(
                    "invalid CORS_ORIGINS '*': list explicit browser origins instead "
                    "(e.g. CORS_ORIGINS=http://localhost:3000)"
                )
            parsed = urlparse(entry)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ConfigurationError(
                    f"invalid CORS_ORIGINS entry {entry!r}: expected an http(s) origin "
                    "like http://localhost:3000"
                )
            if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
                raise ConfigurationError(
                    f"invalid CORS_ORIGINS entry {entry!r}: origins carry no path, "
                    "query, or fragment"
                )

    def _validate_database_url(self) -> None:
        from sqlalchemy.engine.url import make_url
        from sqlalchemy.exc import ArgumentError

        try:
            url = make_url(self.database_url)
        except ArgumentError as exc:
            raise ConfigurationError("invalid DATABASE_URL: not a parseable database URL") from exc
        backend = (url.get_backend_name() or "").lower()
        if not backend.startswith("postgresql"):
            raise ConfigurationError(
                f"invalid DATABASE_URL backend {backend!r}: this service requires "
                "a postgresql database (pgvector); got a different backend"
            )


def get_settings() -> Settings:
    return Settings()
