"""Application configuration via environment variables.

Milestones 3–4 add chunking (token-based: target 800, overlap 120) and
embedding configuration (bge-small-en-v1.5, batch 32, device auto).
Hardware constraints (Lenovo LOQ, 16 GB RAM, 6 GB VRAM,
qwen3:4b / bge-small-en-v1.5 / bge-reranker-base) are documented in
docs/ and must not change without explicit approval.
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


def get_settings() -> Settings:
    return Settings()
