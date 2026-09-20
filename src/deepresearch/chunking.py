"""Deterministic token-based chunking — Milestone 3.

Tokenizer: a small regex word/punctuation tokenizer defined here
(``[\\w']+`` words, single punctuation marks otherwise). It is
deliberately NOT the future embedding-model tokenizer: M3 chunks must
not couple to M4's ``bge-small-en-v1.5`` vocabulary (see ADR-003).
Token counts are therefore "deepresearch tokens" — stable,
dependency-free, and comparable across chunking experiments.

Chunking: sliding window over the token stream with configurable
target/overlap. Pure function of (units, target, overlap): identical
input + identical config → identical chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from deepresearch.parsing import TextUnit

_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
_WS_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?)\]\}%…])")
_WS_AFTER_OPEN_RE = re.compile(r"([(\[{])\s+")
_WS_RUN_RE = re.compile(r"\s+")


class ChunkingError(ValueError):
    """Raised for invalid chunking configuration."""


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    page: int | None
    section: str | None
    token_count: int


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = _WS_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _WS_AFTER_OPEN_RE.sub(r"\1", text)
    return _WS_RUN_RE.sub(" ", text).strip()


def validate_config(*, target_tokens: int, overlap_tokens: int) -> None:
    if target_tokens <= 0:
        raise ChunkingError(f"target_tokens must be > 0, got {target_tokens}")
    if overlap_tokens < 0:
        raise ChunkingError(f"overlap_tokens must be >= 0, got {overlap_tokens}")
    if overlap_tokens >= target_tokens:
        raise ChunkingError(
            f"overlap_tokens ({overlap_tokens}) must be < target_tokens ({target_tokens})"
        )


def chunk_units(
    units: list[TextUnit] | tuple[TextUnit, ...],
    *,
    target_tokens: int = 800,
    overlap_tokens: int = 120,
) -> list[ChunkDraft]:
    """Split ordered text units into overlapping token-window chunks."""
    validate_config(target_tokens=target_tokens, overlap_tokens=overlap_tokens)

    tokens: list[str] = []
    token_meta: list[tuple[int | None, str | None]] = []  # (page, section) per token
    for unit in units:
        unit_tokens = tokenize(unit.text)
        if not unit_tokens:
            continue
        tokens.extend(unit_tokens)
        token_meta.extend([(unit.page, unit.section)] * len(unit_tokens))

    if not tokens:
        return []

    drafts: list[ChunkDraft] = []
    start = 0
    total = len(tokens)
    while start < total:
        end = min(start + target_tokens, total)
        window = tokens[start:end]
        text = detokenize(window)
        if text:  # never emit empty chunks (e.g. punctuation-only windows normalize away)
            page, section = token_meta[start]
            drafts.append(
                ChunkDraft(text=text, page=page, section=section, token_count=end - start)
            )
        if end == total:
            break
        start = end - overlap_tokens  # overlap < target ⇒ forward progress guaranteed
    return drafts
