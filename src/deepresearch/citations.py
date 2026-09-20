"""Source citations for grounded answers — Milestone 11.

Citation contract (association only, NOT correctness — that is M12):

- Each evidence item gets a stable ``[N]`` marker from its position in
  the supplied evidence order (1-based, deterministic, never a UUID).
- ``extract_citations`` maps markers found in answer text back to
  evidence, distinguishing valid from out-of-range references.
- Only strict numeric ``[N]`` markers count; ``[Source A]``, ``[-1]``,
  and similar bracketed text are ignored, never executed.

Standard library parsing only.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from deepresearch.retrieval import RetrievalResult

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Citation:
    """A presentation/provenance object derived from one evidence item."""

    citation_id: int  # 1-based position in the supplied evidence order
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str | None
    document_source: str
    document_type: str
    page: int | None
    section: str | None
    retrieval_method: str
    retrieval_score: float
    retrieval_rank: int


@dataclass(frozen=True)
class InvalidCitationReference:
    """A numeric marker with no corresponding evidence (out of range).

    Retained as metadata for M12 verification; never silently remapped
    to a valid citation and never deleted from the record.
    """

    citation_id: int


@dataclass(frozen=True)
class CitationExtraction:
    """Deterministic extraction result: valid citations in first-use order."""

    citations: list[Citation] = field(default_factory=list)
    invalid: list[InvalidCitationReference] = field(default_factory=list)


def citation_for(position: int, evidence: RetrievalResult) -> Citation:
    """Build the ``[position]`` citation for one evidence item (1-based)."""
    if position < 1:
        raise ValueError(f"citation position must be >= 1, got {position}")
    return Citation(
        citation_id=position,
        chunk_id=evidence.chunk_id,
        document_id=evidence.document_id,
        document_title=evidence.document_title,
        document_source=evidence.document_source,
        document_type=evidence.document_type,
        page=evidence.page,
        section=evidence.section,
        retrieval_method=evidence.retrieval_method,
        retrieval_score=evidence.score,
        retrieval_rank=evidence.rank,
    )


def assign_citations(evidence: list[RetrievalResult]) -> list[Citation]:
    """Number every evidence item in order: 1 → ``[1]``, 2 → ``[2]``, …"""
    return [citation_for(position, item) for position, item in enumerate(evidence, start=1)]


def extract_citations(answer_text: str, evidence: list[RetrievalResult]) -> CitationExtraction:
    """Map numeric ``[N]`` markers in answer text to evidence.

    Valid IDs (``1..len(evidence)``) become ``Citation`` objects;
    out-of-range IDs become ``InvalidCitationReference`` records. Both
    lists are deduplicated preserving first-use order in the answer.
    """
    assigned = assign_citations(evidence)
    by_id = {citation.citation_id: citation for citation in assigned}
    citations: list[Citation] = []
    seen_valid: set[int] = set()
    invalid: list[InvalidCitationReference] = []
    seen_invalid: set[int] = set()
    for match in CITATION_PATTERN.finditer(answer_text or ""):
        number = int(match.group(1))
        if number in by_id:
            if number not in seen_valid:
                seen_valid.add(number)
                citations.append(by_id[number])
        elif number not in seen_invalid:
            seen_invalid.add(number)
            invalid.append(InvalidCitationReference(citation_id=number))
    return CitationExtraction(citations=citations, invalid=invalid)
