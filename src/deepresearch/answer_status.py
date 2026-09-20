"""Answer status and conflict detection — Milestone 13.

Makes the outcome states explicit at the application level:

    no_evidence → nothing retrieved (never calls the LLM)
    conflicting_evidence → detector found contradictions (both kept)
    insufficient_evidence → evidence/verification cannot ground an answer
    answered → grounded response without unresolved conflict

Status is distinct from M12 citation-verification verdicts: verification
judges claim↔evidence pairs, status judges the answer as a whole.
Conflict detection is deterministic and conservative (numeric
contradictions in shared context only) — documented limits, no
theorem prover, no extra model.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Literal

from deepresearch.bm25 import normalize_tokens
from deepresearch.citation_verification import CitationVerificationReport
from deepresearch.retrieval import RetrievalResult

AnswerStatus = Literal[
    "no_evidence",
    "insufficient_evidence",
    "answered",
    "conflicting_evidence",
]

CONTEXT_WINDOW = 4
MIN_SHARED_CONTEXT = 2

_DIGIT_PATTERN = re.compile(r"^\d+$")

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_STOPWORDS = frozenset(
    """
    the a an is was are were be been being in on of and or to for with by
    as at from that this it its into over after before between through
    has have had hasnt havent wasnt werent dont doesnt did do does not no
    nor but if then than so such only also very can could should would may
    might must shall will
    """.split()
)


@dataclass(frozen=True)
class EvidenceConflict:
    """One detected contradiction between two evidence items (both kept)."""

    conflict_id: str  # deterministic: conflict-1, conflict-2, ... in detection order
    conflict_type: str  # currently only "numeric_mismatch"
    description: str
    citation_ids: list[int] = field(default_factory=list)  # 1-based evidence positions
    chunk_ids: list[uuid.UUID] = field(default_factory=list)
    document_ids: list[uuid.UUID] = field(default_factory=list)
    values: list[str] = field(default_factory=list)  # the differing values as written


def _number_at(token: str) -> int | None:
    if _DIGIT_PATTERN.match(token):
        return int(token)
    return _NUMBER_WORDS.get(token)


def _occurrences(text: str) -> list[tuple[int, set[str]]]:
    """(numeric value, surrounding content words) per number mention."""
    tokens = normalize_tokens(text or "")
    found: list[tuple[int, set[str]]] = []
    for position, token in enumerate(tokens):
        value = _number_at(token)
        if value is None:
            continue
        start = max(0, position - CONTEXT_WINDOW)
        end = min(len(tokens), position + CONTEXT_WINDOW + 1)
        context = {
            word
            for word in tokens[start:end]
            if word != token and _number_at(word) is None and word not in _STOPWORDS
        }
        found.append((value, context))
    return found


def detect_conflicts(evidence: list[RetrievalResult]) -> list[EvidenceConflict]:
    """Find explicit numeric contradictions between evidence pairs.

    Two items conflict when a number in each shares at least
    ``MIN_SHARED_CONTEXT`` content words of surrounding context but the
    values differ ("introduced in 2022" vs "introduced in 2024").
    Pairwise over evidence order, first conflicting value-pair per item
    pair, deterministic. Catches only numeric contradictions in shared
    context — not negations, paraphrases, or unit mismatches (see ADR).
    """
    occurrences = [_occurrences(item.text) for item in evidence]
    conflicts: list[EvidenceConflict] = []
    for left in range(len(evidence)):
        for right in range(left + 1, len(evidence)):
            match = _first_mismatch(occurrences[left], occurrences[right])
            if match is None:
                continue
            (left_value, shared), (right_value, _) = match
            first, second = evidence[left], evidence[right]
            conflicts.append(
                EvidenceConflict(
                    conflict_id=f"conflict-{len(conflicts) + 1}",
                    conflict_type="numeric_mismatch",
                    description=(
                        f"Evidence [{left + 1}] states {left_value} where "
                        f"evidence [{right + 1}] states {right_value} "
                        f"(shared context: {', '.join(sorted(shared))})"
                    ),
                    citation_ids=[left + 1, right + 1],
                    chunk_ids=[first.chunk_id, second.chunk_id],
                    document_ids=[first.document_id, second.document_id],
                    values=[str(left_value), str(right_value)],
                )
            )
    return conflicts


def _first_mismatch(
    left: list[tuple[int, set[str]]], right: list[tuple[int, set[str]]]
) -> tuple[tuple[int, set[str]], tuple[int, set[str]]] | None:
    for left_value, left_context in left:
        for right_value, right_context in right:
            shared = left_context & right_context
            if left_value != right_value and len(shared) >= MIN_SHARED_CONTEXT:
                return (left_value, shared), (right_value, right_context)
    return None


def determine_answer_status(
    *,
    has_evidence: bool,
    conflicts: list[EvidenceConflict],
    verification_report: CitationVerificationReport | None = None,
) -> AnswerStatus:
    """Apply the M13 precedence rules (deterministic, no LLM wording involved).

    no_evidence → conflicting_evidence → verification-based
    insufficient_evidence → answered. An invalid marker alone never
    forces insufficient_evidence: secondary citations can be invalid
    while the answer stands (see ADR-013).
    """
    if not has_evidence:
        return "no_evidence"
    if conflicts:
        return "conflicting_evidence"
    if verification_report is not None:
        statuses = {result.status for result in verification_report.results}
        if "unsupported" in statuses or "unverifiable" in statuses:
            return "insufficient_evidence"
    return "answered"
