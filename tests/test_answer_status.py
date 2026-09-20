"""M13 unit tests — statuses, conflict detection, prompt integration (fakes only).

No database, no model, no network. PostgreSQL end-to-end lives in
test_answer_status_postgres.py; live Ollama in test_generation_ollama_live.py.
"""

from __future__ import annotations

import uuid

import pytest

import deepresearch.generation as generation_mod
from deepresearch.answer_status import (
    EvidenceConflict,
    detect_conflicts,
    determine_answer_status,
)
from deepresearch.generation import (
    NO_EVIDENCE_MESSAGE,
    Evidence,
    answer_question,
    build_grounded_prompt,
)
from tests.fakes import FakeLLMProvider, FakeVerifierLLM

SUPPORTED_JSON = '{"verdict": "supported", "explanation": "Stated."}'
UNSUPPORTED_JSON = '{"verdict": "unsupported", "explanation": "Contradicted."}'
WEAK_JSON = '{"verdict": "insufficient_evidence", "explanation": "Partial."}'


def _evidence(text: str, **overrides) -> Evidence:  # type: ignore[no-untyped-def]
    fields = {
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "chunk_index": 0,
        "text": text,
        "score": 0.9,
        "rank": 1,
        "document_title": "Title",
        "document_type": "markdown",
        "document_source": "doc.md",
        "page": None,
        "section": None,
        "chunk_metadata": None,
        "retrieval_method": "reranked",
    }
    fields.update(overrides)
    return Evidence(**fields)


def _stub(monkeypatch, evidence: list[Evidence]):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(generation_mod, "retrieve_hybrid", lambda *a, **k: evidence)
    monkeypatch.setattr(generation_mod, "rerank_results", lambda *a, **k: evidence)


# --- conflict detection ------------------------------------------------------


def test_spec_fixture_year_conflict() -> None:
    (conflict,) = detect_conflicts(
        [
            _evidence("The policy was introduced in 2022."),
            _evidence("The policy was introduced in 2024."),
        ]
    )
    assert conflict.conflict_type == "numeric_mismatch"
    assert conflict.values == ["2022", "2024"]
    assert conflict.citation_ids == [1, 2]
    assert len(conflict.chunk_ids) == 2 and len(conflict.document_ids) == 2
    assert "2022" in conflict.description and "2024" in conflict.description


def test_spec_fixture_count_conflict_and_word_numbers() -> None:
    (conflict,) = detect_conflicts(
        [_evidence("The project has 5 stages."), _evidence("The project has eight stages.")]
    )
    assert conflict.values == ["5", "8"]


def _policy_pair() -> list[Evidence]:
    return [
        _evidence("The policy was introduced in 2022."),
        _evidence("The policy was introduced in 2024."),
    ]


def test_no_conflict_cases() -> None:
    assert detect_conflicts([_evidence("The policy was introduced in 2022.")]) == []
    same = [_evidence("The policy was introduced in 2022.")]
    assert detect_conflicts(_policy_pair()[:1] + same) == []
    unrelated = [_evidence("Budget was 5 million."), _evidence("Rain fell in 2024.")]
    assert detect_conflicts(unrelated) == []
    assert detect_conflicts([]) == []


def test_detection_deterministic() -> None:
    items = _policy_pair()
    first = detect_conflicts(items)
    second = detect_conflicts(items)
    assert [(c.conflict_id, c.description) for c in first] == [
        (c.conflict_id, c.description) for c in second
    ]
    assert first[0].conflict_id == "conflict-1"


# --- status determination ------------------------------------------------------


def test_determine_precedence() -> None:
    assert determine_answer_status(has_evidence=False, conflicts=[]) == "no_evidence"
    conflict = EvidenceConflict("conflict-1", "numeric_mismatch", "d")
    assert determine_answer_status(has_evidence=True, conflicts=[conflict]) == (
        "conflicting_evidence"
    )
    assert determine_answer_status(has_evidence=True, conflicts=[]) == "answered"


def _answer(  # type: ignore[no-untyped-def]
    monkeypatch, evidence, llm, question="Q?", **kwargs
):
    _stub(monkeypatch, evidence)
    return answer_question(None, None, None, llm, question, **kwargs)  # type: ignore[arg-type]


def test_no_evidence_full_contract(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    llm, verifier = FakeLLMProvider(), FakeVerifierLLM(responses=[])
    result = _answer(monkeypatch, [], llm, verify_citations=True, verifier=verifier)
    assert result.status == "no_evidence"
    assert result.answer == NO_EVIDENCE_MESSAGE
    assert result.has_evidence is False
    assert result.citations == [] and result.invalid_citations == []
    assert result.verification_report is not None and result.verification_report.results == []
    assert result.conflicts == []
    assert llm.calls == [] and verifier.calls == []


def test_answered_default_path(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    items = [_evidence("Consistent alpha fact."), _evidence("Consistent beta fact.")]
    result = _answer(monkeypatch, items, FakeLLMProvider(answer="Answer [1]."))
    assert result.status == "answered"
    assert result.evidence == items
    assert [c.citation_id for c in result.citations] == [1]
    assert result.conflicts == []
    assert result.verification_report is None  # default off, unchanged


def test_conflicting_evidence_status_and_preservation(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    items = _policy_pair()
    llm = FakeLLMProvider(answer="Sources disagree [1][2].")
    verifier = FakeVerifierLLM(responses=[SUPPORTED_JSON])
    result = _answer(monkeypatch, items, llm, "When?", verify_citations=True, verifier=verifier)
    assert result.status == "conflicting_evidence"
    assert len(result.conflicts) == 1
    assert result.evidence == items  # both sources kept, no silent winner
    assert [c.citation_id for c in result.citations] == [1, 2]
    assert "disagreement" in llm.calls[0]["system_prompt"]
    assert "2022" in llm.calls[0]["system_prompt"] and "2024" in llm.calls[0]["system_prompt"]


def test_conflict_guidance_only_when_needed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    plain = build_grounded_prompt("Q?", [_evidence("Consistent fact.")])
    assert "disagreement" not in plain.system
    pair = _policy_pair()
    conflicted = build_grounded_prompt("Q?", pair, conflicts=detect_conflicts(pair))
    assert "do not silently choose" in conflicted.system.lower()
    assert "untrusted document content" in conflicted.system


def test_verification_drives_insufficient(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    items = [_evidence("Claimed fact."), _evidence("Other fact.")]
    llm = FakeLLMProvider(answer="Claimed [1]. Other [2].")
    verifier = FakeVerifierLLM(responses=[UNSUPPORTED_JSON, SUPPORTED_JSON])
    bad = _answer(monkeypatch, items, llm, verify_citations=True, verifier=verifier)
    assert bad.status == "insufficient_evidence"
    assert bad.verification_report is not None
    assert bad.citations  # citations preserved alongside the status

    weak_llm = FakeLLMProvider(answer="Claimed [1].")
    weak = _answer(
        monkeypatch,
        items[:1],
        weak_llm,
        verify_citations=True,
        verifier=FakeVerifierLLM(responses=["junk", "junk"]),
    )
    assert weak.status == "insufficient_evidence"

    good_llm = FakeLLMProvider(answer="Claimed [1].")
    good = _answer(
        monkeypatch,
        items[:1],
        good_llm,
        verify_citations=True,
        verifier=FakeVerifierLLM(responses=[SUPPORTED_JSON]),
    )
    assert good.status == "answered"


def test_detector_failure_propagates(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _stub(monkeypatch, [_evidence("Fact.")])

    def boom(evidence):  # type: ignore[no-untyped-def]
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(generation_mod, "detect_conflicts", boom)
    with pytest.raises(RuntimeError, match="detector exploded"):
        answer_question(None, None, None, FakeLLMProvider(), "Q?")  # type: ignore[arg-type]


def test_invalid_citation_does_not_force_insufficient(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Per ADR: a secondary invalid marker can coexist with an answered status.
    _stub(monkeypatch, [_evidence("Fact.")])
    llm = FakeLLMProvider(answer="Fact [1] plus stray [9].")
    result = answer_question(None, None, None, llm, "Q?")  # type: ignore[arg-type]
    assert result.status == "answered"
    assert [c.citation_id for c in result.citations] == [1]
    assert [i.citation_id for i in result.invalid_citations] == [9]
