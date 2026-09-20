"""M11 unit tests — numbering, extraction, prompt integration (no database/LLM)."""

from __future__ import annotations

import uuid

import pytest

import deepresearch.generation as generation_mod
from deepresearch.citations import (
    Citation,
    InvalidCitationReference,
    assign_citations,
    citation_for,
    extract_citations,
)
from deepresearch.generation import (
    NO_EVIDENCE_MESSAGE,
    Evidence,
    answer_question,
    build_grounded_prompt,
)
from tests.fakes import FakeLLMProvider


def _evidence(text: str = "fact", **overrides) -> Evidence:  # type: ignore[no-untyped-def]
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
        "page": 3,
        "section": "Sec",
        "chunk_metadata": None,
        "retrieval_method": "reranked",
    }
    fields.update(overrides)
    return Evidence(**fields)


def _items(n: int) -> list[Evidence]:
    return [_evidence(f"fact {i}", chunk_index=i, rank=i + 1) for i in range(n)]


# --- A. numbering ----------------------------------------------------------


def test_numbering_follows_evidence_order() -> None:
    items = _items(3)
    assigned = assign_citations(items)
    assert [c.citation_id for c in assigned] == [1, 2, 3]
    assert [c.chunk_id for c in assigned] == [item.chunk_id for item in items]
    with pytest.raises(ValueError):
        citation_for(0, items[0])


# --- B/E. extraction + metadata --------------------------------------------


def test_single_and_multiple_citations() -> None:
    items = _items(3)
    result = extract_citations("Claim [2] and more [1].", items)
    assert [c.citation_id for c in result.citations] == [2, 1]
    assert result.invalid == []
    first = result.citations[0]
    assert isinstance(first, Citation)
    assert first.chunk_id == items[1].chunk_id
    assert first.document_id == items[1].document_id
    assert first.document_title == "Title"
    assert first.document_source == "doc.md"
    assert first.document_type == "markdown"
    assert first.page == 3
    assert first.section == "Sec"


def test_repeated_citation_deduplicated() -> None:
    result = extract_citations("A [1]. B [1]. C [1].", _items(2))
    assert [c.citation_id for c in result.citations] == [1]


def test_first_use_ordering() -> None:
    result = extract_citations("A [3]. B [1]. C [3]. D [2].", _items(3))
    assert [c.citation_id for c in result.citations] == [3, 1, 2]


def test_no_citations() -> None:
    result = extract_citations("A plain answer.", _items(2))
    assert result.citations == [] and result.invalid == []


def test_invalid_and_mixed_citations_retained() -> None:
    items = _items(3)
    result = extract_citations("Valid [1], bogus [4], zero [0], again [4].", items)
    assert [c.citation_id for c in result.citations] == [1]
    assert result.invalid == [
        InvalidCitationReference(citation_id=4),
        InvalidCitationReference(citation_id=0),
    ]


def test_strict_bracket_parsing() -> None:
    items = _items(2)
    result = extract_citations("See [Source A], [-1], and [1].", items)
    assert [c.citation_id for c in result.citations] == [1]
    assert result.invalid == []
    assert extract_citations("", items).citations == []


def test_leading_zero_maps_to_evidence() -> None:
    result = extract_citations("Claim [01].", _items(2))
    assert [c.citation_id for c in result.citations] == [1]


def test_deterministic_extraction() -> None:
    items = _items(3)
    text = "A [2]. B [9]. C [1]. D [2]. E [9]."
    first = extract_citations(text, items)
    second = extract_citations(text, items)
    assert [(c.citation_id, c.chunk_id) for c in first.citations] == [
        (c.citation_id, c.chunk_id) for c in second.citations
    ]
    assert first.invalid == second.invalid


# --- F. prompt --------------------------------------------------------------


def test_prompt_blocks_carry_citation_markers_and_rules() -> None:
    prompt = build_grounded_prompt("Q?", _items(2))
    assert "Citation [1]" in prompt.user
    assert "Citation [2]" in prompt.user
    assert "citation markers" in prompt.system
    assert "Never invent markers" in prompt.system
    assert "does not support the claim" in prompt.system
    # M10 grounding rules preserved verbatim.
    assert "using ONLY the evidence" in prompt.system
    assert "untrusted" in prompt.system


# --- G/H. generation integration ---------------------------------------------


def _stub_pipeline(monkeypatch, evidence: list[Evidence]):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(generation_mod, "retrieve_hybrid", lambda *a, **k: evidence)
    monkeypatch.setattr(generation_mod, "rerank_results", lambda *a, **k: evidence)


def test_answer_carries_extracted_citations(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    items = _items(3)
    _stub_pipeline(monkeypatch, items)
    llm = FakeLLMProvider(answer="First [2]. Second [1]. First again [2].")
    result = answer_question(None, None, None, llm, "Q?")  # type: ignore[arg-type]
    assert [c.citation_id for c in result.citations] == [2, 1]
    assert result.citations[0].chunk_id == items[1].chunk_id
    assert result.invalid_citations == []
    assert len(result.evidence) == 3  # uncited evidence preserved for M12


def test_answer_records_invalid_without_remapping(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    items = _items(2)
    _stub_pipeline(monkeypatch, items)
    llm = FakeLLMProvider(answer="Claim [1] and invented [7].")
    result = answer_question(None, None, None, llm, "Q?")  # type: ignore[arg-type]
    assert [c.citation_id for c in result.citations] == [1]
    assert result.invalid_citations == [InvalidCitationReference(citation_id=7)]
    assert "[7]" in result.answer  # answer preserved, not rewritten


def test_no_evidence_has_empty_citations(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(generation_mod, "retrieve_hybrid", lambda *a, **k: [])
    monkeypatch.setattr(generation_mod, "rerank_results", lambda *a, **k: [])
    llm = FakeLLMProvider()
    result = answer_question(None, None, None, llm, "Q?")  # type: ignore[arg-type]
    assert result.answer == NO_EVIDENCE_MESSAGE
    assert result.citations == [] and result.invalid_citations == []
    assert llm.calls == []
