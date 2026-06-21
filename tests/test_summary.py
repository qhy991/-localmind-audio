"""Tests for the structured-summary schema and the map-reduce summarizer."""

from __future__ import annotations

import json

import pytest

from localmind.stt.segment import TranscriptSegment
from localmind.summary import (
    SUMMARY_SCHEMA_VERSION,
    MockSummaryLLM,
    Summarizer,
    SummaryValidationError,
    build_summary_failed,
    validate_summary_against_transcript,
    validate_summary_dict,
    validate_summary_failed_dict,
)
from localmind.summary.summarizer import _chunk_segments_by_chars


def _seg(i, text="hello world"):
    return TranscriptSegment(id=f"seg-{i:04d}", start=float(i), end=float(i + 1), text=text)


def _good_summary():
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "case_id": "case-1",
        "provenance": {"model_id": "m", "prompt_template_hash": "sha256:abcd"},
        "decisions": [{"text": "a decision", "citations": ["seg:seg-0000"]}],
        "action_items": [{
            "text": "do it", "owner": "Maya", "due_date": None,
            "citations": ["seg:seg-0001"],
        }],
        "open_questions": [{"text": "why?", "citations": ["ts:0.0-1.0"]}],
    }


# --------------------------------------------------------------------------- #
# Schema validation                                                           #
# --------------------------------------------------------------------------- #

def test_valid_summary_passes():
    data = _good_summary()
    validate_summary_dict(data)
    validate_summary_against_transcript(data, ["seg-0000", "seg-0001"])


def test_missing_citation_rejected():
    data = _good_summary()
    data["decisions"][0]["citations"] = []
    with pytest.raises(SummaryValidationError, match="citations"):
        validate_summary_dict(data)


def test_bad_due_date_rejected():
    data = _good_summary()
    data["action_items"][0]["due_date"] = "06/28/2026"
    with pytest.raises(SummaryValidationError, match="due_date"):
        validate_summary_dict(data)


def test_unknown_segment_citation_rejected():
    data = _good_summary()
    data["decisions"][0]["citations"] = ["seg:seg-9999"]
    validate_summary_dict(data)  # shape is fine
    with pytest.raises(SummaryValidationError, match="unknown segment"):
        validate_summary_against_transcript(data, ["seg-0000", "seg-0001"])


def test_bad_schema_version_rejected():
    data = _good_summary()
    data["schema_version"] = "wrong"
    with pytest.raises(SummaryValidationError, match="schema_version"):
        validate_summary_dict(data)


def test_summary_failed_roundtrip():
    failed = build_summary_failed(
        "not json", ["output is not valid JSON: ..."],
        case_id="case-1", model_id="m", prompt_template_hash="sha256:abcd",
    )
    validate_summary_failed_dict(failed)
    assert failed["status"] == "failed"
    assert failed["raw_output"] == "not json"
    assert len(failed["errors"]) >= 1


# --------------------------------------------------------------------------- #
# Summarizer: map-reduce, repair, summary_failed                              #
# --------------------------------------------------------------------------- #

def test_summarizer_valid_output():
    segs = [_seg(0), _seg(1), _seg(2)]
    s = Summarizer(MockSummaryLLM("valid"), model_id="mock")
    summary = s.summarize(segs, case_id="case-1")
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["case_id"] == "case-1"
    validate_summary_dict(summary)
    validate_summary_against_transcript(summary, [x.id for x in segs])
    assert len(summary["decisions"]) >= 1


def test_summarizer_repair_succeeds_after_invalid_first_output():
    segs = [_seg(0), _seg(1)]
    llm = MockSummaryLLM("invalid_then_valid")
    s = Summarizer(llm, model_id="mock", max_repair_attempts=1)
    summary = s.summarize(segs, case_id="case-1")
    # First call invalid, second (repair) valid -> 2 calls for the single chunk.
    assert llm.call_count == 2
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    validate_summary_against_transcript(summary, [x.id for x in segs])


def test_summarizer_repair_exhausted_returns_summary_failed():
    segs = [_seg(0), _seg(1)]
    llm = MockSummaryLLM("always_invalid")
    s = Summarizer(llm, model_id="mock", max_repair_attempts=1)
    summary = s.summarize(segs, case_id="case-1")
    validate_summary_failed_dict(summary)
    assert summary["status"] == "failed"
    assert isinstance(summary["raw_output"], str) and summary["raw_output"]
    assert len(summary["errors"]) >= 1
    # 1 initial + 1 repair attempt for the single chunk.
    assert llm.call_count == 2


def test_summarizer_chunks_long_transcript():
    segs = [_seg(i, text="word " * 20) for i in range(10)]  # 100 chars each
    chunks = _chunk_segments_by_chars(segments=segs, max_chars=400)
    assert len(chunks) > 1
    # With sub-budget segments, every chunk fits the budget.
    for ch in chunks:
        assert sum(len(s.text) for s in ch) <= 400

    llm = MockSummaryLLM("valid")
    s = Summarizer(llm, model_id="mock", max_chars_per_chunk=400)
    summary = s.summarize(segs, case_id="case-long")
    validate_summary_dict(summary)
    validate_summary_against_transcript(summary, [x.id for x in segs])
    # Multiple chunks -> multiple LLM calls.
    assert llm.call_count == len(chunks)


def test_summarizer_zero_repair_attempts_fails_on_invalid():
    segs = [_seg(0)]
    llm = MockSummaryLLM("always_invalid")
    s = Summarizer(llm, model_id="mock", max_repair_attempts=0)
    summary = s.summarize(segs, case_id="case-1")
    validate_summary_failed_dict(summary)
    assert llm.call_count == 1  # no repair attempted
