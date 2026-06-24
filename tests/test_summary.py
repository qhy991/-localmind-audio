"""Tests for the structured-summary schema and the map-reduce summarizer."""

from __future__ import annotations

import json
import socket
import sys

import pytest

from localmind.provisioning import Provisioner
from localmind.provisioning.errors import ModelNotProvisionedError
from localmind.stt.segment import TranscriptSegment
from localmind.summary import (
    SUMMARY_SCHEMA_VERSION,
    MlxLmSummaryLLM,
    MockSummaryLLM,
    Summarizer,
    SummaryValidationError,
    build_summary_failed,
    validate_summary_against_transcript,
    validate_summary_dict,
    validate_summary_failed_dict,
)
from localmind.summary.summarizer import _chunk_segments_by_chars


def _seg(i, start=None, end=None, text="hello world"):
    return TranscriptSegment(
        id=f"seg-{i:04d}",
        start=float(i if start is None else start),
        end=float(i + 1 if end is None else end),
        text=text,
    )


def _good_summary():
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "case_id": "case-1",
        "provenance": {
            "model_id": "m", "prompt_template_hash": "sha256:abcd",
            "repaired": False, "repair_attempts_used": 0, "initial_validation_errors": [],
        },
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
    segs = [_seg(0, 0.0, 1.0), _seg(1, 1.0, 2.0)]
    validate_summary_dict(data)
    validate_summary_against_transcript(data, segs)


def test_action_item_missing_owner_rejected():
    data = _good_summary()
    data["action_items"][0] = {"text": "do it", "due_date": None, "citations": ["seg:seg-0001"]}
    with pytest.raises(SummaryValidationError, match="owner is required"):
        validate_summary_dict(data)


def test_action_item_missing_due_date_rejected():
    data = _good_summary()
    data["action_items"][0] = {"text": "do it", "owner": "Maya", "citations": ["seg:seg-0001"]}
    with pytest.raises(SummaryValidationError, match="due_date is required"):
        validate_summary_dict(data)


def test_action_item_empty_string_owner_rejected():
    data = _good_summary()
    data["action_items"][0]["owner"] = "   "
    with pytest.raises(SummaryValidationError, match="owner"):
        validate_summary_dict(data)


def test_action_item_null_owner_and_due_date_accepted():
    data = _good_summary()
    data["action_items"][0]["owner"] = None
    data["action_items"][0]["due_date"] = None
    validate_summary_dict(data)


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
    validate_summary_dict(data)
    with pytest.raises(SummaryValidationError, match="unknown segment"):
        validate_summary_against_transcript(data, [_seg(0, 0.0, 1.0), _seg(1, 1.0, 2.0)])


def test_timestamp_citation_out_of_range_rejected():
    data = _good_summary()
    data["decisions"][0]["citations"] = ["ts:999-1000"]
    validate_summary_dict(data)
    with pytest.raises(SummaryValidationError, match="does not overlap"):
        validate_summary_against_transcript(data, [_seg(0, 0.0, 1.0), _seg(1, 1.0, 2.0)])


def test_timestamp_citation_reversed_rejected():
    data = _good_summary()
    data["decisions"][0]["citations"] = ["ts:5-1"]
    validate_summary_dict(data)
    with pytest.raises(SummaryValidationError, match="start < end"):
        validate_summary_against_transcript(data, [_seg(0, 0.0, 1.0)])


def test_timestamp_citation_no_coverage_rejected():
    data = _good_summary()
    data["decisions"][0]["citations"] = ["ts:100-200"]
    validate_summary_dict(data)
    with pytest.raises(SummaryValidationError, match="does not overlap"):
        validate_summary_against_transcript(data, [_seg(0, 0.0, 2.0)])


def test_timestamp_citation_overlapping_segment_accepted():
    data = _good_summary()
    data["decisions"][0]["citations"] = ["ts:0.5-1.5"]
    segs = [_seg(0, 0.0, 1.0), _seg(1, 1.0, 2.0)]
    validate_summary_against_transcript(data, segs)  # overlaps both segments


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
# Summarizer: map-reduce, repair, provenance, summary_failed                  #
# --------------------------------------------------------------------------- #

def test_summarizer_valid_runs_map_and_reduce():
    segs = [_seg(0), _seg(1), _seg(2)]
    llm = MockSummaryLLM("valid")
    s = Summarizer(llm, model_id="mock")
    summary = s.summarize(segs, case_id="case-1")
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    validate_summary_dict(summary)
    validate_summary_against_transcript(summary, segs)
    # One map call (single chunk) + one reduce call.
    assert llm.map_calls == 1
    assert llm.reduce_calls == 1
    assert summary["provenance"]["repaired"] is False
    assert summary["provenance"]["repair_attempts_used"] == 0


def test_summarizer_repair_recorded_in_provenance():
    segs = [_seg(0), _seg(1)]
    llm = MockSummaryLLM("invalid_then_valid")
    s = Summarizer(llm, model_id="mock", max_repair_attempts=1)
    summary = s.summarize(segs, case_id="case-1")
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    validate_summary_against_transcript(summary, segs)
    prov = summary["provenance"]
    assert prov["repaired"] is True
    assert prov["repair_attempted"] is True
    assert prov["repair_attempts_used"] == 1
    assert len(prov["initial_validation_errors"]) >= 1


def test_summarizer_repair_exhausted_returns_summary_failed():
    segs = [_seg(0), _seg(1)]
    llm = MockSummaryLLM("always_invalid")
    s = Summarizer(llm, model_id="mock", max_repair_attempts=1)
    summary = s.summarize(segs, case_id="case-1")
    validate_summary_failed_dict(summary)
    assert summary["status"] == "failed"
    assert isinstance(summary["raw_output"], str) and summary["raw_output"]
    assert len(summary["errors"]) >= 1
    # Exhausted repair is NOT repaired; it was attempted.
    prov = summary["provenance"]
    assert prov["repaired"] is False
    assert prov["repair_attempted"] is True
    assert prov["repair_attempts_used"] == 1
    # 1 initial map call + 1 map repair; reduce never reached.
    assert llm.map_calls == 2
    assert llm.reduce_calls == 0


def test_summarizer_multi_chunk_runs_reduce_after_map():
    segs = [_seg(i, text="word " * 20) for i in range(10)]  # 100 chars each
    chunks = _chunk_segments_by_chars(segments=segs, max_chars=400)
    assert len(chunks) > 1

    llm = MockSummaryLLM("valid")
    s = Summarizer(llm, model_id="mock", max_chars_per_chunk=400)
    summary = s.summarize(segs, case_id="case-long")
    validate_summary_dict(summary)
    validate_summary_against_transcript(summary, segs)
    # Map ran once per chunk, then exactly one reduce call.
    assert llm.map_calls == len(chunks)
    assert llm.reduce_calls == 1


def test_summarizer_zero_repair_attempts_fails_on_invalid():
    segs = [_seg(0)]
    llm = MockSummaryLLM("always_invalid")
    s = Summarizer(llm, model_id="mock", max_repair_attempts=0)
    summary = s.summarize(segs, case_id="case-1")
    validate_summary_failed_dict(summary)
    assert llm.map_calls == 1  # no repair attempted
    assert llm.reduce_calls == 0


# --------------------------------------------------------------------------- #
# MlxLmSummaryLLM: real adapter with fake mlx_lm                              #
# --------------------------------------------------------------------------- #

def _prov_with_llm_model(tmp_path, model_id="qwen-7b"):
    import hashlib
    model_dir = tmp_path / "models"
    content = b"llm-weights" * 100
    p = model_dir / f"{model_id}.gguf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    (model_dir / "models.json").write_text(json.dumps({
        "schema_version": "1",
        "models": [{
            "model_id": model_id, "name": model_id, "kind": "llm",
            "path": f"{model_id}.gguf", "quant_format": "int4",
            "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "license": "Apache-2.0",
        }],
    }))
    return Provisioner(model_dir)


class _FakeMlxLm:
    """Fake mlx_lm module: load returns dummy objects, generate returns text."""

    def __init__(self, response='{"decisions":[],"action_items":[],"open_questions":[]}'):
        self.response = response
        self.load_calls = []
        self.generate_calls = []

    def load(self, path):
        self.load_calls.append(path)
        return ("fake_model", "fake_tokenizer")

    def generate(self, model, tokenizer, prompt=None, max_tokens=1024):
        self.generate_calls.append(prompt)
        return self.response


def test_mlxlm_llm_generates_with_verified_path(tmp_path, monkeypatch):
    prov = _prov_with_llm_model(tmp_path)
    fake = _FakeMlxLm("a summary text")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    llm = MlxLmSummaryLLM(prov, "qwen-7b")
    result = llm.generate("summarize this")

    assert result == "a summary text"
    assert len(fake.load_calls) == 1
    # mlx_lm.load receives the model DIRECTORY (containing config + weights),
    # not the verified weights file itself.
    assert str(tmp_path) in fake.load_calls[0]
    assert "qwen-7b.gguf" not in fake.load_calls[0]
    assert llm.last_provenance is not None
    assert llm.last_provenance.model_id == "qwen-7b"
    assert len(llm.last_provenance.sha256) == 64


def test_mlxlm_llm_rejects_non_provisioner():
    llm = MlxLmSummaryLLM("not-a-provisioner", "qwen-7b")
    with pytest.raises(TypeError, match="Provisioner"):
        llm.generate("prompt")


def test_mlxlm_llm_raises_without_mlx_lm(tmp_path, monkeypatch):
    prov = _prov_with_llm_model(tmp_path)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    llm = MlxLmSummaryLLM(prov, "qwen-7b")
    with pytest.raises(RuntimeError, match="mlx-lm"):
        llm.generate("prompt")


def test_mlxlm_llm_does_not_touch_network(tmp_path, monkeypatch):
    prov = _prov_with_llm_model(tmp_path)
    fake = _FakeMlxLm("response")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    def _no_network(*_a, **_k):
        raise AssertionError("LLM adapter attempted a network connection")

    monkeypatch.setattr(socket, "socket", _no_network)
    llm = MlxLmSummaryLLM(prov, "qwen-7b")
    llm.generate("prompt")  # would raise via _no_network if network path were taken


def test_mlxlm_llm_rejects_non_llm_kind(tmp_path, monkeypatch):
    """A manifest entry with kind='whisper' cannot be loaded as an LLM."""
    import hashlib
    model_dir = tmp_path / "models"
    content = b"whisper-weights" * 100
    p = model_dir / "whisper-small.mlmodel"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    (model_dir / "models.json").write_text(json.dumps({
        "schema_version": "1",
        "models": [{
            "model_id": "whisper-small", "name": "w", "kind": "whisper",
            "path": "whisper-small.mlmodel", "quant_format": "int4",
            "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "license": "MIT",
        }],
    }))
    prov = Provisioner(model_dir)
    fake = _FakeMlxLm("should not be called")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    llm = MlxLmSummaryLLM(prov, "whisper-small")
    with pytest.raises(ModelNotProvisionedError, match="kind"):
        llm.generate("prompt")
    assert len(fake.load_calls) == 0  # mlx_lm.load never called for non-LLM kind


def test_mlxlm_llm_wrong_kind_missing_file(tmp_path, monkeypatch):
    """Kind check happens BEFORE weight hashing: wrong kind + missing file -> kind error."""
    import hashlib
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Manifest declares a whisper-kind entry; the weight file does NOT exist.
    (model_dir / "models.json").write_text(json.dumps({
        "schema_version": "1",
        "models": [{
            "model_id": "whisper-small", "name": "w", "kind": "whisper",
            "path": "missing.mlmodel", "quant_format": "int4",
            "size_bytes": 100, "sha256": hashlib.sha256(b"x").hexdigest(),
            "license": "MIT",
        }],
    }))
    prov = Provisioner(model_dir)
    fake = _FakeMlxLm("should not be called")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    llm = MlxLmSummaryLLM(prov, "whisper-small")
    with pytest.raises(ModelNotProvisionedError, match="kind"):
        llm.generate("prompt")
    assert len(fake.load_calls) == 0


def test_mlxlm_llm_wrong_kind_tampered_file(tmp_path, monkeypatch):
    """Kind check happens BEFORE SHA verification: wrong kind + tampered file -> kind error."""
    import hashlib
    model_dir = tmp_path / "models"
    original = b"whisper-weights" * 100
    tampered = b"T" * len(original)  # same length, different sha
    p = model_dir / "whisper-small.mlmodel"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(tampered)
    (model_dir / "models.json").write_text(json.dumps({
        "schema_version": "1",
        "models": [{
            "model_id": "whisper-small", "name": "w", "kind": "whisper",
            "path": "whisper-small.mlmodel", "quant_format": "int4",
            "size_bytes": len(original), "sha256": hashlib.sha256(original).hexdigest(),
            "license": "MIT",
        }],
    }))
    prov = Provisioner(model_dir)
    fake = _FakeMlxLm("should not be called")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    llm = MlxLmSummaryLLM(prov, "whisper-small")
    with pytest.raises(ModelNotProvisionedError, match="kind"):
        llm.generate("prompt")
    assert len(fake.load_calls) == 0


def test_fake_llm_backend_skips_mlx_preflight(tmp_path, monkeypatch):
    """Host-independent proof: when a fake mlx_lm is injected into sys.modules,
    MlxLmSummaryLLM.generate must NOT call ensure_mlx_metal_available."""
    def _preflight_must_not_run():
        raise AssertionError("ensure_mlx_metal_available called when fake backend is injected")

    monkeypatch.setattr(
        "localmind.mlx_runtime.ensure_mlx_metal_available", _preflight_must_not_run
    )
    prov = _prov_with_llm_model(tmp_path)
    fake = _FakeMlxLm("a response")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    llm = MlxLmSummaryLLM(prov, "qwen-7b")
    result = llm.generate("summarize this")
    assert result == "a response"
    assert llm.last_provenance is not None
    # preflight was NOT called (no AssertionError)
