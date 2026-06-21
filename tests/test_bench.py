"""Tests for benchmark fixture generation and report schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import wave

from localmind.bench import BENCHMARK_CASES, BenchmarkReport
from localmind.bench.fixtures import (
    BenchmarkCase,
    FixtureNotProvisionedError,
    generate_synthetic_wav,
    is_fixture_provisioned,
    require_fixture,
)
from localmind.bench.report import (
    REPORT_SCHEMA_VERSION,
    PeakMemory,
    StageTiming,
    validate_report_dict,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #

def test_benchmark_cases_cover_10_30_60_minutes():
    durations = sorted(c.duration_min for c in BENCHMARK_CASES)
    assert durations == [10, 30, 60]
    for c in BENCHMARK_CASES:
        assert c.case_id
        assert c.audio_rel_path
        assert c.sample_rate == 16000


def test_generate_synthetic_wav_has_expected_duration_and_shape(tmp_path):
    wav = generate_synthetic_wav(tmp_path / "syn.wav", duration_sec=2.0, seed=42)
    assert wav.is_file()

    with wave.open(str(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 32000  # 2.0s at 16kHz


def test_generate_synthetic_wav_is_deterministic(tmp_path):
    a = generate_synthetic_wav(tmp_path / "a.wav", duration_sec=1.0, seed=7)
    b = generate_synthetic_wav(tmp_path / "b.wav", duration_sec=1.0, seed=7)
    assert a.read_bytes() == b.read_bytes()


def test_require_fixture_fails_clearly_when_missing(tmp_path):
    case = BENCHMARK_CASES[0]
    assert is_fixture_provisioned(tmp_path, case) is False
    with pytest.raises(FixtureNotProvisionedError, match=case.case_id):
        require_fixture(tmp_path, case)


def test_require_fixture_returns_path_when_present(tmp_path):
    case = BENCHMARK_CASES[0]
    p = tmp_path / case.audio_rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake audio bytes")
    assert is_fixture_provisioned(tmp_path, case) is True
    assert require_fixture(tmp_path, case) == p


# --------------------------------------------------------------------------- #
# Report schema                                                                 #
# --------------------------------------------------------------------------- #

def _good_report_dict():
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": "r1",
        "case_id": "bm-10min",
        "audio_duration_sec": 600.0,
        "stages": [
            {"stage": "decode", "duration_sec": 1.2},
            {"stage": "stt", "duration_sec": 40.0},
            {"stage": "llm", "duration_sec": 8.0},
            {"stage": "persist", "duration_sec": 0.1},
        ],
        "total_duration_sec": 49.3,
        "rtf": 0.082,
        "peak_memory": [
            {"bytes": 3_000_000_000, "domain": "cpu", "method": "mach_task_basic_info"},
            {"bytes": 2_000_000_000, "domain": "gpu", "method": "metal_allocated"},
        ],
        "model_tiers": {"stt": "whisper-small", "llm": "qwen2.5-7b"},
        "hardware": {"chip": "M3", "memory_gb": 16},
        "aspirational_targets": {"peak_mem_gb": 6.0, "rtf": 0.08},
        "notes": "",
    }


def test_valid_report_passes_validation():
    validate_report_dict(_good_report_dict())  # no exception


def test_report_missing_stage_is_rejected():
    data = _good_report_dict()
    data["stages"] = [s for s in data["stages"] if s["stage"] != "llm"]
    with pytest.raises(ValueError, match="missing stage"):
        validate_report_dict(data)


def test_report_single_end_to_end_time_without_breakdown_is_rejected():
    data = _good_report_dict()
    data["stages"] = [{"stage": "decode", "duration_sec": 49.3}]  # only one stage
    with pytest.raises(ValueError):
        validate_report_dict(data)


def test_report_peak_memory_without_method_is_rejected():
    data = _good_report_dict()
    data["peak_memory"] = [
        {"bytes": 3_000_000_000, "domain": "cpu"},  # no method
        {"bytes": 2_000_000_000, "domain": "gpu", "method": "metal_allocated"},
    ]
    with pytest.raises(ValueError, match="method"):
        validate_report_dict(data)


def test_report_peak_memory_missing_gpu_domain_is_rejected():
    data = _good_report_dict()
    data["peak_memory"] = [
        {"bytes": 3_000_000_000, "domain": "cpu", "method": "mach_task_basic_info"},
    ]
    with pytest.raises(ValueError, match="cpu.*gpu|both"):
        validate_report_dict(data)


def test_report_wrong_schema_version_rejected():
    data = _good_report_dict()
    data["schema_version"] = "999"
    with pytest.raises(ValueError, match="schema_version"):
        validate_report_dict(data)


def test_report_missing_total_duration_rejected():
    data = _good_report_dict()
    del data["total_duration_sec"]
    with pytest.raises(ValueError, match="total_duration_sec"):
        validate_report_dict(data)


def test_report_negative_total_duration_rejected():
    data = _good_report_dict()
    data["total_duration_sec"] = -1.0
    with pytest.raises(ValueError, match="total_duration_sec"):
        validate_report_dict(data)


def test_report_missing_aspirational_targets_rejected():
    data = _good_report_dict()
    del data["aspirational_targets"]
    with pytest.raises(ValueError, match="aspirational_targets"):
        validate_report_dict(data)


def test_report_aspirational_targets_missing_key_rejected():
    data = _good_report_dict()
    data["aspirational_targets"] = {"peak_mem_gb": 6.0}  # missing rtf
    with pytest.raises(ValueError, match="rtf"):
        validate_report_dict(data)


def test_report_aspirational_targets_non_numeric_rejected():
    data = _good_report_dict()
    data["aspirational_targets"] = {"peak_mem_gb": "6", "rtf": 0.08}
    with pytest.raises(ValueError, match="peak_mem_gb"):
        validate_report_dict(data)


def test_report_aspirational_targets_drift_rejected():
    """Schema v1 pins the plan's targets; a report claiming a different bar is rejected."""
    data = _good_report_dict()
    data["aspirational_targets"] = {"peak_mem_gb": 8.0, "rtf": 0.08}
    with pytest.raises(ValueError, match="peak_mem_gb"):
        validate_report_dict(data)


def test_benchmark_report_roundtrip(tmp_path):
    report = BenchmarkReport(
        schema_version=REPORT_SCHEMA_VERSION,
        run_id="r2",
        case_id="bm-30min",
        audio_duration_sec=1800.0,
        stages=[
            StageTiming("decode", 3.0),
            StageTiming("stt", 120.0),
            StageTiming("llm", 25.0),
            StageTiming("persist", 0.2),
        ],
        total_duration_sec=148.2,
        rtf=0.082,
        peak_memory=[
            PeakMemory(4_000_000_000, "cpu", "mach_task_basic_info"),
            PeakMemory(2_500_000_000, "gpu", "mlx_memory"),
        ],
    )
    text = report.to_json()
    validate_report_dict(json.loads(text))
