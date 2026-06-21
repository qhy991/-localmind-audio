"""Contract tests for the CLI: versioned JSON output, JSONL progress events,
benchmark report validation, cancellation, and error handling."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from localmind.bench.fixtures import generate_synthetic_wav
from localmind.bench.report import validate_report_dict
from localmind.cli import CLI_OUTPUT_SCHEMA_VERSION, main
from localmind.stt import MockTranscriber


def _wav(tmp_path: Path) -> Path:
    return generate_synthetic_wav(tmp_path / "tone.wav", duration_sec=2.0, seed=1)


def _run(argv, tmp_path):
    out, err = io.StringIO(), io.StringIO()
    rc = main(argv, out, err)
    return rc, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# transcribe: versioned JSON contract                                         #
# --------------------------------------------------------------------------- #

def test_transcribe_mock_emits_versioned_json(tmp_path):
    wav = _wav(tmp_path)
    rc, out, _ = _run(
        ["transcribe", str(wav), "--mock", "--model-dir", str(tmp_path / "models"),
         "--chunk-sec", "1", "--overlap-sec", "0.1"],
        tmp_path,
    )
    assert rc == 0
    data = json.loads(out)
    assert data["schema_version"] == CLI_OUTPUT_SCHEMA_VERSION
    assert data["command"] == "transcribe"
    assert data["audio"]["path"] == str(wav)
    assert data["audio"]["duration_sec"] == pytest.approx(2.0)
    assert data["model_tier"] == "mock"
    assert data["mock"] is True
    assert isinstance(data["segments"], list) and len(data["segments"]) >= 1
    seg = data["segments"][0]
    assert {"id", "start", "end", "text"} <= set(seg.keys())
    assert data["provenance"] is None  # mock transcriber records no provenance


def test_transcribe_emits_jsonl_progress_events(tmp_path):
    wav = _wav(tmp_path)
    rc, out, err = _run(
        ["transcribe", str(wav), "--mock", "--model-dir", str(tmp_path / "models"),
         "--chunk-sec", "1", "--overlap-sec", "0.1"],
        tmp_path,
    )
    assert rc == 0
    lines = [ln for ln in err.strip().split("\n") if ln]
    assert len(lines) >= 1
    for ln in lines:
        ev = json.loads(ln)
        assert ev["event"] == "progress"
        assert ev["stage"] == "stt"
        assert 0.0 <= ev["fraction"] <= 1.0


def test_transcribe_no_progress_flag_suppresses_events(tmp_path):
    wav = _wav(tmp_path)
    rc, out, err = _run(
        ["transcribe", str(wav), "--mock", "--no-progress", "--model-dir", str(tmp_path / "models")],
        tmp_path,
    )
    assert rc == 0
    assert err.strip() == ""


def test_cli_output_schema_version_is_pinned():
    """A contract guard: bumping the schema version is a deliberate breaking change."""
    assert CLI_OUTPUT_SCHEMA_VERSION == "1"


# --------------------------------------------------------------------------- #
# benchmark: validated report                                                 #
# --------------------------------------------------------------------------- #

def test_benchmark_emits_valid_report(tmp_path):
    wav = _wav(tmp_path)
    rc, out, _ = _run(
        ["benchmark", str(wav), "--mock", "--model-dir", str(tmp_path / "models"),
         "--chunk-sec", "1", "--overlap-sec", "0.1"],
        tmp_path,
    )
    assert rc == 0
    report = json.loads(out)
    validate_report_dict(report)  # raises if malformed
    stages = {s["stage"]: s["duration_sec"] for s in report["stages"]}
    assert set(stages.keys()) == {"decode", "stt", "llm", "persist"}
    domains = {m["domain"] for m in report["peak_memory"]}
    assert domains == {"cpu", "gpu"}
    assert report["audio_duration_sec"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# error handling + cancellation                                               #
# --------------------------------------------------------------------------- #

def test_transcribe_rejects_unsupported_format(tmp_path):
    flac = tmp_path / "tone.flac"
    flac.write_bytes(b"not audio")
    rc, out, _ = _run(
        ["transcribe", str(flac), "--mock", "--model-dir", str(tmp_path / "models")],
        tmp_path,
    )
    assert rc == 1
    data = json.loads(out)
    assert data["schema_version"] == CLI_OUTPUT_SCHEMA_VERSION
    assert data["error"]["code"] in {"cli_error"}


def test_transcribe_without_mock_errors_when_backend_unavailable(tmp_path):
    """Without --mock the real backend is required; with no mlx_whisper / no
    provisioned model the CLI must fail with a structured error (not a crash)."""
    wav = _wav(tmp_path)
    rc, out, _ = _run(
        ["transcribe", str(wav), "--model-dir", str(tmp_path / "no-such-models")],
        tmp_path,
    )
    assert rc == 1
    data = json.loads(out)
    assert "error" in data
    assert data["error"]["code"] in {"cli_error", "provisioning_error"}


def test_transcribe_cancelled_emits_cancelled_error(tmp_path, monkeypatch):
    wav = _wav(tmp_path)

    def _cancel(self, source, config, provisioner, tier_model_id, on_progress=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(MockTranscriber, "transcribe", _cancel)
    rc, out, _ = _run(
        ["transcribe", str(wav), "--mock", "--model-dir", str(tmp_path / "models")],
        tmp_path,
    )
    assert rc == 130
    data = json.loads(out)
    assert data["error"]["code"] == "cancelled"
