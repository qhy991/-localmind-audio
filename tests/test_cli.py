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


# --------------------------------------------------------------------------- #
# benchmark: decode timing includes source construction                       #
# --------------------------------------------------------------------------- #

def test_benchmark_decode_timing_includes_source_construction(tmp_path, monkeypatch):
    """The decode stage must include source setup (open/probe), so a measurable
    delay injected into source construction shows up in the reported decode time."""
    import time as _time
    import localmind.cli as cli_mod

    real_open = cli_mod.audio_source_from_path

    def slow_open(path, target_sample_rate=16000):
        _time.sleep(0.1)
        return real_open(path, target_sample_rate)

    monkeypatch.setattr(cli_mod, "audio_source_from_path", slow_open)
    wav = _wav(tmp_path)
    rc, out, _ = _run(
        ["benchmark", str(wav), "--mock", "--model-dir", str(tmp_path / "models"),
         "--chunk-sec", "1", "--overlap-sec", "0.1"],
        tmp_path,
    )
    assert rc == 0
    report = json.loads(out)
    stages = {s["stage"]: s["duration_sec"] for s in report["stages"]}
    assert stages["decode"] >= 0.1  # the injected source-construction delay


# --------------------------------------------------------------------------- #
# summarize + analyze (structured-summary via CLI)                            #
# --------------------------------------------------------------------------- #

def _transcript_json(tmp_path, segments):
    p = tmp_path / "transcript.json"
    p.write_text(json.dumps({"segments": segments}))
    return p


def test_summarize_mock_emits_versioned_summary(tmp_path):
    from localmind.summary import SUMMARY_SCHEMA_VERSION
    segs = [{"id": "seg-0000", "start": 0.0, "end": 1.0, "text": "hello"},
            {"id": "seg-0001", "start": 1.0, "end": 2.0, "text": "world"}]
    tjson = _transcript_json(tmp_path, segs)
    rc, out, _ = _run(["summarize", str(tjson), "--mock"], tmp_path)
    assert rc == 0
    summary = json.loads(out)
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["case_id"] == "transcript"
    assert isinstance(summary["decisions"], list) and len(summary["decisions"]) >= 1
    # Citations reference real segment ids.
    cited = summary["decisions"][0]["citations"][0]
    assert cited.endswith("seg-0000") or cited.endswith("seg-0001")


def test_analyze_mock_emits_combined_json(tmp_path):
    from localmind.summary import SUMMARY_SCHEMA_VERSION
    wav = _wav(tmp_path)
    rc, out, err = _run(
        ["analyze", str(wav), "--mock", "--model-dir", str(tmp_path / "models"),
         "--chunk-sec", "1", "--overlap-sec", "0.1"],
        tmp_path,
    )
    assert rc == 0
    data = json.loads(out)
    assert data["schema_version"] == CLI_OUTPUT_SCHEMA_VERSION
    assert data["command"] == "analyze"
    assert len(data["transcript"]["segments"]) >= 1
    assert data["summary"]["schema_version"] == SUMMARY_SCHEMA_VERSION
    # Progress covers both stages.
    stages_seen = {json.loads(ln)["stage"] for ln in err.strip().split("\n") if ln}
    assert "stt" in stages_seen and "summarize" in stages_seen


def test_summarize_rejects_missing_transcript(tmp_path):
    rc, out, _ = _run(["summarize", str(tmp_path / "nope.json"), "--mock"], tmp_path)
    assert rc == 1
    data = json.loads(out)
    assert data["error"]["code"] == "cli_error"


def test_cli_summary_schema_version_pinned():
    from localmind.summary import SUMMARY_SCHEMA_VERSION
    assert SUMMARY_SCHEMA_VERSION == "soundmind.summary.v1"

