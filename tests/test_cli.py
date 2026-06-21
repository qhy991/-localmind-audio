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


# --------------------------------------------------------------------------- #
# summarize non-mock: --model-dir + fake mlx_lm                               #
# --------------------------------------------------------------------------- #

class _FakeMlxLmForCli:
    """Fake mlx_lm that returns valid sections JSON citing seg-0000."""
    def load(self, path):
        return ("model", "tokenizer")

    def generate(self, model, tokenizer, prompt=None, max_tokens=1024):
        return json.dumps({
            "decisions": [{"text": "a decision", "citations": ["seg:seg-0000"]}],
            "action_items": [{"text": "do it", "owner": None, "due_date": None, "citations": ["seg:seg-0000"]}],
            "open_questions": [],
        })


def _llm_manifest(tmp_path, model_id="qwen-7b"):
    import hashlib
    model_dir = tmp_path / "models"
    content = b"llm-weights" * 50
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
    return model_dir


def test_summarize_nonmock_with_valid_llm_manifest(tmp_path, monkeypatch):
    """Non-mock summarize with a valid LLM manifest and fake mlx_lm returns
    structured JSON — not an AttributeError crash."""
    monkeypatch.setitem(__import__("sys").modules, "mlx_lm", _FakeMlxLmForCli())
    segs = [{"id": "seg-0000", "start": 0.0, "end": 1.0, "text": "hello"}]
    tjson = _transcript_json(tmp_path, segs)
    _llm_manifest(tmp_path, "qwen-7b")
    rc, out, _ = _run(
        ["summarize", str(tjson), "--model-dir", str(tmp_path / "models"), "--llm-tier", "qwen-7b"],
        tmp_path,
    )
    assert rc == 0
    summary = json.loads(out)
    assert summary["schema_version"] == "soundmind.summary.v1"
    assert len(summary["decisions"]) >= 1


def test_summarize_nonmock_missing_manifest(tmp_path, monkeypatch):
    """Non-mock summarize with no manifest returns a structured error — not
    an uncaught AttributeError."""
    monkeypatch.setitem(__import__("sys").modules, "mlx_lm", _FakeMlxLmForCli())
    segs = [{"id": "seg-0000", "start": 0.0, "end": 1.0, "text": "hello"}]
    tjson = _transcript_json(tmp_path, segs)
    rc, out, _ = _run(
        ["summarize", str(tjson), "--model-dir", str(tmp_path / "no-models"), "--llm-tier", "qwen-7b"],
        tmp_path,
    )
    assert rc == 1
    data = json.loads(out)
    assert "error" in data
    assert data["error"]["code"] in {"provisioning_error", "cli_error"}


# --------------------------------------------------------------------------- #
# CLI repair / failure contract (structured-summary via CLI)                  #
# --------------------------------------------------------------------------- #

def _patch_llm(monkeypatch, mode):
    from localmind.summary import MockSummaryLLM
    import localmind.cli as cli_mod
    monkeypatch.setattr(cli_mod, "_select_summary_llm", lambda args: MockSummaryLLM(mode))


def test_summarize_repair_contract(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "invalid_then_valid")
    segs = [{"id": "seg-0000", "start": 0.0, "end": 1.0, "text": "hello"},
            {"id": "seg-0001", "start": 1.0, "end": 2.0, "text": "world"}]
    tjson = _transcript_json(tmp_path, segs)
    rc, out, _ = _run(["summarize", str(tjson), "--mock"], tmp_path)
    assert rc == 0
    summary = json.loads(out)
    assert summary["schema_version"] == "soundmind.summary.v1"
    prov = summary["provenance"]
    assert prov["repaired"] is True
    assert prov["repair_attempts_used"] == 1
    assert len(prov["initial_validation_errors"]) >= 1


def test_summarize_failure_contract(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "always_invalid")
    segs = [{"id": "seg-0000", "start": 0.0, "end": 1.0, "text": "hello"}]
    tjson = _transcript_json(tmp_path, segs)
    rc, out, _ = _run(["summarize", str(tjson), "--mock"], tmp_path)
    assert rc == 0
    failed = json.loads(out)
    assert failed["status"] == "failed"
    assert isinstance(failed["raw_output"], str) and failed["raw_output"]
    assert len(failed["errors"]) >= 1
    assert failed["provenance"]["repaired"] is False


def test_analyze_repair_contract(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "invalid_then_valid")
    wav = _wav(tmp_path)
    rc, out, _ = _run(
        ["analyze", str(wav), "--mock", "--model-dir", str(tmp_path / "models"),
         "--chunk-sec", "1", "--overlap-sec", "0.1"],
        tmp_path,
    )
    assert rc == 0
    data = json.loads(out)
    prov = data["summary"]["provenance"]
    assert prov["repaired"] is True
    assert prov["repair_attempts_used"] == 1


def test_analyze_failure_contract(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "always_invalid")
    wav = _wav(tmp_path)
    rc, out, _ = _run(
        ["analyze", str(wav), "--mock", "--model-dir", str(tmp_path / "models"),
         "--chunk-sec", "1", "--overlap-sec", "0.1"],
        tmp_path,
    )
    assert rc == 0
    data = json.loads(out)
    assert data["summary"]["status"] == "failed"
    assert data["summary"]["provenance"]["repaired"] is False
    assert len(data["summary"]["errors"]) >= 1


def test_analyze_persists_to_store(tmp_path):
    wav = _wav(tmp_path)
    db = tmp_path / "s.db"
    rc, out, _ = _run(
        ["analyze", str(wav), "--mock", "--store", str(db),
         "--model-dir", str(tmp_path / "models"), "--chunk-sec", "1", "--overlap-sec", "0.1"],
        tmp_path,
    )
    assert rc == 0
    data = json.loads(out)
    run_id = data["store_run_id"]
    assert run_id

    from localmind.store import Store
    with Store(db) as store:
        run = store.get_run(run_id)
    assert run["audio_asset"]["duration_sec"] == pytest.approx(2.0)
    assert len(run["segments"]) >= 1
    assert run["summary"] is not None
    assert run["inference_run"]["status"] == "ok"
    # Stage metrics + STT/LLM model refs are persisted with complete provenance.
    assert run["inference_run"]["metrics_json"] is not None
    metrics = json.loads(run["inference_run"]["metrics_json"])
    stage_names = {s["stage"] for s in metrics["stages"]}
    assert stage_names == {"decode", "stt", "llm", "persist"}
    ref_kinds = {r["kind"] for r in run["model_manifest_refs"]}
    assert ref_kinds == {"whisper", "llm"}


def test_analyze_persists_failed_summary_to_store(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "always_invalid")
    wav = _wav(tmp_path)
    db = tmp_path / "s.db"
    rc, out, _ = _run(
        ["analyze", str(wav), "--mock", "--store", str(db),
         "--model-dir", str(tmp_path / "models"), "--chunk-sec", "1", "--overlap-sec", "0.1"],
        tmp_path,
    )
    assert rc == 0
    data = json.loads(out)
    run_id = data["store_run_id"]
    from localmind.store import Store
    with Store(db) as store:
        run = store.get_run(run_id)
    assert run["inference_run"]["status"] == "failed"
    assert run["summary"]["status"] == "failed"
    assert run["summary"]["raw_output"]
    # Even a failed-summary run has non-null metrics (persist measured atomically).
    assert run["inference_run"]["metrics_json"] is not None


def test_analyze_store_metrics_match_stdout(tmp_path):
    """Stored metrics_json must include the measured persist stage and match the
    stdout metrics (total/rtf within tolerance)."""
    wav = _wav(tmp_path)
    db = tmp_path / "s.db"
    rc, out, _ = _run(
        ["analyze", str(wav), "--mock", "--store", str(db),
         "--model-dir", str(tmp_path / "models"), "--chunk-sec", "1", "--overlap-sec", "0.1"],
        tmp_path,
    )
    assert rc == 0
    data = json.loads(out)
    stdout_metrics = data["metrics"]

    from localmind.store import Store
    with Store(db) as store:
        run = store.get_run(data["store_run_id"])
    stored_metrics = json.loads(run["inference_run"]["metrics_json"])

    # Persist stage is measured (not the stale 0.0).
    persist = {s["stage"]: s["duration_sec"] for s in stored_metrics["stages"]}["persist"]
    assert persist > 0.0
    # Stored totals match stdout within a small tolerance.
    assert stored_metrics["total_duration_sec"] == pytest.approx(
        stdout_metrics["total_duration_sec"], rel=1e-3, abs=1e-3
    )
    assert {s["stage"] for s in stored_metrics["stages"]} == {"decode", "stt", "llm", "persist"}


