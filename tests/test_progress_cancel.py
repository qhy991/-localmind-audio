"""task13: long multi-chunk progress, mid-run cancellation, and cleanup."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from localmind.bench.fixtures import generate_synthetic_wav
from localmind.cli import main
from localmind.stt import MockTranscriber
from localmind.summary import MockSummaryLLM


def _wav(tmp_path: Path, duration=8.0) -> Path:
    return generate_synthetic_wav(tmp_path / "long.wav", duration_sec=duration, seed=1)


def _run(argv, tmp_path):
    out, err = io.StringIO(), io.StringIO()
    rc = main(argv, out, err)
    return rc, out.getvalue(), err.getvalue()


def _progress_lines(err_text):
    return [json.loads(ln) for ln in err_text.strip().split("\n") if ln]


def test_long_multichunk_emits_incremental_stt_and_summarize_progress(tmp_path):
    wav = _wav(tmp_path, duration=8.0)
    rc, out, err = _run(
        ["analyze", str(wav), "--mock", "--model-dir", str(tmp_path / "models"),
         "--chunk-sec", "2.0", "--overlap-sec", "0.2"],
        tmp_path,
    )
    assert rc == 0
    events = _progress_lines(err)
    stt = [e for e in events if e["stage"] == "stt"]
    summ = [e for e in events if e["stage"] == "summarize"]
    # Multiple chunks -> multiple incremental STT progress events.
    assert len(stt) >= 2
    fractions = [e["fraction"] for e in stt]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    # Summarize progress (start + end).
    assert len(summ) >= 1


def test_cancel_during_stt_returns_cancelled_and_no_store_run(tmp_path, monkeypatch):
    wav = _wav(tmp_path, duration=4.0)
    db = tmp_path / "s.db"

    def _cancel(self, source, config, provisioner, tier_model_id, on_progress=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(MockTranscriber, "transcribe", _cancel)
    rc, out, _ = _run(
        ["analyze", str(wav), "--mock", "--store", str(db),
         "--model-dir", str(tmp_path / "models")],
        tmp_path,
    )
    assert rc == 130
    data = json.loads(out)
    assert data["error"]["code"] == "cancelled"
    # Persistence happens after summarize; cancellation during STT never opens
    # the store, so no db file is created.
    assert not db.exists()


def test_cancel_during_summarize_returns_cancelled_and_no_store_run(tmp_path, monkeypatch):
    wav = _wav(tmp_path, duration=4.0)
    db = tmp_path / "s.db"

    def _cancel(self, prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr(MockSummaryLLM, "generate", _cancel)
    rc, out, _ = _run(
        ["analyze", str(wav), "--mock", "--store", str(db),
         "--model-dir", str(tmp_path / "models")],
        tmp_path,
    )
    assert rc == 130
    data = json.loads(out)
    assert data["error"]["code"] == "cancelled"
    assert not db.exists()


def test_no_temp_files_left_behind_without_store(tmp_path):
    """The pipeline is fileless until storage commit: no temp/staging files."""
    wav = _wav(tmp_path, duration=3.0)
    rc, out, _ = _run(
        ["analyze", str(wav), "--mock", "--model-dir", str(tmp_path / "models")],
        tmp_path,
    )
    assert rc == 0
    # Only the input wav should exist in tmp_path.
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["long.wav"]


def test_no_temp_files_with_store_only_creates_db(tmp_path):
    wav = _wav(tmp_path, duration=3.0)
    db = tmp_path / "s.db"
    rc, out, _ = _run(
        ["analyze", str(wav), "--mock", "--store", str(db),
         "--model-dir", str(tmp_path / "models")],
        tmp_path,
    )
    assert rc == 0
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["long.wav", "s.db"]
