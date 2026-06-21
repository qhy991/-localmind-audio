"""Full-pipeline no-network harness around the analyze path.

The pipeline must run end-to-end with the network fully blocked (mock backends)
and the non-mock path must fail locally (provisioning/backend error) rather than
attempting a download.
"""

from __future__ import annotations

import io
import json
import socket
from pathlib import Path

import pytest

from localmind.bench.fixtures import generate_synthetic_wav
from localmind.cli import main


def _wav(tmp_path: Path) -> Path:
    return generate_synthetic_wav(tmp_path / "tone.wav", duration_sec=2.0, seed=1)


def _block_network(monkeypatch):
    """Make every common network entry point raise, so any egress attempt fails
    the test loudly instead of silently succeeding."""
    def _boom(*_a, **_k):
        raise AssertionError("network access attempted during a no-network run")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(socket, "getaddrinfo", _boom)


def test_analyze_mock_runs_with_network_blocked(tmp_path, monkeypatch):
    """The full analyze pipeline (mock STT + mock LLM + store) needs no network."""
    _block_network(monkeypatch)
    wav = _wav(tmp_path)
    db = tmp_path / "s.db"
    out, err = io.StringIO(), io.StringIO()
    rc = main(
        ["analyze", str(wav), "--mock", "--store", str(db),
         "--model-dir", str(tmp_path / "models"), "--chunk-sec", "1", "--overlap-sec", "0.1"],
        out, err,
    )
    assert rc == 0
    data = json.loads(out.getvalue())
    assert data["command"] == "analyze"
    assert data["store_run_id"]

    from localmind.store import Store
    with Store(db) as store:
        run = store.get_run(data["store_run_id"])
    assert run["inference_run"]["status"] == "ok"
    assert len(run["segments"]) >= 1


def test_analyze_nonmock_fails_locally_not_download(tmp_path, monkeypatch):
    """Without --mock and with no provisioned backend, analyze must fail with a
    structured local error — never attempt a network download."""
    _block_network(monkeypatch)
    wav = _wav(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    rc = main(
        ["analyze", str(wav), "--model-dir", str(tmp_path / "no-such-models")],
        out, err,
    )
    assert rc == 1
    data = json.loads(out.getvalue())
    assert data["error"]["code"] in {"cli_error", "provisioning_error"}
    # A structured error JSON (not an AssertionError traceback) proves no
    # network egress was attempted.


def test_summarize_mock_runs_with_network_blocked(tmp_path, monkeypatch):
    """summarize (mock LLM) also needs no network."""
    _block_network(monkeypatch)
    segs = [{"id": "seg-0000", "start": 0.0, "end": 1.0, "text": "hello"},
            {"id": "seg-0001", "start": 1.0, "end": 2.0, "text": "world"}]
    tjson = tmp_path / "transcript.json"
    tjson.write_text(json.dumps({"segments": segs}))
    out, err = io.StringIO(), io.StringIO()
    rc = main(["summarize", str(tjson), "--mock"], out, err)
    assert rc == 0
    summary = json.loads(out.getvalue())
    assert summary["schema_version"] == "soundmind.summary.v1"
