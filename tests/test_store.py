"""Tests for the normalized store: put/get, reference integrity, FK, and
reopen-in-new-process persistence."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from localmind.stt.segment import TranscriptSegment
from localmind.store import ReferenceIntegrityError, Store, StoreError
from localmind.summary import SUMMARY_SCHEMA_VERSION, build_summary_failed

REPO = Path(__file__).resolve().parents[1]


def _seg(i, start=None, end=None, text="hello"):
    return TranscriptSegment(
        id=f"seg-{i:04d}",
        start=float(i if start is None else start),
        end=float(i + 1 if end is None else end),
        text=text,
    )


def _valid_summary(case_id="c", seg="seg-0000"):
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "case_id": case_id,
        "provenance": {
            "model_id": "m", "prompt_template_hash": "h",
            "repaired": False, "repair_attempted": False,
            "repair_attempts_used": 0, "initial_validation_errors": [],
        },
        "decisions": [{"text": "a decision", "citations": [f"seg:{seg}"]}],
        "action_items": [{"text": "do it", "owner": None, "due_date": None, "citations": [f"seg:{seg}"]}],
        "open_questions": [],
    }


def _seed_run(store, segments=None, summary=None):
    if segments is None:
        segments = [_seg(0, 0.0, 1.0), _seg(1, 1.0, 2.0)]
    if summary is None:
        summary = _valid_summary(seg=segments[0].id)
    status = "failed" if summary.get("status") == "failed" else "ok"
    asset_id = store.put_audio_asset(
        path="/tmp/x.wav", duration_sec=2.0, sample_rate=16000, fmt="wav"
    )
    run_id = store.put_run(asset_id, stt_tier="mock", status=status)
    store.put_segments(run_id, segments)
    store.put_summary(run_id, summary)
    return run_id, segments, summary


def test_put_and_get_run_roundtrips(tmp_path):
    with Store(tmp_path / "s.db") as store:
        run_id, segments, summary = _seed_run(store)
        run = store.get_run(run_id)
    assert run["run_id"] == run_id
    assert run["audio_asset"]["path"] == "/tmp/x.wav"
    assert run["audio_asset"]["duration_sec"] == 2.0
    assert [s["id"] for s in run["segments"]] == [s.id for s in segments]
    assert run["segments"][0]["start"] == 0.0
    assert run["summary"]["case_id"] == summary["case_id"]
    assert run["inference_run"]["status"] == "ok"


def test_summary_citing_unknown_segment_rejected(tmp_path):
    with Store(tmp_path / "s.db") as store:
        run_id, _segs, _ = _seed_run(store)
        bad = _valid_summary()
        bad["decisions"][0]["citations"] = ["seg:seg-9999"]
        with pytest.raises(ReferenceIntegrityError):
            store.put_summary(run_id, bad)


def test_summary_failed_persisted(tmp_path):
    with Store(tmp_path / "s.db") as store:
        run_id, _segs, _ = _seed_run(
            store, summary=build_summary_failed("raw out", ["bad"], case_id="c", model_id="m", prompt_template_hash="h")
        )
        run = store.get_run(run_id)
    assert run["summary"]["status"] == "failed"
    assert run["summary"]["raw_output"] == "raw out"
    assert run["inference_run"]["status"] == "failed"


def test_foreign_key_enforced(tmp_path):
    with Store(tmp_path / "s.db") as store:
        # put_run references a non-existent audio_asset -> FK violation.
        with pytest.raises(sqlite3.IntegrityError):
            store.put_run("nonexistent-asset-id", status="ok")


def test_segments_ordered_and_unique(tmp_path):
    with Store(tmp_path / "s.db") as store:
        run_id, _segs, _ = _seed_run(store)
        ordered = store.stored_segments(run_id)
        assert [s["id"] for s in ordered] == ["seg-0000", "seg-0001"]
        # Duplicate seg_id for the same run is rejected.
        with pytest.raises(sqlite3.IntegrityError):
            store.put_segments(run_id, [_seg(0, 0.0, 1.0)])


def test_get_unknown_run_raises(tmp_path):
    with Store(tmp_path / "s.db") as store:
        with pytest.raises(StoreError):
            store.get_run("does-not-exist")


def test_reopen_in_new_process(tmp_path):
    """A new OS process opening the same store file reads back the full run."""
    db = tmp_path / "s.db"
    with Store(db) as store:
        run_id, segments, _ = _seed_run(store)

    snippet = (
        "import json\n"
        "from localmind.store import Store\n"
        f"s = Store({str(db)!r})\n"
        f"r = s.get_run({run_id!r})\n"
        "print(json.dumps({'n': len(r['segments']), "
        "'first_id': r['segments'][0]['id'], "
        "'status': (r['summary'] or {}).get('status', 'ok')}))\n"
        "s.close()\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet], cwd=str(REPO),
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip())
    assert data["n"] == len(segments)
    assert data["first_id"] == "seg-0000"
    assert data["status"] == "ok"
