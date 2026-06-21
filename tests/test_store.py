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


# --------------------------------------------------------------------------- #
# Atomic full-run persistence, model refs, segment ordering                   #
# --------------------------------------------------------------------------- #

def _refs():
    return [
        {"model_id": "whisper-small", "kind": "whisper", "sha256": "a" * 64, "quant_format": "int4", "path": "m.mlmodel"},
        {"model_id": "qwen-7b", "kind": "llm", "sha256": "", "quant_format": "", "path": ""},
    ]


def test_put_full_run_roundtrip_with_refs(tmp_path):
    with Store(tmp_path / "s.db") as store:
        segments = [_seg(0, 0.0, 1.0), _seg(1, 1.0, 2.0)]
        run_id = store.put_full_run(
            audio={"path": "/tmp/x.wav", "duration_sec": 2.0, "sample_rate": 16000, "format": "wav"},
            run={"stt_tier": "whisper-small", "stt_model_id": "whisper-small",
                 "llm_model_id": "qwen-7b", "prompt_template_hash": "h",
                 "chunk_duration_sec": 30.0, "overlap_sec": 1.0,
                 "schema_version": "1", "status": "ok",
                 "metrics": {"stages": [{"stage": "decode", "duration_sec": 0.1}], "total_duration_sec": 0.5, "rtf": 0.25}},
            model_refs=_refs(),
            segments=segments,
            summary=_valid_summary(seg="seg-0000"),
        )
        run = store.get_run(run_id)
    assert [s["id"] for s in run["segments"]] == ["seg-0000", "seg-0001"]
    assert run["summary"]["case_id"] == _valid_summary(seg="seg-0000")["case_id"]
    refs = run["model_manifest_refs"]
    assert [r["kind"] for r in refs] == ["whisper", "llm"]
    assert refs[0]["model_id"] == "whisper-small"
    assert refs[1]["model_id"] == "qwen-7b"
    assert run["inference_run"]["metrics_json"] is not None


def test_put_full_run_atomic_rollback_on_bad_summary(tmp_path):
    """A summary citing an unknown segment rolls back the entire run."""
    with Store(tmp_path / "s.db") as store:
        bad = _valid_summary(seg="seg-0000")
        bad["decisions"][0]["citations"] = ["seg:seg-9999"]  # not in segments
        with pytest.raises(ReferenceIntegrityError):
            store.put_full_run(
                audio={"path": "/tmp/x.wav", "duration_sec": 2.0, "sample_rate": 16000, "format": "wav"},
                run={"stt_tier": "mock", "status": "ok", "metrics": {"stages": []}},
                model_refs=_refs(),
                segments=[_seg(0, 0.0, 1.0)],
                summary=bad,
            )
        # No orphaned rows survive the rollback.
        n_runs = store._conn.execute("SELECT COUNT(*) AS c FROM inference_run").fetchone()["c"]
        n_segs = store._conn.execute("SELECT COUNT(*) AS c FROM transcript_segment").fetchone()["c"]
        n_assets = store._conn.execute("SELECT COUNT(*) AS c FROM audio_asset").fetchone()["c"]
    assert n_runs == 0
    assert n_segs == 0
    assert n_assets == 0


def test_duplicate_ord_rejected(tmp_path):
    """A second segment batch with a duplicate ordinal for the same run is rejected."""
    with Store(tmp_path / "s.db") as store:
        run_id, _segs, _ = _seed_run(store)
        # put_segments starts ord at 0 again -> duplicate (run_id, ord=0).
        with pytest.raises(sqlite3.IntegrityError):
            store.put_segments(run_id, [_seg(2, 2.0, 3.0)])


def test_get_run_returns_model_manifest_refs(tmp_path):
    with Store(tmp_path / "s.db") as store:
        run_id = store.put_full_run(
            audio={"path": "/tmp/x.wav", "duration_sec": 1.0, "sample_rate": 16000, "format": "wav"},
            run={"stt_tier": "mock", "status": "ok", "metrics": None},
            model_refs=_refs(),
            segments=[_seg(0, 0.0, 1.0)],
            summary=_valid_summary(seg="seg-0000"),
        )
        run = store.get_run(run_id)
    assert len(run["model_manifest_refs"]) == 2
    assert {r["kind"] for r in run["model_manifest_refs"]} == {"whisper", "llm"}


def test_put_full_run_with_metrics_atomic_rollback_on_metrics_failure(tmp_path):
    """If the metrics callback fails, the entire run rolls back — no orphaned
    rows in any of the five store tables."""
    with Store(tmp_path / "s.db") as store:
        def bad_metrics(persist_sec, audio_duration):
            raise RuntimeError("simulated metrics failure")

        with pytest.raises(RuntimeError, match="simulated metrics failure"):
            store.put_full_run_with_metrics(
                audio={"path": "/tmp/x.wav", "duration_sec": 1.0, "sample_rate": 16000, "format": "wav"},
                run={"stt_tier": "mock", "status": "ok"},
                model_refs=_refs(),
                segments=[_seg(0, 0.0, 1.0)],
                summary=_valid_summary(seg="seg-0000"),
                build_metrics=bad_metrics,
            )
        for table in ("audio_asset", "inference_run", "transcript_segment",
                       "summary_artifact", "model_manifest_ref"):
            n = store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert n == 0, f"{table} has {n} orphaned rows"


def test_put_full_run_with_metrics_stores_non_null_metrics(tmp_path):
    """A successful run has non-null metrics_json with the measured persist stage."""
    with Store(tmp_path / "s.db") as store:
        run_id, metrics = store.put_full_run_with_metrics(
            audio={"path": "/tmp/x.wav", "duration_sec": 2.0, "sample_rate": 16000, "format": "wav"},
            run={"stt_tier": "mock", "status": "ok"},
            model_refs=_refs(),
            segments=[_seg(0, 0.0, 1.0), _seg(1, 1.0, 2.0)],
            summary=_valid_summary(seg="seg-0000"),
            build_metrics=lambda persist, dur: {
                "stages": [{"stage": "decode", "duration_sec": 0.1},
                           {"stage": "stt", "duration_sec": 0.2},
                           {"stage": "llm", "duration_sec": 0.3},
                           {"stage": "persist", "duration_sec": persist}],
                "total_duration_sec": 0.1 + 0.2 + 0.3 + persist,
                "rtf": (0.1 + 0.2 + 0.3 + persist) / dur,
            },
        )
        run = store.get_run(run_id)
    assert run["inference_run"]["metrics_json"] is not None
    stored = json.loads(run["inference_run"]["metrics_json"])
    assert stored["stages"][3]["stage"] == "persist"
    assert stored["stages"][3]["duration_sec"] > 0  # measured, not stale 0.0
