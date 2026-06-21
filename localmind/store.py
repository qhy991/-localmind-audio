"""Normalized local storage boundary.

A single authoritative store for pipeline artifacts: SQLite for indexed entities
(``audio_asset``, ordered ``transcript_segment``, ``summary_artifact``,
``inference_run``, ``model_manifest_ref``) with foreign keys enforced, plus the
summary payload stored as JSON. Reference integrity is enforced at write time:
a successful summary's citations must reference transcript segments already
stored for the same run (via the summary schema validator). A failed summary is
stored as a ``summary_failed`` artifact rather than dropped.

A future SwiftData facade can sit on top of this store, never as a second
source of truth.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from localmind.summary.schema import (
    SummaryValidationError,
    validate_summary_against_transcript,
    validate_summary_dict,
    validate_summary_failed_dict,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_asset (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    duration_sec REAL NOT NULL,
    sample_rate INTEGER NOT NULL,
    format TEXT,
    sha256 TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inference_run (
    id TEXT PRIMARY KEY,
    audio_asset_id TEXT NOT NULL REFERENCES audio_asset(id),
    stt_tier TEXT,
    stt_model_id TEXT,
    stt_sha256 TEXT,
    llm_model_id TEXT,
    prompt_template_hash TEXT,
    chunk_duration_sec REAL,
    overlap_sec REAL,
    schema_version TEXT,
    status TEXT NOT NULL,
    metrics_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transcript_segment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES inference_run(id),
    ord INTEGER NOT NULL,
    seg_id TEXT NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(run_id, seg_id),
    UNIQUE(run_id, ord)
);
CREATE TABLE IF NOT EXISTS summary_artifact (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES inference_run(id),
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_manifest_ref (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES inference_run(id),
    model_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    sha256 TEXT,
    quant_format TEXT,
    path TEXT
);
"""


class StoreError(Exception):
    """Base class for storage failures."""


class ReferenceIntegrityError(StoreError):
    """A summary cited transcript segments that are not stored for its run."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return uuid.uuid4().hex


def _seg_field(s, name):
    return s[name] if isinstance(s, dict) else getattr(s, name)


class Store:
    """SQLite-backed normalized store. Use as a context manager or close()."""

    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def put_audio_asset(
        self, *, path, duration_sec: float, sample_rate: int, fmt: str = "", sha256: str = ""
    ) -> str:
        asset_id = _uuid()
        self._conn.execute(
            "INSERT INTO audio_asset(id,path,duration_sec,sample_rate,format,sha256,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (asset_id, str(path), float(duration_sec), int(sample_rate), fmt, sha256, _now()),
        )
        self._conn.commit()
        return asset_id

    def put_run(
        self,
        audio_asset_id: str,
        *,
        stt_tier: str = "",
        stt_model_id: str = "",
        stt_sha256: str = "",
        llm_model_id: str = "",
        prompt_template_hash: str = "",
        chunk_duration_sec: float = 0.0,
        overlap_sec: float = 0.0,
        schema_version: str = "",
        status: str = "ok",
        metrics: Optional[Dict] = None,
    ) -> str:
        run_id = _uuid()
        self._conn.execute(
            "INSERT INTO inference_run(id,audio_asset_id,stt_tier,stt_model_id,stt_sha256,"
            "llm_model_id,prompt_template_hash,chunk_duration_sec,overlap_sec,schema_version,"
            "status,metrics_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, audio_asset_id, stt_tier, stt_model_id, stt_sha256, llm_model_id,
             prompt_template_hash, float(chunk_duration_sec), float(overlap_sec), schema_version,
             status, json.dumps(metrics) if metrics is not None else None, _now()),
        )
        self._conn.commit()
        return run_id

    def put_model_ref(
        self, run_id: str, *, model_id: str, kind: str, sha256: str = "",
        quant_format: str = "", path: str = ""
    ) -> None:
        self._conn.execute(
            "INSERT INTO model_manifest_ref(run_id,model_id,kind,sha256,quant_format,path) "
            "VALUES(?,?,?,?,?,?)",
            (run_id, model_id, kind, sha256, quant_format, path),
        )
        self._conn.commit()

    def put_segments(self, run_id: str, segments) -> None:
        for ord_i, s in enumerate(segments):
            self._conn.execute(
                "INSERT INTO transcript_segment(run_id,ord,seg_id,start,end,text) "
                "VALUES(?,?,?,?,?,?)",
                (run_id, ord_i, str(_seg_field(s, "id")), float(_seg_field(s, "start")),
                 float(_seg_field(s, "end")), str(_seg_field(s, "text"))),
            )
        self._conn.commit()

    def stored_segments(self, run_id: str) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT seg_id,start,end,text FROM transcript_segment WHERE run_id=? ORDER BY ord",
            (run_id,),
        ).fetchall()
        return [{"id": r["seg_id"], "start": r["start"], "end": r["end"], "text": r["text"]} for r in rows]

    def put_summary(self, run_id: str, summary: Dict) -> None:
        """Persist a summary (success or summary_failed) for a run.

        Successful summaries are validated against the run's stored transcript
        segments: any citation referencing an unknown segment raises
        ``ReferenceIntegrityError``.
        """
        if summary.get("status") == "failed":
            validate_summary_failed_dict(summary)
            status_val = "failed"
        else:
            try:
                validate_summary_dict(summary)
                validate_summary_against_transcript(summary, self.stored_segments(run_id))
            except SummaryValidationError as exc:
                raise ReferenceIntegrityError(str(exc)) from exc
            status_val = "ok"
        self._conn.execute(
            "INSERT INTO summary_artifact(run_id,status,payload_json,created_at) "
            "VALUES(?,?,?,?)",
            (run_id, status_val, json.dumps(summary), _now()),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> Dict:
        row = self._conn.execute(
            "SELECT * FROM inference_run WHERE id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise StoreError(f"unknown inference run: {run_id}")
        asset = self._conn.execute(
            "SELECT * FROM audio_asset WHERE id=?", (row["audio_asset_id"],)
        ).fetchone()
        summary = self._conn.execute(
            "SELECT * FROM summary_artifact WHERE run_id=? ORDER BY id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        ref_rows = self._conn.execute(
            "SELECT model_id,kind,sha256,quant_format,path FROM model_manifest_ref "
            "WHERE run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()
        return {
            "run_id": run_id,
            "audio_asset": dict(asset) if asset else None,
            "inference_run": dict(row),
            "segments": self.stored_segments(run_id),
            "summary": json.loads(summary["payload_json"]) if summary else None,
            "model_manifest_refs": [dict(r) for r in ref_rows],
        }

    def put_full_run(
        self,
        *,
        audio: Dict,
        run: Dict,
        model_refs: List[Dict],
        segments,
        summary: Dict,
    ) -> str:
        """Persist a complete run atomically in a single transaction.

        If summary validation fails (e.g. a citation references a segment not in
        ``segments``), the entire run is rolled back — no orphaned
        ``inference_run``/segments/asset rows remain. ``audio`` has keys
        path/duration_sec/sample_rate/format/sha256; ``run`` has the inference_run
        fields (stt_*, llm_model_id, prompt_template_hash, chunk/overlap,
        schema_version, status, metrics); each ``model_refs`` item has
        model_id/kind/sha256/quant_format/path.
        """
        conn = self._conn
        asset_id = _uuid()
        run_id = _uuid()
        try:
            conn.execute(
                "INSERT INTO audio_asset(id,path,duration_sec,sample_rate,format,sha256,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (asset_id, str(audio["path"]), float(audio["duration_sec"]),
                 int(audio.get("sample_rate", 16000)), audio.get("format", ""),
                 audio.get("sha256", ""), _now()),
            )
            conn.execute(
                "INSERT INTO inference_run(id,audio_asset_id,stt_tier,stt_model_id,stt_sha256,"
                "llm_model_id,prompt_template_hash,chunk_duration_sec,overlap_sec,schema_version,"
                "status,metrics_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, asset_id, run.get("stt_tier", ""), run.get("stt_model_id", ""),
                 run.get("stt_sha256", ""), run.get("llm_model_id", ""),
                 run.get("prompt_template_hash", ""), float(run.get("chunk_duration_sec", 0.0)),
                 float(run.get("overlap_sec", 0.0)), run.get("schema_version", ""),
                 run.get("status", "ok"),
                 json.dumps(run["metrics"]) if run.get("metrics") is not None else None, _now()),
            )
            for ref in model_refs:
                conn.execute(
                    "INSERT INTO model_manifest_ref(run_id,model_id,kind,sha256,quant_format,path) "
                    "VALUES(?,?,?,?,?,?)",
                    (run_id, ref.get("model_id", ""), ref.get("kind", ""),
                     ref.get("sha256", ""), ref.get("quant_format", ""), ref.get("path", "")),
                )
            for ord_i, s in enumerate(segments):
                conn.execute(
                    "INSERT INTO transcript_segment(run_id,ord,seg_id,start,end,text) "
                    "VALUES(?,?,?,?,?,?)",
                    (run_id, ord_i, str(_seg_field(s, "id")), float(_seg_field(s, "start")),
                     float(_seg_field(s, "end")), str(_seg_field(s, "text"))),
                )
            # Validate summary against the segments about to be committed (in-memory).
            if summary.get("status") == "failed":
                validate_summary_failed_dict(summary)
                status_val = "failed"
            else:
                try:
                    validate_summary_dict(summary)
                    validate_summary_against_transcript(summary, segments)
                except SummaryValidationError as exc:
                    raise ReferenceIntegrityError(str(exc)) from exc
                status_val = "ok"
            conn.execute(
                "INSERT INTO summary_artifact(run_id,status,payload_json,created_at) "
                "VALUES(?,?,?,?)",
                (run_id, status_val, json.dumps(summary), _now()),
            )
            conn.commit()
            return run_id
        except Exception:
            conn.rollback()
            raise

    def update_run_metrics(self, run_id: str, metrics: Dict) -> None:
        """Update the metrics_json for a committed run.

        Used to record the measured persistence-stage duration, which cannot be
        known until the run's transaction has completed. The run itself (asset,
        segments, summary, refs) is already committed atomically by
        :meth:`put_full_run`; this only updates the metrics payload.
        """
        self._conn.execute(
            "UPDATE inference_run SET metrics_json=? WHERE id=?",
            (json.dumps(metrics), run_id),
        )
        self._conn.commit()

