"""Acceptance tests for task4 framework: segments, chunking, tier selection, mock transcriber (AC-2/AC-2.1)."""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest

from localmind.provisioning import ManifestError, Provisioner
from localmind.stt import (
    MAX_UNCHUNKED_SEC,
    ChunkingConfig,
    MockTranscriber,
    TranscriptSegment,
    WhisperTranscriber,
    chunk_audio,
    select_tier,
    validate_segments,
)
from localmind.stt.segment import SegmentValidationError


# --------------------------------------------------------------------------- #
# Segment validation (AC-2)                                                    #
# --------------------------------------------------------------------------- #

def _seg(i, start, end, text="x"):
    return TranscriptSegment(id=f"seg-{i:04d}", start=start, end=end, text=text)


def test_validate_segments_accepts_ordered_bounded_nonempty():
    segs = [_seg(0, 0.0, 2.0, "hello"), _seg(1, 2.0, 4.5, "world")]
    assert validate_segments(segs, audio_duration_sec=5.0) is segs


def test_validate_segments_rejects_non_monotonic_timestamps():
    # Second segment starts before the first (1.0 < 2.0) -> non-monotonic.
    segs = [_seg(0, 2.0, 4.0), _seg(1, 1.0, 3.0)]
    with pytest.raises(SegmentValidationError, match="monotonically"):
        validate_segments(segs, audio_duration_sec=10.0)


def test_validate_segments_rejects_segment_exceeding_audio_duration():
    segs = [_seg(0, 0.0, 12.0)]
    with pytest.raises(SegmentValidationError, match="exceeds audio duration"):
        validate_segments(segs, audio_duration_sec=10.0)


def test_validate_segments_rejects_empty_text():
    segs = [_seg(0, 0.0, 2.0, "   ")]
    with pytest.raises(SegmentValidationError, match="text"):
        validate_segments(segs, audio_duration_sec=5.0)


def test_validate_segments_rejects_flat_untimed_zero_length():
    """A flat untimed transcript (start==end==0) is rejected."""
    segs = [TranscriptSegment(id="seg-0000", start=0.0, end=0.0, text="all")]
    with pytest.raises(SegmentValidationError, match="end > start"):
        validate_segments(segs, audio_duration_sec=10.0)


def test_validate_segments_rejects_start_after_end():
    segs = [_seg(0, 3.0, 1.0)]
    with pytest.raises(SegmentValidationError, match="end > start"):
        validate_segments(segs, audio_duration_sec=10.0)


def test_validate_segments_rejects_duplicate_ids():
    segs = [_seg(0, 0.0, 1.0), _seg(0, 1.0, 2.0)]
    with pytest.raises(SegmentValidationError, match="duplicate"):
        validate_segments(segs, audio_duration_sec=5.0)


# --------------------------------------------------------------------------- #
# Bounded chunking (AC-2.1)                                                    #
# --------------------------------------------------------------------------- #

def test_chunk_audio_yields_bounded_chunks():
    sr = 16000
    samples = np.zeros(int(120 * sr), dtype=np.float32)  # 2 min
    config = ChunkingConfig(chunk_duration_sec=30.0, overlap_sec=1.0)
    chunks = list(chunk_audio(samples, sr, config))
    assert len(chunks) > 1
    # Each chunk is at most chunk_duration_sec long.
    for _, chunk in chunks:
        assert chunk.size <= int(30 * sr) + 1
    # Starts are non-decreasing.
    starts = [s for s, _ in chunks]
    assert starts == sorted(starts)


def test_chunk_audio_peak_buffer_does_not_scale_with_file_length():
    """AC-2.1: max chunk size is bounded regardless of total file length."""
    sr = 16000
    config = ChunkingConfig(chunk_duration_sec=30.0)
    max_short = max((c.size for _, c in chunk_audio(np.zeros(60 * sr, np.float32), sr, config)), default=0)
    max_long = max((c.size for _, c in chunk_audio(np.zeros(60 * 60 * sr, np.float32), sr, config)), default=0)
    assert max_short == max_long  # bounded by chunk size, not file length


def test_chunking_disabled_on_long_file_is_rejected():
    sr = 16000
    # 60-minute audio with chunking disabled -> rejected (AC-2.1 negative).
    long_samples = np.zeros(int(60 * 60 * sr), dtype=np.float32)
    config = ChunkingConfig(enabled=False)
    with pytest.raises(ValueError, match="mandatory for long audio"):
        list(chunk_audio(long_samples, sr, config))


def test_chunking_disabled_on_short_file_yields_single_chunk():
    sr = 16000
    short = np.zeros(int(60 * sr), dtype=np.float32)  # 1 min, under MAX_UNCHUNKED_SEC
    config = ChunkingConfig(enabled=False)
    chunks = list(chunk_audio(short, sr, config))
    assert len(chunks) == 1
    assert chunks[0][0] == 0.0


def test_invalid_chunking_config_rejected():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_duration_sec=0)
    with pytest.raises(ValueError):
        ChunkingConfig(overlap_sec=5.0, chunk_duration_sec=5.0)  # overlap >= duration


# --------------------------------------------------------------------------- #
# Tier selection (AC-2: missing tier fast-fail, no download)                  #
# --------------------------------------------------------------------------- #

def _prov_with_missing_model(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    import json
    (model_dir / "models.json").write_text(
        json.dumps({"schema_version": "1", "models": []})
    )
    return Provisioner(model_dir)


def test_select_tier_fast_fails_without_download(tmp_path, monkeypatch):
    prov = _prov_with_missing_model(tmp_path)

    def _no_network(*_a, **_k):
        raise AssertionError("select_tier attempted a network download")

    monkeypatch.setattr(socket, "socket", _no_network)
    # select_tier goes through the provisioner, which never downloads.
    with pytest.raises(Exception):  # ModelNotProvisionedError
        select_tier(prov, "whisper-small")


def test_select_tier_returns_path_when_provisioned(tmp_path):
    import hashlib, json
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
    path = select_tier(prov, "whisper-small")
    assert path.is_file()


# --------------------------------------------------------------------------- #
# Mock + Whisper transcribers                                                  #
# --------------------------------------------------------------------------- #

def test_mock_transcriber_emits_ordered_bounded_segments():
    sr = 16000
    samples = np.zeros(int(12 * sr), dtype=np.float32)  # 12s
    t = MockTranscriber(segment_duration_sec=5.0)
    segs = t.transcribe(samples, sr, model_path=Path("/dev/null"))
    # 12s / 5s -> 3 segments
    assert len(segs) == 3
    validate_segments(segs, audio_duration_sec=12.0)  # passes validation
    assert segs[-1].end == pytest.approx(12.0)


def test_whisper_transcriber_raises_clear_error_without_mlx_whisper(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "mlx_whisper":
            raise ImportError("no mlx_whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    t = WhisperTranscriber()
    with pytest.raises(RuntimeError, match="mlx-whisper"):
        t.transcribe(np.zeros(16000, np.float32), 16000, Path("/dev/null"))
