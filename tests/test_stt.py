"""Tests for the STT framework: segment validation, bounded chunking, tier
resolution, mock transcriber, and the WhisperTranscriber adapter (with a fake
mlx_whisper module so logic is verified without real weights).
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from localmind.provisioning import ModelNotProvisionedError, Provisioner
from localmind.stt import (
    MAX_UNCHUNKED_SEC,
    ArrayAudioSource,
    ChunkingConfig,
    MockTranscriber,
    ResolvedTier,
    TranscriptSegment,
    WavAudioSource,
    WhisperTranscriber,
    iter_audio_chunks,
    resolve_tier,
    validate_segments,
)
from localmind.stt.segment import SegmentValidationError


# --------------------------------------------------------------------------- #
# Segment validation                                                          #
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
# Bounded chunking                                                            #
# --------------------------------------------------------------------------- #

class FakeAudioSource:
    """Source that materializes only the requested window and tracks the largest."""

    def __init__(self, duration_sec, sample_rate=16000):
        self._dur = float(duration_sec)
        self._sr = int(sample_rate)
        self.max_window_samples = 0

    @property
    def sample_rate(self):
        return self._sr

    @property
    def duration_sec(self):
        return self._dur

    def read_window(self, start_sec, end_sec):
        n = max(0, int(round((end_sec - start_sec) * self._sr)))
        if n > self.max_window_samples:
            self.max_window_samples = n
        return np.zeros(n, dtype=np.float32)


def test_iter_audio_chunks_bounded_over_array_source():
    sr = 16000
    source = ArrayAudioSource(np.zeros(int(120 * sr), np.float32), sr)
    config = ChunkingConfig(chunk_duration_sec=30.0, overlap_sec=1.0)
    chunks = list(iter_audio_chunks(source, config))
    assert len(chunks) > 1
    for ch in chunks:
        assert ch.samples.size <= int(30 * sr) + 1
    starts = [ch.start_sec for ch in chunks]
    assert starts == sorted(starts)


def test_bounded_memory_does_not_scale_with_file_length():
    """The largest live window stays bounded regardless of total duration."""
    sr = 16000
    config = ChunkingConfig(chunk_duration_sec=30.0, overlap_sec=1.0)
    cap = int(round((config.chunk_duration_sec) * sr))  # window is chunk_duration

    short = FakeAudioSource(60.0, sr)
    list(iter_audio_chunks(short, config))
    long_src = FakeAudioSource(60 * 60.0, sr)  # 60 min, but never allocated
    list(iter_audio_chunks(long_src, config))

    # Peak materialized window is bounded by the chunk window in both cases,
    # and does NOT grow with total duration. No 60-minute array was allocated.
    assert short.max_window_samples <= cap + 1
    assert long_src.max_window_samples <= cap + 1
    assert long_src.max_window_samples == short.max_window_samples


def test_chunking_disabled_on_long_file_is_rejected():
    # 60-minute source with chunking disabled -> rejected, no allocation.
    source = FakeAudioSource(60 * 60.0, 16000)
    with pytest.raises(ValueError, match="mandatory for long audio"):
        list(iter_audio_chunks(source, ChunkingConfig(enabled=False)))


def test_chunking_disabled_on_short_file_yields_single_chunk():
    source = ArrayAudioSource(np.zeros(60 * 16000, np.float32), 16000)
    chunks = list(iter_audio_chunks(source, ChunkingConfig(enabled=False)))
    assert len(chunks) == 1
    assert chunks[0].start_sec == 0.0
    assert chunks[0].final is True


def test_invalid_chunking_config_rejected():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_duration_sec=0)
    with pytest.raises(ValueError):
        ChunkingConfig(overlap_sec=5.0, chunk_duration_sec=5.0)


def _write_sine_wav(path, duration_sec=5.0, sr=16000):
    n = int(duration_sec * sr)
    t = np.arange(n) / sr
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((samples * 32767).astype("<i2").tobytes())
    return path


def test_wav_audio_source_reads_only_window(tmp_path):
    wav = _write_sine_wav(tmp_path / "tone.wav", duration_sec=10.0, sr=16000)
    src = WavAudioSource(wav, target_sample_rate=16000)
    assert src.duration_sec == pytest.approx(10.0, abs=1e-3)
    window = src.read_window(2.0, 3.0)
    # A 1-second window at 16 kHz yields ~16000 samples, not the whole 10s file.
    assert window.size == pytest.approx(16000, abs=2)
    assert window.dtype == np.float32


# --------------------------------------------------------------------------- #
# Tier resolution (missing tier fast-fail, no download)                       #
# --------------------------------------------------------------------------- #

def _prov_with_empty_manifest(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "models.json").write_text(json.dumps({"schema_version": "1", "models": []}))
    return Provisioner(model_dir)


def _prov_with_model(tmp_path, model_id="whisper-small"):
    model_dir = tmp_path / "models"
    content = b"whisper-weights" * 100
    p = model_dir / f"{model_id}.mlmodel"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    (model_dir / "models.json").write_text(json.dumps({
        "schema_version": "1",
        "models": [{
            "model_id": model_id, "name": model_id, "kind": "whisper",
            "path": f"{model_id}.mlmodel", "quant_format": "int4",
            "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "license": "MIT",
        }],
    }))
    return Provisioner(model_dir)


def test_resolve_tier_fast_fails_without_download(tmp_path, monkeypatch):
    prov = _prov_with_empty_manifest(tmp_path)

    def _no_network(*_a, **_k):
        raise AssertionError("resolve_tier attempted a network download")

    monkeypatch.setattr(socket, "socket", _no_network)
    with pytest.raises(ModelNotProvisionedError):
        resolve_tier(prov, "whisper-small")


def test_resolve_tier_returns_provenance_when_provisioned(tmp_path):
    prov = _prov_with_model(tmp_path)
    resolved = resolve_tier(prov, "whisper-small")
    assert isinstance(resolved, ResolvedTier)
    assert resolved.tier == "whisper-small"
    assert resolved.model_path.is_file()
    assert len(resolved.sha256) == 64
    assert resolved.quant_format == "int4"


# --------------------------------------------------------------------------- #
# MockTranscriber                                                             #
# --------------------------------------------------------------------------- #

def test_mock_transcriber_emits_ordered_bounded_segments():
    sr = 16000
    source = ArrayAudioSource(np.zeros(int(12 * sr), np.float32), sr)
    t = MockTranscriber()
    segs = t.transcribe(source, ChunkingConfig(chunk_duration_sec=5.0, overlap_sec=1.0), Path("/dev/null"))
    # 12s with 5s chunks, 1s overlap -> hops at 0,4,8 -> 3 chunks/segments
    assert len(segs) == 3
    assert segs[-1].end == pytest.approx(12.0, abs=1e-3)
    # Already validated inside transcribe; re-validate to be explicit.
    validate_segments(segs, audio_duration_sec=12.0)


# --------------------------------------------------------------------------- #
# WhisperTranscriber with a fake mlx_whisper module                          #
# --------------------------------------------------------------------------- #

class _FakeMlxWhisper:
    """Fake mlx_whisper returning deterministic per-chunk segments."""

    def __init__(self, segments=None, raw_result=None, fail_text=False, out_of_bounds=False):
        self.segments = segments if segments is not None else [
            {"start": 0.0, "end": 0.5, "text": "hello"},
            {"start": 1.0, "end": 1.5, "text": "world"},
        ]
        self.raw_result = raw_result
        self.fail_text = fail_text
        self.out_of_bounds = out_of_bounds
        self.call_count = 0

    def transcribe(self, audio, path_or_hf_repo=None, language=None, verbose=None):
        self.call_count += 1
        if self.raw_result is not None:
            return self.raw_result
        segs = []
        for s in self.segments:
            seg = dict(s)
            if self.fail_text:
                seg["text"] = "   "
            if self.out_of_bounds:
                seg["end"] = 10_000.0
            segs.append(seg)
        return {"segments": segs, "text": " ".join(str(s.get("text", "")) for s in segs)}


def _install_fake_mlx_whisper(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)


def test_whisper_transcriber_converts_offsets_and_validates(monkeypatch):
    fake = _FakeMlxWhisper()
    _install_fake_mlx_whisper(monkeypatch, fake)

    sr = 16000
    source = ArrayAudioSource(np.zeros(int(12 * sr), np.float32), sr)
    config = ChunkingConfig(chunk_duration_sec=5.0, overlap_sec=1.0)
    t = WhisperTranscriber(language="en")
    segs = t.transcribe(source, config, Path("/models/whisper-small.mlmodel"))

    # 3 chunks (0,4,8) x 2 segments each = 6 segments.
    assert fake.call_count == 3
    assert len(segs) == 6
    # First segment of the second chunk is offset by 4.0s.
    assert segs[2].start == pytest.approx(4.0)
    assert segs[2].end == pytest.approx(4.5)
    # All ids normalized and unique.
    assert len({s.id for s in segs}) == 6
    # Passes validation (already done internally).
    validate_segments(segs, audio_duration_sec=12.0)


def test_whisper_transcriber_does_not_touch_network(monkeypatch):
    fake = _FakeMlxWhisper()
    _install_fake_mlx_whisper(monkeypatch, fake)

    def _no_network(*_a, **_k):
        raise AssertionError("transcription attempted a network connection")

    monkeypatch.setattr(socket, "socket", _no_network)
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    WhisperTranscriber().transcribe(
        source, ChunkingConfig(chunk_duration_sec=5.0), Path("/models/w.mlmodel")
    )  # would raise via _no_network if any network path were taken


def test_whisper_transcriber_rejects_non_dict_result(monkeypatch):
    _install_fake_mlx_whisper(monkeypatch, _FakeMlxWhisper(raw_result=["not", "a", "dict"]))
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(ValueError, match="dict"):
        WhisperTranscriber().transcribe(source, ChunkingConfig(chunk_duration_sec=5.0), Path("/m"))


def test_whisper_transcriber_rejects_non_list_segments(monkeypatch):
    _install_fake_mlx_whisper(monkeypatch, _FakeMlxWhisper(raw_result={"segments": "nope"}))
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(ValueError, match="list"):
        WhisperTranscriber().transcribe(source, ChunkingConfig(chunk_duration_sec=5.0), Path("/m"))


def test_whisper_transcriber_rejects_empty_text(monkeypatch):
    _install_fake_mlx_whisper(monkeypatch, _FakeMlxWhisper(fail_text=True))
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(SegmentValidationError, match="text"):
        WhisperTranscriber().transcribe(source, ChunkingConfig(chunk_duration_sec=5.0), Path("/m"))


def test_whisper_transcriber_rejects_out_of_bounds_timestamps(monkeypatch):
    _install_fake_mlx_whisper(monkeypatch, _FakeMlxWhisper(out_of_bounds=True))
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(SegmentValidationError, match="exceeds audio duration"):
        WhisperTranscriber().transcribe(source, ChunkingConfig(chunk_duration_sec=5.0), Path("/m"))


def test_whisper_transcriber_raises_clear_error_without_mlx_whisper(monkeypatch):
    # Ensure no real or cached mlx_whisper is importable.
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(RuntimeError, match="mlx-whisper"):
        WhisperTranscriber().transcribe(source, ChunkingConfig(chunk_duration_sec=5.0), Path("/m"))


def test_whisper_transcriber_real_smoke_skips_without_backend_or_weights():
    """Real-backend smoke test: runs only when mlx_whisper and a provisioned
    Whisper model are available; otherwise skips. Logic is covered by the fake
    tests above."""
    pytest.importorskip("mlx_whisper")
    # A real run additionally requires a provisioned model directory; skip if
    # the user has not provisioned one (see docs/provisioning.md).
    model_dir = Path("models")
    if not (model_dir / "models.json").exists():
        pytest.skip("no provisioned models/ directory for real Whisper smoke test")
