"""Tests for the STT framework: segment validation, bounded chunking (WAV and
compressed sources), tier resolution, mock transcriber, and the WhisperTranscriber
adapter (with a fake mlx_whisper module so logic is verified without weights).
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from localmind.audio.decode import _ffmpeg_exe
from localmind.audio.errors import UnsupportedFormatError
from localmind.provisioning import ModelNotProvisionedError, Provisioner
from localmind.stt import (
    MAX_UNCHUNKED_SEC,
    ArrayAudioSource,
    ChunkingConfig,
    FFmpegAudioSource,
    MockTranscriber,
    ResolvedTier,
    TranscriptSegment,
    WavAudioSource,
    WhisperTranscriber,
    audio_source_from_path,
    iter_audio_chunks,
    resolve_tier,
    validate_segments,
)
from localmind.stt.segment import SegmentValidationError

ffmpeg_available = _ffmpeg_exe() is not None


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _seg(i, start, end, text="x"):
    return TranscriptSegment(id=f"seg-{i:04d}", start=start, end=end, text=text)


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


def _transcode(src: Path, dst: Path) -> Path:
    exe = _ffmpeg_exe()
    assert exe is not None
    subprocess.run([exe, "-loglevel", "error", "-y", "-i", str(src), str(dst)], check=True)
    return dst


# --------------------------------------------------------------------------- #
# Segment validation                                                          #
# --------------------------------------------------------------------------- #

def test_validate_segments_accepts_ordered_bounded_nonempty():
    segs = [_seg(0, 0.0, 2.0, "hello"), _seg(1, 2.0, 4.5, "world")]
    assert validate_segments(segs, audio_duration_sec=5.0) is segs


def test_validate_segments_rejects_non_monotonic_timestamps():
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
    sr = 16000
    config = ChunkingConfig(chunk_duration_sec=30.0, overlap_sec=1.0)
    cap = int(round(config.chunk_duration_sec * sr))

    short = FakeAudioSource(60.0, sr)
    list(iter_audio_chunks(short, config))
    long_src = FakeAudioSource(60 * 60.0, sr)
    list(iter_audio_chunks(long_src, config))

    assert short.max_window_samples <= cap + 1
    assert long_src.max_window_samples <= cap + 1
    assert long_src.max_window_samples == short.max_window_samples


def test_chunking_disabled_on_long_file_is_rejected():
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


def test_wav_audio_source_reads_only_window(tmp_path):
    wav = _write_sine_wav(tmp_path / "tone.wav", duration_sec=10.0, sr=16000)
    src = WavAudioSource(wav, target_sample_rate=16000)
    assert src.duration_sec == pytest.approx(10.0, abs=1e-3)
    window = src.read_window(2.0, 3.0)
    assert window.size == pytest.approx(16000, abs=2)
    assert window.dtype == np.float32


# --------------------------------------------------------------------------- #
# audio_source_from_path factory + FFmpegAudioSource                         #
# --------------------------------------------------------------------------- #

def test_audio_source_from_path_dispatches_by_extension(tmp_path):
    wav = _write_sine_wav(tmp_path / "t.wav", duration_sec=2.0, sr=16000)
    assert isinstance(audio_source_from_path(wav), WavAudioSource)
    with pytest.raises(UnsupportedFormatError):
        audio_source_from_path(tmp_path / "x.flac")


@pytest.mark.skipif(not ffmpeg_available, reason="no ffmpeg binary")
def test_audio_source_from_path_returns_ffmpeg_source_for_compressed(tmp_path):
    wav = _write_sine_wav(tmp_path / "t.wav", duration_sec=3.0, sr=16000)
    m4a = _transcode(wav, tmp_path / "t.m4a")
    mp3 = _transcode(wav, tmp_path / "t.mp3")
    assert isinstance(audio_source_from_path(m4a), FFmpegAudioSource)
    assert isinstance(audio_source_from_path(mp3), FFmpegAudioSource)


@pytest.mark.skipif(not ffmpeg_available, reason="no ffmpeg binary")
def test_ffmpeg_source_probes_duration_and_reads_window(tmp_path):
    wav = _write_sine_wav(tmp_path / "tone.wav", duration_sec=5.0, sr=16000)
    for ext in ("m4a", "mp3"):
        comp = _transcode(wav, tmp_path / f"tone.{ext}")
        src = FFmpegAudioSource(comp, target_sample_rate=16000)
        assert src.duration_sec == pytest.approx(5.0, abs=0.25)
        window = src.read_window(1.0, 2.0)
        # A 1s window at 16 kHz; compressed seek is approximate, allow slack.
        assert window.size == pytest.approx(16000, abs=600)
        assert window.dtype == np.float32


def test_ffmpeg_source_bounded_reads_do_not_scale_with_duration(monkeypatch, tmp_path):
    """A 60-minute compressed source requests only chunk-window durations,
    never a whole-file decode; doubling duration does not increase the largest
    requested window."""
    import localmind.stt.chunking as chunking_mod

    state = {"max_t": 0.0}

    def make_run(duration_label):
        def fake_run(cmd, capture_output=True):
            if "-t" in cmd:
                i = cmd.index("-t")
                t = float(cmd[i + 1])
                if t > state["max_t"]:
                    state["max_t"] = t
                n = int(round(t * 16000))
                return subprocess.CompletedProcess(cmd, 0, stdout=b"\x00" * (n * 4), stderr=b"")
            # duration probe
            return subprocess.CompletedProcess(
                cmd, 1, stdout=b"",
                stderr=f"  Duration: {duration_label}, bitrate: 64 kb/s".encode("utf-8"),
            )
        return fake_run

    config = ChunkingConfig(chunk_duration_sec=30.0, overlap_sec=1.0)

    monkeypatch.setattr(chunking_mod.subprocess, "run", make_run("01:00:00.00"))
    src60 = FFmpegAudioSource(tmp_path / "a.m4a", 16000)
    list(iter_audio_chunks(src60, config))
    cap60 = state["max_t"]

    state["max_t"] = 0.0
    monkeypatch.setattr(chunking_mod.subprocess, "run", make_run("02:00:00.00"))
    src120 = FFmpegAudioSource(tmp_path / "b.m4a", 16000)
    list(iter_audio_chunks(src120, config))
    cap120 = state["max_t"]

    assert cap60 <= config.chunk_duration_sec + 1e-6
    assert cap120 <= config.chunk_duration_sec + 1e-6
    assert cap60 == cap120  # bounded by chunk window, not file length


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

def test_mock_transcriber_emits_ordered_bounded_segments(tmp_path):
    sr = 16000
    source = ArrayAudioSource(np.zeros(int(12 * sr), np.float32), sr)
    t = MockTranscriber()
    # Mock ignores the provisioner/tier id (no real model loaded).
    segs = t.transcribe(
        source, ChunkingConfig(chunk_duration_sec=5.0, overlap_sec=1.0), None, "mock"
    )
    # Overlap is trimmed: 3 chunks -> 3 non-overlapping segments.
    assert len(segs) == 3
    assert segs[-1].end == pytest.approx(12.0, abs=1e-3)
    # No overlap between consecutive segments.
    for a, b in zip(segs, segs[1:]):
        assert b.start >= a.end - 1e-6
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


def test_whisper_transcriber_converts_offsets_and_validates(monkeypatch, tmp_path):
    fake = _FakeMlxWhisper()
    _install_fake_mlx_whisper(monkeypatch, fake)

    sr = 16000
    source = ArrayAudioSource(np.zeros(int(12 * sr), np.float32), sr)
    config = ChunkingConfig(chunk_duration_sec=5.0, overlap_sec=1.0)
    t = WhisperTranscriber(language="en")
    segs = t.transcribe(source, config, _prov_with_model(tmp_path), "whisper-small")

    # 3 chunks; overlap trimming drops the early segment of chunks 1 and 2,
    # leaving 4 segments (chunk0's two + one late segment from each later chunk).
    assert fake.call_count == 3
    assert len(segs) == 4
    assert segs[2].start == pytest.approx(5.0)
    assert segs[2].end == pytest.approx(5.5)
    assert len({s.id for s in segs}) == 4
    validate_segments(segs, audio_duration_sec=12.0)
    # Provenance of the freshly resolved (verified) tier is recorded.
    assert t.last_provenance is not None
    assert t.last_provenance.model_id == "whisper-small"
    assert len(t.last_provenance.sha256) == 64


def test_whisper_transcriber_overlap_merge_regression(monkeypatch, tmp_path):
    """Each chunk returns an early (0-0.5) and a late (4.5-4.8) segment; the
    merged result must stay monotonic and drop the duplicated overlap content."""
    fake = _FakeMlxWhisper(segments=[
        {"start": 0.0, "end": 0.5, "text": "early"},
        {"start": 4.5, "end": 4.8, "text": "late"},
    ])
    _install_fake_mlx_whisper(monkeypatch, fake)

    sr = 16000
    source = ArrayAudioSource(np.zeros(int(13 * sr), np.float32), sr)
    config = ChunkingConfig(chunk_duration_sec=5.0, overlap_sec=1.0)
    segs = WhisperTranscriber().transcribe(
        source, config, _prov_with_model(tmp_path), "whisper-small"
    )

    # chunk0 keeps both; chunks 1 and 2 keep only their late segment.
    assert [s.start for s in segs] == pytest.approx([0.0, 4.5, 8.5, 12.5])
    assert len(segs) == 4
    validate_segments(segs, audio_duration_sec=13.0)


def test_whisper_transcriber_does_not_touch_network(monkeypatch, tmp_path):
    fake = _FakeMlxWhisper()
    _install_fake_mlx_whisper(monkeypatch, fake)

    def _no_network(*_a, **_k):
        raise AssertionError("transcription attempted a network connection")

    monkeypatch.setattr(socket, "socket", _no_network)
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    WhisperTranscriber().transcribe(
        source, ChunkingConfig(chunk_duration_sec=5.0), _prov_with_model(tmp_path), "whisper-small"
    )


def test_whisper_transcriber_rejects_non_dict_result(monkeypatch, tmp_path):
    _install_fake_mlx_whisper(monkeypatch, _FakeMlxWhisper(raw_result=["not", "a", "dict"]))
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(ValueError, match="dict"):
        WhisperTranscriber().transcribe(
            source, ChunkingConfig(chunk_duration_sec=5.0), _prov_with_model(tmp_path), "whisper-small"
        )


def test_whisper_transcriber_rejects_non_list_segments(monkeypatch, tmp_path):
    _install_fake_mlx_whisper(monkeypatch, _FakeMlxWhisper(raw_result={"segments": "nope"}))
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(ValueError, match="list"):
        WhisperTranscriber().transcribe(
            source, ChunkingConfig(chunk_duration_sec=5.0), _prov_with_model(tmp_path), "whisper-small"
        )


def test_whisper_transcriber_rejects_empty_text(monkeypatch, tmp_path):
    _install_fake_mlx_whisper(monkeypatch, _FakeMlxWhisper(fail_text=True))
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(SegmentValidationError, match="text"):
        WhisperTranscriber().transcribe(
            source, ChunkingConfig(chunk_duration_sec=5.0), _prov_with_model(tmp_path), "whisper-small"
        )


def test_whisper_transcriber_rejects_out_of_bounds_timestamps(monkeypatch, tmp_path):
    _install_fake_mlx_whisper(monkeypatch, _FakeMlxWhisper(out_of_bounds=True))
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(SegmentValidationError, match="exceeds audio duration"):
        WhisperTranscriber().transcribe(
            source, ChunkingConfig(chunk_duration_sec=5.0), _prov_with_model(tmp_path), "whisper-small"
        )


# --------------------------------------------------------------------------- #
# Backend boundary: the verified path can only come from the provisioner      #
# --------------------------------------------------------------------------- #

def test_whisper_transcriber_rejects_undeclared_tier_before_backend(monkeypatch, tmp_path):
    """A tier not in the manifest fast-fails before the backend runs."""
    backend = _FakeMlxWhisper()
    _install_fake_mlx_whisper(monkeypatch, backend)
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(ModelNotProvisionedError):
        WhisperTranscriber().transcribe(
            source, ChunkingConfig(chunk_duration_sec=5.0),
            _prov_with_empty_manifest(tmp_path), "whisper-small",
        )
    assert backend.call_count == 0


def test_whisper_transcriber_rejects_existing_file_not_in_manifest(monkeypatch, tmp_path):
    """An existing local file that is NOT declared in any manifest cannot reach
    the backend — the adapter resolves the tier through the provisioner, and an
    undeclared tier fast-fails. (Closes the forged-ResolvedTier vector.)"""
    backend = _FakeMlxWhisper()
    _install_fake_mlx_whisper(monkeypatch, backend)
    # An existing file on disk that no manifest references:
    (tmp_path / "evil.bin").write_bytes(b"not-a-provisioned-model")
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(ModelNotProvisionedError):
        WhisperTranscriber().transcribe(
            source, ChunkingConfig(chunk_duration_sec=5.0),
            _prov_with_empty_manifest(tmp_path), "evil",
        )
    assert backend.call_count == 0


def test_whisper_transcriber_rejects_forged_resolved_tier(monkeypatch, tmp_path):
    """The old forged-tier shape is gone: a hand-built ResolvedTier passed where
    a Provisioner is expected is a TypeError, and the backend is never called."""
    backend = _FakeMlxWhisper()
    _install_fake_mlx_whisper(monkeypatch, backend)
    (tmp_path / "evil.bin").write_bytes(b"not-a-provisioned-model")
    forged = ResolvedTier(
        tier="evil", model_id="evil",
        model_path=tmp_path / "evil.bin", sha256="0" * 64, quant_format="int4",
    )
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(TypeError):
        WhisperTranscriber().transcribe(
            source, ChunkingConfig(chunk_duration_sec=5.0), forged, "evil"
        )
    assert backend.call_count == 0


def test_whisper_transcriber_rejects_raw_path_as_provisioner(tmp_path):
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(TypeError):
        WhisperTranscriber().transcribe(
            source, ChunkingConfig(chunk_duration_sec=5.0), tmp_path / "not-a-provisioner", "x"
        )


def test_whisper_transcriber_rejects_tampered_weight_before_backend(monkeypatch, tmp_path):
    """A manifest-declared weight whose checksum no longer matches fast-fails
    (ChecksumMismatchError) before the backend runs."""
    backend = _FakeMlxWhisper()
    _install_fake_mlx_whisper(monkeypatch, backend)
    prov = _prov_with_model(tmp_path)
    # Tamper with the declared weight after the manifest was written (same length).
    weight = prov.model_dir / "whisper-small.mlmodel"
    original = weight.read_bytes()
    weight.write_bytes(b"X" * len(original) if original else b"X")
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    from localmind.provisioning import ChecksumMismatchError
    with pytest.raises(ChecksumMismatchError):
        WhisperTranscriber().transcribe(
            source, ChunkingConfig(chunk_duration_sec=5.0), prov, "whisper-small"
        )
    assert backend.call_count == 0


def test_whisper_transcriber_raises_clear_error_without_mlx_whisper(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    source = ArrayAudioSource(np.zeros(int(5 * 16000), np.float32), 16000)
    with pytest.raises(RuntimeError, match="mlx-whisper"):
        WhisperTranscriber().transcribe(
            source, ChunkingConfig(chunk_duration_sec=5.0), _prov_with_model(tmp_path), "whisper-small"
        )


def test_whisper_transcriber_real_smoke():
    """Real-backend smoke test: runs only when mlx_whisper and a provisioned
    Whisper model are available; otherwise skips. When it runs, it transcribes a
    tiny local sample and asserts nonempty timestamped segments + provenance."""
    try:
        import mlx_whisper  # noqa: F401
    except Exception as exc:
        pytest.skip(f"mlx_whisper not usable: {exc}")
    model_dir = Path("models")
    manifest = model_dir / "models.json"
    if not manifest.exists():
        pytest.skip("no provisioned models/ directory for real Whisper smoke test")

    # Pick any whisper-kind tier declared in the manifest.
    import json as _json
    declared = _json.loads(manifest.read_text())["models"]
    whisper_tiers = [m["model_id"] for m in declared if m.get("kind") == "whisper"]
    if not whisper_tiers:
        pytest.skip("no whisper-kind tier provisioned in models/models.json")
    tier = whisper_tiers[0]

    import tempfile
    wav = _write_sine_wav(Path(tempfile.mkdtemp()) / "smoke.wav", duration_sec=2.0, sr=16000)

    prov = Provisioner(model_dir)
    source = audio_source_from_path(wav, target_sample_rate=16000)
    t = WhisperTranscriber()
    segments = t.transcribe(source, ChunkingConfig(chunk_duration_sec=5.0), prov, tier)
    assert len(segments) >= 1
    validate_segments(segments, audio_duration_sec=source.duration_sec)
    assert t.last_provenance is not None
    assert t.last_provenance.model_id == tier
