"""Acceptance tests for AC-1: decode local audio to normalized PCM.

WAV paths are fully exercised (stdlib backend, no external deps). Compressed
formats (m4a/mp3) are decoded via the ffmpeg backend, which resolves a binary
from the system PATH or the bundled imageio-ffmpeg package; positive cases run
when any ffmpeg binary is available, and the graceful-unavailable behavior is
always asserted.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from localmind.audio import (
    DecodeError,
    DecoderUnavailableError,
    UnsupportedFormatError,
    decode_audio,
)
from localmind.audio import decode as decode_mod
from localmind.audio.decode import TARGET_SAMPLE_RATE

ffmpeg_available = decode_mod._ffmpeg_available()


# --------------------------------------------------------------------------- #
# Positive tests (expected to PASS)                                            #
# --------------------------------------------------------------------------- #

def test_decode_16k_mono_wav_matches_expected_sample_count(tmp_path, sine_wave, make_wav):
    duration = 1.0
    samples = sine_wave(duration, sample_rate=16000)
    wav_path = make_wav(tmp_path / "tone.wav", samples, sample_rate=16000)

    decoded = decode_audio(wav_path)

    assert decoded.sample_rate == TARGET_SAMPLE_RATE == 16000
    assert decoded.original_format == "wav"
    assert decoded.samples.ndim == 1
    assert decoded.samples.dtype == np.float32
    # 1.0 s at 16 kHz -> 16000 samples (no resampling needed).
    assert decoded.samples.size == 16000
    assert decoded.duration_sec == pytest.approx(1.0, abs=1e-3)


def test_decode_stereo_44100_wav_is_downmixed_and_resampled(tmp_path, sine_wave, make_wav):
    duration = 1.0
    mono = sine_wave(duration, sample_rate=44100)
    # Build a 2-channel signal (identical channels -> averaging is lossless).
    stereo = np.stack([mono, mono], axis=1)
    wav_path = make_wav(tmp_path / "stereo.wav", stereo, sample_rate=44100, nchannels=2)

    decoded = decode_audio(wav_path)

    assert decoded.sample_rate == 16000
    assert decoded.samples.ndim == 1  # mono
    # 1.0 s of audio should yield ~16000 samples after resampling.
    assert decoded.samples.size == pytest.approx(16000, abs=2)
    assert decoded.duration_sec == pytest.approx(1.0, abs=5e-3)
    # Downmixed identical channels preserve amplitude (within float tolerance).
    assert np.max(np.abs(decoded.samples)) > 0.3


def test_decode_wav_reports_true_duration(tmp_path, sine_wave, make_wav):
    duration = 2.5
    samples = sine_wave(duration, sample_rate=16000)
    wav_path = make_wav(tmp_path / "two-half.wav", samples, sample_rate=16000)

    decoded = decode_audio(wav_path)

    assert decoded.duration_sec == pytest.approx(2.5, abs=1e-2)
    assert decoded.samples.size == 2.5 * 16000


# --------------------------------------------------------------------------- #
# Negative tests (expected to FAIL / be rejected)                              #
# --------------------------------------------------------------------------- #

def test_zero_byte_file_is_rejected_without_crash(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")

    with pytest.raises(DecodeError):
        decode_audio(empty)


def test_truncated_corrupt_wav_is_rejected(tmp_path):
    # A file that starts with a RIFF header but is then garbage -> wave raises.
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt not really a wav")

    with pytest.raises(DecodeError):
        decode_audio(corrupt)


def test_random_bytes_not_wav_is_rejected(tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(bytes(range(256)) * 4)

    with pytest.raises(DecodeError):
        decode_audio(junk)


def test_unsupported_format_flac_is_rejected(tmp_path):
    flac = tmp_path / "audio.flac"
    flac.write_bytes(b"some bytes")

    with pytest.raises(UnsupportedFormatError):
        decode_audio(flac)


def test_extensionless_file_is_rejected(tmp_path):
    noext = tmp_path / "audiofile"
    noext.write_bytes(b"RIFF")

    with pytest.raises(UnsupportedFormatError):
        decode_audio(noext)


def test_wav_with_zero_frames_is_rejected(tmp_path, make_wav):
    # Valid WAV header but zero frames -> empty audio must be rejected.
    wav_path = tmp_path / "silent.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"")

    with pytest.raises(DecodeError):
        decode_audio(wav_path)


# --------------------------------------------------------------------------- #
# Compressed-format backend (ffmpeg via imageio-ffmpeg fallback)              #
# --------------------------------------------------------------------------- #

def _make_sine_wav(path: Path, duration: float = 1.0, sr: int = 16000) -> Path:
    n = int(duration * sr)
    t = np.arange(n) / sr
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((samples * 32767).astype("<i2").tobytes())
    return path


def _transcode(src: Path, dst: Path) -> Path:
    import subprocess
    exe = decode_mod._ffmpeg_exe()
    assert exe is not None, "ffmpeg required for transcode fixture"
    subprocess.run(
        [exe, "-loglevel", "error", "-y", "-i", str(src), str(dst)], check=True
    )
    return dst


@pytest.mark.skipif(not ffmpeg_available, reason="no ffmpeg binary available")
def test_decode_mp3_positive(tmp_path):
    wav = _make_sine_wav(tmp_path / "tone.wav")
    mp3 = _transcode(wav, tmp_path / "tone.mp3")

    decoded = decode_audio(mp3)
    assert decoded.sample_rate == 16000
    assert decoded.original_format == "mp3"
    assert decoded.samples.ndim == 1
    assert decoded.samples.dtype == np.float32
    assert decoded.duration_sec == pytest.approx(1.0, abs=0.05)
    assert np.max(np.abs(decoded.samples)) > 0.1  # signal survived round-trip


@pytest.mark.skipif(not ffmpeg_available, reason="no ffmpeg binary available")
def test_decode_m4a_positive(tmp_path):
    wav = _make_sine_wav(tmp_path / "tone.wav")
    m4a = _transcode(wav, tmp_path / "tone.m4a")

    decoded = decode_audio(m4a)
    assert decoded.sample_rate == 16000
    assert decoded.original_format == "m4a"
    assert decoded.samples.ndim == 1
    assert decoded.duration_sec == pytest.approx(1.0, abs=0.05)
    assert np.max(np.abs(decoded.samples)) > 0.1


@pytest.mark.skipif(not ffmpeg_available, reason="no ffmpeg binary available")
def test_decode_mp3_and_m4a_preserve_signal_frequency(tmp_path):
    """Both compressed decodes should preserve the 440 Hz tone (robust to encoder priming)."""
    wav = _make_sine_wav(tmp_path / "tone.wav", duration=1.0)

    for ext in ("mp3", "m4a"):
        comp = _transcode(wav, tmp_path / f"tone.{ext}")
        dec = decode_audio(comp)
        # Duration is ~1s (encoder priming may shift it slightly).
        assert dec.duration_sec == pytest.approx(1.0, abs=0.1)
        # The dominant spectral peak should be at 440 Hz, robust to priming delay.
        spectrum = np.abs(np.fft.rfft(dec.samples.astype(np.float64)))
        freqs = np.fft.rfftfreq(dec.samples.size, d=1.0 / dec.sample_rate)
        peak_freq = freqs[np.argmax(spectrum)]
        assert peak_freq == pytest.approx(440.0, abs=10.0)


def test_compressed_format_unavailable_raises_not_crash(tmp_path, monkeypatch):
    """When no ffmpeg binary is resolvable, compressed formats fail gracefully."""
    monkeypatch.setattr(decode_mod, "shutil", type("S", (), {"which": staticmethod(lambda _: None)})())
    monkeypatch.setattr(decode_mod, "imageio_ffmpeg", None)
    m4a = tmp_path / "clip.m4a"
    m4a.write_bytes(b"fake m4a bytes")

    with pytest.raises(DecoderUnavailableError):
        decode_audio(m4a)


def test_ffmpeg_backend_uses_imageio_fallback_when_system_absent(tmp_path, monkeypatch):
    """imageio-ffmpeg is used when system ffmpeg is missing."""
    if not ffmpeg_available:
        pytest.skip("no ffmpeg binary available to exercise the fallback")

    class _FakeShutil:
        @staticmethod
        def which(_cmd):
            return None  # pretend system ffmpeg is absent

    monkeypatch.setattr(decode_mod, "shutil", _FakeShutil())
    exe = decode_mod._ffmpeg_exe()
    assert exe is not None  # imageio-ffmpeg fallback resolved a binary

    wav = _make_sine_wav(tmp_path / "tone.wav")
    mp3 = _transcode(wav, tmp_path / "tone.mp3")
    decoded = decode_audio(mp3)
    assert decoded.original_format == "mp3"
    assert decoded.samples.size > 0

