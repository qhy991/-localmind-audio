"""Decode local audio files into a normalized PCM representation.

Normalization contract (matches what downstream Whisper expects):

* **Mono** — multi-channel input is averaged to one channel.
* **32-bit float**, range roughly ``[-1.0, 1.0]``.
* **16 kHz** sample rate — input is resampled via linear interpolation.

Supported containers:

* ``.wav`` — decoded with the stdlib :mod:`wave` module (no external deps).
* ``.m4a`` / ``.mp3`` / ``.aac`` — decoded via an ``ffmpeg`` subprocess backend.
  If ``ffmpeg`` is not installed, :class:`DecoderUnavailableError` is raised
  (graceful, not a crash). These formats are *not* decoded at runtime over the
  network — ffmpeg reads the local file only.

Any other extension raises :class:`UnsupportedFormatError`. Corrupt, truncated,
or empty files raise :class:`DecodeError` rather than crashing or silently
yielding an empty buffer.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from localmind.audio.errors import (
    DecodeError,
    DecoderUnavailableError,
    UnsupportedFormatError,
)

TARGET_SAMPLE_RATE = 16000

# Formats handled by the ffmpeg backend (compressed containers).
_FFMPEG_FORMATS = frozenset({"m4a", "mp3", "aac"})
# Formats handled by the stdlib wave backend.
_WAVE_FORMATS = frozenset({"wav"})
# The full set of accepted extensions; anything else is unsupported.
_SUPPORTED_FORMATS = _WAVE_FORMATS | _FFMPEG_FORMATS


@dataclass(frozen=True)
class DecodedAudio:
    """Normalized PCM audio ready for the transcription stage."""

    samples: np.ndarray  # float32, mono, ~[-1, 1]
    sample_rate: int
    duration_sec: float
    original_format: str  # e.g. "wav", "m4a", "mp3"

    def __post_init__(self):
        if self.samples.ndim != 1:
            raise ValueError("DecodedAudio.samples must be mono (1-D)")
        if self.sample_rate <= 0:
            raise ValueError("DecodedAudio.sample_rate must be positive")


def decode_audio(path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> DecodedAudio:
    """Decode a local audio file to normalized mono float32 PCM.

    Raises
    ------
    UnsupportedFormatError
        If the file extension is not in :data:`_SUPPORTED_FORMATS`.
    DecoderUnavailableError
        If a compressed format (m4a/mp3/aac) is requested but ffmpeg is missing.
    DecodeError
        If the file is corrupt, truncated, empty, or unreadable.
    """
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    if not ext:
        raise UnsupportedFormatError(f"unsupported audio format: {p.name!r} has no extension")
    if ext not in _SUPPORTED_FORMATS:
        raise UnsupportedFormatError(f"unsupported audio format: .{ext}")
    if ext in _WAVE_FORMATS:
        return _decode_wav(p, target_sample_rate)
    return _decode_via_ffmpeg(p, target_sample_rate, ext)


# --------------------------------------------------------------------------- #
# WAV backend (stdlib)                                                        #
# --------------------------------------------------------------------------- #

def _decode_wav(path: Path, target_rate: int) -> DecodedAudio:
    try:
        with wave.open(str(path), "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
    except (wave.Error, EOFError, OSError, struct.error) as exc:
        raise DecodeError(f"failed to read WAV {path}: {exc}") from exc

    if not raw or nframes == 0:
        raise DecodeError(f"empty WAV audio (no frames): {path}")

    samples = _bytes_to_float(raw, sampwidth)
    samples = _to_mono(samples, nchannels)
    samples = _resample_linear(samples, framerate, target_rate)

    if samples.size == 0:
        raise DecodeError(f"decoded to empty audio: {path}")

    duration = float(samples.size) / float(target_rate)
    return DecodedAudio(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate=target_rate,
        duration_sec=duration,
        original_format="wav",
    )


def _bytes_to_float(raw: bytes, sampwidth: int) -> np.ndarray:
    """Convert raw PCM bytes to a 2-D float64 array of shape (n_frames, n_channels)."""
    if sampwidth == 2:
        ints = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        return ints / 32768.0
    if sampwidth == 4:
        ints = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        return ints / 2147483648.0
    if sampwidth == 1:
        # 8-bit PCM is unsigned, centered at 128.
        u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        return (u8 - 128.0) / 128.0
    if sampwidth == 3:
        # 24-bit little-endian signed PCM, packed.
        b = np.frombuffer(raw, dtype=np.uint8)
        if b.size % 3 != 0:
            raise DecodeError("corrupt 24-bit PCM: byte count not divisible by 3")
        b = b.reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        # Sign-extend the 24-bit value.
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        return ints.astype(np.float64) / 8388608.0
    raise DecodeError(f"unsupported WAV sample width: {sampwidth} bytes")


def _to_mono(samples: np.ndarray, nchannels: int) -> np.ndarray:
    """Average multi-channel samples down to mono."""
    if nchannels == 1:
        return np.asarray(samples, dtype=np.float64)
    if samples.size % nchannels != 0:
        raise DecodeError("corrupt WAV: frame count not divisible by channel count")
    frames = samples.reshape(-1, nchannels)
    return frames.mean(axis=1)


def _resample_linear(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample 1-D signal from sr_in to sr_out via linear interpolation."""
    if sr_in == sr_out:
        return np.asarray(samples, dtype=np.float64)
    if samples.size <= 1:
        return np.asarray(samples, dtype=np.float64)
    n_out = int(round(samples.size * sr_out / sr_in))
    if n_out <= 0:
        return np.empty(0, dtype=np.float64)
    x_in = np.arange(samples.size, dtype=np.float64)
    x_out = np.linspace(0.0, samples.size - 1, n_out)
    return np.interp(x_out, x_in, samples)


# --------------------------------------------------------------------------- #
# ffmpeg backend (m4a / mp3 / aac)                                            #
# --------------------------------------------------------------------------- #

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _decode_via_ffmpeg(path: Path, target_rate: int, fmt: str) -> DecodedAudio:
    if not _ffmpeg_available():
        raise DecoderUnavailableError(
            f"ffmpeg is not installed; cannot decode .{fmt}. "
            f"Install ffmpeg (out-of-band) to enable .{fmt} decoding. File: {path}"
        )
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(target_rate),
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except OSError as exc:
        raise DecodeError(f"failed to invoke ffmpeg for {path}: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise DecodeError(f"ffmpeg failed for {path}: {stderr}")
    raw = proc.stdout
    if not raw:
        raise DecodeError(f"ffmpeg produced no audio for {path}")
    samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if samples.size == 0:
        raise DecodeError(f"ffmpeg produced empty audio for {path}")
    duration = float(samples.size) / float(target_rate)
    return DecodedAudio(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate=target_rate,
        duration_sec=duration,
        original_format=fmt,
    )
