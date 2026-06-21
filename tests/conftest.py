"""Shared pytest fixtures: synthetic WAV generation and dummy model weights.

Fixtures here are generated in ``tmp_path`` so tests need no committed binary
artifacts and no network access.
"""

from __future__ import annotations

import hashlib
import struct
import wave
from pathlib import Path

import numpy as np
import pytest


def write_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    nchannels: int = 1,
    sampwidth: int = 2,
) -> Path:
    """Write a 1-D or 2-D sample array to a PCM WAV file.

    ``samples`` is float in ~[-1, 1]. For 2-D input the second axis is the
    channel axis. Output is interleaved PCM.
    """
    if samples.ndim == 1:
        mono = samples.reshape(-1, 1)
    else:
        mono = samples
    if mono.shape[1] != nchannels:
        raise ValueError(f"channel mismatch: got {mono.shape[1]} want {nchannels}")

    max_val = (1 << (8 * sampwidth - 1)) - 1
    ints = np.clip(mono * max_val, -max_val, max_val).astype(np.int64)
    # Interleave channels.
    interleaved = ints.reshape(-1)
    raw = b"".join(int(v).to_bytes(sampwidth, "little", signed=True) for v in interleaved)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
    return path


@pytest.fixture
def sine_wave():
    """Return a function building a mono float32 sine of given duration/rate."""
    def _make(duration_sec: float, sample_rate: int = 16000, freq: float = 440.0) -> np.ndarray:
        n = int(round(duration_sec * sample_rate))
        t = np.arange(n, dtype=np.float64) / sample_rate
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return _make


@pytest.fixture
def make_wav():
    return write_wav


@pytest.fixture
def dummy_weight(tmp_path: Path):
    """Return a function writing a deterministic dummy weight file; returns (path, size, sha256)."""
    def _make(name: str, content: bytes) -> tuple:
        p = tmp_path / "models" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        return p, len(content), digest
    return _make


@pytest.fixture
def manifest_writer(tmp_path: Path):
    """Return a function that writes a models.json manifest into tmp_path/models/."""
    def _write(entries):
        import json
        d = tmp_path / "models"
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "1",
            "models": entries,
        }
        (d / "models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return tmp_path / "models"
    return _write
