"""Audio fixtures for benchmarking.

Two kinds of fixture:

1. **Synthetic fixtures** (:func:`generate_synthetic_wav`) — small, deterministic
   sine-tone WAVs generated on the fly for unit tests. No network, no committed
   binaries.

2. **Benchmark case descriptors** (:data:`BENCHMARK_CASES`) — metadata for the
   canonical 10/30/60-minute cases the plan requires. The real audio for these
   long cases is **provisioned out-of-band** (see ``docs/benchmark.md``) exactly
   like model weights; it is far too large to commit. A descriptor records the
   case id, expected duration, sample rate, channels, and the relative path
   where the provisioned file is expected, so the benchmark harness can locate
   it (and skip cleanly when absent).
"""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class BenchmarkCase:
    """Metadata for one canonical benchmark case (10/30/60 min)."""

    case_id: str
    duration_min: int
    audio_rel_path: str  # relative to the benchmark fixtures directory
    sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = 1
    description: str = ""

    @property
    def duration_sec(self) -> float:
        return float(self.duration_min * 60)


# Canonical benchmark cases. Real audio provisioned out-of-band; the harness
# resolves audio_rel_path under the fixtures directory and skips if absent.
BENCHMARK_CASES: List[BenchmarkCase] = [
    BenchmarkCase(
        case_id="bm-10min",
        duration_min=10,
        audio_rel_path="audio/bm-10min.m4a",
        description="10-minute meeting-style fixture for RTF/memory benchmarking",
    ),
    BenchmarkCase(
        case_id="bm-30min",
        duration_min=30,
        audio_rel_path="audio/bm-30min.m4a",
        description="30-minute meeting-style fixture",
    ),
    BenchmarkCase(
        case_id="bm-60min",
        duration_min=60,
        audio_rel_path="audio/bm-60min.m4a",
        description="60-minute long-meeting fixture; exercises bounded chunking",
    ),
]


def generate_synthetic_wav(
    path,
    duration_sec: float,
    sample_rate: int = TARGET_SAMPLE_RATE,
    freq: float = 440.0,
    seed: int = 0,
) -> Path:
    """Write a deterministic synthetic mono 16-bit sine WAV.

    A small amount of seeded noise is added so the signal is not a pure tone
    (keeps resamplers honest). Intended for unit tests, not the 10/30/60-min
    benchmark cases.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = int(round(duration_sec * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    noise = 0.01 * rng.standard_normal(n)
    samples = np.clip(tone + noise, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return p


def fixture_digest(path) -> str:
    """SHA-256 of a fixture file (for integrity / reproducibility records)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FixtureNotProvisionedError(Exception):
    """A required benchmark fixture audio file is not present locally."""


def fixture_path(fixtures_dir, case: "BenchmarkCase") -> Path:
    """Resolve where a benchmark case's audio is expected on disk."""
    return Path(fixtures_dir) / case.audio_rel_path


def is_fixture_provisioned(fixtures_dir, case: "BenchmarkCase") -> bool:
    """True if the benchmark case's audio file exists locally."""
    return fixture_path(fixtures_dir, case).is_file()


def require_fixture(fixtures_dir, case: "BenchmarkCase") -> Path:
    """Return the path to a provisioned benchmark fixture, or fail clearly.

    Real 10/30/60-minute audio is provisioned out-of-band (see
    ``docs/benchmark.md``). A benchmark run calls this so a missing fixture
    fails fast with an explicit message rather than a silent skip or a crash.
    """
    p = fixture_path(fixtures_dir, case)
    if not p.is_file():
        raise FixtureNotProvisionedError(
            f"benchmark fixture not provisioned: {case.case_id} expected at {p}; "
            f"provision it out-of-band (see docs/benchmark.md)"
        )
    return p
