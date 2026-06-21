"""Bounded audio chunking for long-audio transcription (AC-2.1).

Long audio is transcribed in overlapping chunks so peak audio-buffer memory does
not scale with the full file length. Chunking is **mandatory** for audio longer
than :data:`MAX_UNCHUNKED_SEC`; a configuration that disables chunking on a long
file is rejected (AC-2.1 negative test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

# Audio longer than this must be chunked. Keeps peak decode/buffer memory bounded
# regardless of total file length.
MAX_UNCHUNKED_SEC = 300.0  # 5 minutes


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for bounded audio chunking."""

    chunk_duration_sec: float = 30.0
    overlap_sec: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        if self.chunk_duration_sec <= 0:
            raise ValueError("chunk_duration_sec must be positive")
        if self.overlap_sec < 0:
            raise ValueError("overlap_sec must be non-negative")
        if self.overlap_sec >= self.chunk_duration_sec:
            raise ValueError("overlap_sec must be less than chunk_duration_sec")


def chunk_audio(
    samples: np.ndarray,
    sample_rate: int,
    config: ChunkingConfig = ChunkingConfig(),
) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield ``(start_sec, chunk_samples)`` pairs over the audio.

    Each chunk is at most ``chunk_duration_sec`` long, so the largest buffer
    held at once is bounded by the chunk size, not the file length. When
    ``config.enabled`` is False, raises for audio longer than
    :data:`MAX_UNCHUNKED_SEC` (AC-2.1 negative); short audio is yielded as a
    single chunk.
    """
    if samples.ndim != 1:
        raise ValueError("chunk_audio expects mono 1-D samples")

    duration_sec = float(samples.size) / float(sample_rate)

    if not config.enabled:
        if duration_sec > MAX_UNCHUNKED_SEC:
            raise ValueError(
                f"chunking is disabled but audio is {duration_sec:.1f}s (> "
                f"{MAX_UNCHUNKED_SEC}s); chunking is mandatory for long audio"
            )
        yield 0.0, samples
        return

    chunk_n = int(round(config.chunk_duration_sec * sample_rate))
    hop_n = max(1, chunk_n - int(round(config.overlap_sec * sample_rate)))

    start_sample = 0
    while start_sample < samples.size:
        end_sample = min(start_sample + chunk_n, samples.size)
        chunk = samples[start_sample:end_sample]
        start_sec = start_sample / float(sample_rate)
        yield start_sec, chunk
        if end_sample >= samples.size:
            break
        start_sample += hop_n
