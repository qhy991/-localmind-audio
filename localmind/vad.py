"""Voice Activity Detection (VAD) — locate speech segments in decoded PCM.

Energy-based VAD with hysteresis. Pure NumPy, no external dependencies, fully
deterministic: the same audio + config always yields the same speech segments.
This keeps VAD consistent with the project's stdlib-first, zero-network,
reproducible contract.

Algorithm
---------
1. Frame the mono PCM into short windows (frame_ms, hop_ms) and compute the
   RMS energy of each frame.
2. Estimate a robust background-noise floor from the quietest frames (a low
   percentile of per-frame energy), so the threshold adapts to the recording
   rather than being a fixed dB value.
3. Walk the frames with hysteresis: a frame *enters* speech when its energy
   rises ``speech_rise_db`` above the noise floor, and *leaves* speech when it
   falls below ``speech_fall_db``. Hysteresis prevents rapid on/off flapping at
   the boundary of speech.
4. Post-process: discard speech bursts shorter than ``min_speech_sec`` and
   merge gaps shorter than ``min_silence_sec`` (brief pauses inside an
   utterance should not split it).

The result is a list of ``[start_sec, end_sec]`` speech intervals covering the
spoken regions of the audio. Silence is everything in between, which a caller
can skip (e.g. transcribe only speech, or just report gaps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

__all__ = ["VadConfig", "SpeechSegment", "detect_speech"]


@dataclass(frozen=True)
class VadConfig:
    """Configuration for energy-based VAD with hysteresis."""

    frame_ms: float = 30.0
    hop_ms: float = 10.0
    # Background-noise estimate = this percentile of per-frame energies (quiet
    # frames). Robust to a few loud speech frames.
    noise_percentile: float = 10.0
    # Hysteresis thresholds, in dB above the noise floor.
    speech_rise_db: float = 15.0
    speech_fall_db: float = 8.0
    min_speech_sec: float = 0.25
    min_silence_sec: float = 0.30
    # Energies at or below this absolute RMS are pure silence regardless of
    # noise floor (guards against near-zero recordings).
    floor_rms: float = 1e-5

    def __post_init__(self):
        if self.hop_ms <= 0 or self.frame_ms <= 0:
            raise ValueError("frame_ms and hop_ms must be positive")
        if self.hop_ms > self.frame_ms:
            raise ValueError("hop_ms must not exceed frame_ms")
        if not 0.0 <= self.noise_percentile <= 100.0:
            raise ValueError("noise_percentile must be in [0, 100]")
        if self.speech_fall_db > self.speech_rise_db:
            raise ValueError("speech_fall_db must not exceed speech_rise_db")
        if self.min_speech_sec < 0 or self.min_silence_sec < 0:
            raise ValueError("min_speech_sec / min_silence_sec must be non-negative")


@dataclass(frozen=True)
class SpeechSegment:
    """One detected speech interval ``[start_sec, end_sec]`` (end exclusive)."""

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def _frame_energies(samples: np.ndarray, sample_rate: int, cfg: VadConfig) -> np.ndarray:
    """Per-frame RMS energy. Frames are stepped by hop; trailing samples that
    do not fill a frame are ignored (the last partial frame contributes little
    to VAD decisions and is covered by the duration clamp)."""
    if samples.size == 0:
        return np.zeros(0, dtype=np.float64)
    frame = max(1, int(round(cfg.frame_ms / 1000.0 * sample_rate)))
    hop = max(1, int(round(cfg.hop_ms / 1000.0 * sample_rate)))
    n = (samples.size - frame) // hop + 1
    if n <= 0:
        # Signal shorter than one frame: use whatever energy it has, in one bin.
        rms = float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))
        return np.array([rms], dtype=np.float64)
    idx = np.arange(n)[:, None] * hop + np.arange(frame)[None, :]
    frames = np.asarray(samples, dtype=np.float64)[idx]  # (n, frame)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def detect_speech(
    samples: np.ndarray, sample_rate: int, config: VadConfig | None = None
) -> List[SpeechSegment]:
    """Return the spoken intervals in ``samples`` (mono float PCM).

    Parameters
    ----------
    samples : np.ndarray
        Mono float PCM (any float dtype), as produced by ``decode_audio``.
    sample_rate : int
        Sample rate of ``samples`` in Hz.
    config : VadConfig, optional
        VAD configuration. Defaults to :class:`VadConfig`.

    Returns
    -------
    list[SpeechSegment]
        Non-overlapping speech intervals in ascending ``start_sec`` order.
        Empty if the audio contains no speech above the noise floor.
    """
    cfg = config or VadConfig()
    samples = np.asarray(samples, dtype=np.float64)
    duration = samples.size / sample_rate if sample_rate > 0 else 0.0

    energies = _frame_energies(samples, sample_rate, cfg)
    if energies.size == 0:
        return []

    # dB per frame, clamped at the silence floor so log(0) cannot occur.
    safe = np.maximum(energies, cfg.floor_rms)
    db = 20.0 * np.log10(safe)

    noise_db = float(np.percentile(db, cfg.noise_percentile))
    peak_db = float(db.max())
    # If the recording has little dynamic range (peak barely above the noise
    # floor, e.g. near-silent audio), there is no speech to find. Detect via the
    # gap between peak and floor: a real recording has speech well above the
    # silence floor; a flat/near-silent one does not.
    if peak_db - noise_db < cfg.speech_rise_db:
        return []
    rise = noise_db + cfg.speech_rise_db
    fall = noise_db + cfg.speech_fall_db

    hop_sec = cfg.hop_ms / 1000.0
    frame_start = np.arange(energies.size) * hop_sec  # wall-clock time of each frame

    # Hysteresis state machine over frames.
    in_speech = False
    seg_start = 0.0
    raw: List[tuple] = []
    for t, level in zip(frame_start, db):
        if not in_speech:
            if level >= rise:
                in_speech = True
                seg_start = t
        else:
            if level < fall:
                in_speech = False
                raw.append((seg_start, t))
    if in_speech:
        raw.append((seg_start, frame_start[-1] + cfg.frame_ms / 1000.0))

    # Drop speech bursts shorter than min_speech_sec.
    kept = [(s, e) for s, e in raw if (e - s) >= cfg.min_speech_sec]
    if not kept:
        return []

    # Merge speech intervals separated by a gap shorter than min_silence_sec
    # (a short pause inside one utterance should not split it).
    merged: List[tuple] = [kept[0]]
    for s, e in kept[1:]:
        ps, pe = merged[-1]
        if s - pe <= cfg.min_silence_sec:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    segs = [SpeechSegment(start_sec=float(s), end_sec=min(float(e), duration))
            for s, e in merged if e > s]
    # Final pass: merging can create a segment still under min_speech after the
    # duration clamp; drop those.
    return [seg for seg in segs if seg.duration_sec > 0]
