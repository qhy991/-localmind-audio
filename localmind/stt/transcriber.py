"""Transcriber interface, tier selection, and implementations (AC-2 / AC-2.1).

* :class:`Transcriber` — abstract interface.
* :func:`select_tier` — resolve a model tier through the provisioner so a
  missing tier fast-fails with ``ModelNotProvisionedError`` and **never**
  downloads.
* :class:`MockTranscriber` — deterministic, dependency-free implementation for
  tests: emits ordered timestamped segments bounded by audio duration.
* :class:`WhisperTranscriber` — the real adapter over ``mlx-whisper``. It is a
  documented stub here; finalised with provisioned weights in a later milestone.
  Calling it without ``mlx-whisper`` installed raises a clear error rather than
  silently degrading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from localmind.provisioning.provisioner import Provisioner
from localmind.stt.segment import TranscriptSegment

ProgressCallback = Callable[[float], None]


def select_tier(provisioner: Provisioner, tier_model_id: str) -> Path:
    """Resolve a model tier to a verified on-disk path.

    Goes through ``Provisioner.require_model`` so an unprovisioned tier raises
    ``ModelNotProvisionedError`` (no network download). This is the AC-2
    "missing tier fast-fail without download" guarantee.
    """
    return provisioner.require_model(tier_model_id)


class Transcriber(ABC):
    """Abstract speech-to-text interface."""

    @abstractmethod
    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        model_path: Path,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        """Transcribe mono PCM samples into ordered timestamped segments."""
        raise NotImplementedError


class MockTranscriber(Transcriber):
    """Deterministic, dependency-free transcriber for tests.

    Emits one segment per ``segment_duration_sec`` of audio, with strictly
    increasing timestamps bounded by the audio duration. Output passes
    :func:`localmind.stt.segment.validate_segments`.
    """

    def __init__(self, segment_duration_sec: float = 5.0, label: str = "mock"):
        if segment_duration_sec <= 0:
            raise ValueError("segment_duration_sec must be positive")
        self.segment_duration_sec = segment_duration_sec
        self.label = label

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        model_path: Path,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        if samples.ndim != 1:
            raise ValueError("MockTranscriber expects mono 1-D samples")
        total_sec = float(samples.size) / float(sample_rate)
        segments: List[TranscriptSegment] = []
        t = 0.0
        idx = 0
        while t < total_sec - 1e-9:
            end = min(t + self.segment_duration_sec, total_sec)
            segments.append(
                TranscriptSegment(
                    id=f"seg-{idx:04d}",
                    start=t,
                    end=end,
                    text=f"{self.label} segment {idx}",
                    confidence=0.9,
                )
            )
            t = end
            idx += 1
            if on_progress is not None:
                on_progress(min(1.0, t / total_sec if total_sec > 0 else 1.0))
        return segments


class WhisperTranscriber(Transcriber):
    """Real Whisper transcription via mlx-whisper.

    Finalised in a later milestone with provisioned weights and a 3.11+ venv.
    Until then, calling :meth:`transcribe` without ``mlx_whisper`` installed
    raises a clear ``RuntimeError`` — it never silently falls back to a cloud
    API or a download.
    """

    def __init__(self, language: Optional[str] = None):
        self.language = language

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        model_path: Path,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "mlx-whisper is not installed; provision the STT backend "
                "(see docs/provisioning.md) before using WhisperTranscriber"
            ) from exc

        # Real integration (mlx_whisper.transcribe over chunked audio) is
        # implemented once weights are provisioned and the venv is 3.11+.
        raise NotImplementedError(
            "WhisperTranscriber.transcribe is not yet implemented; "
            "use MockTranscriber for tests until the STT milestone"
        )
