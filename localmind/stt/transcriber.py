"""Transcriber interface, tier resolution, and implementations.

* :class:`Transcriber` — abstract interface over a bounded audio source.
* :func:`resolve_tier` — resolve a model tier through the provisioner so a
  missing tier fast-fails with ``ModelNotProvisionedError`` and **never**
  downloads, returning a :class:`ResolvedTier` carrying the provenance fields
  (tier, model_id, model_path, sha256) that downstream run records emit.
* :class:`MockTranscriber` — deterministic, dependency-free transcriber for
  tests: emits ordered timestamped segments bounded by the audio duration.
* :class:`WhisperTranscriber` — the real adapter over ``mlx-whisper``. It
  transcribes chunk-by-chunk from a bounded source, converts backend segments to
  :class:`TranscriptSegment`, offsets chunk-relative timestamps to file time,
  normalizes ids, and validates the result before returning. Without
  ``mlx-whisper`` installed it raises a clear error (never a silent cloud
  fallback or download).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from localmind.provisioning.provisioner import Provisioner
from localmind.stt.chunking import AudioSource, ChunkingConfig, iter_audio_chunks
from localmind.stt.segment import TranscriptSegment, validate_segments

ProgressCallback = Callable[[float], None]


@dataclass(frozen=True)
class ResolvedTier:
    """A verified model tier plus the provenance fields a run record emits."""

    tier: str
    model_id: str
    model_path: Path
    sha256: str
    quant_format: str


def resolve_tier(provisioner: Provisioner, tier_model_id: str) -> ResolvedTier:
    """Resolve a model tier to a verified path plus provenance.

    Goes through ``Provisioner.require_model`` so an unprovisioned tier raises
    ``ModelNotProvisionedError`` (no network download). The manifest entry's
    sha256/quant_format are carried into the returned provenance object.
    """
    manifest = provisioner.load_manifest()
    try:
        entry = manifest.by_id(tier_model_id)
    except KeyError:
        from localmind.provisioning.errors import ModelNotProvisionedError

        raise ModelNotProvisionedError(
            f"model not provisioned: {tier_model_id!r} is not declared in the manifest"
        ) from None
    path = provisioner.require_model(tier_model_id)  # verifies size + sha256
    return ResolvedTier(
        tier=tier_model_id,
        model_id=tier_model_id,
        model_path=path,
        sha256=entry.sha256,
        quant_format=entry.quant_format,
    )


class Transcriber(ABC):
    """Abstract speech-to-text interface over a bounded audio source."""

    @abstractmethod
    def transcribe(
        self,
        source: AudioSource,
        config: ChunkingConfig,
        model_path: Path,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        """Transcribe audio from a bounded source into ordered timestamped segments."""
        raise NotImplementedError


class MockTranscriber(Transcriber):
    """Deterministic, dependency-free transcriber for tests.

    Emits one segment per chunk (respecting the bounded chunking config), with
    strictly increasing timestamps bounded by the audio duration. Output passes
    :func:`validate_segments`.
    """

    def __init__(self, label: str = "mock"):
        self.label = label

    def transcribe(
        self,
        source: AudioSource,
        config: ChunkingConfig,
        model_path: Path,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        duration = source.duration_sec
        segments: List[TranscriptSegment] = []
        idx = 0
        for chunk in iter_audio_chunks(source, config):
            if chunk.samples.size == 0:
                continue
            end = chunk.start_sec + float(chunk.samples.size) / float(chunk.sample_rate)
            segments.append(
                TranscriptSegment(
                    id=f"seg-{idx:04d}",
                    start=chunk.start_sec,
                    end=end,
                    text=f"{self.label} segment {idx}",
                    confidence=0.9,
                )
            )
            idx += 1
            if on_progress is not None:
                on_progress(min(1.0, end / duration if duration > 0 else 1.0))
        return validate_segments(segments, duration)


class WhisperTranscriber(Transcriber):
    """Real Whisper transcription via mlx-whisper, chunk-by-chunk.

    Each bounded chunk is transcribed with ``mlx_whisper.transcribe`` against the
    local ``model_path``; backend segments are converted to
    :class:`TranscriptSegment`, their timestamps offset by the chunk's file
    position, ids normalized, and the merged result validated before return. Bad
    backend output (empty text, out-of-bounds timestamps, non-monotonic order)
    is rejected by :func:`validate_segments`.
    """

    def __init__(self, language: Optional[str] = None):
        self.language = language

    def transcribe(
        self,
        source: AudioSource,
        config: ChunkingConfig,
        model_path: Path,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        try:
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError(
                "mlx-whisper is not installed; provision the STT backend "
                "(see docs/provisioning.md) before using WhisperTranscriber"
            ) from exc

        duration = source.duration_sec
        segments: List[TranscriptSegment] = []
        idx = 0
        for chunk in iter_audio_chunks(source, config):
            if chunk.samples.size == 0:
                continue
            result = mlx_whisper.transcribe(
                np.ascontiguousarray(chunk.samples, dtype=np.float32),
                path_or_hf_repo=str(model_path),
                language=self.language,
                verbose=False,
            )
            backend_segments = self._extract_segments(result)
            for bs in backend_segments:
                start = self._as_float(bs.get("start")) + chunk.start_sec
                end = self._as_float(bs.get("end")) + chunk.start_sec
                text = str(bs.get("text", "")).strip()
                segments.append(
                    TranscriptSegment(
                        id=f"seg-{idx:04d}",
                        start=start,
                        end=end,
                        text=text,
                    )
                )
                idx += 1
            if on_progress is not None:
                on_progress(min(1.0, (chunk.start_sec + chunk.samples.size / chunk.sample_rate) / duration if duration > 0 else 1.0))

        # validate_segments rejects empty text, out-of-bounds timestamps,
        # non-monotonic order, and zero-length (untimed) segments — i.e. bad
        # backend output is rejected before the caller sees it.
        return validate_segments(segments, duration)

    @staticmethod
    def _extract_segments(result) -> list:
        if not isinstance(result, dict):
            raise ValueError("mlx_whisper.transcribe must return a dict")
        segs = result.get("segments")
        if not isinstance(segs, list):
            raise ValueError("mlx_whisper result 'segments' must be a list")
        return segs

    @staticmethod
    def _as_float(value) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"backend segment timestamp must be a number, got {value!r}")
        return float(value)
