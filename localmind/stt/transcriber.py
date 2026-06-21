"""Transcriber interface, tier resolution, and implementations.

* :class:`Transcriber` — abstract interface over a bounded audio source. The
  real backend takes a :class:`ResolvedTier` (not a raw path) so a model tier
  must be resolved through the provisioner first.
* :func:`resolve_tier` — resolve a model tier through the provisioner so a
  missing tier fast-fails with ``ModelNotProvisionedError`` and **never**
  downloads, returning a :class:`ResolvedTier` carrying the provenance fields
  (tier, model_id, model_path, sha256) that downstream run records emit.
* :class:`MockTranscriber` — deterministic, dependency-free transcriber for
  tests: emits ordered timestamped segments bounded by the audio duration.
* :class:`WhisperTranscriber` — the real adapter over ``mlx-whisper``. It
  transcribes chunk-by-chunk from a bounded source, converts backend segments to
  :class:`TranscriptSegment`, merges overlapping chunk outputs without producing
  non-monotonic timestamps, normalizes ids, and validates the result before
  returning. It refuses to run without a verified local model path and without
  ``mlx-whisper`` installed (never a silent cloud fallback or download).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from localmind.provisioning.errors import ModelNotProvisionedError
from localmind.provisioning.provisioner import Provisioner
from localmind.stt.chunking import AudioSource, ChunkingConfig, iter_audio_chunks
from localmind.stt.segment import TranscriptSegment, validate_segments

ProgressCallback = Callable[[float], None]

# A raw per-chunk segment: (absolute_start_sec, absolute_end_sec, text).
_RawSegment = Tuple[float, float, str]


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


def merge_chunk_segments(
    chunk_results: List[Tuple[float, List[_RawSegment]]],
    config: ChunkingConfig,
    duration_sec: float,
) -> List[TranscriptSegment]:
    """Merge per-chunk raw segments into a validated transcript.

    Overlapping chunks would otherwise duplicate content and can produce
    non-monotonic timestamps (a late segment in chunk *i* may start after an
    early segment in chunk *i+1*). For every non-first chunk we define
    ``accept_after = chunk.start_sec + overlap_sec`` and:

    * drop segments whose absolute ``end <= accept_after`` (fully inside the
      overlap region already covered by the previous chunk);
    * clamp a straddling segment's ``start`` up to ``accept_after`` when its
      ``end > accept_after``;
    * keep only segments with ``end > start``.

    IDs are then normalized deterministically and the result is validated.
    """
    accepted: List[_RawSegment] = []
    for i, (chunk_start, raw_segs) in enumerate(chunk_results):
        accept_after = chunk_start + config.overlap_sec if i > 0 else None
        for start, end, text in raw_segs:
            if accept_after is not None:
                if end <= accept_after:
                    continue
                if start < accept_after:
                    start = accept_after
            if end <= start:
                continue
            accepted.append((start, end, text))

    segments = [
        TranscriptSegment(id=f"seg-{i:04d}", start=s, end=e, text=t)
        for i, (s, e, t) in enumerate(accepted)
    ]
    return validate_segments(segments, duration_sec)


class Transcriber(ABC):
    """Abstract speech-to-text interface over a bounded audio source."""

    @abstractmethod
    def transcribe(
        self,
        source: AudioSource,
        config: ChunkingConfig,
        resolved_tier: ResolvedTier,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        """Transcribe audio from a bounded source into ordered timestamped segments."""
        raise NotImplementedError


class MockTranscriber(Transcriber):
    """Deterministic, dependency-free transcriber for tests.

    Emits one segment per chunk (respecting the bounded chunking config), then
    merges overlapping chunk outputs so the result is monotonic and validated.
    The ``resolved_tier`` is accepted for interface compatibility and ignored.
    """

    def __init__(self, label: str = "mock"):
        self.label = label

    def transcribe(
        self,
        source: AudioSource,
        config: ChunkingConfig,
        resolved_tier: ResolvedTier,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        duration = source.duration_sec
        chunk_results: List[Tuple[float, List[_RawSegment]]] = []
        for chunk in iter_audio_chunks(source, config):
            if chunk.samples.size == 0:
                chunk_results.append((chunk.start_sec, []))
                continue
            end = chunk.start_sec + float(chunk.samples.size) / float(chunk.sample_rate)
            chunk_results.append(
                (chunk.start_sec, [(chunk.start_sec, end, f"{self.label} segment")])
            )
            if on_progress is not None:
                on_progress(min(1.0, end / duration if duration > 0 else 1.0))
        return merge_chunk_segments(chunk_results, config, duration)


class WhisperTranscriber(Transcriber):
    """Real Whisper transcription via mlx-whisper, chunk-by-chunk.

    Requires a :class:`ResolvedTier` (from :func:`resolve_tier`) and asserts the
    model path exists locally before invoking the backend, so a Hugging Face
    repo id or any other non-local string cannot be passed through and downloaded
    at runtime. Each bounded chunk is transcribed with ``mlx_whisper.transcribe``
    against the local model path; backend segments are converted to
    :class:`TranscriptSegment`, their timestamps offset by the chunk's file
    position, and the overlapping chunk outputs are merged without producing
    non-monotonic timestamps. Bad backend output (empty text, out-of-bounds
    timestamps, non-monotonic order) is rejected by :func:`validate_segments`.
    """

    def __init__(self, language: Optional[str] = None):
        self.language = language

    def transcribe(
        self,
        source: AudioSource,
        config: ChunkingConfig,
        resolved_tier: ResolvedTier,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        if not isinstance(resolved_tier, ResolvedTier):
            raise TypeError(
                "WhisperTranscriber requires a ResolvedTier from resolve_tier(); "
                "raw paths or repo ids are not accepted"
            )
        if not resolved_tier.model_path.exists():
            raise ModelNotProvisionedError(
                f"model path does not exist: {resolved_tier.model_path}; "
                f"resolve the tier through the provisioner (see docs/provisioning.md)"
            )

        try:
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError(
                "mlx-whisper is not installed; install the ML backend with "
                "`pip install -e '.[ml]'` (see docs/provisioning.md) before using WhisperTranscriber"
            ) from exc

        duration = source.duration_sec
        chunk_results: List[Tuple[float, List[_RawSegment]]] = []
        for chunk in iter_audio_chunks(source, config):
            if chunk.samples.size == 0:
                chunk_results.append((chunk.start_sec, []))
                continue
            result = mlx_whisper.transcribe(
                np.ascontiguousarray(chunk.samples, dtype=np.float32),
                path_or_hf_repo=str(resolved_tier.model_path),
                language=self.language,
                verbose=False,
            )
            raw_segs: List[_RawSegment] = []
            for bs in self._extract_segments(result):
                start = self._as_float(bs.get("start")) + chunk.start_sec
                end = self._as_float(bs.get("end")) + chunk.start_sec
                text = str(bs.get("text", "")).strip()
                raw_segs.append((start, end, text))
            chunk_results.append((chunk.start_sec, raw_segs))
            if on_progress is not None:
                progress = (chunk.start_sec + chunk.samples.size / chunk.sample_rate) / duration
                on_progress(min(1.0, progress if duration > 0 else 1.0))

        return merge_chunk_segments(chunk_results, config, duration)

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
