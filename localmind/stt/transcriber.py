"""Transcriber interface, tier resolution, and implementations.

* :class:`Transcriber` — abstract interface over a bounded audio source. The
  real backend takes a :class:`Provisioner` and a tier id (not a pre-built
  path or tier) so the model is resolved through the provisioner *inside* the
  adapter boundary, immediately before backend invocation.
* :func:`resolve_tier` — resolve a model tier through the provisioner so a
  missing tier fast-fails with ``ModelNotProvisionedError`` and **never**
  downloads, returning a :class:`ResolvedTier` carrying the provenance fields
  (tier, model_id, model_path, sha256) that downstream run records emit.
* :class:`MockTranscriber` — deterministic, dependency-free transcriber for
  tests: emits ordered timestamped segments bounded by the audio duration.
* :class:`WhisperTranscriber` — the real adapter over ``mlx-whisper``. It calls
  :func:`resolve_tier` itself (so the verified path can only come from
  ``Provisioner.require_model``), then transcribes chunk-by-chunk from a bounded
  source, converts backend segments to :class:`TranscriptSegment`, merges
  overlapping chunk outputs without producing non-monotonic timestamps,
  normalizes ids, and validates the result before returning. A hand-built
  :class:`ResolvedTier` is not accepted as the authority for backend loading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from localmind.provisioning.errors import ModelNotProvisionedError
from localmind.provisioning.provisioner import Provisioner
from localmind.stt.chunking import AudioSource, ChunkingConfig, iter_audio_chunks, iter_chunks
from localmind.stt.segment import TranscriptSegment, validate_segments

ProgressCallback = Callable[[float], None]

# A raw per-chunk segment: (absolute_start_sec, absolute_end_sec, text).
_RawSegment = Tuple[float, float, str]


@dataclass(frozen=True)
class ResolvedTier:
    """A verified model tier plus the provenance fields a run record emits.

    This is a *record* of a resolution that happened inside the adapter
    boundary; it is not accepted as input to the real backend (callers pass a
    ``Provisioner`` + tier id, and the adapter resolves fresh).
    """

    tier: str
    model_id: str
    model_path: Path
    sha256: str
    quant_format: str
    kind: str = ""


def resolve_tier(provisioner: Provisioner, tier_model_id: str) -> ResolvedTier:
    """Resolve a model tier to a verified path plus provenance.

    Goes through ``Provisioner.require_model`` so an unprovisioned tier raises
    ``ModelNotProvisionedError`` (no network download) and a tampered weight
    raises ``ChecksumMismatchError``. The manifest entry's sha256/quant_format
    are carried into the returned provenance object.
    """
    manifest = provisioner.load_manifest()
    try:
        entry = manifest.by_id(tier_model_id)
    except KeyError:
        raise ModelNotProvisionedError(
            f"model not provisioned: {tier_model_id!r} is not declared in the manifest"
        ) from None
    path = provisioner.require_model(tier_model_id)  # verifies size + sha256, confines
    return ResolvedTier(
        tier=tier_model_id,
        model_id=tier_model_id,
        model_path=path,
        sha256=entry.sha256,
        quant_format=entry.quant_format,
        kind=entry.kind,
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
        provisioner: Provisioner,
        tier_model_id: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        """Transcribe audio from a bounded source into ordered timestamped segments."""
        raise NotImplementedError


class MockTranscriber(Transcriber):
    """Deterministic, dependency-free transcriber for tests.

    Emits one segment per chunk (respecting the bounded chunking config), then
    merges overlapping chunk outputs so the result is monotonic and validated.
    The ``provisioner``/``tier_model_id`` are accepted for interface
    compatibility and ignored (no real model is loaded).
    """

    def __init__(self, label: str = "mock"):
        self.label = label

    def transcribe(
        self,
        source: AudioSource,
        config: ChunkingConfig,
        provisioner: Provisioner,
        tier_model_id: str,
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

    The adapter owns the model-resolution boundary: callers pass a
    :class:`Provisioner` and a tier id, and the adapter calls
    :func:`resolve_tier` (which runs ``Provisioner.require_model`` — verifying
    size + SHA-256 and confining the path to the model directory) immediately
    before invoking the backend. Only the freshly verified path is passed to
    ``mlx_whisper.transcribe``. A pre-built :class:`ResolvedTier` is not
    accepted, so an arbitrary existing local file cannot reach the backend.

    After a run, ``self.last_provenance`` holds the resolved tier for downstream
    run records.
    """

    def __init__(self, language: Optional[str] = None):
        self.language = language
        self.last_provenance: Optional[ResolvedTier] = None

    def transcribe(
        self,
        source: AudioSource,
        config: ChunkingConfig,
        provisioner: Provisioner,
        tier_model_id: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        if not isinstance(provisioner, Provisioner):
            raise TypeError(
                "WhisperTranscriber requires a Provisioner and a tier id; a "
                "pre-built ResolvedTier or raw path is not accepted"
            )

        # Resolve the tier BEFORE importing the backend: a missing/tampered/
        # unprovisioned model must fail deterministically here, before any
        # backend initialization — regardless of whether a partial backend
        # package happens to be installed.
        resolved = resolve_tier(provisioner, tier_model_id)
        self.last_provenance = resolved

        # Import the backend, but avoid polluting the process with MLX atexit
        # hooks on Metal-unavailable hosts. If a fake backend is already
        # injected (unit tests), use it directly. If explicitly marked as
        # unavailable (sys.modules[name] is None), raise. Only run the
        # subprocess Metal preflight when no module is present at all.
        import sys as _sys
        _mod = _sys.modules.get("mlx_whisper")
        if _mod is not None:
            mlx_whisper = _mod  # injected fake or already-imported real
        elif "mlx_whisper" in _sys.modules:
            raise RuntimeError(
                "mlx-whisper is not installed; install the ML backend with "
                "`pip install -e '.[ml]'` (see docs/provisioning.md) before using WhisperTranscriber"
            )
        else:
            from localmind.mlx_runtime import ensure_mlx_metal_available
            ensure_mlx_metal_available()
            try:
                import mlx_whisper
            except ImportError as exc:
                raise RuntimeError(
                    "mlx-whisper is not installed; install the ML backend with "
                    "`pip install -e '.[ml]'` (see docs/provisioning.md) before using WhisperTranscriber"
                ) from exc

        duration = source.duration_sec
        chunk_results: List[Tuple[float, List[_RawSegment]]] = []
        for chunk in iter_chunks(source, config):
            if chunk.samples.size == 0:
                chunk_results.append((chunk.start_sec, []))
                continue
            result = mlx_whisper.transcribe(
                np.ascontiguousarray(chunk.samples, dtype=np.float32),
                path_or_hf_repo=str(resolved.model_path.parent),
                language=self.language,
                verbose=False,
            )
            raw_segs: List[_RawSegment] = []
            for bs in self._extract_segments(result):
                start = self._as_float(bs.get("start")) + chunk.start_sec
                end = self._as_float(bs.get("end")) + chunk.start_sec
                # Clamp to audio duration: real backends can emit a final
                # segment whose end slightly exceeds the true audio length.
                start = min(start, duration)
                end = min(end, duration)
                text = str(bs.get("text", "")).strip()
                if end <= start:
                    continue  # skip degenerate trailing segment after clamp
                raw_segs.append((start, end, text))
            chunk_results.append((chunk.start_sec, raw_segs))
            if on_progress is not None:
                progress = (chunk.start_sec + chunk.samples.size / chunk.sample_rate) / duration
                on_progress(min(1.0, progress if duration > 0 else 1.0))

        # VAD chunks split at speech boundaries with no overlap, so merge with
        # overlap=0 (the config.overlap_sec is for fixed-window chunking).
        merge_config = config
        if config.use_vad and config.overlap_sec > 0:
            from dataclasses import replace as _replace
            merge_config = _replace(config, overlap_sec=0.0)
        merged = merge_chunk_segments(chunk_results, merge_config, duration)
        # Clamp end timestamps to audio duration: real backends (e.g. mlx_whisper)
        # can emit a final segment whose end slightly exceeds the true audio
        # length (e.g. 16.20s for a 16.10s clip). Clamp so validate_segments does
        # not reject an otherwise-correct transcript.
        for seg in merged:
            if seg.end > duration:
                seg.end = float(duration)
        return merged

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

