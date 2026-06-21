"""Speech-to-text framework: segment model, bounded chunking, tier selection, transcribers.

The transcription backend (`WhisperTranscriber`) runs `mlx-whisper` against a
locally provisioned model; when `mlx-whisper` is not installed it raises a clear
error rather than silently degrading. This package also provides the transcript
segment model and validator, bounded audio chunking (windowed sources so peak
audio-buffer memory does not scale with file length), model-tier resolution
through the provisioner (a missing tier fast-fails without any download), and a
`MockTranscriber` for deterministic tests.
"""

from localmind.stt.chunking import (
    MAX_UNCHUNKED_SEC,
    ArrayAudioSource,
    AudioChunk,
    AudioSource,
    ChunkingConfig,
    FFmpegAudioSource,
    WavAudioSource,
    audio_source_from_path,
    chunk_audio,
    iter_audio_chunks,
)
from localmind.stt.segment import TranscriptSegment, validate_segments
from localmind.stt.transcriber import (
    MockTranscriber,
    ResolvedTier,
    Transcriber,
    WhisperTranscriber,
    resolve_tier,
)

__all__ = [
    "MAX_UNCHUNKED_SEC",
    "ArrayAudioSource",
    "AudioChunk",
    "AudioSource",
    "ChunkingConfig",
    "FFmpegAudioSource",
    "WavAudioSource",
    "audio_source_from_path",
    "chunk_audio",
    "iter_audio_chunks",
    "TranscriptSegment",
    "validate_segments",
    "MockTranscriber",
    "ResolvedTier",
    "Transcriber",
    "WhisperTranscriber",
    "resolve_tier",
]
