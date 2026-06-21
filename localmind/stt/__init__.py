"""Speech-to-text framework: segment model, chunking, tier selection, transcribers.

The real Whisper inference adapter (`WhisperTranscriber`) needs `mlx-whisper`
plus provisioned weights and is finalized in a later milestone; this package
delivers the dependency-free, fully testable framework now: the transcript
segment model and validator (AC-2), bounded audio chunking (AC-2.1), model-tier
selection through the provisioner (missing tier fast-fails without download),
and a `MockTranscriber` for deterministic tests.
"""

from localmind.stt.chunking import MAX_UNCHUNKED_SEC, ChunkingConfig, chunk_audio
from localmind.stt.segment import TranscriptSegment, validate_segments
from localmind.stt.transcriber import (
    MockTranscriber,
    Transcriber,
    WhisperTranscriber,
    select_tier,
)

__all__ = [
    "MAX_UNCHUNKED_SEC",
    "ChunkingConfig",
    "chunk_audio",
    "TranscriptSegment",
    "validate_segments",
    "MockTranscriber",
    "Transcriber",
    "WhisperTranscriber",
    "select_tier",
]
