"""Transcript segment model and validation (AC-2).

A transcript is an ordered list of timestamped segments. Validation enforces the
AC-2 invariants: segments are non-empty, span real time (strictly positive
length), are bounded by the audio duration, and have monotonically
non-decreasing start times. A "flat untimed" transcript (segments with no real
timestamps, e.g. ``start == end == 0``) is rejected because zero-length segments
are invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


class SegmentValidationError(ValueError):
    """Raised when a transcript fails segment validation."""


@dataclass(frozen=True)
class TranscriptSegment:
    """One timestamped transcript segment."""

    id: str
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None


def validate_segments(
    segments: List[TranscriptSegment], audio_duration_sec: float
) -> List[TranscriptSegment]:
    """Validate a transcript against AC-2 invariants.

    Returns the segments unchanged on success; raises SegmentValidationError
    on any violation.
    """
    if not isinstance(segments, list) or len(segments) == 0:
        raise SegmentValidationError("transcript must be a non-empty list of segments")

    if not isinstance(audio_duration_sec, (int, float)) or isinstance(
        audio_duration_sec, bool
    ) or audio_duration_sec <= 0:
        raise SegmentValidationError(
            f"audio_duration_sec must be a positive number, got {audio_duration_sec!r}"
        )

    seen_ids = set()
    prev_start: Optional[float] = None
    for i, seg in enumerate(segments):
        if not isinstance(seg, TranscriptSegment):
            raise SegmentValidationError(f"segment[{i}] is not a TranscriptSegment")

        if not seg.id or not isinstance(seg.id, str):
            raise SegmentValidationError(f"segment[{i}].id must be a non-empty string")
        if seg.id in seen_ids:
            raise SegmentValidationError(f"duplicate segment id: {seg.id!r}")
        seen_ids.add(seg.id)

        # Zero-length / untimed segments are invalid (rejects "flat untimed" output).
        if not isinstance(seg.start, (int, float)) or isinstance(seg.start, bool):
            raise SegmentValidationError(f"segment[{i}].start must be a number")
        if not isinstance(seg.end, (int, float)) or isinstance(seg.end, bool):
            raise SegmentValidationError(f"segment[{i}].end must be a number")
        if seg.start < 0:
            raise SegmentValidationError(
                f"segment[{i}].start must be >= 0, got {seg.start}"
            )
        if seg.end <= seg.start:
            raise SegmentValidationError(
                f"segment[{i}] must have end > start (got start={seg.start}, end={seg.end})"
            )
        if seg.end > audio_duration_sec + 1e-6:
            raise SegmentValidationError(
                f"segment[{i}].end={seg.end} exceeds audio duration {audio_duration_sec}"
            )

        if not isinstance(seg.text, str) or not seg.text.strip():
            raise SegmentValidationError(f"segment[{i}].text must be non-empty")

        if prev_start is not None and seg.start < prev_start - 1e-9:
            raise SegmentValidationError(
                f"segment[{i}].start={seg.start} is before previous start={prev_start} "
                f"(timestamps must be monotonically non-decreasing)"
            )
        prev_start = seg.start

    return segments
