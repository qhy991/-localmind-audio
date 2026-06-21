"""Exception hierarchy for audio decoding."""

from __future__ import annotations


class AudioError(Exception):
    """Base class for all audio decoding failures."""


class UnsupportedFormatError(AudioError):
    """The file extension/container is not in the supported set."""


class DecodeError(AudioError):
    """The file could not be decoded (corrupt, truncated, empty, or bad header)."""


class DecoderUnavailableError(AudioError):
    """A backend required for this format (e.g. ffmpeg for .m4a/.mp3) is not installed.

    Distinct from :class:`UnsupportedFormatError`: the format is supported in
    principle, but the optional decoder binary is missing. Surfaced as an
    explicit error rather than a silent failure or a crash.
    """
