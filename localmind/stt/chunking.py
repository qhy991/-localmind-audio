"""Bounded audio chunking for long-audio transcription.

Long audio is transcribed in overlapping chunks so peak audio-buffer memory does
not scale with the full file length. The bounded-memory guarantee is provided by
:class:`AudioSource`: a source reads only the window it is asked for
(:meth:`read_window`), so :func:`iter_audio_chunks` never holds more than one
chunk's worth of samples at a time. :class:`WavAudioSource` reads WAV frames by
window via the stdlib :mod:`wave` module and never materializes the whole file.

Chunking is **mandatory** for audio longer than :data:`MAX_UNCHUNKED_SEC`; a
configuration that disables chunking on a long file is rejected before any
backend inference runs.
"""

from __future__ import annotations

import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

from localmind.audio.decode import _bytes_to_float, _ffmpeg_exe, _resample_linear, _to_mono
from localmind.audio.errors import DecodeError, DecoderUnavailableError, UnsupportedFormatError

# Audio longer than this must be chunked. Keeps peak decode/buffer memory bounded
# regardless of total file length.
MAX_UNCHUNKED_SEC = 300.0  # 5 minutes


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for bounded audio chunking."""

    chunk_duration_sec: float = 30.0
    overlap_sec: float = 1.0
    enabled: bool = True
    # When True, chunk boundaries follow detected speech (VAD): pure-silence
    # regions are skipped entirely (never transcribed) and chunks split at
    # speech boundaries instead of mid-utterance. See iter_vad_chunks.
    use_vad: bool = False
    # Optional VAD tuning; only used when use_vad is True. Defaults to VadConfig().
    vad_config: "object | None" = None

    def __post_init__(self):
        if self.chunk_duration_sec <= 0:
            raise ValueError("chunk_duration_sec must be positive")
        if self.overlap_sec < 0:
            raise ValueError("overlap_sec must be non-negative")
        if self.overlap_sec >= self.chunk_duration_sec:
            raise ValueError("overlap_sec must be less than chunk_duration_sec")


@dataclass(frozen=True)
class AudioChunk:
    """One bounded chunk of audio, located at ``start_sec`` in file time."""

    start_sec: float
    samples: np.ndarray  # float32, mono, at the source's sample rate
    sample_rate: int
    final: bool  # True if this is the last chunk of the source


@runtime_checkable
class AudioSource(Protocol):
    """A source that exposes bounded windowed reads of decoded audio.

    Implementations must NOT materialize the whole file: :meth:`read_window`
    decodes/reads only the requested ``[start_sec, end_sec]`` window.
    """

    @property
    def sample_rate(self) -> int: ...

    @property
    def duration_sec(self) -> float: ...

    def read_window(self, start_sec: float, end_sec: float) -> np.ndarray: ...


class ArrayAudioSource:
    """In-memory source wrapping a fully decoded sample array.

    Intended for short buffers and tests. It materializes the whole array, so it
    is **not** the long-file path — use :class:`WavAudioSource` for long audio.
    """

    def __init__(self, samples: np.ndarray, sample_rate: int):
        self._samples = np.ascontiguousarray(samples, dtype=np.float32)
        self._sr = int(sample_rate)
        if self._samples.ndim != 1:
            raise ValueError("ArrayAudioSource expects mono 1-D samples")

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def duration_sec(self) -> float:
        return float(self._samples.size) / float(self._sr)

    def read_window(self, start_sec: float, end_sec: float) -> np.ndarray:
        s = max(0, int(round(start_sec * self._sr)))
        e = min(self._samples.size, int(round(end_sec * self._sr)))
        if e <= s:
            return np.empty(0, dtype=np.float32)
        return self._samples[s:e]


class WavAudioSource:
    """Windowed WAV source: reads only the requested frames, never the whole file.

    Decodes each window to mono float32 at ``target_sample_rate`` (default 16 kHz,
    the rate the transcription backend expects). For a 16 kHz mono WAV this is a
    direct frame read; other rates are resampled per window.
    """

    def __init__(self, path, target_sample_rate: int = 16000):
        self.path = Path(path)
        self._target_rate = int(target_sample_rate)
        try:
            with wave.open(str(self.path), "rb") as wf:
                self._nchannels = wf.getnchannels()
                self._sampwidth = wf.getsampwidth()
                self._framerate = wf.getframerate()
                self._nframes = wf.getnframes()
        except Exception as exc:
            raise DecodeError(f"failed to open WAV source {self.path}: {exc}") from exc
        if self._nframes == 0:
            raise DecodeError(f"WAV source has no frames: {self.path}")
        if self._framerate <= 0:
            raise DecodeError(f"invalid sample rate in {self.path}: {self._framerate}")

    @property
    def sample_rate(self) -> int:
        return self._target_rate

    @property
    def duration_sec(self) -> float:
        return float(self._nframes) / float(self._framerate)

    def read_window(self, start_sec: float, end_sec: float) -> np.ndarray:
        start_frame = max(0, int(round(start_sec * self._framerate)))
        end_frame = min(self._nframes, int(round(end_sec * self._framerate)))
        if end_frame <= start_frame:
            return np.empty(0, dtype=np.float32)
        with wave.open(str(self.path), "rb") as wf:
            wf.setpos(start_frame)
            raw = wf.readframes(end_frame - start_frame)
        if not raw:
            return np.empty(0, dtype=np.float32)
        samples = _bytes_to_float(raw, self._sampwidth)
        samples = _to_mono(samples, self._nchannels)
        samples = _resample_linear(samples, self._framerate, self._target_rate)
        return np.ascontiguousarray(samples, dtype=np.float32)


def iter_audio_chunks(
    source: AudioSource,
    config: ChunkingConfig = ChunkingConfig(),
) -> Iterator[AudioChunk]:
    """Yield bounded :class:`AudioChunk` objects over the source.

    Each chunk covers at most ``chunk_duration_sec`` of audio; only one chunk's
    samples are held at a time (callers should process and drop a chunk before
    iterating the next). When ``config.enabled`` is False, raises for sources
    longer than :data:`MAX_UNCHUNKED_SEC`; short sources yield a single chunk.
    """
    duration = source.duration_sec

    if not config.enabled:
        if duration > MAX_UNCHUNKED_SEC:
            raise ValueError(
                f"chunking is disabled but audio is {duration:.1f}s (> "
                f"{MAX_UNCHUNKED_SEC}s); chunking is mandatory for long audio"
            )
        samples = source.read_window(0.0, duration)
        yield AudioChunk(0.0, samples, source.sample_rate, True)
        return

    chunk_dur = config.chunk_duration_sec
    overlap = config.overlap_sec
    hop = chunk_dur - overlap
    t = 0.0
    while t < duration - 1e-9:
        end = min(t + chunk_dur, duration)
        samples = source.read_window(t, end)
        final = end >= duration - 1e-9
        yield AudioChunk(t, samples, source.sample_rate, final)
        if final:
            break
        t += hop


def iter_vad_chunks(
    source: AudioSource,
    config: ChunkingConfig = ChunkingConfig(),
) -> Iterator[AudioChunk]:
    """Yield speech-aligned chunks, skipping silence regions (VAD-driven).

    Unlike :func:`iter_audio_chunks` (fixed windows), this scans the source for
    speech and forms chunks that:

    * start/end at speech boundaries — never cut mid-utterance;
    * skip pure-silence regions longer than ``VadConfig.min_silence_sec``
      entirely (those spans are never read or transcribed, so dead air costs
      nothing);
    * keep continuous audio within a chunk (short pauses inside an utterance
      are preserved so the backend's timestamps stay correct);
    * span at most ``chunk_duration_sec`` of wall-clock audio.

    Peak memory stays bounded: the source is scanned in VAD windows (never the
    whole file at once), and only one chunk's samples are read per yield.
    """
    from localmind.vad import VadConfig, detect_speech

    vcfg = config.vad_config if config.vad_config is not None else VadConfig()
    duration = source.duration_sec
    chunk_max = config.chunk_duration_sec
    # VAD scan window: large enough to estimate a noise floor and cover a few
    # chunks, small enough to keep peak memory bounded for long files.
    vad_window = max(chunk_max * 4.0, 120.0)

    carried = None  # in-progress chunk span [start, end] in file time
    w_start = 0.0
    while w_start < duration - 1e-9:
        w_end = min(w_start + vad_window, duration)
        win = source.read_window(w_start, w_end)
        segs = detect_speech(win, source.sample_rate, vcfg)
        intervals = [(w_start + s.start_sec, w_start + s.end_sec) for s in segs]
        # Slice any single speech interval longer than the budget into
        # budget-sized spans: a single utterance longer than the chunk budget
        # must be cut somewhere, and we cut at time boundaries (like fixed
        # chunking) rather than mid-word at a VAD boundary that doesn't exist.
        sliced = []
        for s, e in intervals:
            while e - s > chunk_max + 1e-9:
                sliced.append((s, s + chunk_max))
                s = s + chunk_max
            sliced.append((s, e))
        intervals = sliced

        cur = carried
        completed = []
        for s, e in intervals:
            if cur is None:
                cur = [s, e]
            else:
                gap = s - cur[1]
                span_with = e - cur[0]
                # Extend the chunk when the gap to the next speech is small
                # (a pause inside one utterance) AND the total span stays under
                # the budget. Otherwise close the chunk and start a new one.
                if gap <= vcfg.min_silence_sec and span_with <= chunk_max:
                    cur[1] = e
                else:
                    completed.append((cur[0], cur[1]))
                    cur = [s, e]
        carried = cur

        window_final = w_end >= duration - 1e-9
        if window_final:
            tail = list(completed)
            if carried is not None:
                tail.append((carried[0], carried[1]))
            for i, (cs, ce) in enumerate(tail):
                yield AudioChunk(cs, source.read_window(cs, ce),
                                 source.sample_rate, i == len(tail) - 1)
            return
        for cs, ce in completed:
            yield AudioChunk(cs, source.read_window(cs, ce),
                             source.sample_rate, False)
        w_start = w_end


def iter_chunks(
    source: AudioSource,
    config: ChunkingConfig = ChunkingConfig(),
) -> Iterator[AudioChunk]:
    """Dispatch to :func:`iter_vad_chunks` when ``config.use_vad`` is set, else
    :func:`iter_audio_chunks`. This is the entry point adapters should use."""
    if config.use_vad:
        yield from iter_vad_chunks(source, config)
    else:
        yield from iter_audio_chunks(source, config)


class FFmpegAudioSource:
    """Windowed source for compressed audio (.m4a/.mp3/.aac) via ffmpeg.

    Reads only the requested window with ``ffmpeg -ss <start> -t <dur> -i <path>
    -f f32le -ac 1 -ar <rate> -``, so the full file is never decoded into memory.
    Duration is discovered from ffmpeg metadata probing (no full decode). Each
    window is returned as mono float32 at ``target_sample_rate``.
    """

    _COMPRESSED_EXTS = frozenset({"m4a", "mp3", "aac"})

    def __init__(self, path, target_sample_rate: int = 16000):
        self.path = Path(path)
        self._target_rate = int(target_sample_rate)
        self._exe = self._require_ffmpeg()
        self._duration = self._probe_duration()

    def _require_ffmpeg(self) -> str:
        exe = _ffmpeg_exe()
        if exe is None:
            raise DecoderUnavailableError(
                f"ffmpeg is not available; cannot create a bounded source for {self.path}"
            )
        return exe

    def _probe_duration(self) -> float:
        # `ffmpeg -i <path>` with no output writes duration to stderr and exits
        # nonzero; that is expected — we parse the stderr, not the exit code.
        proc = subprocess.run(
            [self._exe, "-i", str(self.path)], capture_output=True
        )
        stderr = proc.stderr.decode("utf-8", errors="replace")
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
        if not m:
            raise DecodeError(
                f"could not determine duration of {self.path}; file may be "
                f"unreadable or not a supported audio container"
            )
        hours, minutes, seconds = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration = hours * 3600 + minutes * 60 + seconds
        if duration <= 0:
            raise DecodeError(f"non-positive duration for {self.path}: {duration}")
        return duration

    @property
    def sample_rate(self) -> int:
        return self._target_rate

    @property
    def duration_sec(self) -> float:
        return self._duration

    def read_window(self, start_sec: float, end_sec: float) -> np.ndarray:
        dur = end_sec - start_sec
        if dur <= 0:
            return np.empty(0, dtype=np.float32)
        start_sec = max(0.0, float(start_sec))
        cmd = [
            self._exe,
            "-loglevel", "error",
            "-ss", f"{start_sec:.6f}",
            "-t", f"{dur:.6f}",
            "-i", str(self.path),
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(self._target_rate),
            "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True)
        except OSError as exc:
            raise DecodeError(f"failed to invoke ffmpeg for {self.path}: {exc}") from exc
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise DecodeError(f"ffmpeg failed for {self.path}: {stderr}")
        raw = proc.stdout
        if not raw:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def audio_source_from_path(path, target_sample_rate: int = 16000) -> "AudioSource":
    """Build a bounded audio source appropriate for a file's container.

    ``.wav`` -> :class:`WavAudioSource`; ``.m4a``/``.mp3``/``.aac`` ->
    :class:`FFmpegAudioSource`; anything else raises
    :class:`UnsupportedFormatError`. Both sources read only the window asked
    for, so long-audio transcription never materializes the whole file.
    """
    ext = Path(path).suffix.lower().lstrip(".")
    if ext == "wav":
        return WavAudioSource(path, target_sample_rate)
    if ext in FFmpegAudioSource._COMPRESSED_EXTS:
        return FFmpegAudioSource(path, target_sample_rate)
    raise UnsupportedFormatError(f"unsupported audio format: .{ext}")


def chunk_audio(
    samples: np.ndarray,
    sample_rate: int,
    config: ChunkingConfig = ChunkingConfig(),
) -> Iterator:
    """Convenience wrapper for short in-memory buffers.

    Materializes the full array (via :class:`ArrayAudioSource`), so this is only
    appropriate for short buffers; for long audio use :func:`iter_audio_chunks`
    with a :class:`WavAudioSource`. Yields ``(start_sec, chunk_samples)`` pairs.
    """
    source = ArrayAudioSource(samples, sample_rate)
    for chunk in iter_audio_chunks(source, config):
        yield chunk.start_sec, chunk.samples
