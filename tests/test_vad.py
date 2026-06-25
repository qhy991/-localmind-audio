"""Tests for energy-based VAD. Pure NumPy, deterministic — no audio fixtures
needed; signals are synthesized."""
from __future__ import annotations

import numpy as np
import pytest

from localmind.vad import VadConfig, SpeechSegment, detect_speech

SR = 16000


def _tone(seconds: float, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    n = int(seconds * SR)
    return (np.sin(2 * np.pi * freq * np.arange(n) / SR) * amp).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_pure_silence_yields_no_speech():
    assert detect_speech(_silence(3.0), SR) == []


def test_flat_signal_with_no_dynamic_range_yields_no_speech():
    # A continuous tone with no silence has no measurable background, so there
    # is nothing to detect as speech-above-floor. Real speech always has pauses.
    assert detect_speech(_tone(2.0), SR) == []


def test_two_speech_bursts_separated_by_silence():
    # 1s silence | 2s loud | 1s silence | 2s loud
    x = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.0), _tone(2.0)])
    segs = detect_speech(x, SR)
    assert len(segs) == 2
    # First burst around 1-3s, second around 4-6s.
    assert 0.8 <= segs[0].start_sec <= 1.2
    assert 2.8 <= segs[0].end_sec <= 3.2
    assert 3.8 <= segs[1].start_sec <= 4.2
    assert 5.8 <= segs[1].end_sec <= 6.2
    # Non-overlapping and ascending.
    for a, b in zip(segs, segs[1:]):
        assert a.end_sec <= b.start_sec


def test_short_pause_does_not_split_utterance():
    # 0.5s silence (establishes a floor) | 2s tone | 0.2s pause | 2s tone
    x = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.2), _tone(2.0)])
    segs = detect_speech(x, SR)
    assert len(segs) == 1
    assert segs[0].duration_sec >= 3.5  # spans the pause


def test_long_pause_splits_into_two():
    # 2s tone | 1.5s pause (over min_silence) | 2s tone -> two segments
    x = np.concatenate([_tone(2.0), _silence(1.5), _tone(2.0)])
    segs = detect_speech(x, SR)
    assert len(segs) == 2


def test_min_speech_sec_drops_short_bursts():
    # A 0.05s blip is below min_speech_sec (default 0.25) -> dropped.
    x = np.concatenate([_silence(1.0), _tone(0.05), _silence(1.0)])
    assert detect_speech(x, SR) == []


def test_end_times_clamped_to_duration():
    x = _tone(1.0)  # 1 second
    for seg in detect_speech(x, SR):
        assert seg.end_sec <= 1.0 + 1e-6


def test_deterministic_same_input_same_output():
    x = np.concatenate([_silence(0.5), _tone(1.0), _silence(0.5), _tone(1.0)])
    a = detect_speech(x, SR)
    b = detect_speech(x, SR)
    assert a == b


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        VadConfig(hop_ms=0)
    with pytest.raises(ValueError):
        VadConfig(speech_fall_db=30.0, speech_rise_db=10.0)  # fall > rise
    with pytest.raises(ValueError):
        VadConfig(noise_percentile=150.0)


def test_speech_segment_duration_property():
    seg = SpeechSegment(start_sec=1.0, end_sec=3.5)
    assert seg.duration_sec == 2.5


def test_empty_input_returns_empty():
    assert detect_speech(np.zeros(0, dtype=np.float32), SR) == []


def test_shorter_than_one_frame_does_not_crash():
    # 5ms signal (< 30ms frame) — should not raise; returns 0 or 1 segment.
    segs = detect_speech(_tone(0.005), SR)
    assert isinstance(segs, list)


def test_real_chinese_sample_finds_speech():
    """Optional integration: only runs if the provisioned sample fixture exists."""
    import wave
    from pathlib import Path
    sample = Path("audio/sample-zh.wav")
    if not sample.exists():
        pytest.skip("no audio/sample-zh.wav fixture")
    from localmind.audio.decode import decode_audio
    d = decode_audio(str(sample))
    segs = detect_speech(d.samples, d.sample_rate)
    # A 16s speech sample must yield at least one segment, and speech total
    # should be a sensible fraction of the clip (not ~0%, not >100%).
    assert len(segs) >= 1
    total = sum(s.duration_sec for s in segs)
    assert 0.1 * d.duration_sec < total <= d.duration_sec
