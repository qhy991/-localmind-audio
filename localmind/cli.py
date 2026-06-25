"""Command-line interface: a stable JSON/JSONL contract.

The CLI is the contract that the future SwiftUI wrapper (and any other caller)
sits on top of. It emits:

* a versioned JSON final result on stdout, and
* newline-delimited JSON (JSONL) progress events on stderr.

Subcommands:

* ``transcribe`` — decode + transcribe an audio file into timestamped segments.
* ``summarize`` — summarize a transcript JSON into a versioned summary artifact.
* ``analyze`` — run audio -> transcribe -> summarize in one pass.
* ``benchmark`` — run the transcribe path with per-stage timing and emit a
  validated benchmark report.

The transcribe backend is :class:`~localmind.stt.WhisperTranscriber` when
``mlx-whisper`` is installed and a model tier is provisioned; ``--mock`` uses
:class:`~localmind.stt.MockTranscriber` so the contract is exercisable with no
ML backend. Likewise ``--mock`` uses :class:`~localmind.summary.MockSummaryLLM`
for the summarize/analyze paths.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import IO, List, Optional

from localmind.audio.errors import AudioError
from localmind.bench import REPORT_SCHEMA_VERSION, BenchmarkReport, validate_report_dict
from localmind.bench.report import (
    ASPIRATIONAL_PEAK_MEM_GB,
    ASPIRATIONAL_RTF,
    PeakMemory,
    StageTiming,
)
from localmind.provisioning.errors import ProvisioningError
from localmind.provisioning.provisioner import Provisioner
from localmind.stt import (
    ChunkingConfig,
    MockTranscriber,
    WhisperTranscriber,
    audio_source_from_path,
)
from localmind.stt.segment import TranscriptSegment
from localmind.stt.transcriber import Transcriber
from localmind.summary import MockSummaryLLM, MlxLmSummaryLLM, Summarizer, SummaryLLM

# Bumping this is a breaking change caught by the contract tests.
CLI_OUTPUT_SCHEMA_VERSION = "1"


class CliError(Exception):
    """A user-facing CLI error, rendered as structured JSON."""


def _emit_json(out: IO, obj: dict) -> None:
    json.dump(obj, out, sort_keys=True)
    out.write("\n")
    out.flush()


def _emit_progress(err: IO, event: dict) -> None:
    payload = {"event": "progress"}
    payload.update(event)
    json.dump(payload, err, sort_keys=True)
    err.write("\n")
    err.flush()


def _segment_to_dict(seg) -> dict:
    return {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}


def _provenance_to_dict(provenance) -> Optional[dict]:
    if provenance is None:
        return None
    return {
        "tier": provenance.tier,
        "model_id": provenance.model_id,
        "model_path": str(provenance.model_path),
        "sha256": provenance.sha256,
        "quant_format": provenance.quant_format,
    }


def _select_transcriber(args) -> Transcriber:
    if args.mock:
        return MockTranscriber()
    return WhisperTranscriber(language=getattr(args, "language", None))


def cmd_transcribe(args, out: IO, err: IO) -> int:
    try:
        source = audio_source_from_path(args.audio, target_sample_rate=16000)
    except (AudioError, OSError) as exc:
        raise CliError(f"cannot open audio source {args.audio}: {exc}") from exc

    config = ChunkingConfig(
        chunk_duration_sec=args.chunk_sec, overlap_sec=args.overlap_sec,
        use_vad=bool(args.vad),
    )
    transcriber = _select_transcriber(args)
    provisioner = None if args.mock else Provisioner(args.model_dir)
    tier = "mock" if args.mock else args.tier

    def on_progress(fraction: float) -> None:
        if not args.no_progress:
            _emit_progress(err, {"stage": "stt", "fraction": round(float(fraction), 4)})

    segments = transcriber.transcribe(source, config, provisioner, tier, on_progress)
    provenance = getattr(transcriber, "last_provenance", None)

    result = {
        "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
        "command": "transcribe",
        "audio": {
            "path": str(Path(args.audio)),
            "duration_sec": source.duration_sec,
        },
        "model_tier": tier,
        "mock": bool(args.mock),
        "segments": [_segment_to_dict(s) for s in segments],
        "provenance": _provenance_to_dict(provenance),
    }
    _emit_json(out, result)
    return 0


def _peak_memory(mock: bool = False) -> List[PeakMemory]:
    """Measure CPU RSS and (when available) MLX Metal GPU memory.

    When ``mock`` is True (mock CLI paths), the MLX probe is skipped entirely
    to avoid importing mlx.core, which registers an atexit hook that can write
    a Metal RuntimeError to process stderr on headless sessions. Mock paths
    report GPU memory as ``(0, "metal_unavailable")`` without side effects.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes = int(usage.ru_maxrss * (1024 if sys.platform.startswith("linux") else 1))
    if mock:
        gpu_bytes, gpu_method = 0, "metal_unavailable"
    else:
        gpu_bytes, gpu_method = _try_mlx_gpu_memory()
    return [
        PeakMemory(rss_bytes, "cpu", "resource_tracker"),
        PeakMemory(gpu_bytes, "gpu", gpu_method),
    ]


def _try_mlx_gpu_memory():
    """Try to measure MLX Metal peak GPU memory.

    Returns ``(bytes, method)``: when Metal is available and MLX is installed,
    returns the measured peak memory via ``mx.get_peak_memory()`` (or the legacy
    ``mx.metal.get_peak_memory()``) with method ``"mlx_memory"``. When Metal is
    unavailable (headless/sandbox) or MLX is not installed, returns
    ``(0, "metal_unavailable")``.

    Deprecation warnings from the legacy API are suppressed so they do not
    contaminate the CLI's JSONL progress stream on stderr.
    """
    try:
        import warnings
        import mlx.core as mx
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                peak = mx.get_peak_memory()  # new API (MLX 0.31+)
            except AttributeError:
                peak = mx.metal.get_peak_memory()  # legacy API
        return int(peak), "mlx_memory"
    except Exception:
        return 0, "metal_unavailable"


def cmd_benchmark(args, out: IO, err: IO) -> int:
    config = ChunkingConfig(
        chunk_duration_sec=args.chunk_sec, overlap_sec=args.overlap_sec,
        use_vad=bool(args.vad),
    )
    transcriber = _select_transcriber(args)
    provisioner = None if args.mock else Provisioner(args.model_dir)
    tier = "mock" if args.mock else args.tier

    # The decode stage owns source setup: container open, header parse, and
    # (for compressed sources) ffmpeg duration probing. Per-window reads happen
    # later inside the transcriber and are accounted to the stt stage.
    t_decode_start = time.perf_counter()
    try:
        source = audio_source_from_path(args.audio, target_sample_rate=16000)
    except (AudioError, OSError) as exc:
        raise CliError(f"cannot open audio source {args.audio}: {exc}") from exc
    decode_sec = time.perf_counter() - t_decode_start

    def on_progress(fraction: float) -> None:
        if not args.no_progress:
            _emit_progress(err, {"stage": "stt", "fraction": round(float(fraction), 4)})

    t_stt_start = time.perf_counter()
    segments = transcriber.transcribe(source, config, provisioner, tier, on_progress)
    stt_sec = time.perf_counter() - t_stt_start

    audio_duration = source.duration_sec
    total = decode_sec + stt_sec  # llm + persist stages are not run yet
    rtf = (total / audio_duration) if audio_duration > 0 else 0.0

    report = BenchmarkReport(
        schema_version=REPORT_SCHEMA_VERSION,
        run_id=f"bench-{Path(args.audio).stem}",
        case_id=Path(args.audio).stem,
        audio_duration_sec=audio_duration,
        stages=[
            StageTiming("decode", decode_sec),
            StageTiming("stt", stt_sec),
            StageTiming("llm", 0.0),
            StageTiming("persist", 0.0),
        ],
        total_duration_sec=total,
        rtf=rtf,
        peak_memory=_peak_memory(mock=args.mock),
        model_tiers={"stt": tier},
        hardware={"python": sys.version.split()[0]},
        aspirational_targets={
            "peak_mem_gb": ASPIRATIONAL_PEAK_MEM_GB,
            "rtf": ASPIRATIONAL_RTF,
        },
        notes=(
            "llm/persist stages not run (structured-summary and persistence "
            "work pending); gpu memory is 0 because the ML backend is not installed"
            if args.mock else ""
        ),
    )
    # Validate before emitting so a malformed report never leaves the CLI.
    validate_report_dict(report.to_dict())
    _emit_json(out, report.to_dict())
    return 0


def _select_summary_llm(args) -> SummaryLLM:
    if args.mock:
        return MockSummaryLLM()
    return MlxLmSummaryLLM(Provisioner(args.model_dir), args.llm_tier)


def _load_segments(transcript_path) -> List[TranscriptSegment]:
    try:
        data = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(f"cannot read transcript {transcript_path}: {exc}") from exc
    if isinstance(data, dict):
        segs = data.get("segments", [])
    elif isinstance(data, list):
        segs = data
    else:
        raise CliError("transcript must be a JSON object with 'segments' or a JSON array")
    if not isinstance(segs, list) or not segs:
        raise CliError("transcript has no segments")
    segments = []
    for i, s in enumerate(segs):
        if not isinstance(s, dict) or "text" not in s:
            raise CliError(f"transcript segment {i} is malformed")
        segments.append(TranscriptSegment(
            id=str(s.get("id") or f"seg-{i:04d}"),
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            text=str(s.get("text", "")),
        ))
    return segments


def cmd_summarize(args, out: IO, err: IO) -> int:
    segments = _load_segments(args.transcript)
    llm = _select_summary_llm(args)
    summarizer = Summarizer(llm, model_id="mock-llm" if args.mock else args.llm_tier)
    case_id = Path(args.transcript).stem

    if not args.no_progress:
        _emit_progress(err, {"stage": "summarize", "fraction": 0.0})
    summary = summarizer.summarize(segments, case_id=case_id)
    if not args.no_progress:
        _emit_progress(err, {"stage": "summarize", "fraction": 1.0})

    _emit_json(out, summary)
    return 0


def _build_metrics(decode_sec, stt_sec, llm_sec, persist_sec, audio_duration, mock=False):
    total = decode_sec + stt_sec + llm_sec + persist_sec
    rtf = (total / audio_duration) if audio_duration > 0 else 0.0
    return {
        "stages": [
            {"stage": "decode", "duration_sec": decode_sec},
            {"stage": "stt", "duration_sec": stt_sec},
            {"stage": "llm", "duration_sec": llm_sec},
            {"stage": "persist", "duration_sec": persist_sec},
        ],
        "total_duration_sec": total,
        "rtf": rtf,
        "peak_memory": [m.to_dict() for m in _peak_memory(mock=mock)],
    }


def cmd_analyze(args, out: IO, err: IO) -> int:
    transcriber = _select_transcriber(args)
    provisioner = None if args.mock else Provisioner(args.model_dir)
    stt_tier = "mock" if args.mock else args.tier
    config = ChunkingConfig(chunk_duration_sec=args.chunk_sec, overlap_sec=args.overlap_sec, use_vad=bool(args.vad))

    # Stage: decode (source setup: open/header/probe).
    t0 = time.perf_counter()
    try:
        source = audio_source_from_path(args.audio, target_sample_rate=16000)
    except (AudioError, OSError) as exc:
        raise CliError(f"cannot open audio source {args.audio}: {exc}") from exc
    decode_sec = time.perf_counter() - t0

    def on_stt(fraction: float) -> None:
        if not args.no_progress:
            _emit_progress(err, {"stage": "stt", "fraction": round(float(fraction), 4)})

    # Stage: STT.
    t0 = time.perf_counter()
    segments = transcriber.transcribe(source, config, provisioner, stt_tier, on_stt)
    stt_sec = time.perf_counter() - t0
    stt_provenance = getattr(transcriber, "last_provenance", None)

    # Stage: LLM summarize.
    llm = _select_summary_llm(args)
    summarizer = Summarizer(llm, model_id="mock-llm" if args.mock else args.llm_tier)
    if not args.no_progress:
        _emit_progress(err, {"stage": "summarize", "fraction": 0.0})
    t0 = time.perf_counter()
    summary = summarizer.summarize(segments, case_id=Path(args.audio).stem)
    llm_sec = time.perf_counter() - t0
    if not args.no_progress:
        _emit_progress(err, {"stage": "summarize", "fraction": 1.0})

    # Stage: persist (if --store). The measured persist duration feeds both the
    # stored and the stdout metrics so they match.
    store_run_id = None
    if getattr(args, "store", None):
        store_run_id, metrics = _persist_run(
            args, source, segments, stt_provenance, summarizer, summary, llm,
            decode_sec, stt_sec, llm_sec,
        )
    else:
        metrics = _build_metrics(decode_sec, stt_sec, llm_sec, 0.0, source.duration_sec, mock=bool(args.mock))

    result = {
        "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
        "command": "analyze",
        "audio": {"path": str(Path(args.audio)), "duration_sec": source.duration_sec},
        "stt_tier": stt_tier,
        "mock": bool(args.mock),
        "transcript": {
            "segments": [_segment_to_dict(s) for s in segments],
            "provenance": _provenance_to_dict(stt_provenance),
        },
        "summary": summary,
        "metrics": metrics,
    }
    if store_run_id is not None:
        result["store_run_id"] = store_run_id
    _emit_json(out, result)
    return 0


def _persist_run(args, source, segments, stt_provenance, summarizer, summary, llm,
                 decode_sec, stt_sec, llm_sec):
    """Persist an analyze run atomically with measured metrics; return (run_id, metrics).

    Uses ``put_full_run_with_metrics`` so the full run + metrics are committed
    in a single transaction. The persist duration is measured inside the
    transaction (through the last INSERT), and ``metrics_json`` is written
    before commit. If anything fails, the entire run rolls back — no orphaned
    rows.
    """
    from localmind.store import ReferenceIntegrityError, Store

    stt_prov = _provenance_to_dict(stt_provenance) or {}
    status = "failed" if summary.get("status") == "failed" else "ok"
    stt_ref = {
        "model_id": stt_prov.get("model_id") or ("mock" if args.mock else args.tier),
        "kind": "whisper",
        "sha256": stt_prov.get("sha256", ""),
        "quant_format": stt_prov.get("quant_format", ""),
        "path": stt_prov.get("model_path", ""),
    }
    llm_prov = getattr(llm, "last_provenance", None)
    llm_ref = {
        "model_id": (llm_prov.model_id if llm_prov else summarizer.model_id),
        "kind": "llm",
        "sha256": (llm_prov.sha256 if llm_prov else ""),
        "quant_format": (llm_prov.quant_format if llm_prov else ""),
        "path": (str(llm_prov.model_path) if llm_prov else ""),
    }

    def build_metrics(persist_sec, audio_duration):
        return _build_metrics(decode_sec, stt_sec, llm_sec, persist_sec, audio_duration, mock=bool(args.mock))

    store = Store(args.store)
    try:
        run_id, metrics = store.put_full_run_with_metrics(
            audio={
                "path": args.audio, "duration_sec": source.duration_sec,
                "sample_rate": 16000,
                "format": Path(args.audio).suffix.lower().lstrip("."),
            },
            run={
                "stt_tier": ("mock" if args.mock else args.tier),
                "stt_model_id": stt_ref["model_id"],
                "stt_sha256": stt_ref["sha256"],
                "llm_model_id": summarizer.model_id,
                "prompt_template_hash": summarizer.prompt_template_hash,
                "chunk_duration_sec": args.chunk_sec,
                "overlap_sec": args.overlap_sec,
                "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
                "status": status,
            },
            model_refs=[stt_ref, llm_ref],
            segments=segments,
            summary=summary,
            build_metrics=build_metrics,
        )
        return run_id, metrics
    except ReferenceIntegrityError as exc:
        raise CliError(f"summary failed reference-integrity check: {exc}") from exc
    finally:
        store.close()


def cmd_vad(args, out: IO, err: IO) -> int:
    """Run voice activity detection and emit the speech segments as JSON.

    No model is required — VAD is pure NumPy over decoded PCM. Useful on its
    own (find speech regions / silence) and as a building block for
    transcribe-only-speech.
    """
    from localmind.vad import VadConfig, detect_speech

    _emit_progress(err, {"stage": "vad", "fraction": 0.0})
    try:
        from localmind.audio.decode import decode_audio
        decoded = decode_audio(args.audio)
    except AudioError as exc:
        raise CliError(str(exc)) from exc
    cfg = VadConfig(
        min_speech_sec=args.min_speech_sec,
        min_silence_sec=args.min_silence_sec,
        speech_rise_db=args.speech_rise_db,
        speech_fall_db=args.speech_fall_db,
    )
    segments = detect_speech(decoded.samples, decoded.sample_rate, cfg)
    _emit_progress(err, {"stage": "vad", "fraction": 1.0})

    speech_total = sum(s.duration_sec for s in segments)
    silence_total = max(0.0, decoded.duration_sec - speech_total)
    _emit_json(out, {
        "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
        "command": "vad",
        "audio": {"path": args.audio, "duration_sec": decoded.duration_sec},
        "config": {
            "min_speech_sec": cfg.min_speech_sec,
            "min_silence_sec": cfg.min_silence_sec,
            "speech_rise_db": cfg.speech_rise_db,
            "speech_fall_db": cfg.speech_fall_db,
        },
        "speech_segments": [
            {"start": round(s.start_sec, 3), "end": round(s.end_sec, 3),
             "duration_sec": round(s.duration_sec, 3)}
            for s in segments
        ],
        "speech_total_sec": round(speech_total, 3),
        "silence_total_sec": round(silence_total, 3),
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localmind", description="LocalMind Audio CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("audio", help="path to a .wav/.m4a/.mp3/.aac audio file")
        p.add_argument("--model-dir", default="models", help="provisioned model directory")
        p.add_argument("--tier", default="whisper-small", help="manifest model_id for the STT tier")
        p.add_argument("--mock", action="store_true", help="use MockTranscriber (no ML backend)")
        p.add_argument("--chunk-sec", type=float, default=30.0, help="chunk window seconds")
        p.add_argument("--overlap-sec", type=float, default=1.0, help="chunk overlap seconds")
        p.add_argument("--vad", action="store_true",
                       help="VAD-driven chunking: skip silence and split chunks at speech "
                            "boundaries (faster on talky-with-gaps audio)")
        p.add_argument("--no-progress", action="store_true", help="suppress JSONL progress events")
        p.add_argument("--language", default=None, help="spoken language hint (Whisper backend)")

    p_transcribe = sub.add_parser("transcribe", help="transcribe audio to timestamped segments")
    add_common(p_transcribe)
    p_transcribe.set_defaults(func=cmd_transcribe)

    p_bench = sub.add_parser("benchmark", help="run transcribe with per-stage timing")
    add_common(p_bench)
    p_bench.set_defaults(func=cmd_benchmark)

    p_summarize = sub.add_parser("summarize", help="summarize a transcript JSON")
    p_summarize.add_argument("transcript", help="path to a transcript JSON (transcribe output)")
    p_summarize.add_argument("--model-dir", default="models", help="provisioned model directory")
    p_summarize.add_argument("--mock", action="store_true", help="use MockSummaryLLM (no LLM backend)")
    p_summarize.add_argument("--llm-tier", default="qwen2.5-7b", help="LLM model id (real backend)")
    p_summarize.add_argument("--no-progress", action="store_true", help="suppress JSONL progress events")
    p_summarize.set_defaults(func=cmd_summarize)

    p_analyze = sub.add_parser("analyze", help="transcribe + summarize in one pass")
    add_common(p_analyze)
    p_analyze.add_argument("--llm-tier", default="qwen2.5-7b", help="LLM model id (real backend)")
    p_analyze.add_argument("--store", default=None, help="persist artifacts to this SQLite store path")
    p_analyze.set_defaults(func=cmd_analyze)

    p_vad = sub.add_parser("vad", help="detect speech segments (skip silence) — no model needed")
    p_vad.add_argument("audio", help="path to a .wav/.m4a/.mp3/.aac audio file")
    p_vad.add_argument("--min-speech-sec", type=float, default=0.25,
                       help="drop speech bursts shorter than this (default 0.25)")
    p_vad.add_argument("--min-silence-sec", type=float, default=0.30,
                       help="merge speech separated by a gap shorter than this (default 0.30)")
    p_vad.add_argument("--speech-rise-db", type=float, default=15.0,
                       help="dB above noise floor to ENTER speech (default 15)")
    p_vad.add_argument("--speech-fall-db", type=float, default=8.0,
                       help="dB above noise floor to LEAVE speech (default 8)")
    p_vad.add_argument("--no-progress", action="store_true", help="suppress JSONL progress events")
    p_vad.set_defaults(func=cmd_vad)

    return parser


def main(argv: Optional[List[str]] = None, out: IO = None, err: IO = None) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args, out, err)
    except KeyboardInterrupt:
        _emit_json(out, {
            "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
            "command": args.command,
            "error": {"code": "cancelled", "message": "interrupted by user"},
        })
        return 130
    except CliError as exc:
        _emit_json(out, {
            "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
            "command": args.command,
            "error": {"code": "cli_error", "message": str(exc)},
        })
        return 1
    except ProvisioningError as exc:
        _emit_json(out, {
            "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
            "command": args.command,
            "error": {"code": "provisioning_error", "message": str(exc)},
        })
        return 1
    except RuntimeError as exc:
        _emit_json(out, {
            "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
            "command": args.command,
            "error": {"code": "runtime_error", "message": str(exc)},
        })
        return 1
    except Exception as exc:
        _emit_json(out, {
            "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
            "command": args.command,
            "error": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"},
        })
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
