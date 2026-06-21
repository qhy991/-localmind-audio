"""Command-line interface: a stable JSON/JSONL contract.

The CLI is the contract that the future SwiftUI wrapper (and any other caller)
sits on top of. It emits:

* a versioned JSON final result on stdout, and
* newline-delimited JSON (JSONL) progress events on stderr.

Subcommands (the transcribe path is contract-tested now; ``summarize`` /
``analyze`` arrive with the structured-summary work):

* ``transcribe`` — decode + transcribe an audio file into timestamped segments.
* ``benchmark`` — run the transcribe path with per-stage timing and emit a
  validated benchmark report.

The transcribe backend is :class:`~localmind.stt.WhisperTranscriber` when
``mlx-whisper`` is installed and a model tier is provisioned; ``--mock`` uses
:class:`~localmind.stt.MockTranscriber` so the contract is exercisable with no
ML backend.
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
from localmind.stt.transcriber import Transcriber

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
    try:
        import mlx_whisper  # noqa: F401
    except ImportError as exc:
        raise CliError(
            "mlx-whisper is not installed; pass --mock for an offline contract run, "
            "or install the ML backend with `pip install -e .[ml]` "
            "(see docs/provisioning.md)"
        ) from exc
    return WhisperTranscriber(language=getattr(args, "language", None))


def cmd_transcribe(args, out: IO, err: IO) -> int:
    try:
        source = audio_source_from_path(args.audio, target_sample_rate=16000)
    except (AudioError, OSError) as exc:
        raise CliError(f"cannot open audio source {args.audio}: {exc}") from exc

    config = ChunkingConfig(
        chunk_duration_sec=args.chunk_sec, overlap_sec=args.overlap_sec
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


def _peak_memory() -> List[PeakMemory]:
    """Measure CPU RSS; GPU allocation requires the ML backend (reported as 0)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss_bytes = int(usage.ru_maxrss * (1024 if sys.platform.startswith("linux") else 1))
    return [
        PeakMemory(rss_bytes, "cpu", "resource_tracker"),
        PeakMemory(0, "gpu", "metal_allocated"),
    ]


def cmd_benchmark(args, out: IO, err: IO) -> int:
    try:
        source = audio_source_from_path(args.audio, target_sample_rate=16000)
    except (AudioError, OSError) as exc:
        raise CliError(f"cannot open audio source {args.audio}: {exc}") from exc

    config = ChunkingConfig(
        chunk_duration_sec=args.chunk_sec, overlap_sec=args.overlap_sec
    )
    transcriber = _select_transcriber(args)
    provisioner = None if args.mock else Provisioner(args.model_dir)
    tier = "mock" if args.mock else args.tier

    t_decode_start = time.perf_counter()
    # Source construction already happened above (probes duration / opens the
    # file); account that as the decode stage.
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
        peak_memory=_peak_memory(),
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
        p.add_argument("--no-progress", action="store_true", help="suppress JSONL progress events")
        p.add_argument("--language", default=None, help="spoken language hint (Whisper backend)")

    p_transcribe = sub.add_parser("transcribe", help="transcribe audio to timestamped segments")
    add_common(p_transcribe)
    p_transcribe.set_defaults(func=cmd_transcribe)

    p_bench = sub.add_parser("benchmark", help="run transcribe with per-stage timing")
    add_common(p_bench)
    p_bench.set_defaults(func=cmd_benchmark)

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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
