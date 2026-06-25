# Usage Guide

A practical, copy-pasteable guide to the `localmind` CLI. For the big picture
see the [README](../README.md); for model setup see
[provisioning](provisioning.md).

After `pip install -e ".[ml]"` and `python scripts/provision_models.py`, the
`localmind` command is available in your shell. Five subcommands:

| Command | What it does | Output (stdout) |
|---------|--------------|-----------------|
| `localmind transcribe` | audio → timestamped transcript segments | `{audio, segments[], provenance}` |
| `localmind summarize` | transcript JSON → structured summary | `{summary, provenance}` or `summary_failed` |
| `localmind analyze` | transcribe **+** summarize **+** persist in one pass | full run incl. metrics |
| `localmind benchmark` | transcribe with per-stage timing + memory | benchmark report |
| `localmind vad` | detect speech segments / skip silence — **no model needed** | `{audio, speech_segments[], speech_total, silence_total}` |

Every command writes **versioned JSON to stdout** and **JSONL progress to
stderr** (or nothing on stderr with `--no-progress`). Stdout is always a single
JSON document; stderr is always one JSON object per line. This makes the CLI
safe to drive from scripts and UIs.

---

## 1. Transcribe

```bash
localmind transcribe audio.wav \
    --model-dir models --tier whisper-tiny --language zh
```

- `--model-dir models` — the directory holding `models.json` + weights.
- `--tier whisper-tiny` — the `model_id` of the STT tier (from `models.json`).
- `--language zh` — spoken-language hint (Whisper). Omit for auto-detect.
- `--chunk-sec 30 --overlap-sec 2` — bounded chunking for long audio (default
  already bounded; tune for latency vs. boundary accuracy).

**stdout (abbreviated):**

```json
{
  "schema_version": "1",
  "command": "transcribe",
  "audio": {"path": "audio.wav", "duration_sec": 16.095},
  "model_tier": "whisper-tiny",
  "provenance": {"model_id": "whisper-tiny", "sha256": "0e03a5…", "tier": "whisper-tiny"},
  "segments": [
    {"id": "seg-0000", "start": 0.0, "end": 5.12, "text": "大家好…"},
    {"id": "seg-0001", "start": 5.12, "end": 8.76, "text": "会议讨论了…"}
  ]
}
```

Supported inputs: `.wav`, `.m4a`, `.mp3`, `.aac`. `.wav` decodes with the stdlib;
the rest decode via `ffmpeg` (system `ffmpeg` if present, else the bundled
`imageio-ffmpeg`).

---

## 2. Summarize

Takes a transcript JSON (the `transcribe` output, or any file with a
`segments` array) and produces a structured summary.

```bash
localmind summarize transcript.json \
    --model-dir models --llm-tier qwen3.5-0.8b
```

- `--llm-tier qwen3.5-0.8b` — the `model_id` of the LLM tier (from `models.json`).

**stdout** is a versioned summary (`soundmind.summary.v1`):

```json
{
  "schema_version": "soundmind.summary.v1",
  "case_id": "transcript",
  "provenance": {"model_id": "qwen3.5-0.8b", "repaired": true, "repair_attempts_used": 1},
  "decisions": [
    {"text": "产品路线图和发布计划", "citations": ["seg-0001"]}
  ],
  "action_items": [
    {"text": "张三负责下周前端开发", "owner": "张三", "due_date": "2024-12-15", "citations": ["seg-0002"]}
  ],
  "open_questions": ["发布计划的具体执行细节？"]
}
```

Every decision and action item **cites the transcript segment it came from**. If
the LLM output is invalid, bounded repair runs once; if it still fails, you get
a `summary_failed` artifact (`status: "failed"`, with `errors` and `raw_output`)
— the pipeline never fabricates a summary.

---

## 3. Analyze (transcribe + summarize + persist)

The one-shot end-to-end command most users want:

```bash
localmind analyze meeting.m4a \
    --model-dir models --tier whisper-tiny --language zh \
    --llm-tier qwen3.5-0.8b --store runs.db
```

- `--store runs.db` — persist the full run to SQLite. Omit to print without saving.

The stdout JSON contains **everything**: audio metadata, transcript segments,
summary, and per-stage metrics (decode / stt / llm / persist durations, RTF,
peak CPU RSS + real MLX GPU memory, model provenance for both tiers).
`store_run_id` identifies the persisted run.

---

## 4. Benchmark

Same as transcribe, but emits a validated benchmark report with the project's
aspirational targets for comparison (`peak_mem_gb < 6`, `RTF < 0.08`):

```bash
localmind benchmark audio.m4a \
    --model-dir models --tier whisper-tiny \
    --chunk-sec 30 --overlap-sec 2
```

See [benchmark.md](benchmark.md) for the report schema.

---

## 5. VAD (voice activity detection)

Find the spoken regions of an audio file — and how much is silence. **No model
required**: VAD runs pure NumPy over decoded PCM, so it works with no
provisioned weights and adds nothing to the zero-network footprint.

```bash
localmind vad meeting.m4a
```

**stdout:**

```json
{
  "command": "vad",
  "audio": {"path": "meeting.m4a", "duration_sec": 16.1},
  "speech_segments": [
    {"start": 1.07, "end": 7.27, "duration_sec": 6.2},
    {"start": 9.65, "end": 11.39, "duration_sec": 1.74}
  ],
  "speech_total_sec": 11.89,
  "silence_total_sec": 4.21
}
```

Tune the detector with:

- `--min-speech-sec` (default 0.25) — drop speech bursts shorter than this.
- `--min-silence-sec` (default 0.30) — merge speech separated by a gap shorter
  than this (a short pause inside one utterance shouldn't split it).
- `--speech-rise-db` / `--speech-fall-db` (default 15 / 8) — hysteresis: how
  many dB above the noise floor a frame must rise to *enter* / *leave* speech.

Use cases: estimate how much of a long recording is actual talk before
transcribing, locate the talk to skip dead air, or feed the speech intervals
into downstream segmentation.

---

## Common flags

| Flag | Applies to | Meaning |
|------|-----------|---------|
| `--model-dir DIR` | all | model directory with `models.json` (default `models`) |
| `--tier ID` | transcribe/benchmark/analyze | STT tier `model_id` |
| `--llm-tier ID` | summarize/analyze | LLM tier `model_id` |
| `--language CODE` | transcribe/benchmark/analyze | spoken-language hint |
| `--chunk-sec N` | STT commands | chunk length in seconds |
| `--overlap-sec N` | STT commands | overlap between chunks (smoother segment merge) |
| `--mock` | all | run the full contract **without** any backend (offline smoke) |
| `--no-progress` | all | silence JSONL progress on stderr |
| `--store PATH` | analyze | persist the run to this SQLite file |

Run any command with `--help` for the complete list, e.g. `localmind analyze --help`.

---

## Recipes

**Pipe transcribe → summarize without a temp file:**

```bash
localmind transcribe meeting.wav --model-dir models --tier whisper-tiny --language zh --no-progress \
  | localmind summarize /dev/stdin --model-dir models --llm-tier qwen3.5-0.8b
```

**Test the contract with no models installed (offline):**

```bash
localmind analyze audio.wav --mock --no-progress
```

**Pretty-print JSON output:**

```bash
localmind transcribe audio.wav --model-dir models --tier whisper-tiny --no-progress | jq .
```

**Only show the transcript text, no JSON:**

```bash
localmind transcribe audio.wav --model-dir models --tier whisper-tiny --no-progress \
  | jq -r '.segments[] | "\(.start)s-\(.end)s: \(.text)"'
```

---

## Exit codes & errors

- `0` — success.
- `1` — structured error (JSON on stdout with an `error` object). Codes:
  - `provisioning_error` — model missing/tampered/undeclared (fails before any
    network/backend). Fix by provisioning: `python scripts/provision_models.py`.
  - `runtime_error` — backend unusable (e.g. MLX Metal device unavailable on a
    headless host) or model failed to load.
  - `cli_error` — bad arguments / unreadable input.
  - `internal_error` — unexpected backend failure (catch-all).
- `130` — interrupted with Ctrl-C (a `cancelled` JSON is emitted).

stderr stays pure JSONL throughout (no tracebacks leak), so you can reliably
parse progress and errors from scripts.
