# LocalMind Audio

> 100% on-device macOS voice-to-text + local-LLM summarization, with **zero network access at runtime**.

LocalMind Audio turns a local audio file (`.m4a` / `.wav` / `.mp3`) into a
timestamped transcript, then into a structured summary (meeting minutes /
action items) — all running locally on Apple Silicon. No audio, transcript, or
summary ever leaves your machine. There is no server, no API key, and no
runtime download.

```text
┌──────────┐   decode    ┌─────────────┐  transcribe  ┌──────────────┐  summarize  ┌────────────────┐
│ .m4a/.wav │ ─────────▶ │ mono 16kHz  │ ───────────▶ │ Whisper (MLX)│ ──────────▶ │ structured     │
│  .mp3     │            │  float32 PCM│              │  timestamped │             │ summary (LLM)  │
└──────────┘            └─────────────┘              │  segments    │             │ + provenance   │
                                                     └──────┬───────┘             └───────┬────────┘
                                                            │                             │
                                                            └───────────┬─────────────────┘
                                                                        ▼
                                                          ┌──────────────────────────────┐
                                                          │  normalized SQLite store     │
                                                          │  (audio · transcript ·       │
                                                          │   summary · metrics · refs)  │
                                                          └──────────────────────────────┘
```

---

## Why

Cloud STT/LLM tools leak meeting content, charge subscriptions, and stop
working offline. LocalMind Audio is built for the case where the audio is
sensitive (commercial discussions, medical, legal) and the machine has a GPU
sitting idle. It uses [MLX](https://github.com/ml-explore/mlx) to drive the
Apple-Silicon unified memory directly, so inference is fast and nothing leaves
the process.

**Guarantees baked into the design:**

- **Zero-network runtime.** The pipeline is verified to succeed with the
  network socket layer blocked. A missing or tampered model fails fast with an
  explicit error — it never silently reaches for the network.
- **Integrity-pinned models.** Every weight is pinned by SHA-256 + size in a
  manifest and re-verified inside the adapter boundary, immediately before use.
- **Structured, grounded output.** Summaries conform to a versioned JSON schema;
  every decision/action cites a transcript segment id. Invalid LLM output goes
  through bounded repair and, if still invalid, is persisted as a
  `summary_failed` artifact rather than fabricated.
- **Single normalized store.** All artifacts (audio metadata, transcript,
  summary, metrics, model provenance) land in one SQLite database in a single
  atomic transaction, with referential integrity.

---

## Status

| Area | State |
|------|-------|
| Audio decode (wav/m4a/mp3 → 16kHz mono PCM) | ✅ verified |
| Chunked Whisper transcription (bounded memory) | ✅ verified end-to-end with `whisper-tiny` |
| Structured summary (map→reduce, bounded repair) | ✅ verified end-to-end with `Qwen3.5-0.8B` |
| Normalized SQLite persistence (atomic, reopenable) | ✅ verified |
| Zero-network runtime contract | ✅ verified (socket-layer blocked harness) |
| CLI JSON/JSONL contract + progress/cancel | ✅ verified |
| Offline model provisioning (SHA-256/size) | ✅ verified |
| Per-stage metrics + real MLX GPU memory | ✅ verified |
| Swift/SwiftUI native wrapper (M3) | 🚧 not yet (Python CLI is the stable contract) |

The Python pipeline (M0–M2) is complete and tested. The optional native
wrapper (M3) builds on top of the stable CLI/JSONL contract.

---

## Requirements

- **Apple Silicon Mac** (M1+) with a working Metal device. MLX does not run on
  Intel or headless/VM sessions without a GPU.
- Python **3.11+**
- `ffmpeg` on `PATH` for compressed audio (`.m4a`/`.mp3`), **or** the bundled
  `imageio-ffmpeg` fallback (installed automatically).
- ~2 GB disk for the default model pair, ~4 GB free RAM headroom.

---

## Quickstart

```bash
# 1. Clone + create a venv
git clone <your-fork-url> localmind-audio
cd localmind-audio
python3.11 -m venv .venv && source .venv/bin/activate

# 2. Install the project + ML backend
pip install -e ".[ml]"

# 3. Provision models (one-time, downloads ~1.8 GB; needs network THIS ONCE)
python scripts/provision_models.py

# 4. Run the full pipeline on an audio file (zero network from here on)
python -m localmind.cli analyze path/to/meeting.m4a \
    --model-dir models --tier whisper-tiny --language zh \
    --llm-tier qwen3.5-0.8b --store localmind.db
```

The `analyze` command writes a JSON document to stdout (the full run: audio
metadata, transcript segments, summary, per-stage metrics with real GPU memory)
and persists it to the SQLite store. Progress events stream to stderr as
newline-delimited JSON (JSONL).

---

## CLI

All four subcommands emit **versioned JSON to stdout** and **JSONL progress to
stderr**, so they compose cleanly in scripts and UIs.

```bash
# Transcribe only -> { audio, segments[], provenance }
python -m localmind.cli transcribe audio.wav \
    --model-dir models --tier whisper-tiny --language zh

# Summarize an existing transcript JSON
python -m localmind.cli summarize transcript.json \
    --model-dir models --llm-tier qwen3.5-0.8b

# Transcribe + summarize + persist in one pass
python -m localmind.cli analyze audio.m4a \
    --model-dir models --tier whisper-tiny --language zh \
    --llm-tier qwen3.5-0.8b --store runs.db

# Transcribe with per-stage timing + memory report (benchmark)
python -m localmind.cli benchmark audio.m4a \
    --model-dir models --tier whisper-tiny --chunk-sec 30 --overlap-sec 2
```

Common flags: `--mock` (offline contract run, no backend), `--no-progress`
(silence JSONL on stderr), `--chunk-sec` / `--overlap-sec` (bounded chunking
for long audio). Run any subcommand with `--help` for the full list.

---

## Model provisioning

Models are provisioned **out-of-band** — downloaded once by you, never at
runtime. `scripts/provision_models.py` handles the default pair and writes the
integrity manifest:

```bash
python scripts/provision_models.py                 # whisper-tiny + Qwen3.5-0.8B
python scripts/provision_models.py --stt-only      # just the STT tier
python scripts/provision_models.py \
    --stt mlx-community/whisper-base-mlx \
    --llm mlx-community/Qwen3-1.7B-4bit            # swap in larger tiers
```

The manifest (`models/models.json`) pins each weight by SHA-256 and size and
confines paths to the model directory (no absolute paths, no `..` traversal).
The full layout, schema, and verification rules are documented in
[docs/provisioning.md](docs/provisioning.md).

> **Quality note.** `whisper-tiny` is the smallest STT tier — great for a smoke
> test, but it transcribes Mandarin to Traditional characters with some
> misrecognition. For production Chinese audio, provision `whisper-base` or
> `whisper-medium`. Likewise, `Qwen3.5-0.8B` follows the structured schema but
> is a small model; `Qwen3-1.7B` or larger gives more reliable summaries.

---

## Architecture

```
localmind/
├── audio/            # decode .wav/.m4a/.mp3 -> mono 16kHz float32 PCM
├── provisioning/     # manifest schema + integrity-verified Provisioner
├── stt/              # bounded AudioSource, chunking, WhisperTranscriber
├── summary/          # versioned schema, map->reduce summarizer, bounded repair
├── bench/            # benchmark report schema + synthetic fixtures
├── store.py          # normalized SQLite store (single atomic transaction)
├── mlx_runtime.py    # subprocess Metal preflight (keeps stderr pure)
└── cli.py            # transcribe / summarize / analyze / benchmark
```

**Key boundaries:**

- **Provisioning before backend.** Each adapter resolves its tier through the
  `Provisioner` (re-verifying SHA-256/size) *before* importing the backend
  library. A missing/tampered model fails with a structured `provisioning_error`
  regardless of whether the backend is installed.
- **Bounded memory.** `AudioSource` reads compressed audio in windows via
  ffmpeg (`-ss`/`-t`), so a 60-minute file never materializes in RAM. STT
  processes chunk-by-chunk with overlap-aware segment merging.
- **Subprocess Metal preflight.** `mlx_runtime.ensure_mlx_metal_available()`
  checks Metal in a child process so the parent (and the CLI's JSONL progress
  stream on stderr) never gets polluted by MLX's atexit noise on a headless box.
- **Single normalized store.** `Store.put_full_run_with_metrics()` writes all
  five artifact tables + metrics in one transaction; any failure rolls back with
  no orphaned rows.

---

## Development

```bash
pip install -e ".[dev,ml]"          # dev = pytest; ml = mlx-whisper + mlx-lm
pytest                              # full suite (170+ tests)
pytest tests/test_no_network.py     # the zero-network contract harness
```

The zero-network test blocks the socket layer (`socket`, `create_connection`,
`getaddrinfo`) and asserts the full mock pipeline still succeeds, and that a
non-mock run with a missing model fails locally with `provisioning_error`
rather than attempting a download.

See [docs/benchmark.md](docs/benchmark.md) for the benchmark methodology and
report schema.

---

## Limitations & scope

**In scope (and done):** local import → transcribe → structured summary →
local persistence, zero-network at runtime, on Apple Silicon.

**Deliberately deferred** (out of scope for the lower bound):
multi-device relay (Watch/iPhone/Mac sync), streaming transcription, VAD,
hot-words, speaker diarization, denoising, and the optional Swift/SwiftUI
native wrapper (M3). These do not block the on-device pipeline contract.

**Verified-on hardware:** Apple Silicon with Metal. Intel Macs, and headless
CI runners without a GPU, are not supported by the MLX backend (the code
detects this and fails cleanly).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The test suite is the contract — every
change should keep `pytest` green and the zero-network harness passing. Source
files must stay free of internal plan/milestone terminology.

---

## License

[MIT](LICENSE). The bundled models keep their own licenses (Whisper: MIT;
Qwen: Apache-2.0) — see `models/models.json` after provisioning.
