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

There is no shortage of local STT (whisper.cpp) or local LLM (Ollama, LM Studio)
tools. They are **parts**. LocalMind Audio is a **complete pipeline** where the
"local / trustworthy" property is an enforced, tested contract — not a setting,
and not a marketing claim.

**Two pillars that make it different:**

1. **Verifiable privacy.** This is not "we promise not to upload." The pipeline
   is proven to run with the network socket layer *blocked*
   (`tests/test_no_network.py` locks `socket` / `create_connection` /
   `getaddrinfo` and the pipeline still succeeds). A missing model fails locally
   with `provisioning_error` — it physically cannot reach the network at runtime.
   For sensitive audio (legal, medical, journalism, research, executive),
   "I can prove this recording never left this machine" is the value.

2. **Anti-supply-chain + anti-hallucination, built into the architecture.**
   Every weight is re-verified by SHA-256 + size *inside the adapter boundary,
   immediately before each inference* — not just once at download. Model
   poisoning or local tampering is caught before the weight is used. And every
   summary is *grounded*: each decision/action cites a transcript segment id;
   when the LLM produces garbage, bounded repair runs and, on failure, a
   `summary_failed` artifact is persisted — **never fabrication**.

**How those pillars are enforced (the evidence):**

- **Zero-network runtime.** Verified by the socket-layer harness.
- **Integrity-pinned models.** SHA-256 + size manifest, re-verified in the
  adapter, paths confined to the model directory (no absolute paths, no `..`).
- **Structured, grounded output.** Versioned JSON schema; citations back to
  segments; bounded repair → `summary_failed` instead of made-up content.
- **Single normalized store.** Audio metadata, transcript, summary, metrics, and
  model provenance land in one SQLite database in a single atomic transaction,
  with referential integrity.

---

## Who is this for

- **Privacy-sensitive professions** — lawyers, doctors, journalists,
  researchers, executives, activists — who cannot send audio to a cloud STT/LLM
  and need to *prove* it stayed local.
- **Developers building local-AI products on Apple Silicon** — this is a stable,
  tested foundation (the CLI/JSONL contract is the boundary) you can wrap in a
  UI instead of re-deriving the provisioning / zero-network / grounding rules.
- **Anyone who distrusts cloud AI** — run your own transcription + summary with
  models you pinned and audited.
- **Compliance & security researchers** — an auditable, reproducible on-device
  AI pipeline sample where the trust properties are testable, not asserted.

> **Honest note on model quality.** The default models (`whisper-tiny` +
> `Qwen3.5-0.8B`) are **smoke-tier** — they prove the pipeline runs and the
> contracts hold, but Chinese transcription lands as Traditional characters
> with small errors and the small LLM follows the schema imperfectly. The
> **architecture and contracts are the reusable part**; swap in
> `whisper-base`/`medium` and `Qwen3-1.7B`/`7B` (see
> [Model provisioning](#model-provisioning)) for production-quality output.

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
| Voice activity detection (energy VAD, skip silence) | ✅ verified (no model needed) |
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
localmind analyze path/to/meeting.m4a \
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
stderr**, so they compose cleanly in scripts and UIs. (After `pip install -e .`
the `localmind` command is on your PATH.) For the full guide — flags, exit
codes, recipes, piping — see **[docs/usage.md](docs/usage.md)**.

```bash
# Transcribe only -> { audio, segments[], provenance }
localmind transcribe audio.wav \
    --model-dir models --tier whisper-tiny --language zh

# Summarize an existing transcript JSON
localmind summarize transcript.json \
    --model-dir models --llm-tier qwen3.5-0.8b

# Transcribe + summarize + persist in one pass
localmind analyze audio.m4a \
    --model-dir models --tier whisper-tiny --language zh \
    --llm-tier qwen3.5-0.8b --store runs.db

# Transcribe with per-stage timing + memory report (benchmark)
localmind benchmark audio.m4a \
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

## Roadmap

The contract (zero-network, integrity-pinned, grounded, normalized store) is
fixed. Features are additive on top of it. Roughly ordered by impact:

**Quality — swap the models, not the pipeline**
- Larger STT tiers (`whisper-base` / `medium` / `large-v3`) via the
  `--stt` flag in the provisioning script — better Chinese, fewer errors.
- Larger / specialized LLM tiers (`Qwen3-1.7B` / `7B`) for reliable schema
  adherence.
- Word-level timestamps for finer-grained citations.
- Auto language detection + multi-language transcripts.

**Pipeline features (currently deferred)**
- ~~VAD (voice activity detection) — skip silence, segment on speech.~~ **✅ done**
  (`localmind vad`; pure NumPy, no model).
- Speaker diarization — "who said what."
- Hot-words / custom vocabulary boost for domain terms.
- Streaming / real-time incremental transcription.
- Audio preprocessing (denoising, normalization).

**Interface & output**
- Swift/SwiftUI native app (file picker, live progress, runs DB browser) over
  the stable CLI/JSONL contract.
- Batch processing of many files.
- Live recording (not just file import).
- Export to Markdown / PDF / `.docx` meeting minutes; SRT/VTT subtitles.
- More prompt templates — interview, lecture, standup — not just meeting minutes.

**Retrieve & reuse the archive**
- Full-text search across stored transcripts.
- RAG over past transcripts ("what did we decide about X last quarter?").
- Calendar integration — auto-attach a summary to the meeting that produced it.

**Platform**
- iOS / iPadOS port (CoreML Whisper + on-device LLM).
- Linux / Windows support via whisper.cpp + llama.cpp backends behind the same
  adapter interface.
- Optional local-server mode (still no outbound network) for a browser UI.

**Trust & ops**
- Signed manifests / model provenance attestation.
- Local audit log of every run.
- Quantization A/B harness (compare 4-bit vs 8-bit quality + speed).

See [Limitations & scope](#limitations--scope) for what is deliberately *not*
planned for the lower bound, and open an issue to discuss anything above.

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
