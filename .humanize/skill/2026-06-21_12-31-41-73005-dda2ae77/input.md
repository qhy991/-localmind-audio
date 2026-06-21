# Ask Codex Input

## Question

Review this CANDIDATE implementation plan for reasonability. It already incorporates your first-pass critique. Repository: greenfield macOS Swift project 'SoundMind' inside monorepo Agent4Kernel.

NEW FACTS established by code exploration (corrections to earlier assumptions):
- There is NO MLX Whisper STT and NO whisper.cpp anywhere in the monorepo. mlx-vlm audio modules are VLM encoders, not transcription. => STT requires an EXTERNAL dependency: mlx-whisper (pip) for the Python prototype, or whisper.cpp later.
- llama.cpp Swift API exists (LlamaContext actor, pull-based completion_loop() token streaming) and is the LLM path for the eventual Swift app.
- ANE is NOT reliably usable by MLX/llama.cpp (GPU path). The spec's 'E-cores + ANE' claim is dropped; targets are GPU+CPU based.
- No AVFoundation audio I/O exists yet.

CANDIDATE PLAN v1:
Title: Mac-First Local Voice-to-Summary Vertical Slice (MVP)
Goal: Smallest end-to-end macOS pipeline: import local audio (.m4a/.wav/.mp3) -> local Whisper-Large-v3 transcription (MLX via mlx-whisper) -> local LLM (Qwen-2.5-7B/Llama-3-8B INT4) prompt template -> schema-valid structured JSON summary -> persist all artifacts locally. Zero network at runtime.

Milestones:
M1 Benchmark-only Python CLI: validate RTF + peak memory on real hardware with 10/30/60-min fixtures. Decide concurrent vs sequential model loading. Establish a stable CLI contract (stdin/args -> JSON out).
M2 Structured analysis + persistence: chunked map-reduce summarization for hour-long transcripts; JSON schema (decisions, action_items{owner,due_date}, open_questions, citations to segment IDs); schema validation + repair/fail path; local SQLite/JSON store with entities (audio_asset, transcript_segment, inference_run, prompt_template, summary_artifact, benchmark_metrics, model_manifest).
M3 Swift/SwiftUI wrap: port the validated CLI contract behind a SwiftUI macOS app; LLM via llama.cpp LlamaContext; persistence via SwiftData; STT either Swift-port or Python inference-worker subprocess with stable IPC contract; App Sandbox + no-network entitlement; encryption at rest.

Acceptance criteria (each will get positive+negative TDD tests):
AC-1 import/decode .m4a/.wav/.mp3 offline; AC-2 timestamped transcript segments for 10/30/60-min fixtures; AC-3 schema-valid JSON summary with repair/fail path; AC-4 persist all named entities locally; AC-5 zero-network runtime (verified by blocking egress); AC-6 per-stage timing (decode/STT/LLM/persist) + peak memory recorded and reported; AC-7 stable CLI contract before SwiftUI; AC-8 long-file progress/cancellation/temp-file cleanup.

Deferred to later product phases (out of scope): multi-device relay, streaming real-time, VAD, hot-words, diarization, iPhone/Watch.

Pending user decisions to surface: min Mac hardware; hard-vs-aspirational budgets (<6GB peak, RTF<0.08); Whisper-Large-v3 mandatory vs medium/small for MVP; Python-in-shipped-product vs prototype-only; most-important MVP output; model provisioning (bundled/import/offline-prep).

Respond using EXACTLY these labeled sections:
AGREE: points accepted as reasonable
DISAGREE: points considered unreasonable and why
REQUIRED_CHANGES: must-fix items before convergence
OPTIONAL_IMPROVEMENTS: non-blocking improvements
UNRESOLVED: opposite opinions needing user decisions

## Configuration

- Model: gpt-5.5
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-06-21_12-31-41
- Tool: codex
