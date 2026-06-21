AGREE: points accepted as reasonable

The corrected assumptions are sound: STT is an external dependency for now; llama.cpp is the right Swift LLM path; ANE should be removed from performance claims; AVFoundation work is genuinely new.

The vertical slice is well-scoped: local audio import, local transcription, local summarization, schema validation, artifact persistence, and no runtime network. Deferring streaming, diarization, VAD, hot words, and multi-device relay is the right MVP boundary.

M1 before UI is reasonable. Benchmarking real 10/30/60-minute fixtures before committing to Swift packaging avoids building a polished wrapper around an infeasible pipeline.

M2’s structured summary schema with segment citations is a strong product anchor. It gives the LLM output a testable contract instead of treating summarization as free text.

Progress, cancellation, cleanup, timing, peak memory, and failure paths belong in the MVP because hour-long local inference will otherwise feel broken even when technically working.

DISAGREE: points considered unreasonable and why

“Whisper-Large-v3” as the default MVP target may be unreasonable unless the minimum Mac hardware and memory budget are relaxed. Large-v3 plus a 7B/8B INT4 LLM can easily conflict with `<6GB peak` and `RTF<0.08`, especially if both models are resident.

The plan still blurs prototype and product architecture. M1 uses Python + MLX Whisper; M3 uses Swift + llama.cpp, with STT either Swift-port or Python subprocess. That is acceptable as staged exploration, but not yet a converged implementation plan.

“SwiftData” in M3 conflicts with M2’s “SQLite/JSON store” unless the persistence boundary is clarified. SwiftData may wrap SQLite, but the schema, migration story, and artifact-file layout need one source of truth.

“Zero network runtime” is reasonable, but incomplete. mlx-whisper/model loading often assumes local caches or Hugging Face-style model resolution. The plan needs explicit offline model provisioning and tests with network unavailable from first launch.

“Encryption at rest” is underspecified. For voice transcripts this may be important, but App Sandbox + local model files + SQLite/artifact encryption requires a key-management decision. As written it is a large security feature hidden inside M3.

REQUIRED_CHANGES: must-fix items before convergence

Define the MVP hardware/performance matrix before choosing Large-v3: minimum Mac model/RAM, acceptable RTF, peak memory, battery/thermal assumptions, and whether sequential model loading is mandatory.

Split model choices into tiers: required baseline, preferred quality target, and fallback. Example: `whisper-small/medium` for MVP feasibility, `large-v3` as benchmark target unless hardware supports it.

Make the runtime boundary explicit: either Python worker is allowed in the shipped MVP, or it is prototype-only. If Python may ship, add packaging, sandboxing, IPC, crash recovery, and offline dependency handling to the plan.

Unify persistence: choose canonical storage semantics for `audio_asset`, `transcript_segment`, `inference_run`, `prompt_template`, `summary_artifact`, `benchmark_metrics`, and `model_manifest`. SwiftData can be the app layer, but artifact files and schema migrations need to be specified.

Add model provisioning as a first-class milestone or M0: where weights live, how they are imported, checksum/manifest validation, license tracking, quantization format, disk footprint, and startup behavior when models are missing.

Specify audio decode strategy for both prototype and Swift app. If Python needs `ffmpeg`, that is another external dependency. If Swift uses AVFoundation, benchmark behavior may differ from M1.

Define the JSON repair/fail policy tightly: max repair attempts, validation library/schema versioning, whether repair uses the same local LLM, and what artifact is persisted when repair fails.

Clarify peak memory measurement. On macOS, CPU RSS and Metal/GPU allocations are not the same thing; acceptance criteria should say what is measured and how.

OPTIONAL_IMPROVEMENTS: non-blocking improvements

Add an M0 “fixture and harness” milestone: fixed audio fixtures, expected transcript tolerances, expected summary schema examples, and a benchmark report format.

Add deterministic prompt/version tracking: prompt template hash, model manifest hash, decode parameters, chunking parameters, and app version in each `inference_run`.

Treat cancellation/resume separately from cancellation/cleanup. For 60-minute files, resume from completed transcript/summarization chunks may matter soon after MVP.

Add privacy-oriented logging rules: no transcript text in normal logs, explicit debug mode, local-only crash artifacts.

Consider one intermediate CLI that mirrors the final app contract: `transcribe`, `summarize`, `analyze`, `benchmark`, with JSONL progress events. That will make Swift IPC cleaner later.

UNRESOLVED: opposite opinions needing user decisions

Large-v3 mandatory vs quality-tiered MVP.

Python allowed in shipped macOS app vs prototype-only.

Hard performance budgets vs aspirational benchmark targets.

Minimum supported Mac hardware and RAM.

SwiftData as canonical persistence vs explicit SQLite/artifact store with SwiftData as optional app facade.

Encryption-at-rest required for MVP vs deferred behind clear local privacy constraints.

Bundled models vs user-imported models vs offline-prepared model directory.
