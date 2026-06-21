CORE_RISKS:
- The Mac-first slice is directionally right, but the stated budgets are the riskiest part. `<6 GB` peak for Whisper-Large-v3 4-bit plus a 7B/8B INT4 LLM is tight once KV cache, tokenizer state, audio buffers, Python runtime overhead, and persistence are counted.
- `RTF < 0.08` for transcription plus analysis is an aggressive target, especially for 60-minute audio and long-context summarization. A naive sequential pipeline will likely miss it.
- “E-cores + ANE” is probably an invalid assumption for the chosen stack. MLX/llama.cpp are primarily CPU/GPU paths; ANE scheduling is not something this plan can rely on unless CoreML is in the loop.
- Python CLI first is pragmatic for feasibility, but it creates a packaging cliff later: sandboxing, model paths, subprocess execution, code signing, temp files, and offline dependency management all become macOS product risks.
- “Zero network” is easy to accidentally violate through model downloads, tokenizer fetches, package managers, telemetry, crash reporting, or update checks.
- Summary quality may be poor without timestamps, speaker turns, hotwords, chunking strategy, and confidence metadata. The LLM cannot reliably recover structure from a flat transcript alone.

MISSING_REQUIREMENTS:
- Target hardware matrix: minimum supported Mac, RAM floor, chip generation, macOS version, and whether base M1/M2 machines are in scope.
- Offline model provisioning: where models come from, how large they are, how they are verified, where they are stored, and whether first run requires manual import.
- Audio edge cases: stereo/mono, sample-rate conversion, long files, corrupt files, low-volume recordings, mixed languages, silence-heavy recordings, and huge file cancellation/resume.
- Transcript representation: segment timestamps, token/segment confidence, language tags, speaker placeholder support, and editability.
- Structured summary schema: meeting minutes, action items, owners, dates, decisions, open questions, citations back to transcript spans.
- Local data lifecycle: encryption at rest, deletion semantics, retention policy, temp-file cleanup, export/import, migration, and backup behavior.
- Privacy verification: network entitlement denial, runtime network call tests, no hidden crash upload, no logs containing transcript/audio paths.
- Benchmark protocol: fixed audio corpus, warm/cold runs, peak memory method, RTF definition, thermal state, and pass/fail thresholds.

TECHNICAL_GAPS:
- The plan does not decide whether Whisper and LLM run concurrently or sequentially. Sequential model loading may be required to meet memory; concurrent loading may be required to meet latency.
- Long transcript analysis needs chunking, map-reduce summarization, or hierarchical prompting. A single prompt template will break on context limits for hour-long meetings.
- SwiftData/CoreData persistence is named, but the data model is not: audio asset, transcript segment, inference run, prompt template, summary artifact, benchmark metrics, and model manifest likely need separate entities.
- MLX-to-Swift integration is under-specified. If Python remains in the product, the architecture should explicitly treat it as an inference worker with a stable IPC contract.
- No model abstraction boundary is defined. The app should not couple UI/storage directly to MLX-LM, llama.cpp, or Whisper-specific outputs.
- No observability plan exists for local-only performance: per-stage timing, memory samples, model load time, token throughput, audio duration, and failure reason should be stored locally.
- No deterministic validation exists for LLM structured output. The plan needs schema validation and repair/failure paths.
- The current `plan.md` is still mostly a template plus draft material; it needs concrete acceptance criteria before implementation.

ALTERNATIVE_DIRECTIONS:
- Keep Mac-first, but make the first milestone a benchmark-only Python CLI. Tradeoff: less product UI early, but it directly validates the riskiest assumptions.
- Use smaller models first: Whisper-small/medium plus 3B/4B LLM. Tradeoff: lower quality, but much safer memory and iteration speed.
- Use llama.cpp for the LLM first and MLX only for Whisper, or vice versa. Tradeoff: fewer moving parts at the cost of less ideal Apple Silicon optimization.
- Prioritize a privacy spine first: sandbox, no-network entitlements, local encrypted store, temp-file cleanup, and audit tests. Tradeoff: slower AI demo, stronger product promise.
- Prioritize transcript quality first: timestamps, hotwords, and segment metadata before LLM summaries. Tradeoff: less flashy, but better downstream summary reliability.
- Build CLI plus persisted JSON/SQLite first, then SwiftData later. Tradeoff: less native from day one, but easier to benchmark and regression-test.

QUESTIONS_FOR_USER:
- What is the minimum supported Mac hardware: base M1 8 GB, M2 16 GB, M3 Pro, or something else?
- Is `<6 GB` a hard product requirement or an aspirational benchmark?
- Is Whisper-Large-v3 mandatory for MVP, or can the first slice use medium/small if quality is acceptable?
- Must the first usable app be fully sandboxed and distributable, or is a developer-local prototype acceptable?
- Should Python be allowed in the shipped macOS product, or only in the feasibility prototype?
- Which output matters most for MVP: raw transcript accuracy, action-item extraction, meeting minutes, or privacy proof?
- Are hotwords and diarization truly deferred, or are they required for the first credible demo?
- Should model files be bundled, user-imported, or downloaded by a separate offline preparation step?

CANDIDATE_CRITERIA:
- Import local `.m4a`, `.wav`, and `.mp3` files without network access.
- Produce timestamped transcript segments for a 10, 30, and 60 minute benchmark file.
- Produce schema-valid JSON summary with sections for decisions, action items, owners, due dates, and open questions.
- Persist audio metadata, transcript, summary, model manifest, prompt version, and benchmark metrics locally.
- Pass a no-network test where all outbound network APIs are blocked or fail the run.
- Peak resident memory stays under the chosen target on the minimum supported Mac.
- End-to-end RTF is measured and reported separately for decode, STT, LLM analysis, and persistence.
- Long files support progress, cancellation, failure recovery, and temp-file cleanup.
- Summary claims can cite transcript segment IDs or timestamps.
- First implementation has a stable CLI contract before SwiftUI wraps it.
