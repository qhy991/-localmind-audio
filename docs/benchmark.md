# Benchmark Protocol

This document defines the benchmark harness for acceptance criteria **AC-2**,
**AC-6**, and the supporting expected-output shapes. Implementation lives in
`localmind/bench/`.

## 1. Benchmark cases

Three canonical cases, declared in `localmind/bench/fixtures.py`
(`BENCHMARK_CASES`):

| case_id    | duration | audio_rel_path        | purpose                                  |
|------------|----------|-----------------------|------------------------------------------|
| `bm-10min` | 10 min   | `audio/bm-10min.m4a`  | short-meeting RTF/memory baseline        |
| `bm-30min` | 30 min   | `audio/bm-30min.m4a`  | medium-meeting baseline                  |
| `bm-60min` | 60 min   | `audio/bm-60min.m4a`  | long-meeting; exercises bounded chunking |

**Real audio for these cases is provisioned out-of-band** — exactly like model
weights (see `docs/provisioning.md`). A 60-minute 16 kHz mono file is ~115 MB
of PCM and is never committed to git. The harness resolves `audio_rel_path`
under the benchmark fixtures directory and **skips cleanly** when the file is
absent, so unit tests remain hermetic.

For unit tests that need audio but not the full benchmark corpus, use
`generate_synthetic_wav(path, duration_sec, ...)` — a deterministic, seeded
sine-plus-noise WAV generator (no network, no committed binaries).

## 2. Report schema (`localmind/bench/report.py`)

Every benchmark run produces a machine-readable report validated by
`validate_report_dict` (stdlib only). Versioned via `schema_version` (currently
`"1"`).

Required fields:

* `schema_version`, `run_id`, `case_id`, `audio_duration_sec`
* `stages` — **must include all four stages**: `decode`, `stt`, `llm`,
  `persist`, each with a numeric `duration_sec`. A report missing any stage is
  rejected (AC-6 negative test: no single end-to-end time without breakdown).
* `rtf` — overall real-time factor = `total_duration_sec / audio_duration_sec`.
* `peak_memory` — **must include both `cpu` and `gpu` domains**, each with an
  explicit `method` (one of `resource_tracker`, `psutil_rss`,
  `mach_task_basic_info`, `metal_allocated`, `mlx_memory`). A memory figure
  without a method, or missing a domain, is rejected (AC-6 negative test).
* `aspirational_targets` — the plan's `peak_mem_gb: 6.0` and `rtf: 0.08`,
  recorded for comparison. These are **measure-and-report** targets, not
  pass/fail gates (confirmed user decision: budgets are aspirational).

## 3. Expected-output shapes

Two JSON shape artifacts under `localmind/bench/schemas/`:

* `transcript_expected.json` — expected transcript shape for a benchmark case:
  ordered, timestamped, monotonic non-decreasing segments bounded by audio
  duration, plus a `tolerance` block (`segment_count_relative`, `boundary_sec`,
  `wer_threshold`) describing how a candidate transcript is judged acceptable.
* `summary_example.json` — example structured-summary artifact (the target of
  AC-3): versioned JSON with `decisions`, `action_items` (each with nullable
  `owner`/`due_date`), `open_questions`, and `citations` to segment IDs.

Real expected transcripts are generated **once** from the baseline whisper tier
and stored alongside the provisioned audio (out-of-band); the committed files
document the shape, not the bit-exact content.

## 4. Running a benchmark

1. Provision models per `docs/provisioning.md` (Whisper small/medium, LLM).
2. Provision the 10/30/60-min audio fixtures out-of-band.
3. Run the pipeline against each case (the CLI `benchmark` subcommand, added in
   a later milestone, drives this).
4. The harness emits a validated `BenchmarkReport` JSON per case, comparing
   measured RTF/peak-memory against the aspirational targets.
