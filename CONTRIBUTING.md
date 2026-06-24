# Contributing to LocalMind Audio

Thanks for your interest in improving LocalMind Audio. This is a small project
with a strict contract — the rules below keep the on-device, zero-network
guarantee intact.

## The contract (do not break these)

1. **Zero network at runtime.** The pipeline must succeed with the socket layer
   blocked. Never add a runtime download, telemetry call, or remote model fetch.
   Model acquisition happens out-of-band via `scripts/provision_models.py`.
2. **Integrity-pinned models.** Every weight flows through the `Provisioner` and
   is re-verified (SHA-256 + size) inside the adapter, immediately before use.
   Do not add a code path that loads a model by raw path or repo id.
3. **Structured, grounded output.** Summaries conform to the versioned schema
   (`soundmind.summary.v1`); every decision/action cites a transcript segment.
   Invalid LLM output goes through bounded repair, then `summary_failed` —
   never fabrication.
4. **Single normalized store.** All artifacts persist in one atomic transaction.
   No orphaned rows on partial failure.
5. **Pure CLI contract.** stdout is versioned JSON; stderr is JSONL progress
   (or empty with `--no-progress`). No stray prints, tracebacks, or library
   noise on stderr.

## Development workflow

```bash
pip install -e ".[dev,ml]"
pytest                      # must stay green
pytest tests/test_no_network.py   # the zero-network harness
```

- **Before sending a PR**, run the full suite and confirm `pytest` is green and
  `grep -rnE 'AC-[0-9]|milestone|Milestone' localmind/ tests/` returns nothing
  (source files stay free of internal plan terminology).
- **Keep changes focused.** One concern per PR. Benchmark/quality changes
  belong in their own PR, not bundled with feature work.
- **Add a test** for any new behavior or bug fix. The fake-backend tests
  (`tests/test_stt.py`, `tests/test_summary.py`) let you exercise adapter logic
  without a GPU or provisioned weights.
- **Real-backend changes** (anything touching the MLX path) should be runnable
  as a guarded smoke that skips cleanly on hosts without Metal/weights.

## Project layout

See the README "Architecture" section. The adapters (`stt/`, `summary/`) are
the boundary between provisioning and backend — read `localmind/mlx_runtime.py`
to understand the subprocess preflight pattern before touching backend import
order.

## Reporting issues

Please include: macOS version, chip, Python version, the exact CLI command, the
stdout JSON, and whether `python scripts/provision_models.py` succeeded. A
"Metal device unavailable" error means you are on a headless/VM host without a
GPU — that is expected, not a bug.

## License

By contributing, you agree your contributions are licensed under the project's
[MIT license](LICENSE).
