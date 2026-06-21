"""Benchmark harness: fixture generation and machine-readable report schema.

This package supports benchmark fixture generation and report validation by providing:

* deterministic synthetic-audio fixtures for small test cases,
* descriptors for the 10/30/60-minute benchmark cases (real audio is provisioned
  out-of-band, like model weights — never committed),
* a versioned, machine-readable benchmark report schema with the per-stage
  timing, RTF, and peak-memory (CPU RSS vs GPU, with measurement method) fields
  the plan requires.
"""

from localmind.bench.fixtures import (
    BENCHMARK_CASES,
    BenchmarkCase,
    FixtureNotProvisionedError,
    fixture_path,
    generate_synthetic_wav,
    is_fixture_provisioned,
    require_fixture,
)
from localmind.bench.report import (
    REPORT_SCHEMA_VERSION,
    BenchmarkReport,
    StageTiming,
    validate_report_dict,
)

__all__ = [
    "BENCHMARK_CASES",
    "BenchmarkCase",
    "FixtureNotProvisionedError",
    "fixture_path",
    "generate_synthetic_wav",
    "is_fixture_provisioned",
    "require_fixture",
    "REPORT_SCHEMA_VERSION",
    "BenchmarkReport",
    "StageTiming",
    "validate_report_dict",
]
