"""Versioned, machine-readable benchmark report schema (AC-6).

A benchmark report records the per-stage timing, overall RTF, and peak memory
(CPU RSS vs GPU allocation, each with an explicit measurement method) for a run
against a benchmark fixture, and compares the measured values to the plan's
aspirational targets (``<6 GB`` peak, ``RTF < 0.08`` on Mac).

Validation is stdlib-only so it works in any runtime install.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

REPORT_SCHEMA_VERSION = "1"

# Aspirational targets from the plan (measure-and-report, not pass/fail gates).
ASPIRATIONAL_PEAK_MEM_GB = 6.0
ASPIRATIONAL_RTF = 0.08

_VALID_STAGES = ("decode", "stt", "llm", "persist")
_VALID_MEM_METHODS = frozenset(
    {"resource_tracker", "psutil_rss", "mach_task_basic_info", "metal_allocated", "mlx_memory"}
)


@dataclass
class StageTiming:
    """Timing for one pipeline stage."""

    stage: str  # one of _VALID_STAGES
    duration_sec: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PeakMemory:
    """Peak memory measurement, split by domain with an explicit method.

    ``method`` states how the value was measured so a reader knows whether it is
    CPU RSS, Metal/GPU allocation, etc. — required by AC-6's negative test.
    """

    bytes: int
    domain: str  # "cpu" | "gpu"
    method: str  # how it was measured (must be in _VALID_MEM_METHODS)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """A complete benchmark report for one run against one fixture."""

    schema_version: str
    run_id: str
    case_id: str
    audio_duration_sec: float
    stages: List[StageTiming] = field(default_factory=list)
    total_duration_sec: float = 0.0
    rtf: float = 0.0
    peak_memory: List[PeakMemory] = field(default_factory=list)
    model_tiers: Dict[str, str] = field(default_factory=dict)
    hardware: Dict[str, str] = field(default_factory=dict)
    aspirational_targets: Dict[str, float] = field(
        default_factory=lambda: {
            "peak_mem_gb": ASPIRATIONAL_PEAK_MEM_GB,
            "rtf": ASPIRATIONAL_RTF,
        }
    )
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "audio_duration_sec": self.audio_duration_sec,
            "stages": [s.to_dict() for s in self.stages],
            "total_duration_sec": self.total_duration_sec,
            "rtf": self.rtf,
            "peak_memory": [m.to_dict() for m in self.peak_memory],
            "model_tiers": dict(self.model_tiers),
            "hardware": dict(self.hardware),
            "aspirational_targets": dict(self.aspirational_targets),
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def validate_report_dict(data: Dict) -> None:
    """Validate a benchmark report dict (stdlib only). Raise ValueError on failure."""
    if not isinstance(data, dict):
        raise ValueError(f"report root must be an object, got {type(data).__name__}")

    if data.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported report schema_version {data.get('schema_version')!r}; "
            f"expected {REPORT_SCHEMA_VERSION!r}"
        )

    for req in ("run_id", "case_id", "audio_duration_sec", "stages", "rtf", "peak_memory"):
        if req not in data:
            raise ValueError(f"report missing required field: {req}")

    stages = data["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("report 'stages' must be a non-empty array")
    seen_stages = set()
    for i, s in enumerate(stages):
        if not isinstance(s, dict):
            raise ValueError(f"stages[{i}] must be an object")
        if s.get("stage") not in _VALID_STAGES:
            raise ValueError(
                f"stages[{i}].stage must be one of {sorted(_VALID_STAGES)}, "
                f"got {s.get('stage')!r}"
            )
        if s["stage"] in seen_stages:
            raise ValueError(f"duplicate stage: {s['stage']!r}")
        seen_stages.add(s["stage"])
        if not isinstance(s.get("duration_sec"), (int, float)) or isinstance(
            s.get("duration_sec"), bool
        ):
            raise ValueError(f"stages[{i}].duration_sec must be a number")
    # AC-6 requires per-stage breakdown: every valid stage must be present.
    missing = set(_VALID_STAGES) - seen_stages
    if missing:
        raise ValueError(f"report missing stage(s): {sorted(missing)}")

    if not isinstance(data["rtf"], (int, float)) or isinstance(data["rtf"], bool):
        raise ValueError("report 'rtf' must be a number")

    pm = data["peak_memory"]
    if not isinstance(pm, list) or not pm:
        raise ValueError("report 'peak_memory' must be a non-empty array")
    for i, m in enumerate(pm):
        if not isinstance(m, dict):
            raise ValueError(f"peak_memory[{i}] must be an object")
        if not isinstance(m.get("bytes"), int) or isinstance(m.get("bytes"), bool) or m["bytes"] < 0:
            raise ValueError(f"peak_memory[{i}].bytes must be a non-negative integer")
        if m.get("domain") not in ("cpu", "gpu"):
            raise ValueError(f"peak_memory[{i}].domain must be 'cpu' or 'gpu'")
        if m.get("method") not in _VALID_MEM_METHODS:
            raise ValueError(
                f"peak_memory[{i}].method must be one of {sorted(_VALID_MEM_METHODS)}, "
                f"got {m.get('method')!r}"
            )
    # AC-6 requires distinguishing CPU RSS vs GPU allocation.
    domains = {m["domain"] for m in pm}
    if "cpu" not in domains or "gpu" not in domains:
        raise ValueError("report 'peak_memory' must include both 'cpu' and 'gpu' domains")
