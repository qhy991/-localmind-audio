"""Versioned structured-summary schema and validators.

Schema designed via the task8 analyze route (Codex). A summary is a versioned
JSON object with ``decisions``, ``action_items`` (each with nullable ``owner``
and ``due_date``), ``open_questions``, and ``citations`` back to transcript
segment IDs (or timestamp ranges). Validation is stdlib-only.

When bounded repair is exhausted, a ``summary_failed`` artifact is produced
instead — it carries the raw model output and the validation errors, never a
fabricated summary.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Set

SUMMARY_SCHEMA_VERSION = "soundmind.summary.v1"

_DUE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TS_RE = re.compile(r"^ts:([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)$")


class SummaryValidationError(ValueError):
    """Raised when a summary (or summary_failed) fails validation."""


def _require_nonempty_str(value, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SummaryValidationError(f"{where} must be a non-empty string")


def _validate_citations(citations, where: str) -> None:
    if not isinstance(citations, list) or len(citations) == 0:
        raise SummaryValidationError(f"{where}.citations must be a non-empty list")
    for c in citations:
        if not isinstance(c, str) or not c.strip():
            raise SummaryValidationError(f"{where}.citations entries must be non-empty strings")


def _validate_cited(citation: str, segment_ids: Set[str], where: str) -> None:
    """Validate one citation against the actual transcript segment ids."""
    if citation.startswith("seg:"):
        ref = citation[4:]
        if ref not in segment_ids:
            raise SummaryValidationError(
                f"{where} cites unknown segment id: {citation!r}"
            )
    elif citation.startswith("ts:"):
        if not _TS_RE.match(citation):
            raise SummaryValidationError(f"{where} has malformed timestamp citation: {citation!r}")
        # Timestamp citations are structurally valid; range-bounds checking
        # against audio duration is the caller's concern.
    else:
        # Direct segment-id reference.
        if citation not in segment_ids:
            raise SummaryValidationError(
                f"{where} cites unknown segment id: {citation!r}"
            )


def _validate_item(item, where: str, *, need_owner: bool = False) -> None:
    if not isinstance(item, dict):
        raise SummaryValidationError(f"{where} must be an object")
    _require_nonempty_str(item.get("text"), where)
    _validate_citations(item.get("citations"), where)
    if need_owner:
        owner = item.get("owner")
        if owner is not None and (not isinstance(owner, str) or not owner.strip()):
            raise SummaryValidationError(f"{where}.owner must be null or a non-empty string")
        due = item.get("due_date")
        if due is not None:
            if not isinstance(due, str) or not _DUE_DATE_RE.match(due):
                raise SummaryValidationError(
                    f"{where}.due_date must be null or YYYY-MM-DD, got {due!r}"
                )


def validate_summary_dict(data: Dict) -> None:
    """Validate the summary shape (stdlib). Raise SummaryValidationError on failure."""
    if not isinstance(data, dict):
        raise SummaryValidationError("summary must be an object")
    if data.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise SummaryValidationError(
            f"schema_version must be {SUMMARY_SCHEMA_VERSION!r}, got {data.get('schema_version')!r}"
        )
    _require_nonempty_str(data.get("case_id"), "case_id")
    prov = data.get("provenance")
    if not isinstance(prov, dict):
        raise SummaryValidationError("provenance must be an object")
    _require_nonempty_str(prov.get("model_id"), "provenance.model_id")
    _require_nonempty_str(prov.get("prompt_template_hash"), "provenance.prompt_template_hash")

    for key in ("decisions", "action_items", "open_questions"):
        if not isinstance(data.get(key), list):
            raise SummaryValidationError(f"{key} must be an array")

    for i, item in enumerate(data["decisions"]):
        _validate_item(item, f"decisions[{i}]")
    for i, item in enumerate(data["action_items"]):
        _validate_item(item, f"action_items[{i}]", need_owner=True)
    for i, item in enumerate(data["open_questions"]):
        _validate_item(item, f"open_questions[{i}]")


def validate_summary_against_transcript(
    data: Dict, segment_ids: Iterable[str]
) -> None:
    """Validate that every citation references a real transcript segment id.

    Assumes :func:`validate_summary_dict` has already passed.
    """
    ids = set(segment_ids)
    for section in ("decisions", "action_items", "open_questions"):
        for i, item in enumerate(data.get(section, [])):
            where = f"{section}[{i}]"
            for c in item.get("citations", []):
                _validate_cited(c, ids, where)


def validate_summary_sections(data: Dict, segment_ids: Iterable[str]) -> None:
    """Validate the three summary sections + their citations against segment ids.

    Used for partial (per-chunk) outputs during map-reduce, which do not yet
    carry the top-level schema_version/case_id/provenance wrapper.
    """
    if not isinstance(data, dict):
        raise SummaryValidationError("partial summary must be an object")
    ids = set(segment_ids)
    for key in ("decisions", "action_items", "open_questions"):
        if not isinstance(data.get(key), list):
            raise SummaryValidationError(f"{key} must be an array")
    for i, item in enumerate(data["decisions"]):
        where = f"decisions[{i}]"
        _validate_item(item, where)
        for c in item["citations"]:
            _validate_cited(c, ids, where)
    for i, item in enumerate(data["action_items"]):
        where = f"action_items[{i}]"
        _validate_item(item, where, need_owner=True)
        for c in item["citations"]:
            _validate_cited(c, ids, where)
    for i, item in enumerate(data["open_questions"]):
        where = f"open_questions[{i}]"
        _validate_item(item, where)
        for c in item["citations"]:
            _validate_cited(c, ids, where)


def build_summary_failed(
    raw_output: str,
    errors: Iterable[str],
    *,
    case_id: Optional[str] = None,
    model_id: str = "",
    prompt_template_hash: str = "",
) -> Dict:
    """Construct a ``summary_failed`` artifact (raw output + errors, no fabrication)."""
    error_list = [str(e) for e in errors]
    if not error_list:
        error_list = ["unknown validation error"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "failed",
        "case_id": case_id,
        "raw_output": raw_output if isinstance(raw_output, str) else str(raw_output),
        "errors": error_list,
        "provenance": {
            "model_id": model_id,
            "prompt_template_hash": prompt_template_hash,
        },
    }


def validate_summary_failed_dict(data: Dict) -> None:
    """Validate a ``summary_failed`` artifact shape."""
    if not isinstance(data, dict):
        raise SummaryValidationError("summary_failed must be an object")
    if data.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise SummaryValidationError(
            f"schema_version must be {SUMMARY_SCHEMA_VERSION!r}, got {data.get('schema_version')!r}"
        )
    if data.get("status") != "failed":
        raise SummaryValidationError("summary_failed.status must be 'failed'")
    if not isinstance(data.get("raw_output"), str):
        raise SummaryValidationError("summary_failed.raw_output must be a string")
    errors = data.get("errors")
    if not isinstance(errors, list) or len(errors) == 0:
        raise SummaryValidationError("summary_failed.errors must be a non-empty array")
