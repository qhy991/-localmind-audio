"""Versioned structured-summary schema and validators.

Schema designed via the task8 analyze route (Codex). A summary is a versioned
JSON object with ``decisions``, ``action_items`` (each with ``owner`` and
``due_date`` keys — nullable, but required to be present), ``open_questions``,
and ``citations`` back to transcript segment IDs (or timestamp ranges). Every
citation — including ``ts:<start>-<end>`` ranges — must be grounded in the real
transcript: a segment id must exist, and a timestamp range must overlap at least
one segment's time span. Validation is stdlib-only.

When bounded repair is exhausted, a ``summary_failed`` artifact is produced
instead — it carries the raw model output and the validation errors, never a
fabricated summary.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

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


def _ids_and_spans(segments: Iterable) -> Tuple[Set[str], List[Tuple[float, float]]]:
    """Extract segment ids and (start, end) time spans from segment objects."""
    ids: Set[str] = set()
    spans: List[Tuple[float, float]] = []
    for s in segments:
        sid = s["id"] if isinstance(s, dict) else s.id
        start = s["start"] if isinstance(s, dict) else s.start
        end = s["end"] if isinstance(s, dict) else s.end
        ids.add(str(sid))
        spans.append((float(start), float(end)))
    return ids, spans


def _validate_cited(citation: str, ids: Set[str], spans: List[Tuple[float, float]], where: str) -> None:
    """Validate one citation against the real transcript (ids + time spans)."""
    if citation.startswith("seg:"):
        ref = citation[4:]
        if ref not in ids:
            raise SummaryValidationError(f"{where} cites unknown segment id: {citation!r}")
    elif citation.startswith("ts:"):
        m = _TS_RE.match(citation)
        if not m:
            raise SummaryValidationError(f"{where} has malformed timestamp citation: {citation!r}")
        a, b = float(m.group(1)), float(m.group(2))
        if not a < b:
            raise SummaryValidationError(f"{where} timestamp citation must have start < end: {citation!r}")
        if not any(a < e and b > s for s, e in spans):
            raise SummaryValidationError(
                f"{where} timestamp citation does not overlap any transcript segment: {citation!r}"
            )
    else:
        if citation not in ids:
            raise SummaryValidationError(f"{where} cites unknown segment id: {citation!r}")


def _validate_item(item, where: str, *, need_owner: bool = False) -> None:
    if not isinstance(item, dict):
        raise SummaryValidationError(f"{where} must be an object")
    _require_nonempty_str(item.get("text"), where)
    _validate_citations(item.get("citations"), where)
    if need_owner:
        # owner and due_date keys are REQUIRED (values may be null); absence is invalid.
        if "owner" not in item:
            raise SummaryValidationError(f"{where}.owner is required (may be null)")
        if "due_date" not in item:
            raise SummaryValidationError(f"{where}.due_date is required (may be null)")
        owner = item["owner"]
        if owner is not None and (not isinstance(owner, str) or not owner.strip()):
            raise SummaryValidationError(f"{where}.owner must be null or a non-empty string")
        due = item["due_date"]
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


def _validate_citations_against(segments: Iterable) -> Iterable:
    """Return a helper closure validating a citation list against segments."""
    ids, spans = _ids_and_spans(segments)

    def _check(citations, where: str) -> None:
        for c in citations:
            _validate_cited(c, ids, spans, where)

    return _check


def validate_summary_against_transcript(data: Dict, segments: Iterable) -> None:
    """Validate that every citation is grounded in the real transcript.

    Segment-id citations must reference an existing segment; ``ts:<a>-<b>``
    citations must have ``a < b`` and overlap at least one segment's time span.
    Assumes :func:`validate_summary_dict` has already passed.
    """
    check = _validate_citations_against(segments)
    for section in ("decisions", "action_items", "open_questions"):
        for i, item in enumerate(data.get(section, [])):
            where = f"{section}[{i}]"
            check(item.get("citations", []), where)


def validate_summary_sections(data: Dict, segments: Iterable) -> None:
    """Validate the three summary sections + their citations against segments.

    Used for partial (per-chunk or reduce) outputs during map-reduce, which do
    not yet carry the top-level schema_version/case_id/provenance wrapper.
    """
    if not isinstance(data, dict):
        raise SummaryValidationError("partial summary must be an object")
    check = _validate_citations_against(segments)
    for key in ("decisions", "action_items", "open_questions"):
        if not isinstance(data.get(key), list):
            raise SummaryValidationError(f"{key} must be an array")
    for i, item in enumerate(data["decisions"]):
        where = f"decisions[{i}]"
        _validate_item(item, where)
        check(item.get("citations", []), where)
    for i, item in enumerate(data["action_items"]):
        where = f"action_items[{i}]"
        _validate_item(item, where, need_owner=True)
        check(item.get("citations", []), where)
    for i, item in enumerate(data["open_questions"]):
        where = f"open_questions[{i}]"
        _validate_item(item, where)
        check(item.get("citations", []), where)


def build_summary_failed(
    raw_output: str,
    errors: Iterable[str],
    *,
    case_id: Optional[str] = None,
    model_id: str = "",
    prompt_template_hash: str = "",
    repair_attempted: bool = False,
    repair_attempts_used: int = 0,
) -> Dict:
    """Construct a ``summary_failed`` artifact (raw output + errors, no fabrication).

    A failed artifact is never ``repaired`` (repair did not produce a valid
    summary); ``repair_attempted`` records whether a repair was tried.
    """
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
            "repaired": False,
            "repair_attempted": repair_attempted,
            "repair_attempts_used": repair_attempts_used,
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
