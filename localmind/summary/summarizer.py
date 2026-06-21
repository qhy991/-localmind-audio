"""Structured-summary generation: map-reduce over transcript chunks.

* :class:`SummaryLLM` — abstract local-LLM interface (a string prompt in, a
  string out). The real adapter (mlx-lm / llama.cpp) is added when the LLM
  backend is provisioned; :class:`MockSummaryLLM` makes the pipeline testable
  with no backend.
* :class:`Summarizer` — chunks a transcript by a character budget, asks the LLM
  for a partial summary per chunk, validates each partial's sections and
  citations against the chunk's real segment ids, merges them, and validates the
  final summary against all segment ids. Invalid output gets exactly one bounded
  repair attempt; if repair is exhausted, a ``summary_failed`` artifact is
  returned with the raw output and errors — never a fabricated summary.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from localmind.stt.segment import TranscriptSegment
from localmind.summary.schema import (
    SUMMARY_SCHEMA_VERSION,
    SummaryValidationError,
    build_summary_failed,
    validate_summary_against_transcript,
    validate_summary_dict,
    validate_summary_sections,
)

_SEG_ID_RE = re.compile(r"seg-\d+")


class SummaryLLM(ABC):
    """Abstract local-LLM interface: prompt in, raw string out."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockSummaryLLM(SummaryLLM):
    """Deterministic fake LLM for tests.

    Modes:
      * ``"valid"`` — always returns a valid partial summary citing the segment
        ids found in the prompt.
      * ``"invalid_then_valid"`` — the first call returns an invalid partial
        (a decision with no citation); subsequent calls return a valid one
        (exercises the repair path).
      * ``"always_invalid"`` — always returns an invalid partial (exercises
        repair exhaustion -> summary_failed).
    """

    def __init__(self, mode: str = "valid"):
        if mode not in ("valid", "invalid_then_valid", "always_invalid"):
            raise ValueError(f"unknown MockSummaryLLM mode: {mode!r}")
        self.mode = mode
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        ids = _SEG_ID_RE.findall(prompt)
        if not ids:
            ids = ["seg-0000"]
        first, last = ids[0], ids[-1]

        if self.mode == "always_invalid":
            return json.dumps({
                "decisions": [{"text": "a decision", "citations": []}],
                "action_items": [],
                "open_questions": [],
            })
        if self.mode == "invalid_then_valid" and self.call_count == 1:
            return json.dumps({
                "decisions": [{"text": "a decision", "citations": []}],
                "action_items": [],
                "open_questions": [],
            })

        return json.dumps({
            "decisions": [{"text": "a decision was made", "citations": [f"seg:{first}"]}],
            "action_items": [{
                "text": "follow up on the decision",
                "owner": "Maya",
                "due_date": None,
                "citations": [f"seg:{last}"],
            }],
            "open_questions": [{
                "text": "an open question",
                "citations": [f"seg:{first}"],
            }],
        })


def _chunk_segments_by_chars(
    segments: List[TranscriptSegment], max_chars: int
) -> List[List[TranscriptSegment]]:
    """Group consecutive segments so each chunk's text fits the char budget."""
    chunks: List[List[TranscriptSegment]] = []
    current: List[TranscriptSegment] = []
    used = 0
    for seg in segments:
        seg_len = len(seg.text)
        if current and used + seg_len > max_chars:
            chunks.append(current)
            current = []
            used = 0
        current.append(seg)
        used += seg_len
    if current:
        chunks.append(current)
    return chunks or [list(segments)]


def _build_prompt(chunk: List[TranscriptSegment]) -> str:
    lines = ["Transcript segments (id | start-end | text):"]
    for s in chunk:
        lines.append(f"{s.id} | {s.start:.3f}-{s.end:.3f} | {s.text}")
    lines.append(
        "Produce a JSON object with keys decisions, action_items, open_questions. "
        "Each item has 'text' (non-empty) and 'citations' (non-empty list of "
        "segment ids, formatted 'seg:<id>' using only the ids above). "
        "action_items also have 'owner' (string or null) and 'due_date' "
        "(YYYY-MM-DD or null). Output ONLY the JSON object."
    )
    return "\n".join(lines)


def _parse_and_validate_sections(
    raw: str, chunk_ids: List[str]
) -> Tuple[Optional[Dict], List[str]]:
    """Parse raw LLM output and validate its sections + citations. Returns
    (parsed_sections, errors)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"output is not valid JSON: {exc}"]
    try:
        validate_summary_sections(data, chunk_ids)
    except SummaryValidationError as exc:
        return None, [str(exc)]
    return data, []


class Summarizer:
    """Map-reduce summarizer with bounded repair and a summary_failed fallback."""

    def __init__(
        self,
        llm: SummaryLLM,
        *,
        model_id: str,
        prompt_template: str = "",
        max_chars_per_chunk: int = 4000,
        max_repair_attempts: int = 1,
    ):
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be >= 0")
        self.llm = llm
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.prompt_template_hash = "sha256:" + hashlib.sha256(
            (prompt_template or _build_prompt.__doc__ or "").encode("utf-8")
        ).hexdigest()[:12]
        self.max_chars_per_chunk = max_chars_per_chunk
        self.max_repair_attempts = max_repair_attempts

    def summarize(self, segments: List[TranscriptSegment], case_id: str) -> Dict:
        all_ids = [s.id for s in segments]
        chunks = _chunk_segments_by_chars(segments, self.max_chars_per_chunk)

        merged = {"decisions": [], "action_items": [], "open_questions": []}
        for chunk in chunks:
            partial, raw, errors = self._generate_chunk(chunk)
            if partial is None:
                return build_summary_failed(
                    raw, errors, case_id=case_id,
                    model_id=self.model_id,
                    prompt_template_hash=self.prompt_template_hash,
                )
            for key in merged:
                merged[key].extend(partial.get(key, []))

        final = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "case_id": case_id,
            "provenance": {
                "model_id": self.model_id,
                "prompt_template_hash": self.prompt_template_hash,
            },
            **merged,
        }
        try:
            validate_summary_dict(final)
            validate_summary_against_transcript(final, all_ids)
        except SummaryValidationError as exc:
            return build_summary_failed(
                json.dumps(final), [str(exc)], case_id=case_id,
                model_id=self.model_id,
                prompt_template_hash=self.prompt_template_hash,
            )
        return final

    def _generate_chunk(
        self, chunk: List[TranscriptSegment]
    ) -> Tuple[Optional[Dict], str, List[str]]:
        """Generate + validate one chunk's partial summary with bounded repair.

        Returns (partial_or_None, last_raw_output, errors). errors is non-empty
        only when partial is None.
        """
        chunk_ids = [s.id for s in chunk]
        prompt = _build_prompt(chunk)
        raw = self.llm.generate(prompt)
        partial, errors = _parse_and_validate_sections(raw, chunk_ids)
        if partial is not None:
            return partial, raw, []

        # Bounded repair: re-prompt with the validation error, up to the limit.
        for _ in range(self.max_repair_attempts):
            repair_prompt = (
                prompt + "\n\nYour previous output was invalid: "
                + "; ".join(errors)
                + "\nProduce a valid JSON object now."
            )
            raw = self.llm.generate(repair_prompt)
            partial, errors = _parse_and_validate_sections(raw, chunk_ids)
            if partial is not None:
                return partial, raw, []

        return None, raw, errors or ["repair exhausted"]
