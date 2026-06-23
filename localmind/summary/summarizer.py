"""Structured-summary generation: map-reduce over transcript chunks.

Pipeline:

1. **Map** — chunk the transcript by a character budget; ask the LLM for a
   partial summary per chunk; validate each partial's sections and citations
   against that chunk's real segments.
2. **Reduce** — ask the LLM once over the validated partials for the final
   sections; validate against the full transcript.

Both map and reduce outputs get exactly one bounded repair attempt; if repair is
exhausted, a ``summary_failed`` artifact is returned with the raw output and
errors — never a fabricated summary. Successful repair is recorded in the
summary provenance (``repaired``, ``repair_attempts_used``,
``initial_validation_errors``).

:class:`SummaryLLM` is the abstract local-LLM interface; :class:`MockSummaryLLM`
makes the pipeline testable with no backend and distinguishes map vs reduce
calls so tests can assert the reduce step runs.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from localmind.stt.segment import TranscriptSegment
from localmind.provisioning.provisioner import Provisioner
from localmind.provisioning.errors import ModelNotProvisionedError
from localmind.stt.transcriber import ResolvedTier
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

    A reduce prompt is detected by a leading ``REDUCE`` marker. Modes:
      * ``"valid"`` — always returns valid sections citing the segment ids found
        in the prompt.
      * ``"invalid_then_valid"`` — the first call returns invalid sections (a
        decision with no citation); later calls return valid ones (exercises
        repair at the map or reduce step).
      * ``"always_invalid"`` — always returns invalid sections (exercises repair
        exhaustion -> summary_failed).
    """

    def __init__(self, mode: str = "valid"):
        if mode not in ("valid", "invalid_then_valid", "always_invalid"):
            raise ValueError(f"unknown MockSummaryLLM mode: {mode!r}")
        self.mode = mode
        self.call_count = 0
        self.map_calls = 0
        self.reduce_calls = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        if prompt.startswith("REDUCE"):
            self.reduce_calls += 1
        else:
            self.map_calls += 1

        ids = _SEG_ID_RE.findall(prompt)
        if not ids:
            ids = ["seg-0000"]
        first, last = ids[0], ids[-1]

        if self.mode == "always_invalid":
            return self._invalid()
        if self.mode == "invalid_then_valid" and self.call_count == 1:
            return self._invalid()

        return json.dumps({
            "decisions": [{"text": "a decision was made", "citations": [f"seg:{first}"]}],
            "action_items": [{
                "text": "follow up on the decision",
                "owner": "Maya",
                "due_date": None,
                "citations": [f"seg:{last}"],
            }],
            "open_questions": [{"text": "an open question", "citations": [f"seg:{first}"]}],
        })

    @staticmethod
    def _invalid() -> str:
        return json.dumps({
            "decisions": [{"text": "a decision", "citations": []}],
            "action_items": [],
            "open_questions": [],
        })


def _resolve_llm_tier(provisioner: Provisioner, model_id: str) -> ResolvedTier:
    """Resolve an LLM model tier, checking kind BEFORE weight hashing.

    Unlike the generic :func:`~localmind.stt.transcriber.resolve_tier`, this
    LLM-specific resolver rejects ``kind != "llm"`` immediately after loading
    the manifest entry — before ``require_model`` hashes the weight file. A
    wrong-kind entry with a missing or tampered file produces a kind error, not
    a missing-weight or checksum error.
    """
    manifest = provisioner.load_manifest()
    try:
        entry = manifest.by_id(model_id)
    except KeyError:
        raise ModelNotProvisionedError(
            f"model not provisioned: {model_id!r} is not declared in the manifest"
        ) from None
    if entry.kind != "llm":
        raise ModelNotProvisionedError(
            f"model {model_id!r} has kind={entry.kind!r}, not 'llm'; "
            f"cannot load as an LLM"
        )
    path = provisioner.require_model(model_id)  # verifies size + SHA-256 (AFTER kind)
    return ResolvedTier(
        tier=model_id, model_id=model_id, model_path=path,
        sha256=entry.sha256, quant_format=entry.quant_format, kind=entry.kind,
    )


class MlxLmSummaryLLM(SummaryLLM):
    """Real local LLM adapter over mlx-lm, bound to verified provisioning.

    Like :class:`~localmind.stt.WhisperTranscriber`, this adapter owns the
    model-resolution boundary: callers pass a :class:`Provisioner` and a model
    id, and the adapter resolves the tier internally on the first ``generate``
    call via :func:`_resolve_llm_tier` (which checks ``kind == "llm"`` before
    hashing, then runs ``require_model`` for size + SHA-256 verification). A
    pre-built path or repo id is never accepted.
    """

    def __init__(self, provisioner: Provisioner, model_id: str, max_tokens: int = 1024):
        self.provisioner = provisioner
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.last_provenance: Optional[ResolvedTier] = None
        self._model = None
        self._tokenizer = None

    def generate(self, prompt: str) -> str:
        if not isinstance(self.provisioner, Provisioner):
            raise TypeError(
                "MlxLmSummaryLLM requires a Provisioner; pre-built paths or "
                "repo ids are not accepted"
            )
        if self.last_provenance is None:
            self.last_provenance = _resolve_llm_tier(self.provisioner, self.model_id)

        # Import the backend, avoiding MLX atexit pollution on Metal-unavailable
        # hosts. Skip preflight when a fake/real backend is already injected.
        import sys as _sys
        _mod = _sys.modules.get("mlx_lm")
        if _mod is not None:
            mlx_lm = _mod  # injected fake or already-imported real
        elif "mlx_lm" in _sys.modules:
            raise RuntimeError(
                "mlx-lm is not installed; install the ML backend with "
                "`pip install -e .[ml]` (see docs/provisioning.md)"
            )
        else:
            from localmind.mlx_runtime import ensure_mlx_metal_available
            ensure_mlx_metal_available()
            try:
                import mlx_lm
            except ImportError as exc:
                raise RuntimeError(
                    "mlx-lm is not installed; install the ML backend with "
                    "`pip install -e .[ml]` (see docs/provisioning.md)"
                ) from exc
        if self._model is None:
            self._model, self._tokenizer = mlx_lm.load(str(self.last_provenance.model_path))
        return mlx_lm.generate(
            self._model, self._tokenizer,
            prompt=prompt, max_tokens=self.max_tokens,
        )


def _chunk_segments_by_chars(
    segments: List[TranscriptSegment], max_chars: int
) -> List[List[TranscriptSegment]]:
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


def _build_map_prompt(chunk: List[TranscriptSegment]) -> str:
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


def _build_reduce_prompt(partials: List[Dict], all_segments: List[TranscriptSegment]) -> str:
    lines = ["REDUCE"]
    lines.append("Combine these partial summaries into one final JSON object with keys "
                 "decisions, action_items, open_questions. Resolve duplicates and keep "
                 "citations as 'seg:<id>' referencing only these segment ids:")
    lines.append(", ".join(s.id for s in all_segments))
    lines.append("Partials:")
    lines.append(json.dumps(partials))
    lines.append("Output ONLY the final JSON object.")
    return "\n".join(lines)


def _parse_and_validate_sections(
    raw: str, segments
) -> Tuple[Optional[Dict], List[str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"output is not valid JSON: {exc}"]
    try:
        validate_summary_sections(data, segments)
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
            (prompt_template or "localmind-summary").encode("utf-8")
        ).hexdigest()[:12]
        self.max_chars_per_chunk = max_chars_per_chunk
        self.max_repair_attempts = max_repair_attempts

    def summarize(self, segments: List[TranscriptSegment], case_id: str) -> Dict:
        self._repaired = False
        self._repair_attempted = False
        self._repair_used = 0
        self._initial_errors: List[str] = []

        chunks = _chunk_segments_by_chars(segments, self.max_chars_per_chunk)

        # Map: per-chunk partials.
        partials: List[Dict] = []
        for chunk in chunks:
            partial, raw, errors = self._generate_with_repair(_build_map_prompt(chunk), chunk)
            if partial is None:
                return self._failed(raw, errors, case_id)
            partials.append({
                "decisions": partial.get("decisions", []),
                "action_items": partial.get("action_items", []),
                "open_questions": partial.get("open_questions", []),
            })

        # Reduce: one LLM call over the partials for the final sections.
        reduced, raw, errors = self._generate_with_repair(
            _build_reduce_prompt(partials, segments), segments
        )
        if reduced is None:
            return self._failed(raw, errors, case_id)

        final = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "case_id": case_id,
            "provenance": {
                "model_id": self.model_id,
                "prompt_template_hash": self.prompt_template_hash,
                "repaired": self._repaired,
                "repair_attempted": self._repair_attempted,
                "repair_attempts_used": self._repair_used,
                "initial_validation_errors": list(self._initial_errors),
            },
            "decisions": reduced.get("decisions", []),
            "action_items": reduced.get("action_items", []),
            "open_questions": reduced.get("open_questions", []),
        }
        try:
            validate_summary_dict(final)
            validate_summary_against_transcript(final, segments)
        except SummaryValidationError as exc:
            return self._failed(json.dumps(final), [str(exc)], case_id)
        return final

    def _generate_with_repair(
        self, prompt: str, segments
    ) -> Tuple[Optional[Dict], str, List[str]]:
        """Generate + validate one output (map or reduce) with bounded repair.

        Returns (parsed_or_None, last_raw_output, errors). On success errors is
        empty; on exhaustion parsed is None and errors is non-empty. Repair
        metadata is recorded on the summarizer. ``repaired`` is reserved for a
        repair that produced a valid output; a failed repair records
        ``repair_attempted`` only.
        """
        raw = self.llm.generate(prompt)
        parsed, errors = _parse_and_validate_sections(raw, segments)
        if parsed is not None:
            return parsed, raw, []

        initial_errors = list(errors)
        attempts_used = 0
        for _ in range(self.max_repair_attempts):
            repair_prompt = (
                prompt + "\n\nYour previous output was invalid: "
                + "; ".join(errors)
                + "\nProduce a valid JSON object now."
            )
            raw = self.llm.generate(repair_prompt)
            attempts_used += 1
            parsed, errors = _parse_and_validate_sections(raw, segments)
            if parsed is not None:
                self._record_repair(initial_errors, attempts_used, succeeded=True)
                return parsed, raw, []

        self._record_repair(initial_errors, attempts_used or self.max_repair_attempts, succeeded=False)
        return None, raw, errors or ["repair exhausted"]

    def _record_repair(self, initial_errors: List[str], attempts: int, *, succeeded: bool) -> None:
        self._repair_attempted = True
        self._repair_used += attempts
        if succeeded:
            self._repaired = True
        if not self._initial_errors:
            self._initial_errors = list(initial_errors)

    def _failed(self, raw: str, errors: List[str], case_id: str) -> Dict:
        return build_summary_failed(
            raw, errors, case_id=case_id,
            model_id=self.model_id,
            prompt_template_hash=self.prompt_template_hash,
            repair_attempted=self._repair_attempted,
            repair_attempts_used=self._repair_used,
        )
