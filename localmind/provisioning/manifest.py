"""Versioned model manifest.

The manifest is a small JSON file (``models.json``) living at the root of the
local model directory. It records every model weight the pipeline may load,
together with its expected size and SHA-256 digest. The provisioner validates
each entry against the real file before any inference runs, so a tampered or
truncated weight is rejected deterministically and no network fetch is ever
attempted to "repair" it.

Schema is versioned via ``schema_version``. Bump it when the on-disk shape
changes; the provisioner refuses manifests it does not understand.

Validation is implemented with the standard library only (no ``jsonschema``
dependency) so that a normal runtime install validates manifests deterministically
— the correctness of the integrity check must not depend on a dev-only package
being present.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from localmind.provisioning.errors import ManifestError

MANIFEST_SCHEMA_VERSION = "1"

_VALID_KINDS = frozenset({"whisper", "llm"})
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class ModelEntry:
    """A single model weight file declared in the manifest."""

    model_id: str
    name: str
    kind: str  # "whisper" | "llm"
    path: str  # relative to the model directory, must stay within it
    quant_format: str
    size_bytes: int
    sha256: str
    license: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelManifest:
    """A validated collection of model entries, versioned on disk."""

    schema_version: str
    models: List[ModelEntry] = field(default_factory=list)

    def by_id(self, model_id: str) -> ModelEntry:
        for entry in self.models:
            if entry.model_id == model_id:
                return entry
        raise KeyError(model_id)

    def to_dict(self) -> Dict:
        return {
            "schema_version": self.schema_version,
            "models": [m.to_dict() for m in self.models],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelManifest":
        validate_manifest_dict(data)
        models = [ModelEntry(**m) for m in data["models"]]
        return cls(schema_version=data["schema_version"], models=models)

    @classmethod
    def from_json(cls, text: str) -> "ModelManifest":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path) -> "ModelManifest":
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise ManifestError(f"cannot read manifest at {p}: {exc}") from exc
        return cls.from_json(text)


def _is_hex64(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(c in _HEX_DIGITS for c in value.lower())


def validate_weight_path(rel_path, where: str = "model") -> None:
    """Reject weight paths that are absolute or escape the model directory.

    A manifest path must be relative and must not contain ``..`` components, so
    that a crafted manifest cannot point the provisioner at files outside the
    model directory (e.g. ``../../etc/passwd`` or an absolute path).
    """
    if not isinstance(rel_path, str) or not rel_path:
        raise ManifestError(f"{where} path must be a non-empty string")
    p = Path(rel_path)
    if p.is_absolute():
        raise ManifestError(f"{where} path must be relative, got absolute: {rel_path!r}")
    if any(part == ".." for part in p.parts):
        raise ManifestError(
            f"{where} path must not traverse parent directories: {rel_path!r}"
        )


def validate_manifest_dict(data: Dict) -> None:
    """Validate a manifest dict with stdlib only; raise ManifestError on failure."""
    if not isinstance(data, dict):
        raise ManifestError(f"manifest root must be an object, got {type(data).__name__}")

    version = data.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema_version {version!r}; expected {MANIFEST_SCHEMA_VERSION!r}"
        )

    models = data.get("models")
    if not isinstance(models, list):
        raise ManifestError(
            f"manifest 'models' must be an array, got {type(models).__name__}"
        )

    required_fields = (
        "model_id",
        "name",
        "kind",
        "path",
        "quant_format",
        "size_bytes",
        "sha256",
    )
    seen_ids = set()
    for i, m in enumerate(models):
        if not isinstance(m, dict):
            raise ManifestError(f"models[{i}] must be an object, got {type(m).__name__}")
        for field_name in required_fields:
            if field_name not in m:
                raise ManifestError(f"models[{i}] missing required field: {field_name}")

        model_id = m["model_id"]
        if not isinstance(model_id, str) or not model_id:
            raise ManifestError(f"models[{i}] model_id must be a non-empty string")
        if model_id in seen_ids:
            raise ManifestError(f"duplicate model_id: {model_id!r}")
        seen_ids.add(model_id)

        if not isinstance(m["name"], str):
            raise ManifestError(f"models[{i}] name must be a string")

        kind = m["kind"]
        if kind not in _VALID_KINDS:
            raise ManifestError(
                f"models[{i}] kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}"
            )

        validate_weight_path(m["path"], where=f"models[{i}]")

        if not isinstance(m["quant_format"], str):
            raise ManifestError(f"models[{i}] quant_format must be a string")

        size_bytes = m["size_bytes"]
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ManifestError(
                f"models[{i}] size_bytes must be a non-negative integer, got {size_bytes!r}"
            )

        if not _is_hex64(m["sha256"]):
            raise ManifestError(
                f"models[{i}] sha256 must be 64 lowercase/uppercase hex characters"
            )

        if "license" in m and not isinstance(m["license"], str):
            raise ManifestError(f"models[{i}] license must be a string if present")
