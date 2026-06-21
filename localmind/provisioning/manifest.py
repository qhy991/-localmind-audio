"""Versioned model manifest.

The manifest is a small JSON file (``models.json``) living at the root of the
local model directory. It records every model weight the pipeline may load,
together with its expected size and SHA-256 digest. The provisioner validates
each entry against the real file before any inference runs, so a tampered or
truncated weight is rejected deterministically and no network fetch is ever
attempted to "repair" it.

Schema is versioned via ``schema_version``. Bump it when the on-disk shape
changes; the provisioner refuses manifests it does not understand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from localmind.provisioning.errors import ManifestError

try:  # jsonschema is a dev/test dependency; degrade gracefully if absent.
    import jsonschema
except Exception:  # pragma: no cover - exercised only in odd environments
    jsonschema = None

MANIFEST_SCHEMA_VERSION = "1"

# JSON Schema describing the on-disk manifest shape. Used both to validate
# user-authored manifests and as the contract for the negative tests.
_MANIFEST_SCHEMA: Dict = {
    "type": "object",
    "required": ["schema_version", "models"],
    "properties": {
        "schema_version": {"type": "string"},
        "models": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "model_id",
                    "name",
                    "kind",
                    "path",
                    "quant_format",
                    "size_bytes",
                    "sha256",
                ],
                "properties": {
                    "model_id": {"type": "string", "minLength": 1},
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["whisper", "llm"]},
                    "path": {"type": "string", "minLength": 1},
                    "quant_format": {"type": "string"},
                    "size_bytes": {"type": "integer", "minimum": 0},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "license": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": True,
}


@dataclass(frozen=True)
class ModelEntry:
    """A single model weight file declared in the manifest."""

    model_id: str
    name: str
    kind: str  # "whisper" | "llm"
    path: str  # relative to the model directory
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


def validate_manifest_dict(data: Dict) -> None:
    """Validate a manifest dict against the schema; raise ManifestError on failure."""
    if not isinstance(data, dict):
        raise ManifestError(f"manifest root must be an object, got {type(data).__name__}")
    version = data.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema_version {version!r}; expected {MANIFEST_SCHEMA_VERSION!r}"
        )
    if jsonschema is not None:
        try:
            jsonschema.validate(instance=data, schema=_MANIFEST_SCHEMA)
        except jsonschema.ValidationError as exc:
            raise ManifestError(f"manifest fails schema validation: {exc.message}") from exc
    else:  # pragma: no cover - minimal structural checks without jsonschema
        if not isinstance(data.get("models"), list):
            raise ManifestError("manifest 'models' must be an array")
        for m in data["models"]:
            sha = m.get("sha256", "")
            if not (isinstance(sha, str) and len(sha) == 64):
                raise ManifestError(f"invalid sha256 for {m.get('model_id')}")
