"""The provisioner locates and integrity-checks locally provisioned models.

Design contract (enforced by tests):

* **Zero network at runtime.** The provisioner only reads from the local
  filesystem. It never imports a HTTP client and never downloads a missing
  weight. A missing model raises :class:`ModelNotProvisionedError`; a corrupted
  one raises :class:`ChecksumMismatchError`. Both fail fast.
* **Integrity before inference.** ``require_model`` verifies existence, size,
  and SHA-256 before returning a path, so callers can assume the bytes on disk
  match the manifest.
* **Reproducible.** The manifest pins every weight by digest; the same manifest
  + directory always yields the same verification result.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from localmind.provisioning.errors import (
    ChecksumMismatchError,
    ManifestError,
    ModelNotProvisionedError,
)
from localmind.provisioning.manifest import ModelEntry, ModelManifest, validate_weight_path

# Buffer size for streaming SHA-256; large enough to be fast, small enough to
# keep memory bounded regardless of weight file size.
_HASH_CHUNK = 1 << 20  # 1 MiB

MANIFEST_FILENAME = "models.json"


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of verifying a single manifest entry against the filesystem."""

    model_id: str
    ok: bool
    reason: str = ""


class Provisioner:
    """Verify and locate locally provisioned model weights.

    Parameters
    ----------
    model_dir:
        Directory containing ``models.json`` and the weight files referenced by
        it (paths in the manifest are relative to this directory).
    """

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)

    @property
    def manifest_path(self) -> Path:
        return self.model_dir / MANIFEST_FILENAME

    def load_manifest(self) -> ModelManifest:
        """Load and validate the manifest. Raise ManifestError if absent/invalid."""
        if not self.manifest_path.is_file():
            raise ManifestError(
                f"manifest not found at {self.manifest_path}; provision models out-of-band first"
            )
        return ModelManifest.from_file(self.manifest_path)

    def weight_path(self, entry: ModelEntry) -> Path:
        # Defense in depth: the manifest validator already rejects absolute and
        # traversal paths, but re-check here so a hand-built entry cannot escape.
        validate_weight_path(entry.path, where=entry.model_id)
        return self.model_dir / entry.path

    def _resolve_confined(self, entry: ModelEntry) -> Path:
        """Resolve a weight path and assert it stays within the model directory."""
        candidate = self.weight_path(entry)
        base = self.model_dir.resolve()
        resolved = candidate.resolve()
        # is_relative_to exists on Python 3.9+; fall back to string-prefix check.
        try:
            inside = resolved.is_relative_to(base)
        except AttributeError:  # pragma: no cover
            inside = str(resolved).startswith(str(base) + os.sep) or resolved == base
        if not inside:
            raise ManifestError(
                f"weight path escapes model directory for {entry.model_id!r}: {entry.path!r}"
            )
        return resolved

    @staticmethod
    def sha256_of(path) -> str:
        """Stream a file through SHA-256 without loading it all into memory."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify_entry(self, entry: ModelEntry) -> VerificationResult:
        """Verify one entry: confinement, existence, size, then SHA-256."""
        try:
            path = self._resolve_confined(entry)
        except ManifestError as exc:
            return VerificationResult(entry.model_id, ok=False, reason=str(exc))
        if not path.is_file():
            return VerificationResult(entry.model_id, ok=False, reason="file missing")
        actual_size = path.stat().st_size
        if actual_size != entry.size_bytes:
            return VerificationResult(
                entry.model_id,
                ok=False,
                reason=f"size mismatch: manifest={entry.size_bytes} actual={actual_size}",
            )
        actual_digest = self.sha256_of(path)
        if actual_digest != entry.sha256:
            return VerificationResult(
                entry.model_id,
                ok=False,
                reason="sha256 mismatch",
            )
        return VerificationResult(entry.model_id, ok=True)

    def verify_all(self) -> List[VerificationResult]:
        """Verify every entry in the manifest."""
        manifest = self.load_manifest()
        return [self.verify_entry(e) for e in manifest.models]

    def require_model(self, model_id: str) -> Path:
        """Return the verified on-disk path for ``model_id``.

        Raises
        ------
        ModelNotProvisionedError
            If the manifest is absent, the model is unknown, or its weight file
            is missing. **Never** attempts a network download.
        ChecksumMismatchError
            If the weight file exists but its size or SHA-256 does not match the
            manifest (i.e. it is truncated or tampered).
        """
        manifest = self.load_manifest()
        try:
            entry = manifest.by_id(model_id)
        except KeyError:
            raise ModelNotProvisionedError(
                f"model not provisioned: {model_id!r} is not declared in the manifest"
            ) from None

        path = self._resolve_confined(entry)
        if not path.is_file():
            raise ModelNotProvisionedError(
                f"model not provisioned: weight file missing for {model_id!r} at {path}"
            )

        actual_size = path.stat().st_size
        if actual_size != entry.size_bytes:
            raise ChecksumMismatchError(
                f"checksum mismatch for {model_id!r}: size manifest={entry.size_bytes} "
                f"actual={actual_size}"
            )
        actual_digest = self.sha256_of(path)
        if actual_digest != entry.sha256:
            raise ChecksumMismatchError(
                f"checksum mismatch for {model_id!r}: sha256 manifest={entry.sha256} "
                f"actual={actual_digest}"
            )
        return path
