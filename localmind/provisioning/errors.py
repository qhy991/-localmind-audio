"""Exception hierarchy for offline model provisioning.

All provisioning errors derive from :class:`ProvisioningError`. The provisioner
never attempts a network download to recover; a missing or corrupted model is
always surfaced as one of these errors so callers fail fast and deterministically.
"""

from __future__ import annotations


class ProvisioningError(Exception):
    """Base class for all provisioning failures."""


class ManifestError(ProvisioningError):
    """The model manifest is missing, malformed, or fails schema validation."""


class ModelNotProvisionedError(ProvisioningError):
    """A requested model is not present in the local model directory.

    Raised instead of attempting a network download — the runtime guarantee is
    that provisioning happens out-of-band, never at inference time.
    """


class ChecksumMismatchError(ProvisioningError):
    """A model weight file exists but its SHA-256 or size does not match the manifest."""
