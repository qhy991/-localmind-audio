"""Acceptance tests for AC-9: offline model provisioning with integrity verification.

Every test is hermetic: dummy weights and manifests are generated in ``tmp_path``
and no network is used. The "no download at runtime" guarantee is asserted by
monkeypatching ``socket.socket`` to explode if any networking is attempted.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from localmind.provisioning import (
    ChecksumMismatchError,
    ManifestError,
    ModelNotProvisionedError,
    Provisioner,
)
from localmind.provisioning.manifest import MANIFEST_SCHEMA_VERSION, ModelManifest


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _entry(model_id, path, content):
    return {
        "model_id": model_id,
        "name": model_id,
        "kind": "whisper",
        "path": path,
        "quant_format": "q4",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "license": "MIT",
    }


def _write_weight(model_dir: Path, rel: str, content: bytes) -> Path:
    p = model_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _write_manifest(model_dir: Path, entries):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "models.json").write_text(
        json.dumps({"schema_version": MANIFEST_SCHEMA_VERSION, "models": entries}, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Positive tests (expected to PASS)                                            #
# --------------------------------------------------------------------------- #

def test_verify_entry_ok_for_present_intact_weight(tmp_path):
    model_dir = tmp_path / "models"
    content = b"pretend-whisper-weights" * 100
    _write_weight(model_dir, "whisper-small.mlmodel", content)
    _write_manifest(model_dir, [_entry("whisper-small", "whisper-small.mlmodel", content)])

    prov = Provisioner(model_dir)
    manifest = prov.load_manifest()
    result = prov.verify_entry(manifest.models[0])

    assert result.ok is True
    assert result.reason == ""


def test_require_model_returns_verified_path(tmp_path):
    model_dir = tmp_path / "models"
    content = b"pretend-llm-weights" * 50
    _write_weight(model_dir, "qwen-7b-q4.gguf", content)
    _write_manifest(model_dir, [_entry("qwen-7b", "qwen-7b-q4.gguf", content)])

    prov = Provisioner(model_dir)
    path = prov.require_model("qwen-7b")

    assert path == model_dir / "qwen-7b-q4.gguf"
    assert path.is_file()


def test_verify_all_reports_every_entry_ok(tmp_path):
    model_dir = tmp_path / "models"
    a = b"AAAA" * 10
    b = b"BBBB" * 20
    _write_weight(model_dir, "a.bin", a)
    _write_weight(model_dir, "b.bin", b)
    _write_manifest(
        model_dir,
        [_entry("a", "a.bin", a), _entry("b", "b.bin", b)],
    )

    results = Provisioner(model_dir).verify_all()
    assert [r.ok for r in results] == [True, True]


def test_require_model_never_touches_the_network(tmp_path, monkeypatch):
    """AC-9 runtime guarantee: provisioning reads local files only."""
    model_dir = tmp_path / "models"
    content = b"local-only-weights" * 5
    _write_weight(model_dir, "whisper-small.mlmodel", content)
    _write_manifest(model_dir, [_entry("whisper-small", "whisper-small.mlmodel", content)])

    def _no_network(*_args, **_kwargs):
        raise AssertionError("provisioner attempted a network connection")

    monkeypatch.setattr(socket, "socket", _no_network)
    # If require_model tried the network, the AssertionError above would propagate
    # instead of a clean path return.
    path = Provisioner(model_dir).require_model("whisper-small")
    assert path.is_file()


def test_manifest_roundtrip_preserves_entries(tmp_path):
    model_dir = tmp_path / "models"
    content = b"roundtrip" * 3
    _write_weight(model_dir, "m.bin", content)
    entries = [_entry("m", "m.bin", content)]
    _write_manifest(model_dir, entries)

    manifest = Provisioner(model_dir).load_manifest()
    text = manifest.to_json()
    restored = ModelManifest.from_json(text)

    assert restored.schema_version == MANIFEST_SCHEMA_VERSION
    assert restored.models[0].model_id == "m"
    assert restored.models[0].sha256 == hashlib.sha256(content).hexdigest()


# --------------------------------------------------------------------------- #
# Negative tests (expected to FAIL / be rejected)                              #
# --------------------------------------------------------------------------- #

def test_checksum_mismatch_is_rejected_before_use(tmp_path):
    model_dir = tmp_path / "models"
    content = b"original-bytes" * 8
    _write_weight(model_dir, "w.bin", content)
    _write_manifest(model_dir, [_entry("w", "w.bin", content)])

    # Tamper with the weight file AFTER the manifest was written, keeping the
    # same length so the SHA-256 (not the size) check is what catches it.
    tampered = b"ORIGINAL-BYTES" * 8  # same length, different bytes
    assert len(tampered) == len(content)
    (model_dir / "w.bin").write_bytes(tampered)

    prov = Provisioner(model_dir)
    with pytest.raises(ChecksumMismatchError):
        prov.require_model("w")

    result = prov.verify_entry(prov.load_manifest().models[0])
    assert result.ok is False
    assert "sha256" in result.reason


def test_size_mismatch_is_rejected(tmp_path):
    model_dir = tmp_path / "models"
    content = b"X" * 64
    _write_weight(model_dir, "w.bin", content)
    entry = _entry("w", "w.bin", content)
    entry["size_bytes"] = len(content) + 10  # wrong size in manifest
    _write_manifest(model_dir, [entry])

    with pytest.raises(ChecksumMismatchError):
        Provisioner(model_dir).require_model("w")


def test_missing_weight_fast_fails_without_download(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    # Manifest references a weight that is NOT present on disk.
    _write_manifest(model_dir, [_entry("ghost", "ghost.bin", b"absent" * 4)])

    def _no_network(*_args, **_kwargs):
        raise AssertionError("provisioner attempted a network download for a missing model")

    monkeypatch.setattr(socket, "socket", _no_network)

    with pytest.raises(ModelNotProvisionedError) as exc:
        Provisioner(model_dir).require_model("ghost")
    assert "not provisioned" in str(exc.value).lower()


def test_missing_manifest_raises_manifest_error(tmp_path):
    prov = Provisioner(tmp_path / "does-not-exist")
    with pytest.raises(ManifestError):
        prov.load_manifest()


def test_unknown_model_id_raises_not_provisioned(tmp_path):
    model_dir = tmp_path / "models"
    content = b"present" * 4
    _write_weight(model_dir, "a.bin", content)
    _write_manifest(model_dir, [_entry("a", "a.bin", content)])

    with pytest.raises(ModelNotProvisionedError):
        Provisioner(model_dir).require_model("not-declared")


def test_malformed_manifest_bad_sha256_pattern(tmp_path):
    model_dir = tmp_path / "models"
    _write_weight(model_dir, "w.bin", b"data" * 4)
    entry = _entry("w", "w.bin", b"data" * 4)
    entry["sha256"] = "not-a-hex-digest"  # violates the 64-hex pattern
    _write_manifest(model_dir, [entry])

    with pytest.raises(ManifestError):
        Provisioner(model_dir).load_manifest()


def test_malformed_manifest_missing_required_field(tmp_path):
    model_dir = tmp_path / "models"
    _write_weight(model_dir, "w.bin", b"data" * 4)
    entry = _entry("w", "w.bin", b"data" * 4)
    del entry["size_bytes"]  # required field removed
    _write_manifest(model_dir, [entry])

    with pytest.raises(ManifestError):
        Provisioner(model_dir).load_manifest()


def test_unsupported_manifest_schema_version(tmp_path):
    model_dir = tmp_path / "models"
    (model_dir).mkdir(parents=True, exist_ok=True)
    (model_dir / "models.json").write_text(
        json.dumps({"schema_version": "999", "models": []}),
        encoding="utf-8",
    )
    with pytest.raises(ManifestError):
        Provisioner(model_dir).load_manifest()


def test_corrupt_manifest_json(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "models.json").write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(ManifestError):
        Provisioner(model_dir).load_manifest()
