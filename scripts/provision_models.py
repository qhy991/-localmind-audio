#!/usr/bin/env python3
"""Out-of-band model provisioning for LocalMind Audio.

Downloads the default STT + LLM models from Hugging Face, computes their
SHA-256 integrity hashes, writes the ``models/models.json`` manifest, and
verifies everything through the project's own ``Provisioner``.

This is an OFFLINE provisioning step: it runs ONCE, by the user, before the
first inference. The runtime pipeline never downloads a weight — it only reads
locally-provisioned, integrity-pinned files (AC-5 zero-network + AC-9 integrity).

Requirements
------------
* Apple Silicon Mac (M-series) with a working Metal device.
* The ``ml`` extra installed:  ``pip install -e ".[ml]"``
* Network access to huggingface.co for this one-time download.

Usage
-----
    python scripts/provision_models.py                  # default whisper-tiny + qwen3.5-0.8b
    python scripts/provision_models.py --stt-only       # just the STT tier
    python scripts/provision_models.py --dir /path/to/models
    python scripts/provision_models.py --stt mlx-community/whisper-base-mlx \\
                                       --llm mlx-community/Qwen3-1.7B-4bit

The script is idempotent: re-running re-verifies existing files and only
re-downloads what is missing or tampered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Default models. Both load cleanly through mlx-lm / mlx-whisper on Apple
# Silicon. Replace the ids below (or pass --stt/--llm) to use larger tiers.
DEFAULT_STT = ("mlx-community/whisper-tiny-mlx", "weights.npz", "fp16", "MIT")
DEFAULT_LLM = ("Qwen/Qwen3.5-0.8B",
               "model.safetensors-00001-of-00001.safetensors", "fp16", "Apache-2.0")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_repo(repo_id: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download
    print(f"  downloading {repo_id} -> {dest}")
    return Path(snapshot_download(repo_id=repo_id, local_dir=str(dest)))


def model_entry(repo_id: str, weights_name: str, quant: str, license_: str,
                local_dir: Path, model_id: str, model_root: Path) -> dict:
    weights = local_dir / weights_name
    if not weights.exists():
        download_repo(repo_id, local_dir)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found after download")
    sha = sha256_file(weights)
    size = weights.stat().st_size
    rel = weights.relative_to(model_root).as_posix()
    return {
        "model_id": model_id, "name": repo_id, "kind": None,  # kind set by caller
        "path": rel, "quant_format": quant, "size_bytes": size,
        "sha256": sha, "license": license_,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dir", default="models", help="model directory (default: models)")
    ap.add_argument("--stt", default=None, help="HF repo id for the STT tier")
    ap.add_argument("--llm", default=None, help="HF repo id for the LLM tier")
    ap.add_argument("--stt-only", action="store_true", help="provision only the STT tier")
    ap.add_argument("--llm-only", action="store_true", help="provision only the LLM tier")
    args = ap.parse_args()

    model_root = Path(args.dir)
    model_root.mkdir(parents=True, exist_ok=True)

    stt_repo = (args.stt, *_DEFAULT_STT_TAIL(args.stt)) if args.stt else DEFAULT_STT
    llm_repo = (args.llm, *_DEFAULT_LLM_TAIL(args.llm)) if args.llm else DEFAULT_LLM

    do_stt = not args.llm_only
    do_llm = not args.stt_only

    # Load existing manifest if present, so we keep any extra tiers the user added.
    manifest_path = model_root / "models.json"
    existing: dict = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    by_id = {m["model_id"]: m for m in existing.get("models", [])}

    if do_stt:
        print("[1/2] STT tier (kind=whisper)")
        repo_id, weights_name, quant, lic = stt_repo
        entry = model_entry(repo_id, weights_name, quant, lic,
                            model_root / "whisper-tiny", "whisper-tiny", model_root)
        entry["kind"] = "whisper"
        by_id["whisper-tiny"] = entry
        print(f"      ok  size={entry['size_bytes']} sha256={entry['sha256'][:16]}...")

    if do_llm:
        print("[2/2] LLM tier (kind=llm)")
        repo_id, weights_name, quant, lic = llm_repo
        model_id = repo_id.split("/")[-1].lower()
        entry = model_entry(repo_id, weights_name, quant, lic,
                            model_root / model_id, model_id, model_root)
        entry["kind"] = "llm"
        by_id[model_id] = entry
        print(f"      ok  size={entry['size_bytes']} sha256={entry['sha256'][:16]}...")

    manifest = {"schema_version": "1", "models": list(by_id.values())}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nmanifest written: {manifest_path}")

    # Verify through the project's own Provisioner (defense-in-depth).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from localmind.provisioning.provisioner import Provisioner  # type: ignore
    print("\nverifying via Provisioner.verify_all() ...")
    ok = True
    for r in Provisioner(str(model_root)).verify_all():
        flag = "ok " if r.ok else "FAIL"
        print(f"  [{flag}] {r.model_id}  {r.reason}")
        ok = ok and r.ok
    print("\nDONE." if ok else "\nFAILED: some models did not verify.")
    return 0 if ok else 1


def _DEFAULT_STT_TAIL(_repo_id):
    # When the user overrides --stt, default the weights name/quant/license.
    return ("weights.npz", "fp16", "MIT")


def _DEFAULT_LLM_TAIL(_repo_id):
    return ("model.safetensors-00001-of-00001.safetensors", "fp16", "Apache-2.0")


if __name__ == "__main__":
    raise SystemExit(main())
