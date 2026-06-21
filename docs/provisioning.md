# Offline Model Provisioning

LocalMind Audio runs **100% on-device** with **zero network access at runtime**.
Model weights are therefore provisioned *out-of-band* — downloaded once, by the
user, outside the application — and pinned by an integrity manifest. The
pipeline never fetches a weight at inference time; a missing or tampered model
fails fast with an explicit error rather than a silent network call.

This document is the canonical provisioning workflow (acceptance criterion
**AC-9**).

## 1. Model directory layout

All weights live under a single local model directory (default
`./models/`, overridable by the caller). The manifest sits at its root:

```
models/
├── models.json                 # the integrity manifest (see §2)
├── whisper/
│   ├── whisper-small-q4.mlmodel
│   └── whisper-medium-q4.mlmodel
└── llm/
    ├── qwen2.5-7b-instruct-q4.gguf
    └── llama-3-8b-instruct-q4.gguf
```

Rules:

* Paths in `models.json` are **relative to the model directory** and **must stay
  inside it**. Absolute paths and `..` traversal are rejected by the validator
  (and again by the provisioner at resolve time, defense-in-depth).
* Weight files are **never committed** to git — see `.gitignore` (`models/`,
  `*.gguf`, `*.mlpackage/`, `*.weights`).

## 2. Manifest schema (`models.json`)

Versioned via `schema_version` (currently `"1"`). Validated with the Python
standard library only — validation correctness does **not** depend on any
third-party package being installed.

```json
{
  "schema_version": "1",
  "models": [
    {
      "model_id": "whisper-small",
      "name": "Whisper Small (INT4)",
      "kind": "whisper",
      "path": "whisper/whisper-small-q4.mlmodel",
      "quant_format": "int4",
      "size_bytes": 484477137,
      "sha256": "<64 hex chars>",
      "license": "MIT"
    },
    {
      "model_id": "whisper-medium",
      "name": "Whisper Medium (INT4)",
      "kind": "whisper",
      "path": "whisper/whisper-medium-q4.mlmodel",
      "quant_format": "int4",
      "size_bytes": 772701679,
      "sha256": "<64 hex chars>",
      "license": "MIT"
    },
    {
      "model_id": "qwen2.5-7b",
      "name": "Qwen2.5-7B-Instruct (INT4)",
      "kind": "llm",
      "path": "llm/qwen2.5-7b-instruct-q4.gguf",
      "quant_format": "int4",
      "size_bytes": 4372604098,
      "sha256": "<64 hex chars>",
      "license": "Apache-2.0"
    }
  ]
}
```

Field rules (enforced by `validate_manifest_dict`):

| Field | Type | Constraint |
|-------|------|------------|
| `schema_version` | string | must equal `"1"` |
| `models` | array | required; each entry is an object |
| `model_id` | string | non-empty; **unique** across the manifest |
| `name` | string | human-readable label |
| `kind` | string | one of `"whisper"`, `"llm"` |
| `path` | string | non-empty, **relative**, no `..` components |
| `quant_format` | string | e.g. `"int4"`, `"int8"`, `"f16"` |
| `size_bytes` | integer | non-negative (booleans rejected) |
| `sha256` | string | exactly 64 hex characters |
| `license` | string | optional SPDX identifier |

## 3. One-time weight acquisition (out-of-band)

Perform this **once**, outside the app, on a machine with network access. The
acquired files are then copied to the target Mac's `models/` directory.

1. Obtain the weights from their upstream source (e.g. Hugging Face) and
   quantize/convert to the desired on-device format (`int4` GGUF for LLMs,
   CoreML/MLX `int4` for Whisper).
2. Place each file under `models/` at the relative path declared in the
   manifest.
3. For each file, record its byte size and SHA-256:

   ```bash
   # macOS: size in bytes + SHA-256
   stat -f%z models/whisper/whisper-small-q4.mlmodel
   shasum -a 256 models/whisper/whisper-small-q4.mlmodel
   ```

4. Write the corresponding entry into `models.json` (use the values from step 3
   verbatim). The helper below emits a starter entry for a file:

   ```bash
   python -c "
   import hashlib, os, sys
   p = sys.argv[1]
   mid = sys.argv[2]
   data = open(p, 'rb').read()
   print('model_id:', mid)
   print('size_bytes:', len(data))
   print('sha256:', hashlib.sha256(data).hexdigest())
   " models/whisper/whisper-small-q4.mlmodel whisper-small
   ```

5. Record the `license` (SPDX id) and `quant_format` for each entry — required
   for license tracking and reproducibility.

## 4. Verification at runtime

`Provisioner(model_dir)` exposes:

* `load_manifest()` — parse + validate `models.json` (raises `ManifestError`).
* `verify_all()` — verify every entry (returns `VerificationResult` per entry).
* `require_model(model_id)` — return the **verified** on-disk path for a model.

`require_model` checks, in order: path confinement → file exists → size →
SHA-256. It **never** downloads. Failure modes:

| Condition | Exception |
|-----------|-----------|
| manifest missing/malformed/wrong version | `ManifestError` |
| unknown `model_id` | `ModelNotProvisionedError` |
| weight file missing | `ModelNotProvisionedError` |
| size mismatch | `ChecksumMismatchError` |
| SHA-256 mismatch | `ChecksumMismatchError` |
| path escapes model dir | `ManifestError` |

The "no download" guarantee is covered by a socket-monkeypatch test:
`require_model` for a missing model raises `ModelNotProvisionedError` while
`socket.socket` is patched to explode on any use, proving the runtime path
touches no network.

## 5. Failure behavior (what users see)

* **First run with no `models/`:** `ManifestError("manifest not found at ...;
  provision models out-of-band first")`. The app does not attempt a download.
* **Tampered weight:** `ChecksumMismatchError` naming the model and which check
  failed (size or SHA-256). Re-acquire the file per §3.
* **Missing weight but manifest present:** `ModelNotProvisionedError("model not
  provisioned: weight file missing for ...")`.

## 6. Forward compatibility

Bump `schema_version` and extend `validate_manifest_dict` when the on-disk shape
changes. The provisioner refuses unknown versions, so old builds fail loudly
against a newer manifest rather than silently misbehaving.
