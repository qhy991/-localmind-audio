"""MLX Metal runtime preflight: check Metal availability without polluting the
parent process with MLX atexit side effects.

The core problem: importing ``mlx.core`` registers a nanobind atexit callback
that prints a ``RuntimeError`` to stderr when no Metal device is available.
On headless/sandboxed macOS sessions this pollutes the CLI's JSONL progress
stream. This module checks Metal availability in a **subprocess**, so the
parent process never imports MLX and never registers the atexit hook.
"""

from __future__ import annotations

import subprocess
import sys


def ensure_mlx_metal_available() -> None:
    """Verify that MLX Metal is usable via a subprocess preflight.

    Runs a tiny ``mx.eval(mx.zeros((1,)))`` in a child process. If the child
    exits nonzero, raises ``RuntimeError`` with a message that distinguishes
    "MLX not installed" from "Metal device unavailable". The parent process
    never imports ``mlx.core``, so no atexit hook is registered.

    Raises
    ------
    RuntimeError
        If MLX is not installed or the Metal device is unavailable.
    """
    code = (
        "try:\n"
        "    import mlx.core as mx\n"
        "except ImportError:\n"
        "    print('NOT_INSTALLED'); raise SystemExit(1)\n"
        "try:\n"
        "    mx.eval(mx.zeros((1,)))\n"
        "    print('ok')\n"
        "except Exception as exc:\n"
        "    print(f'METAL_ERROR:{exc}'); raise SystemExit(2)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"MLX Metal preflight failed to execute: {exc}") from exc

    if proc.returncode != 0:
        stdout = proc.stdout.strip()
        if "NOT_INSTALLED" in stdout:
            raise RuntimeError(
                "MLX is not installed; install the ML backend with "
                "`pip install -e .[ml]` (see docs/provisioning.md)"
            )
        if "METAL_ERROR" in stdout:
            detail = stdout.split("METAL_ERROR:")[-1].strip() if "METAL_ERROR:" in stdout else "unknown"
            raise RuntimeError(
                f"MLX Metal device unavailable — backend cannot run in this session. "
                f"Detail: {detail}"
            )
        stderr = proc.stderr.strip()
        detail = stderr.split("\n")[-1] if stderr else "unknown error"
        raise RuntimeError(
            f"MLX Metal preflight failed (rc={proc.returncode}). Detail: {detail}"
        )
