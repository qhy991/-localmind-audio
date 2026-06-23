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
    exits nonzero (Metal unavailable, MLX not installed, etc.), raises
    ``RuntimeError`` with a clear message. The parent process never imports
    ``mlx.core``, so no atexit hook is registered.

    Raises
    ------
    RuntimeError
        If the subprocess cannot verify Metal availability.
    """
    code = (
        "import mlx.core as mx; "
        "mx.eval(mx.zeros((1,))); "
        "print('ok')"
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
        stderr = proc.stderr.strip()
        detail = stderr.split("\n")[-1] if stderr else "unknown error"
        raise RuntimeError(
            f"MLX Metal device unavailable — backend cannot run in this session. "
            f"Detail: {detail}"
        )
