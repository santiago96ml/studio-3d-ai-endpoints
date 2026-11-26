"""Runtime helpers: GPU telemetry, bearer auth, and checked subprocess execution."""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any, Mapping

from fastapi import HTTPException


def query_nvidia_smi() -> dict[str, Any]:
    """Query nvidia-smi for GPU name, memory used/total (MB), and utilization (%)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        if not out:
            return {}
        name, used, total, util = [x.strip() for x in out.splitlines()[0].split(",")]
        return {
            "gpu_name": name,
            "vram_used_mb": int(float(used)),
            "vram_total_mb": int(float(total)),
            "gpu_util_pct": int(float(util)),
        }
    except Exception:
        return {}


def detect_gpu_info() -> dict:
    """Query nvidia-smi for GPU name, memory, driver, and compute capability."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {"available": False, "detail": (proc.stderr or "nvidia-smi unavailable").strip()}
        line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            return {"available": True, "raw": line}
        return {
            "available": True,
            "name": parts[0],
            "vram_total_mb": int(float(parts[1])),
            "vram_free_mb": int(float(parts[2])),
            "driver_version": parts[3],
            "compute_cap": parts[4],
        }
    except Exception as exc:
        return {"available": False, "detail": str(exc)}


def verify_api_token(headers: Mapping[str, str], expected_token: str | None = None) -> None:
    """Enforce optional Bearer or X-API-Key token when STUDIO_API_TOKEN is configured."""
    token = (expected_token if expected_token is not None else os.environ.get("STUDIO_API_TOKEN", "")).strip()
    if not token:
        return
    auth = (headers.get("authorization") or headers.get("Authorization") or "").strip()
    x_key = (headers.get("x-api-key") or headers.get("X-API-Key") or "").strip()
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if bearer == token or x_key == token:
        return
    raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing STUDIO_API_TOKEN")


def run_subprocess_checked(
    cmd: list[str],
    env: dict[str, str] | None = None,
    timeout_sec: int = 240,
) -> tuple[int, str, str, float]:
    """Execute a native inference binary and return (returncode, stdout, stderr, elapsed_sec)."""
    t0 = time.perf_counter()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        env=merged_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
    )
    elapsed = round(time.perf_counter() - t0, 3)
    return proc.returncode, proc.stdout or "", proc.stderr or "", elapsed
