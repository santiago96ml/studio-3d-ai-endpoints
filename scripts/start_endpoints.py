"""Supervisor launching each model in its own isolated Python/CUDA environment + Cloudflare tunnel."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRYCF_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _wait_http_ok(url: str, timeout_sec: float = 45.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _find_cloudflared(runtime_root: Path) -> str:
    local_bin = runtime_root / "bin" / "cloudflared"
    if local_bin.is_file():
        return str(local_bin)
    found = shutil.which("cloudflared")
    if found:
        return found
    raise FileNotFoundError("cloudflared not found; run scripts/setup_gateway.sh first")


def _spawn_uvicorn(python_bin: Path, module_app: str, port: int, extra_env: dict[str, str], log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(extra_env)
    env["PYTHONPATH"] = str(REPO_ROOT)
    log_f = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [
            str(python_bin),
            "-m",
            "uvicorn",
            module_app,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "info",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Start isolated per-model GPU endpoints")
    parser.add_argument("--mode", choices=["all", "flux2", "trellis2"], default="all")
    parser.add_argument("--env-root", default=os.environ.get("ENV_ROOT", "/content/envs"))
    parser.add_argument("--runtime-root", default=os.environ.get("RUNTIME_ROOT", "/content/runtime"))
    parser.add_argument("--no-tunnel", action="store_true")
    args = parser.parse_args()

    env_root = Path(args.env_root)
    runtime_root = Path(args.runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []

    def cleanup(*_args):
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        if args.mode in ("all", "flux2"):
            f_env = _load_env_file(runtime_root / "flux2.env")
            f_py = env_root / "flux2" / "bin" / "python"
            print(f"[start] Launching FLUX.2-klein-4B service on :8001 ({f_py})...")
            procs.append(_spawn_uvicorn(f_py, "services.flux2_klein.app:app", 8001, f_env, runtime_root / "flux2.log"))
            if not _wait_http_ok("http://127.0.0.1:8001/health"):
                print("[warn] FLUX.2 health check did not respond within 45s; inspect", runtime_root / "flux2.log")

        if args.mode in ("all", "trellis2"):
            t_env = _load_env_file(runtime_root / "trellis2.env")
            t_py = env_root / "trellis2" / "bin" / "python"
            print(f"[start] Launching TRELLIS.2-4B service on :8002 ({t_py})...")
            procs.append(_spawn_uvicorn(t_py, "services.trellis2.app:app", 8002, t_env, runtime_root / "trellis2.log"))
            if not _wait_http_ok("http://127.0.0.1:8002/health"):
                print("[warn] TRELLIS.2 health check did not respond within 45s; inspect", runtime_root / "trellis2.log")

        expose_port = 8001 if args.mode == "flux2" else (8002 if args.mode == "trellis2" else 8000)
        if args.mode == "all":
            g_py = env_root / "gateway" / "bin" / "python"
            g_env = {
                "FLUX2_URL": "http://127.0.0.1:8001",
                "TRELLIS2_URL": "http://127.0.0.1:8002",
            }
            print(f"[start] Launching GPU-Lease Gateway on :8000 ({g_py})...")
            procs.append(_spawn_uvicorn(g_py, "gateway.app:app", 8000, g_env, runtime_root / "gateway.log"))
            if not _wait_http_ok("http://127.0.0.1:8000/health"):
                print("[warn] Gateway health check did not respond; inspect", runtime_root / "gateway.log")

        if args.no_tunnel:
            print(f"[ready] Running locally on http://127.0.0.1:{expose_port} (--no-tunnel)")
            while all(p.poll() is None for p in procs):
                time.sleep(1.0)
            return 0

        cf_bin = _find_cloudflared(runtime_root)
        print(f"[tunnel] Starting Cloudflare Quick Tunnel -> http://127.0.0.1:{expose_port} ...")
        cf_proc = subprocess.Popen(
            [cf_bin, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{expose_port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        procs.append(cf_proc)

        public_url = None
        assert cf_proc.stdout is not None
        for line in cf_proc.stdout:
            sys.stdout.flush()
            m = TRYCF_RE.search(line)
            if m and not public_url:
                public_url = m.group(0)
                print("\n" + "=" * 68)
                print(f" PUBLIC URL (pegar en Robótica 3D Studio /admin): {public_url}")
                print(f"  - GET  {public_url}/health")
                if args.mode in ("all", "flux2"):
                    print(f"  - POST {public_url}/v1/concept")
                if args.mode in ("all", "trellis2"):
                    print(f"  - POST {public_url}/v1/image-to-3d")
                if args.mode == "all":
                    print(f"  - POST {public_url}/v1/sketch-to-3d")
                print("=" * 68 + "\n", flush=True)
    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
