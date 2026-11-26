"""Dedicated FLUX.2-klein-4B inference engine backed by native stable-diffusion.cpp CUDA."""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from common.runtime import detect_gpu_info, run_subprocess_checked


@dataclass(frozen=True)
class Flux2Config:
    """Runtime configuration for the isolated FLUX.2-klein-4B worker."""

    bin_path: str
    model_path: str
    llm_path: str
    vae_path: str
    quant: str = "Q4_K_M"
    offload_to_cpu: bool = False
    timeout_sec: int = 240
    backend: str = "sdcpp"

    @classmethod
    def from_env(cls) -> "Flux2Config":
        return cls(
            bin_path=os.environ.get("FLUX2_BIN_PATH", "/content/runtime/stable-diffusion.cpp/build/bin/sd-cli").strip(),
            model_path=os.environ.get("FLUX2_MODEL_PATH", "/content/models/flux2/flux-2-klein-4b-Q4_K_M.gguf").strip(),
            llm_path=os.environ.get("FLUX2_LLM_PATH", "/content/models/flux2/Qwen3-4B-Q4_K_M.gguf").strip(),
            vae_path=os.environ.get("FLUX2_VAE_PATH", "/content/models/flux2/flux2-vae.safetensors").strip(),
            quant=os.environ.get("FLUX2_QUANT", "Q4_K_M").strip(),
            offload_to_cpu=os.environ.get("FLUX2_OFFLOAD_TO_CPU", "0").strip() in ("1", "true", "yes"),
            timeout_sec=int(os.environ.get("FLUX2_TIMEOUT_SEC", "240")),
            backend=os.environ.get("FLUX2_BACKEND", "sdcpp").strip().lower(),
        )


def _clamp_dim(val: int, lo: int = 256, hi: int = 1024) -> int:
    clamped = max(lo, min(hi, int(val)))
    return max(lo, (clamped // 16) * 16)


def build_sdcpp_command(
    cfg: Flux2Config,
    prompt: str,
    out_path: Path,
    ref_paths: Sequence[Path] = (),
    width: int = 768,
    height: int = 768,
    steps: int = 4,
    cfg_scale: float = 1.0,
    seed: int = 42,
    negative_prompt: str = "",
) -> list[str]:
    """Build verified stable-diffusion.cpp CLI command for FLUX.2-klein-4B."""
    w = _clamp_dim(width)
    h = _clamp_dim(height)
    s = max(1, min(30, int(steps)))
    cmd = [
        cfg.bin_path,
        "--diffusion-model",
        cfg.model_path,
        "--vae",
        cfg.vae_path,
        "--llm",
        cfg.llm_path,
        "-p",
        prompt.strip(),
        "-W",
        str(w),
        "-H",
        str(h),
        "--cfg-scale",
        str(float(cfg_scale)),
        "--sampling-method",
        "euler",
        "--steps",
        str(s),
        "--seed",
        str(int(seed)),
        "-o",
        str(out_path),
    ]
    if negative_prompt.strip():
        cmd.extend(["-n", negative_prompt.strip()])
    for rp in ref_paths:
        cmd.extend(["-r", str(rp)])
    if cfg.offload_to_cpu:
        cmd.append("--offload-to-cpu")
    return cmd


class Flux2Engine:
    """Stateless per-request runner so GPU memory is 100% released after each inference."""

    def __init__(self, cfg: Flux2Config | None = None):
        self.cfg = cfg or Flux2Config.from_env()

    def health(self) -> dict:
        files = {
            "bin": Path(self.cfg.bin_path).is_file(),
            "model": Path(self.cfg.model_path).is_file(),
            "llm": Path(self.cfg.llm_path).is_file(),
            "vae": Path(self.cfg.vae_path).is_file(),
        }
        ready = all(files.values())
        return {
            "ok": ready,
            "service": "flux2_klein",
            "model": "black-forest-labs/FLUX.2-klein-4B",
            "backend": f"{self.cfg.backend}-ggml-cuda",
            "quant": self.cfg.quant,
            "offload_to_cpu": self.cfg.offload_to_cpu,
            "files": files,
            "gpu": detect_gpu_info(),
        }

    def generate(
        self,
        prompt: str,
        ref_images: Sequence[bytes] = (),
        negative_prompt: str = "",
        width: int = 768,
        height: int = 768,
        steps: int = 4,
        cfg_scale: float = 1.0,
        seed: int = 42,
    ) -> tuple[bytes, dict]:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        h = self.health()
        if not h["ok"]:
            missing = [k for k, present in h["files"].items() if not present]
            raise RuntimeError(f"FLUX.2-klein-4B assets missing: {missing}")

        with tempfile.TemporaryDirectory(prefix="flux2_job_") as tmp:
            tmp_dir = Path(tmp)
            ref_paths: list[Path] = []
            for idx, raw_bytes in enumerate(ref_images):
                if not raw_bytes:
                    continue
                ref_path = tmp_dir / f"ref_{idx}.png"
                with Image.open(io.BytesIO(raw_bytes)) as im:
                    im.convert("RGB").save(ref_path, format="PNG")
                ref_paths.append(ref_path)

            out_path = tmp_dir / "concept.png"
            cmd = build_sdcpp_command(
                self.cfg,
                prompt=prompt,
                out_path=out_path,
                ref_paths=ref_paths,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                negative_prompt=negative_prompt,
            )
            env = {}
            bin_dir = str(Path(self.cfg.bin_path).parent)
            prev_ld = os.environ.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{bin_dir}:{prev_ld}" if prev_ld else bin_dir
            prev_path = os.environ.get("PATH", "")
            env["PATH"] = f"{bin_dir}{os.pathsep}{prev_path}" if prev_path else bin_dir

            rc, stdout, stderr, elapsed = run_subprocess_checked(cmd, env=env, timeout_sec=self.cfg.timeout_sec)
            if rc != 0:
                tail = (stderr or stdout)[-600:].strip()
                raise RuntimeError(f"sd-cli exited {rc}: {tail}")
            if not out_path.is_file() or out_path.stat().st_size == 0:
                raise RuntimeError("sd-cli completed without producing output PNG")

            png_bytes = out_path.read_bytes()
            return png_bytes, {
                "elapsed_sec": elapsed,
                "width": _clamp_dim(width),
                "height": _clamp_dim(height),
                "steps": max(1, min(30, int(steps))),
                "seed": int(seed),
                "ref_count": len(ref_paths),
            }
