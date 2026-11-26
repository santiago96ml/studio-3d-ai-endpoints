"""Distilled Hunyuan3D-2 headless engine (shape-only FlashVDM, GPU lease safe)."""

from __future__ import annotations

import gc
import io
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class Hunyuan3DConfig:
    variant: str = os.environ.get("HUNYUAN3D_VARIANT", "turbo")  # "turbo" | "mini"
    vram_mode: str = os.environ.get("HUNYUAN3D_VRAM_MODE", "auto")  # "auto" | "resident" | "stage_offload"
    unload_after_request: bool = os.environ.get("HUNYUAN3D_UNLOAD_AFTER_REQUEST", "0") == "1"
    default_steps: int = int(os.environ.get("HUNYUAN3D_STEPS", "5"))
    default_guidance: float = float(os.environ.get("HUNYUAN3D_GUIDANCE", "5.0"))
    default_octree_res: int = int(os.environ.get("HUNYUAN3D_OCTREE_RES", "384"))
    default_max_faces: int = int(os.environ.get("HUNYUAN3D_MAX_FACES", "150000"))
    device: str = os.environ.get("HUNYUAN3D_DEVICE", "cuda")


def query_nvidia_smi() -> dict[str, Any]:
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


class Hunyuan3DEngine:
    def __init__(self, config: Hunyuan3DConfig | None = None):
        self.config = config or Hunyuan3DConfig()
        self._pipelines: dict[str, Any] = {}
        self._active_variant: str | None = None
        self._rembg_session: Any | None = None
        self._lock = threading.Lock()
        self.last_load_seconds: float = 0.0

    def _get_pipeline(self, variant: str | None = None):
        import torch

        from services.hunyuan3d2.hy3d_core.pipeline import DistilledHunyuan3DPipeline

        target_variant = (variant or self.config.variant).strip().lower()
        if target_variant not in ("turbo", "mini"):
            target_variant = "turbo"

        if target_variant in self._pipelines:
            pipe = self._pipelines[target_variant]
            if pipe.vram_mode == "resident":
                pipe.to(self.config.device, torch.float16)
            return pipe

        # Keep only one variant in memory at a time to stay well below Colab 12.7 GB RAM
        if self._pipelines:
            self._pipelines.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        t0 = time.perf_counter()
        dev = self.config.device if (self.config.device != "cuda" or torch.cuda.is_available()) else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.float32
        pipe = DistilledHunyuan3DPipeline.from_variant(
            variant=target_variant,
            device=dev,
            dtype=dtype,
            vram_mode=self.config.vram_mode,
        )
        self.last_load_seconds = round(time.perf_counter() - t0, 2)
        self._pipelines[target_variant] = pipe
        self._active_variant = target_variant
        return pipe

    def _prepare_rgba(self, image_bytes: bytes, bg_removal: str = "auto") -> tuple[Image.Image, str]:
        from services.hunyuan3d2.hy3d_core.preprocessors import fast_white_bg_to_rgba

        img = Image.open(io.BytesIO(image_bytes))
        mode = (bg_removal or "auto").strip().lower()
        if mode == "none":
            return img.convert("RGBA"), "none"

        if mode != "rembg":
            rgba, clean = fast_white_bg_to_rgba(img)
            if clean:
                return rgba, "fast_white_cutout"

        # Lazy fallback for noisy real-world photos ONLY when border isn't clean white/transparent
        try:
            from rembg import new_session, remove

            if self._rembg_session is None:
                self._rembg_session = new_session("u2netp")
            out = remove(img.convert("RGBA"), session=self._rembg_session)
            return out.convert("RGBA"), "rembg_u2netp"
        except Exception:
            return img.convert("RGBA"), "fallback_raw"

    def unload(self) -> dict[str, Any]:
        import torch

        with self._lock:
            for pipe in self._pipelines.values():
                pipe.offload_to_cpu()
            if self.config.unload_after_request:
                self._pipelines.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"status": "unloaded", "gpu": query_nvidia_smi()}

    def status(self) -> dict[str, Any]:
        from services.hunyuan3d2.hy3d_core.pipeline import HUNYUAN_VARIANTS

        variant_key = self._active_variant or self.config.variant
        spec = HUNYUAN_VARIANTS.get(variant_key, HUNYUAN_VARIANTS["turbo"])
        pipe = self._pipelines.get(variant_key)
        return {
            "ok": True,
            "service": "hunyuan3d2",
            "model": f"{spec['dit_repo']}/{spec['dit_subfolder']}",
            "vae": f"{spec['vae_repo']}/{spec['vae_subfolder']}",
            "variant": variant_key,
            "loaded": pipe is not None,
            "vram_mode": pipe.vram_mode if pipe is not None else self.config.vram_mode,
            "has_texture_weights": False,
            "distilled_core": True,
            "last_load_seconds": self.last_load_seconds,
            "ready": True,
            "gpu": query_nvidia_smi(),
        }

    def generate_glb(
        self,
        image_bytes: bytes,
        steps: int | None = None,
        guidance: float | None = None,
        octree_res: int | None = None,
        max_faces: int | None = None,
        seed: int = 42,
        bg_removal: str = "auto",
        ensure_watertight: bool = True,
        variant: str | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        import torch

        from services.hunyuan3d2.hy3d_core.postprocessors import clean_and_repair_mesh

        steps = int(steps if steps is not None else self.config.default_steps)
        guidance = float(guidance if guidance is not None else self.config.default_guidance)
        octree_res = int(octree_res if octree_res is not None else self.config.default_octree_res)
        max_faces = int(max_faces if max_faces is not None else self.config.default_max_faces)

        # Normalize UI slider scales (512 -> 384, 256 -> 256, 384 -> 384)
        octree_res = max(128, min(octree_res, 448))
        if octree_res > 384 and octree_res <= 512:
            octree_res = 384

        with self._lock:
            t0 = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            pipe = self._get_pipeline(variant=variant)
            rgba_img, cutout_method = self._prepare_rgba(image_bytes, bg_removal=bg_removal)

            raw_mesh = pipe(
                image=rgba_img,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=int(seed),
                octree_resolution=octree_res,
            )

            repaired_mesh, mesh_stats = clean_and_repair_mesh(
                raw_mesh,
                max_faces=max_faces,
                ensure_watertight=ensure_watertight,
            )

            buf = io.BytesIO()
            repaired_mesh.export(buf, file_type="glb")
            glb_bytes = buf.getvalue()

            vram_peak_mb = (
                int(round(torch.cuda.max_memory_allocated() / (1024 * 1024)))
                if torch.cuda.is_available()
                else 0
            )
            if self.config.unload_after_request:
                pipe.offload_to_cpu()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = round(time.perf_counter() - t0, 2)
            meta = {
                "engine": "hunyuan3d2-distilled",
                "variant": self._active_variant or self.config.variant,
                "vram_mode": pipe.vram_mode,
                "steps": steps,
                "octree_resolution": octree_res,
                "cutout_method": cutout_method,
                "elapsed_s": elapsed,
                "vram_peak_mb": vram_peak_mb,
                **mesh_stats,
            }
            return glb_bytes, meta
