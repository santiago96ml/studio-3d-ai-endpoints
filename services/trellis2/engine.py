"""Dedicated Microsoft TRELLIS.2-4B inference engine backed by native trellis.cpp GGML CUDA."""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from common.runtime import detect_gpu_info, run_subprocess_checked

REQUIRED_GEOM_FILES = (
    "birefnet.gguf",
    "dinov3.gguf",
    "ss_flow.gguf",
    "ss_dec.gguf",
    "shape_flow_512.gguf",
    "shape_dec.gguf",
)
TEX_FILES = ("tex_flow_512.gguf", "tex_dec.gguf")
RES1024_FILES = ("shape_flow_1024.gguf", "tex_flow_1024.gguf")
VALID_BG_MODES = ("auto", "birefnet", "threshold")


@dataclass(frozen=True)
class Trellis2Config:
    """Runtime configuration for the isolated Microsoft TRELLIS.2-4B worker."""

    bin_path: str
    models_dir: str
    quant: str = "q4"
    default_res: int = 512
    timeout_sec: int = 420
    require_gpu: bool = True

    @classmethod
    def from_env(cls) -> "Trellis2Config":
        return cls(
            bin_path=os.environ.get("TRELLIS2_BIN_PATH", "/content/runtime/trellis_bin/trellis-cli").strip(),
            models_dir=os.environ.get("TRELLIS2_MODELS_DIR", "/content/models/trellis2").strip(),
            quant=os.environ.get("TRELLIS2_QUANT", "q4").strip().lower(),
            default_res=int(os.environ.get("TRELLIS2_DEFAULT_RES", "512")),
            timeout_sec=int(os.environ.get("TRELLIS2_TIMEOUT_SEC", "420")),
            require_gpu=os.environ.get("TRELLIS2_REQUIRE_GPU", "1").strip() in ("1", "true", "yes"),
        )


def build_trellis_command(
    cfg: Trellis2Config,
    input_paths: Sequence[Path],
    out_glb: Path,
    res: int = 512,
    steps: int = 12,
    faces: int = 150000,
    seed: int = 42,
    no_texture: bool = True,
    bg_removal: str = "auto",
    view_angles: Sequence[int] = (),
) -> list[str]:
    """Build verified trellis.cpp CLI invocation supporting single or multi-view inputs."""
    if not input_paths:
        raise ValueError("at least one input image is required")

    resolved_res = 1024 if int(res) >= 1024 else 512
    resolved_steps = max(4, min(30, int(steps)))
    resolved_faces = max(5000, min(500000, int(faces)))
    bg = bg_removal.strip().lower() if bg_removal.strip().lower() in VALID_BG_MODES else "auto"

    if len(input_paths) == 1:
        cmd = [cfg.bin_path, str(input_paths[0]), str(out_glb)]
    else:
        cmd = [cfg.bin_path]
        default_step = 360 // max(1, len(input_paths))
        for idx, p in enumerate(input_paths):
            ang = int(view_angles[idx]) if idx < len(view_angles) else (idx * default_step) % 360
            cmd.extend(["--view", f"{p}:{ang}"])
        cmd.extend(["-o", str(out_glb)])

    cmd.extend([
        "--models",
        cfg.models_dir,
        "--res",
        str(resolved_res),
        "--steps",
        str(resolved_steps),
        "--faces",
        str(resolved_faces),
        "--seed",
        str(int(seed)),
        "--bg-removal",
        bg,
    ])
    if no_texture:
        cmd.append("--no-texture")
    if cfg.require_gpu:
        cmd.append("--require-gpu")
    return cmd


class Trellis2Engine:
    """Stateless per-request runner for TRELLIS.2-4B GGUF reconstruction."""

    def __init__(self, cfg: Trellis2Config | None = None):
        self.cfg = cfg or Trellis2Config.from_env()

    def health(self) -> dict:
        models_path = Path(self.cfg.models_dir)
        bin_ok = Path(self.cfg.bin_path).is_file()
        missing = [fn for fn in REQUIRED_GEOM_FILES if not (models_path / fn).is_file()]
        has_tex = all((models_path / fn).is_file() for fn in TEX_FILES)
        has_1024 = (models_path / "shape_flow_1024.gguf").is_file()
        return {
            "ok": bin_ok and len(missing) == 0,
            "service": "trellis2",
            "model": "microsoft/TRELLIS.2-4B",
            "backend": "trellis.cpp-ggml-cuda",
            "quant": self.cfg.quant,
            "bin_present": bin_ok,
            "has_texture_weights": has_tex,
            "has_1024_weights": has_1024,
            "missing_files": missing,
            "gpu": detect_gpu_info(),
        }

    def reconstruct(
        self,
        images: Sequence[bytes],
        view_angles: Sequence[int] = (),
        res: int = 512,
        steps: int = 12,
        faces: int = 150000,
        seed: int = 42,
        no_texture: bool = True,
        bg_removal: str = "auto",
        strict: bool = False,
    ) -> tuple[bytes, dict]:
        h = self.health()
        if not h["ok"]:
            raise RuntimeError(f"TRELLIS.2-4B assets missing: bin={h['bin_present']} missing={h['missing_files']}")

        effective_no_tex = bool(no_texture)
        texture_fallback = False
        if not effective_no_tex and not h["has_texture_weights"]:
            if strict:
                raise ValueError("Texture generation requested, but tex_*.gguf weights are not downloaded")
            effective_no_tex = True
            texture_fallback = True

        effective_res = 1024 if (int(res) >= 1024 and h["has_1024_weights"]) else 512

        with tempfile.TemporaryDirectory(prefix="trellis2_job_") as tmp:
            tmp_dir = Path(tmp)
            in_paths: list[Path] = []
            for idx, raw in enumerate(images):
                if not raw:
                    continue
                p = tmp_dir / f"view_{idx}.png"
                with Image.open(io.BytesIO(raw)) as im:
                    # Preserve alpha channel when caller sends RGBA transparent PNG.
                    mode = "RGBA" if "A" in im.getbands() else "RGB"
                    im.convert(mode).save(p, format="PNG")
                in_paths.append(p)

            if not in_paths:
                raise ValueError("at least one non-empty PNG/JPG image is required")

            out_glb = tmp_dir / "model.glb"
            cmd = build_trellis_command(
                self.cfg,
                input_paths=in_paths,
                out_glb=out_glb,
                res=effective_res,
                steps=steps,
                faces=faces,
                seed=seed,
                no_texture=effective_no_tex,
                bg_removal=bg_removal,
                view_angles=view_angles,
            )

            bin_dir = str(Path(self.cfg.bin_path).parent)
            prev_ld = os.environ.get("LD_LIBRARY_PATH", "")
            env = {"LD_LIBRARY_PATH": f"{bin_dir}:{prev_ld}" if prev_ld else bin_dir}

            rc, stdout, stderr, elapsed = run_subprocess_checked(cmd, env=env, timeout_sec=self.cfg.timeout_sec)
            if rc != 0:
                tail = (stderr or stdout)[-600:].strip()
                raise RuntimeError(f"trellis-cli exited {rc}: {tail}")
            if not out_glb.is_file() or out_glb.stat().st_size < 16:
                raise RuntimeError("trellis-cli completed without producing valid GLB")

            glb_bytes = out_glb.read_bytes()
            if not glb_bytes.startswith(b"glTF"):
                raise RuntimeError("trellis-cli output did not have valid glTF binary header")

            return glb_bytes, {
                "elapsed_sec": elapsed,
                "res": effective_res,
                "steps": max(4, min(30, int(steps))),
                "faces": max(5000, min(500000, int(faces))),
                "views_used": len(in_paths),
                "no_texture": effective_no_tex,
                "texture_fallback": texture_fallback,
                "bytes": len(glb_bytes),
            }
