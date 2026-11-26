"""Distilled PartField inference engine (pure PyTorch, HF selective ckpt, GPU lease safe)."""

from __future__ import annotations

import gc
import io
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from common.runtime import query_nvidia_smi


@dataclass
class PartFieldConfig:
    repo_id: str = os.environ.get("PARTFIELD_HF_REPO", "mikaelaangel/partfield-ckpt")
    ckpt_filename: str = os.environ.get("PARTFIELD_CKPT", "model_objaverse.ckpt")
    vram_mode: str = os.environ.get("PARTFIELD_VRAM_MODE", "auto")  # "auto" | "resident" | "unload"
    unload_after_request: bool = os.environ.get("PARTFIELD_UNLOAD_AFTER_REQUEST", "0") == "1"
    default_num_parts: int = int(os.environ.get("PARTFIELD_NUM_PARTS", "10"))
    pc_samples: int = int(os.environ.get("PARTFIELD_PC_SAMPLES", "30000"))
    points_per_face: int = int(os.environ.get("PARTFIELD_PTS_PER_FACE", "12"))
    sample_batch_pts: int = int(os.environ.get("PARTFIELD_BATCH_PTS", "12000"))
    device: str = os.environ.get("PARTFIELD_DEVICE", "cuda")


class PartFieldInferenceModel:
    """Thin wrapper loading ONLY pvcnn + triplane_transformer from Lightning model_objaverse.ckpt."""

    def __init__(self, device: str = "cpu"):
        import torch
        import torch.nn as nn
        from services.partfield.pf_core.pvcnn import TriPlanePC2Encoder
        from services.partfield.pf_core.triplane import TriplaneTransformer

        class _Module(nn.Module):
            def __init__(self, dev: str):
                super().__init__()
                self.pvcnn = TriPlanePC2Encoder(z_triplane_channels=256, z_triplane_resolution=128, device=dev)
                self.triplane_transformer = TriplaneTransformer(
                    input_dim=256,
                    transformer_dim=1024,
                    transformer_layers=6,
                    transformer_heads=8,
                    triplane_low_res=32,
                    triplane_high_res=128,
                    triplane_dim=512,
                )

        self.module = _Module(device)
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cuda") -> "PartFieldInferenceModel":
        import torch

        inst = cls(device="cpu")
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        filtered = {
            k: v
            for k, v in sd.items()
            if k.startswith("pvcnn.") or k.startswith("triplane_transformer.")
        }
        missing, unexpected = inst.module.load_state_dict(filtered, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"PartField state_dict mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        inst.module.eval().to(device)
        inst.device = torch.device(device)
        return inst

    def extract_face_features(
        self,
        vertices_norm: np.ndarray,
        faces: np.ndarray,
        pc_samples: int = 30000,
        points_per_face: int = 12,
        sample_batch_pts: int = 12000,
    ) -> np.ndarray:
        import torch
        from services.partfield.pf_core.pvcnn import sample_triplane_feat

        mesh = trimesh.Trimesh(vertices=vertices_norm, faces=faces, process=False)
        pc_np, _ = trimesh.sample.sample_surface(mesh, pc_samples)
        pc_t = torch.from_numpy(pc_np.astype(np.float32)).unsqueeze(0).to(self.device)

        verts_t = torch.from_numpy(vertices_norm.astype(np.float32)).to(self.device)
        faces_t = torch.from_numpy(faces.astype(np.int64)).to(self.device)
        n_f = faces_t.shape[0]

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda"), dtype=torch.float16):
                pc_feat = self.module.pvcnn(pc_t, pc_t)
                planes = self.module.triplane_transformer(pc_feat)
                _, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

            # Barycentric stratified sampling per face in bounded batches to cap VRAM
            u = torch.sqrt(torch.rand((n_f, points_per_face, 1), device=self.device, dtype=torch.float32))
            v = torch.rand((n_f, points_per_face, 1), device=self.device, dtype=torch.float32)
            w0, w1, w2 = 1.0 - u, u * (1.0 - v), u * v
            pts = (
                w0 * verts_t[faces_t[:, 0]].unsqueeze(1)
                + w1 * verts_t[faces_t[:, 1]].unsqueeze(1)
                + w2 * verts_t[faces_t[:, 2]].unsqueeze(1)
            ).reshape(1, -1, 3)

            chunk = max(points_per_face, (sample_batch_pts // points_per_face) * points_per_face)
            part_planes_f32 = part_planes.float()
            out_chunks = []
            for start in range(0, pts.shape[1], chunk):
                sub = pts[:, start : start + chunk, :]
                feat = sample_triplane_feat(part_planes_f32, sub)
                feat = feat.reshape(1, -1, points_per_face, feat.shape[-1]).mean(dim=-2)
                out_chunks.append(feat.squeeze(0).cpu())

        return torch.cat(out_chunks, dim=0).numpy()


class PartFieldEngine:
    def __init__(self, config: PartFieldConfig | None = None):
        self.config = config or PartFieldConfig()
        self._model: PartFieldInferenceModel | None = None
        self.ckpt_path: str | None = None
        self.last_load_seconds: float = 0.0

    def _resolve_ckpt(self) -> str:
        if self.ckpt_path and os.path.isfile(self.ckpt_path):
            return self.ckpt_path
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=self.config.repo_id, filename=self.config.ckpt_filename)
        self.ckpt_path = path
        return path

    def _get_model(self) -> PartFieldInferenceModel:
        import torch

        if self._model is not None:
            return self._model
        t0 = time.perf_counter()
        dev = self.config.device if (self.config.device != "cuda" or torch.cuda.is_available()) else "cpu"
        ckpt = self._resolve_ckpt()
        self._model = PartFieldInferenceModel.from_checkpoint(ckpt, device=dev)
        self.last_load_seconds = round(time.perf_counter() - t0, 2)
        return self._model

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "partfield",
            "distilled_core": True,
            "model": self.config.repo_id,
            "ckpt": self.config.ckpt_filename,
            "loaded": self._model is not None,
            "vram_mode": self.config.vram_mode,
            "last_load_seconds": self.last_load_seconds,
            "gpu": query_nvidia_smi(),
        }

    def unload(self) -> dict[str, Any]:
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return {"ok": True, "status": "unloaded", "gpu": query_nvidia_smi()}

    def segment_glb(
        self,
        glb_bytes: bytes,
        num_parts: int | None = None,
    ) -> dict[str, Any]:
        from services.partfield.pf_core.clustering import cluster_part_features

        t0 = time.perf_counter()
        loaded = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", process=False)
        if isinstance(loaded, trimesh.Scene):
            geoms = []
            for n in loaded.graph.nodes_geometry:
                t, g = loaded.graph[n]
                gm = loaded.geometry[g].copy()
                gm.apply_transform(t)
                geoms.append(gm)
            mesh = trimesh.util.concatenate(geoms)
        else:
            mesh = loaded.copy()

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) == 0 or len(verts) == 0:
            raise ValueError("GLB mesh has no triangles")

        bmin, bmax = verts.min(axis=0), verts.max(axis=0)
        span = max(float(np.max(bmax - bmin)), 1e-6)
        verts_norm = (verts - (bmin + bmax) * 0.5) * (1.8 / span)

        model = self._get_model()
        feats = model.extract_face_features(
            vertices_norm=verts_norm,
            faces=faces,
            pc_samples=self.config.pc_samples,
            points_per_face=self.config.points_per_face,
            sample_batch_pts=self.config.sample_batch_pts,
        )

        target_parts = int(num_parts) if num_parts is not None else self.config.default_num_parts
        face_labels = cluster_part_features(feats, verts_norm, faces, num_parts=target_parts)
        centers_norm = verts_norm[faces].mean(axis=1)

        if self.config.unload_after_request or self.config.vram_mode == "stage_offload":
            self.unload()

        elapsed = round(time.perf_counter() - t0, 2)
        return {
            "ok": True,
            "engine": "partfield-distilled",
            "elapsed_s": elapsed,
            "vertices": int(len(verts)),
            "faces": int(len(faces)),
            "part_count": int(len(np.unique(face_labels))),
            "face_labels": face_labels.astype(int).tolist(),
            "centers_norm": np.round(centers_norm, 4).tolist(),
        }
