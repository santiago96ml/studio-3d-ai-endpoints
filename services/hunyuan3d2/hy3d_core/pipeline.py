# Copyright (C) 2024 Tencent / THL A29 Limited.
# Licensed under the TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT (see ../LICENSE-HUNYUAN3D.txt).
# Distilled pipeline with selective safetensors loading, zero-copy meta init & staged VRAM offload.

import gc
import importlib
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image

from .conditioner import CLIPImageEncoder, DinoImageEncoder, DualImageEncoder, SingleImageEncoder
from .denoiser import Hunyuan3DDiT
from .preprocessors import ImageProcessorV2
from .schedulers import ConsistencyFlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler
from .utils import load_selective_safetensors, logger, smart_load_model, synchronize_timer
from .vae import ShapeVAE


HUNYUAN_VARIANTS = {
    "turbo": {
        "dit_repo": "tencent/Hunyuan3D-2",
        "dit_subfolder": "hunyuan3d-dit-v2-0-turbo",
        "vae_repo": "tencent/Hunyuan3D-2",
        "vae_subfolder": "hunyuan3d-vae-v2-0-turbo",
    },
    "mini": {
        "dit_repo": "tencent/Hunyuan3D-2mini",
        "dit_subfolder": "hunyuan3d-dit-v2-mini-turbo",
        "vae_repo": "tencent/Hunyuan3D-2mini",
        "vae_subfolder": "hunyuan3d-vae-v2-mini-turbo",
    },
}

_TARGET_MAP = {
    "hy3dgen.shapegen.models.Hunyuan3DDiT": Hunyuan3DDiT,
    "hy3dgen.shapegen.models.ShapeVAE": ShapeVAE,
    "hy3dgen.shapegen.models.SingleImageEncoder": SingleImageEncoder,
    "hy3dgen.shapegen.models.DualImageEncoder": DualImageEncoder,
    "hy3dgen.shapegen.models.DinoImageEncoder": DinoImageEncoder,
    "hy3dgen.shapegen.models.CLIPImageEncoder": CLIPImageEncoder,
    "hy3dgen.shapegen.schedulers.FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    "hy3dgen.shapegen.schedulers.ConsistencyFlowMatchEulerDiscreteScheduler": ConsistencyFlowMatchEulerDiscreteScheduler,
    "hy3dgen.shapegen.preprocessors.ImageProcessorV2": ImageProcessorV2,
}


def _resolve_target(target: str):
    norm = target.replace("hy3dshape", "hy3dgen.shapegen")
    if norm in _TARGET_MAP:
        return _TARGET_MAP[norm]
    mod_name, cls_name = norm.rsplit(".", 1)
    return getattr(importlib.import_module(mod_name), cls_name)


def instantiate_from_config(config: dict, **overrides):
    if "target" not in config:
        raise KeyError("Expected key `target` in Hunyuan3D config")
    cls = _resolve_target(config["target"])
    params = dict(config.get("params") or {})
    params.update(overrides)
    return cls(**params)


def _strip_prefix_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in state.items() if k.startswith(prefix)}


def _adapt_conditioner_state_dict(
    cond_sd: dict[str, torch.Tensor], conditioner: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """Support both transformers v4 and v5 Dinov2 naming (q_proj/gate_proj/up_proj)."""
    expected_keys = set(conditioner.state_dict().keys())
    if all(k in expected_keys for k in cond_sd.keys()):
        return cond_sd

    adapted: dict[str, torch.Tensor] = {}
    for k, v in cond_sd.items():
        new_k = (
            k.replace(".attention.attention.query.", ".attention.q_proj.")
            .replace(".attention.attention.key.", ".attention.k_proj.")
            .replace(".attention.attention.value.", ".attention.v_proj.")
            .replace(".attention.output.dense.", ".attention.o_proj.")
            .replace(".mlp.weights_out.", ".mlp.down_proj.")
        )
        if ".mlp.weights_in." in new_k:
            gate_k = new_k.replace(".mlp.weights_in.", ".mlp.gate_proj.")
            up_k = new_k.replace(".mlp.weights_in.", ".mlp.up_proj.")
            if gate_k in expected_keys and up_k in expected_keys:
                g_v, u_v = torch.chunk(v, 2, dim=0)
                adapted[gate_k] = g_v.contiguous()
                adapted[up_k] = u_v.contiguous()
                continue
        adapted[new_k] = v
    return adapted


class DistilledHunyuan3DPipeline:
    """Memory-distilled FlashVDM Flow-Matching 3D pipeline (shape only)."""

    def __init__(
        self,
        vae: ShapeVAE,
        model: Hunyuan3DDiT,
        scheduler,
        conditioner: torch.nn.Module,
        image_processor: ImageProcessorV2,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        vram_mode: str = "resident",
    ):
        self.vae = vae.eval()
        self.model = model.eval()
        self.scheduler = scheduler
        self.conditioner = conditioner.eval()
        self.image_processor = image_processor
        self.device = torch.device(device)
        self.dtype = dtype
        self.vram_mode = vram_mode  # "resident" or "stage_offload"

        if self.vram_mode == "resident":
            self.to(self.device, self.dtype)
        else:
            self.to("cpu", self.dtype)

    @classmethod
    @synchronize_timer("DistilledHunyuan3DPipeline load")
    def from_variant(
        cls,
        variant: str = "turbo",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        vram_mode: str = "auto",
    ) -> "DistilledHunyuan3DPipeline":
        key = variant.strip().lower()
        if key not in HUNYUAN_VARIANTS:
            raise ValueError(f"Unsupported Hunyuan3D variant '{variant}', expected one of {list(HUNYUAN_VARIANTS)}")

        spec = HUNYUAN_VARIANTS[key]
        dit_cfg_path, dit_ckpt_path = smart_load_model(
            spec["dit_repo"],
            subfolder=spec["dit_subfolder"],
            use_safetensors=True,
            variant="fp16",
        )
        vae_cfg_path, vae_ckpt_path = smart_load_model(
            spec["vae_repo"],
            subfolder=spec["vae_subfolder"],
            use_safetensors=True,
            variant="fp16",
        )

        with open(dit_cfg_path, "r", encoding="utf-8") as f:
            dit_cfg = yaml.safe_load(f)
        with open(vae_cfg_path, "r", encoding="utf-8") as f:
            vae_cfg = yaml.safe_load(f)

        # Auto-select stage_offload for <=16 GB GPUs (e.g. T4) or resident for L4/A100
        resolved_mode = vram_mode
        if resolved_mode == "auto":
            if torch.device(device).type == "cuda" and torch.cuda.is_available():
                total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                resolved_mode = "stage_offload" if total_mb < 18000 else "resident"
            else:
                resolved_mode = "resident"

        # 1. Load ONLY model.* + conditioner.* (skip redundant base VAE in dit checkpoint)
        logger.info(f"Selective load DiT + conditioner from {dit_ckpt_path} (dtype={dtype})")
        dit_state = load_selective_safetensors(
            dit_ckpt_path,
            include_prefixes=("model.", "conditioner."),
            target_dtype=dtype,
        )
        model_sd = _strip_prefix_state(dit_state, "model.")
        cond_sd = _strip_prefix_state(dit_state, "conditioner.")
        del dit_state

        # Instantiate DiT + DINOv2 on meta device to avoid allocating ~9 GB fp32 RAM before assign=True
        with torch.device("meta"):
            model = instantiate_from_config(dit_cfg["model"])
            conditioner = instantiate_from_config(dit_cfg["conditioner"])

        model.load_state_dict(model_sd, strict=True, assign=True)
        cond_sd = _adapt_conditioner_state_dict(cond_sd, conditioner)
        conditioner.load_state_dict(cond_sd, strict=True, assign=True)
        del model_sd, cond_sd

        # 2. Load Turbo FlashVDM ShapeVAE excluding encoder.* weights (~250 MB saved)
        logger.info(f"Selective load Turbo ShapeVAE from {vae_ckpt_path} (inference_only=True)")
        vae_state = load_selective_safetensors(
            vae_ckpt_path,
            exclude_prefixes=("encoder.", "pre_kl."),
            target_dtype=dtype,
        )
        prev_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            vae = instantiate_from_config(vae_cfg, inference_only=True)
        finally:
            torch.set_default_dtype(prev_dtype)
        vae.load_state_dict(vae_state, strict=False, assign=True)
        del vae_state

        vae.enable_flashvdm_decoder(enabled=True, topk_mode="mean")
        image_processor = instantiate_from_config(dit_cfg["image_processor"])
        scheduler = instantiate_from_config(dit_cfg["scheduler"])

        gc.collect()
        return cls(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor,
            device=device,
            dtype=dtype,
            vram_mode=resolved_mode,
        )

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            self.conditioner.to(dtype=dtype)
        if device is not None:
            target = torch.device(device)
            self.vae.to(target)
            self.model.to(target)
            self.conditioner.to(target)

    def offload_to_cpu(self):
        self.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _stage_to_device(self, module: torch.nn.Module):
        if self.vram_mode == "stage_offload":
            module.to(self.device, dtype=self.dtype)

    def _stage_to_cpu(self, module: torch.nn.Module):
        if self.vram_mode == "stage_offload":
            module.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
        num_inference_steps: int = 5,
        guidance_scale: float = 5.0,
        seed: int = 42,
        box_v: float = 1.01,
        octree_resolution: int = 384,
        mc_level: float = 0.0,
        num_chunks: int = 12000,
    ) -> trimesh.Trimesh:
        device = self.device
        dtype = self.dtype
        use_autocast = device.type == "cuda"
        amp_ctx = torch.amp.autocast("cuda", dtype=dtype) if use_autocast else nullcontext()

        cond_inputs = self.image_processor(image)
        img_tensor = cond_inputs["image"]
        batch_size = img_tensor.shape[0]

        with amp_ctx:
            # Stage 1: Image Conditioning (DINOv2-giant)
            self._stage_to_device(self.conditioner)
            cond = self.conditioner(image=img_tensor)
            do_cfg = guidance_scale >= 0 and not getattr(self.model, "guidance_embed", False)
            if do_cfg:
                uncond = self.conditioner.unconditional_embedding(batch_size)
                cond = {k: torch.cat([cond[k], uncond[k]], dim=0).to(dtype) for k in cond}
            else:
                cond = {k: v.to(dtype) for k, v in cond.items()}
            self._stage_to_cpu(self.conditioner)

            # Stage 2: 3D Flow Matching DiT (5 steps with Turbo distilled model)
            sigmas = np.linspace(0.0, 1.0, int(num_inference_steps))
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
            timesteps = self.scheduler.timesteps

            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            latents = torch.randn((batch_size, *self.vae.latent_shape), generator=gen, dtype=dtype).to(device)
            latents = latents * getattr(self.scheduler, "init_noise_sigma", 1.0)

            guidance = None
            if getattr(self.model, "guidance_embed", False):
                guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)

            self._stage_to_device(self.model)
            for t in timesteps:
                latent_input = torch.cat([latents] * 2) if do_cfg else latents
                timestep = t.expand(latent_input.shape[0]).to(dtype) / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_input, timestep, cond, guidance=guidance)
                if do_cfg:
                    pred_c, pred_u = noise_pred.chunk(2)
                    noise_pred = pred_u + guidance_scale * (pred_c - pred_u)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            self._stage_to_cpu(self.model)

            # Stage 3: Turbo FlashVDM ShapeVAE + Marching Cubes
            self._stage_to_device(self.vae)
            latents = (1.0 / self.vae.scale_factor) * latents
            decoded = self.vae(latents)
            mesh_out = self.vae.latents2mesh(
                decoded,
                bounds=box_v,
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
            )[0]
            self._stage_to_cpu(self.vae)

        if mesh_out is None:
            raise RuntimeError("Hunyuan3D-2 FlashVDM marching cubes produced an empty surface")

        faces = mesh_out.mesh_f[:, ::-1].copy()
        return trimesh.Trimesh(vertices=mesh_out.mesh_v, faces=faces, process=False)
