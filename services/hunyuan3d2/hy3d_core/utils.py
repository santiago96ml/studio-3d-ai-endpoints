# Copyright (C) 2024 Tencent / THL A29 Limited.
# Licensed under the TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT (see ../LICENSE-HUNYUAN3D.txt).
# Distilled and memory-optimized for studio-3d-ai-endpoints.

import logging
import os
from functools import wraps
from typing import Iterable, Optional

import torch


def get_logger(name: str = "hy3d_core") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger("hy3d_core")


class synchronize_timer:
    """Optional CUDA synchronized timer enabled only when HY3DGEN_DEBUG=1."""

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.time = 0.0
        self._enabled = os.environ.get("HY3DGEN_DEBUG", "0") == "1" and torch.cuda.is_available()

    def __enter__(self):
        if self._enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time
        return None

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self._enabled:
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                logger.info(f"{self.name} takes {self.time:.1f} ms")

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


def smart_load_model(
    model_path: str,
    subfolder: str,
    use_safetensors: bool = True,
    variant: Optional[str] = "fp16",
) -> tuple[str, str]:
    """Download/resolve ONLY config.yaml + target safetensors file for a subfolder."""
    extension = "ckpt" if not use_safetensors else "safetensors"
    suffix = "" if not variant else f".{variant}"
    ckpt_name = f"model{suffix}.{extension}"

    base_dir = os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen")
    local_dir = os.path.expanduser(os.path.join(base_dir, model_path, subfolder))
    local_config = os.path.join(local_dir, "config.yaml")
    local_ckpt = os.path.join(local_dir, ckpt_name)

    if os.path.exists(local_config) and os.path.exists(local_ckpt):
        logger.info(f"Using cached Hunyuan3D subfolder: {local_dir}")
        return local_config, local_ckpt

    # Prefer direct repo path if model_path is a local folder already
    if os.path.isdir(os.path.join(model_path, subfolder)):
        sub_dir = os.path.join(model_path, subfolder)
        cfg = os.path.join(sub_dir, "config.yaml")
        ckpt = os.path.join(sub_dir, ckpt_name)
        if os.path.exists(cfg) and os.path.exists(ckpt):
            return cfg, ckpt

    logger.info(f"Downloading selective files from HF: {model_path}/{subfolder} ({ckpt_name})")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=model_path,
        allow_patterns=[f"{subfolder}/config.yaml", f"{subfolder}/{ckpt_name}"],
    )
    resolved_dir = os.path.join(path, subfolder)
    config_path = os.path.join(resolved_dir, "config.yaml")
    ckpt_path = os.path.join(resolved_dir, ckpt_name)
    if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Expected {config_path} and {ckpt_path} after HF download")
    return config_path, ckpt_path


def load_selective_safetensors(
    ckpt_path: str,
    include_prefixes: Optional[Iterable[str]] = None,
    exclude_prefixes: Optional[Iterable[str]] = None,
    target_dtype: Optional[torch.dtype] = None,
) -> dict[str, torch.Tensor]:
    """Load only matching keys from a .safetensors checkpoint to minimize peak CPU RAM."""
    import safetensors

    inc = tuple(include_prefixes) if include_prefixes else None
    exc = tuple(exclude_prefixes) if exclude_prefixes else None

    state: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if inc is not None and not key.startswith(inc):
                continue
            if exc is not None and key.startswith(exc):
                continue
            tensor = f.get_tensor(key)
            if target_dtype is not None and tensor.is_floating_point() and tensor.dtype != target_dtype:
                tensor = tensor.to(dtype=target_dtype)
            state[key] = tensor
    return state
