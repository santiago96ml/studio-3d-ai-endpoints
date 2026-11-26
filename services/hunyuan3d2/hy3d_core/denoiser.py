# Copyright (C) 2024 Tencent / THL A29 Limited.
# Licensed under the Apache License Version 2.0 and TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT.
# Distilled Hunyuan3DDiT denoiser using native PyTorch SDPA / RMSNorm kernels.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


def _rms_norm(x: Tensor, scale: Tensor, eps: float = 1e-6) -> Tensor:
    x_dtype = x.dtype
    if hasattr(F, "rms_norm"):
        out = F.rms_norm(x.float(), (x.shape[-1],), eps=eps).to(dtype=x_dtype)
    else:
        xf = x.float()
        out = (xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)).to(dtype=x_dtype)
    return out * scale


def attention(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    x = F.scaled_dot_product_attention(q, k, v)
    return rearrange(x, "B H L D -> B L (H D)")


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class GELU(nn.Module):
    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x, approximate=self.approximate)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return _rms_norm(x, self.scale)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Optional[Tensor] = None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        return self.proj(x)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_scale: Optional[float] = None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Optional[Tensor] = None) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)


class Hunyuan3DDiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        context_in_dim: int = 1536,
        hidden_size: int = 1024,
        mlp_ratio: float = 4.0,
        num_heads: int = 16,
        depth: int = 16,
        depth_single_blocks: int = 32,
        axes_dim: List[int] = [64],
        theta: int = 10_000,
        qkv_bias: bool = True,
        time_factor: float = 1000.0,
        guidance_embed: bool = False,
        **kwargs,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.in_channels = in_channels
        self.context_in_dim = context_in_dim
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks
        self.axes_dim = axes_dim
        self.theta = theta
        self.qkv_bias = qkv_bias
        self.time_factor = time_factor
        self.out_channels = in_channels
        self.guidance_embed = guidance_embed

        self.latent_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)
        self.cond_in = nn.Linear(context_in_dim, hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size) if guidance_embed else nn.Identity()

        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias) for _ in range(depth)]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_single_blocks)]
        )
        self.final_layer = LastLayer(hidden_size, 1, self.out_channels)

    def forward(self, x: Tensor, t: Tensor, contexts: dict, **kwargs) -> Tensor:
        cond = contexts["main"]
        latent = self.latent_in(x)
        # Keep positional `self.time_factor` as 3rd arg (`max_period=1000`): matches upstream training/inference
        vec = self.time_in(timestep_embedding(t, 256, self.time_factor).to(dtype=latent.dtype))
        if self.guidance_embed:
            guidance = kwargs.get("guidance")
            if guidance is None:
                raise ValueError("Missing guidance strength for guidance-distilled Hunyuan3DDiT")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256, self.time_factor).to(dtype=latent.dtype))

        cond = self.cond_in(cond)
        for block in self.double_blocks:
            latent, cond = block(img=latent, txt=cond, vec=vec, pe=None)

        latent = torch.cat((cond, latent), dim=1)
        for block in self.single_blocks:
            latent = block(latent, vec=vec, pe=None)

        latent = latent[:, cond.shape[1] :, ...]
        return self.final_layer(latent, vec)
