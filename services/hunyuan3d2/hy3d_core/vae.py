# Copyright (C) 2024 Tencent / THL A29 Limited.
# Licensed under the TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT (see ../LICENSE-HUNYUAN3D.txt).
# Distilled decoder-only ShapeVAE + FlashVDM octree volume decoding + pure Marching Cubes (no diso/torch_cluster).

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from skimage import measure

from .utils import logger, synchronize_timer


class FourierEmbedder(nn.Module):
    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        temp = 1 if (include_input or num_freqs == 0) else 0
        self.out_dim = input_dim * (num_freqs * 2 + temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs <= 0:
            return x
        # Compute trig frequencies in fp32 to prevent angular quantization loss at 2^7 * pi in fp16
        embed = x[..., None].contiguous().float() * self.frequencies.float()
        embed = embed.view(*x.shape[:-1], -1)
        if self.include_input:
            out = torch.cat((x.float(), embed.sin(), embed.cos()), dim=-1)
        else:
            out = torch.cat((embed.sin(), embed.cos()), dim=-1)
        return out.to(dtype=x.dtype)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            rnd.div_(keep_prob)
        return x * rnd


class MLP(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        expand_ratio: int = 4,
        output_width: Optional[int] = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class CrossAttentionProcessor:
    def __call__(self, attn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v)


class FlashVDMCrossAttentionProcessor:
    def __init__(self, topk=None):
        self.topk = topk

    def __call__(self, attn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if k.shape[-2] == 3072:
            topk = 1024
        elif k.shape[-2] == 512:
            topk = 256
        else:
            topk = max(1, k.shape[-2] // 3)

        if self.topk is True:
            q1 = q[:, :, ::100, :]
            sim = torch.mean(q1 @ k.transpose(-1, -2), -2)
            topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1).expand(-1, -1, -1, v.shape[-1])
            out = F.scaled_dot_product_attention(q, torch.gather(k, dim=-2, index=topk_ind), torch.gather(v, dim=-2, index=topk_ind))
        elif self.topk is False or self.topk is None:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            idx, counts = self.topk
            start = 0
            outs = []
            for _, count in zip(idx, counts):
                end = start + count
                q_chunk = q[:, :, start:end, :]
                k0, v0 = self.select_topkv(q_chunk, k, v, topk)
                outs.append(F.scaled_dot_product_attention(q_chunk, k0, v0))
                start = end
            out = torch.cat(outs, dim=-2)
        self.topk = False
        return out

    def select_topkv(self, q_chunk: torch.Tensor, k: torch.Tensor, v: torch.Tensor, topk: int):
        q1 = q_chunk[:, :, ::50, :]
        sim = torch.mean(q1 @ k.transpose(-1, -2), -2)
        topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1).expand(-1, -1, -1, v.shape[-1])
        return torch.gather(k, dim=-2, index=topk_ind), torch.gather(v, dim=-2, index=topk_ind)


class FlashVDMTopMCrossAttentionProcessor(FlashVDMCrossAttentionProcessor):
    def select_topkv(self, q_chunk: torch.Tensor, k: torch.Tensor, v: torch.Tensor, topk: int):
        q1 = q_chunk[:, :, ::30, :]
        sim = torch.mean((q1 @ k.transpose(-1, -2)).softmax(-1), 1)
        activated_token = torch.where(sim > 1e-6)[2]
        index = torch.unique(activated_token).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(-1, v.shape[1], -1, v.shape[-1])
        return torch.gather(k, dim=-2, index=index), torch.gather(v, dim=-2, index=index)


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, heads: int, width: Optional[int] = None, qk_norm: bool = False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_processor = CrossAttentionProcessor()

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = self.q_norm(q.view(bs, n_ctx, self.heads, -1))
        k, v = torch.split(kv.view(bs, n_data, self.heads, -1), attn_ch, dim=-1)
        k = self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, "b n h d -> b h n d"), (q, k, v))
        out = self.attn_processor(self, q, k, v)
        return out.transpose(1, 2).reshape(bs, n_ctx, -1)


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        data_width: Optional[int] = None,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(heads=heads, width=width, norm_layer=norm_layer, qk_norm=qk_norm)

    def forward(self, x: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.attention(self.c_q(x), self.c_kv(data)))


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        data_width: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
    ):
        super().__init__()
        if data_width is None:
            data_width = width
        self.attn = MultiheadCrossAttention(
            width=width,
            heads=heads,
            data_width=data_width,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        return x + self.mlp(self.ln_3(x))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, width: Optional[int] = None, qk_norm: bool = False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        q, k, v = torch.split(qkv.view(bs, n_ctx, self.heads, -1), attn_ch, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, "b n h d -> b h n d"), (q, k, v))
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, width=width, norm_layer=norm_layer, qk_norm=qk_norm)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.c_proj(self.attention(self.c_qkv(x))))


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class CrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        *,
        out_channels: int,
        fourier_embedder: FourierEmbedder,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        downsample_ratio: int = 1,
        enable_ln_post: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.enable_ln_post = enable_ln_post
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.query_proj = nn.Linear(fourier_embedder.out_dim, width)
        if downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)
        if not enable_ln_post:
            qk_norm = False
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            width=width,
            mlp_expand_ratio=mlp_expand_ratio,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        if enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)

    def set_cross_attention_processor(self, processor):
        self.cross_attn_decoder.attn.attention.attn_processor = processor

    def forward(self, queries=None, query_embeddings=None, latents=None):
        if query_embeddings is None:
            query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)
        x = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            x = self.ln_post(x)
        return self.output_proj(x)


class Latent2MeshOutput:
    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


class MCSurfaceExtractor:
    def __call__(self, grid_logits: torch.Tensor, *, mc_level: float = 0.0, bounds: Union[float, list, tuple] = 1.01, close_boundaries: bool = True, **kwargs):
        if isinstance(bounds, (int, float)):
            bounds = [-float(bounds), -float(bounds), -float(bounds), float(bounds), float(bounds), float(bounds)]
        bbox_min = np.array(bounds[0:3], dtype=np.float32)
        bbox_size = np.array(bounds[3:6], dtype=np.float32) - bbox_min

        outputs = []
        for i in range(grid_logits.shape[0]):
            vol = grid_logits[i].detach().float().cpu().numpy()
            vol = np.nan_to_num(vol, nan=-10000.0)
            if close_boundaries and vol.shape[0] > 2 and vol.shape[1] > 2 and vol.shape[2] > 2:
                # Guarantee negative exterior boundary so Marching Cubes seals solids that touch the bbox
                exterior = float(mc_level) - 1.0
                vol[0, :, :] = np.minimum(vol[0, :, :], exterior)
                vol[-1, :, :] = np.minimum(vol[-1, :, :], exterior)
                vol[:, 0, :] = np.minimum(vol[:, 0, :], exterior)
                vol[:, -1, :] = np.minimum(vol[:, -1, :], exterior)
                vol[:, :, 0] = np.minimum(vol[:, :, 0], exterior)
                vol[:, :, -1] = np.minimum(vol[:, :, -1], exterior)

            vertices, faces, _, _ = measure.marching_cubes(vol, float(mc_level), method="lewiner")
            denom = np.maximum(np.array(vol.shape, dtype=np.float32) - 1.0, 1.0)
            vertices = (vertices.astype(np.float32) / denom) * bbox_size + bbox_min
            outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=np.ascontiguousarray(faces)))
        return outputs


def _dilate_binary_mask_3d(mask_3d: torch.Tensor, iterations: int) -> torch.Tensor:
    out = (mask_3d > 0).float().unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        out = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)
    return out.squeeze(0).squeeze(0)


def _extract_near_surface_volume(input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    val = input_tensor + alpha
    valid_mask = val > -9000

    def shift_axis(t: torch.Tensor, shift: int, axis: int) -> torch.Tensor:
        if axis == 0:
            padded = F.pad(t.unsqueeze(0).unsqueeze(0), (0, 0, 0, 0, 1, 1), mode="replicate").squeeze(0).squeeze(0)
            return padded[2:] if shift > 0 else padded[:-2]
        if axis == 1:
            padded = F.pad(t.unsqueeze(0).unsqueeze(0), (0, 0, 1, 1, 0, 0), mode="replicate").squeeze(0).squeeze(0)
            return padded[:, 2:, :] if shift > 0 else padded[:, :-2, :]
        padded = F.pad(t.unsqueeze(0).unsqueeze(0), (1, 1, 0, 0, 0, 0), mode="replicate").squeeze(0).squeeze(0)
        return padded[:, :, 2:] if shift > 0 else padded[:, :, :-2]

    neighbors = [
        shift_axis(val, 1, 0),
        shift_axis(val, -1, 0),
        shift_axis(val, 1, 1),
        shift_axis(val, -1, 1),
        shift_axis(val, 1, 2),
        shift_axis(val, -1, 2),
    ]
    sign = torch.sign(val.float())
    neighbors_sign = torch.stack([torch.sign(torch.where(n > -9000, n, val).float()) for n in neighbors], dim=0)
    same_sign = torch.all(neighbors_sign == sign, dim=0)
    return ((~same_sign) & valid_mask).to(torch.int32)


def _generate_dense_grid_points(bbox_min: np.ndarray, bbox_max: np.ndarray, octree_resolution: int):
    x = np.linspace(bbox_min[0], bbox_max[0], int(octree_resolution) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(octree_resolution) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(octree_resolution) + 1, dtype=np.float32)
    xs, ys, zs = np.meshgrid(x, y, z, indexing="ij")
    xyz = np.stack((xs, ys, zs), axis=-1)
    return xyz, [int(octree_resolution) + 1] * 3


class VanillaVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        device, dtype = latents.device, latents.dtype
        batch_size = latents.shape[0]
        if isinstance(bounds, (int, float)):
            bounds = [-float(bounds), -float(bounds), -float(bounds), float(bounds), float(bounds), float(bounds)]
        bbox_min, bbox_max = np.array(bounds[0:3], dtype=np.float32), np.array(bounds[3:6], dtype=np.float32)
        xyz_samples, grid_size = _generate_dense_grid_points(bbox_min, bbox_max, octree_resolution)
        xyz_flat = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        batch_logits = []
        for start in range(0, xyz_flat.shape[0], num_chunks):
            chunk_queries = repeat(xyz_flat[start : start + num_chunks], "p c -> b p c", b=batch_size)
            batch_logits.append(geo_decoder(queries=chunk_queries, latents=latents))

        return torch.cat(batch_logits, dim=1).view((batch_size, *grid_size)).float()


class FlashVDMVolumeDecoding:
    def __init__(self, topk_mode: str = "mean"):
        self.processor = FlashVDMCrossAttentionProcessor() if topk_mode == "mean" else FlashVDMTopMCrossAttentionProcessor()

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: CrossAttentionDecoder,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = 384,
        min_resolution: int = 63,
        mini_grid_num: int = 4,
        **kwargs,
    ) -> torch.Tensor:
        processor = self.processor
        geo_decoder.set_cross_attention_processor(processor)
        device, dtype = latents.device, latents.dtype

        resolutions = []
        res_cursor = int(octree_resolution)
        if res_cursor < min_resolution:
            resolutions.append(res_cursor)
        while res_cursor >= min_resolution:
            resolutions.append(res_cursor)
            res_cursor = res_cursor // 2
        resolutions.reverse()
        resolutions[0] = round(resolutions[0] / mini_grid_num) * mini_grid_num - 1
        for i in range(len(resolutions) - 1):
            resolutions[i + 1] = resolutions[0] * (2 ** (i + 1))

        if isinstance(bounds, (int, float)):
            bounds = [-float(bounds), -float(bounds), -float(bounds), float(bounds), float(bounds), float(bounds)]
        bbox_min = np.array(bounds[0:3], dtype=np.float32)
        bbox_max = np.array(bounds[3:6], dtype=np.float32)
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size = _generate_dense_grid_points(bbox_min, bbox_max, resolutions[0])
        xyz_t = torch.from_numpy(xyz_samples).to(device, dtype=dtype)
        batch_size = latents.shape[0]
        mini_grid_size = xyz_t.shape[0] // mini_grid_num
        xyz_t = xyz_t.view(
            mini_grid_num,
            mini_grid_size,
            mini_grid_num,
            mini_grid_size,
            mini_grid_num,
            mini_grid_size,
            3,
        ).permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, mini_grid_size ** 3, 3)

        num_batches = max(num_chunks // xyz_t.shape[1], 1)
        batch_logits = []
        for start in range(0, xyz_t.shape[0], num_batches):
            queries = xyz_t[start : start + num_batches]
            batch_latents = repeat(latents.squeeze(0), "p c -> b p c", b=batch_size)
            processor.topk = True
            batch_logits.append(geo_decoder(queries=queries, latents=batch_latents))

        grid_logits = (
            torch.cat(batch_logits, dim=0)
            .reshape(mini_grid_num, mini_grid_num, mini_grid_num, mini_grid_size, mini_grid_size, mini_grid_size)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view((batch_size, grid_size[0], grid_size[1], grid_size[2]))
        )

        bbox_min_t = torch.tensor(bbox_min, dtype=torch.float32, device=device)
        for octree_depth_now in resolutions[1:]:
            grid_dim = octree_depth_now + 1
            step_res = torch.tensor(bbox_size / octree_depth_now, dtype=torch.float32, device=device)
            next_index = torch.zeros((grid_dim, grid_dim, grid_dim), dtype=torch.float32, device=device)
            next_logits = torch.full((grid_dim, grid_dim, grid_dim), -10000.0, dtype=dtype, device=device)

            squeezed = grid_logits.squeeze(0)
            curr_points = _extract_near_surface_volume(squeezed, mc_level) + (squeezed.abs() < 0.95).to(torch.int32)
            expand_num = 0 if octree_depth_now == resolutions[-1] else 1
            if expand_num > 0:
                curr_points = _dilate_binary_mask_3d(curr_points, expand_num)

            cx, cy, cz = torch.where(curr_points > 0)
            next_index[cx * 2, cy * 2, cz * 2] = 1.0
            next_index = _dilate_binary_mask_3d(next_index, 2 - expand_num)
            nidx = torch.where(next_index > 0)

            next_points = torch.stack(nidx, dim=1).float() * step_res + bbox_min_t
            query_grid_num = 6
            min_val = next_points.min(dim=0).values
            max_val = next_points.max(dim=0).values
            span = torch.clamp(max_val - min_val, min=1e-5)
            vol_idx = torch.floor((next_points - min_val) / span * (query_grid_num - 0.001)).long()
            flat_idx = vol_idx[..., 0] * (query_grid_num ** 2) + vol_idx[..., 1] * query_grid_num + vol_idx[..., 2]
            sorted_idx = flat_idx.sort()
            next_points = next_points[sorted_idx.indices].unsqueeze(0).contiguous()
            unique_vals, unique_counts = torch.unique(sorted_idx.values, return_counts=True)

            out_flat = torch.zeros((next_points.shape[1],), dtype=dtype, device=device)
            input_grid = [[], []]
            logits_parts = []
            start_num, sum_num = 0, 0
            for grid_index, count in zip(unique_vals.cpu().tolist(), unique_counts.cpu().tolist()):
                if sum_num + count < num_chunks or sum_num == 0:
                    sum_num += count
                    input_grid[0].append(grid_index)
                    input_grid[1].append(count)
                else:
                    processor.topk = input_grid
                    logits_parts.append(geo_decoder(queries=next_points[:, start_num : start_num + sum_num], latents=latents))
                    start_num += sum_num
                    input_grid = [[grid_index], [count]]
                    sum_num = count
            if sum_num > 0:
                processor.topk = input_grid
                logits_parts.append(geo_decoder(queries=next_points[:, start_num : start_num + sum_num], latents=latents))

            out_flat[sorted_idx.indices] = torch.cat(logits_parts, dim=1).squeeze(0).squeeze(-1)
            next_logits[nidx] = out_flat
            grid_logits = next_logits.unsqueeze(0)

        grid_logits[grid_logits == -10000.0] = float("nan")
        return grid_logits


class ShapeVAE(nn.Module):
    """Decoder-only ShapeVAE: omits PointCrossAttentionEncoder & torch_cluster in inference_only mode."""

    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
        inference_only: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.inference_only = inference_only
        self.scale_factor = float(scale_factor)
        self.latent_shape = (int(num_latents), int(embed_dim))
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.post_kl = nn.Linear(embed_dim, width)
        self.transformer = Transformer(
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )
        self.geo_decoder = CrossAttentionDecoder(
            out_channels=1,
            fourier_embedder=self.fourier_embedder,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=geo_decoder_ln_post,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.volume_decoder = FlashVDMVolumeDecoding("mean")
        self.surface_extractor = MCSurfaceExtractor()

    def enable_flashvdm_decoder(self, enabled: bool = True, topk_mode: str = "mean"):
        self.volume_decoder = FlashVDMVolumeDecoding(topk_mode) if enabled else VanillaVolumeDecoder()
        self.surface_extractor = MCSurfaceExtractor()

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.post_kl(latents))

    @synchronize_timer("ShapeVAE latents2mesh")
    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)
        return self.surface_extractor(grid_logits, **kwargs)
