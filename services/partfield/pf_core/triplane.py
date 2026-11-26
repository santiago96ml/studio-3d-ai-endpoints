"""TriplaneTransformer module preserving exact state_dict parity with model_objaverse.ckpt."""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Self-attention + MLP transformer block matching PF-LRM / PartField checkpoints."""

    def __init__(
        self,
        inner_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        attn_drop: float = 0.0,
        attn_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=attn_bias,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        before_sa = self.norm1(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        inner_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [BasicBlock(inner_dim=inner_dim, num_heads=num_heads, eps=eps) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(inner_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TriplaneTransformer(nn.Module):
    """Multi-plane transformer mapping PVCNN triplanes (128x128) to part feature planes (512ch)."""

    def __init__(
        self,
        input_dim: int = 256,
        transformer_dim: int = 1024,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        triplane_low_res: int = 32,
        triplane_high_res: int = 128,
        triplane_dim: int = 512,
    ):
        super().__init__()
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        self.pos_embed = nn.Parameter(
            torch.randn(1, 3 * (triplane_low_res**2), transformer_dim) * (1.0 / transformer_dim) ** 0.5
        )
        self.transformer = TransformerDecoder(
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
        )
        self.downsampler = nn.Sequential(
            nn.Conv2d(input_dim, transformer_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.upsampler = nn.ConvTranspose2d(transformer_dim, triplane_dim, kernel_size=4, stride=4, padding=0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, triplane_dim),
            nn.ReLU(),
            nn.Linear(triplane_dim, triplane_dim),
        )

    def forward_transformer(self, triplanes: torch.Tensor) -> torch.Tensor:
        n = triplanes.shape[0]
        tokens = torch.einsum("nidhw->nihwd", triplanes).reshape(n, self.pos_embed.shape[1], -1)
        x = self.pos_embed.repeat(n, 1, 1) + tokens
        return self.transformer(x)

    def reshape_downsample(self, triplanes: torch.Tensor) -> torch.Tensor:
        n = triplanes.shape[0]
        h = w = self.triplane_high_res
        x = triplanes.view(n, 3, -1, h, w)
        x = torch.einsum("nidhw->indhw", x).contiguous().view(3 * n, -1, h, w)
        x = self.downsampler(x)
        x = x.view(3, n, *x.shape[-3:])
        return torch.einsum("indhw->nidhw", x).contiguous()

    def reshape_upsample(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[0]
        h = w = self.triplane_low_res
        x = tokens.view(n, 3, h, w, -1)
        x = torch.einsum("nihwd->indhw", x).contiguous().view(3 * n, -1, h, w)
        x = self.upsampler(x)
        x = x.view(3, n, *x.shape[-3:])
        return torch.einsum("indhw->nidhw", x).contiguous()

    def forward(self, triplanes: torch.Tensor) -> torch.Tensor:
        downsampled = self.reshape_downsample(triplanes)
        tokens = self.forward_transformer(downsampled)
        residual = self.reshape_upsample(tokens)

        tri = triplanes.permute(0, 1, 3, 4, 2).contiguous()
        tri = self.mlp(tri)
        tri = tri.permute(0, 1, 4, 2, 3).contiguous()
        return tri + residual
