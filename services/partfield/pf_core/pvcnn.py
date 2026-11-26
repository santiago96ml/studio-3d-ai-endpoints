"""Pure-PyTorch PVCNN + 3D-Aware Triplane UNet matching nv-tlabs/PartField checkpoints.

Removes torch-scatter & C++/CUDA extensions by replacing scatter_mean with native
PyTorch `scatter_add_` + count clamping and using pure `F.grid_sample` devoxelization.
"""

from __future__ import annotations

import functools
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch replacement for `torch_scatter.scatter_mean(src, index, out=out)`."""
    idx_exp = index.expand(-1, src.size(1), -1).long()
    out.scatter_add_(2, idx_exp, src)
    cnt = src.new_zeros((src.size(0), 1, out.size(2)))
    cnt.scatter_add_(2, index.long(), src.new_ones((src.size(0), 1, src.size(2))))
    return out / cnt.clamp(min=1.0)


def _trilinear_devoxelize(c: torch.Tensor, coords: torch.Tensor, r: int) -> torch.Tensor:
    grid = (coords * 2.0 + 1.0) / float(r) - 1.0
    grid = grid.permute(0, 2, 1).reshape(c.shape[0], 1, 1, -1, 3)
    f = F.grid_sample(input=c, grid=grid, padding_mode="border", align_corners=False)
    return f.squeeze(dim=2).squeeze(dim=2)


def _my_voxelization(features: torch.Tensor, coords: torch.Tensor, resolution: int) -> torch.Tensor:
    b, c, n = features.shape
    feat_f = features.float()
    result = torch.zeros(b, c + 1, resolution**3, device=features.device, dtype=torch.float32)
    r = resolution
    r2 = resolution * resolution
    indices = (coords[:, 0] * r2 + coords[:, 1] * r + coords[:, 2]).unsqueeze(1).expand(-1, c + 1, -1).long()
    ones = torch.ones((b, 1, n), device=features.device, dtype=torch.float32)
    out_feature = result.scatter_(index=indices, src=torch.cat([feat_f, ones], dim=1), dim=2, reduce="add")
    cnt = out_feature[:, -1:, :].clamp(min=1e-5)
    vox = out_feature[:, :-1, :] / cnt
    return vox.view(b, c, resolution, resolution, resolution).to(features.dtype)


class SharedMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | list[int], dim: int = 1, device: str = "cpu"):
        super().__init__()
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        bn = nn.InstanceNorm1d
        ocs = [out_channels] if not isinstance(out_channels, (list, tuple)) else list(out_channels)
        layers: list[nn.Module] = []
        ic = in_channels
        for oc in ocs:
            layers.extend([conv(ic, oc, 1, device=device), bn(oc, device=device), nn.ReLU(True)])
            ic = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        return self.layers(inputs)


class Voxelization(nn.Module):
    def __init__(self, resolution: int, normalize: bool = False, eps: float = 0.0, scale_pvcnn: bool = True):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps
        self.scale_pvcnn = scale_pvcnn

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            c_det = coords.detach()
            norm_coords = (c_det + 1.0) / 2.0 if self.scale_pvcnn else c_det
            norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
            vox_coords = torch.round(norm_coords)
        return _my_voxelization(features, vox_coords, self.r), norm_coords


class PVConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        resolution: int,
        with_se: bool = False,
        normalize: bool = False,
        eps: float = 0.0,
        scale_pvcnn: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        del with_se
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps, scale_pvcnn=scale_pvcnn)
        self.voxel_layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, device=device),
            nn.InstanceNorm3d(out_channels, eps=1e-4, device=device),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, device=device),
            nn.InstanceNorm3d(out_channels, eps=1e-4, device=device),
            nn.LeakyReLU(0.1, True),
        )
        self.point_features = SharedMLP(in_channels, out_channels, device=device)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        devox = _trilinear_devoxelize(voxel_features, voxel_coords, self.resolution)
        return devox + self.point_features(features), coords, voxel_features


class PVCNNEncoder(nn.Module):
    def __init__(self, pvcnn_feat_dim: int, device: str = "cpu", in_channels: int = 6):
        super().__init__()
        self.device = device
        self.blocks = ((pvcnn_feat_dim, 1, 32), (128, 2, 16), (256, 1, 8))
        self.append_channel = 2 if in_channels == 6 else 1
        ic = in_channels + self.append_channel
        layers: list[nn.Module] = []
        for oc, num_blocks, vres in self.blocks:
            block = functools.partial(
                PVConv, kernel_size=3, resolution=int(vres), with_se=False, normalize=False, eps=0.0, scale_pvcnn=True, device=device
            )
            for _ in range(num_blocks):
                layers.append(block(ic, int(oc)))
                ic = int(oc)
        self.encoder = nn.ModuleList(layers)

    def forward(self, input_pc: torch.Tensor):
        features = input_pc.permute(0, 2, 1) * 2.0
        coords = features[:, :3, :]
        zero_padding = torch.zeros(features.shape[0], self.append_channel, features.shape[-1], device=features.device, dtype=features.dtype)
        features = torch.cat([features, zero_padding], dim=1)
        out_features_list, voxel_feature_list = [], []
        for layer in self.encoder:
            features, _, voxel_feature = layer((features, coords))
            out_features_list.append(features)
            voxel_feature_list.append(voxel_feature)
        return voxel_feature_list, out_features_list


def _conv3x3(ic: int, oc: int, stride: int = 1, padding: int = 1, bias: bool = True, groups: int = 1):
    return nn.Conv2d(ic, oc, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)


def _conv1x1(ic: int, oc: int, groups: int = 1):
    return nn.Conv2d(ic, oc, kernel_size=1, groups=groups, stride=1)


class ConvTriplane3dAware(nn.Module):
    def __init__(self, internal_conv_f, in_channels: int, out_channels: int, order: str = "xz"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = order
        self.plane_convs = nn.ModuleList([internal_conv_f(3 * in_channels, out_channels) for _ in range(3)])

    def forward(self, triplanes_list):
        inps = list(triplanes_list)
        if self.order == "xz":
            inps[2] = einops.rearrange(inps[2], "b c x z -> b c z x")
        oplanes = [None, None, None]
        for iplane in (0, 1, 2):
            jplane, kplane = (iplane + 1) % 3, (iplane + 2) % 3
            ifeat = inps[iplane]
            jpool = einops.repeat(einops.rearrange(torch.mean(inps[jplane], dim=3, keepdim=True), "b c k 1 -> b c 1 k"), "b c 1 k -> b c j k", j=ifeat.size(2))
            kpool = einops.repeat(einops.rearrange(torch.mean(inps[kplane], dim=2, keepdim=True), "b c 1 j -> b c j 1"), "b c j 1 -> b c j k", k=ifeat.size(3))
            oplanes[iplane] = self.plane_convs[iplane](torch.cat([ifeat, jpool, kpool], dim=1))
        if self.order == "xz":
            oplanes[2] = einops.rearrange(oplanes[2], "b c z x -> b c x z")
        return oplanes


def _normalize_gn(ic: int, num_groups: int = 32):
    return nn.GroupNorm(num_groups=min(ic, num_groups), num_channels=ic, eps=1e-6, affine=True)


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = False):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.with_conv else x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = False):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock3dAware(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None):
        super().__init__()
        oc = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = oc
        self.norm1 = _normalize_gn(in_channels)
        self.conv1 = _conv3x3(in_channels, oc)
        self.norm_mid = _normalize_gn(oc)
        self.conv_3daware = ConvTriplane3dAware(lambda i, o: _conv1x1(i, o), oc, oc, order="xz")
        self.norm2 = _normalize_gn(oc)
        self.conv2 = _conv3x3(oc, oc)
        if in_channels != oc:
            self.nin_shortcut = nn.Conv2d(in_channels, oc, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(_swish(self.norm1(x)))
        h = _swish(self.norm_mid(h))
        unrolled = torch.unbind(einops.rearrange(h, "b c (tri h) w -> b c tri h w", tri=3), dim=2)
        aware = self.conv_3daware(unrolled)
        h = einops.rearrange(torch.stack(aware, dim=2), "b c tri h w -> b c (tri h) w", tri=3)
        h = self.conv2(_swish(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class DownConv3dAware(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True, with_conv: bool = False):
        super().__init__()
        self.block = ResnetBlock3dAware(in_channels, out_channels)
        self.do_downsample = downsample
        self.downsample = Downsample(out_channels, with_conv=with_conv)

    def forward(self, x: torch.Tensor):
        x = self.block(x)
        before_pool = x
        if self.do_downsample:
            x = einops.rearrange(x, "b c (tri h) w -> b (c tri) h w", tri=3)
            x = self.downsample(x)
            x = einops.rearrange(x, "b (c tri) h w -> b c (tri h) w", tri=3)
        return x, before_pool


class UpConv3dAware(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, merge_mode: str = "concat", with_conv: bool = False):
        super().__init__()
        self.merge_mode = merge_mode
        self.upsample = Upsample(in_channels, with_conv)
        in_c = in_channels + out_channels if merge_mode == "concat" else in_channels
        self.norm1 = _normalize_gn(in_c)
        self.block = ResnetBlock3dAware(in_channels=in_c, out_channels=out_channels)

    def forward(self, from_down: torch.Tensor, from_up: torch.Tensor) -> torch.Tensor:
        up = self.upsample(from_up)
        x = torch.cat((up, from_down), 1) if self.merge_mode == "concat" else (up + from_down)
        return self.block(self.norm1(x))


class UNetTriplane3dAware(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: int = 256,
        depth: int = 3,
        start_filts: int = 32,
        use_initial_conv: bool = False,
        merge_mode: str = "concat",
    ):
        super().__init__()
        self.use_initial_conv = use_initial_conv
        if use_initial_conv:
            self.conv_initial = _conv1x1(in_channels, start_filts)

        down_convs, up_convs = [], []
        outs = in_channels
        for i in range(depth):
            ins = start_filts if (i == 0 and use_initial_conv) else (in_channels if i == 0 else outs)
            outs = start_filts * (2**i)
            down_convs.append(DownConv3dAware(ins, outs, downsample=(i < depth - 1)))
        for _ in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_convs.append(UpConv3dAware(ins, outs, merge_mode=merge_mode))

        self.down_convs = nn.ModuleList(down_convs)
        self.up_convs = nn.ModuleList(up_convs)
        self.norm_out = _normalize_gn(outs)
        self.conv_final = _conv1x1(outs, out_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b tri c h w -> b c (tri h) w", tri=3)
        if self.use_initial_conv:
            x = self.conv_initial(x)
        enc_outs = []
        for mod in self.down_convs:
            x, bp = mod(x)
            enc_outs.append(bp)
        for i, mod in enumerate(self.up_convs):
            x = mod(enc_outs[-(i + 2)], x)
        x = self.conv_final(_swish(self.norm_out(x)))
        return einops.rearrange(x, "b c (tri h) w -> b tri c h w", tri=3)


def _normalize_coordinate(p: torch.Tensor, padding: float = 0.0, plane: str = "xz") -> torch.Tensor:
    if plane == "xz":
        xy = p[:, :, [0, 2]]
    elif plane == "xy":
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]
    xy_new = xy / (1.0 + padding + 1e-5) + 0.5
    return torch.clamp(xy_new, 0.0, 1.0 - 1e-5)


def _generate_plane_features(p: torch.Tensor, c: torch.Tensor, resolution: int, plane: str) -> torch.Tensor:
    xy = _normalize_coordinate(p, padding=0.0, plane=plane)
    coords = (xy * resolution).long()
    index = (coords[:, :, 0] + resolution * coords[:, :, 1])[:, None, :]
    fea_plane = c.new_zeros(p.size(0), c.size(1), resolution**2)
    fea_plane = _scatter_mean(c, index, fea_plane)
    return fea_plane.reshape(p.size(0), c.size(1), resolution, resolution)


def sample_triplane_feat(feature_triplane: torch.Tensor, normalized_pos: torch.Tensor) -> torch.Tensor:
    tri_plane = torch.unbind(feature_triplane, dim=1)
    x_feat = F.grid_sample(tri_plane[0], torch.cat([normalized_pos[:, :, 0:1], normalized_pos[:, :, 1:2]], dim=-1).unsqueeze(1), padding_mode="border", align_corners=True)
    y_feat = F.grid_sample(tri_plane[1], torch.cat([normalized_pos[:, :, 1:2], normalized_pos[:, :, 2:3]], dim=-1).unsqueeze(1), padding_mode="border", align_corners=True)
    z_feat = F.grid_sample(tri_plane[2], torch.cat([normalized_pos[:, :, 0:1], normalized_pos[:, :, 2:3]], dim=-1).unsqueeze(1), padding_mode="border", align_corners=True)
    return (x_feat + y_feat + z_feat).squeeze(dim=2).permute(0, 2, 1)


class TriPlanePC2Encoder(nn.Module):
    def __init__(self, z_triplane_channels: int = 256, z_triplane_resolution: int = 128, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.z_triplane_resolution = z_triplane_resolution
        self.shape_min = -1.0
        self.shape_length = 2.0
        self.pc_encoder = PVCNNEncoder(z_triplane_channels, device=device, in_channels=6)
        self.unet_encoder = UNetTriplane3dAware(
            out_channels=z_triplane_channels,
            in_channels=z_triplane_channels,
            depth=3,
            start_filts=32,
            use_initial_conv=False,
        )

    def encode(self, point_cloud_xyz: torch.Tensor, point_cloud_feature: torch.Tensor) -> torch.Tensor:
        xyz = (point_cloud_xyz - self.shape_min) / self.shape_length - 0.5
        pc = torch.cat([xyz, point_cloud_feature], dim=-1)
        _, points_feat = self.pc_encoder(pc)
        pf = points_feat[0]
        p1 = _generate_plane_features(xyz, pf, self.z_triplane_resolution, plane="xy")
        p2 = _generate_plane_features(xyz, pf, self.z_triplane_resolution, plane="yz")
        p3 = _generate_plane_features(xyz, pf, self.z_triplane_resolution, plane="xz")
        pre = torch.stack([p1, p2, p3], dim=1)
        stacked = self.unet_encoder(pre)
        o1, o2, o3 = torch.unbind(stacked, dim=1)
        return torch.stack([o1, o2, o3], dim=1)

    def forward(self, point_cloud_xyz: torch.Tensor, point_cloud_feature: torch.Tensor) -> torch.Tensor:
        return self.encode(point_cloud_xyz, point_cloud_feature)
