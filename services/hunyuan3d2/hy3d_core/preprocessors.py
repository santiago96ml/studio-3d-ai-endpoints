# Copyright (C) 2024 Tencent / THL A29 Limited.
# Licensed under the TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT (see ../LICENSE-HUNYUAN3D.txt).
# Distilled pure-Pillow/NumPy preprocessor (no OpenCV/libGL dependency) + zero-cost white-bg flood cutout.

from collections import deque
from typing import TYPE_CHECKING, Tuple

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch


def array_to_tensor(np_array: np.ndarray) -> "torch.Tensor":
    import torch

    image_pt = torch.from_numpy(np_array).float()
    image_pt = image_pt / 255.0 * 2.0 - 1.0
    return image_pt.permute(2, 0, 1).unsqueeze(0)


def fast_white_bg_to_rgba(
    image: Image.Image,
    white_thresh: int = 238,
    neutral_diff: int = 22,
) -> Tuple[Image.Image, bool]:
    """Cut out clean uniform light backgrounds via border flood-fill in <5 ms without ONNX/VRAM.

    Returns (rgba_image, extracted_cleanly). If the image has neither real alpha nor a clean light
    border (e.g. a real-world photo), returns (original_rgba, False) so the caller can opt into rembg.
    """
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8).copy()
    alpha = arr[:, :, 3]

    # Respect existing alpha masks immediately
    if int(alpha.min()) < 250 and float(np.mean(alpha < 128)) > 0.02:
        return rgba, True

    rgb = arr[:, :, :3].astype(np.int16)
    h, w = rgb.shape[:2]
    is_light = (rgb.min(axis=2) >= white_thresh) & ((rgb.max(axis=2) - rgb.min(axis=2)) <= neutral_diff)

    border_light = (
        float(is_light[0, :].mean() + is_light[-1, :].mean() + is_light[:, 0].mean() + is_light[:, -1].mean())
        / 4.0
    )
    if border_light < 0.80:
        return rgba, False

    bg = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(w):
        if is_light[0, x]:
            bg[0, x] = True
            queue.append((0, x))
        if is_light[h - 1, x] and not bg[h - 1, x]:
            bg[h - 1, x] = True
            queue.append((h - 1, x))
    for y in range(1, h - 1):
        if is_light[y, 0] and not bg[y, 0]:
            bg[y, 0] = True
            queue.append((y, 0))
        if is_light[y, w - 1] and not bg[y, w - 1]:
            bg[y, w - 1] = True
            queue.append((y, w - 1))

    while queue:
        cy, cx = queue.popleft()
        if cy > 0 and is_light[cy - 1, cx] and not bg[cy - 1, cx]:
            bg[cy - 1, cx] = True
            queue.append((cy - 1, cx))
        if cy + 1 < h and is_light[cy + 1, cx] and not bg[cy + 1, cx]:
            bg[cy + 1, cx] = True
            queue.append((cy + 1, cx))
        if cx > 0 and is_light[cy, cx - 1] and not bg[cy, cx - 1]:
            bg[cy, cx - 1] = True
            queue.append((cy, cx - 1))
        if cx + 1 < w and is_light[cy, cx + 1] and not bg[cy, cx + 1]:
            bg[cy, cx + 1] = True
            queue.append((cy, cx + 1))

    fg_fraction = 1.0 - float(bg.mean())
    if fg_fraction < 0.01 or fg_fraction > 0.995:
        return rgba, False

    arr[:, :, 3] = np.where(bg, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA"), True


class ImageProcessorV2:
    """Pillow + NumPy port of Hunyuan3D ImageProcessorV2 without OpenCV dependency."""

    def __init__(self, size: int = 512, border_ratio: float | None = None):
        self.size = int(size)
        self.border_ratio = border_ratio

    @staticmethod
    def recenter(image: np.ndarray, border_ratio: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        if image.shape[-1] == 4:
            mask = image[..., 3]
        else:
            mask_ch = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
            image = np.concatenate([image, mask_ch], axis=-1)
            mask = mask_ch[..., 0]

        h_orig, w_orig, c = image.shape
        size = max(h_orig, w_orig)
        result = np.zeros((size, size, c), dtype=np.uint8)

        coords = np.nonzero(mask)
        if len(coords[0]) == 0:
            raise ValueError("input image is empty")

        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())
        h = max(1, y_max - y_min)
        w = max(1, x_max - x_min)

        desired_size = max(1, int(size * (1.0 - border_ratio)))
        scale = desired_size / max(h, w)
        h2 = max(1, int(h * scale))
        w2 = max(1, int(w * scale))
        y2_min = (size - h2) // 2
        x2_min = (size - w2) // 2

        crop_pil = Image.fromarray(image[y_min:y_max, x_min:x_max], mode="RGBA")
        crop_resized = crop_pil.resize((w2, h2), resample=Image.Resampling.BOX if (w2 < w or h2 < h) else Image.Resampling.BILINEAR)
        result[y2_min : y2_min + h2, x2_min : x2_min + w2] = np.asarray(crop_resized, dtype=np.uint8)

        bg = np.full((size, size, 3), 255.0, dtype=np.float32)
        mask_f = result[..., 3:].astype(np.float32) / 255.0
        comp = result[..., :3].astype(np.float32) * mask_f + bg * (1.0 - mask_f)
        out_rgb = comp.clip(0, 255).astype(np.uint8)
        out_mask = (mask_f[..., 0] * 255.0).clip(0, 255).astype(np.uint8)
        return out_rgb, out_mask

    def load_image(self, image, border_ratio: float = 0.15, to_tensor: bool = True):
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGBA")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGBA")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        rgba_arr = np.asarray(pil_img, dtype=np.uint8)
        rgb_recentered, mask_recentered = self.recenter(rgba_arr, border_ratio=border_ratio)

        rgb_pil = Image.fromarray(rgb_recentered, mode="RGB").resize(
            (self.size, self.size), resample=Image.Resampling.BICUBIC
        )
        mask_pil = Image.fromarray(mask_recentered, mode="L").resize(
            (self.size, self.size), resample=Image.Resampling.NEAREST
        )

        rgb_out = np.asarray(rgb_pil, dtype=np.uint8)
        mask_out = np.asarray(mask_pil, dtype=np.uint8)[..., np.newaxis]

        if to_tensor:
            return array_to_tensor(rgb_out), array_to_tensor(mask_out)
        return rgb_out, mask_out

    def __call__(self, image, border_ratio: float = 0.15, to_tensor: bool = True, **kwargs):
        if self.border_ratio is not None:
            border_ratio = self.border_ratio
        image_t, mask_t = self.load_image(image, border_ratio=border_ratio, to_tensor=to_tensor)
        return {"image": image_t, "mask": mask_t}
