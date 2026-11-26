# Copyright (C) 2024 Tencent / THL A29 Limited.
# Licensed under the Apache License Version 2.0 and TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT.
# Distilled without torchvision dependency (uses native torch.nn.functional.interpolate).

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    Dinov2Config,
    Dinov2Model,
)


def _preprocess_tensor_for_encoder(
    image: torch.Tensor,
    target_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    if value_range is not None:
        low, high = value_range
        image = (image - low) / (high - low)

    _, _, h, w = image.shape
    if h != target_size or w != target_size:
        # Match torchvision.transforms.Resize(target_size) shortest-edge resize + CenterCrop
        if h < w:
            new_h, new_w = target_size, int(round(w * target_size / h))
        else:
            new_h, new_w = int(round(h * target_size / w)), target_size
        image = F.interpolate(
            image,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        top = (new_h - target_size) // 2
        left = (new_w - target_size) // 2
        image = image[:, :, top : top + target_size, left : left + target_size]

    mean_t = torch.tensor(mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean_t) / std_t


class ImageEncoder(nn.Module):
    def __init__(
        self,
        version=None,
        config=None,
        use_cls_token: bool = True,
        image_size: int = 224,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            self.model = self.MODEL_CLASS.from_pretrained(version)
        else:
            self.model = self.MODEL_CLASS(self.MODEL_CONFIG_CLASS.from_dict(config))
        self.model.eval()
        self.model.requires_grad_(False)
        self.use_cls_token = use_cls_token
        self.image_size = int(image_size)
        self.num_patches = (self.image_size // 14) ** 2 + (1 if use_cls_token else 0)

    def forward(self, image: torch.Tensor, mask=None, value_range=(-1.0, 1.0), **kwargs):
        image = image.to(self.model.device, dtype=self.model.dtype)
        inputs = _preprocess_tensor_for_encoder(
            image,
            target_size=self.image_size,
            mean=self.mean,
            std=self.std,
            value_range=value_range,
        )
        # Note: Dinov2 handles arbitrary image_size if interpolate_pos_encoding=True for resolutions != config.image_size
        outputs = (
            self.model(inputs, interpolate_pos_encoding=True)
            if isinstance(self.model, Dinov2Model)
            else self.model(inputs)
        )
        last_hidden_state = outputs.last_hidden_state
        if not self.use_cls_token:
            last_hidden_state = last_hidden_state[:, 1:, :]
        return last_hidden_state

    def unconditional_embedding(self, batch_size: int, **kwargs):
        param = next(self.model.parameters())
        return torch.zeros(
            batch_size,
            self.num_patches,
            self.model.config.hidden_size,
            device=param.device,
            dtype=param.dtype,
        )


class CLIPImageEncoder(ImageEncoder):
    MODEL_CLASS = CLIPVisionModelWithProjection
    MODEL_CONFIG_CLASS = CLIPVisionConfig
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)


class DinoImageEncoder(ImageEncoder):
    MODEL_CLASS = Dinov2Model
    MODEL_CONFIG_CLASS = Dinov2Config
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


def build_image_encoder(config: dict) -> nn.Module:
    enc_type = config["type"]
    if enc_type == "CLIPImageEncoder":
        return CLIPImageEncoder(**config["kwargs"])
    if enc_type == "DinoImageEncoder":
        return DinoImageEncoder(**config["kwargs"])
    raise ValueError(f"Unknown image encoder type: {enc_type}")


class DualImageEncoder(nn.Module):
    def __init__(self, main_image_encoder, additional_image_encoder):
        super().__init__()
        self.main_image_encoder = build_image_encoder(main_image_encoder)
        self.additional_image_encoder = build_image_encoder(additional_image_encoder)

    def forward(self, image, mask=None, **kwargs):
        return {
            "main": self.main_image_encoder(image, mask=mask, **kwargs),
            "additional": self.additional_image_encoder(image, mask=mask, **kwargs),
        }

    def unconditional_embedding(self, batch_size: int, **kwargs):
        return {
            "main": self.main_image_encoder.unconditional_embedding(batch_size, **kwargs),
            "additional": self.additional_image_encoder.unconditional_embedding(batch_size, **kwargs),
        }


class SingleImageEncoder(nn.Module):
    def __init__(self, main_image_encoder):
        super().__init__()
        self.main_image_encoder = build_image_encoder(main_image_encoder)

    def forward(self, image, mask=None, **kwargs):
        return {"main": self.main_image_encoder(image, mask=mask, **kwargs)}

    def unconditional_embedding(self, batch_size: int, **kwargs):
        return {"main": self.main_image_encoder.unconditional_embedding(batch_size, **kwargs)}
