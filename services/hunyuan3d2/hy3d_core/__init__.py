# Copyright (C) 2024 Tencent / THL A29 Limited. Licensed under TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT.
# Distilled geometry-only inference subset derived from Tencent-Hunyuan/Hunyuan3D-2 (hy3dgen/shapegen).

__all__ = [
    "DistilledHunyuan3DPipeline",
    "clean_and_repair_mesh",
]


def __getattr__(name: str):
    if name == "DistilledHunyuan3DPipeline":
        from .pipeline import DistilledHunyuan3DPipeline

        return DistilledHunyuan3DPipeline
    if name == "clean_and_repair_mesh":
        from .postprocessors import clean_and_repair_mesh

        return clean_and_repair_mesh
    raise AttributeError(name)
