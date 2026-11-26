"""FastAPI server for dedicated Microsoft TRELLIS.2-4B image-to-3D reconstruction."""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.runtime import verify_api_token
from services.trellis2.engine import Trellis2Engine

app = FastAPI(title="Microsoft TRELLIS.2-4B Dedicated Endpoint", version="1.0.0")
engine = Trellis2Engine()
_gpu_lock = asyncio.Lock()


def _parse_angles(raw: str) -> list[int]:
    out: list[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)) % 360)
        except ValueError:
            continue
    return out


@app.get("/health")
def health():
    return engine.health()


@app.post("/v1/image-to-3d")
async def image_to_3d(
    request: Request,
    image: UploadFile = File(...),
    extra_views: Optional[List[UploadFile]] = File(None),
    view_angles: str = Form("0,90,180,270"),
    res: int = Form(512),
    steps: int = Form(12),
    faces: int = Form(150000),
    seed: int = Form(42),
    no_texture: bool = Form(True),
    bg_removal: str = Form("auto"),
    strict: bool = Form(False),
    response_format: str = Form("glb"),
):
    verify_api_token(request.headers)
    images: list[bytes] = []
    primary = await image.read()
    if primary:
        images.append(primary)
    if extra_views:
        for item in extra_views:
            raw = await item.read()
            if raw:
                images.append(raw)

    if not images:
        raise HTTPException(status_code=400, detail="image file is required")

    angles = _parse_angles(view_angles)

    async with _gpu_lock:
        try:
            glb_bytes, meta = await asyncio.to_thread(
                engine.reconstruct,
                images=images,
                view_angles=angles,
                res=res,
                steps=steps,
                faces=faces,
                seed=seed,
                no_texture=no_texture,
                bg_removal=bg_removal,
                strict=strict,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if response_format.strip().lower() == "json":
        return {
            "ok": True,
            "service": "trellis2",
            "model": "microsoft/TRELLIS.2-4B",
            "glb_b64": base64.b64encode(glb_bytes).decode("ascii"),
            **meta,
        }

    headers = {
        "X-Inference-Seconds": str(meta["elapsed_sec"]),
        "X-Model-Name": "microsoft/TRELLIS.2-4B",
        "Content-Disposition": 'attachment; filename="trellis2_output.glb"',
    }
    if meta.get("texture_fallback"):
        headers["X-Texture-Fallback"] = "geometry-only"

    return Response(content=glb_bytes, media_type="model/gltf-binary", headers=headers)
