"""Distilled Hunyuan3D-2 Geometry FastAPI Microservice (port 8003)."""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from .engine import Hunyuan3DEngine

app = FastAPI(title="Distilled Hunyuan3D-2 Geometry Endpoint", version="1.0.0")
engine = Hunyuan3DEngine()
_gpu_lock = asyncio.Lock()

API_TOKEN = os.environ.get("STUDIO_API_TOKEN", "").strip()


def verify_auth(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    if not API_TOKEN:
        return
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing Studio API token")


@app.get("/health")
def health(_: None = Depends(verify_auth)):
    return engine.status()


@app.post("/v1/unload")
async def unload_vram(_: None = Depends(verify_auth)):
    async with _gpu_lock:
        return await asyncio.to_thread(engine.unload)


@app.post("/v1/image-to-3d")
async def image_to_3d(
    _: None = Depends(verify_auth),
    image: UploadFile = File(...),
    res: int = Form(default=384),
    steps: int = Form(default=5),
    guidance: float = Form(default=5.0),
    faces: int = Form(default=150000),
    seed: int = Form(default=42),
    bg_removal: str = Form(default="auto"),
    watertight: bool = Form(default=True),
    variant: Optional[str] = Form(default=None),
    response_format: str = Form(default="glb"),
):
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    async with _gpu_lock:
        try:
            glb_bytes, meta = await asyncio.to_thread(
                engine.generate_glb,
                image_bytes=img_bytes,
                steps=steps,
                guidance=guidance,
                octree_res=res,
                max_faces=faces,
                seed=seed,
                bg_removal=bg_removal,
                ensure_watertight=watertight,
                variant=variant,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Hunyuan3D-2 distilled failed: {exc}") from exc

    if response_format.lower() == "b64":
        return JSONResponse(
            {
                "glb_b64": base64.b64encode(glb_bytes).decode("ascii"),
                "meta": meta,
            }
        )

    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": 'attachment; filename="hunyuan3d_output.glb"',
            "X-Engine": str(meta["engine"]),
            "X-Inference-Seconds": str(meta["elapsed_s"]),
            "X-Mesh-Vertices": str(meta["vertices"]),
            "X-Mesh-Faces": str(meta["faces"]),
            "X-Mesh-Watertight": "1" if meta["watertight"] else "0",
            "X-Mesh-NonManifold": str(meta["non_manifold_edges"]),
            "X-VRAM-Peak-MB": str(meta["vram_peak_mb"]),
            "X-Cutout-Method": str(meta["cutout_method"]),
        },
    )
