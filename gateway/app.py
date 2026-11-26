"""Thin GPU-lease Gateway orchestrating FLUX.2-klein-4B (8001) and TRELLIS.2-4B (8002)."""

from __future__ import annotations

import asyncio
import base64
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.runtime import detect_gpu_info, verify_api_token

app = FastAPI(title="Studio 3D GPU-Lease Gateway", version="1.0.0")
gpu_lease_lock = asyncio.Lock()

FLUX2_URL = os.environ.get("FLUX2_URL", "http://127.0.0.1:8001").rstrip("/")
TRELLIS2_URL = os.environ.get("TRELLIS2_URL", "http://127.0.0.1:8002").rstrip("/")


def _forward_headers() -> dict[str, str]:
    tok = os.environ.get("STUDIO_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


async def _probe_service(client: httpx.AsyncClient, url: str) -> dict:
    try:
        resp = await client.get(f"{url}/health", timeout=4.0)
        if resp.status_code == 200:
            return resp.json()
        return {"ok": False, "status_code": resp.status_code}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


@app.get("/health")
async def health():
    async with httpx.AsyncClient() as client:
        flux_h, trellis_h = await asyncio.gather(
            _probe_service(client, FLUX2_URL),
            _probe_service(client, TRELLIS2_URL),
        )
    return {
        "ok": bool(flux_h.get("ok") or trellis_h.get("ok")),
        "service": "studio-3d-gateway",
        "gpu": detect_gpu_info(),
        "services": {
            "flux2_klein": flux_h,
            "trellis2": trellis_h,
        },
    }


@app.post("/v1/concept")
async def concept_endpoint(
    request: Request,
    prompt: str = Form(...),
    ref_image: Optional[UploadFile] = File(None),
    extra_ref_images: Optional[List[UploadFile]] = File(None),
    negative_prompt: str = Form(""),
    width: int = Form(768),
    height: int = Form(768),
    steps: int = Form(4),
    cfg_scale: float = Form(1.0),
    seed: int = Form(42),
    response_format: str = Form("png"),
):
    verify_api_token(request.headers)
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    if ref_image is not None:
        raw = await ref_image.read()
        if raw:
            files.append(("ref_image", (ref_image.filename or "ref.png", raw, ref_image.content_type or "image/png")))
    if extra_ref_images:
        for idx, item in enumerate(extra_ref_images):
            raw = await item.read()
            if raw:
                files.append(("extra_ref_images", (item.filename or f"extra_{idx}.png", raw, item.content_type or "image/png")))

    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": str(width),
        "height": str(height),
        "steps": str(steps),
        "cfg_scale": str(cfg_scale),
        "seed": str(seed),
        "response_format": response_format,
    }

    async with gpu_lease_lock:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{FLUX2_URL}/v1/concept",
                data=data,
                files=files or None,
                headers=_forward_headers(),
            )
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])

    if response_format.strip().lower() == "json":
        return resp.json()
    pass_headers = {
        k: v for k, v in resp.headers.items() if k.lower().startswith("x-")
    }
    return Response(content=resp.content, media_type="image/png", headers=pass_headers)


@app.post("/v1/image-to-3d")
async def image_to_3d_endpoint(
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
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="image file is required")

    files: list[tuple[str, tuple[str, bytes, str]]] = [
        ("image", (image.filename or "input.png", img_bytes, image.content_type or "image/png"))
    ]
    if extra_views:
        for idx, item in enumerate(extra_views):
            raw = await item.read()
            if raw:
                files.append(("extra_views", (item.filename or f"view_{idx}.png", raw, item.content_type or "image/png")))

    data = {
        "view_angles": view_angles,
        "res": str(res),
        "steps": str(steps),
        "faces": str(faces),
        "seed": str(seed),
        "no_texture": "true" if no_texture else "false",
        "bg_removal": bg_removal,
        "strict": "true" if strict else "false",
        "response_format": response_format,
    }

    async with gpu_lease_lock:
        async with httpx.AsyncClient(timeout=480.0) as client:
            resp = await client.post(
                f"{TRELLIS2_URL}/v1/image-to-3d",
                data=data,
                files=files,
                headers=_forward_headers(),
            )
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])

    if response_format.strip().lower() == "json":
        return resp.json()
    pass_headers = {
        k: v for k, v in resp.headers.items() if k.lower().startswith("x-") or k.lower() == "content-disposition"
    }
    return Response(content=resp.content, media_type="model/gltf-binary", headers=pass_headers)


@app.post("/v1/sketch-to-3d")
async def sketch_to_3d_endpoint(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form("3D matte PLA vinyl toy figurine, solid studio lighting, white background"),
    width: int = Form(768),
    height: int = Form(768),
    flux_steps: int = Form(4),
    trellis_res: int = Form(512),
    trellis_steps: int = Form(12),
    faces: int = Form(150000),
    seed: int = Form(42),
    no_texture: bool = Form(True),
    response_format: str = Form("json"),
):
    """End-to-end GPU-serialized pipeline: sketch -> FLUX.2 concept -> TRELLIS.2 GLB."""
    verify_api_token(request.headers)
    raw_sketch = await image.read()
    if not raw_sketch:
        raise HTTPException(status_code=400, detail="image file is required")

    t0 = time.perf_counter()
    async with gpu_lease_lock:
        async with httpx.AsyncClient(timeout=600.0) as client:
            t_f0 = time.perf_counter()
            r_flux = await client.post(
                f"{FLUX2_URL}/v1/concept",
                data={
                    "prompt": prompt,
                    "width": str(width),
                    "height": str(height),
                    "steps": str(flux_steps),
                    "cfg_scale": "1.0",
                    "seed": str(seed),
                    "response_format": "png",
                },
                files=[("ref_image", (image.filename or "sketch.png", raw_sketch, image.content_type or "image/png"))],
                headers=_forward_headers(),
            )
            if r_flux.status_code >= 400:
                raise HTTPException(status_code=r_flux.status_code, detail=f"FLUX.2 failed: {r_flux.text[:400]}")
            concept_bytes = r_flux.content
            flux_sec = round(time.perf_counter() - t_f0, 3)

            t_t0 = time.perf_counter()
            r_3d = await client.post(
                f"{TRELLIS2_URL}/v1/image-to-3d",
                data={
                    "res": str(trellis_res),
                    "steps": str(trellis_steps),
                    "faces": str(faces),
                    "seed": str(seed),
                    "no_texture": "true" if no_texture else "false",
                    "bg_removal": "auto",
                    "response_format": "glb",
                },
                files=[("image", ("concept.png", concept_bytes, "image/png"))],
                headers=_forward_headers(),
            )
            if r_3d.status_code >= 400:
                raise HTTPException(status_code=r_3d.status_code, detail=f"TRELLIS.2 failed: {r_3d.text[:400]}")
            glb_bytes = r_3d.content
            trellis_sec = round(time.perf_counter() - t_t0, 3)

    total_sec = round(time.perf_counter() - t0, 3)
    if response_format.strip().lower() == "glb":
        return Response(
            content=glb_bytes,
            media_type="model/gltf-binary",
            headers={
                "X-Inference-Seconds": str(total_sec),
                "Content-Disposition": 'attachment; filename="sketch_to_3d.glb"',
            },
        )

    return {
        "ok": True,
        "service": "studio-3d-gateway",
        "concept_png_b64": base64.b64encode(concept_bytes).decode("ascii"),
        "glb_b64": base64.b64encode(glb_bytes).decode("ascii"),
        "timings": {
            "flux2_sec": flux_sec,
            "trellis2_sec": trellis_sec,
            "total_sec": total_sec,
        },
    }
