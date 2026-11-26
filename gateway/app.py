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
HUNYUAN3D_URL = os.environ.get("HUNYUAN3D_URL", "http://127.0.0.1:8003").rstrip("/")
DEFAULT_3D_ENGINE = os.environ.get("STUDIO_3D_ENGINE", "auto").strip().lower()


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
        flux_h, trellis_h, hunyuan_h = await asyncio.gather(
            _probe_service(client, FLUX2_URL),
            _probe_service(client, TRELLIS2_URL),
            _probe_service(client, HUNYUAN3D_URL),
        )
    return {
        "ok": bool(flux_h.get("ok") or trellis_h.get("ok") or hunyuan_h.get("ok")),
        "service": "studio-3d-gateway",
        "default_3d_engine": DEFAULT_3D_ENGINE,
        "gpu": detect_gpu_info(),
        "services": {
            "flux2_klein": flux_h,
            "hunyuan3d2": hunyuan_h,
            "trellis2": trellis_h,
        },
    }


async def _select_3d_backend(client: httpx.AsyncClient, requested: str) -> tuple[str, str]:
    req = (requested or DEFAULT_3D_ENGINE or "auto").strip().lower()
    if req in ("hunyuan", "hunyuan3d", "hunyuan3d2", "hy3d"):
        return "hunyuan3d2", HUNYUAN3D_URL
    if req in ("trellis", "trellis2"):
        return "trellis2", TRELLIS2_URL
    if DEFAULT_3D_ENGINE in ("hunyuan", "hunyuan3d", "hunyuan3d2", "hy3d"):
        return "hunyuan3d2", HUNYUAN3D_URL
    if DEFAULT_3D_ENGINE in ("trellis", "trellis2"):
        return "trellis2", TRELLIS2_URL
    # Auto: prefer distilled Hunyuan3D-2 if healthy, otherwise TRELLIS.2
    hy_h, tr_h = await asyncio.gather(
        _probe_service(client, HUNYUAN3D_URL),
        _probe_service(client, TRELLIS2_URL),
    )
    if hy_h.get("ok"):
        return "hunyuan3d2", HUNYUAN3D_URL
    if tr_h.get("ok"):
        return "trellis2", TRELLIS2_URL
    return "hunyuan3d2", HUNYUAN3D_URL


async def _maybe_unload_other_3d(client: httpx.AsyncClient, target_engine: str) -> None:
    if target_engine == "trellis2":
        try:
            await client.post(f"{HUNYUAN3D_URL}/v1/unload", headers=_forward_headers(), timeout=5.0)
        except Exception:
            pass


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
    res: Optional[int] = Form(None),
    steps: Optional[int] = Form(None),
    guidance: float = Form(5.0),
    faces: int = Form(150000),
    seed: int = Form(42),
    no_texture: bool = Form(True),
    bg_removal: str = Form("auto"),
    strict: bool = Form(False),
    watertight: bool = Form(True),
    engine: str = Form("auto"),
    variant: Optional[str] = Form(None),
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

    async with gpu_lease_lock:
        async with httpx.AsyncClient(timeout=480.0) as client:
            chosen_engine, backend_url = await _select_3d_backend(client, engine)
            await _maybe_unload_other_3d(client, chosen_engine)

            if chosen_engine == "hunyuan3d2":
                data = {
                    "res": str(res if res is not None else 384),
                    "steps": str(steps if steps is not None else 5),
                    "guidance": str(guidance),
                    "faces": str(faces),
                    "seed": str(seed),
                    "bg_removal": bg_removal,
                    "watertight": "true" if watertight else "false",
                    "response_format": "b64" if response_format.strip().lower() == "json" else response_format,
                }
                if variant:
                    data["variant"] = variant
                send_files = [files[0]]
            else:
                data = {
                    "view_angles": view_angles,
                    "res": str(res if res is not None else 512),
                    "steps": str(steps if steps is not None else 12),
                    "faces": str(faces),
                    "seed": str(seed),
                    "no_texture": "true" if no_texture else "false",
                    "bg_removal": bg_removal,
                    "strict": "true" if strict else "false",
                    "response_format": response_format,
                }
                send_files = files

            resp = await client.post(
                f"{backend_url}/v1/image-to-3d",
                data=data,
                files=send_files,
                headers=_forward_headers(),
            )
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])

    if response_format.strip().lower() in ("json", "b64"):
        return resp.json()
    pass_headers = {
        k: v for k, v in resp.headers.items() if k.lower().startswith("x-") or k.lower() == "content-disposition"
    }
    pass_headers.setdefault("X-Engine", chosen_engine)
    return Response(content=resp.content, media_type="model/gltf-binary", headers=pass_headers)


@app.post("/v1/sketch-to-3d")
async def sketch_to_3d_endpoint(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form("3D matte PLA vinyl toy figurine, solid studio lighting, white background"),
    width: int = Form(768),
    height: int = Form(768),
    flux_steps: int = Form(4),
    trellis_res: Optional[int] = Form(None),
    trellis_steps: Optional[int] = Form(None),
    faces: int = Form(150000),
    seed: int = Form(42),
    no_texture: bool = Form(True),
    engine: str = Form("auto"),
    response_format: str = Form("json"),
):
    """End-to-end GPU-serialized pipeline: sketch -> FLUX.2 concept -> 3D engine GLB."""
    verify_api_token(request.headers)
    raw_sketch = await image.read()
    if not raw_sketch:
        raise HTTPException(status_code=400, detail="image file is required")

    t0 = time.perf_counter()
    async with gpu_lease_lock:
        async with httpx.AsyncClient(timeout=600.0) as client:
            chosen_engine, backend_url = await _select_3d_backend(client, engine)
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

            await _maybe_unload_other_3d(client, chosen_engine)
            t_t0 = time.perf_counter()
            if chosen_engine == "hunyuan3d2":
                recon_data = {
                    "res": str(trellis_res if trellis_res is not None else 384),
                    "steps": str(trellis_steps if trellis_steps is not None else 5),
                    "faces": str(faces),
                    "seed": str(seed),
                    "bg_removal": "auto",
                    "watertight": "true",
                    "response_format": "glb",
                }
            else:
                recon_data = {
                    "res": str(trellis_res if trellis_res is not None else 512),
                    "steps": str(trellis_steps if trellis_steps is not None else 12),
                    "faces": str(faces),
                    "seed": str(seed),
                    "no_texture": "true" if no_texture else "false",
                    "bg_removal": "auto",
                    "response_format": "glb",
                }

            r_3d = await client.post(
                f"{backend_url}/v1/image-to-3d",
                data=recon_data,
                files=[("image", ("concept.png", concept_bytes, "image/png"))],
                headers=_forward_headers(),
            )
            if r_3d.status_code >= 400:
                raise HTTPException(status_code=r_3d.status_code, detail=f"{chosen_engine} failed: {r_3d.text[:400]}")
            glb_bytes = r_3d.content
            recon_sec = round(time.perf_counter() - t_t0, 3)

    total_sec = round(time.perf_counter() - t0, 3)
    if response_format.strip().lower() == "glb":
        return Response(
            content=glb_bytes,
            media_type="model/gltf-binary",
            headers={
                "X-Inference-Seconds": str(total_sec),
                "X-Engine": chosen_engine,
                "Content-Disposition": 'attachment; filename="sketch_to_3d.glb"',
            },
        )

    return {
        "ok": True,
        "service": "studio-3d-gateway",
        "engine_used": chosen_engine,
        "concept_png_b64": base64.b64encode(concept_bytes).decode("ascii"),
        "glb_b64": base64.b64encode(glb_bytes).decode("ascii"),
        "timings": {
            "flux2_sec": flux_sec,
            "recon_sec": recon_sec,
            "trellis2_sec": recon_sec,
            "total_sec": total_sec,
        },
    }
