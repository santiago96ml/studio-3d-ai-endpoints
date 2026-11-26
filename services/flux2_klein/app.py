"""FastAPI server for dedicated FLUX.2-klein-4B image/sketch-to-concept inference."""

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
from services.flux2_klein.engine import Flux2Engine

app = FastAPI(title="FLUX.2-klein-4B Dedicated Endpoint", version="1.0.0")
engine = Flux2Engine()
_gpu_lock = asyncio.Lock()


@app.get("/health")
def health():
    return engine.health()


@app.post("/v1/concept")
async def generate_concept(
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
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    refs: list[bytes] = []
    if ref_image is not None:
        raw = await ref_image.read()
        if raw:
            refs.append(raw)
    if extra_ref_images:
        for item in extra_ref_images:
            raw = await item.read()
            if raw:
                refs.append(raw)

    async with _gpu_lock:
        try:
            png_bytes, meta = await asyncio.to_thread(
                engine.generate,
                prompt=prompt,
                ref_images=refs,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if response_format.strip().lower() == "json":
        return {
            "ok": True,
            "service": "flux2_klein",
            "model": "black-forest-labs/FLUX.2-klein-4B",
            "image_png_b64": base64.b64encode(png_bytes).decode("ascii"),
            **meta,
        }

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Inference-Seconds": str(meta["elapsed_sec"]),
            "X-Model-Name": "black-forest-labs/FLUX.2-klein-4B",
        },
    )
