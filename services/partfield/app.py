"""Distilled PartField 3D Part Segmentation FastAPI Microservice (port 8004)."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from .engine import PartFieldEngine

app = FastAPI(title="Distilled PartField 3D Segmentation Endpoint", version="1.0.0")
engine = PartFieldEngine()
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


@app.post("/v1/segment-parts")
async def segment_parts(
    _: None = Depends(verify_auth),
    mesh: UploadFile = File(...),
    num_parts: int = Form(default=10),
):
    raw = await mesh.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty 3D mesh upload")
    async with _gpu_lock:
        try:
            return await asyncio.to_thread(engine.segment_glb, glb_bytes=raw, num_parts=num_parts)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PartField segmentation failed: {exc}") from exc
