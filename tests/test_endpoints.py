"""Unit and HTTP contract tests for FLUX.2-klein-4B, TRELLIS.2-4B, and GPU-Lease Gateway."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gateway.app as gw_mod
import services.flux2_klein.app as flux_app_mod
import services.trellis2.app as trellis_app_mod
from services.flux2_klein.engine import Flux2Config, build_sdcpp_command
from services.trellis2.engine import Trellis2Config, build_trellis_command


def _sample_png_bytes(color=(220, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


def test_build_sdcpp_command_flags(tmp_path):
    cfg = Flux2Config(
        bin_path="/opt/sd-cli",
        model_path="/models/flux-2-klein-4b-Q4_K_M.gguf",
        llm_path="/models/Qwen3-4B-Q4_K_M.gguf",
        vae_path="/models/flux2-vae.safetensors",
        offload_to_cpu=True,
    )
    cmd = build_sdcpp_command(
        cfg,
        prompt="3d vinyl toy robot",
        out_path=tmp_path / "out.png",
        ref_paths=[tmp_path / "sketch.png"],
        width=773,
        height=519,
        steps=4,
        cfg_scale=1.0,
        seed=99,
        negative_prompt="blurry",
    )
    assert cmd[0] == "/opt/sd-cli"
    assert "--diffusion-model" in cmd and cfg.model_path in cmd
    assert "--vae" in cmd and cfg.vae_path in cmd
    assert "--llm" in cmd and cfg.llm_path in cmd
    assert "--sampling-method" in cmd and "euler" in cmd
    # Dimensions rounded cleanly down to multiple of 16
    w_idx = cmd.index("-W")
    h_idx = cmd.index("-H")
    assert cmd[w_idx + 1] == "768"
    assert cmd[h_idx + 1] == "512"
    assert "-n" in cmd and "blurry" in cmd
    assert "-r" in cmd and str(tmp_path / "sketch.png") in cmd
    assert "--offload-to-cpu" in cmd


def test_build_trellis_command_single_and_multiview(tmp_path):
    cfg = Trellis2Config(bin_path="/opt/trellis-cli", models_dir="/models/trellis2", require_gpu=True)
    single = build_trellis_command(
        cfg,
        input_paths=[tmp_path / "front.png"],
        out_glb=tmp_path / "out.glb",
        res=512,
        steps=12,
        faces=150000,
        no_texture=True,
    )
    assert single[:3] == ["/opt/trellis-cli", str(tmp_path / "front.png"), str(tmp_path / "out.glb")]
    assert "--no-texture" in single
    assert "--require-gpu" in single
    assert "--faces" not in single
    assert "--bg-removal" not in single  # 'auto' omits --bg-removal in trellis-cli v0.8.1

    mv = build_trellis_command(
        cfg,
        input_paths=[tmp_path / "f.png", tmp_path / "b.png"],
        out_glb=tmp_path / "mv.glb",
        res=1024,
        no_texture=False,
        bg_removal="threshold",
        view_angles=[0, 180],
    )
    assert "--view" in mv
    assert f"{tmp_path / 'f.png'}:0" in mv
    assert f"{tmp_path / 'b.png'}:180" in mv
    assert "--bg-removal" in mv and "threshold" in mv
    assert "--no-texture" not in mv


def test_flux2_endpoint_health_concept_and_auth(monkeypatch):
    monkeypatch.delenv("STUDIO_API_TOKEN", raising=False)
    monkeypatch.setattr(flux_app_mod.engine, "health", lambda: {
        "ok": True,
        "service": "flux2_klein",
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "quant": "Q4_K_M",
    })
    out_png = _sample_png_bytes((20, 120, 240))
    monkeypatch.setattr(
        flux_app_mod.engine,
        "generate",
        lambda **kw: (out_png, {"elapsed_sec": 0.42, "width": 768, "height": 768, "steps": 4, "seed": 42, "ref_count": len(kw["ref_images"])}),
    )

    client = TestClient(flux_app_mod.app)
    assert client.get("/health").json()["service"] == "flux2_klein"

    res_png = client.post(
        "/v1/concept",
        data={"prompt": "toy dragon", "response_format": "png"},
        files={"ref_image": ("sketch.png", _sample_png_bytes(), "image/png")},
    )
    assert res_png.status_code == 200
    assert res_png.headers["content-type"].startswith("image/png")
    assert res_png.content == out_png

    res_json = client.post("/v1/concept", data={"prompt": "toy dragon", "response_format": "json"})
    assert res_json.status_code == 200
    assert base64.b64decode(res_json.json()["image_png_b64"]) == out_png

    monkeypatch.setenv("STUDIO_API_TOKEN", "secret-123")
    unauth = client.post("/v1/concept", data={"prompt": "toy"})
    assert unauth.status_code == 401
    auth = client.post("/v1/concept", data={"prompt": "toy"}, headers={"Authorization": "Bearer secret-123"})
    assert auth.status_code == 200


def test_trellis2_endpoint_health_and_glb(monkeypatch):
    monkeypatch.delenv("STUDIO_API_TOKEN", raising=False)
    monkeypatch.setattr(trellis_app_mod.engine, "health", lambda: {
        "ok": True,
        "service": "trellis2",
        "model": "microsoft/TRELLIS.2-4B",
        "has_texture_weights": False,
        "has_1024_weights": False,
    })
    fake_glb = b"glTF\x02\x00\x00\x00\x20\x00\x00\x00JSON{}"

    def _fake_recon(**kw):
        return fake_glb, {
            "elapsed_sec": 1.25,
            "res": 512,
            "steps": 12,
            "faces": 150000,
            "views_used": len(kw["images"]),
            "no_texture": True,
            "texture_fallback": not kw["no_texture"],
            "bytes": len(fake_glb),
        }

    monkeypatch.setattr(trellis_app_mod.engine, "reconstruct", _fake_recon)
    client = TestClient(trellis_app_mod.app)
    assert client.get("/health").json()["service"] == "trellis2"

    res = client.post(
        "/v1/image-to-3d",
        data={"res": "512", "no_texture": "false"},
        files={"image": ("concept.png", _sample_png_bytes(), "image/png")},
    )
    assert res.status_code == 200
    assert res.content.startswith(b"glTF")
    assert res.headers.get("X-Texture-Fallback") == "geometry-only"


def test_gateway_health_and_sketch_to_3d(monkeypatch):
    monkeypatch.delenv("STUDIO_API_TOKEN", raising=False)
    concept_png = _sample_png_bytes((40, 200, 90))
    fake_glb = b"glTF\x02\x00\x00\x00\x20\x00\x00\x00JSON{}"

    monkeypatch.setattr(flux_app_mod.engine, "health", lambda: {"ok": True, "service": "flux2_klein", "quant": "Q4_K_M"})
    monkeypatch.setattr(flux_app_mod.engine, "generate", lambda **kw: (concept_png, {"elapsed_sec": 0.2}))
    monkeypatch.setattr(trellis_app_mod.engine, "health", lambda: {"ok": True, "service": "trellis2", "quant": "q4"})
    monkeypatch.setattr(trellis_app_mod.engine, "reconstruct", lambda **kw: (fake_glb, {"elapsed_sec": 0.5, "texture_fallback": False}))

    flux_transport = httpx.ASGITransport(app=flux_app_mod.app)
    trellis_transport = httpx.ASGITransport(app=trellis_app_mod.app)

    orig_async_client = httpx.AsyncClient

    class _RouterClient:
        def __init__(self, *args, **kwargs):
            self._fc = orig_async_client(transport=flux_transport, base_url="http://flux2")
            self._tc = orig_async_client(transport=trellis_transport, base_url="http://trellis2")

        async def __aenter__(self):
            await self._fc.__aenter__()
            await self._tc.__aenter__()
            return self

        async def __aexit__(self, *exc):
            await self._fc.__aexit__(*exc)
            await self._tc.__aexit__(*exc)

        async def get(self, url, **kw):
            if url.startswith(gw_mod.FLUX2_URL):
                return await self._fc.get(url.replace(gw_mod.FLUX2_URL, ""), **kw)
            return await self._tc.get(url.replace(gw_mod.TRELLIS2_URL, ""), **kw)

        async def post(self, url, **kw):
            if url.startswith(gw_mod.FLUX2_URL):
                return await self._fc.post(url.replace(gw_mod.FLUX2_URL, ""), **kw)
            return await self._tc.post(url.replace(gw_mod.TRELLIS2_URL, ""), **kw)

    monkeypatch.setattr(gw_mod.httpx, "AsyncClient", _RouterClient)
    gw_client = TestClient(gw_mod.app)

    h = gw_client.get("/health").json()
    assert h["ok"] is True
    assert h["service"] == "studio-3d-gateway"
    assert h["services"]["flux2_klein"]["ok"] is True
    assert h["services"]["trellis2"]["ok"] is True

    res = gw_client.post(
        "/v1/sketch-to-3d",
        data={"prompt": "robot toy", "response_format": "json"},
        files={"image": ("drawing.png", _sample_png_bytes(), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert base64.b64decode(body["concept_png_b64"]) == concept_png
    assert base64.b64decode(body["glb_b64"]) == fake_glb


def test_scripts_lf_and_notebook_valid():
    for sh in (REPO_ROOT / "scripts").glob("*.sh"):
        raw = sh.read_bytes()
        assert b"\r\n" not in raw, f"{sh.name} must use LF line endings, not CRLF"
    nb = json.loads((REPO_ROOT / "colab_endpoints.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "santiago96ml/studio-3d-ai-endpoints" in joined
