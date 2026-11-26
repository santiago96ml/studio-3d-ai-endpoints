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
import services.hunyuan3d2.app as hunyuan_app_mod
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
    monkeypatch.setattr(
        hunyuan_app_mod.engine,
        "status",
        lambda: {"ok": True, "service": "hunyuan3d2", "variant": "turbo", "distilled_core": True},
    )
    monkeypatch.setattr(
        hunyuan_app_mod.engine,
        "generate_glb",
        lambda **kw: (
            fake_glb,
            {
                "engine": "hunyuan3d2-distilled",
                "variant": "turbo",
                "vram_mode": "resident",
                "steps": kw.get("steps", 5),
                "octree_resolution": kw.get("octree_res", 384),
                "cutout_method": "fast_white_cutout",
                "elapsed_s": 0.3,
                "vram_peak_mb": 5100,
                "vertices": 1000,
                "faces": 2000,
                "watertight": True,
                "non_manifold_edges": 0,
            },
        ),
    )
    monkeypatch.setattr(hunyuan_app_mod.engine, "unload", lambda: {"status": "unloaded", "gpu": {}})
    monkeypatch.setattr(trellis_app_mod.engine, "health", lambda: {"ok": True, "service": "trellis2", "quant": "q4"})
    monkeypatch.setattr(trellis_app_mod.engine, "reconstruct", lambda **kw: (fake_glb, {"elapsed_sec": 0.5, "texture_fallback": False}))

    flux_transport = httpx.ASGITransport(app=flux_app_mod.app)
    hunyuan_transport = httpx.ASGITransport(app=hunyuan_app_mod.app)
    trellis_transport = httpx.ASGITransport(app=trellis_app_mod.app)

    orig_async_client = httpx.AsyncClient

    class _RouterClient:
        def __init__(self, *args, **kwargs):
            self._fc = orig_async_client(transport=flux_transport, base_url="http://flux2")
            self._hc = orig_async_client(transport=hunyuan_transport, base_url="http://hunyuan3d2")
            self._tc = orig_async_client(transport=trellis_transport, base_url="http://trellis2")

        async def __aenter__(self):
            await self._fc.__aenter__()
            await self._hc.__aenter__()
            await self._tc.__aenter__()
            return self

        async def __aexit__(self, *exc):
            await self._fc.__aexit__(*exc)
            await self._hc.__aexit__(*exc)
            await self._tc.__aexit__(*exc)

        async def get(self, url, **kw):
            if url.startswith(gw_mod.FLUX2_URL):
                return await self._fc.get(url.replace(gw_mod.FLUX2_URL, ""), **kw)
            if url.startswith(gw_mod.HUNYUAN3D_URL):
                return await self._hc.get(url.replace(gw_mod.HUNYUAN3D_URL, ""), **kw)
            return await self._tc.get(url.replace(gw_mod.TRELLIS2_URL, ""), **kw)

        async def post(self, url, **kw):
            if url.startswith(gw_mod.FLUX2_URL):
                return await self._fc.post(url.replace(gw_mod.FLUX2_URL, ""), **kw)
            if url.startswith(gw_mod.HUNYUAN3D_URL):
                return await self._hc.post(url.replace(gw_mod.HUNYUAN3D_URL, ""), **kw)
            return await self._tc.post(url.replace(gw_mod.TRELLIS2_URL, ""), **kw)

    monkeypatch.setattr(gw_mod.httpx, "AsyncClient", _RouterClient)
    gw_client = TestClient(gw_mod.app)

    h = gw_client.get("/health").json()
    assert h["ok"] is True
    assert h["service"] == "studio-3d-gateway"
    assert h["services"]["flux2_klein"]["ok"] is True
    assert h["services"]["hunyuan3d2"]["ok"] is True
    assert h["services"]["trellis2"]["ok"] is True

    # Default auto routing prefers distilled hunyuan3d2 when healthy
    res = gw_client.post(
        "/v1/sketch-to-3d",
        data={"prompt": "robot toy", "response_format": "json"},
        files={"image": ("drawing.png", _sample_png_bytes(), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["engine_used"] == "hunyuan3d2"
    assert base64.b64decode(body["concept_png_b64"]) == concept_png
    assert base64.b64decode(body["glb_b64"]) == fake_glb

    # Direct hunyuan3d2 endpoint contract
    hy_client = TestClient(hunyuan_app_mod.app)
    r_hy = hy_client.post(
        "/v1/image-to-3d",
        data={"res": "384", "steps": "5"},
        files={"image": ("concept.png", concept_png, "image/png")},
    )
    assert r_hy.status_code == 200
    assert r_hy.content == fake_glb
    assert r_hy.headers["X-Engine"] == "hunyuan3d2-distilled"
    assert r_hy.headers["X-Mesh-Watertight"] == "1"

    # Explicit trellis2 override via gateway
    r_tr = gw_client.post(
        "/v1/sketch-to-3d",
        data={"prompt": "robot toy", "engine": "trellis2", "response_format": "json"},
        files={"image": ("drawing.png", _sample_png_bytes(), "image/png")},
    )
    assert r_tr.status_code == 200
    assert r_tr.json()["engine_used"] == "trellis2"


def test_hunyuan_distilled_core_no_bloat():
    import ast

    hy_dir = REPO_ROOT / "services" / "hunyuan3d2" / "hy3d_core"
    banned = {"gradio", "bpy", "pymeshlab", "torch_cluster", "torchvision", "cv2", "diso", "hy3dgen", "pytorch_lightning", "xatlas", "pygltflib"}
    for py_path in hy_dir.glob("*.py"):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    assert root_mod not in banned, f"Banned import '{alias.name}' in {py_path.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_mod = node.module.split(".")[0]
                assert root_mod not in banned, f"Banned import-from '{node.module}' in {py_path.name}"


def test_hunyuan_white_cutout_and_mesh_repair_cpu(tmp_path):
    numpy = pytest.importorskip("numpy")
    trimesh = pytest.importorskip("trimesh")

    from services.hunyuan3d2.hy3d_core.postprocessors import clean_and_repair_mesh
    from services.hunyuan3d2.hy3d_core.preprocessors import ImageProcessorV2, fast_white_bg_to_rgba

    # Fast pure-PIL/NumPy white cutout (avoids loading ONNX rembg on clean white studio renders)
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    for y in range(20, 44):
        for x in range(20, 44):
            img.putpixel((x, y), (30, 120, 220))
    rgba, clean = fast_white_bg_to_rgba(img)
    assert clean is True
    arr_in = numpy.asarray(rgba)
    assert arr_in[0, 0, 3] == 0
    assert arr_in[63, 63, 3] == 0
    assert arr_in[32, 32, 3] == 255
    _, mask_out = ImageProcessorV2(size=128, border_ratio=0.15).load_image(rgba, to_tensor=False)
    assert mask_out[0, 0, 0] == 0
    assert mask_out[64, 64, 0] == 255

    # Verify floater removal + watertight check purely in memory (without pymeshlab)
    big = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    tiny = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
    tiny.apply_translation([3.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([big, tiny])
    repaired, stats = clean_and_repair_mesh(combined, max_faces=4000, ensure_watertight=True)
    assert stats["watertight"] is True
    assert stats["non_manifold_edges"] == 0
    assert stats["components"] == 1
    assert len(repaired.faces) > 0


def test_hunyuan_distilled_pipeline_modules_if_torch():
    torch = pytest.importorskip("torch")
    from services.hunyuan3d2.hy3d_core.vae import ShapeVAE

    # Verify decoder-only ShapeVAE omits encoder/torch_cluster completely
    vae = ShapeVAE(num_latents=16, embed_dim=8, width=32, heads=2, num_decoder_layers=1, inference_only=True)
    assert not hasattr(vae, "encoder")
    fe = vae.fourier_embedder(torch.ones((1, 4, 3), dtype=torch.float16))
    assert fe.dtype == torch.float16


def test_scripts_lf_and_notebook_valid():
    for sh in (REPO_ROOT / "scripts").glob("*.sh"):
        raw = sh.read_bytes()
        assert b"\r\n" not in raw, f"{sh.name} must use LF line endings, not CRLF"
    nb = json.loads((REPO_ROOT / "colab_endpoints.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "santiago96ml/studio-3d-ai-endpoints" in joined
