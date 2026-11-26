#!/usr/bin/env bash
set -euo pipefail

# Isolated setup for Distilled Hunyuan3D-2 Geometry Endpoint (Port 8003)
# Strips Gradio, Blender, HunyuanDiT text2image, TexGen CUDA extensions, PyMeshLab & diso.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/content/envs}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/content/runtime}"
ENV_DIR="${ENV_ROOT}/hunyuan3d2"

HUNYUAN3D_VARIANT="${HUNYUAN3D_VARIANT:-turbo}"
HUNYUAN3D_VRAM_MODE="${HUNYUAN3D_VRAM_MODE:-auto}"
HUNYUAN3D_UNLOAD_AFTER_REQUEST="${HUNYUAN3D_UNLOAD_AFTER_REQUEST:-0}"
HUNYUAN3D_STEPS="${HUNYUAN3D_STEPS:-5}"
HUNYUAN3D_OCTREE_RES="${HUNYUAN3D_OCTREE_RES:-384}"
HUNYUAN3D_MAX_FACES="${HUNYUAN3D_MAX_FACES:-150000}"

mkdir -p "${ENV_ROOT}" "${RUNTIME_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install -q -U uv
fi

if [ ! -d "${ENV_DIR}" ]; then
  uv venv --python 3.11 "${ENV_DIR}" || uv venv "${ENV_DIR}"
fi

PYTHON_BIN="${ENV_DIR}/bin/python"

echo "[hunyuan3d2] Installing PyTorch CUDA + distilled geometry dependencies into ${ENV_DIR} ..."
uv pip install --python "${PYTHON_BIN}" torch --index-url https://download.pytorch.org/whl/cu124 \
  || uv pip install --python "${PYTHON_BIN}" torch
uv pip install --python "${PYTHON_BIN}" -r "${REPO_ROOT}/services/hunyuan3d2/requirements.txt"

echo "[hunyuan3d2] Verifying PyTorch CUDA availability & selective downloading (${HUNYUAN3D_VARIANT}) ..."
HF_HUB_ENABLE_HF_TRANSFER=1 "${PYTHON_BIN}" - <<PY
import os
import torch
from huggingface_hub import snapshot_download

print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))

variant = "${HUNYUAN3D_VARIANT}".strip().lower()
if variant == "mini":
    dit_repo, dit_sub = "tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"
    vae_repo, vae_sub = "tencent/Hunyuan3D-2mini", "hunyuan3d-vae-v2-mini-turbo"
else:
    dit_repo, dit_sub = "tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo"
    vae_repo, vae_sub = "tencent/Hunyuan3D-2", "hunyuan3d-vae-v2-0-turbo"

snapshot_download(repo_id=dit_repo, allow_patterns=[f"{dit_sub}/config.yaml", f"{dit_sub}/model.fp16.safetensors"])
snapshot_download(repo_id=vae_repo, allow_patterns=[f"{vae_sub}/config.yaml", f"{vae_sub}/model.fp16.safetensors"])
print("[hunyuan3d2] Selective checkpoints cached:", dit_sub, "+", vae_sub)
PY

cat > "${RUNTIME_ROOT}/hunyuan3d2.env" <<EOF
HUNYUAN3D_VARIANT=${HUNYUAN3D_VARIANT}
HUNYUAN3D_VRAM_MODE=${HUNYUAN3D_VRAM_MODE}
HUNYUAN3D_UNLOAD_AFTER_REQUEST=${HUNYUAN3D_UNLOAD_AFTER_REQUEST}
HUNYUAN3D_STEPS=${HUNYUAN3D_STEPS}
HUNYUAN3D_OCTREE_RES=${HUNYUAN3D_OCTREE_RES}
HUNYUAN3D_MAX_FACES=${HUNYUAN3D_MAX_FACES}
HF_HUB_ENABLE_HF_TRANSFER=1
EOF

echo "[hunyuan3d2] Distilled setup complete -> ${RUNTIME_ROOT}/hunyuan3d2.env"
