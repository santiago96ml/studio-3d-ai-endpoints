#!/usr/bin/env bash
set -euo pipefail

# Isolated setup for Distilled PartField 3D Segmentation Endpoint (Port 8004)
# Pure PyTorch (zero torch-scatter, Lightning, Open3D, VTK, PyMeshLab or TetGen).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/content/envs}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/content/runtime}"
ENV_DIR="${ENV_ROOT}/partfield"

PARTFIELD_VRAM_MODE="${PARTFIELD_VRAM_MODE:-auto}"
PARTFIELD_UNLOAD_AFTER_REQUEST="${PARTFIELD_UNLOAD_AFTER_REQUEST:-0}"
PARTFIELD_NUM_PARTS="${PARTFIELD_NUM_PARTS:-10}"
PARTFIELD_PC_SAMPLES="${PARTFIELD_PC_SAMPLES:-30000}"
PARTFIELD_PTS_PER_FACE="${PARTFIELD_PTS_PER_FACE:-12}"

mkdir -p "${ENV_ROOT}" "${RUNTIME_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install -q -U uv
fi

if [ ! -d "${ENV_DIR}" ]; then
  uv venv --python 3.11 "${ENV_DIR}" || uv venv "${ENV_DIR}"
fi

PYTHON_BIN="${ENV_DIR}/bin/python"

echo "[partfield] Installing PyTorch CUDA + distilled segmentation dependencies into ${ENV_DIR} ..."
uv pip install --python "${PYTHON_BIN}" torch --index-url https://download.pytorch.org/whl/cu124 \
  || uv pip install --python "${PYTHON_BIN}" torch
uv pip install --python "${PYTHON_BIN}" -r "${REPO_ROOT}/services/partfield/requirements.txt"

echo "[partfield] Caching PartField Objaverse checkpoint (1.24 GB) ..."
HF_HUB_ENABLE_HF_TRANSFER=1 "${PYTHON_BIN}" - <<PY
import torch
from huggingface_hub import hf_hub_download

print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
ckpt = hf_hub_download(repo_id="mikaelaangel/partfield-ckpt", filename="model_objaverse.ckpt")
print("[partfield] Checkpoint cached at:", ckpt)
PY

cat > "${RUNTIME_ROOT}/partfield.env" <<EOF
PARTFIELD_VRAM_MODE=${PARTFIELD_VRAM_MODE}
PARTFIELD_UNLOAD_AFTER_REQUEST=${PARTFIELD_UNLOAD_AFTER_REQUEST}
PARTFIELD_NUM_PARTS=${PARTFIELD_NUM_PARTS}
PARTFIELD_PC_SAMPLES=${PARTFIELD_PC_SAMPLES}
PARTFIELD_PTS_PER_FACE=${PARTFIELD_PTS_PER_FACE}
HF_HUB_ENABLE_HF_TRANSFER=1
EOF

echo "[partfield] Distilled setup complete -> ${RUNTIME_ROOT}/partfield.env"
