#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/content/envs}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/content/runtime}"
MODELS_ROOT="${MODELS_ROOT:-/content/models}"

FLUX2_VENV="${ENV_ROOT}/flux2"
FLUX2_MODELS="${MODELS_ROOT}/flux2"
SDCPP_DIR="${RUNTIME_ROOT}/stable-diffusion.cpp"
SDCPP_BIN="${SDCPP_DIR}/build/bin/sd-cli"

mkdir -p "${ENV_ROOT}" "${RUNTIME_ROOT}" "${FLUX2_MODELS}"

if ! command -v uv >/dev/null 2>&1; then
  pip install -q uv
fi

if [ ! -x "${FLUX2_VENV}/bin/python" ]; then
  uv venv "${FLUX2_VENV}" --python 3.11 --seed || uv venv "${FLUX2_VENV}" --python 3.10 --seed || uv venv "${FLUX2_VENV}" --seed
fi

uv pip install --python "${FLUX2_VENV}/bin/python" -r "${REPO_ROOT}/services/flux2_klein/requirements.txt"

GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ' || true)"
GPU_ARCH="${GPU_ARCH:-${GPU_CAP:-75}}"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo 15000)"

if [ -z "${FLUX2_QUANT:-}" ]; then
  if [ "${VRAM_MB:-0}" -ge 20000 ] 2>/dev/null; then
    FLUX2_QUANT="Q8_0"
  else
    FLUX2_QUANT="Q4_K_M"
  fi
fi
FLUX2_OFFLOAD_TO_CPU="${FLUX2_OFFLOAD_TO_CPU:-0}"
if [ "${VRAM_MB:-0}" -gt 0 ] && [ "${VRAM_MB:-0}" -lt 12000 ] 2>/dev/null; then
  FLUX2_OFFLOAD_TO_CPU="1"
fi

echo "[flux2] GPU compute arch=${GPU_ARCH}, VRAM=${VRAM_MB} MB, quant=${FLUX2_QUANT}"

if [ ! -x "${SDCPP_BIN}" ]; then
  if [ ! -d "${SDCPP_DIR}/.git" ]; then
    git clone --depth 1 --recursive https://github.com/leejet/stable-diffusion.cpp.git "${SDCPP_DIR}"
  fi
  cmake -S "${SDCPP_DIR}" -B "${SDCPP_DIR}/build" -G Ninja \
    -DSD_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}" \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "${SDCPP_DIR}/build" --target sd-cli -j"$(nproc)"
  find "${SDCPP_DIR}/build" -maxdepth 5 -name "*.so*" -exec cp -Pf {} "${SDCPP_DIR}/build/bin/" \; 2>/dev/null || true
fi

MODEL_FILE="flux-2-klein-4b-${FLUX2_QUANT}.gguf"
LLM_FILE="Qwen3-4B-Q4_K_M.gguf"
VAE_FILE="flux2-vae.safetensors"

"${FLUX2_VENV}/bin/python" - "${FLUX2_MODELS}" "${MODEL_FILE}" "${LLM_FILE}" "${VAE_FILE}" <<'PY'
import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

models_dir = Path(sys.argv[1])
model_file, llm_file, vae_file = sys.argv[2], sys.argv[3], sys.argv[4]

targets = [
    ("unsloth/FLUX.2-klein-4B-GGUF", model_file, model_file),
    ("unsloth/Qwen3-4B-GGUF", llm_file, llm_file),
    ("Comfy-Org/Flux-2-Klein-4B", "split_files/vae/flux2-vae.safetensors", vae_file),
]
for repo_id, remote_name, local_name in targets:
    dest = models_dir / local_name
    if dest.is_file() and dest.stat().st_size > 1024 * 1024:
        print(f"[flux2] Cached: {dest}")
        continue
    print(f"[flux2] Downloading {repo_id}/{remote_name} -> {dest} ...")
    cached = hf_hub_download(repo_id=repo_id, filename=remote_name, token=os.environ.get("HF_TOKEN") or None)
    shutil.copy2(cached, dest)
PY

cat > "${RUNTIME_ROOT}/flux2.env" <<EOF
FLUX2_BIN_PATH=${SDCPP_BIN}
FLUX2_MODEL_PATH=${FLUX2_MODELS}/${MODEL_FILE}
FLUX2_LLM_PATH=${FLUX2_MODELS}/${LLM_FILE}
FLUX2_VAE_PATH=${FLUX2_MODELS}/${VAE_FILE}
FLUX2_QUANT=${FLUX2_QUANT}
FLUX2_OFFLOAD_TO_CPU=${FLUX2_OFFLOAD_TO_CPU}
FLUX2_BACKEND=sdcpp
EOF

echo "[flux2] Isolated FLUX.2-klein-4B setup complete (${RUNTIME_ROOT}/flux2.env)"
