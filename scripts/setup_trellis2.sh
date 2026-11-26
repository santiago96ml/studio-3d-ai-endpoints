#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/content/envs}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/content/runtime}"
MODELS_ROOT="${MODELS_ROOT:-/content/models}"

TRELLIS_VENV="${ENV_ROOT}/trellis2"
TRELLIS_MODELS="${MODELS_ROOT}/trellis2"
TRELLIS_BIN_DIR="${RUNTIME_ROOT}/trellis_bin"
TRELLIS_BIN="${TRELLIS_BIN_DIR}/trellis-cli"
TRELLIS_SRC_DIR="${RUNTIME_ROOT}/trellis.cpp"

mkdir -p "${ENV_ROOT}" "${RUNTIME_ROOT}" "${TRELLIS_MODELS}" "${TRELLIS_BIN_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  pip install -q uv
fi

if [ ! -x "${TRELLIS_VENV}/bin/python" ]; then
  uv venv "${TRELLIS_VENV}" --python 3.11 --seed || uv venv "${TRELLIS_VENV}" --python 3.10 --seed || uv venv "${TRELLIS_VENV}" --seed
fi

uv pip install --python "${TRELLIS_VENV}/bin/python" -r "${REPO_ROOT}/services/trellis2/requirements.txt"

GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ' || true)"
GPU_ARCH="${GPU_ARCH:-${GPU_CAP:-75}}"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo 15000)"
DRIVER_CUDA="$("${TRELLIS_VENV}/bin/python" -c '
import re, subprocess
try:
    out = subprocess.check_output(["nvidia-smi"], text=True, timeout=5)
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    print(m.group(1) if m else "0.0")
except Exception:
    print("0.0")
')"

if [ -z "${TRELLIS2_QUANT:-}" ]; then
  if [ "${VRAM_MB:-0}" -ge 20000 ] 2>/dev/null; then
    TRELLIS2_QUANT="q8"
  else
    TRELLIS2_QUANT="q4"
  fi
fi
TRELLIS2_INCLUDE_TEXTURE="${TRELLIS2_INCLUDE_TEXTURE:-0}"
TRELLIS2_INCLUDE_1024="${TRELLIS2_INCLUDE_1024:-0}"

echo "[trellis2] GPU arch=${GPU_ARCH}, VRAM=${VRAM_MB} MB, host driver CUDA=${DRIVER_CUDA}, quant=${TRELLIS2_QUANT}"

USE_PREBUILT="$("${TRELLIS_VENV}/bin/python" -c "
v = tuple(int(x) for x in '${DRIVER_CUDA}'.split('.')[:2])
print('1' if v >= (13, 1) else '0')
")"

if [ ! -x "${TRELLIS_BIN}" ]; then
  if [ "${USE_PREBUILT}" = "1" ]; then
    echo "[trellis2] Downloading v0.8.1 CUDA prebuilt binary..."
    curl -fsSL "https://github.com/pwilkin/trellis.cpp/releases/download/v0.8.1/trellis-cuda-linux-x64.tar.gz" -o "${RUNTIME_ROOT}/trellis.tar.gz"
    tar -xzf "${RUNTIME_ROOT}/trellis.tar.gz" -C "${TRELLIS_BIN_DIR}"
    rm -f "${RUNTIME_ROOT}/trellis.tar.gz"
    if [ ! -e "${TRELLIS_BIN}" ]; then
      FOUND_BIN="$(find "${TRELLIS_BIN_DIR}" -maxdepth 3 -name "trellis-cli" -type f | head -n1 || true)"
      if [ -n "${FOUND_BIN}" ]; then
        FOUND_DIR="$(dirname "${FOUND_BIN}")"
        if [ "${FOUND_DIR}" != "${TRELLIS_BIN_DIR}" ]; then
          cp -a "${FOUND_DIR}/." "${TRELLIS_BIN_DIR}/"
        fi
      fi
    fi
    chmod +x "${TRELLIS_BIN}" 2>/dev/null || true
    if [ -x "${TRELLIS_BIN}" ] && ! LD_LIBRARY_PATH="${TRELLIS_BIN_DIR}:${LD_LIBRARY_PATH:-}" "${TRELLIS_BIN}" --help >/dev/null 2>&1; then
      echo "[trellis2] Prebuilt binary failed smoke test (CUDA/glibc mismatch); falling back to source build..."
      rm -f "${TRELLIS_BIN}"
    fi
  fi
fi

if [ ! -x "${TRELLIS_BIN}" ]; then
  echo "[trellis2] Compiling trellis.cpp v0.8.1 from source against local CUDA toolkit for sm_${GPU_ARCH}..."
  if [ ! -d "${TRELLIS_SRC_DIR}/.git" ]; then
    git clone --branch v0.8.1 --depth 1 --recursive https://github.com/pwilkin/trellis.cpp.git "${TRELLIS_SRC_DIR}"
  fi
  cmake -S "${TRELLIS_SRC_DIR}" -B "${TRELLIS_SRC_DIR}/build" -G Ninja \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRELLIS_WEBP=OFF
  cmake --build "${TRELLIS_SRC_DIR}/build" --target trellis-cli -j"$(nproc)"
  cp -f "${TRELLIS_SRC_DIR}/build/trellis-cli" "${TRELLIS_BIN}"
  find "${TRELLIS_SRC_DIR}/build" -maxdepth 5 -name "*.so*" -exec cp -Pf {} "${TRELLIS_BIN_DIR}/" \; 2>/dev/null || true
  chmod +x "${TRELLIS_BIN}"
fi

"${TRELLIS_VENV}/bin/python" - "${TRELLIS_MODELS}" "${TRELLIS2_QUANT}" "${TRELLIS2_INCLUDE_TEXTURE}" "${TRELLIS2_INCLUDE_1024}" <<'PY'
import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

models_dir = Path(sys.argv[1])
quant = sys.argv[2].strip().lower()
inc_tex = sys.argv[3].strip() == "1"
inc_1024 = sys.argv[4].strip() == "1"

prefix = "" if quant == "f16" else f"{quant}/"
wanted = [
    "birefnet.gguf",
    "dinov3.gguf",
    "ss_flow.gguf",
    "ss_dec.gguf",
    "shape_flow_512.gguf",
    "shape_dec.gguf",
]
if inc_1024:
    wanted.append("shape_flow_1024.gguf")
if inc_tex:
    wanted.extend(["tex_flow_512.gguf", "tex_dec.gguf"])
    if inc_1024:
        wanted.append("tex_flow_1024.gguf")

for fname in wanted:
    dest = models_dir / fname
    if dest.is_file() and dest.stat().st_size > 1024 * 1024:
        print(f"[trellis2] Cached: {dest}")
        continue
    remote = f"{prefix}{fname}"
    print(f"[trellis2] Downloading ilintar/trellis2-gguf/{remote} -> {dest} ...")
    cached = hf_hub_download(
        repo_id="ilintar/trellis2-gguf",
        filename=remote,
        token=os.environ.get("HF_TOKEN") or None,
    )
    shutil.copy2(cached, dest)
PY

cat > "${RUNTIME_ROOT}/trellis2.env" <<EOF
TRELLIS2_BIN_PATH=${TRELLIS_BIN}
TRELLIS2_MODELS_DIR=${TRELLIS_MODELS}
TRELLIS2_QUANT=${TRELLIS2_QUANT}
TRELLIS2_DEFAULT_RES=512
TRELLIS2_REQUIRE_GPU=1
EOF

echo "[trellis2] Isolated Microsoft TRELLIS.2-4B setup complete (${RUNTIME_ROOT}/trellis2.env)"
