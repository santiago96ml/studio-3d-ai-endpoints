#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/content/envs}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/content/runtime}"
GATEWAY_VENV="${ENV_ROOT}/gateway"

mkdir -p "${ENV_ROOT}" "${RUNTIME_ROOT}/bin"

if ! command -v uv >/dev/null 2>&1; then
  pip install -q uv
fi

if [ ! -x "${GATEWAY_VENV}/bin/python" ]; then
  uv venv "${GATEWAY_VENV}" --python 3.11 --seed || uv venv "${GATEWAY_VENV}" --python 3.10 --seed || uv venv "${GATEWAY_VENV}" --seed
fi

uv pip install --python "${GATEWAY_VENV}/bin/python" -r "${REPO_ROOT}/gateway/requirements.txt"

if ! command -v cloudflared >/dev/null 2>&1 && [ ! -x "${RUNTIME_ROOT}/bin/cloudflared" ]; then
  echo "[gateway] Installing cloudflared tunnel binary..."
  curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o "${RUNTIME_ROOT}/bin/cloudflared"
  chmod +x "${RUNTIME_ROOT}/bin/cloudflared"
fi

echo "[gateway] Isolated gateway environment ready at ${GATEWAY_VENV}"
