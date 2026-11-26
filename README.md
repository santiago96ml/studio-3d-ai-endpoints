# Studio 3D AI Endpoints (Headless Per-Model GPU Microservices)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/santiago96ml/studio-3d-ai-endpoints/blob/main/colab_endpoints.ipynb)

Arquitectura **headless, sin UI pesada ni runtimes genéricos**, diseñada para exprimir al 100% la GPU de **Google Colab (T4 15 GB / L4 24 GB / A100)** manteniendo la RAM por debajo del límite de 12.7 GB mediante **entornos aislados por modelo** y ejecución nativa C++/GGML CUDA:

| Servicio | Modelo Base + Cuantización HF | Motor Nativo | Puerto | VRAM Estimada |
|---|---|---|---|---|
| `flux2_klein` | `black-forest-labs/FLUX.2-klein-4B` (`unsloth/FLUX.2-klein-4B-GGUF` `Q4_K_M` + `Qwen3-4B-Q4_K_M`) | `leejet/stable-diffusion.cpp` (`sd-cli` CUDA) | `8001` | ~5.5–7 GB |
| `trellis2` | `microsoft/TRELLIS.2-4B` (`ilintar/trellis2-gguf` `q4/` / `q8/`) | `pwilkin/trellis.cpp` (`trellis-cli` CUDA) | `8002` | ~7–10 GB |
| `gateway` | Coordinador HTTP con `asyncio.Lock` (GPU Lease serializado) | FastAPI + `httpx` + `cloudflared` | `8000` | Mínimo (<150 MB RAM) |

## Endpoints HTTP

- `GET /health` → Estado de servicios, pesos cargados y telemetría `nvidia-smi`.
- `POST /v1/concept` (`multipart/form-data`: `prompt`, `ref_image`, `width=768`, `height=768`, `steps=4`, `cfg_scale=1.0`, `seed=42`) → Devuelve `image/png`.
- `POST /v1/image-to-3d` (`multipart/form-data`: `image`, `extra_views`, `res=512`, `steps=12`, `faces=150000`, `no_texture=true`) → Devuelve `.glb` (`model/gltf-binary`).
- `POST /v1/sketch-to-3d` (`multipart/form-data`: `image`, `prompt`) → Encadena `FLUX.2-klein-4B` → `TRELLIS.2-4B` bajo un único lock de GPU y devuelve `concept_png_b64` + `glb_b64` (o `.glb` directo con `response_format=glb`).

## Uso rápido en Google Colab

1. Abrí [`colab_endpoints.ipynb`](https://colab.research.google.com/github/santiago96ml/studio-3d-ai-endpoints/blob/main/colab_endpoints.ipynb) con runtime **GPU T4 / L4**.
2. Ejecutá las celdas de instalación y arranque.
3. Copiá la URL pública `https://xxxx.trycloudflare.com` que imprime en consola y pegala en el **Panel Docente (`/admin`)** de Robótica 3D Studio V2.

## Variables de entorno útiles

- `STUDIO_API_TOKEN`: Token Bearer / `X-API-Key` opcional para proteger todos los endpoints.
- `FLUX2_QUANT`: `Q4_K_M` (default T4) o `Q8_0` (recomendado en L4/A100 >= 20 GB VRAM).
- `TRELLIS2_QUANT`: `q4` (default T4) o `q8` (recomendado en L4/A100).
- `TRELLIS2_INCLUDE_TEXTURE`: `0` (solo geometría para impresión 3D multi-filamento, más rápido y liviano) o `1` (descarga también `tex_flow_512` + `tex_dec`).
