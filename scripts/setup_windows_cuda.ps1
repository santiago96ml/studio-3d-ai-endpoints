param(
    [ValidateSet("all", "flux2", "trellis2")]
    [string]$Mode = "all",
    [string]$Flux2Quant = "Q4_K_M",
    [string]$TrellisQuant = "q4",
    [switch]$IncludeTexture,
    [switch]$Include1024
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv CLI is required. Install uv first (e.g. winget install astral-sh.uv)."
}

$EnvRoot = Join-Path $RepoRoot "envs"
$RuntimeRoot = Join-Path $RepoRoot "runtime"
$ModelsRoot = Join-Path $RepoRoot "models"
$SdcppDir = Join-Path $RuntimeRoot "sdcpp_bin"
$TrellisDir = Join-Path $RuntimeRoot "trellis_bin"
$FluxModels = Join-Path $ModelsRoot "flux2"
$TrellisModels = Join-Path $ModelsRoot "trellis2"

New-Item -ItemType Directory -Force $EnvRoot, $RuntimeRoot, $SdcppDir, $TrellisDir, $FluxModels, $TrellisModels | Out-Null

function Ensure-IsolatedVenv {
    param([string]$Name, [string]$ReqFile)
    $VenvDir = Join-Path $EnvRoot $Name
    $PyBin = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path $PyBin)) {
        Write-Host "[$Name] Creating isolated Python venv at $VenvDir..." -ForegroundColor Cyan
        uv venv $VenvDir --python 3.12 --seed
        if ($LASTEXITCODE -ne 0) { uv venv $VenvDir --python 3.11 --seed }
    }
    Write-Host "[$Name] Installing minimal requirements from $ReqFile..." -ForegroundColor Cyan
    uv pip install --python $PyBin -r $ReqFile
    if ($LASTEXITCODE -ne 0) { throw "Failed installing requirements for $Name" }
    return $PyBin
}

function Fetch-And-ExtractZip {
    param([string]$Url, [string]$DestDir, [string]$Tag)
    $ZipPath = Join-Path $RuntimeRoot "$Tag.zip"
    if (Test-Path $ZipPath) {
        if ((Get-Item $ZipPath).Length -eq 0) { Remove-Item -Force $ZipPath -ErrorAction SilentlyContinue }
    }
    if (-not (Test-Path $ZipPath)) {
        Write-Host "[$Tag] Downloading $Url ..." -ForegroundColor Cyan
        curl.exe -fL --retry 3 --silent --show-error -o $ZipPath $Url
        if ($LASTEXITCODE -ne 0) { throw "Failed downloading $Url" }
    }
    Write-Host "[$Tag] Extracting to $DestDir ..." -ForegroundColor Cyan
    Expand-Archive -Path $ZipPath -DestinationPath $DestDir -Force
    Remove-Item -Force $ZipPath -ErrorAction SilentlyContinue
}

$FluxPy = $null
$TrellisPy = $null

if ($Mode -in @("all", "flux2")) {
    $FluxPy = Ensure-IsolatedVenv -Name "flux2" -ReqFile (Join-Path $RepoRoot "services\flux2_klein\requirements.txt")
    $SdCli = Get-ChildItem -Path $SdcppDir -Filter "sd-cli.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $SdCli) {
        Fetch-And-ExtractZip -Url "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-1134-456f52d/sd-master-456f52d-bin-win-cuda12-x64.zip" -DestDir $SdcppDir -Tag "sdcpp_win_cuda12"
        $SdCli = Get-ChildItem -Path $SdcppDir -Filter "sd-cli.exe" -Recurse | Select-Object -First 1
    }
    if (-not $SdCli) { throw "sd-cli.exe not found after extracting Windows CUDA release" }

    $CudartDll = Get-ChildItem -Path $SdcppDir -Filter "cudart64_*.dll" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $CudartDll) {
        Fetch-And-ExtractZip -Url "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-1134-456f52d/cudart-sd-bin-win-cu12-x64.zip" -DestDir $SdcppDir -Tag "sdcpp_cudart12"
    }

    # Ensure CUDA runtime DLLs reside in same folder as sd-cli.exe
    Get-ChildItem -Path $SdcppDir -Filter "*.dll" -Recurse | ForEach-Object {
        $target = Join-Path $SdCli.DirectoryName $_.Name
        if ($_.FullName -ne $target -and -not (Test-Path $target)) {
            Copy-Item $_.FullName $target -Force
        }
    }

    $ModelFile = "flux-2-klein-4b-$Flux2Quant.gguf"
    $LlmFile = "Qwen3-4B-Q4_K_M.gguf"
    $VaeFile = "flux2-vae.safetensors"

    & $FluxPy -c @"
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

models_dir = Path(r'$FluxModels')
targets = [
    ('unsloth/FLUX.2-klein-4B-GGUF', '$ModelFile', '$ModelFile'),
    ('unsloth/Qwen3-4B-GGUF', '$LlmFile', '$LlmFile'),
    ('Comfy-Org/Flux-2-Klein-4B', 'split_files/vae/flux2-vae.safetensors', '$VaeFile'),
]
for repo_id, remote, local in targets:
    dest = models_dir / local
    if dest.is_file() and dest.stat().st_size > 1024 * 1024:
        print('[flux2] Cached:', dest)
        continue
    print('[flux2] Downloading', repo_id, remote)
    cached = hf_hub_download(repo_id=repo_id, filename=remote, token=os.environ.get('HF_TOKEN') or None)
    shutil.copy2(cached, dest)
"@
    if ($LASTEXITCODE -ne 0) { throw "Failed downloading FLUX.2 weights" }

    $FluxEnvContent = @"
FLUX2_BIN_PATH=$($SdCli.FullName)
FLUX2_MODEL_PATH=$(Join-Path $FluxModels $ModelFile)
FLUX2_LLM_PATH=$(Join-Path $FluxModels $LlmFile)
FLUX2_VAE_PATH=$(Join-Path $FluxModels $VaeFile)
FLUX2_QUANT=$Flux2Quant
FLUX2_OFFLOAD_TO_CPU=0
FLUX2_TIMEOUT_SEC=360
FLUX2_BACKEND=sdcpp
"@
    [System.IO.File]::WriteAllText((Join-Path $RuntimeRoot "flux2.env"), ($FluxEnvContent -replace "`r`n", "`n"), [System.Text.UTF8Encoding]::new($false))
}

if ($Mode -in @("all", "trellis2")) {
    $TrellisPy = Ensure-IsolatedVenv -Name "trellis2" -ReqFile (Join-Path $RepoRoot "services\trellis2\requirements.txt")
    $TrellisCli = Get-ChildItem -Path $TrellisDir -Filter "trellis-cli.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $TrellisCli) {
        Fetch-And-ExtractZip -Url "https://github.com/pwilkin/trellis.cpp/releases/download/v0.8.1/trellis-cuda-windows-x64.zip" -DestDir $TrellisDir -Tag "trellis_win_cuda"
        $TrellisCli = Get-ChildItem -Path $TrellisDir -Filter "trellis-cli.exe" -Recurse | Select-Object -First 1
    }
    if (-not $TrellisCli) { throw "trellis-cli.exe not found after extracting Windows CUDA release" }

    $IncTex = if ($IncludeTexture) { "1" } else { "0" }
    $Inc1024 = if ($Include1024) { "1" } else { "0" }
    & $TrellisPy -c @"
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

models_dir = Path(r'$TrellisModels')
quant = '$TrellisQuant'.lower()
prefix = '' if quant == 'f16' else f'{quant}/'
wanted = ['birefnet.gguf', 'dinov3.gguf', 'ss_flow.gguf', 'ss_dec.gguf', 'shape_flow_512.gguf', 'shape_dec.gguf']
if '$Inc1024' == '1':
    wanted.append('shape_flow_1024.gguf')
if '$IncTex' == '1':
    wanted.extend(['tex_flow_512.gguf', 'tex_dec.gguf'])
    if '$Inc1024' == '1':
        wanted.append('tex_flow_1024.gguf')

for fn in wanted:
    dest = models_dir / fn
    if dest.is_file() and dest.stat().st_size > 1024 * 1024:
        print('[trellis2] Cached:', dest)
        continue
    remote = f'{prefix}{fn}'
    print('[trellis2] Downloading ilintar/trellis2-gguf/' + remote)
    cached = hf_hub_download(repo_id='ilintar/trellis2-gguf', filename=remote, token=os.environ.get('HF_TOKEN') or None)
    shutil.copy2(cached, dest)
"@
    if ($LASTEXITCODE -ne 0) { throw "Failed downloading TRELLIS.2 GGUF weights" }

    $TrellisEnvContent = @"
TRELLIS2_BIN_PATH=$($TrellisCli.FullName)
TRELLIS2_MODELS_DIR=$TrellisModels
TRELLIS2_QUANT=$TrellisQuant
TRELLIS2_DEFAULT_RES=512
TRELLIS2_REQUIRE_GPU=1
TRELLIS2_TIMEOUT_SEC=600
"@
    [System.IO.File]::WriteAllText((Join-Path $RuntimeRoot "trellis2.env"), ($TrellisEnvContent -replace "`r`n", "`n"), [System.Text.UTF8Encoding]::new($false))
}

$GwPy = Ensure-IsolatedVenv -Name "gateway" -ReqFile (Join-Path $RepoRoot "gateway\requirements.txt")
Write-Host "Windows isolated CUDA setup complete! Gateway Python: $GwPy" -ForegroundColor Green
