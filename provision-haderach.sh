#!/bin/bash
set -Eeuo pipefail
trap 'printf "[Haderach] Provisioning failed at line %s\n" "$LINENO" >&2' ERR

comfy_dir="${COMFYUI_DIR:-${WORKSPACE:-/workspace}/ComfyUI}"
python_bin="${COMFYUI_PYTHON:-/venv/main/bin/python}"
models_dir="$comfy_dir/models"

log() { printf '[Haderach] %s\n' "$*" >&2; }
die() { log "$*"; exit 1; }
pip_install() { "$python_bin" -m pip install "$@"; }

[[ -f "$comfy_dir/main.py" ]] || die "ComfyUI not found at $comfy_dir"
[[ -x "$python_bin" ]] || die "ComfyUI Python not found at $python_bin"
command -v flock >/dev/null || die 'flock is required'
mkdir -p "$comfy_dir/custom_nodes"
exec 9>"$comfy_dir/.haderach-provision.lock"
flock -x 9

if ! command -v ffmpeg >/dev/null; then
    apt-get update
    apt-get install -y --no-install-recommends ffmpeg
fi

cd "$comfy_dir"
if [[ -n "${HADERACH_COMFYUI_REF:-}" ]] ||
   ! grep -qs MiniMaxH3ReferenceToVideo comfy_extras/nodes_minimax_h3.py; then
    log 'Updating ComfyUI for MiniMax H3 support'
    git fetch origin "${HADERACH_COMFYUI_REF:-HEAD}"
    git checkout --detach FETCH_HEAD
fi
grep -qs MiniMaxH3ReferenceToVideo comfy_extras/nodes_minimax_h3.py ||
    die 'Selected ComfyUI revision lacks MiniMaxH3ReferenceToVideo'

install_node() {
    local repo="$1" target="custom_nodes/${1##*/}"
    if [[ ! -d "$target" ]]; then
        git clone --depth 1 "https://github.com/$repo.git" "$target"
    fi
    [[ -f "$target/__init__.py" ]] || die "Invalid node installation: $target"
}

install_node kijai/ComfyUI-KJNodes
install_node Kosinkadink/ComfyUI-VideoHelperSuite
install_node yolain/ComfyUI-Easy-Use
install_node LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
install_node TFL-TFL/ComfyUI_Text_Translation

# Resolve dependencies together and keep the image's CUDA Torch wheels.
constraints=$(mktemp)
trap 'rm -f -- "$constraints"' EXIT
"$python_bin" - <<'PY' > "$constraints"
from importlib.metadata import PackageNotFoundError, version
for package in ("torch", "torchvision", "torchaudio", "triton"):
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError:
        pass
PY
requirements=(-r requirements.txt)
for node in ComfyUI-KJNodes ComfyUI-VideoHelperSuite ComfyUI-Easy-Use \
    Comfyui_Minimax_h3_latent_Upscaler ComfyUI_Text_Translation; do
    file="custom_nodes/$node/requirements.txt"
    if [[ -f "$file" ]]; then requirements+=(-r "$file"); fi
done
pip_install -c "$constraints" "${requirements[@]}" 'huggingface_hub>=0.34'

log 'Downloading and validating MiniMax H3 model files'
# Source repo, path in that repo, destination directory, optional revision.
"$python_bin" - "$models_dir" <<'PY'
import os
import sys
from pathlib import Path
from time import sleep

from huggingface_hub import hf_hub_download
from safetensors import safe_open

root = Path(sys.argv[1])
assets = [
    ("Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models", "main"),
    ("Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders", "main"),
    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_int8_convrot.safetensors", "vae", "main"),
    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", "vae", "main"),
    ("lightx2v/Minimax-h3-Turbo", "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", "loras", "main"),
    ("fal/MiniMax-H3-Realism-People-LoRA", "h3-realism-people-t2v-i2v-r2v.safetensors", "loras", "main"),
    ("larryvrh/MiniMax-H3-Turbo-Lora", "minimax_h3_turbo_v4_step600_ema.safetensors", "loras", "main"),
    ("LBH-123-AI/Minimax_h3_latent_Upscaler", "minimax_h3_latent_upscaler_3d_fp16.safetensors", "latent_upscale_models", "27236837051793bc7cccf628e5936c6110361dab"),
]

def valid_weights(path):
    try:
        with safe_open(path, framework="pt", device="cpu") as weights:
            return bool(weights.keys())
    except Exception:
        return False

for repo, remote, category, revision in assets:
    name = Path(remote).name
    dest = root / category / name
    if valid_weights(dest):
        print(f"[Haderach] Reusing {dest}", flush=True)
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            downloaded = Path(hf_hub_download(
                repo_id=repo, filename=remote, revision=revision,
                local_dir=dest.parent / ".haderach-downloads",
                force_download=attempt > 0,
            ))
            if not valid_weights(downloaded):
                raise ValueError(f"Invalid safetensors file: {downloaded}")
            os.replace(downloaded, dest)
            break
        except Exception:
            if attempt == 2:
                raise
            sleep(2 ** (attempt + 1))
PY

check_sage() {
    "$python_bin" - <<'PY'
from sageattention.core import (
    attn_false, get_cuda_arch_versions, per_block_int8_triton,
    per_channel_fp8, per_thread_int8_triton, per_warp_int8_cuda,
)
from sageattention import sageattn
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert get_cuda_arch_versions(), "SageAttention cannot detect a CUDA architecture"
q = torch.randn((1, 2, 128, 128), device="cuda", dtype=torch.float16)
out = sageattn(q, q, q)
assert torch.isfinite(out).all().item(), "SageAttention GPU smoke test failed"
PY
}
if [[ -n "${HADERACH_SAGE_WHEEL:-}" ]]; then
    pip_install --no-deps --force-reinstall "$HADERACH_SAGE_WHEEL"
elif ! check_sage; then
    command -v nvcc >/dev/null ||
        die 'SageAttention needs a compatible HADERACH_SAGE_WHEEL or an image with nvcc'
    pip_install setuptools wheel packaging ninja
    log 'Building SageAttention v2.2.0 from upstream source'
    MAX_JOBS="${MAX_JOBS:-1}" EXT_PARALLEL="${EXT_PARALLEL:-1}" \
        pip_install --verbose --no-build-isolation --no-deps \
        --force-reinstall 'git+https://github.com/thu-ml/SageAttention.git@v2.2.0'
fi
check_sage

log 'Provisioning complete'
